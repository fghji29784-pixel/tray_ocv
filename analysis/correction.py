"""[S7] 원인 인자 기반 OCV 구배 보정 + 판정 영향 평가 (최종 목표).

1안(인자 보정): 확정된 온도 인자로 위치 구배 g_hat 추정 → docv7_corr = docv7 - g_hat.
                학습은 정상 셀만 (불량/핫셀 신호를 보정으로 지우지 않도록).
2안(공간 detrend, 폴백/비교): 트레이별 저차 2D 다항식으로 매끄러운 위치 성분 제거.
안전장치: 고립 핫셀(진짜 불량)의 잔차가 보정 후에도 보존되는지 검사.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .fields import structure_table
from .judge import apply_judgment


def fit_factor_correction(cell_tbl: pd.DataFrame, predictors: list[str],
                          target: str = "docv7") -> pd.DataFrame:
    """온도 인자로 트레이 내 구배 추정 → 보정. 정상 셀로 학습.

    ΔT 예측자는 트레이 중앙값 제거 상태이므로 예측자 부분이 곧 위치 구배.
    """
    df = cell_tbl.copy()
    use = [p for p in predictors if p in df.columns]
    if not use:
        df["g_hat_factor"] = 0.0
        df[f"{target}_corr_factor"] = df[target]
        return df

    train = df[~df.get("judge_fail", False).astype(bool)] if "judge_fail" in df else df
    # 타깃도 트레이 중앙값 제거 (정상 셀 기준)
    tmed = train.groupby("tray_id")[target].transform("median")
    d = pd.DataFrame({"y": train[target] - tmed})
    for p in use:
        d[p] = pd.to_numeric(train[p], errors="coerce")
    d = d.dropna()
    if len(d) < max(20, 3 * len(use)):
        df["g_hat_factor"] = 0.0
        df[f"{target}_corr_factor"] = df[target]
        return df

    X = np.column_stack([np.ones(len(d)), d[use].to_numpy()])
    beta, *_ = np.linalg.lstsq(X, d["y"].to_numpy(), rcond=None)

    Xall = df[use].apply(pd.to_numeric, errors="coerce")
    g = beta[0] + Xall.to_numpy() @ beta[1:]
    g = np.nan_to_num(g, nan=0.0)
    df["g_hat_factor"] = g
    df[f"{target}_corr_factor"] = df[target] - g
    return df


def fit_spatial_detrend(cell_tbl: pd.DataFrame, target: str = "docv7",
                        degree: int = 2) -> pd.DataFrame:
    """트레이별 저차 2D 다항식 표면 제거 (원인 불문 위치 성분). 정상 셀로 적합."""
    df = cell_tbl.copy()
    out = np.zeros(len(df))
    fail = df.get("judge_fail", pd.Series(False, index=df.index)).astype(bool)
    for tray, idx in df.groupby("tray_id").groups.items():
        sub = df.loc[idx]
        r = pd.to_numeric(sub["row"], errors="coerce").to_numpy(dtype=float)
        c = pd.to_numeric(sub["col"], errors="coerce").to_numpy(dtype=float)
        y = pd.to_numeric(sub[target], errors="coerce").to_numpy(dtype=float)
        design = _poly_design(r, c, degree)
        train = np.isfinite(y) & np.isfinite(design).all(1) & (~fail.loc[idx].to_numpy())
        if train.sum() < design.shape[1] + 3:
            continue
        beta, *_ = np.linalg.lstsq(design[train], y[train], rcond=None)
        surf = design @ beta
        pos = df.index.get_indexer(idx)
        out[pos] = np.where(np.isfinite(surf), surf, 0.0)
    df["g_hat_spatial"] = out
    df[f"{target}_corr_spatial"] = df[target] - out
    return df


def _poly_design(r, c, degree):
    r = r - np.nanmean(r)
    c = c - np.nanmean(c)
    terms = [np.ones_like(r), r, c]
    if degree >= 2:
        terms += [r * r, c * c, r * c]
    return np.column_stack(terms)


def evaluate(cell_tbl: pd.DataFrame, corrected_cols: dict[str, str],
             cfg, n_rows: int, n_cols: int) -> dict:
    """무보정 vs 각 보정안의 구배 진폭·판정 영향·핫셀 보존 비교.

    corrected_cols: {"factor": "docv7_corr_factor", "spatial": "docv7_corr_spatial"}
    """
    offset = cfg.raw["judge"]["offset_mv"]
    bin_mv = cfg.raw["judge"]["mode_bin_mv"]
    report: dict = {}

    variants = {"none": "docv7", **corrected_cols}
    known_fail = cell_tbl.get("judge_fail", pd.Series(False, index=cell_tbl.index)).astype(bool)

    for name, col in variants.items():
        if col not in cell_tbl.columns:
            continue
        tmp = cell_tbl[["lot", "tray_id", "row", "col", "cell_id"]].copy()
        tmp["docv7"] = cell_tbl[col]
        judged = apply_judgment(tmp, offset_mv=offset, bin_mv=bin_mv)
        st = structure_table(judged, "docv7_dev", n_rows, n_cols, by=("tray_id",))
        report[name] = {
            "amp_p2p_median": float(st["amp_p2p"].median()),
            "amp_p95p5_median": float(st["amp_p95p5"].median()),
            "ring_abs_median": float(st["ring"].abs().median()),
            "n_fail": int(judged["judge_fail"].sum()),
        }
        if name != "none":
            # 핫셀 보존: 무보정에서 불량이던 셀의 보정 후 dev 유지율
            base = variants["none"]
            base_judged = apply_judgment(
                cell_tbl.assign(docv7=cell_tbl[base])[
                    ["lot", "tray_id", "row", "col", "cell_id", "docv7"]],
                offset_mv=offset, bin_mv=bin_mv)
            was_fail = base_judged["judge_fail"].to_numpy()
            now_fail = judged["judge_fail"].to_numpy()
            preserved = int((was_fail & now_fail).sum())
            report[name]["hotcell_preserved"] = preserved
            report[name]["hotcell_total"] = int(was_fail.sum())
            report[name]["flips_fail_to_pass"] = int((was_fail & ~now_fail).sum())
            report[name]["flips_pass_to_fail"] = int((~was_fail & now_fail).sum())
    return report
