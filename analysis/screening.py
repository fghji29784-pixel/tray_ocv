"""[S5] 전 공정 스텝 × OCV 패턴 전수 상관 스크리닝.

각 (예측자=스텝 ΔT/피처, 타깃=ΔOCV1/2/3/ΔdOCV7)에 대해 트레이별 공간 상관의
분포(중앙값, IQR, 부호 일관성)를 구하고, permutation + BH-FDR 로 유의성 판정.
치환 셀 포함/제외 감도분석 병행.

성능: 트레이 그룹 인덱스를 한 번만 만들고, permutation 은 트레이별로
셔플된 y 행렬(n_perm×n)과 고정 x 의 상관을 벡터화해 계산한다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _tray_index_groups(df: pd.DataFrame) -> list[np.ndarray]:
    """트레이별 정수 위치 인덱스 배열 리스트 (한 번만 계산)."""
    codes = pd.factorize(df["tray_id"])[0]
    order = np.argsort(codes, kind="stable")
    codes_sorted = codes[order]
    bounds = np.flatnonzero(np.diff(codes_sorted)) + 1
    return [g for g in np.split(order, bounds)]


def _corr_fixed_x(xv: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """고정 x(n,) 와 Y(m,n) 각 행의 Pearson 상관 → (m,)."""
    xc = xv - xv.mean()
    xden = np.sqrt((xc * xc).sum())
    Yc = Y - Y.mean(axis=1, keepdims=True)
    Yden = np.sqrt((Yc * Yc).sum(axis=1))
    num = Yc @ xc
    den = Yden * xden
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, np.nan)


def _per_tray_arrays(x: np.ndarray, y: np.ndarray, groups: list[np.ndarray],
                     min_cells: int) -> list[tuple[np.ndarray, np.ndarray]]:
    out = []
    for g in groups:
        xv, yv = x[g], y[g]
        m = np.isfinite(xv) & np.isfinite(yv)
        if m.sum() < min_cells:
            continue
        xv, yv = xv[m], yv[m]
        if xv.std() < 1e-12 or yv.std() < 1e-12:
            continue
        out.append((xv, yv))
    return out


def _aggregate(corrs: np.ndarray) -> dict:
    if corrs.size == 0:
        return {"n_tray": 0, "median": np.nan, "iqr": np.nan,
                "sign_consistency": np.nan, "mean_abs": np.nan}
    med = float(np.median(corrs))
    q1, q3 = np.percentile(corrs, [25, 75])
    sign = float(np.mean(np.sign(corrs) == np.sign(med))) if med != 0 else 0.5
    return {"n_tray": int(corrs.size), "median": med, "iqr": float(q3 - q1),
            "sign_consistency": sign, "mean_abs": float(np.mean(np.abs(corrs)))}


def _permutation_p(tray_arrays, observed_med, n_perm, rng) -> float:
    """트레이 내 y 셔플 → median-corr null 분포 (벡터화) → 양측 p-value."""
    if not np.isfinite(observed_med) or n_perm <= 0 or not tray_arrays:
        return np.nan
    null_meds = np.empty((len(tray_arrays), n_perm))
    for i, (xv, yv) in enumerate(tray_arrays):
        Y = np.array([rng.permutation(yv) for _ in range(n_perm)])
        null_meds[i] = _corr_fixed_x(xv, Y)
    med_per_perm = np.nanmedian(null_meds, axis=0)
    ge = np.sum(np.abs(med_per_perm) >= abs(observed_med))
    return (ge + 1) / (n_perm + 1)


def _bh_fdr(pvals: np.ndarray) -> np.ndarray:
    p = np.asarray(pvals, dtype=float)
    ok = np.isfinite(p)
    q = np.full_like(p, np.nan)
    idx = np.where(ok)[0]
    if idx.size == 0:
        return q
    order = idx[np.argsort(p[idx])]
    n = idx.size
    prev = 1.0
    for rank, i in enumerate(reversed(order), start=1):
        r = n - rank + 1
        prev = min(prev, p[i] * n / r)
        q[i] = prev
    return q


def screen(cell_tbl: pd.DataFrame, predictors: list[str], targets: list[str],
           n_perm: int = 500, seed: int = 0, min_cells: int = 8,
           exclude_imputed: bool = False, imputed_col: str = "any_imputed",
           progress=None) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = cell_tbl
    if exclude_imputed and imputed_col in df.columns:
        df = df[~df[imputed_col].astype(bool)]
    groups = _tray_index_groups(df)

    # 예측자/타깃 numpy 캐시
    col_cache = {c: pd.to_numeric(df[c], errors="coerce").to_numpy()
                 for c in set(predictors) | set(targets) if c in df.columns}

    rows = []
    for ti, tgt in enumerate(targets):
        if tgt not in col_cache:
            continue
        if progress:
            progress(f"타깃 {tgt} ({ti + 1}/{len(targets)}) — 예측자 {len(predictors)}개")
        y = col_cache[tgt]
        for pred in predictors:
            if pred not in col_cache:
                continue
            x = col_cache[pred]
            ta = _per_tray_arrays(x, y, groups, min_cells)
            corrs = np.array([np.corrcoef(xv, yv)[0, 1] for xv, yv in ta])
            agg = _aggregate(corrs)
            pval = _permutation_p(ta, agg["median"], n_perm, rng)
            rows.append({"target": tgt, "predictor": pred, **agg, "p_perm": pval})
    res = pd.DataFrame(rows)
    if len(res):
        res["q_fdr"] = _bh_fdr(res["p_perm"].to_numpy())
        res = res.sort_values("mean_abs", ascending=False, ignore_index=True)
    return res


def per_tray_corr_values(cell_tbl: pd.DataFrame, xcol: str, ycol: str,
                         min_cells: int = 8) -> np.ndarray:
    """단일 예측자-타깃의 트레이별 상관값 배열 (진행분석용)."""
    groups = _tray_index_groups(cell_tbl)
    x = pd.to_numeric(cell_tbl[xcol], errors="coerce").to_numpy()
    y = pd.to_numeric(cell_tbl[ycol], errors="coerce").to_numpy()
    ta = _per_tray_arrays(x, y, groups, min_cells)
    return np.array([np.corrcoef(xv, yv)[0, 1] for xv, yv in ta])


def aggregate(corrs: np.ndarray) -> dict:
    return _aggregate(corrs)


def step_similarity(cell_tbl: pd.DataFrame, step_predictors: list[str]) -> pd.DataFrame:
    cols = [c for c in step_predictors if c in cell_tbl.columns]
    return cell_tbl[cols].corr()
