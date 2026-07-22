"""[S8] 구배 발생 추적 — docv7 링 구배가 '어느 공정 단계'에서 생기는지 규명.

중간 OCV(#01~#07) + 전용 OCV(PRIVT #01~#03)의 트레이 내 ΔOCV 필드를 공정 순서대로
만들어, (a) 링 진폭이 어느 단계에서 커지는지, (b) 각 단계 필드가 최종 Δdocv7 필드와
얼마나 공간적으로 일치하는지를 추적한다. 원인 공정(고온에이징 vs 최종에이징 등) 국소화.

이 분석은 '온도로 설명 안 되는' docv7 구배의 발생 시점을 OCV 값 자체로 짚기 위한 것.
공간 detrend(증상 제거)와 달리 원인 공정을 지목한다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .fields import add_tray_delta, ring_score, amplitude_p95p5
from .grids import to_grid
from .screening import per_tray_corr_values, aggregate


def stage_delta_columns(cell: pd.DataFrame) -> list[str]:
    return [c for c in cell.columns if c.startswith("ocvstage::")]


def build_stage_deltas(cell: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """각 단계 OCV의 트레이 중앙값 제거 Δ필드 컬럼 생성."""
    stage_cols = stage_delta_columns(cell)
    dcols = []
    for c in stage_cols:
        out = "d_" + c
        add_tray_delta(cell, c, by=("tray_id",), out_col=out)
        dcols.append(out)
    return cell, dcols


def progression(cell: pd.DataFrame, stage_meta: list[dict], n_rows: int, n_cols: int,
                docv7_field: str = "d_docv7") -> pd.DataFrame:
    """단계별: 링 진폭(트레이 중앙값)·필드 진폭 + 최종 Δdocv7 과의 공간상관 분포.

    corr_with_docv7 이 처음 크게 오르는 단계 = 링 구배가 태어난(또는 확정된) 지점.
    """
    order = {m["label"]: m.get("order", i) for i, m in enumerate(stage_meta)}
    soc = {m["label"]: m.get("soc") for m in stage_meta}
    rows = []
    for m in stage_meta:
        lab = m["label"]
        col = f"ocvstage::{lab}"
        dcol = f"d_{col}"
        if dcol not in cell.columns:
            continue
        # 트레이별 링/진폭
        rings, amps = [], []
        for _, sub in cell.groupby("tray_id", sort=False):
            g = to_grid(sub, dcol, n_rows, n_cols)
            if np.isfinite(g).sum() < 0.3 * n_rows * n_cols:
                continue
            rings.append(ring_score(g))
            amps.append(amplitude_p95p5(g))
        rings = np.array([r for r in rings if np.isfinite(r)])
        amps = np.array([a for a in amps if np.isfinite(a)])
        # 최종 docv7 필드와의 공간 상관 (단계 필드가 docv7 링과 얼마나 닮았나)
        cwd = aggregate(per_tray_corr_values(cell, dcol, docv7_field)) \
            if docv7_field in cell.columns else {}
        rows.append({
            "stage": lab, "order": order.get(lab), "soc": soc.get(lab),
            "ring_abs_median": float(np.median(np.abs(rings))) if rings.size else np.nan,
            "ring_median": float(np.median(rings)) if rings.size else np.nan,
            "amp_p95p5_median": float(np.median(amps)) if amps.size else np.nan,
            "corr_docv7_median": cwd.get("median", np.nan),
            "corr_docv7_signcons": cwd.get("sign_consistency", np.nan),
            "n_tray": cwd.get("n_tray", 0),
        })
    df = pd.DataFrame(rows).sort_values("order", ignore_index=True)
    return df


def selfdischarge_segments(cell: pd.DataFrame, stage_meta: list[dict],
                           n_rows: int, n_cols: int) -> pd.DataFrame:
    """SOC30 동일 단계들(OCV#07, PRIVT#01→#02→#03) 사이 자기방전 구간별 링.

    같은 SOC라 직접 차분 가능. 각 구간 ΔV = 앞단계 - 뒷단계(전압강하)의 링을 본다.
    링이 최종에이징 내내 일정하게 쌓이면 → 최종에이징 챔버 열구배(자기방전) 시사.
    """
    soc30 = [m["label"] for m in sorted(stage_meta, key=lambda x: x.get("order", 0))
             if m.get("soc") == 30 and f"ocvstage::{m['label']}" in cell.columns]
    rows = []
    for a, b in zip(soc30[:-1], soc30[1:]):
        va, vb = f"ocvstage::{a}", f"ocvstage::{b}"
        seg = cell[["tray_id", "row", "col"]].copy()
        seg["drop"] = cell[va] - cell[vb]        # 전압강하(자기방전)
        add_tray_delta(seg, "drop", by=("tray_id",), out_col="d_drop")
        rings = []
        for _, sub in seg.groupby("tray_id", sort=False):
            g = to_grid(sub, "d_drop", n_rows, n_cols)
            if np.isfinite(g).sum() >= 0.3 * n_rows * n_cols:
                rings.append(ring_score(g))
        rings = np.array([r for r in rings if np.isfinite(r)])
        rows.append({
            "segment": f"{a} → {b}",
            "drop_mean": float(np.nanmean(cell[va] - cell[vb])),
            "ring_abs_median": float(np.median(np.abs(rings))) if rings.size else np.nan,
            "ring_median": float(np.median(rings)) if rings.size else np.nan,
        })
    return pd.DataFrame(rows)
