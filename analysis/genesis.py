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


# 각 OCV 단계 직전 공정 (확정된 공정 순서 기준)
# LCI>RT1>OCV1>C1>C2>OCV2>HT1>RT2>OCV3>C3>C4>OCV4>HT2>RT3>OCV5>
# C5>C6>C7>OCV6>D1..D7>OCV7>RT4>PRVT1>RT5>PRVT2>RT6>PRVT3
PROCESS_BEFORE_STAGE = {
    "OCV #01": "LCI+RT1 (baseline)",
    "OCV #02": "C1,C2 (charge→SOC10)",
    "OCV #03": "HT1+RT2 (1st HT aging)",
    "OCV #04": "C3,C4 (charge→SOC70)",
    "OCV #05": "HT2+RT3 (2nd HT aging)",
    "OCV #06": "C5,C6,C7 (charge→SOC100)",
    "OCV #07": "D1~D7 (discharge→SOC30)",
    "PRIVT OCV #01": "RT4 (aging@SOC30)",
    "PRIVT OCV #02": "RT5 (aging@SOC30)",
    "PRIVT OCV #03": "RT6 (aging@SOC30)",
}


def step_contributions(cell: pd.DataFrame, stage_meta: list[dict],
                       n_rows: int, n_cols: int, docv7_field: str = "d_docv7") -> pd.DataFrame:
    """공정별 기여 분해: 인접 OCV 단계의 차(ΔV=뒤−앞) 필드 링 + docv7 상관.

    같은 SOC를 잇는 구간(HT1: OCV2→3, HT2: OCV4→5, 최종에이징: OCV7→PRVT…)은
    충·방전 효과가 상쇄되어 그 공정 고유의 공간 구배만 남는다(matched_soc=True).
    이런 구간의 차-필드 링이 최종 docv7 링과 일치하면 그 공정이 원인.
    """
    stages = [m for m in sorted(stage_meta, key=lambda x: x.get("order", 0))
              if f"ocvstage::{m['label']}" in cell.columns]
    soc = {m["label"]: m.get("soc") for m in stages}
    labels = [m["label"] for m in stages]
    rows = []
    for prev, cur in zip(labels[:-1], labels[1:]):
        d = cell[["tray_id"]].copy()
        d["row"] = cell["row"]; d["col"] = cell["col"]
        d["diff"] = cell[f"ocvstage::{cur}"] - cell[f"ocvstage::{prev}"]
        add_tray_delta(d, "diff", by=("tray_id",), out_col="d_diff")
        rings = []
        for _, sub in d.groupby("tray_id", sort=False):
            g = to_grid(sub, "d_diff", n_rows, n_cols)
            if np.isfinite(g).sum() >= 0.3 * n_rows * n_cols:
                rings.append(ring_score(g))
        rings = np.array([r for r in rings if np.isfinite(r)])
        cwd = {}
        if docv7_field in cell.columns:
            d[docv7_field] = cell[docv7_field].to_numpy()
            cwd = aggregate(per_tray_corr_values(d, "d_diff", docv7_field))
        # RT5/RT6(PRVT1→2→3)는 docv7=PRVT1−PRVT3 을 산술적으로 구성 → 상관이 자명(≈±1)
        composes = prev.startswith("PRIVT") and cur.startswith("PRIVT")
        rows.append({
            "process": PROCESS_BEFORE_STAGE.get(cur, "?"),
            "from_stage": prev, "to_stage": cur,
            "matched_soc": soc.get(prev) == soc.get(cur) and soc.get(cur) is not None,
            "composes_docv7": composes,
            "diff_ring_abs_median": float(np.median(np.abs(rings))) if rings.size else np.nan,
            "corr_docv7_median": cwd.get("median", np.nan),
            "corr_docv7_signcons": cwd.get("sign_consistency", np.nan),
        })
    return pd.DataFrame(rows)


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
