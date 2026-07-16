"""[S4] 셀×스텝 온도 피처 추출.

데이터가 스텝당 온도 1값(wide)이므로 승온기울기(T_rise)는 관측 불가 → 축소 모드:
스텝별 온도값과 그 트레이 내 편차(ΔT)를 피처로 삼는다.

프록시 설계:
  - 각 충방전 스텝 온도는 '직전 에이징에서 나온 직후의 잔존 온도 필드'를 일부 반영
    → 관측 불가한 에이징 챔버 구배의 간접 프록시.
  - 7th Discharge 온도 = 전용OCV 측정 직전의 마지막 관측 필드 (P1 측정시점 프록시).
  - HT 에이징 인접 충전 스텝 온도 = 고온 이월(P3) 프록시.
  - HeatExposure: 아레니우스 가중 exp(-Ea/RT). duration 미관측 시 상대 가중만.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import STEP_META, Config
from .fields import add_tray_delta

R_GAS = 8.314  # J/mol/K


def _arrhenius_weight(temp_c: pd.Series, ea_kj: float) -> pd.Series:
    """exp(-Ea/RT) 상대 가중 (T in Celsius→Kelvin). 정규화 없이 상대값."""
    tk = pd.to_numeric(temp_c, errors="coerce") + 273.15
    return np.exp(-(ea_kj * 1000.0) / (R_GAS * tk))


def build(temp_clean: pd.DataFrame, cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    """반환: (cell_features wide, temp_long_delta).

    cell_features: 셀당 1행. dT_<step> (트레이 내 편차) + 프록시 집계 피처 + id.
    temp_long_delta: temp_clean + d_temp (스텝별 트레이 편차) — 필드 상관용.
    """
    present = cfg.present_steps
    ea_list = cfg.raw["run"]["arrhenius_ea_kj"]

    long = temp_clean.copy()
    long = add_tray_delta(long, "temp", by=("tray_id", "step_name"), out_col="d_temp")

    # wide: 셀 × dT_<step>
    id_cols = ["lot", "tray_id", "row", "col", "cell_id"]
    wide = long.pivot_table(
        index=id_cols, columns="step_name", values="d_temp", aggfunc="first"
    )
    wide.columns = [f"dT::{c}" for c in wide.columns]
    wide = wide.reset_index()

    # 절대 온도값 wide (프록시 집계용)
    wide_abs = long.pivot_table(
        index=id_cols, columns="step_name", values="temp", aggfunc="first"
    )
    abs_cols = {c: f"T::{c}" for c in wide_abs.columns}
    wide_abs = wide_abs.rename(columns=abs_cols).reset_index()
    feat = wide.merge(wide_abs, on=id_cols, how="left")

    # --- 상(phase)별 평균 ΔT 프록시 ---
    def phase_steps(phase: str) -> list[str]:
        return [s for s in present if STEP_META.get(s, {}).get("phase") == phase]

    charge_steps = phase_steps("charge")
    dis_steps = phase_steps("discharge")

    def mean_dT(steps: list[str]) -> pd.Series:
        cols = [f"dT::{s}" for s in steps if f"dT::{s}" in feat.columns]
        return feat[cols].mean(axis=1) if cols else pd.Series(np.nan, index=feat.index)

    feat["dT_charge_mean"] = mean_dT(charge_steps)
    feat["dT_discharge_mean"] = mean_dT(dis_steps)

    # P1 프록시: 전용OCV 직전 마지막 관측 (가장 뒤 discharge)
    last_dis = dis_steps[-1] if dis_steps else (present[-1] if present else None)
    if last_dis and f"dT::{last_dis}" in feat.columns:
        feat["dT_preOCV_proxy"] = feat[f"dT::{last_dis}"]
    else:
        feat["dT_preOCV_proxy"] = np.nan

    # P3 프록시: HT 에이징 직전/직후 충전 스텝 ΔT 평균
    ht_adjacent = _ht_adjacent_steps(present)
    feat["dT_HT_adjacent"] = mean_dT(ht_adjacent)

    # HeatExposure 아레니우스 가중 (충방전 전체, Ea 감도)
    for ea in ea_list:
        w = _arrhenius_weight(long["temp"], ea)
        long[f"heat_ea{int(ea)}"] = w
        he = long.groupby(id_cols)[f"heat_ea{int(ea)}"].sum().rename(
            f"HeatExp_ea{int(ea)}")
        feat = feat.merge(he, on=id_cols, how="left")
        # ΔHeatExposure (트레이 편차)
        add_tray_delta(feat, f"HeatExp_ea{int(ea)}", by=("tray_id",),
                       out_col=f"dHeatExp_ea{int(ea)}")

    return feat, long


def _ht_adjacent_steps(present: list[str]) -> list[str]:
    """HT 에이징 바로 앞/뒤의 충방전(관측되는) 스텝을 프록시로."""
    out: list[str] = []
    for i, s in enumerate(present):
        if STEP_META.get(s, {}).get("high_temp"):
            for j in (i - 1, i + 1):
                if 0 <= j < len(present):
                    nb = present[j]
                    if STEP_META.get(nb, {}).get("phase") in ("charge", "discharge"):
                        out.append(nb)
    return list(dict.fromkeys(out))
