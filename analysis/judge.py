"""판정 로직: 트레이별 docv7 mode + (docv7 - mode) > offset → 불량."""
from __future__ import annotations

import numpy as np
import pandas as pd


def tray_mode(values: np.ndarray, bin_mv: float = 0.05) -> float:
    """연속값의 robust mode: bin_mv 폭 히스토그램의 최빈 bin 중심.

    표본이 적으면 중앙값으로 폴백.
    """
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return np.nan
    if v.size < 5 or bin_mv <= 0:
        return float(np.median(v))
    lo, hi = v.min(), v.max()
    if hi - lo < bin_mv:
        return float(np.median(v))
    nbins = max(1, int(np.ceil((hi - lo) / bin_mv)))
    counts, edges = np.histogram(v, bins=nbins)
    k = int(np.argmax(counts))
    return float(0.5 * (edges[k] + edges[k + 1]))


def apply_judgment(ocv_cell: pd.DataFrame, offset_mv: float = 0.8,
                   bin_mv: float = 0.05) -> pd.DataFrame:
    """docv7_tray_mode, docv7_dev, judge(bool) 컬럼 추가.

    입력 docv7 은 이미 mV 단위(unit_scale_to_mv 적용 완료) 가정.
    """
    df = ocv_cell.copy()
    modes = (
        df.groupby("tray_id")["docv7"]
        .apply(lambda s: tray_mode(s.to_numpy(), bin_mv))
        .rename("docv7_tray_mode")
    )
    df = df.merge(modes, on="tray_id", how="left")
    df["docv7_dev"] = df["docv7"] - df["docv7_tray_mode"]
    df["judge_fail"] = df["docv7_dev"] > offset_mv
    return df
