"""[S2] QC(결측/≤22°C 삭제) + 3×3 이웃 평균 치환.

규칙 (사용자 지정):
  - 결측치 또는 22°C 이하 값 → 무효화
  - 무효 셀은 같은 (tray, step) 필드 내 3×3 이웃(자신 제외 최대 8셀)의
    유효 값 평균으로 치환
  - 치환 마스크(imputed)를 끝까지 유지 → 하류 상관/회귀에서 감도분석에 사용
    (이웃 평균 치환은 필드를 매끄럽게 만들어 공간 상관을 부풀리기 때문)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import convolve2d

from .config import Config
from .grids import infer_shape, to_grid


def impute_neighbor_3x3(
    grid: np.ndarray, min_valid: int = 3, max_iters: int = 3
) -> tuple[np.ndarray, np.ndarray]:
    """NaN-aware 3×3 이웃 평균 치환.

    반환: (치환된 격자, 이번에 치환된 위치 마스크).
    테두리/모서리는 존재하는 이웃만 사용. 유효 이웃 < min_valid 이면 보류(다음 반복).
    """
    g = grid.copy()
    filled = np.zeros_like(grid, dtype=bool)
    kernel = np.ones((3, 3), dtype=float)  # 자신 포함 창이지만 NaN 셀은 valid=0이라 무해
    for _ in range(max_iters):
        nan_mask = np.isnan(g)
        if not nan_mask.any():
            break
        valid = ~np.isnan(g)
        vals = np.where(valid, g, 0.0)
        nsum = convolve2d(vals, kernel, mode="same")
        ncnt = convolve2d(valid.astype(float), kernel, mode="same")
        with np.errstate(invalid="ignore", divide="ignore"):
            fill = nsum / np.where(ncnt > 0, ncnt, np.nan)
        can_fill = nan_mask & (ncnt >= min_valid)
        g[can_fill] = fill[can_fill]
        filled |= can_fill
        if not can_fill.any():
            break
    return g, filled


def run(temp_long: pd.DataFrame, cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    """temp_long → (temp_clean, impute_report).

    temp_clean 컬럼 추가: temp, qc_flag, imputed(bool)
    qc_flag ∈ {ok, was_missing, was_le_min, imputed, imputed_fail}
    """
    pp = cfg.raw["preprocess"]
    tmin = pp["temp_min_valid"]
    min_valid = pp["min_valid_neighbors"]
    max_iters = pp["max_impute_iters"]
    fallback = pp["fallback_tray_median"]
    n_rows_cfg = cfg.raw["tray_shape"]["n_rows"]
    n_cols_cfg = cfg.raw["tray_shape"]["n_cols"]
    n_rows, n_cols = infer_shape(temp_long, n_rows_cfg, n_cols_cfg)

    df = temp_long.copy()
    df["temp_raw"] = pd.to_numeric(df["temp_raw"], errors="coerce")
    # 1) 무효화
    invalid_missing = df["temp_raw"].isna()
    invalid_low = df["temp_raw"] <= tmin
    df["qc_flag"] = "ok"
    df.loc[invalid_low, "qc_flag"] = "was_le_min"
    df.loc[invalid_missing, "qc_flag"] = "was_missing"
    df["temp"] = df["temp_raw"].where(~(invalid_missing | invalid_low), np.nan)
    df["imputed"] = False

    reports = []
    out_parts = []
    for (tray, step), sub in df.groupby(["tray_id", "step_name"], sort=False):
        sub = sub.copy()
        grid = to_grid(sub, "temp", n_rows, n_cols)
        n_cells = np.isfinite(grid).sum() + np.isnan(grid).sum()
        n_invalid_before = int(np.isnan(grid).sum())
        filled_grid, filled_mask = impute_neighbor_3x3(grid, min_valid, max_iters)

        # 폴백: 여전히 NaN → 트레이 중앙값
        still_nan = np.isnan(filled_grid)
        n_fallback = 0
        if still_nan.any() and fallback:
            med = np.nanmedian(filled_grid)
            filled_grid[still_nan] = med
            n_fallback = int(still_nan.sum())

        # 격자값을 sub 로 되쓰기 (row,col 1-based)
        r = pd.to_numeric(sub["row"], errors="coerce")
        c = pd.to_numeric(sub["col"], errors="coerce")
        ok = r.notna() & c.notna()
        ri = (r[ok].astype(int) - 1).to_numpy()
        ci = (c[ok].astype(int) - 1).to_numpy()
        newvals = filled_grid[ri, ci]
        imp_here = filled_mask[ri, ci]
        fb_here = still_nan[ri, ci]  # 폴백으로 채운 곳
        sub_idx = sub.index[ok.to_numpy()]
        sub.loc[sub_idx, "temp"] = newvals
        sub.loc[sub_idx, "imputed"] = imp_here | fb_here
        # 플래그 갱신
        imp_rows = sub_idx[imp_here]
        fb_rows = sub_idx[fb_here]
        sub.loc[imp_rows, "qc_flag"] = "imputed"
        sub.loc[fb_rows, "qc_flag"] = "imputed_fail"
        out_parts.append(sub)

        denom = max(int(n_cells), 1)
        reports.append({
            "tray_id": tray, "step_name": step,
            "n_invalid": n_invalid_before,
            "n_imputed_3x3": int(filled_mask.sum()),
            "n_fallback_median": n_fallback,
            "impute_frac": n_invalid_before / denom,
        })

    temp_clean = pd.concat(out_parts).sort_index()
    rep = pd.DataFrame(reports)
    # 치환율 과다 필드 표시
    if len(rep):
        rep["excluded_high_impute"] = rep["impute_frac"] > pp["max_impute_frac"]
    return temp_clean, rep
