"""[S3] 트레이 필드 구성 + Δ필드 + 구조 지표.

Δ필드: ΔX = X - median_tray(X)  (트레이 간 레벨 제거, 트레이 내 구배만 남김)
구조 지표(온도·전압 공용): 진폭, 링 점수, 방향 구배(행/열), 줄무늬 점수.
기존 [27b] 갤러리와 동일 정의.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .grids import border_mask


# --- 격자 단위 구조 지표 ---------------------------------------------------
def amplitude_p2p(grid: np.ndarray) -> float:
    v = grid[np.isfinite(grid)]
    return float(np.nanmax(v) - np.nanmin(v)) if v.size else np.nan


def amplitude_p95p5(grid: np.ndarray) -> float:
    v = grid[np.isfinite(grid)]
    if v.size < 3:
        return np.nan
    return float(np.percentile(v, 95) - np.percentile(v, 5))


def ring_score(grid: np.ndarray) -> float:
    """테두리 평균 - 중앙 평균. 양수 = 링형(테두리 高)."""
    bm = border_mask(*grid.shape)
    b = grid[bm & np.isfinite(grid)]
    c = grid[(~bm) & np.isfinite(grid)]
    if b.size == 0 or c.size == 0:
        return np.nan
    return float(np.nanmean(b) - np.nanmean(c))


def _grad(grid: np.ndarray, axis: int) -> float:
    """행(axis=0) 또는 열(axis=1) 좌표에 대한 선형 회귀 기울기."""
    idx = np.indices(grid.shape)[axis].astype(float)
    fin = np.isfinite(grid)
    if fin.sum() < 3:
        return np.nan
    x = idx[fin]
    y = grid[fin]
    x = x - x.mean()
    denom = float((x * x).sum())
    if denom == 0:
        return np.nan
    return float((x * (y - y.mean())).sum() / denom)


def grad_row(grid: np.ndarray) -> float:
    return _grad(grid, 0)


def grad_col(grid: np.ndarray) -> float:
    return _grad(grid, 1)


def stripe_score(grid: np.ndarray) -> float:
    """열별 평균의 분산 (세로 줄무늬 강도)."""
    col_means = np.nanmean(grid, axis=0)
    col_means = col_means[np.isfinite(col_means)]
    return float(np.nanvar(col_means)) if col_means.size > 1 else np.nan


STRUCTURE_FUNCS = {
    "amp_p2p": amplitude_p2p,
    "amp_p95p5": amplitude_p95p5,
    "ring": ring_score,
    "grad_row": grad_row,
    "grad_col": grad_col,
    "stripe": stripe_score,
}


def add_tray_delta(df: pd.DataFrame, value_col: str, by=("tray_id", "step_name"),
                   out_col: str | None = None) -> pd.DataFrame:
    """트레이(+스텝) 중앙값을 뺀 Δ 컬럼 추가."""
    out_col = out_col or f"d_{value_col}"
    med = df.groupby(list(by))[value_col].transform("median")
    df[out_col] = df[value_col] - med
    return df


def structure_table(df: pd.DataFrame, value_col: str, n_rows: int, n_cols: int,
                     by=("tray_id", "step_name")) -> pd.DataFrame:
    """(tray[, step]) 별 구조 지표 표."""
    from .grids import to_grid
    rows = []
    for keys, sub in df.groupby(list(by), sort=False):
        grid = to_grid(sub, value_col, n_rows, n_cols)
        rec = dict(zip(by, keys if isinstance(keys, tuple) else (keys,)))
        for name, fn in STRUCTURE_FUNCS.items():
            rec[name] = fn(grid)
        rec["n_valid"] = int(np.isfinite(grid).sum())
        rows.append(rec)
    return pd.DataFrame(rows)
