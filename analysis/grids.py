"""트레이 필드 ↔ 2D 격자 변환 유틸 (여러 모듈 공용)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def infer_shape(df: pd.DataFrame, n_rows_cfg=None, n_cols_cfg=None) -> tuple[int, int]:
    n_rows = n_rows_cfg or int(pd.to_numeric(df["row"], errors="coerce").max())
    n_cols = n_cols_cfg or int(pd.to_numeric(df["col"], errors="coerce").max())
    return n_rows, n_cols


def to_grid(sub: pd.DataFrame, value_col: str, n_rows: int, n_cols: int) -> np.ndarray:
    """(row, col, value) → (n_rows, n_cols) 격자. row/col 은 1-based 가정."""
    g = np.full((n_rows, n_cols), np.nan, dtype=float)
    r = pd.to_numeric(sub["row"], errors="coerce")
    c = pd.to_numeric(sub["col"], errors="coerce")
    v = pd.to_numeric(sub[value_col], errors="coerce")
    ok = r.notna() & c.notna()
    ri = (r[ok].astype(int) - 1).to_numpy()
    ci = (c[ok].astype(int) - 1).to_numpy()
    vv = v[ok].to_numpy()
    inb = (ri >= 0) & (ri < n_rows) & (ci >= 0) & (ci < n_cols)
    g[ri[inb], ci[inb]] = vv[inb]
    return g


def grid_to_series(grid: np.ndarray) -> pd.DataFrame:
    """격자 → long (row, col, value), 1-based."""
    rows, cols = np.indices(grid.shape)
    return pd.DataFrame({
        "row": rows.ravel() + 1,
        "col": cols.ravel() + 1,
        "value": grid.ravel(),
    })


def border_mask(n_rows: int, n_cols: int) -> np.ndarray:
    """테두리=True 격자 (링 점수 계산용)."""
    m = np.zeros((n_rows, n_cols), dtype=bool)
    m[0, :] = m[-1, :] = m[:, 0] = m[:, -1] = True
    return m
