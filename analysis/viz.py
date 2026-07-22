"""시각화: 트레이 필드 갤러리 + 상관 히트맵 + 보정 전후 비교."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .grids import to_grid


def plot_field_gallery(df: pd.DataFrame, value_col: str, n_rows: int, n_cols: int,
                       out_path, by="tray_id", max_trays=24, title="", cmap="RdBu_r"):
    trays = list(df[by].dropna().unique())[:max_trays]
    if not trays:
        return None
    ncols = min(6, len(trays))
    nrows = int(np.ceil(len(trays) / ncols))
    vmax = np.nanpercentile(np.abs(pd.to_numeric(df[value_col], errors="coerce")), 95)
    vmax = vmax if np.isfinite(vmax) and vmax > 0 else 1.0
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.2 * ncols, 2.2 * nrows), squeeze=False)
    for i, tray in enumerate(trays):
        ax = axes[i // ncols][i % ncols]
        grid = to_grid(df[df[by] == tray], value_col, n_rows, n_cols)
        ax.imshow(grid, cmap=cmap, vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_title(str(tray), fontsize=7)
        ax.set_xticks([]); ax.set_yticks([])
    for j in range(len(trays), nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")
    fig.suptitle(title or value_col, fontsize=11)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path


def plot_stage_progression(prog: pd.DataFrame, out_path, title=""):
    """단계별 링 진폭 + docv7 필드 상관을 공정 순서로 (구배 발생 추적)."""
    if prog.empty:
        return None
    x = range(len(prog))
    fig, ax1 = plt.subplots(figsize=(max(7, 0.7 * len(prog)), 4.2))
    ax1.plot(x, prog["ring_abs_median"], "o-", color="#c0392b", label="ring |amplitude| median")
    ax1.set_ylabel("dOCV ring |amplitude| (median)", color="#c0392b")
    ax1.tick_params(axis="y", labelcolor="#c0392b")
    ax1.set_xticks(list(x))
    labels = [f"{r.stage}\n(SOC{int(r.soc)})" if pd.notna(r.soc) else r.stage
              for r in prog.itertuples()]
    ax1.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax2 = ax1.twinx()
    ax2.plot(x, prog["corr_docv7_median"], "s--", color="#2c3e50",
             label="corr with final d_docv7")
    ax2.set_ylabel("corr with final d_docv7 field", color="#2c3e50")
    ax2.tick_params(axis="y", labelcolor="#2c3e50")
    ax2.axhline(0, color="gray", lw=0.5)
    ax1.set_title(title or "OCV stage — gradient genesis tracking")
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def plot_corr_heatmap(screen_df: pd.DataFrame, out_path, value="median", title=""):
    """스텝×타깃 상관 히트맵 (screen 결과 pivot)."""
    if screen_df.empty:
        return None
    piv = screen_df.pivot(index="predictor", columns="target", values=value)
    fig, ax = plt.subplots(figsize=(1.6 * (piv.shape[1] + 2), 0.35 * piv.shape[0] + 2))
    vmax = np.nanmax(np.abs(piv.to_numpy())) or 1.0
    im = ax.imshow(piv.to_numpy(), cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(piv.shape[1])); ax.set_xticklabels(piv.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(piv.shape[0])); ax.set_yticklabels(piv.index, fontsize=7)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.iat[i, j]
            if pd.notna(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6)
    fig.colorbar(im, ax=ax, shrink=0.7, label=f"per-tray corr ({value})")
    ax.set_title(title or "step × target correlation")
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path
