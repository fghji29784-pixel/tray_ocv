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
                       out_path, by="tray_id", max_trays=24, title="", cmap="RdBu_r",
                       per_panel=False):
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
        if per_panel:
            grid = _robust_norm(grid)
            ax.imshow(grid, cmap=cmap, vmin=-1, vmax=1, aspect="equal", interpolation="nearest")
        else:
            ax.imshow(grid, cmap=cmap, vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_title(str(tray), fontsize=7)
        ax.set_xticks([]); ax.set_yticks([])
    for j in range(len(trays), nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")
    note = "  (per-panel contrast-normalized)" if per_panel else ""
    fig.suptitle((title or value_col) + note, fontsize=11)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path


def plot_mean_field(cell, stage_labels, n_rows, n_cols, out_path,
                    target_col="d_docv7", title="", cmap="RdBu_r"):
    """단계별 '전 트레이 평균 필드' — 모든 트레이가 공유하는 공통 지문 여부.

    강하면 설비/지그 고정 원인, 미세하면 트레이별(열이력) 원인.
    """
    cand = [f"d_ocvstage::{s}" for s in stage_labels] + [target_col]
    tit = list(stage_labels) + ["docv7 (final)"]
    cols = [(c, t) for c, t in zip(cand, tit) if c in cell.columns]
    if not cols:
        return None
    nc = min(6, len(cols))
    nr = int(np.ceil(len(cols) / nc))
    fig, axes = plt.subplots(nr, nc, figsize=(2.1 * nc, 2.1 * nr), squeeze=False)
    for k, (col, t) in enumerate(cols):
        ax = axes[k // nc][k % nc]
        mean_grid = (cell.groupby(["row", "col"])[col].mean().reset_index()
                     .pipe(lambda d: to_grid(d.rename(columns={col: "v"}), "v", n_rows, n_cols)))
        vmax = np.nanpercentile(np.abs(mean_grid), 98) or 1.0
        im = ax.imshow(mean_grid, cmap=cmap, vmin=-vmax, vmax=vmax, aspect="equal")
        ax.set_title(f"{t}\n(±{vmax:.3g})", fontsize=7)
        ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    for j in range(len(cols), nr * nc):
        axes[j // nc][j % nc].axis("off")
    fig.suptitle(title or "common fingerprint: mean field across all trays (per stage)", fontsize=11)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
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


def _robust_norm(grid: np.ndarray) -> np.ndarray:
    """패널별 강건 정규화: 중앙값 제거 후 P90(|·|)로 나눠 ±1 클립.

    핫셀/죽은셀이 스케일을 잡아먹는 문제 제거 → 각 패널의 '패턴'이 최대 대비로 보임.
    """
    g = grid.astype(float)
    fin = np.isfinite(g)
    if fin.sum() < 4:
        return g
    c = np.nanmedian(g)
    v = g - c
    scale = np.nanpercentile(np.abs(v[fin]), 90)
    if not np.isfinite(scale) or scale < 1e-12:
        return np.where(fin, 0.0, np.nan)
    return np.clip(v / scale, -1, 1)


def plot_storyboard(cell, stage_labels, trays, n_rows, n_cols, out_path,
                    target_col="d_docv7", title="", cmap="RdBu_r", per_panel=True):
    """대표 트레이(행) × 전 단계(열) small-multiples — 링이 어느 칸에서 켜지는지.

    per_panel=True: 패널마다 강건 정규화(핫셀 무시, 각 패널 패턴을 최대 대비로).
    """
    cand = [f"d_ocvstage::{s}" for s in stage_labels] + [target_col]
    tit = list(stage_labels) + ["docv7 (final)"]
    cols_data = [c for c in cand if c in cell.columns]
    col_titles = [t for c, t in zip(cand, tit) if c in cell.columns]
    if not trays or not cols_data:
        return None
    gvmax = 1.0
    if not per_panel:
        gvmax = np.nanpercentile(np.abs(pd.to_numeric(
            cell[cell["tray_id"].isin(trays)][cols_data].stack(), errors="coerce")), 96) or 1.0
    nr, nc = len(trays), len(cols_data)
    fig, axes = plt.subplots(nr, nc, figsize=(1.7 * nc, 1.8 * nr), squeeze=False)
    for i, tray in enumerate(trays):
        sub = cell[cell["tray_id"] == tray]
        for jj, col in enumerate(cols_data):
            ax = axes[i][jj]
            g = to_grid(sub, col, n_rows, n_cols)
            g = _robust_norm(g) if per_panel else g
            ax.imshow(g, cmap=cmap, vmin=-1 if per_panel else -gvmax,
                      vmax=1 if per_panel else gvmax, aspect="equal", interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_linewidth(0.4); s.set_color("#bbbbbb")
            if i == 0:
                ax.set_title(col_titles[jj], fontsize=8, rotation=32, ha="left")
            if jj == 0:
                ax.set_ylabel(str(tray), fontsize=8)
    note = "  (each panel independently contrast-normalized; hot cells clipped)" if per_panel else ""
    fig.suptitle((title or "gradient genesis storyboard (tray × stage)") + note, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def plot_beta_contributions(beta: pd.DataFrame, out_path, title=""):
    """공정별 증분 투영 β 막대 (matched-SOC 강조, composes_docv7 회색)."""
    if beta.empty:
        return None
    b = beta.copy()
    x = range(len(b))
    colors = ["#95a5a6" if r.composes_docv7 else ("#c0392b" if r.matched_soc else "#7fb3d5")
              for r in b.itertuples()]
    fig, ax = plt.subplots(figsize=(max(7, 0.8 * len(b)), 4.2))
    ax.bar(x, b["beta_median"], color=colors)
    ax.axhline(0, color="gray", lw=0.6)
    ax.set_xticks(list(x))
    ax.set_xticklabels([f"{r.process}\n{r.from_stage}→{r.to_stage}" for r in b.itertuples()],
                       rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("β (increment projection onto Δdocv7)")
    ax.set_title(title or "per-process contribution to final gradient (β)")
    # 범례
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color="#c0392b", label="matched-SOC (clean)"),
                       Patch(color="#7fb3d5", label="charge/discharge"),
                       Patch(color="#95a5a6", label="composes docv7 (trivial)")],
              fontsize=7, loc="best")
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def plot_alignment_moran(align: pd.DataFrame, moran: pd.DataFrame, out_path,
                         changepoint_stage=None, title=""):
    """a_k(누적 정렬) + I_k(Moran's I) 진행 + 변화점 세로선."""
    if align.empty:
        return None
    x = range(len(align))
    fig, ax1 = plt.subplots(figsize=(max(7, 0.7 * len(align)), 4.2))
    ax1.plot(x, align["a_median"], "s-", color="#2c3e50", label="a_k = corr(field, Δdocv7)")
    ax1.set_ylabel("a_k  (alignment with final Δdocv7)", color="#2c3e50")
    ax1.axhline(0, color="gray", lw=0.5)
    ax1.set_xticks(list(x))
    labels = [f"{r.stage}\n(SOC{int(r.soc)})" if pd.notna(r.soc) else r.stage
              for r in align.itertuples()]
    ax1.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    if moran is not None and not moran.empty:
        ax2 = ax1.twinx()
        ax2.plot(x, moran["moran_median"].to_numpy()[:len(align)], "o--", color="#c0392b",
                 label="Moran's I")
        ax2.set_ylabel("Moran's I (spatial structure)", color="#c0392b")
    if changepoint_stage is not None and changepoint_stage in list(align["stage"]):
        kx = list(align["stage"]).index(changepoint_stage)
        ax1.axvline(kx, color="green", ls=":", lw=2, label=f"change-point: {changepoint_stage}")
    ax1.legend(fontsize=7, loc="upper left")
    ax1.set_title(title or "gradient onset: alignment a_k + Moran's I")
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def plot_onset_distribution(onset_df, culprit_df, out_path, n_used=None, title=""):
    """트레이별 발생단계·유발공정 히스토그램 (부호 상쇄 없는 집계)."""
    if onset_df is None or onset_df.empty:
        return None
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.4))
    x = range(len(onset_df))
    ax1.bar(x, onset_df["frac"], color="#2c3e50")
    ax1.set_xticks(list(x))
    ax1.set_xticklabels([f"{r.stage}\n(SOC{int(r.soc)})" if pd.notna(r.soc) else r.stage
                         for r in onset_df.itertuples()], rotation=45, ha="right", fontsize=8)
    ax1.set_ylabel("fraction of trays")
    ax1.set_title("per-tray docv7 onset stage" + (f"  (n={n_used})" if n_used else ""))
    if culprit_df is not None and not culprit_df.empty:
        y = range(len(culprit_df))
        ax2.barh(list(y), culprit_df["frac"], color="#c0392b")
        ax2.set_yticks(list(y))
        ax2.set_yticklabels(culprit_df["process"], fontsize=8)
        ax2.invert_yaxis()
        ax2.set_xlabel("fraction of trays")
        ax2.set_title("per-tray culprit process (matched-SOC)")
    fig.suptitle(title or "gradient genesis — per-tray distribution (sign-robust)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
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
