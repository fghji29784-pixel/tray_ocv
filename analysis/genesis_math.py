"""[S8-math] 구배 발생 지점의 수학적 규명.

각 트레이에서 단계 k 의 ΔOCV 필드를 벡터 fₖ∈ℝⁿ, 최종 판정구배 d=Δdocv7 로 두고:
  a_k = corr(fₖ, d)              누적 정렬도 (패턴 존재)
  β_k = <gₖ,d>/‖d‖², gₖ=fₖ−fₖ₋₁ 증분 투영 (패턴 생성 공정)
  I_k = Moran's I(fₖ)            공간 구조 발생
  k*  = a_k 수열의 변화점        통계적 발생 단계
공통 유효셀에 정렬해 트레이별로 계산 후 집계.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .grids import to_grid, border_mask
from .fields import ring_score


# ---------------------------------------------------------------------------
# 트레이별 단계 행렬 추출 (공통 유효셀 정렬)
# ---------------------------------------------------------------------------
def _tray_matrix(sub: pd.DataFrame, stage_dcols: list[str], target_dcol: str,
                 n_rows: int, n_cols: int, min_cells: int = 20):
    """한 트레이 → (F: K×m, d: m, valid_grid_index). 공통 유효셀만."""
    grids = [to_grid(sub, c, n_rows, n_cols).ravel() for c in stage_dcols]
    d = to_grid(sub, target_dcol, n_rows, n_cols).ravel()
    F = np.vstack(grids) if grids else np.empty((0, n_rows * n_cols))
    valid = np.isfinite(d) & np.all(np.isfinite(F), axis=0)
    if valid.sum() < min_cells:
        return None
    return F[:, valid], d[valid], valid


def _corr(x, y):
    x = x - x.mean(); y = y - y.mean()
    dx, dy = np.sqrt((x * x).sum()), np.sqrt((y * y).sum())
    return float((x * y).sum() / (dx * dy)) if dx > 1e-12 and dy > 1e-12 else np.nan


def alignment_and_projection(cell: pd.DataFrame, stage_meta: list[dict],
                             n_rows: int, n_cols: int,
                             target_dcol: str = "d_docv7") -> tuple[pd.DataFrame, pd.DataFrame]:
    """a_k(단계별 누적 정렬), β_k(공정별 증분 투영) 표 반환."""
    stages = [m for m in sorted(stage_meta, key=lambda x: x.get("order", 0))
              if f"d_ocvstage::{m['label']}" in cell.columns]
    labels = [m["label"] for m in stages]
    soc = {m["label"]: m.get("soc") for m in stages}
    dcols = [f"d_ocvstage::{lab}" for lab in labels]

    a_acc = {lab: [] for lab in labels}
    b_acc = {lab: [] for lab in labels[1:]}  # 증분은 2번째 단계부터
    for _, sub in cell.groupby("tray_id", sort=False):
        got = _tray_matrix(sub, dcols, target_dcol, n_rows, n_cols)
        if got is None:
            continue
        F, d, _ = got
        dd = float((d * d).sum())
        if dd < 1e-18:
            continue
        for i, lab in enumerate(labels):
            a_acc[lab].append(_corr(F[i], d))
        for i in range(1, len(labels)):
            g = F[i] - F[i - 1]
            b_acc[labels[i]].append(float((g * d).sum()) / dd)

    def agg(vals):
        v = np.array([x for x in vals if np.isfinite(x)])
        if v.size == 0:
            return dict(median=np.nan, sign=np.nan, n=0)
        med = float(np.median(v))
        sign = float(np.mean(np.sign(v) == np.sign(med))) if med != 0 else 0.5
        return dict(median=med, sign=sign, n=int(v.size))

    a_rows = [{"stage": lab, "order": i, "soc": soc[lab],
               **{f"a_{k}": v for k, v in agg(a_acc[lab]).items()}}
              for i, lab in enumerate(labels)]
    align = pd.DataFrame(a_rows)

    from .genesis import PROCESS_BEFORE_STAGE
    b_rows = []
    for i in range(1, len(labels)):
        prev, cur = labels[i - 1], labels[i]
        st = agg(b_acc[cur])
        b_rows.append({
            "process": PROCESS_BEFORE_STAGE.get(cur, "?"),
            "from_stage": prev, "to_stage": cur,
            "matched_soc": soc[prev] == soc[cur] and soc[cur] is not None,
            "composes_docv7": prev.startswith("PRIVT") and cur.startswith("PRIVT"),
            "beta_median": st["median"], "beta_sign": st["sign"], "n": st["n"],
        })
    beta = pd.DataFrame(b_rows)
    return align, beta


# ---------------------------------------------------------------------------
# Moran's I (rook 인접, NaN 마스크)
# ---------------------------------------------------------------------------
def morans_i(grid: np.ndarray) -> float:
    """격자 공간 자기상관. 이웃(상하좌우) 유사도. 무작위면 ≈ −1/(n−1)."""
    x = grid.astype(float)
    fin = np.isfinite(x)
    n = int(fin.sum())
    if n < 8:
        return np.nan
    z = np.where(fin, x - np.nanmean(x), 0.0)
    num = 0.0
    W = 0.0
    for sh, ax in ((1, 0), (-1, 0), (1, 1), (-1, 1)):
        zs = np.roll(z, sh, axis=ax)
        fs = np.roll(fin, sh, axis=ax)
        # roll 은 순환 → 경계 wrap 제거
        if ax == 0:
            if sh == 1: fs[0, :] = False
            else: fs[-1, :] = False
        else:
            if sh == 1: fs[:, 0] = False
            else: fs[:, -1] = False
        pair = fin & fs
        num += float((z[pair] * zs[pair]).sum())
        W += float(pair.sum())
    denom = float((z[fin] ** 2).sum())
    if W < 1 or denom < 1e-18:
        return np.nan
    return (n / W) * (num / denom)


def morans_progression(cell: pd.DataFrame, stage_meta: list[dict],
                       n_rows: int, n_cols: int, min_frac: float = 0.3) -> pd.DataFrame:
    """단계별 Moran's I 트레이 분포 + E[I]=−1/(n−1) 대비."""
    stages = [m for m in sorted(stage_meta, key=lambda x: x.get("order", 0))
              if f"d_ocvstage::{m['label']}" in cell.columns]
    rows = []
    for m in stages:
        dcol = f"d_ocvstage::{m['label']}"
        Is = []
        for _, sub in cell.groupby("tray_id", sort=False):
            g = to_grid(sub, dcol, n_rows, n_cols)
            if np.isfinite(g).sum() >= min_frac * n_rows * n_cols:
                Is.append(morans_i(g))
        Is = np.array([v for v in Is if np.isfinite(v)])
        n_eff = n_rows * n_cols
        e_i = -1.0 / (n_eff - 1)
        rows.append({
            "stage": m["label"], "order": m.get("order"), "soc": m.get("soc"),
            "moran_median": float(np.median(Is)) if Is.size else np.nan,
            "moran_E": e_i,
            "frac_structured": float(np.mean(Is > 0.1)) if Is.size else np.nan,
            "n_tray": int(Is.size),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 변화점 (a_k 수열의 계단 상승)
# ---------------------------------------------------------------------------
def _cp_array(y: np.ndarray) -> int | None:
    """수열에서 평균이 가장 크게 도약하는 지점 index (1..K-1)."""
    if np.isfinite(y).sum() < 3:
        return None
    best_k, best_gap = None, -np.inf
    for k in range(1, len(y)):
        b, a = y[:k], y[k:]
        if np.isfinite(b).any() and np.isfinite(a).any():
            gap = np.nanmean(a) - np.nanmean(b)
            if gap > best_gap:
                best_gap, best_k = gap, k
    return best_k


def per_tray_genesis(cell: pd.DataFrame, stage_meta: list[dict], n_rows: int, n_cols: int,
                     target_dcol: str = "d_docv7", min_ring: float | None = None) -> dict:
    """트레이별로 docv7 패턴의 발생 단계·유발 공정을 찾아 '분포'로 집계.

    부호 있는 상관을 트레이 간 평균하면 링/역링이 상쇄되므로, 트레이마다 개별로
    발생단계(|a_k| 변화점)·유발공정(matched-SOC 중 |증분투영| 최대)을 구해 히스토그램.
    구조가 약한(플랫) 트레이는 제외(min_ring).
    """
    stages = [m for m in sorted(stage_meta, key=lambda x: x.get("order", 0))
              if f"d_ocvstage::{m['label']}" in cell.columns]
    labels = [m["label"] for m in stages]
    soc = {m["label"]: m.get("soc") for m in stages}
    dcols = [f"d_ocvstage::{lab}" for lab in labels]
    from .genesis import PROCESS_BEFORE_STAGE

    # docv7 링 진폭으로 강한 트레이 선별 임계
    rings = {}
    for tray, sub in cell.groupby("tray_id", sort=False):
        g = to_grid(sub, target_dcol, n_rows, n_cols)
        rings[tray] = abs(ring_score(g)) if np.isfinite(g).sum() >= 0.3 * n_rows * n_cols else np.nan
    rv = np.array([v for v in rings.values() if np.isfinite(v)])
    thr = min_ring if min_ring is not None else (np.nanmedian(rv) if rv.size else 0.0)

    onset_counts = {lab: 0 for lab in labels}
    culprit_counts = {}
    per_rows = []
    n_used = 0
    for tray, sub in cell.groupby("tray_id", sort=False):
        if not np.isfinite(rings[tray]) or rings[tray] < thr:
            continue
        got = _tray_matrix(sub, dcols, target_dcol, n_rows, n_cols)
        if got is None:
            continue
        F, d, _ = got
        dd = float((d * d).sum())
        if dd < 1e-18:
            continue
        a = np.array([_corr(F[i], d) for i in range(len(labels))])
        k = _cp_array(np.abs(a))
        if k is None:
            continue
        onset = labels[k]
        onset_counts[onset] += 1
        # 유발 공정: matched-SOC & non-composes 증분 투영 최대
        best_proc, best_val = None, -np.inf
        for i in range(1, len(labels)):
            prev, cur = labels[i - 1], labels[i]
            matched = soc[prev] == soc[cur] and soc[cur] is not None
            composes = prev.startswith("PRIVT") and cur.startswith("PRIVT")
            if not matched or composes:
                continue
            g = F[i] - F[i - 1]
            val = abs(float((g * d).sum()) / dd)
            if val > best_val:
                best_val, best_proc = val, PROCESS_BEFORE_STAGE.get(cur, cur)
        if best_proc:
            culprit_counts[best_proc] = culprit_counts.get(best_proc, 0) + 1
        per_rows.append({"tray_id": tray, "onset_stage": onset, "onset_soc": soc[onset],
                         "culprit_process": best_proc, "docv7_ring": rings[tray]})
        n_used += 1

    onset_df = pd.DataFrame(
        [{"stage": lab, "soc": soc[lab], "n_trays_onset": onset_counts[lab],
          "frac": onset_counts[lab] / n_used if n_used else np.nan}
         for lab in labels])
    culprit_df = pd.DataFrame(
        [{"process": p, "n_trays": c, "frac": c / n_used if n_used else np.nan}
         for p, c in sorted(culprit_counts.items(), key=lambda x: -x[1])])
    return {"onset_dist": onset_df, "culprit_dist": culprit_df,
            "per_tray": pd.DataFrame(per_rows), "n_used": n_used, "ring_thr": float(thr)}


def changepoint(align: pd.DataFrame, col: str = "a_median") -> dict:
    """|a_k| 수열에서 평균이 가장 크게 도약하는 지점 k* (단순 최대차 분할)."""
    s = align.sort_values("order")
    y = np.abs(pd.to_numeric(s[col], errors="coerce").to_numpy())
    labels = s["stage"].to_list()
    if np.isfinite(y).sum() < 3:
        return {"ok": False}
    best_k, best_gap = None, -np.inf
    for k in range(1, len(y)):
        before, after = y[:k], y[k:]
        if not (np.isfinite(before).any() and np.isfinite(after).any()):
            continue
        gap = np.nanmean(after) - np.nanmean(before)
        if gap > best_gap:
            best_gap, best_k = gap, k
    if best_k is None:
        return {"ok": False}
    return {"ok": True, "k_index": int(best_k), "stage": labels[best_k],
            "gap": float(best_gap)}
