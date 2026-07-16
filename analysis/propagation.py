"""[S6] 경로 판별 회귀 (P1/P2/P3) + 전파 사슬 + 홀드아웃 검증.

ΔdOCV7 ~ β1·(측정시점 온도차 프록시) + β2·(에이징 자기방전 프록시)
        + β3·(HT 이월 프록시) + 트레이 효과
Δ필드는 이미 트레이 중앙값 제거 상태 → 트레이 고정효과는 근사적으로 반영됨.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

try:
    import statsmodels.api as sm
    _HAS_SM = True
except Exception:  # pragma: no cover
    _HAS_SM = False


def fit_paths(cell_tbl: pd.DataFrame, target: str, predictors: list[str],
              exclude_imputed: bool = False) -> dict:
    """OLS 적합 + 표준화 계수/기여도. 반환 dict."""
    df = cell_tbl.copy()
    if exclude_imputed and "any_imputed" in df.columns:
        df = df[~df["any_imputed"].astype(bool)]
    use = [p for p in predictors if p in df.columns]
    cols = [target] + use
    d = df[cols].apply(pd.to_numeric, errors="coerce").dropna()
    if len(d) < max(20, 3 * len(use)) or not use:
        return {"ok": False, "reason": "표본/예측자 부족", "n": len(d)}

    y = d[target].to_numpy()
    X = d[use].to_numpy()
    # 표준화 (기여도 비교용)
    Xz = (X - X.mean(0)) / (X.std(0) + 1e-12)
    yz = (y - y.mean()) / (y.std() + 1e-12)

    if _HAS_SM:
        Xd = sm.add_constant(Xz)
        model = sm.OLS(yz, Xd).fit()
        coefs = dict(zip(["const"] + use, model.params))
        pvals = dict(zip(["const"] + use, model.pvalues))
        r2 = float(model.rsquared)
    else:  # 최소제곱 폴백
        Xd = np.column_stack([np.ones(len(yz)), Xz])
        beta, *_ = np.linalg.lstsq(Xd, yz, rcond=None)
        coefs = dict(zip(["const"] + use, beta))
        pvals = {k: np.nan for k in ["const"] + use}
        resid = yz - Xd @ beta
        r2 = 1 - float(resid.var() / (yz.var() + 1e-12))

    # 각 예측자 기여 (표준화 계수² 비율 근사)
    contrib = {k: float(coefs[k] ** 2) for k in use}
    tot = sum(contrib.values()) or 1.0
    contrib_frac = {k: v / tot for k, v in contrib.items()}

    return {"ok": True, "n": len(d), "r2": r2, "std_coef": coefs,
            "pvalues": pvals, "contrib_frac": contrib_frac, "predictors": use}


def ocv_progression(cell_tbl: pd.DataFrame, step_predictor: str) -> pd.DataFrame:
    """ΔOCV1→2→3 각각에 대한 스텝 온도 상관 (P2 누적 vs P3 조기완성 판별)."""
    from .screening import per_tray_corr_values, aggregate
    rows = []
    for tgt in ["d_ocv1", "d_ocv2", "d_ocv3", "d_docv7"]:
        if tgt not in cell_tbl.columns or step_predictor not in cell_tbl.columns:
            continue
        agg = aggregate(per_tray_corr_values(cell_tbl, step_predictor, tgt))
        rows.append({"target": tgt, **agg})
    return pd.DataFrame(rows)


def holdout_validate(cell_tbl: pd.DataFrame, target: str, predictors: list[str],
                     seed: int = 0, test_frac: float = 0.2) -> dict:
    """트레이 8:2 분할 → train 계수로 test 예측, 트레이별 공간 상관."""
    use = [p for p in predictors if p in cell_tbl.columns]
    if not use:
        return {"ok": False, "reason": "예측자 없음"}
    trays = np.asarray(cell_tbl["tray_id"].dropna().unique(), dtype=object)
    rng = np.random.default_rng(seed)
    rng.shuffle(trays)
    n_test = max(1, int(len(trays) * test_frac))
    test_trays = set(trays[:n_test])
    train = cell_tbl[~cell_tbl["tray_id"].isin(test_trays)]
    test = cell_tbl[cell_tbl["tray_id"].isin(test_trays)]

    d = train[[target] + use].apply(pd.to_numeric, errors="coerce").dropna()
    if len(d) < max(20, 3 * len(use)):
        return {"ok": False, "reason": "train 표본 부족"}
    X = np.column_stack([np.ones(len(d)), d[use].to_numpy()])
    beta, *_ = np.linalg.lstsq(X, d[target].to_numpy(), rcond=None)

    from .screening import per_tray_corr_values, aggregate
    tt = test.copy()
    Xte = tt[use].apply(pd.to_numeric, errors="coerce")
    tt["_pred"] = beta[0] + Xte.to_numpy() @ beta[1:]
    corrs = per_tray_corr_values(tt.dropna(subset=["_pred", target]), "_pred", target)
    return {"ok": True, "n_test_trays": len(test_trays),
            "pred_vs_actual": aggregate(corrs), "beta": dict(zip(["const"] + use, beta))}
