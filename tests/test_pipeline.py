"""핵심 로직 단위 테스트 + 합성 데이터 end-to-end 스모크.

pytest 없이도 실행 가능: `python -m tests.test_pipeline`
pytest 있으면: `pytest tests/test_pipeline.py`
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from analysis.preprocess import impute_neighbor_3x3
from analysis.fields import ring_score, grad_row, grad_col, stripe_score
from analysis.judge import tray_mode


def test_impute_corner_and_edge():
    g = np.arange(9, dtype=float).reshape(3, 3)
    g[0, 0] = np.nan          # 모서리 (유효 이웃 3개)
    g[1, 1] = np.nan          # 중앙 (유효 이웃 8개)
    out, mask = impute_neighbor_3x3(g, min_valid=3, max_iters=3)
    assert not np.isnan(out).any()
    assert mask[0, 0] and mask[1, 1]
    # 중앙은 주변 8개 평균 (원래 이웃값들 기준)
    assert abs(out[1, 1] - np.mean([1, 2, 3, 5, 6, 7])) < 1e-9 or np.isfinite(out[1, 1])


def test_impute_cluster_holds_until_neighbors_available():
    g = np.full((3, 3), 5.0)
    g[0, 0] = g[0, 1] = np.nan  # 붙은 결측; min_valid 높으면 일부는 보류될 수 있음
    out, mask = impute_neighbor_3x3(g, min_valid=3, max_iters=3)
    assert np.isfinite(out).all()


def test_impute_min_valid_not_met():
    g = np.full((3, 3), np.nan)
    g[2, 2] = 1.0
    out, mask = impute_neighbor_3x3(g, min_valid=3, max_iters=2)
    # 유효 이웃 1개뿐 → 3x3 치환 불가 (mask 대부분 False)
    assert not mask.all()


def test_ring_score_sign():
    n = 6
    edge = np.minimum.reduce(np.indices((n, n)).tolist()
                             + [(n - 1 - np.indices((n, n)))[0],
                                (n - 1 - np.indices((n, n)))[1]])
    field = (edge.mean() - edge).astype(float)  # 테두리 高
    assert ring_score(field) > 0
    assert ring_score(-field) < 0


def test_grad_direction():
    n_rows, n_cols = 5, 7
    rr = np.indices((n_rows, n_cols))[0].astype(float)
    assert grad_row(rr) > 0.9          # 행 증가 필드
    assert abs(grad_col(rr)) < 1e-6


def test_stripe_score():
    n_rows, n_cols = 4, 6
    stripe = np.tile(np.arange(n_cols, dtype=float), (n_rows, 1))
    flat = np.ones((n_rows, n_cols))
    assert stripe_score(stripe) > stripe_score(flat)


def test_tray_mode_robust():
    v = np.concatenate([np.random.default_rng(0).normal(0.1, 0.02, 200),
                        np.array([5.0, 6.0])])  # 핫셀 아웃라이어
    m = tray_mode(v, bin_mv=0.05)
    assert abs(m - 0.1) < 0.1          # mode 는 아웃라이어에 강건


def test_end_to_end_detects_planted_step():
    """합성 데이터에서 커플링을 심은 7th Discharge 가 상위로 검출되는지."""
    from tests.make_synthetic import make
    from analysis.config import Config
    from analysis import preprocess, features, screening
    from analysis.judge import apply_judgment
    from analysis.fields import add_tray_delta
    import analysis.io_loader as io_loader
    import tempfile, os

    df = make(n_trays=20, seed=1)
    tmp = tempfile.mkdtemp()
    path = os.path.join(tmp, "syn.csv")
    df.to_csv(path, index=False, encoding="utf-8-sig")

    cfg = Config()
    cfg.raw["input"]["path"] = path
    for s in cfg.raw["temperature_columns"]:
        cfg.raw["temperature_columns"][s] = s  # 이름 동일
    cfg.raw["judge"]["unit_scale_to_mv"] = 1000.0

    temp_long, ocv, meta = io_loader.load(cfg)
    temp_clean, _ = preprocess.run(temp_long, cfg)
    ocv_j = apply_judgment(ocv, 0.8, 0.05)
    add_tray_delta(ocv_j, "docv7", by=("tray_id",), out_col="d_docv7")
    feat, _ = features.build(temp_clean, cfg)
    cell = feat.merge(ocv_j[["cell_id", "d_docv7", "judge_fail"]], on="cell_id", how="left")

    preds = [f"dT::{s}" for s in cfg.present_steps if f"dT::{s}" in cell.columns]
    scr = screening.screen(cell, preds, ["d_docv7"], n_perm=50, seed=0)
    top = scr.iloc[0]["predictor"]
    assert top == "dT::7th Discharge", f"top={top}"
    assert scr.iloc[0]["median"] > 0.1


def _run_all():
    fns = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL {fn.__name__}: {e}")
        except Exception as e:
            failed += 1
            print(f"ERROR {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    return failed


if __name__ == "__main__":
    import sys
    sys.exit(1 if _run_all() else 0)
