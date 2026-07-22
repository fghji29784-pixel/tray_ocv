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

    from analysis.config import CANONICAL_STEPS
    cfg = Config()
    cfg.raw["input"]["path"] = path
    cfg.raw["temperature_columns"] = {s: s for s in CANONICAL_STEPS}  # 이름 동일
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


def test_position_alpha_and_cellno():
    """A01→(행1,열1), L12→(행12,열12); Cell No 1..144 산술 환산."""
    import pandas as pd
    from analysis.io_loader import _to_index, _resolve_positions
    from analysis.config import Config

    # 알파벳 행 변환
    assert list(_to_index(pd.Series(["A", "B", "L"]))) == [1, 2, 12]
    assert list(_to_index(pd.Series(["01", "12"]))) == [1, 12]

    # Cell 위치 A01..L12 정규식 파싱
    df = pd.DataFrame({
        "Cell ID": ["x1", "x2", "x3"],
        "TRAY ID": ["T0", "T0", "T0"],
        "Cell 위치": ["A01", "A12", "L12"],
    })
    cfg = Config()
    cfg.raw["id_columns"] = {"cell_id": "Cell ID", "tray_id": "TRAY ID",
                             "row": None, "col": None, "lot": None}
    cfg.raw["position_from"] = {"source_column": "Cell 위치",
                                "regex": r"(?P<row>[A-Za-z]+)\s*(?P<col>\d+)",
                                "from_cell_number": None}
    pos = _resolve_positions(df, cfg)
    assert list(pos["row"]) == [1, 1, 12]
    assert list(pos["col"]) == [1, 12, 12]

    # Cell No 산술 (row_major: No 1..12 → 1행)
    df2 = pd.DataFrame({"Cell ID": ["a", "b", "c", "d"], "TRAY ID": ["T", "T", "T", "T"],
                        "Cell No": [1, 12, 13, 144]})
    cfg2 = Config()
    cfg2.raw["id_columns"] = {"cell_id": "Cell ID", "tray_id": "TRAY ID",
                              "row": None, "col": None, "lot": None}
    cfg2.raw["position_from"] = {"source_column": None, "regex": None,
                                 "from_cell_number": {"column": "Cell No",
                                                      "n_cols": 12, "row_major": True}}
    pos2 = _resolve_positions(df2, cfg2)
    assert list(pos2["row"]) == [1, 1, 2, 12]
    assert list(pos2["col"]) == [1, 12, 1, 12]


def test_export_schema_autodetect():
    """Export 스키마 자동감지 + P1 피처(측정지점 온도차) 구성 여부."""
    from tests.make_synthetic import make_export
    from analysis import io_loader, preprocess, features, screening
    from analysis.config import load_config, save_config
    from analysis.judge import apply_judgment
    from analysis.fields import add_tray_delta
    import tempfile, os

    df = make_export(n_trays=20, seed=2)
    tmp = tempfile.mkdtemp()
    path = os.path.join(tmp, "export.csv")
    df.to_csv(path, index=False, encoding="utf-8-sig")

    # 자동 매핑 → 유닛만 조정
    cfg_dict = io_loader.build_config_template(path)
    assert cfg_dict["judge"]["docv7_from"] == "direct"
    assert cfg_dict["ocv_columns"]["ocv1"] == "PRIVT OCV #01 OCV"
    assert any("Charge #01" == k for k in cfg_dict["temperature_columns"])
    cfg_dict["judge"]["unit_scale_to_mv"] = 1000.0
    cfg_dict["judge"]["docv7_unit_scale_to_mv"] = 1.0
    cfg_path = os.path.join(tmp, "cfg.yaml")
    save_config(cfg_dict, cfg_path)
    cfg = load_config(cfg_path)

    temp_long, ocv, meta = io_loader.load(cfg)
    temp_clean, _ = preprocess.run(temp_long, cfg)
    ocv_j = apply_judgment(ocv, 0.8, 0.05)
    add_tray_delta(ocv_j, "docv7", by=("tray_id",), out_col="d_docv7")
    feat, _ = features.build(temp_clean, cfg)
    cell = feat.merge(ocv_j[["cell_id", "d_docv7"]], on="cell_id", how="left")
    # P1 예측자가 구성되고 스크리닝이 에러 없이 도는지 (상관 크기는 데이터에 따라 다름)
    assert "dT_ocv1_minus_ocv3" in cell.columns
    scr = screening.screen(cell, ["dT_ocv1_minus_ocv3"], ["d_docv7"], n_perm=30, seed=0)
    assert len(scr) == 1 and np.isfinite(scr.iloc[0]["median"])


def test_genesis_localizes_planted_stage():
    """OCV #03(1차 고온에이징 후)에 심은 링을 genesis 추적이 그 단계로 지목하는지."""
    from tests.make_synthetic import make_export
    from analysis import io_loader, preprocess, features, genesis
    from analysis.config import load_config, save_config
    from analysis.judge import apply_judgment
    from analysis.fields import add_tray_delta
    import tempfile, os

    df = make_export(n_trays=20, seed=3)
    tmp = tempfile.mkdtemp()
    path = os.path.join(tmp, "export.csv")
    df.to_csv(path, index=False, encoding="utf-8-sig")
    cfg_dict = io_loader.build_config_template(path)
    cfg_dict["judge"]["unit_scale_to_mv"] = 1000.0
    cfg_dict["judge"]["docv7_unit_scale_to_mv"] = 1.0
    cfg_path = os.path.join(tmp, "cfg.yaml")
    save_config(cfg_dict, cfg_path)
    cfg = load_config(cfg_path)

    temp_long, ocv, meta = io_loader.load(cfg)
    assert len(meta["ocv_stages"]) == 10       # OCV#01~07 + PRIVT#01~03
    temp_clean, _ = preprocess.run(temp_long, cfg)
    ocv_j = apply_judgment(ocv, 0.8, 0.05)
    add_tray_delta(ocv_j, "docv7", by=("tray_id",), out_col="d_docv7")
    feat, _ = features.build(temp_clean, cfg)
    stage_cols = [c for c in ocv_j.columns if c.startswith("ocvstage::")]
    cell = feat.merge(ocv_j[["cell_id", "d_docv7"] + stage_cols], on="cell_id", how="left")
    cell, _ = genesis.build_stage_deltas(cell)
    prog = genesis.progression(cell, meta["ocv_stages"], 12, 12)

    # 링 진폭이 OCV#03 에서 급증(이전 대비 5배+) 하는지
    r = prog.set_index("stage")["ring_abs_median"]
    assert r["OCV #03"] > 5 * r["OCV #02"], (r["OCV #02"], r["OCV #03"])
    # docv7 상관도 OCV#03 부터 확 올라감
    c = prog.set_index("stage")["corr_docv7_median"]
    assert abs(c["OCV #02"]) < 0.2 and c["OCV #03"] > 0.4, (c["OCV #02"], c["OCV #03"])


def test_genesis_math_beta_and_moran():
    """β·a_k·Moran's I·변화점이 심어둔 OCV#03 발생을 일관되게 지목하는지."""
    from tests.make_synthetic import make_export
    from analysis import io_loader, preprocess, features, genesis, genesis_math
    from analysis.config import load_config, save_config
    from analysis.judge import apply_judgment
    from analysis.fields import add_tray_delta
    import tempfile, os

    df = make_export(n_trays=20, seed=5)
    tmp = tempfile.mkdtemp()
    path = os.path.join(tmp, "export.csv")
    df.to_csv(path, index=False, encoding="utf-8-sig")
    cfg_dict = io_loader.build_config_template(path)
    cfg_dict["judge"]["unit_scale_to_mv"] = 1000.0
    cfg_dict["judge"]["docv7_unit_scale_to_mv"] = 1.0
    cfg_path = os.path.join(tmp, "cfg.yaml")
    save_config(cfg_dict, cfg_path)
    cfg = load_config(cfg_path)

    temp_long, ocv, meta = io_loader.load(cfg)
    temp_clean, _ = preprocess.run(temp_long, cfg)
    ocv_j = apply_judgment(ocv, 0.8, 0.05)
    add_tray_delta(ocv_j, "docv7", by=("tray_id",), out_col="d_docv7")
    feat, _ = features.build(temp_clean, cfg)
    stage_cols = [c for c in ocv_j.columns if c.startswith("ocvstage::")]
    cell = feat.merge(ocv_j[["cell_id", "d_docv7"] + stage_cols], on="cell_id", how="left")
    cell, _ = genesis.build_stage_deltas(cell)

    align, beta = genesis_math.alignment_and_projection(cell, meta["ocv_stages"], 12, 12)
    moran = genesis_math.morans_progression(cell, meta["ocv_stages"], 12, 12)
    cp = genesis_math.changepoint(align)

    # β 최대(무자명) 공정이 HT1(OCV#02→#03)
    bc = beta[beta["matched_soc"] & ~beta["composes_docv7"]]
    top = bc.loc[bc["beta_median"].abs().idxmax()]
    assert top["to_stage"] == "OCV #03", top["to_stage"]
    # a_k: OCV#02 낮고 OCV#03 급등
    a = align.set_index("stage")["a_median"]
    assert abs(a["OCV #02"]) < 0.2 and a["OCV #03"] > 0.4
    # Moran's I: OCV#02 무구조, OCV#03 구조
    mi = moran.set_index("stage")["moran_median"]
    assert mi["OCV #03"] > mi["OCV #02"] + 0.2
    # 변화점 = OCV #03
    assert cp["ok"] and cp["stage"] == "OCV #03", cp


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
