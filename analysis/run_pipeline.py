"""파이프라인 오케스트레이션 CLI.

사용법:
  # 1) 파일 칼럼 확인 (매핑을 고르기 위해)
  python -m analysis.run_pipeline inspect --input data/raw/파일.xlsx [--sheet 0]

  # 2) 칼럼 매핑 초안 자동 생성 → 검토/수정
  python -m analysis.run_pipeline make-config --input data/raw/파일.xlsx \
         --output analysis/column_config.yaml

  # 3) 전체 분석 실행
  python -m analysis.run_pipeline run --config analysis/column_config.yaml
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import correction, features, io_loader, preprocess, propagation, screening, viz
from .config import Config, load_config, save_config
from .fields import add_tray_delta, structure_table
from .grids import infer_shape
from .judge import apply_judgment


# --- 서브커맨드 ------------------------------------------------------------
def cmd_inspect(args):
    cols = io_loader.list_columns(args.input, args.sheet)
    print(f"[inspect] {args.input} — 칼럼 {len(cols)}개:")
    for i, c in enumerate(cols):
        print(f"  [{i:>3}] {c}")


def cmd_make_config(args):
    cfg = io_loader.build_config_template(args.input, args.sheet)
    save_config(cfg, args.output)
    print(f"[make-config] 초안 저장: {args.output}")
    tmapped = sum(1 for v in cfg["temperature_columns"].values() if v)
    print(f"  온도 스텝 매핑: {tmapped}/{len(cfg['temperature_columns'])}")
    print("  ⚠️  null 로 남은 항목과 잘못 매칭된 칼럼을 직접 확인/수정하세요.")
    unresolved = [k for k, v in cfg["temperature_columns"].items() if not v]
    if unresolved:
        print("  미매핑 스텝:", ", ".join(unresolved))


def _build_cell_table(cfg: Config):
    """S1~S4 실행 후 분석용 셀 테이블·부속 산출물 반환."""
    temp_long, ocv_cell, meta = io_loader.load(cfg)
    temp_clean, impute_rep = preprocess.run(temp_long, cfg)

    # 판정
    ocv_j = apply_judgment(ocv_cell, cfg.raw["judge"]["offset_mv"],
                           cfg.raw["judge"]["mode_bin_mv"])
    # OCV Δ필드 (트레이 중앙값 제거)
    for c in ("ocv1", "ocv2", "ocv3", "docv7"):
        add_tray_delta(ocv_j, c, by=("tray_id",), out_col=f"d_{c}")

    # 피처
    feat, long_delta = features.build(temp_clean, cfg)

    # 셀별 치환 플래그
    any_imp = (temp_clean.groupby("cell_id")["imputed"].any()
               .rename("any_imputed").reset_index())

    id_cols = ["lot", "tray_id", "row", "col", "cell_id"]
    ocv_keep = id_cols + ["ocv1", "ocv2", "ocv3", "docv7", "docv7_tray_mode",
                          "docv7_dev", "judge_fail", "d_ocv1", "d_ocv2", "d_ocv3", "d_docv7"]
    cell = feat.merge(ocv_j[ocv_keep], on=id_cols, how="left")
    cell = cell.merge(any_imp, on="cell_id", how="left")
    cell["any_imputed"] = cell["any_imputed"].fillna(False)
    return cell, temp_clean, long_delta, impute_rep, meta


def _predictor_lists(cfg: Config, cell: pd.DataFrame):
    step_preds = [f"dT::{s}" for s in cfg.present_steps if f"dT::{s}" in cell.columns]
    proxy = [c for c in ["dT_charge_mean", "dT_discharge_mean", "dT_preOCV_proxy",
                         "dT_HT_adjacent"] if c in cell.columns]
    heat = [c for c in cell.columns if c.startswith("dHeatExp_ea")]
    return step_preds, proxy, heat


def cmd_run(args):
    cfg = load_config(args.config)
    problems = cfg.validate()
    if problems:
        print("[run] 설정 오류:\n- " + "\n- ".join(problems))
        return
    seed = cfg.raw["run"]["seed"]
    n_perm = args.perm if args.perm is not None else cfg.raw["run"]["n_permutations"]
    outdir = Path(args.outdir)
    (outdir / "figures").mkdir(parents=True, exist_ok=True)
    proc = Path("data/processed"); proc.mkdir(parents=True, exist_ok=True)

    print("[S1-S4] 로딩·전처리·피처 …")
    cell, temp_clean, long_delta, impute_rep, meta = _build_cell_table(cfg)
    n_rows, n_cols = infer_shape(cell, cfg.raw["tray_shape"]["n_rows"],
                                 cfg.raw["tray_shape"]["n_cols"])
    _save_table(cell, proc / "cell_table")
    impute_rep.to_csv(proc / "impute_report.csv", index=False)
    print(f"  셀 {meta['n_cells']}, 트레이 {meta['n_trays']}, 스텝 {meta['n_steps']}, "
          f"격자 {n_rows}x{n_cols}")

    step_preds, proxy, heat = _predictor_lists(cfg, cell)
    predictors = step_preds + proxy + heat
    targets = [t for t in ["d_ocv1", "d_ocv2", "d_ocv3", "d_docv7"] if t in cell.columns]

    print("[S5] 전수 상관 스크리닝 …")
    scr = screening.screen(cell, predictors, targets, n_perm=n_perm, seed=seed,
                           exclude_imputed=False)
    scr_excl = screening.screen(cell, predictors, targets, n_perm=n_perm, seed=seed,
                                exclude_imputed=True)
    scr.to_csv(proc / "screening_all.csv", index=False)
    scr_excl.to_csv(proc / "screening_exclude_imputed.csv", index=False)
    sim = screening.step_similarity(cell, step_preds)
    sim.to_csv(proc / "step_similarity.csv")
    viz.plot_corr_heatmap(scr[scr["predictor"].isin(step_preds)],
                          outdir / "figures" / "corr_heatmap_steps.png",
                          title="step ΔT × OCV target (per-tray median corr)")

    print("[S6] 경로 판별 회귀 …")
    # 열노출은 Ea별로 서로 단조변환(공선성) → 회귀엔 대표 1개만 사용
    heat_one = [heat[len(heat) // 2]] if heat else []
    path_preds = [c for c in ["dT_preOCV_proxy", "dT_HT_adjacent"] + heat_one
                  if c in cell.columns]
    fit = propagation.fit_paths(cell, "d_docv7", path_preds, exclude_imputed=False)
    top_step = scr.iloc[0]["predictor"] if len(scr) else (step_preds[0] if step_preds else None)
    prog = (propagation.ocv_progression(cell, top_step) if top_step is not None
            else pd.DataFrame())
    hold = propagation.holdout_validate(cell, "d_docv7", path_preds, seed=seed)
    if len(prog):
        prog.to_csv(proc / "ocv_progression.csv", index=False)

    print("[S7] 구배 보정 …")
    corr_preds = fit.get("predictors", path_preds) if fit.get("ok") else path_preds
    cell = correction.fit_factor_correction(cell, corr_preds, target="docv7")
    cell = correction.fit_spatial_detrend(cell, target="docv7", degree=2)
    corr_eval = correction.evaluate(
        cell, {"factor": "docv7_corr_factor", "spatial": "docv7_corr_spatial"},
        cfg, n_rows, n_cols)
    _save_table(cell, proc / "cell_table_corrected")

    # 그림
    viz.plot_field_gallery(cell, "d_docv7", n_rows, n_cols,
                           outdir / "figures" / "gallery_docv7.png",
                           title="ΔdOCV7 (tray-median removed)")
    if top_step is not None:
        viz.plot_field_gallery(cell, top_step, n_rows, n_cols,
                               outdir / "figures" / "gallery_top_step.png",
                               title=f"{top_step}")

    # 요약 리포트
    summary = {
        "meta": meta,
        "impute": {
            "total_invalid": int(impute_rep["n_invalid"].sum()) if len(impute_rep) else 0,
            "total_imputed_3x3": int(impute_rep["n_imputed_3x3"].sum()) if len(impute_rep) else 0,
            "total_fallback": int(impute_rep["n_fallback_median"].sum()) if len(impute_rep) else 0,
            "n_excluded_fields": int(impute_rep.get("excluded_high_impute", pd.Series(dtype=bool)).sum()) if len(impute_rep) else 0,
        },
        "screening_top10": scr.head(10).to_dict("records"),
        "screening_top10_exclude_imputed": scr_excl.head(10).to_dict("records"),
        "path_fit": fit,
        "ocv_progression": prog.to_dict("records") if len(prog) else [],
        "holdout": hold,
        "correction_eval": corr_eval,
        "n_fail_raw": int(cell["judge_fail"].sum()) if "judge_fail" in cell else None,
    }
    with open(outdir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=_json_default)
    _write_markdown(summary, scr, corr_eval, outdir / "report_run.md")
    print(f"[done] 결과: {outdir}/summary.json, {outdir}/report_run.md, {proc}/")


def _save_table(df: pd.DataFrame, path_no_ext: Path):
    """parquet 우선, 없으면 csv 폴백."""
    try:
        df.to_parquet(f"{path_no_ext}.parquet", index=False)
    except Exception:
        df.to_csv(f"{path_no_ext}.csv", index=False, encoding="utf-8-sig")


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return str(o)


def _write_markdown(summary, scr, corr_eval, path):
    lines = ["# 온도–OCV 구배 분석 실행 리포트\n"]
    m = summary["meta"]
    lines.append(f"- 셀 {m['n_cells']}, 트레이 {m['n_trays']}, 스텝 {m['n_steps']}\n")
    lines.append("\n## 상관 스크리닝 상위 (전체)\n")
    lines.append("| predictor | target | median r | sign_consist | q_fdr |")
    lines.append("|---|---|---|---|---|")
    for r in summary["screening_top10"]:
        q = r.get("q_fdr")
        lines.append(f"| {r['predictor']} | {r['target']} | "
                     f"{_f(r['median'])} | {_f(r['sign_consistency'])} | {_f(q)} |")
    lines.append("\n## 보정 전후 판정 영향\n")
    lines.append("| variant | ΔdOCV7 진폭(p2p) median | ring abs median | 불량수 | 핫셀보존 | pass→fail | fail→pass |")
    lines.append("|---|---|---|---|---|---|---|")
    for name, e in corr_eval.items():
        lines.append(f"| {name} | {_f(e.get('amp_p2p_median'))} | {_f(e.get('ring_abs_median'))} | "
                     f"{e.get('n_fail')} | {e.get('hotcell_preserved','-')}/{e.get('hotcell_total','-')} | "
                     f"{e.get('flips_pass_to_fail','-')} | {e.get('flips_fail_to_pass','-')} |")
    fit = summary["path_fit"]
    if fit.get("ok"):
        lines.append(f"\n## 경로 회귀 (R²={_f(fit['r2'])}, n={fit['n']})\n")
        lines.append("| predictor | 표준화계수 | 기여율 |")
        lines.append("|---|---|---|")
        for p in fit["predictors"]:
            lines.append(f"| {p} | {_f(fit['std_coef'].get(p))} | {_f(fit['contrib_frac'].get(p))} |")
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def _f(x):
    try:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return "–"
        return f"{float(x):.3f}"
    except (TypeError, ValueError):
        return str(x)


def build_parser():
    p = argparse.ArgumentParser(description="트레이 온도–OCV 구배 분석 파이프라인")
    sub = p.add_subparsers(dest="cmd", required=True)

    pi = sub.add_parser("inspect", help="파일 칼럼 목록 출력")
    pi.add_argument("--input", required=True)
    pi.add_argument("--sheet", default=0)
    pi.set_defaults(func=cmd_inspect)

    pm = sub.add_parser("make-config", help="칼럼 매핑 초안 자동 생성")
    pm.add_argument("--input", required=True)
    pm.add_argument("--sheet", default=0)
    pm.add_argument("--output", default="analysis/column_config.yaml")
    pm.set_defaults(func=cmd_make_config)

    pr = sub.add_parser("run", help="전체 분석 실행")
    pr.add_argument("--config", required=True)
    pr.add_argument("--outdir", default="reports")
    pr.add_argument("--perm", type=int, default=None, help="permutation 횟수 override")
    pr.set_defaults(func=cmd_run)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    # sheet 가 숫자면 int 로
    if hasattr(args, "sheet"):
        try:
            args.sheet = int(args.sheet)
        except (TypeError, ValueError):
            pass
    args.func(args)


if __name__ == "__main__":
    main()
