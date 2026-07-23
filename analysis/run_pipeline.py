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
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import (correction, features, genesis, genesis_math, io_loader, preprocess,
               propagation, screening, viz)
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


def _build_cell_table(cfg: Config, log: "Progress"):
    """S1~S4 실행 후 분석용 셀 테이블·부속 산출물 반환."""
    log("S1  파일 로딩 + wide→long 변환 …", 1)
    temp_long, ocv_cell, meta = io_loader.load(cfg)
    log(f"S1  완료 — 셀 {meta['n_cells']}, 트레이 {meta['n_trays']}, "
        f"온도측정 스텝 {meta['n_steps']}", 2)

    log("S2  전처리 (결측/≤22°C 삭제 → 3×3 이웃 평균 치환) …", 1)
    temp_clean, impute_rep = preprocess.run(temp_long, cfg)
    if len(impute_rep):
        log(f"S2  완료 — 무효 {int(impute_rep['n_invalid'].sum())}, "
            f"3×3치환 {int(impute_rep['n_imputed_3x3'].sum())}, "
            f"중앙값폴백 {int(impute_rep['n_fallback_median'].sum())}", 2)

    log("판정  트레이 mode + (docv7−mode)>offset …", 1)
    ocv_j = apply_judgment(ocv_cell, cfg.raw["judge"]["offset_mv"],
                           cfg.raw["judge"]["mode_bin_mv"])
    log(f"판정  완료 — 불량 셀 {int(ocv_j['judge_fail'].sum())}", 2)
    # OCV Δ필드 (트레이 중앙값 제거)
    for c in ("ocv1", "ocv2", "ocv3", "docv7"):
        add_tray_delta(ocv_j, c, by=("tray_id",), out_col=f"d_{c}")

    log("S4  온도 피처 추출 (Δ필드 + 프록시 + 열노출) …", 1)
    feat, long_delta = features.build(temp_clean, cfg)
    log(f"S4  완료 — 피처 칼럼 {feat.shape[1]}개", 2)

    # 셀별 치환 플래그
    any_imp = (temp_clean.groupby("cell_id")["imputed"].any()
               .rename("any_imputed").reset_index())

    id_cols = ["lot", "tray_id", "row", "col", "cell_id"]
    stage_cols = [c for c in ocv_j.columns if c.startswith("ocvstage::")]
    ocv_keep = id_cols + ["ocv1", "ocv2", "ocv3", "docv7", "docv7_tray_mode",
                          "docv7_dev", "judge_fail", "d_ocv1", "d_ocv2", "d_ocv3",
                          "d_docv7"] + stage_cols
    cell = feat.merge(ocv_j[ocv_keep], on=id_cols, how="left")
    cell = cell.merge(any_imp, on="cell_id", how="left")
    cell["any_imputed"] = cell["any_imputed"].fillna(False)
    return cell, temp_clean, long_delta, impute_rep, meta


def _predictor_lists(cfg: Config, cell: pd.DataFrame):
    step_preds = [f"dT::{s}" for s in cfg.present_steps if f"dT::{s}" in cell.columns]
    step_preds += [f"dTrise::{s}" for s in cfg.present_steps
                   if f"dTrise::{s}" in cell.columns]
    proxy = [c for c in ["dT_charge_mean", "dT_discharge_mean", "dT_preOCV_proxy",
                         "dT_HT_adjacent", "dT_ocvmeas1", "dT_ocvmeas3",
                         "dT_ocv1_minus_ocv3"] if c in cell.columns]
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
    log = Progress(enabled=not args.quiet)

    log("[S1–S4] 로딩·전처리·판정·피처")
    cell, temp_clean, long_delta, impute_rep, meta = _build_cell_table(cfg, log)
    n_rows, n_cols = infer_shape(cell, cfg.raw["tray_shape"]["n_rows"],
                                 cfg.raw["tray_shape"]["n_cols"])
    _save_table(cell, proc / "cell_table")
    impute_rep.to_csv(proc / "impute_report.csv", index=False)
    log(f"셀 테이블 저장 완료 — 격자 {n_rows}×{n_cols}", 1)

    step_preds, proxy, heat = _predictor_lists(cfg, cell)
    predictors = step_preds + proxy + heat
    targets = [t for t in ["d_ocv1", "d_ocv2", "d_ocv3", "d_docv7"] if t in cell.columns]

    n_combo = len(predictors) * len(targets)
    log(f"[S5] 전수 상관 스크리닝 — 예측자 {len(predictors)} × 타깃 {len(targets)} "
        f"= {n_combo} 조합, permutation {n_perm}회")
    log("S5  (a) 전체 셀 …", 1)
    scr = screening.screen(cell, predictors, targets, n_perm=n_perm, seed=seed,
                           exclude_imputed=False,
                           progress=lambda m: log(m, 2))
    log("S5  (b) 치환 셀 제외 (감도분석) …", 1)
    scr_excl = screening.screen(cell, predictors, targets, n_perm=n_perm, seed=seed,
                                exclude_imputed=True,
                                progress=lambda m: log(m, 2))
    scr.to_csv(proc / "screening_all.csv", index=False)
    scr_excl.to_csv(proc / "screening_exclude_imputed.csv", index=False)
    sim = screening.step_similarity(cell, step_preds)
    sim.to_csv(proc / "step_similarity.csv")
    if len(scr):
        r = scr.iloc[0]
        log(f"S5  완료 — 최상위: {r['predictor']} × {r['target']} "
            f"(median r={r['median']:.3f}, q={r['q_fdr']:.3g})", 2)
    viz.plot_corr_heatmap(scr[scr["predictor"].isin(step_preds)],
                          outdir / "figures" / "corr_heatmap_steps.png",
                          title="step ΔT × OCV target (per-tray median corr)")
    log("S5  히트맵 저장", 2)

    log("[S6] 경로 판별 회귀 + 검증")
    # 열노출은 Ea별로 서로 단조변환(공선성) → 회귀엔 대표 1개만 사용
    heat_one = [heat[len(heat) // 2]] if heat else []
    # P1 직접 예측자(측정시점 온도차)가 있으면 최우선, 없으면 프록시
    p1 = "dT_ocv1_minus_ocv3" if "dT_ocv1_minus_ocv3" in cell.columns else "dT_preOCV_proxy"
    path_preds = [c for c in [p1, "dT_HT_adjacent"] + heat_one if c in cell.columns]
    log(f"S6  회귀 (예측자 {path_preds}) …", 1)
    fit = propagation.fit_paths(cell, "d_docv7", path_preds, exclude_imputed=False)
    if fit.get("ok"):
        log(f"S6  R²={fit['r2']:.3f}, n={fit['n']}", 2)
    else:
        log(f"S6  회귀 생략 — {fit.get('reason')}", 2)
    top_step = scr.iloc[0]["predictor"] if len(scr) else (step_preds[0] if step_preds else None)
    log(f"S6  ΔOCV1→2→3 진행분석 (기준 {top_step}) …", 1)
    prog = (propagation.ocv_progression(cell, top_step) if top_step is not None
            else pd.DataFrame())
    log("S6  홀드아웃 검증 (트레이 8:2) …", 1)
    hold = propagation.holdout_validate(cell, "d_docv7", path_preds, seed=seed)
    if hold.get("ok"):
        log(f"S6  홀드아웃 pred vs actual median r="
            f"{hold['pred_vs_actual']['median']:.3f}", 2)
    if len(prog):
        prog.to_csv(proc / "ocv_progression.csv", index=False)

    # --- S8 구배 발생 추적 (원인 공정 국소화) ---
    stage_meta = meta.get("ocv_stages", [])
    genesis_prog = pd.DataFrame()
    steps = pd.DataFrame()
    if stage_meta:
        log(f"[S8] 구배 발생 추적 — OCV 단계 {len(stage_meta)}개")
        log("S8  단계별 Δ필드 + docv7 상관 …", 1)
        cell, _ = genesis.build_stage_deltas(cell)
        genesis_prog = genesis.progression(cell, stage_meta, n_rows, n_cols)
        steps = genesis.step_contributions(cell, stage_meta, n_rows, n_cols)
        seg = genesis.selfdischarge_segments(cell, stage_meta, n_rows, n_cols)
        genesis_prog.to_csv(proc / "genesis_progression.csv", index=False)
        steps.to_csv(proc / "genesis_step_contributions.csv", index=False)
        if len(seg):
            seg.to_csv(proc / "genesis_selfdischarge_segments.csv", index=False)
        # docv7 링과 처음 크게 일치하는 '단계' 로그
        gp = genesis_prog.dropna(subset=["corr_docv7_median"])
        if len(gp):
            first = gp[gp["corr_docv7_median"].abs() > 0.3]
            if len(first):
                b = first.iloc[0]
                log(f"S8  docv7 링이 처음 나타나는 단계: {b['stage']} "
                    f"(corr={b['corr_docv7_median']:.3f}, 링진폭={b['ring_abs_median']:.3g})", 2)
        # 공정별 기여: matched-SOC & docv7 구성구간 제외 → 링을 가장 크게 유발한 공정
        if len(steps):
            cand = steps[steps["matched_soc"] & ~steps["composes_docv7"]].dropna(
                subset=["diff_ring_abs_median"])
            if len(cand):
                b = cand.loc[cand["diff_ring_abs_median"].idxmax()]
                log(f"S8  구배 유발 유력 공정: {b['process']} "
                    f"(차링={b['diff_ring_abs_median']:.3g}, docv7상관={b['corr_docv7_median']:.3f})", 2)
        viz.plot_stage_progression(genesis_prog,
                                   outdir / "figures" / "genesis_progression.png",
                                   title="OCV stage - gradient genesis tracking")

        # --- 수학적 발생 규명 (a_k, β_k, Moran's I, 변화점) ---
        log("S8  수학 분석: 누적정렬 a_k, 증분투영 β_k, Moran's I, 변화점 …", 1)
        align, beta = genesis_math.alignment_and_projection(cell, stage_meta, n_rows, n_cols)
        moran = genesis_math.morans_progression(cell, stage_meta, n_rows, n_cols)
        cp = genesis_math.changepoint(align)
        align.to_csv(proc / "genesis_alignment.csv", index=False)
        beta.to_csv(proc / "genesis_beta.csv", index=False)
        moran.to_csv(proc / "genesis_moran.csv", index=False)
        genesis_math_out = {"align": align, "beta": beta, "moran": moran, "cp": cp}
        # 종합 판정 로그
        if cp.get("ok"):
            log(f"S8  변화점(a_k 급등): {cp['stage']}", 2)
        bcand = beta[beta["matched_soc"] & ~beta["composes_docv7"]].dropna(subset=["beta_median"])
        if len(bcand):
            bb = bcand.loc[bcand["beta_median"].abs().idxmax()]
            log(f"S8  β 최대 공정(원인): {bb['process']} (β={bb['beta_median']:.3g})", 2)

        # 트레이별 발생단계 분포 (부호 상쇄 없는 집계) — 링/역링 공존 대응
        log("S8  트레이별 발생단계 분포 (부호-강건) …", 1)
        ptg = genesis_math.per_tray_genesis(cell, stage_meta, n_rows, n_cols)
        genesis_math_out["per_tray"] = ptg
        if not ptg["onset_dist"].empty and ptg["n_used"]:
            od = ptg["onset_dist"].sort_values("frac", ascending=False)
            top_onset = od.iloc[0]
            log(f"S8  [트레이 {ptg['n_used']}개] 발생단계 최빈: {top_onset['stage']} "
                f"({top_onset['frac']*100:.0f}%)", 2)
            if not ptg["culprit_dist"].empty:
                tc = ptg["culprit_dist"].iloc[0]
                log(f"S8  유발공정 최빈: {tc['process']} ({tc['frac']*100:.0f}%)", 2)
            ptg["onset_dist"].to_csv(proc / "genesis_onset_distribution.csv", index=False)
            ptg["culprit_dist"].to_csv(proc / "genesis_culprit_distribution.csv", index=False)

        # --- 강한 docv7 링 트레이만 층화 (간헐 신호 희석 방지) ---
        strong_trays = _strong_ring_trays(cell, n_rows, n_cols, frac=0.15, min_n=30)
        if strong_trays:
            cell_s = cell[cell["tray_id"].isin(strong_trays)]
            align_s, beta_s = genesis_math.alignment_and_projection(
                cell_s, stage_meta, n_rows, n_cols)
            moran_s = genesis_math.morans_progression(cell_s, stage_meta, n_rows, n_cols)
            cp_s = genesis_math.changepoint(align_s)
            align_s.to_csv(proc / "genesis_alignment_strongring.csv", index=False)
            beta_s.to_csv(proc / "genesis_beta_strongring.csv", index=False)
            log(f"S8  [강한링 트레이 {len(strong_trays)}개] 변화점="
                f"{cp_s.get('stage')}", 2)
            bcs = beta_s[beta_s["matched_soc"] & ~beta_s["composes_docv7"]].dropna(
                subset=["beta_median"])
            if len(bcs):
                bbs = bcs.loc[bcs["beta_median"].abs().idxmax()]
                log(f"S8  [강한링] β 최대 공정: {bbs['process']} (β={bbs['beta_median']:.3g})", 2)
            genesis_math_out["strong"] = {"align": align_s, "beta": beta_s,
                                          "moran": moran_s, "cp": cp_s,
                                          "n_trays": len(strong_trays)}

        # --- 발표용 시각자료 (패널별 정규화로 핫셀·저진폭 문제 해결) ---
        log("S8  발표용 시각자료 생성 …", 1)
        pres = outdir / "figures" / "presentation"
        trays_curated = _curate_trays(cell, n_rows, n_cols, n_top=8, n_rand=4, seed=seed)
        stage_labels = [m["label"] for m in sorted(stage_meta, key=lambda x: x.get("order", 0))]
        sub_cur = cell[cell["tray_id"].isin(trays_curated)]
        for lab in stage_labels:
            dcol = f"d_ocvstage::{lab}"
            if dcol in cell.columns:
                viz.plot_field_gallery(sub_cur, dcol, n_rows, n_cols,
                                       pres / f"gallery_stage_{_safe(lab)}.png",
                                       max_trays=12, title=f"dOCV field — {lab}",
                                       per_panel=True)
        # 스토리보드(강한링 트레이 우선) / β / a_k·Moran / 공통지문
        story_trays = (strong_trays[:6] if strong_trays else trays_curated[:6])
        viz.plot_storyboard(cell, stage_labels, story_trays, n_rows, n_cols,
                            pres / "storyboard.png", per_panel=True)
        viz.plot_beta_contributions(beta, pres / "beta_contributions.png")
        viz.plot_alignment_moran(align, moran, pres / "alignment_moran.png",
                                 changepoint_stage=cp.get("stage"))
        viz.plot_mean_field(cell, stage_labels, n_rows, n_cols,
                            pres / "common_fingerprint_meanfield.png")
        if genesis_math_out.get("per_tray"):
            ptg = genesis_math_out["per_tray"]
            viz.plot_onset_distribution(ptg["onset_dist"], ptg["culprit_dist"],
                                        pres / "onset_distribution.png", n_used=ptg["n_used"])
    else:
        log("[S8] 구배 발생 추적 생략 — 중간 OCV 단계 칼럼 없음", 0)
        genesis_math_out = None

    log("[S7] 구배 보정")
    corr_preds = fit.get("predictors", path_preds) if fit.get("ok") else path_preds
    log("S7  (1안) 온도 인자 기반 보정 …", 1)
    cell = correction.fit_factor_correction(cell, corr_preds, target="docv7")
    log("S7  (2안) 공간 detrend 폴백 …", 1)
    cell = correction.fit_spatial_detrend(cell, target="docv7", degree=2)
    log("S7  보정 전후 판정 영향 평가 …", 1)
    corr_eval = correction.evaluate(
        cell, {"factor": "docv7_corr_factor", "spatial": "docv7_corr_spatial"},
        cfg, n_rows, n_cols)
    if "factor" in corr_eval:
        e = corr_eval["factor"]
        log(f"S7  factor 보정 — 링 진폭 {corr_eval['none']['ring_abs_median']:.3f}"
            f"→{e['ring_abs_median']:.3f}, 핫셀보존 "
            f"{e.get('hotcell_preserved')}/{e.get('hotcell_total')}", 2)
    _save_table(cell, proc / "cell_table_corrected")

    # 그림
    log("그림 저장 (갤러리) …", 1)
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
        "genesis_progression": genesis_prog.to_dict("records") if len(genesis_prog) else [],
        "n_fail_raw": int(cell["judge_fail"].sum()) if "judge_fail" in cell else None,
    }
    with open(outdir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=_json_default)
    genesis_steps = steps if stage_meta and len(steps) else None
    _write_markdown(summary, scr, corr_eval, outdir / "report_run.md",
                    genesis_prog, genesis_steps, genesis_math_out)
    log(f"[완료] 결과: {outdir}/summary.json, {outdir}/report_run.md, {proc}/")


class Progress:
    """경과시간·flush 포함 진행 로거 (Windows 콘솔 버퍼링 대비)."""
    def __init__(self, enabled: bool = True):
        self.t0 = time.time()
        self.enabled = enabled

    def __call__(self, msg: str, indent: int = 0):
        if not self.enabled:
            return
        el = time.time() - self.t0
        print(f"[{el:6.1f}s] {'  ' * indent}{msg}", flush=True)


def _safe(s: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in str(s)).strip("_")


def _curate_trays(cell: pd.DataFrame, n_rows: int, n_cols: int,
                  n_top: int = 8, n_rand: int = 4, seed: int = 0) -> list:
    """발표 갤러리용 트레이 큐레이션: Δdocv7 링 진폭 상위 + 무작위."""
    from .fields import structure_table
    if "d_docv7" not in cell.columns:
        trays = list(cell["tray_id"].dropna().unique())
        return trays[:n_top + n_rand]
    st = structure_table(cell, "d_docv7", n_rows, n_cols, by=("tray_id",))
    st = st.dropna(subset=["ring"])
    top = st.reindex(st["ring"].abs().sort_values(ascending=False).index)["tray_id"].head(n_top).tolist()
    rng = np.random.default_rng(seed)
    rest = [t for t in cell["tray_id"].dropna().unique() if t not in top]
    rand = list(rng.choice(rest, size=min(n_rand, len(rest)), replace=False)) if rest else []
    return top + rand


def _strong_ring_trays(cell: pd.DataFrame, n_rows: int, n_cols: int,
                       frac: float = 0.15, min_n: int = 30) -> list:
    """Δdocv7 링 진폭 상위 frac 트레이 (간헐 신호 층화용)."""
    from .fields import structure_table
    if "d_docv7" not in cell.columns:
        return []
    st = structure_table(cell, "d_docv7", n_rows, n_cols, by=("tray_id",)).dropna(subset=["ring"])
    if st.empty:
        return []
    st = st.reindex(st["ring"].abs().sort_values(ascending=False).index)
    n = max(min_n, int(len(st) * frac))
    return st["tray_id"].head(n).tolist()


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


def _write_markdown(summary, scr, corr_eval, path, genesis_prog=None, genesis_steps=None,
                    gmath=None):
    lines = ["# 온도–OCV 구배 분석 실행 리포트\n"]
    m = summary["meta"]
    lines.append(f"- 셀 {m['n_cells']}, 트레이 {m['n_trays']}, 스텝 {m['n_steps']}\n")

    if gmath is not None:
        align, beta, moran, cp = gmath["align"], gmath["beta"], gmath["moran"], gmath["cp"]
        # 종합 판정 문장
        verdict = []
        if cp.get("ok"):
            verdict.append(f"변화점(a_k 급등)=**{cp['stage']}**")
        bc = beta[beta["matched_soc"] & ~beta["composes_docv7"]].dropna(subset=["beta_median"])
        if len(bc):
            bb = bc.loc[bc["beta_median"].abs().idxmax()]
            verdict.append(f"β 최대 공정=**{bb['process']}** (β={_f(bb['beta_median'])})")
        mo = moran.dropna(subset=["moran_median"])
        mo_first = mo[mo["moran_median"] > 0.1]
        if len(mo_first):
            verdict.append(f"Moran's I 최초 유의=**{mo_first.iloc[0]['stage']}**")
        lines.append("\n## 구배 발생 — 수학적 종합 판정\n")
        lines.append("> " + " · ".join(verdict) + "\n" if verdict else "> (판정 불가)\n")
        lines.append("\n### 단계별 누적 정렬 a_k / Moran's I\n")
        lines.append("| 단계 | SOC | a_k(정렬) | 부호일관성 | Moran's I |")
        lines.append("|---|---|---|---|---|")
        mo_map = dict(zip(moran["stage"], moran["moran_median"])) if moran is not None else {}
        for r in align.itertuples():
            soc = f"{int(r.soc)}" if pd.notna(r.soc) else "–"
            lines.append(f"| {r.stage} | {soc} | {_f(r.a_median)} | {_f(r.a_sign)} | "
                         f"{_f(mo_map.get(r.stage))} |")
        lines.append("\n### 공정별 증분 투영 β_k (원인 지목)\n")
        lines.append("| 공정 | 구간 | matched SOC | β_k | 부호일관성 |")
        lines.append("|---|---|---|---|---|")
        for r in beta.itertuples():
            flag = "✓" if r.matched_soc else "—"
            if r.composes_docv7:
                flag = "(docv7구성)"
            lines.append(f"| {r.process} | {r.from_stage}→{r.to_stage} | {flag} | "
                         f"{_f(r.beta_median)} | {_f(r.beta_sign)} |")
        lines.append("\n> β_k = 그 공정의 변화가 최종 Δdocv7 방향에 주입한 양. "
                     "matched-SOC(✓) 중 |β| 최대가 원인 공정.\n")

        # 트레이별 발생단계 분포 (부호-강건) — 링/역링 공존 시 핵심
        ptg = gmath.get("per_tray")
        if ptg and not ptg["onset_dist"].empty and ptg.get("n_used"):
            lines.append(f"\n### 트레이별 발생단계 분포 (부호-강건, n={ptg['n_used']} 트레이)\n")
            lines.append("> 링/역링이 공존하면 부호 평균(a_k,β_k)이 상쇄되므로, "
                         "트레이마다 개별 발생단계를 찾아 히스토그램. 상쇄 없음.\n")
            lines.append("| 발생단계 | SOC | 트레이 비율 |")
            lines.append("|---|---|---|")
            for r in ptg["onset_dist"].sort_values("frac", ascending=False).head(6).itertuples():
                soc = f"{int(r.soc)}" if pd.notna(r.soc) else "–"
                lines.append(f"| {r.stage} | {soc} | {_f(r.frac)} |")
            if not ptg["culprit_dist"].empty:
                lines.append("\n| 유발 공정 (matched-SOC) | 트레이 비율 |")
                lines.append("|---|---|")
                for r in ptg["culprit_dist"].head(5).itertuples():
                    lines.append(f"| {r.process} | {_f(r.frac)} |")

        # 강한링 트레이 층화 요약
        strong = gmath.get("strong")
        if strong:
            lines.append(f"\n### 강한 docv7 링 트레이만 ({strong['n_trays']}개) 층화\n")
            cps = strong["cp"]
            bs = strong["beta"]
            bsc = bs[bs["matched_soc"] & ~bs["composes_docv7"]].dropna(subset=["beta_median"])
            msg = f"변화점=**{cps.get('stage')}**"
            if len(bsc):
                bb = bsc.loc[bsc["beta_median"].abs().idxmax()]
                msg += f" · β 최대 공정=**{bb['process']}**"
            lines.append("> " + msg + " (전체 집계와 비교해 신호가 뚜렷해지는지 확인)\n")

    if genesis_steps is not None and len(genesis_steps):
        lines.append("\n## 공정별 구배 기여 (인접 OCV 차이 = 그 공정의 변화)\n")
        lines.append("| 공정 | 구간 | matched SOC | 차-필드 링(median) | Δdocv7 상관 | 부호일관성 |")
        lines.append("|---|---|---|---|---|---|")
        for r in genesis_steps.itertuples():
            lines.append(f"| {r.process} | {r.from_stage}→{r.to_stage} | "
                         f"{'✓' if r.matched_soc else '—'} | "
                         f"{_f(r.diff_ring_abs_median)} | {_f(r.corr_docv7_median)} | "
                         f"{_f(r.corr_docv7_signcons)} |")
        lines.append("\n> matched SOC(✓) 구간은 충·방전 효과가 상쇄돼 그 공정 고유의 "
                     "공간 구배만 남는다. 차-필드 링이 크고 Δdocv7 상관이 높은 공정이 원인.\n")

    if genesis_prog is not None and len(genesis_prog):
        lines.append("\n## 구배 발생 추적 (OCV 단계별 누적)\n")
        lines.append("| 단계 | SOC | 링 진폭(median) | 최종 Δdocv7 상관 | 부호일관성 |")
        lines.append("|---|---|---|---|---|")
        for r in genesis_prog.itertuples():
            soc = f"{int(r.soc)}" if pd.notna(r.soc) else "–"
            lines.append(f"| {r.stage} | {soc} | {_f(r.ring_abs_median)} | "
                         f"{_f(r.corr_docv7_median)} | {_f(r.corr_docv7_signcons)} |")
        lines.append("\n> 링 진폭이 커지고 'Δdocv7 상관'이 처음 높아지는 단계 = "
                     "구배가 태어난(확정된) 공정.\n")
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
    pr.add_argument("--quiet", action="store_true", help="단계별 진행 출력 끄기")
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
