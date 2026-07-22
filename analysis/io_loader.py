"""[S1] 원본 로딩 + 스키마 정규화 (wide 셀×공정 → long).

- inspect: 파일의 칼럼 목록을 보여줌 (사용자가 매핑을 고를 수 있게)
- build_config_template: canonical 스텝과 퍼지 매칭해 column_config.yaml 초안 생성
- to_long: wide 온도 칼럼을 long 포맷으로 melt
"""
from __future__ import annotations

import copy
import difflib
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .config import CANONICAL_STEPS, DEFAULT_CONFIG, Config


def read_table(path: str | Path, sheet=0) -> pd.DataFrame:
    """CSV/Excel 자동 감지 로딩."""
    p = Path(path)
    suf = p.suffix.lower()
    if suf in {".xlsx", ".xls", ".xlsm"}:
        return pd.read_excel(p, sheet_name=sheet)
    # CSV: 인코딩 폴백
    for enc in ("utf-8-sig", "utf-8", "cp949", "euc-kr", "latin-1"):
        try:
            return pd.read_csv(p, encoding=enc)
        except (UnicodeDecodeError, UnicodeError):
            continue
    return pd.read_csv(p)  # 마지막 시도 (에러 노출)


def list_columns(path: str | Path, sheet=0) -> list[str]:
    df = read_table(path, sheet)
    return list(df.columns)


def _norm(s: str) -> str:
    return re.sub(r"[\s_\-]+", "", str(s)).lower()


def _fuzzy_pick(target: str, columns: list[str], cutoff: float = 0.72) -> str | None:
    """정규화 후 근접 매칭. 정확/부분 포함 우선, 없으면 difflib."""
    tn = _norm(target)
    norm_map = {_norm(c): c for c in columns}
    if tn in norm_map:
        return norm_map[tn]
    # 부분 포함 (예: 칼럼에 접미/접두가 붙은 경우)
    contains = [c for c in columns if tn and tn in _norm(c)]
    if len(contains) == 1:
        return contains[0]
    match = difflib.get_close_matches(tn, list(norm_map.keys()), n=1, cutoff=cutoff)
    return norm_map[match[0]] if match else None


def build_config_template(path: str | Path, sheet=0) -> dict:
    """파일 칼럼을 보고 매핑 초안을 자동 생성. 사용자가 검토/수정용."""
    cols = list_columns(path, sheet)
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    cfg["input"]["path"] = str(path)
    cfg["input"]["sheet"] = sheet

    # --- 1) Export 형식 우선 감지 (…평균/최저/최고 온도, …온도) ---
    temp_cols = _detect_export_temperatures(cols)
    if temp_cols:
        cfg["temperature_columns"] = temp_cols
    else:
        # 폴백: canonical 스텝 이름 퍼지 매칭 (구형/단순 파일)
        used: set[str] = set()
        flat: dict = {}
        for step in CANONICAL_STEPS:
            pick = _fuzzy_pick(step, [c for c in cols if c not in used])
            flat[step] = pick
            if pick:
                used.add(pick)
        cfg["temperature_columns"] = flat

    # id/ocv 후보 퍼지 매칭
    def pick_any(names: list[str]) -> str | None:
        for n in names:
            p = _fuzzy_pick(n, cols)
            if p:
                return p
        return None

    cfg["id_columns"]["cell_id"] = pick_any(["Cell ID", "CELL_ID", "cell id", "cellno", "셀id"])
    cfg["id_columns"]["tray_id"] = pick_any(["TRAY ID", "TRAY_ID", "tray", "트레이"])
    cfg["id_columns"]["lot"] = pick_any(["Product Lot", "LOT", "랏", "lot id"])
    cfg["id_columns"]["row"] = pick_any(["ROW", "행", "cell row", "row no"])
    cfg["id_columns"]["col"] = pick_any(["COL", "열", "cell col", "col no"])

    # OCV: PRIVT OCV(전용OCV) 및 Delta OCV(docv7 직접) 감지
    ocv = _detect_ocv(cols)
    cfg["ocv_columns"].update(ocv["columns"])
    if ocv["docv7_direct"]:
        cfg["judge"]["docv7_from"] = "direct"

    # 위치(row/col) 후보 안내: 못 찾으면 position_from 에 힌트 칼럼만 채워둠
    if not (cfg["id_columns"]["row"] and cfg["id_columns"]["col"]):
        pos_hint = pick_any(["Cell 위치", "Cell No", "Cell 위치번호", "위치"])
        cfg["position_from"]["source_column"] = pos_hint
        cfg["position_from"]["regex"] = None  # 값 형식 확인 후 채울 것
    return cfg


# 온도 통계 접미사 → 표준 stat
_TEMP_SUFFIX = [
    (re.compile(r"^(?P<base>.+?)\s*평균\s*온도\s*$"), "mean"),
    (re.compile(r"^(?P<base>.+?)\s*최저\s*온도\s*$"), "min"),
    (re.compile(r"^(?P<base>.+?)\s*최고\s*온도\s*$"), "max"),
    (re.compile(r"^(?P<base>.+?)\s*온도\s*$"), "value"),
]


def _detect_export_temperatures(cols: list[str]) -> dict:
    """'… 평균/최저/최고 온도', '… 온도' 칼럼을 스텝별 {stat: col} 로 묶는다.

    칼럼 등장 순서를 보존해 대략적인 공정 순서를 유지한다.
    """
    out: dict[str, dict] = {}
    for c in cols:
        for rgx, stat in _TEMP_SUFFIX:
            m = rgx.match(str(c))
            if m:
                base = m.group("base").strip()
                out.setdefault(base, {})[stat] = c
                break
    return out


def _detect_ocv(cols: list[str]) -> dict:
    """PRIVT OCV(전용OCV) #01/#02/#03, Delta OCV(docv7) 감지."""
    columns: dict[str, str] = {}
    docv7_direct = False
    privt = re.compile(r"privt\s*ocv\s*#?0*(\d+)\s*ocv\s*$", re.IGNORECASE)
    delta = re.compile(r"delta\s*ocv.*docv", re.IGNORECASE)
    for c in cols:
        cs = str(c)
        m = privt.search(cs)
        if m:
            n = int(m.group(1))
            if n in (1, 2, 3):
                columns[f"ocv{n}"] = c
            continue
        if delta.search(cs):
            columns["docv7"] = c
            docv7_direct = True
    return {"columns": columns, "docv7_direct": docv7_direct}


def _resolve_positions(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """row/col 을 칼럼 또는 정규식 파싱으로 확보."""
    idc = cfg.raw["id_columns"]
    out = pd.DataFrame(index=df.index)
    out["cell_id"] = df[idc["cell_id"]].astype(str)
    out["tray_id"] = df[idc["tray_id"]].astype(str) if idc.get("tray_id") else "T0"
    if idc.get("lot"):
        out["lot"] = df[idc["lot"]].astype(str)
    else:
        out["lot"] = "L0"

    if idc.get("row") and idc.get("col"):
        out["row"] = pd.to_numeric(df[idc["row"]], errors="coerce").astype("Int64")
        out["col"] = pd.to_numeric(df[idc["col"]], errors="coerce").astype("Int64")
    else:
        pos = cfg.raw.get("position_from", {})
        src, rgx = pos.get("source_column"), pos.get("regex")
        if not (src and rgx):
            raise ValueError("row/col 칼럼도 없고 position_from(regex)도 없습니다.")
        ext = df[src].astype(str).str.extract(rgx)
        if "tray" in ext.columns:
            out["tray_id"] = ext["tray"].astype(str)
        out["row"] = pd.to_numeric(ext["row"], errors="coerce").astype("Int64")
        out["col"] = pd.to_numeric(ext["col"], errors="coerce").astype("Int64")
    return out


def load(cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """원본 → (temp_long, ocv_cell, meta).

    temp_long : lot, tray_id, row, col, cell_id, step_name, step_order, temp_raw
    ocv_cell  : lot, tray_id, row, col, cell_id, ocv1, ocv2, ocv3, docv7
    """
    problems = cfg.validate()
    if problems:
        raise ValueError("설정 오류:\n- " + "\n- ".join(problems))

    df = read_table(cfg.raw["input"]["path"], cfg.raw["input"]["sheet"])
    pos = _resolve_positions(df, cfg)

    # --- temp_long (wide → long); 스텝별 대표/최저/최고 온도 ---
    frames = []
    for i, step in enumerate(cfg.temp_steps):
        prim = cfg.primary_temp_col(step)
        if prim not in df.columns:
            raise KeyError(f"온도 칼럼 '{prim}' (step={step}) 이 파일에 없습니다.")
        part = pos.copy()
        part["step_name"] = step
        part["step_order"] = i
        part["temp_raw"] = pd.to_numeric(df[prim], errors="coerce")
        cmin, cmax = cfg.temp_col(step, "min"), cfg.temp_col(step, "max")
        part["temp_min_raw"] = pd.to_numeric(df[cmin], errors="coerce") if cmin else np.nan
        part["temp_max_raw"] = pd.to_numeric(df[cmax], errors="coerce") if cmax else np.nan
        frames.append(part)
    temp_long = pd.concat(frames, ignore_index=True)

    # --- ocv_cell ---
    oc = cfg.raw["ocv_columns"]
    j = cfg.raw["judge"]
    scale = j["unit_scale_to_mv"]
    d_scale = j.get("docv7_unit_scale_to_mv") or scale
    ocv = pos.copy()
    for key in ("ocv1", "ocv2", "ocv3"):
        if oc.get(key):
            ocv[key] = pd.to_numeric(df[oc[key]], errors="coerce") * scale
        else:
            ocv[key] = pd.NA
    if j["docv7_from"] == "direct" and oc.get("docv7"):
        ocv["docv7"] = pd.to_numeric(df[oc["docv7"]], errors="coerce") * d_scale
    else:
        ocv["docv7"] = ocv["ocv1"] - ocv["ocv3"]

    meta = _integrity_report(temp_long, ocv, cfg)
    return temp_long, ocv, meta


def _integrity_report(temp_long: pd.DataFrame, ocv: pd.DataFrame, cfg: Config) -> dict:
    n_rows_cfg = cfg.raw["tray_shape"]["n_rows"]
    n_cols_cfg = cfg.raw["tray_shape"]["n_cols"]
    dup = ocv.duplicated(subset=["tray_id", "row", "col"]).sum()
    inferred_cols = int(pd.to_numeric(ocv["col"], errors="coerce").max()) if len(ocv) else None
    return {
        "n_cells": int(ocv["cell_id"].nunique()),
        "n_trays": int(ocv["tray_id"].nunique()),
        "n_steps": int(temp_long["step_name"].nunique()),
        "steps": cfg.present_steps,
        "dup_position_rows": int(dup),
        "tray_n_rows_cfg": n_rows_cfg,
        "tray_n_cols_cfg": n_cols_cfg,
        "tray_n_cols_inferred": inferred_cols,
        "missing_row_col": int(ocv["row"].isna().sum() + ocv["col"].isna().sum()),
    }
