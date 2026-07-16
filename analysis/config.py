"""공정 순서 상수 + 칼럼 매핑 설정 로딩/저장.

데이터는 wide 포맷 (행 = 셀 ID, 각 공정 스텝 = 온도 칼럼).
사용자는 `column_config.yaml`에 실제 엑셀 칼럼 이름을 넣거나(=매핑),
`run_pipeline inspect` 로 칼럼 목록을 본 뒤 고른다.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# ---------------------------------------------------------------------------
# 확정된 공정 순서 (canonical). 사진의 Process Name 순서 + SOC 프로파일.
# 온도 칼럼 매핑은 이 이름들을 키로 사용한다.
# ---------------------------------------------------------------------------
CANONICAL_STEPS: list[str] = [
    "LCI",
    "1st RT Aging",
    "1st Charging",
    "2nd Charging",
    "1st HT Aging",
    "2nd RT Aging",
    "3rd Charging",
    "4th Charging",
    "2nd HT Aging",
    "3rd RT Aging",
    "5th Charging",
    "6th Charging",
    "7th Charging",
    "1st Discharge",
    "2nd Discharge",
    "3rd Discharge",
    "4th Discharge",
    "5th Discharge",
    "6th Discharge",
    "7th Discharge",
]

# 스텝 메타: phase(charge/discharge/aging/other), 목표 SOC(%), 고온(60C) 여부.
# SOC: 고온1 전 ~10, 고온2 전 ~70, 이후 100 충전 후 30 방전.
STEP_META: dict[str, dict[str, Any]] = {
    "LCI":            {"phase": "other",     "soc": None, "high_temp": False},
    "1st RT Aging":   {"phase": "aging",     "soc": None, "high_temp": False},
    "1st Charging":   {"phase": "charge",    "soc": 10,   "high_temp": False},
    "2nd Charging":   {"phase": "charge",    "soc": 10,   "high_temp": False},
    "1st HT Aging":   {"phase": "aging",     "soc": 10,   "high_temp": True},
    "2nd RT Aging":   {"phase": "aging",     "soc": 10,   "high_temp": False},
    "3rd Charging":   {"phase": "charge",    "soc": 70,   "high_temp": False},
    "4th Charging":   {"phase": "charge",    "soc": 70,   "high_temp": False},
    "2nd HT Aging":   {"phase": "aging",     "soc": 70,   "high_temp": True},
    "3rd RT Aging":   {"phase": "aging",     "soc": 70,   "high_temp": False},
    "5th Charging":   {"phase": "charge",    "soc": 100,  "high_temp": False},
    "6th Charging":   {"phase": "charge",    "soc": 100,  "high_temp": False},
    "7th Charging":   {"phase": "charge",    "soc": 100,  "high_temp": False},
    "1st Discharge":  {"phase": "discharge", "soc": 30,   "high_temp": False},
    "2nd Discharge":  {"phase": "discharge", "soc": 30,   "high_temp": False},
    "3rd Discharge":  {"phase": "discharge", "soc": 30,   "high_temp": False},
    "4th Discharge":  {"phase": "discharge", "soc": 30,   "high_temp": False},
    "5th Discharge":  {"phase": "discharge", "soc": 30,   "high_temp": False},
    "6th Discharge":  {"phase": "discharge", "soc": 30,   "high_temp": False},
    "7th Discharge":  {"phase": "discharge", "soc": 30,   "high_temp": False},
}

# 전용OCV는 7th Discharge 이후 SOC30에서 측정. dOCV7 = OCV1 - OCV3 (기본 가정).
DEFAULT_CONFIG: dict[str, Any] = {
    "input": {
        "path": "data/raw/tray_temp.xlsx",
        "sheet": 0,               # 엑셀 시트 (이름 또는 0-index). CSV면 무시.
    },
    "id_columns": {
        "cell_id": "CELL_ID",
        "tray_id": "TRAY_ID",
        "row": "ROW",             # 트레이 내 행 (1..12). 없으면 null → parse 사용
        "col": "COL",             # 트레이 내 열. 없으면 null.
        "lot": None,              # 랏 칼럼 (선택)
    },
    # row/col 이 별도 칼럼으로 없고 cell_id/다른 칼럼에 인코딩된 경우 정규식으로 파싱.
    # 예: "T012R03C05" -> named group tray/row/col. 미사용 시 null.
    "position_from": {
        "source_column": None,    # 예: "CELL_ID"
        "regex": None,            # 예: r"T(?P<tray>\d+)R(?P<row>\d+)C(?P<col>\d+)"
    },
    "tray_shape": {
        "n_rows": 12,
        "n_cols": None,           # null이면 데이터에서 자동 추론
    },
    # canonical step -> 실제 엑셀 온도 칼럼명. 없는 스텝은 null.
    "temperature_columns": {s: None for s in CANONICAL_STEPS},
    "ocv_columns": {
        "ocv1": "전용OCV1",
        "ocv2": "전용OCV2",
        "ocv3": "전용OCV3",
        "docv7": None,            # 직접 제공되면 칼럼명, 아니면 null → 계산
    },
    "preprocess": {
        "temp_min_valid": 22.0,   # 이 값 '이하'(<=)는 무효 → 결측 처리 후 치환
        "impute_method": "neighbor3x3",
        "min_valid_neighbors": 3, # 3x3 창에서 최소 유효 이웃 수
        "max_impute_iters": 3,    # 덩어리 결측 반복 치환 횟수
        "fallback_tray_median": True,
        "max_impute_frac": 0.30,  # 필드 치환율 > 이 값이면 분석 제외
    },
    "judge": {
        "docv7_from": "ocv1_minus_ocv3",  # "ocv1_minus_ocv3" | "direct"
        "offset_mv": 0.8,         # (셀 docv7 - tray mode) > offset → 불량
        "unit_scale_to_mv": 1.0,  # 원시 전압을 mV로 변환하는 배수 (V면 1000)
        "mode_bin_mv": 0.05,      # tray mode 추정 히스토그램 bin 폭 (mV)
    },
    "run": {
        "seed": 12345,
        "n_permutations": 1000,
        "arrhenius_ea_kj": [40.0, 50.0, 60.0],  # 자기방전 활성화에너지 감도
    },
}


@dataclass
class Config:
    raw: dict[str, Any] = field(default_factory=lambda: copy.deepcopy(DEFAULT_CONFIG))

    # --- 편의 접근자 -------------------------------------------------------
    def get(self, *keys: str, default: Any = None) -> Any:
        node: Any = self.raw
        for k in keys:
            if not isinstance(node, dict) or k not in node:
                return default
            node = node[k]
        return node

    @property
    def temp_map(self) -> dict[str, str]:
        """canonical step -> 실제 칼럼 (null 제외, 존재하는 것만)."""
        return {s: c for s, c in self.raw["temperature_columns"].items() if c}

    @property
    def present_steps(self) -> list[str]:
        """온도 칼럼이 매핑된 스텝을 canonical 순서로."""
        tm = self.temp_map
        return [s for s in CANONICAL_STEPS if s in tm]

    def validate(self) -> list[str]:
        """설정 정합성 검사. 문제 목록 반환 (빈 리스트면 OK)."""
        problems: list[str] = []
        if not self.temp_map:
            problems.append("temperature_columns 에 매핑된 스텝이 하나도 없습니다.")
        idc = self.raw["id_columns"]
        if not idc.get("cell_id"):
            problems.append("id_columns.cell_id 가 비어 있습니다.")
        pos = self.raw.get("position_from", {})
        has_rc = idc.get("row") and idc.get("col")
        has_parse = pos.get("source_column") and pos.get("regex")
        if not (has_rc or has_parse):
            problems.append(
                "트레이 내 위치(row/col)를 얻을 수 없습니다: "
                "id_columns.row/col 을 지정하거나 position_from(regex)을 설정하세요."
            )
        oc = self.raw["ocv_columns"]
        if self.raw["judge"]["docv7_from"] == "ocv1_minus_ocv3":
            if not (oc.get("ocv1") and oc.get("ocv3")):
                problems.append("docv7=ocv1-ocv3 인데 ocv1 또는 ocv3 칼럼이 없습니다.")
        elif self.raw["judge"]["docv7_from"] == "direct":
            if not oc.get("docv7"):
                problems.append("docv7_from=direct 인데 ocv_columns.docv7 이 없습니다.")
        return problems


def load_config(path: str | Path) -> Config:
    with open(path, "r", encoding="utf-8") as f:
        user = yaml.safe_load(f) or {}
    merged = _deep_merge(copy.deepcopy(DEFAULT_CONFIG), user)
    return Config(raw=merged)


def save_config(cfg: dict[str, Any], path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False, default_flow_style=False)


def _deep_merge(base: dict, override: dict) -> dict:
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v
    return base
