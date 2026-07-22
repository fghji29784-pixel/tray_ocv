# tray_ocv — 트레이 내 OCV 구배 원인 분석·보정 파이프라인

트레이 단위 전용OCV / dOCV7 측정에서 나타나는 **셀 위치 의존 전압 구배**의
원인을 공정 단계별 온도 데이터로 규명하고, 원인 인자로 구배를 **보정**한다.

- 배경·인자 정리: [`ocv_gradient_root_cause.md`](ocv_gradient_root_cause.md)
- 분석 방법론: [`analysis_plan_temperature_tracing.md`](analysis_plan_temperature_tracing.md)
- 코드 구조 설계: [`code_design_pipeline.md`](code_design_pipeline.md)

## 데이터 형식

wide 포맷 — **1행 = 셀 1개**. 두 가지 스키마를 자동 인식한다:

1. **Export 형식** (실 생산 데이터): `Charge #01 평균/최저/최고 온도`,
   `DisCharge #07 …`, `Low Current Inspection #01 온도`,
   `PRIVT OCV #01/#02/#03 온도`(= 전용OCV 측정지점 온도), `PRIVT OCV #0N OCV`(전용OCV),
   `Delta OCV #07 DOCV`(= docv7 직접). `make-config`가 이 형식을 자동으로 묶는다.
   - Charge/DisCharge의 (최고−최저)는 **자기발열(승온) 피처**로 자동 추출.
   - `PRIVT OCV #01 온도 − #03 온도` = **P1(측정시점 온도차) 직접 예측자**로
     구성되어 docv7 구배의 상관·회귀·보정에 바로 쓰인다.
2. **단순 형식**: 각 공정 스텝이 온도 칼럼 1개(`1st Charging` 등), `전용OCV1~3`.

트레이 내 위치(`row`/`col`)는 `ROW`/`COL` 칼럼, 또는 `Cell 위치`/`Cell No` 같은
칼럼에서 정규식 파싱(`position_from`)으로 얻는다. Export 파일에 `ROW`/`COL`이
없으면 `column_config.yaml`의 `position_from`에 파싱 규칙을 지정해야 한다(아래).

## 설치

```bash
pip install -r requirements.txt
```

## 사용법 (3단계)

```bash
# 1) 업로드한 파일의 칼럼 목록 확인 (매핑을 고르기 위해)
python -m analysis.run_pipeline inspect --input data/raw/파일.xlsx

# 2) 칼럼 매핑 초안 자동 생성 → analysis/column_config.yaml 을 열어 검토·수정
#    (온도 스텝은 canonical 이름과 퍼지 매칭. null·오매칭을 직접 고친다.)
python -m analysis.run_pipeline make-config --input data/raw/파일.xlsx \
       --output analysis/column_config.yaml

# 3) 전체 분석 실행
python -m analysis.run_pipeline run --config analysis/column_config.yaml
```

`column_config.yaml`에서 반드시 확인할 항목:
- `judge.unit_scale_to_mv`: **ocv1/2/3** 원시 단위가 V면 **1000**, 이미 mV면 1.
- `judge.docv7_unit_scale_to_mv`: **docv7 직접칼럼**(Delta OCV) 단위 배수. 이미 mV면 1,
  V면 1000, null이면 위 값과 동일 적용. (Export는 보통 Delta OCV가 mV → **1**)
- `judge.docv7_from`: `direct`(Delta OCV 칼럼) 또는 `ocv1_minus_ocv3`.
- `preprocess.temp_min_valid`: 기본 22 (이 값 **이하** 삭제 후 3×3 치환).
- `tray_shape.n_rows`: 기본 12. `n_cols`는 null이면 자동 추론.
- **`position_from`**: `ROW`/`COL` 칼럼이 없을 때 위치를 파싱할 규칙.
  `make-config`가 아래를 자동으로 채운다(12×12 기준). 두 방식 지원:
  - `Cell 위치`가 `A01`~`L12` 형식(알파벳=행 A→1, 숫자=열)일 때 (자동 선택):
    ```yaml
    position_from:
      source_column: "Cell 위치"
      regex: "(?P<row>[A-Za-z]+)\\s*(?P<col>\\d+)"
    ```
  - `Cell No`(1~144)만 있을 때 — 행우선 산술 환산:
    ```yaml
    position_from:
      from_cell_number: {column: "Cell No", n_cols: 12, row_major: true}
    ```

온도 칼럼 매핑은 Export 형식에서 스텝별로 이렇게 묶인다:
```yaml
temperature_columns:
  "Charge #01":
    min:  "Charge #01 최저 온도"
    mean: "Charge #01 평균 온도"
    max:  "Charge #01 최고 온도"
  "PRIVT OCV #01": {value: "PRIVT OCV #01 온도"}
```

## 출력

- `reports/report_run.md` — 스크리닝 상위·경로 회귀·보정 전후 판정 영향 표
- `reports/summary.json` — 전체 수치 요약
- `reports/figures/` — 스텝×타깃 상관 히트맵, ΔdOCV7·상위 스텝 필드 갤러리
- `data/processed/` — 셀 테이블, 스크리닝 결과, 치환 리포트 등 (parquet/csv)

## 파이프라인 스테이지

| 모듈 | 역할 |
|---|---|
| `io_loader` (S1) | 로딩, 칼럼 표준화, wide→long, 조인 무결성 |
| `preprocess` (S2) | 결측/≤22°C 삭제 → 3×3 이웃 평균 치환 (치환 마스크 유지) |
| `fields` (S3) | 트레이 필드·Δ필드·구조지표(링/방향/줄무늬) |
| `features` (S4) | 셀×스텝 온도 피처 (환경 vs 자기발열 프록시 분리) |
| `screening` (S5) | 전 스텝 × OCV 전수 상관 (permutation + BH-FDR) |
| `propagation` (S6) | P1/P2/P3 경로 판별 회귀 + 홀드아웃 검증 |
| `genesis` (S8) | **구배 발생 추적** — 중간 OCV(#01~#07)+전용 OCV 단계별 ΔOCV 링을 공정 순서로 훑어 docv7 링이 태어난 공정 국소화 |
| `correction` (S7) | 인자 기반 보정 + 공간 detrend(원인 규명 후 적용) + 판정 영향 평가 |
| `judge` | 트레이 mode + (docv7−mode)>0.8mV 판정 |

### 구배 발생 추적 (S8) — 원인 공정 국소화

`OCV #01~#07`(중간 측정) + `PRIVT OCV`(전용)의 트레이 내 ΔOCV 링 필드를 공정
순서대로 계산한다. **링 진폭이 급증하고 최종 Δdocv7 필드와의 공간 상관이 처음
높아지는 단계 = 구배가 태어난 공정.** 예: OCV #03(1차 고온에이징 후)에서 링이
없다가 나타나면 → 1차 고온에이징 챔버 열구배가 원인. 결과는
`reports/figures/genesis_progression.png` 와 `report_run.md`의 표로 나온다.
(온도로 설명 안 되는 docv7 구배의 발생 시점을 OCV 값 자체로 짚는다 — 공간
detrend 같은 증상 제거가 아니라 원인 공정 지목.)

## 검증

```bash
python -m tests.test_pipeline          # 단위 + end-to-end 스모크 (pytest 불필요)
python -m tests.make_synthetic data/raw/synthetic.csv   # 합성 데이터 생성
```

합성 데이터는 7th Discharge 온도(전용OCV 직전 필드)에 구배를 심어두며,
스크리닝이 이 스텝을 1위로 검출하고 보정이 핫셀을 보존하는지 확인한다.

## 주의 (해석 시)

1. **3×3 이웃 평균 치환은 공간 상관을 부풀린다.** 모든 상관/회귀는 치환 셀
   포함/제외 두 벌로 산출된다(`screening_all` vs `screening_exclude_imputed`).
2. **온도는 충방전 구간에서만 관측**된다. 에이징 챔버 구배는 직접 볼 수 없어
   충방전 스텝 초반 온도를 간접 프록시로 쓴다(한계는 리포트에 명시).
3. **보정은 매끄러운 위치 성분만 제거**하고 고립 핫셀(진짜 불량)은 보존한다.
   보정 후 핫셀 잔차 유지율을 평가표에서 반드시 확인할 것.
