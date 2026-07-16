# tray_ocv — 트레이 내 OCV 구배 원인 분석·보정 파이프라인

트레이 단위 전용OCV / dOCV7 측정에서 나타나는 **셀 위치 의존 전압 구배**의
원인을 공정 단계별 온도 데이터로 규명하고, 원인 인자로 구배를 **보정**한다.

- 배경·인자 정리: [`ocv_gradient_root_cause.md`](ocv_gradient_root_cause.md)
- 분석 방법론: [`analysis_plan_temperature_tracing.md`](analysis_plan_temperature_tracing.md)
- 코드 구조 설계: [`code_design_pipeline.md`](code_design_pipeline.md)

## 데이터 형식

wide 포맷 — **1행 = 셀 1개**, 각 공정 스텝이 **온도 칼럼**, 전용OCV1~3 칼럼 포함.
트레이 내 위치는 `ROW`/`COL` 칼럼 또는 `CELL_ID`에서 정규식 파싱으로 얻는다.

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
- `judge.unit_scale_to_mv`: OCV 원시 단위가 V면 **1000**, 이미 mV면 1.
- `judge.docv7_from`: `ocv1_minus_ocv3`(기본) 또는 `direct`(docv7 칼럼 제공 시).
- `preprocess.temp_min_valid`: 기본 22 (이 값 **이하** 삭제 후 3×3 치환).
- `tray_shape.n_rows`: 기본 12. `n_cols`는 null이면 자동 추론.

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
| `correction` (S7) | 인자 기반 보정 + 공간 detrend 폴백 + 판정 영향 평가 |
| `judge` | 트레이 mode + (docv7−mode)>0.8mV 판정 |

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
