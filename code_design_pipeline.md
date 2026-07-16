# 온도–OCV 구배 분석·보정 파이프라인 — 코드 구성 설계

> 목표: 충방전 구간 온도 데이터를 전처리·분석하여 전용OCV/dOCV7 트레이 내 구배의
> 원인 인자를 찾고, **그 인자를 입력으로 하는 구배 보정 모델**을 만든다.
> (분석 방법론은 `analysis_plan_temperature_tracing.md`, 본 문서는 코드 구조.)

---

## 0. 확정된 공정 순서 및 SOC 프로파일 (canonical step list)

| # | 공정 (Process Name) | SOC / 비고 |
|---|---|---|
| 1 | LCI | |
| 2 | 1st RT Aging | |
| 3 | 1st Charging | → SOC ~10% |
| 4 | 2nd Charging | ↗ |
| 5 | 1st HT Aging | 60 °C, SOC ~10% |
| 6 | 2nd RT Aging | |
| 7 | 3rd Charging | → SOC ~70% |
| 8 | 4th Charging | ↗ |
| 9 | 2nd HT Aging | 60 °C, SOC ~70% |
| 10 | 3rd RT Aging | |
| 11 | 5th Charging | → SOC 100% |
| 12 | 6th Charging | ↗ |
| 13 | 7th Charging | ↗ |
| 14 | 1st Discharge | → SOC 30% |
| 15~20 | 2nd~7th Discharge | ↘ |
| 21 | 에이징 1일 → **전용OCV1** → 에이징 2일 → **전용OCV2** → 에이징 1일 → **전용OCV3** | SOC 30에서 측정 |

- **온도 데이터는 충방전 구간(Charging/Discharge 스텝)에서만 존재** → 에이징 챔버
  내 구배는 직접 관측 불가. 단, 각 충방전 스텝의 **초반 온도는 직전 에이징에서
  나온 직후의 잔존 온도 필드**이므로, 에이징 구배의 간접 프록시로 쓴다
  (특히 7th Discharge 종료 필드가 전용OCV 직전의 마지막 관측).
- dOCV/dT는 SOC 의존 → 전용OCV는 SOC 30에서 측정되므로 P1 경로 해석 시
  SOC 30의 엔트로피 계수를 기준으로 판단.

---

## 1. 디렉토리 / 모듈 구조

```
tray_ocv/
├── data/
│   ├── raw/                  # 원본 (읽기 전용, git-ignore 권장 — 용량/보안)
│   ├── interim/              # 정규화된 long-format parquet (스테이지별 캐시)
│   └── processed/            # 분석 준비 완료 테이블 (셀×스텝 피처 매트릭스)
├── analysis/
│   ├── config.py             # 경로·상수·공정 스텝 canonical 목록
│   ├── io_loader.py          # [S1] 원본 로딩 + 스키마 정규화
│   ├── preprocess.py         # [S2] QC(결측/≤22°C) + 3×3 이웃 평균 치환
│   ├── fields.py             # [S3] 트레이 필드 구성 + Δ필드 + 구조 지표
│   ├── features.py           # [S4] 셀×스텝 온도 피처 추출
│   ├── screening.py          # [S5] 전 스텝 × OCV 전수 상관 스크리닝
│   ├── propagation.py        # [S6] 경로 판별 회귀 (P1/P2/P3) + 전파 사슬
│   ├── correction.py         # [S7] 구배 보정 모델 + 판정 영향 평가
│   ├── viz.py                # 갤러리·히트맵 (모든 스테이지 공용)
│   └── run_pipeline.py       # CLI: 스테이지 단위 실행/재실행
├── reports/
│   ├── figures/              # png 출력
│   └── report_*.md
└── tests/
    └── test_preprocess.py    # 치환 로직 등 핵심 유닛 테스트
```

원칙:
- **스테이지별 캐시**: 각 모듈은 `data/interim/*.parquet`을 읽고 다음 캐시를 쓴다.
  원본 재로딩 없이 하류만 재실행 가능 (`run_pipeline.py --from S5` 식).
- **모든 값에 플래그 동반**: 치환된 셀, 결측이던 셀, 저표본 셀은 boolean 컬럼으로
  끝까지 따라간다 (감도분석용).
- 시각화는 계산과 분리 (`viz.py`는 processed 테이블만 읽음).

## 2. 데이터 모델 (핵심 테이블 스키마)

### T1. `temp_long` — 온도 정규화 테이블 (S1 출력)
```
lot | tray_id | row | col | cell_id | step_name | step_order | timestamp | temp_raw
```
- `step_name`은 config의 canonical 목록으로 매핑(오탈자·명명 차이 흡수)
- 시계열이면 timestamp 유지, 스텝당 1값이면 timestamp=null

### T2. `temp_clean` — QC·치환 후 (S2 출력)
```
... | temp | qc_flag {ok, was_missing, was_le22, imputed_fail} | imputed(bool)
```

### T3. `ocv_cell` — 전압 테이블 (S1 출력)
```
lot | tray_id | row | col | cell_id | ocv1 | ocv2 | ocv3 | docv7 | judge(bool)
```
- `docv7_tray_mode`, `docv7_dev = docv7 − mode` 파생 컬럼 포함 (판정 로직 재현)

### T4. `cell_step_features` — 분석용 피처 매트릭스 (S4 출력, 셀 × 스텝별 피처 wide)
```
cell키 + ΔT_end(step), ΔT_mean(step), ΔT_start(step), ΔT_rise(step), ΔHeatExposure(step), ...
```

## 3. 모듈별 상세 설계

### S1 `io_loader.py`
- 입력 포맷 자동 감지(csv/xlsx, 인코딩), 컬럼명 → 표준 스키마 매핑 테이블
- 셀ID ↔ (tray, row, col) 조인 무결성 검사: 중복/미매칭 리포트 출력
- 산출: `temp_long.parquet`, `ocv_cell.parquet` + `data/README.md`(스키마 기록)

### S2 `preprocess.py` — 사용자 지정 전처리 규칙
1. **무효화**: `temp_raw` 결측 또는 **≤ 22 °C** → NaN 처리, `qc_flag` 기록
   - 22 °C 컷 전후의 값 분포 히스토그램을 자동 저장 (컷이 실데이터를 자르는지 확인)
2. **3×3 이웃 평균 치환**: 같은 (tray, step) 필드 내에서 대상 셀 주변 9셀
   (자신 제외 8셀 + 대각 포함) 중 **유효한 셀만의 평균**으로 치환
   - 모서리/테두리: 존재하는 이웃만 사용 (최소 유효 이웃 수 기준, 기본 3개)
   - 이웃도 대부분 무효인 덩어리(cluster): 1회 반복 치환 후에도 안 되면
     트레이 중앙값으로 폴백하고 `imputed_fail` 플래그
   - 구현: 필드를 2D 배열로 피벗 → NaN-aware 3×3 uniform filter (scipy) → 벡터화
3. **치환 통계 리포트**: 트레이·스텝별 치환 비율. 치환율 > 임계(예: 30%) 필드는
   분석 제외 목록에 등재
4. ⚠️ **설계 주의 — 치환은 공간 상관을 부풀린다**: 치환값은 이웃 평균이므로 필드가
   인위적으로 매끄러워짐 → 이후 모든 상관/회귀는 (a) 전체 셀, (b) `imputed=False`만
   두 가지로 자동 병행 산출 (`sensitivity="exclude_imputed"` 공통 옵션)

### S3 `fields.py`
- 피벗: (tray, step) → 12×N 2D 필드
- Δ필드: `ΔX = X − median_tray(X)` (트레이 간 레벨 제거, 트레이 내 구배만)
- 구조 지표 함수 (온도·전압 공용):
  `amplitude_p2p`, `amplitude_p95p5`, `ring_score`(테두리−중앙),
  `grad_row`/`grad_col`(선형회귀 계수), `stripe_score`(열별 평균 분산)
- 기존 [27b] 갤러리와 동일한 정의 유지 → 결과 직접 비교 가능

### S4 `features.py` — 온도 피처 추출 (환경 vs 자기발열 분리 포함)
충방전 중 셀 온도 = **환경(챔버/에이징 잔열) + 자기발열(I²R)**. 두 성분은 원인이
다르므로 분리 피처로:
- `T_start(step)`: 스텝 초반 온도 = **직전 에이징에서 나온 잔존 온도 필드** (환경 프록시)
- `T_rise(step)`: 스텝 내 승온량(말−초) = **자기발열 프록시** (셀 DCIR 편차 반영)
- `T_end(step)`, `T_mean(step)`: 종합
- `HeatExposure(step)`: ∫exp(−Ea/RT)dt (Ea 40/50/60 kJ/mol 감도), 시계열 없으면
  T_mean × duration 근사
- 에이징 구배 간접 추정: `T_start(다음 충방전) − T_end(직전 충방전)` 의 필드 변화
- 시계열이 없고 스텝당 1값뿐이면 `T_start/T_rise` 생략하고 자동으로 축소 모드

### S5 `screening.py` — 전 스텝 × OCV 전수 상관 (계획 3단계)
- 트레이별 공간 상관 `r(tray; step, feature, target)` → 분포 요약(중앙값, IQR,
  부호 일관성 비율)
- 산출: **스텝×피처 × 타깃(ΔOCV1/2/3, ΔdOCV7) 상관 히트맵** + 상관 상위 조합 표
- 다중비교: BH-FDR + permutation null (트레이–필드 매칭 셔플)
- 스텝 간 ΔT 필드 자기유사성 행렬 → 군집화 → 군집 대표 간 부분상관
- 구조 지표 수준 상관 (링점수↔링점수 등) 병행 → 패턴 유형별 기원 분리

### S6 `propagation.py` — 원인 확정 (계획 4~6단계)
- 혼합효과 회귀: `ΔdOCV7 ~ β₁·(ΔT_ocv1시점 − ΔT_ocv3시점) + β₂·ΔHeatExposure(에이징) + β₃·ΔHeatExposure(HT과도) + (1|tray)`
  - 온도 데이터가 충방전 구간뿐이므로 P1·P2 항은 S4의 프록시(7th Discharge 종료
    필드, 인접 스텝 보간)로 구성하고 프록시 한계를 리포트에 명시
- ΔOCV1→2→3 진행 양상 분석 (P2면 누적 증가, P3이면 OCV1에서 이미 완성)
- 검증: 홀드아웃(트레이 8:2), 부호 일관성, permutation
- 산출: 경로별 기여율 표 + 검증 성적표

### S7 `correction.py` — **최종 목표: 원인 인자 기반 구배 보정**
1. **보정 모델 (1안, 물리 기반)**: S6에서 확정된 인자로
   `docv7_corr = docv7 − f(온도 인자)` (f = S6 회귀의 고정효과 예측치)
   - 학습은 **정상 셀만**으로 (판정 불량·핫셀 제외 — 불량 신호를 보정으로 지우지 않도록)
2. **보정 모델 (2안, 폴백/비교 기준)**: 온도 설명력이 부족할 때 —
   트레이별 공간 저주파 성분 제거 (2D 저차 다항식 or robust 스무딩 잔차)
   - 1안 vs 2안 vs 무보정 3자 비교로 "원인 인자 보정의 실익"을 정량화
3. **안전장치**: 보정은 **매끄러운(저주파) 위치 성분만** 제거해야 함.
   고립 핫셀(진짜 자기방전 불량)의 잔차는 보정 전후 보존되는지 자동 검사
   (알려진 불량핀 셀들의 `docv7_dev` 변화량 리포트)
4. **판정 영향 평가**:
   - 보정 전/후 구배 진폭 (p2p, 링점수) 감소율
   - 판정 뒤집힘 분석: 양→불, 불→양 셀 수와 위치 분포
   - 트레이 mode 재계산 영향 (mode 자체가 보정으로 이동하는 효과 포함)
   - 홀드아웃 랏/트레이에서의 일반화 성능
5. 산출: 보정 계수(운영 적용용), 보정 전후 갤러리, 판정 영향 표

### `viz.py`
- `plot_tray_field(field, ...)`: 단일 트레이 히트맵 (기존 [27b] 스타일, 공통 컬러스케일)
- `plot_gallery(fields, ...)`: 트레이 갤러리 (온도 vs OCV 나란히 비교 모드)
- `plot_corr_heatmap(...)`: 스텝×타깃 상관 히트맵
- `plot_correction_summary(...)`: 보정 전후 비교

### `run_pipeline.py`
```
python -m analysis.run_pipeline --stage all          # 전체
python -m analysis.run_pipeline --from S5            # 스크리닝부터 재실행
python -m analysis.run_pipeline --stage S2 --lot X   # 랏 단위 부분 실행
```
- 각 스테이지는 입력 캐시 존재 검사 → 없으면 상류 자동 실행
- 난수 시드 고정 (permutation 재현성)

## 4. 테스트 계획 (핵심만)

- `test_preprocess`: 3×3 치환 — 모서리/테두리, 유효 이웃 부족, 덩어리 결측,
  22 °C 경계값 케이스
- `test_fields`: 링점수/방향구배가 합성 필드(알려진 링형·일방향)에서 정답 부호·크기
- `test_screening`: 합성 데이터(온도→OCV 인과 심어놓은 가짜 랏)에서 해당 스텝이
  1위로 검출되는지 end-to-end 스모크 테스트
- `test_correction`: 합성 핫셀이 보정 후에도 검출되는지 (안전장치 검증)

## 5. 구현 순서

| 순서 | 작업 | 완료 기준 |
|---|---|---|
| 1 | config + io_loader (S1) | 실데이터 로딩, 조인 무결성 리포트 |
| 2 | preprocess (S2) + 유닛 테스트 | 치환 통계 리포트 출력 |
| 3 | fields + features (S3·S4) | 온도 필드 갤러리 1차 출력 (눈 검증) |
| 4 | screening (S5) | 상관 히트맵 → **여기서 1차 결론 공유** |
| 5 | propagation (S6) | 경로 기여율 + 검증 성적표 |
| 6 | correction (S7) | 보정 전후 판정 영향 표 → **최종 산출물** |

- 4단계(히트맵)가 첫 번째 의사결정 지점: 온도 설명력이 확인되면 5·6으로,
  약하면 비온도 인자(포메이션 채널 등) 데이터 추가 확보로 분기.

## 6. 착수 전 확인 필요 사항 (데이터 업로드 시 함께)

1. 온도 데이터 형태: 셀별인가, 채널/지그별인가? 스텝당 시계열인가 대표값 1개인가?
2. 충방전 장비의 셀 온도 센서 위치 (셀 표면 직접? 지그 온도?)
3. 전용OCV1~3 원본 (셀 단위, 트레이 좌표 포함) 동시 업로드
4. 트레이 크기 확정 (12행 × 몇 열?)
5. "22 °C 이하 삭제" 근거 확인용 — 온도 분포를 보고 컷라인이 적절한지 1회 검토
