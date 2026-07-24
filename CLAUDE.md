# CLAUDE.md — 트레이 내 OCV(dOCV7) 구배 원인 분석 프로젝트 메모리

> 이 파일은 프로젝트의 목표·데이터·파이프라인·**실데이터 분석 결론**·그림 해설·
> 주의점·다음 단계를 담은 지속 메모리다. 새 세션은 이 파일부터 읽고 이어서 작업한다.

---

## 1. 목표

트레이(12×12=144셀) 단위 OCV 측정에서 나타나는 **셀 위치 의존 dOCV7 구배**(링형 등)의
**원인 공정을 규명**하고, 궁극적으로 그 원인 인자로 구배를 **보정**한다.
발표 자료(구배 패턴 + 원인)용 시각자료도 생성.

## 2. 데이터 (실 Export 파일)

- 형식: wide, **1행 = 셀 1개**. 규모: 셀 149,039 / 트레이 1,035 / 온도 스텝 19.
- 위치: `Cell 위치`(A01~L12, 알파벳=행 A→1) 또는 `Cell No`(1~144). 12×12.
- **온도 보유**: Low Current Inspection #01/02, Charge #01~07(최저/평균/최고),
  DisCharge #01~07(최저/평균/최고), PRIVT OCV #01~03. **중간 OCV #01~07은 온도 없음.**
- **OCV 값**: 중간 OCV #01~#07, 전용 PRIVT OCV #01~#03, `Delta OCV #07 DOCV`(=docv7 직접).
- 단위: ocv V→mV (`unit_scale_to_mv=1000`), `Delta OCV`는 이미 mV(`docv7_unit_scale_to_mv=1`).

### 공정 순서 (확정)
```
LCI>RT1>OCV1>C1>C2>OCV2>HT1>RT2>OCV3>C3>C4>OCV4>HT2>RT3>OCV5>
C5>C6>C7>OCV6>D1..D7>OCV7>RT4>PRVT1>RT5>PRVT2>RT6>PRVT3
```
C=충전, D=방전, RT=상온에이징, HT=고온에이징(60°C). SOC: OCV1≈0, OCV2/3=10(HT1 전/후),
OCV4/5=70(HT2 전/후), OCV6=100, OCV7·PRVT=30. **OCV2↔OCV3, OCV4↔OCV5 는 같은 SOC라
그 차이가 HT 에이징 효과만 격리(matched-SOC).** 전용OCV 측정 시 셀은 이미 상온 냉각.

## 3. 판정 로직
- docv7 = `Delta OCV #07` (또는 PRVT OCV1 − PRVT OCV3).
- 트레이별 docv7 **mode** 산출 → `(셀 docv7 − tray mode) > 0.8 mV` → 불량.

## 4. 파이프라인 (`analysis/`, CLI: `python -m analysis.run_pipeline`)
- `inspect` 칼럼 확인 → `make-config` 자동 매핑(Export 형식 인식) → `run` 전체 실행.
- S1 io_loader(로딩·wide→long·위치파싱) · S2 preprocess(결측/≤22°C→3×3 이웃평균 치환) ·
  S3 fields(트레이 Δ필드·링/방향/줄무늬 지표) · S4 features(온도 피처, min/mean/max·자기발열) ·
  S5 screening(온도×OCV 전수 상관) · S6 propagation(경로 회귀) ·
  **S8 genesis / genesis_math(구배 발생 추적 — 원인 공정 규명)** · S7 correction(보정).
- 판정 재현은 `judge.py`(tray mode + offset).

## 5. 실데이터 핵심 결론 (신뢰도 표기)

### [HIGH] docv7 구배는 무작위가 아니라 "공통 지문"이다
- 1035개 트레이 **평균 필드**(common fingerprint)에서 docv7이 **테두리 高 / 중앙 低 링**
  (±0.048 mV)을 보임 → 트레이별 랜덤이 아니라 **모든 트레이 공유 = 위치 고정형 원인**.
- OCV#01(충전 전)도 같은 테두리-高 링(±2.37) → **테두리 셀은 공정 시작부터 다르다.**
- 판정영향: ΔdOCV7 p2p median 0.286 mV (판정오프셋 0.8의 36%) — 실무적으로 유의미.

### [HIGH] 구배는 이르게 나타난다 — OCV#02 (트레이 73%)
- 부호-강건 집계(트레이별 발생단계 분포, n=518): **72.6%가 OCV#02**(1차 충전 직후, HT1 전),
  OCV#03 8.5%, OCV#06 6.4%. OCV#01은 0%.
- ⚠️ 단, OCV#02는 첫 SOC10(가파른 곡선) 측정이라 SNR이 급증하는 지점 → "그때 생성"인지
  "그때부터 측정 가능"인지는 미확정(§7 SOC 교란).

### [HIGH] 온도는 docv7을 설명하지 못한다 (하지만 OCV 레벨엔 강함)
- 경로 회귀 R²=0.058. dT_ocv1_minus_ocv3(P1 측정시점 온도차) 기여 1.3% → 측정 시 냉각돼 무의미.
- 방전 자기발열(dTrise::DisCharge#05)은 **d_ocv1/2/3(레벨)과 강한 음상관(−0.58)**이나,
  세 측정점을 똑같이 밀어 **차분 docv7에서 상쇄** → docv7 열은 약함(≤0.3).

### [MEDIUM] 공정 기여: 고온에이징이 최대 (matched-SOC 한정)
- 유발공정 분포: **HT1 46% · HT2 31% · RT4 22%**. 같은 SOC로 격리한 깨끗한 구간 중 HT 에이징 우세.
- 강한링 트레이(155개) 층화: 변화점 OCV#02, β 최대 HT2+RT3.
- ⚠️ matched-SOC 지표는 충·방전(형성/첫충전)을 후보에서 제외하므로 HT 쪽으로 편향될 수 있음.

### [MEDIUM] 통합 해석 — 테두리(edge) 위치 효과 + 고온에이징 증폭
- 테두리 셀이 중앙과 체계적으로 다름(초기부터). 60°C 고온에이징에서 테두리의 열환경
  (승온/냉각·복사)이 달라 SEI·자기방전 차이 → 테두리-高/중앙-低 링이 docv7에 각인.
- 랜덤 열 이력이 아니라 **트레이 기하에 고정된 효과** + HT 에이징 증폭이 유력.

### [HIGH] 강한 저SOC 구조 vs 약한 docv7은 별개일 수 있음
- Moran's I(공간 구조)는 **OCV#03(0.59)**·OCV#02·#04에서 강하고 SOC30에서 소멸.
  이는 저SOC 가파른 곡선이 용량 편차를 증폭한 것(용량/포메이션 구배). docv7과의 부호상관
  a_k는 약함(≤0.38) → 강한 초기 구조와 최종 docv7이 완전히 같은 것은 아님.

## 6. 산출 그림 카탈로그 (각 이미지 의미) — 발표용

- **common_fingerprint_meanfield.png** ★핵심: 단계별 **전 트레이 평균 필드**. docv7 칸의
  테두리-링이 "공통 지문(위치 고정 원인)"의 직접 증거. OCV#01 링, OCV#03 중앙-高도 보임.
- **onset_distribution.png** ★핵심: (좌) 트레이별 docv7 **발생단계 히스토그램**(OCV#02 73%),
  (우) **유발공정**(HT1 46%/HT2 31%/RT4 22%). 부호 상쇄 없는 정직한 집계.
- **storyboard.png**: 강한링 트레이(행) × OCV 단계(열, 좌→우 공정순) small-multiples,
  패널별 정규화(핫셀 클립). 각 트레이 패턴이 어느 단계에서 켜지는지. OCV#03 칸 구조 뚜렷.
- **alignment_moran.png**: a_k(누적 정렬, 부호有)와 Moran's I(공간 구조, 부호無)+변화점.
  Moran이 OCV#03 정점 후 소멸, a_k는 부호상쇄로 약함 → §7 참고.
- **genesis_progression.png**: 단계별 링 진폭 + docv7 상관(공정 순서). 저SOC서 진폭 큼.
- **beta_contributions.png**: 공정별 증분투영 β. 충·방전(파랑)은 크기 교란으로 폭주 →
  matched-SOC(빨강, HT1/HT2/RT4)만 유효, 부호상쇄 주의.
- **corr_heatmap_steps.png**: 온도 피처(세로) × OCV 타깃(가로) 트레이별 상관. 방전온도가
  d_ocv1/2/3엔 강하고 d_docv7엔 약함(온도의 docv7 상쇄를 시각화).
- **gallery_stage_<OCV>.png / gallery_docv7.png**: 단계별/최종 개별 트레이 필드 갤러리.

## 7. 방법론적 주의점 (해석 시 필수)
1. **부호 상쇄**: 링/역링이 트레이마다 공존 → 부호 있는 평균(a_k, β_k)은 상쇄돼 신호가 죽음.
   → **트레이별 개별 집계(onset_distribution)**·Moran's I(부호無)를 신뢰.
2. **저SOC / SNR 교란**: OCV#02~04는 곡선이 가팔라 구배가 증폭·고SNR. 발생단계가 저SOC로
   쏠리는 게 물리적 생성인지 측정가능성인지 미확정 → **SOC/용량 정규화** 필요(미구현).
3. **β 크기 교란**: 충·방전 증분은 SOC 변화로 절대값이 거대 → matched-SOC만 비교.
4. **docv7 상쇄**: 세 측정점 공통효과(온도 등)는 차분에서 사라짐 → 온도로 안 잡히는 게 정상.
5. **치환 부풀림**: 3×3 이웃평균 치환이 공간상관 과대평가 → 치환 포함/제외 두 벌 산출.
6. **matched-SOC 편향**: 유발공정 후보가 HT/RT뿐이라 충전(형성)이 원인이어도 HT로 잡힐 수 있음.

## 8. 미해결 질문 / 다음 단계 (우선순위)
1. **메타데이터 확보** ★: 공통 지문이 위치 고정이라 하니 **챔버/설비 호기·트레이 슬롯·
   포메이션 채널·Lot·공정경로**로 "어느 설비/위치"를 특정해야 함. (원인 확정의 마지막 퍼즐)
2. **SOC/용량 정규화**: dOCV/dSOC로 ΔOCV→Δ용량 환산 → 저SOC 교란 제거, 발생시점 확정.
3. **테두리 효과 정량화**: 테두리 거리 vs docv7 프로파일 + edge vs center 검정.
4. **OCV#01 링 ↔ docv7 링 상관**: 테두리 효과가 처음부터 있었는지 확인.
5. **강한 초기 구조(OCV#03) 자체 발생추적**: docv7 말고 OCV#03을 타깃으로 → 포메이션/충전 원인.
6. **판정 보정**: 원인 확정 후 factor 보정. spatial detrend는 증상 제거라 원인 규명 전엔 보류.

## 9. 실행법
```bash
pip install -r requirements.txt
python -m analysis.run_pipeline inspect     --input "<파일>.xlsx"
python -m analysis.run_pipeline make-config  --input "<파일>.xlsx" --output analysis/column_config.yaml
# column_config.yaml 에서 unit_scale_to_mv=1000, docv7_unit_scale_to_mv=1, position_from 확인
python -m analysis.run_pipeline run          --config analysis/column_config.yaml
# 산출: reports/report_run.md, reports/figures/presentation/*, data/processed/*.csv
python -m tests.test_pipeline               # 단위+E2E 테스트 (합성 데이터)
```

## 10. 한 줄 결론
> **dOCV7 구배는 무작위가 아니라 "테두리 高/중앙 低" 공통 지문(±0.048mV)이며, 공정 초반
> (OCV#02, 저SOC)부터 나타나 고온에이징이 증폭한다. 측정된 온도로는 설명되지 않으며
> (R²0.06, 측정 시 냉각), 트레이 기하에 고정된 테두리(위치) 효과가 유력 원인이다. "어느
> 설비/위치"의 확정은 메타데이터가 필요하다.**
