"""검증용 합성 데이터 생성 (wide: 셀행 × 공정 온도칼럼 + 전용OCV).

온도→OCV 인과를 의도적으로 심는다:
  - 일부 트레이에 링형 온도 필드(테두리 高) 부여
  - 7th Discharge 온도(전용OCV 직전) 편차가 docv7 구배로 이어지게 (P1 프록시)
  - 고립 핫셀 몇 개 (진짜 자기방전 불량)
  - 결측 + 22°C 이하 값 일부 주입 (전처리 테스트용)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from analysis.config import CANONICAL_STEPS


def make(n_trays=30, n_rows=12, n_cols=8, seed=0, coupling_mv_per_K=0.15):
    rng = np.random.default_rng(seed)
    steps = CANONICAL_STEPS
    rows = []
    rr, cc = np.indices((n_rows, n_cols))
    # 테두리 거리 (링 필드용): 테두리=0, 중앙이 클수록 큼
    edge = np.minimum.reduce([rr, n_rows - 1 - rr, cc, n_cols - 1 - cc]).astype(float)
    ring = (edge.mean() - edge)  # 테두리 高(+), 중앙 低(-)
    ring /= (np.abs(ring).max() + 1e-9)

    for t in range(n_trays):
        tray_id = f"T{t:03d}"
        base_soc_ocv = 3.35  # V 근처 (SOC30)
        # 이 트레이의 링 강도 (간헐성: 절반 정도만 강함)
        ring_amp_C = rng.choice([0.0, 0.0, 1.0, 2.0, 3.0])  # 테두리-중앙 온도차(°C)
        sign = rng.choice([1.0, -1.0])  # 링형/역링형
        env = 25.0 + rng.normal(0, 0.3)  # 트레이 평균 온도
        # 7th Discharge 온도 필드 = 환경 + 링
        T_last = env + sign * ring_amp_C * ring + rng.normal(0, 0.15, size=ring.shape)

        for r in range(n_rows):
            for c in range(n_cols):
                cell_id = f"{tray_id}R{r+1:02d}C{c+1:02d}"
                rec = {"CELL_ID": cell_id, "TRAY_ID": tray_id,
                       "ROW": r + 1, "COL": c + 1, "LOT": "L0"}
                # 각 스텝 온도: 환경 + 충방전시 자기발열 + 약한 링 잔재
                for s in steps:
                    if "Charging" in s:
                        temp = env + rng.normal(1.5, 0.4)  # 자기발열
                    elif "Discharge" in s:
                        temp = env + rng.normal(1.0, 0.4)
                    else:
                        temp = env + rng.normal(0, 0.3)
                    temp += 0.3 * sign * ring_amp_C * ring[r, c]
                    rec[s] = temp
                # 7th Discharge 를 링 필드로 덮어써 전용OCV와 커플링
                rec["7th Discharge"] = T_last[r, c]
                # docv7 = 커플링(측정시점 온도차) + 셀 잡음
                dT = T_last[r, c] - np.median(T_last)
                docv7_mv = coupling_mv_per_K * dT + rng.normal(0, 0.05)
                # ocv1/ocv3 로 환원 (docv7 = ocv1 - ocv3), mV→V
                ocv3 = base_soc_ocv + rng.normal(0, 0.0005)
                ocv1 = ocv3 + docv7_mv / 1000.0
                ocv2 = ocv3 + 0.6 * docv7_mv / 1000.0
                rec["전용OCV1"] = ocv1
                rec["전용OCV2"] = ocv2
                rec["전용OCV3"] = ocv3
                rows.append(rec)

        # 고립 핫셀 2개: docv7 크게 (진짜 불량)
        for _ in range(2):
            ri, ci = rng.integers(0, n_rows), rng.integers(0, n_cols)
            cid = f"{tray_id}R{ri+1:02d}C{ci+1:02d}"
            for rec in rows:
                if rec["CELL_ID"] == cid:
                    rec["전용OCV1"] = rec["전용OCV3"] + (1.5 + rng.random()) / 1000.0
                    break

    df = pd.DataFrame(rows)
    # 결측/22도 이하 주입 (온도 칼럼에만)
    temp_cols = [s for s in steps]
    flat = df[temp_cols].to_numpy(dtype=float).copy()
    n = flat.size
    idx_missing = rng.choice(n, size=int(n * 0.01), replace=False)
    flat.flat[idx_missing] = np.nan
    idx_low = rng.choice(n, size=int(n * 0.01), replace=False)
    flat.flat[idx_low] = 20.0  # 22 이하
    df[temp_cols] = flat
    return df


if __name__ == "__main__":
    import sys
    out = sys.argv[1] if len(sys.argv) > 1 else "data/raw/synthetic.csv"
    make().to_csv(out, index=False, encoding="utf-8-sig")
    print("wrote", out)
