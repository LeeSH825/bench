# P0_00 Deprecated Assumptions

> 작성일: 2026-07-30  
> 표지 규칙: **[확인]** 실험·실측, **[문헌]** 논문·공식 자료, **[분석]** 수식·구조 해석, **[가설]** 검증 대상, **[결정]** 설계 선택, **[보류]** 근거 부족·후속 범위

| 항목 | 내용 |
|---|---|
| 목적 | 현재 연구의 기본 해결책에서 폐기하거나 보류할 가정을 명시하여 같은 방향으로 회귀하지 않도록 한다. |
| 입력 근거 | S0–S4 |
| 확인된 내용 | missing/imputation을 정당화할 핵심 결측이 관찰되지 않았고 단순 enhancement의 명확한 최종 이득이 확인되지 않았음 |
| 미확인 내용 | 향후 다른 데이터셋에서 통신 손실·결측 문제가 새로 나타날지, hardware에서 SNN 이점이 생길지 |
| 결정 상태 | LOCK — 아래 항목은 기본안에서 제외 |
| 남은 TBD | 향후 실제 데이터 분포가 바뀌어 결측·통신 손실이 새로 확인되는 경우에만 별도 문제로 재검토 |
| 다음 Gate | Phase 1 design review에서 deprecated feature가 sensor/estimator input에 다시 들어가지 않았는지 검사 |


## 1. 폐기·보류 목록

| ID | 폐기 가정/해결책 | 근거 | 문제점 | 대체 기본안 | 상태 |
|---|---|---|---|---|---|
| DA-001 | 불규칙 missing measurement가 핵심 문제 | [확인] 실제 데이터에서 핵심 결측 미관찰 | 문제 정의와 데이터 불일치 | timestamped asynchronous multirate 처리 | REJECTED |
| DA-002 | sparse mask/imputation을 연구의 주력 novelty로 사용 | DA-001 | 존재하지 않는 결측을 인위적으로 만듦 | validity는 실제 eclipse/outage/saturation에만 사용 | REJECTED |
| DA-003 | filter prediction을 pseudo-measurement로 삽입 | [분석] 독립 정보가 아니며 correlation을 무시 | overconfidence와 double counting | propagation은 propagation으로만 처리 | REJECTED |
| DA-004 | measurement MSE 감소가 attitude MSE 개선을 보장 | [확인] 단순 enhancement의 명확한 최종 개선 미확인 | bias/time correlation/filter 목적 불일치 | state/calibration + Q/R/gate intervention | REJECTED |
| DA-005 | 저가 IMU를 고정 Gaussian white noise 하나로 대표 | [분석] bias drift·temperature·vibration·scale를 누락 | 연구 문제를 제거함 | 계층형 error catalog | REJECTED |
| DA-006 | Split `G1/G2`를 실제 `Q/R`로 부름 | [분석] factor scale·identifiability ambiguity | 물리 claim과 consistency 분석이 성립하지 않음 | latent factor 또는 explicit structured Q/R | REJECTED |
| DA-007 | scalar SoW 하나가 모든 센서 regime을 충분히 표현 | [가설] 미검증 | 독립 process/measurement 변화가 collapse | scalar→event-local vector ablation | TBD-BY-EXPERIMENT |
| DA-008 | innovation magnitude 하나로 원인을 식별 | [분석] 서로 다른 Q/R 변화가 비슷한 innovation RMS 가능 | 구조적 비식별성 | sensor identity, temporal statistics, telemetry 포함 | REJECTED AS SOLE INPUT |
| DA-009 | SNN 사용만으로 저전력·저지연 주장 | [분석]/S3 | encoding timestep·hardware에 따라 반대 가능 | event count/latency/hardware measurement | REJECTED |
| DA-010 | 전체 Kalman gain 생성기를 곧바로 SNN으로 전환 | [가설] ANN/context 유효성 미확인 | 연구 위험과 디버깅 난도 증가 | fast detector/gate 후보로 제한 | DEFERRED |
| DA-011 | star tracker output을 ground truth로 사용 | S0 | sensor noise/latency/outage를 숨김 | simulator truth attitude를 평가 기준으로 사용 | REJECTED |
| DA-012 | MRP-EKF와 MEKF 결과를 조건 차이 없이 직접 우열 비교 | S3 | state/convention/knowledge가 다름 | legacy baseline로 분리 | REJECTED |
| DA-013 | window-level random split | S2–S4 | orbit phase/event timing leakage | trajectory-level split | REJECTED |
| DA-014 | 모든 외란·오차를 첫 실험부터 동시 주입 | S4 | 실패 원인 식별 불가 | Tier 0→1→2 단계 도입 | REJECTED |
| DA-015 | accelerometer를 즉시 spacecraft attitude update로 사용 | [분석] 궤도 free-fall에서 중력 방향 센서로 단순 해석 불가 | terrestrial IMU 관례의 오적용 | vibration/context proxy로만 후보 | DEFERRED |

## 2. 재도입 조건

폐기 항목은 단지 “싫어서” 제외한 것이 아니다. 다음 조건이 충족되면 별도 연구 질문으로 재도입할 수 있다.

- missing-aware 처리: 실제 flight/communication dataset에서 비정상 결측률과 성능 영향이 확인될 때
- pseudo-measurement: 독립 외부 모델·별도 센서에서 생성되고 cross-covariance를 모델링할 때
- denoising front-end: 최종 attitude/bias metric에서 반복적 이득과 latency 비용을 함께 보일 때
- SNN: ANN/classical detector가 통과한 동일 기능을 event-driven으로 수행하고 측정 가능한 효율 지표가 있을 때
- accelerometer update: 비중력 specific force와 orbit/dynamic model을 명시적으로 다루는 별도 estimator를 설계할 때

## 3. 코드 리뷰 금지 패턴

```text
imputed_measurement = model_prediction       # 독립 측정처럼 update 금지
truth_attitude = star_tracker_output         # 평가 truth로 사용 금지
Q_true = split_branch_1                      # 식별성 검증 없이 명명 금지
R_true = inverse(split_branch_2)             # 동일
random_split(overlapping_windows)            # trajectory leakage
claim_low_power(because='SNN')               # 측정 없이 주장 금지
```
