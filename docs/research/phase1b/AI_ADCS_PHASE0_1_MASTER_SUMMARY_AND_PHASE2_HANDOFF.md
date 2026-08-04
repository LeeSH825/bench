# AI-ADCS Phase 0–1 Master Summary and Phase 2 Handoff

- 작성 기준일: 2026-08-02
- 프로젝트: AI-ADCS / Adaptive Neural MEKF
- Phase 0–1 classical evidence 실행 상태: **완료**
- 최종 P1 Exit: **CONDITIONAL_GO**
- Phase 2 구현 상태: **미착수**

> Authority note: 이 문서는 navigation/handoff summary다. 정확한 수치에는
> machine-frozen result JSON과 specialized final report가, 현재 판정에는
> `P1_EXIT_REVIEW_UPDATED.md`가 항상 우선한다.

## 1. 연구 목표

저가 IMU와 다중 자세센서의 시간변화 오차 및 신뢰도 변화를 반영하는 적응형 신경 칼만 필터 기반 위성 자세 추정 방법을 설계하고, classical MEKF, classical adaptive filter, neural baseline 및 proposed model을 동일한 sensor realization과 평가 계약에서 비교한다.

Handoff에 기록된 연구 제목(Phase 0–1 decision ledger의 별도 title-lock 항목은 아님):

- 국문: 기계학습 기반 저가 IMU 보정과 적응형 신경 칼만 필터를 이용한 위성 자세 추정
- 영문: Spacecraft Attitude Estimation Using Machine-Learning-Based Low-Cost IMU Compensation and Adaptive Neural Kalman Filtering

Phase 0–1에서는 neural model을 구현하지 않고, 연구 문제와 classical benchmark의 존재를 먼저 검증했다.

## 2. Phase 0 — 연구·수학 계약

Nominal state:

```text
q_NB
b_g
```

Local error state:

```text
delta_x = [delta_theta, delta_b_g] in R^6
```

Convention:

```text
scalar-first Hamilton quaternion
active body-to-navigation q_NB
right-multiplicative local error
R_NB: body coordinates -> navigation coordinates
C_BN = R_NB.T
```

Information boundary:

```text
truth/environment -> sensor output -> onboard estimator
```

Estimator 입력에서 제외:

```text
true attitude
true bias
actual Q/R multiplier
event label/window
future sample
oracle context
evaluation metric
```

Oracle 정보는 simulation-only sidecar로 분리한다.

## 3. Phase 1A — Classical Foundation

### 3.1 MEKF math/core

구현:

```text
bench/estimators/mekf.py
```

검증:

- quaternion/SO(3)
- gyro propagation
- gyro bias
- Van Loan discretization
- body-vector Jacobian
- sun tangent Jacobian
- star-tracker tangent residual
- Joseph covariance update
- right injection/reset
- exact-pi q/-q determinism
- immutable/read-only state
- strict SPD Cholesky
- no inverse/pinv/jitter/clipping

결과:

```text
55 passed
```

### 3.2 Typed event and replay

구현:

```text
bench/tasks/generator/mekf_events.py
bench/tasks/generator/unit_st_synthetic.py
```

핵심:

```text
typed gyro/ST events
zero latency
gyro-before-ST order
whole-trajectory split
canonical serialization
semantic hashes
strict generator identity
direct replay
```

결과:

```text
55 passed
```

### 3.3 Basilisk UNIT-ST

구현:

```text
bench/tasks/generator/basilisk_unit_st.py
```

확정:

```text
q_NB = normalize(MRP2EP(sigma_BN))
R_NB = quat_to_dcm(q_NB)
R_NB = MRP2C(sigma_BN).T
```

`omega_BN_B`는 body-frame rad/s이며 Gate A right propagation과 동일 부호다.

Project-owned sensor model:

```text
omega_m = omega_true + b_g + n_g
q_ST = q_true otimes Exp(n_ST)
```

Basilisk built-in star tracker는 사용하지 않았다. right-local covariance, seed, hash, q/-q representation과 estimator contract를 직접 통제하기 위해서다.

결과:

```text
67 passed
```

### 3.4 Canonical metrics

구현:

```text
bench/metrics/mekf.py
```

Metric:

```text
attitude geodesic/right-local error
bias error/RMSE
ST NIS
6D NEES
SPD diagnostics
chi-square consistency summary
```

결과:

```text
43 passed
```

### 3.5 Adapter and runner integration

구현:

```text
bench/models/mekf.py
bench/tasks/bench_generated.py
bench/models/registry.py
bench/runners/run_suite.py
```

고정 ID:

```text
task_family = mekf_unit_st_v1
model_id = mekf_event_replay_v1
```

검증:

```text
direct replay == bridge replay == runner replay
q/b/P/r/S exact equality
fresh generation == verified cache hit
truth-free estimator boundary
no dense float32 coercion
lossless artifact
```

결과:

```text
D1 bridge: 24 passed
CP4 integration: 22 passed
```

Phase 1A foundation: COMPLETE.

## 4. Phase 1B Step 1 — UNIT-ST Classical Benchmark

구현:

```text
bench/tasks/generator/unit_st_regimes.py
bench/experiments/phase1b_unit_st_classical.py
bench/configs/suite_phase1b_unit_st_classical.yaml
```

비교 정책:

```text
F-BASE
F-TUNED
F-MIS-Q-LOW/HIGH
F-MIS-R-LOW/HIGH
ORACLE-QR
WRONG-SIDE
```

실험:

```text
C1 stationary
C2 gyro process uncertainty
C3 ST inlier reliability
C5 RMS-matched process/measurement pair
long horizon
```

결과:

```text
9 conditions
N=50 per condition
1,950 policy/trajectory records
```

Primary baseline:

```text
F-BASE
```

Frozen F-TUNED:

```text
s_Qg=0.125
s_Qb=0.125
s_R_ST=8.0
```

F-TUNED은 short-horizon error를 조금 낮추지만 과도하게 보수적이고 600 s long horizon에서 penalty를 보였다.

핵심 결론:

- C3 measurement reliability 변화는 강한 fixed-filter degradation을 만든다.
- Correct R-side oracle은 accuracy와 consistency를 개선한다.
- C2 process uncertainty는 주로 consistency를 악화시킨다.
- Oracle Qg는 NEES를 개선했지만 attitude RMSE 개선은 확인되지 않았다.
- C5에서 scalar innovation RMS alone은 원인 구분에 부족했지만 raw gyro evidence는 유용했다.

Status:

```text
PASS_P1B_STEP1_UNIT_ST_CLASSICAL
```

## 5. Phase 1B Step 2 — Sensor Fusion and C4

구현:

```text
bench/tasks/generator/mekf_fusion_events.py
bench/tasks/generator/phase1b_sensor_fusion.py
bench/metrics/mekf_fusion.py
bench/experiments/phase1b_sensor_fusion_c4.py
```

Schema:

```text
p1b-mekf-fusion-events-v1
```

Order:

```text
gyro -> magnetometer -> sun -> star tracker
```

### MAIN-FUSION

```text
gyro + mag + sun + low-rate ST
N=50
divergence=0
```

Settled:

```text
mag NIS/DOF=1.023
sun NIS/DOF=1.000
ST NIS/DOF=1.092
full NEES/DOF=1.873
```

Sensor consistency는 matched하지만 posterior state는 과신했다.

### STRESS-MAG

```text
gyro + magnetometer
N=50
```

Result:

```text
magnetic-axis weak direction RMS=0.195676 rad
observable-plane RMS=0.001331 rad
```

단일 자기벡터의 weak/unobservable direction을 숨기지 않고 확인했다.

### C4

```text
slow: gyro-bias random-walk intensity alpha_b
fast: magnetometer covariance alpha_R_mag
```

Full oracle vs F-BASE:

```text
slow bias RMSE improvement=28.56%
fast attitude peak improvement=32.57%
mag NIS improvement=47.20%
NEES improvement=96.32%
```

Status:

```text
PASS_P1B_STEP2_SENSOR_FUSION_C4
```

Initial P1 Exit:

```text
CONDITIONAL_GO
```

## 6. P1 Exit Condition Closure

Independent data:

```text
train=30
validation=20
stationary confirmation=50
C4 confirmation=50
```

Diagnosis on the independent validation split, F-BASE:

```text
initial full NEES/DOF=15.558045
settled full NEES/DOF=1.906245
settled attitude marginal NEES/DOF=1.434813
settled bias marginal NEES/DOF=2.744853
settled attitude-bias P cross relative norm=0.559550
```

Dominant settled cause:

```text
bias-side process/covariance understatement
```

Frozen F-CALIBRATED-v1:

```text
s_P0_att=2
s_P0_bias=4
s_Qg=2
s_Qb=8
all sensor R scales=1
```

Stationary N=50:

```text
F-BASE full NEES/DOF=1.418
F-CALIBRATED=1.021
attitude marginal=0.971
bias marginal=1.312
sensor NIS approximately 1
```

C4 N=50:

```text
bias RMSE degradation vs F-BASE=58.1%
settled mag NIS=1.733
settled sun NIS=1.921
settled ST NIS=4.006
```

따라서 F-CALIBRATED는 stationary-specific comparator이며 C4 primary baseline이 될 수 없다.

Updated P1 Exit:

```text
CONDITIONAL_GO
```

## 7. Frozen Classical Baseline Matrix

Primary:

```text
F-BASE
```

Sensitivity comparator:

```text
F-TUNED
s_Qg=0.125
s_Qb=0.125
s_R_ST=8.0
```

Stationary calibration comparator:

```text
F-CALIBRATED-v1
s_P0_att=2
s_P0_bias=4
s_Qg=2
s_Qb=8
R=1
```

Non-deployable oracle/diagnostic:

```text
ORACLE-PROCESS
ORACLE-MEASUREMENT
ORACLE-FULL
WRONG-PROCESS
WRONG-MEASUREMENT
```

이 baseline은 frozen test 결과를 보고 덮어쓰거나 재튜닝하지 않는다.

## 8. Phase 1 Scientific Conclusions

Supported within this representative benchmark:

1. Time-varying measurement reliability creates strong fixed-filter degradation.
2. Process uncertainty can degrade state consistency even when accuracy change is modest.
3. Correct process/measurement-side actions differ.
4. Wrong-side action may be ineffective or harmful.
5. Scalar innovation RMS alone is insufficient in the constructed C5 pair.
6. Raw gyro and sensor-specific evidence should be retained.
7. The frozen F-CALIBRATED-v1 selected under the declared P0/Q search does not satisfy stationary and C4 acceptance simultaneously; this is not a universal impossibility claim for every fixed calibration.
8. Time-varying, cause-specific adaptation has a justified research role.

Not established:

```text
neural superiority
learned context usefulness
universal identifiability
flight sensor fidelity
WMM/orbit/eclipse fidelity
single-mag-vector full observability
FPGA suitability
closed-loop performance
```

## 9. Phase 2 Design-Review and Implementation Boundary

이 master summary는 Phase 2 Design Review나 implementation을 자동 승인하지 않는다.
Phase 2 Design Review는 별도의 명시적 사용자 요청으로만 시작할 수 있다.
Phase 2 implementation은 미착수 상태이며 Design Review와 별개의 명시적 승인이
필요하다.

향후 별도 승인된 Design Review에서 유지해야 할 조건:

Mandatory:

- retain all frozen classical baselines;
- implement at least one non-neural adaptive classical baseline;
- preserve identical raw sensor realization and trajectory split;
- prevent truth/oracle/event-label leakage;
- evaluate stationary penalty and event benefit together;
- retain sensor-specific evidence;
- compare correct-side, shared-side and wrong-side actions;
- do not use F-CALIBRATED as universal primary baseline;
- do not claim neural necessity from Phase 1 alone.

Phase 2 implementation has not begun.
