# Phase 1B Step 1 UNIT-ST Classical Benchmark Contract

## 1. 목적과 범위

이 계약은 Phase 1A의 동결된 MEKF, typed event, Basilisk UNIT-ST truth,
canonical metric을 이용해 고정 classical filter가 시간변화 uncertainty에 보이는
반응을 측정한다. 결과는 representative-normalized UNIT-ST 연구 benchmark이며
flight representativeness 주장이 아니다.

포함 범위는 C1 stationary, C2 gyro process uncertainty, C3 inlier
star-tracker reliability, C5 innovation-RMS-matched A/B, fixed Q/R tuning,
paired Monte Carlo와 stationary long horizon이다. Magnetometer, sun sensor,
C4 combined event, outage/outlier/gating, neural model, closed-loop control은
포함하지 않는다.

## 2. Raw sensor와 oracle 경계

Raw artifact는 Phase 1A의 세 파일 `manifest.json`, `truth.npz`, `events.npz`
형식을 유지한다. Estimator-facing 입력은 `MEKFEventTable`뿐이다. Sensor
manifest에는 regime code, event window, `alpha_g`, `alpha_b`, `alpha_R_ST`가
없다.

Simulation-only oracle artifact는 별도 디렉터리의
`experiment_manifest.json`과 `oracle_context.npz`다. Sidecar는 event별
trajectory/order, covariance multiplier, window, regime을 가지며 별도 semantic
hash로 보호된다. Oracle replay조차 forward-only cursor로 현재 event scale을
한 번씩만 소비한다. Truth는 estimation 종료 뒤 trajectory ID와 timestamp로
exact join하여 evaluation에만 사용한다.

Fixed/tuned public replay API에는 sidecar, window, label 인수가 없다. Deployable
fixed artifact는 policy ID와 세 개의 고정 scale만 포함한다.

## 3. Regime 및 covariance 의미

`alpha`는 covariance multiplier이며 standard deviation multiplier는
`sqrt(alpha)`다. Base draw에 event-only 독립 draw를 더할 때 증분 표준편차는
`sqrt(alpha - 1)`이므로 전체 covariance가 정확히 `alpha`배가 된다.

| Regime | Event 구간 변경 | 고정 항목 |
|---|---|---|
| C1 | 없음; 모든 alpha=1 | truth, gyro, ST, timing |
| C2 | gyro actual noise covariance에 alpha_g 적용 | alpha_b=alpha_R_ST=1 |
| C3 | ST tangent inlier covariance에 alpha_R_ST 적용 | alpha_g=alpha_b=1 |
| C5-A | C2 medium, alpha_g=4 | ST stream model |
| C5-B | C3, validation에서 alpha_R_ST 선택 | gyro stream model |

Event window는 `[0.4T, 0.6T)`이며 fixed/tuned filter에는 공개하지 않는다.
Gyro/ST 모두 zero latency이고 같은 timestamp에서는 gyro propagation 뒤 ST
update 순서다. Truth, base gyro noise, base ST noise, representation-sign stream은
모든 paired condition에서 동일하다.

## 4. Estimator knowledge와 policy

| Policy | Q/R action | 허용 정보 | 배포 가능 |
|---|---|---|---|
| F-BASE | base Qg/Qb/R | raw typed events | 예 |
| F-MIS-Q-LOW/HIGH | Qg×0.25/4 | raw typed events | 진단용 fixed |
| F-MIS-R-LOW/HIGH | R×0.25/4 | raw typed events | 진단용 fixed |
| F-TUNED | 고정 `(s_Qg,s_Qb,s_R)` | raw typed events | 예 |
| ORACLE-QR | 현재 alpha를 올바른 Q/R side에 적용 | 현재-event cursor | 아니오 |
| WRONG-SIDE | alpha_g→R, alpha_R→Qg | 현재-event cursor | 아니오 |

모든 policy는 같은 condition/trajectory의 동일 raw measurement object를
replay한다. `propagate_state`와 `star_tracker_update`만 호출하며 quaternion 또는
Kalman math를 다시 구현하지 않는다.

## 5. Tuning protocol

Stationary train/validation trajectory만 사용한다. Test ID는 policy freeze 전
접근하지 않는다. 후보는 `{0.25, 0.5, 1, 2, 4}`이고 다음 42개 예산을 고정한다.

1. Qg coordinate 5개
2. Qb coordinate 5개
3. R coordinate 5개
4. 선택점 주변 3×3×3 local grid 27개

목적함수 우선순위는 divergence/SPD failure, attitude geodesic RMSE, bias vector
RMSE, NIS/NEES normalized-mean penalty, scale lexicographic tie-break다. 선택 후
test/long-horizon 결과를 보고 다시 조정하지 않는다.

C5-B는 pilot과 동일 seed/cadence의 validation split 17개에서 predeclared alpha
grid를 평가하고 F-BASE aggregate innovation RMS 차이가 5% 이내인 값을 freeze한
뒤 독립 test 50개에 적용한다.

## 6. Metric과 통계

Attitude geodesic error, bias, NIS, NEES, P/S SPD는 오직
`bench.metrics.mekf`의 Gate C 구현을 사용한다. Primary event metric은 attitude
RMSE/P95/peak, bias vector RMSE, recovery, divergence다.

Recovery는 event 종료 후 attitude error가
`max(1.2 × pre-event RMS, 0.001 rad)` 이하로 3 sample 연속 들어간 첫 시간이다.
Divergence는 nonfinite, P/S Cholesky failure 또는 event attitude error가
0.5 rad를 넘는 경우다. 실패를 inverse, pseudo-inverse, jitter, clipping,
skip/xfail 또는 tolerance 완화로 숨기지 않는다.

각 condition/policy는 exact trajectory pairing을 사용한다. N, mean, median,
P95, divergence, paired difference와 deterministic 2,000-resample 95% bootstrap
CI를 보존한다. C5는 innovation RMS/norm distribution/autocorrelation과 raw gyro
increment statistics도 비교한다.

## 7. Reproducibility와 handoff

Locked config는 `bench/configs/suite_phase1b_unit_st_classical.yaml`이다. Result는
`experiments/phase1b/results/unit_st_classical_v1`, sensor/oracle manifest는
`experiments/phase1b/manifests/unit_st_classical_v1`에 분리 저장한다. 각 pilot
record는 scenario/policy/trajectory 단위 canonical JSON이라 `pilot --resume`이
가능하다. Manifest는 config/source/runtime hash, raw/oracle identity, trajectory
IDs, policy knowledge, Q/P, metric/statistics contract와 artifact path를 기록한다.

Step 2 handoff 시에는 이 Step 1 결과와 경계를 유지하면서 sensor expansion과
C4 combined event를 별도 승인 후 시작한다. 이 계약은 Step 2를 자동 승인하지
않는다.
