# Numeric Evidence Catalog

수치는 값만 단독 인용하지 않는다. 최소한 `experiment + split/dataset + policy +
window + metric`을 함께 적는다. JSON에 더 긴 값이 있으면 JSON의 stored value가
canonical이고, 아래 report 표기의 반올림 값은 사람이 읽기 위한 표현이다.

Master summary의 수치는 navigation copy다. master를 이 catalog의 canonical
numeric source로 사용하지 말고 반드시 아래 machine JSON 또는 specialized
report locator까지 추적한다.

## Phase 1A verification numerics

| ID | 값 | 단위 | 정확한 scope | canonical source / locator |
|---|---:|---|---|---|
| N-GATE-A-TESTS | 55 | passed tests | Gate A final new suite | `P1A_GATE_A_FINAL_APPROVAL.md::§4` |
| N-GATE-A-LEGACY | 18 | passed tests | designated legacy tests; separate 5 subtests | 같은 locator |
| N-EXACT-PI-ORDINARY-SWEEP | 1000 | quaternion pairs | ordinary-angle q/-q invariance | 같은 approval §4 |
| N-EXACT-PI-NEAR-SWEEP | 256 | quaternion pairs | near-pi q/-q invariance | 같은 approval §4 |
| N-GATE-B1-TESTS | 55 | passed tests | B1 after identity Amendment A1 | `P1A_GATE_B1_AMENDMENT_A1_REPORT.md::Regression results` |
| N-GATE-B1-LATENCY | 0 | s | all valid B1 events, arrival minus measurement time | `P1A_EVENT_SCHEMA_CONTRACT.md::Zero-latency rule` |
| N-B2-STATIC-BASIS-ERROR | 4.440892098500626e-16 | matrix absolute error | identity/±90° axes/arbitrary attitude basis proof max | `P1A_GATE_B2_FINAL_APPROVAL.md::§3` |
| N-B2-SHADOW-ERROR | 4.85722573273506e-16 | matrix absolute error | MRP shadow-set proof max | 같은 locator |
| N-B2-DYNAMIC-ATT-ERROR | 4.872566201647101e-16 | rad | spherical-inertia zero-torque attitude proof max | 같은 locator |
| N-B2-DYNAMIC-RATE-ERROR | 3.219646771412954e-14 | rad/s | constant-rate body-rate proof max | 같은 locator |
| N-GATE-B2-TESTS | 67 | passed tests | Gate B2 final suite | B2 approval validation evidence |
| N-GATE-C-TESTS | 43 | passed tests | Gate C suite | `P1A_GATE_C_FINAL_APPROVAL.md::§3` |
| N-GATE-C-NEES-QSIGN-DIFF | 0 | normalized NEES difference | q versus -q invariant NEES check | 같은 locator |
| N-D1-TESTS | 24 | passed tests | D1 replay bridge suite | `P1A_GATE_D1_VALIDATION_REPORT.md::Test results` |
| N-CP4-TESTS | 22 | passed tests | CP4 runner integration suite | `P1A_CP4_VALIDATION_REPORT.md::Regression summary` |
| N-CP4-PRODUCER-SEEDS | 3 | seeds/producer | synthetic and Basilisk each, seeds 6101–6103 | 같은 report §5 |

## Frozen Step 1 policies and workload

| ID | 값 | 단위 | 정확한 scope | canonical source / locator |
|---|---:|---|---|---|
| N-FTUNED-QG | 0.125 | scale | frozen F-TUNED gyro white-noise process covariance | `tuning.json:/fixed_tuning/selected_policy/qg_scale` |
| N-FTUNED-QB | 0.125 | scale | frozen F-TUNED gyro-bias random-walk covariance | `tuning.json:/fixed_tuning/selected_policy/qb_scale` |
| N-FTUNED-RST | 8 | scale | frozen F-TUNED star-tracker covariance | `tuning.json:/fixed_tuning/selected_policy/r_scale` |
| N-STEP1-TESTS | 52 | passed tests | Phase 1B Step 1 new suite | `P1B_STEP1_VALIDATION_REPORT.md::Regression table` |
| N-C1-N | 50 | paired trajectories | C1 matched test group | `pilot_summary.json:/completed_paired_N_per_condition` |
| N-C1-FBASE-ATT-RMSE | 1.6615e-3 | rad | C1 stationary matched, F-BASE attitude event RMSE | `P1B_UNIT_ST_BASELINE_REPORT.md::Stationary row F-BASE` |
| N-C1-FBASE-BIAS-RMSE | 2.0476e-3 | rad/s | same C1 group, bias vector RMSE | 같은 row |
| N-C1-FBASE-NIS | 0.924 | NIS/DOF | same C1 group | 같은 row |
| N-C1-FBASE-NEES | 0.967 | NEES/DOF | same C1 group | 같은 row |
| N-C1-FTUNED-ATT-RMSE | 1.6469e-3 | rad | C1 stationary matched, F-TUNED | 같은 report F-TUNED row |
| N-C1-FTUNED-NIS | 0.150 | NIS/DOF | same F-TUNED group | 같은 row |
| N-C1-FTUNED-NEES | 0.152 | NEES/DOF | same F-TUNED group | 같은 row |

## Step 1 problem-existence and identifiability

| ID | 값 | 단위 | 정확한 scope | canonical source / locator |
|---|---:|---|---|---|
| N-C2-A2-RMSE | 1.6824e-3 | rad | C2 F-BASE, process alpha 2 | `P1B_PROBLEM_EXISTENCE_REPORT.md::C2 severity` |
| N-C2-A4-RMSE | 1.7081e-3 | rad | C2 F-BASE, process alpha 4 | 같은 table |
| N-C2-A8-ATT-RMSE | 1.7520e-3 | rad | C2 F-BASE, process alpha 8 | 같은 table |
| N-C2-A8-NEES | 1.187 | NEES/DOF | C2 F-BASE, process alpha 8 | 같은 table |
| N-C2-A8-ORACLE-NEES | 0.968 | NEES/DOF | C2 correct-side oracle, process alpha 8 | 같은 report oracle comparison |
| N-C3-A2-RMSE | 2.0413e-3 | rad | C3 F-BASE, ST covariance alpha 2 | 같은 report C3 severity |
| N-C3-A4-RMSE | 2.6979e-3 | rad | C3 F-BASE, ST covariance alpha 4 | 같은 table |
| N-C3-A8-ATT-RMSE | 3.6859e-3 | rad | C3 F-BASE, ST covariance alpha 8 | 같은 table |
| N-C3-A8-NIS | 2.245 | NIS/DOF | C3 F-BASE, ST covariance alpha 8 | 같은 table |
| N-C3-A8-ORACLE-RMSE-DELTA | -1.318e-3 | rad | C3 alpha 8 oracle minus F-BASE attitude RMSE | 같은 report oracle comparison |
| N-C5-ALPHA-R | 1.08 | scale | frozen C5 B-pair ST covariance multiplier | `tuning.json:/frozen_c5_B_alpha_R` |
| N-C5-VALIDATION-RMS-DIFF | 0.00396231189955448 | fraction | train/validation-selected C5 innovation-RMS relative difference | `tuning.json:/c5_matching/selected/validation_rms_relative_difference` |
| N-C5-TEST-RMS-DIFF | 0.014851094915976358 | fraction | independent C5 F-BASE paired test RMS relative difference | `pilot_summary.json:/c5_AB_independent_test/F-BASE/independent_test_rms_relative_difference` |
| N-LONG-DURATION | 600 | s/trajectory | stationary long-horizon subset | `long_horizon.json:/duration_s` |
| N-LONG-N | 10 | trajectories | stationary long-horizon subset | `long_horizon.json:/num_trajectories` |
| N-LONG-FBASE-NIS | 0.988 | NIS/DOF | aggregated long-horizon F-BASE | `P1B_UNIT_ST_BASELINE_REPORT.md::Long-horizon row F-BASE` |
| N-LONG-FBASE-NEES | 0.854 | NEES/DOF | same group | 같은 row |
| N-LONG-FTUNED-NIS | 0.186 | NIS/DOF | aggregated long-horizon F-TUNED | 같은 report F-TUNED row |
| N-LONG-FTUNED-NEES | 2.252 | NEES/DOF | same F-TUNED group | 같은 row |

## Step 2 sensor fusion and C4

| ID | 값 | 단위 | 정확한 scope | canonical source / locator |
|---|---:|---|---|---|
| N-STEP2-TESTS | 38 | passed tests | Phase 1B Step 2 new suite | `P1B_STEP2_VALIDATION_REPORT.md::Post-implementation regression` |
| N-MAIN-N | 50 | paired trajectories | original MAIN-FUSION stationary Step 2 test | `P1B_SENSOR_FUSION_BASELINE_REPORT.md::Scenario scope` |
| N-MAIN-WHOLE-ATT-RMSE | 0.0310341 | rad | original MAIN-FUSION F-BASE whole trajectory | 같은 report aggregate table |
| N-MAIN-WHOLE-NEES | 3.8033 | NEES/DOF | original MAIN-FUSION F-BASE whole trajectory | 같은 aggregate table |
| N-MAIN-MAG-NIS | 1.0228982708948622 | NIS/DOF | original MAIN-FUSION F-BASE settled 20%-cut, 6050 mag samples | `settled_consistency.json:/.../mag_nis/normalized_mean` |
| N-MAIN-SUN-NIS | 1.0004866247227868 | NIS/DOF | same run/window, 2215 sun samples | JSON sun pointer |
| N-MAIN-ST-NIS | 1.0918889105362812 | NIS/DOF | same run/window, 1250 ST samples | JSON star-tracker pointer |
| N-MAIN-ORIGINAL-SETTLED-NEES | 1.8730178719854724 | NEES/DOF | original Step 2 MAIN-FUSION F-BASE settled posterior, N=50, 21800 states | JSON NEES pointer |
| N-STRESS-ATT-RMSE | 0.197681 | rad | STRESS-MAG F-BASE whole trajectory, N=50 | `P1B_STRESS_MAG_REPORT.md::Aggregate F-BASE` |
| N-STRESS-WEAK | 0.195676 | rad | STRESS-MAG magnetic-axis-parallel weak component | 같은 report decomposition |
| N-STRESS-PLANE | 0.00133072 | rad | STRESS-MAG observable-plane component | 같은 locator |
| N-C4-ALPHA-B | 100000 | scale | original C4 bias random-walk intensity, 0.2T–0.8T | `P1B_C4_COMBINED_EVENT_REPORT.md::Primary event` |
| N-C4-ALPHA-RMAG | 16 | scale | original C4 mag inlier covariance, 0.45T–0.6T | 같은 locator |
| N-C4-ORIGINAL-FULL-SLOW-IMPROVE | 0.2856 | fraction | original Step 2 C4 full-oracle slow bias improvement, N=50 | 같은 report oracle comparison |
| N-C4-ORIGINAL-FULL-FAST-IMPROVE | 0.3257 | fraction | original Step 2 C4 full-oracle fast attitude peak improvement, N=50 | 같은 locator |

## P1 Exit covariance closure

| ID | 값 | 단위 | 정확한 scope | canonical source / locator |
|---|---:|---|---|---|
| N-CLOSURE-TRAIN | 30 | trajectories | new independent calibration train | `P1_EXIT_TRANSIENT_DIAGNOSTIC_REPORT.md::Split` |
| N-CLOSURE-VALIDATION | 20 | trajectories | new independent calibration validation | 같은 locator |
| N-CLOSURE-CONFIRM-N | 50 | paired trajectories/scenario | frozen-policy stationary and C4 confirmations | `confirmation_summary.json:/completed_N` |
| N-CLOSURE-INITIAL-NEES | 15.55804512442473 | NEES/DOF | independent validation F-BASE initial full whitened error | `updated_exit_review.json:/diagnosed_cause/validation_initial_full_nees_normalized` |
| N-CLOSURE-VALIDATION-SETTLED-ATT | 1.4348134637382222 | NEES/DOF | independent validation F-BASE settled attitude marginal | JSON diagnosed-cause pointer |
| N-CLOSURE-VALIDATION-SETTLED-BIAS | 2.7448526382517784 | NEES/DOF | independent validation F-BASE settled bias marginal | JSON diagnosed-cause pointer |
| N-CLOSURE-VALIDATION-SETTLED-NEES | 1.9062451467732702 | NEES/DOF | independent validation F-BASE settled full posterior aggregate | `diagnosis.json:/groups/validation/aggregate/partitions/settled/full_nees_normalized` |
| N-FCAL-P0-ATT | 2 | scale | frozen F-CALIBRATED-v1 initial attitude covariance | `updated_exit_review.json:/F_CALIBRATED_status/scales/s_P0_att` |
| N-FCAL-P0-BIAS | 4 | scale | frozen F-CALIBRATED-v1 initial bias covariance | JSON scale pointer |
| N-FCAL-QG | 2 | scale | frozen F-CALIBRATED-v1 gyro process covariance | JSON scale pointer |
| N-FCAL-QB | 8 | scale | frozen F-CALIBRATED-v1 bias random-walk covariance | JSON scale pointer |
| N-FCAL-R | 1 | scale | frozen mag/sun/ST R, all unchanged | JSON `/F_CALIBRATED_status/sensor_R_scales` |
| N-CLOSURE-FBASE-CONFIRM-NEES | 1.4180268635870965 | NEES/DOF | independent stationary confirmation F-BASE settled, N=50 | JSON `/confirmation_F_BASE_settled/full_nees_normalized` |
| N-CLOSURE-FCAL-CONFIRM-NEES | 1.0206761630935368 | NEES/DOF | same confirmation F-CALIBRATED-v1 settled | JSON calibrated full-NEES pointer |
| N-CLOSURE-FCAL-ATT-NEES | 0.9708331280650667 | NEES/DOF | same confirmation calibrated attitude marginal | JSON calibrated attitude pointer |
| N-CLOSURE-FCAL-BIAS-NEES | 1.3115928705043793 | NEES/DOF | same confirmation calibrated bias marginal | JSON calibrated bias pointer |
| N-CLOSURE-FCAL-MAG-NIS | 0.9852891819339538 | NIS/DOF | same confirmation calibrated settled mag | JSON calibrated mag pointer |
| N-CLOSURE-FCAL-SUN-NIS | 0.9704377843299701 | NIS/DOF | same confirmation calibrated settled sun | JSON calibrated sun pointer |
| N-CLOSURE-FCAL-ST-NIS | 0.9899380635161372 | NIS/DOF | same confirmation calibrated settled ST | JSON calibrated ST pointer |
| N-CLOSURE-C4-BIAS-DEGRADE | 0.5807472544511563 | fraction | independent closure C4, calibrated whole bias degradation vs F-BASE | JSON `/remaining_classical_limitation/...` |
| N-CLOSURE-C4-FULL-SLOW-IMPROVE | 0.3208727762378273 | fraction | independent closure C4 full-oracle slow-bias improvement | JSON `/acceptance/full_oracle_slow_bias_improvement_fraction` |
| N-CLOSURE-C4-FULL-FAST-IMPROVE | 0.4132085082893989 | fraction | independent closure C4 full-oracle fast-peak improvement | JSON `/acceptance/full_oracle_fast_peak_improvement_fraction` |

## 반드시 구분할 네 NEES

| 값 | dataset/split | policy | window | 의미 |
|---:|---|---|---|---|
| 1.8730178719854724 | original Step 2 MAIN-FUSION test N=50 | F-BASE | settled 20% cut | 최초 exit condition을 만든 값 |
| 1.9062451467732702 | closure independent validation N=20 | F-BASE | settled | calibration 선택 전 validation 진단 |
| 1.4180268635870965 | closure independent confirmation N=50 | F-BASE | settled | 새 confirmation에서 baseline 재측정 |
| 1.0206761630935368 | 같은 confirmation N=50 | F-CALIBRATED-v1 | settled | frozen candidate stationary closure 결과 |

이 네 값을 하나의 시계열이나 “수정 전/후 동일 dataset”으로 표현하면 안 된다.
