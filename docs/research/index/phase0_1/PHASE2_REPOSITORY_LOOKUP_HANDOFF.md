# Phase 2 Repository Lookup Handoff

이 문서는 Phase 2를 설계하거나 승인하지 않는다. 새 Phase 2 관련 대화가 Phase
0–1의 convention, 수치, 구현 또는 결정 이력을 정확히 조회할 때 어느 파일을
먼저 열지 고정하는 read-only handoff다.

## 새 대화의 첫 다섯 파일

다음 순서로 읽는다.

1. `docs/research/phase1b/AI_ADCS_PHASE0_1_MASTER_SUMMARY_AND_PHASE2_HANDOFF.md`
   - Phase 0–1 전체 navigation과 handoff를 읽는다.
   - 수치와 판정은 여기서 멈추지 않고 아래 canonical source까지 추적한다.
2. `experiments/phase1b/reports/P1_EXIT_REVIEW_UPDATED.md`
   - 현재 P1 Exit: `CONDITIONAL_GO`.
   - stationary closure PASS, independent C4 closure FAIL.
   - Phase 2 미승인·미구현.
3. `docs/research/index/phase0_1/DECISION_AND_STATUS_LEDGER.md`
   - current/historical/superseded를 구분한다.
4. `docs/research/index/phase0_1/SOURCE_OF_TRUTH_INDEX.md`
   - 질문 domain의 canonical chain을 선택한다.
5. `docs/research/index/phase0_1/NUMERIC_EVIDENCE_CATALOG.md`
   - 수치를 dataset/split/policy/window scope와 결합한다.

그 다음 특정 질문은 아래 bundle만 추가로 연다. 저장소 전체를 막연히 읽지
않아도 되도록 first-open 순서를 최소화했다.

## Lookup bundle A: 좌표계·quaternion·MEKF math

```text
1. docs/research/phase0a/decision_lock/P0_01_DECISION_LEDGER.md
   → D04-D08
2. docs/research/phase0a/decision_lock/P0_05_MEKF_MATH_CONTRACT.md
   → locked notation and error state
3. docs/research/phase1a/P1A_GATE_A_FINAL_APPROVAL.md
   → exact-pi and immutability
4. bench/estimators/mekf.py
   → MEKFState, align_quaternion, quat_log, star_tracker_residual,
     propagate_state, star_tracker_update
5. tests/test_mekf_conventions.py and tests/test_mekf_core.py
```

잠긴 의미: scalar-first Hamilton `q_NB`, active body-to-navigation,
`R_NB`, `C_BN=R_NB.T`, right-local 6D error `[δθ,δb_g]`.

## Lookup bundle B: typed UNIT-ST data와 semantic identity

```text
1. docs/research/phase1a/P1A_EVENT_SCHEMA_CONTRACT.md
   → schema/order/hash/split and Gate B2 q_NB erratum
2. docs/research/phase1a/P1A_GATE_B1_AMENDMENT_A1_CONTRACT.md
   → schema identity versus generator identity
3. bench/tasks/generator/mekf_events.py
   → MEKFEventTable, save/load, split_trajectory_ids, replay_trajectory
4. bench/tasks/generator/unit_st_synthetic.py
5. tests/test_mekf_events.py, test_mekf_replay.py, test_unit_st_synthetic.py
```

잠긴 제약: valid B1 event는 zero latency, same-time gyro 후 ST, whole-trajectory
split. `generator_id`는 manifest identity에 속하며 schema/physical semantic-hash
domain migration이 아니다.

## Lookup bundle C: Basilisk truth와 frame proof

```text
1. docs/research/phase1a/P1A_GATE_B2_FINAL_APPROVAL.md
   → §2 frame relation, §3 executable proofs, §4 sensor layer
2. docs/research/phase1a/P1A_BASILISK_FRAME_CONVENTION_PROOF.md
3. bench/tasks/generator/basilisk_unit_st.py
   → basilisk_sigma_BN_to_q_NB, run_static_frame_proof,
     run_dynamic_rate_proof, generate_basilisk_unit_st
4. tests/test_basilisk_unit_st_generator.py
```

잠긴 관계:

```text
q_NB = normalize(MRP2EP(sigma_BN))
R_NB = quat_to_dcm(q_NB) = MRP2C(sigma_BN).T
MRP2C(sigma_BN) = C_BN
omega_BN_B = body-frame angular rate [rad/s]
```

Basilisk built-in star tracker를 사용했다고 표현하지 않는다. project-owned
parameterized gyro/ST layer다.

## Lookup bundle D: canonical metrics와 runner artifacts

```text
1. docs/research/phase1a/P1A_GATE_C_FINAL_APPROVAL.md
2. bench/metrics/mekf.py
   → attitude_geodesic_error_rad, bias_error_summary, star_tracker_nis,
     right_local_nees, spd_diagnostics, consistency_summary
3. bench/models/mekf.py
   → DatasetIdentity, MEKFReplayArtifact, MEKFEventReplayBridge
4. experiments/phase1a/reports/P1A_CP4_VALIDATION_REPORT.md
   → §5 fresh/cache CLI evidence
5. bench/runners/run_suite.py
   → _is_p1a_mekf_event_replay_pair, _run_p1a_mekf_event_replay,
     _p1a_exact_truth_join
6. tests/test_mekf_metrics.py, test_mekf_adapter.py,
   test_mekf_runner_integration.py
```

고정 pair는 `task_family=mekf_unit_st_v1`,
`model_id=mekf_event_replay_v1`이다. typed events를 dense float32/zero-filled
sequence로 바꾸지 않는다. truth는 estimation 뒤 trajectory ID와 timestamp로
exact join되어 metric에만 쓰인다.

## Lookup bundle E: Step 1 baseline과 문제 존재

```text
1. docs/research/phase1b/P1B_STEP1_FINAL_APPROVAL_AND_STEP2_HANDOFF.md
   → §2 F-BASE/F-TUNED roles
2. experiments/phase1b/results/unit_st_classical_v1/tuning.json
   → frozen scales, C5, test_split_accessed
3. experiments/phase1b/reports/P1B_UNIT_ST_BASELINE_REPORT.md
   → C1, mismatch, long horizon
4. experiments/phase1b/reports/P1B_PROBLEM_EXISTENCE_REPORT.md
   → C2/C3 severity and oracle effects
5. experiments/phase1b/reports/P1B_IDENTIFIABILITY_PILOT_REPORT.md
   → C5 H3/H4 and limits
6. experiments/phase1b/results/unit_st_classical_v1/pilot_summary.json
7. bench/experiments/phase1b_unit_st_classical.py and its two tests
```

고정 policy roles:

```text
F-BASE: primary classical baseline
F-TUNED: sensitivity comparator only
s_Qg=0.125, s_Qb=0.125, s_R_ST=8.0
C5 alpha_R_ST=1.08
```

C2는 process mismatch의 완만한 evidence, C3는 ST covariance mismatch의 강한
evidence다. C5는 해당 matched-RMS pair에서 scalar innovation RMS만으로 원인을
구별하기 어렵다는 예비 근거이지 일반 정보이론적 불식별성 증명이 아니다.

## Lookup bundle F: Step 2 fusion과 C4

```text
1. docs/research/phase1b/P1B_STEP1_FINAL_APPROVAL_AND_STEP2_HANDOFF.md
   → §4 truth/sensors/order/C4
2. bench/tasks/generator/mekf_fusion_events.py
3. bench/tasks/generator/phase1b_sensor_fusion.py
4. bench/experiments/phase1b_sensor_fusion_c4.py
5. experiments/phase1b/reports/P1B_SENSOR_FUSION_BASELINE_REPORT.md
6. experiments/phase1b/reports/P1B_STRESS_MAG_REPORT.md
7. experiments/phase1b/reports/P1B_C4_COMBINED_EVENT_REPORT.md
8. experiments/phase1b/results/sensor_fusion_c4_v1/settled_consistency.json
9. fusion event/generator/experiment tests
```

잠긴 order는 gyro→mag→sun→ST다. invalid sun은 update skip이며 zero measurement가
아니다. mag primary event는 mean bias/outlier가 아니라 inlier covariance 변화다.
magnetic/sun references는 versioned deterministic benchmark이고 WMM/orbit/eclipse
flight-fidelity claim이 아니다.

## Lookup bundle G: P1 Exit covariance closure

```text
1. docs/research/phase1b/P1_EXIT_CONDITIONAL_GO_AND_CLOSURE_HANDOFF.md
   → original named condition (historical)
2. experiments/phase1b/reports/P1_EXIT_TRANSIENT_DIAGNOSTIC_REPORT.md
3. experiments/phase1b/reports/P1_EXIT_COVARIANCE_CALIBRATION_REPORT.md
4. experiments/phase1b/reports/P1_EXIT_CLOSURE_VALIDATION_REPORT.md
5. experiments/phase1b/results/p1_exit_covariance_closure_v1/updated_exit_review.json
6. experiments/phase1b/reports/P1_EXIT_REVIEW_UPDATED.md
7. bench/experiments/p1_exit_covariance_closure.py and closure test/config
```

새 independent split은 train 30/validation 20/confirmation 50이며 frozen Phase 1
test trajectory와 disjoint다. 허용 변수는 P0 attitude/bias와 Qg/Qb뿐이고 sensor
R는 1로 유지되었다. frozen candidate:

```text
F-CALIBRATED-v1:
s_P0_att=2, s_P0_bias=4, s_Qg=2, s_Qb=8
```

stationary confirmation은 통과했지만 C4 confirmation은 실패했다. 이 두 사실을
함께 유지해야 한다.

## 수치 질의의 first-open matrix

| 필요한 수치 | 먼저 열 파일 | exact locator |
|---|---|---|
| Gate A/B1/B2/C/D1/CP4 test count | `regression_evidence.json` | command records; 각 final approval로 교차검증 |
| F-TUNED/C5 scales | `unit_st_classical_v1/tuning.json` | `/fixed_tuning`, `/frozen_c5_B_alpha_R` |
| C1/C2/C3/C5 paired results | `unit_st_classical_v1/pilot_summary.json` | `/summary`, `/c5_AB_independent_test` |
| long-horizon record | `unit_st_classical_v1/long_horizon.json` | top-level and `/records` |
| original MAIN settled consistency | `sensor_fusion_c4_v1/settled_consistency.json` | `/main_fusion_stationary_F_BASE` |
| original MAIN/STRESS/C4 paired data | `sensor_fusion_c4_v1/pilot_summary.json` | `/summary` and `/paired_differences` |
| closure cause | `p1_exit_covariance_closure_v1/diagnosis.json` | `/groups`, `/likely_source_ranking...` |
| frozen calibration | `.../updated_exit_review.json` | `/F_CALIBRATED_status` selected/freeze fields and source search-manifest hash |
| stationary/C4 closure | `.../confirmation/confirmation_summary.json` | `/groups`, `/acceptance` |
| current decision/scoped headline values | `.../updated_exit_review.json` | named top-level fields |

## 가장 흔한 수치 혼동 방지

| 값 | 올바른 설명 |
|---:|---|
| 1.8730178719854724 | original Step 2 MAIN-FUSION, F-BASE, settled, original test N=50 |
| 1.9062451467732702 | closure independent validation, F-BASE, settled full posterior, N=20 |
| 1.4180268635870965 | closure independent stationary confirmation, F-BASE, settled, N=50 |
| 1.0206761630935368 | 같은 confirmation, F-CALIBRATED-v1, settled, N=50 |

`1.873 → 1.021`을 같은 dataset의 before/after라고 표현하지 않는다.

## Current versus historical review

- Current: `experiments/phase1b/reports/P1_EXIT_REVIEW_UPDATED.md`.
- Historical: `experiments/phase1b/reports/P1_EXIT_REVIEW.md`.
- Original condition handoff도 historical context이며 current decision을 덮지 않는다.
- current review가 없거나 unreadable한 경우에만 historical review를 fallback으로
  보고, 그 사실을 명시한다.

## Audited master-summary provenance

계약이 지정한 master summary는 exact path에 존재하며 expected input digest와
일치한 artifact를 evidence-based documentation correction한 current navigation
문서다. master의 authority note를 지키고 exact numeric claim에는 result JSON,
현재 decision에는 updated P1 Exit review를 우선한다.

## Phase 2 boundary

이 handoff는 Phase 2 아이디어, model, interface, training 또는 implementation을
정의하지 않는다. updated P1 Exit가 별도 승인을 제공하지 않았으므로, repository
evidence lookup을 마친 뒤에도 Phase 2 구현으로 자동 진행하지 않는다. Phase 2
Design Review는 별도의 명시적 사용자 요청으로 시작할 수 있지만 implementation
authorization과는 별개다.
