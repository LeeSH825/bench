# File Path Catalog

모든 경로는 repository root `/home/dss-pc-05/bench` 기준 상대경로다. directory
entry는 그 아래의 immutable artifact family 전체를 뜻하며, exact numeric claim은
항상 개별 JSON/Markdown locator를 사용한다.

## Master navigation/handoff

| 경로 | 역할 | 상태 |
|---|---|---|
| `docs/research/phase1b/AI_ADCS_PHASE0_1_MASTER_SUMMARY_AND_PHASE2_HANDOFF.md` | Phase 0–1 audited navigation과 Phase 2 lookup handoff; exact-number/decision authority 아님 | current |

## Phase 0 canonical decisions

| 경로 | 역할 | 상태 |
|---|---|---|
| `docs/research/phase0a/decision_lock/P0_01_DECISION_LEDGER.md` | D01–D21 결정 ledger | current |
| `docs/research/phase0a/decision_lock/P0_02_TRUTH_SENSOR_ESTIMATOR_BOUNDARY.md` | truth/sensor/estimator/oracle boundary | current |
| `docs/research/phase0a/decision_lock/P0_03_TRUTH_MODEL_SPEC.md` | truth model specification | current |
| `docs/research/phase0a/decision_lock/P0_04_SENSOR_ROLE_AND_MODEL_SPEC.md` | gyro/mag/sun/ST roles | current |
| `docs/research/phase0a/decision_lock/P0_05_MEKF_MATH_CONTRACT.md` | MEKF notation/math | current |
| `docs/research/phase0a/decision_lock/P0_06_NEURAL_INSERTION_OPTIONS.md` | Phase 0 option analysis; implementation authorization 아님 | historical context |
| `docs/research/phase0a/decision_lock/P0_07_CONTEXT_CONTRACT.md` | context taxonomy/leakage/identifiability | current |
| `docs/research/phase0a/decision_lock/P0A_PHASE_0A_SYNTHESIS.md` | Phase 0 synthesis and exit | current |

## Phase 1A decisions and contracts

| 경로 | 역할 | 상태 |
|---|---|---|
| `docs/research/phase1a/P1A_GATE_A_FINAL_APPROVAL.md` | Gate A final GO | current |
| `docs/research/phase1a/P1A_EVENT_SCHEMA_CONTRACT.md` | B1 schema/order/hash/split plus active-q_NB erratum | current; old passive wording superseded |
| `docs/research/phase1a/P1A_GATE_B1_FINAL_APPROVAL.md` | B1 final approval | current |
| `docs/research/phase1a/P1A_GATE_B1_AMENDMENT_A1_CONTRACT.md` | generator identity separation | current |
| `docs/research/phase1a/P1A_BASILISK_UNIT_ST_CONTRACT.md` | Basilisk dataset contract | current |
| `docs/research/phase1a/P1A_BASILISK_FRAME_CONVENTION_PROOF.md` | executable active B→N frame proof | current |
| `docs/research/phase1a/P1A_GATE_B2_FINAL_APPROVAL.md` | B2 final GO | current |
| `docs/research/phase1a/P1A_GATE_C_FINAL_APPROVAL.md` | canonical metrics GO | current |
| `docs/research/phase1a/P1A_CP4_STEP1_FINAL_APPROVAL.md` | D1 bridge approval and D2 shortlist | current |
| `docs/research/phase1a/P1A_FOUNDATION_FINAL_APPROVAL_AND_P1B_HANDOFF.md` | Phase 1A→1B handoff | historical handoff, foundation remains valid |

## Phase 1A implementation and tests

| source path | 주요 symbols / 역할 | supporting tests |
|---|---|---|
| `bench/estimators/mekf.py` | `MEKFState`, quaternion ops, vector models, propagate/update, strict SPD | `test_mekf_core.py`, `test_mekf_conventions.py` |
| `bench/tasks/generator/mekf_events.py` | B1 typed table, serialization/hash/split/direct replay | `test_mekf_events.py`, `test_mekf_replay.py` |
| `bench/tasks/generator/unit_st_synthetic.py` | analytic synthetic UNIT-ST | `test_unit_st_synthetic.py` |
| `bench/tasks/generator/basilisk_unit_st.py` | frame conversion/proofs, parameterized UNIT-ST | `test_basilisk_unit_st_generator.py` |
| `bench/metrics/mekf.py` | right-local error, geodesic, bias, NIS, NEES, SPD | `test_mekf_metrics.py` |
| `bench/models/mekf.py` | `DatasetIdentity`, replay artifact/bridge | `test_mekf_adapter.py` |
| `bench/tasks/bench_generated.py` | `prepare_mekf_unit_st_v1` | `test_mekf_runner_integration.py` |
| `bench/models/registry.py` | model ID registration | same runner test |
| `bench/runners/run_suite.py` | exact task/model early branch, exact truth join, Gate C metrics | same runner test |
| `bench/configs/suite_phase1a_unit_st_smoke.yaml` | synthetic+Basilisk fresh/cache smoke | CP4 report/runner test |

Exact test files:

```text
tests/test_mekf_core.py
tests/test_mekf_conventions.py
tests/test_mekf_events.py
tests/test_mekf_replay.py
tests/test_unit_st_synthetic.py
tests/test_basilisk_unit_st_generator.py
tests/test_mekf_metrics.py
tests/test_mekf_adapter.py
tests/test_mekf_runner_integration.py
```

## Phase 1A specialized reports

| 경로 | 증거 |
|---|---|
| `experiments/phase1a/reports/P1A_MATH_VALIDATION_REPORT.md` | Gate A core numerical validation |
| `experiments/phase1a/reports/P1A_GATE_B1_VALIDATION_REPORT.md` | original B1 validation |
| `experiments/phase1a/reports/P1A_GATE_B1_AMENDMENT_A1_REPORT.md` | identity amendment validation |
| `experiments/phase1a/reports/P1A_GATE_B2_VALIDATION_REPORT.md` | frame/dynamics/sensor validation |
| `experiments/phase1a/reports/P1A_GATE_C_VALIDATION_REPORT.md` | metric reference/property tests |
| `experiments/phase1a/reports/P1A_GATE_D1_VALIDATION_REPORT.md` | bridge equivalence |
| `experiments/phase1a/reports/P1A_CP4_VALIDATION_REPORT.md` | runner fresh/cache, hashes, commands |
| `experiments/phase1a/agent_logs/05D2_20260801T163044Z_cli_verified_fresh.txt` | fresh-generation CLI raw evidence |
| `experiments/phase1a/agent_logs/05D2_20260801T163044Z_cli_verified_cache_hit.txt` | cache-hit raw evidence |

## Phase 1B Step 1

| 종류 | 경로 | 역할 / locator |
|---|---|---|
| decision | `docs/research/phase1b/P1B_STEP1_FINAL_APPROVAL_AND_STEP2_HANDOFF.md` | F-BASE/F-TUNED roles; Step 2 locks |
| source | `bench/experiments/phase1b_unit_st_classical.py` | tune/pilot/long/report orchestration |
| config | `bench/configs/suite_phase1b_unit_st_classical.yaml` | regimes, policies, acceptance |
| tests | `tests/test_phase1b_unit_st_classical.py` | stream/policy/tuning/long behavior |
| tests | `tests/test_phase1b_unit_st_regimes.py` | C2/C3/C5/oracle/wrong-side |
| result | `experiments/phase1b/results/unit_st_classical_v1/tuning.json` | frozen F-TUNED and C5; test leakage flag |
| result | `experiments/phase1b/results/unit_st_classical_v1/pilot_summary.json` | paired N=50 groups and C5 independent test |
| result | `experiments/phase1b/results/unit_st_classical_v1/pilot_workload.json` | workload identity |
| result | `experiments/phase1b/results/unit_st_classical_v1/long_horizon.json` | 600 s/N=10 records |
| result | `experiments/phase1b/results/unit_st_classical_v1/validation.json` | validation outcome |
| report | `experiments/phase1b/reports/P1B_UNIT_ST_BASELINE_REPORT.md` | C1/mismatch/long horizon |
| report | `experiments/phase1b/reports/P1B_PROBLEM_EXISTENCE_REPORT.md` | C2/C3 severity and oracle effects |
| report | `experiments/phase1b/reports/P1B_IDENTIFIABILITY_PILOT_REPORT.md` | C5 H3/H4 scoped interpretation |
| report | `experiments/phase1b/reports/P1B_STEP1_VALIDATION_REPORT.md` | tests, integrity, rerun evidence |
| manifest root | `experiments/phase1b/manifests/unit_st_classical_v1/` | sensor and simulation-only oracle artifacts |

대표 manifest는
`.../pilot/C1-STATIONARY/sensor/manifest.json`과
`.../pilot/C1-STATIONARY/oracle_simulation_only/experiment_manifest.json`이다.
후자는 estimator-facing artifact가 아니다.

## Phase 1B Step 2

| 종류 | 경로 | 역할 / locator |
|---|---|---|
| source | `bench/tasks/generator/mekf_fusion_events.py` | fusion schema, oracle sidecar, ordering |
| source | `bench/tasks/generator/phase1b_sensor_fusion.py` | deterministic mag/sun benchmark generator |
| source | `bench/metrics/mekf_fusion.py` | mag/sun NIS |
| source | `bench/experiments/phase1b_sensor_fusion_c4.py` | fixed/oracle/wrong-side replay and studies |
| config | `bench/configs/suite_phase1b_sensor_fusion_c4.yaml` | MAIN/STRESS/C4 scenarios |
| tests | `tests/test_mekf_fusion_events.py` | schema/order/serialization/boundary |
| tests | `tests/test_phase1b_sensor_fusion.py` | generator and sensor layer |
| tests | `tests/test_phase1b_sensor_fusion_experiments.py` | replay/oracle/metrics/studies |
| result | `experiments/phase1b/results/sensor_fusion_c4_v1/pilot_summary.json` | all paired groups/differences |
| result | `experiments/phase1b/results/sensor_fusion_c4_v1/settled_consistency.json` | original MAIN settled consistency |
| result | `experiments/phase1b/results/sensor_fusion_c4_v1/exit_review.json` | original exit decision machine evidence |
| report | `experiments/phase1b/reports/P1B_SENSOR_FUSION_BASELINE_REPORT.md` | MAIN-FUSION/ablations |
| report | `experiments/phase1b/reports/P1B_STRESS_MAG_REPORT.md` | magnetic observability limitation |
| report | `experiments/phase1b/reports/P1B_C4_COMBINED_EVENT_REPORT.md` | C4 oracle/wrong-side |
| report | `experiments/phase1b/reports/P1B_STEP2_VALIDATION_REPORT.md` | tests/commands/integrity |
| manifest root | `experiments/phase1b/manifests/sensor_fusion_c4_v1/` | scenario sensor/oracle artifacts |
| command log | `experiments/phase1b/agent_logs/02_20260802T020512Z_COMMAND_LOG.md` | Step 2 execution commands |

## P1 Exit original review and covariance closure

| 종류 | 경로 | 역할 / 상태 |
|---|---|---|
| original review | `experiments/phase1b/reports/P1_EXIT_REVIEW.md` | historical; updated review가 대체 |
| closure handoff | `docs/research/phase1b/P1_EXIT_CONDITIONAL_GO_AND_CLOSURE_HANDOFF.md` | historical named-condition contract |
| source | `bench/experiments/p1_exit_covariance_closure.py` | diagnose/search/freeze/confirm/report/exit-review |
| config | `bench/configs/suite_p1_exit_covariance_closure.yaml` | allowed scales and acceptance |
| tests | `tests/test_p1_exit_covariance_closure.py` | 17 closure tests |
| diagnosis | `experiments/phase1b/results/p1_exit_covariance_closure_v1/diagnosis.json` | independent split, marginals, whitened/cross diagnostics |
| search | `experiments/phase1b/results/p1_exit_covariance_closure_v1/search/search_manifest.json` | candidate selection/freeze evidence |
| confirmation | `experiments/phase1b/results/p1_exit_covariance_closure_v1/confirmation/confirmation_summary.json` | stationary/C4 N=50 independent confirmation |
| machine review | `experiments/phase1b/results/p1_exit_covariance_closure_v1/updated_exit_review.json` | current exact values and decision |
| regression | `experiments/phase1b/results/p1_exit_covariance_closure_v1/regression_evidence.json` | final suite command outcomes |
| diagnostic report | `experiments/phase1b/reports/P1_EXIT_TRANSIENT_DIAGNOSTIC_REPORT.md` | initial/settled/marginal/cross interpretation |
| calibration report | `experiments/phase1b/reports/P1_EXIT_COVARIANCE_CALIBRATION_REPORT.md` | train/validation search and freeze |
| validation report | `experiments/phase1b/reports/P1_EXIT_CLOSURE_VALIDATION_REPORT.md` | independent stationary/C4 acceptance |
| current review | `experiments/phase1b/reports/P1_EXIT_REVIEW_UPDATED.md` | current canonical `CONDITIONAL_GO` |
| manifest root | `experiments/phase1b/manifests/p1_exit_covariance_closure_v1/` | independent train/confirmation sensor/oracle artifacts |
| command log | `experiments/phase1b/agent_logs/03_20260802T032016Z_COMMAND_LOG.md` | closure commands |

## Provenance and this index

| 경로 | 역할 |
|---|---|
| `experiments/phase1b/preflight_snapshots/03_20260802T032016Z/FINAL_INTEGRITY.md` | closure dirty-tree integrity |
| `experiments/research_index/preflight_snapshots/01_20260802T_currentZ/` | indexing entry status/patch/hash/runtime snapshot |
| `docs/research/index/phase0_1/phase0_1_evidence_index.json` | machine canonical index |
| `tools/research/validate_phase0_1_evidence_index.py` | stdlib-only validator |
| `tests/test_phase0_1_evidence_index.py` | index regression and 20 lookup self-tests |
| `experiments/research_index/reports/PHASE0_1_REPOSITORY_INDEX_VALIDATION_REPORT.md` | final indexing validation outcome |

## Master amendment provenance

master input artifact SHA는 expected digest와 일치했고 exact path로 복원되었다.
claim audit와 documentation-only correction 내역은
`experiments/research_index/reports/PHASE0_1_MASTER_SUMMARY_AUDIT_REPORT.md`에 있다.
수치 또는 현재 결정을 인용할 때는 master에서 끝내지 않고 해당 result JSON이나
updated review를 연다.
