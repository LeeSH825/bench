# Phase 1A Gate D1 Test Matrix

- Test module: `tests/test_mekf_adapter.py`
- Interpreter: `/home/dss-pc-05/.pyenv/versions/3.10.13/bin/python`
- Result: 24 passed

| ID | Contract | Input | Expected | Tolerance | Actual | Evidence | Status |
|---|---|---|---|---|---|---|---|
| D1-01 | Lightweight, unregistered boundary | clean subprocess import and registry AST | no Basilisk, torch, runner, registry, or visualization import; ID absent from registry | exact | forbidden module list `[]`; ID absent | `test_d1_01_import_boundary_is_lightweight_and_unregistered` | PASS |
| D1-02 | Direct/bridge equivalence | synthetic typed trajectory | q/b/P, ST r/S, times, order, sensor and final state identical | `np.array_equal` | all fields equal | `test_d1_02_synthetic_direct_adapter_exact_equivalence` | PASS |
| D1-03 | Serialization equivalence | strict saved/loaded synthetic dataset | loaded direct and bridge artifacts identical; identity copied | exact | all fields and identity equal | `test_d1_03_serialized_dataset_replays_exactly` | PASS |
| D1-04 | Both Gate B producers | analytic synthetic and Basilisk UNIT-ST | one bridge accepts each typed schema unchanged | exact | 2/2 producer cases equal to direct replay | `test_d1_04_both_gate_b_generators_use_the_same_bridge` | PASS |
| D1-05 | Identity preservation/fail-loud | verified identity and altered dataset hash | artifact preserves all fields; expected-identity mismatch rejected | exact | preserved; mismatch raised `ValueError` before replay | `test_d1_05_identity_is_preserved_and_mismatch_fails` | PASS |
| D1-06 | Same-realization independence | expected-bound and unbound bridge instances | adapter instance choice does not alter data hash or numeric output | exact | dataset hash and q/b/P/r/S equal | `test_d1_06_bridge_instance_identity_does_not_change_data_or_numeric_output` | PASS |
| D1-07 | Truth-free public API | `replay_events` signature and class lifecycle | only required event/state/config/identity inputs; no dense predict/train API | exact names | exact seven-argument method; no predict/train | `test_d1_07_public_estimator_api_is_truth_oracle_and_label_free` | PASS |
| D1-08 | No regeneration or mutation | event arrays, prior, Q_c snapshots | all caller inputs unchanged | bitwise | every snapshot equal after replay | `test_d1_08_bridge_does_not_mutate_or_regenerate_inputs` | PASS |
| D1-09 | Immutable lossless artifact | one synthetic artifact | exact dtype/shape/count, non-writeable arrays, P/S Cholesky success | exact; no repair | all arrays read-only; every P/S Cholesky succeeds | `test_d1_09_artifact_dtype_shape_readonly_counts_and_spd` | PASS |
| D1-10 | Compact ST evidence | valid ST rows in input trajectory | S equals actual valid ST updates; indices select only ST rows | exact | count and index mapping equal | `test_d1_10_st_evidence_is_compact_and_matches_valid_updates` | PASS |
| D1-11 | Post-estimation truth join | artifact plus separately selected exact-time truth | geodesic, bias, NEES, and NIS equal direct evidence | `np.array_equal` | all Gate C outputs equal | `test_d1_11_separate_truth_join_gives_exact_gate_c_metric_evidence` | PASS |
| D1-12 | Raw q/−q invariance | original and globally sign-negated ST quaternion payloads | physical q/b/P/r/S and Gate C geodesic/NIS identical | `np.array_equal` | all numeric artifacts and metrics equal | `test_d1_12_raw_star_tracker_q_sign_has_identical_artifact_and_metrics` | PASS |
| D1-13 | Negative contracts | bad schema/hash/trajectory/time/order/count/index | every invalid case rejected loudly | exact exception | all seven classes rejected by `ValueError` | `test_d1_13_invalid_identity_trajectory_time_count_and_index_fail_loud` | PASS |
| D1-14 | Frozen determinism | two identical calls | no training; repeated numeric artifacts identical | exact | frozen flags true/false as specified; outputs equal | `test_d1_14_bridge_is_deterministic_frozen_and_has_no_training` | PASS |
| D1-15 | No replay/math duplication | monkeypatched replay and source AST | frozen replay called once; no core update imports, inverse/pinv, float32, or truth access | exact/static | one call; all forbidden patterns absent | `test_d1_15_frozen_replay_is_called_once_and_math_is_not_duplicated` | PASS |
| D1-P01 | Synthetic property sweep | seeds 601–605 | direct/bridge artifact equality for every seed | exact | 5/5 passed | `test_d1_synthetic_seed_sweep_exact_equivalence` | PASS |
| D1-P02 | Basilisk property sweep | seeds 701–703 | direct/bridge artifact equality for every seed | exact | 3/3 passed | `test_d1_basilisk_seed_sweep_exact_equivalence` | PASS |

## Regression matrix

| Suite | Command target | Expected | Actual | Status |
|---|---|---:|---:|---|
| Gate A | `test_mekf_conventions.py test_mekf_core.py` | 55 | 55 passed | PASS |
| Gate B1 | `test_mekf_events.py test_unit_st_synthetic.py test_mekf_replay.py` | 55 | 55 passed | PASS |
| Gate B2 | `test_basilisk_unit_st_generator.py` | 67 | 67 passed | PASS |
| Gate C | `test_mekf_metrics.py` | 43 | 43 passed | PASS |
| Legacy | four contract-specified modules | 18 + 5 subtests | 18 passed + 5 subtests | PASS |
