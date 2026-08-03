# P1A-CP4 Integration Test Matrix

Interpreter for every pytest command:
`/home/dss-pc-05/.pyenv/versions/3.10.13/bin/python`, with
`PYTHONDONTWRITEBYTECODE=1` and `pytest -p no:cacheprovider`.

| ID | Input / action | Expected | Tolerance | Actual | Evidence | Status |
|---|---|---|---|---|---|---|
| CP4-01 | Load `suite_phase1a_unit_st_smoke.yaml` | Fixed task/model IDs, two producers, three seeds, Tier-0 metadata | Exact strings/sets | Contract fields present | `test_cp4_01_smoke_yaml_contract_and_fixed_ids` | PASS |
| CP4-02 | Query legacy and typed registries; dispatch synthetic producer | Separate typed bridge; legacy lookup unchanged | Exact class/list behavior | Typed ID resolves only through typed registry | `test_cp4_02_dispatch_and_separate_registry_are_append_only` | PASS |
| CP4-03 | Synthetic seeds 6201–6203, runner fresh then hit | Six successful runs; equal identity and arrays per seed | `np.array_equal` | 3 fresh + 3 verified hits | `test_cp4_03_fresh_and_cache_hit_synthetic_runner_property` | PASS |
| CP4-04 | Basilisk seeds 6301–6303, runner fresh then hit | Six successful runs; equal identity and arrays per seed | `np.array_equal` | 3 fresh + 3 verified hits | `test_cp4_04_fresh_and_cache_hit_basilisk_runner_property` | PASS |
| CP4-05 | Reload a complete synthetic sidecar | Strict verified hit; exactly three files; five hashes unchanged | Exact set/dataclass equality | `manifest.json`, `truth.npz`, `events.npz` only | `test_cp4_05_verified_cache_is_exact_three_file_sidecar` | PASS |
| CP4-06 | Install a self-consistent manifest with stale MEKF source hash | Reject current-source mismatch | No fallback/regeneration | `ValueError: source fingerprint mismatch` | `test_cp4_06_stale_source_fingerprint_cache_is_rejected` | PASS |
| CP4-07 | Replay one serialized realization directly, through D1, and through runner | q/b/P/r/S and all dataset identity fields equal | `np.array_equal`; exact dict equality | All five array groups and identity equal | `test_cp4_07_direct_bridge_runner_q_b_P_r_S_and_identity_are_exact`; exact-equivalence log | PASS |
| CP4-08 | Change model display/training note and nonsemantic task review note | Same sidecar hash, identity, and runner arrays | Exact equality | Unchanged dataset config/identity/q/b/P/r/S | `test_cp4_08_nonsemantic_model_and_task_metadata_preserve_realization` | PASS |
| CP4-09 | Guard D1 replay arguments and inspect runner call order | No truth/oracle/label input; metrics after replay | Exact signature/source order | Typed event table only; replay precedes truth access | `test_cp4_09_truth_never_crosses_estimator_boundary` | PASS |
| CP4-10 | Join valid artifact; perturb one timestamp by one ULP | Valid exact join; perturbed join rejected | Exact float64 equality; no interpolation | Valid shapes match; mismatch raises | `test_cp4_10_truth_join_is_exact_and_never_interpolates` | PASS |
| CP4-11 | Load runner trajectory NPZ with `allow_pickle=False` | Lossless required fields, no object/truth arrays | Exact field set | Twelve estimator/evidence arrays only | `test_cp4_11_lossless_artifact_round_trip_and_no_truth_fields` | PASS |
| CP4-12 | Compare ST compact arrays with ST event positions | One r/S row per ST update with exact index/order/time | `np.array_equal` | Counts and alignments exact | `test_cp4_12_star_tracker_evidence_is_compact_and_exactly_aligned` | PASS |
| CP4-13 | Negate every star-tracker quaternion | Identical q/b/P/r/S | `np.array_equal` | All five groups identical | `test_cp4_13_star_tracker_q_and_negative_q_replay_are_identical` | PASS |
| CP4-14 | Replace dense loader/batcher/adapter methods with failures | Exact pair succeeds without invoking any | Zero calls | Runner succeeds; forbidden methods untouched | `test_cp4_14_exact_pair_never_calls_dense_float32_runner_path` | PASS |
| CP4-15 | Request `trained:frozen` | Fail before generation/replay; no artifact | Exact plan policy | Failed with training/adaptation-disabled message | `test_cp4_15_training_and_adaptation_are_disabled` | PASS |
| CP4-16 | Inject D1 replay failure | Failure record; no final or partial valid artifact | No partial directory | `failure.json` only; no `mekf_replay`/partial | `test_cp4_16_replay_failure_leaves_no_partial_valid_artifact` | PASS |
| CP4-17 | Pair typed task with legacy `oracle_kf` while dense loader is fatal | Fail exact-pair validation before dense lifecycle | Zero dense calls | Exact-pair error returned | `test_cp4_17_partial_pair_fails_before_legacy_lifecycle` | PASS |
| CP4-18 | Inspect canonical manifest, metric fields, NIS/NEES counts, P/S SPD | Complete provenance and finite canonical metrics | Strict SPD; exact counters | Eight identity fields; counts match; min eigenvalues positive | `test_cp4_18_manifest_provenance_metrics_and_spd_contract` | PASS |

## Executed suite results

```text
P1A-CP4 integration: 22 passed
Gate A:              55 passed
Gate B1:             55 passed
Gate B2:             67 passed
Gate C:              43 passed
D1 bridge:           24 passed
Legacy:              18 passed, 5 subtests passed
```

The 22 pytest cases implement the 18 logical rows above; CP4-03 and CP4-04 are
each parameterized over three independent seeds.

Actual smoke CLI ran two tasks × three seeds twice. The first invocation logged
six `fresh_generation` results. The second logged six `verified_cache_hit`
results with the same per-seed dataset hashes.
