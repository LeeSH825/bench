# Phase 1A Gate B1 test matrix

Interpreter for every row: Python 3.10.13 with `PYTHONDONTWRITEBYTECODE=1` and
pytest cache disabled. “Exact” means array equality or contract validation,
not a relaxed numerical threshold.

| Test ID | Contract | Test function | Input | Expected | Tolerance | Result | Evidence |
|---|---|---|---|---|---|---|---|
| E01 | Exact schema layout | `test_exact_dtypes_shapes_and_readonly_arrays` | Representative data | Frozen dtype/shape; read-only | Exact | PASS | B1 pytest log |
| E02 | Fail-loud schema | `test_invalid_dtype_rank_or_shape_fails_loudly` | Wrong dtype/rank/shape cases | Construction rejects | Exact exception | PASS | B1 pytest log |
| E03 | Typed payload ownership | `test_payload_index_mismatch_and_range_fail_loudly` | Out-of-range gyro index | Construction rejects | Exact exception | PASS | B1 pytest log |
| E04 | Unit quaternion | `test_star_quaternion_must_be_normalized` | Scaled ST quaternion | Construction rejects | `2e-13` norm guard | PASS | B1 pytest log |
| E05 | SPD `R_ST` | `test_star_covariance_must_be_spd` | Negative diagonal | Construction rejects | Cholesky | PASS | B1 pytest log |
| E06 | Zero latency | `test_zero_latency_is_exact_and_nonzero_latency_is_rejected` | One-ULP arrival change | Construction rejects | Exact equality | PASS | B1 pytest log |
| E07 | Canonical ordering | `test_event_sort_and_same_time_gyro_before_star_tracker_are_enforced` | Reversed tie/order | Construction rejects | Exact key | PASS | B1 pytest log |
| E08 | Three-file round trip | `test_serialization_round_trip_and_semantic_hash_equality` | Temp artifact | Every array/hash equal | Exact | PASS | B1 pytest log |
| E09 | No pickle/object | `test_object_array_npz_is_rejected_without_pickle` | Object-dtype NPZ | Load rejects | Exact exception | PASS | B1 pytest log |
| E10 | Mutation sensitivity | `test_payload_order_and_config_mutations_change_their_semantic_hashes` | Payload/order/config changes | Corresponding hash changes | SHA-256 | PASS | B1 pytest log |
| E11 | Corruption detection | `test_corrupted_manifest_and_recorded_hash_are_rejected` | Invalid JSON/false hash | Load rejects | Exact exception | PASS | B1 pytest log |
| A101 | Generator-ID round trip / frozen NPZ schema | `test_versioned_generator_identity_round_trip_and_npz_schema_invariance` | Synthetic and Basilisk versioned IDs over the same deterministic fixture | Strict expected-ID load; exact three files, keys, dtype, rank, arrays, and data hashes | Exact/SHA-256 | PASS | `03A1_post_gate_b1.txt` |
| A102 | Expected generator identity | `test_expected_generator_identity_mismatch_fails_loudly` | Synthetic artifact, Basilisk expectation | Explicit `ValueError` | Exact exception | PASS | `03A1_post_gate_b1.txt` |
| A103 | Strict version syntax | `test_empty_malformed_or_unversioned_generator_identity_is_rejected` | Empty, whitespace, unversioned, uppercase, underscore, v0/v01, non-string IDs | Save rejects every case | Exact exception | PASS | `03A1_post_gate_b1.txt` |
| A104 | Identity corruption | `test_generator_identity_tamper_is_detected_by_manifest_hash` | Canonical manifest changed to another valid generator ID without hash update | Load rejects hash mismatch | SHA-256 | PASS | `03A1_post_gate_b1.txt` |
| A105 | Schema identity remains frozen | `test_unsupported_schema_identity_is_rejected_on_save_and_load` | Unsupported `p1a-mekf-events-v2` | Save and load reject | Exact exception | PASS | `03A1_post_gate_b1.txt` |
| A106 | Required generator families | `test_generator_identity_validator_accepts_required_families` | Synthetic/Basilisk v1 IDs | Both accepted as distinct versioned identities | Exact | PASS | `03A1_post_gate_b1.txt` |
| A107 | Synthetic data semantics unchanged | `test_representative_synthetic_data_semantic_hashes_are_unchanged` | Seed 731 representative config | Frozen truth/sensor/event/dataset hashes | SHA-256 | PASS | `03A1_post_gate_b1.txt` |
| G01 | Repeatability/IDs | `test_same_seed_and_config_have_identical_hashes_and_ids` | Same config twice | All hashes/IDs equal | Exact | PASS | B1 pytest log |
| G02 | Sensor seed isolation | `test_sensor_seed_isolation_preserves_truth_and_changes_sensor_hash` | Alternate gyro namespace | Truth same, sensor differs | SHA-256 | PASS | B1 pytest log |
| G03 | Truth seed isolation | `test_truth_seed_change_changes_truth_hash` | Alternate truth namespace | Truth hash differs | SHA-256 | PASS | B1 pytest log |
| G04 | Sign representation | `test_sign_seed_changes_only_quaternion_representation_when_it_changes_raw_hash` | Alternate sign namespace | `|q·q′|=1`, raw hash differs | `2e-15` dot | PASS | B1 pytest log |
| G05 | Exact cadence | `test_schedule_rate_counts_st_subset_and_zero_latency` | 20 Hz / 5 Hz | Counts exact; ST proper subset | Exact | PASS | B1 pytest log |
| G06 | Gyro equation | `test_gyro_equation_sign_and_units_are_locked` | First derived noise draw | `ω_m=ω+b+n` | Exact array | PASS | B1 pytest log |
| G07 | ST equation | `test_star_tracker_right_local_noise_construction_is_locked` | First derived noise draw | Physical quaternion equal | `2e-15` dot | PASS | B1 pytest log |
| G08 | Manifest identity | `test_representative_metadata_is_complete` | Representative manifest | Versions/config/software/source present | Exact | PASS | B1 pytest log |
| G09 | Whole split/selection | `test_whole_trajectory_split_is_disjoint_complete_deterministic_and_order_independent` | Reversed IDs and selected val | Disjoint union; complete selection | Exact sets | PASS | B1 pytest log |
| G10 | Split fail-loud | `test_split_rejects_duplicates_too_few_and_invalid_fractions` | Duplicate/few/bad fractions | Reject all | Exact exception | PASS | B1 pytest log |
| G11 | Split seed isolation | `test_different_split_seed_changes_split_without_changing_data_hashes` | Alternate split namespace | Assignment changes; data hashes same | Exact/SHA-256 | PASS | B1 pytest log |
| R01 | Analytic zero-noise replay | `test_zero_noise_exact_initial_state_tracks_analytic_truth` | Exact initial state/bias | Final analytic truth | `2e-12` rad/vector | PASS | B1 pytest log |
| R02 | Finite low-rate smoke | `test_low_rate_star_tracker_smoke_is_finite_spd_and_bounded` | 4 s, 20/2 Hz | Finite, normalized, SPD | Norm `2e-14` | PASS | B1 pytest log |
| R03 | Replay determinism | `test_same_stream_replays_identically` | Same stream twice | All evidence equal | Exact | PASS | B1 pytest log |
| R04 | Serialized replay | `test_serialization_round_trip_replay_equivalence` | Before/after load | All evidence equal | Exact | PASS | B1 pytest log |
| R05 | `q/-q` replay | `test_star_tracker_q_and_negative_q_have_identical_posteriors` | All ST signs negated | q/b/P/residual/S equal | Exact | PASS | B1 pytest log |
| R06 | Long replay safety | `test_long_sequence_quaternion_norm_and_covariance_spd` | 10 s event stream | Unit q; symmetric SPD P | Norm `2e-14`; Cholesky | PASS | B1 pytest log |
| R07 | Malformed replay | `test_malformed_order_and_unaligned_star_time_fail_loudly` | Reversed tie; shifted ST time | Reject both | Exact exception | PASS | B1 pytest log |
| R08 | Truth-free API | `test_replay_public_api_exposes_no_truth_or_oracle_inputs` | Signature inspection | Exact five inputs only | Exact | PASS | B1 pytest log |
| R09 | Defensive boundaries | `test_replay_does_not_mutate_inputs_and_outputs_are_readonly` | Events/truth/prior/result | Inputs equal; output arrays read-only | Exact | PASS | B1 pytest log |
| R10 | Frozen Gate A state | `test_gate_a_state_immutability_remains_enforced` | Direct q/b/P mutation | All raise | Exact exception | PASS | B1 pytest log |
| P01 | Multi-seed property sweep | standalone fixed-seed sweep | 10 datasets × 4 trajectories | Regen/roundtrip/sign/split/replay all pass | Exact plus SPD/finite | PASS | `03A_property_sweep.txt` |
| A1P01 | Generator identity property sweep | standalone Amendment A1 sweep | 2 IDs × 5 seeds | Strict round trip/hash, match/mismatch, schema/corruption/object rejection, exact arrays/replay | Exact/SHA-256 | PASS | `03A1_generator_identity_property_sweep.txt` |

Gate A and the designated legacy regression are reported separately because
their files and expectations remain frozen.
