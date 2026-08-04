# Phase 1A Gate A Test Matrix

- 실행일: 2026-07-31
- deterministic random seed: `20260731` (`tests/test_mekf_core.py:RNG_SEED`)
- numeric type: float64
- 최초 Gate A 결과: `42 passed in 0.91s`
- Amendment A1 수정 전 재검증: `42 passed in 0.72s`
- Amendment A1 최종 결과: `55 passed in 0.59s`
- Amendment A1 legacy regression: `18 passed, 5 subtests passed in 2.08s`
- 최초 evidence log: `experiments/phase1a/agent_logs/02_math_core_new_tests.txt`
- Amendment evidence logs: `experiments/phase1a/agent_logs/02A_gate_a_before.txt`, `02A_gate_a_after.txt`, `02A_legacy_regression.txt`, `02A_property_sweep.txt`

## 1. Tolerance policy

| 범주 | Tolerance | 근거 |
|---|---:|---|
| scalar/vector algebra | absolute `1e-12` | Phase 0A convention vectors |
| normalized quaternion norm | `|norm-1| <= 1e-12` | Phase 0A convention vectors |
| analytic/central-FD vector Jacobian | relative Frobenius `<=1e-6`, max absolute `<=1e-9` | Package B3/B4 |
| reset finite-difference Jacobian | relative `<=1e-7` | Package B5 |
| covariance relative asymmetry | `<=1e-12` | Phase 0A numerical contract |
| antipodal physical output | absolute/geodesic `<=1e-12` | Package B6 |
| exact-pi scalar tie | `abs(q0) <= 8*eps64 = 1.7763568394002505e-15` | Amendment A1 machine-roundoff-only policy |
| exact-pi deterministic output | byte-identical residual where explicitly asserted; physical output difference `<=1e-12` | Amendment A1 B6 |
| state immutability | writeable flag false and direct assignment raises `ValueError` | Amendment A1 state contract |

Tolerance는 시험 실패를 해소하기 위해 확대하지 않았다.

## 2. Test matrix

| Test ID | 수학 계약 | 시험 파일/함수 | 입력 | expected behavior | tolerance | 결과 | evidence |
|---|---|---|---|---|---|---|---|
| TV-Q01 | identity quaternion/DCM | `test_mekf_conventions.py::test_tv_q01_identity_quaternion_and_dcm` | identity q, vector `[1,2,3]` | `R_NB=C_BN=I`, vector unchanged | abs `1e-12` | PASS | 신규 test log |
| TV-Q02/Q03 | active +90-degree basis mapping | `test_tv_q02_x90_basis_vector_mapping`, `test_tv_q03_z90_basis_vector_mapping`, `test_plus_90_about_each_axis_and_dcm_round_trip` | x/y/z rotations | locked B-to-N basis directions and DCM round trip | abs `1e-12` | PASS | 신규 test log |
| TV-Q04 | Hamilton composition order | `test_tv_q04_hamilton_product_and_composition_order` | x90, y90 | product `[0.5,0.5,0.5,0.5]`, `R(qp)=R(q)R(p)` | abs `1e-12` | PASS | 신규 test log |
| TV-Q05/Q06 | exponential and bias sign | `test_tv_q05_constant_rate_exponential`, `test_tv_q06_bias_cancellation_increment` | z rate; known bias | analytic increment; matched bias cancels | abs `1e-12` | PASS | 신규 test log |
| TV-M01 | body-vector residual sign | `test_tv_m01_body_vector_prediction_and_jacobian_sign` | identity, `r_N=e_x`, +z error | residual y component negative; `H=[h]_x` | abs `1e-12`, local FD `1e-7` | PASS | 신규 test log |
| TV-S01 | deterministic sun tangent basis | `test_tv_s01_sun_tangent_basis_and_jacobian`, `test_sun_tangent_basis_is_orthonormal_and_right_handed` | `h=e_x`, arbitrary unit h | `U=[e_y,e_z]`, orthonormal/right-handed | abs `1e-12` | PASS | 신규 test log |
| TV-ST01 | right-local ST residual | `test_tv_st01_star_tracker_log_residual` | `[0.01,-0.02,0.03]` | `Log(q_hat^-1 q_z)` returns injected error | abs `1e-12` | PASS | 신규 test log |
| TV-INJ01 | right injection order | `test_tv_inj01_right_injection_order` | z90 prior, +0.1 x correction | locked expected quaternion, zero residual to truth | abs `1e-12` | PASS | 신규 test log |
| TV-RST01 | exact/right first-order reset | `test_tv_rst01_exact_and_first_order_right_reset_jacobian` | `[0.1,-0.2,0.05]` | expected first-order and rounded exact `J_r` | abs `5e-9` for rounded table | PASS | 신규 test log |
| TV-SIGN01 | antipodal equivalence | `test_tv_sign01_antipodal_rotation_residual_and_distance` | `q`, `-q` | same DCM, zero residual/distance | abs `1e-12` | PASS | 신규 test log |
| TV-KF01 | simple ST gain/Joseph value | `test_tv_kf01_simple_star_tracker_gain_and_joseph_covariance` | locked diagonal P/R | `S=.05I`, `K=[.8I;0]`, expected P | abs `1e-12` | PASS | 신규 test log |
| TV-COV01 | symmetry/SPD diagnostics | `test_tv_cov01_symmetry_and_spd_diagnostics` | diagonal SPD | zero asymmetry, successful Cholesky | asym `1e-12` | PASS | 신규 test log |
| TV-LONG01 | long-horizon norm/composition | `test_tv_long01_long_horizon_quaternion_norm_and_composition` | 1000 exact increments | norm bound and equivalence to batch Exp | abs `1e-12` | PASS | 신규 test log |
| SO3-EDGE | inverse, normalize, Exp/Log boundaries | `test_quaternion_inverse_conjugate_and_normalization`, `test_exp_log_round_trip_near_zero_and_near_pi` | near zero, ordinary, near pi | algebra identity and shortest-arc round trip | abs/rel `5e-13` | PASS | 신규 test log |
| B1-STATE | zero/constant-rate/bias propagation | `test_mekf_core.py::test_b1_zero_motion_preserves_nominal_attitude_and_bias`, `test_b1_constant_rate_matches_analytic_exponential`, `test_b1_known_bias_cancellation_and_bias_sign` | zero motion, constant omega, bias | unchanged/analytic/correct sign | abs `1e-12` | PASS | 신규 test log |
| B1-FG | continuous dynamics blocks | `test_b1_continuous_error_matrices_shape_sign_and_units_blocks` | arbitrary corrected omega | exact locked F/G blocks and shapes | abs `1e-12` | PASS | 신규 test log |
| B1-PHI | local transition FD | `test_b1_exact_transition_matches_finite_difference_local_error_map` | nonlinear right-error one-step map | central FD Jacobian agrees with exact `exp(Fdt)` | relative `1e-7` | PASS | 신규 test log |
| B1-QD | Van Loan and limiting relations | `test_b1_van_loan_shapes_symmetry_psd_and_substep_composition`, `test_b1_zero_dt_and_first_order_limit` | PSD Qc, full/half/tiny dt | symmetric PSD Qd, substep composition, first-order limit | stated per-test float64 bounds | PASS | 신규 test log |
| B3 | body-vector analytic Jacobian | `test_b3_body_vector_analytic_jacobian_matches_central_difference`, `test_b3_locked_identity_frame_and_residual_sign` | 100 seeded q/reference cases | analytic `H=[h]_x` and locked sign agree with central FD | rel `1e-6`, abs `1e-9` | PASS | 신규 test log |
| B4 | sun tangent analytic Jacobian | `test_b4_sun_tangent_analytic_jacobian_matches_central_difference` | 100 seeded unit-vector cases | U constraints, rank 2, analytic vs central FD | rel `1e-6`, abs `1e-9` | PASS | 신규 test log |
| B5 | injection/reset consistency | `test_b5_known_attitude_and_bias_injection_removes_local_residual`, `test_b5_exact_reset_jacobian_matches_central_difference`, `test_b5_covariance_reset_preserves_symmetry_and_spd` | known correction, seeded SPD P | correction removes error; exact Jr matches FD; SPD retained | residual `1e-12`, reset rel `1e-7` | PASS | 신규 test log |
| B6 | ordinary/near-pi measurement/nominal antipodal update | `test_b6_star_tracker_measurement_antipodes_produce_same_update`, `test_b6_nominal_antipodes_produce_same_physical_posterior_and_covariance` | q/-q measurement and prior pairs | identical residual, correction, physical posterior, covariance | abs/geodesic `1e-12` | PASS | 최초 신규 test log |
| A1-PI-LOG | exact-pi log tie and metric | `test_exact_pi_quat_log_antipodes_use_deterministic_axis_tie_break` | explicit zero-scalar x/y/z/arbitrary-axis q and -q | first significant axis component positive; log byte-identical; geodesic zero | tie `8*eps64`, abs `1e-12` | PASS | Amendment final log/property sweep |
| A1-PI-UPD | exact-pi measurement full update | `test_b6_exact_pi_x_antipodes_produce_identical_full_update`, `test_b6_exact_pi_other_axes_antipodes_produce_identical_update` | x/y/z/arbitrary exact-pi q and -q | identical residual/correction, physical posterior, covariance | exact where asserted; otherwise `1e-12` | PASS | Amendment final log/property sweep |
| A1-PI-NOM | exact-pi nominal sign pair | `test_b6_exact_pi_nominal_sign_flip_preserves_physical_update` | q_hat and -q_hat with common exact-pi measurement | identical residual/correction, physical posterior, covariance | `1e-12` | PASS | Amendment final log/property sweep |
| A1-PI-NEAR | both sides of pi outside tie | `test_near_pi_outside_tie_preserves_shortest_arc_on_both_sides` | `pi +/- 1e-10` | original shortest-arc physical behavior and antipodal equivalence preserved | `5e-13` round-trip, `1e-12` antipodal | PASS | Amendment final log/property sweep |
| A1-IMM-COPY | defensive array ownership | `test_state_initialization_defensively_copies_all_input_arrays` | caller q/b/P mutated after construction | stored state unchanged | exact array equality | PASS | Amendment final log/property sweep |
| A1-IMM-RO | direct mutation rejection | `test_state_q_nb_is_read_only`, `test_state_b_g_is_read_only`, `test_state_p_is_read_only` | direct element assignments | writeable false and `ValueError` | exception required | PASS | Amendment final log/counterexample after |
| A1-IMM-STATE | functional state transitions | `test_failed_propagation_and_update_leave_prior_state_unchanged`, `test_successful_propagation_and_update_return_new_states_without_mutating_prior` | invalid and valid predict/update | prior unchanged; valid call returns distinct read-only state | exact array equality/object identity | PASS | Amendment final log/property sweep |
| NUM-JOSEPH | Joseph symmetry/SPD | `test_numerical_safety_joseph_update_is_symmetric_and_spd` | SPD P/R | symmetric strictly SPD P_c | asym `1e-12` | PASS | 신규 test log |
| NUM-CHOLESKY | valid SPD solve | `test_numerical_safety_valid_spd_cholesky_solve_has_small_residual` | 2x2 SPD system | Cholesky solve residual within roundoff | abs `1e-12` | PASS | 신규 test log |
| NUM-NEGATIVE | deliberate non-SPD P/S | `test_numerical_safety_non_spd_p_and_s_fail_loud` | negative/indefinite matrices | explicit `NumericalSafetyError`, no repair | exception required | PASS | 신규 test log |
| NUM-INPUT | nonfinite/shape fail-loud | `test_numerical_safety_nonfinite_and_invalid_shapes_fail_before_state_mutation` | NaN, Inf, wrong shape | error before immutable state changes | exact exception class/message class | PASS | 신규 test log |
| NUM-NOFALLBACK | prohibited recovery absence | `test_numerical_safety_has_no_inverse_or_pseudoinverse_recovery_path` | implementation source | no explicit/pseudo inverse, clipping, silent perturbation path | exact source assertion | PASS | 신규 test log |
| INFO-BOUNDARY | no truth/oracle input | `test_estimator_public_api_has_no_truth_or_oracle_input` | public signatures | no truth/oracle/event parameter | exact name assertion | PASS | 신규 test log |
| IMPORT-BOUNDARY | runner/Basilisk/torch/viz independence | `test_import_boundary_does_not_load_forbidden_packages` | fresh subprocess import | forbidden module list remains empty | exact empty JSON list | PASS | 신규 test log |
| LEGACY | pre/post existing regression | designated four existing test files | current approved tree | identical pass count before and after implementation | exit code 0 | PASS | baseline before/after logs |

## 3. Execution evidence

New Gate A exact command:

```text
PYTHONDONTWRITEBYTECODE=1 /home/dss-pc-05/.pyenv/versions/3.10.13/bin/python -m pytest -q -p no:cacheprovider tests/test_mekf_conventions.py tests/test_mekf_core.py
```

Result: `42 passed in 0.91s`, exit code 0, measured command duration `1.760958s`.

Legacy exact command:

```text
PYTHONDONTWRITEBYTECODE=1 /home/dss-pc-05/.pyenv/versions/3.10.13/bin/python -m pytest -q -p no:cacheprovider tests/test_basilisk_imu_generator.py tests/test_basilisk_mrp_ekf.py bench/tests/test_generator_contract_tg0.py bench/tests/test_adcs_event_metrics.py
```

Before: `18 passed, 5 subtests passed in 4.40s`, exit code 0.

After: `18 passed, 5 subtests passed in 4.50s`, exit code 0.

Amendment A1 used the same Gate A command before and after the patch.

```text
Before: 42 passed in 0.72s, exit code 0, measured duration 1.222206248s
After:  55 passed in 0.59s, exit code 0, measured duration 0.98792902s
```

The exact legacy command above was then rerun:

```text
18 passed, 5 subtests passed in 2.08s
exit code 0
measured duration 3.089991448s
```

The deterministic Amendment property sweep covered four exact-pi axes, 1,000 ordinary antipodal update pairs, 256 near-pi outside-tie pairs, nominal-sign paired update, defensive copies, direct writes, and prior-state preservation. All declared differences were zero except near-pi shortest-arc round-trip error `1.7763568394002505e-15`, within the unchanged tolerance.

No test was skipped, xfailed, or relaxed. Repository-wide and visualization tests were not collected.
