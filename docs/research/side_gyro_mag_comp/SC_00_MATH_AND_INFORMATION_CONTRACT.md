# SC-00 Math and Information Contract

Status: frozen for the minimal `side_gyro_mag_comp_v1` pilot.

## State, frames, and event order

The nominal state is `(q_NB, b_r^B)`, with scalar-first Hamilton active body-to-navigation quaternion and right-local error

`q_true = q_hat ⊗ Exp(delta_theta)`, `delta_x = [delta_theta, delta_b_r] in R^6`.

Gyro packets arrive in sensor-frame rad/s and are transformed into body-frame propagation inputs; they are not measurement updates. Magnetometer packets likewise arrive in their sensor frame and are transformed into body-frame vector measurements. No value carrying a timestamp later than `t` may influence a quantity computed at `t`. Within `t`, the only legal order is gyro compensation, propagation, magnetometer compensation, magnetometer update. A stage may consume only outputs of earlier stages at `t`, or outputs from times `<t`. Thus `f_g,t` and `f_m,t` may condition `G1_t` and `G2_t`, but the posterior `delta_x_t` and posterior state may not influence `K_t`, `G1_t`, `G2_t`, `f_g,t`, or `f_m,t`.

## Deterministic compensation and residual bias

The gyro sensor model is

`y_g^Sg = A_g C_SgB (omega_true^B + b_r^B) + c_g^Sg + n_g^Sg`.

`A_g` and `A_m` are invertible; the pilot support requires every singular value in `[0.8,1.2]` and condition number `<=1.5`. Mounting matrices are proper rotations.

The causal encoder returns `(omega_tilde_g^B, f_g)`, `f_g in R^8`. Its correction target retains the stochastic residual bias:

`omega_target^B = C_BSg A_g^-1 (y_g^Sg - c_g^Sg) = omega_true^B + b_r^B + transformed noise`.

Propagation therefore uses `omega_tilde_g^B - b_hat_r^B`. The compensator must not remove `b_r`; doing so would double-count the MEKF residual-bias state.

The magnetometer model and inverse compensation are

`y_m^Sm = A_m C_SmB C_BN(q_true) m_N + b_m^Sm + n_m^Sm`,

`z_tilde_m^B = C_BSm A_m^-1 (y_m^Sm - b_m^Sm)`.

Hard iron is subtracted first, inverse soft iron applied second, and the result mounted into body coordinates third. With `h_m = C_BN(q_hat) m_N`, the innovation and Jacobian are

`nu_m = z_tilde_m^B - h_m`, `H_m = [[h_m]_x, 0_3x3]`.

## Split gain and FiLM

The side backbone produces latent factors `G1^0 in R^(6x6)` and `G2^0 in R^(3x3)`. Gyro features modulate only `G1`; magnetometer features modulate only `G2`:

`G1 = FiLM_1(G1^0; f_g)`, `G2 = FiLM_2(G2^0; f_m)`, `f_g,f_m in R^8`.

Feature-off is exact identity (`gamma=1`, `beta=0`). The gain is `K = G1 H_m^T G2 in R^(6x3)` and `delta_x = K nu_m`. Injection is `q+ = q- ⊗ Exp(delta_theta)`, `b_r+ = b_r- + delta_b_r`, followed by the existing right-error reset. `G1`, `G2`, and `K` are neural latent/update factors, not physical `P`, `Q`, `R`, or `S^-1`; neural results carry no NIS/NEES/covariance-validity claim.

## Runtime boundary

Deployable inputs are limited to causal gyro and magnetometer packets, timestamps and validity flags, a train-frozen onboard magnetic reference, prior estimator state/history, and train-frozen normalization constants. Truth, true bias, calibration/event parameters, oracle corrections/scales, event labels/windows, future packets, test-derived normalization, and evaluation metrics are forbidden. Oracle paths are diagnostic sidecars and cannot share a deployable namespace.

## Observability

For a single magnetic vector, `rank([h_m]_x)=2`; instantaneous rotation parallel to `h_m` is weak/unobservable. Every evaluated regime must contain a non-empty weak-axis and observable-plane population, and conclusions are limited to the tested dynamics/reference vector.

## Mandatory red paths

Implementation must make the following tests fail when the named contract is false: `test_sc_qnb_right_injection_fixture_red`, `test_sc_gyro_body_rad_s_right_propagation_red`, `test_sc_deterministic_vs_residual_bias_separation_red`, `test_sc_gyro_oracle_retains_residual_bias_red`, `test_sc_mag_hard_soft_mounting_inverse_red`, `test_sc_mag_jacobian_sign_finite_difference_red`, `test_sc_split_gain_shape_and_right_injection_red`, `test_sc_feature_dim_exactly_eight_red`, `test_sc_film_feature_off_exact_equivalence_red`, `test_sc_film_branch_isolation_red`, `test_sc_causal_prefix_invariance_red`, `test_sc_deployable_namespace_leakage_rejected_red`, and `test_sc_single_mag_weak_axis_red`.

Also mandatory are `test_sc_learned_compensator_residual_bias_retention_red`, `test_sc_intra_timestamp_stage_order_red`, `test_sc_right_error_reset_red`, `test_sc_pairing_split_firewall_red`, `test_sc_frozen_gate_population_red`, and `test_sc_n3s_single_intervention_red`. On the diagnostic split, learned residual retention requires the OLS slope of `(omega_tilde_learned-omega_true)` on `b_r` to lie in `[0.9,1.1]`, and the componentwise mean of `(omega_tilde_learned-omega_target)` to be within `2` standard errors of zero. The intra-timestamp test must reject update-before-propagation. The reset test must reject identity covariance reset in place of `right_jacobian_so3`. The pairing test rejects overlap, realization drift, support leakage, or namespace reuse. The gate test rejects any altered population/seed/cluster/statistic. The N3S test rejects checkpoint/value/timestamp/state changes, any fixed point, or a time-varying permutation.

Red fixtures are non-vacuous: injection/Jacobian fixtures use `q_hat != identity` and a `delta_theta` not parallel to its rotation axis; the magnetometer inverse fixture uses nonzero hard iron, anisotropic `A_m`, and nonidentity mounting. The normative sign anchor is specifically `nu=z_tilde-h`; jointly flipping `(nu,H)` is a classical algebraic gauge, and a global learned-factor sign alone is not accepted as a sign test.
