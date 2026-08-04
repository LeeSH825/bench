from __future__ import annotations

import inspect
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

import bench.estimators.mekf as mekf
from bench.estimators.mekf import (
    MEKFState,
    NumericalSafetyError,
    assert_positive_definite,
    body_vector_jacobian,
    body_vector_prediction,
    cholesky_solve_spd,
    continuous_error_matrices,
    continuous_noise_covariance,
    discretize_van_loan,
    inject_error_state,
    joseph_covariance_update,
    kalman_gain,
    propagate_state,
    quat_exp,
    quat_geodesic_angle,
    quat_inverse,
    quat_log,
    quat_multiply,
    quat_to_dcm,
    reset_covariance,
    right_jacobian_so3,
    skew,
    star_tracker_residual,
    star_tracker_update,
    sun_tangent_jacobian,
)


ATOL = 1.0e-12
RNG_SEED = 20260731


def _state(q: np.ndarray | None = None, b: np.ndarray | None = None) -> MEKFState:
    return MEKFState(
        q_NB=np.array([1.0, 0.0, 0.0, 0.0]) if q is None else q,
        b_g=np.zeros(3) if b is None else b,
        P=np.diag([0.04, 0.03, 0.02, 0.01, 0.008, 0.006]),
    )


def _zero_qc() -> np.ndarray:
    return continuous_noise_covariance(np.zeros((3, 3)), np.zeros((3, 3)))


def _random_quaternion(rng: np.random.Generator) -> np.ndarray:
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    angle = rng.uniform(-2.5, 2.5)
    return quat_exp(axis * angle)


def _one_step_error_map(
    delta_x: np.ndarray,
    q_hat: np.ndarray,
    b_hat: np.ndarray,
    omega_m: np.ndarray,
    dt: float,
) -> np.ndarray:
    q_true = quat_multiply(q_hat, quat_exp(delta_x[:3]))
    b_true = b_hat + delta_x[3:]
    q_hat_next = quat_multiply(q_hat, quat_exp((omega_m - b_hat) * dt))
    q_true_next = quat_multiply(q_true, quat_exp((omega_m - b_true) * dt))
    attitude_error_next = quat_log(quat_multiply(quat_inverse(q_hat_next), q_true_next))
    return np.concatenate((attitude_error_next, b_true - b_hat))


def test_b1_zero_motion_preserves_nominal_attitude_and_bias() -> None:
    state = _state(b=np.array([0.01, -0.02, 0.005]))
    result = propagate_state(state, omega_m=state.b_g, dt=0.25, Q_c=_zero_qc())

    np.testing.assert_allclose(result.state.q_NB, state.q_NB, atol=ATOL, rtol=0.0)
    np.testing.assert_allclose(result.state.b_g, state.b_g, atol=ATOL, rtol=0.0)
    assert abs(float(np.linalg.norm(result.state.q_NB)) - 1.0) <= ATOL


def test_b1_constant_rate_matches_analytic_exponential() -> None:
    state = _state()
    omega = np.array([0.2, -0.1, 0.4])
    dt = 0.125
    result = propagate_state(state, omega_m=omega, dt=dt, Q_c=_zero_qc())

    np.testing.assert_allclose(result.state.q_NB, quat_exp(omega * dt), atol=ATOL, rtol=0.0)
    assert abs(float(np.linalg.norm(result.state.q_NB)) - 1.0) <= ATOL


def test_b1_known_bias_cancellation_and_bias_sign() -> None:
    bias = np.array([0.01, -0.02, 0.005])
    matched = propagate_state(_state(b=bias), omega_m=bias, dt=1.0, Q_c=_zero_qc())
    unmatched = propagate_state(_state(), omega_m=bias, dt=1.0, Q_c=_zero_qc())

    np.testing.assert_allclose(matched.state.q_NB, [1.0, 0.0, 0.0, 0.0], atol=ATOL)
    np.testing.assert_allclose(unmatched.state.q_NB, quat_exp(bias), atol=ATOL, rtol=0.0)
    np.testing.assert_allclose(unmatched.omega_corrected, bias, atol=ATOL, rtol=0.0)


def test_b1_continuous_error_matrices_shape_sign_and_units_blocks() -> None:
    omega = np.array([0.3, -0.2, 0.1])
    f, g = continuous_error_matrices(omega)

    assert f.shape == (6, 6)
    assert g.shape == (6, 6)
    np.testing.assert_allclose(f[:3, :3], -skew(omega), atol=ATOL, rtol=0.0)
    np.testing.assert_allclose(f[:3, 3:], -np.eye(3), atol=ATOL, rtol=0.0)
    np.testing.assert_allclose(f[3:], np.zeros((3, 6)), atol=ATOL, rtol=0.0)
    np.testing.assert_allclose(g, np.diag([-1.0, -1.0, -1.0, 1.0, 1.0, 1.0]), atol=ATOL)


def test_b1_exact_transition_matches_finite_difference_local_error_map() -> None:
    q_hat = quat_exp([0.3, -0.1, 0.2])
    b_hat = np.array([0.01, -0.02, 0.005])
    omega_m = np.array([0.4, -0.3, 0.2])
    dt = 0.01
    f, g = continuous_error_matrices(omega_m - b_hat)
    phi = discretize_van_loan(f, g, np.zeros((6, 6)), dt).Phi
    epsilon = 1.0e-7
    finite_difference = np.empty((6, 6))
    for column in range(6):
        perturbation = np.zeros(6)
        perturbation[column] = epsilon
        plus = _one_step_error_map(perturbation, q_hat, b_hat, omega_m, dt)
        minus = _one_step_error_map(-perturbation, q_hat, b_hat, omega_m, dt)
        finite_difference[:, column] = (plus - minus) / (2.0 * epsilon)

    relative_error = np.linalg.norm(finite_difference - phi) / np.linalg.norm(phi)
    assert relative_error <= 1.0e-7


def test_b1_van_loan_shapes_symmetry_psd_and_substep_composition() -> None:
    f, g = continuous_error_matrices([0.2, -0.1, 0.3])
    q_c = continuous_noise_covariance(2.0e-6 * np.eye(3), 3.0e-8 * np.eye(3))
    full = discretize_van_loan(f, g, q_c, 0.1)
    half = discretize_van_loan(f, g, q_c, 0.05)
    composed_phi = half.Phi @ half.Phi
    composed_q = half.Phi @ half.Q_d @ half.Phi.T + half.Q_d

    assert full.Phi.shape == (6, 6)
    assert full.Q_d.shape == (6, 6)
    np.testing.assert_allclose(full.Q_d, full.Q_d.T, atol=1.0e-15, rtol=0.0)
    assert np.linalg.eigvalsh(full.Q_d)[0] >= -1.0e-12
    np.testing.assert_allclose(full.Phi, composed_phi, atol=5.0e-15, rtol=5.0e-15)
    np.testing.assert_allclose(full.Q_d, composed_q, atol=5.0e-18, rtol=5.0e-13)


def test_b1_zero_dt_and_first_order_limit() -> None:
    f, g = continuous_error_matrices([0.2, 0.1, -0.3])
    q_c = continuous_noise_covariance(1.0e-5 * np.eye(3), 2.0e-7 * np.eye(3))
    zero = discretize_van_loan(f, g, q_c, 0.0)
    tiny_dt = 1.0e-8
    tiny = discretize_van_loan(f, g, q_c, tiny_dt)

    np.testing.assert_allclose(zero.Phi, np.eye(6), atol=0.0, rtol=0.0)
    np.testing.assert_allclose(zero.Q_d, np.zeros((6, 6)), atol=0.0, rtol=0.0)
    np.testing.assert_allclose(tiny.Phi, np.eye(6) + f * tiny_dt, atol=2.0e-16, rtol=0.0)
    np.testing.assert_allclose(tiny.Q_d, g @ q_c @ g.T * tiny_dt, atol=2.0e-20, rtol=1.0e-8)


def test_b3_body_vector_analytic_jacobian_matches_central_difference() -> None:
    rng = np.random.default_rng(RNG_SEED)
    epsilon = 3.0e-7
    worst_relative_error = 0.0
    worst_absolute_error = 0.0
    for _ in range(100):
        q_hat = _random_quaternion(rng)
        reference_n = rng.normal(size=3)
        reference_n /= np.linalg.norm(reference_n)
        prediction = body_vector_prediction(q_hat, reference_n)
        analytic = body_vector_jacobian(q_hat, reference_n)[:, :3]
        finite_difference = np.empty((3, 3))
        for column in range(3):
            step = np.zeros(3)
            step[column] = epsilon
            plus = body_vector_prediction(quat_multiply(q_hat, quat_exp(step)), reference_n) - prediction
            minus = body_vector_prediction(quat_multiply(q_hat, quat_exp(-step)), reference_n) - prediction
            finite_difference[:, column] = (plus - minus) / (2.0 * epsilon)
        error = finite_difference - analytic
        worst_absolute_error = max(worst_absolute_error, float(np.max(np.abs(error))))
        worst_relative_error = max(
            worst_relative_error,
            float(np.linalg.norm(error) / np.linalg.norm(analytic)),
        )

    assert worst_relative_error <= 1.0e-6
    assert worst_absolute_error <= 1.0e-9


def test_b3_locked_identity_frame_and_residual_sign() -> None:
    q = np.array([1.0, 0.0, 0.0, 0.0])
    reference = np.array([1.0, 0.0, 0.0])
    jacobian = body_vector_jacobian(q, reference)[:, :3]
    np.testing.assert_allclose(jacobian, skew(reference), atol=ATOL, rtol=0.0)
    assert jacobian[1, 2] == -1.0


def test_b4_sun_tangent_analytic_jacobian_matches_central_difference() -> None:
    rng = np.random.default_rng(RNG_SEED + 1)
    epsilon = 3.0e-7
    worst_relative_error = 0.0
    worst_absolute_error = 0.0
    for _ in range(100):
        q_hat = _random_quaternion(rng)
        reference_n = rng.normal(size=3)
        reference_n /= np.linalg.norm(reference_n)
        prediction, basis, analytic_full = sun_tangent_jacobian(q_hat, reference_n)
        analytic = analytic_full[:, :3]
        finite_difference = np.empty((2, 3))
        for column in range(3):
            step = np.zeros(3)
            step[column] = epsilon
            plus = body_vector_prediction(quat_multiply(q_hat, quat_exp(step)), reference_n) - prediction
            minus = body_vector_prediction(quat_multiply(q_hat, quat_exp(-step)), reference_n) - prediction
            finite_difference[:, column] = basis.T @ (plus - minus) / (2.0 * epsilon)
        error = finite_difference - analytic
        worst_absolute_error = max(worst_absolute_error, float(np.max(np.abs(error))))
        worst_relative_error = max(
            worst_relative_error,
            float(np.linalg.norm(error) / np.linalg.norm(analytic)),
        )
        np.testing.assert_allclose(basis.T @ basis, np.eye(2), atol=ATOL, rtol=0.0)
        np.testing.assert_allclose(basis.T @ prediction, np.zeros(2), atol=ATOL, rtol=0.0)
        assert np.linalg.matrix_rank(analytic) == 2

    assert worst_relative_error <= 1.0e-6
    assert worst_absolute_error <= 1.0e-9


def test_b5_known_attitude_and_bias_injection_removes_local_residual() -> None:
    q_minus = quat_exp([0.2, -0.3, 0.1])
    b_minus = np.array([0.01, -0.02, 0.005])
    delta_x = np.array([0.03, -0.01, 0.02, -0.002, 0.003, -0.001])
    q_true = quat_multiply(q_minus, quat_exp(delta_x[:3]))
    q_plus, b_plus = inject_error_state(q_minus, b_minus, delta_x)

    assert quat_geodesic_angle(q_plus, q_true) <= 1.0e-12
    np.testing.assert_allclose(star_tracker_residual(q_plus, q_true), np.zeros(3), atol=ATOL)
    np.testing.assert_allclose(b_plus, b_minus + delta_x[3:], atol=ATOL, rtol=0.0)


def test_b5_exact_reset_jacobian_matches_central_difference() -> None:
    correction = np.array([0.1, -0.2, 0.05])
    epsilon = 1.0e-7
    finite_difference = np.empty((3, 3))
    for column in range(3):
        step = np.zeros(3)
        step[column] = epsilon
        plus = quat_log(
            quat_multiply(quat_exp(-correction), quat_exp(correction + step))
        )
        minus = quat_log(
            quat_multiply(quat_exp(-correction), quat_exp(correction - step))
        )
        finite_difference[:, column] = (plus - minus) / (2.0 * epsilon)

    exact = right_jacobian_so3(correction)
    relative_error = np.linalg.norm(finite_difference - exact) / np.linalg.norm(exact)
    assert relative_error <= 1.0e-7


def test_b5_covariance_reset_preserves_symmetry_and_spd() -> None:
    rng = np.random.default_rng(RNG_SEED + 2)
    factor = rng.normal(size=(6, 6))
    p_c = factor @ factor.T + 0.1 * np.eye(6)
    p_plus, reset, correction = reset_covariance(p_c, [0.1, -0.2, 0.05])

    assert reset.shape == (6, 6)
    np.testing.assert_allclose(p_plus, p_plus.T, atol=1.0e-12, rtol=0.0)
    assert_positive_definite(p_plus, name="P_plus")
    assert correction <= 0.5e-12


def test_b6_star_tracker_measurement_antipodes_produce_same_update() -> None:
    state = _state(q=quat_exp([0.2, -0.1, 0.3]))
    measurement = quat_multiply(state.q_NB, quat_exp([0.01, -0.02, 0.03]))
    r = 0.01 * np.eye(3)
    positive = star_tracker_update(state, measurement, r)
    negative = star_tracker_update(state, -measurement, r)

    np.testing.assert_allclose(positive.residual, negative.residual, atol=ATOL, rtol=0.0)
    np.testing.assert_allclose(positive.delta_x, negative.delta_x, atol=ATOL, rtol=0.0)
    np.testing.assert_allclose(positive.state.P, negative.state.P, atol=ATOL, rtol=0.0)
    assert quat_geodesic_angle(positive.state.q_NB, negative.state.q_NB) <= ATOL


def test_b6_nominal_antipodes_produce_same_physical_posterior_and_covariance() -> None:
    q = quat_exp([0.2, -0.1, 0.3])
    state_positive = _state(q=q)
    state_negative = _state(q=-q)
    measurement = quat_multiply(q, quat_exp([0.01, -0.02, 0.03]))
    r = 0.01 * np.eye(3)
    positive = star_tracker_update(state_positive, measurement, r)
    negative = star_tracker_update(state_negative, measurement, r)

    np.testing.assert_allclose(positive.residual, negative.residual, atol=ATOL, rtol=0.0)
    np.testing.assert_allclose(positive.delta_x, negative.delta_x, atol=ATOL, rtol=0.0)
    np.testing.assert_allclose(positive.state.P, negative.state.P, atol=ATOL, rtol=0.0)
    np.testing.assert_allclose(
        quat_to_dcm(positive.state.q_NB),
        quat_to_dcm(negative.state.q_NB),
        atol=ATOL,
        rtol=0.0,
    )


def test_b6_exact_pi_x_antipodes_produce_identical_full_update() -> None:
    state = _state()
    q_pi = np.array([0.0, 1.0, 0.0, 0.0])
    r = np.diag([0.01, 0.02, 0.03])
    positive = star_tracker_update(state, q_pi, r)
    negative = star_tracker_update(state, -q_pi, r)

    np.testing.assert_array_equal(positive.residual, negative.residual)
    assert positive.residual.tobytes() == negative.residual.tobytes()
    np.testing.assert_array_equal(positive.delta_x, negative.delta_x)
    np.testing.assert_array_equal(positive.state.P, negative.state.P)
    assert quat_geodesic_angle(positive.state.q_NB, negative.state.q_NB) == 0.0


@pytest.mark.parametrize(
    ("axis_name", "axis"),
    (
        ("y", np.array([0.0, 1.0, 0.0])),
        ("z", np.array([0.0, 0.0, 1.0])),
        ("arbitrary", np.array([-1.0, 2.0, 3.0]) / np.sqrt(14.0)),
    ),
)
def test_b6_exact_pi_other_axes_antipodes_produce_identical_update(
    axis_name: str,
    axis: np.ndarray,
) -> None:
    del axis_name
    state = _state(q=quat_exp([0.2, -0.1, 0.3]))
    relative_pi = np.concatenate(([0.0], axis))
    measurement = quat_multiply(state.q_NB, relative_pi)
    r = np.diag([0.01, 0.02, 0.03])
    positive = star_tracker_update(state, measurement, r)
    negative = star_tracker_update(state, -measurement, r)

    np.testing.assert_array_equal(positive.residual, negative.residual)
    np.testing.assert_array_equal(positive.delta_x, negative.delta_x)
    np.testing.assert_array_equal(positive.state.P, negative.state.P)
    assert quat_geodesic_angle(positive.state.q_NB, negative.state.q_NB) == 0.0


def test_b6_exact_pi_nominal_sign_flip_preserves_physical_update() -> None:
    nominal = quat_exp([0.2, -0.1, 0.3])
    relative_pi = np.array([0.0, -1.0, 2.0, 3.0]) / np.sqrt(14.0)
    measurement = quat_multiply(nominal, relative_pi)
    positive = star_tracker_update(_state(q=nominal), measurement, 0.01 * np.eye(3))
    negative = star_tracker_update(_state(q=-nominal), measurement, 0.01 * np.eye(3))

    np.testing.assert_allclose(positive.residual, negative.residual, atol=ATOL, rtol=0.0)
    np.testing.assert_allclose(positive.delta_x, negative.delta_x, atol=ATOL, rtol=0.0)
    np.testing.assert_allclose(positive.state.P, negative.state.P, atol=ATOL, rtol=0.0)
    assert quat_geodesic_angle(positive.state.q_NB, negative.state.q_NB) <= ATOL


def test_state_initialization_defensively_copies_all_input_arrays() -> None:
    q = np.array([2.0, 0.0, 0.0, 0.0])
    b = np.array([0.01, -0.02, 0.005])
    p = np.diag([0.04, 0.03, 0.02, 0.01, 0.008, 0.006])
    state = MEKFState(q_NB=q, b_g=b, P=p)
    expected_q = state.q_NB.copy()
    expected_b = state.b_g.copy()
    expected_p = state.P.copy()

    q[:] = np.nan
    b[:] = np.nan
    p[:] = np.nan
    np.testing.assert_array_equal(state.q_NB, expected_q)
    np.testing.assert_array_equal(state.b_g, expected_b)
    np.testing.assert_array_equal(state.P, expected_p)


def test_state_q_nb_is_read_only() -> None:
    state = _state()
    assert not state.q_NB.flags.writeable
    with pytest.raises(ValueError, match="read-only"):
        state.q_NB[0] = 0.0


def test_state_b_g_is_read_only() -> None:
    state = _state()
    assert not state.b_g.flags.writeable
    with pytest.raises(ValueError, match="read-only"):
        state.b_g[0] = 1.0


def test_state_p_is_read_only() -> None:
    state = _state()
    assert not state.P.flags.writeable
    with pytest.raises(ValueError, match="read-only"):
        state.P[0, 0] = 2.0


def test_failed_propagation_and_update_leave_prior_state_unchanged() -> None:
    state = _state(q=quat_exp([0.2, -0.1, 0.3]), b=np.array([0.01, -0.02, 0.005]))
    original_q = state.q_NB.copy()
    original_b = state.b_g.copy()
    original_p = state.P.copy()

    with pytest.raises(NumericalSafetyError, match="finite"):
        propagate_state(state, [np.nan, 0.0, 0.0], 0.1, _zero_qc())
    with pytest.raises(NumericalSafetyError, match="strictly SPD"):
        star_tracker_update(state, quat_exp([0.1, 0.0, 0.0]), np.zeros((3, 3)))
    np.testing.assert_array_equal(state.q_NB, original_q)
    np.testing.assert_array_equal(state.b_g, original_b)
    np.testing.assert_array_equal(state.P, original_p)


def test_successful_propagation_and_update_return_new_states_without_mutating_prior() -> None:
    prior = _state(q=quat_exp([0.2, -0.1, 0.3]), b=np.array([0.01, -0.02, 0.005]))
    original_q = prior.q_NB.copy()
    original_b = prior.b_g.copy()
    original_p = prior.P.copy()

    propagation = propagate_state(prior, [0.2, -0.1, 0.4], 0.1, _zero_qc())
    update = star_tracker_update(prior, quat_exp([0.1, -0.05, 0.2]), 0.01 * np.eye(3))

    assert propagation.state is not prior
    assert update.state is not prior
    np.testing.assert_array_equal(prior.q_NB, original_q)
    np.testing.assert_array_equal(prior.b_g, original_b)
    np.testing.assert_array_equal(prior.P, original_p)
    assert not propagation.state.q_NB.flags.writeable
    assert not update.state.P.flags.writeable


def test_numerical_safety_joseph_update_is_symmetric_and_spd() -> None:
    state = _state()
    h = np.zeros((3, 6))
    h[:, :3] = np.eye(3)
    r = np.diag([0.01, 0.02, 0.03])
    gain = kalman_gain(state.P, h, r)
    p_c, correction = joseph_covariance_update(state.P, gain.K, h, r)

    np.testing.assert_allclose(p_c, p_c.T, atol=ATOL, rtol=0.0)
    assert_positive_definite(p_c, name="P_c")
    assert correction <= 0.5e-12


def test_numerical_safety_valid_spd_cholesky_solve_has_small_residual() -> None:
    matrix = np.array([[2.0, 0.3], [0.3, 1.0]])
    rhs = np.array([1.0, -2.0])
    solution = cholesky_solve_spd(matrix, rhs, name="S")
    np.testing.assert_allclose(matrix @ solution, rhs, atol=ATOL, rtol=0.0)


def test_numerical_safety_non_spd_p_and_s_fail_loud() -> None:
    non_spd_p = np.eye(6)
    non_spd_p[0, 0] = -1.0
    with pytest.raises(NumericalSafetyError, match="P must be strictly SPD"):
        MEKFState(q_NB=np.array([1.0, 0.0, 0.0, 0.0]), b_g=np.zeros(3), P=non_spd_p)

    non_spd_s = np.array([[1.0, 2.0], [2.0, 1.0]])
    with pytest.raises(NumericalSafetyError, match="S must be strictly SPD"):
        cholesky_solve_spd(non_spd_s, np.ones(2), name="S")


def test_numerical_safety_nonfinite_and_invalid_shapes_fail_before_state_mutation() -> None:
    state = _state()
    original_q = state.q_NB.copy()
    original_p = state.P.copy()
    with pytest.raises(NumericalSafetyError, match="finite"):
        propagate_state(state, [np.nan, 0.0, 0.0], 0.1, _zero_qc())
    with pytest.raises(NumericalSafetyError, match="finite"):
        star_tracker_residual(state.q_NB, [np.inf, 0.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="shape"):
        body_vector_prediction(state.q_NB, [1.0, 0.0])
    np.testing.assert_array_equal(state.q_NB, original_q)
    np.testing.assert_array_equal(state.P, original_p)


def test_numerical_safety_has_no_inverse_or_pseudoinverse_recovery_path() -> None:
    source = inspect.getsource(mekf)
    assert "np.linalg.inv" not in source
    assert "np.linalg.pinv" not in source
    assert "eigenvalue clipping" not in source
    assert "diagonal jitter" not in source


def test_estimator_public_api_has_no_truth_or_oracle_input() -> None:
    for function in (propagate_state, star_tracker_update, kalman_gain):
        parameter_names = set(inspect.signature(function).parameters)
        assert not any("true" in name or "oracle" in name or "event" in name for name in parameter_names)


def test_import_boundary_does_not_load_forbidden_packages() -> None:
    repository_root = Path(__file__).resolve().parents[1]
    code = """
import json
import sys
import bench.estimators.mekf
prefixes = (
    'Basilisk', 'torch', 'bench.runners', 'bench.models', 'bench.tasks',
    'bench.metrics', 'viz', 'visualization', 'bench.viz', 'bench.visualization'
)
loaded = sorted(name for name in sys.modules if any(name == p or name.startswith(p + '.') for p in prefixes))
print(json.dumps(loaded))
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(completed.stdout) == []
