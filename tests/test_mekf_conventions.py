from __future__ import annotations

import numpy as np

from bench.estimators.mekf import (
    MEKFState,
    align_quaternion,
    body_vector_jacobian,
    body_vector_prediction,
    covariance_diagnostics,
    dcm_to_quat,
    inject_error_state,
    joseph_covariance_update,
    kalman_gain,
    quat_conjugate,
    quat_exp,
    quat_geodesic_angle,
    quat_inverse,
    quat_log,
    quat_multiply,
    quat_normalize,
    quat_to_dcm,
    right_jacobian_so3,
    skew,
    star_tracker_residual,
    sun_tangent_basis,
    sun_tangent_jacobian,
)


ATOL = 1.0e-12
SQRT_HALF = np.sqrt(0.5)


def test_tv_q01_identity_quaternion_and_dcm() -> None:
    q = np.array([1.0, 0.0, 0.0, 0.0])
    vector_b = np.array([1.0, 2.0, 3.0])
    rotation_nb = quat_to_dcm(q)

    np.testing.assert_allclose(rotation_nb, np.eye(3), atol=ATOL, rtol=0.0)
    np.testing.assert_allclose(rotation_nb @ vector_b, vector_b, atol=ATOL, rtol=0.0)
    np.testing.assert_allclose(rotation_nb.T @ vector_b, vector_b, atol=ATOL, rtol=0.0)


def test_tv_q02_x90_basis_vector_mapping() -> None:
    q_x90 = np.array([SQRT_HALF, SQRT_HALF, 0.0, 0.0])
    expected = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
    rotation_nb = quat_to_dcm(q_x90)

    np.testing.assert_allclose(rotation_nb, expected, atol=ATOL, rtol=0.0)
    np.testing.assert_allclose(rotation_nb @ [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], atol=ATOL)
    np.testing.assert_allclose(rotation_nb @ [0.0, 0.0, 1.0], [0.0, -1.0, 0.0], atol=ATOL)
    np.testing.assert_allclose(rotation_nb.T @ [0.0, 1.0, 0.0], [0.0, 0.0, -1.0], atol=ATOL)


def test_plus_90_about_each_axis_and_dcm_round_trip() -> None:
    expected_images = (
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 0.0, -1.0]),
        np.array([0.0, 1.0, 0.0]),
    )
    for axis, expected_image in zip(np.eye(3), expected_images):
        q = quat_exp(0.5 * np.pi * axis)
        rotation = quat_to_dcm(q)
        recovered = dcm_to_quat(rotation)
        np.testing.assert_allclose(rotation @ [1.0, 0.0, 0.0], expected_image, atol=ATOL)
        np.testing.assert_allclose(quat_to_dcm(recovered), rotation, atol=ATOL, rtol=0.0)


def test_tv_q03_z90_basis_vector_mapping() -> None:
    q_z90 = np.array([SQRT_HALF, 0.0, 0.0, SQRT_HALF])
    expected = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    rotation_nb = quat_to_dcm(q_z90)

    np.testing.assert_allclose(rotation_nb, expected, atol=ATOL, rtol=0.0)
    np.testing.assert_allclose(rotation_nb @ [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], atol=ATOL)
    np.testing.assert_allclose(rotation_nb.T @ [1.0, 0.0, 0.0], [0.0, -1.0, 0.0], atol=ATOL)


def test_tv_q04_hamilton_product_and_composition_order() -> None:
    q_x90 = np.array([SQRT_HALF, SQRT_HALF, 0.0, 0.0])
    q_y90 = np.array([SQRT_HALF, 0.0, SQRT_HALF, 0.0])
    product = quat_multiply(q_x90, q_y90)

    np.testing.assert_allclose(product, [0.5, 0.5, 0.5, 0.5], atol=ATOL, rtol=0.0)
    np.testing.assert_allclose(
        quat_to_dcm(product),
        quat_to_dcm(q_x90) @ quat_to_dcm(q_y90),
        atol=ATOL,
        rtol=0.0,
    )


def test_tv_q05_constant_rate_exponential() -> None:
    expected = np.array([0.9987502603949663, 0.0, 0.0, 0.04997916927067833])
    actual = quat_exp([0.0, 0.0, 0.1])
    np.testing.assert_allclose(actual, expected, atol=ATOL, rtol=0.0)


def test_tv_q06_bias_cancellation_increment() -> None:
    bias = np.array([0.01, -0.02, 0.005])
    dt = 0.1
    np.testing.assert_allclose(quat_exp((bias - bias) * dt), [1.0, 0.0, 0.0, 0.0], atol=ATOL)
    assert quat_geodesic_angle(quat_exp(bias * dt), np.array([1.0, 0.0, 0.0, 0.0])) > 0.0


def test_tv_m01_body_vector_prediction_and_jacobian_sign() -> None:
    q_identity = np.array([1.0, 0.0, 0.0, 0.0])
    reference_n = np.array([1.0, 0.0, 0.0])
    expected = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
    jacobian = body_vector_jacobian(q_identity, reference_n)
    epsilon = 1.0e-7
    q_true = quat_multiply(q_identity, quat_exp([0.0, 0.0, epsilon]))
    residual = body_vector_prediction(q_true, reference_n) - body_vector_prediction(q_identity, reference_n)

    np.testing.assert_allclose(jacobian[:, :3], expected, atol=ATOL, rtol=0.0)
    np.testing.assert_allclose(residual / epsilon, [0.0, -1.0, 0.0], atol=1.0e-7, rtol=0.0)


def test_tv_s01_sun_tangent_basis_and_jacobian() -> None:
    q_identity = np.array([1.0, 0.0, 0.0, 0.0])
    prediction, basis, jacobian = sun_tangent_jacobian(q_identity, [1.0, 0.0, 0.0])

    np.testing.assert_allclose(prediction, [1.0, 0.0, 0.0], atol=ATOL)
    np.testing.assert_allclose(basis, np.column_stack(([0.0, 1.0, 0.0], [0.0, 0.0, 1.0])), atol=ATOL)
    np.testing.assert_allclose(
        jacobian[:, :3],
        [[0.0, 0.0, -1.0], [0.0, 1.0, 0.0]],
        atol=ATOL,
        rtol=0.0,
    )


def test_tv_st01_star_tracker_log_residual() -> None:
    delta_theta = np.array([0.01, -0.02, 0.03])
    expected_q = np.array(
        [
            0.9998250051041071,
            0.0049997083384377,
            -0.0099994166768754,
            0.0149991250153131,
        ]
    )
    measurement = quat_exp(delta_theta)

    np.testing.assert_allclose(measurement, expected_q, atol=ATOL, rtol=0.0)
    np.testing.assert_allclose(
        star_tracker_residual([1.0, 0.0, 0.0, 0.0], measurement),
        delta_theta,
        atol=ATOL,
        rtol=0.0,
    )


def test_tv_inj01_right_injection_order() -> None:
    q_z90 = np.array([SQRT_HALF, 0.0, 0.0, SQRT_HALF])
    expected = np.array(
        [
            0.7062230818371108,
            0.03534060950936696,
            0.03534060950936696,
            0.7062230818371108,
        ]
    )
    q_plus, _ = inject_error_state(q_z90, np.zeros(3), [0.1, 0.0, 0.0, 0.0, 0.0, 0.0])

    np.testing.assert_allclose(q_plus, expected, atol=ATOL, rtol=0.0)
    np.testing.assert_allclose(star_tracker_residual(q_plus, expected), np.zeros(3), atol=ATOL)


def test_tv_rst01_exact_and_first_order_right_reset_jacobian() -> None:
    correction = np.array([0.1, -0.2, 0.05])
    expected_first_order = np.array(
        [[1.0, 0.025, 0.1], [-0.025, 1.0, 0.05], [-0.1, -0.05, 1.0]]
    )
    expected_exact_rounded = np.array(
        [
            [0.99293524, 0.02156622, 0.10039441],
            [-0.02821541, 0.99792213, 0.04811934],
            [-0.09873212, -0.05144393, 0.99168851],
        ]
    )
    exact = right_jacobian_so3(correction)

    np.testing.assert_allclose(np.eye(3) - 0.5 * skew(correction), expected_first_order, atol=ATOL)
    np.testing.assert_allclose(exact, expected_exact_rounded, atol=5.0e-9, rtol=0.0)


def test_tv_sign01_antipodal_rotation_residual_and_distance() -> None:
    q = quat_exp([0.4, -0.2, 0.3])

    np.testing.assert_allclose(quat_to_dcm(q), quat_to_dcm(-q), atol=ATOL, rtol=0.0)
    np.testing.assert_allclose(align_quaternion(-q, q), q, atol=ATOL, rtol=0.0)
    np.testing.assert_allclose(star_tracker_residual(q, -q), np.zeros(3), atol=ATOL)
    assert quat_geodesic_angle(q, -q) == 0.0


def test_tv_kf01_simple_star_tracker_gain_and_joseph_covariance() -> None:
    p_minus = np.diag([0.04, 0.04, 0.04, 0.01, 0.01, 0.01])
    h = np.zeros((3, 6))
    h[:, :3] = np.eye(3)
    r = 0.01 * np.eye(3)
    gain = kalman_gain(p_minus, h, r)
    p_plus, _ = joseph_covariance_update(p_minus, gain.K, h, r)

    np.testing.assert_allclose(gain.S, 0.05 * np.eye(3), atol=ATOL, rtol=0.0)
    np.testing.assert_allclose(gain.K[:3], 0.8 * np.eye(3), atol=ATOL, rtol=0.0)
    np.testing.assert_allclose(gain.K[3:], np.zeros((3, 3)), atol=ATOL, rtol=0.0)
    np.testing.assert_allclose(
        p_plus,
        np.diag([0.008, 0.008, 0.008, 0.01, 0.01, 0.01]),
        atol=ATOL,
        rtol=0.0,
    )


def test_tv_cov01_symmetry_and_spd_diagnostics() -> None:
    diagnostics = covariance_diagnostics(np.diag([1.0, 2.0, 3.0]), require_spd=True)
    assert diagnostics.relative_asymmetry == 0.0
    assert diagnostics.minimum_eigenvalue == 1.0
    assert diagnostics.cholesky_succeeded


def test_tv_long01_long_horizon_quaternion_norm_and_composition() -> None:
    increment = quat_exp([0.0, 0.0, 0.1])
    q = np.array([1.0, 0.0, 0.0, 0.0])
    max_norm_error = 0.0
    for _ in range(1000):
        q = quat_normalize(quat_multiply(q, increment))
        max_norm_error = max(max_norm_error, abs(float(np.linalg.norm(q)) - 1.0))

    assert max_norm_error <= 1.0e-12
    assert abs(float(q @ quat_exp([0.0, 0.0, 100.0]))) >= 1.0 - 1.0e-12


def test_quaternion_inverse_conjugate_and_normalization() -> None:
    q = quat_normalize([2.0, -1.0, 0.5, 0.25])
    identity = quat_multiply(q, quat_inverse(q))
    np.testing.assert_allclose(quat_inverse(q), quat_conjugate(q), atol=ATOL, rtol=0.0)
    np.testing.assert_allclose(identity, [1.0, 0.0, 0.0, 0.0], atol=ATOL, rtol=0.0)
    assert abs(float(np.linalg.norm(q)) - 1.0) <= ATOL


def test_exp_log_round_trip_near_zero_and_near_pi() -> None:
    cases = (
        np.array([1.0e-14, -2.0e-14, 3.0e-14]),
        np.array([0.2, -0.1, 0.4]),
        (np.pi - 1.0e-10) * np.array([1.0, 2.0, -1.0]) / np.sqrt(6.0),
    )
    for rotation_vector in cases:
        np.testing.assert_allclose(
            quat_log(quat_exp(rotation_vector)),
            rotation_vector,
            atol=5.0e-13,
            rtol=5.0e-13,
        )


def test_exact_pi_quat_log_antipodes_use_deterministic_axis_tie_break() -> None:
    axes = (
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
        np.array([-1.0, 2.0, 3.0]) / np.sqrt(14.0),
    )
    for axis in axes:
        q_pi = np.concatenate(([0.0], axis))
        expected_axis = axis.copy()
        first_significant = np.flatnonzero(np.abs(expected_axis) > 8.0 * np.finfo(np.float64).eps)[0]
        if expected_axis[first_significant] < 0.0:
            expected_axis = -expected_axis

        positive = quat_log(q_pi)
        negative = quat_log(-q_pi)
        np.testing.assert_array_equal(positive, negative)
        assert positive.tobytes() == negative.tobytes()
        np.testing.assert_allclose(positive, np.pi * expected_axis, atol=ATOL, rtol=0.0)
        assert quat_geodesic_angle(q_pi, -q_pi) == 0.0


def test_near_pi_outside_tie_preserves_shortest_arc_on_both_sides() -> None:
    axis = np.array([1.0, -2.0, 3.0]) / np.sqrt(14.0)
    offset = 1.0e-10
    below_pi = quat_exp((np.pi - offset) * axis)
    above_pi = quat_exp((np.pi + offset) * axis)

    assert abs(float(below_pi[0])) > 8.0 * np.finfo(np.float64).eps
    assert abs(float(above_pi[0])) > 8.0 * np.finfo(np.float64).eps
    np.testing.assert_allclose(
        quat_log(below_pi),
        (np.pi - offset) * axis,
        atol=5.0e-13,
        rtol=5.0e-13,
    )
    np.testing.assert_allclose(quat_log(-below_pi), quat_log(below_pi), atol=ATOL, rtol=0.0)
    np.testing.assert_allclose(
        quat_log(above_pi),
        -(np.pi - offset) * axis,
        atol=5.0e-13,
        rtol=5.0e-13,
    )
    np.testing.assert_allclose(quat_log(-above_pi), quat_log(above_pi), atol=ATOL, rtol=0.0)


def test_sun_tangent_basis_is_orthonormal_and_right_handed() -> None:
    h = np.array([1.0, 2.0, 3.0])
    h /= np.linalg.norm(h)
    basis = sun_tangent_basis(h)

    np.testing.assert_allclose(basis.T @ basis, np.eye(2), atol=ATOL, rtol=0.0)
    np.testing.assert_allclose(basis.T @ h, np.zeros(2), atol=ATOL, rtol=0.0)
    np.testing.assert_allclose(np.cross(h, basis[:, 0]), basis[:, 1], atol=ATOL, rtol=0.0)


def test_state_initialization_normalizes_quaternion_only_at_boundary() -> None:
    state = MEKFState(q_NB=np.array([2.0, 0.0, 0.0, 0.0]), b_g=np.zeros(3), P=np.eye(6))
    np.testing.assert_allclose(state.q_NB, [1.0, 0.0, 0.0, 0.0], atol=ATOL)
