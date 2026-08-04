from __future__ import annotations

import ast
import subprocess
from dataclasses import FrozenInstanceError, fields
from pathlib import Path

import numpy as np
import pytest
from scipy.linalg import cholesky, solve_triangular

from bench.estimators.mekf import MEKFState, NumericalSafetyError, quat_exp, quat_multiply
from bench.metrics.mekf import (
    attitude_geodesic_error_deg,
    attitude_geodesic_error_rad,
    bias_error_summary,
    consistency_summary,
    right_local_nees,
    right_local_state_error,
    spd_diagnostics,
    star_tracker_nis,
)
from bench.tasks.generator.basilisk_unit_st import (
    BasiliskUnitSTConfig,
    generate_basilisk_unit_st,
)
from bench.tasks.generator.mekf_events import replay_trajectory


PYTHON = "/home/dss-pc-05/.pyenv/versions/3.10.13/bin/python"
IDENTITY_Q = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
ZERO_BIAS = np.zeros(3, dtype=np.float64)


def _as_batch(value: np.ndarray) -> np.ndarray:
    return np.asarray(value, dtype=np.float64)[None, ...]


def _full_spd(dimension: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    factor = rng.normal(size=(dimension, dimension))
    return np.asarray(factor @ factor.T + np.eye(dimension) * 0.2, dtype=np.float64)


def _cholesky_quadratic_form(vector: np.ndarray, covariance: np.ndarray) -> float:
    lower = cholesky(covariance, lower=True, check_finite=True)
    intermediate = solve_triangular(lower, vector, lower=True, check_finite=True)
    solved = solve_triangular(lower.T, intermediate, lower=False, check_finite=True)
    return float(vector @ solved)


def test_c01_attitude_identity_and_quaternion_sign_invariance() -> None:
    q_hat = np.stack((IDENTITY_Q, -IDENTITY_Q))
    q_true = np.stack((IDENTITY_Q, IDENTITY_Q))
    observed = attitude_geodesic_error_rad(q_hat, q_true)
    assert np.array_equal(observed, np.zeros(2, dtype=np.float64))
    assert np.array_equal(
        observed,
        attitude_geodesic_error_rad(-q_hat, -q_true),
    )
    assert np.array_equal(
        attitude_geodesic_error_deg(q_hat, q_true),
        np.zeros(2, dtype=np.float64),
    )


@pytest.mark.parametrize(
    "rotation_vector",
    (
        np.asarray([0.3, 0.0, 0.0], dtype=np.float64),
        np.asarray([0.0, -0.4, 0.0], dtype=np.float64),
        np.asarray([0.0, 0.0, 0.7], dtype=np.float64),
        np.asarray([0.2, -0.3, 0.4], dtype=np.float64),
    ),
    ids=("x", "y", "z", "arbitrary"),
)
def test_c02_known_axis_angle_magnitudes(rotation_vector: np.ndarray) -> None:
    observed = float(attitude_geodesic_error_rad(IDENTITY_Q, quat_exp(rotation_vector)))
    assert observed == pytest.approx(np.linalg.norm(rotation_vector), rel=0.0, abs=2.0e-15)


@pytest.mark.parametrize(
    "magnitude",
    (1.0e-13, np.pi, np.pi - 1.0e-12),
    ids=("near-zero", "exact-pi", "near-pi"),
)
def test_c03_near_zero_exact_pi_and_near_pi_are_stable(magnitude: float) -> None:
    axis = np.asarray([2.0, -3.0, 4.0], dtype=np.float64)
    axis /= np.linalg.norm(axis)
    q_true = quat_exp(axis * magnitude)
    positive = attitude_geodesic_error_rad(IDENTITY_Q, q_true)
    negative = attitude_geodesic_error_rad(IDENTITY_Q, -q_true)
    assert float(positive) == pytest.approx(magnitude, rel=0.0, abs=3.0e-15)
    assert np.array_equal(positive, negative)


def test_c04_known_right_local_state_error_recovery() -> None:
    q_hat = quat_exp(np.asarray([0.4, -0.1, 0.2], dtype=np.float64))
    expected_theta = np.asarray([0.02, -0.03, 0.01], dtype=np.float64)
    q_true = quat_multiply(q_hat, quat_exp(expected_theta))
    b_hat = np.zeros(3, dtype=np.float64)
    expected_bias = np.asarray([-0.004, 0.005, 0.006], dtype=np.float64)
    result = right_local_state_error(q_hat, b_hat, q_true, b_hat + expected_bias)
    assert np.allclose(result.delta_theta_rad, expected_theta, rtol=0.0, atol=3.0e-16)
    assert np.array_equal(result.delta_bias_rad_s, expected_bias)
    assert np.allclose(
        result.state_error,
        np.concatenate((expected_theta, expected_bias)),
        rtol=0.0,
        atol=3.0e-16,
    )


def test_c05_closed_form_bias_error_and_rmse() -> None:
    scale = 0.01
    b_hat = np.zeros((2, 3), dtype=np.float64)
    b_true = scale * np.asarray([[1.0, 2.0, 3.0], [-1.0, -2.0, -3.0]], dtype=np.float64)
    result = bias_error_summary(b_hat, b_true)
    assert np.array_equal(result.per_axis_error_rad_s, b_true)
    expected_norm = scale * np.sqrt(14.0)
    assert np.allclose(result.vector_norm_rad_s, expected_norm, rtol=0.0, atol=2.0e-18)
    assert np.allclose(
        result.per_axis_rmse_rad_s,
        scale * np.asarray([1.0, 2.0, 3.0]),
        rtol=0.0,
        atol=2.0e-18,
    )
    assert result.vector_rmse_rad_s == pytest.approx(expected_norm, rel=0.0, abs=2.0e-18)


@pytest.mark.parametrize(
    "covariance",
    (
        np.diag(np.asarray([2.0, 3.0, 4.0], dtype=np.float64)),
        np.asarray([[2.0, 0.2, -0.1], [0.2, 1.5, 0.3], [-0.1, 0.3, 1.2]], dtype=np.float64),
    ),
    ids=("diagonal", "full-spd"),
)
def test_c06_closed_form_star_tracker_nis(covariance: np.ndarray) -> None:
    residual = np.asarray([0.3, -0.2, 0.1], dtype=np.float64)
    expected = _cholesky_quadratic_form(residual, covariance)
    observed = float(star_tracker_nis(residual, covariance))
    assert observed == pytest.approx(expected, rel=2.0e-15, abs=2.0e-16)


@pytest.mark.parametrize("covariance", (np.eye(6, dtype=np.float64), _full_spd(6, 701)))
def test_c07_closed_form_right_local_nees(covariance: np.ndarray) -> None:
    delta_theta = np.asarray([0.08, -0.04, 0.02], dtype=np.float64)
    delta_bias = np.asarray([0.003, -0.002, 0.001], dtype=np.float64)
    q_true = quat_exp(delta_theta)
    expected_error = np.concatenate((delta_theta, delta_bias))
    expected = _cholesky_quadratic_form(expected_error, covariance)
    observed = float(
        right_local_nees(IDENTITY_Q, ZERO_BIAS, covariance, q_true, delta_bias)
    )
    assert observed == pytest.approx(expected, rel=3.0e-15, abs=3.0e-16)


def test_c08_nees_is_invariant_to_each_quaternion_sign() -> None:
    q_hat = quat_exp(np.asarray([0.2, -0.5, 0.1], dtype=np.float64))
    q_true = quat_multiply(q_hat, quat_exp(np.asarray([0.1, 0.02, -0.03], dtype=np.float64)))
    covariance = _full_spd(6, 702)
    reference = right_local_nees(q_hat, ZERO_BIAS, covariance, q_true, ZERO_BIAS)
    for estimate_sign, truth_sign in ((-1.0, 1.0), (1.0, -1.0), (-1.0, -1.0)):
        observed = right_local_nees(
            estimate_sign * q_hat,
            ZERO_BIAS,
            covariance,
            truth_sign * q_true,
            ZERO_BIAS,
        )
        assert np.array_equal(observed, reference)


def test_c09_consistency_summary_and_chi_square_batch_bounds() -> None:
    values = np.asarray([1.0, 2.0, 3.0, 6.0], dtype=np.float64)
    result = consistency_summary(values, dof_per_sample=3, confidence_level=0.95)
    assert result.count == 4
    assert result.dof_per_sample == 3
    assert result.sum == 12.0
    assert result.mean == 3.0
    assert result.normalized_mean == 1.0
    assert result.chi_square_sum_lower < result.chi_square_sum_upper
    assert result.chi_square_sum_lower > 0.0


def test_c10_spd_diagnostics_and_negative_matrices_fail_loudly() -> None:
    covariance = np.stack((np.eye(3), np.diag([2.0, 3.0, 4.0]))).astype(np.float64)
    before = covariance.copy()
    result = spd_diagnostics(covariance, name="S")
    assert result.dimension == 3
    assert np.array_equal(result.relative_asymmetry, np.zeros(2, dtype=np.float64))
    assert np.array_equal(result.minimum_eigenvalue, np.asarray([1.0, 2.0]))
    assert np.array_equal(result.cholesky_succeeded, np.ones(2, dtype=np.bool_))
    assert np.array_equal(covariance, before)

    asymmetric = np.eye(3, dtype=np.float64)
    asymmetric[0, 1] = 0.1
    with pytest.raises(NumericalSafetyError, match="asymmetry"):
        star_tracker_nis(np.ones(3, dtype=np.float64), asymmetric)
    non_spd_s = np.diag(np.asarray([1.0, -1.0, 1.0], dtype=np.float64))
    with pytest.raises(NumericalSafetyError, match="strictly SPD"):
        star_tracker_nis(np.ones(3, dtype=np.float64), non_spd_s)
    non_spd_p = np.diag(np.asarray([1.0] * 5 + [-1.0], dtype=np.float64))
    with pytest.raises(NumericalSafetyError, match="strictly SPD"):
        right_local_nees(IDENTITY_Q, ZERO_BIAS, non_spd_p, IDENTITY_Q, ZERO_BIAS)
    nonfinite = np.eye(3, dtype=np.float64)
    nonfinite[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        spd_diagnostics(nonfinite)


@pytest.mark.parametrize(
    ("call", "error_type"),
    (
        (lambda: attitude_geodesic_error_rad(IDENTITY_Q.astype(np.float32), IDENTITY_Q), TypeError),
        (lambda: bias_error_summary(np.zeros((0, 3), dtype=np.float64), np.zeros((0, 3), dtype=np.float64)), ValueError),
        (lambda: bias_error_summary(np.zeros((2, 3), dtype=np.float64), np.zeros((3, 3), dtype=np.float64)), ValueError),
        (lambda: star_tracker_nis(np.zeros((2, 3), dtype=np.float64), np.eye(3, dtype=np.float64)), ValueError),
        (lambda: right_local_nees(IDENTITY_Q, ZERO_BIAS, np.eye(5, dtype=np.float64), IDENTITY_Q, ZERO_BIAS), ValueError),
        (lambda: consistency_summary(np.zeros(0, dtype=np.float64), dof_per_sample=3), ValueError),
        (lambda: consistency_summary(np.ones(2, dtype=np.float32), dof_per_sample=3), TypeError),
        (lambda: consistency_summary(np.ones(2, dtype=np.float64), dof_per_sample=0), ValueError),
    ),
)
def test_c11_shape_dtype_batch_and_empty_validation(call, error_type: type[Exception]) -> None:
    with pytest.raises(error_type):
        call()


def test_c12_inputs_are_not_mutated_and_array_results_are_read_only() -> None:
    q_hat = _as_batch(quat_exp(np.asarray([0.2, -0.1, 0.05], dtype=np.float64)))
    q_true = _as_batch(
        quat_multiply(q_hat[0], quat_exp(np.asarray([0.01, 0.02, -0.03], dtype=np.float64)))
    )
    b_hat = np.asarray([[0.001, 0.002, 0.003]], dtype=np.float64)
    b_true = np.asarray([[0.004, -0.001, 0.005]], dtype=np.float64)
    covariance = _as_batch(_full_spd(6, 703))
    inputs = (q_hat, q_true, b_hat, b_true, covariance)
    before = tuple(value.copy() for value in inputs)
    state = right_local_state_error(q_hat, b_hat, q_true, b_true)
    nees = right_local_nees(q_hat, b_hat, covariance, q_true, b_true)
    bias = bias_error_summary(b_hat, b_true)
    diagnostics = spd_diagnostics(covariance)
    for value, snapshot in zip(inputs, before):
        assert np.array_equal(value, snapshot)
    arrays = [
        state.delta_theta_rad,
        state.delta_bias_rad_s,
        state.state_error,
        nees,
        bias.per_axis_error_rad_s,
        bias.vector_norm_rad_s,
        bias.per_axis_rmse_rad_s,
        diagnostics.relative_asymmetry,
        diagnostics.minimum_eigenvalue,
        diagnostics.cholesky_succeeded,
    ]
    for value in arrays:
        assert value.flags.writeable is False
        with pytest.raises(ValueError, match="read-only"):
            value.flat[0] = 0
    with pytest.raises(FrozenInstanceError):
        state.state_error = np.zeros(6, dtype=np.float64)


def test_c13_import_boundary_is_metrics_and_gate_a_only() -> None:
    source_path = Path("bench/metrics/mekf.py")
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    forbidden = (
        "Basilisk",
        "torch",
        "bench.models",
        "bench.runners",
        "bench.tasks",
        "viz",
        "visualization",
    )
    assert not any(
        name == item or name.startswith(item + ".")
        for name in imported
        for item in forbidden
    )
    completed = subprocess.run(
        [
            PYTHON,
            "-c",
            (
                "import sys; import bench.metrics.mekf; "
                "print(','.join(sorted(name for name in sys.modules if "
                "name == 'Basilisk' or name.startswith('Basilisk.') or "
                "name.startswith('bench.runners') or name.startswith('bench.models') or "
                "name.startswith('bench.tasks') or name.startswith('viz'))))"
            ),
        ],
        check=True,
        capture_output=True,
        text=True,
        env={"PYTHONDONTWRITEBYTECODE": "1"},
    )
    assert completed.stdout.strip() == ""


def test_c14_source_has_no_forbidden_numerical_fallback() -> None:
    source = Path("bench/metrics/mekf.py").read_text(encoding="utf-8").lower()
    forbidden = (
        "np.linalg.inv(",
        "numpy.linalg.inv(",
        "scipy.linalg.inv(",
        "pinv(",
        "lstsq(",
        "jitter",
        "np.clip(",
        "numpy.clip(",
    )
    for token in forbidden:
        assert token not in source


@pytest.mark.parametrize("seed", (801, 802, 803))
def test_c15_small_deterministic_b2_replay_metric_smoke(seed: int) -> None:
    generated = generate_basilisk_unit_st(
        BasiliskUnitSTConfig(
            num_trajectories=3,
            duration_s=0.2,
            gyro_rate_hz=20,
            star_tracker_rate_hz=5,
            master_seed=seed,
        )
    )
    truth = generated.dataset.truth
    trajectory_id = int(truth.trajectory_id[0])
    truth_start = int(truth.truth_offsets[0])
    truth_stop = int(truth.truth_offsets[1])
    prior = MEKFState(
        q_NB=quat_multiply(
            truth.q_true_NB[truth_start],
            quat_exp(np.asarray([0.01, -0.015, 0.02], dtype=np.float64)),
        ),
        b_g=np.zeros(3, dtype=np.float64),
        P=np.eye(6, dtype=np.float64) * 1.0e-3,
    )
    process_noise = np.diag(
        np.asarray([1.0e-10] * 3 + [1.0e-12] * 3, dtype=np.float64)
    )
    replay = replay_trajectory(
        generated.dataset.events,
        trajectory_id,
        prior,
        0.0,
        process_noise,
    )
    trajectory_truth_time = truth.truth_time_s[truth_start:truth_stop]
    truth_indices = np.asarray(
        [int(np.flatnonzero(trajectory_truth_time == time)[0]) for time in replay.event_time_s],
        dtype=np.int64,
    )
    q_true = truth.q_true_NB[truth_start:truth_stop][truth_indices]
    b_true = truth.gyro_bias_rad_s[truth_start:truth_stop][truth_indices]
    trajectory_ids = np.full(replay.event_time_s.shape, trajectory_id, dtype=np.int64)

    attitude = attitude_geodesic_error_rad(replay.q_NB_history, q_true)
    bias = bias_error_summary(replay.b_g_history, b_true)
    nees = right_local_nees(
        replay.q_NB_history,
        replay.b_g_history,
        replay.P_history,
        q_true,
        b_true,
        estimate_time_s=replay.event_time_s,
        covariance_time_s=replay.event_time_s,
        truth_time_s=replay.event_time_s,
        estimate_trajectory_id=trajectory_ids,
        covariance_trajectory_id=trajectory_ids,
        truth_trajectory_id=trajectory_ids,
    )
    order_to_row = {int(order): index for index, order in enumerate(replay.event_order)}
    star_rows = np.asarray(
        [order_to_row[int(order)] for order in replay.star_tracker_event_order],
        dtype=np.int64,
    )
    star_times = replay.event_time_s[star_rows]
    star_ids = trajectory_ids[star_rows]
    nis = star_tracker_nis(
        replay.star_tracker_residual,
        replay.star_tracker_S,
        residual_time_s=star_times,
        covariance_time_s=star_times,
        residual_trajectory_id=star_ids,
        covariance_trajectory_id=star_ids,
    )

    assert np.all(np.isfinite(attitude)) and np.all(attitude >= 0.0)
    assert np.all(np.isfinite(bias.per_axis_error_rad_s))
    assert np.all(np.isfinite(nees)) and np.all(nees >= 0.0)
    assert np.all(np.isfinite(nis)) and np.all(nis >= 0.0)
    assert np.all(spd_diagnostics(replay.P_history, name="P").cholesky_succeeded)
    assert np.all(spd_diagnostics(replay.star_tracker_S, name="S").cholesky_succeeded)
    assert np.array_equal(attitude, attitude_geodesic_error_rad(replay.q_NB_history, -q_true))
    assert np.array_equal(
        nees,
        right_local_nees(
            -replay.q_NB_history,
            replay.b_g_history,
            replay.P_history,
            q_true,
            b_true,
        ),
    )


def test_c16_timestamp_posterior_and_trajectory_pairing_mismatch_fails_loudly() -> None:
    residual = np.zeros((2, 3), dtype=np.float64)
    covariance_s = np.repeat(np.eye(3, dtype=np.float64)[None, :, :], 2, axis=0)
    times = np.asarray([1.0, 2.0], dtype=np.float64)
    wrong_times = np.asarray([1.0, 2.1], dtype=np.float64)
    ids = np.asarray([7, 7], dtype=np.int64)
    wrong_ids = np.asarray([7, 8], dtype=np.int64)
    with pytest.raises(ValueError, match="match exactly"):
        star_tracker_nis(
            residual,
            covariance_s,
            residual_time_s=times,
            covariance_time_s=wrong_times,
        )
    with pytest.raises(ValueError, match="match exactly"):
        star_tracker_nis(
            residual,
            covariance_s,
            residual_trajectory_id=ids,
            covariance_trajectory_id=wrong_ids,
        )

    q = np.repeat(IDENTITY_Q[None, :], 2, axis=0)
    bias = np.zeros((2, 3), dtype=np.float64)
    covariance_p = np.repeat(np.eye(6, dtype=np.float64)[None, :, :], 2, axis=0)
    with pytest.raises(ValueError, match="match exactly"):
        right_local_nees(
            q,
            bias,
            covariance_p,
            q,
            bias,
            estimate_time_s=times,
            covariance_time_s=wrong_times,
            truth_time_s=times,
        )
    with pytest.raises(ValueError, match="supplied together"):
        right_local_nees(
            q,
            bias,
            covariance_p,
            q,
            bias,
            estimate_time_s=times,
        )


@pytest.mark.parametrize("seed", range(10))
def test_metric_property_sweep_against_linear_solve(seed: int) -> None:
    rng = np.random.default_rng(9000 + seed)
    q_hat = quat_exp(rng.normal(size=3).astype(np.float64))
    delta_theta = rng.normal(scale=0.1, size=3).astype(np.float64)
    q_true = quat_multiply(q_hat, quat_exp(delta_theta))
    b_hat = rng.normal(scale=0.01, size=3).astype(np.float64)
    delta_bias = rng.normal(scale=0.002, size=3).astype(np.float64)
    b_true = b_hat + delta_bias
    posterior = _full_spd(6, 9100 + seed)
    residual = rng.normal(scale=0.01, size=3).astype(np.float64)
    innovation_covariance = _full_spd(3, 9200 + seed)
    input_snapshots = tuple(
        value.copy()
        for value in (q_hat, q_true, b_hat, b_true, posterior, residual, innovation_covariance)
    )

    state = right_local_state_error(q_hat, b_hat, q_true, b_true)
    observed_nees = float(right_local_nees(q_hat, b_hat, posterior, q_true, b_true))
    observed_nis = float(star_tracker_nis(residual, innovation_covariance))
    expected_error = np.concatenate((delta_theta, delta_bias))
    expected_nees = _cholesky_quadratic_form(expected_error, posterior)
    expected_nis = _cholesky_quadratic_form(residual, innovation_covariance)

    assert np.allclose(state.delta_theta_rad, delta_theta, rtol=0.0, atol=8.0e-16)
    assert np.allclose(state.delta_bias_rad_s, delta_bias, rtol=0.0, atol=2.0e-18)
    assert observed_nees == pytest.approx(expected_nees, rel=8.0e-15, abs=5.0e-16)
    assert observed_nis == pytest.approx(expected_nis, rel=8.0e-15, abs=5.0e-16)
    assert np.array_equal(
        right_local_nees(-q_hat, b_hat, posterior, q_true, b_true),
        right_local_nees(q_hat, b_hat, posterior, q_true, b_true),
    )
    for value, snapshot in zip(
        (q_hat, q_true, b_hat, b_true, posterior, residual, innovation_covariance),
        input_snapshots,
    ):
        assert np.array_equal(value, snapshot)


def test_public_result_dataclasses_have_only_locked_fields() -> None:
    state = right_local_state_error(IDENTITY_Q, ZERO_BIAS, IDENTITY_Q, ZERO_BIAS)
    assert tuple(field.name for field in fields(state)) == (
        "delta_theta_rad",
        "delta_bias_rad_s",
        "state_error",
    )
