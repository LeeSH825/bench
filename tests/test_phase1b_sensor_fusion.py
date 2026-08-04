from __future__ import annotations

import numpy as np
import pytest

from bench.estimators.mekf import (
    body_vector_jacobian,
    body_vector_prediction,
    quat_exp,
    quat_multiply,
    sun_tangent_jacobian,
)
from bench.tasks.generator.phase1b_sensor_fusion import (
    FusionScenarioCode,
    SensorFusionConfig,
    fusion_gyro_st_as_phase1a,
    generate_sensor_fusion,
    magnetic_reference_N,
    sun_is_valid,
    sun_reference_N,
)


def _config(scenario: FusionScenarioCode, seed: int = 27201, duration: float = 5.0):
    return SensorFusionConfig(
        num_trajectories=5,
        duration_s=duration,
        master_seed=seed,
        scenario_code=int(scenario),
        bias_random_walk_enabled=scenario != FusionScenarioCode.UNIT_ST_REDUCTION,
    )


def test_reference_profiles_are_unit_and_geometry_guarded() -> None:
    for time_s in np.linspace(0.0, 30.0, 101):
        mag = magnetic_reference_N(float(time_s), 30.0)
        sun = sun_reference_N(float(time_s), 30.0)
        assert np.linalg.norm(mag) == pytest.approx(1.0, abs=2e-15)
        assert np.linalg.norm(sun) == pytest.approx(1.0, abs=2e-15)
        angle = np.degrees(np.arccos(np.clip(mag @ sun, -1.0, 1.0)))
        assert 20.0 <= angle <= 160.0


def test_magnetometer_jacobian_matches_right_error_finite_difference() -> None:
    q = quat_exp(np.array([0.3, -0.2, 0.1], dtype=np.float64))
    reference = magnetic_reference_N(2.0, 10.0)
    analytic = body_vector_jacobian(q, reference)[:, :3]
    epsilon = 1e-6
    finite = np.empty((3, 3), dtype=np.float64)
    h0 = body_vector_prediction(q, reference)
    for axis in range(3):
        delta = np.zeros(3, dtype=np.float64)
        delta[axis] = epsilon
        plus = body_vector_prediction(quat_multiply(q, quat_exp(delta)), reference) - h0
        minus = body_vector_prediction(quat_multiply(q, quat_exp(-delta)), reference) - h0
        finite[:, axis] = (plus - minus) / (2.0 * epsilon)
    assert np.allclose(analytic, finite, rtol=1e-6, atol=1e-9)


def test_sun_tangent_jacobian_matches_right_error_finite_difference() -> None:
    q = quat_exp(np.array([-0.1, 0.25, 0.2], dtype=np.float64))
    reference = sun_reference_N(1.0, 10.0)
    prediction, basis, analytic = sun_tangent_jacobian(q, reference)
    epsilon = 1e-6
    finite = np.empty((2, 3), dtype=np.float64)
    for axis in range(3):
        delta = np.zeros(3, dtype=np.float64)
        delta[axis] = epsilon
        plus = body_vector_prediction(quat_multiply(q, quat_exp(delta)), reference) - prediction
        minus = body_vector_prediction(quat_multiply(q, quat_exp(-delta)), reference) - prediction
        finite[:, axis] = basis.T @ (plus - minus) / (2.0 * epsilon)
    assert np.allclose(analytic[:, :3], finite, rtol=1e-6, atol=1e-9)


def test_sun_validity_schedule_has_valid_and_invalid_epochs() -> None:
    validity = [sun_is_valid(float(t), 30.0, 0) for t in np.linspace(0.5, 30.0, 60)]
    assert any(validity) and not all(validity)


def test_generator_is_deterministic() -> None:
    left = generate_sensor_fusion(_config(FusionScenarioCode.MAIN_FUSION_STATIONARY))
    right = generate_sensor_fusion(_config(FusionScenarioCode.MAIN_FUSION_STATIONARY))
    assert left.semantic_hashes == right.semantic_hashes
    assert left.oracle_context.semantic_hash == right.oracle_context.semantic_hash


def test_stress_mag_contains_only_gyro_and_magnetometer() -> None:
    generated = generate_sensor_fusion(_config(FusionScenarioCode.STRESS_MAG))
    assert set(map(int, np.unique(generated.dataset.events.sensor_code))) == {1, 3}


def test_unit_st_reduction_is_exact() -> None:
    generated = generate_sensor_fusion(_config(FusionScenarioCode.UNIT_ST_REDUCTION))
    converted = fusion_gyro_st_as_phase1a(generated.dataset.events)
    base = generated.base_unit_st.dataset.events
    for name in (
        "trajectory_id",
        "sensor_code",
        "measurement_time_s",
        "arrival_time_s",
        "event_order",
        "valid",
        "payload_index",
        "gyro_omega_rad_s",
        "star_tracker_q_NB",
        "star_tracker_R_rad2",
    ):
        assert np.array_equal(getattr(converted, name), getattr(base, name))


def test_main_stress_and_c4_preserve_intended_stream_pairing() -> None:
    main = generate_sensor_fusion(_config(FusionScenarioCode.MAIN_FUSION_STATIONARY, seed=27211))
    stress = generate_sensor_fusion(
        _config(FusionScenarioCode.STRESS_MAG, seed=27211), base_unit_st=main.base_unit_st
    )
    c4 = generate_sensor_fusion(
        _config(FusionScenarioCode.C4_COMBINED, seed=27211), base_unit_st=main.base_unit_st
    )
    assert np.array_equal(main.dataset.truth.q_true_NB, stress.dataset.truth.q_true_NB)
    assert np.array_equal(main.dataset.truth.q_true_NB, c4.dataset.truth.q_true_NB)
    assert np.array_equal(main.dataset.truth.omega_true_B_rad_s, c4.dataset.truth.omega_true_B_rad_s)
    assert np.array_equal(main.dataset.events.sun_z_sun_B, c4.dataset.events.sun_z_sun_B)
    assert np.array_equal(
        main.dataset.events.star_tracker_q_ST_NB, c4.dataset.events.star_tracker_q_ST_NB
    )
    assert not np.array_equal(
        main.dataset.events.gyro_omega_m_B_rad_s, c4.dataset.events.gyro_omega_m_B_rad_s
    )
    fast = c4.oracle_context.fast_window
    mag_rows = c4.oracle_context.scenario_code == int(FusionScenarioCode.C4_COMBINED)
    assert np.any(fast & mag_rows)


def test_sensor_noise_is_finite_inlier_and_covariance_is_full_rank() -> None:
    generated = generate_sensor_fusion(_config(FusionScenarioCode.MAIN_FUSION_STATIONARY, duration=10.0))
    events = generated.dataset.events
    assert np.all(np.isfinite(events.magnetometer_z_mag_B))
    assert np.all(np.linalg.eigvalsh(events.magnetometer_R_mag) > 0.0)
    assert np.all(np.linalg.eigvalsh(events.sun_R_sun_tangent_rad2) > 0.0)
    assert np.allclose(np.linalg.norm(events.sun_z_sun_B, axis=1), 1.0, atol=2e-13)
