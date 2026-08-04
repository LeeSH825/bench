from __future__ import annotations

import inspect
from dataclasses import fields, replace

import numpy as np
import pytest

from bench.estimators.mekf import MEKFState, quat_geodesic_angle
from bench.tasks.generator.mekf_events import (
    MEKFEventTable,
    load_event_dataset,
    replay_trajectory,
    save_event_dataset,
)
from bench.tasks.generator.unit_st_synthetic import UnitSTSyntheticConfig, generate_unit_st


def _initial_state(generated, trajectory_index: int = 0, *, exact_bias: bool = False) -> MEKFState:
    truth = generated.dataset.truth
    start = int(truth.truth_offsets[trajectory_index])
    bias = truth.gyro_bias_rad_s[start] if exact_bias else np.zeros(3, dtype=np.float64)
    return MEKFState(
        q_NB=truth.q_true_NB[start],
        b_g=bias,
        P=np.eye(6, dtype=np.float64) * 1.0e-5,
    )


def _process_noise() -> np.ndarray:
    return np.diag(np.array([1.0e-10] * 3 + [1.0e-12] * 3, dtype=np.float64))


def _event_kwargs(table: MEKFEventTable) -> dict[str, np.ndarray]:
    return {field.name: np.array(getattr(table, field.name), copy=True) for field in fields(table)}


def _assert_replay_arrays_equal(left, right) -> None:
    for name in (
        "event_time_s",
        "event_order",
        "sensor_code",
        "q_NB_history",
        "b_g_history",
        "P_history",
        "attitude_step_rad",
        "star_tracker_event_order",
        "star_tracker_residual",
        "star_tracker_S",
    ):
        assert np.array_equal(getattr(left, name), getattr(right, name))


def test_zero_noise_exact_initial_state_tracks_analytic_truth() -> None:
    config = UnitSTSyntheticConfig(
        num_trajectories=4,
        duration_s=1.0,
        gyro_noise_std_rad_s=0.0,
        star_tracker_noise_std_rad=0.0,
        randomize_star_tracker_sign=True,
        master_seed=110,
    )
    generated = generate_unit_st(config)
    trajectory_id = int(generated.dataset.truth.trajectory_id[0])
    result = replay_trajectory(
        generated.dataset.events,
        trajectory_id,
        _initial_state(generated, exact_bias=True),
        0.0,
        _process_noise(),
    )
    stop = int(generated.dataset.truth.truth_offsets[1]) - 1
    expected = generated.dataset.truth.q_true_NB[stop]
    assert quat_geodesic_angle(result.final_state.q_NB, expected) < 2.0e-12
    assert np.linalg.norm(result.final_state.b_g - generated.dataset.truth.gyro_bias_rad_s[stop]) < 2.0e-12


def test_low_rate_star_tracker_smoke_is_finite_spd_and_bounded() -> None:
    generated = generate_unit_st(
        UnitSTSyntheticConfig(
            num_trajectories=4,
            duration_s=4.0,
            gyro_rate_hz=20,
            star_tracker_rate_hz=2,
            master_seed=120,
        )
    )
    trajectory_id = int(generated.dataset.truth.trajectory_id[0])
    result = replay_trajectory(
        generated.dataset.events,
        trajectory_id,
        _initial_state(generated),
        0.0,
        _process_noise(),
    )
    assert np.all(np.isfinite(result.q_NB_history))
    assert np.all(np.isfinite(result.b_g_history))
    assert np.all(np.isfinite(result.P_history))
    assert np.allclose(np.linalg.norm(result.q_NB_history, axis=1), 1.0, atol=2.0e-14)
    assert all(np.all(np.linalg.eigvalsh(matrix) > 0.0) for matrix in result.P_history)
    assert np.max(result.attitude_step_rad) <= np.pi


def test_same_stream_replays_identically() -> None:
    generated = generate_unit_st(UnitSTSyntheticConfig(num_trajectories=4, duration_s=0.8))
    trajectory_id = int(generated.dataset.truth.trajectory_id[0])
    prior = _initial_state(generated)
    first = replay_trajectory(generated.dataset.events, trajectory_id, prior, 0.0, _process_noise())
    second = replay_trajectory(generated.dataset.events, trajectory_id, prior, 0.0, _process_noise())
    _assert_replay_arrays_equal(first, second)


def test_serialization_round_trip_replay_equivalence(tmp_path) -> None:
    generated = generate_unit_st(UnitSTSyntheticConfig(num_trajectories=4, duration_s=0.8))
    path = tmp_path / "artifact"
    save_event_dataset(path, generated.dataset, generated.manifest)
    loaded, _, _ = load_event_dataset(path)
    trajectory_id = int(generated.dataset.truth.trajectory_id[0])
    prior = _initial_state(generated)
    before = replay_trajectory(generated.dataset.events, trajectory_id, prior, 0.0, _process_noise())
    after = replay_trajectory(loaded.events, trajectory_id, prior, 0.0, _process_noise())
    _assert_replay_arrays_equal(before, after)


def test_star_tracker_q_and_negative_q_have_identical_posteriors() -> None:
    generated = generate_unit_st(UnitSTSyntheticConfig(num_trajectories=4, duration_s=1.0))
    kwargs = _event_kwargs(generated.dataset.events)
    kwargs["star_tracker_q_NB"] *= -1.0
    negated = MEKFEventTable(**kwargs)
    trajectory_id = int(generated.dataset.truth.trajectory_id[0])
    prior = _initial_state(generated)
    positive = replay_trajectory(
        generated.dataset.events, trajectory_id, prior, 0.0, _process_noise()
    )
    negative = replay_trajectory(negated, trajectory_id, prior, 0.0, _process_noise())
    assert np.array_equal(positive.q_NB_history, negative.q_NB_history)
    assert np.array_equal(positive.b_g_history, negative.b_g_history)
    assert np.array_equal(positive.P_history, negative.P_history)
    assert np.array_equal(positive.star_tracker_residual, negative.star_tracker_residual)
    assert np.array_equal(positive.star_tracker_S, negative.star_tracker_S)


def test_long_sequence_quaternion_norm_and_covariance_spd() -> None:
    generated = generate_unit_st(
        UnitSTSyntheticConfig(num_trajectories=4, duration_s=10.0, master_seed=400)
    )
    trajectory_id = int(generated.dataset.truth.trajectory_id[-1])
    result = replay_trajectory(
        generated.dataset.events,
        trajectory_id,
        _initial_state(generated, trajectory_index=3),
        0.0,
        _process_noise(),
    )
    assert np.allclose(np.linalg.norm(result.q_NB_history, axis=1), 1.0, atol=2.0e-14)
    for covariance in result.P_history:
        assert np.array_equal(covariance, covariance.T)
        np.linalg.cholesky(covariance)


def test_malformed_order_and_unaligned_star_time_fail_loudly() -> None:
    generated = generate_unit_st(UnitSTSyntheticConfig(num_trajectories=4, duration_s=0.8))
    kwargs = _event_kwargs(generated.dataset.events)
    first_id = generated.dataset.truth.trajectory_id[0]
    same_time = np.flatnonzero(
        (kwargs["trajectory_id"] == first_id) & (kwargs["measurement_time_s"] == 0.2)
    )
    assert same_time.size == 2
    kwargs["event_order"][same_time] = kwargs["event_order"][same_time[::-1]]
    with pytest.raises(ValueError, match="sorted"):
        MEKFEventTable(**kwargs)

    kwargs = _event_kwargs(generated.dataset.events)
    star_row = int(np.flatnonzero(kwargs["sensor_code"] == 2)[0])
    kwargs["measurement_time_s"][star_row] += 0.01
    kwargs["arrival_time_s"][star_row] += 0.01
    unaligned = MEKFEventTable(**kwargs)
    trajectory_id = int(generated.dataset.truth.trajectory_id[0])
    with pytest.raises(ValueError, match="current propagated time"):
        replay_trajectory(
            unaligned, trajectory_id, _initial_state(generated), 0.0, _process_noise()
        )


def test_replay_public_api_exposes_no_truth_or_oracle_inputs() -> None:
    parameters = set(inspect.signature(replay_trajectory).parameters)
    assert parameters == {
        "event_table",
        "trajectory_id",
        "initial_state",
        "initial_time_s",
        "Q_c",
    }
    assert not parameters & {"truth", "oracle", "label", "future"}
    source = inspect.getsource(replay_trajectory)
    assert "propagate_state(" in source
    assert "star_tracker_update(" in source
    for forbidden in (
        "dataset.truth",
        "q_true_NB",
        "gyro_bias_rad_s",
        "omega_true_rad_s",
        "oracle",
        "future",
    ):
        assert forbidden not in source


def test_replay_does_not_mutate_inputs_and_outputs_are_readonly() -> None:
    generated = generate_unit_st(UnitSTSyntheticConfig(num_trajectories=4, duration_s=0.8))
    event_copies = _event_kwargs(generated.dataset.events)
    truth_copies = {
        field.name: np.array(getattr(generated.dataset.truth, field.name), copy=True)
        for field in fields(generated.dataset.truth)
    }
    prior = _initial_state(generated)
    prior_q = prior.q_NB.copy()
    prior_b = prior.b_g.copy()
    prior_P = prior.P.copy()
    result = replay_trajectory(
        generated.dataset.events,
        int(generated.dataset.truth.trajectory_id[0]),
        prior,
        0.0,
        _process_noise(),
    )
    for field in fields(generated.dataset.events):
        assert np.array_equal(getattr(generated.dataset.events, field.name), event_copies[field.name])
    for field in fields(generated.dataset.truth):
        assert np.array_equal(getattr(generated.dataset.truth, field.name), truth_copies[field.name])
    assert np.array_equal(prior.q_NB, prior_q)
    assert np.array_equal(prior.b_g, prior_b)
    assert np.array_equal(prior.P, prior_P)
    for field in fields(result):
        value = getattr(result, field.name)
        if isinstance(value, np.ndarray):
            assert not value.flags.writeable


def test_gate_a_state_immutability_remains_enforced() -> None:
    state = MEKFState(
        q_NB=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
        b_g=np.zeros(3, dtype=np.float64),
        P=np.eye(6, dtype=np.float64),
    )
    with pytest.raises(ValueError):
        state.q_NB[0] = 0.0
    with pytest.raises(ValueError):
        state.b_g[0] = 1.0
    with pytest.raises(ValueError):
        state.P[0, 0] = 2.0
