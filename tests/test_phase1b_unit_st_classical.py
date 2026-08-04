from __future__ import annotations

import inspect
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

import bench.experiments.phase1b_unit_st_classical as experiment
from bench.estimators.mekf import MEKFState
from bench.tasks.generator.mekf_events import MEKFEventTable
from bench.tasks.generator.unit_st_regimes import (
    RegimeCode,
    UnitSTRegimeConfig,
    generate_base_unit_st,
    generate_unit_st_regime,
)


@pytest.fixture(scope="module")
def stationary():
    config = UnitSTRegimeConfig(
        truth_source="synthetic",
        num_trajectories=5,
        duration_s=2.0,
        gyro_rate_hz=10,
        star_tracker_rate_hz=2,
        master_seed=202,
    )
    return config, generate_unit_st_regime(config)


@pytest.fixture(scope="module")
def step_cases(stationary):
    config, _generated = stationary
    base = generate_base_unit_st(config)
    c2 = generate_unit_st_regime(
        UnitSTRegimeConfig(
            **{
                **asdict(config),
                "regime_code": int(RegimeCode.C2_GYRO_PROCESS_STEP),
                "event_covariance_multiplier": 4.0,
            }
        ),
        base_generated=base,
    )
    c3 = generate_unit_st_regime(
        UnitSTRegimeConfig(
            **{
                **asdict(config),
                "regime_code": int(RegimeCode.C3_STAR_TRACKER_RELIABILITY_STEP),
                "event_covariance_multiplier": 4.0,
            }
        ),
        base_generated=base,
    )
    return c2, c3


def _q_c(config: UnitSTRegimeConfig) -> np.ndarray:
    return experiment.base_process_covariance(config, bias_psd=1.0e-12)


def _trajectory(generated) -> int:
    return int(generated.dataset.truth.trajectory_id[0])


def test_fixed_policy_public_api_has_no_oracle_or_hidden_label() -> None:
    parameters = inspect.signature(experiment.replay_fixed_policy).parameters
    assert set(parameters) == {
        "event_table",
        "trajectory_id",
        "initial_state",
        "initial_time_s",
        "base_Q_c",
        "policy",
    }
    assert not {"oracle", "context", "label", "window"} & set(parameters)


def test_deployable_policy_artifact_has_no_oracle_context() -> None:
    artifact = experiment.FixedPolicy("F-TUNED", 2.0, 0.5, 4.0).deployable_artifact()
    assert set(artifact) == {"policy_contract_version", "policy_id", "qg_scale", "qb_scale", "r_scale"}
    assert "oracle" not in str(artifact).lower()


def test_all_one_fixed_replay_is_bit_exact_phase1a(stationary) -> None:
    config, generated = stationary
    for trajectory_id in generated.dataset.truth.trajectory_id:
        experiment.assert_all_one_replay_exact(
            generated.dataset.events,
            int(trajectory_id),
            experiment.default_initial_state(),
            0.0,
            _q_c(config),
        )


def test_replay_does_not_mutate_raw_measurement_stream(stationary) -> None:
    config, generated = stationary
    events = generated.dataset.events
    before = [np.array(getattr(events, name), copy=True) for name in (
        "gyro_omega_rad_s", "star_tracker_q_NB", "star_tracker_R_rad2"
    )]
    experiment.replay_fixed_policy(
        events,
        _trajectory(generated),
        experiment.default_initial_state(),
        0.0,
        _q_c(config),
        experiment.F_BASE,
    )
    for name, expected in zip(("gyro_omega_rad_s", "star_tracker_q_NB", "star_tracker_R_rad2"), before):
        assert np.array_equal(getattr(events, name), expected)


def test_fixed_q_and_r_scales_reach_gate_a_calls(monkeypatch, stationary) -> None:
    config, generated = stationary
    q_c = _q_c(config)
    seen_q: list[np.ndarray] = []
    seen_r: list[np.ndarray] = []
    real_propagate = experiment.propagate_state
    real_update = experiment.star_tracker_update

    def propagation(*args, **kwargs):
        seen_q.append(np.array(args[3], copy=True))
        return real_propagate(*args, **kwargs)

    def update(*args, **kwargs):
        seen_r.append(np.array(args[2], copy=True))
        return real_update(*args, **kwargs)

    monkeypatch.setattr(experiment, "propagate_state", propagation)
    monkeypatch.setattr(experiment, "star_tracker_update", update)
    policy = experiment.FixedPolicy("fixture", qg_scale=2.0, qb_scale=0.5, r_scale=4.0)
    experiment.replay_fixed_policy(
        generated.dataset.events,
        _trajectory(generated),
        experiment.default_initial_state(),
        0.0,
        q_c,
        policy,
    )
    assert all(np.array_equal(value[:3, :3], q_c[:3, :3] * 2.0) for value in seen_q)
    assert all(np.array_equal(value[3:, 3:], q_c[3:, 3:] * 0.5) for value in seen_q)
    nominal_r = generated.dataset.events.star_tracker_R_rad2[0]
    assert all(np.array_equal(value, nominal_r * 4.0) for value in seen_r)


@pytest.mark.parametrize("case_index,expected_q,expected_r", [(0, 4.0, 1.0), (1, 1.0, 4.0)])
def test_oracle_maps_current_scale_to_correct_side(monkeypatch, step_cases, case_index, expected_q, expected_r) -> None:
    generated = step_cases[case_index]
    config_alpha = 4.0
    seen_q: list[float] = []
    seen_r: list[float] = []
    q_c = experiment.base_process_covariance(
        UnitSTRegimeConfig(
            truth_source="synthetic", num_trajectories=3, duration_s=1.0, gyro_rate_hz=10, star_tracker_rate_hz=2
        ),
        bias_psd=1.0e-12,
    )
    nominal_r = generated.dataset.events.star_tracker_R_rad2[0]
    real_propagate = experiment.propagate_state
    real_update = experiment.star_tracker_update

    def propagation(*args, **kwargs):
        seen_q.append(float(args[3][0, 0] / q_c[0, 0]))
        return real_propagate(*args, **kwargs)

    def update(*args, **kwargs):
        seen_r.append(float(args[2][0, 0] / nominal_r[0, 0]))
        return real_update(*args, **kwargs)

    monkeypatch.setattr(experiment, "propagate_state", propagation)
    monkeypatch.setattr(experiment, "star_tracker_update", update)
    experiment.replay_oracle_policy(
        generated.dataset.events,
        _trajectory(generated),
        experiment.default_initial_state(),
        0.0,
        q_c,
        generated.oracle_context,
    )
    assert expected_q in seen_q if case_index == 0 else all(value == 1.0 for value in seen_q)
    assert expected_r in seen_r if case_index == 1 else all(value == 1.0 for value in seen_r)
    assert config_alpha == 4.0


@pytest.mark.parametrize("case_index,expected_q,expected_r", [(0, 1.0, 4.0), (1, 4.0, 1.0)])
def test_wrong_side_swaps_process_and_measurement_actions(monkeypatch, step_cases, case_index, expected_q, expected_r) -> None:
    generated = step_cases[case_index]
    q_c = np.diag(np.asarray([1.0e-8] * 3 + [1.0e-12] * 3, dtype=np.float64))
    nominal_r = generated.dataset.events.star_tracker_R_rad2[0]
    seen_q: list[float] = []
    seen_r: list[float] = []
    real_propagate = experiment.propagate_state
    real_update = experiment.star_tracker_update

    def propagation(*args, **kwargs):
        seen_q.append(float(args[3][0, 0] / q_c[0, 0]))
        return real_propagate(*args, **kwargs)

    def update(*args, **kwargs):
        seen_r.append(float(args[2][0, 0] / nominal_r[0, 0]))
        return real_update(*args, **kwargs)

    monkeypatch.setattr(experiment, "propagate_state", propagation)
    monkeypatch.setattr(experiment, "star_tracker_update", update)
    experiment.replay_wrong_side_policy(
        generated.dataset.events,
        _trajectory(generated),
        experiment.default_initial_state(),
        0.0,
        q_c,
        generated.oracle_context,
    )
    assert expected_q in seen_q if case_index == 1 else all(value == 1.0 for value in seen_q)
    assert expected_r in seen_r if case_index == 0 else all(value == 1.0 for value in seen_r)


def test_oracle_cursor_is_consumed_for_every_event(step_cases) -> None:
    generated = step_cases[0]
    cursor = generated.oracle_context.cursor(_trajectory(generated))
    rows = np.flatnonzero(generated.dataset.events.trajectory_id == np.int64(_trajectory(generated)))
    for order in generated.dataset.events.event_order[rows]:
        cursor.consume(int(order))
    with pytest.raises(ValueError):
        cursor.consume(len(rows))


def test_quaternion_sign_flip_produces_same_replay(stationary) -> None:
    config, generated = stationary
    source = generated.dataset.events
    flipped = MEKFEventTable(
        trajectory_id=source.trajectory_id,
        sensor_code=source.sensor_code,
        measurement_time_s=source.measurement_time_s,
        arrival_time_s=source.arrival_time_s,
        event_order=source.event_order,
        valid=source.valid,
        payload_index=source.payload_index,
        gyro_omega_rad_s=source.gyro_omega_rad_s,
        star_tracker_q_NB=-source.star_tracker_q_NB,
        star_tracker_R_rad2=source.star_tracker_R_rad2,
    )
    trajectory_id = _trajectory(generated)
    first = experiment.replay_fixed_policy(source, trajectory_id, experiment.default_initial_state(), 0.0, _q_c(config), experiment.F_BASE)
    second = experiment.replay_fixed_policy(flipped, trajectory_id, experiment.default_initial_state(), 0.0, _q_c(config), experiment.F_BASE)
    for name in ("q_NB_history", "b_g_history", "P_history", "star_tracker_residual", "star_tracker_S"):
        assert np.array_equal(getattr(first, name), getattr(second, name))


@pytest.mark.parametrize("case_index", [0, 1])
def test_policy_replay_covariances_remain_spd(step_cases, case_index: int) -> None:
    generated = step_cases[case_index]
    q_c = np.diag(np.asarray([2.5e-8] * 3 + [1.0e-12] * 3, dtype=np.float64))
    for policy in (experiment.F_BASE, "ORACLE-QR", "WRONG-SIDE"):
        replay = experiment._run_policy(generated, _trajectory(generated), q_c, policy)
        assert all(np.linalg.cholesky(item) is not None for item in replay.P_history)
        assert all(np.linalg.cholesky(item) is not None for item in replay.star_tracker_S)


def test_evaluation_uses_canonical_metrics_and_exact_truth_join(stationary) -> None:
    config, generated = stationary
    replay = experiment.replay_fixed_policy(
        generated.dataset.events, _trajectory(generated), experiment.default_initial_state(), 0.0, _q_c(config), experiment.F_BASE
    )
    result = experiment.evaluate_replay(
        generated.dataset,
        generated.oracle_context,
        replay,
        scenario_id="fixture",
        policy_id="F-BASE",
        recovery_floor_rad=1.0e-3,
        divergence_threshold_rad=1.0,
    )
    assert result["trajectory_id"] == replay.trajectory_id
    assert result["event_count"] == replay.processed_event_count
    assert result["minimum_p_eigenvalue"] > 0.0
    assert result["minimum_s_eigenvalue"] > 0.0


def test_fixed_policy_is_independent_of_event_boundary(stationary) -> None:
    config, generated = stationary
    changed_config = UnitSTRegimeConfig(
        **{**asdict(config), "event_start_fraction": 0.2, "event_end_fraction": 0.8}
    )
    changed = generate_unit_st_regime(changed_config)
    trajectory_id = _trajectory(generated)
    first = experiment.replay_fixed_policy(generated.dataset.events, trajectory_id, experiment.default_initial_state(), 0.0, _q_c(config), experiment.F_BASE)
    second = experiment.replay_fixed_policy(changed.dataset.events, trajectory_id, experiment.default_initial_state(), 0.0, _q_c(config), experiment.F_BASE)
    assert np.array_equal(first.q_NB_history, second.q_NB_history)
    assert np.array_equal(first.P_history, second.P_history)


def test_recovery_time_fixture() -> None:
    times = np.arange(8, dtype=np.float64)
    errors = np.asarray([0.1, 0.1, 0.5, 0.4, 0.11, 0.11, 0.11, 0.2], dtype=np.float64)
    windows = np.asarray([0, 0, 1, 1, 2, 2, 2, 2], dtype=np.int8)
    assert experiment.recovery_time_s(times, errors, windows, absolute_floor_rad=0.01, sustained_samples=3) == 1.0


def test_recovery_time_none_when_not_sustained() -> None:
    times = np.arange(7, dtype=np.float64)
    errors = np.asarray([0.1, 0.1, 0.5, 0.4, 0.11, 0.4, 0.11], dtype=np.float64)
    windows = np.asarray([0, 0, 1, 1, 2, 2, 2], dtype=np.int8)
    assert experiment.recovery_time_s(times, errors, windows, absolute_floor_rad=0.01, sustained_samples=3) is None


def test_paired_bootstrap_is_deterministic_and_requires_2000() -> None:
    values = np.asarray([-1.0, 0.0, 1.0, 2.0], dtype=np.float64)
    first = experiment.paired_bootstrap_ci(values, seed=9, resamples=2000)
    second = experiment.paired_bootstrap_ci(values, seed=9, resamples=2000)
    assert first == second
    with pytest.raises(ValueError, match="at least 2000"):
        experiment.paired_bootstrap_ci(values, seed=9, resamples=1999)


def test_summary_pairs_by_exact_trajectory_id() -> None:
    records = []
    for policy, offset in (("F-BASE", 0.0), ("ORACLE-QR", -0.1)):
        for trajectory_id in (5, 7):
            records.append(
                {
                    "scenario_id": "C2",
                    "policy_id": policy,
                    "trajectory_id": trajectory_id,
                    "diverged": False,
                    "attitude_event_rmse_rad": trajectory_id + offset,
                    "attitude_event_p95_rad": 1.0,
                    "attitude_event_peak_rad": 1.0,
                    "bias_vector_rmse_rad_s": 1.0,
                    "event_innovation_rms_rad": 1.0,
                    "event_innovation_norm_median_rad": 1.0,
                    "event_innovation_norm_p95_rad": 1.0,
                    "event_innovation_lag1": 0.0,
                    "event_raw_gyro_measurement_rms_rad_s": 1.0,
                    "event_raw_gyro_increment_rms_rad_s": 1.0,
                    "recovery_time_s": 1.0,
                    "nis_normalized_mean": 1.0,
                    "nees_normalized_mean": 1.0,
                }
            )
    summary = experiment.summarize_records(records, bootstrap_seed=3, resamples=2000)
    paired = summary["paired_differences"]["C2/ORACLE-QR-minus-F-BASE"]
    assert paired["N"] == 2
    assert paired["mean_attitude_event_rmse_difference_rad"] == pytest.approx(-0.1)


def test_tuning_evaluates_exact_budget_without_test_ids(monkeypatch, stationary) -> None:
    config, generated = stationary
    seen_ids: set[int] = set()

    def fake_evaluate(_generated, trajectory_ids, _q, policy, **_kwargs):
        ids = tuple(int(item) for item in trajectory_ids)
        seen_ids.update(ids)
        return [
            {
                "diverged": False,
                "attitude_rmse_rad": policy.qg_scale + policy.qb_scale + policy.r_scale,
                "bias_vector_rmse_rad_s": policy.qb_scale,
                "nis_normalized_mean": policy.r_scale,
                "nees_normalized_mean": policy.qg_scale,
            }
            for _item in ids
        ]

    monkeypatch.setattr(experiment, "_replay_and_evaluate_fixed", fake_evaluate)
    selected, manifest = experiment.tune_fixed_policy(generated, _q_c(config))
    assert isinstance(selected, experiment.FixedPolicy)
    assert manifest["evaluated_candidate_count"] == 42
    assert manifest["candidate_budget"] == 42
    assert manifest["test_split_accessed"] is False
    assert not seen_ids & set(int(item) for item in generated.trajectory_split.test_ids)


def test_tuning_tie_break_is_deterministic(monkeypatch, stationary) -> None:
    config, generated = stationary

    def tied(_generated, trajectory_ids, _q, _policy, **_kwargs):
        return [
            {
                "diverged": False,
                "attitude_rmse_rad": 1.0,
                "bias_vector_rmse_rad_s": 1.0,
                "nis_normalized_mean": 1.0,
                "nees_normalized_mean": 1.0,
            }
            for _item in trajectory_ids
        ]

    monkeypatch.setattr(experiment, "_replay_and_evaluate_fixed", tied)
    first, _ = experiment.tune_fixed_policy(generated, _q_c(config))
    second, _ = experiment.tune_fixed_policy(generated, _q_c(config))
    assert first == second
    assert (first.qg_scale, first.qb_scale, first.r_scale) == (0.125, 0.125, 0.125)


def test_c5_matching_uses_validation_and_freezes_before_test(monkeypatch, stationary, step_cases) -> None:
    config, _stationary_generated = stationary
    c2, _c3 = step_cases
    seen_ids: set[int] = set()

    def fake_evaluate(generated, trajectory_ids, _q, _policy, **_kwargs):
        ids = tuple(int(item) for item in trajectory_ids)
        seen_ids.update(ids)
        alpha_r = float(np.max(generated.oracle_context.alpha_R_ST))
        value = 2.0 if alpha_r == 1.0 else 2.0 * np.sqrt(alpha_r)
        return [{"event_innovation_rms_rad": value} for _item in ids]

    monkeypatch.setattr(experiment, "_replay_and_evaluate_fixed", fake_evaluate)
    selected, manifest = experiment.match_c5_innovation_rms(
        c2,
        config,
        _q_c(config),
        candidate_alpha_R=(1.0, 1.1, 2.0),
    )
    assert selected == 1.0
    assert manifest["match_within_tolerance"] is True
    assert manifest["test_split_accessed"] is False
    assert manifest["frozen_before_test"] is True
    assert not seen_ids & set(int(item) for item in c2.trajectory_split.test_ids)


def test_divergence_threshold_is_reported_not_hidden(stationary) -> None:
    config, generated = stationary
    replay = experiment.replay_fixed_policy(
        generated.dataset.events, _trajectory(generated), experiment.default_initial_state(), 0.0, _q_c(config), experiment.F_BASE
    )
    result = experiment.evaluate_replay(
        generated.dataset,
        generated.oracle_context,
        replay,
        scenario_id="fixture",
        policy_id="F-BASE",
        recovery_floor_rad=1.0e-3,
        divergence_threshold_rad=1.0e-12,
    )
    assert result["diverged"] is True


def test_workload_locks_minimum_paired_n_50() -> None:
    config = experiment._load_config(Path("bench/configs/suite_phase1b_unit_st_classical.yaml"))
    workload = experiment.pilot_workload(config)
    assert workload["required_test_trajectories_per_condition"] == 50
    assert workload["scenario_count"] == 9
    assert workload["filter_event_steps"] > 0


def test_small_multiseed_report_generation(tmp_path: Path) -> None:
    config = experiment._load_config(Path("bench/configs/suite_phase1b_unit_st_classical.yaml"))
    config = {**config, "paths": {"results_root": str(tmp_path / "results"), "manifests_root": str(tmp_path / "manifests")}}
    records_root = tmp_path / "results" / "pilot" / "records" / "C1" / "F-BASE"
    records_root.mkdir(parents=True)
    for trajectory_id in (11, 13, 17):
        record = {
            "scenario_id": "C1",
            "policy_id": "F-BASE",
            "trajectory_id": trajectory_id,
            "diverged": False,
            "attitude_event_rmse_rad": 0.001,
            "attitude_event_p95_rad": 0.002,
            "attitude_event_peak_rad": 0.003,
            "bias_vector_rmse_rad_s": 0.0001,
            "event_innovation_rms_rad": 0.001,
            "event_innovation_norm_median_rad": 0.001,
            "event_innovation_norm_p95_rad": 0.002,
            "event_innovation_lag1": 0.0,
            "event_raw_gyro_measurement_rms_rad_s": 0.1,
            "event_raw_gyro_increment_rms_rad_s": 0.001,
            "recovery_time_s": 0.1,
            "nis_normalized_mean": 1.0,
            "nees_normalized_mean": 1.0,
        }
        experiment._write_json(records_root / f"{trajectory_id}.json", record)
    experiment._write_json(
        tmp_path / "results" / "pilot" / "pilot_manifest.json",
        {"status": "PARTIAL", "required_paired_N_per_condition": 50, "completed_paired_N_per_condition": 3},
    )
    output = experiment._report_command(config, include_long_horizon=False)
    assert output["completed_paired_N_per_condition"] == 3
    assert output["summary"]["groups"]["C1/F-BASE"]["N"] == 3
    assert (tmp_path / "results" / "pilot_summary.json").is_file()


def test_source_has_no_filter_math_duplication_or_failure_hiding() -> None:
    source = Path("bench/experiments/phase1b_unit_st_classical.py").read_text(encoding="utf-8")
    forbidden = ("np.linalg.inv", "np.linalg.pinv", "np.clip", "pseudo-inverse", "xfail", "torch", "kalman_net", "magnetometer", "sun_sensor")
    for token in forbidden:
        assert token not in source.lower()
    assert "propagate_state(" in source
    assert "star_tracker_update(" in source


def test_default_initial_state_is_defensively_read_only() -> None:
    state = experiment.default_initial_state()
    assert isinstance(state, MEKFState)
    for array in (state.q_NB, state.b_g, state.P):
        assert not array.flags.writeable
        with pytest.raises(ValueError):
            array.flat[0] = array.flat[0]
