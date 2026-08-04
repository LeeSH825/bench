from __future__ import annotations

import inspect

import numpy as np
import pytest

from bench.estimators.mekf import MEKFState
from bench.experiments.phase1b_sensor_fusion_c4 import (
    F_BASE,
    ORACLE_FULL,
    ORACLE_MEASUREMENT,
    ORACLE_PROCESS,
    WRONG_MEASUREMENT,
    WRONG_PROCESS,
    _oracle_scales,
    assert_all_one_oracle_exact,
    base_process_covariance,
    evaluate_fusion_replay,
    replay_fixed_policy,
    replay_oracle_policy,
)
from bench.experiments.phase1b_unit_st_classical import default_initial_state
from bench.metrics.mekf_fusion import magnetometer_nis, summarize_sensor_consistency, sun_sensor_nis
from bench.tasks.generator.phase1b_sensor_fusion import (
    FusionScenarioCode,
    SensorFusionConfig,
    generate_sensor_fusion,
)


@pytest.fixture(scope="module")
def main_generated():
    return generate_sensor_fusion(
        SensorFusionConfig(
            num_trajectories=5,
            duration_s=5.0,
            master_seed=27301,
            scenario_code=int(FusionScenarioCode.MAIN_FUSION_STATIONARY),
        )
    )


def test_fixed_api_has_no_truth_oracle_label_window(main_generated) -> None:
    names = set(inspect.signature(replay_fixed_policy).parameters)
    assert not ({"truth", "oracle", "event_window", "hidden_label"} & names)
    assert "oracle_context" in inspect.signature(replay_oracle_policy).parameters


def test_all_one_oracle_is_exact(main_generated) -> None:
    trajectory_id = int(main_generated.trajectory_split.test_ids[0])
    assert_all_one_oracle_exact(
        main_generated.dataset.events,
        main_generated.oracle_context,
        trajectory_id,
        default_initial_state(),
        base_process_covariance(
            SensorFusionConfig(
                num_trajectories=5,
                duration_s=5.0,
                master_seed=27301,
                scenario_code=int(FusionScenarioCode.MAIN_FUSION_STATIONARY),
            )
        ),
    )


def test_invalid_sun_update_is_exact_skip(main_generated) -> None:
    cfg = SensorFusionConfig(
        num_trajectories=5,
        duration_s=5.0,
        master_seed=27301,
        scenario_code=int(FusionScenarioCode.MAIN_FUSION_STATIONARY),
    )
    trajectory_id = next(
        int(item)
        for item in main_generated.dataset.truth.trajectory_id
        if np.any(
            ~main_generated.dataset.events.valid[
                main_generated.dataset.events.trajectory_id == item
            ]
        )
    )
    replay = replay_fixed_policy(
        main_generated.dataset.events,
        trajectory_id,
        default_initial_state(),
        0.0,
        base_process_covariance(cfg),
        F_BASE,
    )
    assert replay.sun_skipped_event_order.size > 0
    for order in replay.sun_skipped_event_order:
        row = int(np.flatnonzero(replay.event_order == order)[0])
        assert np.array_equal(replay.q_NB_history[row], replay.q_NB_history[row - 1])
        assert np.array_equal(replay.P_history[row], replay.P_history[row - 1])


def test_star_tracker_antipodes_produce_identical_replay(main_generated) -> None:
    from dataclasses import fields
    from bench.tasks.generator.mekf_fusion_events import FusionEventTable

    table = main_generated.dataset.events
    values = {item.name: np.array(getattr(table, item.name), copy=True) for item in fields(table)}
    values["star_tracker_q_ST_NB"] *= -1.0
    antipodal = FusionEventTable(**values)
    cfg = SensorFusionConfig(num_trajectories=5, duration_s=5.0, master_seed=27301)
    trajectory_id = int(main_generated.trajectory_split.test_ids[0])
    left = replay_fixed_policy(table, trajectory_id, default_initial_state(), 0.0, base_process_covariance(cfg))
    right = replay_fixed_policy(
        antipodal, trajectory_id, default_initial_state(), 0.0, base_process_covariance(cfg)
    )
    for name in ("q_NB_history", "b_g_history", "P_history", "star_tracker_residual"):
        assert np.array_equal(getattr(left, name), getattr(right, name))


@pytest.mark.parametrize(
    ("policy", "expected"),
    [
        (ORACLE_PROCESS, (7.0, 1.0)),
        (ORACLE_MEASUREMENT, (1.0, 11.0)),
        (ORACLE_FULL, (7.0, 11.0)),
        (WRONG_PROCESS, (11.0, 1.0)),
        (WRONG_MEASUREMENT, (1.0, 7.0)),
    ],
)
def test_oracle_and_wrong_side_mappings_are_exact(policy, expected) -> None:
    assert _oracle_scales(policy, 7.0, 11.0) == expected


def test_sensor_nis_closed_forms_and_counts() -> None:
    mag_r = np.asarray([[1.0, 2.0, 3.0]], dtype=np.float64)
    mag_s = np.asarray([np.diag([2.0, 4.0, 8.0])], dtype=np.float64)
    assert magnetometer_nis(mag_r, mag_s)[0] == pytest.approx(1 / 2 + 4 / 4 + 9 / 8)
    sun_r = np.asarray([[2.0, 3.0]], dtype=np.float64)
    sun_s = np.asarray([np.diag([4.0, 9.0])], dtype=np.float64)
    assert sun_sensor_nis(sun_r, sun_s)[0] == pytest.approx(2.0)
    summary = summarize_sensor_consistency(
        sun_r,
        sun_s,
        sensor_name="sun_sensor",
        degrees_of_freedom=2,
        total_event_count=3,
        skip_count=2,
    )
    assert (summary.update_count, summary.skip_count, summary.total_event_count) == (1, 2, 3)


def test_sensor_nis_fails_loudly_on_non_spd() -> None:
    with pytest.raises(ValueError):
        sun_sensor_nis(
            np.ones((1, 2), dtype=np.float64),
            np.asarray([[[1.0, 0.0], [0.0, 0.0]]], dtype=np.float64),
        )


def test_stationary_replay_is_finite_spd_and_evaluable(main_generated) -> None:
    cfg = SensorFusionConfig(num_trajectories=5, duration_s=5.0, master_seed=27301)
    trajectory_id = int(main_generated.trajectory_split.test_ids[0])
    replay = replay_fixed_policy(
        main_generated.dataset.events,
        trajectory_id,
        default_initial_state(),
        0.0,
        base_process_covariance(cfg),
    )
    assert np.all(np.isfinite(replay.q_NB_history))
    assert np.all(np.linalg.eigvalsh(replay.P_history) > 0.0)
    record = evaluate_fusion_replay(
        main_generated.dataset,
        replay,
        scenario_id="TEST",
        policy_id="F-BASE",
        duration_s=cfg.duration_s,
        divergence_threshold_rad=0.75,
    )
    assert record["mag_nis_count"] > 0 and record["sun_nis_count"] > 0
    assert not record["diverged"]
