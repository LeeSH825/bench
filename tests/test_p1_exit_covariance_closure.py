from __future__ import annotations

import ast
import inspect
from pathlib import Path

import numpy as np
import pytest

from bench.experiments.p1_exit_covariance_closure import (
    CalibrationScales,
    _frozen_phase1_test_ids,
    _generator_config,
    closure_train_validation_split,
    coordinate_candidates,
    covariance_decomposition,
    deterministic_select,
    first_settling_bin,
    load_closure_config,
    local_combined_grid,
    partition_masks,
    replay_calibrated_fixed,
    scaled_initial_state,
    validation_guard,
)
from bench.experiments.phase1b_sensor_fusion_c4 import (
    F_BASE,
    base_process_covariance,
    replay_fixed_policy,
)
from bench.experiments.phase1b_unit_st_classical import default_initial_state
from bench.tasks.generator.phase1b_sensor_fusion import (
    FusionScenarioCode,
    generate_sensor_fusion,
)


CONFIG_PATH = Path("bench/configs/suite_p1_exit_covariance_closure.yaml")


def _config():
    return load_closure_config(CONFIG_PATH)


def _aggregate(
    *,
    full: float = 1.0,
    attitude: float = 1.0,
    bias: float = 1.0,
    attitude_rmse: float = 0.01,
    bias_rmse: float = 0.001,
    sensor_nis: float = 1.0,
):
    partition = {
        "full_nees_normalized": full,
        "attitude_nees_normalized": attitude,
        "bias_nees_normalized": bias,
        "attitude_rmse_rad": attitude_rmse,
        "attitude_p95_rad_mean": attitude_rmse * 1.5,
        "bias_rmse_rad_s": bias_rmse,
        "bias_p95_rad_s_mean": bias_rmse * 1.5,
        "mag_nis_normalized": sensor_nis,
        "sun_nis_normalized": sensor_nis,
        "st_nis_normalized": sensor_nis,
    }
    return {
        "N": 20,
        "divergence_count": 0,
        "minimum_P_eigenvalue": 1e-9,
        "minimum_S_eigenvalue": 1e-9,
        "partitions": {
            "whole": dict(partition),
            "initial": dict(partition),
            "middle": dict(partition),
            "settled": dict(partition),
        },
    }


def _candidate(scales: CalibrationScales, full: float, identifier: str):
    baseline = _aggregate()["partitions"]["settled"]
    aggregate = _aggregate(full=full)
    return {
        "candidate_id": identifier,
        "scales": scales.as_dict(),
        "aggregate": aggregate,
        "baseline_settled": baseline,
        "guard": {"passed": True, "checks": {}},
    }


def test_config_locks_f_base_f_tuned_sensor_r_and_30_20_50_50() -> None:
    config = _config()
    frozen = config["frozen_foundation"]
    assert frozen["fixed_primary"] == "F-BASE"
    assert frozen["frozen_sensitivity"] == {
        "policy_id": "F-TUNED",
        "s_Qg": 0.125,
        "s_Qb": 0.125,
        "s_R_ST": 8.0,
    }
    assert set(map(float, frozen["sensor_R_scales"].values())) == {1.0}
    data = config["data"]
    assert (data["train_N"], data["validation_N"]) == (30, 20)
    assert (data["confirmation_stationary_N"], data["confirmation_c4_N"]) == (50, 50)


def test_closure_split_is_deterministic_disjoint_and_whole_trajectory() -> None:
    ids = np.arange(1000, 1050, dtype=np.int64)
    first = closure_train_validation_split(
        ids, train_count=30, validation_count=20, split_seed=71
    )
    second = closure_train_validation_split(
        ids, train_count=30, validation_count=20, split_seed=71
    )
    assert np.array_equal(first[0], second[0])
    assert np.array_equal(first[1], second[1])
    assert first[0].size == 30 and first[1].size == 20
    assert not (set(map(int, first[0])) & set(map(int, first[1])))
    assert set(map(int, np.concatenate(first))) == set(map(int, ids))


def test_calibration_and_confirmation_seed_namespaces_are_new_and_disjoint_from_frozen() -> None:
    config = _config()
    calibration = generate_sensor_fusion(
        _generator_config(
            config,
            scenario=FusionScenarioCode.MAIN_FUSION_STATIONARY,
            master_seed=int(config["data"]["calibration_master_seed"]),
            count=5,
        )
    )
    confirmation = generate_sensor_fusion(
        _generator_config(
            config,
            scenario=FusionScenarioCode.MAIN_FUSION_STATIONARY,
            master_seed=int(config["data"]["confirmation_master_seed"]),
            count=5,
        )
    )
    calibration_ids = set(map(int, calibration.dataset.truth.trajectory_id))
    confirmation_ids = set(map(int, confirmation.dataset.truth.trajectory_id))
    frozen = _frozen_phase1_test_ids()
    assert not (calibration_ids & confirmation_ids)
    assert not (calibration_ids & frozen)
    assert not (confirmation_ids & frozen)


def test_scaled_initial_state_applies_only_pre_replay_P0_blocks() -> None:
    base = default_initial_state()
    scaled = scaled_initial_state(CalibrationScales(2.0, 4.0, 1.0, 1.0))
    assert np.array_equal(scaled.q_NB, base.q_NB)
    assert np.array_equal(scaled.b_g, base.b_g)
    assert np.array_equal(scaled.P[:3, :3], base.P[:3, :3] * 2.0)
    assert np.array_equal(scaled.P[3:, 3:], base.P[3:, 3:] * 4.0)
    assert np.array_equal(scaled.P[:3, 3:], base.P[:3, 3:])
    assert not scaled.P.flags.writeable


def test_full_marginal_and_whitened_closed_forms() -> None:
    error = np.asarray([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]], dtype=np.float64)
    diagonal = np.asarray([2.0, 4.0, 8.0, 16.0, 32.0, 64.0], dtype=np.float64)
    covariance = np.diag(diagonal)[None, :, :]
    result = covariance_decomposition(error, covariance)
    expected_energy = error[0] ** 2 / diagonal
    assert result.full_nees[0] == pytest.approx(float(np.sum(expected_energy)))
    assert result.attitude_nees[0] == pytest.approx(float(np.sum(expected_energy[:3])))
    assert result.bias_nees[0] == pytest.approx(float(np.sum(expected_energy[3:])))
    assert np.allclose(result.whitened_error[0], error[0] / np.sqrt(diagonal))
    assert np.allclose(result.whitened_energy[0], expected_energy)
    assert result.cross_relative_norm[0] == 0.0
    assert np.array_equal(result.cross_correlation_block[0], np.zeros((3, 3)))


def test_cross_covariance_is_retained_and_correlation_normalized() -> None:
    covariance = np.eye(6, dtype=np.float64)
    covariance[0, 3] = covariance[3, 0] = 0.2
    covariance[1, 4] = covariance[4, 1] = -0.1
    result = covariance_decomposition(
        np.ones((1, 6), dtype=np.float64), covariance[None, :, :]
    )
    assert result.cross_relative_norm[0] > 0.0
    assert result.cross_correlation_block[0, 0, 0] == pytest.approx(0.2)
    assert result.cross_correlation_block[0, 1, 1] == pytest.approx(-0.1)


def test_covariance_decomposition_fails_loudly_on_non_spd() -> None:
    covariance = np.eye(6, dtype=np.float64)
    covariance[-1, -1] = 0.0
    with pytest.raises((ValueError, np.linalg.LinAlgError)):
        covariance_decomposition(
            np.ones((1, 6), dtype=np.float64), covariance[None, :, :]
        )


def test_partition_boundaries_are_exact_and_settled_includes_T() -> None:
    times = np.asarray([0.0, 1.999, 2.0, 5.999, 6.0, 10.0], dtype=np.float64)
    masks = partition_masks(
        times,
        10.0,
        {"initial": [0.0, 0.2], "middle": [0.2, 0.6], "settled": [0.6, 1.0]},
    )
    assert np.array_equal(masks["initial"], [True, True, False, False, False, False])
    assert np.array_equal(masks["middle"], [False, False, True, True, False, False])
    assert np.array_equal(masks["settled"], [False, False, False, False, True, True])


def test_known_settling_fixture_requires_three_consecutive_bins() -> None:
    def record(value: float):
        return {
            "attitude_rmse_rad": value,
            "bias_rmse_rad_s": value,
            "full_nees_normalized": 1.0,
            "mag_nis_normalized": 1.0,
            "sun_nis_normalized": 1.0,
            "st_nis_normalized": 1.0,
        }

    records = [record(2.0), record(0.5), record(2.0), record(0.5), record(0.5), record(0.5)]
    assert first_settling_bin(
        records,
        consecutive_bins=3,
        attitude_rmse_max_rad=1.0,
        bias_rmse_max_rad_s=1.0,
        full_nees_band=(0.8, 1.2),
        sensor_nis_band=(0.8, 1.2),
    ) == 3


def test_coordinate_and_local_candidate_budget_is_exact() -> None:
    grid = (0.5, 1.0, 2.0, 4.0, 8.0)
    center = CalibrationScales(1.0, 2.0, 2.0, 4.0)
    for field in ("s_P0_att", "s_P0_bias", "s_Qg", "s_Qb"):
        candidates = coordinate_candidates(center, field, grid)
        assert len(candidates) == 5
        assert {getattr(item, field) for item in candidates} == set(grid)
    local = local_combined_grid(center, grid)
    assert len(local) == 81
    assert len(set(local)) == 81
    assert {item.s_P0_att for item in local} == {0.5, 1.0, 2.0}
    assert {item.s_Qb for item in local} == {2.0, 4.0, 8.0}


def test_local_grid_boundary_uses_nearest_three_locked_values_and_keeps_budget() -> None:
    local = local_combined_grid(
        CalibrationScales(0.5, 1.0, 2.0, 8.0),
        (0.5, 1.0, 2.0, 4.0, 8.0),
    )
    assert len(local) == 81 and len(set(local)) == 81
    assert {item.s_P0_att for item in local} == {0.5, 1.0, 2.0}
    assert {item.s_Qb for item in local} == {2.0, 4.0, 8.0}


def test_deterministic_selection_uses_consistency_then_scale_then_lexicographic() -> None:
    candidates = [
        _candidate(CalibrationScales(1, 1, 2, 1), 1.2, "a"),
        _candidate(CalibrationScales(1, 1, 1, 2), 1.1, "b"),
        _candidate(CalibrationScales(2, 1, 1, 1), 1.1, "c"),
    ]
    selected = deterministic_select(candidates, stage="settled")
    assert selected["candidate_id"] == "b"
    shuffled = list(reversed(candidates))
    assert deterministic_select(shuffled, stage="settled")["candidate_id"] == "b"


def test_guard_rejects_sensor_nis_and_accuracy_degradation() -> None:
    config = _config()
    baseline = _aggregate()
    candidate = _aggregate(attitude_rmse=0.02, bias_rmse=0.002, sensor_nis=1.5)
    guard = validation_guard(candidate, baseline, config)
    assert not guard["passed"]
    assert not guard["checks"]["mag_nis"]
    assert not guard["checks"]["attitude_accuracy"]
    assert not guard["checks"]["bias_accuracy"]


def test_all_one_calibrated_replay_is_exactly_f_base() -> None:
    config = _config()
    generated = generate_sensor_fusion(
        _generator_config(
            config,
            scenario=FusionScenarioCode.MAIN_FUSION_STATIONARY,
            master_seed=20261101,
            count=3,
        )
    )
    trajectory_id = int(generated.dataset.truth.trajectory_id[0])
    generator_config = _generator_config(
        config,
        scenario=FusionScenarioCode.MAIN_FUSION_STATIONARY,
        master_seed=20261101,
        count=3,
    )
    Q_c = base_process_covariance(generator_config)
    fixed = replay_fixed_policy(
        generated.dataset.events,
        trajectory_id,
        default_initial_state(),
        0.0,
        Q_c,
        F_BASE,
    )
    calibrated = replay_calibrated_fixed(
        generated.dataset.events, trajectory_id, Q_c, CalibrationScales()
    )
    for name in (
        "q_NB_history",
        "b_g_history",
        "P_history",
        "star_tracker_residual",
        "star_tracker_S",
        "magnetometer_residual",
        "magnetometer_S",
        "sun_residual",
        "sun_S",
    ):
        assert np.array_equal(getattr(fixed, name), getattr(calibrated, name))


def test_calibrated_fixed_api_has_no_truth_oracle_event_or_label() -> None:
    names = set(inspect.signature(replay_calibrated_fixed).parameters)
    assert not ({"truth", "oracle", "event_window", "hidden_label"} & names)


def test_search_does_not_call_confirmation_generator() -> None:
    source = Path("bench/experiments/p1_exit_covariance_closure.py").read_text(
        encoding="utf-8"
    )
    tree = ast.parse(source)
    search = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_search"
    )
    calls = {
        node.func.id
        for node in ast.walk(search)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "_ensure_confirmation_bundles" not in calls


def test_source_has_no_forbidden_numerical_or_neural_path() -> None:
    source = Path("bench/experiments/p1_exit_covariance_closure.py").read_text(
        encoding="utf-8"
    ).lower()
    forbidden = (
        "np.linalg.inv(",
        "numpy.linalg.inv(",
        "pinv(",
        "lstsq(",
        "np.clip(",
        "import torch",
        "from torch",
        "import tensorflow",
        "from tensorflow",
    )
    for token in forbidden:
        assert token not in source
