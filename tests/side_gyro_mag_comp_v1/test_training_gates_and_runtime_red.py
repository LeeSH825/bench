from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
import torch

from bench.side_gyro_mag_comp_v1.data import (
    RuntimeSensorPacket,
    RuntimeTrajectoryBatch,
    WholeTrajectorySplit,
    freeze_train_normalization,
    generate_dataset,
    strip_runtime_trajectory,
    strip_runtime_normalization,
    validate_deployable_namespace,
)
from bench.side_gyro_mag_comp_v1.model import SideEstimator
from bench.side_gyro_mag_comp_v1.study import (
    differentiable_trajectory_unroll,
    deployable_replay,
    evaluate_fractional_improvement_gate,
    evaluate_g0_gate,
    evaluate_g1_gate,
    evaluate_g2_gate,
    evaluate_g3_gate,
    evaluate_g4_gate,
    load_config,
    paired_cluster_bootstrap_ci,
    select_earliest_minimum_attitude_epoch,
    weak_observable_metrics,
)


CONFIG = Path("bench/configs/side_gyro_mag_comp_v1.yaml")


def test_sc_checkpoint_selection_attitude_ranking_disagreement_red() -> None:
    sensor_auxiliary_loss = [0.1, 0.2]
    validation_attitude_rmse = [0.3, 0.2]
    assert int(np.argmin(sensor_auxiliary_loss)) + 1 == 1
    assert select_earliest_minimum_attitude_epoch(validation_attitude_rmse) == 2
    assert select_earliest_minimum_attitude_epoch([0.2, 0.2]) == 1


def test_sc_differentiable_right_local_unroll_uses_all_loss_weights_red() -> None:
    dataset = generate_dataset(population={"train": 1, "validation": 1, "test": 2}, sequence_length=8)
    trajectory_id = next(item for item in dataset.split.train_ids if dataset.sensor[item].regime == "R3")
    config = load_config(CONFIG)
    estimator = SideEstimator("learned", feature_enabled=True)
    total, attitude = differentiable_trajectory_unroll(
        estimator, dataset, trajectory_id, "N3", config["training"]["loss_weights"],
    )
    assert total.requires_grad and attitude.requires_grad and torch.isfinite(total + attitude)
    total.backward()
    assert estimator.backbone.g1_head.bias.grad is not None
    for name in config["training"]["loss_weights"]:
        estimator = SideEstimator("learned", feature_enabled=True)
        changed = dict(config["training"]["loss_weights"]); changed[name] = 0.0
        mutated, _ = differentiable_trajectory_unroll(estimator, dataset, trajectory_id, "N3", changed)
        assert not torch.isclose(total.detach(), mutated.detach(), rtol=0.0, atol=1e-18), name


def test_sc_runtime_packet_allowlist_recursive_red() -> None:
    dataset = generate_dataset(population={"train": 1, "validation": 1, "test": 2}, sequence_length=8)
    runtime = strip_runtime_trajectory(next(iter(dataset.sensor.values())))
    validate_deployable_namespace(runtime)

    @dataclass(frozen=True)
    class BadPacket:
        timestamp_s: float
        regime: str

    with pytest.raises(ValueError, match="regime"):
        validate_deployable_namespace(BadPacket(0.1, "R0"))
    for key in ("A_m.weight", "b_m_buffer", "C_BSm", "A_g.weight", "c_g", "C_SgB"):
        with pytest.raises(ValueError):
            validate_deployable_namespace({"state_dict": {key: torch.zeros(1)}})


def test_sc_duplicate_split_and_same_label_different_bytes_red() -> None:
    dataset = generate_dataset(population={"train": 1, "validation": 1, "test": 2}, sequence_length=8)
    split = dataset.split
    with pytest.raises(ValueError, match="duplicate"):
        WholeTrajectorySplit(
            split.train_ids + (split.train_ids[0],), split.validation_ids, split.test_ids,
            split.regime_by_id, split.stream_namespace_by_id,
            split.data_generation_seed, split.split_seed,
        )
    runtime = strip_runtime_trajectory(next(iter(dataset.sensor.values())))
    packet = runtime.packets[0]
    altered_packet = RuntimeSensorPacket(
        packet.timestamp_s, packet.event_order, packet.sensor,
        packet.measurement_S + np.array([1e-6, 0., 0.]), packet.valid,
    )
    with pytest.raises(ValueError, match="digest"):
        RuntimeTrajectoryBatch(
            runtime.trajectory_id, runtime.realization_sha256,
            (altered_packet,) + runtime.packets[1:],
        )


def test_sc_g0_g1_g2_g4_boundary_arithmetic_red() -> None:
    reference = np.ones((12, 3))
    assert evaluate_g0_gate(np.full((12, 3), .899), reference)["passed"]
    assert not evaluate_g0_gate(np.full((12, 3), .901), reference)["passed"]
    two_of_three = np.tile(np.array([.8, .8, 1.1]), (12, 1))
    assert evaluate_g2_gate(two_of_three, reference)["passed"]
    assert not evaluate_g2_gate(np.tile(np.array([.8, 1.1, 1.1]), (12, 1)), reference)["passed"]
    assert evaluate_g1_gate(
        np.full((12, 3), .9), reference, np.full((12, 3), .9), reference,
        np.full((12, 3), .9), reference, two_of_three, reference,
    )["passed"]
    assert not evaluate_g1_gate(
        reference, reference, np.full((12, 3), .9), reference,
        np.full((12, 3), .9), reference, two_of_three, reference,
    )["passed"]
    zeros = np.zeros((12, 3), dtype=bool)
    assert evaluate_g4_gate(np.full((12, 3), 1.029), reference, zeros, zeros)["passed"]
    assert not evaluate_g4_gate(np.full((12, 3), 1.031), reference, zeros, zeros)["passed"]
    added = zeros.copy(); added[0, 0] = True
    assert not evaluate_g4_gate(np.full((12, 3), 1.0), reference, added, zeros)["passed"]


def test_sc_g3_repaired_sc01_predicate_boundary_red() -> None:
    n2 = np.ones((20, 3)); n3 = np.full((20, 3), .9)
    assert evaluate_g3_gate(n2, n3, np.full((20, 3), 1.01))["decision"] == "PASS"
    assert evaluate_g3_gate(n2, n3, np.full((20, 3), .90))["decision"] == "FAIL"
    crossing = np.full((20, 3), .95)
    crossing[:10] += .1; crossing[10:] -= .1
    assert evaluate_g3_gate(n2, n3, crossing)["decision"] == "INCONCLUSIVE_UNDERPOWERED"
    config = load_config(CONFIG)
    assert config["gates"]["G3"]["contrast"] == "N3S-0.5*N2-0.5*N3"
    assert config["gates"]["G3"]["old_disjunct_allowed"] is False
    assert config["gates"]["prompt05_g3_status"] == "superseded_by_repaired_audited_SC_01_only"


def test_sc_cluster_bootstrap_resamples_heterogeneous_trajectory_ids_red() -> None:
    contrast = np.array([
        [10., -2., -8.], [-9., 7., 2.], [6., 1., -7.], [-4., -3., 7.],
        [8., -6., -2.], [-7., -1., 8.], [3., -9., 6.], [-2., 5., -3.],
    ])
    reference = np.zeros_like(contrast)
    clustered = paired_cluster_bootstrap_ci(contrast, reference, resamples=2000, seed=45173)
    assert clustered == paired_cluster_bootstrap_ci(
        contrast, reference, resamples=2000, seed=45173,
    )
    assert clustered == pytest.approx((0.0, 0.0), abs=1e-15)

    rng = np.random.default_rng(45173)
    flat = contrast.reshape(-1)
    flat_stats = np.mean(flat[rng.integers(0, flat.size, size=(2000, flat.size))], axis=1)
    independent_cell_ci = tuple(np.percentile(flat_stats, [2.5, 97.5]))

    rng = np.random.default_rng(45173)
    per_seed_stats = np.empty(2000)
    for draw in range(2000):
        independently_sampled_columns = [
            contrast[rng.integers(0, contrast.shape[0], size=contrast.shape[0]), seed]
            for seed in range(contrast.shape[1])
        ]
        per_seed_stats[draw] = np.mean(np.stack(independently_sampled_columns, axis=1))
    independent_per_seed_ci = tuple(np.percentile(per_seed_stats, [2.5, 97.5]))
    assert not np.allclose(clustered, independent_cell_ci, rtol=0.0, atol=1e-12)
    assert not np.allclose(clustered, independent_per_seed_ci, rtol=0.0, atol=1e-12)


def test_sc_actual_weak_and_observable_metrics_nonempty_red() -> None:
    dataset = generate_dataset(population={"train": 1, "validation": 1, "test": 2}, sequence_length=8)
    normalization = freeze_train_normalization(dataset)
    runtime_normalization = strip_runtime_normalization(normalization)
    trajectory_id = dataset.split.test_ids[0]
    runtime = strip_runtime_trajectory(dataset.sensor[trajectory_id])
    replay = deployable_replay(
        runtime, SideEstimator("raw", feature_enabled=False), runtime_normalization,
        dataset.m_model_N_onboard, variant="N0",
    )
    metrics = weak_observable_metrics(
        replay, dataset.truth[trajectory_id].q_true_NB,
        dataset.truth[trajectory_id].residual_bias_B_rad_s, dataset.m_model_N_onboard,
    )
    assert metrics["weak_axis_count"] == 8 and metrics["observable_plane_count"] == 8
    assert np.isfinite(metrics["weak_axis_rms_rad"])
    assert np.isfinite(metrics["observable_plane_rms_rad"])
    with pytest.raises(ValueError, match="nonempty"):
        weak_observable_metrics(
            copy.deepcopy(replay).__class__(
                replay.trajectory_id, replay.variant, replay.realization_id,
                replay.timestamp_s[:0], replay.q_hat_NB[:0], replay.b_hat_B_rad_s[:0],
                replay.corrected_gyro_B[:0], replay.corrected_mag_B[:0],
                replay.gyro_feature[:0], replay.mag_feature[:0], tuple(),
                replay.initial_state_sha256, replay.recurrent_history_owner_token,
                replay.recurrent_history_provenance_sha256,
            ),
            dataset.truth[trajectory_id].q_true_NB[:0],
            dataset.truth[trajectory_id].residual_bias_B_rad_s[:0], dataset.m_model_N_onboard,
        )
