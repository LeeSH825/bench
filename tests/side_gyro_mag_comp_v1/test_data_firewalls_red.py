from __future__ import annotations

import copy
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from bench.estimators.mekf import MEKFState, quat_exp, quat_to_dcm
from bench.metrics.mekf import attitude_geodesic_error_rad
from bench.side_gyro_mag_comp_v1.data import (
    FORBIDDEN_DEPLOYABLE_KEYS,
    SensorTrajectory,
    SplitFirewallRecord,
    WholeTrajectorySplit,
    assert_same_realization,
    freeze_train_normalization,
    generate_dataset,
    strip_runtime_trajectory,
    strip_runtime_normalization,
    validate_deployable_namespace,
    validate_firewall,
)
from bench.side_gyro_mag_comp_v1.model import SideEstimator
from bench.side_gyro_mag_comp_v1.study import (
    deployable_replay,
    fixed_derangement,
    load_config,
    n3s_replay_namespace,
    protected_replay_hashes,
    verify_n3s_bridge,
    state_dict_digest,
)


CONFIG = Path("bench/configs/side_gyro_mag_comp_v1.yaml")


@pytest.fixture(scope="module")
def dataset():
    return generate_dataset(population={"train": 4, "validation": 2, "test": 4}, sequence_length=8)


def test_sc_identity_compensation_red(dataset) -> None:
    tid = next(item for item in dataset.sensor if dataset.sensor[item].regime == "R0")
    trajectory, target = dataset.sensor[tid], dataset.oracle[tid]
    gyro = np.stack([event.measurement_S for event in trajectory.events if event.sensor == "gyro"])
    mag = np.stack([event.measurement_S for event in trajectory.events if event.sensor == "magnetometer"])
    assert np.array_equal(gyro, target.gyro_target_B_rad_s)
    assert np.array_equal(mag, target.mag_target_B)


def test_sc_noise_free_oracle_exact_correction_red(dataset) -> None:
    for tid, trajectory in dataset.sensor.items():
        sidecar = dataset.oracle[tid]
        calibration = sidecar.calibration
        gyro = np.stack([event.measurement_S for event in trajectory.events if event.sensor == "gyro"])
        mag = np.stack([event.measurement_S for event in trajectory.events if event.sensor == "magnetometer"])
        recovered_g = np.linalg.solve(calibration.A_g, (gyro - calibration.c_g).T).T
        recovered_m = (calibration.C_BSm @ np.linalg.solve(calibration.A_m, (mag - calibration.b_m).T)).T
        assert np.allclose(recovered_g, sidecar.gyro_target_B_rad_s, atol=1e-14)
        assert np.allclose(recovered_m, sidecar.mag_target_B, atol=1e-14)


def test_sc_wrong_sign_frame_fixture_red() -> None:
    q = quat_exp(np.array([.3, -.2, .1])); reference = np.array([.3, -.2, .93])
    correct = quat_to_dcm(q).T @ reference
    wrong = quat_to_dcm(q) @ reference
    assert np.linalg.norm(correct - wrong) > .1
    innovation = np.array([.01, -.02, .03])
    assert not np.array_equal(innovation, -innovation)


def test_sc_q_sign_metric_invariance_red() -> None:
    q = quat_exp(np.array([.2, -.1, .3]))
    estimates = np.stack((q, -q)).astype(np.float64)
    truths = np.stack((quat_exp(np.array([.21, -.1, .3])), -quat_exp(np.array([.21, -.1, .3])))).astype(np.float64)
    values = attitude_geodesic_error_rad(estimates, truths)
    assert values[0] == pytest.approx(values[1], abs=1e-14)


def test_sc_future_sample_injection_rejection_red() -> None:
    from bench.side_gyro_mag_comp_v1.model import GyroEncoder
    encoder = GyroEncoder(); encoder.forward_step(np.ones(3), .2, True)
    with pytest.raises(ValueError, match="future/reordered"):
        encoder.forward_step(np.ones(3), .1, True)


def test_sc_deployable_namespace_leakage_rejected_red() -> None:
    for symbol in ("A_m", "b_m", "C_BSm", "A_g", "c_g", "C_SgB"):
        with pytest.raises(ValueError):
            validate_deployable_namespace({symbol: np.eye(3)})
    for alias in ("mag_calibration_matrix", "gyroCalibration", "inverse_A_m", "oracle_context", "truth_state", "future_packet"):
        with pytest.raises(ValueError):
            validate_deployable_namespace({alias: np.eye(3)})


def test_sc_pairing_split_firewall_red(dataset) -> None:
    normalization = freeze_train_normalization(dataset)
    runtime_normalization = strip_runtime_normalization(normalization)
    assert set(normalization.source_trajectory_ids) == set(dataset.split.train_ids)
    good = SplitFirewallRecord(
        normalization.source_trajectory_ids, dataset.split.train_ids,
        dataset.split.validation_ids, dataset.split.validation_ids, dataset.split.train_ids,
    )
    validate_firewall(dataset.split, good)
    contaminated = copy.deepcopy(good)
    object.__setattr__(contaminated, "normalization_ids", good.normalization_ids + (dataset.split.test_ids[0],))
    with pytest.raises(ValueError, match="training-only|test trajectory"):
        validate_firewall(dataset.split, contaminated)
    with pytest.raises(ValueError):
        WholeTrajectorySplit((1, 2), (2, 3), (4,), {1:"R0",2:"R0",3:"R0",4:"R0"},
                             {1:"a",2:"b",3:"c",4:"d"}, 271828, 314159)


def test_sc_split_firewall_requires_exact_nonempty_unique_source_sets_red(dataset) -> None:
    good = SplitFirewallRecord(
        dataset.split.train_ids, dataset.split.train_ids,
        dataset.split.validation_ids, dataset.split.validation_ids, dataset.split.train_ids,
    )
    train_fields = ("normalization_ids", "training_loss_ids", "threshold_setting_ids")
    validation_fields = ("early_stopping_ids", "checkpoint_selection_ids")
    for field_name in train_fields + validation_fields:
        values = getattr(good, field_name)
        for invalid, message in (
            (tuple(), "nonempty"),
            (values[:-1], "complete"),
            (values + (values[0],), "duplicate"),
            (values + (dataset.split.test_ids[0],), "test trajectory"),
        ):
            with pytest.raises(ValueError, match=message):
                validate_firewall(dataset.split, replace(good, **{field_name: invalid}))


def test_sc_different_raw_realization_across_variants_rejected_red(dataset) -> None:
    first, second = list(dataset.sensor.values())[:2]
    with pytest.raises(ValueError, match="same trajectory and raw realization"):
        assert_same_realization([first, second])


def test_sc_frozen_gate_population_red(tmp_path: Path) -> None:
    config = load_config(CONFIG)
    assert config["training"]["seeds"] == [31001, 31002, 31003]
    assert config["evaluation"]["bootstrap_resamples"] == 10000
    altered = copy.deepcopy(config); altered["evaluation"]["bootstrap_seed"] = 9
    path = tmp_path / "altered.yaml"
    import yaml
    path.write_text(yaml.safe_dump(altered))
    with pytest.raises(ValueError, match="frozen field"):
        load_config(path)


def test_sc_n3s_single_intervention_red(dataset) -> None:
    ids = [item for item in dataset.split.test_ids if dataset.sensor[item].regime == "R3"]
    first = fixed_derangement(ids, regime="R3", training_seed=31001)
    second = fixed_derangement(ids, regime="R3", training_seed=31001)
    assert first == second and all(key != value for key, value in first.items())
    # Whole-sequence/time-invariant association: one source ID per target ID.
    for target in ids:
        assert isinstance(first[target], int)
    with pytest.raises(ValueError):
        fixed_derangement(ids[:1], regime="R3", training_seed=31001)
    normalization = freeze_train_normalization(dataset)
    runtime_normalization = strip_runtime_normalization(normalization)
    estimator = SideEstimator("learned", feature_enabled=True)
    # Make the feature path nontrivial without training or test-driven tuning.
    with torch.no_grad():
        estimator.backbone.gyro_film.weight[0, 0] = .05
        estimator.backbone.mag_film.weight[0, 0] = .05
    checkpoint = copy.deepcopy(estimator.state_dict())
    target_id = ids[0]
    baseline_estimator = SideEstimator("learned", feature_enabled=True)
    baseline_estimator.load_state_dict(checkpoint, strict=True)
    runtime_by_id = {item: strip_runtime_trajectory(dataset.sensor[item]) for item in ids}
    baseline = deployable_replay(
        runtime_by_id[target_id], baseline_estimator, runtime_normalization,
        dataset.m_model_N_onboard, variant="N3",
    )
    shuffled, evidence = n3s_replay_namespace(
        runtime_by_id, tuple(ids), 3, dataset.m_model_N_onboard,
        target_id, 31001, checkpoint, "checkpoint-file-digest", runtime_normalization,
    )
    assert evidence["n3_state_dict_sha256"] == state_dict_digest(checkpoint)
    assert evidence["fixed_point_count"] == 0 and evidence["source_trajectory_id"] != target_id
    assert np.array_equal(shuffled.timestamp_s, baseline.timestamp_s)
    assert np.array_equal(shuffled.corrected_gyro_B, baseline.corrected_gyro_B)
    assert np.array_equal(shuffled.corrected_mag_B, baseline.corrected_mag_B)
    assert shuffled.realization_id == baseline.realization_id
    assert not np.array_equal(shuffled.gyro_feature, baseline.gyro_feature)
    assert not np.array_equal(shuffled.mag_feature, baseline.mag_feature)
    baseline_hashes = protected_replay_hashes(runtime_by_id[target_id], baseline)
    shuffled_hashes = protected_replay_hashes(runtime_by_id[target_id], shuffled)
    verify_n3s_bridge(
        baseline_hashes, shuffled_hashes, evidence,
        n3s_recurrent_owner_token=shuffled.recurrent_history_owner_token,
        n3s_recurrent_history_sha256=shuffled.recurrent_history_provenance_sha256,
    )
    assert baseline.initial_state_sha256 == shuffled.initial_state_sha256
    assert baseline.recurrent_history_provenance_sha256 != shuffled.recurrent_history_provenance_sha256
    assert evidence["source_recurrent_history_sha256"] != shuffled.recurrent_history_provenance_sha256

    source_history_substitution = dict(evidence)
    source_history_substitution["n3s_recurrent_history_sha256"] = evidence["source_recurrent_history_sha256"]
    with pytest.raises(ValueError, match="emitted recurrent lineage"):
        verify_n3s_bridge(
            baseline_hashes, shuffled_hashes, source_history_substitution,
            n3s_recurrent_owner_token=shuffled.recurrent_history_owner_token,
            n3s_recurrent_history_sha256=shuffled.recurrent_history_provenance_sha256,
        )

    altered_initial_state = MEKFState(
        q_NB=np.array([1., 0., 0., 0.]),
        b_g=np.array([1e-4, 0., 0.]),
        P=np.diag(np.r_[np.full(3, 2e-2), np.full(3, 1e-5)]),
    )
    altered, altered_evidence = n3s_replay_namespace(
        runtime_by_id, tuple(ids), 3, dataset.m_model_N_onboard,
        target_id, 31001, checkpoint, "checkpoint-file-digest", runtime_normalization,
        initial_state=altered_initial_state,
    )
    assert altered.initial_state_sha256 != baseline.initial_state_sha256
    with pytest.raises(ValueError, match="protected"):
        verify_n3s_bridge(
            baseline_hashes, protected_replay_hashes(runtime_by_id[target_id], altered), altered_evidence,
            n3s_recurrent_owner_token=altered.recurrent_history_owner_token,
            n3s_recurrent_history_sha256=altered.recurrent_history_provenance_sha256,
        )

    with pytest.raises(ValueError, match="source history"):
        verify_n3s_bridge(
            baseline_hashes, shuffled_hashes, evidence,
            n3s_recurrent_owner_token=evidence["source_recurrent_owner_token"],
            n3s_recurrent_history_sha256=shuffled.recurrent_history_provenance_sha256,
        )
    for key in baseline_hashes:
        mutation = dict(shuffled_hashes); mutation[key] = "mutated"
        with pytest.raises(ValueError):
            verify_n3s_bridge(
                baseline_hashes, mutation, evidence,
                n3s_recurrent_owner_token=shuffled.recurrent_history_owner_token,
                n3s_recurrent_history_sha256=shuffled.recurrent_history_provenance_sha256,
            )
    for key in ("n3_checkpoint_file_sha256", "n3_state_dict_sha256"):
        mutation = dict(evidence)
        counterpart = "n3s_checkpoint_file_sha256" if "file" in key else "n3s_state_dict_sha256"
        mutation[counterpart] = "mutated"
        with pytest.raises(ValueError):
            verify_n3s_bridge(
                baseline_hashes, shuffled_hashes, mutation,
                n3s_recurrent_owner_token=shuffled.recurrent_history_owner_token,
                n3s_recurrent_history_sha256=shuffled.recurrent_history_provenance_sha256,
            )


def test_sc_nominal_no_op_red(dataset) -> None:
    tid = next(item for item in dataset.sensor if dataset.sensor[item].regime == "R0")
    trajectory, sidecar = dataset.sensor[tid], dataset.oracle[tid]
    raw_g = np.stack([item.measurement_S for item in trajectory.events if item.sensor == "gyro"])
    raw_m = np.stack([item.measurement_S for item in trajectory.events if item.sensor == "magnetometer"])
    assert np.array_equal(raw_g, sidecar.gyro_target_B_rad_s)
    assert np.array_equal(raw_m, sidecar.mag_target_B)


def test_sc_weak_axis_nonempty_population_red(dataset) -> None:
    for regime in ("R0", "R1", "R2", "R3", "R4"):
        ids = [item for item in dataset.split.test_ids if dataset.sensor[item].regime == regime]
        assert ids
        for tid in ids:
            h = dataset.oracle[tid].mag_target_B
            assert h.shape[0] > 0 and np.all(np.linalg.norm(h, axis=1) > 0)


def test_sc_three_axis_excitation_certificate_red(dataset) -> None:
    for truth in dataset.truth.values():
        centered = truth.omega_true_B_rad_s - truth.omega_true_B_rad_s.mean(0)
        certificate = np.linalg.eigvalsh(centered.T @ centered / centered.shape[0])[0]
        assert certificate >= 1e-5
    rank_deficient_adversary = np.tile(np.array([[.01, .0, .0]]), (8, 1))
    centered = rank_deficient_adversary - rank_deficient_adversary.mean(0)
    assert np.linalg.eigvalsh(centered.T @ centered / 8)[0] < 1e-5
