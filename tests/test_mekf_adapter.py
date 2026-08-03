from __future__ import annotations

import ast
import inspect
import json
import subprocess
from dataclasses import fields, replace
from pathlib import Path

import numpy as np
import pytest

from bench.estimators.mekf import MEKFState
from bench.metrics.mekf import (
    attitude_geodesic_error_rad,
    bias_error_summary,
    right_local_nees,
    star_tracker_nis,
)
from bench.models.mekf import (
    ADAPTER_ID,
    ADAPTER_VERSION,
    DatasetIdentity,
    MEKFEventReplayBridge,
    MEKFReplayArtifact,
)
from bench.tasks.generator.basilisk_unit_st import (
    BasiliskUnitSTConfig,
    generate_basilisk_unit_st,
)
from bench.tasks.generator.mekf_events import (
    MEKFDataset,
    MEKFEventTable,
    SensorCode,
    compute_semantic_hashes,
    load_event_dataset,
    replay_trajectory,
    save_event_dataset,
)
from bench.tasks.generator.unit_st_synthetic import (
    UnitSTSyntheticConfig,
    generate_unit_st,
)


PYTHON = "/home/dss-pc-05/.pyenv/versions/3.10.13/bin/python"
SOURCE = Path("bench/models/mekf.py")


@pytest.fixture(scope="module")
def synthetic_generated():
    return generate_unit_st(
        UnitSTSyntheticConfig(
            num_trajectories=4,
            duration_s=0.4,
            gyro_rate_hz=20,
            star_tracker_rate_hz=5,
            master_seed=510,
        )
    )


@pytest.fixture(scope="module")
def basilisk_generated():
    return generate_basilisk_unit_st(
        BasiliskUnitSTConfig(
            num_trajectories=4,
            duration_s=0.2,
            gyro_rate_hz=20,
            star_tracker_rate_hz=5,
            master_seed=520,
        )
    )


def _identity(generated) -> DatasetIdentity:
    return DatasetIdentity.from_verified(generated.manifest, generated.semantic_hashes)


def _initial_state(generated, trajectory_index: int = 0) -> MEKFState:
    truth = generated.dataset.truth
    start = int(truth.truth_offsets[trajectory_index])
    return MEKFState(
        q_NB=truth.q_true_NB[start],
        b_g=np.zeros(3, dtype=np.float64),
        P=np.eye(6, dtype=np.float64) * 1.0e-5,
    )


def _process_noise() -> np.ndarray:
    return np.diag(np.asarray([1.0e-10] * 3 + [1.0e-12] * 3, dtype=np.float64))


def _event_kwargs(table: MEKFEventTable) -> dict[str, np.ndarray]:
    return {field.name: np.array(getattr(table, field.name), copy=True) for field in fields(table)}


def _bridge_replay(generated, trajectory_index: int = 0) -> MEKFReplayArtifact:
    trajectory_id = int(generated.dataset.truth.trajectory_id[trajectory_index])
    identity = _identity(generated)
    return MEKFEventReplayBridge(expected_dataset_identity=identity).replay_events(
        generated.dataset.events,
        trajectory_id,
        _initial_state(generated, trajectory_index),
        0.0,
        _process_noise(),
        identity,
    )


def _direct_replay(generated, trajectory_index: int = 0):
    trajectory_id = int(generated.dataset.truth.trajectory_id[trajectory_index])
    return replay_trajectory(
        generated.dataset.events,
        trajectory_id,
        _initial_state(generated, trajectory_index),
        0.0,
        _process_noise(),
    )


def _assert_artifact_matches_direct(artifact: MEKFReplayArtifact, direct) -> None:
    assert artifact.trajectory_id == direct.trajectory_id
    assert artifact.processed_event_count == direct.processed_event_count
    assert np.array_equal(artifact.timestamp_s, direct.event_time_s)
    assert np.array_equal(artifact.event_order, direct.event_order)
    assert np.array_equal(artifact.sensor_code, direct.sensor_code)
    assert np.array_equal(artifact.q_hat_NB, direct.q_NB_history)
    assert np.array_equal(artifact.b_hat_rad_s, direct.b_g_history)
    assert np.array_equal(artifact.P, direct.P_history)
    assert np.array_equal(artifact.st_event_order, direct.star_tracker_event_order)
    assert np.array_equal(artifact.st_residual, direct.star_tracker_residual)
    assert np.array_equal(artifact.st_S, direct.star_tracker_S)
    assert np.array_equal(artifact.final_state.q_NB, direct.final_state.q_NB)
    assert np.array_equal(artifact.final_state.b_g, direct.final_state.b_g)
    assert np.array_equal(artifact.final_state.P, direct.final_state.P)


def _truth_join(generated, artifact: MEKFReplayArtifact) -> tuple[np.ndarray, np.ndarray]:
    truth = generated.dataset.truth
    trajectory_position = int(
        np.flatnonzero(truth.trajectory_id == np.int64(artifact.trajectory_id))[0]
    )
    start = int(truth.truth_offsets[trajectory_position])
    stop = int(truth.truth_offsets[trajectory_position + 1])
    truth_time = truth.truth_time_s[start:stop]
    locations = np.searchsorted(truth_time, artifact.timestamp_s)
    assert np.array_equal(truth_time[locations], artifact.timestamp_s)
    return truth.q_true_NB[start:stop][locations], truth.gyro_bias_rad_s[start:stop][locations]


def test_d1_01_import_boundary_is_lightweight_and_unregistered() -> None:
    code = """
import json
import sys
import bench.models.mekf
forbidden = ('Basilisk', 'torch', 'bench.runners', 'bench.models.registry', 'visualization')
print(json.dumps(sorted(name for name in sys.modules if any(token in name for token in forbidden))))
"""
    completed = subprocess.run(
        [PYTHON, "-c", code],
        check=True,
        capture_output=True,
        text=True,
        env={"PYTHONDONTWRITEBYTECODE": "1"},
    )
    assert json.loads(completed.stdout) == []
    registry_tree = ast.parse(Path("bench/models/registry.py").read_text(encoding="utf-8"))
    registered_literals = {
        node.value for node in ast.walk(registry_tree) if isinstance(node, ast.Constant)
    }
    assert ADAPTER_ID not in registered_literals


def test_d1_02_synthetic_direct_adapter_exact_equivalence(synthetic_generated) -> None:
    artifact = _bridge_replay(synthetic_generated)
    direct = _direct_replay(synthetic_generated)
    _assert_artifact_matches_direct(artifact, direct)


def test_d1_03_serialized_dataset_replays_exactly(tmp_path, synthetic_generated) -> None:
    target = tmp_path / "unit-st"
    save_event_dataset(target, synthetic_generated.dataset, synthetic_generated.manifest)
    loaded, manifest, hashes = load_event_dataset(
        target,
        expected_generator_id=synthetic_generated.manifest["generator_id"],
    )
    identity = DatasetIdentity.from_verified(manifest, hashes)
    trajectory_id = int(loaded.truth.trajectory_id[0])
    prior = _initial_state(synthetic_generated)
    artifact = MEKFEventReplayBridge(expected_dataset_identity=identity).replay_events(
        loaded.events,
        trajectory_id,
        prior,
        0.0,
        _process_noise(),
        identity,
    )
    direct = replay_trajectory(loaded.events, trajectory_id, prior, 0.0, _process_noise())
    _assert_artifact_matches_direct(artifact, direct)
    assert artifact.provenance.dataset_identity == identity


@pytest.mark.parametrize("source_name", ("synthetic", "basilisk"))
def test_d1_04_both_gate_b_generators_use_the_same_bridge(
    request: pytest.FixtureRequest,
    source_name: str,
) -> None:
    generated = request.getfixturevalue(f"{source_name}_generated")
    artifact = _bridge_replay(generated)
    _assert_artifact_matches_direct(artifact, _direct_replay(generated))
    assert artifact.provenance.generator_id == generated.manifest["generator_id"]


def test_d1_05_identity_is_preserved_and_mismatch_fails(synthetic_generated) -> None:
    identity = _identity(synthetic_generated)
    artifact = _bridge_replay(synthetic_generated)
    assert artifact.provenance.dataset_identity.as_dict() == identity.as_dict()
    mismatch = replace(identity, dataset_hash="0" * 64)
    bridge = MEKFEventReplayBridge(expected_dataset_identity=identity)
    with pytest.raises(ValueError, match="does not match"):
        bridge.replay_events(
            synthetic_generated.dataset.events,
            int(synthetic_generated.dataset.truth.trajectory_id[0]),
            _initial_state(synthetic_generated),
            0.0,
            _process_noise(),
            mismatch,
        )


def test_d1_06_bridge_instance_identity_does_not_change_data_or_numeric_output(
    synthetic_generated,
) -> None:
    identity = _identity(synthetic_generated)
    first = _bridge_replay(synthetic_generated)
    second = MEKFEventReplayBridge().replay_events(
        synthetic_generated.dataset.events,
        int(synthetic_generated.dataset.truth.trajectory_id[0]),
        _initial_state(synthetic_generated),
        0.0,
        _process_noise(),
        identity,
    )
    assert first.provenance.dataset_hash == second.provenance.dataset_hash
    assert first.provenance.adapter_id == second.provenance.adapter_id == ADAPTER_ID
    assert first.provenance.adapter_version == second.provenance.adapter_version == ADAPTER_VERSION
    assert np.array_equal(first.q_hat_NB, second.q_hat_NB)
    assert np.array_equal(first.b_hat_rad_s, second.b_hat_rad_s)
    assert np.array_equal(first.P, second.P)
    assert np.array_equal(first.st_residual, second.st_residual)
    assert np.array_equal(first.st_S, second.st_S)


def test_d1_07_public_estimator_api_is_truth_oracle_and_label_free() -> None:
    parameters = tuple(inspect.signature(MEKFEventReplayBridge.replay_events).parameters)
    assert parameters == (
        "self",
        "event_table",
        "trajectory_id",
        "initial_state",
        "initial_time_s",
        "Q_c",
        "dataset_identity",
    )
    forbidden = ("truth", "oracle", "label", "future", "metric")
    assert not any(token in name.lower() for name in parameters for token in forbidden)
    assert MEKFEventReplayBridge.__bases__ == (object,)
    assert not hasattr(MEKFEventReplayBridge, "predict")
    assert not hasattr(MEKFEventReplayBridge, "train")


def test_d1_08_bridge_does_not_mutate_or_regenerate_inputs(synthetic_generated) -> None:
    events = synthetic_generated.dataset.events
    event_before = {
        field.name: np.array(getattr(events, field.name), copy=True) for field in fields(events)
    }
    prior = _initial_state(synthetic_generated)
    prior_before = (prior.q_NB.copy(), prior.b_g.copy(), prior.P.copy())
    process_noise = _process_noise()
    process_before = process_noise.copy()
    identity = _identity(synthetic_generated)
    MEKFEventReplayBridge().replay_events(
        events,
        int(synthetic_generated.dataset.truth.trajectory_id[0]),
        prior,
        0.0,
        process_noise,
        identity,
    )
    for name, expected in event_before.items():
        assert np.array_equal(getattr(events, name), expected)
    assert np.array_equal(prior.q_NB, prior_before[0])
    assert np.array_equal(prior.b_g, prior_before[1])
    assert np.array_equal(prior.P, prior_before[2])
    assert np.array_equal(process_noise, process_before)


def test_d1_09_artifact_dtype_shape_readonly_counts_and_spd(synthetic_generated) -> None:
    artifact = _bridge_replay(synthetic_generated)
    expected = {
        "event_index": (np.dtype(np.int64), (artifact.processed_event_count,)),
        "event_order": (np.dtype(np.int64), (artifact.processed_event_count,)),
        "timestamp_s": (np.dtype(np.float64), (artifact.processed_event_count,)),
        "sensor_code": (np.dtype(np.int16), (artifact.processed_event_count,)),
        "q_hat_NB": (np.dtype(np.float64), (artifact.processed_event_count, 4)),
        "b_hat_rad_s": (np.dtype(np.float64), (artifact.processed_event_count, 3)),
        "P": (np.dtype(np.float64), (artifact.processed_event_count, 6, 6)),
        "st_event_index": (np.dtype(np.int64), (artifact.star_tracker_update_count,)),
        "st_event_order": (np.dtype(np.int64), (artifact.star_tracker_update_count,)),
        "st_timestamp_s": (np.dtype(np.float64), (artifact.star_tracker_update_count,)),
        "st_residual": (np.dtype(np.float64), (artifact.star_tracker_update_count, 3)),
        "st_S": (np.dtype(np.float64), (artifact.star_tracker_update_count, 3, 3)),
    }
    for name, (dtype, shape) in expected.items():
        value = getattr(artifact, name)
        assert value.dtype == dtype
        assert value.shape == shape
        assert not value.flags.writeable
        with pytest.raises(ValueError):
            value.flat[0] = value.flat[0]
    assert all(np.linalg.cholesky(matrix).shape == (6, 6) for matrix in artifact.P)
    assert all(np.linalg.cholesky(matrix).shape == (3, 3) for matrix in artifact.st_S)
    with pytest.raises(ValueError):
        artifact.final_state.q_NB[0] = 0.0


def test_d1_10_st_evidence_is_compact_and_matches_valid_updates(synthetic_generated) -> None:
    artifact = _bridge_replay(synthetic_generated)
    events = synthetic_generated.dataset.events
    rows = np.flatnonzero(events.trajectory_id == np.int64(artifact.trajectory_id))
    expected = int(
        np.count_nonzero(
            (events.sensor_code[rows] == np.int16(SensorCode.STAR_TRACKER))
            & events.valid[rows]
        )
    )
    assert artifact.star_tracker_update_count == expected
    assert artifact.st_residual.shape[0] == expected
    assert artifact.st_S.shape[0] == expected
    star_positions = np.flatnonzero(
        artifact.sensor_code == np.int16(SensorCode.STAR_TRACKER)
    )
    assert np.array_equal(artifact.st_event_index, artifact.event_index[star_positions])


def test_d1_11_separate_truth_join_gives_exact_gate_c_metric_evidence(
    synthetic_generated,
) -> None:
    artifact = _bridge_replay(synthetic_generated)
    direct = _direct_replay(synthetic_generated)
    q_true, b_true = _truth_join(synthetic_generated, artifact)
    trajectory_ids = np.full(
        artifact.processed_event_count, artifact.trajectory_id, dtype=np.int64
    )
    artifact_geodesic = attitude_geodesic_error_rad(artifact.q_hat_NB, q_true)
    direct_geodesic = attitude_geodesic_error_rad(direct.q_NB_history, q_true)
    assert np.array_equal(artifact_geodesic, direct_geodesic)
    artifact_bias = bias_error_summary(artifact.b_hat_rad_s, b_true)
    direct_bias = bias_error_summary(direct.b_g_history, b_true)
    assert np.array_equal(
        artifact_bias.per_axis_error_rad_s,
        direct_bias.per_axis_error_rad_s,
    )
    artifact_nees = right_local_nees(
        artifact.q_hat_NB,
        artifact.b_hat_rad_s,
        artifact.P,
        q_true,
        b_true,
        estimate_time_s=artifact.timestamp_s,
        covariance_time_s=artifact.timestamp_s,
        truth_time_s=artifact.timestamp_s,
        estimate_trajectory_id=trajectory_ids,
        covariance_trajectory_id=trajectory_ids,
        truth_trajectory_id=trajectory_ids,
    )
    direct_nees = right_local_nees(
        direct.q_NB_history,
        direct.b_g_history,
        direct.P_history,
        q_true,
        b_true,
        estimate_time_s=direct.event_time_s,
        covariance_time_s=direct.event_time_s,
        truth_time_s=direct.event_time_s,
        estimate_trajectory_id=trajectory_ids,
        covariance_trajectory_id=trajectory_ids,
        truth_trajectory_id=trajectory_ids,
    )
    assert np.array_equal(artifact_nees, direct_nees)
    artifact_nis = star_tracker_nis(
        artifact.st_residual,
        artifact.st_S,
        residual_time_s=artifact.st_timestamp_s,
        covariance_time_s=artifact.st_timestamp_s,
    )
    direct_nis = star_tracker_nis(
        direct.star_tracker_residual,
        direct.star_tracker_S,
        residual_time_s=artifact.st_timestamp_s,
        covariance_time_s=artifact.st_timestamp_s,
    )
    assert np.array_equal(artifact_nis, direct_nis)


def test_d1_12_raw_star_tracker_q_sign_has_identical_artifact_and_metrics(
    synthetic_generated,
) -> None:
    original_events = synthetic_generated.dataset.events
    kwargs = _event_kwargs(original_events)
    kwargs["star_tracker_q_NB"] *= -1.0
    negated_events = MEKFEventTable(**kwargs)
    negated_dataset = MEKFDataset(events=negated_events, truth=synthetic_generated.dataset.truth)
    negated_hashes = compute_semantic_hashes(negated_dataset, synthetic_generated.manifest)
    original_identity = _identity(synthetic_generated)
    negated_identity = DatasetIdentity.from_verified(synthetic_generated.manifest, negated_hashes)
    trajectory_id = int(synthetic_generated.dataset.truth.trajectory_id[0])
    prior = _initial_state(synthetic_generated)
    original = MEKFEventReplayBridge().replay_events(
        original_events, trajectory_id, prior, 0.0, _process_noise(), original_identity
    )
    negated = MEKFEventReplayBridge().replay_events(
        negated_events, trajectory_id, prior, 0.0, _process_noise(), negated_identity
    )
    for name in ("q_hat_NB", "b_hat_rad_s", "P", "st_residual", "st_S"):
        assert np.array_equal(getattr(original, name), getattr(negated, name))
    q_true, _ = _truth_join(synthetic_generated, original)
    assert np.array_equal(
        attitude_geodesic_error_rad(original.q_hat_NB, q_true),
        attitude_geodesic_error_rad(negated.q_hat_NB, q_true),
    )
    assert np.array_equal(
        star_tracker_nis(original.st_residual, original.st_S),
        star_tracker_nis(negated.st_residual, negated.st_S),
    )


def test_d1_13_invalid_identity_trajectory_time_count_and_index_fail_loud(
    synthetic_generated,
) -> None:
    identity = _identity(synthetic_generated)
    with pytest.raises(ValueError, match="schema_version"):
        replace(identity, schema_version="wrong")
    with pytest.raises(ValueError, match="SHA-256"):
        replace(identity, dataset_hash="ABC")
    bridge = MEKFEventReplayBridge()
    with pytest.raises(ValueError, match="trajectory_id"):
        bridge.replay_events(
            synthetic_generated.dataset.events,
            -1,
            _initial_state(synthetic_generated),
            0.0,
            _process_noise(),
            identity,
        )
    with pytest.raises(ValueError, match="strictly later"):
        bridge.replay_events(
            synthetic_generated.dataset.events,
            int(synthetic_generated.dataset.truth.trajectory_id[0]),
            _initial_state(synthetic_generated),
            1.0,
            _process_noise(),
            identity,
        )
    invalid_order = _event_kwargs(synthetic_generated.dataset.events)
    first_star = int(
        np.flatnonzero(
            invalid_order["sensor_code"] == np.int16(SensorCode.STAR_TRACKER)
        )[0]
    )
    preceding_gyro = first_star - 1
    invalid_order["event_order"][[preceding_gyro, first_star]] = invalid_order[
        "event_order"
    ][[first_star, preceding_gyro]]
    with pytest.raises(ValueError, match="sorted|gyro before"):
        MEKFEventTable(**invalid_order)
    artifact = _bridge_replay(synthetic_generated)
    with pytest.raises(ValueError, match="gyro_event_count"):
        replace(artifact, gyro_event_count=artifact.gyro_event_count + 1)
    bad_index = artifact.st_event_index.copy()
    bad_index[0] += 1
    with pytest.raises(ValueError, match="st_event_index"):
        replace(artifact, st_event_index=bad_index)


def test_d1_14_bridge_is_deterministic_frozen_and_has_no_training(
    synthetic_generated,
) -> None:
    bridge = MEKFEventReplayBridge()
    assert bridge.is_frozen is True
    assert bridge.supports_training is False
    first = _bridge_replay(synthetic_generated)
    second = _bridge_replay(synthetic_generated)
    _assert_artifact_matches_direct(first, _direct_replay(synthetic_generated))
    for name in ("q_hat_NB", "b_hat_rad_s", "P", "st_residual", "st_S"):
        assert np.array_equal(getattr(first, name), getattr(second, name))


def test_d1_15_frozen_replay_is_called_once_and_math_is_not_duplicated(
    monkeypatch: pytest.MonkeyPatch,
    synthetic_generated,
) -> None:
    import bench.models.mekf as adapter_module

    direct = _direct_replay(synthetic_generated)
    calls: list[tuple[object, ...]] = []

    def recorded_call(*args):
        calls.append(args)
        return direct

    monkeypatch.setattr(adapter_module, "replay_trajectory", recorded_call)
    artifact = _bridge_replay(synthetic_generated)
    assert len(calls) == 1
    _assert_artifact_matches_direct(artifact, direct)
    for artifact_name, replay_name in (
        ("q_hat_NB", "q_NB_history"),
        ("b_hat_rad_s", "b_g_history"),
        ("P", "P_history"),
        ("st_residual", "star_tracker_residual"),
        ("st_S", "star_tracker_S"),
    ):
        assert not np.shares_memory(
            getattr(artifact, artifact_name),
            getattr(direct, replay_name),
        )

    source = SOURCE.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported_names = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }
    assert "propagate_state" not in imported_names
    assert "star_tracker_update" not in imported_names
    assert "ModelAdapter" not in imported_names
    lowered = source.lower()
    assert "pinv(" not in lowered
    assert ".inv(" not in lowered
    assert "float32" not in lowered
    assert not any(
        isinstance(node, ast.Attribute) and node.attr == "truth" for node in ast.walk(tree)
    )


@pytest.mark.parametrize("seed", (601, 602, 603, 604, 605))
def test_d1_synthetic_seed_sweep_exact_equivalence(seed: int) -> None:
    generated = generate_unit_st(
        UnitSTSyntheticConfig(
            num_trajectories=3,
            duration_s=0.2,
            master_seed=seed,
        )
    )
    _assert_artifact_matches_direct(_bridge_replay(generated), _direct_replay(generated))


@pytest.mark.parametrize("seed", (701, 702, 703))
def test_d1_basilisk_seed_sweep_exact_equivalence(seed: int) -> None:
    generated = generate_basilisk_unit_st(
        BasiliskUnitSTConfig(
            num_trajectories=3,
            duration_s=0.2,
            gyro_rate_hz=20,
            star_tracker_rate_hz=5,
            master_seed=seed,
        )
    )
    _assert_artifact_matches_direct(_bridge_replay(generated), _direct_replay(generated))
