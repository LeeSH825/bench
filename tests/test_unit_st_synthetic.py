from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from bench.estimators.mekf import quat_exp, quat_multiply, quat_normalize
from bench.tasks.generator.mekf_events import (
    SensorCode,
    select_trajectories,
    split_trajectory_ids,
)
from bench.tasks.generator.unit_st_synthetic import UnitSTSyntheticConfig, generate_unit_st


def _config(**kwargs) -> UnitSTSyntheticConfig:
    return UnitSTSyntheticConfig(
        num_trajectories=8,
        duration_s=0.8,
        gyro_rate_hz=20,
        star_tracker_rate_hz=5,
        master_seed=731,
        **kwargs,
    )


def test_same_seed_and_config_have_identical_hashes_and_ids() -> None:
    first = generate_unit_st(_config())
    second = generate_unit_st(_config())
    assert first.semantic_hashes == second.semantic_hashes
    assert first.dataset.truth.trajectory_id.dtype == np.int64
    assert np.array_equal(first.dataset.truth.trajectory_id, second.dataset.truth.trajectory_id)
    assert np.unique(first.dataset.truth.trajectory_id).size == 8


def test_sensor_seed_isolation_preserves_truth_and_changes_sensor_hash() -> None:
    base = generate_unit_st(_config())
    changed = generate_unit_st(
        replace(_config(), gyro_noise_seed_namespace="gyro-noise-alternate")
    )
    assert changed.semantic_hashes.truth_hash == base.semantic_hashes.truth_hash
    assert changed.semantic_hashes.sensor_payload_hash != base.semantic_hashes.sensor_payload_hash


def test_truth_seed_change_changes_truth_hash() -> None:
    base = generate_unit_st(_config())
    changed = generate_unit_st(replace(_config(), truth_seed_namespace="truth-alternate"))
    assert changed.semantic_hashes.truth_hash != base.semantic_hashes.truth_hash


def test_sign_seed_changes_only_quaternion_representation_when_it_changes_raw_hash() -> None:
    base = generate_unit_st(_config())
    changed = generate_unit_st(
        replace(_config(), star_tracker_sign_seed_namespace="star-tracker-sign-alternate")
    )
    assert np.array_equal(
        base.dataset.events.gyro_omega_rad_s, changed.dataset.events.gyro_omega_rad_s
    )
    assert np.allclose(
        np.abs(np.sum(base.dataset.events.star_tracker_q_NB * changed.dataset.events.star_tracker_q_NB, axis=1)),
        1.0,
        rtol=0.0,
        atol=2.0e-15,
    )
    assert changed.semantic_hashes.sensor_payload_hash != base.semantic_hashes.sensor_payload_hash


def test_schedule_rate_counts_st_subset_and_zero_latency() -> None:
    config = _config()
    generated = generate_unit_st(config)
    events = generated.dataset.events
    gyro_count = config.num_trajectories * int(config.duration_s * config.gyro_rate_hz)
    star_count = config.num_trajectories * int(config.duration_s * config.star_tracker_rate_hz)
    assert np.count_nonzero(events.sensor_code == int(SensorCode.GYRO)) == gyro_count
    assert np.count_nonzero(events.sensor_code == int(SensorCode.STAR_TRACKER)) == star_count
    assert np.array_equal(events.arrival_time_s, events.measurement_time_s)
    for trajectory_id in generated.dataset.truth.trajectory_id:
        rows = events.trajectory_id == trajectory_id
        gyro_times = set(events.measurement_time_s[rows & (events.sensor_code == 1)])
        star_times = set(events.measurement_time_s[rows & (events.sensor_code == 2)])
        assert star_times < gyro_times
        assert min(gyro_times) > 0.0


def test_gyro_equation_sign_and_units_are_locked() -> None:
    config = _config()
    generated = generate_unit_st(config)
    truth = generated.dataset.truth
    events = generated.dataset.events
    trajectory_id = int(truth.trajectory_id[0])
    truth_start = int(truth.truth_offsets[0])
    omega = truth.omega_true_rad_s[truth_start]
    bias = truth.gyro_bias_rad_s[truth_start]
    seed = generated.manifest["derived_seeds"]["per_trajectory"][str(trajectory_id)]["gyro_noise"]
    expected_noise = np.random.default_rng(seed).normal(
        0.0, config.gyro_noise_std_rad_s, size=3
    )
    row = np.flatnonzero(
        (events.trajectory_id == trajectory_id) & (events.sensor_code == int(SensorCode.GYRO))
    )[0]
    measured = events.gyro_omega_rad_s[int(events.payload_index[row])]
    assert np.array_equal(measured, omega + bias + expected_noise)


def test_star_tracker_right_local_noise_construction_is_locked() -> None:
    config = _config()
    generated = generate_unit_st(config)
    truth = generated.dataset.truth
    events = generated.dataset.events
    trajectory_id = int(truth.trajectory_id[0])
    seed = generated.manifest["derived_seeds"]["per_trajectory"][str(trajectory_id)][
        "star_tracker_noise"
    ]
    noise = np.random.default_rng(seed).normal(
        0.0, config.star_tracker_noise_std_rad, size=3
    )
    step = config.gyro_rate_hz // config.star_tracker_rate_hz
    q_true = truth.q_true_NB[int(truth.truth_offsets[0]) + step]
    expected = quat_normalize(quat_multiply(q_true, quat_exp(noise)))
    row = np.flatnonzero(
        (events.trajectory_id == trajectory_id)
        & (events.sensor_code == int(SensorCode.STAR_TRACKER))
    )[0]
    measured = events.star_tracker_q_NB[int(events.payload_index[row])]
    assert abs(float(np.dot(expected, measured))) == pytest.approx(1.0, abs=2.0e-15)


def test_representative_metadata_is_complete() -> None:
    generated = generate_unit_st(_config())
    manifest = generated.manifest
    assert manifest["schema_version"] == "p1a-mekf-events-v1"
    assert manifest["generator_id"] == "synthetic-unit-st-v1"
    assert manifest["zero_latency"] is True
    assert manifest["same_timestamp_order"] == ["gyro", "star_tracker"]
    assert set(manifest["software_versions"]) == {"python", "numpy", "scipy"}
    assert len(manifest["source_fingerprints"]) == 3
    assert manifest["generator_config"]["master_seed"] == 731


def test_whole_trajectory_split_is_disjoint_complete_deterministic_and_order_independent() -> None:
    generated = generate_unit_st(_config())
    ids = generated.dataset.truth.trajectory_id
    split = generated.trajectory_split
    groups = [set(map(int, split.train_ids)), set(map(int, split.val_ids)), set(map(int, split.test_ids))]
    assert not groups[0] & groups[1]
    assert not groups[0] & groups[2]
    assert not groups[1] & groups[2]
    assert set.union(*groups) == set(map(int, ids))
    repeated = split_trajectory_ids(ids[::-1], split_seed=split.split_seed)
    assert set(repeated.train_ids) == set(split.train_ids)
    assert set(repeated.val_ids) == set(split.val_ids)
    assert set(repeated.test_ids) == set(split.test_ids)

    selected = select_trajectories(generated.dataset, split.val_ids)
    assert set(map(int, selected.truth.trajectory_id)) == groups[1]
    assert set(map(int, np.unique(selected.events.trajectory_id))) == groups[1]
    assert selected.events.gyro_omega_rad_s.shape[0] == np.count_nonzero(
        selected.events.sensor_code == int(SensorCode.GYRO)
    )
    assert selected.events.star_tracker_q_NB.shape[0] == np.count_nonzero(
        selected.events.sensor_code == int(SensorCode.STAR_TRACKER)
    )


@pytest.mark.parametrize(
    "ids,kwargs",
    [
        ([1, 1, 2], {}),
        ([1, 2], {}),
        ([1, 2, 3], {"train_fraction": 0.5, "val_fraction": 0.5, "test_fraction": 0.5}),
        ([1, 2, 3], {"train_fraction": 1.0, "val_fraction": 0.0, "test_fraction": 0.0}),
    ],
)
def test_split_rejects_duplicates_too_few_and_invalid_fractions(ids, kwargs) -> None:
    with pytest.raises((TypeError, ValueError)):
        split_trajectory_ids(ids, split_seed=1, **kwargs)


def test_different_split_seed_changes_split_without_changing_data_hashes() -> None:
    base = generate_unit_st(_config())
    changed = generate_unit_st(replace(_config(), split_seed_namespace="split-alternate"))
    assert base.semantic_hashes.truth_hash == changed.semantic_hashes.truth_hash
    assert base.semantic_hashes.sensor_payload_hash == changed.semantic_hashes.sensor_payload_hash
    assert base.semantic_hashes.event_order_hash == changed.semantic_hashes.event_order_hash
    assert base.semantic_hashes.dataset_hash == changed.semantic_hashes.dataset_hash
    assert not np.array_equal(base.trajectory_split.train_ids, changed.trajectory_split.train_ids)
