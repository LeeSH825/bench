from __future__ import annotations

import json
from dataclasses import fields, replace

import numpy as np
import pytest

from bench.tasks.generator.mekf_fusion_events import (
    GENERATOR_ID,
    FusionEventTable,
    FusionSensorCode,
    load_fusion_dataset,
    load_fusion_oracle,
    save_fusion_dataset,
    save_fusion_oracle,
    select_fusion_sensors,
    validate_generator_id,
)
from bench.tasks.generator.phase1b_sensor_fusion import (
    FusionScenarioCode,
    SensorFusionConfig,
    generate_sensor_fusion,
)


@pytest.fixture(scope="module")
def generated():
    return generate_sensor_fusion(
        SensorFusionConfig(
            num_trajectories=5,
            duration_s=5.0,
            master_seed=27101,
            scenario_code=int(FusionScenarioCode.MAIN_FUSION_STATIONARY),
        )
    )


def test_schema_exact_dtype_shape_and_read_only(generated) -> None:
    events = generated.dataset.events
    expected = {
        "trajectory_id": np.dtype(np.int64),
        "sensor_code": np.dtype(np.int16),
        "measurement_time_s": np.dtype(np.float64),
        "arrival_time_s": np.dtype(np.float64),
        "event_order": np.dtype(np.int64),
        "valid": np.dtype(np.bool_),
        "payload_index": np.dtype(np.int64),
    }
    for name, dtype in expected.items():
        value = getattr(events, name)
        assert value.dtype == dtype
        assert value.flags.c_contiguous and not value.flags.writeable
    for item in fields(events):
        assert not getattr(events, item.name).flags.writeable
    for item in fields(generated.dataset.truth):
        assert not getattr(generated.dataset.truth, item.name).flags.writeable


def test_sensor_code_and_same_time_order(generated) -> None:
    table = generated.dataset.events
    assert set(map(int, np.unique(table.sensor_code))) == {1, 2, 3, 4}
    assert np.array_equal(table.measurement_time_s, table.arrival_time_s)
    rank = {1: 0, 3: 1, 4: 2, 2: 3}
    for trajectory_id in np.unique(table.trajectory_id):
        rows = np.flatnonzero(table.trajectory_id == trajectory_id)
        assert np.array_equal(table.event_order[rows], np.arange(rows.size))
        for time_s in np.unique(table.measurement_time_s[rows]):
            same = rows[table.measurement_time_s[rows] == time_s]
            codes = [int(item) for item in table.sensor_code[same]]
            assert codes[0] == 1
            assert [rank[item] for item in codes] == sorted(rank[item] for item in codes)


def test_payload_ownership_is_one_to_one(generated) -> None:
    table = generated.dataset.events
    counts = {
        1: table.gyro_omega_m_B_rad_s.shape[0],
        2: table.star_tracker_q_ST_NB.shape[0],
        3: table.magnetometer_z_mag_B.shape[0],
        4: table.sun_z_sun_B.shape[0],
    }
    for code, count in counts.items():
        owned = table.payload_index[table.sensor_code == code]
        assert np.array_equal(np.sort(owned), np.arange(count))


def test_invalid_policy_is_sun_skip_with_nonzero_unit_payload(generated) -> None:
    table = generated.dataset.events
    invalid = np.flatnonzero(~table.valid)
    assert invalid.size > 0
    assert np.all(table.sensor_code[invalid] == int(FusionSensorCode.SUN_SENSOR))
    payload = table.payload_index[invalid]
    assert np.allclose(np.linalg.norm(table.sun_z_sun_B[payload], axis=1), 1.0)
    assert np.all(np.linalg.norm(table.sun_z_sun_B[payload], axis=1) > 0.0)


def test_strict_serialization_hash_and_oracle_round_trip(generated, tmp_path) -> None:
    sensor = tmp_path / "sensor"
    oracle = tmp_path / "oracle"
    saved = save_fusion_dataset(sensor, generated.dataset, generated.sensor_manifest)
    save_fusion_oracle(oracle, generated.oracle_context, dataset_hash=saved.dataset_hash)
    loaded, manifest, hashes = load_fusion_dataset(
        sensor, expected_generator_id=GENERATOR_ID
    )
    loaded_oracle = load_fusion_oracle(oracle, expected_dataset_hash=hashes.dataset_hash)
    assert hashes == saved == generated.semantic_hashes
    assert loaded_oracle.semantic_hash == generated.oracle_context.semantic_hash
    assert manifest["generator_id"] == GENERATOR_ID
    for item in fields(FusionEventTable):
        assert np.array_equal(
            getattr(loaded.events, item.name), getattr(generated.dataset.events, item.name)
        )


def test_strict_identity_and_corruption_rejection(generated, tmp_path) -> None:
    sensor = tmp_path / "sensor"
    save_fusion_dataset(sensor, generated.dataset, generated.sensor_manifest)
    with pytest.raises(ValueError, match="generator identity"):
        load_fusion_dataset(sensor, expected_generator_id="another-generator-v1")
    manifest_path = sensor / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="ascii"))
    manifest["semantic_hashes"]["dataset_hash"] = "0" * 64
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")), encoding="ascii"
    )
    with pytest.raises(ValueError, match="semantic hash"):
        load_fusion_dataset(sensor, expected_generator_id=GENERATOR_ID)


@pytest.mark.parametrize(
    "value",
    ["", " bad-v1", "Bad-v1", "bad_generator-v1", "bad", "bad-v0", "bad-v01"],
)
def test_generator_identity_validation(value: str) -> None:
    with pytest.raises(ValueError):
        validate_generator_id(value)


def test_whole_trajectory_split_is_disjoint_and_complete(generated) -> None:
    split = generated.trajectory_split
    groups = [set(map(int, item)) for item in (split.train_ids, split.val_ids, split.test_ids)]
    assert not (groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])
    assert set.union(*groups) == set(map(int, generated.dataset.truth.trajectory_id))


def test_oracle_cursor_is_forward_only(generated) -> None:
    trajectory_id = int(generated.dataset.truth.trajectory_id[0])
    cursor = generated.oracle_context.cursor(trajectory_id)
    assert cursor.consume(0) == (1.0, 1.0)
    with pytest.raises(ValueError, match="strict order"):
        cursor.consume(2)


def test_sensor_selection_compacts_order_and_payload(generated) -> None:
    selected = select_fusion_sensors(
        generated.dataset.events,
        [FusionSensorCode.GYRO, FusionSensorCode.STAR_TRACKER],
    )
    assert set(map(int, np.unique(selected.sensor_code))) == {1, 2}
    for trajectory_id in np.unique(selected.trajectory_id):
        rows = np.flatnonzero(selected.trajectory_id == trajectory_id)
        assert np.array_equal(selected.event_order[rows], np.arange(rows.size))


def test_caller_arrays_cannot_alias_table(generated) -> None:
    original = generated.dataset.events
    values = {item.name: np.array(getattr(original, item.name), copy=True) for item in fields(original)}
    cloned = FusionEventTable(**values)
    values["gyro_omega_m_B_rad_s"][0, 0] += 1.0
    assert not np.array_equal(values["gyro_omega_m_B_rad_s"], cloned.gyro_omega_m_B_rad_s)
    with pytest.raises(ValueError):
        cloned.gyro_omega_m_B_rad_s[0, 0] = 0.0
