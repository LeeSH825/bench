from __future__ import annotations

import json
from dataclasses import fields

import numpy as np
import pytest

from bench.tasks.generator.mekf_events import (
    MEKFDataset,
    MEKFEventTable,
    compute_semantic_hashes,
    load_event_dataset,
    save_event_dataset,
    validate_generator_id,
)
from bench.tasks.generator.unit_st_synthetic import UnitSTSyntheticConfig, generate_unit_st


def _generated():
    return generate_unit_st(
        UnitSTSyntheticConfig(num_trajectories=4, duration_s=0.4, master_seed=901)
    )


def _event_kwargs(table: MEKFEventTable) -> dict[str, np.ndarray]:
    return {field.name: np.array(getattr(table, field.name), copy=True) for field in fields(table)}


def test_exact_dtypes_shapes_and_readonly_arrays() -> None:
    table = _generated().dataset.events
    assert table.trajectory_id.dtype == np.int64
    assert table.sensor_code.dtype == np.int16
    assert table.measurement_time_s.dtype == np.float64
    assert table.arrival_time_s.dtype == np.float64
    assert table.event_order.dtype == np.int64
    assert table.valid.dtype == np.bool_
    assert table.payload_index.dtype == np.int64
    assert table.gyro_omega_rad_s.shape[1:] == (3,)
    assert table.star_tracker_q_NB.shape[1:] == (4,)
    assert table.star_tracker_R_rad2.shape[1:] == (3, 3)
    assert all(not getattr(table, field.name).flags.writeable for field in fields(table))


@pytest.mark.parametrize(
    ("field", "mutation", "error"),
    [
        ("sensor_code", lambda value: value.astype(np.int64), TypeError),
        ("measurement_time_s", lambda value: value[:, None], ValueError),
        ("gyro_omega_rad_s", lambda value: value[:, :2], ValueError),
        ("star_tracker_q_NB", lambda value: value[:, :3], ValueError),
        ("star_tracker_R_rad2", lambda value: value[:, :2, :2], ValueError),
    ],
)
def test_invalid_dtype_rank_or_shape_fails_loudly(field, mutation, error) -> None:
    kwargs = _event_kwargs(_generated().dataset.events)
    kwargs[field] = mutation(kwargs[field])
    with pytest.raises(error):
        MEKFEventTable(**kwargs)


def test_payload_index_mismatch_and_range_fail_loudly() -> None:
    kwargs = _event_kwargs(_generated().dataset.events)
    kwargs["payload_index"][0] = kwargs["gyro_omega_rad_s"].shape[0]
    with pytest.raises(ValueError, match="one-to-one"):
        MEKFEventTable(**kwargs)


def test_star_quaternion_must_be_normalized() -> None:
    kwargs = _event_kwargs(_generated().dataset.events)
    kwargs["star_tracker_q_NB"][0] *= 1.01
    with pytest.raises(ValueError, match="normalized"):
        MEKFEventTable(**kwargs)


def test_star_covariance_must_be_spd() -> None:
    kwargs = _event_kwargs(_generated().dataset.events)
    kwargs["star_tracker_R_rad2"][0, 0, 0] = -1.0
    with pytest.raises(ValueError, match="positive definite"):
        MEKFEventTable(**kwargs)


def test_zero_latency_is_exact_and_nonzero_latency_is_rejected() -> None:
    table = _generated().dataset.events
    assert np.array_equal(table.arrival_time_s, table.measurement_time_s)
    kwargs = _event_kwargs(table)
    kwargs["arrival_time_s"][0] = np.nextafter(kwargs["arrival_time_s"][0], np.inf)
    with pytest.raises(ValueError, match="zero latency"):
        MEKFEventTable(**kwargs)


def test_event_sort_and_same_time_gyro_before_star_tracker_are_enforced() -> None:
    table = _generated().dataset.events
    trajectory_id = table.trajectory_id[0]
    rows = np.flatnonzero(table.trajectory_id == trajectory_id)
    assert np.all(np.diff(table.event_order[rows]) > 0)
    same_time = np.flatnonzero(
        (table.trajectory_id == trajectory_id) & (table.measurement_time_s == 0.2)
    )
    assert same_time.size == 2
    kwargs = _event_kwargs(table)
    kwargs["event_order"][same_time] = kwargs["event_order"][same_time[::-1]]
    with pytest.raises(ValueError, match="sorted"):
        MEKFEventTable(**kwargs)

    kwargs = _event_kwargs(table)
    kwargs["sensor_code"][same_time] = kwargs["sensor_code"][same_time[::-1]]
    kwargs["payload_index"][same_time] = kwargs["payload_index"][same_time[::-1]]
    with pytest.raises(ValueError):
        MEKFEventTable(**kwargs)


def test_serialization_round_trip_and_semantic_hash_equality(tmp_path) -> None:
    generated = _generated()
    artifact = tmp_path / "artifact"
    written = save_event_dataset(artifact, generated.dataset, generated.manifest)
    loaded, manifest, loaded_hashes = load_event_dataset(artifact)
    assert written == generated.semantic_hashes
    assert loaded_hashes == written
    assert compute_semantic_hashes(loaded, manifest) == written
    for field in fields(loaded.events):
        assert np.array_equal(
            getattr(loaded.events, field.name), getattr(generated.dataset.events, field.name)
        )
    for field in fields(loaded.truth):
        assert np.array_equal(
            getattr(loaded.truth, field.name), getattr(generated.dataset.truth, field.name)
        )


def test_object_array_npz_is_rejected_without_pickle(tmp_path) -> None:
    generated = _generated()
    artifact = tmp_path / "artifact"
    save_event_dataset(artifact, generated.dataset, generated.manifest)
    arrays = _event_kwargs(generated.dataset.events)
    arrays["sensor_code"] = np.array([object()], dtype=object)
    np.savez(artifact / "events.npz", **arrays)
    with pytest.raises(ValueError, match="strict NPZ"):
        load_event_dataset(artifact)


def test_payload_order_and_config_mutations_change_their_semantic_hashes() -> None:
    generated = _generated()
    original = generated.semantic_hashes

    payload_kwargs = _event_kwargs(generated.dataset.events)
    payload_kwargs["gyro_omega_rad_s"][0, 0] += 1.0e-12
    payload_dataset = MEKFDataset(
        events=MEKFEventTable(**payload_kwargs), truth=generated.dataset.truth
    )
    payload_hashes = compute_semantic_hashes(payload_dataset, generated.manifest)
    assert payload_hashes.sensor_payload_hash != original.sensor_payload_hash
    assert payload_hashes.dataset_hash != original.dataset_hash

    order_kwargs = _event_kwargs(generated.dataset.events)
    order_kwargs["event_order"] += np.int64(100)
    order_dataset = MEKFDataset(
        events=MEKFEventTable(**order_kwargs), truth=generated.dataset.truth
    )
    order_hashes = compute_semantic_hashes(order_dataset, generated.manifest)
    assert order_hashes.event_order_hash != original.event_order_hash
    assert order_hashes.dataset_hash != original.dataset_hash

    manifest = dict(generated.manifest)
    manifest["audit_note"] = "config mutation"
    config_hashes = compute_semantic_hashes(generated.dataset, manifest)
    assert config_hashes.manifest_hash != original.manifest_hash


def test_corrupted_manifest_and_recorded_hash_are_rejected(tmp_path) -> None:
    generated = _generated()
    corrupt_json = tmp_path / "corrupt-json"
    save_event_dataset(corrupt_json, generated.dataset, generated.manifest)
    with (corrupt_json / "manifest.json").open("ab") as stream:
        stream.write(b"!")
    with pytest.raises(ValueError, match="manifest"):
        load_event_dataset(corrupt_json)

    corrupt_hash = tmp_path / "corrupt-hash"
    save_event_dataset(corrupt_hash, generated.dataset, generated.manifest)
    path = corrupt_hash / "manifest.json"
    manifest = json.loads(path.read_text(encoding="ascii"))
    manifest["semantic_hashes"]["truth_hash"] = "0" * 64
    path.write_text(
        json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=True),
        encoding="ascii",
    )
    with pytest.raises(ValueError, match="hash mismatch"):
        load_event_dataset(corrupt_hash)


@pytest.mark.parametrize(
    "generator_id",
    ["synthetic-unit-st-v1", "basilisk-unit-st-v1"],
)
def test_versioned_generator_identity_round_trip_and_npz_schema_invariance(
    tmp_path, generator_id
) -> None:
    generated = _generated()
    manifest = dict(generated.manifest)
    manifest["generator_id"] = generator_id
    artifact = tmp_path / generator_id

    written = save_event_dataset(artifact, generated.dataset, manifest)
    loaded, loaded_manifest, loaded_hashes = load_event_dataset(
        artifact, expected_generator_id=generator_id
    )

    assert {path.name for path in artifact.iterdir()} == {
        "manifest.json",
        "truth.npz",
        "events.npz",
    }
    assert loaded_manifest["schema_version"] == generated.manifest["schema_version"]
    assert loaded_manifest["generator_id"] == generator_id
    assert loaded_hashes == written
    assert written.truth_hash == generated.semantic_hashes.truth_hash
    assert written.sensor_payload_hash == generated.semantic_hashes.sensor_payload_hash
    assert written.event_order_hash == generated.semantic_hashes.event_order_hash
    assert written.dataset_hash == generated.semantic_hashes.dataset_hash
    if generator_id != generated.manifest["generator_id"]:
        assert written.manifest_hash != generated.semantic_hashes.manifest_hash

    for table_name in ("events", "truth"):
        expected_table = getattr(generated.dataset, table_name)
        loaded_table = getattr(loaded, table_name)
        expected_fields = [field.name for field in fields(expected_table)]
        with np.load(artifact / f"{table_name}.npz", allow_pickle=False) as archive:
            assert set(archive.files) == set(expected_fields)
            for field in fields(expected_table):
                stored = archive[field.name]
                expected_array = getattr(expected_table, field.name)
                assert not stored.dtype.hasobject
                assert stored.dtype == expected_array.dtype
                assert stored.ndim == expected_array.ndim
                assert np.array_equal(stored, expected_array)
                assert np.array_equal(getattr(loaded_table, field.name), expected_array)


def test_expected_generator_identity_mismatch_fails_loudly(tmp_path) -> None:
    generated = _generated()
    artifact = tmp_path / "artifact"
    save_event_dataset(artifact, generated.dataset, generated.manifest)
    with pytest.raises(ValueError, match="generator_id mismatch"):
        load_event_dataset(artifact, expected_generator_id="basilisk-unit-st-v1")


@pytest.mark.parametrize(
    "generator_id",
    [
        "",
        " ",
        "\t",
        "synthetic-unit-st",
        "Synthetic-unit-st-v1",
        "synthetic_unit_st-v1",
        "synthetic-unit-st-v0",
        "synthetic-unit-st-v01",
        None,
    ],
)
def test_empty_malformed_or_unversioned_generator_identity_is_rejected(
    tmp_path, generator_id
) -> None:
    generated = _generated()
    manifest = dict(generated.manifest)
    manifest["generator_id"] = generator_id
    with pytest.raises(ValueError, match="generator_id"):
        save_event_dataset(tmp_path / "artifact", generated.dataset, manifest)


def test_generator_identity_tamper_is_detected_by_manifest_hash(tmp_path) -> None:
    generated = _generated()
    artifact = tmp_path / "artifact"
    save_event_dataset(artifact, generated.dataset, generated.manifest)
    path = artifact / "manifest.json"
    manifest = json.loads(path.read_text(encoding="ascii"))
    manifest["generator_id"] = "basilisk-unit-st-v1"
    path.write_text(
        json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=True),
        encoding="ascii",
    )
    with pytest.raises(ValueError, match="hash mismatch"):
        load_event_dataset(artifact)


def test_unsupported_schema_identity_is_rejected_on_save_and_load(tmp_path) -> None:
    generated = _generated()
    unsupported = dict(generated.manifest)
    unsupported["schema_version"] = "p1a-mekf-events-v2"
    with pytest.raises(ValueError, match="schema_version"):
        save_event_dataset(tmp_path / "unsupported-save", generated.dataset, unsupported)

    artifact = tmp_path / "unsupported-load"
    save_event_dataset(artifact, generated.dataset, generated.manifest)
    path = artifact / "manifest.json"
    manifest = json.loads(path.read_text(encoding="ascii"))
    manifest["schema_version"] = "p1a-mekf-events-v2"
    path.write_text(
        json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=True),
        encoding="ascii",
    )
    with pytest.raises(ValueError, match="schema_version"):
        load_event_dataset(artifact)


def test_generator_identity_validator_accepts_required_families() -> None:
    assert validate_generator_id("synthetic-unit-st-v1") == "synthetic-unit-st-v1"
    assert validate_generator_id("basilisk-unit-st-v1") == "basilisk-unit-st-v1"


def test_representative_synthetic_data_semantic_hashes_are_unchanged() -> None:
    generated = generate_unit_st(
        UnitSTSyntheticConfig(
            num_trajectories=8,
            duration_s=0.8,
            gyro_rate_hz=20,
            star_tracker_rate_hz=5,
            master_seed=731,
        )
    )
    assert generated.semantic_hashes.truth_hash == (
        "9b9545b069cdf3c0feb5e636e45213a1a17bf49dd18cfb1d7ef0c53a8152a71d"
    )
    assert generated.semantic_hashes.sensor_payload_hash == (
        "2fe16d091f43d3c0c24cde6044ecbba043ff7f1f8b8bf20ffb1edbe19e6da38a"
    )
    assert generated.semantic_hashes.event_order_hash == (
        "02bdea51896c359f66dd489f363aecd5d779cd0b64fa29c612d30f62f65ef125"
    )
    assert generated.semantic_hashes.dataset_hash == (
        "60607c5f078fd170392ec58846b44e8c3e43157509e1a7f74628d1ba9fa798e7"
    )
