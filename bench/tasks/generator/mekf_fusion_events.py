"""Typed Phase 1B four-sensor MEKF events and deterministic persistence.

This module is deliberately separate from the frozen Phase 1A UNIT-ST schema.
Physical sensor/truth files and the simulation-only oracle sidecar have distinct
types, serializers, manifests, and semantic hashes.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import re
from dataclasses import dataclass, fields
from enum import IntEnum
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from bench.tasks.generator.mekf_events import TrajectorySplit, split_trajectory_ids


SCHEMA_VERSION = "p1b-mekf-fusion-events-v1"
GENERATOR_ID = "basilisk-sensor-fusion-regimes-v1"
CONVENTION_ID = "qNB-scalar-first-hamilton-right-v1"
SEED_POLICY_VERSION = "p1b-fusion-separated-streams-v1"
ORACLE_SCHEMA_VERSION = "p1b-mekf-fusion-oracle-context-v1"
SAME_TIME_ORDER_ID = "gyro-mag-sun-star-tracker-v1"
_GENERATOR_PATTERN = re.compile(r"[a-z][a-z0-9]*(?:-[a-z0-9]+)*-v[1-9][0-9]*\Z")


class FusionSensorCode(IntEnum):
    GYRO = 1
    STAR_TRACKER = 2
    MAGNETOMETER = 3
    SUN_SENSOR = 4


_GYRO = np.int16(FusionSensorCode.GYRO)
_ST = np.int16(FusionSensorCode.STAR_TRACKER)
_MAG = np.int16(FusionSensorCode.MAGNETOMETER)
_SUN = np.int16(FusionSensorCode.SUN_SENSOR)
_SAME_TIME_RANK = {_GYRO: 0, _MAG: 1, _SUN: 2, _ST: 3}


def _array(value: Any, dtype: Any, ndim: int, name: str) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{name} must be a numpy.ndarray")
    expected = np.dtype(dtype)
    if value.dtype != expected:
        raise TypeError(f"{name} must have dtype {expected}, got {value.dtype}")
    if value.ndim != ndim:
        raise ValueError(f"{name} must have rank {ndim}, got {value.shape}")
    result = np.array(value, dtype=expected, order="C", copy=True)
    result.setflags(write=False)
    return result


def _finite(value: np.ndarray, name: str) -> None:
    if not np.all(np.isfinite(value)):
        raise ValueError(f"{name} must contain only finite values")


def _unit_rows(value: np.ndarray, name: str, tolerance: float = 2.0e-13) -> None:
    if value.shape[1:] != (3,):
        raise ValueError(f"{name} must have shape [N,3]")
    if not np.allclose(np.linalg.norm(value, axis=1), 1.0, rtol=0.0, atol=tolerance):
        raise ValueError(f"{name} rows must be unit vectors")


def _unit_quaternions(value: np.ndarray, name: str) -> None:
    if value.shape[1:] != (4,):
        raise ValueError(f"{name} must have shape [N,4]")
    if not np.allclose(np.linalg.norm(value, axis=1), 1.0, rtol=0.0, atol=2.0e-13):
        raise ValueError(f"{name} rows must be unit quaternions")


def _spd_stack(value: np.ndarray, dimension: int, name: str) -> None:
    if value.shape[1:] != (dimension, dimension):
        raise ValueError(f"{name} must have shape [N,{dimension},{dimension}]")
    for index, matrix in enumerate(value):
        if not np.array_equal(matrix, matrix.T):
            raise ValueError(f"{name}[{index}] must be exactly symmetric")
        try:
            np.linalg.cholesky(matrix)
        except np.linalg.LinAlgError as error:
            raise ValueError(f"{name}[{index}] must be strictly SPD") from error


@dataclass(frozen=True)
class FusionEventTable:
    trajectory_id: np.ndarray
    sensor_code: np.ndarray
    measurement_time_s: np.ndarray
    arrival_time_s: np.ndarray
    event_order: np.ndarray
    valid: np.ndarray
    payload_index: np.ndarray
    gyro_omega_m_B_rad_s: np.ndarray
    star_tracker_q_ST_NB: np.ndarray
    star_tracker_R_ST_rad2: np.ndarray
    magnetometer_z_mag_B: np.ndarray
    magnetometer_r_mag_N_model: np.ndarray
    magnetometer_R_mag: np.ndarray
    sun_z_sun_B: np.ndarray
    sun_r_sun_N_model: np.ndarray
    sun_R_sun_tangent_rad2: np.ndarray

    def __post_init__(self) -> None:
        specs = {
            "trajectory_id": (np.int64, 1),
            "sensor_code": (np.int16, 1),
            "measurement_time_s": (np.float64, 1),
            "arrival_time_s": (np.float64, 1),
            "event_order": (np.int64, 1),
            "valid": (np.bool_, 1),
            "payload_index": (np.int64, 1),
            "gyro_omega_m_B_rad_s": (np.float64, 2),
            "star_tracker_q_ST_NB": (np.float64, 2),
            "star_tracker_R_ST_rad2": (np.float64, 3),
            "magnetometer_z_mag_B": (np.float64, 2),
            "magnetometer_r_mag_N_model": (np.float64, 2),
            "magnetometer_R_mag": (np.float64, 3),
            "sun_z_sun_B": (np.float64, 2),
            "sun_r_sun_N_model": (np.float64, 2),
            "sun_R_sun_tangent_rad2": (np.float64, 3),
        }
        for name, (dtype, ndim) in specs.items():
            object.__setattr__(self, name, _array(getattr(self, name), dtype, ndim, name))
        validate_fusion_event_table(self)

    @property
    def event_count(self) -> int:
        return int(self.trajectory_id.size)


@dataclass(frozen=True)
class FusionTruthTable:
    trajectory_id: np.ndarray
    truth_offsets: np.ndarray
    truth_time_s: np.ndarray
    q_true_NB: np.ndarray
    gyro_bias_true_rad_s: np.ndarray
    omega_true_B_rad_s: np.ndarray
    r_mag_N_true: np.ndarray
    r_sun_N_true: np.ndarray

    def __post_init__(self) -> None:
        specs = {
            "trajectory_id": (np.int64, 1),
            "truth_offsets": (np.int64, 1),
            "truth_time_s": (np.float64, 1),
            "q_true_NB": (np.float64, 2),
            "gyro_bias_true_rad_s": (np.float64, 2),
            "omega_true_B_rad_s": (np.float64, 2),
            "r_mag_N_true": (np.float64, 2),
            "r_sun_N_true": (np.float64, 2),
        }
        for name, (dtype, ndim) in specs.items():
            object.__setattr__(self, name, _array(getattr(self, name), dtype, ndim, name))
        validate_fusion_truth_table(self)


@dataclass(frozen=True)
class FusionDataset:
    events: FusionEventTable
    truth: FusionTruthTable

    def __post_init__(self) -> None:
        if not isinstance(self.events, FusionEventTable) or not isinstance(
            self.truth, FusionTruthTable
        ):
            raise TypeError("events/truth must use the fusion table types")
        if set(map(int, np.unique(self.events.trajectory_id))) != set(
            map(int, self.truth.trajectory_id)
        ):
            raise ValueError("event and truth trajectory IDs must match")


@dataclass(frozen=True)
class FusionSemanticHashes:
    truth_reference_hash: str
    sensor_payload_hash: str
    event_order_hash: str
    manifest_hash: str
    dataset_hash: str

    def as_dict(self) -> dict[str, str]:
        return {item.name: str(getattr(self, item.name)) for item in fields(self)}


@dataclass(frozen=True)
class FusionOracleSidecar:
    trajectory_id: np.ndarray
    event_order: np.ndarray
    alpha_b: np.ndarray
    alpha_R_mag: np.ndarray
    slow_window: np.ndarray
    fast_window: np.ndarray
    scenario_code: np.ndarray

    def __post_init__(self) -> None:
        specs = {
            "trajectory_id": (np.int64, 1),
            "event_order": (np.int64, 1),
            "alpha_b": (np.float64, 1),
            "alpha_R_mag": (np.float64, 1),
            "slow_window": (np.bool_, 1),
            "fast_window": (np.bool_, 1),
            "scenario_code": (np.int8, 1),
        }
        for name, (dtype, ndim) in specs.items():
            object.__setattr__(self, name, _array(getattr(self, name), dtype, ndim, name))
        count = self.trajectory_id.size
        if count == 0 or any(getattr(self, item).size != count for item in specs):
            raise ValueError("oracle fields must be nonempty and equal length")
        if not np.all(np.isfinite(self.alpha_b)) or not np.all(np.isfinite(self.alpha_R_mag)):
            raise ValueError("oracle scales must be finite")
        if np.any(self.alpha_b < 1.0) or np.any(self.alpha_R_mag < 1.0):
            raise ValueError("oracle scales must be at least one")
        for trajectory_id in np.unique(self.trajectory_id):
            rows = np.flatnonzero(self.trajectory_id == trajectory_id)
            if not np.array_equal(self.event_order[rows], np.arange(rows.size, dtype=np.int64)):
                raise ValueError("oracle event order must be contiguous per trajectory")

    @property
    def semantic_hash(self) -> str:
        return _hash_named_arrays("p1b-fusion-oracle-v1", _named_arrays(self))

    def cursor(self, trajectory_id: int) -> "CurrentFusionOracleCursor":
        return CurrentFusionOracleCursor(self, trajectory_id)


class CurrentFusionOracleCursor:
    """Forward-only view exposing exactly one current event at a time."""

    __slots__ = ("_alpha_b", "_alpha_r", "_next")

    def __init__(self, sidecar: FusionOracleSidecar, trajectory_id: int) -> None:
        if not isinstance(sidecar, FusionOracleSidecar):
            raise TypeError("sidecar must be FusionOracleSidecar")
        rows = np.flatnonzero(sidecar.trajectory_id == np.int64(trajectory_id))
        if rows.size == 0:
            raise ValueError("trajectory is absent from oracle sidecar")
        self._alpha_b = np.array(sidecar.alpha_b[rows], copy=True)
        self._alpha_r = np.array(sidecar.alpha_R_mag[rows], copy=True)
        self._next = 0

    def consume(self, event_order: int) -> tuple[float, float]:
        order = int(event_order)
        if order != self._next or order >= self._alpha_b.size:
            raise ValueError("oracle cursor must be consumed once in strict order")
        result = float(self._alpha_b[order]), float(self._alpha_r[order])
        self._next += 1
        return result


def validate_fusion_event_table(table: FusionEventTable) -> None:
    metadata = (
        table.trajectory_id,
        table.sensor_code,
        table.measurement_time_s,
        table.arrival_time_s,
        table.event_order,
        table.valid,
        table.payload_index,
    )
    count = table.trajectory_id.size
    if count == 0 or any(item.size != count for item in metadata):
        raise ValueError("event metadata fields must be nonempty and equal length")
    for name in (
        "measurement_time_s",
        "arrival_time_s",
        "gyro_omega_m_B_rad_s",
        "star_tracker_q_ST_NB",
        "star_tracker_R_ST_rad2",
        "magnetometer_z_mag_B",
        "magnetometer_r_mag_N_model",
        "magnetometer_R_mag",
        "sun_z_sun_B",
        "sun_r_sun_N_model",
        "sun_R_sun_tangent_rad2",
    ):
        _finite(getattr(table, name), name)
    if np.any(table.measurement_time_s < 0.0) or not np.array_equal(
        table.measurement_time_s, table.arrival_time_s
    ):
        raise ValueError("fusion events require finite nonnegative exact zero latency")
    allowed = np.asarray([_GYRO, _ST, _MAG, _SUN], dtype=np.int16)
    if not np.all(np.isin(table.sensor_code, allowed)):
        raise ValueError("unknown fusion sensor code")
    if np.any(table.payload_index < 0):
        raise ValueError("payload indices must be nonnegative")
    if np.any(~table.valid[table.sensor_code != _SUN]):
        raise ValueError("only sun events may be invalid in the Step 2 primary schema")

    payloads = {
        _GYRO: table.gyro_omega_m_B_rad_s.shape[0],
        _ST: table.star_tracker_q_ST_NB.shape[0],
        _MAG: table.magnetometer_z_mag_B.shape[0],
        _SUN: table.sun_z_sun_B.shape[0],
    }
    for code, payload_count in payloads.items():
        indices = table.payload_index[table.sensor_code == code]
        if not np.array_equal(np.sort(indices), np.arange(payload_count, dtype=np.int64)):
            raise ValueError("each typed payload must be owned exactly once")
    if table.gyro_omega_m_B_rad_s.shape[1:] != (3,):
        raise ValueError("gyro payload must have shape [G,3]")
    _unit_quaternions(table.star_tracker_q_ST_NB, "star_tracker_q_ST_NB")
    _spd_stack(table.star_tracker_R_ST_rad2, 3, "star_tracker_R_ST_rad2")
    _unit_rows(table.magnetometer_r_mag_N_model, "magnetometer_r_mag_N_model")
    if table.magnetometer_z_mag_B.shape[1:] != (3,):
        raise ValueError("magnetometer_z_mag_B must have shape [M,3]")
    _spd_stack(table.magnetometer_R_mag, 3, "magnetometer_R_mag")
    _unit_rows(table.sun_z_sun_B, "sun_z_sun_B")
    _unit_rows(table.sun_r_sun_N_model, "sun_r_sun_N_model")
    _spd_stack(table.sun_R_sun_tangent_rad2, 2, "sun_R_sun_tangent_rad2")

    for trajectory_id in np.unique(table.trajectory_id):
        rows = np.flatnonzero(table.trajectory_id == trajectory_id)
        if not np.array_equal(table.event_order[rows], np.arange(rows.size, dtype=np.int64)):
            raise ValueError("event_order must be contiguous per trajectory")
        times = table.measurement_time_s[rows]
        if np.any(np.diff(times) < 0.0):
            raise ValueError("event times must be monotonic per trajectory")
        for time_s in np.unique(times):
            same = rows[times == time_s]
            ranks = np.asarray([_SAME_TIME_RANK[item] for item in table.sensor_code[same]])
            if not np.array_equal(ranks, np.sort(ranks)) or table.sensor_code[same[0]] != _GYRO:
                raise ValueError("same-time order must be gyro, magnetometer, sun, ST")


def validate_fusion_truth_table(table: FusionTruthTable) -> None:
    count = table.trajectory_id.size
    if count < 1 or np.unique(table.trajectory_id).size != count:
        raise ValueError("truth trajectory IDs must be nonempty and unique")
    if table.truth_offsets.shape != (count + 1,) or table.truth_offsets[0] != 0:
        raise ValueError("truth offsets are malformed")
    if np.any(np.diff(table.truth_offsets) <= 0):
        raise ValueError("every truth trajectory must have samples")
    total = int(table.truth_offsets[-1])
    if any(
        getattr(table, name).shape[0] != total
        for name in (
            "truth_time_s",
            "q_true_NB",
            "gyro_bias_true_rad_s",
            "omega_true_B_rad_s",
            "r_mag_N_true",
            "r_sun_N_true",
        )
    ):
        raise ValueError("truth arrays do not match ragged offsets")
    for name in (
        "truth_time_s",
        "q_true_NB",
        "gyro_bias_true_rad_s",
        "omega_true_B_rad_s",
        "r_mag_N_true",
        "r_sun_N_true",
    ):
        _finite(getattr(table, name), name)
    _unit_quaternions(table.q_true_NB, "q_true_NB")
    _unit_rows(table.r_mag_N_true, "r_mag_N_true")
    _unit_rows(table.r_sun_N_true, "r_sun_N_true")
    if table.gyro_bias_true_rad_s.shape[1:] != (3,) or table.omega_true_B_rad_s.shape[1:] != (3,):
        raise ValueError("truth bias/rate arrays must have shape [T,3]")
    for index in range(count):
        start, stop = int(table.truth_offsets[index]), int(table.truth_offsets[index + 1])
        if np.any(np.diff(table.truth_time_s[start:stop]) <= 0.0):
            raise ValueError("truth time must increase strictly per trajectory")


def validate_generator_id(value: Any) -> str:
    if not isinstance(value, str) or value != value.strip() or not value:
        raise ValueError("generator_id must be a nonempty unpadded string")
    if _GENERATOR_PATTERN.fullmatch(value) is None:
        raise ValueError("generator_id must be a lowercase versioned identity")
    return value


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    try:
        return json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
        ).encode("ascii")
    except (TypeError, ValueError) as error:
        raise ValueError("manifest must be finite canonical JSON") from error


def _canonical_array(value: np.ndarray) -> np.ndarray:
    if value.dtype.hasobject:
        raise TypeError("object arrays are forbidden")
    dtype = value.dtype.newbyteorder("<") if value.dtype.itemsize > 1 else value.dtype
    return np.ascontiguousarray(value.astype(dtype, copy=False))


def _named_arrays(record: Any) -> list[tuple[str, np.ndarray]]:
    return [(item.name, getattr(record, item.name)) for item in fields(record)]


def _hash_named_arrays(domain: str, arrays: Sequence[tuple[str, np.ndarray]]) -> str:
    digest = hashlib.sha256(domain.encode("ascii") + b"\0")
    for name, raw in arrays:
        value = _canonical_array(raw)
        metadata = _canonical_json(
            {"dtype": value.dtype.str, "field": name, "shape": list(value.shape)}
        )
        digest.update(len(metadata).to_bytes(8, "little"))
        digest.update(metadata)
        payload = value.tobytes(order="C")
        digest.update(len(payload).to_bytes(8, "little"))
        digest.update(payload)
    return digest.hexdigest()


def _manifest_identity(manifest: Mapping[str, Any]) -> dict[str, Any]:
    value = copy.deepcopy(dict(manifest))
    value.pop("semantic_hashes", None)
    return value


def _validated_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    value = _manifest_identity(manifest)
    expected = {
        "schema_version": SCHEMA_VERSION,
        "convention_id": CONVENTION_ID,
        "seed_policy_version": SEED_POLICY_VERSION,
        "same_time_order_id": SAME_TIME_ORDER_ID,
    }
    for name, locked in expected.items():
        if value.get(name) != locked:
            raise ValueError(f"manifest {name} must equal {locked!r}")
    validate_generator_id(value.get("generator_id"))
    return value


def compute_fusion_semantic_hashes(
    dataset: FusionDataset, manifest: Mapping[str, Any]
) -> FusionSemanticHashes:
    if not isinstance(dataset, FusionDataset):
        raise TypeError("dataset must be FusionDataset")
    identity = _validated_manifest(manifest)
    truth_hash = _hash_named_arrays("p1b-fusion-truth-reference-v1", _named_arrays(dataset.truth))
    routing = {
        "trajectory_id",
        "sensor_code",
        "measurement_time_s",
        "arrival_time_s",
        "event_order",
        "valid",
        "payload_index",
    }
    event_arrays = _named_arrays(dataset.events)
    event_hash = _hash_named_arrays(
        "p1b-fusion-event-order-v1", [(n, a) for n, a in event_arrays if n in routing]
    )
    sensor_hash = _hash_named_arrays(
        "p1b-fusion-sensor-payload-v1", [(n, a) for n, a in event_arrays if n not in routing]
    )
    manifest_hash = hashlib.sha256(_canonical_json(identity)).hexdigest()
    digest = hashlib.sha256(b"p1b-fusion-dataset-v1\0")
    for name, value in (
        ("truth_reference_hash", truth_hash),
        ("sensor_payload_hash", sensor_hash),
        ("event_order_hash", event_hash),
    ):
        digest.update(name.encode("ascii") + b"=" + value.encode("ascii") + b"\0")
    return FusionSemanticHashes(truth_hash, sensor_hash, event_hash, manifest_hash, digest.hexdigest())


def save_fusion_dataset(
    directory: os.PathLike[str] | str,
    dataset: FusionDataset,
    manifest: Mapping[str, Any],
) -> FusionSemanticHashes:
    target = Path(directory)
    if target.exists():
        if not target.is_dir() or any(target.iterdir()):
            raise FileExistsError("fusion serialization target must be a new empty directory")
    else:
        target.mkdir(parents=True, exist_ok=False)
    identity = _validated_manifest(manifest)
    hashes = compute_fusion_semantic_hashes(dataset, identity)
    serialized = copy.deepcopy(identity)
    serialized["semantic_hashes"] = hashes.as_dict()
    (target / "manifest.json").write_bytes(_canonical_json(serialized))
    np.savez(target / "truth.npz", **dict(_named_arrays(dataset.truth)))
    np.savez(target / "events.npz", **dict(_named_arrays(dataset.events)))
    return hashes


def _load_npz(path: Path, record_type: type[Any]) -> dict[str, np.ndarray]:
    expected = [item.name for item in fields(record_type)]
    try:
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != set(expected):
                raise ValueError("missing or extra NPZ fields")
            values = {name: np.array(archive[name], copy=True) for name in expected}
    except (OSError, TypeError, ValueError) as error:
        raise ValueError(f"failed strict load of {path.name}") from error
    if any(item.dtype.hasobject for item in values.values()):
        raise ValueError("object arrays are forbidden")
    return values


def load_fusion_dataset(
    directory: os.PathLike[str] | str,
    *,
    expected_generator_id: str = GENERATOR_ID,
) -> tuple[FusionDataset, dict[str, Any], FusionSemanticHashes]:
    source = Path(directory)
    if not source.is_dir() or {item.name for item in source.iterdir()} != {
        "manifest.json",
        "truth.npz",
        "events.npz",
    }:
        raise ValueError("fusion artifact must contain exactly manifest/truth/events")
    raw = (source / "manifest.json").read_bytes()
    try:
        manifest = json.loads(raw.decode("ascii"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("invalid fusion manifest JSON") from error
    if not isinstance(manifest, dict) or _canonical_json(manifest) != raw:
        raise ValueError("fusion manifest is not canonical")
    identity = _validated_manifest(manifest)
    if identity["generator_id"] != validate_generator_id(expected_generator_id):
        raise ValueError("fusion generator identity mismatch")
    events = FusionEventTable(**_load_npz(source / "events.npz", FusionEventTable))
    truth = FusionTruthTable(**_load_npz(source / "truth.npz", FusionTruthTable))
    dataset = FusionDataset(events, truth)
    hashes = compute_fusion_semantic_hashes(dataset, manifest)
    if manifest.get("semantic_hashes") != hashes.as_dict():
        raise ValueError("fusion semantic hash mismatch")
    return dataset, manifest, hashes


def save_fusion_oracle(
    directory: os.PathLike[str] | str,
    oracle: FusionOracleSidecar,
    *,
    dataset_hash: str,
) -> str:
    target = Path(directory)
    if target.exists():
        if not target.is_dir() or any(target.iterdir()):
            raise FileExistsError("oracle target must be a new empty directory")
    else:
        target.mkdir(parents=True, exist_ok=False)
    oracle_hash = oracle.semantic_hash
    manifest = {
        "dataset_hash": str(dataset_hash),
        "oracle_hash": oracle_hash,
        "schema_version": ORACLE_SCHEMA_VERSION,
    }
    (target / "manifest.json").write_bytes(_canonical_json(manifest))
    np.savez(target / "oracle.npz", **dict(_named_arrays(oracle)))
    return oracle_hash


def load_fusion_oracle(
    directory: os.PathLike[str] | str,
    *,
    expected_dataset_hash: str,
) -> FusionOracleSidecar:
    source = Path(directory)
    if not source.is_dir() or {item.name for item in source.iterdir()} != {
        "manifest.json",
        "oracle.npz",
    }:
        raise ValueError("oracle artifact must contain exactly manifest/oracle")
    raw = (source / "manifest.json").read_bytes()
    try:
        manifest = json.loads(raw.decode("ascii"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("invalid oracle manifest") from error
    if _canonical_json(manifest) != raw or manifest.get("schema_version") != ORACLE_SCHEMA_VERSION:
        raise ValueError("oracle manifest is not canonical or has wrong schema")
    if manifest.get("dataset_hash") != expected_dataset_hash:
        raise ValueError("oracle is linked to a different physical dataset")
    oracle = FusionOracleSidecar(**_load_npz(source / "oracle.npz", FusionOracleSidecar))
    if oracle.semantic_hash != manifest.get("oracle_hash"):
        raise ValueError("oracle semantic hash mismatch")
    return oracle


def split_fusion_trajectory_ids(
    trajectory_ids: Sequence[int] | np.ndarray,
    *,
    split_seed: int,
    train_fraction: float,
    val_fraction: float,
    test_fraction: float,
) -> TrajectorySplit:
    return split_trajectory_ids(
        trajectory_ids,
        split_seed=split_seed,
        train_fraction=train_fraction,
        val_fraction=val_fraction,
        test_fraction=test_fraction,
    )


def select_fusion_sensors(
    table: FusionEventTable, sensor_codes: Sequence[int | FusionSensorCode]
) -> FusionEventTable:
    """Select complete typed sensor streams and compact order/payload indices."""

    requested = {np.int16(int(item)) for item in sensor_codes}
    if _GYRO not in requested or not requested.issubset({_GYRO, _ST, _MAG, _SUN}):
        raise ValueError("sensor subset must contain gyro and only known sensors")
    rows = np.flatnonzero(np.isin(table.sensor_code, np.asarray(sorted(requested), dtype=np.int16)))
    codes = table.sensor_code[rows]
    old_payload = table.payload_index[rows]
    new_payload = np.empty(rows.size, dtype=np.int64)
    payload_rows: dict[np.int16, np.ndarray] = {}
    for code in (_GYRO, _ST, _MAG, _SUN):
        owned = old_payload[codes == code]
        payload_rows[code] = owned
        new_payload[codes == code] = np.arange(owned.size, dtype=np.int64)
    new_orders = np.empty(rows.size, dtype=np.int64)
    for trajectory_id in np.unique(table.trajectory_id[rows]):
        selected = np.flatnonzero(table.trajectory_id[rows] == trajectory_id)
        new_orders[selected] = np.arange(selected.size, dtype=np.int64)
    return FusionEventTable(
        trajectory_id=table.trajectory_id[rows],
        sensor_code=codes,
        measurement_time_s=table.measurement_time_s[rows],
        arrival_time_s=table.arrival_time_s[rows],
        event_order=new_orders,
        valid=table.valid[rows],
        payload_index=new_payload,
        gyro_omega_m_B_rad_s=table.gyro_omega_m_B_rad_s[payload_rows[_GYRO]],
        star_tracker_q_ST_NB=table.star_tracker_q_ST_NB[payload_rows[_ST]],
        star_tracker_R_ST_rad2=table.star_tracker_R_ST_rad2[payload_rows[_ST]],
        magnetometer_z_mag_B=table.magnetometer_z_mag_B[payload_rows[_MAG]],
        magnetometer_r_mag_N_model=table.magnetometer_r_mag_N_model[payload_rows[_MAG]],
        magnetometer_R_mag=table.magnetometer_R_mag[payload_rows[_MAG]],
        sun_z_sun_B=table.sun_z_sun_B[payload_rows[_SUN]],
        sun_r_sun_N_model=table.sun_r_sun_N_model[payload_rows[_SUN]],
        sun_R_sun_tangent_rad2=table.sun_R_sun_tangent_rad2[payload_rows[_SUN]],
    )


__all__ = [
    "CONVENTION_ID",
    "CurrentFusionOracleCursor",
    "FusionDataset",
    "FusionEventTable",
    "FusionOracleSidecar",
    "FusionSemanticHashes",
    "FusionSensorCode",
    "FusionTruthTable",
    "GENERATOR_ID",
    "ORACLE_SCHEMA_VERSION",
    "SAME_TIME_ORDER_ID",
    "SCHEMA_VERSION",
    "SEED_POLICY_VERSION",
    "compute_fusion_semantic_hashes",
    "load_fusion_dataset",
    "load_fusion_oracle",
    "save_fusion_dataset",
    "save_fusion_oracle",
    "select_fusion_sensors",
    "split_fusion_trajectory_ids",
]
