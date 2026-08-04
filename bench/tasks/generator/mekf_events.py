"""Typed Phase 1A MEKF events, semantic serialization, splitting, and replay.

This module is intentionally independent of the legacy float32 sequence format.
Truth is carried only by :class:`MEKFDataset`; :func:`replay_trajectory` accepts
an event table and therefore cannot consume truth accidentally.
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

from bench.estimators.mekf import (
    MEKFState,
    propagate_state,
    quat_geodesic_angle,
    star_tracker_update,
)


SCHEMA_VERSION = "p1a-mekf-events-v1"
# Backward-compatible identity of the analytic synthetic generator.  The
# serializer accepts any generator identity satisfying the versioned contract.
GENERATOR_ID = "synthetic-unit-st-v1"
SEED_POLICY_VERSION = "p1a-separated-streams-v1"
CONVENTION_ID = "qNB-scalar-first-hamilton-right-v1"
_GENERATOR_ID_PATTERN = re.compile(r"[a-z][a-z0-9]*(?:-[a-z0-9]+)*-v[1-9][0-9]*\Z")


class SensorCode(IntEnum):
    """Frozen integer sensor codes used on disk and in memory."""

    GYRO = 1
    STAR_TRACKER = 2


_GYRO = np.int16(SensorCode.GYRO)
_STAR_TRACKER = np.int16(SensorCode.STAR_TRACKER)
_HASH_KEYS = (
    "truth_hash",
    "sensor_payload_hash",
    "event_order_hash",
    "manifest_hash",
    "dataset_hash",
)


def _require_array(value: Any, *, dtype: np.dtype[Any], ndim: int, name: str) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{name} must be a numpy.ndarray")
    expected = np.dtype(dtype)
    if value.dtype != expected:
        raise TypeError(f"{name} must have dtype {expected}, got {value.dtype}")
    if value.ndim != ndim:
        raise ValueError(f"{name} must have rank {ndim}, got shape {value.shape}")
    result = np.array(value, dtype=expected, order="C", copy=True)
    result.setflags(write=False)
    return result


def _readonly(value: np.ndarray, dtype: np.dtype[Any]) -> np.ndarray:
    result = np.array(value, dtype=dtype, order="C", copy=True)
    result.setflags(write=False)
    return result


def _require_finite(value: np.ndarray, name: str) -> None:
    if not np.all(np.isfinite(value)):
        raise ValueError(f"{name} must contain only finite values")


def _require_normalized_quaternions(value: np.ndarray, name: str) -> None:
    norms = np.linalg.norm(value, axis=1)
    if not np.allclose(norms, 1.0, rtol=0.0, atol=2.0e-13):
        raise ValueError(f"{name} must contain normalized quaternions")


def _require_spd_stack(value: np.ndarray, name: str) -> None:
    for index, matrix in enumerate(value):
        if not np.array_equal(matrix, matrix.T):
            raise ValueError(f"{name}[{index}] must be exactly symmetric")
        try:
            np.linalg.cholesky(matrix)
        except np.linalg.LinAlgError as error:
            raise ValueError(f"{name}[{index}] must be positive definite") from error


@dataclass(frozen=True)
class MEKFEventTable:
    """Struct-of-arrays event table with disjoint typed payload tables."""

    trajectory_id: np.ndarray
    sensor_code: np.ndarray
    measurement_time_s: np.ndarray
    arrival_time_s: np.ndarray
    event_order: np.ndarray
    valid: np.ndarray
    payload_index: np.ndarray
    gyro_omega_rad_s: np.ndarray
    star_tracker_q_NB: np.ndarray
    star_tracker_R_rad2: np.ndarray

    def __post_init__(self) -> None:
        specifications = {
            "trajectory_id": (np.dtype(np.int64), 1),
            "sensor_code": (np.dtype(np.int16), 1),
            "measurement_time_s": (np.dtype(np.float64), 1),
            "arrival_time_s": (np.dtype(np.float64), 1),
            "event_order": (np.dtype(np.int64), 1),
            "valid": (np.dtype(np.bool_), 1),
            "payload_index": (np.dtype(np.int64), 1),
            "gyro_omega_rad_s": (np.dtype(np.float64), 2),
            "star_tracker_q_NB": (np.dtype(np.float64), 2),
            "star_tracker_R_rad2": (np.dtype(np.float64), 3),
        }
        for name, (dtype, ndim) in specifications.items():
            object.__setattr__(
                self,
                name,
                _require_array(getattr(self, name), dtype=dtype, ndim=ndim, name=name),
            )
        validate_event_table(self)

    @property
    def event_count(self) -> int:
        return int(self.trajectory_id.size)


@dataclass(frozen=True)
class MEKFTruthTable:
    """Ragged analytic truth stored separately from estimator inputs."""

    trajectory_id: np.ndarray
    truth_offsets: np.ndarray
    truth_time_s: np.ndarray
    q_true_NB: np.ndarray
    gyro_bias_rad_s: np.ndarray
    omega_true_rad_s: np.ndarray

    def __post_init__(self) -> None:
        specifications = {
            "trajectory_id": (np.dtype(np.int64), 1),
            "truth_offsets": (np.dtype(np.int64), 1),
            "truth_time_s": (np.dtype(np.float64), 1),
            "q_true_NB": (np.dtype(np.float64), 2),
            "gyro_bias_rad_s": (np.dtype(np.float64), 2),
            "omega_true_rad_s": (np.dtype(np.float64), 2),
        }
        for name, (dtype, ndim) in specifications.items():
            object.__setattr__(
                self,
                name,
                _require_array(getattr(self, name), dtype=dtype, ndim=ndim, name=name),
            )
        validate_truth_table(self)


@dataclass(frozen=True)
class MEKFDataset:
    """Typed sensor events plus inaccessible-by-replay truth."""

    events: MEKFEventTable
    truth: MEKFTruthTable

    def __post_init__(self) -> None:
        if not isinstance(self.events, MEKFEventTable):
            raise TypeError("events must be an MEKFEventTable")
        if not isinstance(self.truth, MEKFTruthTable):
            raise TypeError("truth must be an MEKFTruthTable")
        event_ids = set(int(item) for item in np.unique(self.events.trajectory_id))
        truth_ids = set(int(item) for item in self.truth.trajectory_id)
        if event_ids != truth_ids:
            raise ValueError("event and truth trajectory IDs must match exactly")


@dataclass(frozen=True)
class SemanticHashes:
    truth_hash: str
    sensor_payload_hash: str
    event_order_hash: str
    manifest_hash: str
    dataset_hash: str

    def as_dict(self) -> dict[str, str]:
        return {field.name: str(getattr(self, field.name)) for field in fields(self)}


@dataclass(frozen=True)
class TrajectorySplit:
    train_ids: np.ndarray
    val_ids: np.ndarray
    test_ids: np.ndarray
    split_seed: int

    def __post_init__(self) -> None:
        for name in ("train_ids", "val_ids", "test_ids"):
            array = _require_array(
                getattr(self, name), dtype=np.dtype(np.int64), ndim=1, name=name
            )
            if np.unique(array).size != array.size:
                raise ValueError(f"{name} contains duplicate trajectory IDs")
            object.__setattr__(self, name, array)
        groups = [set(map(int, getattr(self, name))) for name in ("train_ids", "val_ids", "test_ids")]
        if groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2]:
            raise ValueError("trajectory splits must be disjoint")

    @property
    def all_ids(self) -> np.ndarray:
        return _readonly(
            np.concatenate((self.train_ids, self.val_ids, self.test_ids)), np.int64
        )


@dataclass(frozen=True)
class ReplayResult:
    """Defensive replay trace. ST evidence is stored only for ST updates."""

    trajectory_id: int
    processed_event_count: int
    event_time_s: np.ndarray
    event_order: np.ndarray
    sensor_code: np.ndarray
    q_NB_history: np.ndarray
    b_g_history: np.ndarray
    P_history: np.ndarray
    attitude_step_rad: np.ndarray
    star_tracker_event_order: np.ndarray
    star_tracker_residual: np.ndarray
    star_tracker_S: np.ndarray
    final_state: MEKFState

    def __post_init__(self) -> None:
        if not isinstance(self.trajectory_id, (int, np.integer)):
            raise TypeError("trajectory_id must be an integer")
        object.__setattr__(self, "trajectory_id", int(self.trajectory_id))
        specifications = {
            "event_time_s": (np.float64, 1),
            "event_order": (np.int64, 1),
            "sensor_code": (np.int16, 1),
            "q_NB_history": (np.float64, 2),
            "b_g_history": (np.float64, 2),
            "P_history": (np.float64, 3),
            "attitude_step_rad": (np.float64, 1),
            "star_tracker_event_order": (np.int64, 1),
            "star_tracker_residual": (np.float64, 2),
            "star_tracker_S": (np.float64, 3),
        }
        for name, (dtype, ndim) in specifications.items():
            value = _require_array(
                getattr(self, name), dtype=np.dtype(dtype), ndim=ndim, name=name
            )
            _require_finite(value, name)
            object.__setattr__(self, name, value)
        count = int(self.processed_event_count)
        if count < 0:
            raise ValueError("processed_event_count must be nonnegative")
        if any(
            int(getattr(self, name).shape[0]) != count
            for name in (
                "event_time_s",
                "event_order",
                "sensor_code",
                "q_NB_history",
                "b_g_history",
                "P_history",
                "attitude_step_rad",
            )
        ):
            raise ValueError("replay event histories must agree with processed_event_count")
        if self.q_NB_history.shape[1:] != (4,):
            raise ValueError("q_NB_history must have shape [E,4]")
        if self.b_g_history.shape[1:] != (3,):
            raise ValueError("b_g_history must have shape [E,3]")
        if self.P_history.shape[1:] != (6, 6):
            raise ValueError("P_history must have shape [E,6,6]")
        st_count = int(self.star_tracker_event_order.size)
        if self.star_tracker_residual.shape != (st_count, 3):
            raise ValueError("star_tracker_residual must have shape [S,3]")
        if self.star_tracker_S.shape != (st_count, 3, 3):
            raise ValueError("star_tracker_S must have shape [S,3,3]")
        if not isinstance(self.final_state, MEKFState):
            raise TypeError("final_state must be an MEKFState")


def validate_event_table(table: MEKFEventTable) -> None:
    event_count = int(table.trajectory_id.size)
    for name in (
        "sensor_code",
        "measurement_time_s",
        "arrival_time_s",
        "event_order",
        "valid",
        "payload_index",
    ):
        if int(getattr(table, name).size) != event_count:
            raise ValueError(f"{name} length must equal trajectory_id length")
    if table.gyro_omega_rad_s.shape[1:] != (3,):
        raise ValueError("gyro_omega_rad_s must have shape [G,3]")
    if table.star_tracker_q_NB.shape[1:] != (4,):
        raise ValueError("star_tracker_q_NB must have shape [S,4]")
    if table.star_tracker_R_rad2.shape[1:] != (3, 3):
        raise ValueError("star_tracker_R_rad2 must have shape [S,3,3]")
    for name in (
        "measurement_time_s",
        "arrival_time_s",
        "gyro_omega_rad_s",
        "star_tracker_q_NB",
        "star_tracker_R_rad2",
    ):
        _require_finite(getattr(table, name), name)
    if np.any(table.measurement_time_s < 0.0) or np.any(table.arrival_time_s < 0.0):
        raise ValueError("event times must be nonnegative")
    if not np.array_equal(table.arrival_time_s, table.measurement_time_s):
        raise ValueError("Gate B1 requires exact zero latency for every event")
    allowed = np.isin(table.sensor_code, np.array([_GYRO, _STAR_TRACKER], dtype=np.int16))
    if not np.all(allowed):
        raise ValueError("sensor_code contains an unknown code")
    if np.any(table.payload_index < 0):
        raise ValueError("payload_index must be nonnegative")
    _require_normalized_quaternions(table.star_tracker_q_NB, "star_tracker_q_NB")
    _require_spd_stack(table.star_tracker_R_rad2, "star_tracker_R_rad2")

    gyro_rows = table.sensor_code == _GYRO
    star_rows = table.sensor_code == _STAR_TRACKER
    gyro_indices = table.payload_index[gyro_rows]
    star_indices = table.payload_index[star_rows]
    if not np.array_equal(
        np.sort(gyro_indices), np.arange(table.gyro_omega_rad_s.shape[0], dtype=np.int64)
    ):
        raise ValueError("gyro payloads must have one-to-one event ownership")
    if not np.array_equal(
        np.sort(star_indices), np.arange(table.star_tracker_q_NB.shape[0], dtype=np.int64)
    ):
        raise ValueError("star-tracker payloads must have one-to-one event ownership")
    if table.star_tracker_q_NB.shape[0] != table.star_tracker_R_rad2.shape[0]:
        raise ValueError("star-tracker quaternion and covariance counts must match")

    for trajectory_id in np.unique(table.trajectory_id):
        rows = np.flatnonzero(table.trajectory_id == trajectory_id)
        times = table.arrival_time_s[rows]
        orders = table.event_order[rows]
        if np.unique(orders).size != orders.size:
            raise ValueError("event_order must be unique within each trajectory")
        expected = np.lexsort((orders, times))
        if not np.array_equal(expected, np.arange(rows.size)):
            raise ValueError("events must be sorted by (arrival_time_s, event_order)")
        for time in np.unique(times):
            same_time = rows[times == time]
            codes = table.sensor_code[same_time]
            gyro_positions = np.flatnonzero(codes == _GYRO)
            star_positions = np.flatnonzero(codes == _STAR_TRACKER)
            if gyro_positions.size and star_positions.size:
                if int(np.max(gyro_positions)) > int(np.min(star_positions)):
                    raise ValueError("same-time events must process gyro before star tracker")


def validate_truth_table(table: MEKFTruthTable) -> None:
    trajectory_count = int(table.trajectory_id.size)
    if np.unique(table.trajectory_id).size != trajectory_count:
        raise ValueError("truth trajectory_id values must be unique")
    if table.truth_offsets.shape != (trajectory_count + 1,):
        raise ValueError("truth_offsets must have shape [N+1]")
    if trajectory_count == 0:
        raise ValueError("truth must contain at least one trajectory")
    if int(table.truth_offsets[0]) != 0:
        raise ValueError("truth_offsets must start at zero")
    if np.any(np.diff(table.truth_offsets) <= 0):
        raise ValueError("each trajectory must own at least one truth sample")
    total = int(table.truth_offsets[-1])
    if total != int(table.truth_time_s.size):
        raise ValueError("truth_offsets final value must equal truth sample count")
    if table.q_true_NB.shape != (total, 4):
        raise ValueError("q_true_NB must have shape [total,4]")
    if table.gyro_bias_rad_s.shape != (total, 3):
        raise ValueError("gyro_bias_rad_s must have shape [total,3]")
    if table.omega_true_rad_s.shape != (total, 3):
        raise ValueError("omega_true_rad_s must have shape [total,3]")
    for name in ("truth_time_s", "q_true_NB", "gyro_bias_rad_s", "omega_true_rad_s"):
        _require_finite(getattr(table, name), name)
    if np.any(table.truth_time_s < 0.0):
        raise ValueError("truth_time_s must be nonnegative")
    _require_normalized_quaternions(table.q_true_NB, "q_true_NB")
    for index in range(trajectory_count):
        start = int(table.truth_offsets[index])
        stop = int(table.truth_offsets[index + 1])
        if np.any(np.diff(table.truth_time_s[start:stop]) <= 0.0):
            raise ValueError("truth time must be strictly increasing per trajectory")


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError) as error:
        raise ValueError("manifest must be finite canonical-JSON data") from error


def _manifest_identity(manifest: Mapping[str, Any]) -> dict[str, Any]:
    identity = copy.deepcopy(dict(manifest))
    identity.pop("semantic_hashes", None)
    return identity


def validate_generator_id(generator_id: Any) -> str:
    """Return a strict, deterministic dataset-generator identity."""

    if not isinstance(generator_id, str):
        raise ValueError("manifest generator_id must be a string")
    if not generator_id or generator_id != generator_id.strip():
        raise ValueError("manifest generator_id must be nonempty with no surrounding whitespace")
    if _GENERATOR_ID_PATTERN.fullmatch(generator_id) is None:
        raise ValueError(
            "manifest generator_id must match "
            "'<lowercase-family>-v<positive-integer>'"
        )
    return generator_id


def _validated_manifest_identity(manifest: Mapping[str, Any]) -> dict[str, Any]:
    identity = _manifest_identity(manifest)
    required = {
        "schema_version": SCHEMA_VERSION,
        "seed_policy_version": SEED_POLICY_VERSION,
        "convention_id": CONVENTION_ID,
    }
    for key, expected in required.items():
        if identity.get(key) != expected:
            raise ValueError(f"manifest {key} must equal {expected!r}")
    validate_generator_id(identity.get("generator_id"))
    return identity


def _canonical_array(array: np.ndarray) -> np.ndarray:
    dtype = array.dtype
    if dtype.hasobject:
        raise TypeError("object arrays are forbidden")
    canonical_dtype = dtype.newbyteorder("<") if dtype.itemsize > 1 else dtype
    return np.ascontiguousarray(array.astype(canonical_dtype, copy=False))


def _hash_named_arrays(domain: str, arrays: Sequence[tuple[str, np.ndarray]]) -> str:
    digest = hashlib.sha256()
    digest.update(domain.encode("ascii") + b"\0")
    for name, raw in arrays:
        array = _canonical_array(raw)
        metadata = {
            "dtype": array.dtype.str,
            "field": name,
            "shape": list(array.shape),
        }
        encoded = _canonical_json_bytes(metadata)
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
        payload = array.tobytes(order="C")
        digest.update(len(payload).to_bytes(8, "little"))
        digest.update(payload)
    return digest.hexdigest()


def compute_semantic_hashes(
    dataset: MEKFDataset, manifest: Mapping[str, Any]
) -> SemanticHashes:
    if not isinstance(dataset, MEKFDataset):
        raise TypeError("dataset must be an MEKFDataset")
    truth_hash = _hash_named_arrays(
        "p1a-truth-v1",
        [
            ("trajectory_id", dataset.truth.trajectory_id),
            ("truth_offsets", dataset.truth.truth_offsets),
            ("truth_time_s", dataset.truth.truth_time_s),
            ("q_true_NB", dataset.truth.q_true_NB),
            ("gyro_bias_rad_s", dataset.truth.gyro_bias_rad_s),
            ("omega_true_rad_s", dataset.truth.omega_true_rad_s),
        ],
    )
    sensor_payload_hash = _hash_named_arrays(
        "p1a-sensor-payload-v1",
        [
            ("gyro_omega_rad_s", dataset.events.gyro_omega_rad_s),
            ("star_tracker_q_NB", dataset.events.star_tracker_q_NB),
            ("star_tracker_R_rad2", dataset.events.star_tracker_R_rad2),
        ],
    )
    event_order_hash = _hash_named_arrays(
        "p1a-event-order-v1",
        [
            ("trajectory_id", dataset.events.trajectory_id),
            ("sensor_code", dataset.events.sensor_code),
            ("measurement_time_s", dataset.events.measurement_time_s),
            ("arrival_time_s", dataset.events.arrival_time_s),
            ("event_order", dataset.events.event_order),
            ("valid", dataset.events.valid),
            ("payload_index", dataset.events.payload_index),
        ],
    )
    manifest_hash = hashlib.sha256(_canonical_json_bytes(_manifest_identity(manifest))).hexdigest()
    combined = hashlib.sha256()
    combined.update(b"p1a-dataset-v1\0")
    for name, value in (
        ("truth_hash", truth_hash),
        ("sensor_payload_hash", sensor_payload_hash),
        ("event_order_hash", event_order_hash),
    ):
        combined.update(name.encode("ascii") + b"=" + value.encode("ascii") + b"\0")
    dataset_hash = combined.hexdigest()
    return SemanticHashes(
        truth_hash=truth_hash,
        sensor_payload_hash=sensor_payload_hash,
        event_order_hash=event_order_hash,
        manifest_hash=manifest_hash,
        dataset_hash=dataset_hash,
    )


def _event_arrays(table: MEKFEventTable) -> dict[str, np.ndarray]:
    return {field.name: getattr(table, field.name) for field in fields(table)}


def _truth_arrays(table: MEKFTruthTable) -> dict[str, np.ndarray]:
    return {field.name: getattr(table, field.name) for field in fields(table)}


def save_event_dataset(
    directory: os.PathLike[str] | str,
    dataset: MEKFDataset,
    manifest: Mapping[str, Any],
) -> SemanticHashes:
    """Write the three-file deterministic Gate B1 artifact."""

    target = Path(directory)
    if target.exists():
        if not target.is_dir() or any(target.iterdir()):
            raise FileExistsError(f"serialization target must be a new empty directory: {target}")
    else:
        target.mkdir(parents=True, exist_ok=False)
    identity = _validated_manifest_identity(manifest)
    hashes = compute_semantic_hashes(dataset, identity)
    serialized_manifest = copy.deepcopy(identity)
    serialized_manifest["semantic_hashes"] = hashes.as_dict()
    (target / "manifest.json").write_bytes(_canonical_json_bytes(serialized_manifest))
    np.savez(target / "truth.npz", **_truth_arrays(dataset.truth))
    np.savez(target / "events.npz", **_event_arrays(dataset.events))
    return hashes


def _load_exact_npz(path: Path, expected: Sequence[str]) -> dict[str, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != set(expected):
                raise ValueError(f"{path.name} has missing or unexpected fields")
            result = {name: np.array(archive[name], copy=True) for name in expected}
    except (OSError, ValueError, TypeError) as error:
        raise ValueError(f"failed to load strict NPZ artifact {path.name}") from error
    if any(array.dtype.hasobject for array in result.values()):
        raise ValueError(f"{path.name} contains forbidden object arrays")
    return result


def load_event_dataset(
    directory: os.PathLike[str] | str,
    *,
    expected_generator_id: str | None = None,
) -> tuple[MEKFDataset, dict[str, Any], SemanticHashes]:
    """Strictly load and verify a complete Gate B1 artifact and its identity."""

    source = Path(directory)
    expected_files = {"manifest.json", "truth.npz", "events.npz"}
    if not source.is_dir() or {path.name for path in source.iterdir()} != expected_files:
        raise ValueError("artifact directory must contain exactly manifest.json, truth.npz, events.npz")
    try:
        manifest = json.loads((source / "manifest.json").read_text(encoding="ascii"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("manifest.json is not valid canonical JSON") from error
    if not isinstance(manifest, dict):
        raise ValueError("manifest root must be an object")
    if _canonical_json_bytes(manifest) != (source / "manifest.json").read_bytes():
        raise ValueError("manifest.json is not in canonical sorted compact form")
    identity = _validated_manifest_identity(manifest)
    if expected_generator_id is not None:
        expected = validate_generator_id(expected_generator_id)
        if identity["generator_id"] != expected:
            raise ValueError(
                "manifest generator_id mismatch: "
                f"expected {expected!r}, recorded {identity['generator_id']!r}"
            )
    recorded = manifest.get("semantic_hashes")
    if not isinstance(recorded, dict) or set(recorded) != set(_HASH_KEYS):
        raise ValueError("manifest semantic_hashes block is missing or malformed")
    event_names = [field.name for field in fields(MEKFEventTable)]
    truth_names = [field.name for field in fields(MEKFTruthTable)]
    events = MEKFEventTable(**_load_exact_npz(source / "events.npz", event_names))
    truth = MEKFTruthTable(**_load_exact_npz(source / "truth.npz", truth_names))
    dataset = MEKFDataset(events=events, truth=truth)
    computed = compute_semantic_hashes(dataset, manifest)
    if computed.as_dict() != recorded:
        raise ValueError("artifact semantic hash mismatch")
    return dataset, manifest, computed


def _validated_trajectory_ids(trajectory_ids: Sequence[int] | np.ndarray) -> np.ndarray:
    values = np.asarray(trajectory_ids)
    if values.ndim != 1:
        raise ValueError("trajectory_ids must be one-dimensional")
    if values.dtype.kind not in "iu":
        raise TypeError("trajectory_ids must be integers")
    if values.size < 3:
        raise ValueError("at least three trajectories are required for train/val/test")
    converted = values.astype(np.int64, copy=True)
    if np.unique(converted).size != converted.size:
        raise ValueError("trajectory_ids must not contain duplicates")
    return converted


def _split_score(split_seed: int, trajectory_id: int) -> bytes:
    return hashlib.sha256(
        b"p1a-whole-trajectory-split-v1\0"
        + str(int(split_seed)).encode("ascii")
        + b"\0"
        + str(int(trajectory_id)).encode("ascii")
    ).digest()


def split_trajectory_ids(
    trajectory_ids: Sequence[int] | np.ndarray,
    *,
    split_seed: int,
    train_fraction: float = 0.6,
    val_fraction: float = 0.2,
    test_fraction: float = 0.2,
) -> TrajectorySplit:
    """Deterministically split whole trajectories, independent of input order."""

    ids = _validated_trajectory_ids(trajectory_ids)
    fractions = np.array([train_fraction, val_fraction, test_fraction], dtype=np.float64)
    if not np.all(np.isfinite(fractions)) or np.any(fractions <= 0.0):
        raise ValueError("split fractions must be finite and strictly positive")
    if not math.isclose(float(np.sum(fractions)), 1.0, rel_tol=0.0, abs_tol=1.0e-12):
        raise ValueError("split fractions must sum to one")
    if ids.size < 3:
        raise ValueError("too few trajectories for three nonempty splits")
    raw_counts = fractions * ids.size
    counts = np.floor(raw_counts).astype(np.int64)
    counts = np.maximum(counts, 1)
    while int(np.sum(counts)) > ids.size:
        candidates = np.flatnonzero(counts > 1)
        if candidates.size == 0:
            raise ValueError("too few trajectories for requested split fractions")
        index = int(candidates[np.argmin(raw_counts[candidates] - counts[candidates])])
        counts[index] -= 1
    while int(np.sum(counts)) < ids.size:
        index = int(np.argmax(raw_counts - counts))
        counts[index] += 1
    ordered = np.array(
        sorted((int(item) for item in ids), key=lambda item: (_split_score(split_seed, item), item)),
        dtype=np.int64,
    )
    first = int(counts[0])
    second = first + int(counts[1])
    return TrajectorySplit(
        train_ids=ordered[:first],
        val_ids=ordered[first:second],
        test_ids=ordered[second:],
        split_seed=int(split_seed),
    )


def select_trajectories(
    dataset: MEKFDataset, trajectory_ids: Sequence[int] | np.ndarray
) -> MEKFDataset:
    """Select complete truth and event records, compacting payload indices."""

    if not isinstance(dataset, MEKFDataset):
        raise TypeError("dataset must be an MEKFDataset")
    requested_raw = np.asarray(trajectory_ids)
    if requested_raw.ndim != 1 or requested_raw.dtype.kind not in "iu":
        raise TypeError("trajectory_ids must be a one-dimensional integer sequence")
    requested = requested_raw.astype(np.int64, copy=False)
    if requested.size == 0 or np.unique(requested).size != requested.size:
        raise ValueError("trajectory_ids must be nonempty and unique")
    available = set(map(int, dataset.truth.trajectory_id))
    if not set(map(int, requested)).issubset(available):
        raise ValueError("requested trajectory ID is not present")
    requested_set = set(map(int, requested))
    ordered_ids = np.array(
        [item for item in dataset.truth.trajectory_id if int(item) in requested_set], dtype=np.int64
    )

    truth_indices: list[np.ndarray] = []
    offsets = [0]
    truth_lookup = {int(item): index for index, item in enumerate(dataset.truth.trajectory_id)}
    for trajectory_id in ordered_ids:
        index = truth_lookup[int(trajectory_id)]
        start = int(dataset.truth.truth_offsets[index])
        stop = int(dataset.truth.truth_offsets[index + 1])
        rows = np.arange(start, stop, dtype=np.int64)
        truth_indices.append(rows)
        offsets.append(offsets[-1] + rows.size)
    truth_rows = np.concatenate(truth_indices)
    truth = MEKFTruthTable(
        trajectory_id=ordered_ids,
        truth_offsets=np.array(offsets, dtype=np.int64),
        truth_time_s=dataset.truth.truth_time_s[truth_rows],
        q_true_NB=dataset.truth.q_true_NB[truth_rows],
        gyro_bias_rad_s=dataset.truth.gyro_bias_rad_s[truth_rows],
        omega_true_rad_s=dataset.truth.omega_true_rad_s[truth_rows],
    )

    event_rows = np.flatnonzero(np.isin(dataset.events.trajectory_id, ordered_ids))
    codes = dataset.events.sensor_code[event_rows]
    old_payload = dataset.events.payload_index[event_rows]
    gyro_old = old_payload[codes == _GYRO]
    star_old = old_payload[codes == _STAR_TRACKER]
    new_payload = np.empty(event_rows.size, dtype=np.int64)
    new_payload[codes == _GYRO] = np.arange(gyro_old.size, dtype=np.int64)
    new_payload[codes == _STAR_TRACKER] = np.arange(star_old.size, dtype=np.int64)
    events = MEKFEventTable(
        trajectory_id=dataset.events.trajectory_id[event_rows],
        sensor_code=codes,
        measurement_time_s=dataset.events.measurement_time_s[event_rows],
        arrival_time_s=dataset.events.arrival_time_s[event_rows],
        event_order=dataset.events.event_order[event_rows],
        valid=dataset.events.valid[event_rows],
        payload_index=new_payload,
        gyro_omega_rad_s=dataset.events.gyro_omega_rad_s[gyro_old],
        star_tracker_q_NB=dataset.events.star_tracker_q_NB[star_old],
        star_tracker_R_rad2=dataset.events.star_tracker_R_rad2[star_old],
    )
    return MEKFDataset(events=events, truth=truth)


def replay_trajectory(
    event_table: MEKFEventTable,
    trajectory_id: int,
    initial_state: MEKFState,
    initial_time_s: float,
    Q_c: np.ndarray,
) -> ReplayResult:
    """Replay one typed stream by composing only the frozen Gate A API."""

    if not isinstance(event_table, MEKFEventTable):
        raise TypeError("event_table must be an MEKFEventTable")
    if not isinstance(initial_state, MEKFState):
        raise TypeError("initial_state must be an MEKFState")
    current_time = float(initial_time_s)
    if not np.isfinite(current_time) or current_time < 0.0:
        raise ValueError("initial_time_s must be finite and nonnegative")
    process_noise = np.asarray(Q_c)
    if process_noise.dtype != np.dtype(np.float64) or process_noise.shape != (6, 6):
        raise TypeError("Q_c must be a float64 array with shape [6,6]")
    process_noise = np.array(process_noise, copy=True)
    rows = np.flatnonzero(event_table.trajectory_id == np.int64(trajectory_id))
    if rows.size == 0:
        raise ValueError("trajectory_id is not present in event_table")

    current_state = initial_state
    times: list[float] = []
    orders: list[int] = []
    sensors: list[int] = []
    quaternions: list[np.ndarray] = []
    biases: list[np.ndarray] = []
    covariances: list[np.ndarray] = []
    attitude_steps: list[float] = []
    st_orders: list[int] = []
    st_residuals: list[np.ndarray] = []
    st_covariances: list[np.ndarray] = []

    for row in rows:
        code = event_table.sensor_code[row]
        event_time = float(event_table.measurement_time_s[row])
        before_q = current_state.q_NB
        if code == _GYRO:
            if not bool(event_table.valid[row]):
                raise ValueError("invalid gyro events are forbidden")
            if not event_time > current_time:
                raise ValueError("gyro event time must be strictly later than filter time")
            payload = int(event_table.payload_index[row])
            result = propagate_state(
                current_state,
                event_table.gyro_omega_rad_s[payload],
                event_time - current_time,
                process_noise,
            )
            current_state = result.state
            current_time = event_time
        elif code == _STAR_TRACKER:
            if event_time != current_time:
                raise ValueError("star-tracker event time must equal current propagated time")
            if not bool(event_table.valid[row]):
                continue
            payload = int(event_table.payload_index[row])
            result = star_tracker_update(
                current_state,
                event_table.star_tracker_q_NB[payload],
                event_table.star_tracker_R_rad2[payload],
            )
            current_state = result.state
            st_orders.append(int(event_table.event_order[row]))
            st_residuals.append(result.residual)
            st_covariances.append(result.S)
        else:  # validated tables make this unreachable
            raise ValueError("unknown sensor code")
        times.append(current_time)
        orders.append(int(event_table.event_order[row]))
        sensors.append(int(code))
        quaternions.append(current_state.q_NB)
        biases.append(current_state.b_g)
        covariances.append(current_state.P)
        attitude_steps.append(quat_geodesic_angle(before_q, current_state.q_NB))

    count = len(times)
    return ReplayResult(
        trajectory_id=int(trajectory_id),
        processed_event_count=count,
        event_time_s=np.array(times, dtype=np.float64),
        event_order=np.array(orders, dtype=np.int64),
        sensor_code=np.array(sensors, dtype=np.int16),
        q_NB_history=np.array(quaternions, dtype=np.float64).reshape(count, 4),
        b_g_history=np.array(biases, dtype=np.float64).reshape(count, 3),
        P_history=np.array(covariances, dtype=np.float64).reshape(count, 6, 6),
        attitude_step_rad=np.array(attitude_steps, dtype=np.float64),
        star_tracker_event_order=np.array(st_orders, dtype=np.int64),
        star_tracker_residual=np.array(st_residuals, dtype=np.float64).reshape(-1, 3),
        star_tracker_S=np.array(st_covariances, dtype=np.float64).reshape(-1, 3, 3),
        final_state=current_state,
    )
