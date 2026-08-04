"""Unregistered typed-event MEKF replay bridge for Phase 1A Gate D1.

The bridge deliberately does not implement :class:`ModelAdapter`: that API is
based on dense sequence arrays and cannot carry the frozen typed event table or
the compact star-tracker innovation evidence without loss.  This module owns no
filter math.  It calls the Gate B1 replay entry point and packages its result as
an immutable, truth-free estimator artifact.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, fields
from typing import Any, Mapping

import numpy as np

from bench.estimators.mekf import MEKFState
from bench.tasks.generator.mekf_events import (
    CONVENTION_ID,
    SCHEMA_VERSION,
    MEKFEventTable,
    ReplayResult,
    SemanticHashes,
    SensorCode,
    replay_trajectory,
    validate_generator_id,
)


ADAPTER_ID = "mekf-event-replay-bridge"
ADAPTER_VERSION = "p1a-gate-d1-v1"
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")
_GYRO = np.int16(SensorCode.GYRO)
_STAR_TRACKER = np.int16(SensorCode.STAR_TRACKER)


def _require_sha256(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase 64-character SHA-256 hex digest")
    return value


def _readonly_array(
    value: Any,
    *,
    dtype: np.dtype[Any] | type[np.generic],
    ndim: int,
    name: str,
) -> np.ndarray:
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


def _require_finite(value: np.ndarray, *, name: str) -> None:
    if not np.all(np.isfinite(value)):
        raise ValueError(f"{name} must contain only finite values")


def _require_spd_stack(value: np.ndarray, *, name: str) -> None:
    for index in range(value.shape[0]):
        matrix = value[index]
        if not np.array_equal(matrix, matrix.T):
            raise ValueError(f"{name}[{index}] must be exactly symmetric")
        try:
            np.linalg.cholesky(matrix)
        except np.linalg.LinAlgError as error:
            raise ValueError(f"{name}[{index}] must be positive definite") from error


@dataclass(frozen=True)
class DatasetIdentity:
    """Verified Gate B semantic identity, copied without recomputation."""

    schema_version: str
    generator_id: str
    convention_id: str
    truth_hash: str
    sensor_payload_hash: str
    event_order_hash: str
    manifest_hash: str
    dataset_hash: str

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"schema_version must equal {SCHEMA_VERSION!r}")
        object.__setattr__(self, "generator_id", validate_generator_id(self.generator_id))
        if self.convention_id != CONVENTION_ID:
            raise ValueError(f"convention_id must equal {CONVENTION_ID!r}")
        for name in (
            "truth_hash",
            "sensor_payload_hash",
            "event_order_hash",
            "manifest_hash",
            "dataset_hash",
        ):
            object.__setattr__(self, name, _require_sha256(getattr(self, name), name=name))

    @classmethod
    def from_verified(
        cls,
        manifest: Mapping[str, Any],
        semantic_hashes: SemanticHashes,
    ) -> "DatasetIdentity":
        """Copy identity returned by the strict Gate B loader or generator."""

        if not isinstance(manifest, Mapping):
            raise TypeError("manifest must be a mapping")
        if not isinstance(semantic_hashes, SemanticHashes):
            raise TypeError("semantic_hashes must be a SemanticHashes instance")
        recorded = manifest.get("semantic_hashes")
        if recorded is not None and recorded != semantic_hashes.as_dict():
            raise ValueError("manifest semantic_hashes do not match the verified hashes")
        return cls(
            schema_version=manifest.get("schema_version"),
            generator_id=manifest.get("generator_id"),
            convention_id=manifest.get("convention_id"),
            **semantic_hashes.as_dict(),
        )

    def as_dict(self) -> dict[str, str]:
        return {field.name: str(getattr(self, field.name)) for field in fields(self)}


@dataclass(frozen=True)
class ArtifactProvenance:
    """Dataset identity plus the non-semantic adapter implementation identity."""

    schema_version: str
    generator_id: str
    convention_id: str
    truth_hash: str
    sensor_payload_hash: str
    event_order_hash: str
    manifest_hash: str
    dataset_hash: str
    adapter_id: str
    adapter_version: str

    def __post_init__(self) -> None:
        identity_names = tuple(field.name for field in fields(DatasetIdentity))
        DatasetIdentity(**{name: getattr(self, name) for name in identity_names})
        if self.adapter_id != ADAPTER_ID:
            raise ValueError(f"adapter_id must equal {ADAPTER_ID!r}")
        if self.adapter_version != ADAPTER_VERSION:
            raise ValueError(f"adapter_version must equal {ADAPTER_VERSION!r}")

    @classmethod
    def from_identity(cls, identity: DatasetIdentity) -> "ArtifactProvenance":
        if not isinstance(identity, DatasetIdentity):
            raise TypeError("identity must be a DatasetIdentity")
        return cls(
            **identity.as_dict(),
            adapter_id=ADAPTER_ID,
            adapter_version=ADAPTER_VERSION,
        )

    @property
    def dataset_identity(self) -> DatasetIdentity:
        identity_names = tuple(field.name for field in fields(DatasetIdentity))
        return DatasetIdentity(**{name: getattr(self, name) for name in identity_names})


@dataclass(frozen=True)
class MEKFReplayArtifact:
    """Read-only posterior trace and compact star-tracker update evidence."""

    trajectory_id: int
    event_index: np.ndarray
    event_order: np.ndarray
    timestamp_s: np.ndarray
    sensor_code: np.ndarray
    q_hat_NB: np.ndarray
    b_hat_rad_s: np.ndarray
    P: np.ndarray
    st_event_index: np.ndarray
    st_event_order: np.ndarray
    st_timestamp_s: np.ndarray
    st_residual: np.ndarray
    st_S: np.ndarray
    final_state: MEKFState
    processed_event_count: int
    gyro_event_count: int
    star_tracker_update_count: int
    provenance: ArtifactProvenance

    def __post_init__(self) -> None:
        if not isinstance(self.trajectory_id, (int, np.integer)):
            raise TypeError("trajectory_id must be an integer")
        object.__setattr__(self, "trajectory_id", int(self.trajectory_id))
        specifications = {
            "event_index": (np.int64, 1),
            "event_order": (np.int64, 1),
            "timestamp_s": (np.float64, 1),
            "sensor_code": (np.int16, 1),
            "q_hat_NB": (np.float64, 2),
            "b_hat_rad_s": (np.float64, 2),
            "P": (np.float64, 3),
            "st_event_index": (np.int64, 1),
            "st_event_order": (np.int64, 1),
            "st_timestamp_s": (np.float64, 1),
            "st_residual": (np.float64, 2),
            "st_S": (np.float64, 3),
        }
        for name, (dtype, ndim) in specifications.items():
            value = _readonly_array(
                getattr(self, name), dtype=dtype, ndim=ndim, name=name
            )
            if value.dtype.kind == "f":
                _require_finite(value, name=name)
            object.__setattr__(self, name, value)

        event_count = int(self.processed_event_count)
        gyro_count = int(self.gyro_event_count)
        st_count = int(self.star_tracker_update_count)
        for name, value in (
            ("processed_event_count", event_count),
            ("gyro_event_count", gyro_count),
            ("star_tracker_update_count", st_count),
        ):
            if not isinstance(getattr(self, name), (int, np.integer)) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
            object.__setattr__(self, name, value)
        if event_count == 0:
            raise ValueError("artifact must contain at least one processed event")
        event_fields = (
            "event_index",
            "event_order",
            "timestamp_s",
            "sensor_code",
            "q_hat_NB",
            "b_hat_rad_s",
            "P",
        )
        if any(getattr(self, name).shape[0] != event_count for name in event_fields):
            raise ValueError("event artifact lengths must match processed_event_count")
        if self.q_hat_NB.shape[1:] != (4,):
            raise ValueError("q_hat_NB must have shape [E,4]")
        if self.b_hat_rad_s.shape[1:] != (3,):
            raise ValueError("b_hat_rad_s must have shape [E,3]")
        if self.P.shape[1:] != (6, 6):
            raise ValueError("P must have shape [E,6,6]")
        if np.any(self.event_index < 0) or np.unique(self.event_index).size != event_count:
            raise ValueError("event_index must contain unique nonnegative table indices")
        if np.unique(self.event_order).size != event_count:
            raise ValueError("event_order must be unique within the trajectory")
        if not np.array_equal(
            np.lexsort((self.event_order, self.timestamp_s)),
            np.arange(event_count, dtype=np.int64),
        ):
            raise ValueError("artifact events must remain in timestamp/event_order order")
        if not np.all(np.isin(self.sensor_code, np.array([_GYRO, _STAR_TRACKER]))):
            raise ValueError("sensor_code contains an unknown code")
        if not np.allclose(
            np.linalg.norm(self.q_hat_NB, axis=1), 1.0, rtol=0.0, atol=2.0e-13
        ):
            raise ValueError("q_hat_NB must contain normalized quaternions")
        _require_spd_stack(self.P, name="P")

        star_positions = np.flatnonzero(self.sensor_code == _STAR_TRACKER)
        if gyro_count != int(np.count_nonzero(self.sensor_code == _GYRO)):
            raise ValueError("gyro_event_count does not match sensor_code")
        if st_count != int(star_positions.size):
            raise ValueError("star_tracker_update_count does not match sensor_code")
        st_fields = (
            "st_event_index",
            "st_event_order",
            "st_timestamp_s",
            "st_residual",
            "st_S",
        )
        if any(getattr(self, name).shape[0] != st_count for name in st_fields):
            raise ValueError("compact ST artifact lengths must match update count")
        if self.st_residual.shape[1:] != (3,):
            raise ValueError("st_residual must have shape [S,3]")
        if self.st_S.shape[1:] != (3, 3):
            raise ValueError("st_S must have shape [S,3,3]")
        if not np.array_equal(self.st_event_index, self.event_index[star_positions]):
            raise ValueError("st_event_index does not match processed ST events")
        if not np.array_equal(self.st_event_order, self.event_order[star_positions]):
            raise ValueError("st_event_order does not match processed ST events")
        if not np.array_equal(self.st_timestamp_s, self.timestamp_s[star_positions]):
            raise ValueError("st_timestamp_s does not match processed ST events")
        _require_spd_stack(self.st_S, name="st_S")

        if not isinstance(self.final_state, MEKFState):
            raise TypeError("final_state must be an MEKFState")
        final_copy = MEKFState(
            q_NB=self.final_state.q_NB,
            b_g=self.final_state.b_g,
            P=self.final_state.P,
        )
        if not np.array_equal(final_copy.q_NB, self.q_hat_NB[-1]):
            raise ValueError("final_state quaternion does not match final posterior")
        if not np.array_equal(final_copy.b_g, self.b_hat_rad_s[-1]):
            raise ValueError("final_state bias does not match final posterior")
        if not np.array_equal(final_copy.P, self.P[-1]):
            raise ValueError("final_state covariance does not match final posterior")
        object.__setattr__(self, "final_state", final_copy)
        if not isinstance(self.provenance, ArtifactProvenance):
            raise TypeError("provenance must be ArtifactProvenance")


class MEKFEventReplayBridge:
    """Frozen, no-training bridge from typed events to a MEKF artifact."""

    adapter_id = ADAPTER_ID
    adapter_version = ADAPTER_VERSION
    is_frozen = True
    supports_training = False

    def __init__(self, *, expected_dataset_identity: DatasetIdentity | None = None) -> None:
        if expected_dataset_identity is not None and not isinstance(
            expected_dataset_identity, DatasetIdentity
        ):
            raise TypeError("expected_dataset_identity must be a DatasetIdentity or None")
        self._expected_dataset_identity = expected_dataset_identity

    def replay_events(
        self,
        event_table: MEKFEventTable,
        trajectory_id: int,
        initial_state: MEKFState,
        initial_time_s: float,
        Q_c: np.ndarray,
        dataset_identity: DatasetIdentity,
    ) -> MEKFReplayArtifact:
        """Replay one trajectory and return exact posterior/update evidence."""

        if not isinstance(dataset_identity, DatasetIdentity):
            raise TypeError("dataset_identity must be a DatasetIdentity")
        if (
            self._expected_dataset_identity is not None
            and dataset_identity != self._expected_dataset_identity
        ):
            raise ValueError("dataset_identity does not match the bridge expectation")
        result = replay_trajectory(
            event_table,
            trajectory_id,
            initial_state,
            initial_time_s,
            Q_c,
        )
        if not isinstance(result, ReplayResult):
            raise TypeError("replay_trajectory must return ReplayResult")
        return self._package_result(event_table, result, dataset_identity)

    @staticmethod
    def _package_result(
        event_table: MEKFEventTable,
        result: ReplayResult,
        dataset_identity: DatasetIdentity,
    ) -> MEKFReplayArtifact:
        trajectory_rows = np.flatnonzero(
            event_table.trajectory_id == np.int64(result.trajectory_id)
        ).astype(np.int64, copy=False)
        trajectory_orders = event_table.event_order[trajectory_rows]
        sorter = np.argsort(trajectory_orders, kind="stable")
        sorted_orders = trajectory_orders[sorter]
        locations = np.searchsorted(sorted_orders, result.event_order)
        if np.any(locations >= sorted_orders.size) or not np.array_equal(
            sorted_orders[locations], result.event_order
        ):
            raise ValueError("replay event_order cannot be mapped to the input event table")
        event_index = trajectory_rows[sorter[locations]]
        expected_count = int(
            np.count_nonzero(event_table.sensor_code[trajectory_rows] == _GYRO)
            + np.count_nonzero(
                (event_table.sensor_code[trajectory_rows] == _STAR_TRACKER)
                & event_table.valid[trajectory_rows]
            )
        )
        if result.processed_event_count != expected_count:
            raise ValueError("replay processed_event_count does not match valid input events")
        if not np.array_equal(event_table.event_order[event_index], result.event_order):
            raise ValueError("replay event_order differs from input event rows")
        if not np.array_equal(
            event_table.measurement_time_s[event_index], result.event_time_s
        ):
            raise ValueError("replay timestamps differ from input event rows")
        if not np.array_equal(event_table.sensor_code[event_index], result.sensor_code):
            raise ValueError("replay sensor codes differ from input event rows")

        star_positions = np.flatnonzero(result.sensor_code == _STAR_TRACKER)
        if not np.array_equal(
            result.event_order[star_positions], result.star_tracker_event_order
        ):
            raise ValueError("replay compact ST evidence does not match ST event order")
        return MEKFReplayArtifact(
            trajectory_id=result.trajectory_id,
            event_index=event_index,
            event_order=result.event_order,
            timestamp_s=result.event_time_s,
            sensor_code=result.sensor_code,
            q_hat_NB=result.q_NB_history,
            b_hat_rad_s=result.b_g_history,
            P=result.P_history,
            st_event_index=event_index[star_positions],
            st_event_order=result.star_tracker_event_order,
            st_timestamp_s=result.event_time_s[star_positions],
            st_residual=result.star_tracker_residual,
            st_S=result.star_tracker_S,
            final_state=result.final_state,
            processed_event_count=result.processed_event_count,
            gyro_event_count=int(np.count_nonzero(result.sensor_code == _GYRO)),
            star_tracker_update_count=int(result.star_tracker_event_order.size),
            provenance=ArtifactProvenance.from_identity(dataset_identity),
        )


__all__ = [
    "ADAPTER_ID",
    "ADAPTER_VERSION",
    "ArtifactProvenance",
    "DatasetIdentity",
    "MEKFEventReplayBridge",
    "MEKFReplayArtifact",
]
