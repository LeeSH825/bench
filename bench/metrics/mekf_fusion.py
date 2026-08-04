"""Sensor-specific consistency metrics for Phase 1B MEKF fusion updates."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from bench.estimators.mekf import cholesky_solve_spd
from bench.metrics.mekf import ConsistencySummary, consistency_summary


@dataclass(frozen=True)
class SensorConsistencyEvidence:
    sensor_name: str
    degrees_of_freedom: int
    total_event_count: int
    update_count: int
    skip_count: int
    nis: np.ndarray
    summary: ConsistencySummary

    def __post_init__(self) -> None:
        if not isinstance(self.sensor_name, str) or not self.sensor_name:
            raise ValueError("sensor_name must be nonempty")
        if self.degrees_of_freedom not in (2, 3):
            raise ValueError("sensor degrees of freedom must be two or three")
        if self.update_count != self.nis.size:
            raise ValueError("update count must equal compact NIS count")
        if self.total_event_count != self.update_count + self.skip_count:
            raise ValueError("sensor total must equal updates plus skips")
        value = np.array(self.nis, dtype=np.float64, order="C", copy=True)
        value.setflags(write=False)
        object.__setattr__(self, "nis", value)


def _require_float64(value: np.ndarray, ndim: int, name: str) -> np.ndarray:
    if not isinstance(value, np.ndarray) or value.dtype != np.dtype(np.float64):
        raise TypeError(f"{name} must be an exact float64 ndarray")
    if value.ndim != ndim:
        raise ValueError(f"{name} must have rank {ndim}")
    if not np.all(np.isfinite(value)):
        raise ValueError(f"{name} must contain only finite values")
    return value


def sensor_nis(
    residual: np.ndarray,
    innovation_covariance: np.ndarray,
    *,
    degrees_of_freedom: int,
    sensor_name: str,
) -> np.ndarray:
    """Return compact per-update NIS via the frozen strict Cholesky path."""

    dimension = int(degrees_of_freedom)
    if dimension not in (2, 3):
        raise ValueError("degrees_of_freedom must be two or three")
    r = _require_float64(residual, 2, "residual")
    s = _require_float64(innovation_covariance, 3, "innovation_covariance")
    if r.shape[0] == 0:
        raise ValueError("NIS requires at least one actual update")
    if r.shape != (s.shape[0], dimension) or s.shape[1:] != (dimension, dimension):
        raise ValueError("residual and innovation covariance shapes do not pair")
    values = np.empty(r.shape[0], dtype=np.float64)
    for index in range(r.shape[0]):
        solved = cholesky_solve_spd(
            s[index], r[index], name=f"{sensor_name}_S[{index}]"
        )
        values[index] = float(r[index] @ solved)
    if not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("NIS must be finite and nonnegative")
    values.setflags(write=False)
    return values


def magnetometer_nis(residual: np.ndarray, innovation_covariance: np.ndarray) -> np.ndarray:
    return sensor_nis(
        residual,
        innovation_covariance,
        degrees_of_freedom=3,
        sensor_name="magnetometer",
    )


def sun_sensor_nis(residual: np.ndarray, innovation_covariance: np.ndarray) -> np.ndarray:
    return sensor_nis(
        residual,
        innovation_covariance,
        degrees_of_freedom=2,
        sensor_name="sun_sensor",
    )


def summarize_sensor_consistency(
    residual: np.ndarray,
    innovation_covariance: np.ndarray,
    *,
    sensor_name: str,
    degrees_of_freedom: int,
    total_event_count: int,
    skip_count: int,
    confidence_level: float = 0.95,
) -> SensorConsistencyEvidence:
    nis = sensor_nis(
        residual,
        innovation_covariance,
        degrees_of_freedom=degrees_of_freedom,
        sensor_name=sensor_name,
    )
    updates = int(nis.size)
    total = int(total_event_count)
    skips = int(skip_count)
    if min(total, skips) < 0:
        raise ValueError("sensor event counts must be nonnegative")
    return SensorConsistencyEvidence(
        sensor_name=sensor_name,
        degrees_of_freedom=int(degrees_of_freedom),
        total_event_count=total,
        update_count=updates,
        skip_count=skips,
        nis=nis,
        summary=consistency_summary(
            nis,
            dof_per_sample=int(degrees_of_freedom),
            confidence_level=float(confidence_level),
        ),
    )


__all__ = [
    "SensorConsistencyEvidence",
    "magnetometer_nis",
    "sensor_nis",
    "summarize_sensor_consistency",
    "sun_sensor_nis",
]
