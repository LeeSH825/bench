"""Canonical Phase 1A metrics for the six-dimensional kinematic MEKF.

The convention is scalar-first Hamilton, active body-to-navigation ``q_NB``
with the right-local error

``q_true = q_hat (x) Exp_q(delta_theta)``.

This module is evaluation-only.  It deliberately has no generator, replay,
runner, registry, model, or visualization dependency.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
from scipy.stats import chi2

from bench.estimators.mekf import (
    assert_positive_definite,
    cholesky_solve_spd,
    quat_inverse,
    quat_log,
    quat_multiply,
    quat_normalize,
)


_ATTITUDE_DIMENSION: Final = 3
_BIAS_DIMENSION: Final = 3
_STATE_DIMENSION: Final = 6
_QUATERNION_DIMENSION: Final = 4


@dataclass(frozen=True)
class RightLocalStateError:
    """Right-local attitude, bias, and combined state error arrays."""

    delta_theta_rad: np.ndarray
    delta_bias_rad_s: np.ndarray
    state_error: np.ndarray


@dataclass(frozen=True)
class BiasErrorSummary:
    """Per-sample bias errors and whole-batch RMSE values in rad/s."""

    per_axis_error_rad_s: np.ndarray
    vector_norm_rad_s: np.ndarray
    per_axis_rmse_rad_s: np.ndarray
    vector_rmse_rad_s: float


@dataclass(frozen=True)
class SPDDiagnostics:
    """Read-only diagnostics for one matrix or a batch of SPD matrices."""

    relative_asymmetry: np.ndarray
    minimum_eigenvalue: np.ndarray
    cholesky_succeeded: np.ndarray
    dimension: int


@dataclass(frozen=True)
class ConsistencySummary:
    """Batch NIS/NEES summary and chi-square interval for the batch sum."""

    count: int
    dof_per_sample: int
    sum: float
    mean: float
    normalized_mean: float
    confidence_level: float
    chi_square_sum_lower: float
    chi_square_sum_upper: float


def _require_float64_array(
    value: object,
    *,
    name: str,
    trailing_shape: tuple[int, ...] | None = None,
    allow_empty: bool = False,
) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{name} must be a numpy.ndarray")
    if value.dtype != np.dtype(np.float64):
        raise TypeError(f"{name} must have dtype float64, got {value.dtype}")
    if trailing_shape is not None:
        if value.ndim < len(trailing_shape) or value.shape[-len(trailing_shape) :] != trailing_shape:
            raise ValueError(
                f"{name} must have trailing shape {trailing_shape}, got {value.shape}"
            )
    if not allow_empty and value.size == 0:
        raise ValueError(f"{name} must be nonempty")
    if not np.all(np.isfinite(value)):
        raise ValueError(f"{name} must contain only finite float64 values")
    return value


def _readonly(value: np.ndarray) -> np.ndarray:
    result = np.array(value, dtype=np.float64, copy=True)
    result.setflags(write=False)
    return result


def _readonly_bool(value: np.ndarray) -> np.ndarray:
    result = np.array(value, dtype=np.bool_, copy=True)
    result.setflags(write=False)
    return result


def _require_same_leading_shape(
    reference: np.ndarray,
    candidate: np.ndarray,
    *,
    reference_trailing_dimensions: int,
    candidate_trailing_dimensions: int,
    reference_name: str,
    candidate_name: str,
) -> tuple[int, ...]:
    reference_shape = reference.shape[:-reference_trailing_dimensions]
    candidate_shape = candidate.shape[:-candidate_trailing_dimensions]
    if reference_shape != candidate_shape:
        raise ValueError(
            f"{reference_name}/{candidate_name} batch shapes must match exactly, "
            f"got {reference_shape} and {candidate_shape}"
        )
    return reference_shape


def _right_local_attitude_error(q_hat_NB: np.ndarray, q_true_NB: np.ndarray) -> np.ndarray:
    batch_shape = _require_same_leading_shape(
        q_hat_NB,
        q_true_NB,
        reference_trailing_dimensions=1,
        candidate_trailing_dimensions=1,
        reference_name="q_hat_NB",
        candidate_name="q_true_NB",
    )
    estimate_rows = q_hat_NB.reshape(-1, _QUATERNION_DIMENSION)
    truth_rows = q_true_NB.reshape(-1, _QUATERNION_DIMENSION)
    errors = np.empty((estimate_rows.shape[0], _ATTITUDE_DIMENSION), dtype=np.float64)
    for index, (estimate, truth) in enumerate(zip(estimate_rows, truth_rows)):
        normalized_estimate = quat_normalize(estimate, name="q_hat_NB")
        normalized_truth = quat_normalize(truth, name="q_true_NB")
        delta_quaternion = quat_multiply(
            quat_inverse(normalized_estimate),
            normalized_truth,
        )
        errors[index] = quat_log(delta_quaternion)
    return errors.reshape(batch_shape + (_ATTITUDE_DIMENSION,))


def right_local_state_error(
    q_hat_NB: np.ndarray,
    b_hat_rad_s: np.ndarray,
    q_true_NB: np.ndarray,
    b_true_rad_s: np.ndarray,
) -> RightLocalStateError:
    """Return the canonical six-dimensional right-local MEKF state error."""

    estimate_q = _require_float64_array(
        q_hat_NB, name="q_hat_NB", trailing_shape=(_QUATERNION_DIMENSION,)
    )
    truth_q = _require_float64_array(
        q_true_NB, name="q_true_NB", trailing_shape=(_QUATERNION_DIMENSION,)
    )
    estimate_bias = _require_float64_array(
        b_hat_rad_s, name="b_hat_rad_s", trailing_shape=(_BIAS_DIMENSION,)
    )
    truth_bias = _require_float64_array(
        b_true_rad_s, name="b_true_rad_s", trailing_shape=(_BIAS_DIMENSION,)
    )
    batch_shape = _require_same_leading_shape(
        estimate_q,
        truth_q,
        reference_trailing_dimensions=1,
        candidate_trailing_dimensions=1,
        reference_name="q_hat_NB",
        candidate_name="q_true_NB",
    )
    for candidate, candidate_name in (
        (estimate_bias, "b_hat_rad_s"),
        (truth_bias, "b_true_rad_s"),
    ):
        if candidate.shape[:-1] != batch_shape:
            raise ValueError(
                f"all state inputs must have batch shape {batch_shape}; "
                f"{candidate_name} has {candidate.shape[:-1]}"
            )
    delta_theta = _right_local_attitude_error(estimate_q, truth_q)
    delta_bias = truth_bias - estimate_bias
    state_error = np.concatenate((delta_theta, delta_bias), axis=-1)
    return RightLocalStateError(
        delta_theta_rad=_readonly(delta_theta),
        delta_bias_rad_s=_readonly(delta_bias),
        state_error=_readonly(state_error),
    )


def attitude_geodesic_error_rad(
    q_hat_NB: np.ndarray,
    q_true_NB: np.ndarray,
) -> np.ndarray:
    """Return the per-sample shortest right-local attitude angle in radians."""

    estimate = _require_float64_array(
        q_hat_NB, name="q_hat_NB", trailing_shape=(_QUATERNION_DIMENSION,)
    )
    truth = _require_float64_array(
        q_true_NB, name="q_true_NB", trailing_shape=(_QUATERNION_DIMENSION,)
    )
    delta_theta = _right_local_attitude_error(estimate, truth)
    return _readonly(np.linalg.norm(delta_theta, axis=-1))


def attitude_geodesic_error_deg(
    q_hat_NB: np.ndarray,
    q_true_NB: np.ndarray,
) -> np.ndarray:
    """Return :func:`attitude_geodesic_error_rad` converted to degrees."""

    return _readonly(np.rad2deg(attitude_geodesic_error_rad(q_hat_NB, q_true_NB)))


def bias_error_summary(
    b_hat_rad_s: np.ndarray,
    b_true_rad_s: np.ndarray,
) -> BiasErrorSummary:
    """Return per-sample bias errors and closed-form batch RMSE summaries."""

    estimate = _require_float64_array(
        b_hat_rad_s, name="b_hat_rad_s", trailing_shape=(_BIAS_DIMENSION,)
    )
    truth = _require_float64_array(
        b_true_rad_s, name="b_true_rad_s", trailing_shape=(_BIAS_DIMENSION,)
    )
    _require_same_leading_shape(
        estimate,
        truth,
        reference_trailing_dimensions=1,
        candidate_trailing_dimensions=1,
        reference_name="b_hat_rad_s",
        candidate_name="b_true_rad_s",
    )
    error = truth - estimate
    rows = error.reshape(-1, _BIAS_DIMENSION)
    vector_norm = np.linalg.norm(error, axis=-1)
    per_axis_rmse = np.sqrt(np.mean(rows * rows, axis=0))
    vector_rmse = float(np.sqrt(np.mean(np.sum(rows * rows, axis=1))))
    if not np.isfinite(vector_rmse):
        raise ValueError("bias vector RMSE is not finite")
    return BiasErrorSummary(
        per_axis_error_rad_s=_readonly(error),
        vector_norm_rad_s=_readonly(vector_norm),
        per_axis_rmse_rad_s=_readonly(per_axis_rmse),
        vector_rmse_rad_s=vector_rmse,
    )


def _require_pairing_array(
    value: object,
    *,
    name: str,
    batch_shape: tuple[int, ...],
    dtype: np.dtype,
) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{name} must be a numpy.ndarray")
    if value.dtype != dtype:
        raise TypeError(f"{name} must have dtype {dtype}, got {value.dtype}")
    if value.shape != batch_shape:
        raise ValueError(f"{name} must have shape {batch_shape}, got {value.shape}")
    if dtype == np.dtype(np.float64) and not np.all(np.isfinite(value)):
        raise ValueError(f"{name} must contain only finite values")
    return value


def _validate_pairing_group(
    values: tuple[object | None, ...],
    *,
    names: tuple[str, ...],
    batch_shape: tuple[int, ...],
    dtype: np.dtype,
) -> None:
    present = tuple(value is not None for value in values)
    if not any(present):
        return
    if not all(present):
        raise ValueError(f"{', '.join(names)} must be supplied together")
    checked = tuple(
        _require_pairing_array(value, name=name, batch_shape=batch_shape, dtype=dtype)
        for value, name in zip(values, names)
    )
    reference = checked[0]
    for candidate, name in zip(checked[1:], names[1:]):
        if not np.array_equal(reference, candidate):
            raise ValueError(f"{names[0]} and {name} must match exactly; alignment is not inferred")


def spd_diagnostics(matrix: np.ndarray, *, name: str = "matrix") -> SPDDiagnostics:
    """Validate strict SPD and return diagnostics without changing the matrix."""

    value = _require_float64_array(matrix, name=name)
    if value.ndim < 2 or value.shape[-1] != value.shape[-2]:
        raise ValueError(f"{name} must have trailing square matrix dimensions, got {value.shape}")
    dimension = int(value.shape[-1])
    if dimension == 0:
        raise ValueError(f"{name} matrix dimension must be positive")
    batch_shape = value.shape[:-2]
    rows = value.reshape(-1, dimension, dimension)
    asymmetry = np.empty(rows.shape[0], dtype=np.float64)
    minimum = np.empty(rows.shape[0], dtype=np.float64)
    succeeded = np.empty(rows.shape[0], dtype=np.bool_)
    for index, row in enumerate(rows):
        diagnostics = assert_positive_definite(row, name=f"{name}[{index}]")
        asymmetry[index] = diagnostics.relative_asymmetry
        minimum[index] = diagnostics.minimum_eigenvalue
        succeeded[index] = diagnostics.cholesky_succeeded
    return SPDDiagnostics(
        relative_asymmetry=_readonly(asymmetry.reshape(batch_shape)),
        minimum_eigenvalue=_readonly(minimum.reshape(batch_shape)),
        cholesky_succeeded=_readonly_bool(succeeded.reshape(batch_shape)),
        dimension=dimension,
    )


def _quadratic_form_batch(
    vector: np.ndarray,
    matrix: np.ndarray,
    *,
    vector_name: str,
    matrix_name: str,
) -> np.ndarray:
    dimension = int(vector.shape[-1])
    batch_shape = _require_same_leading_shape(
        vector,
        matrix,
        reference_trailing_dimensions=1,
        candidate_trailing_dimensions=2,
        reference_name=vector_name,
        candidate_name=matrix_name,
    )
    vector_rows = vector.reshape(-1, dimension)
    matrix_rows = matrix.reshape(-1, dimension, dimension)
    values = np.empty(vector_rows.shape[0], dtype=np.float64)
    for index, (row_vector, row_matrix) in enumerate(zip(vector_rows, matrix_rows)):
        solved = cholesky_solve_spd(
            row_matrix,
            row_vector,
            name=f"{matrix_name}[{index}]",
        )
        result = float(row_vector @ solved)
        if not np.isfinite(result) or result < 0.0:
            raise ValueError(f"{matrix_name}[{index}] quadratic form must be finite and nonnegative")
        values[index] = result
    return _readonly(values.reshape(batch_shape))


def star_tracker_nis(
    residual_rad: np.ndarray,
    innovation_covariance_rad2: np.ndarray,
    *,
    residual_time_s: np.ndarray | None = None,
    covariance_time_s: np.ndarray | None = None,
    residual_trajectory_id: np.ndarray | None = None,
    covariance_trajectory_id: np.ndarray | None = None,
) -> np.ndarray:
    """Compute NIS only for supplied star-tracker update residual evidence."""

    residual = _require_float64_array(
        residual_rad, name="residual_rad", trailing_shape=(_ATTITUDE_DIMENSION,)
    )
    covariance = _require_float64_array(
        innovation_covariance_rad2,
        name="innovation_covariance_rad2",
        trailing_shape=(_ATTITUDE_DIMENSION, _ATTITUDE_DIMENSION),
    )
    batch_shape = _require_same_leading_shape(
        residual,
        covariance,
        reference_trailing_dimensions=1,
        candidate_trailing_dimensions=2,
        reference_name="residual_rad",
        candidate_name="innovation_covariance_rad2",
    )
    _validate_pairing_group(
        (residual_time_s, covariance_time_s),
        names=("residual_time_s", "covariance_time_s"),
        batch_shape=batch_shape,
        dtype=np.dtype(np.float64),
    )
    _validate_pairing_group(
        (residual_trajectory_id, covariance_trajectory_id),
        names=("residual_trajectory_id", "covariance_trajectory_id"),
        batch_shape=batch_shape,
        dtype=np.dtype(np.int64),
    )
    return _quadratic_form_batch(
        residual,
        covariance,
        vector_name="residual_rad",
        matrix_name="innovation_covariance_rad2",
    )


def right_local_nees(
    q_hat_NB: np.ndarray,
    b_hat_rad_s: np.ndarray,
    posterior_covariance: np.ndarray,
    q_true_NB: np.ndarray,
    b_true_rad_s: np.ndarray,
    *,
    estimate_time_s: np.ndarray | None = None,
    covariance_time_s: np.ndarray | None = None,
    truth_time_s: np.ndarray | None = None,
    estimate_trajectory_id: np.ndarray | None = None,
    covariance_trajectory_id: np.ndarray | None = None,
    truth_trajectory_id: np.ndarray | None = None,
) -> np.ndarray:
    """Compute six-dimensional NEES in the matching right-local posterior tangent."""

    state = right_local_state_error(q_hat_NB, b_hat_rad_s, q_true_NB, b_true_rad_s)
    covariance = _require_float64_array(
        posterior_covariance,
        name="posterior_covariance",
        trailing_shape=(_STATE_DIMENSION, _STATE_DIMENSION),
    )
    batch_shape = _require_same_leading_shape(
        state.state_error,
        covariance,
        reference_trailing_dimensions=1,
        candidate_trailing_dimensions=2,
        reference_name="state_error",
        candidate_name="posterior_covariance",
    )
    _validate_pairing_group(
        (estimate_time_s, covariance_time_s, truth_time_s),
        names=("estimate_time_s", "covariance_time_s", "truth_time_s"),
        batch_shape=batch_shape,
        dtype=np.dtype(np.float64),
    )
    _validate_pairing_group(
        (estimate_trajectory_id, covariance_trajectory_id, truth_trajectory_id),
        names=(
            "estimate_trajectory_id",
            "covariance_trajectory_id",
            "truth_trajectory_id",
        ),
        batch_shape=batch_shape,
        dtype=np.dtype(np.int64),
    )
    return _quadratic_form_batch(
        state.state_error,
        covariance,
        vector_name="state_error",
        matrix_name="posterior_covariance",
    )


def consistency_summary(
    values: np.ndarray,
    *,
    dof_per_sample: int,
    confidence_level: float = 0.95,
) -> ConsistencySummary:
    """Summarize NIS/NEES values and the matched-Gaussian batch-sum interval.

    The chi-square interval is diagnostic only.  It assumes independent,
    correctly paired samples from the matched Gaussian model.
    """

    samples = _require_float64_array(values, name="values")
    if samples.ndim != 1:
        raise ValueError(f"values must be one-dimensional, got {samples.shape}")
    if np.any(samples < 0.0):
        raise ValueError("values must be nonnegative")
    if isinstance(dof_per_sample, bool) or not isinstance(dof_per_sample, (int, np.integer)):
        raise TypeError("dof_per_sample must be an integer")
    dof = int(dof_per_sample)
    if dof <= 0:
        raise ValueError("dof_per_sample must be positive")
    confidence = float(confidence_level)
    if not np.isfinite(confidence) or not 0.0 < confidence < 1.0:
        raise ValueError("confidence_level must be finite and strictly between zero and one")
    count = int(samples.size)
    total = float(np.sum(samples, dtype=np.float64))
    mean = total / count
    alpha = 1.0 - confidence
    batch_dof = count * dof
    lower = float(chi2.ppf(alpha / 2.0, batch_dof))
    upper = float(chi2.ppf(1.0 - alpha / 2.0, batch_dof))
    if not all(np.isfinite(value) for value in (total, mean, lower, upper)):
        raise ValueError("consistency summary is not finite")
    return ConsistencySummary(
        count=count,
        dof_per_sample=dof,
        sum=total,
        mean=mean,
        normalized_mean=mean / dof,
        confidence_level=confidence,
        chi_square_sum_lower=lower,
        chi_square_sum_upper=upper,
    )


__all__ = [
    "BiasErrorSummary",
    "ConsistencySummary",
    "RightLocalStateError",
    "SPDDiagnostics",
    "attitude_geodesic_error_deg",
    "attitude_geodesic_error_rad",
    "bias_error_summary",
    "consistency_summary",
    "right_local_nees",
    "right_local_state_error",
    "spd_diagnostics",
    "star_tracker_nis",
]
