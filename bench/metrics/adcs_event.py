from __future__ import annotations

import math
from typing import Dict, Tuple

import numpy as np


def _as_ntd_state(x: np.ndarray, *, name: str) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim == 2:
        arr = arr[None, ...]
    if arr.ndim != 3:
        raise ValueError(f"{name} must have shape [T,D] or [N,T,D], got {arr.shape}")
    if int(arr.shape[-1]) < 6:
        raise ValueError(f"{name} must contain at least [MRP(3), omega(3)], got D={arr.shape[-1]}")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} contains non-finite values")
    return arr


def _state_pair(x_true: np.ndarray, x_pred: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    truth = _as_ntd_state(x_true, name="x_true")
    pred = _as_ntd_state(x_pred, name="x_pred")
    if truth.shape != pred.shape:
        raise ValueError(f"x_true/x_pred shapes must match, got {truth.shape} and {pred.shape}")
    return truth, pred


def _event_mask(event_flag_seq: np.ndarray, *, n_seq: int, n_step: int) -> np.ndarray:
    flag = np.asarray(event_flag_seq)
    if flag.ndim == 1:
        if flag.shape != (n_step,):
            raise ValueError(f"1D event_flag_seq must have shape ({n_step},), got {flag.shape}")
        flag = np.broadcast_to(flag[None, :], (n_seq, n_step))
    elif flag.ndim == 2:
        if flag.shape == (n_seq, n_step):
            pass
        elif flag.shape == (1, n_step):
            flag = np.broadcast_to(flag, (n_seq, n_step))
        elif flag.shape == (n_step, 1):
            flag = np.broadcast_to(flag[:, 0][None, :], (n_seq, n_step))
        else:
            raise ValueError(
                "2D event_flag_seq must have shape [N,T], [1,T], or [T,1], "
                f"got {flag.shape} for N={n_seq}, T={n_step}"
            )
    elif flag.ndim == 3:
        if flag.shape == (n_seq, n_step, 1):
            flag = flag[..., 0]
        elif flag.shape == (1, n_step, 1):
            flag = np.broadcast_to(flag[..., 0], (n_seq, n_step))
        else:
            raise ValueError(
                f"3D event_flag_seq must have shape [N,T,1] or [1,T,1], got {flag.shape}"
            )
    else:
        raise ValueError(
            "event_flag_seq must have shape [T], [T,1], [N,T], or [N,T,1], "
            f"got {flag.shape}"
        )
    return np.asarray(flag > 0.5, dtype=bool)


def _mrp_to_unit_quaternion(sigma: np.ndarray) -> np.ndarray:
    """
    Convert MRP sigma to scalar-first unit quaternion [q0,q1,q2,q3].

    q0 = (1 - ||sigma||^2) / (1 + ||sigma||^2)
    qv = 2 sigma / (1 + ||sigma||^2)
    """
    sigma_arr = np.asarray(sigma, dtype=np.float64)
    if sigma_arr.ndim < 1 or sigma_arr.shape[-1] != 3:
        raise ValueError(f"MRP input must have trailing shape [3], got {sigma_arr.shape}")
    sigma_sq = np.sum(sigma_arr * sigma_arr, axis=-1)
    denom = 1.0 + sigma_sq
    scalar = (1.0 - sigma_sq) / denom
    vector = (2.0 / denom)[..., None] * sigma_arr
    return np.concatenate([scalar[..., None], vector], axis=-1)


def attitude_error_deg(x_true: np.ndarray, x_pred: np.ndarray) -> np.ndarray:
    """
    Return the exact shortest relative-attitude angle in degrees for MRP states.

    MRPs are converted to unit quaternions. For unit quaternions q_true and
    q_pred, the shortest geodesic rotation angle is
    ``2 * arccos(abs(dot(q_true, q_pred)))``. The absolute dot product handles
    the equivalent quaternion signs and MRP shadow-set representation.
    """
    truth, pred = _state_pair(x_true, x_pred)
    q_true = _mrp_to_unit_quaternion(truth[..., 0:3])
    q_pred = _mrp_to_unit_quaternion(pred[..., 0:3])
    quat_dot = np.sum(q_true * q_pred, axis=-1)
    angle_rad = 2.0 * np.arccos(np.clip(np.abs(quat_dot), 0.0, 1.0))
    return np.rad2deg(angle_rad)


def attitude_rmse_deg(x_true: np.ndarray, x_pred: np.ndarray) -> float:
    angle_deg = attitude_error_deg(x_true, x_pred)
    return float(math.sqrt(float(np.mean(angle_deg * angle_deg))))


def angular_velocity_rmse(x_true: np.ndarray, x_pred: np.ndarray) -> float:
    truth, pred = _state_pair(x_true, x_pred)
    omega_error = pred[..., 3:6] - truth[..., 3:6]
    squared_norm = np.sum(omega_error * omega_error, axis=-1)
    return float(math.sqrt(float(np.mean(squared_norm))))


def event_attitude_rmse_deg(
    x_true: np.ndarray,
    x_pred: np.ndarray,
    event_flag_seq: np.ndarray,
) -> float:
    angle_deg = attitude_error_deg(x_true, x_pred)
    mask = _event_mask(event_flag_seq, n_seq=int(angle_deg.shape[0]), n_step=int(angle_deg.shape[1]))
    if not np.any(mask):
        return float("nan")
    selected = angle_deg[mask]
    return float(math.sqrt(float(np.mean(selected * selected))))


def event_peak_attitude_error_deg(
    x_true: np.ndarray,
    x_pred: np.ndarray,
    event_flag_seq: np.ndarray,
) -> float:
    angle_deg = attitude_error_deg(x_true, x_pred)
    mask = _event_mask(event_flag_seq, n_seq=int(angle_deg.shape[0]), n_step=int(angle_deg.shape[1]))
    if not np.any(mask):
        return float("nan")
    return float(np.max(angle_deg[mask]))


def compute_adcs_event_metrics(
    *,
    x_true: np.ndarray,
    x_pred: np.ndarray,
    event_flag_seq: np.ndarray,
) -> Dict[str, float]:
    """Compute the P0 ADCS full-trajectory and event-window metric set."""
    return {
        "attitude_rmse_deg": attitude_rmse_deg(x_true, x_pred),
        "angular_velocity_rmse": angular_velocity_rmse(x_true, x_pred),
        "event_attitude_rmse_deg": event_attitude_rmse_deg(x_true, x_pred, event_flag_seq),
        "event_peak_attitude_error_deg": event_peak_attitude_error_deg(
            x_true,
            x_pred,
            event_flag_seq,
        ),
    }
