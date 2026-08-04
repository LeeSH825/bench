from __future__ import annotations

import numpy as np


def _as_mrp(sigma: np.ndarray) -> np.ndarray:
    arr = np.asarray(sigma, dtype=np.float64)
    if arr.ndim < 1 or arr.shape[-1] != 3:
        raise ValueError(f"sigma must have trailing shape [3], got shape={arr.shape}")
    return arr


def mrp_to_quat(sigma: np.ndarray) -> np.ndarray:
    """
    Convert benchmark MRP ``sigma_BN`` to scalar-first quaternion ``[q0,q1,q2,q3]``.

    Leading dimensions are preserved. This convention is used only for benchmark
    visualization artifacts and does not change official benchmark metrics.
    """
    arr = _as_mrp(sigma)
    sigma_sq = np.sum(arr * arr, axis=-1)
    denom = 1.0 + sigma_sq
    scalar = (1.0 - sigma_sq) / denom
    vector = (2.0 / denom)[..., None] * arr
    return np.concatenate([scalar[..., None], vector], axis=-1)


def mrp_to_dcm(sigma: np.ndarray) -> np.ndarray:
    """
    Convert MRP ``sigma_BN`` to the passive direction cosine matrix ``C_BN``.

    The implementation follows the benchmark/Basilisk-style MRP convention:
    ``C_BN = I + (8*S@S - 4*(1-sigma^2)*S)/(1+sigma^2)^2``, where
    ``S`` is the cross-product matrix of ``sigma``. Leading dimensions are
    preserved and the result has trailing shape ``[3,3]``.
    """
    arr = _as_mrp(sigma)
    sigma_sq = np.sum(arr * arr, axis=-1)
    skew = np.zeros(arr.shape[:-1] + (3, 3), dtype=np.float64)
    skew[..., 0, 1] = -arr[..., 2]
    skew[..., 0, 2] = arr[..., 1]
    skew[..., 1, 0] = arr[..., 2]
    skew[..., 1, 2] = -arr[..., 0]
    skew[..., 2, 0] = -arr[..., 1]
    skew[..., 2, 1] = arr[..., 0]
    skew_sq = np.matmul(skew, skew)
    identity = np.broadcast_to(np.eye(3, dtype=np.float64), skew.shape)
    denom = ((1.0 + sigma_sq) ** 2)[..., None, None]
    return identity + (
        8.0 * skew_sq
        - 4.0 * (1.0 - sigma_sq)[..., None, None] * skew
    ) / denom


def mrp_to_euler321(sigma: np.ndarray) -> np.ndarray:
    """
    Convert MRP ``sigma_BN`` to 3-2-1 roll, pitch, yaw angles in radians.

    The returned order is ``[roll, pitch, yaw]`` and uses the passive ``C_BN``
    convention from :func:`mrp_to_dcm`. At Euler gimbal lock, yaw is set to zero
    and the observable combined rotation is assigned to roll. This conversion
    is for timestamp-level visualization/reporting only.
    """
    dcm = mrp_to_dcm(sigma)
    pitch = np.arcsin(np.clip(-dcm[..., 0, 2], -1.0, 1.0))
    regular = np.abs(np.cos(pitch)) > 1.0e-10

    roll_regular = np.arctan2(dcm[..., 1, 2], dcm[..., 2, 2])
    yaw_regular = np.arctan2(dcm[..., 0, 1], dcm[..., 0, 0])
    roll_lock_pos = np.arctan2(dcm[..., 1, 0], dcm[..., 1, 1])
    roll_lock_neg = np.arctan2(-dcm[..., 1, 0], dcm[..., 1, 1])
    roll_lock = np.where(pitch >= 0.0, roll_lock_pos, roll_lock_neg)

    roll = np.where(regular, roll_regular, roll_lock)
    yaw = np.where(regular, yaw_regular, 0.0)
    return np.stack([roll, pitch, yaw], axis=-1)


def wrap_angle_rad(angle: np.ndarray) -> np.ndarray:
    """Wrap radians elementwise to the half-open interval ``[-pi, pi)``."""
    arr = np.asarray(angle, dtype=np.float64)
    return (arr + np.pi) % (2.0 * np.pi) - np.pi


def euler321_error(euler_hat: np.ndarray, euler_true: np.ndarray) -> np.ndarray:
    """Return wrapped elementwise 3-2-1 Euler error ``hat - true`` in radians."""
    estimated = np.asarray(euler_hat, dtype=np.float64)
    truth = np.asarray(euler_true, dtype=np.float64)
    if estimated.shape != truth.shape:
        raise ValueError(
            f"Euler shapes must match, got estimated={estimated.shape}, true={truth.shape}"
        )
    if estimated.ndim < 1 or estimated.shape[-1] != 3:
        raise ValueError(
            f"Euler arrays must have trailing shape [3], got shape={estimated.shape}"
        )
    return wrap_angle_rad(estimated - truth)


def mrp_error_norm(sigma_hat: np.ndarray, sigma_true: np.ndarray) -> np.ndarray:
    """Return Euclidean MRP component error norm over the trailing dimension."""
    estimated = _as_mrp(sigma_hat)
    truth = _as_mrp(sigma_true)
    if estimated.shape != truth.shape:
        raise ValueError(
            f"MRP shapes must match, got estimated={estimated.shape}, true={truth.shape}"
        )
    return np.linalg.norm(estimated - truth, axis=-1)
