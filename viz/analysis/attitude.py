from __future__ import annotations

import numpy as np


def normalize_quat(q):
    arr = np.asarray(q, dtype=np.float64)
    norm = np.linalg.norm(arr, axis=-1, keepdims=True)
    return arr / np.maximum(norm, np.finfo(np.float64).eps)


def mrp_to_quat(sigma):
    sig = np.asarray(sigma, dtype=np.float64)
    sigma_sq = np.sum(sig * sig, axis=-1)
    denom = 1.0 + sigma_sq
    scalar = (1.0 - sigma_sq) / denom
    vector = (2.0 / denom)[..., None] * sig
    return np.concatenate([scalar[..., None], vector], axis=-1)


def quat_to_mrp(q):
    quat = normalize_quat(q)
    scalar = quat[..., 0]
    vector = quat[..., 1:4]
    denom = 1.0 + scalar
    near = np.abs(denom) <= np.finfo(np.float64).eps
    sigma = vector / np.where(near, np.nan, denom)[..., None]
    if np.any(near):
        alt = -vector / np.where(near, 1.0 - scalar, np.nan)[..., None]
        sigma = np.where(near[..., None], alt, sigma)
    return sigma


def continuous_quat_sign(q):
    out = np.asarray(q, dtype=np.float64).copy()
    if out.ndim != 2 or out.shape[1] != 4:
        return out
    for idx in range(1, out.shape[0]):
        if float(np.dot(out[idx - 1], out[idx])) < 0.0:
            out[idx] *= -1.0
    return out


def mrp_to_quat_continuous(sigma):
    return continuous_quat_sign(mrp_to_quat(sigma))


def shadow_mrp(sigma, threshold=1.0):
    sig = np.asarray(sigma, dtype=np.float64)
    norm_sq = np.sum(sig * sig, axis=-1, keepdims=True)
    mask = norm_sq > np.asarray(threshold, dtype=np.float64) ** 2
    shadow = -sig / np.maximum(norm_sq, np.finfo(np.float64).eps)
    return np.where(mask, shadow, sig)


def quat_conjugate(q):
    quat = np.asarray(q, dtype=np.float64)
    out = quat.copy()
    out[..., 1:4] *= -1.0
    return out


def quat_multiply(q_left, q_right):
    a = np.asarray(q_left, dtype=np.float64)
    b = np.asarray(q_right, dtype=np.float64)
    aw, ax, ay, az = np.moveaxis(a, -1, 0)
    bw, bx, by, bz = np.moveaxis(b, -1, 0)
    return np.stack(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        axis=-1,
    )


def relative_quat(q_true, q_hat):
    return normalize_quat(quat_multiply(q_hat, quat_conjugate(q_true)))


def geodesic_angle_rad(q_a, q_b):
    qa = normalize_quat(q_a)
    qb = normalize_quat(q_b)
    dot = np.sum(qa * qb, axis=-1)
    dot = np.clip(np.abs(dot), 0.0, 1.0)
    return 2.0 * np.arccos(dot)


def quat_to_rotvec(q):
    quat = normalize_quat(q)
    quat = np.where((quat[..., 0:1] < 0.0), -quat, quat)
    scalar = np.clip(quat[..., 0], -1.0, 1.0)
    vector = quat[..., 1:4]
    vec_norm = np.linalg.norm(vector, axis=-1)
    angle = 2.0 * np.arctan2(vec_norm, scalar)
    scale = np.zeros_like(vec_norm)
    mask = vec_norm > np.finfo(np.float64).eps
    scale[mask] = angle[mask] / vec_norm[mask]
    scale[~mask] = 2.0
    return vector * scale[..., None]


def mrp_axis_error_rad(sigma_true, sigma_hat):
    return quat_to_rotvec(relative_quat(mrp_to_quat(sigma_true), mrp_to_quat(sigma_hat)))


def quat_from_euler321(roll, pitch, yaw):
    half = 0.5
    cr = np.cos(np.asarray(roll, dtype=np.float64) * half)
    sr = np.sin(np.asarray(roll, dtype=np.float64) * half)
    cp = np.cos(np.asarray(pitch, dtype=np.float64) * half)
    sp = np.sin(np.asarray(pitch, dtype=np.float64) * half)
    cy = np.cos(np.asarray(yaw, dtype=np.float64) * half)
    sy = np.sin(np.asarray(yaw, dtype=np.float64) * half)
    return normalize_quat(
        np.stack(
            [
                cr * cp * cy + sr * sp * sy,
                sr * cp * cy - cr * sp * sy,
                cr * sp * cy + sr * cp * sy,
                cr * cp * sy - sr * sp * cy,
            ],
            axis=-1,
        )
    )


def euler321_from_quat(q):
    quat = normalize_quat(q)
    w = quat[..., 0]
    x = quat[..., 1]
    y = quat[..., 2]
    z = quat[..., 3]
    roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    sin_pitch = 2.0 * (w * y - z * x)
    pitch = np.arcsin(np.clip(sin_pitch, -1.0, 1.0))
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return np.stack([roll, pitch, yaw], axis=-1)
