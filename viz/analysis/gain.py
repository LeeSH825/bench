from __future__ import annotations

import numpy as np


def gain_norm(gain, axis=(-2, -1)):
    return np.linalg.norm(np.asarray(gain, dtype=np.float64), axis=axis)


def normalize_gain_trace(trace, mode="initial"):
    arr = np.asarray(trace, dtype=np.float64)
    if mode == "initial":
        ref = np.take(arr, 0, axis=-1)
    elif mode == "max":
        ref = np.nanmax(np.abs(arr), axis=-1)
    else:
        raise ValueError(f"unsupported gain normalization mode={mode!r}")
    return arr / np.maximum(np.expand_dims(np.abs(ref), axis=-1), np.finfo(np.float64).eps)


def extract_gain_trajectory(gain, traj_idx=0, normalize=None):
    arr = np.asarray(gain, dtype=np.float64)
    if arr.ndim == 4:
        selected = arr[int(traj_idx)]
    elif arr.ndim == 3:
        selected = arr
    else:
        raise ValueError(f"gain must have shape [N,T,n,m] or [T,n,m], got {arr.shape}")
    norm = gain_norm(selected)
    if normalize is not None:
        norm = normalize_gain_trace(norm, mode=str(normalize))
    return {"gain": selected, "gain_norm": norm}
