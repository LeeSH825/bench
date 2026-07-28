from __future__ import annotations

import numpy as np


def regime_intervals(flag):
    values = np.asarray(flag)
    if values.ndim != 1:
        raise ValueError(f"flag must be one-dimensional, got {values.shape}")
    intervals = []
    if values.size == 0:
        return intervals
    start = 0
    current = values[0].item()
    for idx in range(1, values.size):
        value = values[idx].item()
        if value != current:
            intervals.append({"start": start, "end": idx, "value": current})
            start = idx
            current = value
    intervals.append({"start": start, "end": int(values.size), "value": current})
    return intervals


def true_intervals(flag):
    return [item for item in regime_intervals(np.asarray(flag, dtype=bool)) if bool(item["value"])]


def convergence_time(t, signal, threshold, consecutive=1):
    time = np.asarray(t, dtype=np.float64)
    values = np.asarray(signal, dtype=np.float64)
    if time.ndim != 1 or values.ndim != 1 or time.shape[0] != values.shape[0]:
        raise ValueError("t and signal must be one-dimensional arrays with equal length")
    ok = np.abs(values) <= float(threshold)
    need = int(consecutive)
    if need <= 0:
        raise ValueError("consecutive must be positive")
    for idx in range(0, ok.size - need + 1):
        if bool(np.all(ok[idx : idx + need])):
            return float(time[idx])
    return np.nan


def settling_time(t, signal, threshold):
    time = np.asarray(t, dtype=np.float64)
    values = np.asarray(signal, dtype=np.float64)
    if time.ndim != 1 or values.ndim != 1 or time.shape[0] != values.shape[0]:
        raise ValueError("t and signal must be one-dimensional arrays with equal length")
    ok = np.abs(values) <= float(threshold)
    for idx in range(ok.size):
        if bool(np.all(ok[idx:])):
            return float(time[idx])
    return np.nan
