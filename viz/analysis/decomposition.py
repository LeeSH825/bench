from __future__ import annotations

import numpy as np


def reconstruct_imu_error(bias_component, noise_component):
    return np.asarray(bias_component, dtype=np.float64) + np.asarray(noise_component, dtype=np.float64)


def decomposition_residual(bias_component, noise_component, imu_error):
    return reconstruct_imu_error(bias_component, noise_component) - np.asarray(imu_error, dtype=np.float64)


def contribution_fractions(bias_component, noise_component):
    bias = np.asarray(bias_component, dtype=np.float64)
    noise = np.asarray(noise_component, dtype=np.float64)
    bias_energy = float(np.sum(bias * bias))
    noise_energy = float(np.sum(noise * noise))
    total = bias_energy + noise_energy
    if total <= np.finfo(np.float64).eps:
        return {
            "deterministic": np.nan,
            "stochastic": np.nan,
            "total_energy": total,
        }
    return {
        "deterministic": bias_energy / total,
        "stochastic": noise_energy / total,
        "total_energy": total,
    }
