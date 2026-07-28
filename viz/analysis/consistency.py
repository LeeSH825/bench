from __future__ import annotations

import numpy as np
from scipy.stats import chi2 as scipy_chi2


def _quadratic_form(vector, matrix):
    vec = np.asarray(vector, dtype=np.float64)
    mat = np.asarray(matrix, dtype=np.float64)
    if vec.shape[-1] != mat.shape[-1] or mat.shape[-1] != mat.shape[-2]:
        raise ValueError(f"shape mismatch: vector={vec.shape}, matrix={mat.shape}")
    try:
        solved = np.linalg.solve(mat, vec[..., None])[..., 0]
    except np.linalg.LinAlgError:
        solved = np.einsum("...ij,...j->...i", np.linalg.pinv(mat), vec)
    return np.einsum("...i,...i->...", vec, solved)


def nees(error, covariance):
    return _quadratic_form(error, covariance)


def nis(innovation, innovation_covariance, valid=None):
    out = _quadratic_form(innovation, innovation_covariance)
    if valid is None:
        return out
    mask = np.asarray(valid, dtype=bool)
    return np.where(mask, out, np.nan)


def chi2_ppf(probability, dof):
    k = np.asarray(dof, dtype=np.float64)
    if np.any(k <= 0.0):
        raise ValueError("chi-square dof must be positive")
    return np.asarray(scipy_chi2.ppf(probability, k), dtype=np.float64)


def chi2_mean_bounds(*, probability=(0.005, 0.995), n_samples, dim):
    n = int(n_samples)
    d = int(dim)
    if n <= 0 or d <= 0:
        raise ValueError("n_samples and dim must be positive")
    dof = n * d
    probs = np.asarray(probability, dtype=np.float64)
    return chi2_ppf(probs, dof) / np.asarray(n, dtype=np.float64)


def three_sigma_coverage(error, covariance):
    err = np.asarray(error, dtype=np.float64)
    cov = np.asarray(covariance, dtype=np.float64)
    diag = np.diagonal(cov, axis1=-2, axis2=-1)
    sigma = np.sqrt(np.maximum(diag, 0.0))
    covered = np.abs(err) <= (3.0 * sigma)
    return {
        "axis_coverage": np.mean(covered, axis=tuple(range(covered.ndim - 1))),
        "overall_coverage": float(np.mean(covered)),
    }


def predicted_sigma_mean(covariance):
    cov = np.asarray(covariance, dtype=np.float64)
    diag = np.diagonal(cov, axis1=-2, axis2=-1)
    sigma = np.sqrt(np.maximum(diag, 0.0))
    if sigma.ndim >= 3:
        return np.mean(sigma, axis=0)
    return sigma


def ensemble_relative_standard_error(n_samples):
    n = int(n_samples)
    if n < 2:
        raise ValueError("at least two samples are required")
    return float(1.0 / np.sqrt(2.0 * (n - 1)))


def ensemble_sigma_confidence_interval(emp_std, n_samples, confidence_z=1.959963984540054):
    sigma = np.asarray(emp_std, dtype=np.float64)
    relative_standard_error = ensemble_relative_standard_error(n_samples)
    delta = np.asarray(confidence_z, dtype=np.float64) * relative_standard_error * sigma
    return np.stack([np.maximum(sigma - delta, 0.0), sigma + delta], axis=0)


def ensemble_sigma(error, confidence_z=1.959963984540054):
    err = np.asarray(error, dtype=np.float64)
    if err.ndim < 2:
        raise ValueError("ensemble error must have at least sample and state dimensions")
    n_samples = int(err.shape[0])
    emp_std = np.std(err, axis=0, ddof=1)
    relative_standard_error = ensemble_relative_standard_error(n_samples)
    return {
        "emp_std": emp_std,
        "n_samples": n_samples,
        "relative_standard_error": float(relative_standard_error),
        "confidence_interval": ensemble_sigma_confidence_interval(
            emp_std,
            n_samples=n_samples,
            confidence_z=confidence_z,
        ),
    }


def innovation_acf(innovation, max_lag):
    arr = np.asarray(innovation, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[None, :, None]
    elif arr.ndim == 2:
        arr = arr[None, :, :]
    if arr.ndim != 3:
        raise ValueError(f"innovation must have shape [T], [T,D], or [N,T,D], got {arr.shape}")
    lag_max = int(max_lag)
    if lag_max < 0:
        raise ValueError("max_lag must be non-negative")

    out = np.empty((lag_max + 1, arr.shape[-1]), dtype=np.float64)
    centered = arr - np.mean(arr, axis=1, keepdims=True)
    denom = np.sum(centered * centered, axis=1)
    denom = np.maximum(denom, np.finfo(np.float64).eps)
    for lag in range(lag_max + 1):
        if lag == 0:
            acf_n = np.ones((arr.shape[0], arr.shape[-1]), dtype=np.float64)
        else:
            acf_n = np.sum(centered[:, :-lag, :] * centered[:, lag:, :], axis=1) / denom
        out[lag] = np.mean(acf_n, axis=0)
    return out


def whiteness_bounds(n_time, confidence_z=1.959963984540054):
    return np.asarray(confidence_z, dtype=np.float64) / np.sqrt(np.asarray(n_time, dtype=np.float64))
