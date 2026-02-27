from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np


def _safe_log10(x: float, eps: float = 1e-30) -> float:
    return math.log10(max(x, eps))


def mse_per_step(x_hat: np.ndarray, x_gt: np.ndarray) -> np.ndarray:
    """
    Per-step MSE_t.
    Inputs:
      x_hat: [B,T,D]
      x_gt : [B,T,D]
    Returns:
      mse_t: [T] where mse_t[t] = mean_{b} mean_{d} (x_hat[b,t,d]-x_gt[b,t,d])^2
    """
    assert x_hat.ndim == 3 and x_gt.ndim == 3
    assert x_hat.shape == x_gt.shape
    # Use float64 accumulation to reduce overflow risk on large-magnitude predictions.
    xh = np.asarray(x_hat, dtype=np.float64)
    xg = np.asarray(x_gt, dtype=np.float64)
    err2 = (xh - xg) ** 2  # [B,T,D]
    return err2.mean(axis=(0, 2))  # [T]


def mse_scalar(x_hat: np.ndarray, x_gt: np.ndarray) -> float:
    """
    Scalar MSE = mean_{b,t} ||e||^2  (dimension 평균 포함)
    """
    xh = np.asarray(x_hat, dtype=np.float64)
    xg = np.asarray(x_gt, dtype=np.float64)
    return float(((xh - xg) ** 2).mean())


def rmse_scalar(x_hat: np.ndarray, x_gt: np.ndarray) -> float:
    return float(math.sqrt(max(mse_scalar(x_hat, x_gt), 0.0)))


def mse_db_scalar(mse: float) -> float:
    return float(10.0 * _safe_log10(mse))


def compute_shift_recovery_k(
    mse_t: np.ndarray,
    t0: int,
    W: int = 20,
    eps: float = 0.05,
    failure_policy: str = "cap",
) -> Dict[str, Any]:
    """
    METRICS.md 정의를 그대로 구현.

    E_pre = mean_{t=t0-W..t0-1} MSE_t
    recovery_k = min k>=0 s.t. mean_{t=t0+k..t0+k+W-1} MSE_t <= (1+eps)*E_pre

    failure_policy:
      - "cap": recovery_k = (T - t0)
      - "NA" : recovery_k = None
    """
    T = int(mse_t.shape[0])
    out: Dict[str, Any] = {
        "t0": int(t0),
        "W": int(W),
        "eps": float(eps),
        "E_pre": None,
        "recovery_k": None,
        "recovered": False,
        "failure_policy": failure_policy,
    }

    if T <= 0:
        return out

    t0 = int(max(0, min(t0, T)))
    if t0 < W:
        # pre window가 부족하면 정의가 애매하므로, 가능한 범위로 축소
        pre_start = 0
        pre_end = t0
    else:
        pre_start = t0 - W
        pre_end = t0

    if pre_end <= pre_start:
        # pre window 자체가 없으면 복구 정의 불가
        out["E_pre"] = None
        if failure_policy.lower() == "cap":
            out["recovery_k"] = int(T - t0)
        else:
            out["recovery_k"] = None
        return out

    E_pre = float(np.mean(mse_t[pre_start:pre_end]))
    out["E_pre"] = E_pre
    threshold = (1.0 + float(eps)) * E_pre

    # search k
    for k in range(0, T - t0):
        win_start = t0 + k
        win_end = min(T, win_start + W)
        if win_end - win_start < W:
            # 마지막 구간에서 window 미만이면 중단(정의 상 W 필요)
            break
        win_mean = float(np.mean(mse_t[win_start:win_end]))
        if win_mean <= threshold:
            out["recovery_k"] = int(k)
            out["recovered"] = True
            return out

    # not recovered
    if failure_policy.lower() == "cap":
        out["recovery_k"] = int(T - t0)
    else:
        out["recovery_k"] = None
    out["recovered"] = False
    return out


def gaussian_nll_per_step(
    err: np.ndarray,
    cov: np.ndarray,
) -> np.ndarray:
    """
    Optional Gaussian NLL per step (METRICS.md):
      NLL_t = mean_b [ 0.5*log|Sigma| + 0.5*e^T Sigma^{-1} e + (d/2)*log(2*pi) ]

    Inputs:
      err: [B,T,D]
      cov: [B,T,D,D]  (full covariance)
    Returns:
      nll_t: [T]
    """
    assert err.ndim == 3
    assert cov.ndim == 4
    B, T, D = err.shape
    assert cov.shape[:3] == (B, T, D) and cov.shape[3] == D

    nll_t = np.zeros((T,), dtype=np.float64)
    const = 0.5 * D * math.log(2.0 * math.pi)

    for t in range(T):
        acc = 0.0
        for b in range(B):
            Sigma = cov[b, t]  # [D,D]
            e = err[b, t]      # [D]
            # 안정성: 작은 jitter
            Sigma = Sigma + np.eye(D) * 1e-9
            sign, logdet = np.linalg.slogdet(Sigma)
            if sign <= 0:
                # 비정상 공분산이면 큰 값으로 패널티
                acc += 1e9
                continue
            inv = np.linalg.inv(Sigma)
            quad = float(e.T @ inv @ e)
            acc += 0.5 * logdet + 0.5 * quad + const
        nll_t[t] = acc / float(B)
    return nll_t
