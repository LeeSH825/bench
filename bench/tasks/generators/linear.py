from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import numpy as np


# ============================================================
# KalmanNet_TSP canonical definitions (Simulations/.../parameters.py)
# - m = n (suite_shift uses x_dim=y_dim=5)
# - F = I; and F[0,:] = 1
# - H:
#   - if n==2: H = I
#   - if n>2: "reverse canonical form"
#       H = zeros(n,n)
#       H[0,:] = 1
#       for i in range(n): H[i, n-1-i] = 1
# ============================================================

def kalmannet_tsp_F_linear_canonical(n: int) -> np.ndarray:
    F = np.eye(n, dtype=np.float64)
    F[0, :] = 1.0
    return F


def kalmannet_tsp_H_reverse_canonical(n: int) -> np.ndarray:
    if n == 2:
        return np.eye(n, dtype=np.float64)
    H = np.zeros((n, n), dtype=np.float64)
    H[0, :] = 1.0
    for i in range(n):
        H[i, n - 1 - i] = 1.0
    return H


# ============================================================
# v0 matrix builders
# ============================================================

def _stable_F_from_rng(rng: np.random.Generator, x_dim: int) -> np.ndarray:
    """
    Non-canonical fallback: random stable linear dynamics with eigenvalues in (0.7, 0.99).
    Deterministic given rng state.
    """
    A = rng.standard_normal((x_dim, x_dim))
    Q, _ = np.linalg.qr(A)  # orthonormal
    eig = rng.uniform(0.7, 0.99, size=(x_dim,))
    F = (Q @ np.diag(eig) @ Q.T).astype(np.float64)
    return F


def _H_from_spec(rng: np.random.Generator, x_dim: int, y_dim: int, H_spec: str) -> np.ndarray:
    """
    suite 스펙의 H 키워드 해석.

    MVP 범위(필수):
      - "canonical_inverse": KalmanNet_TSP linear_canonical 정렬(reverse canonical form)
      - "identity": identity (y_dim==x_dim)

    Optional:
      - "select_first_two_dims": y_dim==2일 때 [I2 0]
      - else: random linear observation
    """
    spec = (H_spec or "").strip()

    if spec == "canonical_inverse":
        if y_dim != x_dim:
            raise ValueError(f"H_spec={spec} requires y_dim==x_dim (got y_dim={y_dim}, x_dim={x_dim})")
        return kalmannet_tsp_H_reverse_canonical(x_dim)

    if spec in ("identity", "I"):
        if y_dim != x_dim:
            raise ValueError(f"H_spec={spec} requires y_dim==x_dim (got y_dim={y_dim}, x_dim={x_dim})")
        return np.eye(x_dim, dtype=np.float64)

    if spec == "select_first_two_dims":
        if y_dim != 2:
            raise ValueError("select_first_two_dims requires y_dim==2")
        H = np.zeros((2, x_dim), dtype=np.float64)
        H[0, 0] = 1.0
        H[1, 1] = 1.0
        return H

    # fallback: random full-rank linear observation
    H = rng.standard_normal((y_dim, x_dim)).astype(np.float64)
    return H


def build_linear_system_matrices_v0(
    *,
    rng: np.random.Generator,
    x_dim: int,
    y_dim: int,
    H_spec: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (F,H) in float64 for numerical stability; caller can cast to float32.
    """
    H = _H_from_spec(rng, x_dim=x_dim, y_dim=y_dim, H_spec=H_spec)

    # If this is canonical_inverse, we MUST align F with KalmanNet_TSP linear_canonical
    if (H_spec or "").strip() == "canonical_inverse":
        F = kalmannet_tsp_F_linear_canonical(x_dim)
    else:
        F = _stable_F_from_rng(rng, x_dim=x_dim)

    return F, H


# ============================================================
# v0 sequence generator (MVP: gaussian obs only)
# ============================================================

def _sample_obs_noise(
    rng: np.random.Generator,
    n_seq: int,
    y_dim: int,
    r2: float,
    dist_name: str,
    dist_params: Dict[str, Any],
) -> np.ndarray:
    """
    v0 MVP: gaussian only.
    dist shift(student_t 등)은 C_shift_dist_v0에서 enabled:false이므로
    여기서는 gaussian 외엔 NotImplementedError로 둔다(확장 지점).
    """
    name = (dist_name or "gaussian").lower()
    if name == "gaussian":
        return (math.sqrt(r2) * rng.standard_normal((n_seq, y_dim))).astype(np.float64)

    raise NotImplementedError(f"obs_distribution '{dist_name}' is not implemented in v0 MVP")


def generate_linear_gaussian_sequences_v0(
    *,
    rng: np.random.Generator,
    n_seq: int,
    T: int,
    F: np.ndarray,
    H: np.ndarray,
    q2: float,
    r2: float,
    t0_shift: Optional[int],
    r_scale_post: float,
    obs_dist_name: str,
    obs_dist_params: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      x: [N,T,x_dim], y: [N,T,y_dim] in float64 (caller may cast to float32)
    """
    x_dim = int(F.shape[0])
    y_dim = int(H.shape[0])

    x = np.zeros((n_seq, T, x_dim), dtype=np.float64)
    y = np.zeros((n_seq, T, y_dim), dtype=np.float64)

    # initial state
    x_t = rng.standard_normal((n_seq, x_dim)).astype(np.float64)

    for t in range(T):
        # observation noise variance (shifted or not)
        if t0_shift is not None and t >= int(t0_shift):
            r2_t = float(r2) * float(r_scale_post)
        else:
            r2_t = float(r2)

        v_t = _sample_obs_noise(
            rng=rng,
            n_seq=n_seq,
            y_dim=y_dim,
            r2=r2_t,
            dist_name=obs_dist_name,
            dist_params=obs_dist_params,
        )

        # y_t = H x_t + v_t  (ensure H is actually applied)
        y_t = (x_t @ H.T) + v_t

        x[:, t, :] = x_t
        y[:, t, :] = y_t

        # evolve: x_{t+1} = F x_t + w_t
        w_t = (math.sqrt(float(q2)) * rng.standard_normal((n_seq, x_dim))).astype(np.float64)
        x_t = (x_t @ F.T) + w_t

    return x, y

