"""Float64 reference math for the Phase 1A six-dimensional kinematic MEKF.

The convention is scalar-first Hamilton quaternion algebra, active body-to-
navigation attitude ``q_NB``, and a right-multiplicative local error:

``q_true = q_hat (x) Exp_q(delta_theta)``.

This module deliberately has no runner, model, task, sensor-generator,
visualization, or training dependency.  It implements only pure Gate A math.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.linalg import cholesky, expm, solve_triangular


ArrayLike = Sequence[float] | np.ndarray
SYMMETRY_TOL = 1.0e-12
PSD_TOL = 1.0e-12
UNIT_QUATERNION_TOL = 1.0e-12
_EXACT_PI_TIE_TOL = 8.0 * np.finfo(np.float64).eps


class NumericalSafetyError(ValueError):
    """Raised when an input or covariance violates the fail-loud contract."""


@dataclass(frozen=True)
class CovarianceDiagnostics:
    """Numerical diagnostics for a symmetric covariance-like matrix."""

    relative_asymmetry: float
    minimum_eigenvalue: float
    maximum_eigenvalue: float
    cholesky_succeeded: bool


@dataclass(frozen=True)
class DiscretizationResult:
    """Exact constant-coefficient Van Loan discretization result."""

    Phi: np.ndarray
    Q_d: np.ndarray
    qd_symmetrization_relative_correction: float


@dataclass(frozen=True)
class KalmanGainResult:
    """Innovation covariance and gain computed without a matrix inverse."""

    K: np.ndarray
    S: np.ndarray
    s_symmetrization_relative_correction: float


@dataclass(frozen=True)
class MEKFState:
    """MEKF nominal state and its six-dimensional right-local covariance."""

    q_NB: np.ndarray
    b_g: np.ndarray
    P: np.ndarray

    def __post_init__(self) -> None:
        q = quat_normalize(self.q_NB, name="q_NB").copy()
        b = _vector(self.b_g, 3, "b_g").copy()
        p = _matrix(self.P, (6, 6), "P").copy()
        assert_positive_definite(p, name="P")
        q.setflags(write=False)
        b.setflags(write=False)
        p.setflags(write=False)
        object.__setattr__(self, "q_NB", q)
        object.__setattr__(self, "b_g", b)
        object.__setattr__(self, "P", p)


@dataclass(frozen=True)
class PropagationResult:
    """State propagation result and matrices needed for independent checks."""

    state: MEKFState
    omega_corrected: np.ndarray
    F: np.ndarray
    G: np.ndarray
    Phi: np.ndarray
    Q_d: np.ndarray
    qd_symmetrization_relative_correction: float
    p_symmetrization_relative_correction: float


@dataclass(frozen=True)
class StarTrackerUpdateResult:
    """Complete local ST update, injection, and covariance reset evidence."""

    state: MEKFState
    residual: np.ndarray
    H: np.ndarray
    R: np.ndarray
    S: np.ndarray
    K: np.ndarray
    delta_x: np.ndarray
    P_c: np.ndarray
    G_reset: np.ndarray
    s_symmetrization_relative_correction: float
    pc_symmetrization_relative_correction: float
    p_reset_symmetrization_relative_correction: float


def _array(value: ArrayLike, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if not np.all(np.isfinite(result)):
        raise NumericalSafetyError(f"{name} must contain only finite float64 values")
    return result


def _vector(value: ArrayLike, size: int, name: str) -> np.ndarray:
    result = _array(value, name)
    if result.shape != (size,):
        raise ValueError(f"{name} must have shape ({size},), got {result.shape}")
    return result


def _matrix(value: ArrayLike, shape: tuple[int, int], name: str) -> np.ndarray:
    result = _array(value, name)
    if result.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {result.shape}")
    return result


def _square_matrix(value: ArrayLike, name: str) -> np.ndarray:
    result = _array(value, name)
    if result.ndim != 2 or result.shape[0] != result.shape[1]:
        raise ValueError(f"{name} must be square, got {result.shape}")
    return result


def _relative_asymmetry(matrix: np.ndarray) -> float:
    denominator = max(float(np.linalg.norm(matrix, ord="fro")), np.finfo(np.float64).eps)
    return float(np.linalg.norm(matrix - matrix.T, ord="fro") / denominator)


def _symmetrize_roundoff(
    matrix: ArrayLike,
    *,
    name: str,
    tolerance: float = SYMMETRY_TOL,
) -> tuple[np.ndarray, float]:
    value = _square_matrix(matrix, name)
    relative_asymmetry = _relative_asymmetry(value)
    if relative_asymmetry > tolerance:
        raise NumericalSafetyError(
            f"{name} relative asymmetry {relative_asymmetry:.3e} exceeds {tolerance:.3e}"
        )
    symmetric = 0.5 * (value + value.T)
    denominator = max(float(np.linalg.norm(value, ord="fro")), np.finfo(np.float64).eps)
    relative_correction = float(np.linalg.norm(symmetric - value, ord="fro") / denominator)
    return symmetric, relative_correction


def covariance_diagnostics(
    matrix: ArrayLike,
    *,
    name: str = "matrix",
    require_spd: bool = False,
    symmetry_tolerance: float = SYMMETRY_TOL,
    psd_tolerance: float = PSD_TOL,
) -> CovarianceDiagnostics:
    """Validate symmetry and PSD/SPD status without correcting eigenvalues."""

    value = _square_matrix(matrix, name)
    relative_asymmetry = _relative_asymmetry(value)
    if relative_asymmetry > symmetry_tolerance:
        raise NumericalSafetyError(
            f"{name} relative asymmetry {relative_asymmetry:.3e} exceeds "
            f"{symmetry_tolerance:.3e}"
        )
    symmetric = 0.5 * (value + value.T)
    eigenvalues = np.linalg.eigvalsh(symmetric)
    minimum = float(eigenvalues[0])
    maximum = float(eigenvalues[-1])
    cholesky_succeeded = False
    if require_spd:
        try:
            cholesky(symmetric, lower=True, check_finite=True)
        except (ValueError, np.linalg.LinAlgError) as exc:
            raise NumericalSafetyError(f"{name} must be strictly SPD") from exc
        cholesky_succeeded = True
    elif minimum < -psd_tolerance:
        raise NumericalSafetyError(
            f"{name} must be PSD; minimum eigenvalue {minimum:.3e} is below "
            f"{-psd_tolerance:.3e}"
        )
    return CovarianceDiagnostics(
        relative_asymmetry=relative_asymmetry,
        minimum_eigenvalue=minimum,
        maximum_eigenvalue=maximum,
        cholesky_succeeded=cholesky_succeeded,
    )


def assert_positive_definite(matrix: ArrayLike, *, name: str = "matrix") -> CovarianceDiagnostics:
    """Require a symmetric positive-definite matrix and return diagnostics."""

    return covariance_diagnostics(matrix, name=name, require_spd=True)


def assert_positive_semidefinite(matrix: ArrayLike, *, name: str = "matrix") -> CovarianceDiagnostics:
    """Require a symmetric positive-semidefinite matrix and return diagnostics."""

    return covariance_diagnostics(matrix, name=name, require_spd=False)


def skew(vector: ArrayLike) -> np.ndarray:
    """Return ``[a]_x`` such that ``[a]_x b == cross(a, b)``."""

    x, y, z = _vector(vector, 3, "vector")
    return np.array(
        [[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]],
        dtype=np.float64,
    )


def quat_normalize(quaternion: ArrayLike, *, name: str = "quaternion") -> np.ndarray:
    """Normalize a scalar-first quaternion, rejecting degenerate input."""

    q = _vector(quaternion, 4, name)
    norm = float(np.linalg.norm(q))
    if norm <= np.finfo(np.float64).tiny:
        raise NumericalSafetyError(f"{name} norm is too small to normalize")
    return q / norm


def quat_conjugate(quaternion: ArrayLike) -> np.ndarray:
    """Return the Hamilton quaternion conjugate without normalizing."""

    q = _vector(quaternion, 4, "quaternion")
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float64)


def quat_inverse(quaternion: ArrayLike) -> np.ndarray:
    """Return the algebraic inverse of a nonzero Hamilton quaternion."""

    q = _vector(quaternion, 4, "quaternion")
    norm_squared = float(q @ q)
    if norm_squared <= np.finfo(np.float64).tiny:
        raise NumericalSafetyError("quaternion norm is too small to invert")
    return quat_conjugate(q) / norm_squared


def quat_multiply(left: ArrayLike, right: ArrayLike) -> np.ndarray:
    """Hamilton product for scalar-first quaternions."""

    q = _vector(left, 4, "left")
    p = _vector(right, 4, "right")
    scalar = q[0] * p[0] - float(q[1:] @ p[1:])
    vector = q[0] * p[1:] + p[0] * q[1:] + np.cross(q[1:], p[1:])
    return np.concatenate((np.array([scalar], dtype=np.float64), vector))


def quat_exp(rotation_vector: ArrayLike) -> np.ndarray:
    """SO(3) exponential as a unit scalar-first quaternion."""

    phi = _vector(rotation_vector, 3, "rotation_vector")
    theta = float(np.linalg.norm(phi))
    theta_squared = theta * theta
    if theta < 1.0e-8:
        scalar = 1.0 - theta_squared / 8.0 + theta_squared * theta_squared / 384.0
        scale = 0.5 - theta_squared / 48.0 + theta_squared * theta_squared / 3840.0
    else:
        half_theta = 0.5 * theta
        scalar = float(np.cos(half_theta))
        scale = float(np.sin(half_theta) / theta)
    return quat_normalize(np.concatenate(([scalar], scale * phi)), name="Exp_q(rotation_vector)")


def quat_log(quaternion: ArrayLike) -> np.ndarray:
    """Shortest-arc SO(3) logarithm with estimate-independent hemisphere choice."""

    q = quat_normalize(quaternion)
    if abs(float(q[0])) <= _EXACT_PI_TIE_TOL:
        significant = np.flatnonzero(np.abs(q[1:]) > _EXACT_PI_TIE_TOL)
        if significant.size and q[1 + int(significant[0])] < 0.0:
            q = -q
        q[q == 0.0] = 0.0
    elif q[0] < 0.0:
        q = -q
    vector_norm = float(np.linalg.norm(q[1:]))
    if vector_norm < 1.0e-14:
        return 2.0 * q[1:]
    angle = 2.0 * float(np.arctan2(vector_norm, q[0]))
    return (angle / vector_norm) * q[1:]


def align_quaternion(quaternion: ArrayLike, reference: ArrayLike) -> np.ndarray:
    """Normalize and align a quaternion to the reference hemisphere."""

    q = quat_normalize(quaternion, name="quaternion")
    ref = quat_normalize(reference, name="reference")
    return -q if float(q @ ref) < 0.0 else q


def quat_to_dcm(quaternion: ArrayLike) -> np.ndarray:
    """Return active ``R_NB`` mapping body coordinates to navigation coordinates."""

    q = quat_normalize(quaternion)
    scalar = q[0]
    vector = q[1:]
    return (
        (scalar * scalar - float(vector @ vector)) * np.eye(3, dtype=np.float64)
        + 2.0 * np.outer(vector, vector)
        + 2.0 * scalar * skew(vector)
    )


def dcm_to_quat(rotation: ArrayLike) -> np.ndarray:
    """Convert a proper active rotation matrix to a deterministic quaternion."""

    matrix = _matrix(rotation, (3, 3), "rotation")
    orthogonality_error = float(np.linalg.norm(matrix.T @ matrix - np.eye(3), ord="fro"))
    determinant = float(np.linalg.det(matrix))
    if orthogonality_error > 1.0e-10 or abs(determinant - 1.0) > 1.0e-10:
        raise NumericalSafetyError(
            "rotation must be a proper orthogonal matrix; "
            f"orthogonality_error={orthogonality_error:.3e}, det={determinant:.16g}"
        )

    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = 2.0 * np.sqrt(trace + 1.0)
        q = np.array(
            [
                0.25 * scale,
                (matrix[2, 1] - matrix[1, 2]) / scale,
                (matrix[0, 2] - matrix[2, 0]) / scale,
                (matrix[1, 0] - matrix[0, 1]) / scale,
            ],
            dtype=np.float64,
        )
    else:
        index = int(np.argmax(np.diag(matrix)))
        if index == 0:
            scale = 2.0 * np.sqrt(max(0.0, 1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]))
            q = np.array(
                [
                    (matrix[2, 1] - matrix[1, 2]) / scale,
                    0.25 * scale,
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                ],
                dtype=np.float64,
            )
        elif index == 1:
            scale = 2.0 * np.sqrt(max(0.0, 1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]))
            q = np.array(
                [
                    (matrix[0, 2] - matrix[2, 0]) / scale,
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    0.25 * scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                ],
                dtype=np.float64,
            )
        else:
            scale = 2.0 * np.sqrt(max(0.0, 1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]))
            q = np.array(
                [
                    (matrix[1, 0] - matrix[0, 1]) / scale,
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                    0.25 * scale,
                ],
                dtype=np.float64,
            )
    q = quat_normalize(q, name="dcm quaternion")
    return -q if q[0] < 0.0 else q


def quat_geodesic_angle(left: ArrayLike, right: ArrayLike) -> float:
    """Sign-invariant SO(3) geodesic angle in radians."""

    q = quat_normalize(left, name="left")
    p = quat_normalize(right, name="right")
    dot = min(1.0, max(0.0, abs(float(q @ p))))
    return 2.0 * float(np.arccos(dot))


def right_jacobian_so3(rotation_vector: ArrayLike) -> np.ndarray:
    """Exact SO(3) right Jacobian with a stable near-zero series."""

    phi = _vector(rotation_vector, 3, "rotation_vector")
    theta = float(np.linalg.norm(phi))
    theta_squared = theta * theta
    cross = skew(phi)
    if theta < 1.0e-6:
        a = 0.5 - theta_squared / 24.0 + theta_squared * theta_squared / 720.0
        b = 1.0 / 6.0 - theta_squared / 120.0 + theta_squared * theta_squared / 5040.0
    else:
        a = (1.0 - float(np.cos(theta))) / theta_squared
        b = (theta - float(np.sin(theta))) / (theta_squared * theta)
    return np.eye(3, dtype=np.float64) - a * cross + b * (cross @ cross)


def body_vector_prediction(q_NB: ArrayLike, reference_N: ArrayLike) -> np.ndarray:
    """Predict a navigation-frame reference vector in body coordinates."""

    reference = _vector(reference_N, 3, "reference_N")
    return quat_to_dcm(q_NB).T @ reference


def body_vector_jacobian(q_NB: ArrayLike, reference_N: ArrayLike) -> np.ndarray:
    """Right-error Jacobian of residual ``measurement_B - prediction_B``."""

    prediction = body_vector_prediction(q_NB, reference_N)
    result = np.zeros((3, 6), dtype=np.float64)
    result[:, :3] = skew(prediction)
    return result


def sun_tangent_basis(prediction_B: ArrayLike) -> np.ndarray:
    """Deterministic orthonormal tangent basis ``U(h)`` for a unit vector."""

    prediction = _vector(prediction_B, 3, "prediction_B")
    norm = float(np.linalg.norm(prediction))
    if abs(norm - 1.0) > 1.0e-12:
        raise NumericalSafetyError(f"prediction_B must be unit length, got norm={norm:.16g}")
    basis_index = int(np.argmin(np.abs(prediction)))
    candidate = np.eye(3, dtype=np.float64)[:, basis_index]
    first = candidate - float(candidate @ prediction) * prediction
    first /= np.linalg.norm(first)
    second = np.cross(prediction, first)
    second /= np.linalg.norm(second)
    return np.column_stack((first, second))


def sun_tangent_jacobian(
    q_NB: ArrayLike,
    reference_N: ArrayLike,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return unit prediction, deterministic tangent basis, and 2x6 Jacobian."""

    reference = _vector(reference_N, 3, "reference_N")
    if abs(float(np.linalg.norm(reference)) - 1.0) > 1.0e-12:
        raise NumericalSafetyError("reference_N must be unit length for a sun tangent update")
    prediction = body_vector_prediction(q_NB, reference)
    basis = sun_tangent_basis(prediction)
    jacobian = np.zeros((2, 6), dtype=np.float64)
    jacobian[:, :3] = basis.T @ skew(prediction)
    return prediction, basis, jacobian


def star_tracker_residual(q_hat_NB: ArrayLike, q_measurement_NB: ArrayLike) -> np.ndarray:
    """Return the right-local three-dimensional star-tracker residual."""

    estimate = quat_normalize(q_hat_NB, name="q_hat_NB")
    measurement = align_quaternion(q_measurement_NB, estimate)
    relative = quat_multiply(quat_inverse(estimate), measurement)
    return quat_log(relative)


def continuous_error_matrices(omega_corrected: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    """Return the locked random-walk-bias continuous ``F`` and ``G``."""

    omega = _vector(omega_corrected, 3, "omega_corrected")
    f = np.zeros((6, 6), dtype=np.float64)
    f[:3, :3] = -skew(omega)
    f[:3, 3:] = -np.eye(3, dtype=np.float64)
    g = np.zeros((6, 6), dtype=np.float64)
    g[:3, :3] = -np.eye(3, dtype=np.float64)
    g[3:, 3:] = np.eye(3, dtype=np.float64)
    return f, g


def continuous_noise_covariance(S_g: ArrayLike, S_b: ArrayLike) -> np.ndarray:
    """Build ``Q_c = blockdiag(S_g, S_b)`` after PSD validation."""

    gyro = _matrix(S_g, (3, 3), "S_g")
    bias = _matrix(S_b, (3, 3), "S_b")
    assert_positive_semidefinite(gyro, name="S_g")
    assert_positive_semidefinite(bias, name="S_b")
    q_c = np.zeros((6, 6), dtype=np.float64)
    q_c[:3, :3] = gyro
    q_c[3:, 3:] = bias
    return q_c


def discretize_van_loan(
    F: ArrayLike,
    G: ArrayLike,
    Q_c: ArrayLike,
    dt: float,
) -> DiscretizationResult:
    """Exact constant-coefficient ``Phi`` and ``Q_d`` via the locked block form."""

    f = _square_matrix(F, "F")
    dimension = f.shape[0]
    g = _matrix(G, (dimension, dimension), "G")
    q_c = _matrix(Q_c, (dimension, dimension), "Q_c")
    assert_positive_semidefinite(q_c, name="Q_c")
    delta_time = float(dt)
    if not np.isfinite(delta_time) or delta_time < 0.0:
        raise NumericalSafetyError(f"dt must be finite and nonnegative, got {dt!r}")
    if delta_time == 0.0:
        return DiscretizationResult(
            Phi=np.eye(dimension, dtype=np.float64),
            Q_d=np.zeros((dimension, dimension), dtype=np.float64),
            qd_symmetrization_relative_correction=0.0,
        )
    diffusion = g @ q_c @ g.T
    van_loan = np.zeros((2 * dimension, 2 * dimension), dtype=np.float64)
    van_loan[:dimension, :dimension] = f
    van_loan[:dimension, dimension:] = diffusion
    van_loan[dimension:, dimension:] = -f.T
    exponential = expm(van_loan * delta_time)
    phi = exponential[:dimension, :dimension]
    q_d_raw = exponential[:dimension, dimension:] @ phi.T
    q_d, correction = _symmetrize_roundoff(q_d_raw, name="Q_d")
    assert_positive_semidefinite(q_d, name="Q_d")
    return DiscretizationResult(
        Phi=phi,
        Q_d=q_d,
        qd_symmetrization_relative_correction=correction,
    )


def cholesky_solve_spd(matrix: ArrayLike, rhs: ArrayLike, *, name: str = "matrix") -> np.ndarray:
    """Solve an SPD system using Cholesky and triangular solves only."""

    value = _square_matrix(matrix, name)
    assert_positive_definite(value, name=name)
    right_hand_side = _array(rhs, "rhs")
    if right_hand_side.ndim not in (1, 2) or right_hand_side.shape[0] != value.shape[0]:
        raise ValueError(
            f"rhs must have leading dimension {value.shape[0]}, got {right_hand_side.shape}"
        )
    lower = cholesky(value, lower=True, check_finite=True)
    intermediate = solve_triangular(lower, right_hand_side, lower=True, check_finite=True)
    return solve_triangular(lower.T, intermediate, lower=False, check_finite=True)


def kalman_gain(P_minus: ArrayLike, H: ArrayLike, R: ArrayLike) -> KalmanGainResult:
    """Compute ``S`` and ``K`` with an explicit fail-loud SPD solve."""

    p = _square_matrix(P_minus, "P_minus")
    assert_positive_definite(p, name="P_minus")
    h = _array(H, "H")
    if h.ndim != 2 or h.shape[1] != p.shape[0]:
        raise ValueError(f"H must have shape (m, {p.shape[0]}), got {h.shape}")
    r = _matrix(R, (h.shape[0], h.shape[0]), "R")
    assert_positive_definite(r, name="R")
    s_raw = h @ p @ h.T + r
    s, correction = _symmetrize_roundoff(s_raw, name="S")
    assert_positive_definite(s, name="S")
    pht = p @ h.T
    gain = cholesky_solve_spd(s, pht.T, name="S").T
    return KalmanGainResult(K=gain, S=s, s_symmetrization_relative_correction=correction)


def joseph_covariance_update(
    P_minus: ArrayLike,
    K: ArrayLike,
    H: ArrayLike,
    R: ArrayLike,
) -> tuple[np.ndarray, float]:
    """Return Joseph-form covariance before multiplicative reset."""

    p = _square_matrix(P_minus, "P_minus")
    assert_positive_definite(p, name="P_minus")
    h = _array(H, "H")
    if h.ndim != 2 or h.shape[1] != p.shape[0]:
        raise ValueError(f"H must have shape (m, {p.shape[0]}), got {h.shape}")
    k = _matrix(K, (p.shape[0], h.shape[0]), "K")
    r = _matrix(R, (h.shape[0], h.shape[0]), "R")
    assert_positive_definite(r, name="R")
    identity_minus_kh = np.eye(p.shape[0], dtype=np.float64) - k @ h
    raw = identity_minus_kh @ p @ identity_minus_kh.T + k @ r @ k.T
    updated, correction = _symmetrize_roundoff(raw, name="P_c")
    assert_positive_definite(updated, name="P_c")
    return updated, correction


def inject_error_state(
    q_minus_NB: ArrayLike,
    b_minus_g: ArrayLike,
    delta_x: ArrayLike,
) -> tuple[np.ndarray, np.ndarray]:
    """Right-inject attitude correction and add the gyro-bias correction."""

    q_minus = quat_normalize(q_minus_NB, name="q_minus_NB")
    b_minus = _vector(b_minus_g, 3, "b_minus_g")
    correction = _vector(delta_x, 6, "delta_x")
    q_plus = quat_normalize(
        quat_multiply(q_minus, quat_exp(correction[:3])),
        name="q_plus_NB",
    )
    b_plus = b_minus + correction[3:]
    return q_plus, b_plus


def reset_covariance(P_c: ArrayLike, delta_theta: ArrayLike) -> tuple[np.ndarray, np.ndarray, float]:
    """Transport covariance to the reset right-local tangent with exact ``J_r``."""

    p_c = _matrix(P_c, (6, 6), "P_c")
    assert_positive_definite(p_c, name="P_c")
    attitude_correction = _vector(delta_theta, 3, "delta_theta")
    reset = np.eye(6, dtype=np.float64)
    reset[:3, :3] = right_jacobian_so3(attitude_correction)
    raw = reset @ p_c @ reset.T
    p_plus, correction = _symmetrize_roundoff(raw, name="P_plus")
    assert_positive_definite(p_plus, name="P_plus")
    return p_plus, reset, correction


def propagate_state(
    state: MEKFState,
    omega_m: ArrayLike,
    dt: float,
    Q_c: ArrayLike,
) -> PropagationResult:
    """Propagate nominal ``[q_NB, b_g]`` and its six-dimensional covariance."""

    if not isinstance(state, MEKFState):
        raise TypeError("state must be an MEKFState")
    measurement = _vector(omega_m, 3, "omega_m")
    delta_time = float(dt)
    if not np.isfinite(delta_time) or delta_time < 0.0:
        raise NumericalSafetyError(f"dt must be finite and nonnegative, got {dt!r}")
    corrected = measurement - state.b_g
    f, g = continuous_error_matrices(corrected)
    discretization = discretize_van_loan(f, g, Q_c, delta_time)
    q_minus = quat_normalize(
        quat_multiply(state.q_NB, quat_exp(corrected * delta_time)),
        name="propagated q_NB",
    )
    p_raw = (
        discretization.Phi @ state.P @ discretization.Phi.T
        + discretization.Q_d
    )
    p_minus, p_correction = _symmetrize_roundoff(p_raw, name="P_minus")
    assert_positive_definite(p_minus, name="P_minus")
    propagated = MEKFState(q_NB=q_minus, b_g=state.b_g, P=p_minus)
    return PropagationResult(
        state=propagated,
        omega_corrected=corrected,
        F=f,
        G=g,
        Phi=discretization.Phi,
        Q_d=discretization.Q_d,
        qd_symmetrization_relative_correction=(
            discretization.qd_symmetrization_relative_correction
        ),
        p_symmetrization_relative_correction=p_correction,
    )


def star_tracker_update(
    state: MEKFState,
    q_measurement_NB: ArrayLike,
    R_ST: ArrayLike,
) -> StarTrackerUpdateResult:
    """Perform local ST update, right injection, Joseph update, and reset."""

    if not isinstance(state, MEKFState):
        raise TypeError("state must be an MEKFState")
    residual = star_tracker_residual(state.q_NB, q_measurement_NB)
    h = np.zeros((3, 6), dtype=np.float64)
    h[:, :3] = np.eye(3, dtype=np.float64)
    r = _matrix(R_ST, (3, 3), "R_ST")
    gain_result = kalman_gain(state.P, h, r)
    delta_x = gain_result.K @ residual
    p_c, pc_correction = joseph_covariance_update(state.P, gain_result.K, h, r)
    q_plus, b_plus = inject_error_state(state.q_NB, state.b_g, delta_x)
    p_plus, reset, reset_correction = reset_covariance(p_c, delta_x[:3])
    posterior = MEKFState(q_NB=q_plus, b_g=b_plus, P=p_plus)
    return StarTrackerUpdateResult(
        state=posterior,
        residual=residual,
        H=h,
        R=r.copy(),
        S=gain_result.S,
        K=gain_result.K,
        delta_x=delta_x,
        P_c=p_c,
        G_reset=reset,
        s_symmetrization_relative_correction=(
            gain_result.s_symmetrization_relative_correction
        ),
        pc_symmetrization_relative_correction=pc_correction,
        p_reset_symmetrization_relative_correction=reset_correction,
    )
