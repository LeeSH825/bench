"""Causal compensation encoders and right-local Split gain estimator.

This module is the deployable model namespace.  It deliberately has no
generator-truth, calibration, diagnostic-oracle, event-label, or metric input.
The reference ``P`` carried by :class:`MEKFState` is used only for propagation
and tangent reset mechanics; learned factors are not physical covariances.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
from torch import nn

from bench.estimators.mekf import (
    MEKFState,
    body_vector_jacobian,
    body_vector_prediction,
    inject_error_state,
    joseph_covariance_update,
    kalman_gain,
    propagate_state,
    reset_covariance,
    right_jacobian_so3,
)


FEATURE_DIM = 8
GAIN_SHAPE = (6, 3)


def mekf_reset_state_digest(initial_state: MEKFState, initial_time_s: float) -> str:
    """Digest the actual state object and timestamp supplied to a trajectory reset."""

    digest = hashlib.sha256(b"side-gyro-mag-actual-reset-state-v1\0")
    for value in (initial_state.q_NB, initial_state.b_g, initial_state.P):
        array = np.asarray(value)
        digest.update(str(array.dtype).encode() + b"\0")
        digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
        digest.update(np.ascontiguousarray(array).tobytes())
    digest.update(np.asarray([float(initial_time_s)], dtype="<f8").tobytes())
    return digest.hexdigest()


def _hidden_tensor_lineage_payload(label: str, value: torch.Tensor) -> bytes:
    """Serialize one actual post-transition hidden tensor without lossy conversion."""

    tensor = value.detach().cpu().contiguous()
    array = tensor.numpy()
    return b"".join((
        label.encode(), b"\0", str(tensor.dtype).encode(), b"\0",
        np.asarray(tensor.shape, dtype="<i8").tobytes(),
        np.ascontiguousarray(array).tobytes(),
    ))


def _tensor3(value: torch.Tensor | np.ndarray, name: str) -> torch.Tensor:
    result = value.to(dtype=torch.float64) if isinstance(value, torch.Tensor) else torch.tensor(
        np.array(value, dtype=np.float64, copy=True), dtype=torch.float64,
    )
    if result.shape != (3,) or not torch.isfinite(result).all():
        raise ValueError(f"{name} must be finite with shape (3,)")
    return result


@dataclass(frozen=True)
class EncoderOutput:
    corrected_B: torch.Tensor
    feature: torch.Tensor


@dataclass(frozen=True)
class GainOutput:
    G1: torch.Tensor
    G2: torch.Tensor
    K: torch.Tensor


@dataclass(frozen=True)
class EstimatorStep:
    state: MEKFState
    gyro_corrected_B: np.ndarray
    mag_corrected_B: np.ndarray
    gyro_feature: np.ndarray
    mag_feature: np.ndarray
    innovation: np.ndarray
    H: np.ndarray
    G1: np.ndarray
    G2: np.ndarray
    K: np.ndarray
    delta_x: np.ndarray
    stage_order: tuple[str, ...]


class _CausalSensorEncoder(nn.Module):
    """One strictly forward GRUCell with corrected-vector and 8D heads."""

    def __init__(self, hidden_dim: int = 16) -> None:
        super().__init__()
        if hidden_dim != 16:
            raise ValueError("the frozen encoder hidden width is 16")
        self.hidden_dim = hidden_dim
        self.cell = nn.GRUCell(5, hidden_dim, dtype=torch.float64)
        self.correction_head = nn.Linear(hidden_dim, 3, dtype=torch.float64)
        self.feature_head = nn.Linear(hidden_dim, FEATURE_DIM, dtype=torch.float64)
        self.register_buffer("input_mean", torch.zeros(3, dtype=torch.float64))
        self.register_buffer("input_std", torch.ones(3, dtype=torch.float64))
        nn.init.zeros_(self.correction_head.weight)
        nn.init.zeros_(self.correction_head.bias)
        self._hidden: torch.Tensor | None = None
        self._last_timestamp: float | None = None

    def reset_trajectory(self) -> None:
        self._hidden = None
        self._last_timestamp = None

    def install_normalization(self, mean: np.ndarray, std: np.ndarray) -> None:
        mean_value, std_value = _tensor3(mean, "mean"), _tensor3(std, "std")
        if torch.any(std_value <= 0):
            raise ValueError("normalization std must be positive")
        self.input_mean.copy_(mean_value)
        self.input_std.copy_(std_value)

    def forward_step(
        self,
        measurement_S: torch.Tensor | np.ndarray,
        timestamp_s: float,
        valid: bool,
    ) -> EncoderOutput:
        measurement = _tensor3(measurement_S, "measurement_S")
        timestamp = float(timestamp_s)
        if not np.isfinite(timestamp) or timestamp < 0:
            raise ValueError("timestamp must be finite and nonnegative")
        if self._last_timestamp is not None and timestamp < self._last_timestamp:
            raise ValueError("future/reordered sample injection is forbidden")
        dt = 0.0 if self._last_timestamp is None else timestamp - self._last_timestamp
        if self._hidden is None:
            self._hidden = torch.zeros(self.hidden_dim, dtype=torch.float64, device=measurement.device)
        normalized = (measurement - self.input_mean) / self.input_std
        inputs = torch.cat((normalized, torch.tensor(
            [dt, float(bool(valid))], dtype=torch.float64, device=measurement.device,
        )))
        self._hidden = self.cell(inputs, self._hidden)
        self._last_timestamp = timestamp
        correction = measurement + bool(valid) * self.correction_head(self._hidden)
        feature = self.feature_head(self._hidden)
        if feature.shape != (FEATURE_DIM,):
            raise RuntimeError("feature head violated the fixed 8D contract")
        return EncoderOutput(corrected_B=correction, feature=feature)


class GyroEncoder(_CausalSensorEncoder):
    """Causal sensor-frame gyro encoder producing body-rate and an 8D feature."""


class MagEncoder(_CausalSensorEncoder):
    """Causal sensor-frame magnetometer encoder producing body vector and 8D feature."""


class SplitGainBackbone(nn.Module):
    """Right-local ``K = G1 H.T G2`` with branch-specific FiLM."""

    def __init__(self, prior_hidden_dim: int = 32, measurement_hidden_dim: int = 32) -> None:
        super().__init__()
        if (prior_hidden_dim, measurement_hidden_dim) != (32, 32):
            raise ValueError("frozen Split recurrent widths are 32 and 32")
        self.prior_cell = nn.GRUCell(6, prior_hidden_dim, dtype=torch.float64)
        self.measurement_cell = nn.GRUCell(3, measurement_hidden_dim, dtype=torch.float64)
        self.g1_head = nn.Linear(prior_hidden_dim, 36, dtype=torch.float64)
        self.g2_head = nn.Linear(measurement_hidden_dim, 9, dtype=torch.float64)
        self.gyro_film = nn.Linear(FEATURE_DIM, 72, dtype=torch.float64)
        self.mag_film = nn.Linear(FEATURE_DIM, 18, dtype=torch.float64)
        self._prior_hidden: torch.Tensor | None = None
        self._measurement_hidden: torch.Tensor | None = None
        self._transition_count = 0
        nn.init.zeros_(self.g1_head.weight)
        nn.init.zeros_(self.g2_head.weight)
        base_g1 = torch.eye(6, dtype=torch.float64)
        base_g1[3:, :3] = 0.02 * torch.eye(3, dtype=torch.float64)
        with torch.no_grad():
            self.g1_head.bias.copy_(base_g1.reshape(-1))
            self.g2_head.bias.copy_(torch.eye(3, dtype=torch.float64).reshape(-1))
        # gamma_delta=0 and beta=0 is exact identity at initialization.
        nn.init.zeros_(self.gyro_film.weight)
        nn.init.zeros_(self.gyro_film.bias)
        nn.init.zeros_(self.mag_film.weight)
        nn.init.zeros_(self.mag_film.bias)

    def reset_trajectory(self) -> None:
        self._prior_hidden = None
        self._measurement_hidden = None
        self._transition_count = 0

    @property
    def transition_count(self) -> int:
        return self._transition_count

    def forward_step(
        self,
        prior_input: torch.Tensor | np.ndarray,
        innovation: torch.Tensor | np.ndarray,
        H: torch.Tensor | np.ndarray,
        gyro_feature: torch.Tensor | np.ndarray,
        mag_feature: torch.Tensor | np.ndarray,
        *,
        feature_enabled: bool,
    ) -> GainOutput:
        prior = torch.as_tensor(prior_input, dtype=torch.float64)
        residual = _tensor3(innovation, "innovation")
        jacobian = torch.as_tensor(H, dtype=torch.float64)
        gyro_context = torch.as_tensor(gyro_feature, dtype=torch.float64)
        mag_context = torch.as_tensor(mag_feature, dtype=torch.float64)
        if prior.shape != (6,) or jacobian.shape != (3, 6):
            raise ValueError("prior/H shapes must be (6,) and (3,6)")
        if gyro_context.shape != (FEATURE_DIM,) or mag_context.shape != (FEATURE_DIM,):
            raise ValueError("each branch feature must be exactly 8D")
        if self._prior_hidden is None:
            self._prior_hidden = torch.zeros(32, dtype=torch.float64, device=prior.device)
        if self._measurement_hidden is None:
            self._measurement_hidden = torch.zeros(32, dtype=torch.float64, device=residual.device)
        self._prior_hidden = self.prior_cell(prior, self._prior_hidden)
        self._measurement_hidden = self.measurement_cell(residual, self._measurement_hidden)
        self._transition_count += 1
        base_g1 = self.g1_head(self._prior_hidden).reshape(6, 6)
        base_g2 = self.g2_head(self._measurement_hidden).reshape(3, 3)
        if feature_enabled:
            g1_affine = self.gyro_film(gyro_context)
            g2_affine = self.mag_film(mag_context)
            g1_delta, g1_beta = g1_affine[:36].reshape(6, 6), g1_affine[36:].reshape(6, 6)
            g2_delta, g2_beta = g2_affine[:9].reshape(3, 3), g2_affine[9:].reshape(3, 3)
            g1 = (1.0 + g1_delta) * base_g1 + g1_beta
            g2 = (1.0 + g2_delta) * base_g2 + g2_beta
        else:
            # Bypass is bitwise exact, independent of feature values.
            g1, g2 = base_g1, base_g2
        gain = g1 @ jacobian.T @ g2
        if gain.shape != GAIN_SHAPE:
            raise RuntimeError("Split gain must have shape (6,3)")
        return GainOutput(G1=g1, G2=g2, K=gain)


class SideEstimator(nn.Module):
    """Reset-per-trajectory causal estimator for N0, N2 and N3 deployable paths."""

    def __init__(self, mode: Literal["raw", "learned"], *, feature_enabled: bool) -> None:
        super().__init__()
        if mode not in ("raw", "learned"):
            raise ValueError("deployable mode must be raw or learned")
        if mode == "raw" and feature_enabled:
            raise ValueError("raw N0 path has FiLM off")
        self.mode = mode
        self.feature_enabled = bool(feature_enabled)
        self.gyro_encoder = GyroEncoder()
        self.mag_encoder = MagEncoder()
        self.backbone = SplitGainBackbone()
        self._state: MEKFState | None = None
        self._last_time: float | None = None
        self._pending_gyro_time: float | None = None
        self._initial_state_sha256: str | None = None
        self._trajectory_owner_token: str | None = None
        self._recurrent_transition_lineage: list[bytes] = []
        self._Q_c = np.diag(np.r_[np.full(3, 1e-8), np.full(3, 1e-12)]).astype(np.float64)

    def install_normalization(
        self,
        gyro_mean: np.ndarray,
        gyro_std: np.ndarray,
        mag_mean: np.ndarray,
        mag_std: np.ndarray,
    ) -> None:
        self.gyro_encoder.install_normalization(gyro_mean, gyro_std)
        self.mag_encoder.install_normalization(mag_mean, mag_std)

    def reset_trajectory(
        self,
        initial_state: MEKFState,
        initial_time_s: float = 0.0,
        *,
        trajectory_owner_token: str = "direct-runtime",
    ) -> None:
        if not isinstance(initial_state, MEKFState):
            raise TypeError("initial_state must be MEKFState")
        if not trajectory_owner_token:
            raise ValueError("trajectory_owner_token must be nonempty")
        self._state = initial_state
        self._last_time = float(initial_time_s)
        self._pending_gyro_time = None
        self._initial_state_sha256 = mekf_reset_state_digest(initial_state, initial_time_s)
        self._trajectory_owner_token = str(trajectory_owner_token)
        self._recurrent_transition_lineage = []
        self.gyro_encoder.reset_trajectory()
        self.mag_encoder.reset_trajectory()
        self.backbone.reset_trajectory()

    @property
    def state(self) -> MEKFState:
        if self._state is None:
            raise RuntimeError("reset_trajectory is required")
        return self._state

    @property
    def initial_state_sha256(self) -> str:
        if self._initial_state_sha256 is None:
            raise RuntimeError("reset_trajectory is required")
        return self._initial_state_sha256

    @property
    def recurrent_history_owner_token(self) -> str:
        if self._trajectory_owner_token is None:
            raise RuntimeError("reset_trajectory is required")
        return self._trajectory_owner_token

    def recurrent_history_provenance_sha256(self) -> str:
        """Hash lineage events recorded by the target's executed GRU transitions."""

        if self._trajectory_owner_token is None:
            raise RuntimeError("reset_trajectory is required")
        digest = hashlib.sha256(b"side-gyro-mag-actual-recurrent-lineage-v1\0")
        digest.update(self._trajectory_owner_token.encode() + b"\0")
        for event in self._recurrent_transition_lineage:
            digest.update(event)
        return digest.hexdigest()

    def compensate_gyro(self, measurement_S: np.ndarray, timestamp_s: float, valid: bool = True) -> EncoderOutput:
        if self._pending_gyro_time is not None:
            raise ValueError("magnetometer update must complete before next gyro")
        if self._last_time is None or timestamp_s <= self._last_time:
            raise ValueError("gyro must advance trajectory time strictly")
        if self.mode == "learned":
            output = self.gyro_encoder.forward_step(measurement_S, timestamp_s, valid)
        else:
            raw = _tensor3(measurement_S, "measurement_S")
            output = EncoderOutput(raw, torch.zeros(FEATURE_DIM, dtype=torch.float64))
        self._pending_gyro_time = float(timestamp_s)
        return output

    def propagate(self, gyro_output: EncoderOutput) -> None:
        if self._pending_gyro_time is None or self._last_time is None:
            raise ValueError("gyro compensation must precede propagation")
        result = propagate_state(
            self.state,
            gyro_output.corrected_B.detach().cpu().numpy(),
            self._pending_gyro_time - self._last_time,
            self._Q_c,
        )
        self._state = result.state
        self._last_time = self._pending_gyro_time

    def compensate_magnetometer(self, measurement_S: np.ndarray, timestamp_s: float, valid: bool = True) -> EncoderOutput:
        if self._pending_gyro_time is None or timestamp_s != self._pending_gyro_time or timestamp_s != self._last_time:
            raise ValueError("magnetometer compensation requires completed same-time propagation")
        if self.mode == "learned":
            return self.mag_encoder.forward_step(measurement_S, timestamp_s, valid)
        raw = _tensor3(measurement_S, "measurement_S")
        return EncoderOutput(raw, torch.zeros(FEATURE_DIM, dtype=torch.float64))

    def update(
        self,
        gyro_output: EncoderOutput,
        mag_output: EncoderOutput,
        m_model_N_onboard: np.ndarray,
    ) -> EstimatorStep:
        if self._pending_gyro_time is None or self._last_time != self._pending_gyro_time:
            raise ValueError("update-before-propagation is forbidden")
        prediction = body_vector_prediction(self.state.q_NB, m_model_N_onboard)
        innovation_np = mag_output.corrected_B.detach().cpu().numpy() - prediction
        H_np = body_vector_jacobian(self.state.q_NB, m_model_N_onboard)
        prior = np.r_[gyro_output.corrected_B.detach().cpu().numpy(), self.state.b_g]
        gain = self.backbone.forward_step(
            prior, innovation_np, H_np, gyro_output.feature, mag_output.feature,
            feature_enabled=self.feature_enabled,
        )
        transition_index = self.backbone.transition_count
        if gain.K.shape != GAIN_SHAPE or gain.G1.shape != (6, 6) or gain.G2.shape != (3, 3):
            raise RuntimeError("SideEstimator rejected invalid Split factor/gain shape")
        if transition_index > len(self._recurrent_transition_lineage):
            if self.backbone._prior_hidden is None or self.backbone._measurement_hidden is None:
                raise RuntimeError("backbone transition lineage is incomplete")
            self._recurrent_transition_lineage.append(
                b"transition\0" + np.asarray([transition_index], dtype="<i8").tobytes()
                + _hidden_tensor_lineage_payload("prior_hidden", self.backbone._prior_hidden)
                + _hidden_tensor_lineage_payload(
                    "measurement_hidden", self.backbone._measurement_hidden,
                )
            )
        delta = gain.K @ torch.as_tensor(innovation_np, dtype=torch.float64)
        delta_np = delta.detach().cpu().numpy()
        q_plus, b_plus = inject_error_state(self.state.q_NB, self.state.b_g, delta_np)
        # Shadow/reference covariance is only transported to the reset tangent.
        p_plus, reset_matrix, _ = reset_covariance(self.state.P, delta_np[:3])
        expected_reset = np.eye(6, dtype=np.float64)
        expected_reset[:3, :3] = right_jacobian_so3(delta_np[:3])
        if not np.allclose(reset_matrix, expected_reset, rtol=0.0, atol=1e-14):
            raise RuntimeError("SideEstimator rejected non-right-Jacobian reset")
        self._state = MEKFState(q_plus, b_plus, p_plus)
        self._pending_gyro_time = None
        return EstimatorStep(
            state=self.state,
            gyro_corrected_B=gyro_output.corrected_B.detach().cpu().numpy(),
            mag_corrected_B=mag_output.corrected_B.detach().cpu().numpy(),
            gyro_feature=gyro_output.feature.detach().cpu().numpy(),
            mag_feature=mag_output.feature.detach().cpu().numpy(),
            innovation=innovation_np, H=H_np,
            G1=gain.G1.detach().cpu().numpy(), G2=gain.G2.detach().cpu().numpy(),
            K=gain.K.detach().cpu().numpy(), delta_x=delta_np,
            stage_order=("gyro_compensation", "propagation", "mag_compensation", "mag_update"),
        )

    def step_pair(
        self,
        gyro_measurement_S: np.ndarray,
        mag_measurement_S: np.ndarray,
        timestamp_s: float,
        m_model_N_onboard: np.ndarray,
        *,
        gyro_valid: bool = True,
        mag_valid: bool = True,
    ) -> EstimatorStep:
        gyro = self.compensate_gyro(gyro_measurement_S, timestamp_s, gyro_valid)
        self.propagate(gyro)
        mag = self.compensate_magnetometer(mag_measurement_S, timestamp_s, mag_valid)
        return self.update(gyro, mag, m_model_N_onboard)


def classical_vector_update(state: MEKFState, corrected_mag_B: np.ndarray, m_model_N_onboard: np.ndarray) -> MEKFState:
    """Classical shadow/reference update used by C0/C1 only; no learned claim."""

    prediction = body_vector_prediction(state.q_NB, m_model_N_onboard)
    innovation = np.asarray(corrected_mag_B, dtype=np.float64) - prediction
    H = body_vector_jacobian(state.q_NB, m_model_N_onboard)
    R = np.eye(3, dtype=np.float64) * 1e-4
    gain = kalman_gain(state.P, H, R).K
    delta = gain @ innovation
    p_c, _ = joseph_covariance_update(state.P, gain, H, R)
    q_plus, b_plus = inject_error_state(state.q_NB, state.b_g, delta)
    p_plus, _, _ = reset_covariance(p_c, delta[:3])
    return MEKFState(q_plus, b_plus, p_plus)
