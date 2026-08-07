"""Training, deployable replay, diagnostic oracle replay, and tiny smoke CLI."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import yaml

from bench.estimators.mekf import MEKFState, body_vector_prediction, propagate_state, quat_to_dcm
from bench.metrics.mekf import attitude_geodesic_error_rad, right_local_state_error
from bench.side_gyro_mag_comp_pilot.data import (
    GeneratedDataset,
    NormalizationRecord,
    REGIMES,
    RuntimeTrajectoryBatch,
    RuntimeNormalization,
    SensorTrajectory,
    SplitFirewallRecord,
    assert_same_realization,
    freeze_train_normalization,
    generate_dataset,
    strip_runtime_trajectory,
    strip_runtime_normalization,
    validate_deployable_namespace,
    validate_firewall,
)
from bench.side_gyro_mag_comp_pilot.model import (
    FEATURE_DIM,
    EncoderOutput,
    EstimatorStep,
    SideEstimator,
    classical_vector_update,
    mekf_reset_state_digest,
)


VARIANTS = ("C0", "C1", "N0", "N1", "N2", "N3")
TRAINABLE_VARIANTS = ("N0", "N1", "N2", "N3")
REGIME_NAMES = {
    "R0": "R0_NOMINAL",
    "R1": "R1_GYRO_BIAS_SCALE",
    "R2": "R2_MAG_HARD_SOFT_IRON",
    "R3": "R3_COMBINED",
    "R4": "R4_COMBINED_OOD",
}
VARIANT_NAMES = {
    "C0": "C0_RAW_MEKF",
    "C1": "C1_ORACLE_COMP_MEKF",
    "N0": "N0_RAW_SPLIT_KNET",
    "N1": "N1_ORACLE_COMP_SPLIT_KNET",
    "N2": "N2_LEARNED_COMP_ONLY_SPLIT_KNET",
    "N3": "N3_LEARNED_COMP_FEATURE_SPLIT_KNET",
    "N3S": "N3S",
}


@dataclass(frozen=True)
class ReplayResult:
    trajectory_id: int
    variant: str
    realization_id: str
    timestamp_s: np.ndarray
    q_hat_NB: np.ndarray
    b_hat_B_rad_s: np.ndarray
    corrected_gyro_B: np.ndarray
    corrected_mag_B: np.ndarray
    gyro_feature: np.ndarray
    mag_feature: np.ndarray
    stage_order: tuple[tuple[str, ...], ...]
    initial_state_sha256: str
    recurrent_history_owner_token: str
    recurrent_history_provenance_sha256: str


@dataclass(frozen=True)
class TrainingResult:
    variant: str
    training_seed: int
    selected_epoch: int
    validation_attitude_rmse_by_epoch: tuple[float, ...]
    training_total_loss_by_epoch: tuple[float, ...]
    normalization_sha256: str
    normalization_source_ids: tuple[int, ...]
    training_ids: tuple[int, ...]
    validation_ids: tuple[int, ...]
    checkpoint_path: str
    checkpoint_sha256: str


def _initial_state() -> MEKFState:
    return MEKFState(
        q_NB=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
        b_g=np.zeros(3, dtype=np.float64),
        P=np.diag(np.r_[np.full(3, 2e-2), np.full(3, 1e-5)]).astype(np.float64),
    )


def _event_pairs(trajectory: SensorTrajectory | RuntimeTrajectoryBatch) -> list[tuple[Any, Any]]:
    events = trajectory.events if isinstance(trajectory, SensorTrajectory) else trajectory.packets
    pairs = []
    for index in range(0, len(events), 2):
        gyro, mag = events[index:index + 2]
        if gyro.sensor != "gyro" or mag.sensor != "magnetometer" or gyro.timestamp_s != mag.timestamp_s:
            raise ValueError("trajectory violates gyro-propagate-mag-update order")
        pairs.append((gyro, mag))
    return pairs


def _install_normalization(
    estimator: SideEstimator, normalization: NormalizationRecord | RuntimeNormalization,
) -> None:
    estimator.install_normalization(
        normalization.gyro_mean, normalization.gyro_std,
        normalization.mag_mean, normalization.mag_std,
    )


def deployable_replay(
    trajectory: RuntimeTrajectoryBatch,
    estimator: SideEstimator,
    normalization: RuntimeNormalization,
    m_model_N_onboard: np.ndarray,
    *,
    variant: str,
    initial_state: MEKFState | None = None,
) -> ReplayResult:
    """Truth-free causal replay for N0/N2/N3."""

    if variant not in ("N0", "N2", "N3"):
        raise ValueError("deployable replay supports only N0, N2 and N3")
    validate_deployable_namespace({
        "trajectory": trajectory,
        "normalization": normalization,
        "variant": variant,
        "m_model_N_onboard": np.asarray(m_model_N_onboard).tolist(),
    })
    _install_normalization(estimator, normalization)
    reset_state = _initial_state() if initial_state is None else initial_state
    estimator.reset_trajectory(
        reset_state, 0.0, trajectory_owner_token=trajectory.realization_sha256,
    )
    steps: list[EstimatorStep] = []
    timestamps = []
    for gyro, mag in _event_pairs(trajectory):
        steps.append(estimator.step_pair(
            gyro.measurement_S, mag.measurement_S, gyro.timestamp_s, m_model_N_onboard,
            gyro_valid=gyro.valid, mag_valid=mag.valid,
        ))
        timestamps.append(gyro.timestamp_s)
    return _assemble_replay(trajectory, variant, timestamps, steps, estimator)


def _assemble_replay(
    trajectory: RuntimeTrajectoryBatch,
    variant: str,
    timestamps: list[float],
    steps: list[EstimatorStep],
    estimator: SideEstimator,
) -> ReplayResult:
    return ReplayResult(
        trajectory_id=trajectory.trajectory_id,
        variant=variant, realization_id=trajectory.realization_sha256,
        timestamp_s=np.asarray(timestamps, dtype=np.float64),
        q_hat_NB=np.stack([step.state.q_NB for step in steps]),
        b_hat_B_rad_s=np.stack([step.state.b_g for step in steps]),
        corrected_gyro_B=np.stack([step.gyro_corrected_B for step in steps]),
        corrected_mag_B=np.stack([step.mag_corrected_B for step in steps]),
        gyro_feature=np.stack([step.gyro_feature for step in steps]),
        mag_feature=np.stack([step.mag_feature for step in steps]),
        stage_order=tuple(step.stage_order for step in steps),
        initial_state_sha256=estimator.initial_state_sha256,
        recurrent_history_owner_token=estimator.recurrent_history_owner_token,
        recurrent_history_provenance_sha256=estimator.recurrent_history_provenance_sha256(),
    )


def diagnostic_oracle_replay(
    dataset: GeneratedDataset,
    trajectory_id: int,
    variant: str,
    estimator: SideEstimator | None = None,
) -> ReplayResult:
    """Separate diagnostic namespace for C1/N1; never called by deployable replay."""

    if variant not in ("C1", "N1"):
        raise ValueError("diagnostic oracle replay supports C1 or N1")
    trajectory = dataset.sensor[trajectory_id]
    sidecar = dataset.oracle[trajectory_id]
    if variant == "N1":
        if estimator is None:
            estimator = SideEstimator("raw", feature_enabled=False)
        normalization = freeze_train_normalization(dataset)
        _install_normalization(estimator, normalization)
        estimator.reset_trajectory(
            _initial_state(), 0.0, trajectory_owner_token=trajectory.realization_id,
        )
        steps, timestamps = [], []
        for index, (gyro, _) in enumerate(_event_pairs(trajectory)):
            steps.append(estimator.step_pair(
                sidecar.gyro_target_B_rad_s[index], sidecar.mag_target_B[index],
                gyro.timestamp_s, dataset.m_model_N_onboard,
            ))
            timestamps.append(gyro.timestamp_s)
        return _assemble_replay(
            strip_runtime_trajectory(trajectory), variant, timestamps, steps, estimator,
        )
    return _classical_replay(dataset, trajectory_id, oracle_enabled=True)


def _classical_replay(dataset: GeneratedDataset, trajectory_id: int, *, oracle_enabled: bool) -> ReplayResult:
    trajectory = dataset.sensor[trajectory_id]
    sidecar = dataset.oracle[trajectory_id]
    state, current_time = _initial_state(), 0.0
    initial_state_sha256 = mekf_reset_state_digest(state, current_time)
    lineage = hashlib.sha256(b"side-gyro-mag-actual-classical-lineage-v1\0")
    lineage.update(trajectory.realization_id.encode() + b"\0")
    Q_c = np.diag(np.r_[np.full(3, 1e-8), np.full(3, 1e-12)]).astype(np.float64)
    q_hist, b_hist, gyro_hist, mag_hist, timestamps = [], [], [], [], []
    for index, (gyro, mag) in enumerate(_event_pairs(trajectory)):
        gyro_value = sidecar.gyro_target_B_rad_s[index] if oracle_enabled else gyro.measurement_S
        mag_value = sidecar.mag_target_B[index] if oracle_enabled else mag.measurement_S
        state = propagate_state(state, gyro_value, gyro.timestamp_s - current_time, Q_c).state
        current_time = gyro.timestamp_s
        state = classical_vector_update(state, mag_value, dataset.m_model_N_onboard)
        lineage.update(f"{trajectory.realization_id}/{index + 1}\n".encode())
        q_hist.append(state.q_NB); b_hist.append(state.b_g)
        gyro_hist.append(gyro_value); mag_hist.append(mag_value); timestamps.append(current_time)
    count = len(timestamps)
    return ReplayResult(
        trajectory_id, "C1" if oracle_enabled else "C0",
        trajectory.realization_id, np.asarray(timestamps), np.stack(q_hist), np.stack(b_hist),
        np.stack(gyro_hist), np.stack(mag_hist),
        np.zeros((count, FEATURE_DIM)), np.zeros((count, FEATURE_DIM)),
        tuple(("gyro_compensation", "propagation", "mag_compensation", "mag_update") for _ in range(count)),
        initial_state_sha256, trajectory.realization_id, lineage.hexdigest(),
    )


def fixed_derangement(trajectory_ids: list[int], *, regime: str, training_seed: int) -> dict[int, int]:
    """One fixed-point-free whole-sequence mapping per (regime, seed)."""

    ids = sorted(map(int, trajectory_ids))
    if len(ids) < 2 or len(set(ids)) != len(ids):
        raise ValueError("N3S stratum requires at least two unique trajectories")
    digest = hashlib.sha256(f"N3S/{regime}/{training_seed}".encode()).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "big"))
    permuted = ids.copy()
    for _ in range(1000):
        rng.shuffle(permuted)
        if all(left != right for left, right in zip(ids, permuted)):
            return dict(zip(ids, permuted))
    # Deterministic rotation is always a derangement for n>=2.
    return dict(zip(ids, ids[1:] + ids[:1]))


def n3s_replay_namespace(
    runtime_trajectories: Mapping[int, RuntimeTrajectoryBatch],
    stratum_trajectory_ids: tuple[int, ...],
    stratum_nonce: int,
    m_model_N_onboard: np.ndarray,
    trajectory_id: int,
    training_seed: int,
    n3_checkpoint: Mapping[str, torch.Tensor],
    checkpoint_file_sha256: str,
    n3_reference_state_dict_sha256: str,
    normalization: RuntimeNormalization,
    initial_state: MEKFState | None = None,
) -> tuple[ReplayResult, dict[str, Any]]:
    """Evaluate the single N3S intervention using the identical N3 checkpoint."""

    validate_deployable_namespace({
        "runtime_trajectory_ids": list(runtime_trajectories),
        "stratum_trajectory_ids": list(stratum_trajectory_ids),
        "stratum_nonce": int(stratum_nonce),
        "m_model_N_onboard": np.asarray(m_model_N_onboard).tolist(),
    })
    mapping = fixed_derangement(
        list(stratum_trajectory_ids), regime=str(stratum_nonce), training_seed=training_seed,
    )
    source_id = mapping[trajectory_id]
    target = runtime_trajectories[trajectory_id]
    source = runtime_trajectories[source_id]
    if len(_event_pairs(target)) != len(_event_pairs(source)):
        raise ValueError("N3S feature sequences must have equal length")

    source_encoder = SideEstimator("learned", feature_enabled=True)
    source_encoder.load_state_dict(n3_checkpoint, strict=True)
    _install_normalization(source_encoder, normalization)
    reset_state = _initial_state() if initial_state is None else initial_state
    source_encoder.reset_trajectory(
        reset_state, 0.0, trajectory_owner_token=source.realization_sha256,
    )
    source_features: list[tuple[torch.Tensor, torch.Tensor]] = []
    for gyro, mag in _event_pairs(source):
        g = source_encoder.compensate_gyro(gyro.measurement_S, gyro.timestamp_s, gyro.valid)
        source_encoder.propagate(g)
        m = source_encoder.compensate_magnetometer(mag.measurement_S, mag.timestamp_s, mag.valid)
        source_features.append((g.feature.detach().clone(), m.feature.detach().clone()))
        source_encoder.update(g, m, m_model_N_onboard)

    estimator = SideEstimator("learned", feature_enabled=True)
    estimator.load_state_dict(n3_checkpoint, strict=True)
    _install_normalization(estimator, normalization)
    estimator.reset_trajectory(
        reset_state, 0.0, trajectory_owner_token=target.realization_sha256,
    )
    steps, timestamps = [], []
    target_own_features: list[tuple[torch.Tensor, torch.Tensor]] = []
    for index, (gyro, mag) in enumerate(_event_pairs(target)):
        own_g = estimator.compensate_gyro(gyro.measurement_S, gyro.timestamp_s, gyro.valid)
        estimator.propagate(own_g)
        own_m = estimator.compensate_magnetometer(mag.measurement_S, mag.timestamp_s, mag.valid)
        target_own_features.append((own_g.feature.detach().clone(), own_m.feature.detach().clone()))
        shuffled_g, shuffled_m = source_features[index]
        g = EncoderOutput(own_g.corrected_B, shuffled_g)
        m = EncoderOutput(own_m.corrected_B, shuffled_m)
        steps.append(estimator.update(g, m, m_model_N_onboard))
        timestamps.append(gyro.timestamp_s)
    result = _assemble_replay(target, "N3S", timestamps, steps, estimator)
    source_gyro = np.stack([item[0].cpu().numpy() for item in source_features])
    source_mag = np.stack([item[1].cpu().numpy() for item in source_features])
    target_gyro = np.stack([item[0].cpu().numpy() for item in target_own_features])
    target_mag = np.stack([item[1].cpu().numpy() for item in target_own_features])
    evidence = {
        "stratum_nonce": int(stratum_nonce), "training_seed": training_seed,
        "mapping": {str(key): value for key, value in mapping.items()},
        "fixed_point_count": sum(key == value for key, value in mapping.items()),
        "n3_checkpoint_file_sha256": checkpoint_file_sha256,
        "n3s_checkpoint_file_sha256": checkpoint_file_sha256,
        "n3_state_dict_sha256": n3_reference_state_dict_sha256,
        "n3s_state_dict_sha256": state_dict_digest(estimator.state_dict()),
        "source_trajectory_id": source_id, "target_trajectory_id": trajectory_id,
        "expected_target_recurrent_owner_token": target.realization_sha256,
        "n3s_recurrent_owner_token": estimator.recurrent_history_owner_token,
        "n3s_recurrent_history_sha256": result.recurrent_history_provenance_sha256,
        "n3s_recurrent_transition_count": estimator.backbone.transition_count,
        "source_recurrent_owner_token": source_encoder.recurrent_history_owner_token,
        "source_recurrent_history_sha256": source_encoder.recurrent_history_provenance_sha256(),
        "source_gyro_feature": source_gyro.tolist(),
        "source_mag_feature": source_mag.tolist(),
        "target_own_gyro_feature": target_gyro.tolist(),
        "target_own_mag_feature": target_mag.tolist(),
        "n3s_applied_gyro_feature": result.gyro_feature.tolist(),
        "n3s_applied_mag_feature": result.mag_feature.tolist(),
        "intervention": "whole_feature_sequence_association_only",
    }
    return result, evidence


def _array_digest(label: str, *values: np.ndarray) -> str:
    digest = hashlib.sha256(label.encode() + b"\0")
    for value in values:
        array = np.asarray(value)
        digest.update(str(array.dtype).encode() + b"\0")
        digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def protected_replay_hashes(runtime: RuntimeTrajectoryBatch, replay: ReplayResult) -> dict[str, str]:
    return {
        "raw_packets_sha256": runtime.realization_sha256,
        "corrected_values_sha256": _array_digest(
            "corrected-values-v1", replay.corrected_gyro_B, replay.corrected_mag_B,
        ),
        "timestamps_sha256": _array_digest("timestamps-v1", replay.timestamp_s),
        "initial_state_sha256": replay.initial_state_sha256,
        "recurrent_owner_sha256": hashlib.sha256(replay.recurrent_history_owner_token.encode()).hexdigest(),
    }


def verify_n3s_bridge(
    n3_hashes: Mapping[str, str],
    n3s_hashes: Mapping[str, str],
    evidence: Mapping[str, Any],
    *,
    n3s_recurrent_owner_token: str,
    n3s_recurrent_history_sha256: str,
) -> None:
    if dict(n3_hashes) != dict(n3s_hashes):
        raise ValueError("N3S protected raw/corrected/time/init/owner hash changed")
    if evidence["n3_checkpoint_file_sha256"] != evidence["n3s_checkpoint_file_sha256"]:
        raise ValueError("N3S checkpoint file digest changed")
    if evidence["n3_state_dict_sha256"] != evidence["n3s_state_dict_sha256"]:
        raise ValueError("N3S state_dict digest changed")
    source_gyro = np.asarray(evidence["source_gyro_feature"], dtype=np.float64)
    source_mag = np.asarray(evidence["source_mag_feature"], dtype=np.float64)
    target_gyro = np.asarray(evidence["target_own_gyro_feature"], dtype=np.float64)
    target_mag = np.asarray(evidence["target_own_mag_feature"], dtype=np.float64)
    applied_gyro = np.asarray(evidence["n3s_applied_gyro_feature"], dtype=np.float64)
    applied_mag = np.asarray(evidence["n3s_applied_mag_feature"], dtype=np.float64)
    if not np.array_equal(applied_gyro, source_gyro) or not np.array_equal(applied_mag, source_mag):
        raise ValueError("N3S applied features do not equal the source trajectory features")
    if np.array_equal(source_gyro, target_gyro) or np.array_equal(source_mag, target_mag):
        raise ValueError("N3S feature association did not change from the target trajectory")
    if n3s_recurrent_owner_token != evidence["expected_target_recurrent_owner_token"]:
        raise ValueError("N3S target recurrent history was replaced by source history")
    if evidence["n3s_recurrent_owner_token"] != n3s_recurrent_owner_token:
        raise ValueError("N3S emitted recurrent owner evidence changed")
    if evidence["n3s_recurrent_history_sha256"] != n3s_recurrent_history_sha256:
        raise ValueError("N3S independently emitted recurrent lineage digest changed")
    if len(n3s_recurrent_history_sha256) != 64 or int(evidence["n3s_recurrent_transition_count"]) <= 0:
        raise ValueError("N3S recurrent lineage evidence is empty or malformed")
    if evidence["source_recurrent_owner_token"] == evidence["expected_target_recurrent_owner_token"]:
        raise ValueError("N3S source and target recurrent lineage owners must differ")
    if int(evidence["fixed_point_count"]) != 0:
        raise ValueError("N3S derangement contains a fixed point")


def state_dict_digest(state_dict: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for key in sorted(state_dict):
        value = state_dict[key].detach().cpu().contiguous()
        digest.update(key.encode()); digest.update(str(value.dtype).encode())
        digest.update(np.asarray(value.numpy()).tobytes())
    return digest.hexdigest()


def _torch_quat_multiply(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    scalar = left[0] * right[0] - torch.dot(left[1:], right[1:])
    vector = left[0] * right[1:] + right[0] * left[1:] + torch.linalg.cross(left[1:], right[1:])
    value = torch.cat((scalar.reshape(1), vector))
    return value / torch.linalg.vector_norm(value)


def _torch_quat_exp(rotation: torch.Tensor) -> torch.Tensor:
    theta = torch.linalg.vector_norm(rotation)
    half = 0.5 * theta
    scale = torch.where(theta < 1e-8, 0.5 - theta * theta / 48.0, torch.sin(half) / theta)
    return torch.cat((torch.cos(half).reshape(1), scale * rotation))


def _torch_dcm(q: torch.Tensor) -> torch.Tensor:
    q = q / torch.linalg.vector_norm(q)
    s, v = q[0], q[1:]
    cross = torch.stack((
        torch.stack((v[0] * 0, -v[2], v[1])),
        torch.stack((v[2], v[0] * 0, -v[0])),
        torch.stack((-v[1], v[0], v[0] * 0)),
    ))
    return (s * s - torch.dot(v, v)) * torch.eye(3, dtype=torch.float64) + 2 * torch.outer(v, v) + 2 * s * cross


def _torch_skew(value: torch.Tensor) -> torch.Tensor:
    zero = value[0] * 0
    return torch.stack((
        torch.stack((zero, -value[2], value[1])),
        torch.stack((value[2], zero, -value[0])),
        torch.stack((-value[1], value[0], zero)),
    ))


def _torch_attitude_angle(q_hat: torch.Tensor, q_true: torch.Tensor) -> torch.Tensor:
    dot = torch.clamp(torch.abs(torch.dot(q_hat / torch.linalg.vector_norm(q_hat), q_true)), 0.0, 1.0 - 1e-12)
    return 2.0 * torch.acos(dot)


def differentiable_trajectory_unroll(
    estimator: SideEstimator,
    dataset: GeneratedDataset,
    trajectory_id: int,
    variant: str,
    loss_weights: Mapping[str, float],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Approved causal right-local differentiable unroll used for train/validation."""

    trajectory, sidecar, truth = (
        dataset.sensor[trajectory_id], dataset.oracle[trajectory_id], dataset.truth[trajectory_id]
    )
    estimator.gyro_encoder.reset_trajectory(); estimator.mag_encoder.reset_trajectory(); estimator.backbone.reset_trajectory()
    q = torch.tensor([1., 0., 0., 0.], dtype=torch.float64)
    bias = torch.zeros(3, dtype=torch.float64)
    m_onboard = torch.tensor(np.array(dataset.m_model_N_onboard, copy=True), dtype=torch.float64)
    previous_time = 0.0
    gyro_losses: list[torch.Tensor] = []
    mag_losses: list[torch.Tensor] = []
    attitude_errors: list[torch.Tensor] = []
    bias_losses: list[torch.Tensor] = []
    for index, (gyro, mag) in enumerate(_event_pairs(trajectory)):
        if variant in ("N2", "N3"):
            gyro_output = estimator.gyro_encoder.forward_step(gyro.measurement_S, gyro.timestamp_s, gyro.valid)
            mag_output = estimator.mag_encoder.forward_step(mag.measurement_S, mag.timestamp_s, mag.valid)
            gyro_value, mag_value = gyro_output.corrected_B, mag_output.corrected_B
            gyro_feature, mag_feature = gyro_output.feature, mag_output.feature
        else:
            gyro_value = torch.tensor(np.array(
                sidecar.gyro_target_B_rad_s[index] if variant == "N1" else gyro.measurement_S, copy=True,
            ), dtype=torch.float64)
            mag_value = torch.tensor(np.array(
                sidecar.mag_target_B[index] if variant == "N1" else mag.measurement_S, copy=True,
            ), dtype=torch.float64)
            gyro_feature = torch.zeros(FEATURE_DIM, dtype=torch.float64)
            mag_feature = torch.zeros(FEATURE_DIM, dtype=torch.float64)
        gyro_target = torch.tensor(np.array(sidecar.gyro_target_B_rad_s[index], copy=True), dtype=torch.float64)
        mag_target = torch.tensor(np.array(sidecar.mag_target_B[index], copy=True), dtype=torch.float64)
        dt = gyro.timestamp_s - previous_time
        q = _torch_quat_multiply(q, _torch_quat_exp((gyro_value - bias) * dt))
        previous_time = gyro.timestamp_s
        h = _torch_dcm(q).T @ m_onboard
        H = torch.zeros((3, 6), dtype=torch.float64)
        H[:, :3] = _torch_skew(h)
        innovation = mag_value - h
        gain = estimator.backbone.forward_step(
            torch.cat((gyro_value, bias)), innovation, H, gyro_feature, mag_feature,
            feature_enabled=(variant == "N3"),
        )
        if gain.K.shape != (6, 3):
            raise RuntimeError("trajectory unroll requires a 6x3 gain")
        delta = gain.K @ innovation
        q = _torch_quat_multiply(q, _torch_quat_exp(delta[:3]))
        bias = bias + delta[3:]
        q_true = torch.tensor(np.array(truth.q_true_NB[index], copy=True), dtype=torch.float64)
        b_true = torch.tensor(np.array(truth.residual_bias_B_rad_s[index], copy=True), dtype=torch.float64)
        gyro_losses.append(torch.mean((gyro_value - gyro_target) ** 2))
        mag_losses.append(torch.mean((mag_value - mag_target) ** 2))
        attitude_errors.append(_torch_attitude_angle(q, q_true))
        bias_losses.append(torch.mean((bias - b_true) ** 2))
    attitude_rmse = torch.sqrt(torch.mean(torch.stack(attitude_errors) ** 2))
    total = (
        float(loss_weights["corrected_gyro"]) * torch.mean(torch.stack(gyro_losses))
        + float(loss_weights["corrected_magnetometer"]) * torch.mean(torch.stack(mag_losses))
        + float(loss_weights["downstream_attitude"]) * attitude_rmse ** 2
        + float(loss_weights["residual_bias"]) * torch.mean(torch.stack(bias_losses))
    )
    return total, attitude_rmse


def _trajectory_epoch(
    estimator: SideEstimator,
    dataset: GeneratedDataset,
    trajectory_ids: tuple[int, ...],
    optimizer: torch.optim.Optimizer | None,
    variant: str,
    *,
    gradient_clip_norm: float,
    batch_trajectories: int,
    loss_weights: Mapping[str, float],
) -> tuple[float, float]:
    training = optimizer is not None
    estimator.train(training)
    losses: list[float] = []
    attitude_rmse: list[float] = []
    batch_losses: list[torch.Tensor] = []
    for trajectory_id in trajectory_ids:
        per_trajectory, trajectory_attitude = differentiable_trajectory_unroll(
            estimator, dataset, trajectory_id, variant, loss_weights,
        )
        if training:
            batch_losses.append(per_trajectory)
            if len(batch_losses) == batch_trajectories:
                optimizer.zero_grad(set_to_none=True)
                torch.stack(batch_losses).mean().backward()
                torch.nn.utils.clip_grad_norm_(estimator.parameters(), gradient_clip_norm)
                optimizer.step()
                batch_losses.clear()
        losses.append(float(per_trajectory.detach()))
        attitude_rmse.append(float(trajectory_attitude.detach()))
    if training and batch_losses:
        optimizer.zero_grad(set_to_none=True)
        torch.stack(batch_losses).mean().backward()
        torch.nn.utils.clip_grad_norm_(estimator.parameters(), gradient_clip_norm)
        optimizer.step()
    if not losses or not np.all(np.isfinite(losses)):
        raise ValueError("training/validation population must be nonempty and finite")
    return float(np.mean(losses)), float(np.sqrt(np.mean(np.asarray(attitude_rmse) ** 2)))


def select_earliest_minimum_attitude_epoch(validation_attitude_rmse: list[float]) -> int:
    if not validation_attitude_rmse or not np.all(np.isfinite(validation_attitude_rmse)):
        raise ValueError("validation attitude RMSE history must be finite and nonempty")
    return int(np.argmin(np.asarray(validation_attitude_rmse))) + 1


def paired_cluster_bootstrap_ci(
    candidate: np.ndarray,
    reference: np.ndarray,
    *,
    resamples: int = 10000,
    seed: int = 45173,
) -> tuple[float, float]:
    """SC-01 cluster bootstrap: rows are trajectory IDs, columns training seeds."""

    left, right = np.asarray(candidate, dtype=np.float64), np.asarray(reference, dtype=np.float64)
    if left.shape != right.shape or left.ndim != 2 or left.shape[0] == 0 or left.shape[1] != 3:
        raise ValueError("paired gate arrays must have nonempty [trajectory_id,3 seeds] shape")
    contrast_by_id = np.mean(left - right, axis=1)
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, left.shape[0], size=(resamples, left.shape[0]))
    statistics = np.mean(contrast_by_id[sampled], axis=1)
    lower, upper = np.percentile(statistics, [2.5, 97.5])
    return float(lower), float(upper)


def evaluate_fractional_improvement_gate(
    candidate: np.ndarray,
    reference: np.ndarray,
    *,
    minimum_fractional_reduction: float,
    require_two_of_three: bool,
) -> dict[str, Any]:
    ci = paired_cluster_bootstrap_ci(candidate, reference)
    candidate_mean, reference_mean = float(np.mean(candidate)), float(np.mean(reference))
    reduction = (reference_mean - candidate_mean) / reference_mean
    seed_directions = [float(np.mean(candidate[:, seed] - reference[:, seed])) < 0 for seed in range(3)]
    passed = reduction >= minimum_fractional_reduction and ci[1] < 0
    if require_two_of_three:
        passed = passed and sum(seed_directions) >= 2
    return {"passed": bool(passed), "fractional_reduction": reduction, "ci": ci, "seed_directions": seed_directions}


def evaluate_g0_gate(n1_r3: np.ndarray, n0_r3: np.ndarray) -> dict[str, Any]:
    return evaluate_fractional_improvement_gate(
        n1_r3, n0_r3, minimum_fractional_reduction=0.10, require_two_of_three=False,
    )


def evaluate_g1_gate(
    n2_r1_gyro_rate: np.ndarray,
    n0_r1_gyro_rate: np.ndarray,
    n2_r1_increment: np.ndarray,
    n0_r1_increment: np.ndarray,
    n2_r2_mag_angle: np.ndarray,
    n0_r2_mag_angle: np.ndarray,
    n2_r3_attitude: np.ndarray,
    n0_r3_attitude: np.ndarray,
) -> dict[str, Any]:
    primary = evaluate_fractional_improvement_gate(
        n2_r3_attitude, n0_r3_attitude,
        minimum_fractional_reduction=0.05, require_two_of_three=True,
    )
    secondary = {
        "gyro_rate_strict": float(np.mean(n2_r1_gyro_rate)) < float(np.mean(n0_r1_gyro_rate)),
        "gyro_increment_strict": float(np.mean(n2_r1_increment)) < float(np.mean(n0_r1_increment)),
        "mag_angle_strict": float(np.mean(n2_r2_mag_angle)) < float(np.mean(n0_r2_mag_angle)),
    }
    return {**primary, **secondary, "passed": bool(primary["passed"] and all(secondary.values()))}


def evaluate_g2_gate(n3_r4: np.ndarray, n2_r4: np.ndarray) -> dict[str, Any]:
    return evaluate_fractional_improvement_gate(
        n3_r4, n2_r4, minimum_fractional_reduction=0.05, require_two_of_three=True,
    )


def evaluate_g3_gate(n2: np.ndarray, n3: np.ndarray, n3s: np.ndarray) -> dict[str, Any]:
    """Frozen pilot G3: shuffling removes half the gain or is null vs N2."""

    n2v, n3v, n3sv = map(lambda value: np.asarray(value, dtype=np.float64), (n2, n3, n3s))
    if n2v.shape != n3v.shape or n2v.shape != n3sv.shape:
        raise ValueError("G3 arrays must have identical paired shapes")
    feature_gain = float(np.mean(n2v) - np.mean(n3v))
    shuffled_loss = float(np.mean(n3sv) - np.mean(n3v))
    ci_n3s_minus_n2 = paired_cluster_bootstrap_ci(n3sv, n2v)
    loses_half = feature_gain > 0.0 and shuffled_loss >= 0.5 * feature_gain
    includes_zero = ci_n3s_minus_n2[0] <= 0.0 <= ci_n3s_minus_n2[1]
    return {
        "passed": bool(loses_half or includes_zero),
        "feature_gain": feature_gain,
        "shuffled_loss": shuffled_loss,
        "loses_at_least_half": bool(loses_half),
        "n3s_minus_n2_ci": ci_n3s_minus_n2,
        "ci_includes_zero": bool(includes_zero),
    }


def evaluate_g4_gate(
    n3: np.ndarray,
    n0: np.ndarray,
    n3_divergence: np.ndarray,
    n0_divergence: np.ndarray,
) -> dict[str, Any]:
    ratio = (float(np.mean(n3)) - float(np.mean(n0))) / float(np.mean(n0))
    no_added = all(
        int(np.sum(n3_divergence[:, seed])) <= int(np.sum(n0_divergence[:, seed]))
        for seed in range(3)
    )
    return {"passed": bool(ratio <= 0.03 and no_added), "penalty_ratio": ratio, "no_added_divergence": no_added}


def train_variant(
    dataset: GeneratedDataset,
    variant: str,
    normalization: NormalizationRecord,
    config: Mapping[str, Any],
    training_seed: int,
    checkpoint_path: Path,
    *,
    smoke: bool,
) -> tuple[SideEstimator, TrainingResult]:
    if variant not in TRAINABLE_VARIANTS:
        raise ValueError("only N0-N3 are trainable")
    torch.manual_seed(training_seed)
    mode = "learned" if variant in ("N2", "N3") else "raw"
    estimator = SideEstimator(mode, feature_enabled=(variant == "N3"))
    _install_normalization(estimator, normalization)
    train_ids, val_ids = tuple(dataset.split.train_ids), tuple(dataset.split.validation_ids)
    firewall = SplitFirewallRecord(
        normalization.source_trajectory_ids, train_ids, val_ids, val_ids, train_ids,
    )
    validate_firewall(dataset.split, firewall)
    training = config["training"]
    optimizer = torch.optim.Adam(
        estimator.parameters(), lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    epoch_count = int(training["smoke_epochs"] if smoke else training["max_epochs"])
    validation_history: list[float] = []
    training_history: list[float] = []
    snapshots: list[dict[str, torch.Tensor]] = []
    for _ in range(epoch_count):
        training_loss, _ = _trajectory_epoch(
            estimator, dataset, train_ids, optimizer, variant,
            gradient_clip_norm=float(training["gradient_clip_norm"]),
            batch_trajectories=int(training["batch_trajectories"]),
            loss_weights=training["loss_weights"],
        )
        _, validation_attitude = _trajectory_epoch(
            estimator, dataset, val_ids, None, variant,
            gradient_clip_norm=float(training["gradient_clip_norm"]),
            batch_trajectories=int(training["batch_trajectories"]),
            loss_weights=training["loss_weights"],
        )
        training_history.append(training_loss)
        validation_history.append(validation_attitude)
        snapshots.append(copy.deepcopy(estimator.state_dict()))
        if not smoke:
            best_index = select_earliest_minimum_attitude_epoch(validation_history) - 1
            if len(validation_history) - 1 - best_index >= int(training["early_stopping_patience"]):
                break
    selected_epoch = select_earliest_minimum_attitude_epoch(validation_history)
    estimator.load_state_dict(snapshots[selected_epoch - 1], strict=True)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "side-study-checkpoint-v1", "variant": variant,
        "training_seed": training_seed, "selected_epoch": selected_epoch,
        "normalization_sha256": normalization.sha256,
        "normalization_source_ids": list(normalization.source_trajectory_ids),
        "state_dict": estimator.state_dict(),
    }
    validate_deployable_namespace(payload)
    torch.save(payload, checkpoint_path)
    checkpoint_sha = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    result = TrainingResult(
        variant, training_seed, selected_epoch, tuple(validation_history), tuple(training_history), normalization.sha256,
        normalization.source_trajectory_ids, train_ids, val_ids,
        str(checkpoint_path), checkpoint_sha,
    )
    return estimator, result


def _metric(dataset: GeneratedDataset, replay: ReplayResult) -> float:
    truth = dataset.truth[replay.trajectory_id]
    values = attitude_geodesic_error_rad(replay.q_hat_NB.astype(np.float64), truth.q_true_NB.astype(np.float64))
    return float(np.sqrt(np.mean(values * values)))


def weak_observable_metrics(
    replay: ReplayResult,
    truth_q_NB: np.ndarray,
    truth_bias_B: np.ndarray,
    m_true_N: np.ndarray,
    valid_magnetometer: np.ndarray,
) -> dict[str, float | int]:
    """Exact whole-window weak-axis/observable-plane diagnostic from the pilot spec."""

    errors = right_local_state_error(
        replay.q_hat_NB.astype(np.float64), replay.b_hat_B_rad_s.astype(np.float64),
        np.asarray(truth_q_NB, dtype=np.float64), np.asarray(truth_bias_B, dtype=np.float64),
    ).delta_theta_rad
    mask = np.asarray(valid_magnetometer, dtype=np.bool_)
    if mask.shape != (errors.shape[0],) or not np.any(mask):
        raise ValueError("trajectory has zero valid magnetometer-update samples")
    directions = np.stack([
        quat_to_dcm(q).T @ np.asarray(m_true_N, dtype=np.float64) for q in truth_q_NB
    ])
    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    if not np.all(np.isfinite(norms)) or np.any(norms <= 0):
        raise ValueError("true body-frame magnetic direction is invalid")
    directions = directions / norms
    selected_errors, selected_directions = errors[mask], directions[mask]
    parallel_signed = np.sum(selected_errors * selected_directions, axis=1)
    observable = selected_errors - parallel_signed[:, None] * selected_directions
    weak = float(np.sqrt(np.mean(parallel_signed ** 2)))
    plane = float(np.sqrt(np.mean(np.sum(observable ** 2, axis=1))))
    if not np.isfinite(weak) or not np.isfinite(plane):
        raise ValueError("weak/observable diagnostics must be finite")
    return {
        "valid_magnetometer_sample_count": int(np.sum(mask)),
        "weak_axis_rmse_rad": weak,
        "observable_plane_rmse_rad": plane,
    }


def load_config(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text())
    expected = {
        ("data", "namespace"): "side-gyro-mag-comp-pilot-data-v1",
        ("data", "generation_seed"): 271828,
        ("data", "split_seed"): 314159,
        ("data", "sequence_length"): 16,
        ("data", "smoke_population_per_regime"): {"train": 4, "validation": 2, "test": 4},
        ("data", "method_lock_population_per_regime"): {"train": 40, "validation": 10, "test": 30},
        ("data", "training_regimes"): ["R0", "R1", "R2", "R3"],
        ("data", "test_regimes"): ["R0", "R1", "R2", "R3", "R4"],
        ("data", "normalization_source"): "R0-R3_training_split_only",
        ("model", "dtype"): "float64",
        ("model", "smoke_device"): "cpu",
        ("model", "pilot_device"): "cuda_if_available",
        ("model", "sensor_input_dim"): 5,
        ("model", "encoder_hidden_dim"): 16,
        ("model", "feature_dim_per_sensor"): 8,
        ("model", "split_prior_hidden_dim"): 32,
        ("model", "split_measurement_hidden_dim"): 32,
        ("model", "g1_head_dim"): 36,
        ("model", "g2_head_dim"): 9,
        ("model", "film_g1_head_dim"): 72,
        ("model", "film_g2_head_dim"): 18,
        ("model", "conditioning"): "branch_specific_film",
        ("model", "gain_shape"): [6, 3],
        ("training", "seeds"): [31001, 31002, 31003],
        ("training", "optimizer"): "Adam",
        ("training", "learning_rate"): 0.001,
        ("training", "weight_decay"): 0.0,
        ("training", "batch_trajectories"): 8,
        ("training", "max_epochs"): 20,
        ("training", "smoke_epochs"): 2,
        ("training", "early_stopping_patience"): 4,
        ("training", "checkpoint_rule"): "earliest_epoch_attaining_minimum_R0_R3_validation_attitude_rmse",
        ("training", "gradient_clip_norm"): 1.0,
        ("training", "loss_weights"): {
            "corrected_gyro": 1.0, "corrected_magnetometer": 1.0,
            "downstream_attitude": 1.0, "residual_bias": 0.25,
        },
        ("evaluation", "bootstrap_resamples"): 10000,
        ("evaluation", "bootstrap_seed"): 45173,
        ("evaluation", "divergence_threshold_rad"): 1.0,
        ("evaluation", "n3s"): "fixed_whole_sequence_derangement_per_regime_and_training_seed",
        ("gates", "canonical_precedence"): "experiments/side_gyro_mag_comp_pilot/PILOT_SPEC.md",
        ("gates", "contrast_order"): "candidate_minus_reference",
        ("gates", "G0"): {
            "regime": "R3", "candidate": "N1", "reference": "N0",
            "minimum_fractional_reduction": 0.10, "ci_upper_strictly_below": 0.0,
        },
        ("gates", "G1"): {
            "regime": "R3", "candidate": "N2", "reference": "N0",
            "minimum_fractional_reduction": 0.05, "ci_upper_strictly_below": 0.0,
            "seed_direction_count": 2, "gyro_regime": "R1", "mag_regime": "R2",
        },
        ("gates", "G2"): {
            "regime": "R4", "candidate": "N3", "reference": "N2",
            "minimum_fractional_reduction": 0.05, "ci_upper_strictly_below": 0.0,
            "seed_direction_count": 2,
        },
        ("gates", "G3"): {
            "regime": "R4",
            "pass": "loses_half_of_N3_feature_gain_or_N3S_vs_N2_CI_includes_zero",
        },
        ("gates", "G4"): {
            "regime": "R0", "candidate": "N3", "reference": "N0",
            "maximum_fractional_penalty": 0.03, "added_divergence_allowed": 0,
        },
    }
    for path_parts, frozen in expected.items():
        actual = value[path_parts[0]][path_parts[1]]
        if actual != frozen:
            raise ValueError(f"config violates frozen field {'.'.join(path_parts)}")
    return value


def run_tiny_smoke(config_path: Path, output_dir: Path) -> dict[str, Any]:
    config = load_config(config_path)
    population = config["data"]["smoke_population_per_regime"]
    dataset = generate_dataset(
        population=population, sequence_length=int(config["data"]["sequence_length"]),
        dt_s=float(config["data"]["dt_s"]), generation_seed=int(config["data"]["generation_seed"]),
        split_seed=int(config["data"]["split_seed"]),
    )
    normalization = freeze_train_normalization(dataset)
    runtime_normalization = strip_runtime_normalization(normalization)
    output_dir.mkdir(parents=True, exist_ok=True)
    training_records: dict[str, Any] = {}
    estimators: dict[str, SideEstimator] = {}
    runtime_by_id = {item: strip_runtime_trajectory(trajectory) for item, trajectory in dataset.sensor.items()}
    for variant in TRAINABLE_VARIANTS:
        estimator, result = train_variant(
            dataset, variant, normalization, config, int(config["training"]["seeds"][0]),
            output_dir / "checkpoints" / f"{variant}.pt", smoke=True,
        )
        estimators[variant] = estimator
        training_records[variant] = asdict(result)

    results: list[dict[str, Any]] = []
    n3_state = copy.deepcopy(estimators["N3"].state_dict())
    n3_digest = state_dict_digest(n3_state)
    n3_checkpoint_file_digest = training_records["N3"]["checkpoint_sha256"]
    n3s_evidence: list[dict[str, Any]] = []
    for regime in REGIMES:
        test_ids = [item for item in dataset.split.test_ids if dataset.sensor[item].regime == regime]
        if not test_ids:
            raise ValueError(f"empty smoke population for {regime}")
        for trajectory_id in test_ids:
            trajectory = dataset.sensor[trajectory_id]
            runtime = runtime_by_id[trajectory_id]
            replays = {
                "C0": _classical_replay(dataset, trajectory_id, oracle_enabled=False),
                "C1": diagnostic_oracle_replay(dataset, trajectory_id, "C1"),
            }
            replay_estimators: dict[str, SideEstimator] = {}
            for variant in ("N0", "N2", "N3"):
                estimator = SideEstimator(
                    "learned" if variant in ("N2", "N3") else "raw",
                    feature_enabled=(variant == "N3"),
                )
                estimator.load_state_dict(estimators[variant].state_dict(), strict=True)
                replays[variant] = deployable_replay(
                    runtime, estimator, runtime_normalization, dataset.m_model_N_onboard, variant=variant,
                )
                replay_estimators[variant] = estimator
            n1_estimator = SideEstimator("raw", feature_enabled=False)
            n1_estimator.load_state_dict(estimators["N1"].state_dict(), strict=True)
            replays["N1"] = diagnostic_oracle_replay(dataset, trajectory_id, "N1", n1_estimator)
            n3s, evidence = n3s_replay_namespace(
                runtime_by_id, tuple(test_ids), REGIMES.index(regime), dataset.m_model_N_onboard,
                trajectory_id, int(config["training"]["seeds"][0]), n3_state,
                n3_checkpoint_file_digest,
                state_dict_digest(replay_estimators["N3"].state_dict()), runtime_normalization,
            )
            n3_protected = protected_replay_hashes(runtime, replays["N3"])
            n3s_protected = protected_replay_hashes(runtime, n3s)
            evidence["n3_protected_hashes"] = n3_protected
            evidence["n3s_protected_hashes"] = n3s_protected
            evidence["n3_recurrent_history_sha256"] = (
                replays["N3"].recurrent_history_provenance_sha256
            )
            verify_n3s_bridge(
                n3_protected, n3s_protected, evidence,
                n3s_recurrent_owner_token=n3s.recurrent_history_owner_token,
                n3s_recurrent_history_sha256=n3s.recurrent_history_provenance_sha256,
            )
            if evidence["n3_state_dict_sha256"] != n3_digest:
                raise RuntimeError("N3S checkpoint/derangement invariant failed")
            replays["N3S"] = n3s; n3s_evidence.append(evidence)
            assert_same_realization([trajectory for _ in replays])
            for variant, replay in replays.items():
                if replay.realization_id != runtime.realization_sha256:
                    raise RuntimeError("variant raw realization provenance drift")
                weak_metrics = weak_observable_metrics(
                    replay, dataset.truth[trajectory_id].q_true_NB,
                    dataset.truth[trajectory_id].residual_bias_B_rad_s,
                    dataset.truth[trajectory_id].m_true_N,
                    np.asarray([mag.valid for _, mag in _event_pairs(trajectory)], dtype=np.bool_),
                )
                results.append({
                    "experiment": "side-gyro-mag-comp-pilot-tiny-smoke",
                    "regime": REGIME_NAMES[regime], "split": "test", "model": VARIANT_NAMES[variant],
                    "window": "whole_trajectory", "metric": "attitude_geodesic_rmse_rad",
                    "seed": int(config["training"]["seeds"][0]), "trajectory_id": trajectory_id,
                    "realization_sha256": replay.realization_id, "value": _metric(dataset, replay),
                    "finite": bool(np.isfinite(replay.q_hat_NB).all()),
                    "population_count": 1,
                    "weak_observable": weak_metrics,
                })
    summary = {
        "schema_version": "side-gyro-mag-comp-pilot-smoke-v1", "status": "PASS_SMOKE",
        "performance_claim": False, "covariance_claim_valid": False,
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "normalization_sha256": normalization.sha256,
        "normalization_source_ids": list(normalization.source_trajectory_ids),
        "split": {
            "train_ids": list(dataset.split.train_ids), "validation_ids": list(dataset.split.validation_ids),
            "test_ids": list(dataset.split.test_ids), "data_generation_seed": dataset.split.data_generation_seed,
        },
        "training": training_records, "results": results, "n3s": n3s_evidence,
        "populations": {
            regime: sum(1 for item in dataset.split.test_ids if dataset.sensor[item].regime == regime)
            for regime in REGIMES
        },
        "variants": list(VARIANTS) + ["N3S"],
    }
    (output_dir / "SMOKE_RESULT.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tiny-smoke", action="store_true", required=True)
    args = parser.parse_args()
    result = run_tiny_smoke(args.config, args.output_dir)
    print(json.dumps({"status": result["status"], "variants": result["variants"], "populations": result["populations"]}, sort_keys=True))


if __name__ == "__main__":
    main()
