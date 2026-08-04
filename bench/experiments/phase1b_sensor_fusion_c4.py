"""Phase 1B Step 2 classical sensor-fusion and C4 experiment driver."""

from __future__ import annotations

import argparse
import copy
import hashlib
import inspect
import json
import math
import platform
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import yaml

from bench.estimators.mekf import (
    MEKFState,
    body_vector_jacobian,
    body_vector_prediction,
    inject_error_state,
    joseph_covariance_update,
    kalman_gain,
    propagate_state,
    reset_covariance,
    star_tracker_update,
    sun_tangent_jacobian,
)
from bench.experiments.phase1b_unit_st_classical import (
    default_initial_state,
    paired_bootstrap_ci,
    stable_statistics_seed,
)
from bench.metrics.mekf import (
    attitude_geodesic_error_rad,
    bias_error_summary,
    consistency_summary,
    right_local_nees,
    right_local_state_error,
    spd_diagnostics,
    star_tracker_nis,
)
from bench.metrics.mekf_fusion import magnetometer_nis, sun_sensor_nis
from bench.tasks.generator.mekf_events import replay_trajectory
from bench.tasks.generator.mekf_fusion_events import (
    GENERATOR_ID,
    FusionEventTable,
    FusionOracleSidecar,
    FusionSensorCode,
    load_fusion_dataset,
    load_fusion_oracle,
    save_fusion_dataset,
    save_fusion_oracle,
    select_fusion_sensors,
)
from bench.tasks.generator.phase1b_sensor_fusion import (
    FusionScenarioCode,
    GeneratedSensorFusion,
    SensorFusionConfig,
    fusion_gyro_st_as_phase1a,
    generate_sensor_fusion,
)


EXPERIMENT_VERSION = "p1b-sensor-fusion-c4-v1"
POLICY_VERSION = "p1b-classical-fusion-policy-v1"


@dataclass(frozen=True)
class FusionPolicy:
    policy_id: str
    qg_scale: float = 1.0
    qb_scale: float = 1.0
    r_st_scale: float = 1.0
    r_mag_scale: float = 1.0
    r_sun_scale: float = 1.0
    oracle_mode: str = "none"

    def __post_init__(self) -> None:
        if not isinstance(self.policy_id, str) or not self.policy_id:
            raise ValueError("policy_id must be nonempty")
        for name in ("qg_scale", "qb_scale", "r_st_scale", "r_mag_scale", "r_sun_scale"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
            object.__setattr__(self, name, value)
        if self.oracle_mode not in {
            "none",
            "process",
            "measurement",
            "full",
            "wrong-process",
            "wrong-measurement",
        }:
            raise ValueError("unknown oracle mode")

    def deployable_artifact(self) -> dict[str, Any]:
        if self.oracle_mode != "none":
            raise ValueError("oracle policies are simulation-only")
        return {
            "policy_version": POLICY_VERSION,
            "policy_id": self.policy_id,
            "qg_scale": self.qg_scale,
            "qb_scale": self.qb_scale,
            "r_st_scale": self.r_st_scale,
            "r_mag_scale": self.r_mag_scale,
            "r_sun_scale": self.r_sun_scale,
        }


F_BASE = FusionPolicy("F-BASE")
F_TUNED = FusionPolicy(
    "F-TUNED", qg_scale=0.125, qb_scale=0.125, r_st_scale=8.0
)
ORACLE_PROCESS = FusionPolicy("ORACLE-PROCESS", oracle_mode="process")
ORACLE_MEASUREMENT = FusionPolicy("ORACLE-MEASUREMENT", oracle_mode="measurement")
ORACLE_FULL = FusionPolicy("ORACLE-FULL", oracle_mode="full")
WRONG_PROCESS = FusionPolicy("WRONG-PROCESS", oracle_mode="wrong-process")
WRONG_MEASUREMENT = FusionPolicy(
    "WRONG-MEASUREMENT", oracle_mode="wrong-measurement"
)


@dataclass(frozen=True)
class FusionReplayResult:
    trajectory_id: int
    processed_event_count: int
    event_time_s: np.ndarray
    event_order: np.ndarray
    sensor_code: np.ndarray
    q_NB_history: np.ndarray
    b_g_history: np.ndarray
    P_history: np.ndarray
    star_tracker_event_order: np.ndarray
    star_tracker_residual: np.ndarray
    star_tracker_S: np.ndarray
    magnetometer_event_order: np.ndarray
    magnetometer_residual: np.ndarray
    magnetometer_S: np.ndarray
    sun_event_order: np.ndarray
    sun_residual: np.ndarray
    sun_S: np.ndarray
    sun_skipped_event_order: np.ndarray
    final_state: MEKFState

    def __post_init__(self) -> None:
        specs = {
            "event_time_s": (np.float64, 1),
            "event_order": (np.int64, 1),
            "sensor_code": (np.int16, 1),
            "q_NB_history": (np.float64, 2),
            "b_g_history": (np.float64, 2),
            "P_history": (np.float64, 3),
            "star_tracker_event_order": (np.int64, 1),
            "star_tracker_residual": (np.float64, 2),
            "star_tracker_S": (np.float64, 3),
            "magnetometer_event_order": (np.int64, 1),
            "magnetometer_residual": (np.float64, 2),
            "magnetometer_S": (np.float64, 3),
            "sun_event_order": (np.int64, 1),
            "sun_residual": (np.float64, 2),
            "sun_S": (np.float64, 3),
            "sun_skipped_event_order": (np.int64, 1),
        }
        for name, (dtype, ndim) in specs.items():
            value = getattr(self, name)
            if not isinstance(value, np.ndarray) or value.dtype != np.dtype(dtype):
                raise TypeError(f"{name} must have exact dtype {np.dtype(dtype)}")
            if value.ndim != ndim or not np.all(np.isfinite(value)):
                raise ValueError(f"{name} has invalid rank or nonfinite values")
            result = np.array(value, dtype=dtype, order="C", copy=True)
            result.setflags(write=False)
            object.__setattr__(self, name, result)
        count = int(self.processed_event_count)
        if count <= 0 or any(
            getattr(self, name).shape[0] != count
            for name in (
                "event_time_s",
                "event_order",
                "sensor_code",
                "q_NB_history",
                "b_g_history",
                "P_history",
            )
        ):
            raise ValueError("processed event histories do not align")
        if self.q_NB_history.shape[1:] != (4,) or self.b_g_history.shape[1:] != (3,):
            raise ValueError("state histories have invalid shape")
        if self.P_history.shape[1:] != (6, 6):
            raise ValueError("P history must have shape [E,6,6]")
        for prefix, dimension in (("star_tracker", 3), ("magnetometer", 3), ("sun", 2)):
            orders = getattr(self, f"{prefix}_event_order")
            residual = getattr(self, f"{prefix}_residual")
            covariance = getattr(self, f"{prefix}_S")
            if residual.shape != (orders.size, dimension) or covariance.shape != (
                orders.size,
                dimension,
                dimension,
            ):
                raise ValueError(f"{prefix} compact evidence does not align")
        if not isinstance(self.final_state, MEKFState):
            raise TypeError("final_state must be MEKFState")


def base_process_covariance(config: SensorFusionConfig) -> np.ndarray:
    return np.diag(
        np.asarray(
            [config.base_Q_g_rad2_s] * 3 + [config.bias_psd_rad2_s3] * 3,
            dtype=np.float64,
        )
    )


def _scaled_process(Q_c: np.ndarray, qg_scale: float, qb_scale: float) -> np.ndarray:
    if not isinstance(Q_c, np.ndarray) or Q_c.dtype != np.dtype(np.float64) or Q_c.shape != (6, 6):
        raise TypeError("Q_c must be a float64 [6,6] array")
    value = np.array(Q_c, copy=True)
    value[:3, :3] *= float(qg_scale)
    value[3:, 3:] *= float(qb_scale)
    return value


def _vector_update(
    state: MEKFState,
    residual: np.ndarray,
    H: np.ndarray,
    R: np.ndarray,
) -> tuple[MEKFState, np.ndarray]:
    gain = kalman_gain(state.P, H, R)
    delta_x = gain.K @ residual
    P_c, _ = joseph_covariance_update(state.P, gain.K, H, R)
    q_plus, b_plus = inject_error_state(state.q_NB, state.b_g, delta_x)
    P_plus, _, _ = reset_covariance(P_c, delta_x[:3])
    return MEKFState(q_NB=q_plus, b_g=b_plus, P=P_plus), gain.S


def _oracle_scales(policy: FusionPolicy, alpha_b: float, alpha_r: float) -> tuple[float, float]:
    if policy.oracle_mode == "process":
        return alpha_b, 1.0
    if policy.oracle_mode == "measurement":
        return 1.0, alpha_r
    if policy.oracle_mode == "full":
        return alpha_b, alpha_r
    if policy.oracle_mode == "wrong-process":
        return alpha_r, 1.0
    if policy.oracle_mode == "wrong-measurement":
        return 1.0, alpha_b
    return 1.0, 1.0


def _replay(
    table: FusionEventTable,
    trajectory_id: int,
    initial_state: MEKFState,
    initial_time_s: float,
    Q_c: np.ndarray,
    policy: FusionPolicy,
    oracle: FusionOracleSidecar | None,
) -> FusionReplayResult:
    if not isinstance(table, FusionEventTable):
        raise TypeError("table must be FusionEventTable")
    if not isinstance(initial_state, MEKFState) or not isinstance(policy, FusionPolicy):
        raise TypeError("initial state/policy type mismatch")
    if (policy.oracle_mode == "none") != (oracle is None):
        raise ValueError("fixed replay has no oracle; oracle replay requires a sidecar")
    current_time = float(initial_time_s)
    if not np.isfinite(current_time) or current_time < 0.0:
        raise ValueError("initial time must be finite and nonnegative")
    rows = np.flatnonzero(table.trajectory_id == np.int64(trajectory_id))
    if rows.size == 0:
        raise ValueError("trajectory is absent from event table")
    cursor = oracle.cursor(trajectory_id) if oracle is not None else None
    state = initial_state
    times: list[float] = []
    orders: list[int] = []
    sensors: list[int] = []
    quaternions: list[np.ndarray] = []
    biases: list[np.ndarray] = []
    covariances: list[np.ndarray] = []
    st_orders: list[int] = []
    st_residual: list[np.ndarray] = []
    st_S: list[np.ndarray] = []
    mag_orders: list[int] = []
    mag_residual: list[np.ndarray] = []
    mag_S: list[np.ndarray] = []
    sun_orders: list[int] = []
    sun_residual: list[np.ndarray] = []
    sun_S: list[np.ndarray] = []
    sun_skips: list[int] = []

    for row in rows:
        order = int(table.event_order[row])
        code = int(table.sensor_code[row])
        event_time = float(table.measurement_time_s[row])
        alpha_b, alpha_r = cursor.consume(order) if cursor is not None else (1.0, 1.0)
        process_oracle, measurement_oracle = _oracle_scales(policy, alpha_b, alpha_r)
        if code == int(FusionSensorCode.GYRO):
            if not event_time > current_time:
                raise ValueError("gyro must advance filter time strictly")
            payload = int(table.payload_index[row])
            result = propagate_state(
                state,
                table.gyro_omega_m_B_rad_s[payload],
                event_time - current_time,
                _scaled_process(
                    Q_c,
                    policy.qg_scale,
                    policy.qb_scale * process_oracle,
                ),
            )
            state = result.state
            current_time = event_time
        elif event_time != current_time:
            raise ValueError("updates must occur at the current propagated time")
        elif code == int(FusionSensorCode.MAGNETOMETER):
            payload = int(table.payload_index[row])
            prediction = body_vector_prediction(
                state.q_NB, table.magnetometer_r_mag_N_model[payload]
            )
            residual = table.magnetometer_z_mag_B[payload] - prediction
            H = body_vector_jacobian(
                state.q_NB, table.magnetometer_r_mag_N_model[payload]
            )
            R = table.magnetometer_R_mag[payload] * (
                policy.r_mag_scale * measurement_oracle
            )
            state, S = _vector_update(state, residual, H, R)
            mag_orders.append(order)
            mag_residual.append(residual)
            mag_S.append(S)
        elif code == int(FusionSensorCode.SUN_SENSOR):
            if not bool(table.valid[row]):
                sun_skips.append(order)
            else:
                payload = int(table.payload_index[row])
                prediction, basis, H = sun_tangent_jacobian(
                    state.q_NB, table.sun_r_sun_N_model[payload]
                )
                residual = basis.T @ (table.sun_z_sun_B[payload] - prediction)
                R = table.sun_R_sun_tangent_rad2[payload] * policy.r_sun_scale
                state, S = _vector_update(state, residual, H, R)
                sun_orders.append(order)
                sun_residual.append(residual)
                sun_S.append(S)
        elif code == int(FusionSensorCode.STAR_TRACKER):
            payload = int(table.payload_index[row])
            result = star_tracker_update(
                state,
                table.star_tracker_q_ST_NB[payload],
                table.star_tracker_R_ST_rad2[payload] * policy.r_st_scale,
            )
            state = result.state
            st_orders.append(order)
            st_residual.append(result.residual)
            st_S.append(result.S)
        else:
            raise ValueError("unknown sensor code")
        times.append(current_time)
        orders.append(order)
        sensors.append(code)
        quaternions.append(state.q_NB)
        biases.append(state.b_g)
        covariances.append(state.P)

    count = len(times)
    return FusionReplayResult(
        trajectory_id=int(trajectory_id),
        processed_event_count=count,
        event_time_s=np.asarray(times, dtype=np.float64),
        event_order=np.asarray(orders, dtype=np.int64),
        sensor_code=np.asarray(sensors, dtype=np.int16),
        q_NB_history=np.asarray(quaternions, dtype=np.float64).reshape(count, 4),
        b_g_history=np.asarray(biases, dtype=np.float64).reshape(count, 3),
        P_history=np.asarray(covariances, dtype=np.float64).reshape(count, 6, 6),
        star_tracker_event_order=np.asarray(st_orders, dtype=np.int64),
        star_tracker_residual=np.asarray(st_residual, dtype=np.float64).reshape(-1, 3),
        star_tracker_S=np.asarray(st_S, dtype=np.float64).reshape(-1, 3, 3),
        magnetometer_event_order=np.asarray(mag_orders, dtype=np.int64),
        magnetometer_residual=np.asarray(mag_residual, dtype=np.float64).reshape(-1, 3),
        magnetometer_S=np.asarray(mag_S, dtype=np.float64).reshape(-1, 3, 3),
        sun_event_order=np.asarray(sun_orders, dtype=np.int64),
        sun_residual=np.asarray(sun_residual, dtype=np.float64).reshape(-1, 2),
        sun_S=np.asarray(sun_S, dtype=np.float64).reshape(-1, 2, 2),
        sun_skipped_event_order=np.asarray(sun_skips, dtype=np.int64),
        final_state=state,
    )


def replay_fixed_policy(
    table: FusionEventTable,
    trajectory_id: int,
    initial_state: MEKFState,
    initial_time_s: float,
    Q_c: np.ndarray,
    policy: FusionPolicy = F_BASE,
) -> FusionReplayResult:
    """Truth-free deployable replay boundary; no oracle parameter exists."""

    if policy.oracle_mode != "none":
        raise ValueError("replay_fixed_policy accepts fixed policies only")
    return _replay(table, trajectory_id, initial_state, initial_time_s, Q_c, policy, None)


def replay_oracle_policy(
    table: FusionEventTable,
    oracle_context: FusionOracleSidecar,
    trajectory_id: int,
    initial_state: MEKFState,
    initial_time_s: float,
    Q_c: np.ndarray,
    policy: FusionPolicy,
) -> FusionReplayResult:
    """Simulation-only replay using a current-event, forward-only sidecar."""

    if policy.oracle_mode == "none":
        raise ValueError("oracle replay requires an oracle policy")
    return _replay(
        table, trajectory_id, initial_state, initial_time_s, Q_c, policy, oracle_context
    )


def assert_all_one_oracle_exact(
    table: FusionEventTable,
    oracle_context: FusionOracleSidecar,
    trajectory_id: int,
    initial_state: MEKFState,
    Q_c: np.ndarray,
) -> None:
    rows = np.flatnonzero(oracle_context.trajectory_id == np.int64(trajectory_id))
    if not np.array_equal(oracle_context.alpha_b[rows], np.ones(rows.size)) or not np.array_equal(
        oracle_context.alpha_R_mag[rows], np.ones(rows.size)
    ):
        raise ValueError("all-one equivalence requires an all-one sidecar")
    fixed = replay_fixed_policy(table, trajectory_id, initial_state, 0.0, Q_c, F_BASE)
    oracle = replay_oracle_policy(
        table, oracle_context, trajectory_id, initial_state, 0.0, Q_c, ORACLE_FULL
    )
    for name in (
        "event_time_s",
        "event_order",
        "sensor_code",
        "q_NB_history",
        "b_g_history",
        "P_history",
        "star_tracker_residual",
        "star_tracker_S",
        "magnetometer_residual",
        "magnetometer_S",
        "sun_residual",
        "sun_S",
    ):
        if not np.array_equal(getattr(fixed, name), getattr(oracle, name)):
            raise AssertionError(f"all-one fixed/oracle mismatch at {name}")


def _truth_join(dataset: Any, replay: FusionReplayResult) -> dict[str, np.ndarray]:
    truth = dataset.truth
    matches = np.flatnonzero(truth.trajectory_id == np.int64(replay.trajectory_id))
    if matches.size != 1:
        raise ValueError("trajectory truth join is not unique")
    index = int(matches[0])
    start, stop = int(truth.truth_offsets[index]), int(truth.truth_offsets[index + 1])
    times = truth.truth_time_s[start:stop]
    lookup = {float(value): row for row, value in enumerate(times)}
    rows: list[int] = []
    for value in replay.event_time_s:
        row = lookup.get(float(value))
        if row is None or times[row] != value:
            raise ValueError("truth join requires exact timestamp equality")
        rows.append(start + row)
    selected = np.asarray(rows, dtype=np.int64)
    return {
        "q": truth.q_true_NB[selected],
        "b": truth.gyro_bias_true_rad_s[selected],
        "mag": truth.r_mag_N_true[selected],
        "sun": truth.r_sun_N_true[selected],
    }


def _window_metrics(values: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    selected = values[mask]
    if selected.size == 0:
        raise ValueError("metric window must be nonempty")
    return {
        "rmse": float(np.sqrt(np.mean(selected * selected))),
        "p95": float(np.quantile(selected, 0.95)),
        "peak": float(np.max(selected)),
    }


def evaluate_fusion_replay(
    dataset: Any,
    replay: FusionReplayResult,
    *,
    scenario_id: str,
    policy_id: str,
    duration_s: float,
    divergence_threshold_rad: float,
    confidence_level: float = 0.95,
) -> dict[str, Any]:
    joined = _truth_join(dataset, replay)
    attitude = attitude_geodesic_error_rad(replay.q_NB_history, joined["q"])
    bias = bias_error_summary(replay.b_g_history, joined["b"])
    nees = right_local_nees(
        replay.q_NB_history,
        replay.b_g_history,
        replay.P_history,
        joined["q"],
        joined["b"],
    )
    nees_summary = consistency_summary(nees, dof_per_sample=6, confidence_level=confidence_level)
    p_diag = spd_diagnostics(replay.P_history, name="posterior_P")
    whole = np.ones(replay.event_time_s.size, dtype=np.bool_)
    slow = (replay.event_time_s >= 0.2 * duration_s) & (
        replay.event_time_s < 0.8 * duration_s
    )
    fast = (replay.event_time_s >= 0.45 * duration_s) & (
        replay.event_time_s < 0.6 * duration_s
    )
    post = replay.event_time_s >= 0.8 * duration_s
    attitude_whole = _window_metrics(attitude, whole)
    attitude_fast = _window_metrics(attitude, fast)
    bias_slow = _window_metrics(bias.vector_norm_rad_s, slow)
    bias_post = _window_metrics(bias.vector_norm_rad_s, post)

    result: dict[str, Any] = {
        "scenario_id": str(scenario_id),
        "policy_id": str(policy_id),
        "trajectory_id": int(replay.trajectory_id),
        "event_count": int(replay.processed_event_count),
        "attitude_rmse_rad": attitude_whole["rmse"],
        "attitude_p95_rad": attitude_whole["p95"],
        "attitude_peak_rad": attitude_whole["peak"],
        "fast_attitude_rmse_rad": attitude_fast["rmse"],
        "fast_attitude_p95_rad": attitude_fast["p95"],
        "fast_attitude_peak_rad": attitude_fast["peak"],
        "bias_vector_rmse_rad_s": bias.vector_rmse_rad_s,
        "slow_bias_rmse_rad_s": bias_slow["rmse"],
        "slow_bias_p95_rad_s": bias_slow["p95"],
        "post_bias_rmse_rad_s": bias_post["rmse"],
        "nees_count": nees_summary.count,
        "nees_normalized_mean": nees_summary.normalized_mean,
        "minimum_P_eigenvalue": float(np.min(p_diag.minimum_eigenvalue)),
        "diverged": bool(
            not np.all(np.isfinite(attitude))
            or np.max(attitude) > float(divergence_threshold_rad)
        ),
        "sun_update_count": int(replay.sun_event_order.size),
        "sun_skip_count": int(replay.sun_skipped_event_order.size),
    }
    if replay.star_tracker_residual.shape[0]:
        values = star_tracker_nis(replay.star_tracker_residual, replay.star_tracker_S)
        summary = consistency_summary(values, dof_per_sample=3, confidence_level=confidence_level)
        result.update(
            st_nis_count=summary.count,
            st_nis_normalized_mean=summary.normalized_mean,
            minimum_ST_S_eigenvalue=float(
                np.min(spd_diagnostics(replay.star_tracker_S, name="ST_S").minimum_eigenvalue)
            ),
        )
    else:
        result.update(st_nis_count=0, st_nis_normalized_mean=None, minimum_ST_S_eigenvalue=None)
    if replay.magnetometer_residual.shape[0]:
        values = magnetometer_nis(replay.magnetometer_residual, replay.magnetometer_S)
        summary = consistency_summary(values, dof_per_sample=3, confidence_level=confidence_level)
        result.update(
            mag_nis_count=summary.count,
            mag_nis_normalized_mean=summary.normalized_mean,
            minimum_mag_S_eigenvalue=float(
                np.min(spd_diagnostics(replay.magnetometer_S, name="mag_S").minimum_eigenvalue)
            ),
        )
    else:
        result.update(mag_nis_count=0, mag_nis_normalized_mean=None, minimum_mag_S_eigenvalue=None)
    if replay.sun_residual.shape[0]:
        values = sun_sensor_nis(replay.sun_residual, replay.sun_S)
        summary = consistency_summary(values, dof_per_sample=2, confidence_level=confidence_level)
        result.update(
            sun_nis_count=summary.count,
            sun_nis_normalized_mean=summary.normalized_mean,
            minimum_sun_S_eigenvalue=float(
                np.min(spd_diagnostics(replay.sun_S, name="sun_S").minimum_eigenvalue)
            ),
        )
    else:
        result.update(sun_nis_count=0, sun_nis_normalized_mean=None, minimum_sun_S_eigenvalue=None)

    mag_rows = np.flatnonzero(replay.sensor_code == int(FusionSensorCode.MAGNETOMETER))
    if mag_rows.size:
        errors = right_local_state_error(
            replay.q_NB_history[mag_rows],
            replay.b_g_history[mag_rows],
            joined["q"][mag_rows],
            joined["b"][mag_rows],
        ).delta_theta_rad
        directions = np.stack(
            [
                body_vector_prediction(replay.q_NB_history[row], joined["mag"][row])
                for row in mag_rows
            ]
        )
        parallel = np.abs(np.sum(errors * directions, axis=1))
        perpendicular = np.linalg.norm(errors - np.sum(errors * directions, axis=1)[:, None] * directions, axis=1)
        result["mag_axis_attitude_rms_rad"] = float(np.sqrt(np.mean(parallel * parallel)))
        result["mag_observable_plane_attitude_rms_rad"] = float(
            np.sqrt(np.mean(perpendicular * perpendicular))
        )
    else:
        result["mag_axis_attitude_rms_rad"] = None
        result["mag_observable_plane_attitude_rms_rad"] = None
    return result


@dataclass(frozen=True)
class ScenarioData:
    dataset: Any
    oracle: FusionOracleSidecar
    manifest: dict[str, Any]
    hashes: Any
    train_ids: np.ndarray
    val_ids: np.ndarray
    test_ids: np.ndarray


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("ascii")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_canonical_json(value))


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="ascii"))


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict) or value.get("experiment_version") != EXPERIMENT_VERSION:
        raise ValueError("wrong or malformed Step 2 config")
    frozen = value.get("filter", {}).get("frozen_sensitivity", {})
    if (
        frozen.get("s_Qg"),
        frozen.get("s_Qb"),
        frozen.get("s_R_ST"),
    ) != (0.125, 0.125, 8.0):
        raise ValueError("F-TUNED values differ from the frozen Step 1 comparator")
    if value.get("filter", {}).get("fixed_primary") != "F-BASE":
        raise ValueError("F-BASE must remain the primary classical baseline")
    return value


def _fusion_config(
    config: Mapping[str, Any],
    scenario: FusionScenarioCode,
    *,
    num_trajectories: int | None = None,
    duration_s: float | None = None,
    master_seed: int | None = None,
) -> SensorFusionConfig:
    raw = config["fusion"]
    c4 = config["c4"]
    return SensorFusionConfig(
        num_trajectories=int(num_trajectories or raw["num_trajectories"]),
        duration_s=float(duration_s or raw["duration_s"]),
        master_seed=int(master_seed or raw["master_seed"]),
        gyro_rate_hz=int(raw["gyro_rate_hz"]),
        magnetometer_rate_hz=int(raw["magnetometer_rate_hz"]),
        sun_sensor_rate_hz=int(raw["sun_sensor_rate_hz"]),
        star_tracker_rate_hz=int(raw["star_tracker_rate_hz"]),
        initial_attitude_max_rad=float(raw["initial_attitude_max_rad"]),
        angular_rate_max_rad_s=float(raw["angular_rate_max_rad_s"]),
        gyro_bias_max_rad_s=float(raw["gyro_bias_max_rad_s"]),
        gyro_noise_std_rad_s=float(raw["gyro_noise_std_rad_s"]),
        star_tracker_R_rad2=tuple(tuple(float(item) for item in row) for row in raw["star_tracker_R_rad2"]),
        magnetometer_R=tuple(tuple(float(item) for item in row) for row in raw["magnetometer_R"]),
        sun_tangent_R_rad2=tuple(
            tuple(float(item) for item in row) for row in raw["sun_tangent_R_rad2"]
        ),
        bias_psd_rad2_s3=float(config["filter"]["bias_psd_rad2_s3"]),
        scenario_code=int(scenario),
        alpha_b=float(c4["alpha_b"]),
        alpha_R_mag=float(c4["alpha_R_mag"]),
        slow_window_start_fraction=float(c4["slow_window_fraction"][0]),
        slow_window_end_fraction=float(c4["slow_window_fraction"][1]),
        fast_window_start_fraction=float(c4["fast_window_fraction"][0]),
        fast_window_end_fraction=float(c4["fast_window_fraction"][1]),
        bias_random_walk_enabled=scenario != FusionScenarioCode.UNIT_ST_REDUCTION,
        train_fraction=float(raw["train_fraction"]),
        val_fraction=float(raw["val_fraction"]),
        test_fraction=float(raw["test_fraction"]),
    )


def _artifact_roots(config: Mapping[str, Any]) -> tuple[Path, Path]:
    return Path(config["paths"]["results_root"]), Path(config["paths"]["manifests_root"])


def _scenario_directory_name(scenario: FusionScenarioCode) -> str:
    return {
        FusionScenarioCode.MAIN_FUSION_STATIONARY: "main_fusion_stationary",
        FusionScenarioCode.STRESS_MAG: "stress_mag",
        FusionScenarioCode.C4_COMBINED: "c4_combined",
        FusionScenarioCode.UNIT_ST_REDUCTION: "unit_st_reduction",
    }[scenario]


def _scenario_data_from_generated(generated: GeneratedSensorFusion) -> ScenarioData:
    return ScenarioData(
        dataset=generated.dataset,
        oracle=generated.oracle_context,
        manifest=generated.sensor_manifest,
        hashes=generated.semantic_hashes,
        train_ids=generated.trajectory_split.train_ids,
        val_ids=generated.trajectory_split.val_ids,
        test_ids=generated.trajectory_split.test_ids,
    )


def _save_generated(root: Path, scenario: FusionScenarioCode, generated: GeneratedSensorFusion) -> None:
    scenario_root = root / _scenario_directory_name(scenario)
    hashes = save_fusion_dataset(scenario_root / "sensor", generated.dataset, generated.sensor_manifest)
    if hashes != generated.semantic_hashes:
        raise AssertionError("saved physical dataset identity changed")
    save_fusion_oracle(
        scenario_root / "oracle_simulation_only",
        generated.oracle_context,
        dataset_hash=hashes.dataset_hash,
    )


def _load_scenario(root: Path, scenario: FusionScenarioCode) -> ScenarioData:
    scenario_root = root / _scenario_directory_name(scenario)
    dataset, manifest, hashes = load_fusion_dataset(
        scenario_root / "sensor", expected_generator_id=GENERATOR_ID
    )
    oracle = load_fusion_oracle(
        scenario_root / "oracle_simulation_only", expected_dataset_hash=hashes.dataset_hash
    )
    split = manifest["split"]
    return ScenarioData(
        dataset=dataset,
        oracle=oracle,
        manifest=manifest,
        hashes=hashes,
        train_ids=np.asarray(split["train"], dtype=np.int64),
        val_ids=np.asarray(split["val"], dtype=np.int64),
        test_ids=np.asarray(split["test"], dtype=np.int64),
    )


def _load_or_generate_scenario(
    config: Mapping[str, Any],
    scenario: FusionScenarioCode,
    *,
    base_unit_st: Any | None = None,
) -> tuple[ScenarioData, GeneratedSensorFusion | None]:
    _, manifests_root = _artifact_roots(config)
    scenario_root = manifests_root / _scenario_directory_name(scenario)
    if scenario_root.exists():
        return _load_scenario(manifests_root, scenario), None
    generated = generate_sensor_fusion(_fusion_config(config, scenario), base_unit_st=base_unit_st)
    _save_generated(manifests_root, scenario, generated)
    return _scenario_data_from_generated(generated), generated


def _run_policy(
    data: ScenarioData,
    trajectory_id: int,
    fusion_config: SensorFusionConfig,
    policy: FusionPolicy,
    *,
    table: FusionEventTable | None = None,
) -> FusionReplayResult:
    events = table or data.dataset.events
    Q_c = base_process_covariance(fusion_config)
    if policy.oracle_mode == "none":
        return replay_fixed_policy(events, trajectory_id, default_initial_state(), 0.0, Q_c, policy)
    if table is not None:
        raise ValueError("oracle policies require the unmodified event-order sidecar")
    return replay_oracle_policy(
        events,
        data.oracle,
        trajectory_id,
        default_initial_state(),
        0.0,
        Q_c,
        policy,
    )


def _evaluate(
    config: Mapping[str, Any],
    data: ScenarioData,
    replay: FusionReplayResult,
    scenario_id: str,
    policy_id: str,
) -> dict[str, Any]:
    return evaluate_fusion_replay(
        data.dataset,
        replay,
        scenario_id=scenario_id,
        policy_id=policy_id,
        duration_s=float(data.manifest["generator_config"]["duration_s"]),
        divergence_threshold_rad=float(config["metrics"]["divergence_threshold_rad"]),
        confidence_level=float(config["statistics"]["confidence_level"]),
    )


def _array_fields_equal(left: Any, right: Any, names: Sequence[str]) -> None:
    for name in names:
        if not np.array_equal(getattr(left, name), getattr(right, name)):
            raise AssertionError(f"exact array mismatch at {name}")


def _validation_command(config: Mapping[str, Any]) -> dict[str, Any]:
    reduction_cfg = _fusion_config(
        config,
        FusionScenarioCode.UNIT_ST_REDUCTION,
        num_trajectories=3,
        duration_s=2.0,
        master_seed=20261801,
    )
    reduction = generate_sensor_fusion(reduction_cfg)
    phase1a = fusion_gyro_st_as_phase1a(reduction.dataset.events)
    _array_fields_equal(
        phase1a,
        reduction.base_unit_st.dataset.events,
        (
            "trajectory_id",
            "sensor_code",
            "measurement_time_s",
            "arrival_time_s",
            "event_order",
            "valid",
            "payload_index",
            "gyro_omega_rad_s",
            "star_tracker_q_NB",
            "star_tracker_R_rad2",
        ),
    )
    trajectory_id = int(reduction.trajectory_split.test_ids[0])
    Q_c = base_process_covariance(reduction_cfg)
    direct = replay_trajectory(
        phase1a, trajectory_id, default_initial_state(), 0.0, Q_c
    )
    fusion = replay_fixed_policy(
        reduction.dataset.events,
        trajectory_id,
        default_initial_state(),
        0.0,
        Q_c,
        F_BASE,
    )
    for fusion_name, direct_name in (
        ("event_time_s", "event_time_s"),
        ("event_order", "event_order"),
        ("sensor_code", "sensor_code"),
        ("q_NB_history", "q_NB_history"),
        ("b_g_history", "b_g_history"),
        ("P_history", "P_history"),
        ("star_tracker_residual", "star_tracker_residual"),
        ("star_tracker_S", "star_tracker_S"),
    ):
        if not np.array_equal(getattr(fusion, fusion_name), getattr(direct, direct_name)):
            raise AssertionError(f"Phase 1A reduction replay mismatch at {fusion_name}")

    main_cfg = _fusion_config(
        config,
        FusionScenarioCode.MAIN_FUSION_STATIONARY,
        num_trajectories=3,
        duration_s=10.0,
        master_seed=20261802,
    )
    main = generate_sensor_fusion(main_cfg)
    main_tid = int(main.trajectory_split.test_ids[0])
    assert_all_one_oracle_exact(
        main.dataset.events,
        main.oracle_context,
        main_tid,
        default_initial_state(),
        base_process_covariance(main_cfg),
    )
    replay = replay_fixed_policy(
        main.dataset.events,
        main_tid,
        default_initial_state(),
        0.0,
        base_process_covariance(main_cfg),
        F_BASE,
    )
    if replay.sun_skipped_event_order.size == 0:
        raise AssertionError("validation case must exercise sun validity skip")
    for order in replay.sun_skipped_event_order:
        index = int(np.flatnonzero(replay.event_order == order)[0])
        if index == 0 or not (
            np.array_equal(replay.q_NB_history[index], replay.q_NB_history[index - 1])
            and np.array_equal(replay.P_history[index], replay.P_history[index - 1])
        ):
            raise AssertionError("invalid sun event changed state or covariance")
    with tempfile.TemporaryDirectory(prefix="p1b-fusion-validation-") as tmp:
        sensor_path = Path(tmp) / "sensor"
        oracle_path = Path(tmp) / "oracle"
        saved = save_fusion_dataset(sensor_path, main.dataset, main.sensor_manifest)
        save_fusion_oracle(
            oracle_path, main.oracle_context, dataset_hash=saved.dataset_hash
        )
        loaded, _, hashes = load_fusion_dataset(
            sensor_path, expected_generator_id=GENERATOR_ID
        )
        loaded_oracle = load_fusion_oracle(
            oracle_path, expected_dataset_hash=hashes.dataset_hash
        )
        if hashes != saved or loaded_oracle.semantic_hash != main.oracle_context.semantic_hash:
            raise AssertionError("fusion serialization identity changed")
        _array_fields_equal(loaded.events, main.dataset.events, [item.name for item in fields(FusionEventTable)])
    fixed_parameters = inspect.signature(replay_fixed_policy).parameters
    forbidden = {"truth", "oracle", "event_window", "hidden_label"}
    if forbidden & set(fixed_parameters):
        raise AssertionError("fixed estimator API contains forbidden information")
    evaluated = evaluate_fusion_replay(
        main.dataset,
        replay,
        scenario_id="VALIDATION-MAIN",
        policy_id="F-BASE",
        duration_s=main_cfg.duration_s,
        divergence_threshold_rad=float(config["metrics"]["divergence_threshold_rad"]),
    )
    return {
        "status": "PASS",
        "command": "validate",
        "phase1a_reduction_exact": True,
        "all_one_fixed_oracle_exact": True,
        "fixed_api_truth_free": True,
        "invalid_sun_skip_count": int(replay.sun_skipped_event_order.size),
        "mag_nis_count": evaluated["mag_nis_count"],
        "sun_nis_count": evaluated["sun_nis_count"],
        "dataset_hash": main.semantic_hashes.dataset_hash,
        "oracle_hash": main.oracle_context.semantic_hash,
    }


def _debug_command(config: Mapping[str, Any], scenario_name: str, seed: int | None) -> dict[str, Any]:
    mapping = {
        "main_fusion_stationary": FusionScenarioCode.MAIN_FUSION_STATIONARY,
        "stress_mag": FusionScenarioCode.STRESS_MAG,
        "c4_medium": FusionScenarioCode.C4_COMBINED,
    }
    scenario = mapping[scenario_name]
    cfg = _fusion_config(
        config,
        scenario,
        num_trajectories=4,
        duration_s=4.0,
        master_seed=int(seed or 8101),
    )
    generated = generate_sensor_fusion(cfg)
    data = _scenario_data_from_generated(generated)
    policies = [F_BASE, ORACLE_FULL] if scenario != FusionScenarioCode.C4_COMBINED else [
        F_BASE,
        ORACLE_PROCESS,
        ORACLE_MEASUREMENT,
        ORACLE_FULL,
        WRONG_PROCESS,
        WRONG_MEASUREMENT,
    ]
    records = []
    for trajectory_id in data.test_ids:
        for policy in policies:
            replay = _run_policy(data, int(trajectory_id), cfg, policy)
            records.append(_evaluate(config, data, replay, scenario_name, policy.policy_id))
    if any(item["diverged"] for item in records):
        raise RuntimeError("debug scenario diverged")
    return {
        "status": "PASS",
        "command": "debug",
        "scenario": scenario_name,
        "trajectory_count": int(data.test_ids.size),
        "record_count": len(records),
        "records": records,
    }


def pilot_workload(config: Mapping[str, Any]) -> dict[str, Any]:
    raw = config["fusion"]
    primary_n = int(config["pilot"]["required_test_trajectories_per_condition"])
    ablation_n = int(config["pilot"]["ablation_trajectories"])
    events_full = int(
        float(raw["duration_s"])
        * (
            int(raw["gyro_rate_hz"])
            + int(raw["magnetometer_rate_hz"])
            + int(raw["sun_sensor_rate_hz"])
            + int(raw["star_tracker_rate_hz"])
        )
    )
    policy_trajectories = primary_n * (3 + 2 + 7) + ablation_n * 4
    return {
        "generated_trajectories": int(raw["num_trajectories"]),
        "primary_paired_N": primary_n,
        "ablation_N": ablation_n,
        "duration_s": float(raw["duration_s"]),
        "events_per_full_trajectory": events_full,
        "policy_trajectory_records": policy_trajectories,
        "upper_bound_filter_event_steps": policy_trajectories * events_full,
        "checkpoint_unit": "scenario/policy/trajectory canonical JSON",
        "estimated_runtime_s": 300,
        "estimated_storage_bytes": 100_000_000,
    }


def _record_path(root: Path, scenario: str, policy: str, trajectory_id: int) -> Path:
    return root / "pilot" / "records" / scenario / policy / f"{int(trajectory_id)}.json"


def _pilot_command(
    config: Mapping[str, Any], *, max_trajectories: int | None, resume: bool
) -> dict[str, Any]:
    results_root, manifests_root = _artifact_roots(config)
    records_root = results_root / "pilot" / "records"
    required_n = int(config["pilot"]["required_test_trajectories_per_condition"])
    ablation_n = int(config["pilot"]["ablation_trajectories"])
    if max_trajectories is not None and max_trajectories <= 0:
        raise ValueError("max-trajectories must be positive")
    target_n = min(required_n, int(max_trajectories or required_n))
    start_time = time.monotonic()

    main, main_generated = _load_or_generate_scenario(
        config, FusionScenarioCode.MAIN_FUSION_STATIONARY
    )
    reusable_base = main_generated.base_unit_st if main_generated is not None else None
    stress, _ = _load_or_generate_scenario(
        config, FusionScenarioCode.STRESS_MAG, base_unit_st=reusable_base
    )
    c4, _ = _load_or_generate_scenario(
        config, FusionScenarioCode.C4_COMBINED, base_unit_st=reusable_base
    )
    datasets = {
        "MAIN-FUSION-STATIONARY": (main, FusionScenarioCode.MAIN_FUSION_STATIONARY),
        "STRESS-MAG": (stress, FusionScenarioCode.STRESS_MAG),
        "C4-COMBINED": (c4, FusionScenarioCode.C4_COMBINED),
    }
    policies = {
        "MAIN-FUSION-STATIONARY": [F_BASE, F_TUNED, ORACLE_FULL],
        "STRESS-MAG": [F_BASE, ORACLE_FULL],
        "C4-COMBINED": [
            F_BASE,
            F_TUNED,
            ORACLE_PROCESS,
            ORACLE_MEASUREMENT,
            ORACLE_FULL,
            WRONG_PROCESS,
            WRONG_MEASUREMENT,
        ],
    }
    written = 0
    reused = 0
    for scenario_id, (data, scenario_code) in datasets.items():
        cfg = _fusion_config(config, scenario_code)
        if data.test_ids.size < required_n:
            raise RuntimeError("whole-trajectory test split does not contain required paired N")
        for trajectory_id_raw in data.test_ids[:target_n]:
            trajectory_id = int(trajectory_id_raw)
            for policy in policies[scenario_id]:
                path = _record_path(records_root.parent.parent, scenario_id, policy.policy_id, trajectory_id)
                if path.exists():
                    if not resume:
                        raise FileExistsError("pilot checkpoint exists; use --resume")
                    reused += 1
                    continue
                replay = _run_policy(data, trajectory_id, cfg, policy)
                record = _evaluate(config, data, replay, scenario_id, policy.policy_id)
                record.update(
                    dataset_hash=data.hashes.dataset_hash,
                    oracle_hash=data.oracle.semantic_hash,
                    estimator_knowledge=(
                        "fixed typed events and fixed Q/R only"
                        if policy.oracle_mode == "none"
                        else "simulation-only current-event forward-only sidecar"
                    ),
                    policy=(
                        policy.deployable_artifact()
                        if policy.oracle_mode == "none"
                        else {"policy_id": policy.policy_id, "oracle_mode": policy.oracle_mode}
                    ),
                )
                _write_json(path, record)
                written += 1

    ablations = {
        "ABLATION-GYRO-ST": [FusionSensorCode.GYRO, FusionSensorCode.STAR_TRACKER],
        "ABLATION-GYRO-MAG-ST": [
            FusionSensorCode.GYRO,
            FusionSensorCode.MAGNETOMETER,
            FusionSensorCode.STAR_TRACKER,
        ],
        "ABLATION-GYRO-SUN-ST": [
            FusionSensorCode.GYRO,
            FusionSensorCode.SUN_SENSOR,
            FusionSensorCode.STAR_TRACKER,
        ],
        "ABLATION-FULL": [
            FusionSensorCode.GYRO,
            FusionSensorCode.MAGNETOMETER,
            FusionSensorCode.SUN_SENSOR,
            FusionSensorCode.STAR_TRACKER,
        ],
    }
    main_cfg = _fusion_config(config, FusionScenarioCode.MAIN_FUSION_STATIONARY)
    target_ablation = min(ablation_n, int(max_trajectories or ablation_n))
    for scenario_id, sensor_codes in ablations.items():
        table = select_fusion_sensors(main.dataset.events, sensor_codes)
        for trajectory_id_raw in main.test_ids[:target_ablation]:
            trajectory_id = int(trajectory_id_raw)
            path = _record_path(records_root.parent.parent, scenario_id, F_BASE.policy_id, trajectory_id)
            if path.exists():
                if not resume:
                    raise FileExistsError("ablation checkpoint exists; use --resume")
                reused += 1
                continue
            replay = _run_policy(main, trajectory_id, main_cfg, F_BASE, table=table)
            record = _evaluate(config, main, replay, scenario_id, F_BASE.policy_id)
            record.update(
                dataset_hash=main.hashes.dataset_hash,
                selected_sensor_codes=[int(item) for item in sensor_codes],
                estimator_knowledge="fixed typed event subset and fixed Q/R only",
                policy=F_BASE.deployable_artifact(),
            )
            _write_json(path, record)
            written += 1

    all_records = [
        _read_json(path) for path in sorted((results_root / "pilot" / "records").glob("*/*/*.json"))
    ]
    counts: dict[str, int] = {}
    for record in all_records:
        key = f"{record['scenario_id']}/{record['policy_id']}"
        counts[key] = counts.get(key, 0) + 1
    required_counts = {
        f"{scenario}/{policy.policy_id}": required_n
        for scenario, policy_values in policies.items()
        for policy in policy_values
    }
    required_counts.update({f"{scenario}/F-BASE": ablation_n for scenario in ablations})
    complete = all(counts.get(key, 0) >= value for key, value in required_counts.items())
    output = {
        "status": "COMPLETE" if complete else "PARTIAL",
        "experiment_version": EXPERIMENT_VERSION,
        "required_paired_N_per_primary_condition": required_n,
        "completed_target_N_this_invocation": target_n,
        "required_ablation_N": ablation_n,
        "counts": counts,
        "required_counts": required_counts,
        "record_count": len(all_records),
        "written_this_invocation": written,
        "reused_checkpoints": reused,
        "scenario_dataset_hashes": {
            "MAIN-FUSION-STATIONARY": main.hashes.dataset_hash,
            "STRESS-MAG": stress.hashes.dataset_hash,
            "C4-COMBINED": c4.hashes.dataset_hash,
        },
        "same_realization": {
            "trajectory_ids_equal": bool(
                np.array_equal(main.dataset.truth.trajectory_id, stress.dataset.truth.trajectory_id)
                and np.array_equal(main.dataset.truth.trajectory_id, c4.dataset.truth.trajectory_id)
            ),
            "attitude_truth_equal": bool(
                np.array_equal(main.dataset.truth.q_true_NB, stress.dataset.truth.q_true_NB)
                and np.array_equal(main.dataset.truth.q_true_NB, c4.dataset.truth.q_true_NB)
            ),
            "rate_truth_equal": bool(
                np.array_equal(main.dataset.truth.omega_true_B_rad_s, stress.dataset.truth.omega_true_B_rad_s)
                and np.array_equal(main.dataset.truth.omega_true_B_rad_s, c4.dataset.truth.omega_true_B_rad_s)
            ),
            "sun_payload_unaffected_C4": bool(
                np.array_equal(main.dataset.events.sun_z_sun_B, c4.dataset.events.sun_z_sun_B)
            ),
            "star_tracker_payload_unaffected_C4": bool(
                np.array_equal(
                    main.dataset.events.star_tracker_q_ST_NB,
                    c4.dataset.events.star_tracker_q_ST_NB,
                )
            ),
        },
        "configuration_hash": _sha256_json(config),
        "elapsed_s": time.monotonic() - start_time,
        "runtime": {"python": platform.python_version(), "numpy": np.__version__},
        "checkpoint_layout": "scenario/policy/trajectory canonical JSON",
        "restartable": True,
        "manifests_root": str(manifests_root),
    }
    _write_json(results_root / "pilot" / "pilot_manifest.json", output)
    return output


def _numeric_summary(values: Sequence[Mapping[str, Any]], metric: str) -> dict[str, float] | None:
    samples = [float(item[metric]) for item in values if item.get(metric) is not None]
    if not samples:
        return None
    array = np.asarray(samples, dtype=np.float64)
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p95": float(np.quantile(array, 0.95)),
    }


def summarize_records(
    records: Sequence[Mapping[str, Any]], *, bootstrap_seed: int, resamples: int
) -> dict[str, Any]:
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for record in records:
        grouped.setdefault((str(record["scenario_id"]), str(record["policy_id"])), []).append(record)
    metric_names = (
        "attitude_rmse_rad",
        "attitude_p95_rad",
        "attitude_peak_rad",
        "fast_attitude_rmse_rad",
        "fast_attitude_p95_rad",
        "fast_attitude_peak_rad",
        "bias_vector_rmse_rad_s",
        "slow_bias_rmse_rad_s",
        "post_bias_rmse_rad_s",
        "mag_nis_normalized_mean",
        "sun_nis_normalized_mean",
        "st_nis_normalized_mean",
        "nees_normalized_mean",
        "mag_axis_attitude_rms_rad",
        "mag_observable_plane_attitude_rms_rad",
    )
    groups: dict[str, Any] = {}
    for (scenario, policy), values in sorted(grouped.items()):
        ordered = sorted(values, key=lambda item: int(item["trajectory_id"]))
        groups[f"{scenario}/{policy}"] = {
            "N": len(ordered),
            "trajectory_ids": [int(item["trajectory_id"]) for item in ordered],
            "divergence_count": sum(bool(item["diverged"]) for item in ordered),
            "sun_update_count": sum(int(item["sun_update_count"]) for item in ordered),
            "sun_skip_count": sum(int(item["sun_skip_count"]) for item in ordered),
            "metrics": {
                name: _numeric_summary(ordered, name) for name in metric_names
            },
        }

    paired: dict[str, Any] = {}
    for scenario in sorted({str(item["scenario_id"]) for item in records}):
        base = {
            int(item["trajectory_id"]): item
            for item in records
            if item["scenario_id"] == scenario and item["policy_id"] == "F-BASE"
        }
        policies = sorted(
            {
                str(item["policy_id"])
                for item in records
                if item["scenario_id"] == scenario and item["policy_id"] != "F-BASE"
            }
        )
        for policy in policies:
            candidate = {
                int(item["trajectory_id"]): item
                for item in records
                if item["scenario_id"] == scenario and item["policy_id"] == policy
            }
            ids = sorted(set(base) & set(candidate))
            if not ids:
                continue
            comparisons: dict[str, Any] = {}
            for metric in (
                "attitude_rmse_rad",
                "fast_attitude_peak_rad",
                "slow_bias_rmse_rad_s",
                "mag_nis_normalized_mean",
                "nees_normalized_mean",
            ):
                difference = np.asarray(
                    [float(candidate[item][metric]) - float(base[item][metric]) for item in ids],
                    dtype=np.float64,
                )
                low, high = paired_bootstrap_ci(
                    difference,
                    seed=stable_statistics_seed(bootstrap_seed, scenario, policy, metric),
                    resamples=resamples,
                )
                base_mean = float(np.mean([float(base[item][metric]) for item in ids]))
                comparisons[metric] = {
                    "candidate_minus_base_mean": float(np.mean(difference)),
                    "paired_bootstrap_95_ci": [low, high],
                    "improvement_fraction": (
                        float(-np.mean(difference) / base_mean) if base_mean > 0.0 else None
                    ),
                }
            paired[f"{scenario}/{policy}-minus-F-BASE"] = {
                "N": len(ids),
                "metrics": comparisons,
            }
    return {"groups": groups, "paired_differences": paired}


def _report_command(config: Mapping[str, Any]) -> dict[str, Any]:
    results_root, _ = _artifact_roots(config)
    manifest = _read_json(results_root / "pilot" / "pilot_manifest.json")
    records = [
        _read_json(path) for path in sorted((results_root / "pilot" / "records").glob("*/*/*.json"))
    ]
    summary = summarize_records(
        records,
        bootstrap_seed=int(config["statistics"]["bootstrap_seed"]),
        resamples=int(config["statistics"]["bootstrap_resamples"]),
    )
    output = {
        "experiment_version": EXPERIMENT_VERSION,
        "pilot_status": manifest["status"],
        "required_paired_N_per_primary_condition": manifest[
            "required_paired_N_per_primary_condition"
        ],
        "record_count": len(records),
        "summary": summary,
        "same_realization": manifest["same_realization"],
        "unsupported_claims": [
            "flight-orbit/WMM/eclipse fidelity",
            "full attitude observability from one magnetic vector",
            "general information-theoretic identifiability",
            "flight-product sensor accuracy",
        ],
    }
    _write_json(results_root / "pilot_summary.json", output)
    return output


def _exit_review_command(config: Mapping[str, Any]) -> dict[str, Any]:
    results_root, _ = _artifact_roots(config)
    report = _read_json(results_root / "pilot_summary.json")
    if report["pilot_status"] != "COMPLETE":
        output = {
            "decision": "DEFERRED",
            "status": "PASS_P1B_STEP2_IMPLEMENTATION_ONLY",
            "reason": "required paired N=50 primary pilots are incomplete",
        }
        _write_json(results_root / "exit_review.json", output)
        return output
    groups = report["summary"]["groups"]
    paired = report["summary"]["paired_differences"]
    main = groups["MAIN-FUSION-STATIONARY/F-BASE"]
    c4_process = paired["C4-COMBINED/ORACLE-PROCESS-minus-F-BASE"]["metrics"]
    c4_measurement = paired["C4-COMBINED/ORACLE-MEASUREMENT-minus-F-BASE"]["metrics"]
    c4_full = paired["C4-COMBINED/ORACLE-FULL-minus-F-BASE"]["metrics"]
    threshold = float(config["exit"]["oracle_practical_improvement_fraction"])
    process_useful = (
        c4_process["slow_bias_rmse_rad_s"]["improvement_fraction"] or -math.inf
    ) >= threshold
    measurement_useful = (
        c4_measurement["fast_attitude_peak_rad"]["improvement_fraction"] or -math.inf
    ) >= threshold
    full_useful = any(
        (c4_full[name]["improvement_fraction"] or -math.inf) >= threshold
        for name in ("slow_bias_rmse_rad_s", "fast_attitude_peak_rad")
    )
    mandatory = {
        "primary_N_complete": all(
            groups[key]["N"] >= 50
            for key in (
                "MAIN-FUSION-STATIONARY/F-BASE",
                "STRESS-MAG/F-BASE",
                "C4-COMBINED/F-BASE",
            )
        ),
        "stationary_no_divergence": main["divergence_count"] == 0,
        "mag_nis_available": main["metrics"]["mag_nis_normalized_mean"] is not None,
        "sun_nis_available": main["metrics"]["sun_nis_normalized_mean"] is not None,
        "oracle_useful": process_useful or measurement_useful or full_useful,
    }
    if not all(mandatory.values()):
        decision = "STOP"
    else:
        decision = "CONDITIONAL_GO"
    output = {
        "decision": decision,
        "status": "PASS_P1B_STEP2_SENSOR_FUSION_C4" if decision != "STOP" else "STOP",
        "mandatory_evidence": mandatory,
        "oracle_effects": {
            "process_useful_2pct": process_useful,
            "measurement_useful_2pct": measurement_useful,
            "full_useful_2pct": full_useful,
        },
        "condition_if_go": (
            "retain F-TUNED as a frozen sensitivity comparator and retain classical "
            "process/measurement/wrong-side oracle ablations in any future Phase 2 design"
            if decision == "CONDITIONAL_GO"
            else None
        ),
        "phase2_implemented": False,
    }
    _write_json(results_root / "exit_review.json", output)
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("validate")
    debug = commands.add_parser("debug")
    debug.add_argument(
        "--scenario",
        choices=("main_fusion_stationary", "stress_mag", "c4_medium"),
        required=True,
    )
    debug.add_argument("--seed", type=int)
    pilot = commands.add_parser("pilot")
    pilot.add_argument("--max-trajectories", type=int)
    pilot.add_argument("--resume", action="store_true")
    commands.add_parser("report")
    commands.add_parser("exit-review")
    commands.add_parser("workload")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = _load_config(args.config)
    results_root, manifests_root = _artifact_roots(config)
    results_root.mkdir(parents=True, exist_ok=True)
    manifests_root.mkdir(parents=True, exist_ok=True)
    if args.command == "validate":
        output = _validation_command(config)
    elif args.command == "debug":
        output = _debug_command(config, args.scenario, args.seed)
    elif args.command == "pilot":
        output = _pilot_command(
            config, max_trajectories=args.max_trajectories, resume=args.resume
        )
    elif args.command == "report":
        output = _report_command(config)
    elif args.command == "exit-review":
        output = _exit_review_command(config)
    elif args.command == "workload":
        output = pilot_workload(config)
        _write_json(results_root / "pilot_workload.json", output)
    else:
        raise RuntimeError("unreachable command")
    sys.stdout.write(json.dumps(output, sort_keys=True, indent=2, allow_nan=False) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "EXPERIMENT_VERSION",
    "F_BASE",
    "F_TUNED",
    "FusionPolicy",
    "FusionReplayResult",
    "ORACLE_FULL",
    "ORACLE_MEASUREMENT",
    "ORACLE_PROCESS",
    "WRONG_MEASUREMENT",
    "WRONG_PROCESS",
    "assert_all_one_oracle_exact",
    "base_process_covariance",
    "evaluate_fusion_replay",
    "pilot_workload",
    "replay_fixed_policy",
    "replay_oracle_policy",
    "summarize_records",
]
