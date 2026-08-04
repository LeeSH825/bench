"""Phase 1B Step 1 classical UNIT-ST experiment and command-line driver.

This module owns experiment orchestration only.  Every propagation and
star-tracker correction calls the frozen Gate A implementation, while all
reported consistency quantities call the frozen Gate C metrics.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import yaml

from bench.estimators.mekf import (
    MEKFState,
    propagate_state,
    quat_geodesic_angle,
    star_tracker_update,
)
from bench.metrics.mekf import (
    attitude_geodesic_error_rad,
    bias_error_summary,
    consistency_summary,
    right_local_nees,
    spd_diagnostics,
    star_tracker_nis,
)
from bench.tasks.generator.mekf_events import (
    MEKFDataset,
    MEKFEventTable,
    ReplayResult,
    SensorCode,
    replay_trajectory,
)
from bench.tasks.generator.unit_st_regimes import (
    BASILISK_REGIME_GENERATOR_ID,
    SYNTHETIC_REGIME_GENERATOR_ID,
    CurrentOracleCursor,
    GeneratedUnitSTRegime,
    OracleContextSidecar,
    RegimeCode,
    UnitSTRegimeConfig,
    WindowCode,
    generate_base_unit_st,
    generate_unit_st_regime,
    save_unit_st_regime,
)


EXPERIMENT_VERSION = "p1b-unit-st-classical-v1"
POLICY_CONTRACT_VERSION = "p1b-fixed-policy-v1"
_GYRO = np.int16(SensorCode.GYRO)
_STAR_TRACKER = np.int16(SensorCode.STAR_TRACKER)


def _canonical_json_bytes(value: Mapping[str, Any] | Sequence[Any]) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _write_json(path: Path, value: Mapping[str, Any] | Sequence[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_canonical_json_bytes(value))


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="ascii"))


def _sha256_json(value: Mapping[str, Any] | Sequence[Any]) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _source_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _runtime_and_source_provenance() -> dict[str, Any]:
    repository_root = Path(__file__).resolve().parents[2]
    paths = (
        "bench/estimators/mekf.py",
        "bench/tasks/generator/mekf_events.py",
        "bench/tasks/generator/unit_st_regimes.py",
        "bench/metrics/mekf.py",
        "bench/experiments/phase1b_unit_st_classical.py",
    )
    return {
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": importlib.metadata.version("scipy"),
            "basilisk": importlib.metadata.version("bsk"),
        },
        "source_fingerprints": {
            item: _source_sha256(repository_root / item) for item in paths
        },
    }


def _require_positive(value: float, name: str) -> float:
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result


@dataclass(frozen=True)
class FixedPolicy:
    """Deployable, scenario-wide fixed Q/R policy with no context fields."""

    policy_id: str
    qg_scale: float = 1.0
    qb_scale: float = 1.0
    r_scale: float = 1.0

    def __post_init__(self) -> None:
        if not isinstance(self.policy_id, str) or not self.policy_id:
            raise ValueError("policy_id must be a nonempty string")
        for name in ("qg_scale", "qb_scale", "r_scale"):
            object.__setattr__(self, name, _require_positive(getattr(self, name), name))

    def deployable_artifact(self) -> dict[str, Any]:
        """Return the complete estimator policy artifact; it contains no oracle data."""

        return {
            "policy_contract_version": POLICY_CONTRACT_VERSION,
            "policy_id": self.policy_id,
            "qg_scale": self.qg_scale,
            "qb_scale": self.qb_scale,
            "r_scale": self.r_scale,
        }


F_BASE = FixedPolicy("F-BASE")
F_MIS_Q_LOW = FixedPolicy("F-MIS-Q-LOW", qg_scale=0.25)
F_MIS_Q_HIGH = FixedPolicy("F-MIS-Q-HIGH", qg_scale=4.0)
F_MIS_R_LOW = FixedPolicy("F-MIS-R-LOW", r_scale=0.25)
F_MIS_R_HIGH = FixedPolicy("F-MIS-R-HIGH", r_scale=4.0)


def base_process_covariance(config: UnitSTRegimeConfig, *, bias_psd: float) -> np.ndarray:
    """Return the representative normalized continuous Q_c reference."""

    if not isinstance(config, UnitSTRegimeConfig):
        raise TypeError("config must be a UnitSTRegimeConfig")
    qb = _require_positive(bias_psd, "bias_psd")
    return np.diag(
        np.asarray([config.base_Q_g_rad2_s] * 3 + [qb] * 3, dtype=np.float64)
    )


def default_initial_state() -> MEKFState:
    """Return the locked common prior used by every policy in a comparison."""

    covariance = np.diag(
        np.asarray([0.25**2] * 3 + [0.01**2] * 3, dtype=np.float64)
    )
    return MEKFState(
        q_NB=np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
        b_g=np.zeros(3, dtype=np.float64),
        P=covariance,
    )


def _scaled_process_covariance(
    base_Q_c: np.ndarray, qg_scale: float, qb_scale: float
) -> np.ndarray:
    if not isinstance(base_Q_c, np.ndarray) or base_Q_c.dtype != np.dtype(np.float64):
        raise TypeError("base_Q_c must be a float64 numpy.ndarray")
    if base_Q_c.shape != (6, 6):
        raise ValueError("base_Q_c must have shape [6,6]")
    result = np.array(base_Q_c, dtype=np.float64, copy=True)
    result[:3, :3] *= _require_positive(qg_scale, "qg_scale")
    result[3:, 3:] *= _require_positive(qb_scale, "qb_scale")
    return result


def _replay_with_scale_source(
    event_table: MEKFEventTable,
    trajectory_id: int,
    initial_state: MEKFState,
    initial_time_s: float,
    base_Q_c: np.ndarray,
    *,
    fixed_policy: FixedPolicy | None,
    oracle_cursor: CurrentOracleCursor | None,
    wrong_side: bool,
) -> ReplayResult:
    """Replay with fixed or forward-only current-event covariance scales."""

    if not isinstance(event_table, MEKFEventTable):
        raise TypeError("event_table must be an MEKFEventTable")
    if not isinstance(initial_state, MEKFState):
        raise TypeError("initial_state must be an MEKFState")
    if (fixed_policy is None) == (oracle_cursor is None):
        raise ValueError("exactly one fixed policy or oracle cursor must be supplied")
    current_time = float(initial_time_s)
    if not np.isfinite(current_time) or current_time < 0.0:
        raise ValueError("initial_time_s must be finite and nonnegative")
    rows = np.flatnonzero(event_table.trajectory_id == np.int64(trajectory_id))
    if rows.size == 0:
        raise ValueError("trajectory_id is not present in event table")

    current_state = initial_state
    times: list[float] = []
    orders: list[int] = []
    sensors: list[int] = []
    quaternions: list[np.ndarray] = []
    biases: list[np.ndarray] = []
    covariances: list[np.ndarray] = []
    attitude_steps: list[float] = []
    st_orders: list[int] = []
    st_residuals: list[np.ndarray] = []
    st_covariances: list[np.ndarray] = []

    for row in rows:
        code = event_table.sensor_code[row]
        event_time = float(event_table.measurement_time_s[row])
        order = int(event_table.event_order[row])
        before_q = current_state.q_NB
        if fixed_policy is not None:
            qg_scale = fixed_policy.qg_scale
            qb_scale = fixed_policy.qb_scale
            r_scale = fixed_policy.r_scale
        else:
            if oracle_cursor is None:
                raise RuntimeError("internal oracle cursor state is invalid")
            alpha_g, alpha_b, alpha_r = oracle_cursor.consume(order)
            if wrong_side:
                qg_scale, qb_scale, r_scale = alpha_r, alpha_b, alpha_g
            else:
                qg_scale, qb_scale, r_scale = alpha_g, alpha_b, alpha_r
        process_noise = _scaled_process_covariance(base_Q_c, qg_scale, qb_scale)
        if code == _GYRO:
            if not bool(event_table.valid[row]):
                raise ValueError("invalid gyro events are forbidden")
            if not event_time > current_time:
                raise ValueError("gyro event time must be strictly later than filter time")
            payload = int(event_table.payload_index[row])
            result = propagate_state(
                current_state,
                event_table.gyro_omega_rad_s[payload],
                event_time - current_time,
                process_noise,
            )
            current_state = result.state
            current_time = event_time
        elif code == _STAR_TRACKER:
            if event_time != current_time:
                raise ValueError("star-tracker time must equal the current propagation time")
            if not bool(event_table.valid[row]):
                continue
            payload = int(event_table.payload_index[row])
            result = star_tracker_update(
                current_state,
                event_table.star_tracker_q_NB[payload],
                event_table.star_tracker_R_rad2[payload] * r_scale,
            )
            current_state = result.state
            st_orders.append(order)
            st_residuals.append(result.residual)
            st_covariances.append(result.S)
        else:
            raise ValueError("unknown sensor code")
        times.append(current_time)
        orders.append(order)
        sensors.append(int(code))
        quaternions.append(current_state.q_NB)
        biases.append(current_state.b_g)
        covariances.append(current_state.P)
        attitude_steps.append(quat_geodesic_angle(before_q, current_state.q_NB))

    count = len(times)
    return ReplayResult(
        trajectory_id=int(trajectory_id),
        processed_event_count=count,
        event_time_s=np.asarray(times, dtype=np.float64),
        event_order=np.asarray(orders, dtype=np.int64),
        sensor_code=np.asarray(sensors, dtype=np.int16),
        q_NB_history=np.asarray(quaternions, dtype=np.float64).reshape(count, 4),
        b_g_history=np.asarray(biases, dtype=np.float64).reshape(count, 3),
        P_history=np.asarray(covariances, dtype=np.float64).reshape(count, 6, 6),
        attitude_step_rad=np.asarray(attitude_steps, dtype=np.float64),
        star_tracker_event_order=np.asarray(st_orders, dtype=np.int64),
        star_tracker_residual=np.asarray(st_residuals, dtype=np.float64).reshape(-1, 3),
        star_tracker_S=np.asarray(st_covariances, dtype=np.float64).reshape(-1, 3, 3),
        final_state=current_state,
    )


def replay_fixed_policy(
    event_table: MEKFEventTable,
    trajectory_id: int,
    initial_state: MEKFState,
    initial_time_s: float,
    base_Q_c: np.ndarray,
    policy: FixedPolicy,
) -> ReplayResult:
    """Deployable replay API.  It deliberately accepts no oracle context."""

    if not isinstance(policy, FixedPolicy):
        raise TypeError("policy must be a FixedPolicy")
    return _replay_with_scale_source(
        event_table,
        trajectory_id,
        initial_state,
        initial_time_s,
        base_Q_c,
        fixed_policy=policy,
        oracle_cursor=None,
        wrong_side=False,
    )


def replay_oracle_policy(
    event_table: MEKFEventTable,
    trajectory_id: int,
    initial_state: MEKFState,
    initial_time_s: float,
    base_Q_c: np.ndarray,
    oracle_context: OracleContextSidecar,
) -> ReplayResult:
    """Simulation-only replay using only each currently consumed event's scale."""

    if not isinstance(oracle_context, OracleContextSidecar):
        raise TypeError("oracle_context must be an OracleContextSidecar")
    return _replay_with_scale_source(
        event_table,
        trajectory_id,
        initial_state,
        initial_time_s,
        base_Q_c,
        fixed_policy=None,
        oracle_cursor=oracle_context.cursor(trajectory_id),
        wrong_side=False,
    )


def replay_wrong_side_policy(
    event_table: MEKFEventTable,
    trajectory_id: int,
    initial_state: MEKFState,
    initial_time_s: float,
    base_Q_c: np.ndarray,
    oracle_context: OracleContextSidecar,
) -> ReplayResult:
    """Diagnostic control mapping alpha_g to R and alpha_R to Q_g."""

    if not isinstance(oracle_context, OracleContextSidecar):
        raise TypeError("oracle_context must be an OracleContextSidecar")
    return _replay_with_scale_source(
        event_table,
        trajectory_id,
        initial_state,
        initial_time_s,
        base_Q_c,
        fixed_policy=None,
        oracle_cursor=oracle_context.cursor(trajectory_id),
        wrong_side=True,
    )


def assert_all_one_replay_exact(
    event_table: MEKFEventTable,
    trajectory_id: int,
    initial_state: MEKFState,
    initial_time_s: float,
    base_Q_c: np.ndarray,
) -> None:
    """Require bit-exact equality with the frozen Gate B1 direct replay."""

    direct = replay_trajectory(
        event_table, trajectory_id, initial_state, initial_time_s, base_Q_c
    )
    fixed = replay_fixed_policy(
        event_table, trajectory_id, initial_state, initial_time_s, base_Q_c, F_BASE
    )
    array_names = (
        "event_time_s",
        "event_order",
        "sensor_code",
        "q_NB_history",
        "b_g_history",
        "P_history",
        "attitude_step_rad",
        "star_tracker_event_order",
        "star_tracker_residual",
        "star_tracker_S",
    )
    for name in array_names:
        if not np.array_equal(getattr(direct, name), getattr(fixed, name)):
            raise AssertionError(f"all-one fixed replay differs from Gate B1 in {name}")
    for name in ("q_NB", "b_g", "P"):
        if not np.array_equal(getattr(direct.final_state, name), getattr(fixed.final_state, name)):
            raise AssertionError(f"all-one final state differs from Gate B1 in {name}")


def _truth_join(dataset: MEKFDataset, replay: ReplayResult) -> tuple[np.ndarray, np.ndarray]:
    """Exact post-estimation join by trajectory ID and timestamp."""

    matches = np.flatnonzero(dataset.truth.trajectory_id == np.int64(replay.trajectory_id))
    if matches.size != 1:
        raise ValueError("truth trajectory identity is not unique")
    index = int(matches[0])
    start = int(dataset.truth.truth_offsets[index])
    stop = int(dataset.truth.truth_offsets[index + 1])
    truth_time = dataset.truth.truth_time_s[start:stop]
    rows = np.searchsorted(truth_time, replay.event_time_s)
    if np.any(rows >= truth_time.size) or not np.array_equal(
        truth_time[rows], replay.event_time_s
    ):
        raise ValueError("estimator/truth timestamps do not join exactly")
    return (
        np.asarray(dataset.truth.q_true_NB[start:stop][rows], dtype=np.float64),
        np.asarray(dataset.truth.gyro_bias_rad_s[start:stop][rows], dtype=np.float64),
    )


def _window_for_replay(
    context: OracleContextSidecar, replay: ReplayResult
) -> np.ndarray:
    rows = np.flatnonzero(context.trajectory_id == np.int64(replay.trajectory_id))
    if not np.array_equal(context.event_order[rows], replay.event_order):
        raise ValueError("oracle/evaluation event ordering does not match replay")
    return np.asarray(context.event_window_id[rows], dtype=np.int8)


def recovery_time_s(
    event_time_s: np.ndarray,
    attitude_error_rad: np.ndarray,
    event_window_id: np.ndarray,
    *,
    absolute_floor_rad: float,
    multiplier: float = 1.2,
    sustained_samples: int = 3,
) -> float | None:
    """First sustained post-event recovery to the locked pre-event threshold."""

    if not all(isinstance(item, np.ndarray) for item in (event_time_s, attitude_error_rad, event_window_id)):
        raise TypeError("recovery inputs must be numpy arrays")
    if event_time_s.shape != attitude_error_rad.shape or event_time_s.shape != event_window_id.shape:
        raise ValueError("recovery input shapes must match")
    pre = attitude_error_rad[event_window_id == np.int8(WindowCode.PRE_EVENT)]
    event = np.flatnonzero(event_window_id == np.int8(WindowCode.EVENT))
    recovery = np.flatnonzero(event_window_id == np.int8(WindowCode.RECOVERY))
    if pre.size == 0 or event.size == 0 or recovery.size < sustained_samples:
        return None
    threshold = max(
        _require_positive(absolute_floor_rad, "absolute_floor_rad"),
        _require_positive(multiplier, "multiplier") * float(np.sqrt(np.mean(pre * pre))),
    )
    event_end_time = float(event_time_s[event[-1]])
    for offset in range(recovery.size - sustained_samples + 1):
        rows = recovery[offset : offset + sustained_samples]
        if bool(np.all(attitude_error_rad[rows] <= threshold)):
            return float(event_time_s[rows[0]] - event_end_time)
    return None


def _lag_one_autocorrelation(residual: np.ndarray) -> float | None:
    if residual.shape[0] < 3:
        return None
    norms = np.linalg.norm(residual, axis=1)
    centered = norms - float(np.mean(norms))
    denominator = float(centered @ centered)
    if denominator <= np.finfo(np.float64).eps:
        return None
    return float(centered[:-1] @ centered[1:] / denominator)


def _event_gyro_statistics(
    dataset: MEKFDataset,
    context: OracleContextSidecar,
    trajectory_id: int,
) -> dict[str, float]:
    """Return raw-measurement statistics without consulting truth or policy output."""

    rows = np.flatnonzero(dataset.events.trajectory_id == np.int64(trajectory_id))
    context_rows = np.flatnonzero(context.trajectory_id == np.int64(trajectory_id))
    if not np.array_equal(dataset.events.event_order[rows], context.event_order[context_rows]):
        raise ValueError("raw gyro/context event identities do not match")
    mask = (
        (dataset.events.sensor_code[rows] == _GYRO)
        & (context.event_window_id[context_rows] == np.int8(WindowCode.EVENT))
    )
    gyro_rows = rows[mask]
    payload = dataset.events.payload_index[gyro_rows]
    values = dataset.events.gyro_omega_rad_s[payload]
    if values.shape[0] < 2:
        raise ValueError("event window must contain at least two gyro measurements")
    increments = np.diff(values, axis=0)
    return {
        "event_raw_gyro_measurement_rms_rad_s": float(np.sqrt(np.mean(values * values))),
        "event_raw_gyro_increment_rms_rad_s": float(
            np.sqrt(np.mean(increments * increments))
        ),
    }


def evaluate_replay(
    dataset: MEKFDataset,
    context: OracleContextSidecar,
    replay: ReplayResult,
    *,
    scenario_id: str,
    policy_id: str,
    recovery_floor_rad: float,
    divergence_threshold_rad: float,
) -> dict[str, Any]:
    """Evaluate one completed estimator trace using only canonical Gate C metrics."""

    q_true, bias_true = _truth_join(dataset, replay)
    windows = _window_for_replay(context, replay)
    attitude = attitude_geodesic_error_rad(replay.q_NB_history, q_true)
    bias = bias_error_summary(replay.b_g_history, bias_true)
    spd_p = spd_diagnostics(replay.P_history, name="posterior covariance")
    nees = right_local_nees(
        replay.q_NB_history,
        replay.b_g_history,
        replay.P_history,
        q_true,
        bias_true,
        estimate_time_s=replay.event_time_s,
        covariance_time_s=replay.event_time_s,
        truth_time_s=replay.event_time_s,
        estimate_trajectory_id=np.full(replay.processed_event_count, replay.trajectory_id, dtype=np.int64),
        covariance_trajectory_id=np.full(replay.processed_event_count, replay.trajectory_id, dtype=np.int64),
        truth_trajectory_id=np.full(replay.processed_event_count, replay.trajectory_id, dtype=np.int64),
    )
    nis = star_tracker_nis(replay.star_tracker_residual, replay.star_tracker_S)
    spd_s = spd_diagnostics(replay.star_tracker_S, name="innovation covariance")
    nis_summary = consistency_summary(nis, dof_per_sample=3)
    nees_summary = consistency_summary(nees, dof_per_sample=6)
    event_rows = windows == np.int8(WindowCode.EVENT)
    if not np.any(event_rows):
        raise ValueError("evaluation trajectory has no event-window samples")
    st_window_lookup = {int(order): int(window) for order, window in zip(replay.event_order, windows)}
    st_event_mask = np.asarray(
        [st_window_lookup[int(order)] == int(WindowCode.EVENT) for order in replay.star_tracker_event_order],
        dtype=np.bool_,
    )
    event_residual = replay.star_tracker_residual[st_event_mask]
    if event_residual.size == 0:
        raise ValueError("evaluation trajectory has no event-window star-tracker updates")
    recovered = recovery_time_s(
        replay.event_time_s,
        attitude,
        windows,
        absolute_floor_rad=recovery_floor_rad,
    )
    peak = float(np.max(attitude[event_rows]))
    divergent = bool(
        not np.all(spd_p.cholesky_succeeded)
        or not np.all(spd_s.cholesky_succeeded)
        or peak > _require_positive(divergence_threshold_rad, "divergence_threshold_rad")
    )
    innovation_norm = np.linalg.norm(event_residual, axis=1)
    raw_gyro = _event_gyro_statistics(dataset, context, replay.trajectory_id)
    return {
        "experiment_version": EXPERIMENT_VERSION,
        "scenario_id": scenario_id,
        "policy_id": policy_id,
        "trajectory_id": int(replay.trajectory_id),
        "event_count": int(replay.processed_event_count),
        "star_tracker_update_count": int(replay.star_tracker_event_order.size),
        "attitude_rmse_rad": float(np.sqrt(np.mean(attitude * attitude))),
        "attitude_event_rmse_rad": float(np.sqrt(np.mean(attitude[event_rows] ** 2))),
        "attitude_event_p95_rad": float(np.quantile(attitude[event_rows], 0.95)),
        "attitude_event_peak_rad": peak,
        "bias_vector_rmse_rad_s": bias.vector_rmse_rad_s,
        "event_innovation_rms_rad": float(np.sqrt(np.mean(event_residual * event_residual))),
        "event_innovation_norm_median_rad": float(np.median(innovation_norm)),
        "event_innovation_norm_p95_rad": float(np.quantile(innovation_norm, 0.95)),
        "event_innovation_lag1": _lag_one_autocorrelation(event_residual),
        "nis_normalized_mean": nis_summary.normalized_mean,
        "nees_normalized_mean": nees_summary.normalized_mean,
        "minimum_p_eigenvalue": float(np.min(spd_p.minimum_eigenvalue)),
        "minimum_s_eigenvalue": float(np.min(spd_s.minimum_eigenvalue)),
        "recovery_time_s": recovered,
        "diverged": divergent,
        **raw_gyro,
    }


def paired_bootstrap_ci(
    paired_differences: np.ndarray,
    *,
    seed: int,
    resamples: int = 2000,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Deterministic paired percentile bootstrap confidence interval."""

    if not isinstance(paired_differences, np.ndarray) or paired_differences.dtype != np.float64:
        raise TypeError("paired_differences must be a float64 numpy array")
    if paired_differences.ndim != 1 or paired_differences.size == 0:
        raise ValueError("paired_differences must be nonempty and one-dimensional")
    if not np.all(np.isfinite(paired_differences)):
        raise ValueError("paired_differences must be finite")
    if int(resamples) != resamples or resamples < 2000:
        raise ValueError("resamples must be an integer of at least 2000")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie strictly between zero and one")
    rng = np.random.default_rng(int(seed))
    means = np.empty(int(resamples), dtype=np.float64)
    for index in range(int(resamples)):
        sample = rng.integers(0, paired_differences.size, size=paired_differences.size)
        means[index] = float(np.mean(paired_differences[sample]))
    tail = (1.0 - confidence) / 2.0
    return float(np.quantile(means, tail)), float(np.quantile(means, 1.0 - tail))


def summarize_records(
    records: Sequence[Mapping[str, Any]], *, bootstrap_seed: int, resamples: int
) -> dict[str, Any]:
    """Summarize per-trajectory records and paired differences by exact ID."""

    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for record in records:
        grouped.setdefault((str(record["scenario_id"]), str(record["policy_id"])), []).append(record)
    summaries: dict[str, Any] = {}
    for (scenario, policy), values in sorted(grouped.items()):
        ordered = sorted(values, key=lambda item: int(item["trajectory_id"]))
        metrics: dict[str, Any] = {}
        for metric in (
            "attitude_event_rmse_rad",
            "attitude_event_p95_rad",
            "attitude_event_peak_rad",
            "bias_vector_rmse_rad_s",
            "event_innovation_rms_rad",
            "event_innovation_norm_median_rad",
            "event_innovation_norm_p95_rad",
            "event_raw_gyro_measurement_rms_rad_s",
            "event_raw_gyro_increment_rms_rad_s",
            "nis_normalized_mean",
            "nees_normalized_mean",
        ):
            samples = np.asarray([float(item[metric]) for item in ordered], dtype=np.float64)
            metrics[metric] = {
                "mean": float(np.mean(samples)),
                "median": float(np.median(samples)),
                "p95": float(np.quantile(samples, 0.95)),
            }
        lag_values = [
            float(item["event_innovation_lag1"])
            for item in ordered
            if item["event_innovation_lag1"] is not None
        ]
        recovery_values = [
            float(item["recovery_time_s"])
            for item in ordered
            if item["recovery_time_s"] is not None
        ]
        summaries[f"{scenario}/{policy}"] = {
            "N": len(ordered),
            "trajectory_ids": [int(item["trajectory_id"]) for item in ordered],
            "divergence_count": sum(bool(item["diverged"]) for item in ordered),
            "recovered_count": len(recovery_values),
            "mean_recovery_time_s": (
                float(np.mean(recovery_values)) if recovery_values else None
            ),
            "mean_event_innovation_lag1": (
                float(np.mean(lag_values)) if lag_values else None
            ),
            "metrics": metrics,
        }
    paired: dict[str, Any] = {}
    scenarios = sorted({str(item["scenario_id"]) for item in records})
    for scenario in scenarios:
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
            difference = np.asarray(
                [
                    float(candidate[item]["attitude_event_rmse_rad"])
                    - float(base[item]["attitude_event_rmse_rad"])
                    for item in ids
                ],
                dtype=np.float64,
            )
            low, high = paired_bootstrap_ci(
                difference, seed=stable_statistics_seed(bootstrap_seed, scenario, policy), resamples=resamples
            )
            paired[f"{scenario}/{policy}-minus-F-BASE"] = {
                "N": len(ids),
                "mean_attitude_event_rmse_difference_rad": float(np.mean(difference)),
                "paired_bootstrap_95_ci_rad": [low, high],
            }
    return {"groups": summaries, "paired_differences": paired}


def summarize_c5_pair(
    records: Sequence[Mapping[str, Any]], *, bootstrap_seed: int, resamples: int
) -> dict[str, Any]:
    """Compare the frozen C5-A/B conditions by exact test trajectory pairing."""

    metrics = (
        "event_innovation_rms_rad",
        "event_innovation_norm_p95_rad",
        "event_raw_gyro_measurement_rms_rad_s",
        "event_raw_gyro_increment_rms_rad_s",
        "attitude_event_rmse_rad",
        "bias_vector_rmse_rad_s",
        "nis_normalized_mean",
        "nees_normalized_mean",
    )
    result: dict[str, Any] = {}
    for policy in ("F-BASE", "F-TUNED", "ORACLE-QR", "WRONG-SIDE"):
        first = {
            int(item["trajectory_id"]): item
            for item in records
            if item["scenario_id"] == "C5-A-GYRO-MEDIUM" and item["policy_id"] == policy
        }
        second = {
            int(item["trajectory_id"]): item
            for item in records
            if item["scenario_id"] == "C5-B-ST-MATCHED-RMS" and item["policy_id"] == policy
        }
        ids = sorted(set(first) & set(second))
        if not ids:
            continue
        comparisons: dict[str, Any] = {}
        for metric in metrics:
            difference = np.asarray(
                [float(second[item][metric]) - float(first[item][metric]) for item in ids],
                dtype=np.float64,
            )
            low, high = paired_bootstrap_ci(
                difference,
                seed=stable_statistics_seed(bootstrap_seed, "C5-B-minus-A", policy, metric),
                resamples=resamples,
            )
            comparisons[metric] = {
                "A_mean": float(np.mean([float(first[item][metric]) for item in ids])),
                "B_mean": float(np.mean([float(second[item][metric]) for item in ids])),
                "B_minus_A_mean": float(np.mean(difference)),
                "paired_bootstrap_95_ci": [low, high],
            }
        lag_a = [
            float(first[item]["event_innovation_lag1"])
            for item in ids
            if first[item]["event_innovation_lag1"] is not None
        ]
        lag_b = [
            float(second[item]["event_innovation_lag1"])
            for item in ids
            if second[item]["event_innovation_lag1"] is not None
        ]
        comparisons["event_innovation_lag1"] = {
            "A_mean": float(np.mean(lag_a)) if lag_a else None,
            "B_mean": float(np.mean(lag_b)) if lag_b else None,
        }
        a_rms = comparisons["event_innovation_rms_rad"]["A_mean"]
        b_rms = comparisons["event_innovation_rms_rad"]["B_mean"]
        result[policy] = {
            "N": len(ids),
            "independent_test_rms_relative_difference": abs(b_rms - a_rms) / a_rms,
            "metrics": comparisons,
        }
    return result


def stable_statistics_seed(master_seed: int, *parts: str) -> int:
    payload = "\0".join((EXPERIMENT_VERSION, str(int(master_seed)), *parts)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little", signed=False)


def _objective(records: Sequence[Mapping[str, Any]], policy: FixedPolicy) -> tuple[Any, ...]:
    divergence = sum(bool(item["diverged"]) for item in records)
    attitude = float(np.mean([float(item["attitude_rmse_rad"]) for item in records]))
    bias = float(np.mean([float(item["bias_vector_rmse_rad_s"]) for item in records]))
    consistency = float(
        np.mean(
            [
                abs(float(item["nis_normalized_mean"]) - 1.0)
                + abs(float(item["nees_normalized_mean"]) - 1.0)
                for item in records
            ]
        )
    )
    return (
        divergence,
        attitude,
        bias,
        consistency,
        policy.qg_scale,
        policy.qb_scale,
        policy.r_scale,
    )


def _neighbor_triplet(grid: Sequence[float], selected: float) -> tuple[float, float, float]:
    ordered = tuple(sorted({_require_positive(item, "candidate scale") for item in grid}))
    index = ordered.index(float(selected))
    lower = ordered[index - 1] if index else ordered[index] / 2.0
    upper = ordered[index + 1] if index + 1 < len(ordered) else ordered[index] * 2.0
    return lower, ordered[index], upper


def _replay_and_evaluate_fixed(
    generated: GeneratedUnitSTRegime,
    trajectory_ids: Iterable[int],
    base_Q_c: np.ndarray,
    policy: FixedPolicy,
    *,
    scenario_id: str,
    recovery_floor_rad: float,
    divergence_threshold_rad: float,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for trajectory_id in trajectory_ids:
        replay = replay_fixed_policy(
            generated.dataset.events,
            int(trajectory_id),
            default_initial_state(),
            0.0,
            base_Q_c,
            policy,
        )
        records.append(
            evaluate_replay(
                generated.dataset,
                generated.oracle_context,
                replay,
                scenario_id=scenario_id,
                policy_id=policy.policy_id,
                recovery_floor_rad=recovery_floor_rad,
                divergence_threshold_rad=divergence_threshold_rad,
            )
        )
    return records


def tune_fixed_policy(
    generated: GeneratedUnitSTRegime,
    base_Q_c: np.ndarray,
    *,
    candidates: Sequence[float] = (0.25, 0.5, 1.0, 2.0, 4.0),
    recovery_floor_rad: float = 1.0e-3,
    divergence_threshold_rad: float = 0.5,
) -> tuple[FixedPolicy, dict[str, Any]]:
    """Execute the predeclared 5+5+5+27 train/validation-only search."""

    if not isinstance(generated, GeneratedUnitSTRegime):
        raise TypeError("generated must be a GeneratedUnitSTRegime")
    if RegimeCode(int(generated.oracle_context.regime_code[0])) != RegimeCode.STATIONARY:
        raise ValueError("fixed tuning must use the stationary matched scenario")
    grid = tuple(sorted({_require_positive(item, "candidate") for item in candidates}))
    if len(grid) != 5 or 1.0 not in grid:
        raise ValueError("staged tuning requires five unique positive candidates including one")
    train_ids = tuple(int(item) for item in generated.trajectory_split.train_ids)
    val_ids = tuple(int(item) for item in generated.trajectory_split.val_ids)
    evaluated: list[dict[str, Any]] = []

    def evaluate(policy: FixedPolicy, stage: str) -> tuple[Any, ...]:
        train_records = _replay_and_evaluate_fixed(
            generated,
            train_ids,
            base_Q_c,
            policy,
            scenario_id="C1-STATIONARY-TRAIN",
            recovery_floor_rad=recovery_floor_rad,
            divergence_threshold_rad=divergence_threshold_rad,
        )
        val_records = _replay_and_evaluate_fixed(
            generated,
            val_ids,
            base_Q_c,
            policy,
            scenario_id="C1-STATIONARY-VAL",
            recovery_floor_rad=recovery_floor_rad,
            divergence_threshold_rad=divergence_threshold_rad,
        )
        train_objective = _objective(train_records, policy)
        val_objective = _objective(val_records, policy)
        evaluated.append(
            {
                "stage": stage,
                "policy": policy.deployable_artifact(),
                "train_objective": list(train_objective),
                "validation_objective": list(val_objective),
                "train_trajectory_ids": list(train_ids),
                "validation_trajectory_ids": list(val_ids),
            }
        )
        return val_objective

    stage_qg = [(evaluate(FixedPolicy(f"T-QG-{value:g}", qg_scale=value), "Qg-coordinate"), value) for value in grid]
    selected_qg = min(stage_qg)[1]
    stage_qb = [
        (
            evaluate(
                FixedPolicy(f"T-QB-{value:g}", qg_scale=selected_qg, qb_scale=value),
                "Qb-coordinate",
            ),
            value,
        )
        for value in grid
    ]
    selected_qb = min(stage_qb)[1]
    stage_r = [
        (
            evaluate(
                FixedPolicy(
                    f"T-R-{value:g}", qg_scale=selected_qg, qb_scale=selected_qb, r_scale=value
                ),
                "R-coordinate",
            ),
            value,
        )
        for value in grid
    ]
    selected_r = min(stage_r)[1]
    local: list[tuple[tuple[Any, ...], FixedPolicy]] = []
    for qg in _neighbor_triplet(grid, selected_qg):
        for qb in _neighbor_triplet(grid, selected_qb):
            for r in _neighbor_triplet(grid, selected_r):
                policy = FixedPolicy("F-TUNED", qg_scale=qg, qb_scale=qb, r_scale=r)
                local.append((evaluate(policy, "local-3x3x3"), policy))
    selected = min(local, key=lambda item: item[0])[1]
    manifest = {
        "experiment_version": EXPERIMENT_VERSION,
        "tuning_contract": "stationary train/validation only; test untouched until freeze",
        "candidate_budget": 42,
        "candidate_grid": list(grid),
        "objective_priority": [
            "divergence_or_spd_failure",
            "attitude_geodesic_rmse",
            "bias_vector_rmse",
            "nis_nees_normalized_mean_penalty",
            "lexicographic_scale_tie_break",
        ],
        "selected_policy": selected.deployable_artifact(),
        "candidate_results": evaluated,
        "evaluated_candidate_count": len(evaluated),
        "test_split_accessed": False,
        "frozen_before_test": True,
        "raw_sensor_stream_hash": generated.semantic_hashes.dataset_hash,
    }
    if len(evaluated) != 42:
        raise RuntimeError("staged tuning candidate budget changed")
    return selected, manifest


def match_c5_innovation_rms(
    c2_generated: GeneratedUnitSTRegime,
    base_config: UnitSTRegimeConfig,
    base_Q_c: np.ndarray,
    *,
    candidate_alpha_R: Sequence[float],
    tolerance_fraction: float = 0.05,
    recovery_floor_rad: float = 1.0e-3,
    divergence_threshold_rad: float = 0.5,
) -> tuple[float, dict[str, Any]]:
    """Freeze C5-B on validation trajectories before independent test replay."""

    if RegimeCode(int(c2_generated.oracle_context.regime_code[0])) != RegimeCode.C2_GYRO_PROCESS_STEP:
        raise ValueError("C5-A must be the C2 medium process-uncertainty condition")
    val_ids = tuple(int(item) for item in c2_generated.trajectory_split.val_ids)
    target_records = _replay_and_evaluate_fixed(
        c2_generated,
        val_ids,
        base_Q_c,
        F_BASE,
        scenario_id="C5-A-VALIDATION",
        recovery_floor_rad=recovery_floor_rad,
        divergence_threshold_rad=divergence_threshold_rad,
    )
    target = float(np.mean([float(item["event_innovation_rms_rad"]) for item in target_records]))
    candidates: list[dict[str, Any]] = []
    base_generated = generate_base_unit_st(base_config)
    for alpha in candidate_alpha_R:
        value = _require_positive(alpha, "candidate_alpha_R")
        if value < 1.0:
            raise ValueError("candidate alpha_R must be at least one")
        config = UnitSTRegimeConfig(
            **{
                **asdict(base_config),
                "regime_code": int(RegimeCode.C3_STAR_TRACKER_RELIABILITY_STEP),
                "event_covariance_multiplier": value,
            }
        )
        generated = generate_unit_st_regime(config, base_generated=base_generated)
        records = _replay_and_evaluate_fixed(
            generated,
            val_ids,
            base_Q_c,
            F_BASE,
            scenario_id=f"C5-B-ALPHA-{value:g}-VALIDATION",
            recovery_floor_rad=recovery_floor_rad,
            divergence_threshold_rad=divergence_threshold_rad,
        )
        rms = float(np.mean([float(item["event_innovation_rms_rad"]) for item in records]))
        relative = abs(rms - target) / target
        candidates.append({"alpha_R": value, "innovation_rms_rad": rms, "relative_difference": relative})
    selected = min(candidates, key=lambda item: (item["relative_difference"], item["alpha_R"]))
    manifest = {
        "experiment_version": EXPERIMENT_VERSION,
        "pair": "C5 matched innovation-RMS A/B",
        "A": "C2 medium alpha_g with F-BASE",
        "B": "C3 alpha_R selected on validation with F-BASE",
        "target_innovation_rms_rad": target,
        "tolerance_fraction": float(tolerance_fraction),
        "candidate_results": candidates,
        "selected_alpha_R": selected["alpha_R"],
        "selected_relative_difference": selected["relative_difference"],
        "match_within_tolerance": bool(selected["relative_difference"] <= tolerance_fraction),
        "validation_trajectory_ids": list(val_ids),
        "test_split_accessed": False,
        "frozen_before_test": True,
    }
    return float(selected["alpha_R"]), manifest


def _load_config(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("experiment config root must be a mapping")
    if value.get("experiment_version") != EXPERIMENT_VERSION:
        raise ValueError("experiment config version mismatch")
    return value


def _regime_config(config: Mapping[str, Any], **overrides: Any) -> UnitSTRegimeConfig:
    raw = dict(config["unit_st"])
    raw.update(overrides)
    return UnitSTRegimeConfig(**raw)


def _base_q(config: Mapping[str, Any], regime_config: UnitSTRegimeConfig) -> np.ndarray:
    return base_process_covariance(regime_config, bias_psd=float(config["filter"]["bias_psd"]))


def _run_policy(
    generated: GeneratedUnitSTRegime,
    trajectory_id: int,
    base_Q_c: np.ndarray,
    policy: FixedPolicy | str,
) -> ReplayResult:
    if isinstance(policy, FixedPolicy):
        return replay_fixed_policy(
            generated.dataset.events,
            trajectory_id,
            default_initial_state(),
            0.0,
            base_Q_c,
            policy,
        )
    if policy == "ORACLE-QR":
        return replay_oracle_policy(
            generated.dataset.events,
            trajectory_id,
            default_initial_state(),
            0.0,
            base_Q_c,
            generated.oracle_context,
        )
    if policy == "WRONG-SIDE":
        return replay_wrong_side_policy(
            generated.dataset.events,
            trajectory_id,
            default_initial_state(),
            0.0,
            base_Q_c,
            generated.oracle_context,
        )
    raise ValueError("unknown policy")


def _policy_id(policy: FixedPolicy | str) -> str:
    return policy.policy_id if isinstance(policy, FixedPolicy) else policy


def _evaluation_options(config: Mapping[str, Any]) -> dict[str, float]:
    return {
        "recovery_floor_rad": float(config["metrics"]["recovery_absolute_floor_rad"]),
        "divergence_threshold_rad": float(config["metrics"]["divergence_threshold_rad"]),
    }


def _artifact_roots(config: Mapping[str, Any]) -> tuple[Path, Path]:
    return Path(config["paths"]["results_root"]), Path(config["paths"]["manifests_root"])


def _validation_command(config: Mapping[str, Any]) -> dict[str, Any]:
    results_root, _ = _artifact_roots(config)
    cfg = _regime_config(
        config,
        truth_source="synthetic",
        num_trajectories=3,
        duration_s=1.0,
        gyro_rate_hz=10,
        star_tracker_rate_hz=2,
        regime_code=int(RegimeCode.STATIONARY),
        event_covariance_multiplier=1.0,
    )
    generated = generate_unit_st_regime(cfg)
    q_c = _base_q(config, cfg)
    for trajectory_id in generated.dataset.truth.trajectory_id:
        assert_all_one_replay_exact(
            generated.dataset.events,
            int(trajectory_id),
            default_initial_state(),
            0.0,
            q_c,
        )
    output = {
        "status": "PASS",
        "command": "validate",
        "all_one_exact_trajectory_count": int(cfg.num_trajectories),
        "raw_sensor_stream_hash": generated.semantic_hashes.dataset_hash,
        "oracle_context_hash": generated.oracle_context.semantic_hash,
        "policy_boundary": "fixed replay signature contains no oracle argument",
    }
    _write_json(results_root / "validation.json", output)
    return output


def _debug_command(config: Mapping[str, Any], scenario: str, seed: int | None) -> dict[str, Any]:
    results_root, manifests_root = _artifact_roots(config)
    mapping = {
        "c1": (RegimeCode.STATIONARY, 1.0),
        "c2": (RegimeCode.C2_GYRO_PROCESS_STEP, 4.0),
        "c3": (RegimeCode.C3_STAR_TRACKER_RELIABILITY_STEP, 4.0),
    }
    if scenario not in mapping:
        raise ValueError("debug scenario must be c1, c2, or c3")
    regime, alpha = mapping[scenario]
    cfg = _regime_config(
        config,
        num_trajectories=int(config["debug"]["num_trajectories"]),
        duration_s=float(config["debug"]["duration_s"]),
        master_seed=int(seed if seed is not None else config["unit_st"]["master_seed"]),
        regime_code=int(regime),
        event_covariance_multiplier=alpha,
    )
    generated = generate_unit_st_regime(cfg)
    q_c = _base_q(config, cfg)
    policies: list[FixedPolicy | str] = [F_BASE, "ORACLE-QR", "WRONG-SIDE"]
    records: list[dict[str, Any]] = []
    for trajectory_id in generated.dataset.truth.trajectory_id:
        for policy in policies:
            replay = _run_policy(generated, int(trajectory_id), q_c, policy)
            records.append(
                evaluate_replay(
                    generated.dataset,
                    generated.oracle_context,
                    replay,
                    scenario_id=f"DEBUG-{scenario.upper()}",
                    policy_id=_policy_id(policy),
                    **_evaluation_options(config),
                )
            )
    sensor_dir = manifests_root / "debug" / scenario / "sensor"
    oracle_dir = manifests_root / "debug" / scenario / "oracle_simulation_only"
    if sensor_dir.exists() or oracle_dir.exists():
        raise FileExistsError("debug artifacts already exist; choose a new output root")
    save_unit_st_regime(sensor_dir, oracle_dir, generated)
    output = {
        "status": "PASS",
        "command": "debug",
        "scenario": scenario,
        "records": records,
        "raw_sensor_stream_hash": generated.semantic_hashes.dataset_hash,
        "oracle_context_hash": generated.oracle_context.semantic_hash,
    }
    _write_json(results_root / "debug" / f"{scenario}.json", output)
    return output


def _tuning_command(config: Mapping[str, Any]) -> dict[str, Any]:
    results_root, manifests_root = _artifact_roots(config)
    tuning = config["tuning"]
    cfg = _regime_config(
        config,
        num_trajectories=int(tuning["num_trajectories"]),
        duration_s=float(tuning["duration_s"]),
        master_seed=int(tuning["master_seed"]),
        regime_code=int(RegimeCode.STATIONARY),
        event_covariance_multiplier=1.0,
    )
    generated = generate_unit_st_regime(cfg)
    q_c = _base_q(config, cfg)
    selected, tuning_manifest = tune_fixed_policy(
        generated,
        q_c,
        candidates=tuple(float(item) for item in tuning["candidate_scales"]),
        **_evaluation_options(config),
    )
    pilot = config["pilot"]
    c5_base_cfg = _regime_config(
        config,
        num_trajectories=int(pilot["generated_trajectories"]),
        duration_s=float(pilot["duration_s"]),
        master_seed=int(pilot["master_seed"]),
        train_fraction=float(pilot["train_fraction"]),
        val_fraction=float(pilot["val_fraction"]),
        test_fraction=float(pilot["test_fraction"]),
        regime_code=int(RegimeCode.STATIONARY),
        event_covariance_multiplier=1.0,
    )
    c2_cfg = UnitSTRegimeConfig(
        **{
            **asdict(c5_base_cfg),
            "regime_code": int(RegimeCode.C2_GYRO_PROCESS_STEP),
            "event_covariance_multiplier": float(config["regimes"]["medium_alpha"]),
        }
    )
    c2_generated = generate_unit_st_regime(c2_cfg)
    selected_c5, c5_manifest = match_c5_innovation_rms(
        c2_generated,
        c5_base_cfg,
        _base_q(config, c5_base_cfg),
        candidate_alpha_R=tuple(float(item) for item in config["c5"]["candidate_alpha_R"]),
        tolerance_fraction=float(config["c5"]["rms_match_tolerance_fraction"]),
        **_evaluation_options(config),
    )
    output = {
        "status": "PASS",
        "command": "tune",
        "fixed_tuning": tuning_manifest,
        "c5_matching": c5_manifest,
        "frozen_policy": selected.deployable_artifact(),
        "frozen_c5_B_alpha_R": selected_c5,
        "test_split_accessed": False,
    }
    _write_json(results_root / "tuning.json", output)
    sensor_dir = manifests_root / "tuning" / "sensor"
    oracle_dir = manifests_root / "tuning" / "oracle_simulation_only"
    if not sensor_dir.exists() and not oracle_dir.exists():
        save_unit_st_regime(sensor_dir, oracle_dir, generated)
    elif not sensor_dir.is_dir() or not oracle_dir.is_dir():
        raise ValueError("existing tuning artifact pair is incomplete")
    return output


def _load_frozen_tuning(config: Mapping[str, Any]) -> tuple[FixedPolicy, float, dict[str, Any]]:
    results_root, _ = _artifact_roots(config)
    manifest = _read_json(results_root / "tuning.json")
    if manifest.get("status") != "PASS" or manifest.get("test_split_accessed") is not False:
        raise ValueError("tuning artifact is not a frozen train/validation-only result")
    raw = manifest["frozen_policy"]
    policy = FixedPolicy(
        "F-TUNED",
        qg_scale=float(raw["qg_scale"]),
        qb_scale=float(raw["qb_scale"]),
        r_scale=float(raw["r_scale"]),
    )
    return policy, float(manifest["frozen_c5_B_alpha_R"]), manifest


def _scenario_specs(config: Mapping[str, Any], c5_alpha_r: float) -> list[tuple[str, RegimeCode, float]]:
    mild = float(config["regimes"]["mild_alpha"])
    medium = float(config["regimes"]["medium_alpha"])
    severe = float(config["regimes"]["severe_alpha"])
    return [
        ("C1-STATIONARY", RegimeCode.STATIONARY, 1.0),
        ("C2-GYRO-MILD", RegimeCode.C2_GYRO_PROCESS_STEP, mild),
        ("C2-GYRO-MEDIUM", RegimeCode.C2_GYRO_PROCESS_STEP, medium),
        ("C2-GYRO-SEVERE", RegimeCode.C2_GYRO_PROCESS_STEP, severe),
        ("C3-ST-MILD", RegimeCode.C3_STAR_TRACKER_RELIABILITY_STEP, mild),
        ("C3-ST-MEDIUM", RegimeCode.C3_STAR_TRACKER_RELIABILITY_STEP, medium),
        ("C3-ST-SEVERE", RegimeCode.C3_STAR_TRACKER_RELIABILITY_STEP, severe),
        ("C5-A-GYRO-MEDIUM", RegimeCode.C2_GYRO_PROCESS_STEP, medium),
        ("C5-B-ST-MATCHED-RMS", RegimeCode.C3_STAR_TRACKER_RELIABILITY_STEP, c5_alpha_r),
    ]


def _scenario_policies(scenario: str, tuned: FixedPolicy) -> list[FixedPolicy | str]:
    if scenario == "C1-STATIONARY":
        return [
            F_BASE,
            tuned,
            F_MIS_Q_LOW,
            F_MIS_Q_HIGH,
            F_MIS_R_LOW,
            F_MIS_R_HIGH,
            "ORACLE-QR",
        ]
    return [F_BASE, tuned, "ORACLE-QR", "WRONG-SIDE"]


def _pilot_command(
    config: Mapping[str, Any], *, max_trajectories: int | None, resume: bool
) -> dict[str, Any]:
    results_root, manifests_root = _artifact_roots(config)
    tuned, c5_alpha_r, tuning_manifest = _load_frozen_tuning(config)
    pilot = config["pilot"]
    base_cfg = _regime_config(
        config,
        num_trajectories=int(pilot["generated_trajectories"]),
        duration_s=float(pilot["duration_s"]),
        master_seed=int(pilot["master_seed"]),
        train_fraction=float(pilot["train_fraction"]),
        val_fraction=float(pilot["val_fraction"]),
        test_fraction=float(pilot["test_fraction"]),
        regime_code=int(RegimeCode.STATIONARY),
        event_covariance_multiplier=1.0,
    )
    paired_base = generate_base_unit_st(base_cfg)
    base_generated = generate_unit_st_regime(base_cfg, base_generated=paired_base)
    q_c = _base_q(config, base_cfg)
    required_n = int(pilot["required_test_trajectories_per_condition"])
    test_ids = [int(item) for item in base_generated.trajectory_split.test_ids]
    if len(test_ids) < required_n:
        raise ValueError("pilot split has fewer test trajectories than the required paired N")
    selected_ids = test_ids[:required_n]
    if max_trajectories is not None:
        if int(max_trajectories) != max_trajectories or max_trajectories <= 0:
            raise ValueError("max_trajectories must be a positive integer")
        selected_ids = selected_ids[: int(max_trajectories)]
    records_root = results_root / "pilot" / "records"
    previous_manifest_path = results_root / "pilot" / "pilot_manifest.json"
    previous_hashes: dict[str, str] = {}
    if previous_manifest_path.is_file():
        previous_manifest = _read_json(previous_manifest_path)
        previous_hashes = {
            str(item["scenario_id"]): str(item["raw_sensor_stream_hash"])
            for item in previous_manifest.get("scenario_manifests", [])
        }
    specs = _scenario_specs(config, c5_alpha_r)
    scenario_manifests: list[dict[str, Any]] = []
    start = time.monotonic()
    for scenario, regime, alpha in specs:
        scenario_cfg = UnitSTRegimeConfig(
            **{
                **asdict(base_cfg),
                "regime_code": int(regime),
                "event_covariance_multiplier": alpha,
            }
        )
        generated = generate_unit_st_regime(scenario_cfg, base_generated=paired_base)
        artifact_root = manifests_root / "pilot" / scenario / generated.semantic_hashes.dataset_hash
        sensor_dir = artifact_root / "sensor"
        oracle_dir = artifact_root / "oracle_simulation_only"
        if not sensor_dir.exists() and not oracle_dir.exists():
            save_unit_st_regime(sensor_dir, oracle_dir, generated)
        elif not resume:
            raise FileExistsError("pilot artifacts exist; pass --resume to continue")
        scenario_manifests.append(
            {
                "experiment_id": EXPERIMENT_VERSION,
                "contract_version": POLICY_CONTRACT_VERSION,
                "scenario_id": scenario,
                "regime_code": int(regime),
                "covariance_multiplier": alpha,
                "truth_config_hash": generated.semantic_hashes.truth_hash,
                "scenario_config_hash": _sha256_json(asdict(scenario_cfg)),
                "raw_sensor_stream_hash": generated.semantic_hashes.dataset_hash,
                "oracle_context_hash": generated.oracle_context.semantic_hash,
                "event_window_fraction": [
                    scenario_cfg.event_start_fraction,
                    scenario_cfg.event_end_fraction,
                ],
                "master_seed": scenario_cfg.master_seed,
                "sensor_artifact_directory": str(sensor_dir),
                "oracle_artifact_directory": str(oracle_dir),
            }
        )
        for trajectory_id in selected_ids:
            for policy in _scenario_policies(scenario, tuned):
                policy_id = _policy_id(policy)
                record_path = records_root / scenario / policy_id / f"{trajectory_id}.json"
                if record_path.exists():
                    if resume:
                        existing = _read_json(record_path)
                        required_evaluation_fields = {
                            "event_innovation_norm_median_rad",
                            "event_innovation_norm_p95_rad",
                            "event_raw_gyro_measurement_rms_rad_s",
                            "event_raw_gyro_increment_rms_rad_s",
                        }
                        if required_evaluation_fields.issubset(existing) and previous_hashes.get(
                            scenario
                        ) == generated.semantic_hashes.dataset_hash:
                            existing["raw_sensor_stream_hash"] = generated.semantic_hashes.dataset_hash
                            existing["oracle_context_hash"] = generated.oracle_context.semantic_hash
                            _write_json(record_path, existing)
                            continue
                    else:
                        raise FileExistsError(f"pilot record already exists: {record_path}")
                replay = _run_policy(generated, trajectory_id, q_c, policy)
                record = evaluate_replay(
                    generated.dataset,
                    generated.oracle_context,
                    replay,
                    scenario_id=scenario,
                    policy_id=policy_id,
                    **_evaluation_options(config),
                )
                record["raw_sensor_stream_hash"] = generated.semantic_hashes.dataset_hash
                record["oracle_context_hash"] = generated.oracle_context.semantic_hash
                _write_json(record_path, record)
    records = [_read_json(path) for path in sorted(records_root.glob("*/*/*.json"))]
    completed_n = min(
        sum(
            1
            for item in records
            if item["scenario_id"] == scenario and item["policy_id"] == "F-BASE"
        )
        for scenario, _regime, _alpha in specs
    )
    output = {
        "experiment_id": EXPERIMENT_VERSION,
        "experiment_version": EXPERIMENT_VERSION,
        "contract_version": POLICY_CONTRACT_VERSION,
        "status": "COMPLETE" if completed_n >= required_n else "PARTIAL",
        "required_paired_N_per_condition": required_n,
        "completed_paired_N_per_condition": completed_n,
        "selected_test_trajectory_ids": selected_ids,
        "scenario_manifests": scenario_manifests,
        "frozen_tuning_manifest_hash": _sha256_json(tuning_manifest),
        "frozen_tuned_policy": tuned.deployable_artifact(),
        "frozen_c5_B_alpha_R": c5_alpha_r,
        "estimator_id": "phase1b-classical-mekf-replay-v1",
        "estimator_knowledge": {
            "F-BASE/F-TUNED/mismatch": "typed raw sensor events and fixed Q/R only",
            "ORACLE-QR/WRONG-SIDE": "current-event forward-only simulation sidecar",
            "truth": "evaluation-only exact join after estimation",
        },
        "policy_set": {
            "fixed": [
                F_BASE.deployable_artifact(),
                tuned.deployable_artifact(),
                F_MIS_Q_LOW.deployable_artifact(),
                F_MIS_Q_HIGH.deployable_artifact(),
                F_MIS_R_LOW.deployable_artifact(),
                F_MIS_R_HIGH.deployable_artifact(),
            ],
            "simulation_only": ["ORACLE-QR", "WRONG-SIDE"],
        },
        "initial_state": {
            "q_NB": default_initial_state().q_NB.tolist(),
            "b_g_rad_s": default_initial_state().b_g.tolist(),
            "P": default_initial_state().P.tolist(),
        },
        "base_Q_c": q_c.tolist(),
        "metric_contract": {
            "implementation": "bench.metrics.mekf canonical Gate C",
            "recovery_absolute_floor_rad": config["metrics"]["recovery_absolute_floor_rad"],
            "recovery_multiplier": config["metrics"]["recovery_multiplier"],
            "recovery_sustained_samples": config["metrics"]["recovery_sustained_samples"],
            "divergence_threshold_rad": config["metrics"]["divergence_threshold_rad"],
        },
        "statistics": dict(config["statistics"]),
        "configuration_hash": _sha256_json(config),
        "checkpoint_layout": "one canonical JSON per scenario/policy/trajectory",
        "restartable": True,
        "elapsed_s": time.monotonic() - start,
        "result_records_root": str(records_root),
        "record_count": len(records),
        **_runtime_and_source_provenance(),
    }
    _write_json(results_root / "pilot" / "pilot_manifest.json", output)
    return output


def _long_horizon_command(config: Mapping[str, Any], tuned: FixedPolicy) -> dict[str, Any]:
    results_root, manifests_root = _artifact_roots(config)
    raw = config["long_horizon"]
    cfg = _regime_config(
        config,
        num_trajectories=int(raw["num_trajectories"]),
        duration_s=float(raw["duration_s"]),
        gyro_rate_hz=int(raw["gyro_rate_hz"]),
        star_tracker_rate_hz=int(raw["star_tracker_rate_hz"]),
        master_seed=int(raw["master_seed"]),
        regime_code=int(RegimeCode.STATIONARY),
        event_covariance_multiplier=1.0,
    )
    generated = generate_unit_st_regime(cfg)
    q_c = _base_q(config, cfg)
    records: list[dict[str, Any]] = []
    for trajectory_id in generated.dataset.truth.trajectory_id:
        for policy in (F_BASE, tuned):
            replay = _run_policy(generated, int(trajectory_id), q_c, policy)
            records.append(
                evaluate_replay(
                    generated.dataset,
                    generated.oracle_context,
                    replay,
                    scenario_id="C1-LONG-HORIZON-STATIONARY",
                    policy_id=policy.policy_id,
                    **_evaluation_options(config),
                )
            )
    sensor_dir = manifests_root / "long_horizon" / "sensor"
    oracle_dir = manifests_root / "long_horizon" / "oracle_simulation_only"
    if not sensor_dir.exists() and not oracle_dir.exists():
        save_unit_st_regime(sensor_dir, oracle_dir, generated)
    output = {
        "status": "PASS",
        "num_trajectories": cfg.num_trajectories,
        "duration_s": cfg.duration_s,
        "records": records,
    }
    _write_json(results_root / "long_horizon.json", output)
    return output


def _report_command(config: Mapping[str, Any], *, include_long_horizon: bool) -> dict[str, Any]:
    results_root, _ = _artifact_roots(config)
    manifest = _read_json(results_root / "pilot" / "pilot_manifest.json")
    records = [_read_json(path) for path in sorted((results_root / "pilot" / "records").glob("*/*/*.json"))]
    summary = summarize_records(
        records,
        bootstrap_seed=int(config["statistics"]["bootstrap_seed"]),
        resamples=int(config["statistics"]["bootstrap_resamples"]),
    )
    output: dict[str, Any] = {
        "experiment_version": EXPERIMENT_VERSION,
        "pilot_status": manifest["status"],
        "required_paired_N_per_condition": manifest["required_paired_N_per_condition"],
        "completed_paired_N_per_condition": manifest["completed_paired_N_per_condition"],
        "summary": summary,
        "c5_AB_independent_test": summarize_c5_pair(
            records,
            bootstrap_seed=int(config["statistics"]["bootstrap_seed"]),
            resamples=int(config["statistics"]["bootstrap_resamples"]),
        ),
    }
    if include_long_horizon:
        tuned, _c5, _manifest = _load_frozen_tuning(config)
        output["long_horizon"] = _long_horizon_command(config, tuned)
    _write_json(results_root / "pilot_summary.json", output)
    return output


def pilot_workload(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return the locked workload estimate before any full pilot run."""

    pilot = config["pilot"]
    duration = float(pilot["duration_s"])
    gyro_rate = int(config["unit_st"]["gyro_rate_hz"])
    st_rate = int(config["unit_st"]["star_tracker_rate_hz"])
    events_per_trajectory = int(round(duration * (gyro_rate + st_rate)))
    scenarios = 9
    policy_runs = 7 + 8 * 4
    required_n = int(pilot["required_test_trajectories_per_condition"])
    return {
        "generated_trajectories": int(pilot["generated_trajectories"]),
        "required_test_trajectories_per_condition": required_n,
        "duration_s": duration,
        "gyro_rate_hz": gyro_rate,
        "star_tracker_rate_hz": st_rate,
        "events_per_trajectory": events_per_trajectory,
        "scenario_count": scenarios,
        "policy_trajectory_runs": policy_runs * required_n,
        "filter_event_steps": policy_runs * required_n * events_per_trajectory,
        "estimated_runtime_s": 180,
        "estimated_storage_bytes": 25_000_000,
        "estimate_basis": "single-host Basilisk truth reuse and 6x6 Gate A replay calibration",
        "checkpoint_unit": "scenario/policy/trajectory JSON record",
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("validate")
    debug = subparsers.add_parser("debug")
    debug.add_argument("--scenario", choices=("c1", "c2", "c3"), required=True)
    debug.add_argument("--seed", type=int)
    subparsers.add_parser("tune")
    pilot = subparsers.add_parser("pilot")
    pilot.add_argument("--max-trajectories", type=int)
    pilot.add_argument("--resume", action="store_true")
    report = subparsers.add_parser("report")
    report.add_argument("--include-long-horizon", action="store_true")
    subparsers.add_parser("workload")
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
    elif args.command == "tune":
        output = _tuning_command(config)
    elif args.command == "pilot":
        output = _pilot_command(
            config, max_trajectories=args.max_trajectories, resume=args.resume
        )
    elif args.command == "report":
        output = _report_command(config, include_long_horizon=args.include_long_horizon)
    elif args.command == "workload":
        output = pilot_workload(config)
        results_root.mkdir(parents=True, exist_ok=True)
        _write_json(results_root / "pilot_workload.json", output)
    else:
        raise RuntimeError("unreachable command")
    sys.stdout.write(json.dumps(output, sort_keys=True, indent=2, allow_nan=False) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
