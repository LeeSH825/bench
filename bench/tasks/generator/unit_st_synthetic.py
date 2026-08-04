"""Analytic, deterministic synthetic UNIT-ST trajectories for Phase 1A Gate B1."""

from __future__ import annotations

import hashlib
import json
import platform
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import scipy

from bench.estimators import mekf as mekf_core
from bench.estimators.mekf import quat_exp, quat_multiply, quat_normalize
from bench.tasks.generator import mekf_events as event_schema
from bench.tasks.generator.mekf_events import (
    CONVENTION_ID,
    GENERATOR_ID,
    SCHEMA_VERSION,
    SEED_POLICY_VERSION,
    MEKFDataset,
    MEKFEventTable,
    MEKFTruthTable,
    SemanticHashes,
    SensorCode,
    TrajectorySplit,
    compute_semantic_hashes,
    split_trajectory_ids,
)
from bench.utils.seeding import stable_int_seed_v0


@dataclass(frozen=True)
class UnitSTSyntheticConfig:
    """Complete frozen configuration for the analytic generator."""

    num_trajectories: int = 8
    duration_s: float = 2.0
    gyro_rate_hz: int = 20
    star_tracker_rate_hz: int = 5
    master_seed: int = 20260731
    initial_attitude_max_rad: float = 0.8
    angular_rate_max_rad_s: float = 0.25
    gyro_bias_max_rad_s: float = 0.015
    gyro_noise_std_rad_s: float = 8.0e-4
    star_tracker_noise_std_rad: float = 1.5e-3
    star_tracker_R_diagonal_rad2: tuple[float, float, float] = (
        2.25e-6,
        2.25e-6,
        2.25e-6,
    )
    randomize_star_tracker_sign: bool = True
    train_fraction: float = 0.6
    val_fraction: float = 0.2
    test_fraction: float = 0.2
    truth_seed_namespace: str = "truth"
    gyro_noise_seed_namespace: str = "gyro-noise"
    star_tracker_noise_seed_namespace: str = "star-tracker-noise"
    star_tracker_sign_seed_namespace: str = "star-tracker-sign"
    split_seed_namespace: str = "trajectory-split"

    def __post_init__(self) -> None:
        if int(self.num_trajectories) != self.num_trajectories or self.num_trajectories < 3:
            raise ValueError("num_trajectories must be an integer of at least three")
        if int(self.gyro_rate_hz) != self.gyro_rate_hz or self.gyro_rate_hz <= 0:
            raise ValueError("gyro_rate_hz must be a positive integer")
        if (
            int(self.star_tracker_rate_hz) != self.star_tracker_rate_hz
            or self.star_tracker_rate_hz <= 0
        ):
            raise ValueError("star_tracker_rate_hz must be a positive integer")
        if self.gyro_rate_hz % self.star_tracker_rate_hz != 0:
            raise ValueError("gyro_rate_hz must be an integer multiple of star_tracker_rate_hz")
        steps = float(self.duration_s) * int(self.gyro_rate_hz)
        if not np.isfinite(steps) or self.duration_s <= 0.0:
            raise ValueError("duration_s must be finite and positive")
        if not np.isclose(steps, round(steps), rtol=0.0, atol=1.0e-12):
            raise ValueError("duration_s * gyro_rate_hz must be an integer")
        finite_nonnegative = (
            self.initial_attitude_max_rad,
            self.angular_rate_max_rad_s,
            self.gyro_bias_max_rad_s,
            self.gyro_noise_std_rad_s,
            self.star_tracker_noise_std_rad,
        )
        if not np.all(np.isfinite(finite_nonnegative)) or np.any(
            np.asarray(finite_nonnegative) < 0.0
        ):
            raise ValueError("motion and noise magnitudes must be finite and nonnegative")
        covariance = np.asarray(self.star_tracker_R_diagonal_rad2, dtype=np.float64)
        if covariance.shape != (3,) or not np.all(np.isfinite(covariance)) or np.any(covariance <= 0.0):
            raise ValueError("star_tracker_R_diagonal_rad2 must contain three positive values")
        namespaces = (
            self.truth_seed_namespace,
            self.gyro_noise_seed_namespace,
            self.star_tracker_noise_seed_namespace,
            self.star_tracker_sign_seed_namespace,
            self.split_seed_namespace,
        )
        if any(not isinstance(item, str) or not item for item in namespaces):
            raise ValueError("seed namespaces must be nonempty strings")
        if len(set(namespaces)) != len(namespaces):
            raise ValueError("seed namespaces must be distinct")


@dataclass(frozen=True)
class GeneratedUnitST:
    dataset: MEKFDataset
    trajectory_split: TrajectorySplit
    manifest: dict[str, Any]
    semantic_hashes: SemanticHashes


def _stream_seed(config: UnitSTSyntheticConfig, namespace: str, trajectory_id: int | None = None) -> int:
    parts: list[Any] = [SEED_POLICY_VERSION, int(config.master_seed), namespace]
    if trajectory_id is not None:
        parts.append(int(trajectory_id))
    return stable_int_seed_v0(*parts)


def _trajectory_id(config: UnitSTSyntheticConfig, index: int) -> int:
    digest = hashlib.sha256(
        f"{GENERATOR_ID}|trajectory-id|{int(config.master_seed)}|{int(index)}".encode("ascii")
    ).digest()
    return int.from_bytes(digest[:8], "little", signed=False) & ((1 << 63) - 1)


def _sample_vector_in_ball(rng: np.random.Generator, maximum_norm: float) -> np.ndarray:
    if maximum_norm == 0.0:
        return np.zeros(3, dtype=np.float64)
    direction = rng.normal(size=3)
    direction /= np.linalg.norm(direction)
    radius = maximum_norm * float(rng.random()) ** (1.0 / 3.0)
    return np.asarray(direction * radius, dtype=np.float64)


def _source_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_safe_config(config: UnitSTSyntheticConfig) -> dict[str, Any]:
    return json.loads(
        json.dumps(asdict(config), sort_keys=True, separators=(",", ":"), allow_nan=False)
    )


def _split_hash(split: TrajectorySplit) -> str:
    payload = {
        "split_seed": int(split.split_seed),
        "test_ids": [int(item) for item in split.test_ids],
        "train_ids": [int(item) for item in split.train_ids],
        "val_ids": [int(item) for item in split.val_ids],
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def generate_unit_st(config: UnitSTSyntheticConfig | None = None) -> GeneratedUnitST:
    """Generate constant-rate/bias truth and zero-latency gyro/ST events."""

    cfg = config or UnitSTSyntheticConfig()
    if not isinstance(cfg, UnitSTSyntheticConfig):
        raise TypeError("config must be a UnitSTSyntheticConfig")
    trajectory_ids = np.array(
        [_trajectory_id(cfg, index) for index in range(cfg.num_trajectories)], dtype=np.int64
    )
    if np.unique(trajectory_ids).size != trajectory_ids.size:
        raise RuntimeError("deterministic trajectory ID collision")
    step_count = int(round(cfg.duration_s * cfg.gyro_rate_hz))
    truth_grid = np.arange(step_count + 1, dtype=np.float64) / float(cfg.gyro_rate_hz)
    star_stride = int(cfg.gyro_rate_hz // cfg.star_tracker_rate_hz)
    star_count_per_trajectory = step_count // star_stride

    truth_times: list[np.ndarray] = []
    truth_quaternions: list[np.ndarray] = []
    truth_biases: list[np.ndarray] = []
    truth_rates: list[np.ndarray] = []
    truth_offsets = [0]
    event_trajectory_ids: list[int] = []
    sensor_codes: list[int] = []
    measurement_times: list[float] = []
    event_orders: list[int] = []
    payload_indices: list[int] = []
    gyro_payloads: list[np.ndarray] = []
    star_quaternions: list[np.ndarray] = []
    star_covariances: list[np.ndarray] = []
    per_trajectory_seeds: dict[str, dict[str, int]] = {}
    star_covariance = np.diag(
        np.asarray(cfg.star_tracker_R_diagonal_rad2, dtype=np.float64)
    )

    for trajectory_id in trajectory_ids:
        tid = int(trajectory_id)
        seeds = {
            "truth": _stream_seed(cfg, cfg.truth_seed_namespace, tid),
            "gyro_noise": _stream_seed(cfg, cfg.gyro_noise_seed_namespace, tid),
            "star_tracker_noise": _stream_seed(cfg, cfg.star_tracker_noise_seed_namespace, tid),
            "star_tracker_sign": _stream_seed(cfg, cfg.star_tracker_sign_seed_namespace, tid),
        }
        per_trajectory_seeds[str(tid)] = seeds
        truth_rng = np.random.default_rng(seeds["truth"])
        gyro_rng = np.random.default_rng(seeds["gyro_noise"])
        star_rng = np.random.default_rng(seeds["star_tracker_noise"])
        sign_rng = np.random.default_rng(seeds["star_tracker_sign"])

        initial_rotation = _sample_vector_in_ball(truth_rng, cfg.initial_attitude_max_rad)
        q_initial = quat_exp(initial_rotation)
        omega_true = _sample_vector_in_ball(truth_rng, cfg.angular_rate_max_rad_s)
        gyro_bias = _sample_vector_in_ball(truth_rng, cfg.gyro_bias_max_rad_s)
        q_trajectory = np.stack(
            [
                quat_normalize(
                    quat_multiply(q_initial, quat_exp(omega_true * time_s)),
                    name="synthetic truth quaternion",
                )
                for time_s in truth_grid
            ],
            axis=0,
        )
        truth_times.append(truth_grid.copy())
        truth_quaternions.append(q_trajectory)
        truth_biases.append(np.repeat(gyro_bias[None, :], step_count + 1, axis=0))
        truth_rates.append(np.repeat(omega_true[None, :], step_count + 1, axis=0))
        truth_offsets.append(truth_offsets[-1] + step_count + 1)

        order = 0
        star_number = 0
        for step in range(1, step_count + 1):
            time_s = float(truth_grid[step])
            gyro_payload_index = len(gyro_payloads)
            gyro_payloads.append(
                omega_true
                + gyro_bias
                + gyro_rng.normal(0.0, cfg.gyro_noise_std_rad_s, size=3)
            )
            event_trajectory_ids.append(tid)
            sensor_codes.append(int(SensorCode.GYRO))
            measurement_times.append(time_s)
            event_orders.append(order)
            payload_indices.append(gyro_payload_index)
            order += 1

            if step % star_stride == 0:
                noise = star_rng.normal(0.0, cfg.star_tracker_noise_std_rad, size=3)
                measured = quat_normalize(
                    quat_multiply(q_trajectory[step], quat_exp(noise)),
                    name="synthetic star-tracker quaternion",
                )
                if cfg.randomize_star_tracker_sign and int(sign_rng.integers(0, 2)):
                    measured = -measured
                star_payload_index = len(star_quaternions)
                star_quaternions.append(measured)
                star_covariances.append(star_covariance.copy())
                event_trajectory_ids.append(tid)
                sensor_codes.append(int(SensorCode.STAR_TRACKER))
                measurement_times.append(time_s)
                event_orders.append(order)
                payload_indices.append(star_payload_index)
                order += 1
                star_number += 1
        if star_number != star_count_per_trajectory:
            raise RuntimeError("internal star-tracker cadence mismatch")

    event_count = len(event_trajectory_ids)
    events = MEKFEventTable(
        trajectory_id=np.asarray(event_trajectory_ids, dtype=np.int64),
        sensor_code=np.asarray(sensor_codes, dtype=np.int16),
        measurement_time_s=np.asarray(measurement_times, dtype=np.float64),
        arrival_time_s=np.asarray(measurement_times, dtype=np.float64),
        event_order=np.asarray(event_orders, dtype=np.int64),
        valid=np.ones(event_count, dtype=np.bool_),
        payload_index=np.asarray(payload_indices, dtype=np.int64),
        gyro_omega_rad_s=np.asarray(gyro_payloads, dtype=np.float64).reshape(-1, 3),
        star_tracker_q_NB=np.asarray(star_quaternions, dtype=np.float64).reshape(-1, 4),
        star_tracker_R_rad2=np.asarray(star_covariances, dtype=np.float64).reshape(-1, 3, 3),
    )
    truth = MEKFTruthTable(
        trajectory_id=trajectory_ids,
        truth_offsets=np.asarray(truth_offsets, dtype=np.int64),
        truth_time_s=np.concatenate(truth_times).astype(np.float64, copy=False),
        q_true_NB=np.concatenate(truth_quaternions, axis=0).astype(np.float64, copy=False),
        gyro_bias_rad_s=np.concatenate(truth_biases, axis=0).astype(np.float64, copy=False),
        omega_true_rad_s=np.concatenate(truth_rates, axis=0).astype(np.float64, copy=False),
    )
    dataset = MEKFDataset(events=events, truth=truth)
    split_seed = _stream_seed(cfg, cfg.split_seed_namespace)
    trajectory_split = split_trajectory_ids(
        trajectory_ids,
        split_seed=split_seed,
        train_fraction=cfg.train_fraction,
        val_fraction=cfg.val_fraction,
        test_fraction=cfg.test_fraction,
    )
    module_path = Path(__file__).resolve()
    event_path = Path(event_schema.__file__).resolve()
    core_path = Path(mekf_core.__file__).resolve()
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "generator_id": GENERATOR_ID,
        "seed_policy_version": SEED_POLICY_VERSION,
        "convention_id": CONVENTION_ID,
        "zero_latency": True,
        "same_timestamp_order": ["gyro", "star_tracker"],
        "event_sort_key": ["arrival_time_s", "event_order"],
        "generator_config": _json_safe_config(cfg),
        "master_seed": int(cfg.master_seed),
        "derived_seeds": {
            "stream_roots": {
                "truth": _stream_seed(cfg, cfg.truth_seed_namespace),
                "gyro_noise": _stream_seed(cfg, cfg.gyro_noise_seed_namespace),
                "star_tracker_noise": _stream_seed(cfg, cfg.star_tracker_noise_seed_namespace),
                "star_tracker_sign": _stream_seed(cfg, cfg.star_tracker_sign_seed_namespace),
                "trajectory_split": split_seed,
            },
            "per_trajectory": per_trajectory_seeds,
        },
        "trajectory_ids": [int(item) for item in trajectory_ids],
        "trajectory_split": {
            "split_seed": split_seed,
            "split_hash": _split_hash(trajectory_split),
            "train_ids": [int(item) for item in trajectory_split.train_ids],
            "val_ids": [int(item) for item in trajectory_split.val_ids],
            "test_ids": [int(item) for item in trajectory_split.test_ids],
        },
        "software_versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
        "source_fingerprints": {
            "bench/estimators/mekf.py": _source_sha256(core_path),
            "bench/tasks/generator/mekf_events.py": _source_sha256(event_path),
            "bench/tasks/generator/unit_st_synthetic.py": _source_sha256(module_path),
        },
    }
    hashes = compute_semantic_hashes(dataset, manifest)
    return GeneratedUnitST(
        dataset=dataset,
        trajectory_split=trajectory_split,
        manifest=manifest,
        semantic_hashes=hashes,
    )
