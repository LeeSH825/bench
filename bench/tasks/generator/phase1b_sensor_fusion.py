"""Deterministic Basilisk-backed Phase 1B sensor-fusion regimes.

Basilisk supplies only rigid-body attitude/rate truth. Magnetic/sun reference
profiles, validity, and every sensor output are project-owned parameterized
benchmark layers and are not flight-environment claims.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import platform
from dataclasses import asdict, dataclass
from enum import IntEnum
from pathlib import Path
from typing import Any

import numpy as np

from bench.estimators.mekf import (
    body_vector_prediction,
    quat_exp,
    quat_multiply,
    quat_normalize,
    sun_tangent_basis,
)
from bench.tasks.generator.mekf_events import MEKFEventTable
from bench.tasks.generator.mekf_fusion_events import (
    CONVENTION_ID,
    GENERATOR_ID,
    SAME_TIME_ORDER_ID,
    SCHEMA_VERSION,
    SEED_POLICY_VERSION,
    FusionDataset,
    FusionEventTable,
    FusionOracleSidecar,
    FusionSemanticHashes,
    FusionSensorCode,
    FusionTruthTable,
    compute_fusion_semantic_hashes,
)
from bench.tasks.generator.unit_st_regimes import (
    GeneratedUnitSTRegime,
    RegimeCode,
    UnitSTRegimeConfig,
    generate_unit_st_regime,
)
from bench.utils.seeding import stable_int_seed_v0


GENERATOR_CONTRACT_VERSION = "p1b-sensor-fusion-regimes-v1"
SENSOR_MODEL_VERSION = "parameterized-mag-sun-gyro-st-v1"
MAGNETIC_REFERENCE_PROFILE = "normalized-magnetic-reference-v1"
SUN_REFERENCE_PROFILE = "normalized-sun-reference-v1"
SUN_VALIDITY_PROFILE = "deterministic-fov-eclipselike-v1"


class FusionScenarioCode(IntEnum):
    MAIN_FUSION_STATIONARY = 1
    STRESS_MAG = 2
    C4_COMBINED = 3
    UNIT_ST_REDUCTION = 4


@dataclass(frozen=True)
class SensorFusionConfig:
    num_trajectories: int = 84
    duration_s: float = 30.0
    master_seed: int = 20260852
    gyro_rate_hz: int = 10
    magnetometer_rate_hz: int = 5
    sun_sensor_rate_hz: int = 2
    star_tracker_rate_hz: int = 1
    initial_attitude_max_rad: float = 0.5
    angular_rate_max_rad_s: float = 0.12
    gyro_bias_max_rad_s: float = 0.004
    gyro_noise_std_rad_s: float = 5.0e-4
    star_tracker_R_rad2: tuple[tuple[float, float, float], ...] = (
        (2.25e-6, 0.0, 0.0),
        (0.0, 2.25e-6, 0.0),
        (0.0, 0.0, 2.25e-6),
    )
    magnetometer_R: tuple[tuple[float, float, float], ...] = (
        (4.0e-6, 0.0, 0.0),
        (0.0, 4.0e-6, 0.0),
        (0.0, 0.0, 4.0e-6),
    )
    sun_tangent_R_rad2: tuple[tuple[float, float], ...] = (
        (9.0e-6, 0.0),
        (0.0, 9.0e-6),
    )
    bias_psd_rad2_s3: float = 1.0e-12
    scenario_code: int = int(FusionScenarioCode.MAIN_FUSION_STATIONARY)
    alpha_b: float = 100000.0
    alpha_R_mag: float = 16.0
    slow_window_start_fraction: float = 0.2
    slow_window_end_fraction: float = 0.8
    fast_window_start_fraction: float = 0.45
    fast_window_end_fraction: float = 0.6
    bias_random_walk_enabled: bool = True
    train_fraction: float = 0.2
    val_fraction: float = 0.2
    test_fraction: float = 0.6

    def __post_init__(self) -> None:
        if int(self.num_trajectories) != self.num_trajectories or self.num_trajectories < 3:
            raise ValueError("num_trajectories must be an integer of at least three")
        rates = (
            self.gyro_rate_hz,
            self.magnetometer_rate_hz,
            self.sun_sensor_rate_hz,
            self.star_tracker_rate_hz,
        )
        if any(int(rate) != rate or rate <= 0 for rate in rates):
            raise ValueError("all sensor rates must be positive integers")
        if any(self.gyro_rate_hz % rate for rate in rates[1:]):
            raise ValueError("all update rates must divide the gyro rate")
        steps = float(self.duration_s) * self.gyro_rate_hz
        if not np.isfinite(steps) or steps <= 0.0 or not np.isclose(
            steps, round(steps), rtol=0.0, atol=1.0e-12
        ):
            raise ValueError("duration times gyro rate must be a positive integer")
        if self.angular_rate_max_rad_s > 0.2:
            raise ValueError("Basilisk Gate B2 compatible rate must not exceed 0.2 rad/s")
        try:
            scenario = FusionScenarioCode(int(self.scenario_code))
        except ValueError as error:
            raise ValueError("unsupported fusion scenario code") from error
        if scenario == FusionScenarioCode.UNIT_ST_REDUCTION and self.bias_random_walk_enabled:
            raise ValueError("UNIT-ST exact reduction requires bias_random_walk_enabled=false")
        for name, dimension in (
            ("star_tracker_R_rad2", 3),
            ("magnetometer_R", 3),
            ("sun_tangent_R_rad2", 2),
        ):
            covariance = np.asarray(getattr(self, name), dtype=np.float64)
            if covariance.shape != (dimension, dimension) or not np.all(np.isfinite(covariance)):
                raise ValueError(f"{name} has the wrong finite shape")
            if not np.array_equal(covariance, covariance.T):
                raise ValueError(f"{name} must be exactly symmetric")
            try:
                np.linalg.cholesky(covariance)
            except np.linalg.LinAlgError as error:
                raise ValueError(f"{name} must be strictly SPD") from error
        scalars = np.asarray(
            (
                self.initial_attitude_max_rad,
                self.angular_rate_max_rad_s,
                self.gyro_bias_max_rad_s,
                self.gyro_noise_std_rad_s,
                self.bias_psd_rad2_s3,
            ),
            dtype=np.float64,
        )
        if not np.all(np.isfinite(scalars)) or np.any(scalars < 0.0):
            raise ValueError("motion/noise magnitudes must be finite and nonnegative")
        if not np.isfinite(self.alpha_b) or not np.isfinite(self.alpha_R_mag):
            raise ValueError("C4 scales must be finite")
        if self.alpha_b < 1.0 or self.alpha_R_mag < 1.0:
            raise ValueError("C4 scales must be at least one")
        if not (
            0.0 < self.slow_window_start_fraction < self.slow_window_end_fraction < 1.0
            and 0.0 < self.fast_window_start_fraction < self.fast_window_end_fraction < 1.0
        ):
            raise ValueError("C4 windows must be strict interior intervals")
        fractions = np.asarray(
            (self.train_fraction, self.val_fraction, self.test_fraction), dtype=np.float64
        )
        if np.any(fractions <= 0.0) or not math.isclose(
            float(np.sum(fractions)), 1.0, rel_tol=0.0, abs_tol=1.0e-12
        ):
            raise ValueError("split fractions must be positive and sum to one")

    @property
    def base_Q_g_rad2_s(self) -> float:
        return self.gyro_noise_std_rad_s**2 / float(self.gyro_rate_hz)


@dataclass(frozen=True)
class GeneratedSensorFusion:
    dataset: FusionDataset
    trajectory_split: Any
    sensor_manifest: dict[str, Any]
    semantic_hashes: FusionSemanticHashes
    oracle_context: FusionOracleSidecar
    base_unit_st: GeneratedUnitSTRegime


def _unit(value: np.ndarray) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64)
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm <= np.finfo(np.float64).tiny:
        raise ValueError("reference vector must be finite and nonzero")
    return vector / norm


def magnetic_reference_N(time_s: float, duration_s: float, phase: float = 0.0) -> np.ndarray:
    """Versioned representative-normalized inertial magnetic profile."""

    angle = 2.0 * math.pi * (0.13 * float(time_s) / float(duration_s) + phase)
    return _unit(np.array([0.48 + 0.08 * math.cos(angle), -0.18 + 0.06 * math.sin(angle), 0.86]))


def sun_reference_N(time_s: float, duration_s: float, phase: float = 0.0) -> np.ndarray:
    """Versioned deterministic inertial unit-sun profile."""

    angle = 2.0 * math.pi * (0.09 * float(time_s) / float(duration_s) + phase)
    return _unit(np.array([-0.25 + 0.04 * math.sin(angle), 0.91, 0.32 + 0.04 * math.cos(angle)]))


def sun_is_valid(time_s: float, duration_s: float, trajectory_index: int) -> bool:
    """Deterministic FOV/eclipselike benchmark mask; not an eclipse model."""

    phase = (float(time_s) / float(duration_s) + 0.13 * (trajectory_index % 5)) % 1.0
    return not (0.32 <= phase < 0.42)


def _stream_seed(master_seed: int, namespace: str, trajectory_id: int) -> int:
    return stable_int_seed_v0(
        GENERATOR_CONTRACT_VERSION, int(master_seed), namespace, int(trajectory_id)
    )


def _base_config(config: SensorFusionConfig) -> UnitSTRegimeConfig:
    return UnitSTRegimeConfig(
        truth_source="basilisk",
        num_trajectories=config.num_trajectories,
        duration_s=config.duration_s,
        gyro_rate_hz=config.gyro_rate_hz,
        star_tracker_rate_hz=config.star_tracker_rate_hz,
        master_seed=config.master_seed,
        initial_attitude_max_rad=config.initial_attitude_max_rad,
        angular_rate_max_rad_s=config.angular_rate_max_rad_s,
        gyro_bias_max_rad_s=config.gyro_bias_max_rad_s,
        gyro_noise_std_rad_s=config.gyro_noise_std_rad_s,
        star_tracker_R_rad2=config.star_tracker_R_rad2,
        randomize_star_tracker_sign=True,
        regime_code=int(RegimeCode.STATIONARY),
        event_covariance_multiplier=1.0,
        train_fraction=config.train_fraction,
        val_fraction=config.val_fraction,
        test_fraction=config.test_fraction,
    )


def _require_compatible_base(config: SensorFusionConfig, base: GeneratedUnitSTRegime) -> None:
    if not isinstance(base, GeneratedUnitSTRegime):
        raise TypeError("base_unit_st must be GeneratedUnitSTRegime")
    expected = _base_config(config)
    cadence = base.sensor_manifest.get("cadence")
    if not isinstance(cadence, dict):
        raise ValueError("base UNIT-ST manifest lacks cadence")
    checks = {
        "duration_s": expected.duration_s,
        "gyro_rate_hz": expected.gyro_rate_hz,
        "star_tracker_rate_hz": expected.star_tracker_rate_hz,
    }
    for name, value in checks.items():
        if cadence.get(name) != value:
            raise ValueError(f"base UNIT-ST is incompatible at {name}")
    if base.sensor_manifest.get("master_seed") != expected.master_seed:
        raise ValueError("base UNIT-ST is incompatible at master_seed")
    if len(base.sensor_manifest.get("trajectory_ids", [])) != expected.num_trajectories:
        raise ValueError("base UNIT-ST is incompatible at num_trajectories")


def _time_index(times: np.ndarray, time_s: float) -> int:
    matches = np.flatnonzero(times == np.float64(time_s))
    if matches.size != 1:
        raise RuntimeError("sensor timestamp must exactly select one truth sample")
    return int(matches[0])


def _base_payload_maps(table: MEKFEventTable, trajectory_id: int) -> tuple[dict[float, np.ndarray], dict[float, tuple[np.ndarray, np.ndarray]]]:
    gyro: dict[float, np.ndarray] = {}
    star: dict[float, tuple[np.ndarray, np.ndarray]] = {}
    rows = np.flatnonzero(table.trajectory_id == np.int64(trajectory_id))
    for row in rows:
        time_s = float(table.measurement_time_s[row])
        payload = int(table.payload_index[row])
        if int(table.sensor_code[row]) == 1:
            gyro[time_s] = table.gyro_omega_rad_s[payload]
        elif int(table.sensor_code[row]) == 2:
            star[time_s] = (
                table.star_tracker_q_NB[payload],
                table.star_tracker_R_rad2[payload],
            )
    return gyro, star


def _runtime() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": importlib.metadata.version("scipy"),
        "basilisk": importlib.metadata.version("bsk"),
    }


def _source_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def generate_sensor_fusion(
    config: SensorFusionConfig | None = None,
    *,
    base_unit_st: GeneratedUnitSTRegime | None = None,
) -> GeneratedSensorFusion:
    cfg = config or SensorFusionConfig()
    if not isinstance(cfg, SensorFusionConfig):
        raise TypeError("config must be SensorFusionConfig")
    base = base_unit_st or generate_unit_st_regime(_base_config(cfg))
    _require_compatible_base(cfg, base)
    scenario = FusionScenarioCode(int(cfg.scenario_code))
    include_mag = scenario != FusionScenarioCode.UNIT_ST_REDUCTION
    include_sun = scenario in {
        FusionScenarioCode.MAIN_FUSION_STATIONARY,
        FusionScenarioCode.C4_COMBINED,
    }
    include_st = scenario != FusionScenarioCode.STRESS_MAG
    c4 = scenario == FusionScenarioCode.C4_COMBINED

    truth = base.dataset.truth
    trajectory_ids = np.array(truth.trajectory_id, copy=True)
    truth_times: list[np.ndarray] = []
    truth_q: list[np.ndarray] = []
    truth_bias: list[np.ndarray] = []
    truth_omega: list[np.ndarray] = []
    truth_mag: list[np.ndarray] = []
    truth_sun: list[np.ndarray] = []
    offsets = [0]

    event_tid: list[int] = []
    event_code: list[int] = []
    event_time: list[float] = []
    event_order: list[int] = []
    event_valid: list[bool] = []
    event_payload: list[int] = []
    gyro_values: list[np.ndarray] = []
    st_q: list[np.ndarray] = []
    st_r: list[np.ndarray] = []
    mag_z: list[np.ndarray] = []
    mag_ref: list[np.ndarray] = []
    mag_r: list[np.ndarray] = []
    sun_z: list[np.ndarray] = []
    sun_ref: list[np.ndarray] = []
    sun_r: list[np.ndarray] = []
    oracle_tid: list[int] = []
    oracle_order: list[int] = []
    oracle_alpha_b: list[float] = []
    oracle_alpha_r: list[float] = []
    oracle_slow: list[bool] = []
    oracle_fast: list[bool] = []
    oracle_scenario: list[int] = []
    seed_ledger: dict[str, dict[str, int]] = {}
    separation_degrees: list[float] = []

    mag_cov = np.asarray(cfg.magnetometer_R, dtype=np.float64)
    mag_chol = np.linalg.cholesky(mag_cov)
    sun_cov = np.asarray(cfg.sun_tangent_R_rad2, dtype=np.float64)
    sun_chol = np.linalg.cholesky(sun_cov)
    gyro_stride_mag = cfg.gyro_rate_hz // cfg.magnetometer_rate_hz
    gyro_stride_sun = cfg.gyro_rate_hz // cfg.sun_sensor_rate_hz
    gyro_stride_st = cfg.gyro_rate_hz // cfg.star_tracker_rate_hz
    dt = 1.0 / float(cfg.gyro_rate_hz)

    for trajectory_index, trajectory_id_raw in enumerate(trajectory_ids):
        trajectory_id = int(trajectory_id_raw)
        start = int(truth.truth_offsets[trajectory_index])
        stop = int(truth.truth_offsets[trajectory_index + 1])
        times = np.array(truth.truth_time_s[start:stop], copy=True)
        q_values = np.array(truth.q_true_NB[start:stop], copy=True)
        omega_values = np.array(truth.omega_true_rad_s[start:stop], copy=True)
        initial_bias = np.array(truth.gyro_bias_rad_s[start], copy=True)
        seeds = {
            name: _stream_seed(cfg.master_seed, name, trajectory_id)
            for name in (
                "bias-rw-base",
                "bias-rw-c4-extra",
                "mag-base-noise",
                "mag-c4-extra-noise",
                "sun-tangent-noise",
            )
        }
        seed_ledger[str(trajectory_id)] = seeds
        bias_rng = np.random.default_rng(seeds["bias-rw-base"])
        bias_extra_rng = np.random.default_rng(seeds["bias-rw-c4-extra"])
        mag_rng = np.random.default_rng(seeds["mag-base-noise"])
        mag_extra_rng = np.random.default_rng(seeds["mag-c4-extra-noise"])
        sun_rng = np.random.default_rng(seeds["sun-tangent-noise"])

        biases = np.repeat(initial_bias[None, :], times.size, axis=0)
        if cfg.bias_random_walk_enabled:
            for index in range(1, times.size):
                current = float(times[index])
                slow = (
                    cfg.slow_window_start_fraction * cfg.duration_s
                    <= current
                    < cfg.slow_window_end_fraction * cfg.duration_s
                )
                increment = math.sqrt(cfg.bias_psd_rad2_s3 * dt) * bias_rng.normal(size=3)
                if c4 and slow:
                    increment += math.sqrt(
                        (cfg.alpha_b - 1.0) * cfg.bias_psd_rad2_s3 * dt
                    ) * bias_extra_rng.normal(size=3)
                biases[index] = biases[index - 1] + increment

        phase = 0.011 * (trajectory_index % 7)
        mag_true = np.stack(
            [magnetic_reference_N(item, cfg.duration_s, phase) for item in times]
        )
        sun_true = np.stack([sun_reference_N(item, cfg.duration_s, phase) for item in times])
        truth_times.append(times)
        truth_q.append(q_values)
        truth_bias.append(biases)
        truth_omega.append(omega_values)
        truth_mag.append(mag_true)
        truth_sun.append(sun_true)
        offsets.append(offsets[-1] + times.size)

        gyro_base, star_base = _base_payload_maps(base.dataset.events, trajectory_id)
        local_order = 0
        for gyro_index in range(1, times.size):
            time_s = float(times[gyro_index])
            slow = (
                c4
                and cfg.slow_window_start_fraction * cfg.duration_s
                <= time_s
                < cfg.slow_window_end_fraction * cfg.duration_s
            )
            fast = (
                c4
                and cfg.fast_window_start_fraction * cfg.duration_s
                <= time_s
                < cfg.fast_window_end_fraction * cfg.duration_s
            )

            def add_event(code: FusionSensorCode, valid: bool, payload: int) -> None:
                nonlocal local_order
                event_tid.append(trajectory_id)
                event_code.append(int(code))
                event_time.append(time_s)
                event_order.append(local_order)
                event_valid.append(valid)
                event_payload.append(payload)
                oracle_tid.append(trajectory_id)
                oracle_order.append(local_order)
                oracle_alpha_b.append(cfg.alpha_b if slow else 1.0)
                oracle_alpha_r.append(cfg.alpha_R_mag if fast else 1.0)
                oracle_slow.append(slow)
                oracle_fast.append(fast)
                oracle_scenario.append(int(scenario))
                local_order += 1

            base_gyro = np.asarray(gyro_base[time_s], dtype=np.float64)
            gyro_values.append(
                base_gyro + biases[gyro_index] - initial_bias
                if cfg.bias_random_walk_enabled
                else np.array(base_gyro, copy=True)
            )
            add_event(FusionSensorCode.GYRO, True, len(gyro_values) - 1)

            if include_mag and gyro_index % gyro_stride_mag == 0:
                reference = mag_true[gyro_index]
                prediction = body_vector_prediction(q_values[gyro_index], reference)
                noise = mag_chol @ mag_rng.normal(size=3)
                if fast:
                    noise += math.sqrt(cfg.alpha_R_mag - 1.0) * (
                        mag_chol @ mag_extra_rng.normal(size=3)
                    )
                mag_z.append(prediction + noise)
                mag_ref.append(reference)
                mag_r.append(mag_cov)
                add_event(FusionSensorCode.MAGNETOMETER, True, len(mag_z) - 1)

            if include_sun and gyro_index % gyro_stride_sun == 0:
                reference = sun_true[gyro_index]
                prediction = body_vector_prediction(q_values[gyro_index], reference)
                basis = sun_tangent_basis(prediction)
                tangent_noise = sun_chol @ sun_rng.normal(size=2)
                measurement = _unit(prediction + basis @ tangent_noise)
                valid = sun_is_valid(time_s, cfg.duration_s, trajectory_index)
                sun_z.append(measurement)
                sun_ref.append(reference)
                sun_r.append(sun_cov)
                add_event(FusionSensorCode.SUN_SENSOR, valid, len(sun_z) - 1)
                if valid:
                    dot = float(np.clip(reference @ mag_true[gyro_index], -1.0, 1.0))
                    separation_degrees.append(math.degrees(math.acos(dot)))

            if include_st and gyro_index % gyro_stride_st == 0:
                q_measurement, covariance = star_base[time_s]
                st_q.append(q_measurement)
                st_r.append(covariance)
                add_event(FusionSensorCode.STAR_TRACKER, True, len(st_q) - 1)

    event_table = FusionEventTable(
        trajectory_id=np.asarray(event_tid, dtype=np.int64),
        sensor_code=np.asarray(event_code, dtype=np.int16),
        measurement_time_s=np.asarray(event_time, dtype=np.float64),
        arrival_time_s=np.asarray(event_time, dtype=np.float64),
        event_order=np.asarray(event_order, dtype=np.int64),
        valid=np.asarray(event_valid, dtype=np.bool_),
        payload_index=np.asarray(event_payload, dtype=np.int64),
        gyro_omega_m_B_rad_s=np.asarray(gyro_values, dtype=np.float64).reshape(-1, 3),
        star_tracker_q_ST_NB=np.asarray(st_q, dtype=np.float64).reshape(-1, 4),
        star_tracker_R_ST_rad2=np.asarray(st_r, dtype=np.float64).reshape(-1, 3, 3),
        magnetometer_z_mag_B=np.asarray(mag_z, dtype=np.float64).reshape(-1, 3),
        magnetometer_r_mag_N_model=np.asarray(mag_ref, dtype=np.float64).reshape(-1, 3),
        magnetometer_R_mag=np.asarray(mag_r, dtype=np.float64).reshape(-1, 3, 3),
        sun_z_sun_B=np.asarray(sun_z, dtype=np.float64).reshape(-1, 3),
        sun_r_sun_N_model=np.asarray(sun_ref, dtype=np.float64).reshape(-1, 3),
        sun_R_sun_tangent_rad2=np.asarray(sun_r, dtype=np.float64).reshape(-1, 2, 2),
    )
    truth_table = FusionTruthTable(
        trajectory_id=trajectory_ids,
        truth_offsets=np.asarray(offsets, dtype=np.int64),
        truth_time_s=np.concatenate(truth_times),
        q_true_NB=np.concatenate(truth_q),
        gyro_bias_true_rad_s=np.concatenate(truth_bias),
        omega_true_B_rad_s=np.concatenate(truth_omega),
        r_mag_N_true=np.concatenate(truth_mag),
        r_sun_N_true=np.concatenate(truth_sun),
    )
    dataset = FusionDataset(event_table, truth_table)
    oracle = FusionOracleSidecar(
        trajectory_id=np.asarray(oracle_tid, dtype=np.int64),
        event_order=np.asarray(oracle_order, dtype=np.int64),
        alpha_b=np.asarray(oracle_alpha_b, dtype=np.float64),
        alpha_R_mag=np.asarray(oracle_alpha_r, dtype=np.float64),
        slow_window=np.asarray(oracle_slow, dtype=np.bool_),
        fast_window=np.asarray(oracle_fast, dtype=np.bool_),
        scenario_code=np.asarray(oracle_scenario, dtype=np.int8),
    )
    config_dict = json.loads(json.dumps(asdict(cfg), sort_keys=True, allow_nan=False))
    split = base.trajectory_split
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "generator_id": GENERATOR_ID,
        "convention_id": CONVENTION_ID,
        "seed_policy_version": SEED_POLICY_VERSION,
        "same_time_order_id": SAME_TIME_ORDER_ID,
        "generator_contract_version": GENERATOR_CONTRACT_VERSION,
        "sensor_model_version": SENSOR_MODEL_VERSION,
        "truth_source": "Basilisk spherical-inertia zero-torque rigid-body attitude/rate",
        "environment_claim": "none; deterministic normalized benchmark references",
        "reference_profiles": {
            "magnetic": MAGNETIC_REFERENCE_PROFILE,
            "sun": SUN_REFERENCE_PROFILE,
            "sun_validity": SUN_VALIDITY_PROFILE,
            "true_equals_model": True,
        },
        "generator_config": config_dict,
        "trajectory_ids": [int(item) for item in trajectory_ids],
        "split": {
            "train": [int(item) for item in split.train_ids],
            "val": [int(item) for item in split.val_ids],
            "test": [int(item) for item in split.test_ids],
            "split_seed": int(split.split_seed),
        },
        "derived_seeds": seed_ledger,
        "base_unit_st_dataset_hash": base.semantic_hashes.dataset_hash,
        "oracle_schema_version": "p1b-mekf-fusion-oracle-context-v1",
        "oracle_hash": oracle.semantic_hash,
        "sun_invalid_policy": "valid=false; nonzero unit payload; update skipped",
        "valid_reference_separation_deg": {
            "minimum": min(separation_degrees) if separation_degrees else None,
            "maximum": max(separation_degrees) if separation_degrees else None,
            "guard": [20.0, 160.0],
        },
        "runtime": _runtime(),
        "source_fingerprints": {
            "fusion_schema": _source_sha(Path(__file__).with_name("mekf_fusion_events.py")),
            "fusion_generator": _source_sha(Path(__file__)),
        },
    }
    if separation_degrees and (
        min(separation_degrees) < 20.0 or max(separation_degrees) > 160.0
    ):
        raise RuntimeError("MAIN-FUSION reference geometry guard failed")
    hashes = compute_fusion_semantic_hashes(dataset, manifest)
    return GeneratedSensorFusion(dataset, split, manifest, hashes, oracle, base)


def fusion_gyro_st_as_phase1a(table: FusionEventTable) -> MEKFEventTable:
    """Losslessly convert a gyro+ST fusion subset to the frozen Phase 1A type."""

    if np.any(~np.isin(table.sensor_code, np.asarray([1, 2], dtype=np.int16))):
        raise ValueError("fusion table must contain only gyro and star tracker")
    return MEKFEventTable(
        trajectory_id=np.array(table.trajectory_id, copy=True),
        sensor_code=np.array(table.sensor_code, copy=True),
        measurement_time_s=np.array(table.measurement_time_s, copy=True),
        arrival_time_s=np.array(table.arrival_time_s, copy=True),
        event_order=np.array(table.event_order, copy=True),
        valid=np.array(table.valid, copy=True),
        payload_index=np.array(table.payload_index, copy=True),
        gyro_omega_rad_s=np.array(table.gyro_omega_m_B_rad_s, copy=True),
        star_tracker_q_NB=np.array(table.star_tracker_q_ST_NB, copy=True),
        star_tracker_R_rad2=np.array(table.star_tracker_R_ST_rad2, copy=True),
    )


__all__ = [
    "FusionScenarioCode",
    "GeneratedSensorFusion",
    "GENERATOR_CONTRACT_VERSION",
    "MAGNETIC_REFERENCE_PROFILE",
    "SENSOR_MODEL_VERSION",
    "SUN_REFERENCE_PROFILE",
    "SUN_VALIDITY_PROFILE",
    "SensorFusionConfig",
    "fusion_gyro_st_as_phase1a",
    "generate_sensor_fusion",
    "magnetic_reference_N",
    "sun_is_valid",
    "sun_reference_N",
]
