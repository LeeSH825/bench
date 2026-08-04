"""Paired UNIT-ST uncertainty regimes for Phase 1B classical experiments.

The typed sensor artifact and the simulation-only oracle context are separate
artifacts.  A deployable fixed policy receives only :class:`MEKFEventTable`;
it cannot inspect regime labels, event boundaries, or covariance multipliers.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
from dataclasses import asdict, dataclass, fields
from enum import IntEnum
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from bench.estimators import mekf as mekf_core
from bench.estimators.mekf import quat_exp, quat_multiply, quat_normalize
from bench.tasks.generator import basilisk_unit_st, mekf_events, unit_st_synthetic
from bench.tasks.generator.basilisk_unit_st import (
    BasiliskUnitSTConfig,
    generate_basilisk_unit_st,
)
from bench.tasks.generator.mekf_events import (
    CONVENTION_ID,
    SCHEMA_VERSION,
    SEED_POLICY_VERSION,
    MEKFDataset,
    MEKFEventTable,
    SemanticHashes,
    SensorCode,
    TrajectorySplit,
    compute_semantic_hashes,
    load_event_dataset,
    save_event_dataset,
)
from bench.tasks.generator.unit_st_synthetic import UnitSTSyntheticConfig, generate_unit_st
from bench.utils.seeding import stable_int_seed_v0


REGIME_CONTRACT_VERSION = "p1b-unit-st-regimes-v1"
BASILISK_REGIME_GENERATOR_ID = "basilisk-unit-st-regimes-v1"
SYNTHETIC_REGIME_GENERATOR_ID = "synthetic-unit-st-regimes-v1"
ORACLE_CONTEXT_SCHEMA_VERSION = "p1b-unit-st-oracle-context-v1"


class RegimeCode(IntEnum):
    """Simulation-only regime labels.  These never enter a sensor artifact."""

    STATIONARY = 1
    C2_GYRO_PROCESS_STEP = 2
    C3_STAR_TRACKER_RELIABILITY_STEP = 3


class WindowCode(IntEnum):
    """Simulation-only event-window labels."""

    PRE_EVENT = 0
    EVENT = 1
    RECOVERY = 2


_GYRO = np.int16(SensorCode.GYRO)
_STAR_TRACKER = np.int16(SensorCode.STAR_TRACKER)
_CONTEXT_FIELDS = (
    "trajectory_id",
    "event_order",
    "alpha_g",
    "alpha_b",
    "alpha_R_ST",
    "event_window_id",
    "regime_code",
)


def _readonly_array(
    value: Any, *, dtype: np.dtype[Any] | type[np.generic], ndim: int, name: str
) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{name} must be a numpy.ndarray")
    expected = np.dtype(dtype)
    if value.dtype != expected:
        raise TypeError(f"{name} must have dtype {expected}, got {value.dtype}")
    if value.ndim != ndim:
        raise ValueError(f"{name} must have rank {ndim}, got {value.shape}")
    result = np.array(value, dtype=expected, order="C", copy=True)
    result.setflags(write=False)
    return result


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _source_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_array(array: np.ndarray) -> np.ndarray:
    if array.dtype.hasobject:
        raise TypeError("object arrays are forbidden")
    dtype = array.dtype.newbyteorder("<") if array.dtype.itemsize > 1 else array.dtype
    return np.ascontiguousarray(array.astype(dtype, copy=False))


def _context_hash(context: "OracleContextSidecar") -> str:
    digest = hashlib.sha256(b"p1b-unit-st-oracle-context-v1\0")
    for field in _CONTEXT_FIELDS:
        array = _canonical_array(getattr(context, field))
        metadata = _canonical_json_bytes(
            {"dtype": array.dtype.str, "field": field, "shape": list(array.shape)}
        )
        digest.update(len(metadata).to_bytes(8, "little"))
        digest.update(metadata)
        payload = array.tobytes(order="C")
        digest.update(len(payload).to_bytes(8, "little"))
        digest.update(payload)
    return digest.hexdigest()


@dataclass(frozen=True)
class UnitSTRegimeConfig:
    """Complete representative-normalized UNIT-ST regime configuration."""

    truth_source: str = "basilisk"
    num_trajectories: int = 15
    duration_s: float = 10.0
    gyro_rate_hz: int = 10
    star_tracker_rate_hz: int = 2
    master_seed: int = 20260802
    initial_attitude_max_rad: float = 0.5
    angular_rate_max_rad_s: float = 0.12
    gyro_bias_max_rad_s: float = 0.004
    gyro_noise_std_rad_s: float = 5.0e-4
    star_tracker_R_rad2: tuple[tuple[float, float, float], ...] = (
        (2.25e-6, 0.0, 0.0),
        (0.0, 2.25e-6, 0.0),
        (0.0, 0.0, 2.25e-6),
    )
    randomize_star_tracker_sign: bool = True
    regime_code: int = int(RegimeCode.STATIONARY)
    event_covariance_multiplier: float = 1.0
    event_start_fraction: float = 0.4
    event_end_fraction: float = 0.6
    train_fraction: float = 0.6
    val_fraction: float = 0.2
    test_fraction: float = 0.2
    base_gyro_noise_seed_namespace: str = "p1b-base-gyro-noise"
    event_gyro_noise_seed_namespace: str = "p1b-event-gyro-noise"
    base_star_tracker_noise_seed_namespace: str = "p1b-base-st-noise"
    event_star_tracker_noise_seed_namespace: str = "p1b-event-st-noise"
    star_tracker_sign_seed_namespace: str = "p1b-st-sign"

    def __post_init__(self) -> None:
        if self.truth_source not in {"basilisk", "synthetic"}:
            raise ValueError("truth_source must be 'basilisk' or 'synthetic'")
        if int(self.num_trajectories) != self.num_trajectories or self.num_trajectories < 3:
            raise ValueError("num_trajectories must be an integer of at least three")
        for name in ("gyro_rate_hz", "star_tracker_rate_hz"):
            value = getattr(self, name)
            if int(value) != value or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.gyro_rate_hz % self.star_tracker_rate_hz:
            raise ValueError("gyro_rate_hz must be an integer multiple of star_tracker_rate_hz")
        steps = self.duration_s * self.gyro_rate_hz
        if not np.isfinite(steps) or steps <= 0.0 or not np.isclose(
            steps, round(steps), rtol=0.0, atol=1.0e-12
        ):
            raise ValueError("duration_s * gyro_rate_hz must be a positive integer")
        if self.angular_rate_max_rad_s > 0.2:
            raise ValueError("Basilisk-compatible angular rate must not exceed 0.2 rad/s")
        nonnegative = np.asarray(
            (
                self.initial_attitude_max_rad,
                self.angular_rate_max_rad_s,
                self.gyro_bias_max_rad_s,
                self.gyro_noise_std_rad_s,
            ),
            dtype=np.float64,
        )
        if not np.all(np.isfinite(nonnegative)) or np.any(nonnegative < 0.0):
            raise ValueError("motion and noise magnitudes must be finite and nonnegative")
        covariance = np.asarray(self.star_tracker_R_rad2, dtype=np.float64)
        if covariance.shape != (3, 3) or not np.all(np.isfinite(covariance)):
            raise ValueError("star_tracker_R_rad2 must be a finite 3x3 matrix")
        if not np.array_equal(covariance, covariance.T):
            raise ValueError("star_tracker_R_rad2 must be exactly symmetric")
        try:
            np.linalg.cholesky(covariance)
        except np.linalg.LinAlgError as error:
            raise ValueError("star_tracker_R_rad2 must be strictly positive definite") from error
        try:
            regime = RegimeCode(int(self.regime_code))
        except ValueError as error:
            raise ValueError("regime_code is not a supported UNIT-ST regime") from error
        alpha = float(self.event_covariance_multiplier)
        if not np.isfinite(alpha) or alpha < 1.0:
            raise ValueError("event_covariance_multiplier must be finite and at least one")
        if regime == RegimeCode.STATIONARY and alpha != 1.0:
            raise ValueError("stationary regime requires an all-one covariance multiplier")
        if not 0.0 < self.event_start_fraction < self.event_end_fraction < 1.0:
            raise ValueError("event fractions must satisfy 0 < start < end < 1")
        fractions = np.asarray(
            (self.train_fraction, self.val_fraction, self.test_fraction), dtype=np.float64
        )
        if np.any(fractions <= 0.0) or not math.isclose(
            float(np.sum(fractions)), 1.0, rel_tol=0.0, abs_tol=1.0e-12
        ):
            raise ValueError("split fractions must be positive and sum to one")
        namespaces = (
            self.base_gyro_noise_seed_namespace,
            self.event_gyro_noise_seed_namespace,
            self.base_star_tracker_noise_seed_namespace,
            self.event_star_tracker_noise_seed_namespace,
            self.star_tracker_sign_seed_namespace,
        )
        if any(not isinstance(item, str) or not item for item in namespaces):
            raise ValueError("seed namespaces must be nonempty strings")
        if len(set(namespaces)) != len(namespaces):
            raise ValueError("seed namespaces must be distinct")

    @property
    def generator_id(self) -> str:
        if self.truth_source == "basilisk":
            return BASILISK_REGIME_GENERATOR_ID
        return SYNTHETIC_REGIME_GENERATOR_ID

    @property
    def base_Q_g_rad2_s(self) -> float:
        """Continuous gyro white-noise PSD matching the sampled base variance."""

        return self.gyro_noise_std_rad_s**2 / float(self.gyro_rate_hz)


@dataclass(frozen=True)
class OracleContextSidecar:
    """Read-only simulation truth inaccessible to deployable replay policies."""

    trajectory_id: np.ndarray
    event_order: np.ndarray
    alpha_g: np.ndarray
    alpha_b: np.ndarray
    alpha_R_ST: np.ndarray
    event_window_id: np.ndarray
    regime_code: np.ndarray

    def __post_init__(self) -> None:
        specifications = {
            "trajectory_id": (np.int64, 1),
            "event_order": (np.int64, 1),
            "alpha_g": (np.float64, 1),
            "alpha_b": (np.float64, 1),
            "alpha_R_ST": (np.float64, 1),
            "event_window_id": (np.int8, 1),
            "regime_code": (np.int8, 1),
        }
        for name, (dtype, ndim) in specifications.items():
            object.__setattr__(
                self,
                name,
                _readonly_array(getattr(self, name), dtype=dtype, ndim=ndim, name=name),
            )
        count = self.trajectory_id.size
        if any(getattr(self, name).size != count for name in _CONTEXT_FIELDS):
            raise ValueError("oracle context fields must have identical length")
        if count == 0:
            raise ValueError("oracle context must be nonempty")
        scales = np.column_stack((self.alpha_g, self.alpha_b, self.alpha_R_ST))
        if not np.all(np.isfinite(scales)) or np.any(scales < 1.0):
            raise ValueError("oracle covariance multipliers must be finite and at least one")
        valid_windows = np.asarray([int(item) for item in WindowCode], dtype=np.int8)
        valid_regimes = np.asarray([int(item) for item in RegimeCode], dtype=np.int8)
        if not np.all(np.isin(self.event_window_id, valid_windows)):
            raise ValueError("oracle context contains an invalid event-window code")
        if not np.all(np.isin(self.regime_code, valid_regimes)):
            raise ValueError("oracle context contains an invalid regime code")
        for trajectory_id in np.unique(self.trajectory_id):
            rows = np.flatnonzero(self.trajectory_id == trajectory_id)
            if not np.array_equal(self.event_order[rows], np.arange(rows.size, dtype=np.int64)):
                raise ValueError("oracle event_order must be contiguous per trajectory")

    @property
    def semantic_hash(self) -> str:
        return _context_hash(self)

    def cursor(self, trajectory_id: int) -> "CurrentOracleCursor":
        return CurrentOracleCursor(self, trajectory_id)


class CurrentOracleCursor:
    """Forward-only oracle view; future scales have no public read method."""

    __slots__ = ("_alpha_b", "_alpha_g", "_alpha_r", "_next_order")

    def __init__(self, context: OracleContextSidecar, trajectory_id: int) -> None:
        if not isinstance(context, OracleContextSidecar):
            raise TypeError("context must be an OracleContextSidecar")
        rows = np.flatnonzero(context.trajectory_id == np.int64(trajectory_id))
        if rows.size == 0:
            raise ValueError("trajectory_id is absent from oracle context")
        self._alpha_g = np.array(context.alpha_g[rows], copy=True)
        self._alpha_b = np.array(context.alpha_b[rows], copy=True)
        self._alpha_r = np.array(context.alpha_R_ST[rows], copy=True)
        self._next_order = 0

    def consume(self, event_order: int) -> tuple[float, float, float]:
        """Return only the current event's scales and advance exactly once."""

        order = int(event_order)
        if order != self._next_order or order >= self._alpha_g.size:
            raise ValueError("oracle cursor must be consumed once in strict event order")
        result = (
            float(self._alpha_g[order]),
            float(self._alpha_b[order]),
            float(self._alpha_r[order]),
        )
        self._next_order += 1
        return result


@dataclass(frozen=True)
class GeneratedUnitSTRegime:
    dataset: MEKFDataset
    trajectory_split: TrajectorySplit
    sensor_manifest: dict[str, Any]
    semantic_hashes: SemanticHashes
    oracle_context: OracleContextSidecar
    experiment_manifest: dict[str, Any]


def _stream_seed(config: UnitSTRegimeConfig, namespace: str, trajectory_id: int) -> int:
    return stable_int_seed_v0(
        REGIME_CONTRACT_VERSION, int(config.master_seed), namespace, int(trajectory_id)
    )


def covariance_increment_scale(alpha: float) -> float:
    """Standard-deviation scale of independent noise added to a base draw."""

    value = float(alpha)
    if not np.isfinite(value) or value < 1.0:
        raise ValueError("alpha must be finite and at least one")
    return math.sqrt(value - 1.0)


def _base_generation(config: UnitSTRegimeConfig) -> Any:
    common = {
        "num_trajectories": config.num_trajectories,
        "duration_s": config.duration_s,
        "gyro_rate_hz": config.gyro_rate_hz,
        "star_tracker_rate_hz": config.star_tracker_rate_hz,
        "master_seed": config.master_seed,
        "initial_attitude_max_rad": config.initial_attitude_max_rad,
        "angular_rate_max_rad_s": config.angular_rate_max_rad_s,
        "gyro_bias_max_rad_s": config.gyro_bias_max_rad_s,
        "randomize_star_tracker_sign": False,
        "train_fraction": config.train_fraction,
        "val_fraction": config.val_fraction,
        "test_fraction": config.test_fraction,
    }
    if config.truth_source == "basilisk":
        return generate_basilisk_unit_st(
            BasiliskUnitSTConfig(
                **common,
                gyro_noise_std_rad_s=0.0,
                star_tracker_R_rad2=config.star_tracker_R_rad2,
                star_tracker_noise_scale=0.0,
            )
        )
    diagonal = np.diag(np.asarray(config.star_tracker_R_rad2, dtype=np.float64))
    return generate_unit_st(
        UnitSTSyntheticConfig(
            **common,
            gyro_noise_std_rad_s=0.0,
            star_tracker_noise_std_rad=0.0,
            star_tracker_R_diagonal_rad2=tuple(float(item) for item in diagonal),
        )
    )


def generate_base_unit_st(config: UnitSTRegimeConfig) -> Any:
    """Generate the common frozen Phase 1A truth realization for paired regimes."""

    if not isinstance(config, UnitSTRegimeConfig):
        raise TypeError("config must be a UnitSTRegimeConfig")
    return _base_generation(config)


def _validate_compatible_base(config: UnitSTRegimeConfig, generated: Any) -> None:
    if not hasattr(generated, "dataset") or not hasattr(generated, "trajectory_split"):
        raise TypeError("base_generated must be a frozen Phase 1A generated UNIT-ST object")
    manifest = generated.manifest
    expected = (
        basilisk_unit_st.GENERATOR_ID
        if config.truth_source == "basilisk"
        else unit_st_synthetic.GENERATOR_ID
    )
    if manifest.get("generator_id") != expected:
        raise ValueError("base generation truth source is incompatible with regime config")
    source = manifest.get("generator_config", {})
    checks = {
        "num_trajectories": config.num_trajectories,
        "duration_s": config.duration_s,
        "gyro_rate_hz": config.gyro_rate_hz,
        "star_tracker_rate_hz": config.star_tracker_rate_hz,
        "master_seed": config.master_seed,
    }
    if any(source.get(name) != value for name, value in checks.items()):
        raise ValueError("base generation does not match the regime truth/cadence identity")


def _truth_sample(dataset: MEKFDataset, trajectory_id: int, time_s: float) -> tuple[np.ndarray, ...]:
    truth_index = int(np.flatnonzero(dataset.truth.trajectory_id == trajectory_id)[0])
    start = int(dataset.truth.truth_offsets[truth_index])
    stop = int(dataset.truth.truth_offsets[truth_index + 1])
    local_times = dataset.truth.truth_time_s[start:stop]
    local = int(np.searchsorted(local_times, time_s))
    if local >= local_times.size or float(local_times[local]) != time_s:
        raise RuntimeError("event/truth timestamps do not join exactly")
    row = start + local
    return (
        dataset.truth.q_true_NB[row],
        dataset.truth.gyro_bias_rad_s[row],
        dataset.truth.omega_true_rad_s[row],
    )


def _sensor_manifest(config: UnitSTRegimeConfig, base_generated: Any) -> dict[str, Any]:
    """Build a sensor-only manifest with no regime timing, label, or oracle scale."""

    module_path = Path(__file__).resolve()
    base_module = (
        Path(basilisk_unit_st.__file__).resolve()
        if config.truth_source == "basilisk"
        else Path(unit_st_synthetic.__file__).resolve()
    )
    split = base_generated.trajectory_split
    return {
        "schema_version": SCHEMA_VERSION,
        "generator_id": config.generator_id,
        "generator_version": REGIME_CONTRACT_VERSION,
        "seed_policy_version": SEED_POLICY_VERSION,
        "convention_id": CONVENTION_ID,
        "zero_latency": True,
        "same_timestamp_order": ["gyro", "star_tracker"],
        "event_sort_key": ["arrival_time_s", "event_order"],
        "truth_source": config.truth_source,
        "sensor_contract": "typed gyro/quaternion-star-tracker; hidden regimes external",
        "representativeness": "representative-normalized UNIT-ST; no flight claim",
        "base_sensor_parameters": {
            "gyro_noise_std_rad_s": config.gyro_noise_std_rad_s,
            "star_tracker_R_rad2": [list(row) for row in config.star_tracker_R_rad2],
            "randomize_star_tracker_sign": config.randomize_star_tracker_sign,
        },
        "cadence": {
            "duration_s": config.duration_s,
            "gyro_rate_hz": config.gyro_rate_hz,
            "star_tracker_rate_hz": config.star_tracker_rate_hz,
        },
        "master_seed": int(config.master_seed),
        "trajectory_ids": [int(item) for item in base_generated.dataset.truth.trajectory_id],
        "trajectory_split": {
            "split_seed": int(split.split_seed),
            "train_ids": [int(item) for item in split.train_ids],
            "val_ids": [int(item) for item in split.val_ids],
            "test_ids": [int(item) for item in split.test_ids],
        },
        "software_versions": {"python": platform.python_version(), "numpy": np.__version__},
        "source_fingerprints": {
            "bench/estimators/mekf.py": _source_sha256(Path(mekf_core.__file__).resolve()),
            "bench/tasks/generator/mekf_events.py": _source_sha256(
                Path(mekf_events.__file__).resolve()
            ),
            str(base_module.relative_to(module_path.parents[3])): _source_sha256(base_module),
            "bench/tasks/generator/unit_st_regimes.py": _source_sha256(module_path),
        },
    }


def generate_unit_st_regime(
    config: UnitSTRegimeConfig | None = None,
    *,
    base_generated: Any | None = None,
) -> GeneratedUnitSTRegime:
    """Generate paired UNIT-ST sensors plus a separate oracle-only sidecar."""

    cfg = config or UnitSTRegimeConfig()
    if not isinstance(cfg, UnitSTRegimeConfig):
        raise TypeError("config must be a UnitSTRegimeConfig")
    base = _base_generation(cfg) if base_generated is None else base_generated
    _validate_compatible_base(cfg, base)
    base_dataset = base.dataset
    regime = RegimeCode(int(cfg.regime_code))
    alpha = float(cfg.event_covariance_multiplier)
    covariance = np.asarray(cfg.star_tracker_R_rad2, dtype=np.float64)
    star_cholesky = np.linalg.cholesky(covariance)
    increment = covariance_increment_scale(alpha)

    gyro_payloads: list[np.ndarray] = []
    star_payloads: list[np.ndarray] = []
    star_covariances: list[np.ndarray] = []
    context_trajectory: list[int] = []
    context_order: list[int] = []
    alpha_g: list[float] = []
    alpha_b: list[float] = []
    alpha_r: list[float] = []
    window_codes: list[int] = []
    regime_codes: list[int] = []
    seeds: dict[str, dict[str, int]] = {}

    for trajectory_id in base_dataset.truth.trajectory_id:
        tid = int(trajectory_id)
        derived = {
            "base_gyro": _stream_seed(cfg, cfg.base_gyro_noise_seed_namespace, tid),
            "event_gyro": _stream_seed(cfg, cfg.event_gyro_noise_seed_namespace, tid),
            "base_star_tracker": _stream_seed(
                cfg, cfg.base_star_tracker_noise_seed_namespace, tid
            ),
            "event_star_tracker": _stream_seed(
                cfg, cfg.event_star_tracker_noise_seed_namespace, tid
            ),
            "star_tracker_sign": _stream_seed(
                cfg, cfg.star_tracker_sign_seed_namespace, tid
            ),
        }
        seeds[str(tid)] = derived
        base_gyro_rng = np.random.default_rng(derived["base_gyro"])
        event_gyro_rng = np.random.default_rng(derived["event_gyro"])
        base_st_rng = np.random.default_rng(derived["base_star_tracker"])
        event_st_rng = np.random.default_rng(derived["event_star_tracker"])
        sign_rng = np.random.default_rng(derived["star_tracker_sign"])
        rows = np.flatnonzero(base_dataset.events.trajectory_id == trajectory_id)
        start_s = cfg.duration_s * cfg.event_start_fraction
        end_s = cfg.duration_s * cfg.event_end_fraction
        for row in rows:
            time_s = float(base_dataset.events.measurement_time_s[row])
            in_event = start_s <= time_s < end_s
            window = (
                WindowCode.PRE_EVENT
                if time_s < start_s
                else WindowCode.EVENT
                if in_event
                else WindowCode.RECOVERY
            )
            current_alpha_g = alpha if in_event and regime == RegimeCode.C2_GYRO_PROCESS_STEP else 1.0
            current_alpha_r = (
                alpha
                if in_event and regime == RegimeCode.C3_STAR_TRACKER_RELIABILITY_STEP
                else 1.0
            )
            q_true, bias_true, omega_true = _truth_sample(base_dataset, tid, time_s)
            code = base_dataset.events.sensor_code[row]
            if code == _GYRO:
                base_noise = cfg.gyro_noise_std_rad_s * base_gyro_rng.normal(size=3)
                event_noise = cfg.gyro_noise_std_rad_s * event_gyro_rng.normal(size=3)
                total_noise = base_noise
                if in_event and regime == RegimeCode.C2_GYRO_PROCESS_STEP:
                    total_noise = total_noise + increment * event_noise
                gyro_payloads.append(omega_true + bias_true + total_noise)
            elif code == _STAR_TRACKER:
                base_noise = star_cholesky @ base_st_rng.normal(size=3)
                event_noise = star_cholesky @ event_st_rng.normal(size=3)
                total_noise = base_noise
                if in_event and regime == RegimeCode.C3_STAR_TRACKER_RELIABILITY_STEP:
                    total_noise = total_noise + increment * event_noise
                measured = quat_normalize(
                    quat_multiply(q_true, quat_exp(total_noise)),
                    name="Phase 1B UNIT-ST measurement",
                )
                if cfg.randomize_star_tracker_sign and int(sign_rng.integers(0, 2)):
                    measured = -measured
                star_payloads.append(measured)
                star_covariances.append(covariance.copy())
            else:
                raise RuntimeError("unexpected sensor code in validated base dataset")
            context_trajectory.append(tid)
            context_order.append(int(base_dataset.events.event_order[row]))
            alpha_g.append(current_alpha_g)
            alpha_b.append(1.0)
            alpha_r.append(current_alpha_r)
            window_codes.append(int(window))
            regime_codes.append(int(regime))

    events = MEKFEventTable(
        trajectory_id=base_dataset.events.trajectory_id,
        sensor_code=base_dataset.events.sensor_code,
        measurement_time_s=base_dataset.events.measurement_time_s,
        arrival_time_s=base_dataset.events.arrival_time_s,
        event_order=base_dataset.events.event_order,
        valid=base_dataset.events.valid,
        payload_index=base_dataset.events.payload_index,
        gyro_omega_rad_s=np.asarray(gyro_payloads, dtype=np.float64).reshape(-1, 3),
        star_tracker_q_NB=np.asarray(star_payloads, dtype=np.float64).reshape(-1, 4),
        star_tracker_R_rad2=np.asarray(star_covariances, dtype=np.float64).reshape(-1, 3, 3),
    )
    dataset = MEKFDataset(events=events, truth=base_dataset.truth)
    context = OracleContextSidecar(
        trajectory_id=np.asarray(context_trajectory, dtype=np.int64),
        event_order=np.asarray(context_order, dtype=np.int64),
        alpha_g=np.asarray(alpha_g, dtype=np.float64),
        alpha_b=np.asarray(alpha_b, dtype=np.float64),
        alpha_R_ST=np.asarray(alpha_r, dtype=np.float64),
        event_window_id=np.asarray(window_codes, dtype=np.int8),
        regime_code=np.asarray(regime_codes, dtype=np.int8),
    )
    sensor_manifest = _sensor_manifest(cfg, base)
    hashes = compute_semantic_hashes(dataset, sensor_manifest)
    experiment_manifest = {
        "experiment_contract_version": REGIME_CONTRACT_VERSION,
        "oracle_context_schema_version": ORACLE_CONTEXT_SCHEMA_VERSION,
        "sensor_generator_id": cfg.generator_id,
        "regime_config": asdict(cfg),
        "raw_sensor_stream_hash": hashes.dataset_hash,
        "oracle_context_hash": context.semantic_hash,
        "base_generator_id": base.manifest["generator_id"],
        "base_truth_hash": base.semantic_hashes.truth_hash,
        "pairing_identity": {
            "master_seed": int(cfg.master_seed),
            "truth_hash": hashes.truth_hash,
            "seed_streams": seeds,
        },
        "trajectory_split": sensor_manifest["trajectory_split"],
        "policy_knowledge_boundary": {
            "fixed_and_tuned": "typed sensor event table only",
            "oracle_and_wrong_side": "forward-only current-event oracle cursor",
            "evaluation": "truth and full sidecar only after estimation",
        },
    }
    return GeneratedUnitSTRegime(
        dataset=dataset,
        trajectory_split=base.trajectory_split,
        sensor_manifest=sensor_manifest,
        semantic_hashes=hashes,
        oracle_context=context,
        experiment_manifest=experiment_manifest,
    )


def save_oracle_context(
    directory: os.PathLike[str] | str,
    context: OracleContextSidecar,
    experiment_manifest: Mapping[str, Any],
) -> None:
    """Write the two-file oracle sidecar outside the sensor artifact directory."""

    if not isinstance(context, OracleContextSidecar):
        raise TypeError("context must be an OracleContextSidecar")
    target = Path(directory)
    if target.exists():
        if not target.is_dir() or any(target.iterdir()):
            raise FileExistsError(f"oracle target must be a new empty directory: {target}")
    else:
        target.mkdir(parents=True, exist_ok=False)
    manifest = dict(experiment_manifest)
    if manifest.get("oracle_context_schema_version") != ORACLE_CONTEXT_SCHEMA_VERSION:
        raise ValueError("oracle manifest schema version mismatch")
    if manifest.get("oracle_context_hash") != context.semantic_hash:
        raise ValueError("oracle manifest/context hash mismatch")
    (target / "experiment_manifest.json").write_bytes(_canonical_json_bytes(manifest))
    np.savez(target / "oracle_context.npz", **{name: getattr(context, name) for name in _CONTEXT_FIELDS})


def load_oracle_context(
    directory: os.PathLike[str] | str,
    *,
    expected_raw_sensor_stream_hash: str,
) -> tuple[OracleContextSidecar, dict[str, Any]]:
    """Strictly load and semantically verify a separate oracle sidecar."""

    source = Path(directory)
    expected_files = {"experiment_manifest.json", "oracle_context.npz"}
    if not source.is_dir() or {item.name for item in source.iterdir()} != expected_files:
        raise ValueError("oracle directory must contain exactly two sidecar files")
    raw = (source / "experiment_manifest.json").read_bytes()
    try:
        manifest = json.loads(raw.decode("ascii"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("oracle experiment manifest is invalid") from error
    if not isinstance(manifest, dict) or _canonical_json_bytes(manifest) != raw:
        raise ValueError("oracle experiment manifest must be canonical JSON")
    if manifest.get("oracle_context_schema_version") != ORACLE_CONTEXT_SCHEMA_VERSION:
        raise ValueError("oracle context schema version mismatch")
    if manifest.get("raw_sensor_stream_hash") != expected_raw_sensor_stream_hash:
        raise ValueError("oracle sidecar is paired with a different raw sensor stream")
    try:
        with np.load(source / "oracle_context.npz", allow_pickle=False) as archive:
            if set(archive.files) != set(_CONTEXT_FIELDS):
                raise ValueError("oracle NPZ contains missing or unexpected fields")
            arrays = {name: np.array(archive[name], copy=True) for name in _CONTEXT_FIELDS}
    except (OSError, TypeError, ValueError) as error:
        raise ValueError("failed to load strict oracle NPZ") from error
    context = OracleContextSidecar(**arrays)
    if manifest.get("oracle_context_hash") != context.semantic_hash:
        raise ValueError("oracle context semantic hash mismatch")
    return context, manifest


def save_unit_st_regime(
    sensor_directory: os.PathLike[str] | str,
    oracle_directory: os.PathLike[str] | str,
    generated: GeneratedUnitSTRegime,
) -> SemanticHashes:
    """Persist separate sensor and oracle artifacts without crossing the boundary."""

    if not isinstance(generated, GeneratedUnitSTRegime):
        raise TypeError("generated must be a GeneratedUnitSTRegime")
    hashes = save_event_dataset(sensor_directory, generated.dataset, generated.sensor_manifest)
    if hashes != generated.semantic_hashes:
        raise RuntimeError("saved sensor semantic identity changed")
    save_oracle_context(
        oracle_directory, generated.oracle_context, generated.experiment_manifest
    )
    return hashes


def load_unit_st_regime(
    sensor_directory: os.PathLike[str] | str,
    oracle_directory: os.PathLike[str] | str,
    *,
    expected_generator_id: str,
) -> tuple[MEKFDataset, dict[str, Any], SemanticHashes, OracleContextSidecar, dict[str, Any]]:
    """Strict-load both artifacts while retaining their API separation."""

    dataset, sensor_manifest, hashes = load_event_dataset(
        sensor_directory, expected_generator_id=expected_generator_id
    )
    context, experiment_manifest = load_oracle_context(
        oracle_directory, expected_raw_sensor_stream_hash=hashes.dataset_hash
    )
    if not np.array_equal(context.trajectory_id, dataset.events.trajectory_id):
        raise ValueError("oracle sidecar event identity does not match sensor events")
    if not np.array_equal(context.event_order, dataset.events.event_order):
        raise ValueError("oracle sidecar event ordering does not match sensor events")
    return dataset, sensor_manifest, hashes, context, experiment_manifest
