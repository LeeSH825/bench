"""Typed generator-side records and split/normalization firewalls.

Calibration and truth exist only in this generator/diagnostic module.  Runtime
model objects receive :class:`SensorTrajectory` values, which contain sensor
packets and frozen onboard constants but no truth or calibration fields.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass
from typing import Any, Iterable, Literal, Mapping

import numpy as np

from bench.estimators.mekf import quat_exp, quat_multiply, quat_normalize, quat_to_dcm


REGIMES = ("R0", "R1", "R2", "R3", "R4")
TRAIN_REGIMES = ("R0", "R1", "R2", "R3")
SENSORS = ("gyro", "magnetometer")
FORBIDDEN_DEPLOYABLE_KEYS = frozenset(
    {
        "A_g", "c_g", "C_SgB", "A_m", "b_m", "C_BSm",
        "gyro_calibration", "mag_calibration", "calibration", "oracle",
        "truth", "future", "regime", "event_label", "event_window", "evaluation_metric",
        "inverse_A_g", "inverse_A_m", "A_g_inv", "A_m_inv",
    }
)


def _vec(value: Any, size: int, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (size,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite with shape ({size},)")
    result = array.copy()
    result.setflags(write=False)
    return result


def _array(value: Any, shape: tuple[int, ...], name: str, dtype: Any = np.float64) -> np.ndarray:
    array = np.asarray(value, dtype=dtype)
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    if array.dtype.kind == "f" and not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    result = array.copy()
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class SensorFrameEvent:
    trajectory_id: int
    regime: str
    timestamp_s: float
    event_order: int
    sensor: Literal["gyro", "magnetometer"]
    measurement_S: np.ndarray
    valid: bool
    realization_id: str
    stream_namespace: str

    def __post_init__(self) -> None:
        if self.regime not in REGIMES or self.sensor not in SENSORS:
            raise ValueError("unknown regime or sensor")
        if self.trajectory_id < 0 or self.event_order < 0 or self.timestamp_s < 0:
            raise ValueError("event identifiers and time must be nonnegative")
        object.__setattr__(self, "measurement_S", _vec(self.measurement_S, 3, "measurement_S"))
        if not self.realization_id or not self.stream_namespace:
            raise ValueError("realization and stream namespaces are required")


@dataclass(frozen=True)
class SensorTrajectory:
    """The complete deployable packet sequence for one trajectory."""

    trajectory_id: int
    regime: str
    realization_id: str
    stream_namespace: str
    events: tuple[SensorFrameEvent, ...]

    def __post_init__(self) -> None:
        if not self.events:
            raise ValueError("trajectory events must be nonempty")
        expected = list(range(len(self.events)))
        if [event.event_order for event in self.events] != expected:
            raise ValueError("event order must be contiguous")
        if any(event.trajectory_id != self.trajectory_id for event in self.events):
            raise ValueError("event trajectory IDs must agree")
        computed_realization = raw_realization_digest(self.events)
        if self.realization_id != computed_realization:
            raise ValueError("realization_id must be derived from raw packet bytes")
        if any(event.realization_id != computed_realization for event in self.events):
            raise ValueError("event realization labels must match raw packet provenance")
        by_time: dict[float, list[str]] = {}
        for event in self.events:
            by_time.setdefault(event.timestamp_s, []).append(event.sensor)
        if any(sensors != ["gyro", "magnetometer"] for sensors in by_time.values()):
            raise ValueError("same-time order must be gyro then magnetometer")


@dataclass(frozen=True)
class RuntimeSensorPacket:
    """Stripped deployable packet with an explicit field allowlist."""

    timestamp_s: float
    event_order: int
    sensor: Literal["gyro", "magnetometer"]
    measurement_S: np.ndarray
    valid: bool

    def __post_init__(self) -> None:
        if not np.isfinite(self.timestamp_s) or self.timestamp_s < 0 or self.event_order < 0:
            raise ValueError("runtime packet time/order must be finite and nonnegative")
        if self.sensor not in SENSORS:
            raise ValueError("runtime packet sensor must be gyro or magnetometer")
        object.__setattr__(self, "measurement_S", _vec(self.measurement_S, 3, "measurement_S"))


@dataclass(frozen=True)
class RuntimeTrajectoryBatch:
    """Deployable whole-trajectory batch; no regime, event label, or sidecar."""

    trajectory_id: int
    realization_sha256: str
    packets: tuple[RuntimeSensorPacket, ...]

    def __post_init__(self) -> None:
        if self.trajectory_id < 0 or not self.packets:
            raise ValueError("runtime trajectory must be nonempty")
        if [packet.event_order for packet in self.packets] != list(range(len(self.packets))):
            raise ValueError("runtime event order must be contiguous")
        if self.realization_sha256 != runtime_realization_digest(self.packets):
            raise ValueError("runtime realization digest does not match packet bytes")


@dataclass(frozen=True)
class RuntimeNormalization:
    gyro_mean: np.ndarray
    gyro_std: np.ndarray
    mag_mean: np.ndarray
    mag_std: np.ndarray
    normalization_sha256: str

    def __post_init__(self) -> None:
        for name in ("gyro_mean", "gyro_std", "mag_mean", "mag_std"):
            object.__setattr__(self, name, _vec(getattr(self, name), 3, name))
        if np.any(self.gyro_std <= 0) or np.any(self.mag_std <= 0):
            raise ValueError("runtime normalization std must be positive")
        if len(self.normalization_sha256) != 64:
            raise ValueError("runtime normalization digest must be sha256")


@dataclass(frozen=True)
class TrajectoryTruth:
    trajectory_id: int
    timestamp_s: np.ndarray
    q_true_NB: np.ndarray
    omega_true_B_rad_s: np.ndarray
    residual_bias_B_rad_s: np.ndarray
    m_true_N: np.ndarray

    def __post_init__(self) -> None:
        count = len(self.timestamp_s)
        object.__setattr__(self, "timestamp_s", _array(self.timestamp_s, (count,), "timestamp_s"))
        object.__setattr__(self, "q_true_NB", _array(self.q_true_NB, (count, 4), "q_true_NB"))
        object.__setattr__(self, "omega_true_B_rad_s", _array(self.omega_true_B_rad_s, (count, 3), "omega_true"))
        object.__setattr__(self, "residual_bias_B_rad_s", _array(self.residual_bias_B_rad_s, (count, 3), "residual_bias"))
        object.__setattr__(self, "m_true_N", _vec(self.m_true_N, 3, "m_true_N"))


@dataclass(frozen=True)
class CalibrationTruth:
    """Generator-only deterministic sensor calibration."""

    A_g: np.ndarray
    c_g: np.ndarray
    C_SgB: np.ndarray
    A_m: np.ndarray
    b_m: np.ndarray
    C_BSm: np.ndarray

    def __post_init__(self) -> None:
        for name in ("A_g", "C_SgB", "A_m", "C_BSm"):
            object.__setattr__(self, name, _array(getattr(self, name), (3, 3), name))
        object.__setattr__(self, "c_g", _vec(self.c_g, 3, "c_g"))
        object.__setattr__(self, "b_m", _vec(self.b_m, 3, "b_m"))


@dataclass(frozen=True)
class OracleSidecar:
    """Diagnostic-only targets and calibration; never accepted by deployable replay."""

    trajectory_id: int
    calibration: CalibrationTruth
    gyro_target_B_rad_s: np.ndarray
    mag_target_B: np.ndarray

    def __post_init__(self) -> None:
        count = self.gyro_target_B_rad_s.shape[0]
        object.__setattr__(self, "gyro_target_B_rad_s", _array(self.gyro_target_B_rad_s, (count, 3), "gyro_target"))
        object.__setattr__(self, "mag_target_B", _array(self.mag_target_B, (count, 3), "mag_target"))


@dataclass(frozen=True)
class WholeTrajectorySplit:
    train_ids: tuple[int, ...]
    validation_ids: tuple[int, ...]
    test_ids: tuple[int, ...]
    regime_by_id: Mapping[int, str]
    stream_namespace_by_id: Mapping[int, str]
    data_generation_seed: int
    split_seed: int

    def __post_init__(self) -> None:
        groups = [set(self.train_ids), set(self.validation_ids), set(self.test_ids)]
        if any(len(values) != len(set(values)) for values in (self.train_ids, self.validation_ids, self.test_ids)):
            raise ValueError("duplicate trajectory IDs within a split are forbidden")
        if any(not group for group in groups):
            raise ValueError("all trajectory splits must be nonempty")
        if groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2]:
            raise ValueError("trajectory splits must be disjoint")
        union = groups[0] | groups[1] | groups[2]
        if union != set(self.regime_by_id) or union != set(self.stream_namespace_by_id):
            raise ValueError("split metadata must cover every trajectory exactly once")
        if any(self.regime_by_id[item] == "R4" for item in groups[0] | groups[1]):
            raise ValueError("R4 is test-only")
        train_val_namespaces = {self.stream_namespace_by_id[item] for item in groups[0] | groups[1]}
        r4_namespaces = {self.stream_namespace_by_id[item] for item in groups[2] if self.regime_by_id[item] == "R4"}
        if train_val_namespaces & r4_namespaces:
            raise ValueError("R4 RNG namespaces must be disjoint from train/validation")
        for regime in TRAIN_REGIMES:
            if not any(self.regime_by_id[item] == regime for item in groups[0]):
                raise ValueError(f"empty training population for {regime}")
            if not any(self.regime_by_id[item] == regime for item in groups[1]):
                raise ValueError(f"empty validation population for {regime}")
        for regime in REGIMES:
            if not any(self.regime_by_id[item] == regime for item in groups[2]):
                raise ValueError(f"empty test population for {regime}")


@dataclass(frozen=True)
class NormalizationRecord:
    gyro_mean: np.ndarray
    gyro_std: np.ndarray
    mag_mean: np.ndarray
    mag_std: np.ndarray
    source_trajectory_ids: tuple[int, ...]
    frozen_before_test: bool
    sha256: str

    def __post_init__(self) -> None:
        for name in ("gyro_mean", "gyro_std", "mag_mean", "mag_std"):
            object.__setattr__(self, name, _vec(getattr(self, name), 3, name))
        if np.any(self.gyro_std <= 0) or np.any(self.mag_std <= 0):
            raise ValueError("normalization standard deviations must be positive")
        if not self.frozen_before_test:
            raise ValueError("normalization must be frozen before test access")
        if self.sha256 != normalization_digest(self):
            raise ValueError("normalization digest mismatch")


@dataclass(frozen=True)
class SplitFirewallRecord:
    normalization_ids: tuple[int, ...]
    training_loss_ids: tuple[int, ...]
    early_stopping_ids: tuple[int, ...]
    checkpoint_selection_ids: tuple[int, ...]
    threshold_setting_ids: tuple[int, ...]


@dataclass(frozen=True)
class GeneratedDataset:
    sensor: Mapping[int, SensorTrajectory]
    truth: Mapping[int, TrajectoryTruth]
    oracle: Mapping[int, OracleSidecar]
    split: WholeTrajectorySplit
    m_model_N_onboard: np.ndarray

    def __post_init__(self) -> None:
        ids = set(self.sensor)
        if ids != set(self.truth) or ids != set(self.oracle):
            raise ValueError("sensor/truth/oracle trajectory IDs must pair exactly")
        object.__setattr__(self, "m_model_N_onboard", _vec(self.m_model_N_onboard, 3, "m_model_N_onboard"))
        if any(np.array_equal(self.m_model_N_onboard, truth.m_true_N) for truth in self.truth.values()):
            raise ValueError("onboard magnetic model must be distinct from truth reference")


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def normalization_digest(record: NormalizationRecord) -> str:
    payload = {
        "gyro_mean": np.asarray(record.gyro_mean).tolist(), "gyro_std": np.asarray(record.gyro_std).tolist(),
        "mag_mean": np.asarray(record.mag_mean).tolist(), "mag_std": np.asarray(record.mag_std).tolist(),
        "source_trajectory_ids": list(record.source_trajectory_ids), "frozen_before_test": bool(record.frozen_before_test),
    }
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _forbidden_token(token: str) -> bool:
    normalized = token.lower().replace("-", "_")
    compact = "".join(character for character in normalized if character.isalnum())
    exact_compact = {"am", "bm", "cbsm", "ag", "cg", "csgb"}
    calibration_prefixes = ("a_m", "b_m", "c_bsm", "a_g", "c_g", "c_sgb")
    return token in FORBIDDEN_DEPLOYABLE_KEYS or compact in exact_compact or normalized.startswith(calibration_prefixes) or any(
        marker in normalized for marker in (
            "truth", "oracle", "future", "calibr", "event_label", "event_window",
            "evaluation_metric", "regime",
        )
    ) or ("inverse" in normalized and any(marker in normalized for marker in ("gyro", "mag", "a_g", "a_m")))


def validate_deployable_namespace(value: Any, path: str = "runtime") -> None:
    """Reject forbidden names, aliases, inverses, truth and oracle objects recursively."""

    if isinstance(value, (TrajectoryTruth, CalibrationTruth, OracleSidecar, SensorFrameEvent, SensorTrajectory, GeneratedDataset)):
        raise ValueError(f"forbidden diagnostic object at {path}")
    if dataclasses.is_dataclass(value):
        for field in dataclasses.fields(value):
            if _forbidden_token(field.name):
                raise ValueError(f"forbidden deployable field {field.name!r} at {path}")
            validate_deployable_namespace(getattr(value, field.name), f"{path}.{field.name}")
    if isinstance(value, Mapping):
        for key, child in value.items():
            token = str(key)
            if _forbidden_token(token):
                raise ValueError(f"forbidden deployable key {token!r} at {path}")
            validate_deployable_namespace(child, f"{path}.{token}")
    elif isinstance(value, (tuple, list)):
        for index, child in enumerate(value):
            validate_deployable_namespace(child, f"{path}[{index}]")


def validate_firewall(split: WholeTrajectorySplit, record: SplitFirewallRecord) -> None:
    train, validation, test = set(split.train_ids), set(split.validation_ids), set(split.test_ids)
    source_contracts = (
        ("normalization_ids", record.normalization_ids, train, "training"),
        ("training_loss_ids", record.training_loss_ids, train, "training"),
        ("threshold_setting_ids", record.threshold_setting_ids, train, "training"),
        ("early_stopping_ids", record.early_stopping_ids, validation, "validation"),
        ("checkpoint_selection_ids", record.checkpoint_selection_ids, validation, "validation"),
    )
    for field_name, values, expected, split_name in source_contracts:
        if not values:
            raise ValueError(f"{field_name} must be nonempty")
        if len(values) != len(set(values)):
            raise ValueError(f"{field_name} contains duplicate trajectory IDs")
        actual = set(values)
        if actual & test:
            raise ValueError("test trajectory contributed to training or selection")
        if actual != expected:
            raise ValueError(f"{field_name} must equal the complete {split_name} trajectory-ID set")


def freeze_train_normalization(dataset: GeneratedDataset) -> NormalizationRecord:
    train_ids = tuple(sorted(dataset.split.train_ids))
    if any(dataset.split.regime_by_id[item] not in TRAIN_REGIMES for item in train_ids):
        raise ValueError("normalization accepts only R0-R3 training IDs")
    gyro, mag = [], []
    for trajectory_id in train_ids:
        for event in dataset.sensor[trajectory_id].events:
            (gyro if event.sensor == "gyro" else mag).append(event.measurement_S)
    gyro_values, mag_values = np.stack(gyro), np.stack(mag)
    payload = {
        "gyro_mean": gyro_values.mean(0), "gyro_std": np.maximum(gyro_values.std(0), 1e-12),
        "mag_mean": mag_values.mean(0), "mag_std": np.maximum(mag_values.std(0), 1e-12),
        "source_trajectory_ids": train_ids, "frozen_before_test": True,
    }
    placeholder = object.__new__(NormalizationRecord)
    for key, value in payload.items():
        object.__setattr__(placeholder, key, value)
    digest = normalization_digest(placeholder)
    return NormalizationRecord(**payload, sha256=digest)


def oracle_correct(sidecar: OracleSidecar) -> tuple[np.ndarray, np.ndarray]:
    """Explicit diagnostic namespace for exact oracle targets."""

    return sidecar.gyro_target_B_rad_s.copy(), sidecar.mag_target_B.copy()


def _hash_packet_parts(parts: Iterable[tuple[float, int, str, np.ndarray, bool]]) -> str:
    digest = hashlib.sha256(b"side-gyro-mag-raw-realization-v1\0")
    for timestamp, order, sensor, measurement, valid in parts:
        digest.update(np.asarray([timestamp], dtype="<f8").tobytes())
        digest.update(np.asarray([order], dtype="<i8").tobytes())
        digest.update(sensor.encode("ascii") + b"\0")
        digest.update(np.asarray(measurement, dtype="<f8").tobytes())
        digest.update(b"\x01" if valid else b"\x00")
    return digest.hexdigest()


def raw_realization_digest(events: Iterable[SensorFrameEvent]) -> str:
    return _hash_packet_parts(
        (event.timestamp_s, event.event_order, event.sensor, event.measurement_S, event.valid)
        for event in events
    )


def runtime_realization_digest(packets: Iterable[RuntimeSensorPacket]) -> str:
    return _hash_packet_parts(
        (packet.timestamp_s, packet.event_order, packet.sensor, packet.measurement_S, packet.valid)
        for packet in packets
    )


def strip_runtime_trajectory(trajectory: SensorTrajectory) -> RuntimeTrajectoryBatch:
    packets = tuple(RuntimeSensorPacket(
        timestamp_s=event.timestamp_s,
        event_order=event.event_order,
        sensor=event.sensor,
        measurement_S=event.measurement_S,
        valid=event.valid,
    ) for event in trajectory.events)
    result = RuntimeTrajectoryBatch(
        trajectory_id=trajectory.trajectory_id,
        realization_sha256=runtime_realization_digest(packets),
        packets=packets,
    )
    validate_deployable_namespace(result)
    return result


def strip_runtime_normalization(record: NormalizationRecord) -> RuntimeNormalization:
    result = RuntimeNormalization(
        record.gyro_mean, record.gyro_std, record.mag_mean, record.mag_std, record.sha256,
    )
    validate_deployable_namespace(result)
    return result


def _signed(rng: np.random.Generator, low: float, high: float) -> np.ndarray:
    return rng.choice(np.array([-1.0, 1.0]), 3) * rng.uniform(low, high, 3)


def _regime_calibration(regime: str, rng: np.random.Generator, m_norm: float) -> CalibrationTruth:
    if regime == "R0":
        c_g, dg, b_m, dm = np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3)
    elif regime in ("R1", "R2", "R3"):
        c_g = _signed(rng, 2e-4, 8e-4) if regime in ("R1", "R3") else np.zeros(3)
        dg = _signed(rng, 0.005, 0.015) if regime in ("R1", "R3") else np.zeros(3)
        b_m = _signed(rng, 0.02, 0.06) * m_norm if regime in ("R2", "R3") else np.zeros(3)
        dm = _signed(rng, 0.02, 0.06) if regime in ("R2", "R3") else np.zeros(3)
    elif regime == "R4":
        c_g, dg = _signed(rng, 1.2e-3, 1.8e-3), _signed(rng, 0.025, 0.040)
        b_m, dm = _signed(rng, 0.10, 0.16) * m_norm, _signed(rng, 0.10, 0.16)
    else:
        raise ValueError("unknown regime")
    # Identity mounting is the explicit pilot invariant.  A_m is SPD here.
    return CalibrationTruth(
        A_g=np.diag(1.0 + dg), c_g=c_g, C_SgB=np.eye(3),
        A_m=np.diag(1.0 + dm), b_m=b_m, C_BSm=np.eye(3),
    )


def generate_dataset(
    *, population: Mapping[str, int], sequence_length: int = 16, dt_s: float = 0.1,
    generation_seed: int = 271828, split_seed: int = 314159,
) -> GeneratedDataset:
    """Generate R0--R4 with exact whole-trajectory split populations."""

    if set(population) != {"train", "validation", "test"} or min(population.values()) <= 0:
        raise ValueError("population must define positive train/validation/test counts")
    if sequence_length < 4 or dt_s <= 0:
        raise ValueError("sequence length and dt must be positive")
    m_true = np.array([0.30, -0.20, 0.932], dtype=np.float64)
    m_model = np.array([0.31, -0.18, 0.933], dtype=np.float64)
    calibrations = {
        regime: _regime_calibration(regime, np.random.default_rng(generation_seed + 101 * index), np.linalg.norm(m_true))
        for index, regime in enumerate(REGIMES)
    }
    sensor: dict[int, SensorTrajectory] = {}
    truth: dict[int, TrajectoryTruth] = {}
    oracle: dict[int, OracleSidecar] = {}
    train_ids: list[int] = []
    validation_ids: list[int] = []
    test_ids: list[int] = []
    regime_by_id: dict[int, str] = {}
    namespace_by_id: dict[int, str] = {}
    trajectory_id = 0
    for regime_index, regime in enumerate(REGIMES):
        split_counts = {"test": population["test"]} if regime == "R4" else population
        for split_name, count in split_counts.items():
            for local_index in range(count):
                tid = trajectory_id
                trajectory_id += 1
                namespace = f"data-{generation_seed}/{regime}/{split_name}/{local_index}"
                rng = np.random.default_rng(generation_seed + 100000 * regime_index + 1000 * ("train validation test".split().index(split_name)) + local_index)
                time = np.arange(1, sequence_length + 1, dtype=np.float64) * dt_s
                phase = rng.uniform(-np.pi, np.pi, 3)
                sample = np.arange(sequence_length, dtype=np.float64)
                omega = np.column_stack([
                    0.035 * np.sin(2.0 * np.pi * sample / sequence_length + phase[0]),
                    0.040 * np.sin(4.0 * np.pi * sample / sequence_length + phase[1]),
                    0.045 * np.sin(6.0 * np.pi * sample / sequence_length + phase[2]),
                ])
                centered = omega - omega.mean(0)
                excitation = float(np.linalg.eigvalsh(centered.T @ centered / sequence_length)[0])
                if excitation < 1e-5:
                    raise RuntimeError(f"excitation certificate failed: {excitation:.6g}")
                residual_bias = np.empty((sequence_length, 3), dtype=np.float64)
                residual_bias[0] = rng.normal(0.0, 2e-5, 3)
                for index in range(1, sequence_length):
                    residual_bias[index] = residual_bias[index - 1] + rng.normal(0.0, 2e-7, 3)
                q = np.empty((sequence_length, 4), dtype=np.float64)
                q_prev = quat_normalize(np.r_[1.0, rng.normal(0.0, 0.03, 3)])
                calibration = calibrations[regime]
                events: list[SensorFrameEvent] = []
                gyro_target, mag_target = [], []
                realization_id = "pending"
                for index, timestamp in enumerate(time):
                    q_prev = quat_normalize(quat_multiply(q_prev, quat_exp(omega[index] * dt_s)))
                    q[index] = q_prev
                    gyro_noise = rng.normal(0.0, 1e-5, 3)
                    mag_noise = rng.normal(0.0, 2e-4, 3)
                    gyro_body = omega[index] + residual_bias[index] + gyro_noise
                    mag_body = quat_to_dcm(q_prev).T @ m_true + mag_noise
                    y_g = calibration.A_g @ gyro_body + calibration.c_g
                    y_m = calibration.A_m @ mag_body + calibration.b_m
                    gyro_target.append(gyro_body)
                    mag_target.append(mag_body)
                    for sensor_name, measurement in (("gyro", y_g), ("magnetometer", y_m)):
                        events.append(SensorFrameEvent(
                            trajectory_id=tid, regime=regime, timestamp_s=float(timestamp),
                            event_order=len(events), sensor=sensor_name, measurement_S=measurement,
                            valid=True, realization_id=realization_id, stream_namespace=namespace,
                        ))
                realization_id = raw_realization_digest(events)
                events = [dataclasses.replace(event, realization_id=realization_id) for event in events]
                sensor[tid] = SensorTrajectory(tid, regime, realization_id, namespace, tuple(events))
                truth[tid] = TrajectoryTruth(tid, time, q, omega, residual_bias, m_true)
                oracle[tid] = OracleSidecar(tid, calibration, np.stack(gyro_target), np.stack(mag_target))
                regime_by_id[tid] = regime
                namespace_by_id[tid] = namespace
                {"train": train_ids, "validation": validation_ids, "test": test_ids}[split_name].append(tid)
    split = WholeTrajectorySplit(
        tuple(train_ids), tuple(validation_ids), tuple(test_ids), regime_by_id,
        namespace_by_id, generation_seed, split_seed,
    )
    return GeneratedDataset(sensor, truth, oracle, split, m_model)


def assert_same_realization(trajectories: Iterable[SensorTrajectory]) -> None:
    values = list(trajectories)
    if not values:
        raise ValueError("pairing population must be nonempty")
    key = (values[0].trajectory_id, raw_realization_digest(values[0].events))
    if any((item.trajectory_id, raw_realization_digest(item.events)) != key for item in values[1:]):
        raise ValueError("compared variants must use the same trajectory and raw realization")
