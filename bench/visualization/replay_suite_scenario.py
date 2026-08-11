from __future__ import annotations

import hashlib
import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

from bench.utils.seeding import stable_int_seed_v0

from .adcs_schema import (
    ADCSStateSchema,
    adcs_state_schema_to_dict,
    parse_adcs_state_schema,
)


REPLAY_SCENARIO_FILENAME = "replay_scenario.npz"
REPLAY_SCENARIO_META_FILENAME = "replay_scenario_meta.json"


@dataclass(frozen=True)
class ReplaySuiteScenario:
    suite_name: str
    suite_version: str | None
    task_id: str
    task_name: str | None
    seed: int
    scenario_id: str
    time_s: np.ndarray
    x_true: np.ndarray
    y_obs: np.ndarray
    trajectory_id: np.ndarray
    meta: dict[str, Any]


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} is required and must be a mapping")
    return dict(value)


def _positive_int(value: Any, name: str, *, minimum: int = 1) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise ValueError(f"{name} must be an integer, got {value!r}")
    parsed = int(value)
    if parsed < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {parsed}")
    return parsed


def _finite_float(value: Any, name: str, *, positive: bool = False) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be numeric, got {value!r}")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric, got {value!r}") from exc
    if not np.isfinite(parsed):
        raise ValueError(f"{name} must be finite, got {parsed}")
    if positive and parsed <= 0.0:
        raise ValueError(f"{name} must be > 0, got {parsed}")
    return parsed


def _vector3(value: Any, name: str) -> np.ndarray:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{name} must contain exactly 3 numeric values")
    if len(value) != 3:
        raise ValueError(
            f"{name} must contain exactly 3 values, got {len(value)}"
        )
    try:
        vector = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain numeric values") from exc
    if vector.shape != (3,) or not np.isfinite(vector).all():
        raise ValueError(f"{name} must contain 3 finite numeric values")
    return vector


def load_suite_yaml(path: str | Path) -> dict[str, Any]:
    suite_path = Path(path).expanduser().resolve()
    if not suite_path.exists():
        raise FileNotFoundError(f"suite YAML not found: {suite_path}")
    try:
        loaded = yaml.safe_load(suite_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"invalid suite YAML: {suite_path}: {exc}") from exc
    if not isinstance(loaded, Mapping):
        raise ValueError(f"suite YAML must contain a mapping: {suite_path}")
    suite_cfg = dict(loaded)
    suite_cfg["__source_suite_yaml__"] = str(suite_path)
    return suite_cfg


def select_task_from_suite(
    suite_cfg: Mapping[str, Any],
    task_id: str,
) -> dict[str, Any]:
    requested = str(task_id).strip()
    if not requested:
        raise ValueError("task_id must be a non-empty string")
    tasks = suite_cfg.get("tasks")
    if not isinstance(tasks, list):
        raise ValueError("suite.tasks is required and must be a list")
    matches = [
        task
        for task in tasks
        if isinstance(task, Mapping)
        and str(task.get("task_id", "")).strip() == requested
    ]
    if not matches:
        available = [
            str(task.get("task_id"))
            for task in tasks
            if isinstance(task, Mapping) and task.get("task_id") is not None
        ]
        raise ValueError(
            f"task_id={requested!r} not found in suite; available={available}"
        )
    if len(matches) > 1:
        raise ValueError(f"suite contains duplicate task_id={requested!r}")
    return dict(matches[0])


def _observed_indices(
    observation: Mapping[str, Any],
    *,
    x_dim: int,
) -> tuple[int, ...]:
    raw = observation.get("observed_state")
    if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
        raise ValueError(
            "task.observation.observed_state is required and must be a sequence"
        )
    if len(raw) == 0:
        raise ValueError("task.observation.observed_state must not be empty")
    indices: list[int] = []
    for position, value in enumerate(raw):
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, np.integer)
        ):
            raise ValueError(
                "task.observation.observed_state"
                f"[{position}] must be an integer, got {value!r}"
            )
        index = int(value)
        if index < 0 or index >= x_dim:
            raise ValueError(
                "task.observation.observed_state"
                f"[{position}]={index} is out of range for x_dim={x_dim}"
            )
        indices.append(index)
    if len(set(indices)) != len(indices):
        raise ValueError(
            "task.observation.observed_state contains duplicate indices: "
            f"{indices}"
        )
    return tuple(indices)


def _noise_standard_deviations(
    task_cfg: Mapping[str, Any],
    *,
    observed_indices: tuple[int, ...],
    schema: ADCSStateSchema,
) -> tuple[np.ndarray, str]:
    noise = task_cfg.get("noise", {})
    if noise is None:
        noise = {}
    if not isinstance(noise, Mapping):
        raise ValueError("task.noise must be a mapping")
    replay_measurement = noise.get("replay_measurement")
    if replay_measurement is None:
        return np.zeros(len(observed_indices), dtype=np.float64), "zero_fallback"
    if not isinstance(replay_measurement, Mapping):
        raise ValueError("task.noise.replay_measurement must be a mapping")

    values = {
        "attitude_mrp": _finite_float(
            replay_measurement.get("attitude_mrp", 0.0),
            "task.noise.replay_measurement.attitude_mrp",
        ),
        "angular_rate_rad_s": _finite_float(
            replay_measurement.get("angular_rate_rad_s", 0.0),
            "task.noise.replay_measurement.angular_rate_rad_s",
        ),
        "gyro_bias_rad_s": _finite_float(
            replay_measurement.get("gyro_bias_rad_s", 0.0),
            "task.noise.replay_measurement.gyro_bias_rad_s",
        ),
    }
    for name, value in values.items():
        if value < 0.0:
            raise ValueError(
                f"task.noise.replay_measurement.{name} must be >= 0, got {value}"
            )

    attitude = set(schema.attitude_indices)
    angular_rate = set(schema.angular_rate_indices)
    gyro_bias = set(schema.gyro_bias_indices or ())
    standard_deviations: list[float] = []
    for index in observed_indices:
        if index in attitude:
            standard_deviations.append(values["attitude_mrp"])
        elif index in angular_rate:
            standard_deviations.append(values["angular_rate_rad_s"])
        elif index in gyro_bias:
            standard_deviations.append(values["gyro_bias_rad_s"])
        else:
            standard_deviations.append(0.0)
    return np.asarray(standard_deviations, dtype=np.float64), "explicit"


def _validate_process_noise(dynamics: Mapping[str, Any]) -> None:
    process_noise = dynamics.get("process_noise_std", {})
    if process_noise is None:
        return
    if not isinstance(process_noise, Mapping):
        raise ValueError("task.dynamics.process_noise_std must be a mapping")
    for key in (
        "attitude_mrp",
        "angular_rate_rad_s",
        "gyro_bias_rad_s",
    ):
        value = _finite_float(
            process_noise.get(key, 0.0),
            f"task.dynamics.process_noise_std.{key}",
        )
        if value < 0.0:
            raise ValueError(
                f"task.dynamics.process_noise_std.{key} must be >= 0, got {value}"
            )
        if value != 0.0:
            raise ValueError(
                "Phase 6A simple_attitude_bias supports zero process noise "
                f"only; task.dynamics.process_noise_std.{key}={value}"
            )


def validate_adcs_replay_task(
    suite_cfg: Mapping[str, Any],
    task_cfg: Mapping[str, Any],
    *,
    seed: int,
) -> dict[str, Any]:
    suite = _mapping(suite_cfg.get("suite"), "suite.suite")
    suite_name = str(suite.get("name", "")).strip()
    if not suite_name:
        raise ValueError("suite.suite.name is required")

    seeds = suite_cfg.get("seeds")
    if seeds is not None:
        if not isinstance(seeds, list):
            raise ValueError("suite.seeds must be a list when provided")
        parsed_seeds = [
            _positive_int(value, "suite.seeds[]", minimum=0) for value in seeds
        ]
        if int(seed) not in parsed_seeds:
            raise ValueError(
                f"seed={seed} is not declared by suite.seeds={parsed_seeds}"
            )

    task_id = str(task_cfg.get("task_id", "")).strip()
    if not task_id:
        raise ValueError("task.task_id is required")
    if not bool(task_cfg.get("enabled", True)):
        raise ValueError(f"task {task_id!r} is disabled")
    system_type = str(task_cfg.get("system_type", "")).strip().lower()
    if system_type not in {"adcs_replay", "adcs"}:
        raise ValueError(
            "task.system_type must be 'adcs_replay' or 'adcs', "
            f"got {system_type!r}"
        )

    x_dim = _positive_int(task_cfg.get("x_dim"), "task.x_dim")
    if x_dim not in {6, 9}:
        raise ValueError(f"task.x_dim must be 6 or 9, got {x_dim}")
    sequence_length = _positive_int(
        task_cfg.get("sequence_length_T"),
        "task.sequence_length_T",
        minimum=2,
    )
    time_cfg = _mapping(task_cfg.get("time"), "task.time")
    dt_s = _finite_float(
        time_cfg.get("dt_s"),
        "task.time.dt_s",
        positive=True,
    )
    duration_s = float((sequence_length - 1) * dt_s)
    if time_cfg.get("duration_s") is not None:
        configured_duration = _finite_float(
            time_cfg["duration_s"],
            "task.time.duration_s",
        )
        if not np.isclose(
            configured_duration,
            duration_s,
            rtol=1.0e-6,
            atol=max(1.0e-9, dt_s * 1.0e-6),
        ):
            raise ValueError(
                "task.time.duration_s is inconsistent with "
                "sequence_length_T and dt_s: "
                f"configured={configured_duration}, expected={duration_s}"
            )

    raw_schema = _mapping(task_cfg.get("state_schema"), "task.state_schema")
    schema = parse_adcs_state_schema(
        {
            "state_schema": raw_schema,
            "attitude_convention": str(
                task_cfg.get("attitude_convention", "MRP sigma_BN")
            ),
            "time_unit": "s",
        },
        x_dim=x_dim,
    )
    if schema.angular_rate_type != "rad_s":
        raise ValueError(
            "task.state_schema.angular_rate.type must be 'rad_s', "
            f"got {schema.angular_rate_type!r}"
        )
    if (
        schema.gyro_bias_indices is not None
        and schema.gyro_bias_type != "rad_s"
    ):
        raise ValueError(
            "task.state_schema.gyro_bias.type must be 'rad_s', "
            f"got {schema.gyro_bias_type!r}"
        )
    all_schema_indices = (
        list(schema.attitude_indices)
        + list(schema.angular_rate_indices)
        + list(schema.gyro_bias_indices or ())
    )
    if len(all_schema_indices) != len(set(all_schema_indices)):
        raise ValueError(
            "task.state_schema vectors must not overlap; "
            f"indices={all_schema_indices}"
        )

    initial_state = _mapping(
        task_cfg.get("initial_state"),
        "task.initial_state",
    )
    sigma0 = _vector3(
        initial_state.get("sigma_BN"),
        "task.initial_state.sigma_BN",
    )
    omega0 = _vector3(
        initial_state.get("omega_BN_B_rad_s"),
        "task.initial_state.omega_BN_B_rad_s",
    )
    bias0 = None
    if schema.gyro_bias_indices is not None:
        bias0 = _vector3(
            initial_state.get("gyro_bias_rad_s"),
            "task.initial_state.gyro_bias_rad_s",
        )

    dynamics = _mapping(task_cfg.get("dynamics"), "task.dynamics")
    dynamics_type = str(dynamics.get("type", "")).strip()
    if dynamics_type != "simple_attitude_bias":
        raise ValueError(
            "task.dynamics.type must be 'simple_attitude_bias', "
            f"got {dynamics_type!r}"
        )
    attitude_motion = str(
        dynamics.get("attitude_motion", "constant_rate")
    ).strip()
    if attitude_motion != "constant_rate":
        raise ValueError(
            "task.dynamics.attitude_motion must be 'constant_rate', "
            f"got {attitude_motion!r}"
        )
    gyro_bias_model = str(
        dynamics.get("gyro_bias_model", "constant")
    ).strip()
    if schema.gyro_bias_indices is not None and gyro_bias_model != "constant":
        raise ValueError(
            "task.dynamics.gyro_bias_model must be 'constant', "
            f"got {gyro_bias_model!r}"
        )
    _validate_process_noise(dynamics)

    observation = _mapping(
        task_cfg.get("observation"),
        "task.observation",
    )
    observed_indices = _observed_indices(observation, x_dim=x_dim)
    y_dim = _positive_int(task_cfg.get("y_dim"), "task.y_dim")
    if y_dim != len(observed_indices):
        raise ValueError(
            "task.y_dim must equal len(task.observation.observed_state): "
            f"y_dim={y_dim}, observed={len(observed_indices)}"
        )
    measurement_std, measurement_noise_source = _noise_standard_deviations(
        task_cfg,
        observed_indices=observed_indices,
        schema=schema,
    )

    ground_truth = _mapping(
        task_cfg.get("ground_truth"),
        "task.ground_truth",
    )
    if ground_truth.get("has_gt") is not True:
        raise ValueError("task.ground_truth.has_gt must be true")

    replay = _mapping(task_cfg.get("replay"), "task.replay")
    if replay.get("enabled") is not True:
        raise ValueError("task.replay.enabled must be true")

    dataset_sizes = _mapping(
        task_cfg.get("dataset_sizes"),
        "task.dataset_sizes",
    )
    raw_n_test = dataset_sizes.get("N_test")
    n_test = (
        1
        if raw_n_test is None or raw_n_test == 0
        else _positive_int(raw_n_test, "task.dataset_sizes.N_test")
    )

    return {
        "suite_name": suite_name,
        "suite_version": (
            None if suite.get("version") is None else str(suite.get("version"))
        ),
        "task_id": task_id,
        "task_name": (
            None
            if task_cfg.get("task_name") is None
            else str(task_cfg.get("task_name"))
        ),
        "seed": int(seed),
        "x_dim": x_dim,
        "y_dim": y_dim,
        "sequence_length_T": sequence_length,
        "dt_s": dt_s,
        "duration_s": duration_s,
        "n_test": n_test,
        "dataset_sizes": _jsonable(dataset_sizes),
        "schema": schema,
        "state_schema": adcs_state_schema_to_dict(schema),
        "sigma0": sigma0,
        "omega0": omega0,
        "bias0": bias0,
        "initial_state": _jsonable(initial_state),
        "dynamics": _jsonable(dynamics),
        "observation": _jsonable(observation),
        "observed_indices": observed_indices,
        "noise": _jsonable(task_cfg.get("noise", {}) or {}),
        "measurement_std": measurement_std,
        "measurement_noise_source": measurement_noise_source,
        "vizard": _jsonable(task_cfg.get("vizard", {}) or {}),
        "replay": _jsonable(replay),
    }


def _scenario_hash_fields(
    suite_cfg: Mapping[str, Any],
    task_cfg: Mapping[str, Any],
    *,
    seed: int,
) -> dict[str, Any]:
    suite = _mapping(suite_cfg.get("suite"), "suite.suite")
    return {
        "suite_name": str(suite.get("name")),
        "task_id": str(task_cfg.get("task_id")),
        "seed": int(seed),
        "sequence_length_T": task_cfg.get("sequence_length_T"),
        "dt_s": _mapping(task_cfg.get("time"), "task.time").get("dt_s"),
        "initial_state": _jsonable(task_cfg.get("initial_state")),
        "dynamics": _jsonable(task_cfg.get("dynamics")),
        "observation": _jsonable(task_cfg.get("observation")),
        "noise": _jsonable(task_cfg.get("noise", {}) or {}),
    }


def _scenario_id(hash_fields: Mapping[str, Any]) -> str:
    canonical = json.dumps(
        _jsonable(hash_fields),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    digest = hashlib.sha1(canonical.encode("utf-8")).hexdigest()[:8]
    return f"scenario_{digest}"


def materialize_adcs_replay_task(
    suite_cfg: Mapping[str, Any],
    task_cfg: Mapping[str, Any],
    *,
    seed: int,
) -> ReplaySuiteScenario:
    resolved = validate_adcs_replay_task(suite_cfg, task_cfg, seed=seed)
    n_test = int(resolved["n_test"])
    sequence_length = int(resolved["sequence_length_T"])
    x_dim = int(resolved["x_dim"])
    dt_s = float(resolved["dt_s"])
    schema: ADCSStateSchema = resolved["schema"]
    time_s = np.arange(sequence_length, dtype=np.float64) * dt_s

    base_state = np.zeros((sequence_length, x_dim), dtype=np.float64)
    sigma = (
        resolved["sigma0"][None, :]
        + 0.25 * time_s[:, None] * resolved["omega0"][None, :]
    )
    sigma_norm = np.linalg.norm(sigma, axis=-1)
    max_sigma_norm = float(np.max(sigma_norm))
    if max_sigma_norm >= 1.0:
        raise ValueError(
            "simple replay propagation produces MRP norm >= 1.0 "
            f"(max={max_sigma_norm:.6g}); use a shorter duration or smaller "
            "angular rate"
        )
    base_state[:, schema.attitude_indices] = sigma
    base_state[:, schema.angular_rate_indices] = resolved["omega0"]
    if schema.gyro_bias_indices is not None:
        base_state[:, schema.gyro_bias_indices] = resolved["bias0"]

    x_true = np.broadcast_to(
        base_state[None, :, :],
        (n_test, sequence_length, x_dim),
    ).copy()
    observed_indices = np.asarray(
        resolved["observed_indices"],
        dtype=np.int64,
    )
    y_clean = np.take(x_true, observed_indices, axis=-1)
    hash_fields = _scenario_hash_fields(suite_cfg, task_cfg, seed=seed)
    scenario_id = _scenario_id(hash_fields)
    measurement_seed = stable_int_seed_v0(
        "phase6a_measurement",
        resolved["suite_name"],
        resolved["task_id"],
        scenario_id,
        int(seed),
    )
    rng = np.random.default_rng(measurement_seed)
    measurement_std = np.asarray(
        resolved["measurement_std"],
        dtype=np.float64,
    )
    measurement_noise = rng.normal(
        loc=0.0,
        scale=measurement_std,
        size=y_clean.shape,
    )
    y_obs = y_clean + measurement_noise
    trajectory_id = np.arange(n_test, dtype=np.int64)

    source_suite_yaml = suite_cfg.get("__source_suite_yaml__")
    meta = {
        "schema_version": "phase6a_replay_input_v1",
        "suite_name": resolved["suite_name"],
        "suite_version": resolved["suite_version"],
        "task_id": resolved["task_id"],
        "task_name": resolved["task_name"],
        "seed": int(seed),
        "scenario_id": scenario_id,
        "source_suite_yaml": (
            None if source_suite_yaml is None else str(source_suite_yaml)
        ),
        "time": {
            "dt_s": dt_s,
            "sequence_length_T": sequence_length,
            "duration_s": float(time_s[-1]),
        },
        "dataset_sizes": resolved["dataset_sizes"],
        "num_trajectories": n_test,
        "state_dim": x_dim,
        "measurement_dim": int(resolved["y_dim"]),
        "state_schema": resolved["state_schema"],
        "initial_state": resolved["initial_state"],
        "dynamics": resolved["dynamics"],
        "observation": resolved["observation"],
        "noise": resolved["noise"],
        "measurement_noise_source": resolved["measurement_noise_source"],
        "measurement_seed": int(measurement_seed),
        "measurement_std_by_observed_state": measurement_std.tolist(),
        "vizard": resolved["vizard"],
        "replay": resolved["replay"],
        "trajectory_generation": "exact_repeat_for_n_test",
        "process_noise_policy": (
            "zero_only_constant_rate_constant_bias_propagation"
        ),
        "scenario_hash_fields": hash_fields,
        "notes": (
            "Phase 6A replay input is generated from an existing-style suite "
            "YAML task and is not a high-fidelity spacecraft dynamics simulator."
        ),
    }
    return ReplaySuiteScenario(
        suite_name=str(resolved["suite_name"]),
        suite_version=resolved["suite_version"],
        task_id=str(resolved["task_id"]),
        task_name=resolved["task_name"],
        seed=int(seed),
        scenario_id=scenario_id,
        time_s=time_s.astype(np.float32),
        x_true=x_true.astype(np.float32),
        y_obs=y_obs.astype(np.float32),
        trajectory_id=trajectory_id,
        meta=meta,
    )


def save_replay_input_npz(
    scenario: ReplaySuiteScenario,
    out_dir: str | Path,
) -> tuple[Path, Path]:
    output_dir = Path(out_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    npz_path = output_dir / REPLAY_SCENARIO_FILENAME
    meta_path = output_dir / REPLAY_SCENARIO_META_FILENAME

    n_test, sequence_length, x_dim = scenario.x_true.shape
    if scenario.y_obs.ndim != 3 or scenario.y_obs.shape[:2] != (
        n_test,
        sequence_length,
    ):
        raise ValueError(
            "scenario.y_obs must have shape [N,T,Dy] matching x_true[:2], "
            f"got x_true={scenario.x_true.shape}, y_obs={scenario.y_obs.shape}"
        )
    if scenario.time_s.shape != (sequence_length,):
        raise ValueError(
            f"scenario.time_s must have shape {(sequence_length,)}, "
            f"got {scenario.time_s.shape}"
        )
    if scenario.trajectory_id.shape != (n_test,):
        raise ValueError(
            f"scenario.trajectory_id must have shape {(n_test,)}, "
            f"got {scenario.trajectory_id.shape}"
        )
    for name, values in (
        ("time_s", scenario.time_s),
        ("x_true", scenario.x_true),
        ("y_obs", scenario.y_obs),
        ("trajectory_id", scenario.trajectory_id),
    ):
        if not np.isfinite(values).all():
            raise ValueError(f"scenario.{name} contains NaN or Inf")
    if x_dim not in {6, 9}:
        raise ValueError(f"scenario.x_true state dimension must be 6 or 9, got {x_dim}")

    with tempfile.TemporaryDirectory(
        prefix=".phase6a_replay_",
        dir=output_dir,
    ) as tmp:
        staging_dir = Path(tmp)
        staged_npz = staging_dir / REPLAY_SCENARIO_FILENAME
        staged_meta = staging_dir / REPLAY_SCENARIO_META_FILENAME
        np.savez_compressed(
            staged_npz,
            time_s=np.asarray(scenario.time_s, dtype=np.float32),
            x_true=np.asarray(scenario.x_true, dtype=np.float32),
            y_obs=np.asarray(scenario.y_obs, dtype=np.float32),
            trajectory_id=np.asarray(scenario.trajectory_id, dtype=np.int64),
        )
        staged_meta.write_text(
            json.dumps(
                _jsonable(scenario.meta),
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        staged_npz.replace(npz_path)
        staged_meta.replace(meta_path)
    return npz_path, meta_path
