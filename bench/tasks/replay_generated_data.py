from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

from bench.utils.seeding import stable_int_seed_v0
from bench.visualization.replay_suite_scenario import (
    load_suite_yaml,
    materialize_adcs_replay_task,
    select_task_from_suite,
    validate_adcs_replay_task,
)

from .data_format import (
    CANONICAL_LAYOUT_V0,
    DatasetArtifactsV0,
    DatasetSplitV0,
    save_npz_split_v0,
)
from .generator.contract import GeneratorOutput, coerce_ntd_float32_output
from .generator.datasets.common import INTERNAL_SPLIT_PAYLOADS_KEY
from .generator.schema import enforce_meta_v1
from .generator.validate import validate_artifacts


REPLAY_GENERATED_TASK_FAMILY = "adcs_replay_v0"


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _positive_int(value: Any, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer, got {value!r}")
    parsed = int(value)
    if parsed < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {parsed}")
    return parsed


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} is required and must be a mapping")
    return dict(value)


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
    payload = json.dumps(_jsonable(hash_fields), sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:8]
    return f"scenario_{digest}"


def _task_cache_dir(cache_root: Path, suite_name: str, task_id: str, scenario_id: str, seed: int) -> Path:
    return cache_root / suite_name / task_id / f"scenario_{scenario_id}" / f"seed_{seed}"


def _observed_state_indices(task_cfg: Mapping[str, Any]) -> list[int]:
    observation = _mapping(task_cfg.get("observation"), "task.observation")
    raw = observation.get("observed_state")
    if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
        raise ValueError("task.observation.observed_state is required and must be a sequence")
    indices: list[int] = []
    for position, value in enumerate(raw):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise ValueError(
                "task.observation.observed_state"
                f"[{position}] must be an integer, got {value!r}"
            )
        indices.append(int(value))
    if len(indices) != len(set(indices)):
        raise ValueError(f"task.observation.observed_state contains duplicate indices: {indices}")
    return indices


def _process_noise_std(task_cfg: Mapping[str, Any]) -> np.ndarray:
    dynamics = _mapping(task_cfg.get("dynamics"), "task.dynamics")
    process_noise = dynamics.get("process_noise_std", {}) or {}
    if not isinstance(process_noise, Mapping):
        raise ValueError("task.dynamics.process_noise_std must be a mapping")
    attitude = float(process_noise.get("attitude_mrp", 0.0))
    omega = float(process_noise.get("angular_rate_rad_s", 0.0))
    bias = float(process_noise.get("gyro_bias_rad_s", 0.0))
    if min(attitude, omega, bias) < 0.0:
        raise ValueError("task.dynamics.process_noise_std values must be >= 0")
    return np.asarray([attitude, attitude, attitude, omega, omega, omega, bias, bias, bias], dtype=np.float64)


def build_replay_generated_system_model(
    *,
    task_cfg: Mapping[str, Any],
    dt_s: float,
) -> dict[str, Any]:
    observed = _observed_state_indices(task_cfg)
    if observed != [0, 1, 2, 3, 4, 5]:
        raise ValueError(
            "Phase 6G replay_generated bridge currently requires the direct-observation "
            "layout observed_state=[0,1,2,3,4,5]"
        )

    q_std = _process_noise_std(task_cfg)
    noise = _mapping(task_cfg.get("noise"), "task.noise")
    replay_measurement = _mapping(noise.get("replay_measurement"), "task.noise.replay_measurement")
    meas_std = np.asarray(
        [
            float(replay_measurement.get("attitude_mrp", 0.0)),
            float(replay_measurement.get("attitude_mrp", 0.0)),
            float(replay_measurement.get("attitude_mrp", 0.0)),
            float(replay_measurement.get("angular_rate_rad_s", 0.0)),
            float(replay_measurement.get("angular_rate_rad_s", 0.0)),
            float(replay_measurement.get("angular_rate_rad_s", 0.0)),
        ],
        dtype=np.float64,
    )
    if np.any(meas_std < 0.0):
        raise ValueError("task.noise.replay_measurement values must be >= 0")
    q_diag = np.maximum(q_std * q_std, 1.0e-8)
    r_diag = np.maximum(meas_std * meas_std, 1.0e-10)

    F = np.eye(9, dtype=np.float64)
    F[0:3, 3:6] = 0.25 * float(dt_s) * np.eye(3, dtype=np.float64)
    H = np.zeros((6, 9), dtype=np.float64)
    H[np.arange(6), np.arange(6)] = 1.0

    return {
        "schema_version": "kalmannet_tsp_system_model_v1",
        "format": "linear_F_H",
        "state_dim": 9,
        "measurement_dim": 6,
        "observed_state": list(observed),
        "F": F.astype(np.float32).tolist(),
        "H": H.astype(np.float32).tolist(),
        "Q": np.diag(q_diag).astype(np.float32).tolist(),
        "R": np.diag(r_diag).astype(np.float32).tolist(),
        "time_unit": "s",
        "attitude_representation": "MRP sigma_BN",
        "angular_rate_representation": "omega_BN_B rad/s",
        "position_frame": "visualization_only",
        "velocity_frame": "visualization_only",
    }


def _clone_suite_for_split(suite_cfg: Mapping[str, Any], split_seed: int) -> dict[str, Any]:
    cloned = copy.deepcopy(dict(suite_cfg))
    cloned["seeds"] = [int(split_seed)]
    return cloned


def _clone_task_for_split(task_cfg: Mapping[str, Any], n_test: int) -> dict[str, Any]:
    cloned = copy.deepcopy(dict(task_cfg))
    dataset_sizes = dict(cloned.get("dataset_sizes", {}) or {})
    dataset_sizes["N_train"] = 0
    dataset_sizes["N_val"] = 0
    dataset_sizes["N_test"] = int(n_test)
    cloned["dataset_sizes"] = dataset_sizes
    cloned["task_family"] = REPLAY_GENERATED_TASK_FAMILY
    cloned["system_type"] = "adcs_replay"
    return cloned


def _split_seed(
    *,
    suite_name: str,
    task_id: str,
    scenario_id: str,
    seed: int,
    split_name: str,
) -> int:
    return stable_int_seed_v0("replay_generated_split", suite_name, task_id, scenario_id, int(seed), split_name)


def _empty_split(*, T: int, x_dim: int, y_dim: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.zeros((0, T, x_dim), dtype=np.float32)
    y = np.zeros((0, T, y_dim), dtype=np.float32)
    trajectory_id = np.zeros((0,), dtype=np.int64)
    return x, y, trajectory_id


@dataclass(frozen=True)
class ReplayGeneratedDatasetV0:
    suite_name: str
    suite_version: str | None
    task_id: str
    task_name: str | None
    seed: int
    scenario_id: str
    time_s: np.ndarray
    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    trajectory_id_train: np.ndarray
    trajectory_id_val: np.ndarray
    trajectory_id_test: np.ndarray
    meta: dict[str, Any]
    system_model: dict[str, Any]
    split_payloads: dict[str, dict[str, Any]]


def _build_split(
    *,
    suite_cfg: Mapping[str, Any],
    task_cfg: Mapping[str, Any],
    base_seed: int,
    split_seed: int,
    scenario_id: str,
    split_name: str,
    n_samples: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    split_suite = _clone_suite_for_split(suite_cfg, int(split_seed))
    validated = validate_adcs_replay_task(split_suite, task_cfg, seed=int(split_seed))
    T = int(validated["sequence_length_T"])
    x_dim = int(validated["x_dim"])
    y_dim = int(validated["y_dim"])
    dt_s = float(validated["dt_s"])
    time_s = np.arange(T, dtype=np.float32) * np.float32(dt_s)

    if int(n_samples) <= 0:
        x, y, trajectory_id = _empty_split(T=T, x_dim=x_dim, y_dim=y_dim)
        meta = {
            "split": split_name,
            "scenario_id": scenario_id,
            "seed": int(base_seed),
            "generation_seed": int(split_seed),
            "time_s": time_s.tolist(),
        }
        return x, y, trajectory_id, meta

    split_task = _clone_task_for_split(task_cfg, int(n_samples))
    scenario = materialize_adcs_replay_task(split_suite, split_task, seed=int(split_seed))

    x = np.asarray(scenario.x_true, dtype=np.float32)
    y = np.asarray(scenario.y_obs, dtype=np.float32)
    trajectory_id = np.asarray(scenario.trajectory_id, dtype=np.int64)
    meta = dict(scenario.meta)
    meta["split"] = split_name
    meta["scenario_id"] = str(scenario_id)
    meta["seed"] = int(base_seed)
    meta["generation_seed"] = int(split_seed)
    meta["time_s"] = time_s.tolist()
    meta["data_mode"] = "replay_generated"
    return x, y, trajectory_id, meta


def build_replay_generated_dataset(
    suite_cfg: Mapping[str, Any],
    task_cfg: Mapping[str, Any],
    *,
    seed: int,
) -> ReplayGeneratedDatasetV0:
    validated = validate_adcs_replay_task(suite_cfg, task_cfg, seed=int(seed))
    suite_name = str(validated["suite_name"])
    task_id = str(validated["task_id"])
    task_name = validated.get("task_name")
    suite_version = validated.get("suite_version")
    T = int(validated["sequence_length_T"])
    dt_s = float(validated["dt_s"])
    time_s = np.arange(T, dtype=np.float32) * np.float32(dt_s)

    hash_fields = _scenario_hash_fields(suite_cfg, task_cfg, seed=int(seed))
    scenario_id = _scenario_id(hash_fields)

    sizes = dict(validated["dataset_sizes"])
    n_train = _positive_int(sizes.get("N_train", 0), "task.dataset_sizes.N_train", minimum=0)
    n_val = _positive_int(sizes.get("N_val", 0), "task.dataset_sizes.N_val", minimum=0)
    n_test = _positive_int(sizes.get("N_test", 0), "task.dataset_sizes.N_test", minimum=0)

    split_specs = {
        "train": n_train,
        "val": n_val,
        "test": n_test,
    }
    split_results: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]] = {}
    for split_name, n_samples in split_specs.items():
        split_seed = _split_seed(
            suite_name=suite_name,
            task_id=task_id,
            scenario_id=scenario_id,
            seed=int(seed),
            split_name=split_name,
        )
        split_results[split_name] = _build_split(
            suite_cfg=suite_cfg,
            task_cfg=task_cfg,
            base_seed=int(seed),
            split_seed=int(split_seed),
            scenario_id=scenario_id,
            split_name=split_name,
            n_samples=int(n_samples),
        )

    x_train, y_train, tid_train, meta_train = split_results["train"]
    x_val, y_val, tid_val, meta_val = split_results["val"]
    x_test, y_test, tid_test, meta_test = split_results["test"]

    system_model = build_replay_generated_system_model(
        task_cfg=task_cfg,
        dt_s=dt_s,
    )

    meta_common: dict[str, Any] = {
        "schema_version": 1,
        "schema_tag": "generator_contract_v1",
        "task_family": REPLAY_GENERATED_TASK_FAMILY,
        "suite_name": suite_name,
        "suite_version": suite_version,
        "task_id": task_id,
        "task_name": task_name,
        "seed": int(seed),
        "scenario_id": scenario_id,
        "dims": {
            "x_dim": int(validated["x_dim"]),
            "y_dim": int(validated["y_dim"]),
            "T": int(T),
        },
        "splits": {
            "train": {"N": int(n_train)},
            "val": {"N": int(n_val)},
            "test": {"N": int(n_test)},
        },
        "ssm": {
            "true": {
                "system_type": "replay_generated_adcs",
                "observed_state": list(validated["observed_indices"]),
                "F": system_model["F"],
                "H": system_model["H"],
                "Q": system_model["Q"],
                "R": system_model["R"],
                "time_s": time_s.tolist(),
            },
            "assumed": {
                "system_type": "replay_generated_adcs",
                "observed_state": list(validated["observed_indices"]),
                "F": system_model["F"],
                "H": system_model["H"],
                "Q": system_model["Q"],
                "R": system_model["R"],
                "time_s": time_s.tolist(),
            },
        },
        "mismatch": {
            "enabled": False,
            "kind": "none",
            "params": {},
        },
        "noise_schedule": {
            "enabled": False,
            "kind": "stationary",
            "q2_t": "replay_generated_q2_scalar",
            "r2_t": "replay_generated_r2_scalar",
            "SoW_t": "replay_generated_q2_over_r2",
            "SoW_hat_t": None,
            "params": {
                "q2_base": float(np.mean(np.diag(np.asarray(system_model["Q"], dtype=np.float64)))),
                "r2_base": float(np.mean(np.diag(np.asarray(system_model["R"], dtype=np.float64)))),
            },
        },
        "switching": {
            "enabled": False,
            "models": [],
            "t_change": None,
            "retrain_window": 0,
        },
        "state_schema": _jsonable(validated["state_schema"]),
        "initial_state": _jsonable(validated["initial_state"]),
        "dynamics": _jsonable(validated["dynamics"]),
        "observation": _jsonable(validated["observation"]),
        "noise": _jsonable(validated["noise"]),
        "vizard": _jsonable(validated["vizard"]),
        "replay": _jsonable(validated["replay"]),
        "data_mode": "replay_generated",
        "replay_generated": {
            "base_seed": int(seed),
            "split_seeds": {
                split_name: int(
                    _split_seed(
                        suite_name=suite_name,
                        task_id=task_id,
                        scenario_id=scenario_id,
                        seed=int(seed),
                        split_name=split_name,
                    )
                )
                for split_name in split_specs
            },
            "observed_state": list(validated["observed_indices"]),
            "time_unit": "s",
            "state_representation": "sigma_BN/omega_BN_B/gyro_bias",
        },
    }

    def _payload(x: np.ndarray, y: np.ndarray, trajectory_id: np.ndarray, meta: dict[str, Any]) -> dict[str, Any]:
        return {
            "x": np.asarray(x, dtype=np.float32),
            "y": np.asarray(y, dtype=np.float32),
            "extras": {
                "trajectory_id": np.asarray(trajectory_id, dtype=np.float32),
                "time_s": np.asarray(time_s, dtype=np.float32),
            },
            "meta": meta,
            "F": np.asarray(system_model["F"], dtype=np.float32),
            "H": np.asarray(system_model["H"], dtype=np.float32),
        }

    split_payloads = {
        "train": _payload(x_train, y_train, tid_train, meta_train),
        "val": _payload(x_val, y_val, tid_val, meta_val),
        "test": _payload(x_test, y_test, tid_test, meta_test),
    }

    x_proxy = np.concatenate(
        [arr for arr in (x_train, x_val, x_test) if int(arr.shape[0]) > 0],
        axis=0,
    ) if any(int(arr.shape[0]) > 0 for arr in (x_train, x_val, x_test)) else np.zeros((0, T, 9), dtype=np.float32)
    y_proxy = np.concatenate(
        [arr for arr in (y_train, y_val, y_test) if int(arr.shape[0]) > 0],
        axis=0,
    ) if any(int(arr.shape[0]) > 0 for arr in (y_train, y_val, y_test)) else np.zeros((0, T, 6), dtype=np.float32)

    out = GeneratorOutput(
        x=x_proxy,
        y=y_proxy,
        meta=meta_common,
        extras={INTERNAL_SPLIT_PAYLOADS_KEY: split_payloads},
    )
    out = coerce_ntd_float32_output(out)

    return ReplayGeneratedDatasetV0(
        suite_name=suite_name,
        suite_version=suite_version,
        task_id=task_id,
        task_name=task_name,
        seed=int(seed),
        scenario_id=scenario_id,
        time_s=time_s,
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        x_test=x_test,
        y_test=y_test,
        trajectory_id_train=tid_train,
        trajectory_id_val=tid_val,
        trajectory_id_test=tid_test,
        meta=out.meta,
        system_model=system_model,
        split_payloads=split_payloads,
    )


def generate_replay_generated_v0(
    *,
    suite_name: str,
    task_cfg_dict: dict[str, Any],
    scenario_cfg: dict[str, Any],
    seed: int,
    scenario_id: str,
    task_family: str = REPLAY_GENERATED_TASK_FAMILY,
) -> tuple[GeneratorOutput, np.ndarray | None, np.ndarray | None]:
    suite_cfg = {
        "suite": {"name": suite_name, "version": None},
        "seeds": [int(seed)],
        "tasks": [copy.deepcopy(task_cfg_dict)],
    }
    task_cfg = copy.deepcopy(task_cfg_dict)
    dataset = build_replay_generated_dataset(suite_cfg, task_cfg, seed=int(seed))
    meta = dict(dataset.meta)
    meta["scenario_id"] = str(scenario_id)
    meta["seed"] = int(seed)
    meta["task_family"] = str(task_family)

    if dataset.x_train.shape[0] or dataset.x_val.shape[0] or dataset.x_test.shape[0]:
        x_proxy = np.concatenate(
            [arr for arr in (dataset.x_train, dataset.x_val, dataset.x_test) if int(arr.shape[0]) > 0],
            axis=0,
        )
        y_proxy = np.concatenate(
            [arr for arr in (dataset.y_train, dataset.y_val, dataset.y_test) if int(arr.shape[0]) > 0],
            axis=0,
        )
    else:
        x_proxy = np.zeros((0, int(dataset.time_s.shape[0]), 9), dtype=np.float32)
        y_proxy = np.zeros((0, int(dataset.time_s.shape[0]), 6), dtype=np.float32)

    proxy = GeneratorOutput(
        x=np.asarray(x_proxy, dtype=np.float32),
        y=np.asarray(y_proxy, dtype=np.float32),
        meta=meta,
        extras={INTERNAL_SPLIT_PAYLOADS_KEY: dict(dataset.split_payloads)},
    )
    proxy = coerce_ntd_float32_output(proxy)
    return proxy, np.asarray(dataset.system_model["F"], dtype=np.float32), np.asarray(dataset.system_model["H"], dtype=np.float32)


def load_suite_task(
    suite_yaml: str | Path,
    task_id: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    suite_cfg = load_suite_yaml(suite_yaml)
    task_cfg = select_task_from_suite(suite_cfg, task_id)
    return suite_cfg, task_cfg


def save_replay_generated_cache(
    *,
    suite_cfg: Mapping[str, Any],
    task_cfg: Mapping[str, Any],
    seed: int,
    cache_root: Path,
    scenario_cfg: Mapping[str, Any] | None = None,
) -> DatasetArtifactsV0:
    scenario_cfg = dict(scenario_cfg or {})
    suite_name = str(_mapping(suite_cfg.get("suite"), "suite.suite").get("name"))
    hash_fields = _scenario_hash_fields(suite_cfg, task_cfg, seed=int(seed))
    scenario_id = _scenario_id(hash_fields)
    out_dir = _task_cache_dir(cache_root, suite_name, str(task_cfg.get("task_id")), scenario_id, int(seed))
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset = build_replay_generated_dataset(suite_cfg, task_cfg, seed=int(seed))

    split_cfg = _mapping(task_cfg.get("dataset_sizes"), "task.dataset_sizes")
    task_cfg_obj = dict(task_cfg)
    task_cfg_obj["task_family"] = REPLAY_GENERATED_TASK_FAMILY
    task_cfg_obj["system_type"] = "adcs_replay"

    def _save_split(name: str, x: np.ndarray, y: np.ndarray, tid: np.ndarray, meta: dict[str, Any]) -> Path:
        path = out_dir / f"{name}.npz"
        meta_obj = enforce_meta_v1(
            meta,
            task_cfg=None,
            split_cfg=None,
            x=x,
            y=y,
            extras={
                "trajectory_id": np.asarray(tid, dtype=np.float32),
                "time_s": np.asarray(dataset.time_s, dtype=np.float32),
                "F": np.asarray(dataset.system_model["F"], dtype=np.float32),
                "H": np.asarray(dataset.system_model["H"], dtype=np.float32),
            },
        )
        meta_obj.update({
            "schema_version": 1,
            "schema_tag": "generator_contract_v1",
            "task_family": REPLAY_GENERATED_TASK_FAMILY,
            "suite_name": dataset.suite_name,
            "suite_version": dataset.suite_version,
            "task_id": dataset.task_id,
            "task_name": dataset.task_name,
            "seed": int(seed),
            "scenario_id": scenario_id,
            "dims": {"x_dim": 9, "y_dim": 6, "T": int(dataset.time_s.shape[0])},
            "splits": {
                "train": {"N": int(dataset.x_train.shape[0])},
                "val": {"N": int(dataset.x_val.shape[0])},
                "test": {"N": int(dataset.x_test.shape[0])},
                "active_split": name,
            },
            "ssm": dataset.meta["ssm"],
            "mismatch": dataset.meta["mismatch"],
            "noise_schedule": dataset.meta["noise_schedule"],
            "switching": dataset.meta["switching"],
        })
        meta_obj["split"] = name
        validate_artifacts(x, y, meta_obj, strict=True)
        save_npz_split_v0(
            path=path,
            x=x,
            y=y,
            u=None,
            F=np.asarray(dataset.system_model["F"], dtype=np.float32),
            H=np.asarray(dataset.system_model["H"], dtype=np.float32),
            meta=meta_obj,
            extras={
                "trajectory_id": np.asarray(tid, dtype=np.float32),
                "time_s": np.asarray(dataset.time_s, dtype=np.float32),
            },
        )
        return path

    train_path = _save_split("train", dataset.x_train, dataset.y_train, dataset.trajectory_id_train, dict(dataset.meta))
    val_path = _save_split("val", dataset.x_val, dataset.y_val, dataset.trajectory_id_val, dict(dataset.meta))
    test_path = _save_split("test", dataset.x_test, dataset.y_test, dataset.trajectory_id_test, dict(dataset.meta))

    return DatasetArtifactsV0(
        format_version="0.1",
        canonical_layout=CANONICAL_LAYOUT_V0,
        suite_name=dataset.suite_name,
        task_id=dataset.task_id,
        scenario_id=scenario_id,
        seed=int(seed),
        cache_dir=out_dir,
        train=DatasetSplitV0(path=train_path, split="train"),
        val=DatasetSplitV0(path=val_path, split="val"),
        test=DatasetSplitV0(path=test_path, split="test"),
        meta_common=dict(dataset.meta),
    )
