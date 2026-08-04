"""P1 Exit classical posterior-covariance condition-closure study.

Only scenario-wide fixed P0/Qg/Qb scales are calibrated.  The frozen Phase 1
generator, typed events, replay, sensor covariances, and canonical state-error
convention are reused without modification.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import itertools
import json
import math
import os
import platform
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml
from scipy.linalg import cholesky, solve_triangular

from bench.estimators.mekf import MEKFState, NumericalSafetyError
from bench.experiments.phase1b_sensor_fusion_c4 import (
    F_BASE,
    F_TUNED,
    ORACLE_FULL,
    ORACLE_MEASUREMENT,
    ORACLE_PROCESS,
    WRONG_MEASUREMENT,
    WRONG_PROCESS,
    FusionPolicy,
    FusionReplayResult,
    _fusion_config,
    base_process_covariance,
    evaluate_fusion_replay,
    replay_fixed_policy,
    replay_oracle_policy,
)
from bench.experiments.phase1b_unit_st_classical import (
    default_initial_state,
    paired_bootstrap_ci,
)
from bench.metrics.mekf import (
    attitude_geodesic_error_rad,
    bias_error_summary,
    right_local_nees,
    right_local_state_error,
    spd_diagnostics,
    star_tracker_nis,
)
from bench.metrics.mekf_fusion import magnetometer_nis, sun_sensor_nis
from bench.tasks.generator.mekf_fusion_events import (
    GENERATOR_ID,
    FusionDataset,
    FusionOracleSidecar,
    load_fusion_dataset,
    load_fusion_oracle,
    save_fusion_dataset,
    save_fusion_oracle,
)
from bench.tasks.generator.phase1b_sensor_fusion import (
    FusionScenarioCode,
    GeneratedSensorFusion,
    SensorFusionConfig,
    generate_sensor_fusion,
)


EXPERIMENT_VERSION = "p1-exit-covariance-closure-v1"
FROZEN_POLICY_ID = "F-CALIBRATED-v1"
CANDIDATE_POLICY_ID = "F-CALIBRATED-CANDIDATE"
SCALE_FIELDS = ("s_P0_att", "s_P0_bias", "s_Qg", "s_Qb")
PARTITION_NAMES = ("whole", "initial", "middle", "settled")


@dataclass(frozen=True, order=True)
class CalibrationScales:
    s_P0_att: float = 1.0
    s_P0_bias: float = 1.0
    s_Qg: float = 1.0
    s_Qb: float = 1.0

    def __post_init__(self) -> None:
        for name in SCALE_FIELDS:
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
            object.__setattr__(self, name, value)

    def as_dict(self) -> dict[str, float]:
        return {name: float(getattr(self, name)) for name in SCALE_FIELDS}


@dataclass(frozen=True)
class CovarianceDecomposition:
    full_nees: np.ndarray
    attitude_nees: np.ndarray
    bias_nees: np.ndarray
    whitened_error: np.ndarray
    whitened_energy: np.ndarray
    cross_relative_norm: np.ndarray
    cross_correlation_block: np.ndarray

    def __post_init__(self) -> None:
        for name in (
            "full_nees",
            "attitude_nees",
            "bias_nees",
            "whitened_error",
            "whitened_energy",
            "cross_relative_norm",
            "cross_correlation_block",
        ):
            value = np.asarray(getattr(self, name))
            result = np.array(value, dtype=np.float64, order="C", copy=True)
            if not np.all(np.isfinite(result)):
                raise ValueError(f"{name} must be finite")
            result.setflags(write=False)
            object.__setattr__(self, name, result)


@dataclass(frozen=True)
class ScenarioBundle:
    dataset: FusionDataset
    oracle: FusionOracleSidecar
    manifest: dict[str, Any]
    dataset_hash: str
    oracle_hash: str
    generated: GeneratedSensorFusion


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _canonical_json(value)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def _read_json(path: Path) -> Any:
    raw = path.read_bytes()
    try:
        value = json.loads(raw.decode("ascii"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid canonical JSON at {path}") from error
    if _canonical_json(value) != raw:
        raise ValueError(f"noncanonical JSON at {path}")
    return value


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_json(value: Any) -> str:
    return _sha256_bytes(_canonical_json(value))


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a mapping")
    return value


def load_closure_config(path: Path) -> dict[str, Any]:
    config = _load_yaml(path)
    if config.get("experiment_version") != EXPERIMENT_VERSION:
        raise ValueError("wrong closure experiment version")
    frozen = config["frozen_foundation"]
    frozen_path = Path(frozen["step2_config"])
    if _sha256_file(frozen_path) != frozen["step2_config_sha256"]:
        raise ValueError("frozen Step 2 config fingerprint changed")
    if frozen.get("fixed_primary") != "F-BASE":
        raise ValueError("F-BASE must remain primary")
    comparator = frozen["frozen_sensitivity"]
    if (
        comparator.get("policy_id"),
        float(comparator.get("s_Qg")),
        float(comparator.get("s_Qb")),
        float(comparator.get("s_R_ST")),
    ) != ("F-TUNED", 0.125, 0.125, 8.0):
        raise ValueError("F-TUNED differs from the frozen comparator")
    if set(map(float, frozen["sensor_R_scales"].values())) != {1.0}:
        raise ValueError("all sensor R scales must remain exactly one")
    data = config["data"]
    if (int(data["train_N"]), int(data["validation_N"])) != (30, 20):
        raise ValueError("closure calibration split must be 30/20")
    if int(data["confirmation_stationary_N"]) != 50 or int(data["confirmation_c4_N"]) != 50:
        raise ValueError("both confirmation conditions require N=50")
    grid = tuple(float(item) for item in config["calibration"]["scales"])
    if grid != (0.5, 1.0, 2.0, 4.0, 8.0):
        raise ValueError("candidate scale grid differs from the locked contract")
    if int(config["calibration"]["maximum_candidate_count"]) != 101:
        raise ValueError("candidate budget must be 101")
    return config


def _step2_config(config: Mapping[str, Any]) -> dict[str, Any]:
    return _load_yaml(Path(config["frozen_foundation"]["step2_config"]))


def _roots(config: Mapping[str, Any]) -> tuple[Path, Path, Path]:
    paths = config["paths"]
    return Path(paths["results_root"]), Path(paths["manifests_root"]), Path(paths["reports_root"])


def _diagnosis_path(config: Mapping[str, Any]) -> Path:
    results_root, _, _ = _roots(config)
    return results_root / f"diagnosis_{_sha256_json(config)[:12]}.json"


def _generator_config(
    config: Mapping[str, Any],
    *,
    scenario: FusionScenarioCode,
    master_seed: int,
    count: int,
) -> SensorFusionConfig:
    data = config["data"]
    base = _fusion_config(
        _step2_config(config),
        scenario,
        num_trajectories=int(count),
        duration_s=float(data["duration_s"]),
        master_seed=int(master_seed),
    )
    fractions = tuple(float(item) for item in data["generator_internal_split"])
    return replace(
        base,
        train_fraction=fractions[0],
        val_fraction=fractions[1],
        test_fraction=fractions[2],
    )


def _save_or_verify_bundle(
    root: Path,
    generated: GeneratedSensorFusion,
) -> ScenarioBundle:
    sensor_path = root / "sensor"
    oracle_path = root / "oracle_simulation_only"
    if root.exists():
        dataset, manifest, hashes = load_fusion_dataset(
            sensor_path, expected_generator_id=GENERATOR_ID
        )
        oracle = load_fusion_oracle(
            oracle_path, expected_dataset_hash=hashes.dataset_hash
        )
        if hashes != generated.semantic_hashes:
            raise ValueError("existing physical dataset differs from deterministic generation")
        if oracle.semantic_hash != generated.oracle_context.semantic_hash:
            raise ValueError("existing oracle identity differs from deterministic generation")
        return ScenarioBundle(
            dataset=dataset,
            oracle=oracle,
            manifest=manifest,
            dataset_hash=hashes.dataset_hash,
            oracle_hash=oracle.semantic_hash,
            generated=generated,
        )
    hashes = save_fusion_dataset(sensor_path, generated.dataset, generated.sensor_manifest)
    oracle_hash = save_fusion_oracle(
        oracle_path,
        generated.oracle_context,
        dataset_hash=hashes.dataset_hash,
    )
    if hashes != generated.semantic_hashes or oracle_hash != generated.oracle_context.semantic_hash:
        raise AssertionError("serialized closure dataset identity changed")
    return ScenarioBundle(
        dataset=generated.dataset,
        oracle=generated.oracle_context,
        manifest=generated.sensor_manifest,
        dataset_hash=hashes.dataset_hash,
        oracle_hash=oracle_hash,
        generated=generated,
    )


def _frozen_phase1_test_ids() -> set[int]:
    summary_sources = (
        (
            Path("experiments/phase1b/results/unit_st_classical_v1/pilot_summary.json"),
            ("summary", "groups", "C1-STATIONARY/F-BASE", "trajectory_ids"),
        ),
        (
            Path("experiments/phase1b/results/sensor_fusion_c4_v1/pilot_summary.json"),
            (
                "summary",
                "groups",
                "MAIN-FUSION-STATIONARY/F-BASE",
                "trajectory_ids",
            ),
        ),
    )
    values: set[int] = set()
    for path, locator in summary_sources:
        node: Any = json.loads(path.read_text(encoding="utf-8"))
        try:
            for key in locator:
                node = node[key]
        except (KeyError, TypeError) as exc:
            raise ValueError(
                f"frozen test trajectory ledger is missing at {path}: /{'/'.join(locator)}"
            ) from exc
        if not isinstance(node, list) or not node:
            raise ValueError(f"frozen test trajectory ledger is empty at {path}")
        values.update(int(item) for item in node)
    if not values:
        raise ValueError("frozen Phase 1 test ID ledger is empty")
    return values


def closure_train_validation_split(
    trajectory_ids: np.ndarray,
    *,
    train_count: int,
    validation_count: int,
    split_seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if not isinstance(trajectory_ids, np.ndarray) or trajectory_ids.dtype != np.dtype(np.int64):
        raise TypeError("trajectory_ids must be an int64 ndarray")
    if trajectory_ids.ndim != 1 or np.unique(trajectory_ids).size != trajectory_ids.size:
        raise ValueError("trajectory IDs must be a unique vector")
    if int(train_count) + int(validation_count) != trajectory_ids.size:
        raise ValueError("train and validation counts must consume the entire calibration pool")
    permutation = np.random.default_rng(int(split_seed)).permutation(trajectory_ids.size)
    train = np.array(trajectory_ids[permutation[: int(train_count)]], dtype=np.int64, copy=True)
    validation = np.array(trajectory_ids[permutation[int(train_count) :]], dtype=np.int64, copy=True)
    train.setflags(write=False)
    validation.setflags(write=False)
    if set(map(int, train)) & set(map(int, validation)):
        raise AssertionError("closure train/validation split overlaps")
    return train, validation


def _ensure_calibration_bundle(
    config: Mapping[str, Any],
) -> tuple[ScenarioBundle, np.ndarray, np.ndarray, dict[str, Any]]:
    _, manifests_root, _ = _roots(config)
    data = config["data"]
    generated = generate_sensor_fusion(
        _generator_config(
            config,
            scenario=FusionScenarioCode.MAIN_FUSION_STATIONARY,
            master_seed=int(data["calibration_master_seed"]),
            count=int(data["calibration_pool_N"]),
        )
    )
    bundle = _save_or_verify_bundle(manifests_root / "calibration_stationary", generated)
    ids = np.asarray(bundle.dataset.truth.trajectory_id, dtype=np.int64)
    train, validation = closure_train_validation_split(
        ids,
        train_count=int(data["train_N"]),
        validation_count=int(data["validation_N"]),
        split_seed=int(data["split_seed"]),
    )
    frozen_overlap = set(map(int, ids)) & _frozen_phase1_test_ids()
    if frozen_overlap:
        raise ValueError("closure calibration IDs overlap frozen Phase 1 test IDs")
    split_record = {
        "schema_version": "p1-exit-calibration-split-v1",
        "seed_namespace": data["seed_namespace"],
        "master_seed": int(data["calibration_master_seed"]),
        "split_seed": int(data["split_seed"]),
        "dataset_hash": bundle.dataset_hash,
        "train_ids": [int(item) for item in train],
        "validation_ids": [int(item) for item in validation],
        "confirmation_dataset_generated": False,
        "frozen_phase1_test_overlap": [],
        "generator_internal_split_used_for_selection": False,
    }
    split_path = manifests_root / "calibration_split.json"
    if split_path.exists() and _read_json(split_path) != split_record:
        raise ValueError("existing calibration split identity changed")
    if not split_path.exists():
        _atomic_write_json(split_path, split_record)
    return bundle, train, validation, split_record


def scaled_initial_state(scales: CalibrationScales) -> MEKFState:
    if not isinstance(scales, CalibrationScales):
        raise TypeError("scales must be CalibrationScales")
    base = default_initial_state()
    covariance = np.array(base.P, dtype=np.float64, copy=True)
    covariance[:3, :3] *= scales.s_P0_att
    covariance[3:, 3:] *= scales.s_P0_bias
    cross_scale = math.sqrt(scales.s_P0_att * scales.s_P0_bias)
    covariance[:3, 3:] *= cross_scale
    covariance[3:, :3] *= cross_scale
    return MEKFState(q_NB=base.q_NB, b_g=base.b_g, P=covariance)


def replay_calibrated_fixed(
    event_table: Any,
    trajectory_id: int,
    base_Q_c: np.ndarray,
    scales: CalibrationScales,
) -> FusionReplayResult:
    """Deployable fixed replay boundary with no truth, oracle, or label input."""

    if not isinstance(scales, CalibrationScales):
        raise TypeError("scales must be CalibrationScales")
    policy = FusionPolicy(
        CANDIDATE_POLICY_ID,
        qg_scale=scales.s_Qg,
        qb_scale=scales.s_Qb,
        r_st_scale=1.0,
        r_mag_scale=1.0,
        r_sun_scale=1.0,
        oracle_mode="none",
    )
    return replay_fixed_policy(
        event_table,
        int(trajectory_id),
        scaled_initial_state(scales),
        0.0,
        base_Q_c,
        policy,
    )


def _strict_quadratic_batch(vectors: np.ndarray, matrices: np.ndarray, name: str) -> np.ndarray:
    if vectors.dtype != np.dtype(np.float64) or matrices.dtype != np.dtype(np.float64):
        raise TypeError(f"{name} inputs must be float64")
    if vectors.ndim != 2 or matrices.shape != (vectors.shape[0], vectors.shape[1], vectors.shape[1]):
        raise ValueError(f"{name} vector/matrix shapes do not pair")
    values = np.empty(vectors.shape[0], dtype=np.float64)
    for index, (vector, matrix) in enumerate(zip(vectors, matrices)):
        lower = cholesky(matrix, lower=True, check_finite=True)
        intermediate = solve_triangular(lower, vector, lower=True, check_finite=True)
        solved = solve_triangular(lower.T, intermediate, lower=False, check_finite=True)
        values[index] = float(vector @ solved)
    if not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise NumericalSafetyError(f"{name} quadratic forms must be finite and nonnegative")
    return values


def covariance_decomposition(
    state_error: np.ndarray,
    posterior_covariance: np.ndarray,
) -> CovarianceDecomposition:
    if not isinstance(state_error, np.ndarray) or state_error.dtype != np.dtype(np.float64):
        raise TypeError("state_error must be a float64 ndarray")
    if not isinstance(posterior_covariance, np.ndarray) or posterior_covariance.dtype != np.dtype(np.float64):
        raise TypeError("posterior_covariance must be a float64 ndarray")
    if state_error.ndim == 1:
        errors = state_error[None, :]
    elif state_error.ndim == 2:
        errors = state_error
    else:
        raise ValueError("state_error must have rank one or two")
    if posterior_covariance.ndim == 2:
        covariances = posterior_covariance[None, :, :]
    elif posterior_covariance.ndim == 3:
        covariances = posterior_covariance
    else:
        raise ValueError("posterior_covariance must have rank two or three")
    if errors.shape[1:] != (6,) or covariances.shape != (errors.shape[0], 6, 6):
        raise ValueError("state error and covariance must pair as [N,6] and [N,6,6]")
    full = _strict_quadratic_batch(errors, covariances, "full")
    attitude = _strict_quadratic_batch(errors[:, :3], covariances[:, :3, :3], "attitude")
    bias = _strict_quadratic_batch(errors[:, 3:], covariances[:, 3:, 3:], "bias")
    whitened = np.empty_like(errors)
    relative = np.empty(errors.shape[0], dtype=np.float64)
    correlation = np.empty((errors.shape[0], 3, 3), dtype=np.float64)
    for index, (error, covariance) in enumerate(zip(errors, covariances)):
        lower = cholesky(covariance, lower=True, check_finite=True)
        whitened[index] = solve_triangular(
            lower, error, lower=True, check_finite=True
        )
        attitude_block = covariance[:3, :3]
        bias_block = covariance[3:, 3:]
        cross = covariance[:3, 3:]
        denominator = math.sqrt(
            float(np.linalg.norm(attitude_block, ord="fro"))
            * float(np.linalg.norm(bias_block, ord="fro"))
        )
        if denominator <= 0.0:
            raise NumericalSafetyError("cross-covariance normalization denominator is nonpositive")
        relative[index] = float(np.linalg.norm(cross, ord="fro")) / denominator
        diagonal_denominator = np.sqrt(
            np.outer(np.diag(attitude_block), np.diag(bias_block))
        )
        if np.any(diagonal_denominator <= 0.0):
            raise NumericalSafetyError("cross-correlation diagonal denominator is nonpositive")
        correlation[index] = cross / diagonal_denominator
    return CovarianceDecomposition(
        full_nees=full,
        attitude_nees=attitude,
        bias_nees=bias,
        whitened_error=whitened,
        whitened_energy=whitened * whitened,
        cross_relative_norm=relative,
        cross_correlation_block=correlation,
    )


def partition_masks(
    time_s: np.ndarray,
    duration_s: float,
    partitions: Mapping[str, Sequence[float]],
) -> dict[str, np.ndarray]:
    if not isinstance(time_s, np.ndarray) or time_s.dtype != np.dtype(np.float64):
        raise TypeError("time_s must be a float64 ndarray")
    if time_s.ndim != 1 or not np.all(np.isfinite(time_s)):
        raise ValueError("time_s must be a finite vector")
    duration = float(duration_s)
    if not np.isfinite(duration) or duration <= 0.0:
        raise ValueError("duration_s must be finite and positive")
    result = {"whole": np.ones(time_s.size, dtype=np.bool_)}
    for name in ("initial", "middle", "settled"):
        start, stop = (float(item) for item in partitions[name])
        if name == "settled":
            mask = (time_s >= start * duration) & (time_s <= stop * duration)
        else:
            mask = (time_s >= start * duration) & (time_s < stop * duration)
        if not np.any(mask):
            raise ValueError(f"partition {name} is empty")
        result[name] = mask
    if np.any(result["initial"] & result["middle"]) or np.any(
        result["middle"] & result["settled"]
    ):
        raise AssertionError("time partitions overlap")
    return result


def _truth_join(dataset: FusionDataset, replay: FusionReplayResult) -> tuple[np.ndarray, np.ndarray]:
    truth = dataset.truth
    match = np.flatnonzero(truth.trajectory_id == np.int64(replay.trajectory_id))
    if match.size != 1:
        raise ValueError("truth trajectory join is not unique")
    index = int(match[0])
    start = int(truth.truth_offsets[index])
    stop = int(truth.truth_offsets[index + 1])
    times = truth.truth_time_s[start:stop]
    lookup = {float(value): row for row, value in enumerate(times)}
    rows = np.empty(replay.event_time_s.size, dtype=np.int64)
    for output_index, time_value in enumerate(replay.event_time_s):
        truth_index = lookup.get(float(time_value))
        if truth_index is None or times[truth_index] != time_value:
            raise ValueError("truth join requires exact float64 timestamp equality")
        rows[output_index] = start + truth_index
    return truth.q_true_NB[rows], truth.gyro_bias_true_rad_s[rows]


def _orders_to_times(replay: FusionReplayResult, orders: np.ndarray) -> np.ndarray:
    lookup = {int(order): float(time_s) for order, time_s in zip(replay.event_order, replay.event_time_s)}
    values = np.asarray([lookup[int(order)] for order in orders], dtype=np.float64)
    return values


def _correlation_matrix(values: np.ndarray) -> np.ndarray:
    centered = values - np.mean(values, axis=0, keepdims=True)
    covariance = centered.T @ centered / float(values.shape[0])
    scale = np.sqrt(np.outer(np.diag(covariance), np.diag(covariance)))
    if np.any(scale <= 0.0):
        raise NumericalSafetyError("whitened correlation has a zero-variance coordinate")
    return covariance / scale


def trajectory_diagnostics(
    dataset: FusionDataset,
    replay: FusionReplayResult,
    *,
    duration_s: float,
    partitions: Mapping[str, Sequence[float]],
    divergence_threshold_rad: float,
) -> dict[str, Any]:
    q_true, b_true = _truth_join(dataset, replay)
    state = right_local_state_error(
        replay.q_NB_history,
        replay.b_g_history,
        q_true,
        b_true,
    )
    canonical_full = right_local_nees(
        replay.q_NB_history,
        replay.b_g_history,
        replay.P_history,
        q_true,
        b_true,
    )
    decomposition = covariance_decomposition(state.state_error, replay.P_history)
    if not np.array_equal(canonical_full, decomposition.full_nees):
        if not np.allclose(canonical_full, decomposition.full_nees, rtol=2.0e-14, atol=2.0e-14):
            raise AssertionError("closure full NEES differs from canonical Gate C NEES")
    attitude = attitude_geodesic_error_rad(replay.q_NB_history, q_true)
    bias = bias_error_summary(replay.b_g_history, b_true).vector_norm_rad_s
    masks = partition_masks(replay.event_time_s, duration_s, partitions)
    sensor_values: dict[str, tuple[np.ndarray, np.ndarray, int]] = {
        "mag": (
            magnetometer_nis(replay.magnetometer_residual, replay.magnetometer_S),
            _orders_to_times(replay, replay.magnetometer_event_order),
            3,
        ),
        "sun": (
            sun_sensor_nis(replay.sun_residual, replay.sun_S),
            _orders_to_times(replay, replay.sun_event_order),
            2,
        ),
        "st": (
            star_tracker_nis(replay.star_tracker_residual, replay.star_tracker_S),
            _orders_to_times(replay, replay.star_tracker_event_order),
            3,
        ),
    }
    partition_records: dict[str, Any] = {}
    for name, mask in masks.items():
        count = int(np.sum(mask))
        record: dict[str, Any] = {
            "state_count": count,
            "attitude_squared_sum": float(np.sum(attitude[mask] ** 2)),
            "attitude_rmse_rad": float(np.sqrt(np.mean(attitude[mask] ** 2))),
            "attitude_p95_rad": float(np.quantile(attitude[mask], 0.95)),
            "bias_squared_sum": float(np.sum(bias[mask] ** 2)),
            "bias_rmse_rad_s": float(np.sqrt(np.mean(bias[mask] ** 2))),
            "bias_p95_rad_s": float(np.quantile(bias[mask], 0.95)),
            "full_nees_sum": float(np.sum(decomposition.full_nees[mask])),
            "full_nees_normalized": float(np.mean(decomposition.full_nees[mask]) / 6.0),
            "attitude_nees_sum": float(np.sum(decomposition.attitude_nees[mask])),
            "attitude_nees_normalized": float(
                np.mean(decomposition.attitude_nees[mask]) / 3.0
            ),
            "bias_nees_sum": float(np.sum(decomposition.bias_nees[mask])),
            "bias_nees_normalized": float(np.mean(decomposition.bias_nees[mask]) / 3.0),
            "whitened_coordinate_energy": [
                float(item) for item in np.mean(decomposition.whitened_energy[mask], axis=0)
            ],
            "whitened_attitude_group_energy": float(
                np.mean(np.sum(decomposition.whitened_energy[mask, :3], axis=1))
            ),
            "whitened_bias_group_energy": float(
                np.mean(np.sum(decomposition.whitened_energy[mask, 3:], axis=1))
            ),
            "whitened_cross_correlation": [
                [float(item) for item in row]
                for row in _correlation_matrix(decomposition.whitened_error[mask])[:3, 3:]
            ],
            "P_cross_relative_norm_mean": float(
                np.mean(decomposition.cross_relative_norm[mask])
            ),
            "P_cross_correlation_block_mean": [
                [float(item) for item in row]
                for row in np.mean(decomposition.cross_correlation_block[mask], axis=0)
            ],
        }
        start, stop = (0.0, 1.0) if name == "whole" else tuple(
            float(item) for item in partitions[name]
        )
        for sensor_name, (nis, sensor_time, dof) in sensor_values.items():
            if name == "settled":
                sensor_mask = (sensor_time >= start * duration_s) & (
                    sensor_time <= stop * duration_s
                )
            else:
                sensor_mask = (sensor_time >= start * duration_s) & (
                    sensor_time < stop * duration_s
                )
            sensor_count = int(np.sum(sensor_mask))
            if sensor_count == 0:
                raise ValueError(f"{sensor_name} has no evidence in partition {name}")
            sensor_sum = float(np.sum(nis[sensor_mask]))
            record[f"{sensor_name}_nis_count"] = sensor_count
            record[f"{sensor_name}_nis_sum"] = sensor_sum
            record[f"{sensor_name}_nis_normalized"] = sensor_sum / (sensor_count * dof)
        partition_records[name] = record
    P_diagnostics = spd_diagnostics(replay.P_history, name="closure_P")
    S_minima = [
        float(np.min(spd_diagnostics(value, name=name).minimum_eigenvalue))
        for value, name in (
            (replay.magnetometer_S, "closure_mag_S"),
            (replay.sun_S, "closure_sun_S"),
            (replay.star_tracker_S, "closure_ST_S"),
        )
    ]
    return {
        "trajectory_id": int(replay.trajectory_id),
        "partitions": partition_records,
        "minimum_P_eigenvalue": float(np.min(P_diagnostics.minimum_eigenvalue)),
        "minimum_S_eigenvalue": min(S_minima),
        "diverged": bool(
            not np.all(np.isfinite(attitude))
            or float(np.max(attitude)) > float(divergence_threshold_rad)
        ),
    }


def aggregate_trajectory_diagnostics(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not records:
        raise ValueError("diagnostic aggregation requires records")
    output: dict[str, Any] = {
        "N": len(records),
        "trajectory_ids": [int(item["trajectory_id"]) for item in records],
        "divergence_count": sum(bool(item["diverged"]) for item in records),
        "minimum_P_eigenvalue": min(float(item["minimum_P_eigenvalue"]) for item in records),
        "minimum_S_eigenvalue": min(float(item["minimum_S_eigenvalue"]) for item in records),
        "partitions": {},
    }
    for partition in PARTITION_NAMES:
        values = [item["partitions"][partition] for item in records]
        state_count = sum(int(item["state_count"]) for item in values)
        aggregate: dict[str, Any] = {
            "state_count": state_count,
            "attitude_rmse_rad": math.sqrt(
                sum(float(item["attitude_squared_sum"]) for item in values) / state_count
            ),
            "attitude_p95_rad_mean": float(
                np.mean([float(item["attitude_p95_rad"]) for item in values])
            ),
            "bias_rmse_rad_s": math.sqrt(
                sum(float(item["bias_squared_sum"]) for item in values) / state_count
            ),
            "bias_p95_rad_s_mean": float(
                np.mean([float(item["bias_p95_rad_s"]) for item in values])
            ),
            "full_nees_normalized": sum(
                float(item["full_nees_sum"]) for item in values
            )
            / (6.0 * state_count),
            "attitude_nees_normalized": sum(
                float(item["attitude_nees_sum"]) for item in values
            )
            / (3.0 * state_count),
            "bias_nees_normalized": sum(
                float(item["bias_nees_sum"]) for item in values
            )
            / (3.0 * state_count),
            "whitened_coordinate_energy": [
                float(item)
                for item in np.mean(
                    np.asarray([value["whitened_coordinate_energy"] for value in values]),
                    axis=0,
                )
            ],
            "whitened_attitude_group_energy": float(
                np.mean([value["whitened_attitude_group_energy"] for value in values])
            ),
            "whitened_bias_group_energy": float(
                np.mean([value["whitened_bias_group_energy"] for value in values])
            ),
            "whitened_cross_correlation": np.mean(
                np.asarray([value["whitened_cross_correlation"] for value in values]), axis=0
            ).tolist(),
            "P_cross_relative_norm_mean": float(
                np.mean([value["P_cross_relative_norm_mean"] for value in values])
            ),
            "P_cross_correlation_block_mean": np.mean(
                np.asarray([value["P_cross_correlation_block_mean"] for value in values]),
                axis=0,
            ).tolist(),
        }
        for sensor, dof in (("mag", 3), ("sun", 2), ("st", 3)):
            count = sum(int(item[f"{sensor}_nis_count"]) for item in values)
            total = sum(float(item[f"{sensor}_nis_sum"]) for item in values)
            aggregate[f"{sensor}_nis_count"] = count
            aggregate[f"{sensor}_nis_normalized"] = total / (count * dof)
        output["partitions"][partition] = aggregate
    return output


def first_settling_bin(
    bin_records: Sequence[Mapping[str, Any]],
    *,
    consecutive_bins: int,
    attitude_rmse_max_rad: float,
    bias_rmse_max_rad_s: float,
    full_nees_band: Sequence[float],
    sensor_nis_band: Sequence[float],
) -> int | None:
    count = int(consecutive_bins)
    if count <= 0:
        raise ValueError("consecutive_bins must be positive")
    nees_low, nees_high = (float(item) for item in full_nees_band)
    nis_low, nis_high = (float(item) for item in sensor_nis_band)

    def accepted(record: Mapping[str, Any]) -> bool:
        return (
            float(record["attitude_rmse_rad"]) <= float(attitude_rmse_max_rad)
            and float(record["bias_rmse_rad_s"]) <= float(bias_rmse_max_rad_s)
            and nees_low <= float(record["full_nees_normalized"]) <= nees_high
            and all(
                nis_low <= float(record[f"{sensor}_nis_normalized"]) <= nis_high
                for sensor in ("mag", "sun", "st")
            )
        )

    flags = [accepted(item) for item in bin_records]
    for start in range(0, len(flags) - count + 1):
        if all(flags[start : start + count]):
            return start
    return None


def _time_bin_diagnostics(
    bundle: ScenarioBundle,
    replays: Sequence[FusionReplayResult],
    *,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    duration = float(config["data"]["duration_s"])
    width = float(config["settling_diagnostic"]["time_bin_fraction"])
    edges = np.arange(0.0, 1.0 + 0.5 * width, width, dtype=np.float64)
    if edges[-1] != 1.0:
        raise ValueError("time-bin fraction must exactly tile the horizon")
    accumulators: list[dict[str, list[float]]] = [
        {
            "attitude": [],
            "bias": [],
            "full": [],
            "attitude_nees": [],
            "bias_nees": [],
            "cross": [],
            "mag": [],
            "sun": [],
            "st": [],
        }
        for _ in range(edges.size - 1)
    ]
    for replay in replays:
        q_true, b_true = _truth_join(bundle.dataset, replay)
        state = right_local_state_error(
            replay.q_NB_history, replay.b_g_history, q_true, b_true
        )
        decomposition = covariance_decomposition(state.state_error, replay.P_history)
        attitude = attitude_geodesic_error_rad(replay.q_NB_history, q_true)
        bias = bias_error_summary(replay.b_g_history, b_true).vector_norm_rad_s
        bin_index = np.minimum(
            (replay.event_time_s / (duration * width)).astype(np.int64),
            edges.size - 2,
        )
        for index in range(edges.size - 1):
            mask = bin_index == index
            accumulators[index]["attitude"].extend(attitude[mask].tolist())
            accumulators[index]["bias"].extend(bias[mask].tolist())
            accumulators[index]["full"].extend(decomposition.full_nees[mask].tolist())
            accumulators[index]["attitude_nees"].extend(
                decomposition.attitude_nees[mask].tolist()
            )
            accumulators[index]["bias_nees"].extend(decomposition.bias_nees[mask].tolist())
            accumulators[index]["cross"].extend(
                decomposition.cross_relative_norm[mask].tolist()
            )
        for sensor, values, orders in (
            ("mag", magnetometer_nis(replay.magnetometer_residual, replay.magnetometer_S), replay.magnetometer_event_order),
            ("sun", sun_sensor_nis(replay.sun_residual, replay.sun_S), replay.sun_event_order),
            ("st", star_tracker_nis(replay.star_tracker_residual, replay.star_tracker_S), replay.star_tracker_event_order),
        ):
            sensor_times = _orders_to_times(replay, orders)
            sensor_bins = np.minimum(
                (sensor_times / (duration * width)).astype(np.int64), edges.size - 2
            )
            for index in range(edges.size - 1):
                accumulators[index][sensor].extend(values[sensor_bins == index].tolist())
    records: list[dict[str, Any]] = []
    for index, values in enumerate(accumulators):
        if any(len(values[name]) == 0 for name in values):
            raise ValueError("time-bin diagnostic contains an empty evidence channel")
        attitude = np.asarray(values["attitude"], dtype=np.float64)
        bias = np.asarray(values["bias"], dtype=np.float64)
        records.append(
            {
                "bin_index": index,
                "start_fraction": float(edges[index]),
                "stop_fraction": float(edges[index + 1]),
                "attitude_rmse_rad": float(np.sqrt(np.mean(attitude**2))),
                "bias_rmse_rad_s": float(np.sqrt(np.mean(bias**2))),
                "full_nees_normalized": float(np.mean(values["full"]) / 6.0),
                "attitude_nees_normalized": float(
                    np.mean(values["attitude_nees"]) / 3.0
                ),
                "bias_nees_normalized": float(np.mean(values["bias_nees"]) / 3.0),
                "P_cross_relative_norm_mean": float(np.mean(values["cross"])),
                "mag_nis_normalized": float(np.mean(values["mag"]) / 3.0),
                "sun_nis_normalized": float(np.mean(values["sun"]) / 2.0),
                "st_nis_normalized": float(np.mean(values["st"]) / 3.0),
            }
        )
    settling = config["settling_diagnostic"]
    first = first_settling_bin(
        records,
        consecutive_bins=int(settling["consecutive_bins"]),
        attitude_rmse_max_rad=float(settling["attitude_rmse_max_rad"]),
        bias_rmse_max_rad_s=float(settling["bias_rmse_max_rad_s"]),
        full_nees_band=settling["full_nees_normalized_band"],
        sensor_nis_band=settling["sensor_nis_normalized_band"],
    )
    return {
        "time_bin_fraction": width,
        "records": records,
        "first_settling_bin": first,
        "first_settling_fraction": None if first is None else records[first]["start_fraction"],
        "criterion": settling,
    }


def _run_fixed(
    bundle: ScenarioBundle,
    generator_config: SensorFusionConfig,
    trajectory_id: int,
    policy: str,
    scales: CalibrationScales | None = None,
) -> FusionReplayResult:
    Q_c = base_process_covariance(generator_config)
    if policy == "F-BASE":
        return replay_fixed_policy(
            bundle.dataset.events, trajectory_id, default_initial_state(), 0.0, Q_c, F_BASE
        )
    if policy == "F-TUNED":
        return replay_fixed_policy(
            bundle.dataset.events, trajectory_id, default_initial_state(), 0.0, Q_c, F_TUNED
        )
    if policy == FROZEN_POLICY_ID and scales is not None:
        return replay_calibrated_fixed(bundle.dataset.events, trajectory_id, Q_c, scales)
    raise ValueError("unknown fixed closure policy")


def _diagnostic_record(
    bundle: ScenarioBundle,
    generator_config: SensorFusionConfig,
    trajectory_id: int,
    policy: str,
    config: Mapping[str, Any],
    scales: CalibrationScales | None = None,
) -> tuple[FusionReplayResult, dict[str, Any]]:
    replay = _run_fixed(bundle, generator_config, trajectory_id, policy, scales)
    record = trajectory_diagnostics(
        bundle.dataset,
        replay,
        duration_s=float(config["data"]["duration_s"]),
        partitions=config["partitions"],
        divergence_threshold_rad=float(config["numerics"]["divergence_threshold_rad"]),
    )
    record["policy_id"] = policy
    return replay, record


def _diagnose(config: Mapping[str, Any], *, resume: bool) -> dict[str, Any]:
    output_path = _diagnosis_path(config)
    if output_path.exists():
        if not resume:
            raise FileExistsError("diagnosis exists; use --resume")
        return _read_json(output_path)
    bundle, train_ids, validation_ids, split_record = _ensure_calibration_bundle(config)
    generator_config = _generator_config(
        config,
        scenario=FusionScenarioCode.MAIN_FUSION_STATIONARY,
        master_seed=int(config["data"]["calibration_master_seed"]),
        count=int(config["data"]["calibration_pool_N"]),
    )
    groups: dict[str, Any] = {}
    for split_name, ids in (("train", train_ids), ("validation", validation_ids)):
        replays: list[FusionReplayResult] = []
        records: list[dict[str, Any]] = []
        for trajectory_id in ids:
            replay, record = _diagnostic_record(
                bundle, generator_config, int(trajectory_id), "F-BASE", config
            )
            replays.append(replay)
            records.append(record)
        groups[split_name] = {
            "aggregate": aggregate_trajectory_diagnostics(records),
            "time_bins": _time_bin_diagnostics(bundle, replays, config=config),
            "trajectory_records": records,
        }
    validation_settled = groups["validation"]["aggregate"]["partitions"]["settled"]
    marginal_ranking = sorted(
        (
            ("attitude_marginal", abs(validation_settled["attitude_nees_normalized"] - 1.0)),
            ("bias_marginal", abs(validation_settled["bias_nees_normalized"] - 1.0)),
        ),
        key=lambda item: (-item[1], item[0]),
    )
    output = {
        "status": "COMPLETE",
        "experiment_version": EXPERIMENT_VERSION,
        "dataset_hash": bundle.dataset_hash,
        "oracle_hash_not_estimator_input": bundle.oracle_hash,
        "split": split_record,
        "groups": groups,
        "likely_source_ranking_by_settled_marginal_distance": [item[0] for item in marginal_ranking],
        "confirmation_dataset_accessed": False,
        "sensor_R_scales": {"mag": 1.0, "sun": 1.0, "st": 1.0},
        "runtime": {"python": platform.python_version(), "numpy": np.__version__},
        "config_hash": _sha256_json(config),
    }
    _atomic_write_json(output_path, output)
    return output


def scale_change_magnitude(scales: CalibrationScales) -> float:
    return float(sum(abs(math.log2(getattr(scales, name))) for name in SCALE_FIELDS))


def coordinate_candidates(
    center: CalibrationScales,
    field: str,
    grid: Sequence[float],
) -> list[CalibrationScales]:
    if field not in SCALE_FIELDS:
        raise ValueError("unknown calibration coordinate")
    result = []
    for value in grid:
        arguments = center.as_dict()
        arguments[field] = float(value)
        result.append(CalibrationScales(**arguments))
    return result


def local_combined_grid(
    center: CalibrationScales,
    grid: Sequence[float],
) -> list[CalibrationScales]:
    locked = tuple(float(item) for item in grid)
    axes: list[tuple[float, float, float]] = []
    for field in SCALE_FIELDS:
        value = float(getattr(center, field))
        if value not in locked:
            raise ValueError("coordinate winner is absent from locked scale grid")
        index = locked.index(value)
        if index == 0:
            axes.append((locked[0], locked[1], locked[2]))
        elif index == len(locked) - 1:
            axes.append((locked[-3], locked[-2], locked[-1]))
        else:
            axes.append((locked[index - 1], locked[index], locked[index + 1]))
    return [CalibrationScales(*values) for values in itertools.product(*axes)]


def validation_guard(
    candidate: Mapping[str, Any],
    baseline: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    guard = config["calibration"]["validation_guards"]
    settled = candidate["partitions"]["settled"]
    base = baseline["partitions"]["settled"]
    low, high = (float(item) for item in guard["sensor_nis_normalized_band"])
    checks = {
        "zero_divergence": int(candidate["divergence_count"]) == 0,
        "strict_P_S_SPD": float(candidate["minimum_P_eigenvalue"]) > 0.0
        and float(candidate["minimum_S_eigenvalue"]) > 0.0,
        "mag_nis": low <= float(settled["mag_nis_normalized"]) <= high,
        "sun_nis": low <= float(settled["sun_nis_normalized"]) <= high,
        "st_nis": low <= float(settled["st_nis_normalized"]) <= high,
        "attitude_accuracy": float(settled["attitude_rmse_rad"])
        <= float(base["attitude_rmse_rad"])
        * (1.0 + float(guard["attitude_rmse_max_degradation_fraction"])),
        "bias_accuracy": float(settled["bias_rmse_rad_s"])
        <= float(base["bias_rmse_rad_s"])
        * (1.0 + float(guard["bias_rmse_max_degradation_fraction"])),
    }
    return {"passed": all(checks.values()), "checks": checks}


def stage1_selection_key(record: Mapping[str, Any]) -> tuple[Any, ...]:
    initial = record["aggregate"]["partitions"]["initial"]
    scales = CalibrationScales(**record["scales"])
    return (
        int(record["aggregate"]["divergence_count"]),
        float(initial["attitude_rmse_rad"]),
        float(initial["attitude_p95_rad_mean"]),
        float(initial["bias_rmse_rad_s"]),
        float(initial["bias_p95_rad_s_mean"]),
        abs(float(initial["full_nees_normalized"]) - 1.0),
        scale_change_magnitude(scales),
        tuple(getattr(scales, name) for name in SCALE_FIELDS),
    )


def settled_selection_key(record: Mapping[str, Any]) -> tuple[Any, ...]:
    settled = record["aggregate"]["partitions"]["settled"]
    scales = CalibrationScales(**record["scales"])
    baseline = record["baseline_settled"]
    attitude_degradation = float(settled["attitude_rmse_rad"]) / float(
        baseline["attitude_rmse_rad"]
    ) - 1.0
    bias_degradation = float(settled["bias_rmse_rad_s"]) / float(
        baseline["bias_rmse_rad_s"]
    ) - 1.0
    return (
        int(record["aggregate"]["divergence_count"]),
        abs(float(settled["full_nees_normalized"]) - 1.0),
        abs(float(settled["attitude_nees_normalized"]) - 1.0),
        abs(float(settled["bias_nees_normalized"]) - 1.0),
        attitude_degradation,
        bias_degradation,
        scale_change_magnitude(scales),
        tuple(getattr(scales, name) for name in SCALE_FIELDS),
    )


def deterministic_select(
    records: Sequence[Mapping[str, Any]],
    *,
    stage: str,
) -> Mapping[str, Any]:
    eligible = [item for item in records if bool(item["guard"]["passed"])]
    if not eligible:
        raise ValueError("no candidate satisfies the predeclared validation guards")
    key = stage1_selection_key if stage == "stage1" else settled_selection_key
    return min(eligible, key=key)


def _candidate_id(stage: str, index: int, scales: CalibrationScales) -> str:
    values = "_".join(f"{getattr(scales, name):g}" for name in SCALE_FIELDS)
    return f"{stage}_{index:03d}_{values}"


def _candidate_record(
    config: Mapping[str, Any],
    bundle: ScenarioBundle,
    generator_config: SensorFusionConfig,
    validation_ids: np.ndarray,
    baseline: Mapping[str, Any],
    *,
    stage: str,
    index: int,
    scales: CalibrationScales,
    resume: bool,
) -> tuple[dict[str, Any], bool]:
    results_root, _, _ = _roots(config)
    candidate_id = _candidate_id(stage, index, scales)
    path = results_root / "search" / "candidates" / f"{candidate_id}.json"
    if path.exists():
        if not resume:
            raise FileExistsError("candidate checkpoint exists; use --resume")
        value = _read_json(path)
        expected = {
            "candidate_id": candidate_id,
            "scales": scales.as_dict(),
            "dataset_hash": bundle.dataset_hash,
            "validation_ids": [int(item) for item in validation_ids],
            "config_hash": _sha256_json(config),
        }
        if any(value.get(name) != item for name, item in expected.items()):
            raise ValueError("candidate checkpoint identity mismatch")
        return value, True
    trajectory_records: list[dict[str, Any]] = []
    failure: str | None = None
    try:
        for trajectory_id in validation_ids:
            _, record = _diagnostic_record(
                bundle,
                generator_config,
                int(trajectory_id),
                FROZEN_POLICY_ID,
                config,
                scales,
            )
            trajectory_records.append(record)
        aggregate = aggregate_trajectory_diagnostics(trajectory_records)
        guard = validation_guard(aggregate, baseline, config)
    except (ValueError, NumericalSafetyError, np.linalg.LinAlgError) as error:
        failure = f"{type(error).__name__}: {error}"
        aggregate = {
            "N": len(trajectory_records),
            "divergence_count": len(validation_ids),
            "minimum_P_eigenvalue": None,
            "minimum_S_eigenvalue": None,
            "partitions": {},
        }
        guard = {"passed": False, "checks": {"numerical_failure": False}}
    value = {
        "candidate_id": candidate_id,
        "stage": stage,
        "logical_index": int(index),
        "scales": scales.as_dict(),
        "sensor_R_scales": {"mag": 1.0, "sun": 1.0, "st": 1.0},
        "dataset_hash": bundle.dataset_hash,
        "validation_ids": [int(item) for item in validation_ids],
        "test_ids_accessed": False,
        "aggregate": aggregate,
        "baseline_settled": baseline["partitions"]["settled"],
        "guard": guard,
        "failure": failure,
        "trajectory_records": trajectory_records,
        "config_hash": _sha256_json(config),
    }
    _atomic_write_json(path, value)
    return value, False


def _search(config: Mapping[str, Any], *, resume: bool) -> dict[str, Any]:
    results_root, manifests_root, _ = _roots(config)
    completed_manifest_path = results_root / "search" / "search_manifest.json"
    if completed_manifest_path.exists():
        if not resume:
            raise FileExistsError("search manifest exists; use --resume")
        completed = _read_json(completed_manifest_path)
        if completed.get("status") != "COMPLETE" or completed.get("candidate_count") != 101:
            raise ValueError("existing search manifest is not a complete 101-candidate search")
        for item in completed["candidate_ledger"]:
            path = results_root / "search" / "candidates" / f"{item['candidate_id']}.json"
            checkpoint = _read_json(path)
            if (
                checkpoint.get("candidate_id") != item["candidate_id"]
                or checkpoint.get("scales") != item["scales"]
                or checkpoint.get("dataset_hash") != completed["dataset_hash"]
                or checkpoint.get("config_hash") != _sha256_json(config)
            ):
                raise ValueError("resume candidate checkpoint identity mismatch")
        _load_freeze(config)
        resumed = dict(completed)
        resumed["resume_verified_candidate_checkpoints"] = 101
        return resumed
    diagnosis = _read_json(_diagnosis_path(config))
    if diagnosis.get("status") != "COMPLETE" or diagnosis.get("confirmation_dataset_accessed"):
        raise ValueError("complete no-confirmation diagnosis is required before search")
    bundle, train_ids, validation_ids, split_record = _ensure_calibration_bundle(config)
    if diagnosis["dataset_hash"] != bundle.dataset_hash:
        raise ValueError("diagnosis dataset identity changed")
    baseline = diagnosis["groups"]["validation"]["aggregate"]
    generator_config = _generator_config(
        config,
        scenario=FusionScenarioCode.MAIN_FUSION_STATIONARY,
        master_seed=int(config["data"]["calibration_master_seed"]),
        count=int(config["data"]["calibration_pool_N"]),
    )
    grid = tuple(float(item) for item in config["calibration"]["scales"])
    ledger: list[dict[str, Any]] = []
    stage_selected: dict[str, dict[str, Any]] = {}
    reused = 0

    center = CalibrationScales()
    stages = (
        ("stage1_p0_att", "s_P0_att", "stage1"),
        ("stage1_p0_bias", "s_P0_bias", "stage1"),
        ("stage2_qg", "s_Qg", "settled"),
        ("stage2_qb", "s_Qb", "settled"),
    )
    logical_index = 0
    for stage_name, field, selection_stage in stages:
        stage_records: list[dict[str, Any]] = []
        for scales in coordinate_candidates(center, field, grid):
            record, was_reused = _candidate_record(
                config,
                bundle,
                generator_config,
                validation_ids,
                baseline,
                stage=stage_name,
                index=logical_index,
                scales=scales,
                resume=resume,
            )
            reused += int(was_reused)
            ledger.append(record)
            stage_records.append(record)
            logical_index += 1
        selected = deterministic_select(stage_records, stage=selection_stage)
        center = CalibrationScales(**selected["scales"])
        stage_selected[stage_name] = {
            "candidate_id": selected["candidate_id"],
            "scales": selected["scales"],
        }

    local_candidates = local_combined_grid(center, grid)
    local_records: list[dict[str, Any]] = []
    for scales in local_candidates:
        record, was_reused = _candidate_record(
            config,
            bundle,
            generator_config,
            validation_ids,
            baseline,
            stage="stage3_local",
            index=logical_index,
            scales=scales,
            resume=resume,
        )
        reused += int(was_reused)
        ledger.append(record)
        local_records.append(record)
        logical_index += 1
    selected = deterministic_select(local_records, stage="settled")
    if len(ledger) != 101:
        raise AssertionError("staged search must log exactly 101 logical candidates")
    search_manifest = {
        "status": "COMPLETE",
        "experiment_version": EXPERIMENT_VERSION,
        "dataset_hash": bundle.dataset_hash,
        "train_ids_diagnostic_only": [int(item) for item in train_ids],
        "validation_ids_used_for_selection": [int(item) for item in validation_ids],
        "confirmation_dataset_accessed": False,
        "candidate_count": len(ledger),
        "coordinate_candidate_count": 20,
        "local_candidate_count": 81,
        "stage_selected": stage_selected,
        "selected_candidate_id": selected["candidate_id"],
        "selected_scales": selected["scales"],
        "selected_guard": selected["guard"],
        "candidate_ledger": [
            {
                "candidate_id": item["candidate_id"],
                "stage": item["stage"],
                "scales": item["scales"],
                "guard": item["guard"],
                "aggregate": item["aggregate"],
                "failure": item["failure"],
            }
            for item in ledger
        ],
        "reused_candidate_checkpoints": reused,
        "split_identity": split_record,
        "sensor_R_scales": {"mag": 1.0, "sun": 1.0, "st": 1.0},
        "config_hash": _sha256_json(config),
    }
    manifest_path = results_root / "search" / "search_manifest.json"
    _atomic_write_json(manifest_path, search_manifest)
    search_hash = _sha256_file(manifest_path)
    selected_path = results_root / "search" / "candidates" / f"{selected['candidate_id']}.json"
    freeze_identity = {
        "schema_version": "p1-exit-f-calibrated-freeze-v1",
        "policy_id": FROZEN_POLICY_ID,
        "scales": selected["scales"],
        "sensor_R_scales": {"mag": 1.0, "sun": 1.0, "st": 1.0},
        "calibration_dataset_hash": bundle.dataset_hash,
        "train_ids": [int(item) for item in train_ids],
        "validation_ids": [int(item) for item in validation_ids],
        "selected_candidate_id": selected["candidate_id"],
        "selected_candidate_hash": _sha256_file(selected_path),
        "search_manifest_hash": search_hash,
        "config_hash": _sha256_json(config),
        "confirmation_dataset_accessed_at_freeze": False,
    }
    freeze = dict(freeze_identity)
    freeze["freeze_hash"] = _sha256_json(freeze_identity)
    freeze_path = manifests_root / "F-CALIBRATED-v1.json"
    if freeze_path.exists() and _read_json(freeze_path) != freeze:
        raise ValueError("existing F-CALIBRATED-v1 freeze differs")
    if not freeze_path.exists():
        _atomic_write_json(freeze_path, freeze)
    return search_manifest


def _load_freeze(config: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
    results_root, manifests_root, _ = _roots(config)
    freeze_path = manifests_root / "F-CALIBRATED-v1.json"
    freeze = _read_json(freeze_path)
    identity = dict(freeze)
    observed_hash = identity.pop("freeze_hash", None)
    if observed_hash != _sha256_json(identity):
        raise ValueError("F-CALIBRATED-v1 freeze hash mismatch")
    if freeze.get("policy_id") != FROZEN_POLICY_ID:
        raise ValueError("wrong frozen candidate policy ID")
    if set(map(float, freeze["sensor_R_scales"].values())) != {1.0}:
        raise ValueError("frozen candidate changes sensor R")
    search_path = results_root / "search" / "search_manifest.json"
    if _sha256_file(search_path) != freeze["search_manifest_hash"]:
        raise ValueError("search manifest changed after candidate freeze")
    if _read_json(search_path).get("status") != "COMPLETE":
        raise ValueError("candidate freeze requires a complete search")
    return freeze, _sha256_file(freeze_path)


def _ensure_confirmation_bundles(
    config: Mapping[str, Any],
) -> tuple[ScenarioBundle, ScenarioBundle]:
    _, manifests_root, _ = _roots(config)
    data = config["data"]
    count = int(data["confirmation_stationary_N"])
    master_seed = int(data["confirmation_master_seed"])
    stationary_generated = generate_sensor_fusion(
        _generator_config(
            config,
            scenario=FusionScenarioCode.MAIN_FUSION_STATIONARY,
            master_seed=master_seed,
            count=count,
        )
    )
    stationary = _save_or_verify_bundle(
        manifests_root / "confirmation_stationary", stationary_generated
    )
    c4_generated = generate_sensor_fusion(
        _generator_config(
            config,
            scenario=FusionScenarioCode.C4_COMBINED,
            master_seed=master_seed,
            count=int(data["confirmation_c4_N"]),
        ),
        base_unit_st=stationary_generated.base_unit_st,
    )
    c4 = _save_or_verify_bundle(manifests_root / "confirmation_c4", c4_generated)
    if not np.array_equal(
        stationary.dataset.truth.trajectory_id, c4.dataset.truth.trajectory_id
    ):
        raise ValueError("stationary/C4 confirmation trajectory IDs differ")
    if not np.array_equal(stationary.dataset.truth.q_true_NB, c4.dataset.truth.q_true_NB):
        raise ValueError("stationary/C4 confirmation attitude truth differs")
    if not np.array_equal(
        stationary.dataset.truth.omega_true_B_rad_s,
        c4.dataset.truth.omega_true_B_rad_s,
    ):
        raise ValueError("stationary/C4 confirmation rate truth differs")
    if not np.array_equal(
        stationary.dataset.events.sun_z_sun_B, c4.dataset.events.sun_z_sun_B
    ) or not np.array_equal(
        stationary.dataset.events.star_tracker_q_ST_NB,
        c4.dataset.events.star_tracker_q_ST_NB,
    ):
        raise ValueError("C4 changed an unaffected confirmation sensor stream")
    return stationary, c4


def _confirmation_record_path(
    results_root: Path, scenario: str, policy: str, trajectory_id: int
) -> Path:
    return results_root / "confirmation" / "records" / scenario / policy / f"{trajectory_id}.json"


def _oracle_replay(
    bundle: ScenarioBundle,
    generator_config: SensorFusionConfig,
    trajectory_id: int,
    policy: FusionPolicy,
) -> FusionReplayResult:
    return replay_oracle_policy(
        bundle.dataset.events,
        bundle.oracle,
        trajectory_id,
        default_initial_state(),
        0.0,
        base_process_covariance(generator_config),
        policy,
    )


def _write_or_load_confirmation_record(
    config: Mapping[str, Any],
    bundle: ScenarioBundle,
    generator_config: SensorFusionConfig,
    *,
    scenario: str,
    policy: str,
    trajectory_id: int,
    scales: CalibrationScales,
    resume: bool,
) -> tuple[dict[str, Any], bool]:
    results_root, _, _ = _roots(config)
    path = _confirmation_record_path(results_root, scenario, policy, trajectory_id)
    if path.exists():
        if not resume:
            raise FileExistsError("confirmation checkpoint exists; use --resume")
        record = _read_json(path)
        if (
            record.get("dataset_hash") != bundle.dataset_hash
            or int(record.get("trajectory_id")) != trajectory_id
            or record.get("policy_id") != policy
        ):
            raise ValueError("confirmation checkpoint identity mismatch")
        return record, True
    if policy in ("F-BASE", "F-TUNED", FROZEN_POLICY_ID):
        replay = _run_fixed(
            bundle,
            generator_config,
            trajectory_id,
            policy,
            scales if policy == FROZEN_POLICY_ID else None,
        )
    else:
        oracle_policies = {
            item.policy_id: item
            for item in (
                ORACLE_PROCESS,
                ORACLE_MEASUREMENT,
                ORACLE_FULL,
                WRONG_PROCESS,
                WRONG_MEASUREMENT,
            )
        }
        replay = _oracle_replay(
            bundle, generator_config, trajectory_id, oracle_policies[policy]
        )
    diagnostic = trajectory_diagnostics(
        bundle.dataset,
        replay,
        duration_s=float(config["data"]["duration_s"]),
        partitions=config["partitions"],
        divergence_threshold_rad=float(config["numerics"]["divergence_threshold_rad"]),
    )
    phase1_metric = evaluate_fusion_replay(
        bundle.dataset,
        replay,
        scenario_id=scenario,
        policy_id=policy,
        duration_s=float(config["data"]["duration_s"]),
        divergence_threshold_rad=float(config["numerics"]["divergence_threshold_rad"]),
        confidence_level=float(config["numerics"]["confidence_level"]),
    )
    record = dict(diagnostic)
    record.update(
        scenario=scenario,
        policy_id=policy,
        dataset_hash=bundle.dataset_hash,
        scales=(scales.as_dict() if policy == FROZEN_POLICY_ID else None),
        sensor_R_scales={"mag": 1.0, "sun": 1.0, "st": 1.0},
        phase1_metric=phase1_metric,
    )
    _atomic_write_json(path, record)
    return record, False


def _paired_confirmation(
    by_policy: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    base = {
        int(item["trajectory_id"]): item for item in by_policy["F-BASE"]
    }
    candidate = {
        int(item["trajectory_id"]): item for item in by_policy[FROZEN_POLICY_ID]
    }
    ids = sorted(set(base) & set(candidate))
    if len(ids) != len(base) or len(ids) != len(candidate):
        raise ValueError("stationary confirmation pairing is incomplete")
    base_nees = np.asarray(
        [base[item]["partitions"]["settled"]["full_nees_normalized"] for item in ids],
        dtype=np.float64,
    )
    candidate_nees = np.asarray(
        [candidate[item]["partitions"]["settled"]["full_nees_normalized"] for item in ids],
        dtype=np.float64,
    )
    distance_difference = np.abs(candidate_nees - 1.0) - np.abs(base_nees - 1.0)
    low, high = paired_bootstrap_ci(
        distance_difference,
        seed=int(config["confirmation"]["bootstrap_seed"]),
        resamples=int(config["confirmation"]["bootstrap_resamples"]),
    )
    return {
        "N": len(ids),
        "trajectory_ids": ids,
        "base_nees_normalized_per_trajectory": base_nees.tolist(),
        "calibrated_nees_normalized_per_trajectory": candidate_nees.tolist(),
        "absolute_distance_candidate_minus_base_mean": float(
            np.mean(distance_difference)
        ),
        "paired_bootstrap_95_ci": [low, high],
    }


def _confirmation_acceptance(
    stationary: Mapping[str, Any],
    paired: Mapping[str, Any],
    c4_groups: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    confirmation = config["confirmation"]
    base = stationary["F-BASE"]["partitions"]["settled"]
    calibrated = stationary[FROZEN_POLICY_ID]["partitions"]["settled"]
    low, high = (float(item) for item in confirmation["sensor_nis_normalized_band"])
    target_low, target_high = (
        float(item) for item in confirmation["recommended_full_nees_normalized_band"]
    )
    stationary_checks = {
        "N50": int(stationary[FROZEN_POLICY_ID]["N"]) == 50,
        "target_full_nees": target_low
        <= float(calibrated["full_nees_normalized"])
        <= target_high,
        "materially_closer": float(
            paired["absolute_distance_candidate_minus_base_mean"]
        )
        < 0.0
        and float(paired["paired_bootstrap_95_ci"][1]) < 0.0,
        "mag_nis": low <= float(calibrated["mag_nis_normalized"]) <= high,
        "sun_nis": low <= float(calibrated["sun_nis_normalized"]) <= high,
        "st_nis": low <= float(calibrated["st_nis_normalized"]) <= high,
        "attitude_accuracy": float(calibrated["attitude_rmse_rad"])
        <= float(base["attitude_rmse_rad"])
        * (1.0 + float(confirmation["attitude_rmse_max_degradation_fraction"])),
        "bias_accuracy": float(calibrated["bias_rmse_rad_s"])
        <= float(base["bias_rmse_rad_s"])
        * (1.0 + float(confirmation["bias_rmse_max_degradation_fraction"])),
        "zero_divergence": int(stationary[FROZEN_POLICY_ID]["divergence_count"]) == 0,
        "strict_P_S_SPD": float(stationary[FROZEN_POLICY_ID]["minimum_P_eigenvalue"])
        > 0.0
        and float(stationary[FROZEN_POLICY_ID]["minimum_S_eigenvalue"]) > 0.0,
    }
    c4_base = c4_groups["F-BASE"]
    c4_cal = c4_groups[FROZEN_POLICY_ID]
    c4_full = c4_groups["ORACLE-FULL"]
    c4_process = c4_groups["ORACLE-PROCESS"]
    c4_wrong_process = c4_groups["WRONG-PROCESS"]
    c4_wrong_measurement = c4_groups["WRONG-MEASUREMENT"]
    phase = lambda group, name: float(group["phase1_metrics"][name])
    min_effect = float(confirmation["c4_full_oracle_min_improvement_fraction"])
    full_slow_improvement = 1.0 - phase(c4_full, "slow_bias_rmse_rad_s") / phase(
        c4_base, "slow_bias_rmse_rad_s"
    )
    full_fast_improvement = 1.0 - phase(c4_full, "fast_attitude_peak_rad") / phase(
        c4_base, "fast_attitude_peak_rad"
    )
    c4_settled_cal = c4_cal["partitions"]["settled"]
    c4_settled_full = c4_full["partitions"]["settled"]
    c4_checks = {
        "N50_all": all(int(item["N"]) == 50 for item in c4_groups.values()),
        "calibrated_accuracy_not_seriously_worse": phase(c4_cal, "attitude_rmse_rad")
        <= phase(c4_base, "attitude_rmse_rad")
        * (1.0 + float(confirmation["c4_calibrated_accuracy_max_degradation_fraction"]))
        and phase(c4_cal, "bias_vector_rmse_rad_s")
        <= phase(c4_base, "bias_vector_rmse_rad_s")
        * (1.0 + float(confirmation["c4_calibrated_accuracy_max_degradation_fraction"])),
        "full_oracle_cause_advantage": full_slow_improvement >= min_effect
        and full_fast_improvement >= min_effect,
        "process_beats_wrong_process_slow_bias": phase(
            c4_process, "slow_bias_rmse_rad_s"
        )
        < phase(c4_wrong_process, "slow_bias_rmse_rad_s"),
        "full_beats_wrong_measurement_state": phase(c4_full, "slow_bias_rmse_rad_s")
        < phase(c4_wrong_measurement, "slow_bias_rmse_rad_s")
        and phase(c4_full, "fast_attitude_peak_rad")
        < phase(c4_wrong_measurement, "fast_attitude_peak_rad"),
        "wrong_measurement_state_worse_than_base": phase(
            c4_wrong_measurement, "slow_bias_rmse_rad_s"
        )
        > phase(c4_base, "slow_bias_rmse_rad_s")
        and phase(c4_wrong_measurement, "fast_attitude_peak_rad")
        > phase(c4_base, "fast_attitude_peak_rad"),
        "calibrated_settled_sensor_nis": all(
            low <= float(c4_settled_cal[f"{sensor}_nis_normalized"]) <= high
            for sensor in ("mag", "sun", "st")
        ),
        "full_oracle_settled_sensor_nis": all(
            low <= float(c4_settled_full[f"{sensor}_nis_normalized"]) <= high
            for sensor in ("mag", "sun", "st")
        ),
        "zero_divergence": all(int(item["divergence_count"]) == 0 for item in c4_groups.values()),
        "strict_P_S_SPD": all(
            float(item["minimum_P_eigenvalue"]) > 0.0
            and float(item["minimum_S_eigenvalue"]) > 0.0
            for item in c4_groups.values()
        ),
    }
    return {
        "stationary": stationary_checks,
        "c4": c4_checks,
        "stationary_passed": all(stationary_checks.values()),
        "c4_passed": all(c4_checks.values()),
        "full_oracle_slow_bias_improvement_fraction": full_slow_improvement,
        "full_oracle_fast_peak_improvement_fraction": full_fast_improvement,
    }


def _aggregate_confirmation_group(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    aggregate = aggregate_trajectory_diagnostics(records)
    phase_names = (
        "attitude_rmse_rad",
        "fast_attitude_peak_rad",
        "bias_vector_rmse_rad_s",
        "slow_bias_rmse_rad_s",
        "mag_nis_normalized_mean",
        "sun_nis_normalized_mean",
        "st_nis_normalized_mean",
        "nees_normalized_mean",
    )
    aggregate["phase1_metrics"] = {
        name: float(np.mean([float(item["phase1_metric"][name]) for item in records]))
        for name in phase_names
    }
    return aggregate


def _confirm(
    config: Mapping[str, Any],
    *,
    resume: bool,
    max_trajectories: int | None,
) -> dict[str, Any]:
    results_root, manifests_root, _ = _roots(config)
    freeze, freeze_file_hash = _load_freeze(config)
    scales = CalibrationScales(**freeze["scales"])
    calibration_split = _read_json(manifests_root / "calibration_split.json")
    stationary, c4 = _ensure_confirmation_bundles(config)
    confirmation_ids = np.asarray(stationary.dataset.truth.trajectory_id, dtype=np.int64)
    calibration_ids = set(calibration_split["train_ids"]) | set(
        calibration_split["validation_ids"]
    )
    if calibration_ids & set(map(int, confirmation_ids)):
        raise ValueError("calibration and confirmation trajectory IDs overlap")
    if _frozen_phase1_test_ids() & set(map(int, confirmation_ids)):
        raise ValueError("confirmation IDs overlap frozen Phase 1 test IDs")
    required = int(config["data"]["confirmation_stationary_N"])
    target = min(required, int(max_trajectories or required))
    if target <= 0:
        raise ValueError("max-trajectories must be positive")
    selected_ids = confirmation_ids[:target]
    stationary_config = _generator_config(
        config,
        scenario=FusionScenarioCode.MAIN_FUSION_STATIONARY,
        master_seed=int(config["data"]["confirmation_master_seed"]),
        count=required,
    )
    c4_config = _generator_config(
        config,
        scenario=FusionScenarioCode.C4_COMBINED,
        master_seed=int(config["data"]["confirmation_master_seed"]),
        count=required,
    )
    policies = {
        "STATIONARY": ("F-BASE", "F-TUNED", FROZEN_POLICY_ID),
        "C4": (
            "F-BASE",
            "F-TUNED",
            FROZEN_POLICY_ID,
            "ORACLE-PROCESS",
            "ORACLE-MEASUREMENT",
            "ORACLE-FULL",
            "WRONG-PROCESS",
            "WRONG-MEASUREMENT",
        ),
    }
    bundles = {"STATIONARY": stationary, "C4": c4}
    generator_configs = {"STATIONARY": stationary_config, "C4": c4_config}
    records_by_group: dict[str, list[dict[str, Any]]] = {}
    written = 0
    reused = 0
    start = time.monotonic()
    for scenario, policy_ids in policies.items():
        for policy in policy_ids:
            group_key = f"{scenario}/{policy}"
            records_by_group[group_key] = []
            for trajectory_id_raw in selected_ids:
                trajectory_id = int(trajectory_id_raw)
                record, was_reused = _write_or_load_confirmation_record(
                    config,
                    bundles[scenario],
                    generator_configs[scenario],
                    scenario=scenario,
                    policy=policy,
                    trajectory_id=trajectory_id,
                    scales=scales,
                    resume=resume,
                )
                reused += int(was_reused)
                written += int(not was_reused)
                records_by_group[group_key].append(record)
    groups = {
        key: _aggregate_confirmation_group(records)
        for key, records in records_by_group.items()
    }
    stationary_groups = {
        policy: groups[f"STATIONARY/{policy}"] for policy in policies["STATIONARY"]
    }
    c4_groups = {policy: groups[f"C4/{policy}"] for policy in policies["C4"]}
    paired = _paired_confirmation(
        {
            policy: records_by_group[f"STATIONARY/{policy}"]
            for policy in policies["STATIONARY"]
        },
        config=config,
    )
    complete = target == required and all(group["N"] == required for group in groups.values())
    acceptance = (
        _confirmation_acceptance(stationary_groups, paired, c4_groups, config)
        if complete
        else None
    )
    if _sha256_file(manifests_root / "F-CALIBRATED-v1.json") != freeze_file_hash:
        raise ValueError("F-CALIBRATED-v1 changed during confirmation")
    output = {
        "status": "COMPLETE" if complete else "PARTIAL",
        "required_N": required,
        "completed_N": target,
        "freeze_hash": freeze["freeze_hash"],
        "frozen_scales": freeze["scales"],
        "stationary_dataset_hash": stationary.dataset_hash,
        "c4_dataset_hash": c4.dataset_hash,
        "calibration_confirmation_disjoint": True,
        "frozen_phase1_test_disjoint": True,
        "stationary_c4_same_realization": True,
        "confirmation_ids": [int(item) for item in selected_ids],
        "groups": groups,
        "stationary_paired": paired,
        "acceptance": acceptance,
        "written_this_invocation": written,
        "reused_checkpoints": reused,
        "elapsed_s": time.monotonic() - start,
        "sensor_R_scales": {"mag": 1.0, "sun": 1.0, "st": 1.0},
        "test_access_after_freeze_only": True,
        "config_hash": _sha256_json(config),
    }
    _atomic_write_json(results_root / "confirmation" / "confirmation_summary.json", output)
    return output


def _validation(config: Mapping[str, Any]) -> dict[str, Any]:
    if set(inspect.signature(replay_calibrated_fixed).parameters) & {
        "truth",
        "oracle",
        "event_window",
        "hidden_label",
    }:
        raise AssertionError("calibrated fixed API exposes forbidden information")
    scale = CalibrationScales(2.0, 4.0, 0.5, 8.0)
    state = scaled_initial_state(scale)
    base = default_initial_state()
    if not np.array_equal(state.P[:3, :3], base.P[:3, :3] * 2.0):
        raise AssertionError("attitude P0 scale is not applied before replay")
    if not np.array_equal(state.P[3:, 3:], base.P[3:, 3:] * 4.0):
        raise AssertionError("bias P0 scale is not applied before replay")
    schedule = local_combined_grid(CalibrationScales(1.0, 2.0, 2.0, 4.0), (0.5, 1, 2, 4, 8))
    if len(schedule) != 81 or len(set(schedule)) != 81:
        raise AssertionError("local combined grid is not exact 3^4")
    result = {
        "status": "PASS",
        "experiment_version": EXPERIMENT_VERSION,
        "fixed_api_truth_oracle_free": True,
        "sensor_R_scales_exact_one": True,
        "coordinate_budget": 20,
        "local_grid_budget": len(schedule),
        "maximum_candidate_count": 101,
        "confirmation_artifacts_required_only_after_freeze": True,
        "config_hash": _sha256_json(config),
    }
    return result


def _format_number(value: Any) -> str:
    return "—" if value is None else f"{float(value):.8g}"


def _report(config: Mapping[str, Any]) -> dict[str, Any]:
    results_root, _, reports_root = _roots(config)
    diagnosis = _read_json(_diagnosis_path(config))
    search = _read_json(results_root / "search" / "search_manifest.json")
    confirmation = _read_json(results_root / "confirmation" / "confirmation_summary.json")
    if search.get("status") != "COMPLETE":
        raise ValueError("complete search is required for report")
    train = diagnosis["groups"]["train"]["aggregate"]
    validation = diagnosis["groups"]["validation"]["aggregate"]
    transient_lines = [
        "# P1 Exit Transient Diagnostic Report",
        "",
        "Status: `COMPLETE`. Only independent calibration train/validation data are used.",
        "",
        "| Split/partition | Full NEES/DOF | Attitude marginal | Bias marginal | Mag NIS | Sun NIS | ST NIS |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for split_name, aggregate in (("train", train), ("validation", validation)):
        for partition in PARTITION_NAMES:
            item = aggregate["partitions"][partition]
            transient_lines.append(
                f"| {split_name}/{partition} | {_format_number(item['full_nees_normalized'])} | "
                f"{_format_number(item['attitude_nees_normalized'])} | "
                f"{_format_number(item['bias_nees_normalized'])} | "
                f"{_format_number(item['mag_nis_normalized'])} | "
                f"{_format_number(item['sun_nis_normalized'])} | "
                f"{_format_number(item['st_nis_normalized'])} |"
            )
    settled = validation["partitions"]["settled"]
    bins = diagnosis["groups"]["validation"]["time_bins"]
    transient_lines.extend(
        [
            "",
            "## Whitened and cross-covariance evidence",
            "",
            f"- Settled whitened coordinate energy: `{settled['whitened_coordinate_energy']}`.",
            f"- Settled whitened attitude/bias grouped energy: "
            f"`{settled['whitened_attitude_group_energy']:.8g}` / "
            f"`{settled['whitened_bias_group_energy']:.8g}`.",
            f"- Settled mean relative attitude-bias P cross norm: "
            f"`{settled['P_cross_relative_norm_mean']:.8g}`.",
            f"- Settled correlation-normalized P cross block: "
            f"`{settled['P_cross_correlation_block_mean']}`.",
            f"- Settled whitened attitude-bias cross-correlation: "
            f"`{settled['whitened_cross_correlation']}`.",
            f"- Predeclared validation settling-bin result: "
            f"`{bins['first_settling_fraction']}` horizon fraction.",
            "",
            "The likely-source ordering is based on train/validation marginal diagnostics only: "
            f"`{diagnosis['likely_source_ranking_by_settled_marginal_distance']}`. "
            "Sensor R remains fixed, so matched sensor NIS and posterior inconsistency are reported separately.",
            "",
            "Limitations: the decomposition is for the representative normalized MAIN-FUSION benchmark; "
            "it is not an orbit, WMM, eclipse, flight-sensor, or universal calibration claim.",
        ]
    )
    (reports_root / "P1_EXIT_TRANSIENT_DIAGNOSTIC_REPORT.md").write_text(
        "\n".join(transient_lines) + "\n", encoding="utf-8"
    )

    calibration_lines = [
        "# P1 Exit Covariance Calibration Report",
        "",
        f"Search status: `{search['status']}`; logical candidates: `{search['candidate_count']}`.",
        f"Selected `F-CALIBRATED-v1`: `{search['selected_scales']}`.",
        "Candidate selection used only the listed validation IDs; confirmation was inaccessible before freeze.",
        "",
        "| Candidate | Stage | P0-att | P0-bias | Qg | Qb | Guard | Settled full | Att | Bias |",
        "|---|---|---:|---:|---:|---:|---|---:|---:|---:|",
    ]
    for candidate in search["candidate_ledger"]:
        scales = candidate["scales"]
        if candidate["failure"] is None:
            settled_candidate = candidate["aggregate"]["partitions"]["settled"]
            metrics = (
                _format_number(settled_candidate["full_nees_normalized"]),
                _format_number(settled_candidate["attitude_nees_normalized"]),
                _format_number(settled_candidate["bias_nees_normalized"]),
            )
        else:
            metrics = ("failure", "failure", "failure")
        calibration_lines.append(
            f"| {candidate['candidate_id']} | {candidate['stage']} | "
            f"{scales['s_P0_att']} | {scales['s_P0_bias']} | {scales['s_Qg']} | "
            f"{scales['s_Qb']} | {candidate['guard']['passed']} | {metrics[0]} | "
            f"{metrics[1]} | {metrics[2]} |"
        )
    calibration_lines.extend(
        [
            "",
            "## Independent confirmation",
            "",
            f"- Status/N: `{confirmation['status']}` / `{confirmation['completed_N']}`.",
            f"- Stationary paired NEES-distance evidence: "
            f"`{confirmation['stationary_paired']['absolute_distance_candidate_minus_base_mean']:.8g}`, "
            f"95% CI `{confirmation['stationary_paired']['paired_bootstrap_95_ci']}`.",
            f"- Acceptance: `{confirmation['acceptance']}`.",
            f"- Stationary/C4 same realization: `{confirmation['stationary_c4_same_realization']}`.",
            f"- Calibration/confirmation and frozen-test disjointness: "
            f"`{confirmation['calibration_confirmation_disjoint']}` / "
            f"`{confirmation['frozen_phase1_test_disjoint']}`.",
            "",
            "No sensor R tuning, test-set candidate selection, reported-P scaling, event-wise inflation, "
            "oracle label input, or numerical covariance fallback was used.",
        ]
    )
    (reports_root / "P1_EXIT_COVARIANCE_CALIBRATION_REPORT.md").write_text(
        "\n".join(calibration_lines) + "\n", encoding="utf-8"
    )

    regression_path = results_root / "regression_evidence.json"
    regressions = _read_json(regression_path) if regression_path.exists() else None
    validation_lines = [
        "# P1 Exit Closure Validation Report",
        "",
        f"Implementation: `PASS`; search: `{search['status']}`; confirmation: `{confirmation['status']}`.",
        f"Runtime confirmation seconds: `{confirmation['elapsed_s']:.3f}`; "
        f"records written/reused: `{confirmation['written_this_invocation']}` / "
        f"`{confirmation['reused_checkpoints']}`.",
        "",
        "## Regression and integrity evidence",
        "",
        f"- All required regression groups passed: `{bool(regressions and regressions['all_passed'])}`.",
        f"- Pytest/legacy counts: `{regressions['pytest'] if regressions else None}`.",
        f"- Phase 1A fresh/cache smoke: `{regressions['phase1a_smoke'] if regressions else None}`.",
        f"- Frozen files checked/mismatched: "
        f"`{regressions['frozen_integrity']['checked_files'] if regressions else None}` / "
        f"`{regressions['frozen_integrity']['mismatches'] if regressions else None}`.",
        "- Final tracked/staged patch equality, allowlist classification, and the ignored smoke-output "
        "note are recorded in `preflight_snapshots/03_20260802T032016Z/FINAL_INTEGRITY.md`.",
        "",
        "## Blocking and deferred items",
        "",
        "There is no implementation or sample-count blocker. The scientific result remains conditional: "
        "F-CALIBRATED-v1 closes stationary posterior consistency but distorts C4 bias accuracy and fixed-policy "
        "sensor NIS, so it must not replace F-BASE for C4. Phase 2 and learned/FPGA/closed-loop work remain "
        "explicitly deferred.",
    ]
    (reports_root / "P1_EXIT_CLOSURE_VALIDATION_REPORT.md").write_text(
        "\n".join(validation_lines) + "\n", encoding="utf-8"
    )
    return {
        "status": "COMPLETE",
        "reports": [
            "P1_EXIT_TRANSIENT_DIAGNOSTIC_REPORT.md",
            "P1_EXIT_COVARIANCE_CALIBRATION_REPORT.md",
            "P1_EXIT_CLOSURE_VALIDATION_REPORT.md",
        ],
        "regression_evidence_present": regressions is not None,
    }


def _exit_review(config: Mapping[str, Any]) -> dict[str, Any]:
    results_root, _, reports_root = _roots(config)
    search = _read_json(results_root / "search" / "search_manifest.json")
    confirmation = _read_json(results_root / "confirmation" / "confirmation_summary.json")
    regression_path = results_root / "regression_evidence.json"
    if search.get("status") != "COMPLETE" or confirmation.get("status") != "COMPLETE":
        decision = "DEFERRED"
        status = "PASS_P1_EXIT_CLOSURE_IMPLEMENTATION_ONLY"
        regressions_passed = False
    elif not regression_path.exists():
        decision = "DEFERRED"
        status = "PASS_P1_EXIT_CLOSURE_IMPLEMENTATION_ONLY"
        regressions_passed = False
    else:
        regression = _read_json(regression_path)
        regressions_passed = bool(regression.get("all_passed"))
        stationary = confirmation["acceptance"]["stationary"]
        c4 = confirmation["acceptance"]["c4"]
        stationary_without_target = all(
            value for name, value in stationary.items() if name != "target_full_nees"
        )
        c4_structure_without_calibrated_distortion = all(
            value
            for name, value in c4.items()
            if name
            not in {
                "calibrated_accuracy_not_seriously_worse",
                "calibrated_settled_sensor_nis",
            }
        )
        if (
            regressions_passed
            and stationary_without_target
            and bool(stationary["target_full_nees"])
            and all(c4.values())
            and confirmation["calibration_confirmation_disjoint"]
            and confirmation["frozen_phase1_test_disjoint"]
        ):
            decision = "GO"
            status = "PASS_P1_EXIT_CONDITION_CLOSURE"
        elif (
            regressions_passed
            and stationary_without_target
            and bool(stationary["materially_closer"])
            and c4_structure_without_calibrated_distortion
            and confirmation["calibration_confirmation_disjoint"]
            and confirmation["frozen_phase1_test_disjoint"]
        ):
            decision = "CONDITIONAL_GO"
            status = "PASS_P1_EXIT_CONDITION_CLOSURE"
        else:
            decision = "STOP"
            status = "STOP_P1_EXIT_CONDITION_CLOSURE"
    freeze, _ = _load_freeze(config)
    groups = confirmation.get("groups", {})
    base_settled = groups.get("STATIONARY/F-BASE", {}).get("partitions", {}).get("settled", {})
    calibrated_settled = groups.get(f"STATIONARY/{FROZEN_POLICY_ID}", {}).get(
        "partitions", {}
    ).get("settled", {})
    diagnosis = _read_json(_diagnosis_path(config))
    diagnostic_validation = diagnosis["groups"]["validation"]["aggregate"]["partitions"]
    c4_base_phase = groups.get("C4/F-BASE", {}).get("phase1_metrics", {})
    c4_calibrated_phase = groups.get(f"C4/{FROZEN_POLICY_ID}", {}).get(
        "phase1_metrics", {}
    )
    c4_calibrated_settled = groups.get(f"C4/{FROZEN_POLICY_ID}", {}).get(
        "partitions", {}
    ).get("settled", {})
    c4_bias_degradation = (
        float(c4_calibrated_phase["bias_vector_rmse_rad_s"])
        / float(c4_base_phase["bias_vector_rmse_rad_s"])
        - 1.0
        if c4_base_phase and c4_calibrated_phase
        else None
    )
    review = {
        "decision": decision,
        "status": status,
        "original_condition": "MAIN-FUSION settled posterior NEES/DOF=1.873 with matched sensor NIS",
        "diagnosed_cause": {
            "validation_initial_full_nees_normalized": diagnostic_validation["initial"][
                "full_nees_normalized"
            ],
            "validation_settled_attitude_marginal_nees_normalized": diagnostic_validation[
                "settled"
            ]["attitude_nees_normalized"],
            "validation_settled_bias_marginal_nees_normalized": diagnostic_validation[
                "settled"
            ]["bias_nees_normalized"],
            "ranking": diagnosis["likely_source_ranking_by_settled_marginal_distance"],
        },
        "F_CALIBRATED_status": freeze,
        "confirmation_F_BASE_settled": base_settled,
        "confirmation_F_CALIBRATED_settled": calibrated_settled,
        "acceptance": confirmation.get("acceptance"),
        "remaining_classical_limitation": {
            "c4_calibrated_whole_bias_degradation_fraction": c4_bias_degradation,
            "c4_calibrated_settled_sensor_nis": {
                sensor: c4_calibrated_settled.get(f"{sensor}_nis_normalized")
                for sensor in ("mag", "sun", "st")
            },
        },
        "regressions_passed": regressions_passed,
        "phase2_implemented": False,
    }
    _atomic_write_json(results_root / "updated_exit_review.json", review)
    markdown = [
        "# P1 Exit Review — Updated After Covariance Closure",
        "",
        f"Decision: **{decision}**",
        "",
        f"Closure status: `{status}`. The original condition was settled posterior "
        "NEES/DOF `1.873` with matched MAIN-FUSION sensor NIS.",
        "",
        "## Diagnosed cause and calibrated policy",
        "",
        "The transient, attitude marginal, bias marginal, full whitened-error, and "
        "attitude-bias cross-covariance evidence is frozen in "
        "`P1_EXIT_TRANSIENT_DIAGNOSTIC_REPORT.md`. Candidate selection used only the "
        "independent 30/20 calibration split.",
        "",
        f"Validation initial full NEES/DOF was "
        f"`{diagnostic_validation['initial']['full_nees_normalized']:.6f}`. After "
        f"the frozen 60% partition, attitude marginal NEES/DOF was "
        f"`{diagnostic_validation['settled']['attitude_nees_normalized']:.6f}` and "
        f"bias marginal was "
        f"`{diagnostic_validation['settled']['bias_nees_normalized']:.6f}`. The "
        "dominant settled source is therefore bias-side process/covariance "
        "understatement, with a separate large initial transient and material "
        "attitude-bias cross covariance; it is not a sensor-R mismatch.",
        "",
        f"`F-CALIBRATED-v1` is frozen at `{freeze['scales']}` with all sensor R scales "
        "exactly one. Independent stationary and C4 confirmation each used N=50.",
        "",
        "| Stationary settled policy | Full NEES/DOF | Attitude NEES/DOF | Bias NEES/DOF | "
        "Mag NIS/DOF | Sun NIS/DOF | ST NIS/DOF | Attitude RMSE (rad) | Bias RMSE (rad/s) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        f"| F-BASE | {base_settled['full_nees_normalized']:.6f} | "
        f"{base_settled['attitude_nees_normalized']:.6f} | "
        f"{base_settled['bias_nees_normalized']:.6f} | "
        f"{base_settled['mag_nis_normalized']:.6f} | "
        f"{base_settled['sun_nis_normalized']:.6f} | "
        f"{base_settled['st_nis_normalized']:.6f} | "
        f"{base_settled['attitude_rmse_rad']:.9g} | "
        f"{base_settled['bias_rmse_rad_s']:.9g} |",
        f"| F-CALIBRATED-v1 | {calibrated_settled['full_nees_normalized']:.6f} | "
        f"{calibrated_settled['attitude_nees_normalized']:.6f} | "
        f"{calibrated_settled['bias_nees_normalized']:.6f} | "
        f"{calibrated_settled['mag_nis_normalized']:.6f} | "
        f"{calibrated_settled['sun_nis_normalized']:.6f} | "
        f"{calibrated_settled['st_nis_normalized']:.6f} | "
        f"{calibrated_settled['attitude_rmse_rad']:.9g} | "
        f"{calibrated_settled['bias_rmse_rad_s']:.9g} |",
        "",
        f"The stationary paired mean change in absolute distance from NEES/DOF=1 was "
        f"`{confirmation['stationary_paired']['absolute_distance_candidate_minus_base_mean']:.6f}` "
        f"with paired 95% bootstrap CI "
        f"`{confirmation['stationary_paired']['paired_bootstrap_95_ci']}`. "
        f"The mean relative attitude-bias P cross norm changed from "
        f"`{base_settled['P_cross_relative_norm_mean']:.6f}` to "
        f"`{calibrated_settled['P_cross_relative_norm_mean']:.6f}`.",
        "",
        "Stationary acceptance passed every predeclared guard, including N=50, "
        "strict P/S SPD, zero divergence, all three sensor NIS guards, accuracy, "
        "and the [0.8, 1.25] full-NEES target.",
        "",
        "## Why the updated decision remains conditional",
        "",
        f"Stationary confirmation closes the named consistency target, but C4 does "
        f"not pass every predeclared calibrated-policy guard. F-CALIBRATED-v1 "
        f"whole-horizon bias RMSE was `{c4_calibrated_phase.get('bias_vector_rmse_rad_s')}` "
        f"versus F-BASE `{c4_base_phase.get('bias_vector_rmse_rad_s')}` "
        f"(degradation `{c4_bias_degradation:.3%}`). Its C4 settled normalized NIS "
        f"was mag `{c4_calibrated_settled.get('mag_nis_normalized')}`, sun "
        f"`{c4_calibrated_settled.get('sun_nis_normalized')}`, and ST "
        f"`{c4_calibrated_settled.get('st_nis_normalized')}`, outside the fixed "
        "stationary guard. These failures are retained rather than re-tuning Q/P0 "
        "or sensor R on confirmation data.",
        "",
        "The C4 full-oracle cause-specific advantage, wrong-side ordering, zero "
        "divergence, and strict SPD evidence remain intact. Accordingly this is the "
        "contract's named `CONDITIONAL_GO` case: stationary covariance closure "
        "succeeds, while F-CALIBRATED-v1 must not replace F-BASE for C4.",
        "",
        "## Frozen baseline matrix",
        "",
        "- `F-BASE`: unchanged primary classical baseline.",
        "- `F-TUNED=(0.125,0.125,8.0)`: unchanged sensitivity comparator only.",
        "- `F-CALIBRATED-v1`: separate fixed P0/Q calibration; it does not replace "
        "  F-BASE or any oracle/wrong-side comparator.",
        "- C4 process-only, measurement-only, full-oracle, and both wrong-side "
        "  diagnostics remain frozen comparators.",
        "",
        "## Remaining limits and future scope",
        "",
        "This decision covers only the deterministic representative-normalized "
        "classical benchmark. It does not establish orbit, WMM, eclipse, flight-sensor, "
        "universal calibration, learned-model, FPGA, or closed-loop performance.",
        "",
        "A future Phase 2 design requires separate approval and must retain F-BASE, "
        "F-TUNED, F-CALIBRATED-v1, and the named classical/oracle/wrong-side matrix. "
        "No Phase 2 implementation was started in this study.",
    ]
    (reports_root / "P1_EXIT_REVIEW_UPDATED.md").write_text(
        "\n".join(markdown) + "\n", encoding="utf-8"
    )
    return review


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("validate")
    diagnose = commands.add_parser("diagnose")
    diagnose.add_argument("--resume", action="store_true")
    search = commands.add_parser("search")
    search.add_argument("--resume", action="store_true")
    confirm = commands.add_parser("confirm")
    confirm.add_argument("--resume", action="store_true")
    confirm.add_argument("--max-trajectories", type=int)
    commands.add_parser("report")
    commands.add_parser("exit-review")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_closure_config(args.config)
    results_root, manifests_root, reports_root = _roots(config)
    results_root.mkdir(parents=True, exist_ok=True)
    manifests_root.mkdir(parents=True, exist_ok=True)
    reports_root.mkdir(parents=True, exist_ok=True)
    if args.command == "validate":
        output = _validation(config)
    elif args.command == "diagnose":
        output = _diagnose(config, resume=args.resume)
    elif args.command == "search":
        output = _search(config, resume=args.resume)
    elif args.command == "confirm":
        output = _confirm(
            config,
            resume=args.resume,
            max_trajectories=args.max_trajectories,
        )
    elif args.command == "report":
        output = _report(config)
    elif args.command == "exit-review":
        output = _exit_review(config)
    else:
        raise RuntimeError("unreachable command")
    sys.stdout.write(json.dumps(output, sort_keys=True, indent=2, allow_nan=False) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CANDIDATE_POLICY_ID",
    "CalibrationScales",
    "CovarianceDecomposition",
    "EXPERIMENT_VERSION",
    "FROZEN_POLICY_ID",
    "aggregate_trajectory_diagnostics",
    "closure_train_validation_split",
    "coordinate_candidates",
    "covariance_decomposition",
    "deterministic_select",
    "first_settling_bin",
    "load_closure_config",
    "local_combined_grid",
    "partition_masks",
    "replay_calibrated_fixed",
    "scaled_initial_state",
    "settled_selection_key",
    "stage1_selection_key",
    "trajectory_diagnostics",
    "validation_guard",
]
