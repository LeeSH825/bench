from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np


ARTIFACT_VERSION = "1.1"
SUPPORTED_ARTIFACT_VERSIONS = {"1.0", ARTIFACT_VERSION}
DEFAULT_V1_COVARIANCE_SPACE = "mrp"
VALID_COVARIANCE_SPACES = {"mrp", "rotation_vector_rad"}
VALID_DATA_SPLITS = {"train", "validation", "test", "unknown"}
VALID_SPLIT_SOURCES = {"explicit", "inferred", "legacy_unknown"}

CAPABILITY_KEYS = (
    "covariance",
    "gain",
    "innovation",
    "innovation_cov",
    "context",
    "regime_labels",
    "bias_state",
)

SOURCE_KEY_CANDIDATES = {
    "ref_mask": ("ref_mask_seq",),
    "bias_component": ("bias_component_seq",),
    "noise_component": ("noise_component_seq",),
    "imu_error": ("imu_error_seq",),
    "eclipse_flag": ("eclipse_flag_seq",),
    "event_flag": ("event_flag_seq",),
    "b_true": ("gyro_bias_seq",),
}

F32_TRAJ_KEYS = {
    "t",
    "x_true",
    "x_hat",
    "q_true",
    "q_hat",
    "sigma_nominal",
    "innov",
    "bias_component",
    "noise_component",
    "imu_error",
    "b_true",
    "torque_cmd",
    "sow_hat",
    "gate",
}

F16_TRAJ_KEYS = {"P", "S", "gain", "gain_g1", "gain_g2"}

BOOL_TRAJ_KEYS = {"innov_valid", "eclipse_flag", "event_flag", "ref_mask"}

SPLIT_GAIN_COMPONENT_SEMANTICS = {
    "gain": "learned_combined_kalman_gain",
    "gain_g1": "learned_split_factor_g1",
    "gain_g2": "learned_split_factor_g2",
    "validity_mask": "innov_valid",
}


class ContractError(ValueError):
    """Raised when a visualization artifact violates the v1 contract."""


class UnsupportedArtifactVersion(ContractError):
    """Raised when loader sees an artifact version it cannot interpret."""


def deterministic_traj_index(n_total: int, k: int = 8) -> List[int]:
    n = int(n_total)
    kk = int(k)
    if n <= 0 or kk <= 0:
        return []
    step = max(1, n // kk)
    return list(range(n))[::step][:kk]


def formulation_for_task(task_family: str, task_id: str = "") -> str:
    family = str(task_family or "").strip().lower()
    task_text = str(task_id or "").strip().lower()
    haystack = f"{family} {task_text}"
    if "sparse_ref" in haystack or "bias" in haystack:
        return "imu_meas_mrp_omega_bias_v0"
    if family == "basilisk_imu_adcs_v0" or "basilisk_imu_adcs" in haystack:
        return "imu_meas_mrp_omega_v0"
    if family == "basilisk_adcs_v0" or "basilisk_adcs" in haystack:
        return "full_state_mrp_omega_v0"
    sanitized = re.sub(r"[^a-z0-9_]+", "_", family or task_text or "unknown").strip("_")
    return f"{sanitized}_formulation"


def sanity_benchmark_only(formulation: str) -> bool:
    return str(formulation) == "full_state_mrp_omega_v0"


def source_key_map(extras: Optional[Mapping[str, Any]]) -> Dict[str, str]:
    if not isinstance(extras, Mapping):
        return {}
    out: Dict[str, str] = {}
    for artifact_key, candidates in SOURCE_KEY_CANDIDATES.items():
        for source_key in candidates:
            if source_key in extras:
                out[artifact_key] = source_key
                break
    return out


def empty_capabilities() -> Dict[str, bool]:
    return {key: False for key in CAPABILITY_KEYS}


def capabilities_for(
    *,
    model_id: str,
    diagnostics: Optional[Mapping[str, Any]],
    source_map: Mapping[str, str],
) -> Dict[str, bool]:
    _ = model_id
    caps = empty_capabilities()
    diag = diagnostics if isinstance(diagnostics, Mapping) else {}
    caps["covariance"] = diag.get("P") is not None
    caps["gain"] = any(diag.get(key) is not None for key in ("gain", "gain_g1", "gain_g2"))
    caps["innovation"] = diag.get("innov") is not None
    caps["innovation_cov"] = diag.get("S") is not None
    caps["context"] = False
    caps["regime_labels"] = "event_flag" in source_map or "eclipse_flag" in source_map or "ref_mask" in source_map
    caps["bias_state"] = "b_true" in source_map
    return caps


def state_spec_for(formulation: str, x_dim: int) -> Dict[str, Any]:
    form = str(formulation)
    if form in {"full_state_mrp_omega_v0", "imu_meas_mrp_omega_v0", "imu_meas_mrp_omega_bias_v0"}:
        layout: List[Dict[str, Any]] = [
            {"name": "sigma_BN", "dim": 3, "unit": "MRP", "kind": "attitude"},
        ]
        if int(x_dim) >= 6:
            layout.append({"name": "omega_BN_B", "dim": 3, "unit": "rad/s", "kind": "angular_rate"})
        if int(x_dim) >= 9 and form == "imu_meas_mrp_omega_bias_v0":
            layout.append({"name": "gyro_bias", "dim": 3, "unit": "rad/s", "kind": "bias"})
        attitude_repr: Optional[str] = "MRP"
        shadow_set = True
        rpy_reference_frame: Optional[str] = "inertial_N"
    else:
        layout = [{"name": "x", "dim": int(x_dim), "unit": "unknown", "kind": "state"}]
        attitude_repr = None
        shadow_set = False
        rpy_reference_frame = None
    return {
        "layout": layout,
        "attitude_repr": attitude_repr,
        "covariance_space": DEFAULT_V1_COVARIANCE_SPACE,
        "shadow_set": shadow_set,
        "nominal_attitude_stored": False,
        "rpy_reference_frame": rpy_reference_frame,
    }


def normalize_meta(meta: Mapping[str, Any]) -> Dict[str, Any]:
    normalized = dict(meta)
    version = normalized.get("artifact_version")
    if version not in SUPPORTED_ARTIFACT_VERSIONS:
        raise UnsupportedArtifactVersion(f"unsupported artifact_version={version!r}")
    state_spec = normalized.get("state_spec")
    if not isinstance(state_spec, Mapping):
        raise ContractError("meta.state_spec must be a mapping")
    normalized_state = dict(state_spec)
    if version == "1.0" and "covariance_space" not in normalized_state:
        normalized_state["covariance_space"] = DEFAULT_V1_COVARIANCE_SPACE
    covariance_space = normalized_state.get("covariance_space")
    if covariance_space not in VALID_COVARIANCE_SPACES:
        raise ContractError(f"unknown state_spec.covariance_space={covariance_space!r}")
    normalized["state_spec"] = normalized_state
    raw_data_spec = normalized.get("data_spec")
    if isinstance(raw_data_spec, Mapping):
        data_spec = dict(raw_data_spec)
    else:
        traj_index = normalized.get("traj_index")
        legacy_stored = len(traj_index) if isinstance(traj_index, list) else 0
        legacy_total = max(int(normalized.get("N_test", 0) or 0), legacy_stored)
        data_spec = {
            "split": "unknown",
            "split_source": "legacy_unknown",
            "num_trajectories": legacy_total,
            "num_stored_trajectories": legacy_stored,
            "trajectory_selection": "legacy_unspecified",
            "is_live": False,
        }
    normalized["data_spec"] = data_spec
    return normalized


def meas_spec_for(task_family: str, y_dim: int) -> Dict[str, Any]:
    family = str(task_family or "").strip().lower()
    yd = int(y_dim)
    if "sparse_ref" in family and yd >= 9:
        channels = [
            {"name": "gyro", "dim": 3, "unit": "rad/s", "role": "measurement", "columns": [0, 1, 2]},
            {"name": "delta_theta", "dim": 3, "unit": "rad", "role": "measurement", "columns": [3, 4, 5]},
            {
                "name": "sigma_ref",
                "dim": 3,
                "unit": "MRP",
                "role": "measurement",
                "columns": [6, 7, 8],
                "gated_by": "ref_mask",
            },
        ]
    elif "imu" in family and yd >= 6:
        channels = [
            {"name": "gyro", "dim": 3, "unit": "rad/s", "role": "measurement", "columns": [0, 1, 2]},
            {"name": "delta_theta", "dim": 3, "unit": "rad", "role": "measurement", "columns": [3, 4, 5]},
        ]
    elif yd >= 6:
        channels = [
            {"name": "sigma_meas", "dim": 3, "unit": "MRP", "role": "measurement", "columns": [0, 1, 2]},
            {"name": "omega_meas", "dim": 3, "unit": "rad/s", "role": "measurement", "columns": [3, 4, 5]},
        ]
    else:
        channels = [{"name": "y", "dim": yd, "unit": "unknown", "role": "measurement"}]
    return {"channels": channels}


def validate_meta(meta: Mapping[str, Any]) -> None:
    normalized = normalize_meta(meta)
    caps = meta.get("capabilities")
    if not isinstance(caps, Mapping):
        raise ContractError("meta.capabilities must be a mapping")
    missing = [key for key in CAPABILITY_KEYS if key not in caps]
    if missing:
        raise ContractError(f"meta.capabilities missing keys: {missing}")
    non_bool = [key for key in CAPABILITY_KEYS if not isinstance(caps.get(key), bool)]
    if non_bool:
        raise ContractError(f"meta.capabilities values must be bool for keys: {non_bool}")
    if normalized["state_spec"].get("covariance_space") not in VALID_COVARIANCE_SPACES:
        raise ContractError("meta.state_spec.covariance_space is invalid")
    data_spec = normalized.get("data_spec")
    if not isinstance(data_spec, Mapping):
        raise ContractError("meta.data_spec must be a mapping")
    split = data_spec.get("split")
    if split not in VALID_DATA_SPLITS:
        raise ContractError(f"unknown data_spec.split={split!r}")
    split_source = data_spec.get("split_source")
    if split_source not in VALID_SPLIT_SOURCES:
        raise ContractError(f"unknown data_spec.split_source={split_source!r}")
    for key in ("num_trajectories", "num_stored_trajectories"):
        value = data_spec.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ContractError(f"meta.data_spec.{key} must be a non-negative integer")
    if int(data_spec["num_stored_trajectories"]) > int(data_spec["num_trajectories"]):
        raise ContractError("meta.data_spec.num_stored_trajectories exceeds num_trajectories")
    selection = data_spec.get("trajectory_selection")
    if not isinstance(selection, str) or not selection:
        raise ContractError("meta.data_spec.trajectory_selection must be a non-empty string")
    if data_spec.get("is_live") is not False:
        raise ContractError("only offline artifacts with data_spec.is_live=false are supported")
    manifest = normalized.get("trajectories")
    if manifest is not None:
        validate_trajectory_manifest(normalized, manifest)
    diagnostic_semantics = normalized.get("diagnostic_semantics", {})
    if not isinstance(diagnostic_semantics, Mapping):
        raise ContractError("meta.diagnostic_semantics must be a mapping")
    for key, expected in SPLIT_GAIN_COMPONENT_SEMANTICS.items():
        if key in diagnostic_semantics and diagnostic_semantics.get(key) != expected:
            raise ContractError(
                f"meta.diagnostic_semantics.{key} must be {expected!r}, "
                f"got {diagnostic_semantics.get(key)!r}"
            )


def validate_trajectory_manifest(meta: Mapping[str, Any], manifest: Any) -> None:
    if not isinstance(manifest, list):
        raise ContractError("meta.trajectories must be a list")
    stored_indices: set[int] = set()
    source_ids: set[tuple[str, str]] = set()
    files: set[str] = set()
    for position, item in enumerate(manifest):
        if not isinstance(item, Mapping):
            raise ContractError(f"meta.trajectories[{position}] must be a mapping")
        stored_index = item.get("stored_index")
        if not isinstance(stored_index, int) or isinstance(stored_index, bool) or stored_index < 0:
            raise ContractError(f"meta.trajectories[{position}].stored_index must be a non-negative integer")
        if stored_index in stored_indices:
            raise ContractError(f"duplicate stored trajectory index: {stored_index}")
        stored_indices.add(stored_index)
        source_id = item.get("source_trajectory_id")
        if source_id is None or isinstance(source_id, (dict, list, bool)):
            raise ContractError(f"meta.trajectories[{position}].source_trajectory_id must be a scalar")
        source_key = (type(source_id).__name__, str(source_id))
        if source_key in source_ids:
            raise ContractError(f"duplicate source trajectory ID: {source_id!r}")
        source_ids.add(source_key)
        file_name = item.get("file")
        expected_file = f"series/traj_{stored_index:04d}.npz"
        if file_name != expected_file:
            raise ContractError(
                f"meta.trajectories[{position}].file must be {expected_file!r}, got {file_name!r}"
            )
        if file_name in files:
            raise ContractError(f"duplicate trajectory file in manifest: {file_name}")
        files.add(file_name)
        length = item.get("length_T")
        if not isinstance(length, int) or isinstance(length, bool) or length <= 0:
            raise ContractError(f"meta.trajectories[{position}].length_T must be a positive integer")
        for key in ("has_event", "has_eclipse"):
            if item.get(key) is not None and not isinstance(item.get(key), bool):
                raise ContractError(f"meta.trajectories[{position}].{key} must be bool or null")
    expected_indices = set(range(len(manifest)))
    if stored_indices != expected_indices:
        raise ContractError(
            f"stored trajectory indices must be contiguous from zero; got {sorted(stored_indices)}"
        )
    data_spec = meta.get("data_spec", {})
    if int(data_spec.get("num_stored_trajectories", -1)) != len(manifest):
        raise ContractError("meta.data_spec.num_stored_trajectories does not match trajectory manifest")


def validate_traj_arrays(arrays: Mapping[str, np.ndarray]) -> None:
    for key, arr in arrays.items():
        if key in F32_TRAJ_KEYS and np.asarray(arr).dtype != np.float32:
            raise ContractError(f"{key} must be float32, got {np.asarray(arr).dtype}")
        if key in F16_TRAJ_KEYS and np.asarray(arr).dtype != np.float16:
            raise ContractError(f"{key} must be float16, got {np.asarray(arr).dtype}")
        if key in BOOL_TRAJ_KEYS and np.asarray(arr).dtype != np.bool_:
            raise ContractError(f"{key} must be bool, got {np.asarray(arr).dtype}")


def validate_trajectory_capabilities(meta: Mapping[str, Any], arrays: Mapping[str, np.ndarray]) -> None:
    caps = meta.get("capabilities")
    if not isinstance(caps, Mapping):
        raise ContractError("meta.capabilities must be a mapping")
    requirements = {
        "covariance": ("P",),
        "innovation": ("innov",),
        "innovation_cov": ("S",),
        "gain": ("gain", "gain_g1", "gain_g2"),
    }
    for cap_key, array_keys in requirements.items():
        if bool(caps.get(cap_key)) and not any(key in arrays for key in array_keys):
            raise ContractError(f"capabilities.{cap_key}=true but trajectory is missing one of {array_keys}")
    strict_false_keys = {
        "covariance": ("P",),
        "innovation": ("innov",),
        "innovation_cov": ("S",),
        "gain": ("gain", "gain_g1", "gain_g2"),
    }
    for cap_key, array_keys in strict_false_keys.items():
        present = [key for key in array_keys if key in arrays]
        if present and not bool(caps.get(cap_key)):
            raise ContractError(f"trajectory contains {present} but capabilities.{cap_key}=false")

    t = np.asarray(arrays.get("t"))
    if t.ndim != 1:
        raise ContractError(f"trajectory t must have shape [T], got {t.shape}")
    t_len = int(t.shape[0])
    expected_ranks = {
        "innov": 2,
        "P": 3,
        "S": 3,
        "gain": 3,
        "gain_g1": 3,
        "gain_g2": 3,
    }
    for key, rank in expected_ranks.items():
        if key not in arrays:
            continue
        arr = np.asarray(arrays[key])
        if arr.ndim != rank:
            raise ContractError(f"{key} must have rank {rank} with time first, got {arr.shape}")
        if int(arr.shape[0]) != t_len:
            raise ContractError(f"{key} time dimension {arr.shape[0]} does not match T={t_len}")
    for key in ("P", "S", "gain_g1", "gain_g2"):
        if key in arrays:
            arr = np.asarray(arrays[key])
            if arr.shape[1] != arr.shape[2]:
                raise ContractError(f"{key} must contain square matrices, got {arr.shape}")

    component_keys = {key for key in ("gain_g1", "gain_g2") if key in arrays}
    if component_keys and component_keys != {"gain_g1", "gain_g2"}:
        raise ContractError("Split gain components must contain both gain_g1 and gain_g2")
    if component_keys and "gain" not in arrays:
        raise ContractError("Split gain components require the combined gain key")
    if component_keys:
        semantics = meta.get("diagnostic_semantics")
        if not isinstance(semantics, Mapping):
            raise ContractError("Split gain components require meta.diagnostic_semantics")
        for key, expected in SPLIT_GAIN_COMPONENT_SEMANTICS.items():
            if semantics.get(key) != expected:
                raise ContractError(
                    f"Split gain component semantic {key!r} must be {expected!r}, got {semantics.get(key)!r}"
                )
        gain_arr = np.asarray(arrays["gain"])
        g1_arr = np.asarray(arrays["gain_g1"])
        g2_arr = np.asarray(arrays["gain_g2"])
        if gain_arr.shape[1:] != (g1_arr.shape[1], g2_arr.shape[1]):
            raise ContractError(
                "Split gain component dimensions do not match combined gain: "
                f"gain={gain_arr.shape}, gain_g1={g1_arr.shape}, gain_g2={g2_arr.shape}"
            )

    valid_mask = np.ones((t_len,), dtype=bool)
    if "innov_valid" in arrays:
        valid_mask = np.asarray(arrays["innov_valid"])
        if valid_mask.ndim != 1 or valid_mask.shape[0] != t_len:
            raise ContractError(f"innov_valid must have shape [{t_len}], got {valid_mask.shape}")
    for key in ("innov", "gain", "gain_g1", "gain_g2"):
        if key in arrays and not np.all(np.isfinite(np.asarray(arrays[key])[valid_mask])):
            raise ContractError(f"{key} contains NaN/Inf at a valid diagnostic timestep")
    for key in ("P", "S"):
        if key in arrays and not np.all(np.isfinite(np.asarray(arrays[key]))):
            raise ContractError(f"{key} contains NaN/Inf")


def sorted_run_key(meta: Mapping[str, Any], run_dir: Path) -> tuple[Any, ...]:
    return (
        str(meta.get("suite", "")),
        str(meta.get("task", "")),
        str(meta.get("scenario_id", "")),
        str(meta.get("model_id", "")),
        int(meta.get("seed", -1)) if str(meta.get("seed", "")).lstrip("-").isdigit() else -1,
        str(run_dir),
    )


def require_overlay_compatible(base_meta: Mapping[str, Any], overlay_meta: Mapping[str, Any]) -> None:
    base = normalize_meta(base_meta)
    overlay = normalize_meta(overlay_meta)
    base_split = base["data_spec"]["split"]
    overlay_split = overlay["data_spec"]["split"]
    if base_split != overlay_split:
        raise ContractError(
            f"overlay blocked: data split mismatch (base={base_split!r}, overlay={overlay_split!r})"
        )
    for key in ("task", "scenario_id", "seed", "formulation"):
        if base.get(key) != overlay.get(key):
            raise ContractError(
                f"overlay blocked: {key} differs ({base.get(key)!r} != {overlay.get(key)!r})"
            )
    if base.get("state_spec") != overlay.get("state_spec"):
        raise ContractError("overlay blocked: state_spec differs")
