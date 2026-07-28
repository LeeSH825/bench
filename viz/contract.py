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
    "correction_attitude",
    "correction_bias",
}

F16_TRAJ_KEYS = {"P", "S", "gain", "gain_g1", "gain_g2"}

BOOL_TRAJ_KEYS = {"innov_valid", "eclipse_flag", "event_flag", "ref_mask", "correction_valid"}

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


def _layout_blocks(state_spec: Mapping[str, Any]) -> Dict[str, tuple[int, int, Mapping[str, Any]]]:
    blocks: Dict[str, tuple[int, int, Mapping[str, Any]]] = {}
    offset = 0
    for item in state_spec.get("layout", []):
        if not isinstance(item, Mapping):
            continue
        dim = int(item.get("dim", 0))
        kind = str(item.get("kind", "state"))
        blocks.setdefault(kind, (offset, offset + dim, item))
        offset += dim
    return blocks


def _expanded_state_semantics(state_spec: Mapping[str, Any]) -> tuple[list[str], list[str]]:
    names: list[str] = []
    units: list[str] = []
    axis_names = ("x", "y", "z")
    for item in state_spec.get("layout", []):
        if not isinstance(item, Mapping):
            continue
        dim = int(item.get("dim", 0))
        name = str(item.get("name", "state"))
        unit = str(item.get("unit", "unknown"))
        for index in range(dim):
            suffix = axis_names[index] if index < len(axis_names) else str(index)
            names.append(f"{name}_{suffix}")
            units.append(unit)
    return names, units


def _expanded_measurement_semantics(meas_spec: Mapping[str, Any]) -> tuple[list[str], list[str]]:
    names: list[str] = []
    units: list[str] = []
    axis_names = ("x", "y", "z")
    for item in meas_spec.get("channels", []):
        if not isinstance(item, Mapping):
            continue
        dim = int(item.get("dim", 0))
        name = str(item.get("name", "measurement"))
        unit = str(item.get("unit", "unknown"))
        for index in range(dim):
            suffix = axis_names[index] if index < len(axis_names) else str(index)
            names.append(f"{name}_{suffix}")
            units.append(unit)
    return names, units


def comparison_spec_for(
    *,
    state_spec: Mapping[str, Any],
    meas_spec: Mapping[str, Any],
    capabilities: Mapping[str, bool],
    diagnostic_semantics: Mapping[str, Any],
    adapter_meta: Mapping[str, Any],
    identity: Mapping[str, Any],
    n_samples: int,
) -> Dict[str, Any]:
    blocks = _layout_blocks(state_spec)
    state_order, state_units = _expanded_state_semantics(state_spec)
    measurement_order, measurement_units = _expanded_measurement_semantics(meas_spec)
    attitude_block_info = blocks.get("attitude") or blocks.get("attitude_error")
    bias_block_info = blocks.get("bias")
    attitude_block = (
        [int(attitude_block_info[0]), int(attitude_block_info[1])]
        if attitude_block_info is not None
        else None
    )
    bias_block = [int(bias_block_info[0]), int(bias_block_info[1])] if bias_block_info is not None else None

    attitude_coordinate_space = None
    if attitude_block_info is not None:
        attitude_unit = str(attitude_block_info[2].get("unit", ""))
        if attitude_unit == "MRP":
            attitude_coordinate_space = "mrp"
        elif attitude_unit in {"rad", "rotation_vector_rad"}:
            attitude_coordinate_space = "rotation_vector_rad"

    attitude_available = attitude_block is not None and attitude_coordinate_space is not None
    bias_available = bias_block is not None and bool(capabilities.get("bias_state"))
    covariance_available = bool(capabilities.get("covariance"))
    innovation_available = bool(capabilities.get("innovation"))
    gain_available = bool(capabilities.get("gain"))
    innovation_dimension = len(measurement_order)
    state_dimension = len(state_order)
    gain_semantic = diagnostic_semantics.get("gain") or adapter_meta.get("gain_semantics")
    if gain_semantic is None:
        gain_semantic = (
            "model_based_kalman_gain"
            if covariance_available and bool(capabilities.get("innovation_cov"))
            else "gain_mapping_measurement_residual_to_state_correction"
        )

    return {
        "schema_version": "1",
        "comparison_source": "explicit_writer_v1_1",
        "identity": dict(identity),
        "attitude": {
            "available": attitude_available,
            "reason": None if attitude_available else "state_spec has no supported attitude state",
            "estimate_key": "q_hat" if attitude_available else None,
            "truth_key": "q_true" if attitude_available else None,
            "representation": "quaternion" if attitude_available else None,
            "quaternion_order": "wxyz" if attitude_available else None,
            "rotation_direction": "body_to_inertial" if attitude_available else None,
            "frame_from": "body_B" if attitude_available else None,
            "frame_to": "inertial_N" if attitude_available else None,
            "rpy_sequence": "ZYX" if attitude_available else None,
            "rpy_convention": "intrinsic" if attitude_available else None,
            "rpy_output_order": ["roll", "pitch", "yaw"] if attitude_available else None,
            "error_definition": "geodesic_relative_rotation" if attitude_available else None,
            "state_block": attitude_block,
            "state_coordinate_space": attitude_coordinate_space,
        },
        "bias": {
            "available": bias_available,
            "reason": None if bias_available else "bias state truth is not available",
            "estimate_key": "x_hat" if bias_available else None,
            "estimate_block": bias_block,
            "truth_key": "b_true" if bias_available else None,
            "units": "rad_per_s" if bias_available else None,
            "frame": "body_B" if bias_available else None,
        },
        "covariance": {
            "available": covariance_available,
            "physical": covariance_available,
            "key": "P" if covariance_available else None,
            "space": state_spec.get("covariance_space") if covariance_available else None,
            "state_order": state_order,
            "attitude_block": attitude_block if covariance_available else None,
            "bias_block": bias_block if covariance_available else None,
            "posterior_stage": "posterior_filter_state" if covariance_available else None,
        },
        "innovation": {
            "available": innovation_available,
            "key": "innov" if innovation_available else None,
            "measurement_type": "ordered_measurement_channels" if innovation_available else None,
            "residual_definition": "measurement_minus_predicted_measurement" if innovation_available else None,
            "units": measurement_units if innovation_available else None,
            "frame": "declared_per_channel_or_unspecified" if innovation_available else None,
            "dimension": innovation_dimension if innovation_available else None,
            "measurement_order": measurement_order if innovation_available else None,
            "valid_mask_key": "innov_valid" if innovation_available else None,
        },
        "gain": {
            "available": gain_available,
            "key": "gain" if gain_available else None,
            "semantic": str(gain_semantic) if gain_available else None,
            "row_state_order": state_order if gain_available else None,
            "column_measurement_order": measurement_order if gain_available else None,
            "row_units": state_units if gain_available else None,
            "column_units": measurement_units if gain_available else None,
            "state_scaling": "native_state_coordinates" if gain_available else None,
            "measurement_scaling": "native_measurement_coordinates" if gain_available else None,
            "shape": [state_dimension, innovation_dimension] if gain_available else None,
            "attitude_block": attitude_block if gain_available else None,
            "bias_block": bias_block if gain_available else None,
        },
        "correction": {
            "available": gain_available and innovation_available,
            "source": "reconstructed_gain_times_innovation" if gain_available and innovation_available else None,
            "actual_applied": False if gain_available and innovation_available else None,
            "gain_key": "gain" if gain_available and innovation_available else None,
            "innovation_key": "innov" if gain_available and innovation_available else None,
            "valid_mask_key": "innov_valid" if gain_available and innovation_available else None,
            "attitude_block": attitude_block if gain_available and innovation_available else None,
            "attitude_coordinate_space": attitude_coordinate_space if gain_available and innovation_available else None,
            "bias_block": bias_block if gain_available and innovation_available else None,
            "bias_units": "rad_per_s" if bias_block is not None and gain_available and innovation_available else None,
        },
        "empirical_uncertainty": {
            "available": int(n_samples) >= 2,
            "key": "emp_std" if int(n_samples) >= 2 else None,
            "source": "sample_standard_deviation_of_trajectory_estimation_errors" if int(n_samples) >= 2 else None,
            "physical": False,
            "sample_count": int(n_samples),
        },
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
    comparison_spec = normalized.get("comparison_spec")
    if comparison_spec is not None:
        validate_comparison_spec(comparison_spec)


def _validate_block(value: Any, *, dimension: int, label: str, required: bool = False) -> None:
    if value is None and not required:
        return
    if not isinstance(value, list) or len(value) != 2:
        raise ContractError(f"{label} must be [start,end] or null")
    start, end = value
    if (
        not isinstance(start, int)
        or isinstance(start, bool)
        or not isinstance(end, int)
        or isinstance(end, bool)
        or start < 0
        or end <= start
        or end > int(dimension)
    ):
        raise ContractError(f"{label}={value!r} is outside dimension {dimension}")


def validate_comparison_spec(spec: Mapping[str, Any]) -> None:
    if not isinstance(spec, Mapping):
        raise ContractError("meta.comparison_spec must be a mapping")
    if spec.get("schema_version") != "1":
        raise ContractError(f"unsupported comparison_spec.schema_version={spec.get('schema_version')!r}")
    if spec.get("comparison_source") not in {"explicit_writer_v1_1", "synthetic_fixture", "explicit_adapter"}:
        raise ContractError(f"unknown comparison_spec.comparison_source={spec.get('comparison_source')!r}")
    identity = spec.get("identity")
    if not isinstance(identity, Mapping):
        raise ContractError("comparison_spec.identity must be a mapping")
    if not isinstance(identity.get("physical_scenario_id"), str) or not identity.get("physical_scenario_id"):
        raise ContractError("comparison_spec.identity.physical_scenario_id must be a non-empty string")
    truth_fingerprints = identity.get("truth_fingerprints")
    if not isinstance(truth_fingerprints, Mapping):
        raise ContractError("comparison_spec.identity.truth_fingerprints must be a mapping")

    for section_name in (
        "attitude",
        "bias",
        "covariance",
        "innovation",
        "gain",
        "correction",
        "empirical_uncertainty",
    ):
        if not isinstance(spec.get(section_name), Mapping):
            raise ContractError(f"comparison_spec.{section_name} must be a mapping")

    covariance = spec["covariance"]
    state_order = covariance.get("state_order")
    if not isinstance(state_order, list) or not all(isinstance(value, str) and value for value in state_order):
        raise ContractError("comparison_spec.covariance.state_order must be a list of names")
    state_dim = len(state_order)

    attitude_spec = spec["attitude"]
    if bool(attitude_spec.get("available")):
        expected = {
            "representation": "quaternion",
            "quaternion_order": "wxyz",
            "rotation_direction": "body_to_inertial",
            "frame_from": "body_B",
            "frame_to": "inertial_N",
            "rpy_sequence": "ZYX",
            "rpy_convention": "intrinsic",
            "rpy_output_order": ["roll", "pitch", "yaw"],
            "error_definition": "geodesic_relative_rotation",
        }
        for key, value in expected.items():
            if attitude_spec.get(key) != value:
                raise ContractError(f"unsupported comparison_spec.attitude.{key}={attitude_spec.get(key)!r}")
        if attitude_spec.get("estimate_key") != "q_hat" or attitude_spec.get("truth_key") != "q_true":
            raise ContractError("comparison_spec.attitude must use q_hat/q_true canonical keys")
        if attitude_spec.get("state_coordinate_space") not in VALID_COVARIANCE_SPACES:
            raise ContractError("comparison_spec.attitude.state_coordinate_space is unsupported")
        if not isinstance(truth_fingerprints.get("attitude_truth"), str):
            raise ContractError("comparison attitude requires an attitude_truth fingerprint")
        _validate_block(attitude_spec.get("state_block"), dimension=state_dim, label="comparison_spec.attitude.state_block", required=True)

    bias_spec = spec["bias"]
    if bool(bias_spec.get("available")):
        if bias_spec.get("estimate_key") != "x_hat" or bias_spec.get("truth_key") != "b_true":
            raise ContractError("comparison_spec.bias must use x_hat/b_true canonical keys")
        if bias_spec.get("units") != "rad_per_s" or bias_spec.get("frame") != "body_B":
            raise ContractError("comparison_spec.bias units/frame must be rad_per_s/body_B")
        if not isinstance(truth_fingerprints.get("bias_truth"), str):
            raise ContractError("comparison bias requires a bias_truth fingerprint")
        _validate_block(bias_spec.get("estimate_block"), dimension=state_dim, label="comparison_spec.bias.estimate_block", required=True)

    if bool(covariance.get("available")):
        if covariance.get("physical") is not True or covariance.get("key") != "P":
            raise ContractError("available comparison covariance must be physical trajectory key P")
        if covariance.get("space") not in VALID_COVARIANCE_SPACES:
            raise ContractError(f"unknown comparison covariance space={covariance.get('space')!r}")
        _validate_block(covariance.get("attitude_block"), dimension=state_dim, label="comparison_spec.covariance.attitude_block")
        _validate_block(covariance.get("bias_block"), dimension=state_dim, label="comparison_spec.covariance.bias_block")
    elif covariance.get("physical") not in {False, None}:
        raise ContractError("comparison covariance cannot be physical when unavailable")

    innovation = spec["innovation"]
    if bool(innovation.get("available")):
        dimension = innovation.get("dimension")
        order = innovation.get("measurement_order")
        units = innovation.get("units")
        if not isinstance(dimension, int) or isinstance(dimension, bool) or dimension <= 0:
            raise ContractError("comparison_spec.innovation.dimension must be positive")
        if not isinstance(order, list) or len(order) != dimension or not all(isinstance(value, str) for value in order):
            raise ContractError("comparison_spec.innovation.measurement_order does not match dimension")
        if not isinstance(units, list) or len(units) != dimension or not all(isinstance(value, str) for value in units):
            raise ContractError("comparison_spec.innovation.units does not match dimension")
        if innovation.get("key") != "innov" or innovation.get("valid_mask_key") != "innov_valid":
            raise ContractError("comparison_spec.innovation must use innov/innov_valid keys")

    gain_spec = spec["gain"]
    if bool(gain_spec.get("available")):
        shape = gain_spec.get("shape")
        if not isinstance(shape, list) or len(shape) != 2 or shape[0] != state_dim:
            raise ContractError("comparison_spec.gain.shape does not match state dimension")
        measurement_dim = innovation.get("dimension")
        if not isinstance(measurement_dim, int) or shape[1] != measurement_dim:
            raise ContractError("comparison_spec.gain.shape does not match innovation dimension")
        for key, expected_length in (
            ("row_state_order", shape[0]),
            ("row_units", shape[0]),
            ("column_measurement_order", shape[1]),
            ("column_units", shape[1]),
        ):
            value = gain_spec.get(key)
            if not isinstance(value, list) or len(value) != expected_length:
                raise ContractError(f"comparison_spec.gain.{key} length mismatch")
        _validate_block(gain_spec.get("attitude_block"), dimension=state_dim, label="comparison_spec.gain.attitude_block")
        _validate_block(gain_spec.get("bias_block"), dimension=state_dim, label="comparison_spec.gain.bias_block")

    correction = spec["correction"]
    if bool(correction.get("available")):
        if correction.get("source") not in {"reconstructed_gain_times_innovation", "actual_applied"}:
            raise ContractError(f"unsupported comparison correction source={correction.get('source')!r}")
        if correction.get("source") == "reconstructed_gain_times_innovation" and correction.get("actual_applied") is not False:
            raise ContractError("reconstructed correction must declare actual_applied=false")
        _validate_block(correction.get("attitude_block"), dimension=state_dim, label="comparison_spec.correction.attitude_block")
        _validate_block(correction.get("bias_block"), dimension=state_dim, label="comparison_spec.correction.bias_block")

    empirical = spec["empirical_uncertainty"]
    if empirical.get("physical") is not False:
        raise ContractError("comparison empirical uncertainty must declare physical=false")
    if bool(empirical.get("available")) and empirical.get("key") != "emp_std":
        raise ContractError("comparison empirical uncertainty must use aggregate key emp_std")


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
    validate_comparison_trajectory(meta, arrays)


def validate_comparison_trajectory(meta: Mapping[str, Any], arrays: Mapping[str, np.ndarray]) -> None:
    spec = meta.get("comparison_spec")
    if spec is None:
        return
    validate_comparison_spec(spec)
    t_len = int(np.asarray(arrays.get("t")).shape[0])
    state_dim = int(np.asarray(arrays.get("x_hat")).shape[1]) if "x_hat" in arrays else 0

    attitude_spec = spec["attitude"]
    if bool(attitude_spec.get("available")):
        for key in ("q_true", "q_hat"):
            if key not in arrays:
                raise ContractError(f"comparison_spec.attitude declares {key} but trajectory key is missing")
            q = np.asarray(arrays[key], dtype=np.float64)
            if q.shape != (t_len, 4):
                raise ContractError(f"{key} must have shape [{t_len},4], got {q.shape}")
            if not np.all(np.isfinite(q)):
                raise ContractError(f"{key} contains NaN/Inf")
            norms = np.linalg.norm(q, axis=1)
            if not np.allclose(norms, 1.0, rtol=1e-4, atol=1e-4):
                raise ContractError(f"{key} is not normalized within tolerance")

    bias_spec = spec["bias"]
    if bool(bias_spec.get("available")):
        if "b_true" not in arrays:
            raise ContractError("comparison_spec.bias declares b_true but trajectory key is missing")
        b_true = np.asarray(arrays["b_true"])
        if b_true.shape != (t_len, 3) or not np.all(np.isfinite(b_true)):
            raise ContractError(f"b_true must be finite with shape [{t_len},3], got {b_true.shape}")

    covariance = spec["covariance"]
    if bool(covariance.get("available")):
        p = np.asarray(arrays.get("P"))
        if p.shape != (t_len, state_dim, state_dim):
            raise ContractError(
                f"comparison covariance P must have shape [{t_len},{state_dim},{state_dim}], got {p.shape}"
            )

    innovation = spec["innovation"]
    if bool(innovation.get("available")):
        innov = np.asarray(arrays.get("innov"))
        expected = (t_len, int(innovation["dimension"]))
        if innov.shape != expected:
            raise ContractError(f"comparison innovation must have shape {expected}, got {innov.shape}")

    gain_spec = spec["gain"]
    if bool(gain_spec.get("available")):
        gain_arr = np.asarray(arrays.get("gain"))
        expected = (t_len, int(gain_spec["shape"][0]), int(gain_spec["shape"][1]))
        if gain_arr.shape != expected:
            raise ContractError(f"comparison gain must have shape {expected}, got {gain_arr.shape}")

    correction = spec["correction"]
    for key in ("correction_attitude", "correction_bias"):
        if key not in arrays:
            continue
        value = np.asarray(arrays[key])
        if value.shape != (t_len, 3):
            raise ContractError(f"{key} must have shape [{t_len},3], got {value.shape}")
        valid = np.asarray(arrays.get("correction_valid", np.ones(t_len, dtype=bool)), dtype=bool)
        if valid.shape != (t_len,) or not np.all(np.isfinite(value[valid])):
            raise ContractError(f"{key} contains invalid values at a valid correction timestep")
    if "correction_valid" in arrays and np.asarray(arrays["correction_valid"]).shape != (t_len,):
        raise ContractError(f"correction_valid must have shape [{t_len}]")


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
