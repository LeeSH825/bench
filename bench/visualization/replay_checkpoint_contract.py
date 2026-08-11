from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


REPLAY_CHECKPOINT_CONTRACT_FILENAME = "replay_contract.json"
REPLAY_CHECKPOINT_CONTRACT_SCHEMA_VERSION = "replay_checkpoint_contract_v1"
SUPPORTED_INPUT_LAYOUTS = {"NTD"}
SUPPORTED_OUTPUT_LAYOUTS = {"NTD"}
SUPPORTED_TIME_LAYOUTS = {"time_s_T"}
SUPPORTED_SYSTEM_MODEL_FORMATS = {
    "none",
    "linear_F_H",
    "adcs_simple_attitude_bias",
}
SUPPORTED_NORMALIZATION_FORMATS = {"none", "standard_scaler"}
_PATH_FIELDS = (
    "checkpoint_path",
    "model_config_path",
    "normalizer_path",
    "system_model_path",
    "training_summary_path",
)
_REQUIRED_FIELDS = (
    "schema_version",
    "package_id",
    "created_at_utc",
    "model_id",
    "adapter_id",
    *_PATH_FIELDS,
    "state_dim",
    "measurement_dim",
    "observed_state",
    "input_layout",
    "output_layout",
    "time_layout",
    "training_suite_name",
    "training_task_id",
    "training_seed",
    "training_run_dir",
    "checkpoint_step",
    "checkpoint_metric",
    "checkpoint_metric_value",
    "requires_system_model",
    "system_model_format",
    "requires_normalization",
    "normalization_format",
    "hidden_state_initialization",
    "preprocessing",
    "compatibility",
    "is_mock",
    "not_for_benchmark_reporting",
    "warnings",
    "notes",
)


def _positive_int(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    if value <= 0:
        raise ValueError(f"{name} must be > 0, got {value}")
    return int(value)


def _optional_int(name: str, value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer or null, got {value!r}")
    return int(value)


def _nonempty_string(name: str, value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _optional_string(name: str, value: Any) -> str | None:
    if value is None:
        return None
    return _nonempty_string(name, value)


def _bool(name: str, value: Any) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean, got {value!r}")
    return bool(value)


def _observed_state(
    value: Any,
    *,
    state_dim: int,
    measurement_dim: int,
) -> list[int]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError("observed_state must be a sequence of integers")
    indices: list[int] = []
    for position, item in enumerate(value):
        if isinstance(item, bool) or not isinstance(item, int):
            raise ValueError(
                f"observed_state[{position}] must be an integer, got {item!r}"
            )
        index = int(item)
        if index < 0 or index >= state_dim:
            raise ValueError(
                f"observed_state[{position}]={index} is out of bounds for "
                f"state_dim={state_dim}"
            )
        indices.append(index)
    if len(set(indices)) != len(indices):
        raise ValueError(
            f"observed_state contains duplicate indices: {indices}"
        )
    if len(indices) != measurement_dim:
        raise ValueError(
            "len(observed_state) must equal measurement_dim for "
            f"replay_checkpoint_contract_v1: observed={len(indices)}, "
            f"measurement_dim={measurement_dim}"
        )
    return indices


def _string_list(name: str, value: Any) -> list[str]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a list of strings")
    result: list[str] = []
    for position, item in enumerate(value):
        if not isinstance(item, str):
            raise ValueError(
                f"{name}[{position}] must be a string, got {item!r}"
            )
        result.append(item)
    return result


def _resolve_reference(
    field: str,
    value: str | None,
    *,
    package_dir: Path | None,
) -> str | None:
    if value is None:
        return None
    path = Path(value).expanduser()
    if not path.is_absolute():
        if package_dir is None:
            raise ValueError(
                f"{field} is relative but package_dir was not provided"
            )
        path = package_dir / path
    resolved = path.resolve()
    if not resolved.exists():
        raise ValueError(f"{field} referenced file does not exist: {resolved}")
    if not resolved.is_file():
        raise ValueError(f"{field} must reference a file: {resolved}")
    return str(resolved)


def _mapping(name: str, value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return dict(value)


def _validate_hidden_state(value: Any) -> dict[str, Any]:
    hidden = _mapping("hidden_state_initialization", value)
    for field in ("method", "source", "details"):
        if field not in hidden:
            raise ValueError(
                f"hidden_state_initialization.{field} is required"
            )
    hidden["method"] = _nonempty_string(
        "hidden_state_initialization.method",
        hidden["method"],
    )
    hidden["source"] = _nonempty_string(
        "hidden_state_initialization.source",
        hidden["source"],
    )
    hidden["details"] = _mapping(
        "hidden_state_initialization.details",
        hidden["details"],
    )
    return hidden


def _validate_preprocessing(value: Any) -> dict[str, Any]:
    preprocessing = _mapping("preprocessing", value)
    for field in ("input_transform", "output_inverse_transform", "assumptions"):
        if field not in preprocessing:
            raise ValueError(f"preprocessing.{field} is required")
    preprocessing["input_transform"] = _nonempty_string(
        "preprocessing.input_transform",
        preprocessing["input_transform"],
    )
    preprocessing["output_inverse_transform"] = _nonempty_string(
        "preprocessing.output_inverse_transform",
        preprocessing["output_inverse_transform"],
    )
    preprocessing["assumptions"] = _string_list(
        "preprocessing.assumptions",
        preprocessing["assumptions"],
    )
    return preprocessing


def _validate_compatibility(value: Any) -> dict[str, Any]:
    compatibility = _mapping("compatibility", value)
    for field in (
        "compatible_replay_schema_versions",
        "compatible_state_schema",
        "compatible_observation_schema",
    ):
        if field not in compatibility:
            raise ValueError(f"compatibility.{field} is required")
    compatibility["compatible_replay_schema_versions"] = _string_list(
        "compatibility.compatible_replay_schema_versions",
        compatibility["compatible_replay_schema_versions"],
    )
    if not compatibility["compatible_replay_schema_versions"]:
        raise ValueError(
            "compatibility.compatible_replay_schema_versions must not be empty"
        )
    compatibility["compatible_state_schema"] = _mapping(
        "compatibility.compatible_state_schema",
        compatibility["compatible_state_schema"],
    )
    compatibility["compatible_observation_schema"] = _mapping(
        "compatibility.compatible_observation_schema",
        compatibility["compatible_observation_schema"],
    )
    return compatibility


def load_replay_checkpoint_contract(
    path_or_package_dir: str | Path,
) -> dict[str, Any]:
    path = Path(path_or_package_dir).expanduser().resolve()
    if path.is_dir():
        path = path / REPLAY_CHECKPOINT_CONTRACT_FILENAME
    if not path.exists():
        raise FileNotFoundError(f"replay checkpoint contract not found: {path}")
    if not path.is_file():
        raise ValueError(f"replay checkpoint contract must be a file: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"invalid replay checkpoint contract JSON at {path}: {exc}"
        ) from exc
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return validate_replay_checkpoint_contract(
        value,
        package_dir=path.parent,
    )


def validate_replay_checkpoint_contract(
    contract: Mapping[str, Any],
    *,
    package_dir: str | Path | None = None,
    expected_state_dim: int | None = None,
    expected_measurement_dim: int | None = None,
    expected_observed_state: list[int] | None = None,
) -> dict[str, Any]:
    if not isinstance(contract, Mapping):
        raise ValueError("replay checkpoint contract must be an object")
    missing = [field for field in _REQUIRED_FIELDS if field not in contract]
    if missing:
        raise ValueError(
            f"replay checkpoint contract is missing required fields: {missing}"
        )

    normalized = dict(contract)
    schema_version = _nonempty_string(
        "schema_version",
        normalized["schema_version"],
    )
    if schema_version != REPLAY_CHECKPOINT_CONTRACT_SCHEMA_VERSION:
        raise ValueError(
            "schema_version must be "
            f"{REPLAY_CHECKPOINT_CONTRACT_SCHEMA_VERSION!r}, "
            f"got {schema_version!r}"
        )
    normalized["schema_version"] = schema_version
    for field in (
        "package_id",
        "created_at_utc",
        "model_id",
        "adapter_id",
    ):
        normalized[field] = _nonempty_string(field, normalized[field])

    state_dim = _positive_int("state_dim", normalized["state_dim"])
    measurement_dim = _positive_int(
        "measurement_dim",
        normalized["measurement_dim"],
    )
    observed_state = _observed_state(
        normalized["observed_state"],
        state_dim=state_dim,
        measurement_dim=measurement_dim,
    )
    normalized["state_dim"] = state_dim
    normalized["measurement_dim"] = measurement_dim
    normalized["observed_state"] = observed_state

    if expected_state_dim is not None and state_dim != int(expected_state_dim):
        raise ValueError(
            f"state_dim mismatch: contract={state_dim}, "
            f"expected={int(expected_state_dim)}"
        )
    if (
        expected_measurement_dim is not None
        and measurement_dim != int(expected_measurement_dim)
    ):
        raise ValueError(
            f"measurement_dim mismatch: contract={measurement_dim}, "
            f"expected={int(expected_measurement_dim)}"
        )
    if expected_observed_state is not None:
        expected = [int(index) for index in expected_observed_state]
        if observed_state != expected:
            raise ValueError(
                f"observed_state mismatch: contract={observed_state}, "
                f"expected={expected}"
            )

    for field, supported in (
        ("input_layout", SUPPORTED_INPUT_LAYOUTS),
        ("output_layout", SUPPORTED_OUTPUT_LAYOUTS),
        ("time_layout", SUPPORTED_TIME_LAYOUTS),
    ):
        value = _nonempty_string(field, normalized[field])
        if value not in supported:
            raise ValueError(
                f"{field}={value!r} is unsupported; "
                f"supported values are {sorted(supported)}"
            )
        normalized[field] = value

    normalized["requires_system_model"] = _bool(
        "requires_system_model",
        normalized["requires_system_model"],
    )
    system_format = _nonempty_string(
        "system_model_format",
        normalized["system_model_format"],
    )
    if system_format not in SUPPORTED_SYSTEM_MODEL_FORMATS:
        raise ValueError(
            f"system_model_format={system_format!r} is unsupported; "
            f"supported values are {sorted(SUPPORTED_SYSTEM_MODEL_FORMATS)}"
        )
    if normalized["requires_system_model"] and system_format == "none":
        raise ValueError(
            "system_model_format must not be 'none' when "
            "requires_system_model=true"
        )
    normalized["system_model_format"] = system_format

    normalized["requires_normalization"] = _bool(
        "requires_normalization",
        normalized["requires_normalization"],
    )
    normalization_format = _nonempty_string(
        "normalization_format",
        normalized["normalization_format"],
    )
    if normalization_format not in SUPPORTED_NORMALIZATION_FORMATS:
        raise ValueError(
            f"normalization_format={normalization_format!r} is unsupported; "
            f"supported values are {sorted(SUPPORTED_NORMALIZATION_FORMATS)}"
        )
    if normalized["requires_normalization"] and normalization_format == "none":
        raise ValueError(
            "normalization_format must not be 'none' when "
            "requires_normalization=true"
        )
    normalized["normalization_format"] = normalization_format

    normalized["hidden_state_initialization"] = _validate_hidden_state(
        normalized["hidden_state_initialization"]
    )
    normalized["preprocessing"] = _validate_preprocessing(
        normalized["preprocessing"]
    )
    normalized["compatibility"] = _validate_compatibility(
        normalized["compatibility"]
    )
    normalized["is_mock"] = _bool("is_mock", normalized["is_mock"])
    normalized["not_for_benchmark_reporting"] = _bool(
        "not_for_benchmark_reporting",
        normalized["not_for_benchmark_reporting"],
    )
    if normalized["is_mock"] and not normalized["not_for_benchmark_reporting"]:
        raise ValueError(
            "not_for_benchmark_reporting must be true when is_mock=true"
        )
    normalized["warnings"] = _string_list(
        "warnings",
        normalized["warnings"],
    )
    if not isinstance(normalized["notes"], str):
        raise ValueError("notes must be a string")

    for field in (
        "training_suite_name",
        "training_task_id",
        "training_run_dir",
        "checkpoint_metric",
    ):
        normalized[field] = _optional_string(field, normalized[field])
    normalized["training_seed"] = _optional_int(
        "training_seed",
        normalized["training_seed"],
    )
    normalized["checkpoint_step"] = _optional_int(
        "checkpoint_step",
        normalized["checkpoint_step"],
    )
    metric_value = normalized["checkpoint_metric_value"]
    if metric_value is not None:
        if isinstance(metric_value, bool) or not isinstance(
            metric_value,
            (int, float),
        ):
            raise ValueError(
                "checkpoint_metric_value must be numeric or null"
            )
        normalized["checkpoint_metric_value"] = float(metric_value)

    resolved_package_dir = (
        None
        if package_dir is None
        else Path(package_dir).expanduser().resolve()
    )
    resolved_paths: dict[str, str | None] = {}
    for field in _PATH_FIELDS:
        value = _optional_string(field, normalized[field])
        normalized[field] = value
        resolved_paths[field] = _resolve_reference(
            field,
            value,
            package_dir=resolved_package_dir,
        )
    if resolved_paths["checkpoint_path"] is None:
        raise ValueError("checkpoint_path must not be null")
    if (
        normalized["requires_system_model"]
        and resolved_paths["system_model_path"] is None
    ):
        raise ValueError(
            "system_model_path is required when requires_system_model=true"
        )
    if (
        normalized["requires_normalization"]
        and resolved_paths["normalizer_path"] is None
    ):
        raise ValueError(
            "normalizer_path is required when requires_normalization=true"
        )

    normalized["_package_dir"] = (
        None if resolved_package_dir is None else str(resolved_package_dir)
    )
    normalized["_resolved_paths"] = resolved_paths
    return normalized


def _portable_contract(contract: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in contract.items()
        if not str(key).startswith("_")
    }


def save_replay_checkpoint_contract(
    contract: Mapping[str, Any],
    package_dir: str | Path,
) -> Path:
    output_dir = Path(package_dir).expanduser().resolve()
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        normalized = validate_replay_checkpoint_contract(
            contract,
            package_dir=output_dir,
        )
        output_path = output_dir / REPLAY_CHECKPOINT_CONTRACT_FILENAME
        temporary = output_path.with_name(f".{output_path.name}.tmp")
        temporary.write_text(
            json.dumps(
                _portable_contract(normalized),
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        temporary.replace(output_path)
    except OSError as exc:
        raise RuntimeError(
            f"failed to write replay checkpoint contract under {output_dir}"
        ) from exc
    return output_path


def summarize_replay_checkpoint_contract(
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": contract.get("schema_version"),
        "package_id": contract.get("package_id"),
        "model_id": contract.get("model_id"),
        "adapter_id": contract.get("adapter_id"),
        "state_dim": contract.get("state_dim"),
        "measurement_dim": contract.get("measurement_dim"),
        "observed_state": list(contract.get("observed_state", [])),
        "input_layout": contract.get("input_layout"),
        "output_layout": contract.get("output_layout"),
        "time_layout": contract.get("time_layout"),
        "requires_system_model": contract.get("requires_system_model"),
        "system_model_format": contract.get("system_model_format"),
        "requires_normalization": contract.get("requires_normalization"),
        "normalization_format": contract.get("normalization_format"),
        "hidden_state_initialization": contract.get(
            "hidden_state_initialization"
        ),
        "is_mock": contract.get("is_mock"),
        "not_for_benchmark_reporting": contract.get(
            "not_for_benchmark_reporting"
        ),
        "package_dir": contract.get("_package_dir"),
    }
