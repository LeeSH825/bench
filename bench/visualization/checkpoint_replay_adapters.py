from __future__ import annotations

from collections.abc import Mapping
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from bench.models.kalmannet_tsp import KalmanNetTSPAdapter

from .checkpoint_contract_probe import probe_checkpoint_contract
from .replay_checkpoint_contract import (
    REPLAY_CHECKPOINT_CONTRACT_FILENAME,
    load_replay_checkpoint_contract,
    summarize_replay_checkpoint_contract,
    validate_replay_checkpoint_contract,
)


MOCK_CHECKPOINT_MODEL_ID = "mock_checkpoint_adapter"
KALMANNET_TSP_MODEL_ID = "kalmannet_tsp"


@dataclass(frozen=True)
class CheckpointReplayResult:
    x_hat: np.ndarray
    metadata: dict[str, Any]


def _load_checkpoint(path: Path) -> Any:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch is required for checkpoint-backed replay"
        ) from exc
    try:
        try:
            return torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            return torch.load(path, map_location="cpu")
    except Exception as exc:
        raise ValueError(
            f"failed to load checkpoint {path}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc


def _observed_state(replay_meta: Mapping[str, Any]) -> list[int]:
    observation = replay_meta.get("observation")
    raw = (
        observation.get("observed_state")
        if isinstance(observation, Mapping)
        else replay_meta.get("observed_state")
    )
    if not isinstance(raw, (list, tuple)) or not raw:
        raise ValueError(
            "replay metadata observation.observed_state must be a "
            "non-empty list"
        )
    indices: list[int] = []
    for position, value in enumerate(raw):
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, np.integer)
        ):
            raise ValueError(
                "observation.observed_state"
                f"[{position}] must be an integer, got {value!r}"
            )
        indices.append(int(value))
    if len(set(indices)) != len(indices):
        raise ValueError(
            "observation.observed_state contains duplicate indices: "
            f"{indices}"
        )
    return indices


def _state_dim(replay_meta: Mapping[str, Any]) -> int:
    value = replay_meta.get("state_dim")
    if isinstance(value, bool):
        raise ValueError("replay metadata state_dim must be a positive integer")
    try:
        state_dim = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "replay metadata state_dim must be a positive integer"
        ) from exc
    if state_dim <= 0:
        raise ValueError("replay metadata state_dim must be a positive integer")
    return state_dim


def _measurement_dim(
    replay_meta: Mapping[str, Any],
    *,
    y_obs: np.ndarray,
) -> int:
    value = replay_meta.get("measurement_dim", y_obs.shape[-1])
    if isinstance(value, bool):
        raise ValueError(
            "replay metadata measurement_dim must be a positive integer"
        )
    try:
        measurement_dim = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "replay metadata measurement_dim must be a positive integer"
        ) from exc
    if measurement_dim <= 0:
        raise ValueError(
            "replay metadata measurement_dim must be a positive integer"
        )
    if measurement_dim != int(y_obs.shape[-1]):
        raise ValueError(
            f"replay metadata measurement_dim={measurement_dim} does not "
            f"match y_obs Dy={y_obs.shape[-1]}"
        )
    return measurement_dim


def _contract_path(value: Path) -> Path | None:
    if value.is_dir():
        return value / REPLAY_CHECKPOINT_CONTRACT_FILENAME
    if value.name == REPLAY_CHECKPOINT_CONTRACT_FILENAME:
        return value
    return None


def _load_adapter_contract(
    *,
    model_id: str,
    checkpoint: Path,
    model_config: Path | None,
    y_obs: np.ndarray,
    replay_meta: dict[str, Any],
) -> dict[str, Any] | None:
    contract_source: Path | None = None
    if model_config is not None:
        contract_source = _contract_path(model_config)
    if contract_source is None and checkpoint.name == "checkpoint.pt":
        sibling = checkpoint.parent / REPLAY_CHECKPOINT_CONTRACT_FILENAME
        if sibling.exists():
            contract_source = sibling
    if contract_source is None:
        return None

    contract = load_replay_checkpoint_contract(contract_source)
    observed_state = _observed_state(replay_meta)
    validated = validate_replay_checkpoint_contract(
        contract,
        package_dir=Path(contract["_package_dir"]),
        expected_state_dim=_state_dim(replay_meta),
        expected_measurement_dim=_measurement_dim(
            replay_meta,
            y_obs=y_obs,
        ),
        expected_observed_state=observed_state,
    )
    if validated["model_id"] != model_id:
        raise ValueError(
            f"replay contract model_id={validated['model_id']!r} does not "
            f"match requested model_id={model_id!r}"
        )
    contract_checkpoint = Path(
        validated["_resolved_paths"]["checkpoint_path"]
    ).resolve()
    if checkpoint.resolve() != contract_checkpoint:
        raise ValueError(
            "checkpoint does not match replay contract checkpoint_path: "
            f"checkpoint={checkpoint.resolve()}, "
            f"contract={contract_checkpoint}"
        )

    replay_schema = str(replay_meta.get("schema_version", "")).strip()
    compatible_versions = validated["compatibility"][
        "compatible_replay_schema_versions"
    ]
    if replay_schema and replay_schema not in compatible_versions:
        raise ValueError(
            f"replay schema_version={replay_schema!r} is not compatible "
            f"with contract versions={compatible_versions}"
        )
    compatible_observation = validated["compatibility"][
        "compatible_observation_schema"
    ]
    declared_observed = compatible_observation.get("observed_state")
    if declared_observed is not None and [
        int(index) for index in declared_observed
    ] != observed_state:
        raise ValueError(
            "replay observation schema does not match contract "
            f"compatible_observation_schema: replay={observed_state}, "
            f"contract={declared_observed}"
        )
    return validated


def _finite_scalar(checkpoint: Mapping[str, Any], key: str) -> float:
    value = checkpoint.get(key)
    if isinstance(value, bool):
        raise ValueError(f"mock checkpoint {key} must be a finite scalar")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"mock checkpoint {key} must be a finite scalar"
        ) from exc
    if not np.isfinite(result):
        raise ValueError(f"mock checkpoint {key} must be finite")
    return result


def _json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid {label} JSON at {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object: {path}")
    return value


def _run_kalmannet_tsp_checkpoint_adapter(
    *,
    checkpoint: Path,
    model_config: Path | None,
    y_obs: np.ndarray,
    replay_meta: dict[str, Any],
    device: str,
    strict: bool,
    replay_contract: dict[str, Any] | None,
) -> CheckpointReplayResult:
    if replay_contract is None:
        raise ValueError("kalmannet_tsp replay requires replay_contract.json")

    resolved = dict(replay_contract.get("_resolved_paths", {}) or {})
    model_config_path = resolved.get("model_config_path")
    system_model_path = resolved.get("system_model_path")
    if not model_config_path:
        raise ValueError("kalmannet_tsp replay requires model_config_path in replay_contract")
    if not system_model_path:
        raise ValueError("kalmannet_tsp replay requires system_model_path in replay_contract")

    model_cfg = _json_object(Path(model_config_path), label="model_config")
    system_model = _json_object(Path(system_model_path), label="system_model")

    adapter = KalmanNetTSPAdapter()
    x_dim = _state_dim(replay_meta)
    measurement_dim = _measurement_dim(replay_meta, y_obs=np.asarray(y_obs))
    observed_state = _observed_state(replay_meta)
    if len(observed_state) != measurement_dim:
        raise ValueError(
            "observed_state length must match measurement_dim: "
            f"observed={len(observed_state)}, measurement_dim={measurement_dim}"
        )

    if int(model_cfg.get("state_dim", x_dim)) != x_dim:
        raise ValueError(
            f"model_config.state_dim mismatch: expected {x_dim}, got {model_cfg.get('state_dim')}"
        )
    if int(model_cfg.get("measurement_dim", measurement_dim)) != measurement_dim:
        raise ValueError(
            "model_config.measurement_dim mismatch: expected "
            f"{measurement_dim}, got {model_cfg.get('measurement_dim')}"
        )

    y_values = np.asarray(y_obs, dtype=np.float32)
    if y_values.ndim != 3:
        raise ValueError(
            "y_obs must have shape [N,T,Dy] for KalmanNet_TSP replay, "
            f"got {tuple(y_values.shape)}"
        )
    if strict and not np.isfinite(y_values).all():
        raise ValueError("y_obs contains NaN or Inf values")

    system_info = {
        "x_dim": int(x_dim),
        "y_dim": int(measurement_dim),
        "T": int(y_values.shape[1]),
        "F": np.asarray(system_model["F"], dtype=np.float32),
        "H": np.asarray(system_model["H"], dtype=np.float32),
        "Q": np.asarray(system_model["Q"], dtype=np.float32),
        "R": np.asarray(system_model["R"], dtype=np.float32),
        "meta": dict(replay_meta),
    }
    run_ctx = {
        "seed": int(replay_meta.get("seed", 0) or 0),
        "deterministic": True,
        "device": str(device),
    }
    adapter.setup(dict(model_cfg), system_info, run_ctx)
    adapter.load(str(checkpoint))
    x_hat = adapter.predict(y_values, context={})
    x_hat_np = np.asarray(x_hat, dtype=np.float32)
    if x_hat_np.shape != (y_values.shape[0], y_values.shape[1], x_dim):
        raise ValueError(
            "KalmanNet_TSP replay output shape mismatch: "
            f"expected {(y_values.shape[0], y_values.shape[1], x_dim)}, got {x_hat_np.shape}"
        )
    if not np.isfinite(x_hat_np).all():
        raise ValueError("KalmanNet_TSP replay produced NaN or Inf values")

    probe = probe_checkpoint_contract(
        checkpoint,
        model_id=KALMANNET_TSP_MODEL_ID,
        model_config=Path(model_config_path),
    )
    return CheckpointReplayResult(
        x_hat=x_hat_np,
        metadata={
            "adapter_name": "KalmanNetTSPReplayAdapter",
            "model_id": KALMANNET_TSP_MODEL_ID,
            "checkpoint_path": str(checkpoint),
            "model_config_path": str(model_config_path),
            "checkpoint_loaded": True,
            "checkpoint_format_summary": {
                "top_level_type": probe["top_level_type"],
                "top_level_keys": probe["top_level_keys"],
                "checkpoint_size_bytes": probe["checkpoint_size_bytes"],
            },
            "normalization_applied": False,
            "system_model_used": True,
            "hidden_state_initialization": str(
                replay_contract.get("hidden_state_initialization", {})
            ),
            "input_layout": str(replay_contract.get("input_layout", "NTD")),
            "output_layout": str(replay_contract.get("output_layout", "NTD")),
            "assumptions": [
                "KalmanNet_TSP replay uses the replay_contract package contract",
                "direct NTD y_obs layout is mapped through the adapter's BTD interface",
            ],
            "warnings": [],
            "device": str(device),
            "is_mock_adapter": False,
            "is_real_checkpoint_adapter": True,
            "not_for_benchmark_reporting": bool(
                replay_contract.get("not_for_benchmark_reporting", False)
            ),
            "replay_contract_summary": summarize_replay_checkpoint_contract(replay_contract),
            "package_dir": replay_contract.get("_package_dir"),
            "model_config_loaded": True,
            "system_model_format": str(system_model.get("format", "")),
        },
    )


def _run_mock_checkpoint_adapter(
    *,
    checkpoint: Path,
    model_config: Path | None,
    y_obs: np.ndarray,
    replay_meta: dict[str, Any],
    device: str,
    strict: bool,
    replay_contract: dict[str, Any] | None,
) -> CheckpointReplayResult:
    loaded = _load_checkpoint(checkpoint)
    if not isinstance(loaded, Mapping):
        raise ValueError("mock checkpoint must contain a mapping")
    saved_model_id = str(loaded.get("model_id", "")).strip()
    if saved_model_id != MOCK_CHECKPOINT_MODEL_ID:
        raise ValueError(
            "mock checkpoint model_id must be "
            f"{MOCK_CHECKPOINT_MODEL_ID!r}, got {saved_model_id!r}"
        )

    gain = _finite_scalar(loaded, "gain")
    bias = _finite_scalar(loaded, "bias")
    if replay_contract is not None:
        if not replay_contract["is_mock"]:
            raise ValueError(
                "mock_checkpoint_adapter requires replay contract is_mock=true"
            )
        if replay_contract["requires_normalization"]:
            raise ValueError(
                "mock_checkpoint_adapter does not implement normalization"
            )
        if replay_contract["requires_system_model"]:
            raise ValueError(
                "mock_checkpoint_adapter does not use a system model"
            )
    values = np.asarray(y_obs)
    if values.ndim != 3:
        raise ValueError(
            "y_obs must have shape [N,T,Dy] for checkpoint replay, "
            f"got shape={tuple(values.shape)}"
        )
    if not np.issubdtype(values.dtype, np.number):
        raise ValueError(f"y_obs must be numeric, got dtype={values.dtype}")
    if strict and not np.isfinite(values).all():
        raise ValueError("y_obs contains NaN or Inf values")

    observed_state = _observed_state(replay_meta)
    if len(observed_state) != values.shape[-1]:
        raise ValueError(
            "len(observation.observed_state) must equal y_obs Dy: "
            f"observed={len(observed_state)}, Dy={values.shape[-1]}"
        )
    state_dim = _state_dim(replay_meta)
    for position, index in enumerate(observed_state):
        if index < 0 or index >= state_dim:
            raise ValueError(
                "observation.observed_state"
                f"[{position}]={index} is out of bounds for Dx={state_dim}"
            )

    x_hat = np.zeros(
        (values.shape[0], values.shape[1], state_dim),
        dtype=np.result_type(values.dtype, np.float32),
    )
    transformed = values * gain + bias
    for measurement_index, state_index in enumerate(observed_state):
        x_hat[..., state_index] = transformed[..., measurement_index]

    probe = probe_checkpoint_contract(
        checkpoint,
        model_id=MOCK_CHECKPOINT_MODEL_ID,
        model_config=(
            model_config
            if replay_contract is not None
            else None
        ),
    )
    return CheckpointReplayResult(
        x_hat=x_hat,
        metadata={
            "adapter_name": "Phase6DMockCheckpointReplayAdapter",
            "model_id": MOCK_CHECKPOINT_MODEL_ID,
            "checkpoint_path": str(checkpoint),
            "model_config_path": (
                None if model_config is None else str(model_config)
            ),
            "checkpoint_loaded": True,
            "checkpoint_format_summary": {
                "top_level_type": probe["top_level_type"],
                "top_level_keys": probe["top_level_keys"],
                "checkpoint_size_bytes": probe["checkpoint_size_bytes"],
            },
            "normalization_applied": False,
            "system_model_used": False,
            "hidden_state_initialization": (
                "not_applicable_stateless_mock"
                if replay_contract is None
                else replay_contract["hidden_state_initialization"]
            ),
            "input_layout": (
                "NTD [N,T,Dy]"
                if replay_contract is None
                else replay_contract["input_layout"]
            ),
            "output_layout": (
                "NTD [N,T,Dx]"
                if replay_contract is None
                else replay_contract["output_layout"]
            ),
            "assumptions": [
                "observed_state maps y_obs channels directly to state indices",
                "unobserved state dimensions are zero",
                f"mock affine transform uses gain={gain} and bias={bias}",
            ],
            "warnings": [
                "This adapter is test-only and must not be used for "
                "benchmark reporting."
            ],
            "device": str(device),
            "is_mock_adapter": True,
            "is_real_checkpoint_adapter": False,
            "not_for_benchmark_reporting": True,
            "replay_contract_summary": (
                None
                if replay_contract is None
                else summarize_replay_checkpoint_contract(replay_contract)
            ),
            "package_dir": (
                None
                if replay_contract is None
                else replay_contract["_package_dir"]
            ),
        },
    )


ReplayAdapter = Callable[..., CheckpointReplayResult]
_REAL_CHECKPOINT_REPLAY_ADAPTERS: dict[str, ReplayAdapter] = {
    KALMANNET_TSP_MODEL_ID: _run_kalmannet_tsp_checkpoint_adapter,
}
_TEST_CHECKPOINT_REPLAY_ADAPTERS: dict[str, ReplayAdapter] = {
    MOCK_CHECKPOINT_MODEL_ID: _run_mock_checkpoint_adapter,
}
SUPPORTED_CHECKPOINT_REPLAY_MODEL_IDS = frozenset(
    set(_REAL_CHECKPOINT_REPLAY_ADAPTERS)
    | set(_TEST_CHECKPOINT_REPLAY_ADAPTERS)
)


def get_real_checkpoint_replay_model_ids() -> list[str]:
    return sorted(_REAL_CHECKPOINT_REPLAY_ADAPTERS)


def get_test_checkpoint_replay_model_ids() -> list[str]:
    return sorted(_TEST_CHECKPOINT_REPLAY_ADAPTERS)


def get_supported_checkpoint_replay_model_ids(
    include_test: bool = False,
) -> list[str]:
    supported = set(_REAL_CHECKPOINT_REPLAY_ADAPTERS)
    if include_test:
        supported.update(_TEST_CHECKPOINT_REPLAY_ADAPTERS)
    return sorted(supported)


def run_checkpoint_replay_adapter(
    *,
    model_id: str,
    checkpoint: str | Path,
    model_config: str | Path | None,
    y_obs: np.ndarray,
    replay_meta: dict[str, Any],
    device: str = "cpu",
    strict: bool = True,
) -> CheckpointReplayResult:
    requested_model_id = str(model_id).strip()
    checkpoint_input = Path(checkpoint).expanduser().resolve()
    if not checkpoint_input.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_input}")
    config_path: Path | None = None
    if model_config is not None:
        config_path = Path(model_config).expanduser().resolve()
        if not config_path.exists():
            raise FileNotFoundError(f"model config not found: {config_path}")
        if not config_path.is_file() and not config_path.is_dir():
            raise ValueError(
                f"model config must be a file or package directory: "
                f"{config_path}"
            )

    if checkpoint_input.is_dir():
        package_contract = load_replay_checkpoint_contract(checkpoint_input)
        checkpoint_path = Path(
            package_contract["_resolved_paths"]["checkpoint_path"]
        )
        if config_path is None:
            config_path = checkpoint_input
    elif checkpoint_input.is_file():
        checkpoint_path = checkpoint_input
    else:
        raise ValueError(
            f"checkpoint must be a file or package directory: "
            f"{checkpoint_input}"
        )

    values = np.asarray(y_obs)
    replay_contract = _load_adapter_contract(
        model_id=requested_model_id,
        checkpoint=checkpoint_path,
        model_config=config_path,
        y_obs=values,
        replay_meta=replay_meta,
    )

    adapter = _REAL_CHECKPOINT_REPLAY_ADAPTERS.get(requested_model_id)
    is_real = adapter is not None
    if adapter is None:
        adapter = _TEST_CHECKPOINT_REPLAY_ADAPTERS.get(requested_model_id)
    if adapter is None:
        detail = ""
        if requested_model_id == "kalmannet_tsp":
            detail = (
                " A Phase 6F package may be structurally valid, but "
                "KalmanNet_TSP still requires a real 9x6-compatible "
                "checkpoint trained for the Phase 6A direct-observation "
                "contract, verified model_config and system_model files, "
                "normalization/preprocessing metadata, hidden-state "
                "initialization, and explicit real adapter registration."
            )
        raise NotImplementedError(
            f"Checkpoint replay for model_id={requested_model_id!r} is not "
            "supported through Phase 6F. Real IDs: "
            f"{get_real_checkpoint_replay_model_ids()}; test IDs: "
            f"{get_test_checkpoint_replay_model_ids()}.{detail}"
        )
    if is_real and replay_contract is None:
        raise ValueError(
            f"real checkpoint adapter model_id={requested_model_id!r} "
            "requires replay_contract.json"
        )

    return adapter(
        checkpoint=checkpoint_path,
        model_config=config_path,
        y_obs=values,
        replay_meta=replay_meta,
        device=str(device),
        strict=bool(strict),
        replay_contract=replay_contract,
    )


def validate_checkpoint_replay_output(
    *,
    x_hat: np.ndarray,
    x_true: np.ndarray,
    model_id: str,
) -> None:
    estimate = np.asarray(x_hat)
    truth = np.asarray(x_true)
    if estimate.ndim != 3:
        raise ValueError(
            f"x_hat from model_id={model_id!r} must have shape [N,T,Dx] "
            f"(rank 3), got shape={tuple(estimate.shape)}"
        )
    if estimate.shape != truth.shape:
        raise ValueError(
            f"x_hat from model_id={model_id!r} must match x_true shape: "
            f"x_true={tuple(truth.shape)}, x_hat={tuple(estimate.shape)}"
        )
    if not np.issubdtype(estimate.dtype, np.number):
        raise ValueError(
            f"x_hat from model_id={model_id!r} must be numeric, "
            f"got dtype={estimate.dtype}"
        )
    if not np.isfinite(estimate).all():
        raise ValueError(
            f"x_hat from model_id={model_id!r} contains NaN or Inf values"
        )
