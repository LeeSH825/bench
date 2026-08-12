from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any, Literal, Mapping

import numpy as np

from .checkpoint_contract_probe import probe_checkpoint_contract
from .checkpoint_replay_adapters import (
    CheckpointReplayResult,
    get_supported_checkpoint_replay_model_ids,
    run_checkpoint_replay_adapter,
    validate_checkpoint_replay_output,
)
from .pred_artifact import (
    PRED_ARTIFACT_FILENAME,
    PRED_META_FILENAME,
    load_pred_artifact,
    save_pred_artifact,
)
from .replay_suite_scenario import (
    REPLAY_SCENARIO_FILENAME,
    REPLAY_SCENARIO_META_FILENAME,
)


ReplayModelMode = Literal[
    "replay_identity_baseline",
    "checkpoint_adapter",
]

IDENTITY_MODEL_ID = "replay_identity_baseline"
SUPPORTED_CHECKPOINT_REPLAY_MODEL_IDS = tuple(
    get_supported_checkpoint_replay_model_ids()
)
_REPLAY_KEYS = ("time_s", "x_true", "y_obs", "trajectory_id")
_REQUIRED_META_KEYS = (
    "state_schema",
    "scenario_id",
    "task_id",
    "suite_name",
    "seed",
)


def _shape(values: np.ndarray) -> tuple[int, ...]:
    return tuple(int(value) for value in values.shape)


def _require_numeric(name: str, values: np.ndarray) -> None:
    if not np.issubdtype(values.dtype, np.number):
        raise ValueError(
            f"{name} must be a numeric array, got dtype={values.dtype}"
        )


def _require_finite(name: str, values: np.ndarray) -> None:
    _require_numeric(name, values)
    if not np.isfinite(values).all():
        raise ValueError(f"{name} contains NaN or Inf values")


def _load_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON metadata at {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _validate_replay_arrays(
    arrays: Mapping[str, np.ndarray],
    *,
    strict: bool,
) -> None:
    x_true = arrays["x_true"]
    y_obs = arrays["y_obs"]
    time_s = arrays["time_s"]
    trajectory_id = arrays["trajectory_id"]

    if x_true.ndim != 3:
        raise ValueError(
            "x_true must have shape [N,T,Dx] (rank 3), "
            f"got shape={_shape(x_true)}"
        )
    if y_obs.ndim != 3:
        raise ValueError(
            "y_obs must have shape [N,T,Dy] (rank 3), "
            f"got shape={_shape(y_obs)}"
        )
    if y_obs.shape[:2] != x_true.shape[:2]:
        raise ValueError(
            "y_obs [N,T] dimensions must match x_true: "
            f"x_true={_shape(x_true)}, y_obs={_shape(y_obs)}"
        )

    n_trajectory, n_step = x_true.shape[:2]
    if time_s.ndim != 1 or time_s.shape != (n_step,):
        raise ValueError(
            f"time_s must have shape [T] with T={n_step}, "
            f"got shape={_shape(time_s)}"
        )
    if trajectory_id.ndim != 1 or trajectory_id.shape != (n_trajectory,):
        raise ValueError(
            f"trajectory_id must have shape [N] with N={n_trajectory}, "
            f"got shape={_shape(trajectory_id)}"
        )

    for name, values in arrays.items():
        _require_numeric(name, values)
        if strict:
            _require_finite(name, values)


def _validate_replay_meta(meta: Mapping[str, Any]) -> None:
    for key in _REQUIRED_META_KEYS:
        if key not in meta:
            raise ValueError(
                f"replay_scenario_meta.json is missing required field {key!r}"
            )
    if not isinstance(meta["state_schema"], Mapping):
        raise ValueError("replay metadata state_schema must be a mapping")
    if "observation" not in meta and "observed_state" not in meta:
        raise ValueError(
            "replay metadata is missing observation.observed_state"
        )
    if "observation" in meta and not isinstance(
        meta["observation"], Mapping
    ):
        raise ValueError("replay metadata observation must be a mapping")
    for key in ("scenario_id", "task_id", "suite_name"):
        if not str(meta[key]).strip():
            raise ValueError(f"replay metadata {key} must be non-empty")
    if isinstance(meta["seed"], bool):
        raise ValueError("replay metadata seed must be an integer")
    try:
        int(meta["seed"])
    except (TypeError, ValueError) as exc:
        raise ValueError("replay metadata seed must be an integer") from exc


def load_replay_input(
    replay_input_dir: str | Path,
    *,
    strict: bool = True,
) -> tuple[dict[str, Any], dict[str, Any]]:
    input_dir = Path(replay_input_dir).expanduser().resolve()
    npz_path = input_dir / REPLAY_SCENARIO_FILENAME
    meta_path = input_dir / REPLAY_SCENARIO_META_FILENAME
    if not npz_path.exists():
        raise FileNotFoundError(f"replay input artifact not found: {npz_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"replay input metadata not found: {meta_path}")

    with np.load(npz_path, allow_pickle=False) as data:
        missing = [key for key in _REPLAY_KEYS if key not in data.files]
        if missing:
            raise ValueError(
                f"{npz_path} is missing required replay keys: {missing}"
            )
        arrays: dict[str, Any] = {
            key: np.array(data[key], copy=True) for key in _REPLAY_KEYS
        }
    _validate_replay_arrays(arrays, strict=bool(strict))

    meta = _load_json_object(meta_path)
    _validate_replay_meta(meta)
    arrays["npz_path"] = npz_path
    arrays["meta_path"] = meta_path
    return arrays, meta


def infer_observed_state_indices(meta: dict[str, Any]) -> list[int]:
    observation = meta.get("observation")
    if isinstance(observation, Mapping):
        raw = observation.get("observed_state")
    else:
        raw = meta.get("observed_state")
    if not isinstance(raw, (list, tuple)) or not raw:
        raise ValueError(
            "replay metadata observation.observed_state must be a non-empty list"
        )

    observed_state: list[int] = []
    for position, value in enumerate(raw):
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, np.integer)
        ):
            raise ValueError(
                "observation.observed_state"
                f"[{position}] must be an integer, got {value!r}"
            )
        observed_state.append(int(value))
    if len(set(observed_state)) != len(observed_state):
        raise ValueError(
            "observation.observed_state contains duplicate indices: "
            f"{observed_state}"
        )
    return observed_state


def run_replay_identity_baseline(
    *,
    x_true: np.ndarray,
    y_obs: np.ndarray,
    observed_state: list[int],
    allow_true_fill: bool,
) -> np.ndarray:
    x_true_array = np.asarray(x_true)
    y_obs_array = np.asarray(y_obs)
    if x_true_array.ndim != 3:
        raise ValueError(
            "x_true must have shape [N,T,Dx] (rank 3), "
            f"got shape={_shape(x_true_array)}"
        )
    if y_obs_array.ndim != 3 or y_obs_array.shape[:2] != x_true_array.shape[:2]:
        raise ValueError(
            "y_obs must have shape [N,T,Dy] matching x_true[:2], "
            f"got x_true={_shape(x_true_array)}, y_obs={_shape(y_obs_array)}"
        )
    if len(observed_state) != y_obs_array.shape[-1]:
        raise ValueError(
            "len(observation.observed_state) must equal y_obs Dy: "
            f"observed={len(observed_state)}, Dy={y_obs_array.shape[-1]}"
        )
    if len(set(observed_state)) != len(observed_state):
        raise ValueError(
            "observation.observed_state contains duplicate indices: "
            f"{observed_state}"
        )

    x_dim = int(x_true_array.shape[-1])
    for position, index in enumerate(observed_state):
        if isinstance(index, (bool, np.bool_)) or not isinstance(
            index, (int, np.integer)
        ):
            raise ValueError(
                "observation.observed_state"
                f"[{position}] must be an integer, got {index!r}"
            )
        if int(index) < 0 or int(index) >= x_dim:
            raise ValueError(
                "observation.observed_state"
                f"[{position}]={index} is out of bounds for Dx={x_dim}"
            )

    x_hat = (
        np.array(x_true_array, copy=True)
        if allow_true_fill
        else np.zeros_like(x_true_array)
    )
    for measurement_index, state_index in enumerate(observed_state):
        x_hat[..., int(state_index)] = y_obs_array[..., measurement_index]
    return x_hat


def run_checkpoint_adapter_replay(
    *,
    model_id: str,
    checkpoint: str | Path,
    model_config: str | Path | None,
    y_obs: np.ndarray,
    replay_meta: dict[str, Any],
    device: str,
) -> np.ndarray:
    result = run_checkpoint_replay_adapter(
        model_id=model_id,
        checkpoint=checkpoint,
        model_config=model_config,
        y_obs=y_obs,
        replay_meta=replay_meta,
        device=device,
    )
    return result.x_hat


def _resolve_optional_file(
    value: str | Path | None,
    *,
    label: str,
    allow_directory: bool = False,
) -> Path | None:
    if value is None:
        return None
    path = Path(value).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    if not path.is_file() and not (allow_directory and path.is_dir()):
        expected = "a file or directory" if allow_directory else "a file"
        raise ValueError(f"{label} must be {expected}: {path}")
    return path


def _replay_meta_summary(meta: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: meta.get(key)
        for key in (
            "schema_version",
            "suite_version",
            "task_name",
            "time",
            "dataset_sizes",
            "num_trajectories",
            "state_dim",
            "measurement_dim",
            "vizard",
            "replay",
        )
        if key in meta
    }


def _publish_staged_artifact(
    staged_artifact: Path,
    staged_meta: Path,
    *,
    final_artifacts_dir: Path,
) -> tuple[Path, Path]:
    final_artifacts_dir.mkdir(parents=True, exist_ok=True)
    final_artifact = final_artifacts_dir / PRED_ARTIFACT_FILENAME
    final_meta = final_artifacts_dir / PRED_META_FILENAME
    staged_artifact.replace(final_artifact)
    staged_meta.replace(final_meta)
    return final_artifact, final_meta


def run_phase6b_replay(
    replay_input_dir: str | Path,
    *,
    out_dir: str | Path,
    model_id: str = IDENTITY_MODEL_ID,
    checkpoint: str | Path | None = None,
    model_config: str | Path | None = None,
    device: str = "cpu",
    allow_true_fill: bool = False,
    allow_fallback: bool = False,
    strict: bool = True,
) -> tuple[Path, Path]:
    requested_model_id = str(model_id).strip()
    if not requested_model_id:
        raise ValueError("model_id must be a non-empty string")
    requested_device = str(device).strip()
    if not requested_device:
        raise ValueError("device must be a non-empty string")

    arrays, replay_meta = load_replay_input(
        replay_input_dir,
        strict=bool(strict),
    )
    x_true = np.asarray(arrays["x_true"])
    y_obs = np.asarray(arrays["y_obs"])
    time_s = np.asarray(arrays["time_s"])
    trajectory_id = np.asarray(arrays["trajectory_id"])
    observed_state = infer_observed_state_indices(replay_meta)

    checkpoint_path: Path | None = None
    model_config_path = _resolve_optional_file(
        model_config,
        label="model config",
        allow_directory=True,
    )
    actual_model_id = requested_model_id
    is_trained_checkpoint = False
    is_mock_adapter = False
    used_fallback = False
    fallback_reason: str | None = None
    adapter_metadata: dict[str, Any] | None = None
    checkpoint_probe: dict[str, Any] | None = None

    if requested_model_id == IDENTITY_MODEL_ID:
        x_hat = run_replay_identity_baseline(
            x_true=x_true,
            y_obs=y_obs,
            observed_state=observed_state,
            allow_true_fill=bool(allow_true_fill),
        )
        phase_note = (
            "This is a deterministic replay fallback for visualization "
            "pipeline validation, not a trained model result."
        )
    else:
        if checkpoint is None:
            raise FileNotFoundError(
                f"checkpoint is required for model_id={requested_model_id!r}"
            )
        checkpoint_path = _resolve_optional_file(
            checkpoint,
            label="checkpoint",
            allow_directory=True,
        )
        try:
            checkpoint_probe = probe_checkpoint_contract(
                checkpoint_path,
                model_id=requested_model_id,
                model_config=model_config_path,
                run_dir=Path(replay_input_dir).expanduser().resolve(),
            )
            adapter_result: CheckpointReplayResult = (
                run_checkpoint_replay_adapter(
                    model_id=requested_model_id,
                    checkpoint=checkpoint_path,
                    model_config=model_config_path,
                    y_obs=y_obs,
                    replay_meta=replay_meta,
                    device=requested_device,
                    strict=bool(strict),
                )
            )
            x_hat = np.asarray(adapter_result.x_hat)
            adapter_metadata = dict(adapter_result.metadata)
            validate_checkpoint_replay_output(
                x_hat=x_hat,
                x_true=x_true,
                model_id=requested_model_id,
            )
            is_mock_adapter = bool(
                adapter_metadata.get("is_mock_adapter", False)
            )
            is_trained_checkpoint = not is_mock_adapter
            if is_mock_adapter:
                phase_note = (
                    "This artifact was generated by a test-only mock "
                    "checkpoint adapter and is not a trained model result."
                )
            else:
                phase_note = (
                    "This artifact was generated from a trained checkpoint "
                    "replay path."
                )
        except Exception as exc:
            if not allow_fallback:
                raise
            used_fallback = True
            fallback_reason = f"{type(exc).__name__}: {exc}"
            actual_model_id = IDENTITY_MODEL_ID
            is_trained_checkpoint = False
            is_mock_adapter = False
            adapter_metadata = None
            x_hat = run_replay_identity_baseline(
                x_true=x_true,
                y_obs=y_obs,
                observed_state=observed_state,
                allow_true_fill=bool(allow_true_fill),
            )
            phase_note = (
                "Checkpoint replay failed or was unsupported; fallback was "
                "used only because allow_fallback=true. This is a "
                "deterministic replay fallback for visualization pipeline "
                "validation, not a trained model result."
            )

    x_hat = np.asarray(x_hat)
    if x_hat.ndim != 3 or x_hat.shape != x_true.shape:
        raise ValueError(
            "x_hat must have shape identical to x_true: "
            f"x_true={_shape(x_true)}, x_hat={_shape(x_hat)}"
        )
    _require_finite("x_hat", x_hat)

    output_root = Path(out_dir).expanduser().resolve()
    try:
        output_root.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise RuntimeError(
            f"failed to create Phase 6B output directory: {output_root}"
        ) from exc

    not_for_benchmark_reporting = (
        actual_model_id == IDENTITY_MODEL_ID or is_mock_adapter
    )
    output_meta = {
        "phase": (
            "phase6b_checkpoint_replay"
            if requested_model_id == IDENTITY_MODEL_ID
            else "phase6d_checkpoint_replay"
        ),
        "source_replay_input_dir": str(
            Path(replay_input_dir).expanduser().resolve()
        ),
        "source_replay_npz": str(arrays["npz_path"]),
        "source_replay_meta": str(arrays["meta_path"]),
        "model_id": actual_model_id,
        "requested_model_id": requested_model_id,
        "checkpoint_path": (
            None if checkpoint_path is None else str(checkpoint_path)
        ),
        "model_config_path": (
            None if model_config_path is None else str(model_config_path)
        ),
        "device": requested_device,
        "is_trained_checkpoint": bool(is_trained_checkpoint),
        "is_mock_adapter": bool(is_mock_adapter),
        "used_fallback": bool(used_fallback),
        "fallback_reason": fallback_reason,
        "original_model_id": (
            requested_model_id if used_fallback else None
        ),
        "original_checkpoint_path": (
            str(checkpoint_path)
            if used_fallback and checkpoint_path is not None
            else None
        ),
        "allow_true_fill": bool(allow_true_fill),
        "strict": bool(strict),
        "suite_name": replay_meta["suite_name"],
        "task_id": replay_meta["task_id"],
        "seed": int(replay_meta["seed"]),
        "scenario_id": replay_meta["scenario_id"],
        "state_schema": replay_meta["state_schema"],
        "observation": replay_meta.get(
            "observation",
            {"observed_state": observed_state},
        ),
        "attitude_convention": replay_meta.get(
            "attitude_convention",
            "MRP sigma_BN",
        ),
        "time_unit": replay_meta.get("time_unit", "s"),
        "replay_meta_summary": _replay_meta_summary(replay_meta),
        "output_shape_summary": {
            "time_s": list(time_s.shape),
            "x_true": list(x_true.shape),
            "y_obs": list(y_obs.shape),
            "x_hat": list(x_hat.shape),
            "trajectory_id": list(trajectory_id.shape),
        },
        "identity_fill_policy": (
            (
                "observed_from_y_obs_unobserved_from_x_true"
                if allow_true_fill
                else "observed_from_y_obs_unobserved_zero"
            )
            if actual_model_id == IDENTITY_MODEL_ID
            else None
        ),
        "checkpoint_replay_adapter_metadata": adapter_metadata,
        "checkpoint_contract_probe_summary": checkpoint_probe,
        "normalization_applied": (
            None
            if adapter_metadata is None
            else adapter_metadata.get("normalization_applied", "unknown")
        ),
        "hidden_state_initialization": (
            None
            if adapter_metadata is None
            else adapter_metadata.get("hidden_state_initialization")
        ),
        "input_layout": (
            None
            if adapter_metadata is None
            else adapter_metadata.get("input_layout")
        ),
        "output_layout": (
            None
            if adapter_metadata is None
            else adapter_metadata.get("output_layout")
        ),
        "assumptions": (
            [] if adapter_metadata is None
            else list(adapter_metadata.get("assumptions", []))
        ),
        "warnings": (
            [] if adapter_metadata is None
            else list(adapter_metadata.get("warnings", []))
        ),
        "purpose": (
            "pipeline fallback only"
            if actual_model_id == IDENTITY_MODEL_ID
            else (
                "test-only checkpoint replay"
                if is_mock_adapter
                else "trained checkpoint replay"
            )
        ),
        "not_for_benchmark_reporting": not_for_benchmark_reporting,
        "phase6b_notes": phase_note,
    }

    try:
        with tempfile.TemporaryDirectory(
            prefix=".phase6b_replay_",
            dir=output_root,
        ) as temporary:
            staged_artifacts_dir = Path(temporary) / "artifacts"
            staged_artifact, staged_meta = save_pred_artifact(
                staged_artifacts_dir,
                time_s=time_s,
                x_true=x_true,
                y_obs=y_obs,
                x_hat=x_hat,
                trajectory_id=trajectory_id,
                meta=output_meta,
            )
            staged_meta_obj = _load_json_object(staged_meta)
            default_note = str(staged_meta_obj.get("notes", "")).strip()
            staged_meta_obj["notes"] = (
                f"{default_note} {phase_note}".strip()
            )
            staged_meta_tmp = staged_meta.with_name(
                f".{staged_meta.name}.phase6b.tmp"
            )
            staged_meta_tmp.write_text(
                json.dumps(staged_meta_obj, indent=2, ensure_ascii=False)
                + "\n",
                encoding="utf-8",
            )
            staged_meta_tmp.replace(staged_meta)
            load_pred_artifact(staged_artifact)
            return _publish_staged_artifact(
                staged_artifact,
                staged_meta,
                final_artifacts_dir=output_root / "artifacts",
            )
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"failed to write Phase 6B prediction artifact under "
            f"{output_root}: {exc}"
        ) from exc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run Phase 6B replay inference and write a standard Phase 1 "
            "prediction artifact."
        )
    )
    parser.add_argument("--replay-input-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--model-id", default=IDENTITY_MODEL_ID)
    parser.add_argument("--checkpoint")
    parser.add_argument("--model-config")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--allow-true-fill", action="store_true")
    parser.add_argument("--allow-fallback", action="store_true")
    parser.add_argument(
        "--no-strict",
        action="store_true",
        help="Disable strict finite-value checks while loading replay input.",
    )
    args = parser.parse_args(argv)

    artifact_path, meta_path = run_phase6b_replay(
        args.replay_input_dir,
        out_dir=args.out_dir,
        model_id=args.model_id,
        checkpoint=args.checkpoint,
        model_config=args.model_config,
        device=args.device,
        allow_true_fill=bool(args.allow_true_fill),
        allow_fallback=bool(args.allow_fallback),
        strict=not bool(args.no_strict),
    )
    print(f"wrote {artifact_path}")
    print(f"wrote {meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
