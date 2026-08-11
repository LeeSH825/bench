from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .checkpoint_contract_probe import (
    probe_checkpoint_contract,
    save_checkpoint_contract_probe,
)
from .phase6e_checkpoint_package import (
    build_replay_checkpoint_package,
    validate_replay_checkpoint_package,
)
from .replay_checkpoint_contract import (
    load_replay_checkpoint_contract,
    save_replay_checkpoint_contract,
)


KALMANNET_TSP_MODEL_ID = "kalmannet_tsp"
KALMANNET_TSP_REPLAY_ADAPTER_ID = "kalmannet_tsp_replay_adapter_v1"
KALMANNET_MODEL_CONFIG_SCHEMA_VERSION = "kalmannet_tsp_model_config_v1"
KALMANNET_SYSTEM_MODEL_SCHEMA_VERSION = "kalmannet_tsp_system_model_v1"
PHASE6F_STATE_DIM = 9
PHASE6F_MEASUREMENT_DIM = 6
PHASE6F_OBSERVED_STATE = [0, 1, 2, 3, 4, 5]


def _file(
    value: str | Path | None,
    *,
    label: str,
    required: bool = False,
) -> Path | None:
    if value is None:
        if required:
            raise ValueError(f"{label} is required for KalmanNet_TSP export")
        return None
    path = Path(value).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    if not path.is_file():
        raise ValueError(f"{label} must be a file: {path}")
    return path


def _json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid {label} JSON at {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object: {path}")
    return value


def _positive_int(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return int(value)


def _validate_dimensions(
    value: Mapping[str, Any],
    *,
    label: str,
) -> None:
    state_dim = _positive_int(f"{label}.state_dim", value.get("state_dim"))
    measurement_dim = _positive_int(
        f"{label}.measurement_dim",
        value.get("measurement_dim"),
    )
    if state_dim != PHASE6F_STATE_DIM:
        raise ValueError(
            f"{label}.state_dim must be {PHASE6F_STATE_DIM}, "
            f"got {state_dim}"
        )
    if measurement_dim != PHASE6F_MEASUREMENT_DIM:
        raise ValueError(
            f"{label}.measurement_dim must be "
            f"{PHASE6F_MEASUREMENT_DIM}, got {measurement_dim}"
        )


def _validate_model_config(path: Path) -> dict[str, Any]:
    config = _json_object(path, label="model_config")
    if config.get("schema_version") != KALMANNET_MODEL_CONFIG_SCHEMA_VERSION:
        raise ValueError(
            "model_config.schema_version must be "
            f"{KALMANNET_MODEL_CONFIG_SCHEMA_VERSION!r}"
        )
    if config.get("model_id") != KALMANNET_TSP_MODEL_ID:
        raise ValueError(
            "model_config.model_id must be 'kalmannet_tsp'"
        )
    _validate_dimensions(config, label="model_config")
    for field in ("input_layout", "output_layout"):
        if config.get(field) != "NTD":
            raise ValueError(f"model_config.{field} must be 'NTD'")
    hidden = config.get("hidden_state_initialization")
    if not isinstance(hidden, Mapping):
        raise ValueError(
            "model_config.hidden_state_initialization must be an object"
        )
    method = str(hidden.get("method", "")).strip()
    if method not in {"zeros", "model_default"}:
        raise ValueError(
            "model_config.hidden_state_initialization.method must be "
            "'zeros' or 'model_default'"
        )
    normalization = config.get("normalization")
    if not isinstance(normalization, Mapping):
        raise ValueError("model_config.normalization must be an object")
    if not isinstance(normalization.get("enabled"), bool):
        raise ValueError(
            "model_config.normalization.enabled must be a boolean"
        )
    return config


def _numeric_matrix(
    value: Any,
    *,
    name: str,
    shape: tuple[int, int],
) -> np.ndarray:
    try:
        matrix = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"system_model.{name} must be numeric") from exc
    if matrix.shape != shape:
        raise ValueError(
            f"system_model.{name} must have shape {shape}, "
            f"got {tuple(matrix.shape)}"
        )
    if not np.isfinite(matrix).all():
        raise ValueError(
            f"system_model.{name} contains NaN or Inf values"
        )
    return matrix


def _validate_system_model(path: Path) -> dict[str, Any]:
    system = _json_object(path, label="system_model")
    if system.get("schema_version") != KALMANNET_SYSTEM_MODEL_SCHEMA_VERSION:
        raise ValueError(
            "system_model.schema_version must be "
            f"{KALMANNET_SYSTEM_MODEL_SCHEMA_VERSION!r}"
        )
    if system.get("format") != "linear_F_H":
        raise ValueError("system_model.format must be 'linear_F_H'")
    _validate_dimensions(system, label="system_model")
    _numeric_matrix(
        system.get("F"),
        name="F",
        shape=(PHASE6F_STATE_DIM, PHASE6F_STATE_DIM),
    )
    _numeric_matrix(
        system.get("H"),
        name="H",
        shape=(PHASE6F_MEASUREMENT_DIM, PHASE6F_STATE_DIM),
    )
    _numeric_matrix(
        system.get("Q"),
        name="Q",
        shape=(PHASE6F_STATE_DIM, PHASE6F_STATE_DIM),
    )
    _numeric_matrix(
        system.get("R"),
        name="R",
        shape=(PHASE6F_MEASUREMENT_DIM, PHASE6F_MEASUREMENT_DIM),
    )
    return system


def _validate_normalizer(
    path: Path | None,
    *,
    required: bool,
) -> dict[str, Any] | None:
    if path is None:
        if required:
            raise ValueError(
                "normalizer is required because model_config declares "
                "normalization.enabled=true"
            )
        return None
    normalizer = _json_object(path, label="normalizer")
    if normalizer.get("format") != "standard_scaler":
        raise ValueError(
            "normalizer.format must be 'standard_scaler'"
        )
    return normalizer


def _discover_file(
    run_dir: Path,
    explicit: str | Path | None,
    *,
    label: str,
    candidates: Sequence[str],
    required: bool,
) -> Path | None:
    if explicit is not None:
        return _file(explicit, label=label, required=required)
    for candidate in candidates:
        path = run_dir / candidate
        if path.is_file():
            return path.resolve()
    if required:
        joined = ", ".join(candidates)
        raise FileNotFoundError(
            f"{label} was not supplied and no source-run candidate exists "
            f"under {run_dir}: {joined}"
        )
    return None


def _publish_directory(
    staging: Path,
    destination: Path,
    *,
    overwrite: bool,
) -> None:
    backup: Path | None = None
    if destination.exists():
        if not overwrite:
            raise FileExistsError(
                f"KalmanNet replay package already exists: {destination}; "
                "use overwrite=True or --overwrite"
            )
        backup = destination.with_name(
            f".{destination.name}.backup-{os.getpid()}"
        )
        if backup.exists():
            shutil.rmtree(backup)
        destination.replace(backup)
    try:
        staging.replace(destination)
    except Exception:
        if backup is not None and backup.exists() and not destination.exists():
            backup.replace(destination)
        raise
    if backup is not None and backup.exists():
        shutil.rmtree(backup)


def _relocate_probe_paths(
    value: Any,
    *,
    source_dir: Path,
    destination_dir: Path,
) -> Any:
    if isinstance(value, dict):
        return {
            key: _relocate_probe_paths(
                item,
                source_dir=source_dir,
                destination_dir=destination_dir,
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [
            _relocate_probe_paths(
                item,
                source_dir=source_dir,
                destination_dir=destination_dir,
            )
            for item in value
        ]
    if isinstance(value, str):
        source = str(source_dir)
        if value == source or value.startswith(f"{source}{os.sep}"):
            return f"{destination_dir}{value[len(source):]}"
    return value


def export_kalmannet_tsp_replay_package(
    *,
    source_run_dir: str | Path | None = None,
    checkpoint: str | Path | None = None,
    package_dir: str | Path,
    model_config: str | Path | None = None,
    system_model: str | Path | None = None,
    normalizer: str | Path | None = None,
    training_summary: str | Path | None = None,
    training_suite_name: str | None = None,
    training_task_id: str | None = None,
    training_seed: int | None = None,
    checkpoint_step: int | None = None,
    overwrite: bool = False,
) -> Path:
    source_run = (
        None
        if source_run_dir is None
        else Path(source_run_dir).expanduser().resolve()
    )
    if source_run is not None:
        if not source_run.exists():
            raise FileNotFoundError(
                f"source_run_dir not found: {source_run}"
            )
        if not source_run.is_dir():
            raise ValueError(
                f"source_run_dir must be a directory: {source_run}"
            )
    if source_run is None and checkpoint is None:
        raise ValueError(
            "checkpoint is required when source_run_dir is not provided"
        )

    if source_run is None:
        checkpoint_path = _file(
            checkpoint,
            label="checkpoint",
            required=True,
        )
        model_config_path = _file(
            model_config,
            label="model_config",
            required=True,
        )
        system_model_path = _file(
            system_model,
            label="system_model",
            required=True,
        )
        normalizer_path = _file(normalizer, label="normalizer")
        training_summary_path = _file(
            training_summary,
            label="training_summary",
        )
    else:
        checkpoint_path = _discover_file(
            source_run,
            checkpoint,
            label="checkpoint",
            candidates=(
                "checkpoint.pt",
                "checkpoints/model.pt",
                "checkpoints/checkpoint.pt",
            ),
            required=True,
        )
        model_config_path = _discover_file(
            source_run,
            model_config,
            label="model_config",
            candidates=(
                "model_config.json",
                "artifacts/model_config.json",
            ),
            required=True,
        )
        system_model_path = _discover_file(
            source_run,
            system_model,
            label="system_model",
            candidates=(
                "system_model.json",
                "artifacts/system_model.json",
            ),
            required=True,
        )
        normalizer_path = _discover_file(
            source_run,
            normalizer,
            label="normalizer",
            candidates=(
                "normalizer.json",
                "artifacts/normalizer.json",
            ),
            required=False,
        )
        training_summary_path = _discover_file(
            source_run,
            training_summary,
            label="training_summary",
            candidates=(
                "training_summary.json",
                "checkpoints/train_state.json",
                "artifacts/training_summary.json",
            ),
            required=False,
        )
    assert checkpoint_path is not None
    assert model_config_path is not None
    assert system_model_path is not None

    config = _validate_model_config(model_config_path)
    _validate_system_model(system_model_path)
    normalization_required = bool(config["normalization"]["enabled"])
    _validate_normalizer(
        normalizer_path,
        required=normalization_required,
    )
    summary = (
        {}
        if training_summary_path is None
        else _json_object(
            training_summary_path,
            label="training_summary",
        )
    )
    reporting_recommended = bool(
        summary.get("benchmark_reporting_recommended", False)
    )
    smoke_training = bool(summary.get("smoke_training", True))
    hidden_method = str(
        config["hidden_state_initialization"]["method"]
    )

    output_dir = Path(package_dir).expanduser().resolve()
    if output_dir.exists() and not overwrite:
        raise FileExistsError(
            f"KalmanNet replay package already exists: {output_dir}; "
            "use overwrite=True or --overwrite"
        )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary_root = Path(
        tempfile.mkdtemp(
            prefix=f".{output_dir.name}.phase6f-staging-",
            dir=output_dir.parent,
        )
    )
    staging_package = temporary_root / "package"
    try:
        build_replay_checkpoint_package(
            checkpoint=checkpoint_path,
            model_id=KALMANNET_TSP_MODEL_ID,
            package_dir=staging_package,
            state_dim=PHASE6F_STATE_DIM,
            measurement_dim=PHASE6F_MEASUREMENT_DIM,
            observed_state=list(PHASE6F_OBSERVED_STATE),
            adapter_id=KALMANNET_TSP_REPLAY_ADAPTER_ID,
            model_config=model_config_path,
            normalizer=normalizer_path,
            system_model=system_model_path,
            training_summary=training_summary_path,
            training_suite_name=training_suite_name,
            training_task_id=training_task_id,
            training_seed=training_seed,
            training_run_dir=source_run,
            checkpoint_step=checkpoint_step,
            requires_system_model=True,
            requires_normalization=normalization_required,
            system_model_format="linear_F_H",
            normalization_format=(
                "standard_scaler"
                if normalization_required
                else "none"
            ),
            hidden_state_initialization_method=hidden_method,
            is_mock=False,
            not_for_benchmark_reporting=not reporting_recommended,
        )
        contract = load_replay_checkpoint_contract(staging_package)
        contract["benchmark_reporting_recommended"] = (
            reporting_recommended
        )
        contract["smoke_training"] = smoke_training
        contract["checkpoint_compatibility_verified"] = False
        contract["model_config_schema_version"] = (
            KALMANNET_MODEL_CONFIG_SCHEMA_VERSION
        )
        contract["system_model_schema_version"] = (
            KALMANNET_SYSTEM_MODEL_SCHEMA_VERSION
        )
        contract["warnings"] = list(contract["warnings"]) + [
            "The KalmanNet_TSP adapter is structurally registered, but "
            "checkpoint/runtime compatibility remains unverified until this "
            "exact trained 9x6 package is exercised through the real "
            "upstream-backed adapter."
        ]
        contract["notes"] = (
            "Phase 6F 9x6 ADCS KalmanNet_TSP replay package. Structural "
            "validation does not prove that checkpoint weights were trained "
            "for this observation contract."
        )
        save_replay_checkpoint_contract(contract, staging_package)
        validate_replay_checkpoint_package(
            staging_package,
            expected_state_dim=PHASE6F_STATE_DIM,
            expected_measurement_dim=PHASE6F_MEASUREMENT_DIM,
            expected_observed_state=list(PHASE6F_OBSERVED_STATE),
        )
        staging_probe = probe_checkpoint_contract(
            staging_package,
            model_id=KALMANNET_TSP_MODEL_ID,
        )
        final_probe = _relocate_probe_paths(
            staging_probe,
            source_dir=staging_package.resolve(),
            destination_dir=output_dir,
        )
        save_checkpoint_contract_probe(final_probe, staging_package)
        _publish_directory(
            staging_package,
            output_dir,
            overwrite=bool(overwrite),
        )
    finally:
        if temporary_root.exists():
            shutil.rmtree(temporary_root)

    return output_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Export a strict 9x6 ADCS KalmanNet_TSP replay package."
        )
    )
    parser.add_argument("--source-run-dir")
    parser.add_argument("--checkpoint")
    parser.add_argument("--package-dir", required=True)
    parser.add_argument("--model-config")
    parser.add_argument("--system-model")
    parser.add_argument("--normalizer")
    parser.add_argument("--training-summary")
    parser.add_argument("--training-suite-name")
    parser.add_argument("--training-task-id")
    parser.add_argument("--training-seed", type=int)
    parser.add_argument("--checkpoint-step", type=int)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    if args.source_run_dir is None and args.checkpoint is None:
        parser.error("one of --source-run-dir or --checkpoint is required")

    package = export_kalmannet_tsp_replay_package(
        source_run_dir=args.source_run_dir,
        checkpoint=args.checkpoint,
        package_dir=args.package_dir,
        model_config=args.model_config,
        system_model=args.system_model,
        normalizer=args.normalizer,
        training_summary=args.training_summary,
        training_suite_name=args.training_suite_name,
        training_task_id=args.training_task_id,
        training_seed=args.training_seed,
        checkpoint_step=args.checkpoint_step,
        overwrite=bool(args.overwrite),
    )
    print(f"wrote {package / 'replay_contract.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
