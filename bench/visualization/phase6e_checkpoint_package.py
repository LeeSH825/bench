from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .replay_checkpoint_contract import (
    REPLAY_CHECKPOINT_CONTRACT_FILENAME,
    REPLAY_CHECKPOINT_CONTRACT_SCHEMA_VERSION,
    load_replay_checkpoint_contract,
    save_replay_checkpoint_contract,
    validate_replay_checkpoint_contract,
)


_PACKAGE_FILENAMES = {
    "checkpoint_path": "checkpoint.pt",
    "model_config_path": "model_config.json",
    "normalizer_path": "normalizer.json",
    "system_model_path": "system_model.json",
    "training_summary_path": "training_summary.json",
}


def _source_file(value: str | Path | None, *, label: str) -> Path | None:
    if value is None:
        return None
    path = Path(value).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    if not path.is_file():
        raise ValueError(f"{label} must be a file: {path}")
    return path


def _checkpoint_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _package_id(
    *,
    model_id: str,
    checkpoint: Path,
    state_dim: int,
    measurement_dim: int,
    observed_state: list[int],
) -> str:
    payload = {
        "model_id": model_id,
        "checkpoint_sha256": _checkpoint_digest(checkpoint),
        "state_dim": int(state_dim),
        "measurement_dim": int(measurement_dim),
        "observed_state": [int(index) for index in observed_state],
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"replay_ckpt_{hashlib.sha256(encoded).hexdigest()[:12]}"


def _default_state_schema(state_dim: int) -> dict[str, Any]:
    if state_dim >= 6:
        schema: dict[str, Any] = {
            "attitude": {
                "type": "mrp",
                "name": "sigma_BN",
                "indices": [0, 1, 2],
            },
            "angular_rate": {
                "type": "rad_s",
                "name": "omega_BN_B",
                "indices": [3, 4, 5],
            },
        }
        if state_dim >= 9:
            schema["gyro_bias"] = {
                "type": "rad_s",
                "name": "gyro_bias",
                "indices": [6, 7, 8],
                "optional": True,
            }
        return schema
    return {"state_dim": int(state_dim)}


def _copy_package_file(
    source: Path | None,
    *,
    staging_dir: Path,
    contract_field: str,
) -> str | None:
    if source is None:
        return None
    filename = _PACKAGE_FILENAMES[contract_field]
    shutil.copy2(source, staging_dir / filename)
    return filename


def _publish_package(
    staging_dir: Path,
    package_dir: Path,
    *,
    overwrite: bool,
) -> None:
    backup: Path | None = None
    if package_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"checkpoint package already exists: {package_dir}; "
                "use overwrite=True or --overwrite"
            )
        backup = package_dir.with_name(
            f".{package_dir.name}.backup-{os.getpid()}"
        )
        if backup.exists():
            shutil.rmtree(backup)
        package_dir.replace(backup)
    try:
        staging_dir.replace(package_dir)
    except Exception:
        if backup is not None and backup.exists() and not package_dir.exists():
            backup.replace(package_dir)
        raise
    if backup is not None and backup.exists():
        shutil.rmtree(backup)


def build_replay_checkpoint_package(
    *,
    checkpoint: str | Path,
    model_id: str,
    package_dir: str | Path,
    state_dim: int,
    measurement_dim: int,
    observed_state: list[int],
    adapter_id: str | None = None,
    model_config: str | Path | None = None,
    normalizer: str | Path | None = None,
    system_model: str | Path | None = None,
    training_summary: str | Path | None = None,
    training_suite_name: str | None = None,
    training_task_id: str | None = None,
    training_seed: int | None = None,
    training_run_dir: str | Path | None = None,
    checkpoint_step: int | None = None,
    requires_system_model: bool = False,
    requires_normalization: bool = False,
    system_model_format: str = "none",
    normalization_format: str = "none",
    hidden_state_initialization_method: str = "zeros",
    is_mock: bool = False,
    not_for_benchmark_reporting: bool = False,
    overwrite: bool = False,
) -> Path:
    checkpoint_source = _source_file(checkpoint, label="checkpoint")
    assert checkpoint_source is not None
    source_files = {
        "model_config_path": _source_file(
            model_config,
            label="model config",
        ),
        "normalizer_path": _source_file(
            normalizer,
            label="normalizer",
        ),
        "system_model_path": _source_file(
            system_model,
            label="system model",
        ),
        "training_summary_path": _source_file(
            training_summary,
            label="training summary",
        ),
    }
    requested_model_id = str(model_id).strip()
    if not requested_model_id:
        raise ValueError("model_id must be a non-empty string")
    resolved_adapter_id = str(adapter_id or requested_model_id).strip()
    if not resolved_adapter_id:
        raise ValueError("adapter_id must be a non-empty string")

    output_dir = Path(package_dir).expanduser().resolve()
    if output_dir.exists() and not overwrite:
        raise FileExistsError(
            f"checkpoint package already exists: {output_dir}; "
            "use overwrite=True or --overwrite"
        )
    try:
        output_dir.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise RuntimeError(
            f"failed to create checkpoint package parent: {output_dir.parent}"
        ) from exc

    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{output_dir.name}.staging-",
            dir=output_dir.parent,
        )
    )
    try:
        checkpoint_path = _copy_package_file(
            checkpoint_source,
            staging_dir=temporary,
            contract_field="checkpoint_path",
        )
        copied_paths = {
            field: _copy_package_file(
                source,
                staging_dir=temporary,
                contract_field=field,
            )
            for field, source in source_files.items()
        }
        created_at = (
            datetime.now(timezone.utc)
            .isoformat(timespec="seconds")
            .replace("+00:00", "Z")
        )
        normalized_observed_state = [
            int(index) for index in observed_state
        ]
        contract = {
            "schema_version": REPLAY_CHECKPOINT_CONTRACT_SCHEMA_VERSION,
            "package_id": _package_id(
                model_id=requested_model_id,
                checkpoint=checkpoint_source,
                state_dim=int(state_dim),
                measurement_dim=int(measurement_dim),
                observed_state=normalized_observed_state,
            ),
            "created_at_utc": created_at,
            "model_id": requested_model_id,
            "adapter_id": resolved_adapter_id,
            "checkpoint_path": checkpoint_path,
            **copied_paths,
            "state_dim": int(state_dim),
            "measurement_dim": int(measurement_dim),
            "observed_state": normalized_observed_state,
            "input_layout": "NTD",
            "output_layout": "NTD",
            "time_layout": "time_s_T",
            "training_suite_name": training_suite_name,
            "training_task_id": training_task_id,
            "training_seed": training_seed,
            "training_run_dir": (
                None
                if training_run_dir is None
                else str(Path(training_run_dir).expanduser().resolve())
            ),
            "checkpoint_step": checkpoint_step,
            "checkpoint_metric": None,
            "checkpoint_metric_value": None,
            "requires_system_model": bool(requires_system_model),
            "system_model_format": str(system_model_format),
            "requires_normalization": bool(requires_normalization),
            "normalization_format": str(normalization_format),
            "hidden_state_initialization": {
                "method": str(hidden_state_initialization_method),
                "source": "replay_contract",
                "details": {},
            },
            "preprocessing": {
                "input_transform": (
                    "standard_scaler"
                    if requires_normalization
                    else "identity"
                ),
                "output_inverse_transform": (
                    "standard_scaler_inverse"
                    if requires_normalization
                    else "identity"
                ),
                "assumptions": [],
            },
            "compatibility": {
                "compatible_replay_schema_versions": [
                    "phase6a_replay_input_v1"
                ],
                "compatible_state_schema": _default_state_schema(
                    int(state_dim)
                ),
                "compatible_observation_schema": {
                    "observed_state": normalized_observed_state,
                },
            },
            "is_mock": bool(is_mock),
            "not_for_benchmark_reporting": bool(
                not_for_benchmark_reporting
            ),
            "warnings": (
                [
                    "This package uses a test-only mock checkpoint adapter "
                    "and is not benchmark performance."
                ]
                if is_mock
                else []
            ),
            "notes": (
                "Replay-compatible checkpoint package generated by Phase 6E."
            ),
            "file_storage": {
                "checkpoint_path": "copied",
                **{
                    field: ("copied" if path is not None else "absent")
                    for field, path in copied_paths.items()
                },
            },
        }
        save_replay_checkpoint_contract(contract, temporary)
        validate_replay_checkpoint_package(
            temporary,
            expected_state_dim=int(state_dim),
            expected_measurement_dim=int(measurement_dim),
            expected_observed_state=normalized_observed_state,
        )
        _publish_package(
            temporary,
            output_dir,
            overwrite=bool(overwrite),
        )
    except FileExistsError:
        raise
    except Exception as exc:
        if isinstance(exc, (ValueError, FileNotFoundError)):
            raise
        raise RuntimeError(
            f"failed to build replay checkpoint package at {output_dir}: {exc}"
        ) from exc
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)

    validate_replay_checkpoint_package(
        output_dir,
        expected_state_dim=int(state_dim),
        expected_measurement_dim=int(measurement_dim),
        expected_observed_state=normalized_observed_state,
    )
    return output_dir


def validate_replay_checkpoint_package(
    package_dir: str | Path,
    *,
    expected_state_dim: int | None = None,
    expected_measurement_dim: int | None = None,
    expected_observed_state: list[int] | None = None,
) -> dict[str, Any]:
    directory = Path(package_dir).expanduser().resolve()
    if not directory.exists():
        raise FileNotFoundError(f"checkpoint package not found: {directory}")
    if not directory.is_dir():
        raise ValueError(f"checkpoint package must be a directory: {directory}")
    contract = load_replay_checkpoint_contract(directory)
    return validate_replay_checkpoint_contract(
        contract,
        package_dir=directory,
        expected_state_dim=expected_state_dim,
        expected_measurement_dim=expected_measurement_dim,
        expected_observed_state=expected_observed_state,
    )


def _parse_indices(value: str | None, *, label: str) -> list[int] | None:
    if value is None:
        return None
    try:
        result = [int(item.strip()) for item in value.split(",")]
    except ValueError as exc:
        raise ValueError(
            f"{label} must be a comma-separated integer list"
        ) from exc
    if not result:
        raise ValueError(f"{label} must not be empty")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build or validate a Phase 6E replay checkpoint package."
    )
    parser.add_argument("--validate-package")
    parser.add_argument("--checkpoint")
    parser.add_argument("--model-id")
    parser.add_argument("--package-dir")
    parser.add_argument("--state-dim", type=int)
    parser.add_argument("--measurement-dim", type=int)
    parser.add_argument("--observed-state")
    parser.add_argument("--adapter-id")
    parser.add_argument("--model-config")
    parser.add_argument("--normalizer")
    parser.add_argument("--system-model")
    parser.add_argument("--training-summary")
    parser.add_argument("--training-suite-name")
    parser.add_argument("--training-task-id")
    parser.add_argument("--training-seed", type=int)
    parser.add_argument("--training-run-dir")
    parser.add_argument("--checkpoint-step", type=int)
    parser.add_argument("--requires-system-model", action="store_true")
    parser.add_argument("--requires-normalization", action="store_true")
    parser.add_argument("--system-model-format", default="none")
    parser.add_argument("--normalization-format", default="none")
    parser.add_argument(
        "--hidden-state-initialization",
        default="zeros",
    )
    parser.add_argument("--is-mock", action="store_true")
    parser.add_argument("--not-for-benchmark-reporting", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--expected-state-dim", type=int)
    parser.add_argument("--expected-measurement-dim", type=int)
    parser.add_argument("--expected-observed-state")
    args = parser.parse_args(argv)

    if args.validate_package:
        if args.checkpoint or args.package_dir:
            parser.error(
                "--validate-package cannot be combined with build arguments"
            )
        validated = validate_replay_checkpoint_package(
            args.validate_package,
            expected_state_dim=args.expected_state_dim,
            expected_measurement_dim=args.expected_measurement_dim,
            expected_observed_state=_parse_indices(
                args.expected_observed_state,
                label="expected_observed_state",
            ),
        )
        print(
            "validated "
            f"{Path(args.validate_package).expanduser().resolve()} "
            f"({validated['model_id']} "
            f"{validated['state_dim']}x{validated['measurement_dim']})"
        )
        return 0

    required = {
        "--checkpoint": args.checkpoint,
        "--model-id": args.model_id,
        "--package-dir": args.package_dir,
        "--state-dim": args.state_dim,
        "--measurement-dim": args.measurement_dim,
        "--observed-state": args.observed_state,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        parser.error(f"build mode is missing required arguments: {missing}")

    package = build_replay_checkpoint_package(
        checkpoint=args.checkpoint,
        model_id=args.model_id,
        package_dir=args.package_dir,
        state_dim=args.state_dim,
        measurement_dim=args.measurement_dim,
        observed_state=_parse_indices(
            args.observed_state,
            label="observed_state",
        )
        or [],
        adapter_id=args.adapter_id,
        model_config=args.model_config,
        normalizer=args.normalizer,
        system_model=args.system_model,
        training_summary=args.training_summary,
        training_suite_name=args.training_suite_name,
        training_task_id=args.training_task_id,
        training_seed=args.training_seed,
        training_run_dir=args.training_run_dir,
        checkpoint_step=args.checkpoint_step,
        requires_system_model=bool(args.requires_system_model),
        requires_normalization=bool(args.requires_normalization),
        system_model_format=args.system_model_format,
        normalization_format=args.normalization_format,
        hidden_state_initialization_method=(
            args.hidden_state_initialization
        ),
        is_mock=bool(args.is_mock),
        not_for_benchmark_reporting=bool(
            args.not_for_benchmark_reporting
        ),
        overwrite=bool(args.overwrite),
    )
    print(f"wrote {package / REPLAY_CHECKPOINT_CONTRACT_FILENAME}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
