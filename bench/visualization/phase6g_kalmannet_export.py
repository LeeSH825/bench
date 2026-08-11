from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any, Mapping

import yaml

from bench.tasks.replay_generated_data import build_replay_generated_system_model

from .phase6f_kalmannet_export import (
    KALMANNET_MODEL_CONFIG_SCHEMA_VERSION,
    KALMANNET_SYSTEM_MODEL_SCHEMA_VERSION,
    KALMANNET_TSP_MODEL_ID,
    export_kalmannet_tsp_replay_package,
)


def _read_json(path: Path) -> dict[str, Any]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return obj


def _read_yaml(path: Path) -> dict[str, Any]:
    obj = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError(f"{path} must contain a mapping")
    return obj


def _discover_checkpoint(source_run_dir: Path) -> Path:
    candidates = [
        source_run_dir / "checkpoints" / "model.pt",
        source_run_dir / "checkpoints" / "checkpoint.pt",
        source_run_dir / "checkpoint.pt",
    ]
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        f"no checkpoint.pt/model.pt found under {source_run_dir}"
    )


def _build_model_config_from_run(run_cfg: Mapping[str, Any]) -> dict[str, Any]:
    model_cfg = dict(run_cfg.get("model", {}) or {})
    task_cfg = dict(run_cfg.get("task", {}) or {})
    x_dim = int(task_cfg.get("x_dim", 9))
    y_dim = int(task_cfg.get("y_dim", 6))
    return {
        "schema_version": KALMANNET_MODEL_CONFIG_SCHEMA_VERSION,
        "model_id": KALMANNET_TSP_MODEL_ID,
        "state_dim": x_dim,
        "measurement_dim": y_dim,
        "input_layout": "NTD",
        "output_layout": "NTD",
        "repo": dict(model_cfg.get("repo", {}) or {"path": "third_party/KalmanNet_TSP"}),
        "batch_size": int(model_cfg.get("batch_size", 4)),
        "lr": float(model_cfg.get("lr", 1.0e-4)),
        "weight_decay": float(model_cfg.get("weight_decay", 1.0e-3)),
        "in_mult_KNet": int(model_cfg.get("in_mult_KNet", 5)),
        "out_mult_KNet": int(model_cfg.get("out_mult_KNet", 40)),
        "normalization": {
            "enabled": bool(model_cfg.get("normalization", {}).get("enabled", False))
            if isinstance(model_cfg.get("normalization", {}), Mapping)
            else False,
            "format": "standard_scaler",
        },
        "hidden_state_initialization": {
            "method": str(
                model_cfg.get("hidden_state_initialization", {}).get("method", "zeros")
            )
            if isinstance(model_cfg.get("hidden_state_initialization", {}), Mapping)
            else "zeros",
        },
        "notes": "Derived from a Phase 6G training run config_snapshot.yaml.",
    }


def _build_system_model_from_run(run_cfg: Mapping[str, Any]) -> dict[str, Any]:
    task_cfg = dict(run_cfg.get("task", {}) or {})
    time_cfg = dict(task_cfg.get("time", {}) or {})
    dt_s = float(time_cfg.get("dt_s", 0.5))
    return build_replay_generated_system_model(task_cfg=task_cfg, dt_s=dt_s)


def _build_training_summary_from_run(run_dir: Path) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "smoke_training": True,
        "benchmark_reporting_recommended": False,
        "source_run_dir": str(run_dir),
    }
    for rel in ("checkpoints/train_state.json", "metrics.json", "run_plan.json"):
        path = run_dir / rel
        if not path.exists():
            continue
        try:
            summary[Path(rel).stem] = _read_json(path)
        except Exception:
            continue
    return summary


def export_phase6g_kalmannet_tsp_package(
    *,
    source_run_dir: str | Path,
    package_dir: str | Path,
    overwrite: bool = False,
) -> Path:
    run_dir = Path(source_run_dir).expanduser().resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"source_run_dir not found: {run_dir}")
    if not run_dir.is_dir():
        raise ValueError(f"source_run_dir must be a directory: {run_dir}")

    config_path = run_dir / "config_snapshot.yaml"
    if not config_path.exists():
        raise FileNotFoundError(
            f"config_snapshot.yaml not found in source_run_dir: {config_path}"
        )
    run_cfg = _read_yaml(config_path)
    checkpoint_path = _discover_checkpoint(run_dir)
    model_config = _build_model_config_from_run(run_cfg)
    system_model = _build_system_model_from_run(run_cfg)
    training_summary = _build_training_summary_from_run(run_dir)

    with tempfile.TemporaryDirectory(prefix=".phase6g_kalmannet_", dir=str(Path(package_dir).expanduser().resolve().parent)) as tmp:
        staging = Path(tmp)
        model_config_path = staging / "model_config.json"
        system_model_path = staging / "system_model.json"
        training_summary_path = staging / "training_summary.json"
        model_config_path.write_text(
            json.dumps(model_config, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        system_model_path.write_text(
            json.dumps(system_model, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        training_summary_path.write_text(
            json.dumps(training_summary, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return export_kalmannet_tsp_replay_package(
            checkpoint=checkpoint_path,
            package_dir=package_dir,
            model_config=model_config_path,
            system_model=system_model_path,
            training_summary=training_summary_path,
            training_suite_name=str((run_cfg.get("suite", {}) or {}).get("name", "")),
            training_task_id=str((run_cfg.get("task", {}) or {}).get("task_id", "")),
            training_seed=int((run_cfg.get("seed", 0)) if isinstance(run_cfg.get("seed", 0), int) else 0),
            checkpoint_step=int((run_cfg.get("checkpoints", {}) or {}).get("best_step", 0)) if isinstance(run_cfg.get("checkpoints", {}), Mapping) else None,
            overwrite=bool(overwrite),
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export a replay package from a Phase 6G KalmanNet training run.")
    parser.add_argument("--source-run-dir", required=True)
    parser.add_argument("--package-dir", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    package = export_phase6g_kalmannet_tsp_package(
        source_run_dir=args.source_run_dir,
        package_dir=args.package_dir,
        overwrite=bool(args.overwrite),
    )
    print(f"wrote {package / 'replay_contract.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
