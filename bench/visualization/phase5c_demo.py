from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from .adcs_plots import make_adcs_plots
from .adcs_timeseries import build_adcs_timeseries
from .pred_artifact import save_pred_artifact
from .vizard_basilisk_wrapper import build_basilisk_vizard_offline_input
from .vizard_export import export_vizard_offline
from .vizard_native_bridge import (
    NATIVE_BRIDGE_MANIFEST_FILENAME,
    NATIVE_PLAYBACK_FILENAME,
    run_vizard_native_bridge,
)
from .vizard_phase5c_review import (
    REVIEW_MANIFEST_FILENAME,
    REVIEW_README_FILENAME,
    REVIEW_ZIP_FILENAME,
    build_phase5c_review_package,
    convert_frame_check_fixtures_to_native,
)


DEMO_SUMMARY_FILENAME = "phase5c_demo_summary.json"
DEMO_CONFIG_FILENAME = "phase5c_demo_config.json"
DEMO_TRAINING_TRACE_FILENAME = "toy_training_trace.json"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_name(f".{path.name}.tmp")
    try:
        tmp.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        tmp.replace(path)
    finally:
        if tmp.exists():
            tmp.unlink()


def _state_schema_meta() -> dict[str, Any]:
    return {
        "state_schema": {
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
            "gyro_bias": {
                "type": "rad_s",
                "name": "gyro_bias",
                "indices": [6, 7, 8],
                "optional": True,
            },
        },
        "attitude_convention": "MRP sigma_BN",
        "time_unit": "s",
        "time_source": "synthetic_uniform_demo",
        "dt_s": 0.5,
        "demo_only": True,
    }


def _synthetic_adcs(
    *,
    rng: np.random.Generator,
    n_seq: int,
    n_step: int,
    time_s: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    states = np.empty((n_seq, n_step, 9), dtype=np.float64)
    observations = np.empty_like(states)
    noise_scale = np.array(
        [0.004, 0.004, 0.004, 0.002, 0.002, 0.002, 0.0005, 0.0005, 0.0005],
        dtype=np.float64,
    )
    sensor_offset = np.array(
        [0.0015, -0.001, 0.0008, 0.0006, -0.0004, 0.0003, 0.0002, 0.0, -0.0001],
        dtype=np.float64,
    )
    for index in range(n_seq):
        phase = 0.3 * index + rng.uniform(-0.1, 0.1)
        sigma = np.stack(
            [
                0.06 * np.sin(0.22 * time_s + phase),
                0.04 * np.cos(0.17 * time_s + 0.5 * phase),
                0.05 * np.sin(0.13 * time_s + 0.2 + phase),
            ],
            axis=-1,
        )
        omega = np.stack(
            [
                0.018 * np.cos(0.22 * time_s + phase),
                -0.012 * np.sin(0.17 * time_s + 0.5 * phase),
                0.011 * np.cos(0.13 * time_s + 0.2 + phase),
            ],
            axis=-1,
        )
        bias_value = rng.normal(scale=0.0008, size=3)
        bias = np.broadcast_to(bias_value, (n_step, 3))
        state = np.concatenate([sigma, omega, bias], axis=-1)
        observation = (
            state
            + sensor_offset
            + rng.normal(scale=noise_scale, size=state.shape)
        )
        states[index] = state
        observations[index] = observation
    return states, observations


def _train_toy_bias_calibrator(
    x_train: np.ndarray,
    y_train: np.ndarray,
    *,
    max_train_steps: int,
) -> tuple[np.ndarray, list[dict[str, float]]]:
    correction = np.zeros(x_train.shape[-1], dtype=np.float64)
    target_correction = np.mean(
        x_train - y_train,
        axis=(0, 1),
    )
    trace: list[dict[str, float]] = []
    learning_rate = 0.6
    for step in range(max_train_steps):
        correction += learning_rate * (target_correction - correction)
        prediction = y_train + correction
        loss = float(np.mean(np.square(prediction - x_train)))
        trace.append(
            {
                "step": float(step + 1),
                "train_mse": loss,
                "correction_norm": float(np.linalg.norm(correction)),
            }
        )
    return correction, trace


def run_phase5c_tiny_demo(
    *,
    out_root: str | Path,
    device: str = "cpu",
    seed: int = 0,
    max_train_steps: int = 5,
    trajectory_id: int = 0,
    require_native_success: bool = False,
) -> Path:
    if str(device).lower() != "cpu":
        raise ValueError("Phase 5C tiny demo is CPU-only; device must be 'cpu'")
    if int(max_train_steps) <= 0:
        raise ValueError(
            f"max_train_steps must be positive, got {max_train_steps}"
        )
    if int(trajectory_id) not in (0, 1):
        raise ValueError("trajectory_id must be 0 or 1 for the tiny demo")

    root = Path(out_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    demo_run_dir = (
        root
        / f"phase5c_tiny_demo_seed_{int(seed)}_steps_{int(max_train_steps)}"
    )
    artifacts_dir = demo_run_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    try:
        rng = np.random.default_rng(int(seed))
        time_s = np.arange(24, dtype=np.float64) * 0.5
        x_train, y_train = _synthetic_adcs(
            rng=rng,
            n_seq=8,
            n_step=time_s.size,
            time_s=time_s,
        )
        x_test, y_test = _synthetic_adcs(
            rng=rng,
            n_seq=2,
            n_step=time_s.size,
            time_s=time_s,
        )
        correction, training_trace = _train_toy_bias_calibrator(
            x_train,
            y_train,
            max_train_steps=int(max_train_steps),
        )
        x_hat = y_test + correction
        test_mse = float(np.mean(np.square(x_hat - x_test)))
        test_rmse = float(np.sqrt(test_mse))
    except Exception as exc:
        raise RuntimeError(
            f"Phase 5C tiny training failed: {type(exc).__name__}: {exc}"
        ) from exc

    config = {
        "schema_version": "phase5c_demo_config_v1",
        "seed": int(seed),
        "device": "cpu",
        "max_train_steps": int(max_train_steps),
        "trajectory_id": int(trajectory_id),
        "train_shape": list(x_train.shape),
        "test_shape": list(x_test.shape),
        "model": "phase5c_toy_bias_calibrated_identity_estimator",
        "official_benchmark_run": False,
    }
    _write_json(artifacts_dir / DEMO_CONFIG_FILENAME, config)
    _write_json(
        artifacts_dir / DEMO_TRAINING_TRACE_FILENAME,
        {
            "schema_version": "phase5c_toy_training_trace_v1",
            "updates": training_trace,
            "learned_bias_correction": correction.tolist(),
        },
    )

    try:
        pred_path, pred_meta_path = save_pred_artifact(
            artifacts_dir,
            time_s=time_s,
            x_true=x_test,
            y_obs=y_test,
            x_hat=x_hat,
            trajectory_id=np.arange(x_test.shape[0], dtype=np.int64),
            meta=_state_schema_meta(),
        )
        timeseries_path, _ = build_adcs_timeseries(
            pred_path,
            pred_meta_path=pred_meta_path,
            trajectory_id=int(trajectory_id),
            out_dir=artifacts_dir,
        )
        plot_paths, _ = make_adcs_plots(
            timeseries_path,
            out_dir=artifacts_dir / "plots",
            trajectory_id=int(trajectory_id),
        )
        vizard_states_path, _ = export_vizard_offline(
            demo_run_dir,
            trajectory_id=int(trajectory_id),
            position_source="dummy_circular_orbit",
        )
        basilisk_input_path, _, _ = build_basilisk_vizard_offline_input(
            demo_run_dir
        )
        native_manifest_path, _ = run_vizard_native_bridge(
            demo_run_dir,
            mode="attempt-native",
            require_native_success=bool(require_native_success),
        )
    except Exception as exc:
        raise RuntimeError(
            f"Phase 5C demo artifact pipeline failed: {type(exc).__name__}: {exc}"
        ) from exc

    native_manifest = json.loads(
        native_manifest_path.read_text(encoding="utf-8")
    )
    native_status = str(native_manifest["native_conversion_status"])
    native_playback_path = native_manifest_path.parent / NATIVE_PLAYBACK_FILENAME
    if not native_playback_path.exists():
        native_playback: str | None = None
    else:
        native_playback = str(native_playback_path)
    if require_native_success:
        convert_frame_check_fixtures_to_native(
            artifacts_dir / "vizard" / "basilisk" / "frame_check",
            require_native_success=True,
        )

    review_dir = artifacts_dir / "vizard" / "phase5c_review"
    review_manifest_path = review_dir / REVIEW_MANIFEST_FILENAME
    review_readme_path = review_dir / REVIEW_README_FILENAME
    review_zip_path = review_dir / REVIEW_ZIP_FILENAME
    summary_path = artifacts_dir / DEMO_SUMMARY_FILENAME
    summary = {
        "schema_version": "phase5c_demo_summary_v1",
        "demo_run_dir": str(demo_run_dir),
        "seed": int(seed),
        "device": "cpu",
        "max_train_steps": int(max_train_steps),
        "model_or_adapter_used": (
            "phase5c_toy_bias_calibrated_identity_estimator"
        ),
        "dataset_description": (
            "Deterministic synthetic ADCS smoke data with "
            "[sigma_BN(3), omega_BN_B(3), gyro_bias(3)]; "
            "8 train and 2 test trajectories, T=24."
        ),
        "artifact_paths": {
            "preds_test": str(pred_path),
            "preds_test_meta": str(pred_meta_path),
            "adcs_timeseries": str(timeseries_path),
            "plots_dir": str(artifacts_dir / "plots"),
            "vizard_spacecraft_states": str(vizard_states_path),
            "dataFileToViz_input": str(basilisk_input_path),
            "native_bridge_manifest": str(native_manifest_path),
            "native_playback_bin": native_playback,
            "review_package_dir": str(review_dir),
            "review_manifest": str(review_manifest_path),
            "review_readme": str(review_readme_path),
            "review_bundle_zip": str(review_zip_path),
        },
        "plot_paths": [str(path) for path in plot_paths],
        "test_metric_summary": {
            "mse": test_mse,
            "rmse": test_rmse,
            "classification": "demo_diagnostic_only_not_official",
        },
        "native_conversion_status": native_status,
        "official_metrics_affected": False,
        "notes": (
            "This is a tiny smoke/demo run for visualization verification, "
            "not a scientific benchmark result."
        ),
    }
    _write_json(summary_path, summary)

    try:
        build_phase5c_review_package(demo_run_dir)
    except Exception as exc:
        raise RuntimeError(
            f"Phase 5C review packaging failed: {type(exc).__name__}: {exc}"
        ) from exc
    for required in (
        review_manifest_path,
        review_readme_path,
        review_zip_path,
    ):
        if not required.exists():
            raise RuntimeError(f"Phase 5C demo missing required output: {required}")
    return demo_run_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the tiny CPU-only Phase 5C visualization demo."
    )
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-train-steps", type=int, default=5)
    parser.add_argument("--trajectory-id", type=int, default=0)
    parser.add_argument("--require-native-success", action="store_true")
    args = parser.parse_args(argv)

    run_dir = run_phase5c_tiny_demo(
        out_root=args.out_root,
        device=args.device,
        seed=args.seed,
        max_train_steps=args.max_train_steps,
        trajectory_id=args.trajectory_id,
        require_native_success=args.require_native_success,
    )
    print(f"demo_run_dir: {run_dir}")
    print(f"summary: {run_dir / 'artifacts' / DEMO_SUMMARY_FILENAME}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
