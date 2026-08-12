from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .adcs_plots import ADCS_PLOT_MANIFEST_FILENAME, make_adcs_plots
from .adcs_timeseries import build_adcs_timeseries
from .basilisk_api_probe import BASILISK_API_PROBE_FILENAME
from .pred_artifact import (
    PRED_ARTIFACT_FILENAME,
    PRED_META_FILENAME,
    load_pred_artifact,
)
from .vizard_basilisk_wrapper import build_basilisk_vizard_offline_input
from .vizard_export import export_vizard_offline
from .vizard_native_bridge import (
    NATIVE_BRIDGE_LOG_FILENAME,
    NATIVE_BRIDGE_MANIFEST_FILENAME,
    NATIVE_LEGACY_INPUT_FILENAME,
    NATIVE_OUTPUT_MANIFEST_FILENAME,
    NATIVE_PLAYBACK_FILENAME,
    run_vizard_native_bridge,
)
from .vizard_phase5c_review import (
    REVIEW_ZIP_FILENAME,
    build_phase5c_review_package,
)


PHASE6C_SUMMARY_FILENAME = "phase6c_replay_visualization_summary.json"
_SUPPORTED_POSITION_SOURCES = ("fixed_origin", "dummy_circular_orbit")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON at {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _select_source_time(
    time_s: np.ndarray,
    trajectory_ids: np.ndarray,
    *,
    trajectory_id: int,
) -> tuple[np.ndarray, int]:
    ids = np.asarray(trajectory_ids)
    matches = np.flatnonzero(ids == int(trajectory_id))
    if matches.size == 0:
        raise ValueError(
            f"selected trajectory_id={trajectory_id} does not exist; "
            f"available IDs={ids.tolist()}"
        )
    if matches.size != 1:
        raise ValueError(
            f"trajectory_id={trajectory_id} is ambiguous; "
            f"matching positions={matches.tolist()}"
        )

    values = np.asarray(time_s)
    selected = values if values.ndim == 1 else values[int(matches[0])]
    selected = np.asarray(selected, dtype=np.float64)
    if selected.ndim != 1 or selected.size == 0:
        raise ValueError(
            f"selected time_s must be a non-empty [T] vector, got {selected.shape}"
        )
    if not np.isfinite(selected).all():
        raise ValueError("selected time_s contains NaN or Inf values")
    if selected.size > 1 and not np.all(np.diff(selected) > 0.0):
        raise ValueError("selected time_s must be strictly increasing")
    return selected, int(matches[0])


def _numeric_time_column(path: Path) -> np.ndarray:
    try:
        frame = pd.read_csv(path)
    except pd.errors.EmptyDataError as exc:
        raise ValueError(f"generated CSV is empty: {path}") from exc
    if "time_s" not in frame.columns:
        raise ValueError(f"generated CSV is missing time_s: {path}")
    try:
        values = pd.to_numeric(frame["time_s"], errors="raise").to_numpy(
            dtype=np.float64
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"generated CSV has non-numeric time_s: {path}") from exc
    if not np.isfinite(values).all():
        raise ValueError(f"generated CSV time_s contains NaN or Inf: {path}")
    return values


def _validate_timeline(
    path: Path,
    expected_time: np.ndarray,
    *,
    repetitions: int,
    stage: str,
) -> None:
    values = _numeric_time_column(path)
    expected = np.repeat(expected_time, int(repetitions))
    if values.shape != expected.shape or not np.allclose(
        values,
        expected,
        rtol=1.0e-7,
        atol=1.0e-8,
    ):
        raise RuntimeError(
            f"{stage} changed the selected timeline: "
            f"expected shape={expected.shape}, got={values.shape}"
        )


def _clear_managed_native_outputs(native_dir: Path) -> None:
    for filename in (
        BASILISK_API_PROBE_FILENAME,
        NATIVE_BRIDGE_MANIFEST_FILENAME,
        NATIVE_BRIDGE_LOG_FILENAME,
        NATIVE_LEGACY_INPUT_FILENAME,
        NATIVE_PLAYBACK_FILENAME,
        NATIVE_OUTPUT_MANIFEST_FILENAME,
    ):
        path = native_dir / filename
        if path.is_file():
            path.unlink()


def _validate_native_timeline(
    manifest_path: Path,
    expected_time: np.ndarray,
) -> None:
    manifest = _read_json(manifest_path)
    if int(manifest.get("num_timestamps", -1)) != int(expected_time.size):
        raise RuntimeError(
            "native conversion changed the number of timestamps: "
            f"expected={expected_time.size}, "
            f"got={manifest.get('num_timestamps')}"
        )
    source_offset = float(manifest.get("source_time_offset_s", np.nan))
    if not np.isclose(source_offset, expected_time[0]):
        raise RuntimeError(
            "native conversion source time offset does not match the selected "
            f"timeline: expected={expected_time[0]}, got={source_offset}"
        )
    if expected_time.size > 1:
        expected_dt = float(expected_time[1] - expected_time[0])
        native_dt = float(manifest.get("dt_s", np.nan))
        if not np.isclose(native_dt, expected_dt):
            raise RuntimeError(
                "native conversion sampling interval does not match the "
                f"selected timeline: expected={expected_dt}, got={native_dt}"
            )


def run_phase6c_replay_visualization(
    pred_run_dir: str | Path,
    *,
    trajectory_id: int = 0,
    position_source: str = "dummy_circular_orbit",
    vizard_convention: str | Path | None = None,
    require_native_success: bool = False,
    include_review_bundle: bool = True,
    create_zip: bool = True,
) -> Path:
    run_dir = Path(pred_run_dir).expanduser().resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(
            f"Phase 6B prediction run directory not found: {run_dir}"
        )
    source = str(position_source)
    if source not in _SUPPORTED_POSITION_SOURCES:
        raise ValueError(
            f"unsupported position_source={source!r}; "
            f"expected one of {_SUPPORTED_POSITION_SOURCES}"
        )

    artifacts_dir = run_dir / "artifacts"
    pred_path = artifacts_dir / PRED_ARTIFACT_FILENAME
    pred_meta_path = artifacts_dir / PRED_META_FILENAME
    if not pred_path.exists():
        raise FileNotFoundError(f"prediction artifact not found: {pred_path}")
    if not pred_meta_path.exists():
        raise FileNotFoundError(
            f"prediction artifact metadata not found: {pred_meta_path}"
        )

    prediction = load_pred_artifact(pred_path)
    x_true = np.asarray(prediction["x_true"])
    x_hat = np.asarray(prediction["x_hat"])
    y_obs = np.asarray(prediction["y_obs"])
    trajectory_ids = np.asarray(prediction["trajectory_id"])
    if x_true.shape != x_hat.shape:
        raise ValueError(
            "x_true and x_hat must have identical shape: "
            f"x_true={x_true.shape}, x_hat={x_hat.shape}"
        )
    selected_time, _ = _select_source_time(
        prediction["time_s"],
        trajectory_ids,
        trajectory_id=int(trajectory_id),
    )

    timeseries_csv, timeseries_meta = build_adcs_timeseries(
        pred_path,
        pred_meta_path=pred_meta_path,
        trajectory_id=int(trajectory_id),
        out_dir=artifacts_dir,
    )
    _validate_timeline(
        timeseries_csv,
        selected_time,
        repetitions=1,
        stage="Phase 2 ADCS timeseries",
    )

    plot_paths, plot_manifest = make_adcs_plots(
        timeseries_csv,
        out_dir=artifacts_dir / "plots",
        trajectory_id=int(trajectory_id),
    )

    vizard_csv, vizard_manifest = export_vizard_offline(
        run_dir,
        trajectory_id=int(trajectory_id),
        position_source=source,  # type: ignore[arg-type]
        vizard_convention=vizard_convention,
    )
    _validate_timeline(
        vizard_csv,
        selected_time,
        repetitions=2,
        stage="Phase 4 Vizard export",
    )

    basilisk_csv, basilisk_manifest, _ = (
        build_basilisk_vizard_offline_input(run_dir)
    )
    _validate_timeline(
        basilisk_csv,
        selected_time,
        repetitions=2,
        stage="Phase 5A Basilisk wrapper",
    )

    native_manifest_path: Path | None = None
    native_log_path: Path | None = None
    native_status = "not_attempted"
    native_error: str | None = None
    native_dir = artifacts_dir / "vizard" / "basilisk" / "native"
    native_dir.mkdir(parents=True, exist_ok=True)
    _clear_managed_native_outputs(native_dir)
    try:
        native_manifest_path, native_log_path = run_vizard_native_bridge(
            run_dir,
            mode="attempt-native",
            require_native_success=bool(require_native_success),
        )
        native_manifest = _read_json(native_manifest_path)
        native_status = str(
            native_manifest.get(
                "native_conversion_status",
                "unknown",
            )
        )
        native_error = native_manifest.get("native_conversion_error")
        if native_status == "attempted_success":
            _validate_native_timeline(
                native_dir / NATIVE_OUTPUT_MANIFEST_FILENAME,
                selected_time,
            )
    except Exception as exc:
        _clear_managed_native_outputs(native_dir)
        native_manifest_path = None
        native_log_path = None
        if require_native_success:
            raise RuntimeError(
                "Phase 6C native playback conversion did not succeed"
            ) from exc
        native_status = "orchestrator_native_bridge_failed"
        native_error = f"{type(exc).__name__}: {exc}"

    native_playback = native_dir / NATIVE_PLAYBACK_FILENAME
    native_playback_path = native_playback if native_playback.is_file() else None
    native_output_manifest = native_dir / NATIVE_OUTPUT_MANIFEST_FILENAME
    native_output_manifest_path = (
        native_output_manifest if native_output_manifest.is_file() else None
    )

    review_manifest_path: Path | None = None
    review_readme_path: Path | None = None
    review_zip_path: Path | None = None
    if include_review_bundle:
        try:
            review_manifest_path, review_readme_path = (
                build_phase5c_review_package(
                    run_dir,
                    include_plots=True,
                    include_native_playback=True,
                    include_frame_checks=True,
                    create_zip=bool(create_zip),
                )
            )
        except Exception as exc:
            raise RuntimeError(
                f"Phase 6C review package generation failed: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        candidate_zip = review_manifest_path.parent / REVIEW_ZIP_FILENAME
        if candidate_zip.is_file():
            review_zip_path = candidate_zip
        review_status = "created"
    else:
        review_status = "not_requested"

    prediction_meta = prediction["meta"]
    notes = [
        (
            "Phase 6C orchestrates existing visualization artifacts from a "
            "Phase 6B prediction run."
        ),
        (
            "Phase 6C does not run model inference or alter benchmark metrics."
        ),
    ]
    if prediction_meta.get("model_id") == "replay_identity_baseline":
        notes.append(
            "The input prediction artifact was generated by "
            "replay_identity_baseline and is for visualization pipeline "
            "validation only."
        )
    if native_error:
        notes.append(f"Native conversion detail: {native_error}")

    summary_path = artifacts_dir / PHASE6C_SUMMARY_FILENAME
    summary = {
        "schema_version": "phase6c_replay_visualization_summary_v1",
        "pred_run_dir": str(run_dir),
        "input_pred_npz": str(pred_path),
        "input_pred_meta": str(pred_meta_path),
        "trajectory_id": int(trajectory_id),
        "position_source": source,
        "require_native_success": bool(require_native_success),
        "include_review_bundle": bool(include_review_bundle),
        "create_zip": bool(create_zip),
        "time_range_s": {
            "start": float(selected_time[0]),
            "end": float(selected_time[-1]),
            "duration": float(selected_time[-1] - selected_time[0]),
        },
        "num_timestamps": int(selected_time.size),
        "trajectory_count": int(trajectory_ids.size),
        "state_dim": int(x_true.shape[-1]),
        "measurement_dim": int(y_obs.shape[-1]),
        "model_id": prediction_meta.get("model_id"),
        "scenario_id": prediction_meta.get("scenario_id"),
        "suite_name": prediction_meta.get("suite_name"),
        "task_id": prediction_meta.get("task_id"),
        "seed": prediction_meta.get("seed"),
        "generated_artifacts": {
            "adcs_timeseries_csv": str(timeseries_csv),
            "adcs_timeseries_meta": str(timeseries_meta),
            "plots_dir": str(plot_manifest.parent),
            "plot_files": [str(path) for path in plot_paths],
            "adcs_plot_manifest": str(plot_manifest),
            "vizard_spacecraft_states_csv": str(vizard_csv),
            "vizard_export_manifest": str(vizard_manifest),
            "dataFileToViz_input_csv": str(basilisk_csv),
            "dataFileToViz_input_manifest": str(basilisk_manifest),
            "native_bridge_manifest": (
                None
                if native_manifest_path is None
                else str(native_manifest_path)
            ),
            "native_bridge_log": (
                None if native_log_path is None else str(native_log_path)
            ),
            "native_playback_bin": (
                None
                if native_playback_path is None
                else str(native_playback_path)
            ),
            "native_conversion_output_manifest": (
                None
                if native_output_manifest_path is None
                else str(native_output_manifest_path)
            ),
            "vizard_convention_file": (
                None
                if vizard_convention is None
                else str(Path(vizard_convention).expanduser().resolve())
            ),
            "review_package_dir": (
                None
                if review_manifest_path is None
                else str(review_manifest_path.parent)
            ),
            "review_manifest": (
                None
                if review_manifest_path is None
                else str(review_manifest_path)
            ),
            "review_readme": (
                None
                if review_readme_path is None
                else str(review_readme_path)
            ),
            "review_bundle_zip": (
                None if review_zip_path is None else str(review_zip_path)
            ),
        },
        "timeline_validation": {
            "adcs_timeseries": "preserved",
            "vizard_spacecraft_states": "preserved",
            "dataFileToViz_input": "preserved",
            "native_playback": (
                "preserved"
                if native_status == "attempted_success"
                else "not_generated"
            ),
        },
        "native_conversion_status": native_status,
        "review_package_status": review_status,
        "official_metrics_affected": False,
        "notes": " ".join(notes),
    }
    _write_json_atomic(summary_path, summary)
    return summary_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run the Phase 2-5C replay visualization pipeline from a "
            "Phase 6B prediction run directory."
        )
    )
    parser.add_argument("--pred-run-dir", required=True)
    parser.add_argument("--trajectory-id", type=int, default=0)
    parser.add_argument(
        "--position-source",
        choices=_SUPPORTED_POSITION_SOURCES,
        default="dummy_circular_orbit",
    )
    parser.add_argument("--vizard-convention", default=None)
    parser.add_argument("--require-native-success", action="store_true")
    parser.add_argument("--no-review-bundle", action="store_true")
    parser.add_argument("--no-zip", action="store_true")
    args = parser.parse_args(argv)

    summary_path = run_phase6c_replay_visualization(
        args.pred_run_dir,
        trajectory_id=args.trajectory_id,
        position_source=args.position_source,
        vizard_convention=args.vizard_convention,
        require_native_success=bool(args.require_native_success),
        include_review_bundle=not bool(args.no_review_bundle),
        create_zip=not bool(args.no_zip),
    )
    print(f"wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
