from __future__ import annotations

import argparse
import json
import tempfile
import traceback
from pathlib import Path
from typing import Any, Literal, Optional

import numpy as np
import pandas as pd

from .basilisk_api_probe import load_probe, probe_basilisk_vizard_api
from .vizard_basilisk_wrapper import BASILISK_INPUT_FILENAME
from .vizard_frame_checks import generate_frame_check_fixtures


NativeBridgeMode = Literal["probe-only", "attempt-native"]

NATIVE_BRIDGE_MANIFEST_FILENAME = "native_bridge_manifest.json"
NATIVE_BRIDGE_LOG_FILENAME = "native_bridge_log.txt"
NATIVE_LEGACY_INPUT_FILENAME = "native_dataFileToViz_input.csv"
NATIVE_PLAYBACK_FILENAME = "vizard_playback.bin"
NATIVE_OUTPUT_MANIFEST_FILENAME = "native_conversion_output_manifest.json"

_SUPPORTED_MODES = ("probe-only", "attempt-native")
_STRING_COLUMNS = ("sc_name", "attitude_type")
_NUMERIC_COLUMNS = (
    "time_s",
    "r_BN_N_x_m",
    "r_BN_N_y_m",
    "r_BN_N_z_m",
    "v_BN_N_x_m_s",
    "v_BN_N_y_m_s",
    "v_BN_N_z_m_s",
    "sigma_BN_1",
    "sigma_BN_2",
    "sigma_BN_3",
    "omega_BN_B_x_rad_s",
    "omega_BN_B_y_rad_s",
    "omega_BN_B_z_rad_s",
)
_REQUIRED_COLUMNS = ("time_s", "sc_name", "attitude_type") + tuple(
    column for column in _NUMERIC_COLUMNS if column != "time_s"
)


def _load_native_input(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Basilisk wrapper input CSV not found: {path}")
    try:
        frame = pd.read_csv(path)
    except pd.errors.EmptyDataError as exc:
        raise ValueError(f"Basilisk wrapper input CSV is empty: {path}") from exc
    if frame.empty:
        raise ValueError(f"Basilisk wrapper input CSV contains no rows: {path}")

    missing = [column for column in _REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(
            f"{path} is missing required native bridge columns: {missing}"
        )
    for column in _NUMERIC_COLUMNS:
        try:
            frame[column] = pd.to_numeric(frame[column], errors="raise")
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"native bridge column {column!r} must be numeric"
            ) from exc
        if not np.isfinite(
            frame[column].to_numpy(dtype=np.float64, copy=False)
        ).all():
            raise ValueError(
                f"native bridge column {column!r} contains NaN or Inf"
            )

    for column in _STRING_COLUMNS:
        if frame[column].isna().any():
            raise ValueError(f"native bridge column {column!r} contains missing values")
        frame[column] = frame[column].astype(str).str.strip()
        if frame[column].eq("").any():
            raise ValueError(f"native bridge column {column!r} contains empty values")
    if not frame["attitude_type"].eq("MRP").all():
        invalid = sorted(frame.loc[frame["attitude_type"] != "MRP", "attitude_type"].unique())
        raise ValueError(
            "attitude_type must be 'MRP' for every Phase 5B row, "
            f"got={invalid}"
        )

    duplicate = frame.duplicated(subset=["time_s", "sc_name"], keep=False)
    if duplicate.any():
        pairs = (
            frame.loc[duplicate, ["time_s", "sc_name"]]
            .drop_duplicates()
            .to_dict(orient="records")
        )
        raise ValueError(f"duplicate (time_s, sc_name) pairs are invalid: {pairs}")
    return frame.sort_values(["time_s", "sc_name"], kind="stable").reset_index(
        drop=True
    )


def _legacy_contract_data(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str], np.ndarray, float]:
    spacecraft_names = sorted(frame["sc_name"].unique().tolist())
    times = np.sort(frame["time_s"].unique().astype(np.float64))
    if times.size < 2:
        raise ValueError("native DataFileToViz conversion requires at least two timestamps")
    expected_rows = int(times.size * len(spacecraft_names))
    if len(frame) != expected_rows:
        raise ValueError(
            "native DataFileToViz conversion requires a complete rectangular "
            f"time/spacecraft grid: expected={expected_rows}, got={len(frame)}"
        )
    counts = frame.groupby("sc_name", sort=False)["time_s"].nunique()
    if not counts.eq(times.size).all():
        raise ValueError(
            "native DataFileToViz conversion requires every spacecraft at every timestamp"
        )
    deltas = np.diff(times)
    dt_s = float(deltas[0])
    if dt_s <= 0.0 or not np.allclose(deltas, dt_s, rtol=1.0e-9, atol=1.0e-12):
        raise ValueError(
            "native DataFileToViz conversion requires uniformly spaced timestamps"
        )

    columns = ["time_s"]
    for name in spacecraft_names:
        columns.extend(
            [
                f"{name}_r_BN_N_x_m",
                f"{name}_r_BN_N_y_m",
                f"{name}_r_BN_N_z_m",
                f"{name}_v_BN_N_x_m_s",
                f"{name}_v_BN_N_y_m_s",
                f"{name}_v_BN_N_z_m_s",
                f"{name}_sigma_BN_1",
                f"{name}_sigma_BN_2",
                f"{name}_sigma_BN_3",
                f"{name}_omega_BN_B_x_rad_s",
                f"{name}_omega_BN_B_y_rad_s",
                f"{name}_omega_BN_B_z_rad_s",
            ]
        )

    rows: list[list[float]] = []
    state_columns = [
        "r_BN_N_x_m",
        "r_BN_N_y_m",
        "r_BN_N_z_m",
        "v_BN_N_x_m_s",
        "v_BN_N_y_m_s",
        "v_BN_N_z_m_s",
        "sigma_BN_1",
        "sigma_BN_2",
        "sigma_BN_3",
        "omega_BN_B_x_rad_s",
        "omega_BN_B_y_rad_s",
        "omega_BN_B_z_rad_s",
    ]
    time_offset = float(times[0])
    for time_value in times:
        row = [float(time_value - time_offset)]
        for name in spacecraft_names:
            selected = frame.loc[
                (frame["time_s"] == time_value) & (frame["sc_name"] == name),
                state_columns,
            ]
            if len(selected) != 1:
                raise ValueError(
                    f"expected one row for time_s={time_value}, sc_name={name!r}"
                )
            row.extend(selected.iloc[0].to_numpy(dtype=np.float64).tolist())
        rows.append(row)
    return pd.DataFrame(rows, columns=columns), spacecraft_names, times, dt_s


def _attempt_native_conversion(
    frame: pd.DataFrame,
    staging_dir: Path,
    *,
    basilisk_version: Optional[str],
) -> tuple[list[Path], dict[str, Any]]:
    from Basilisk.simulation import dataFileToViz, vizInterface
    from Basilisk.utilities import SimulationBaseClass, macros

    legacy, spacecraft_names, times, dt_s = _legacy_contract_data(frame)
    legacy_path = staging_dir / NATIVE_LEGACY_INPUT_FILENAME
    playback_path = staging_dir / NATIVE_PLAYBACK_FILENAME
    output_manifest_path = staging_dir / NATIVE_OUTPUT_MANIFEST_FILENAME
    legacy.to_csv(legacy_path, index=False)

    simulation = SimulationBaseClass.SimBaseClass()
    process = simulation.CreateNewProcess("vizardNativeBridgeProcess")
    task_name = "vizardNativeBridgeTask"
    process.addTask(
        simulation.CreateNewTask(task_name, macros.sec2nano(dt_s))
    )

    reader = dataFileToViz.DataFileToViz()
    reader.ModelTag = "dataFileToViz"
    reader.setNumOfSatellites(len(spacecraft_names))
    reader.attitudeType = 0
    reader.dataFileName = str(legacy_path)
    reader.delimiter = ","
    reader.headerLine = True
    reader.convertPosToMeters = 1.0
    simulation.AddModelToTask(task_name, reader)

    viz = vizInterface.VizInterface()
    viz.ModelTag = "vizInterface"
    viz.settings = vizInterface.VizSettings()
    viz.saveFile = True
    viz.protoFilename = str(playback_path)
    viz.liveStream = False
    viz.broadcastStream = False
    viz.scData.clear()
    for index, name in enumerate(spacecraft_names):
        sc_data = vizInterface.VizSpacecraftData()
        sc_data.spacecraftName = name
        sc_data.scStateInMsg.subscribeTo(reader.scStateOutMsgs[index])
        viz.scData.push_back(sc_data)
    simulation.AddModelToTask(task_name, viz)

    simulation.InitializeSimulation()
    simulation.ConfigureStopTime(
        macros.sec2nano(float(times[-1] - times[0]))
    )
    simulation.ExecuteSimulation()
    if not playback_path.exists() or playback_path.stat().st_size <= 0:
        raise RuntimeError("Basilisk vizInterface did not produce a playback file")

    output_manifest = {
        "schema_version": "native_conversion_output_v1",
        "basilisk_version": basilisk_version,
        "legacy_input_csv": NATIVE_LEGACY_INPUT_FILENAME,
        "playback_file": NATIVE_PLAYBACK_FILENAME,
        "spacecraft_names": spacecraft_names,
        "num_timestamps": int(times.size),
        "dt_s": dt_s,
        "source_time_offset_s": float(times[0]),
        "attitude_type": 0,
        "attitude_representation": "MRP sigma_BN",
        "convert_position_to_meters": 1.0,
        "live_stream": False,
        "broadcast_stream": False,
    }
    output_manifest_path.write_text(
        json.dumps(output_manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return (
        [legacy_path, playback_path, output_manifest_path],
        output_manifest,
    )


def _log_text(
    *,
    input_csv: Path,
    mode: str,
    probe: dict[str, Any],
    status: str,
    error: Optional[str],
    traceback_text: Optional[str],
    generated_outputs: list[str],
) -> str:
    lines = [
        "Native Basilisk/Vizard Bridge Log",
        f"input_csv: {input_csv}",
        f"mode: {mode}",
        f"basilisk_available: {probe.get('basilisk_available')}",
        f"basilisk_version: {probe.get('basilisk_version')}",
        (
            "native_contract_discoverable: "
            f"{probe.get('native_contract_discoverable')}"
        ),
        f"native_conversion_status: {status}",
        f"native_conversion_error: {error or 'none'}",
        "generated_outputs:",
    ]
    lines.extend(f"- {path}" for path in generated_outputs)
    if traceback_text:
        lines.extend(["conversion_traceback:", traceback_text.rstrip()])
    lines.extend(
        [
            "next_manual_steps:",
            "- Open the generated .bin with a compatible Vizard installation.",
            "- Run the frame-check fixtures and verify MRP yaw sign/direction.",
            "- Confirm SC_true and SC_estimated naming/orientation mapping.",
            "- Do not use this path for official benchmark metrics.",
        ]
    )
    return "\n".join(lines) + "\n"


def run_vizard_native_bridge(
    run_dir: str | Path | None = None,
    *,
    mode: NativeBridgeMode = "probe-only",
    out_dir: str | Path | None = None,
    input_csv: str | Path | None = None,
    require_basilisk: bool = False,
    require_native_success: bool = False,
) -> tuple[Path, Path]:
    bridge_mode = str(mode)
    if bridge_mode not in _SUPPORTED_MODES:
        raise ValueError(
            f"unsupported mode={bridge_mode!r}; expected one of {_SUPPORTED_MODES}"
        )
    resolved_run_dir = (
        None if run_dir is None else Path(run_dir).expanduser().resolve()
    )
    if input_csv is None:
        if resolved_run_dir is None:
            raise ValueError("run_dir or input_csv is required")
        source_csv = (
            resolved_run_dir
            / "artifacts"
            / "vizard"
            / "basilisk"
            / BASILISK_INPUT_FILENAME
        )
    else:
        source_csv = Path(input_csv).expanduser().resolve()

    if out_dir is None:
        if resolved_run_dir is None:
            raise ValueError("out_dir is required when using input_csv directly")
        output_dir = (
            resolved_run_dir
            / "artifacts"
            / "vizard"
            / "basilisk"
            / "native"
        )
        frame_check_dir = output_dir.parent / "frame_check"
    else:
        output_dir = Path(out_dir).expanduser().resolve()
        frame_check_dir = output_dir.parent / "frame_check"

    frame = _load_native_input(source_csv)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame_paths, frame_readme = generate_frame_check_fixtures(frame_check_dir)
    probe_path = probe_basilisk_vizard_api(output_dir, require_basilisk=False)
    probe = load_probe(probe_path)

    status = "not_attempted_probe_only"
    error: Optional[str] = None
    attempted = False
    traceback_text: Optional[str] = None
    generated_outputs: list[str] = [str(path) for path in frame_paths]
    generated_outputs.extend([str(frame_readme), str(probe_path)])

    if bridge_mode == "attempt-native":
        if not probe["basilisk_available"]:
            status = "not_attempted_basilisk_unavailable"
            error = "Basilisk imports are unavailable; see API probe errors."
        elif not probe.get("native_contract_discoverable", False):
            status = "not_attempted_contract_unknown"
            error = (
                "The installed API and local scenarioDataToViz contract could "
                "not both be discovered."
            )
        else:
            attempted = True
            try:
                with tempfile.TemporaryDirectory(
                    prefix=".native_conversion_",
                    dir=output_dir,
                ) as tmp:
                    staging_dir = Path(tmp)
                    staged_outputs, _ = _attempt_native_conversion(
                        frame,
                        staging_dir,
                        basilisk_version=probe.get("basilisk_version"),
                    )
                    for staged in staged_outputs:
                        final = output_dir / staged.name
                        staged.replace(final)
                        generated_outputs.append(str(final))
                status = "attempted_success"
            except Exception as exc:
                status = "attempted_failed"
                error = f"{type(exc).__name__}: {exc}"
                traceback_text = traceback.format_exc()

    if require_basilisk and not probe["basilisk_available"]:
        error = error or "Basilisk is required but unavailable."

    manifest_path = output_dir / NATIVE_BRIDGE_MANIFEST_FILENAME
    log_path = output_dir / NATIVE_BRIDGE_LOG_FILENAME
    manifest = {
        "schema_version": "vizard_native_bridge_v1",
        "input_csv": str(source_csv),
        "run_dir": (
            None if resolved_run_dir is None else str(resolved_run_dir)
        ),
        "out_dir": str(output_dir),
        "probe_json": str(probe_path),
        "mode": bridge_mode,
        "basilisk_available": bool(probe["basilisk_available"]),
        "basilisk_version": probe.get("basilisk_version"),
        "native_contract_discoverable": bool(
            probe.get("native_contract_discoverable", False)
        ),
        "native_conversion_attempted": attempted,
        "native_conversion_status": status,
        "native_conversion_error": error,
        "native_conversion_traceback": traceback_text,
        "generated_outputs": generated_outputs,
        "num_rows": int(len(frame)),
        "spacecraft_names": sorted(frame["sc_name"].unique().tolist()),
        "time_unit": "s",
        "attitude_representation": "MRP sigma_BN",
        "angular_rate_representation": "omega_BN_B rad/s",
        "official_metrics_affected": False,
        "notes": (
            "Probe-first native Basilisk/Vizard offline bridge. Live streaming "
            "and online inference are not used."
        ),
    }
    log_text = _log_text(
        input_csv=source_csv,
        mode=bridge_mode,
        probe=probe,
        status=status,
        error=error,
        traceback_text=traceback_text,
        generated_outputs=generated_outputs,
    )
    with tempfile.TemporaryDirectory(
        prefix=".native_bridge_meta_",
        dir=output_dir,
    ) as tmp:
        staging_dir = Path(tmp)
        staged_manifest = staging_dir / NATIVE_BRIDGE_MANIFEST_FILENAME
        staged_log = staging_dir / NATIVE_BRIDGE_LOG_FILENAME
        staged_manifest.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        staged_log.write_text(log_text, encoding="utf-8")
        staged_manifest.replace(manifest_path)
        staged_log.replace(log_path)

    if require_basilisk and not probe["basilisk_available"]:
        raise RuntimeError(
            f"Basilisk is required but unavailable; see {manifest_path}"
        )
    if require_native_success and status != "attempted_success":
        raise RuntimeError(
            "native Basilisk/Vizard conversion did not succeed "
            f"(status={status}); see {manifest_path}"
        )
    return manifest_path, log_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Probe and optionally run the native Basilisk/Vizard offline bridge."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--run-dir")
    source.add_argument("--input-csv")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--mode", choices=_SUPPORTED_MODES, default="probe-only")
    parser.add_argument("--require-basilisk", action="store_true")
    parser.add_argument("--require-native-success", action="store_true")
    args = parser.parse_args(argv)
    if args.input_csv is not None and args.out_dir is None:
        parser.error("--out-dir is required with --input-csv")

    manifest_path, log_path = run_vizard_native_bridge(
        args.run_dir,
        mode=args.mode,
        out_dir=args.out_dir,
        input_csv=args.input_csv,
        require_basilisk=args.require_basilisk,
        require_native_success=args.require_native_success,
    )
    print(f"wrote {manifest_path}")
    print(f"wrote {log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
