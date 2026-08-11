from __future__ import annotations

import argparse
import importlib
import json
import tempfile
from pathlib import Path
from typing import Any, Literal, Mapping, Optional

import numpy as np
import pandas as pd

from .vizard_export import VIZARD_MANIFEST_FILENAME, VIZARD_STATES_FILENAME


WrapperMode = Literal["data-file"]

BASILISK_INPUT_FILENAME = "dataFileToViz_input.csv"
BASILISK_INPUT_MANIFEST_FILENAME = "dataFileToViz_input_manifest.json"
MANUAL_CHECK_FILENAME = "README_manual_vizard_check.md"

_SUPPORTED_MODES = ("data-file",)
_STRING_COLUMNS = ("sc_name", "source")
_NUMERIC_COLUMNS = (
    "time_s",
    "traj_id",
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
_REQUIRED_COLUMNS = ("time_s", "traj_id", "sc_name", "source") + tuple(
    column for column in _NUMERIC_COLUMNS if column not in {"time_s", "traj_id"}
)
_OUTPUT_COLUMNS = (
    "time_s",
    "sc_name",
    "r_BN_N_x_m",
    "r_BN_N_y_m",
    "r_BN_N_z_m",
    "v_BN_N_x_m_s",
    "v_BN_N_y_m_s",
    "v_BN_N_z_m_s",
    "attitude_type",
    "sigma_BN_1",
    "sigma_BN_2",
    "sigma_BN_3",
    "omega_BN_B_x_rad_s",
    "omega_BN_B_y_rad_s",
    "omega_BN_B_z_rad_s",
)


def _detect_basilisk() -> tuple[bool, Optional[str], Optional[str]]:
    try:
        basilisk = importlib.import_module("Basilisk")
        data_file_module = importlib.import_module(
            "Basilisk.simulation.dataFileToViz"
        )
        viz_module = importlib.import_module("Basilisk.simulation.vizInterface")
        if not hasattr(data_file_module, "DataFileToViz"):
            return False, None, "Basilisk dataFileToViz module lacks DataFileToViz"
        if not hasattr(viz_module, "VizInterface"):
            return False, None, "Basilisk vizInterface module lacks VizInterface"
        version = getattr(basilisk, "__version__", None)
        return True, (None if version is None else str(version)), None
    except Exception as exc:
        return False, None, f"{type(exc).__name__}: {exc}"


def detect_basilisk_available() -> bool:
    """Return whether Basilisk dataFileToViz and vizInterface import safely."""
    available, _, _ = _detect_basilisk()
    return available


def _read_phase4_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in Phase 4 Vizard manifest: {path}") from exc
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must contain a JSON object")
    return dict(value)


def _load_vizard_states(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Vizard spacecraft states CSV not found: {path}")
    try:
        frame = pd.read_csv(path)
    except pd.errors.EmptyDataError as exc:
        raise ValueError(f"Vizard spacecraft states CSV is empty: {path}") from exc
    if frame.empty:
        raise ValueError(f"Vizard spacecraft states CSV contains no rows: {path}")

    missing = [column for column in _REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(
            f"{path} is missing required Basilisk wrapper columns: {missing}"
        )

    for column in _NUMERIC_COLUMNS:
        try:
            frame[column] = pd.to_numeric(frame[column], errors="raise")
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Vizard spacecraft state column {column!r} must be numeric"
            ) from exc
        values = frame[column].to_numpy(dtype=np.float64, copy=False)
        if not np.isfinite(values).all():
            raise ValueError(
                f"Vizard spacecraft state column {column!r} contains NaN or Inf"
            )

    for column in _STRING_COLUMNS:
        if frame[column].isna().any():
            raise ValueError(
                f"Vizard spacecraft state column {column!r} contains missing values"
            )
        values = frame[column].astype(str).str.strip()
        if values.eq("").any():
            raise ValueError(
                f"Vizard spacecraft state column {column!r} contains empty values"
            )
        frame[column] = values

    duplicate = frame.duplicated(subset=["time_s", "sc_name"], keep=False)
    if duplicate.any():
        pairs = (
            frame.loc[duplicate, ["time_s", "sc_name"]]
            .drop_duplicates()
            .to_dict(orient="records")
        )
        raise ValueError(
            "duplicate (time_s, sc_name) pairs are invalid for Basilisk wrapper "
            f"input: {pairs}"
        )

    frame["_input_order"] = np.arange(len(frame), dtype=np.int64)
    return frame.sort_values(
        ["time_s", "_input_order"],
        kind="stable",
    ).reset_index(drop=True)


def _build_output(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.loc[
        :,
        [column for column in _OUTPUT_COLUMNS if column != "attitude_type"],
    ].copy()
    output.insert(8, "attitude_type", "MRP")
    output = output.loc[:, _OUTPUT_COLUMNS]
    if len(output) != len(frame):
        raise ValueError(
            f"generated wrapper row count={len(output)} does not match input={len(frame)}"
        )
    return output


def _manual_check_text(
    *,
    input_csv: Path,
    output_csv: Path,
    manifest_path: Path,
    spacecraft_names: list[str],
    position_source: str,
    basilisk_available: bool,
) -> str:
    spacecraft = "\n".join(f"- `{name}`" for name in spacecraft_names)
    return f"""# Manual Basilisk/Vizard Check

## Generated Files

- Source Phase 4 timeline: `{input_csv}`
- Benchmark-owned Basilisk bridge CSV: `{output_csv}`
- Wrapper manifest: `{manifest_path}`

This phase does not launch Basilisk, Vizard, Unity, or a live socket connection.
`dataFileToViz_input.csv` is an offline bridge artifact for a later Basilisk
`dataFileToViz` or `vizInterface` conversion wrapper.

## Spacecraft

{spacecraft}

The Phase 4 MVP normally provides `SC_true` and `SC_estimated`.

## Frame And Unit Conventions

- `time_s`: seconds
- `r_BN_N`: spacecraft position in the inertial N frame, meters
- `v_BN_N`: spacecraft velocity in the inertial N frame, meters/second
- `sigma_BN`: modified Rodrigues parameters
- `omega_BN_B`: body angular rate, rad/s
- Position source recorded by Phase 4: `{position_source}`
- Basilisk import available while generating this file: `{str(basilisk_available).lower()}`

## Manual Verification Cases

1. A zero MRP attitude should align body and inertial frame axes.
2. A small yaw-like MRP rotation should rotate in the expected direction.
3. `SC_true` and `SC_estimated` should show different orientations when their
   MRP values differ.
4. Confirm the position source is visually appropriate. `fixed_origin` is for
   attitude-only debugging and `dummy_circular_orbit` is synthetic.

## Known Limitations

- True-orbit playback depends on standardized scenario position/velocity metadata.
- This CSV is not claimed to be Basilisk's native legacy text format.
- Direct `dataFileToViz` conversion is deferred until its exact versioned loader
  contract is implemented and verified.
- Vizard launch and live streaming are deferred.
- Final MRP sign and frame conventions must be checked visually once Vizard is connected.
"""


def _validate_staged_output(
    csv_path: Path,
    *,
    expected_rows: int,
) -> None:
    generated = pd.read_csv(csv_path)
    if list(generated.columns) != list(_OUTPUT_COLUMNS):
        raise ValueError(
            f"generated Basilisk wrapper columns are invalid: {list(generated.columns)}"
        )
    if len(generated) != expected_rows:
        raise ValueError(
            f"generated Basilisk wrapper row count={len(generated)}, "
            f"expected={expected_rows}"
        )
    if not generated["attitude_type"].eq("MRP").all():
        raise ValueError("generated attitude_type must be 'MRP' for every row")


def build_basilisk_vizard_offline_input(
    run_dir: str | Path | None = None,
    *,
    mode: WrapperMode = "data-file",
    out_dir: str | Path | None = None,
    input_csv: str | Path | None = None,
    require_basilisk: bool = False,
) -> tuple[Path, Path, Path]:
    """
    Build the Phase 5A offline bridge without launching Basilisk or Vizard.

    Basilisk availability is detected lazily. Even when installed, native
    conversion is deliberately not attempted because this phase standardizes
    the benchmark-owned input contract only.
    """
    wrapper_mode = str(mode)
    if wrapper_mode not in _SUPPORTED_MODES:
        raise ValueError(
            f"unsupported mode={wrapper_mode!r}; expected one of {_SUPPORTED_MODES}"
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
            / VIZARD_STATES_FILENAME
        )
    else:
        source_csv = Path(input_csv).expanduser().resolve()

    if out_dir is None:
        if resolved_run_dir is not None:
            output_dir = (
                resolved_run_dir / "artifacts" / "vizard" / "basilisk"
            )
        else:
            raise ValueError("out_dir is required when using input_csv directly")
    else:
        output_dir = Path(out_dir).expanduser().resolve()

    frame = _load_vizard_states(source_csv)
    phase4_manifest_path = source_csv.with_name(VIZARD_MANIFEST_FILENAME)
    phase4_manifest = _read_phase4_manifest(phase4_manifest_path)
    position_source = str(phase4_manifest.get("position_source", "unknown"))

    basilisk_available, basilisk_version, basilisk_error = _detect_basilisk()
    if require_basilisk and not basilisk_available:
        detail = f": {basilisk_error}" if basilisk_error else ""
        raise RuntimeError(
            "Basilisk dataFileToViz/vizInterface is unavailable"
            f"{detail}"
        )

    output = _build_output(frame)
    spacecraft_names = frame["sc_name"].drop_duplicates().tolist()
    sources = frame["source"].drop_duplicates().tolist()
    spacecraft_sources = (
        frame.loc[:, ["sc_name", "source"]]
        .drop_duplicates()
        .to_dict(orient="records")
    )

    output_csv = output_dir / BASILISK_INPUT_FILENAME
    manifest_path = output_dir / BASILISK_INPUT_MANIFEST_FILENAME
    readme_path = output_dir / MANUAL_CHECK_FILENAME
    manifest: dict[str, Any] = {
        "schema_version": "basilisk_vizard_offline_input_v1",
        "input_csv": str(source_csv),
        "input_phase4_manifest": (
            str(phase4_manifest_path) if phase4_manifest_path.exists() else None
        ),
        "output_csv": str(output_csv),
        "run_dir": (
            None if resolved_run_dir is None else str(resolved_run_dir)
        ),
        "mode": wrapper_mode,
        "num_rows": int(len(output)),
        "num_spacecraft": int(len(spacecraft_names)),
        "spacecraft_names": spacecraft_names,
        "spacecraft_sources": spacecraft_sources,
        "sources": sources,
        "time_unit": "s",
        "position_frame": "N",
        "velocity_frame": "N",
        "attitude_representation": "MRP sigma_BN",
        "angular_rate_representation": "omega_BN_B rad/s",
        "position_source": position_source,
        "basilisk_required": bool(require_basilisk),
        "basilisk_available": bool(basilisk_available),
        "basilisk_version": basilisk_version,
        "basilisk_detection_error": basilisk_error,
        "basilisk_conversion_attempted": False,
        "basilisk_conversion_status": "not_attempted",
        "official_metrics_affected": False,
        "manual_verification_readme": str(readme_path),
        "notes": (
            "Offline Basilisk/Vizard wrapper input generated from Phase 4 "
            "Vizard export artifact."
        ),
    }
    readme = _manual_check_text(
        input_csv=source_csv,
        output_csv=output_csv,
        manifest_path=manifest_path,
        spacecraft_names=spacecraft_names,
        position_source=position_source,
        basilisk_available=basilisk_available,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".basilisk_wrapper_",
        dir=output_dir,
    ) as tmp:
        staging_dir = Path(tmp)
        staged_csv = staging_dir / BASILISK_INPUT_FILENAME
        staged_manifest = staging_dir / BASILISK_INPUT_MANIFEST_FILENAME
        staged_readme = staging_dir / MANUAL_CHECK_FILENAME
        output.to_csv(staged_csv, index=False)
        _validate_staged_output(staged_csv, expected_rows=len(output))
        staged_manifest.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        staged_readme.write_text(readme, encoding="utf-8")
        staged_csv.replace(output_csv)
        staged_manifest.replace(manifest_path)
        staged_readme.replace(readme_path)

    return output_csv, manifest_path, readme_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build an offline Basilisk/Vizard data-file bridge artifact."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--run-dir", help="Benchmark run directory.")
    source.add_argument(
        "--input-csv",
        help="Direct path to vizard_spacecraft_states.csv.",
    )
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--mode", choices=_SUPPORTED_MODES, default="data-file")
    parser.add_argument("--require-basilisk", action="store_true")
    args = parser.parse_args(argv)

    if args.input_csv is not None and args.out_dir is None:
        parser.error("--out-dir is required with --input-csv")

    output_csv, manifest_path, readme_path = (
        build_basilisk_vizard_offline_input(
            args.run_dir,
            mode=args.mode,
            out_dir=args.out_dir,
            input_csv=args.input_csv,
            require_basilisk=args.require_basilisk,
        )
    )
    print(f"wrote {output_csv}")
    print(f"wrote {manifest_path}")
    print(f"wrote {readme_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
