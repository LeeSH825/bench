from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any, Literal, Mapping

import numpy as np
import pandas as pd

from .adcs_timeseries import (
    ADCS_TIMESERIES_FILENAME,
    ADCS_TIMESERIES_META_FILENAME,
)
from .vizard_convention import (
    apply_vizard_convention_to_frame,
    build_vizard_convention,
    convention_summary,
    load_vizard_convention,
)


PositionSource = Literal["fixed_origin", "dummy_circular_orbit"]

VIZARD_STATES_FILENAME = "vizard_spacecraft_states.csv"
VIZARD_MANIFEST_FILENAME = "vizard_export_manifest.json"

_SUPPORTED_POSITION_SOURCES = ("fixed_origin", "dummy_circular_orbit")
_REQUIRED_COLUMNS = (
    "traj_id",
    "t_idx",
    "time_s",
    "sigma1_true",
    "sigma2_true",
    "sigma3_true",
    "sigma1_hat",
    "sigma2_hat",
    "sigma3_hat",
    "omega_x_true_rad_s",
    "omega_y_true_rad_s",
    "omega_z_true_rad_s",
    "omega_x_hat_rad_s",
    "omega_y_hat_rad_s",
    "omega_z_hat_rad_s",
)
_OUTPUT_COLUMNS = (
    "time_s",
    "traj_id",
    "sc_name",
    "source",
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


def _load_adcs_timeseries(
    csv_path: Path,
    meta_path: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not csv_path.exists():
        raise FileNotFoundError(f"ADCS timeseries CSV not found: {csv_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"ADCS timeseries metadata not found: {meta_path}")

    try:
        meta_obj = json.loads(meta_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in ADCS timeseries metadata: {meta_path}") from exc
    if not isinstance(meta_obj, Mapping):
        raise ValueError(f"{meta_path} must contain a JSON object")
    if meta_obj.get("schema_version") != "adcs_timeseries_v1":
        raise ValueError(
            f"{meta_path} has invalid schema_version="
            f"{meta_obj.get('schema_version')!r}"
        )

    try:
        frame = pd.read_csv(csv_path)
    except pd.errors.EmptyDataError as exc:
        raise ValueError(f"ADCS timeseries CSV is empty: {csv_path}") from exc

    missing = [column for column in _REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(
            f"{csv_path} is missing required Vizard export columns: {missing}"
        )

    for column in _REQUIRED_COLUMNS:
        try:
            frame[column] = pd.to_numeric(frame[column], errors="raise")
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"ADCS timeseries column {column!r} must contain numeric values"
            ) from exc
        values = frame[column].to_numpy(dtype=np.float64, copy=False)
        if not np.isfinite(values).all():
            raise ValueError(
                f"ADCS timeseries column {column!r} contains NaN or Inf values"
            )

    for column in ("traj_id", "t_idx"):
        values = frame[column].to_numpy(dtype=np.float64, copy=False)
        if not np.equal(values, np.floor(values)).all():
            raise ValueError(
                f"ADCS timeseries column {column!r} must contain integer values"
            )
        frame[column] = values.astype(np.int64)

    return frame, dict(meta_obj)


def _select_trajectory(frame: pd.DataFrame, trajectory_id: int) -> pd.DataFrame:
    selected = frame.loc[frame["traj_id"] == int(trajectory_id)]
    if selected.empty:
        available = sorted(int(value) for value in frame["traj_id"].unique())
        raise ValueError(
            f"selected trajectory_id={trajectory_id} does not exist; "
            f"available IDs={available}"
        )
    return selected.sort_values("t_idx", kind="stable").reset_index(drop=True)


def _make_fixed_origin(time_s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    shape = (int(np.asarray(time_s).size), 3)
    return np.zeros(shape, dtype=np.float64), np.zeros(shape, dtype=np.float64)


def _make_dummy_circular_orbit(
    time_s: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    radius_m = 7000.0e3
    period_s = 5400.0
    angular_rate = 2.0 * np.pi / period_s
    theta = angular_rate * np.asarray(time_s, dtype=np.float64)
    zeros = np.zeros_like(theta)
    position = np.stack(
        [
            radius_m * np.cos(theta),
            radius_m * np.sin(theta),
            zeros,
        ],
        axis=-1,
    )
    velocity = np.stack(
        [
            -radius_m * angular_rate * np.sin(theta),
            radius_m * angular_rate * np.cos(theta),
            zeros,
        ],
        axis=-1,
    )
    return position, velocity


def _position_timeline(
    time_s: np.ndarray,
    position_source: str,
) -> tuple[np.ndarray, np.ndarray, str]:
    if position_source == "fixed_origin":
        position, velocity = _make_fixed_origin(time_s)
        notes = (
            "Visualization-only attitude debug frame with position and velocity "
            "fixed at zero."
        )
    elif position_source == "dummy_circular_orbit":
        position, velocity = _make_dummy_circular_orbit(time_s)
        notes = (
            "Synthetic visualization-only circular orbit: radius_m=7000000, "
            "period_s=5400.0. It is not scenario truth."
        )
    else:
        raise ValueError(
            f"unsupported position_source={position_source!r}; "
            f"expected one of {_SUPPORTED_POSITION_SOURCES}"
        )
    return position, velocity, notes


def _interleave_vectors(
    true_values: np.ndarray,
    estimated_values: np.ndarray,
) -> np.ndarray:
    return np.stack([true_values, estimated_values], axis=1).reshape(-1, 3)


def _build_spacecraft_rows(
    selected: pd.DataFrame,
    *,
    position: np.ndarray,
    velocity: np.ndarray,
    sc_name_true: str,
    sc_name_estimated: str,
) -> pd.DataFrame:
    time_s = selected["time_s"].to_numpy(dtype=np.float64)
    traj_id = selected["traj_id"].to_numpy(dtype=np.int64)
    sigma_true = selected[
        ["sigma1_true", "sigma2_true", "sigma3_true"]
    ].to_numpy(dtype=np.float64)
    sigma_hat = selected[
        ["sigma1_hat", "sigma2_hat", "sigma3_hat"]
    ].to_numpy(dtype=np.float64)
    omega_true = selected[
        [
            "omega_x_true_rad_s",
            "omega_y_true_rad_s",
            "omega_z_true_rad_s",
        ]
    ].to_numpy(dtype=np.float64)
    omega_hat = selected[
        [
            "omega_x_hat_rad_s",
            "omega_y_hat_rad_s",
            "omega_z_hat_rad_s",
        ]
    ].to_numpy(dtype=np.float64)

    position_rows = np.repeat(position, 2, axis=0)
    velocity_rows = np.repeat(velocity, 2, axis=0)
    sigma_rows = _interleave_vectors(sigma_true, sigma_hat)
    omega_rows = _interleave_vectors(omega_true, omega_hat)
    rows = pd.DataFrame(
        {
            "time_s": np.repeat(time_s, 2),
            "traj_id": np.repeat(traj_id, 2),
            "sc_name": np.tile([sc_name_true, sc_name_estimated], len(selected)),
            "source": np.tile(["true", "estimated"], len(selected)),
            "r_BN_N_x_m": position_rows[:, 0],
            "r_BN_N_y_m": position_rows[:, 1],
            "r_BN_N_z_m": position_rows[:, 2],
            "v_BN_N_x_m_s": velocity_rows[:, 0],
            "v_BN_N_y_m_s": velocity_rows[:, 1],
            "v_BN_N_z_m_s": velocity_rows[:, 2],
            "sigma_BN_1": sigma_rows[:, 0],
            "sigma_BN_2": sigma_rows[:, 1],
            "sigma_BN_3": sigma_rows[:, 2],
            "omega_BN_B_x_rad_s": omega_rows[:, 0],
            "omega_BN_B_y_rad_s": omega_rows[:, 1],
            "omega_BN_B_z_rad_s": omega_rows[:, 2],
        },
        columns=_OUTPUT_COLUMNS,
    )

    numeric_columns = [
        column for column in _OUTPUT_COLUMNS if column not in {"sc_name", "source"}
    ]
    values = rows[numeric_columns].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("generated Vizard spacecraft state rows contain NaN or Inf")
    return rows


def _validate_spacecraft_names(
    sc_name_true: str,
    sc_name_estimated: str,
) -> tuple[str, str]:
    true_name = str(sc_name_true).strip()
    estimated_name = str(sc_name_estimated).strip()
    if not true_name:
        raise ValueError("sc_name_true must be a non-empty string")
    if not estimated_name:
        raise ValueError("sc_name_estimated must be a non-empty string")
    if true_name == estimated_name:
        raise ValueError("true and estimated spacecraft names must be distinct")
    return true_name, estimated_name


def export_vizard_offline(
    run_dir: str | Path,
    *,
    trajectory_id: int = 0,
    position_source: PositionSource = "fixed_origin",
    out_dir: str | Path | None = None,
    sc_name_true: str = "SC_true",
    sc_name_estimated: str = "SC_estimated",
    vizard_convention: str | Path | Mapping[str, Any] | None = None,
) -> tuple[Path, Path]:
    """
    Export true and estimated ADCS timelines without launching Basilisk or Vizard.

    The output is a stable intermediate CSV for a future ``dataFileToViz`` or
    ``vizInterface`` wrapper. It does not affect official benchmark metrics.
    """
    source = str(position_source)
    if source not in _SUPPORTED_POSITION_SOURCES:
        raise ValueError(
            f"unsupported position_source={source!r}; "
            f"expected one of {_SUPPORTED_POSITION_SOURCES}"
        )
    true_name, estimated_name = _validate_spacecraft_names(
        sc_name_true,
        sc_name_estimated,
    )

    input_run_dir = Path(run_dir).expanduser().resolve()
    artifacts_dir = input_run_dir / "artifacts"
    timeseries_path = artifacts_dir / ADCS_TIMESERIES_FILENAME
    timeseries_meta_path = artifacts_dir / ADCS_TIMESERIES_META_FILENAME
    frame, _ = _load_adcs_timeseries(timeseries_path, timeseries_meta_path)
    selected = _select_trajectory(frame, int(trajectory_id))

    time_s = selected["time_s"].to_numpy(dtype=np.float64)
    position, velocity, position_notes = _position_timeline(time_s, source)
    rows = _build_spacecraft_rows(
        selected,
        position=position,
        velocity=velocity,
        sc_name_true=true_name,
        sc_name_estimated=estimated_name,
    )

    convention_file: str | None
    if vizard_convention is None:
        convention = build_vizard_convention("direct")
        convention_file = None
    elif isinstance(vizard_convention, (str, Path)):
        convention_file = str(Path(vizard_convention).expanduser().resolve())
        convention = load_vizard_convention(vizard_convention)
    else:
        convention = load_vizard_convention(vizard_convention)
        convention_file = None
    rows = apply_vizard_convention_to_frame(rows, convention)
    convention_info = convention_summary(convention)

    output_dir = (
        Path(out_dir).expanduser().resolve()
        if out_dir is not None
        else artifacts_dir / "vizard"
    )
    output_csv = output_dir / VIZARD_STATES_FILENAME
    manifest_path = output_dir / VIZARD_MANIFEST_FILENAME
    manifest = {
        "schema_version": "vizard_export_v1",
        "input_run_dir": str(input_run_dir),
        "input_timeseries_csv": str(timeseries_path),
        "input_timeseries_meta": str(timeseries_meta_path),
        "output_csv": str(output_csv),
        "trajectory_id": int(trajectory_id),
        "num_timestamps": int(len(selected)),
        "num_rows": int(len(rows)),
        "spacecraft": [
            {"sc_name": true_name, "source": "true"},
            {"sc_name": estimated_name, "source": "estimated"},
        ],
        "attitude_representation": "MRP sigma_BN",
        "angular_rate_representation": "omega_BN_B rad/s",
        "convention_id": convention_info["convention_id"],
        "convention_file": convention_file,
        "attitude_mrp_mapping": convention_info["attitude_mrp_mapping"],
        "omega_mapping": convention_info["omega_mapping"],
        "requires_manual_vizard_confirmation": True,
        "manual_confirmation_status": convention_info[
            "manual_confirmation_status"
        ],
        "confirmed_by": convention_info["confirmed_by"],
        "confirmed_at_utc": convention_info["confirmed_at_utc"],
        "position_source": source,
        "position_source_notes": position_notes,
        "intended_next_step": (
            "Use this CSV as an intermediate input for a Basilisk/Vizard "
            "dataFileToViz or vizInterface wrapper."
        ),
        "official_metrics_affected": False,
        "notes": (
            "Offline Vizard-ready spacecraft state timeline generated from "
            "ADCS timeseries artifacts."
        ),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".vizard_export_",
        dir=output_dir,
    ) as tmp:
        staging_dir = Path(tmp)
        staged_csv = staging_dir / VIZARD_STATES_FILENAME
        staged_manifest = staging_dir / VIZARD_MANIFEST_FILENAME
        rows.to_csv(staged_csv, index=False)
        staged_manifest.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        staged_csv.replace(output_csv)
        staged_manifest.replace(manifest_path)

    return output_csv, manifest_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate an offline Vizard-ready spacecraft state timeline."
    )
    parser.add_argument("--run-dir", required=True, help="Benchmark run directory.")
    parser.add_argument("--trajectory-id", type=int, default=0)
    parser.add_argument(
        "--position-source",
        choices=_SUPPORTED_POSITION_SOURCES,
        default="fixed_origin",
    )
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--sc-name-true", default="SC_true")
    parser.add_argument("--sc-name-estimated", default="SC_estimated")
    parser.add_argument("--vizard-convention", default=None)
    args = parser.parse_args(argv)

    csv_path, manifest_path = export_vizard_offline(
        args.run_dir,
        trajectory_id=args.trajectory_id,
        position_source=args.position_source,
        out_dir=args.out_dir,
        sc_name_true=args.sc_name_true,
        sc_name_estimated=args.sc_name_estimated,
        vizard_convention=args.vizard_convention,
    )
    print(f"wrote {csv_path}")
    print(f"wrote {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
