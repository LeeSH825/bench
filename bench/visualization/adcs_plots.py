from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import matplotlib
import numpy as np
import pandas as pd


matplotlib.use("Agg")
import matplotlib.pyplot as plt


ADCS_PLOT_MANIFEST_FILENAME = "adcs_plot_manifest.json"

_BASE_COLUMNS = ("traj_id", "t_idx", "time_s")
_RPY_COLUMNS = (
    "roll_true_rad",
    "pitch_true_rad",
    "yaw_true_rad",
    "roll_hat_rad",
    "pitch_hat_rad",
    "yaw_hat_rad",
    "roll_err_rad",
    "pitch_err_rad",
    "yaw_err_rad",
)
_OMEGA_COLUMNS = (
    "omega_x_true_rad_s",
    "omega_y_true_rad_s",
    "omega_z_true_rad_s",
    "omega_x_hat_rad_s",
    "omega_y_hat_rad_s",
    "omega_z_hat_rad_s",
    "omega_err_norm_rad_s",
)
_ATTITUDE_ERROR_COLUMNS = ("mrp_err_norm",)
_REQUIRED_COLUMNS = (
    _BASE_COLUMNS + _RPY_COLUMNS + _OMEGA_COLUMNS + _ATTITUDE_ERROR_COLUMNS
)
_BIAS_COLUMNS = (
    "bias_x_true_rad_s",
    "bias_y_true_rad_s",
    "bias_z_true_rad_s",
    "bias_x_hat_rad_s",
    "bias_y_hat_rad_s",
    "bias_z_hat_rad_s",
    "bias_err_norm_rad_s",
)

_REQUIRED_PLOT_FILENAMES = (
    "rpy_true_vs_hat.png",
    "rpy_error.png",
    "omega_true_vs_hat.png",
    "omega_error_norm.png",
    "mrp_error_norm.png",
)
_BIAS_PLOT_FILENAMES = (
    "bias_true_vs_hat.png",
    "bias_error_norm.png",
)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    tmp = path.with_name(f".{path.name}.tmp")
    try:
        tmp.write_text(
            json.dumps(dict(payload), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        tmp.replace(path)
    finally:
        if tmp.exists():
            tmp.unlink()


def _load_timeseries(path: Path) -> tuple[pd.DataFrame, bool]:
    if not path.exists():
        raise FileNotFoundError(f"ADCS timeseries CSV not found: {path}")
    try:
        frame = pd.read_csv(path)
    except pd.errors.EmptyDataError as exc:
        raise ValueError(f"ADCS timeseries CSV is empty: {path}") from exc

    missing = [column for column in _REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(
            f"{path} is missing required ADCS timeseries columns: {missing}"
        )

    present_bias = [column for column in _BIAS_COLUMNS if column in frame.columns]
    if present_bias and len(present_bias) != len(_BIAS_COLUMNS):
        missing_bias = [
            column for column in _BIAS_COLUMNS if column not in frame.columns
        ]
        raise ValueError(
            "ADCS timeseries CSV contains partial gyro-bias columns; "
            f"missing={missing_bias}, present={present_bias}"
        )
    has_bias = len(present_bias) == len(_BIAS_COLUMNS)

    numeric_columns = list(_REQUIRED_COLUMNS)
    if has_bias:
        numeric_columns.extend(_BIAS_COLUMNS)
    for column in numeric_columns:
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

    return frame, has_bias


def _select_trajectory(
    frame: pd.DataFrame,
    trajectory_id: Optional[int],
) -> pd.DataFrame:
    selected = frame
    if trajectory_id is not None:
        selected = frame.loc[frame["traj_id"] == int(trajectory_id)]
        if selected.empty:
            available = sorted(frame["traj_id"].unique().tolist())
            raise ValueError(
                f"selected trajectory_id={trajectory_id} does not exist; "
                f"available IDs={available}"
            )
    if selected.empty:
        raise ValueError("ADCS timeseries selection contains no rows to plot")
    return selected.sort_values(["traj_id", "t_idx"], kind="stable").reset_index(
        drop=True
    )


def _plot_series(
    frame: pd.DataFrame,
    *,
    series: Sequence[str],
    title: str,
    ylabel: str,
    out_path: Path,
    dpi: int,
) -> None:
    fig, ax = plt.subplots()
    try:
        trajectory_count = int(frame["traj_id"].nunique())
        for traj_id, trajectory in frame.groupby("traj_id", sort=False):
            time_s = trajectory["time_s"].to_numpy()
            for column in series:
                label = (
                    column
                    if trajectory_count == 1
                    else f"{column} [traj={traj_id:g}]"
                )
                ax.plot(time_s, trajectory[column].to_numpy(), label=label)
        ax.set_title(title)
        ax.set_xlabel("time [s]")
        ax.set_ylabel(ylabel)
        ax.grid(True)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_path, dpi=dpi, format="png")
    finally:
        plt.close(fig)


def _render_plots(
    frame: pd.DataFrame,
    *,
    staging_dir: Path,
    include_bias: bool,
    dpi: int,
) -> list[str]:
    plot_specs = [
        (
            "rpy_true_vs_hat.png",
            (
                "roll_true_rad",
                "roll_hat_rad",
                "pitch_true_rad",
                "pitch_hat_rad",
                "yaw_true_rad",
                "yaw_hat_rad",
            ),
            "RPY True vs Estimated",
            "angle [rad]",
        ),
        (
            "rpy_error.png",
            ("roll_err_rad", "pitch_err_rad", "yaw_err_rad"),
            "RPY Error",
            "error [rad]",
        ),
        (
            "omega_true_vs_hat.png",
            (
                "omega_x_true_rad_s",
                "omega_x_hat_rad_s",
                "omega_y_true_rad_s",
                "omega_y_hat_rad_s",
                "omega_z_true_rad_s",
                "omega_z_hat_rad_s",
            ),
            "Angular Rate True vs Estimated",
            "angular rate [rad/s]",
        ),
        (
            "omega_error_norm.png",
            ("omega_err_norm_rad_s",),
            "Angular-Rate Error Norm",
            "angular-rate error norm [rad/s]",
        ),
        (
            "mrp_error_norm.png",
            ("mrp_err_norm",),
            "MRP Error Norm",
            "MRP error norm",
        ),
    ]
    if include_bias:
        plot_specs.extend(
            [
                (
                    "bias_true_vs_hat.png",
                    (
                        "bias_x_true_rad_s",
                        "bias_x_hat_rad_s",
                        "bias_y_true_rad_s",
                        "bias_y_hat_rad_s",
                        "bias_z_true_rad_s",
                        "bias_z_hat_rad_s",
                    ),
                    "Gyro Bias True vs Estimated",
                    "gyro bias [rad/s]",
                ),
                (
                    "bias_error_norm.png",
                    ("bias_err_norm_rad_s",),
                    "Gyro-Bias Error Norm",
                    "gyro-bias error norm [rad/s]",
                ),
            ]
        )

    filenames: list[str] = []
    for filename, series, title, ylabel in plot_specs:
        _plot_series(
            frame,
            series=series,
            title=title,
            ylabel=ylabel,
            out_path=staging_dir / filename,
            dpi=dpi,
        )
        filenames.append(filename)
    return filenames


def make_adcs_plots(
    timeseries_csv: str | Path,
    *,
    out_dir: str | Path | None = None,
    trajectory_id: int | None = None,
    plot_bias: bool = True,
    dpi: int = 150,
) -> tuple[list[Path], Path]:
    """Generate downstream ADCS sanity-check plots from a Phase 2 timeseries CSV."""
    if int(dpi) <= 0:
        raise ValueError(f"dpi must be positive, got {dpi}")

    csv_path = Path(timeseries_csv).expanduser().resolve()
    output_dir = (
        Path(out_dir).expanduser().resolve()
        if out_dir is not None
        else csv_path.parent / "plots"
    )
    frame, has_bias = _load_timeseries(csv_path)
    selected = _select_trajectory(frame, trajectory_id)
    include_bias = bool(plot_bias and has_bias)

    skipped_plots: list[dict[str, str]] = []
    if not has_bias:
        skipped_plots.extend(
            {
                "plot": filename,
                "reason": "complete gyro-bias columns are absent",
            }
            for filename in _BIAS_PLOT_FILENAMES
        )
    elif not plot_bias:
        skipped_plots.extend(
            {
                "plot": filename,
                "reason": "bias plotting disabled by plot_bias=False",
            }
            for filename in _BIAS_PLOT_FILENAMES
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".adcs_plots_",
        dir=output_dir,
    ) as tmp:
        staging_dir = Path(tmp)
        generated_filenames = _render_plots(
            selected,
            staging_dir=staging_dir,
            include_bias=include_bias,
            dpi=int(dpi),
        )
        for filename in generated_filenames:
            (staging_dir / filename).replace(output_dir / filename)

    if not include_bias:
        for filename in _BIAS_PLOT_FILENAMES:
            stale_path = output_dir / filename
            if stale_path.exists():
                stale_path.unlink()

    generated_paths = [output_dir / name for name in generated_filenames]
    manifest_path = output_dir / ADCS_PLOT_MANIFEST_FILENAME
    manifest = {
        "schema_version": "adcs_plot_manifest_v1",
        "input_timeseries_csv": str(csv_path),
        "output_dir": str(output_dir),
        "trajectory_id": (
            None if trajectory_id is None else int(trajectory_id)
        ),
        "num_rows_plotted": int(len(selected)),
        "generated_plots": [str(path) for path in generated_paths],
        "skipped_plots": skipped_plots,
        "notes": (
            "ADCS sanity-check plots generated from adcs_timeseries.csv. "
            "These plots do not affect official benchmark metrics."
        ),
    }
    _write_json(manifest_path, manifest)
    return generated_paths, manifest_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate ADCS sanity-check PNG plots from adcs_timeseries.csv."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--run-dir", help="Benchmark run directory.")
    source.add_argument("--timeseries-csv", help="Path to adcs_timeseries.csv.")
    parser.add_argument(
        "--out-dir",
        help="Output directory for direct CSV mode; defaults to a sibling plots directory.",
    )
    parser.add_argument("--trajectory-id", type=int, default=None)
    parser.add_argument("--no-bias", action="store_true")
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args(argv)

    if args.run_dir is not None:
        if args.out_dir is not None:
            parser.error("--out-dir cannot be used with --run-dir")
        run_dir = Path(args.run_dir).expanduser().resolve()
        csv_path = run_dir / "artifacts" / "adcs_timeseries.csv"
        output_dir = run_dir / "artifacts" / "plots"
    else:
        csv_path = Path(args.timeseries_csv).expanduser().resolve()
        output_dir = (
            Path(args.out_dir).expanduser().resolve()
            if args.out_dir is not None
            else csv_path.parent / "plots"
        )

    plot_paths, manifest_path = make_adcs_plots(
        csv_path,
        out_dir=output_dir,
        trajectory_id=args.trajectory_id,
        plot_bias=not args.no_bias,
        dpi=args.dpi,
    )
    for path in plot_paths:
        print(f"wrote {path}")
    print(f"wrote {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
