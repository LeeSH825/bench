from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np

from .adcs_schema import (
    ADCSStateSchema,
    adcs_state_schema_to_dict,
    parse_adcs_state_schema,
)
from .attitude import euler321_error, mrp_error_norm, mrp_to_euler321
from .pred_artifact import (
    PRED_ARTIFACT_FILENAME,
    PRED_META_FILENAME,
    validate_pred_artifact,
)


ADCS_TIMESERIES_FILENAME = "adcs_timeseries.csv"
ADCS_TIMESERIES_META_FILENAME = "adcs_timeseries_meta.json"

_PRED_REQUIRED_KEYS = ("time_s", "x_true", "y_obs", "x_hat", "trajectory_id")


def _load_prediction_source(
    pred_npz_path: Path,
    pred_meta_path: Path,
) -> dict[str, Any]:
    if not pred_npz_path.exists():
        raise FileNotFoundError(f"prediction artifact not found: {pred_npz_path}")
    if not pred_meta_path.exists():
        raise FileNotFoundError(f"prediction artifact metadata not found: {pred_meta_path}")

    with np.load(pred_npz_path, allow_pickle=False) as data:
        missing = [key for key in _PRED_REQUIRED_KEYS if key not in data.files]
        if missing:
            raise ValueError(
                f"{pred_npz_path} is missing required prediction keys: {missing}"
            )
        arrays = {key: np.array(data[key], copy=True) for key in _PRED_REQUIRED_KEYS}

    validate_pred_artifact(
        time_s=arrays["time_s"],
        x_true=arrays["x_true"],
        y_obs=arrays["y_obs"],
        x_hat=arrays["x_hat"],
        trajectory_id=arrays["trajectory_id"],
        strict=True,
    )
    meta_obj = json.loads(pred_meta_path.read_text(encoding="utf-8"))
    if not isinstance(meta_obj, Mapping):
        raise ValueError(f"{pred_meta_path} must contain a JSON object")
    if meta_obj.get("schema_version") != "pred_artifact_v1":
        raise ValueError(
            f"{pred_meta_path} has invalid schema_version={meta_obj.get('schema_version')!r}"
        )

    expected_meta = {
        "layout": "NTD",
        "required_keys": list(_PRED_REQUIRED_KEYS),
        "x_shape": list(arrays["x_true"].shape),
        "y_shape": list(arrays["y_obs"].shape),
        "time_shape": list(arrays["time_s"].shape),
    }
    for key, expected in expected_meta.items():
        if meta_obj.get(key) != expected:
            raise ValueError(
                f"{pred_meta_path} has invalid {key}: "
                f"expected={expected!r}, got={meta_obj.get(key)!r}"
            )
    return {**arrays, "meta": dict(meta_obj)}


def _selected_positions(
    trajectory_ids: np.ndarray,
    *,
    trajectory_id: Optional[int],
    all_trajectories: bool,
) -> tuple[np.ndarray, np.ndarray]:
    ids = np.asarray(trajectory_ids)
    if ids.ndim != 1 or not np.issubdtype(ids.dtype, np.integer):
        raise ValueError(
            f"trajectory_id must be an integer array with shape [N], got "
            f"shape={ids.shape}, dtype={ids.dtype}"
        )
    ids = ids.astype(np.int64, copy=False)
    if np.unique(ids).size != ids.size:
        raise ValueError(f"trajectory_id contains duplicate values: {ids.tolist()}")

    if all_trajectories:
        positions = np.arange(ids.size, dtype=np.int64)
        return positions, ids

    selected_id = 0 if trajectory_id is None else int(trajectory_id)
    matches = np.flatnonzero(ids == selected_id)
    if matches.size == 0:
        raise ValueError(
            f"selected trajectory_id={selected_id} does not exist; "
            f"available IDs={ids.tolist()}"
        )
    return matches.astype(np.int64), ids[matches]


def _extract_state(
    state: np.ndarray,
    indices: tuple[int, int, int],
) -> np.ndarray:
    return np.take(state, np.asarray(indices, dtype=np.int64), axis=-1)


def _flat(arr: np.ndarray) -> np.ndarray:
    return np.asarray(arr).reshape(-1)


def _build_columns(
    *,
    x_true: np.ndarray,
    x_hat: np.ndarray,
    time_s: np.ndarray,
    trajectory_ids: np.ndarray,
    schema: ADCSStateSchema,
) -> dict[str, np.ndarray]:
    n_selected, n_step, _ = x_true.shape
    sigma_true = _extract_state(x_true, schema.attitude_indices)
    sigma_hat = _extract_state(x_hat, schema.attitude_indices)
    euler_true = mrp_to_euler321(sigma_true)
    euler_hat = mrp_to_euler321(sigma_hat)
    euler_err = euler321_error(euler_hat, euler_true)
    omega_true = _extract_state(x_true, schema.angular_rate_indices)
    omega_hat = _extract_state(x_hat, schema.angular_rate_indices)

    columns: dict[str, np.ndarray] = {
        "traj_id": np.repeat(trajectory_ids.astype(np.int64), n_step),
        "t_idx": np.tile(np.arange(n_step, dtype=np.int64), n_selected),
        "time_s": _flat(time_s),
        "sigma1_true": _flat(sigma_true[..., 0]),
        "sigma2_true": _flat(sigma_true[..., 1]),
        "sigma3_true": _flat(sigma_true[..., 2]),
        "sigma1_hat": _flat(sigma_hat[..., 0]),
        "sigma2_hat": _flat(sigma_hat[..., 1]),
        "sigma3_hat": _flat(sigma_hat[..., 2]),
        "mrp_err_norm": _flat(mrp_error_norm(sigma_hat, sigma_true)),
        "roll_true_rad": _flat(euler_true[..., 0]),
        "pitch_true_rad": _flat(euler_true[..., 1]),
        "yaw_true_rad": _flat(euler_true[..., 2]),
        "roll_hat_rad": _flat(euler_hat[..., 0]),
        "pitch_hat_rad": _flat(euler_hat[..., 1]),
        "yaw_hat_rad": _flat(euler_hat[..., 2]),
        "roll_err_rad": _flat(euler_err[..., 0]),
        "pitch_err_rad": _flat(euler_err[..., 1]),
        "yaw_err_rad": _flat(euler_err[..., 2]),
        "omega_x_true_rad_s": _flat(omega_true[..., 0]),
        "omega_y_true_rad_s": _flat(omega_true[..., 1]),
        "omega_z_true_rad_s": _flat(omega_true[..., 2]),
        "omega_x_hat_rad_s": _flat(omega_hat[..., 0]),
        "omega_y_hat_rad_s": _flat(omega_hat[..., 1]),
        "omega_z_hat_rad_s": _flat(omega_hat[..., 2]),
        "omega_err_norm_rad_s": _flat(
            np.linalg.norm(omega_hat - omega_true, axis=-1)
        ),
    }

    if schema.gyro_bias_indices is not None:
        bias_true = _extract_state(x_true, schema.gyro_bias_indices)
        bias_hat = _extract_state(x_hat, schema.gyro_bias_indices)
        columns.update(
            {
                "bias_x_true_rad_s": _flat(bias_true[..., 0]),
                "bias_y_true_rad_s": _flat(bias_true[..., 1]),
                "bias_z_true_rad_s": _flat(bias_true[..., 2]),
                "bias_x_hat_rad_s": _flat(bias_hat[..., 0]),
                "bias_y_hat_rad_s": _flat(bias_hat[..., 1]),
                "bias_z_hat_rad_s": _flat(bias_hat[..., 2]),
                "bias_err_norm_rad_s": _flat(
                    np.linalg.norm(bias_hat - bias_true, axis=-1)
                ),
            }
        )

    expected_rows = int(n_selected * n_step)
    for name, values in columns.items():
        if values.shape != (expected_rows,):
            raise ValueError(
                f"generated column {name} has shape={values.shape}, "
                f"expected={(expected_rows,)}"
            )
        if not np.isfinite(values).all():
            raise ValueError(f"generated column {name} contains NaN or Inf values")
    return columns


def _write_csv(path: Path, columns: Mapping[str, np.ndarray]) -> None:
    tmp = path.with_name(f".{path.name}.tmp")
    names = list(columns.keys())
    try:
        with tmp.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(names)
            writer.writerows(zip(*(columns[name] for name in names)))
        tmp.replace(path)
    finally:
        if tmp.exists():
            tmp.unlink()


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


def build_adcs_timeseries(
    pred_npz_path: str | Path,
    *,
    pred_meta_path: str | Path | None = None,
    trajectory_id: int | None = None,
    all_trajectories: bool = False,
    out_dir: str | Path | None = None,
) -> tuple[Path, Path]:
    if all_trajectories and trajectory_id is not None:
        raise ValueError("trajectory_id and all_trajectories are mutually exclusive")

    pred_path = Path(pred_npz_path).expanduser().resolve()
    meta_path = (
        Path(pred_meta_path).expanduser().resolve()
        if pred_meta_path is not None
        else pred_path.with_name(PRED_META_FILENAME)
    )
    source = _load_prediction_source(pred_path, meta_path)
    x_true_all = np.asarray(source["x_true"])
    x_hat_all = np.asarray(source["x_hat"])
    time_all = np.asarray(source["time_s"])
    source_meta = source["meta"]

    schema = parse_adcs_state_schema(source_meta, x_dim=int(x_true_all.shape[2]))
    positions, selected_ids = _selected_positions(
        source["trajectory_id"],
        trajectory_id=trajectory_id,
        all_trajectories=bool(all_trajectories),
    )
    x_true = x_true_all[positions]
    x_hat = x_hat_all[positions]
    if time_all.ndim == 1:
        selected_time = np.broadcast_to(
            time_all[None, :],
            (positions.size, time_all.shape[0]),
        )
    else:
        selected_time = time_all[positions]

    columns = _build_columns(
        x_true=x_true,
        x_hat=x_hat,
        time_s=selected_time,
        trajectory_ids=selected_ids,
        schema=schema,
    )

    output_dir = (
        Path(out_dir).expanduser().resolve()
        if out_dir is not None
        else pred_path.parent
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / ADCS_TIMESERIES_FILENAME
    output_meta_path = output_dir / ADCS_TIMESERIES_META_FILENAME

    output_meta: dict[str, Any] = {
        "schema_version": "adcs_timeseries_v1",
        "source_pred_schema_version": source_meta.get("schema_version"),
        "input_pred_artifact": str(pred_path),
        "input_pred_meta": str(meta_path),
        "output_csv": str(csv_path),
        "selected_trajectory_ids": [int(v) for v in selected_ids.tolist()],
        "num_rows": int(len(columns["traj_id"])),
        "num_trajectories": int(selected_ids.size),
        "state_schema": adcs_state_schema_to_dict(schema),
        "schema_source": schema.schema_source,
        "attitude_convention": schema.attitude_convention,
        "euler_convention": "321",
        "time_unit": schema.time_unit,
        "notes": "ADCS timestamp-level timeseries derived from prediction artifact.",
    }
    for key in ("time_source", "time_warning", "dt_s"):
        if key in source_meta:
            output_meta[key] = source_meta[key]

    _write_csv(csv_path, columns)
    _write_json(output_meta_path, output_meta)
    return csv_path, output_meta_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build an ADCS timestamp-level CSV from a benchmark prediction artifact."
    )
    parser.add_argument("--run-dir", required=True, help="Benchmark run directory.")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--trajectory-id", type=int, default=None)
    selection.add_argument("--all-trajectories", action="store_true")
    args = parser.parse_args(argv)

    run_dir = Path(args.run_dir).expanduser().resolve()
    artifacts_dir = run_dir / "artifacts"
    csv_path, meta_path = build_adcs_timeseries(
        artifacts_dir / PRED_ARTIFACT_FILENAME,
        pred_meta_path=artifacts_dir / PRED_META_FILENAME,
        trajectory_id=(None if args.all_trajectories else args.trajectory_id),
        all_trajectories=bool(args.all_trajectories),
        out_dir=artifacts_dir,
    )
    print(f"wrote {csv_path}")
    print(f"wrote {meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
