from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np


PRED_ARTIFACT_FILENAME = "preds_test.npz"
PRED_META_FILENAME = "preds_test_meta.json"

_REQUIRED_KEYS = ("time_s", "x_true", "y_obs", "x_hat", "trajectory_id")


def _shape(arr: np.ndarray) -> tuple[int, ...]:
    return tuple(int(v) for v in arr.shape)


def _require_numeric(key: str, arr: np.ndarray) -> None:
    if not np.issubdtype(arr.dtype, np.number):
        raise ValueError(f"{key} must be a numeric array, got dtype={arr.dtype}")


def _require_finite(key: str, arr: np.ndarray) -> None:
    _require_numeric(key, arr)
    if not np.isfinite(arr).all():
        raise ValueError(f"{key} contains NaN or Inf values")


def validate_pred_artifact(
    *,
    time_s: np.ndarray,
    x_true: np.ndarray,
    y_obs: np.ndarray,
    x_hat: np.ndarray,
    trajectory_id: Optional[np.ndarray] = None,
    strict: bool = True,
) -> None:
    time_arr = np.asarray(time_s)
    x_true_arr = np.asarray(x_true)
    y_obs_arr = np.asarray(y_obs)
    x_hat_arr = np.asarray(x_hat)

    if x_true_arr.ndim != 3:
        raise ValueError(f"x_true must have shape [N,T,Dx] (rank 3), got shape={_shape(x_true_arr)}")
    if x_hat_arr.ndim != 3:
        raise ValueError(f"x_hat must have shape [N,T,Dx] (rank 3), got shape={_shape(x_hat_arr)}")
    if x_true_arr.shape != x_hat_arr.shape:
        raise ValueError(
            "x_hat shape must match x_true shape: "
            f"x_true={_shape(x_true_arr)}, x_hat={_shape(x_hat_arr)}"
        )
    if y_obs_arr.ndim != 3:
        raise ValueError(f"y_obs must have shape [N,T,Dy] (rank 3), got shape={_shape(y_obs_arr)}")
    if x_true_arr.shape[0] != y_obs_arr.shape[0]:
        raise ValueError(
            "y_obs N dimension must match x_true: "
            f"x_true={_shape(x_true_arr)}, y_obs={_shape(y_obs_arr)}"
        )
    if x_true_arr.shape[1] != y_obs_arr.shape[1]:
        raise ValueError(
            "y_obs T dimension must match x_true: "
            f"x_true={_shape(x_true_arr)}, y_obs={_shape(y_obs_arr)}"
        )

    n_seq, n_step = int(x_true_arr.shape[0]), int(x_true_arr.shape[1])
    if time_arr.ndim == 1:
        if time_arr.shape[0] != n_step:
            raise ValueError(
                f"time_s shape [T] must use T={n_step}, got shape={_shape(time_arr)}"
            )
    elif time_arr.ndim == 2:
        if time_arr.shape != (n_seq, n_step):
            raise ValueError(
                "time_s shape [N,T] must match x_true[:2]: "
                f"expected={(n_seq, n_step)}, got shape={_shape(time_arr)}"
            )
    else:
        raise ValueError(
            f"time_s must have shape [T] or [N,T] (rank 1 or 2), got shape={_shape(time_arr)}"
        )

    trajectory_arr: Optional[np.ndarray] = None
    if trajectory_id is not None:
        trajectory_arr = np.asarray(trajectory_id)
        if trajectory_arr.ndim != 1 or trajectory_arr.shape[0] != n_seq:
            raise ValueError(
                f"trajectory_id must have shape [N] with N={n_seq}, got shape={_shape(trajectory_arr)}"
            )

    for key, arr in (
        ("time_s", time_arr),
        ("x_true", x_true_arr),
        ("y_obs", y_obs_arr),
        ("x_hat", x_hat_arr),
    ):
        _require_numeric(key, arr)
        if strict:
            _require_finite(key, arr)
    if trajectory_arr is not None:
        _require_numeric("trajectory_id", trajectory_arr)
        if strict:
            _require_finite("trajectory_id", trajectory_arr)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def save_pred_artifact(
    out_dir: str | Path,
    *,
    time_s: np.ndarray,
    x_true: np.ndarray,
    y_obs: np.ndarray,
    x_hat: np.ndarray,
    trajectory_id: Optional[np.ndarray] = None,
    meta: Optional[Mapping[str, Any]] = None,
) -> tuple[Path, Path]:
    out_path = Path(out_dir).expanduser()
    out_path.mkdir(parents=True, exist_ok=True)

    x_true_arr = np.asarray(x_true)
    y_obs_arr = np.asarray(y_obs)
    x_hat_arr = np.asarray(x_hat)
    time_arr = np.asarray(time_s)
    n_seq = int(x_true_arr.shape[0]) if x_true_arr.ndim >= 1 else 0
    trajectory_arr = (
        np.arange(n_seq, dtype=np.int64)
        if trajectory_id is None
        else np.asarray(trajectory_id)
    )

    validate_pred_artifact(
        time_s=time_arr,
        x_true=x_true_arr,
        y_obs=y_obs_arr,
        x_hat=x_hat_arr,
        trajectory_id=trajectory_arr,
        strict=True,
    )

    arrays = {
        "time_s": time_arr.astype(np.float32, copy=False),
        "x_true": x_true_arr.astype(np.float32, copy=False),
        "y_obs": y_obs_arr.astype(np.float32, copy=False),
        "x_hat": x_hat_arr.astype(np.float32, copy=False),
        "trajectory_id": trajectory_arr.astype(np.int64, copy=False),
    }
    default_meta = {
        "schema_version": "pred_artifact_v1",
        "layout": "NTD",
        "required_keys": list(_REQUIRED_KEYS),
        "x_shape": list(arrays["x_true"].shape),
        "y_shape": list(arrays["y_obs"].shape),
        "time_shape": list(arrays["time_s"].shape),
        "notes": "Prediction artifact for benchmark visualization and downstream analysis.",
    }
    merged_meta = dict(_jsonable(meta or {}))
    merged_meta.update(default_meta)

    artifact_path = out_path / PRED_ARTIFACT_FILENAME
    meta_path = out_path / PRED_META_FILENAME
    artifact_tmp = out_path / f".{PRED_ARTIFACT_FILENAME}.tmp.npz"
    meta_tmp = out_path / f".{PRED_META_FILENAME}.tmp"

    try:
        np.savez_compressed(artifact_tmp, **arrays)
        meta_tmp.write_text(
            json.dumps(merged_meta, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        artifact_tmp.replace(artifact_path)
        meta_tmp.replace(meta_path)
    finally:
        if artifact_tmp.exists():
            artifact_tmp.unlink()
        if meta_tmp.exists():
            meta_tmp.unlink()

    return artifact_path, meta_path


def _resolve_paths(path: str | Path) -> tuple[Path, Path]:
    p = Path(path).expanduser()
    if p.is_dir():
        return p / PRED_ARTIFACT_FILENAME, p / PRED_META_FILENAME
    if p.name == PRED_META_FILENAME:
        return p.with_name(PRED_ARTIFACT_FILENAME), p
    return p, p.with_name(PRED_META_FILENAME)


def load_pred_artifact(path: str | Path) -> dict[str, Any]:
    artifact_path, meta_path = _resolve_paths(path)
    if not artifact_path.exists():
        raise FileNotFoundError(f"prediction artifact not found: {artifact_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"prediction artifact metadata not found: {meta_path}")

    with np.load(artifact_path, allow_pickle=False) as data:
        missing = [key for key in _REQUIRED_KEYS if key not in data.files]
        if missing:
            raise ValueError(
                f"{artifact_path} is missing required prediction keys: {missing}"
            )
        arrays = {key: np.array(data[key], copy=True) for key in _REQUIRED_KEYS}

    validate_pred_artifact(
        time_s=arrays["time_s"],
        x_true=arrays["x_true"],
        y_obs=arrays["y_obs"],
        x_hat=arrays["x_hat"],
        trajectory_id=arrays["trajectory_id"],
        strict=True,
    )
    meta_obj = json.loads(meta_path.read_text(encoding="utf-8"))
    if not isinstance(meta_obj, dict):
        raise ValueError(f"{meta_path} must contain a JSON object")
    if meta_obj.get("schema_version") != "pred_artifact_v1":
        raise ValueError(
            f"{meta_path} has invalid schema_version={meta_obj.get('schema_version')!r}"
        )
    expected_meta = {
        "layout": "NTD",
        "required_keys": list(_REQUIRED_KEYS),
        "x_shape": list(arrays["x_true"].shape),
        "y_shape": list(arrays["y_obs"].shape),
        "time_shape": list(arrays["time_s"].shape),
    }
    for key, expected in expected_meta.items():
        if meta_obj.get(key) != expected:
            raise ValueError(
                f"{meta_path} has invalid {key}: expected={expected!r}, got={meta_obj.get(key)!r}"
            )

    return {
        **arrays,
        "meta": meta_obj,
        "artifact_path": artifact_path,
        "meta_path": meta_path,
    }
