from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np

try:
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore


def to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    if torch is not None and isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _float_or_none(value: Any) -> Optional[float]:
    try:
        f = float(value)
    except Exception:
        return None
    if math.isnan(f) or math.isinf(f):
        return f
    return f


def array_stats(value: Any) -> Dict[str, Any]:
    arr = to_numpy(value)
    stats: Dict[str, Any] = {
        "shape": [int(x) for x in arr.shape],
        "dtype": str(arr.dtype),
        "size": int(arr.size),
        "finite": bool(np.isfinite(arr).all()) if arr.size else True,
        "nan_count": int(np.isnan(arr).sum()) if np.issubdtype(arr.dtype, np.number) else 0,
        "inf_count": int(np.isinf(arr).sum()) if np.issubdtype(arr.dtype, np.number) else 0,
    }
    if arr.size == 0 or not np.issubdtype(arr.dtype, np.number):
        stats.update({"min": None, "max": None, "mean": None, "std": None, "norm": None})
        return stats

    finite_vals = arr[np.isfinite(arr)]
    if finite_vals.size == 0:
        stats.update({"min": None, "max": None, "mean": None, "std": None, "norm": None})
        return stats

    stats.update(
        {
            "min": _float_or_none(np.min(finite_vals)),
            "max": _float_or_none(np.max(finite_vals)),
            "mean": _float_or_none(np.mean(finite_vals)),
            "std": _float_or_none(np.std(finite_vals)),
            "norm": _float_or_none(np.linalg.norm(finite_vals.ravel(), ord=2)),
        }
    )
    return stats


def format_array_stats(label: str, value: Any) -> str:
    stats = array_stats(value)
    return (
        f"{label}: shape={tuple(stats['shape'])} dtype={stats['dtype']} "
        f"min={stats['min']} max={stats['max']} mean={stats['mean']} std={stats['std']} "
        f"norm={stats['norm']} nan={stats['nan_count']} inf={stats['inf_count']}"
    )


def has_nonfinite(value: Any) -> bool:
    arr = to_numpy(value)
    return bool(arr.size and (not np.isfinite(arr).all()))


def _shape_tuple(value: Any) -> Tuple[int, ...]:
    if hasattr(value, "shape"):
        try:
            return tuple(int(x) for x in value.shape)
        except Exception:
            pass
    arr = to_numpy(value)
    return tuple(int(x) for x in arr.shape)


def validate_exact_layout(
    value: Any,
    *,
    expected: Tuple[int, int, int],
    axis_names: Tuple[str, str, str],
    label: str,
) -> None:
    shape = _shape_tuple(value)
    expected_shape = tuple(int(x) for x in expected)
    if len(shape) != 3:
        raise ValueError(
            f"shape_mismatch: {label} must be rank-3 with layout [{axis_names[0]},{axis_names[1]},{axis_names[2]}], "
            f"got shape={shape}"
        )
    if shape == expected_shape:
        return

    transposed_hint = ""
    a0, a1, a2 = axis_names
    if shape == (expected_shape[0], expected_shape[2], expected_shape[1]):
        transposed_hint = (
            f" Detected likely [{a0},{a2},{a1}] transpose. "
            f"Fix the adapter permutation before returning {label}."
        )
    elif shape == (expected_shape[1], expected_shape[0], expected_shape[2]):
        transposed_hint = (
            f" Detected likely [{a1},{a0},{a2}] transpose. "
            f"Fix the adapter permutation before returning {label}."
        )

    raise ValueError(
        f"shape_mismatch: expected {label} [{axis_names[0]},{axis_names[1]},{axis_names[2]}]={expected_shape}, "
        f"got {shape}.{transposed_hint}"
    )


def short_window(value: Any, *, batch_limit: int = 1, time_limit: int = 32) -> np.ndarray:
    arr = to_numpy(value)
    if arr.ndim >= 1:
        arr = arr[: max(1, int(batch_limit))]
    if arr.ndim >= 2:
        arr = arr[:, : max(1, int(time_limit))]
    return np.array(arr, copy=True)


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def write_diagnostic_dump(
    *,
    run_dir: Path,
    stats: Mapping[str, Any],
    arrays: Mapping[str, Any],
    extra_arrays: Optional[Mapping[str, Any]] = None,
) -> Dict[str, str]:
    diag_dir = run_dir / "diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)

    stats_path = diag_dir / "stats.json"
    stats_path.write_text(json.dumps(_jsonable(dict(stats)), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    npz_path = diag_dir / "first_batch_dump.npz"
    save_payload: Dict[str, np.ndarray] = {}
    for key, value in arrays.items():
        save_payload[str(key)] = short_window(value)
    for key, value in (extra_arrays or {}).items():
        save_payload[str(key)] = short_window(value)
    np.savez_compressed(npz_path, **save_payload)
    return {"diagnostics_dir": str(diag_dir), "stats_path": str(stats_path), "npz_path": str(npz_path)}


def summarize_mapping_arrays(prefix: str, payload: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if not payload:
        return out
    for key, value in payload.items():
        if isinstance(value, (dict, list, tuple)):
            continue
        try:
            out[f"{prefix}.{key}"] = array_stats(value)
        except Exception:
            continue
    return out
