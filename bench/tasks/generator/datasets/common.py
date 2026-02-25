from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np

INTERNAL_SPLIT_PAYLOADS_KEY = "__split_payloads__"


class DatasetMissingError(FileNotFoundError):
    """
    Raised when an external dataset is not configured or incomplete.

    Subclassing FileNotFoundError keeps run-suite failure classification aligned
    with io_error.
    """

    error_code = "io_error"

    def __init__(self, *, dataset: str, env_var: str, message: str) -> None:
        self.dataset = str(dataset)
        self.env_var = str(env_var)
        self.error_code = "io_error"
        super().__init__(f"[{self.error_code}] dataset={self.dataset} env_var={self.env_var}: {message}")


def dataset_root_is_available(env_var: str) -> bool:
    raw = os.environ.get(str(env_var), "").strip()
    if not raw:
        return False
    p = Path(raw).expanduser()
    return p.exists() and p.is_dir()


def resolve_dataset_root(
    *,
    dataset: str,
    env_var: str,
    expected_layout_lines: Sequence[str],
    prep_hint: str,
) -> Path:
    raw = os.environ.get(str(env_var), "").strip()
    if not raw:
        layout = "\n".join(f"  - {line}" for line in expected_layout_lines)
        raise DatasetMissingError(
            dataset=dataset,
            env_var=env_var,
            message=(
                f"Environment variable '{env_var}' is not set.\n"
                f"Expected dataset root layout examples:\n{layout}\n"
                f"{prep_hint}"
            ),
        )

    root = Path(raw).expanduser().resolve()
    if not root.exists():
        raise DatasetMissingError(
            dataset=dataset,
            env_var=env_var,
            message=f"Path from '{env_var}' does not exist: {root}",
        )
    if not root.is_dir():
        raise DatasetMissingError(
            dataset=dataset,
            env_var=env_var,
            message=f"Path from '{env_var}' is not a directory: {root}",
        )
    return root


def find_first_existing(paths: Iterable[Path]) -> Optional[Path]:
    for p in paths:
        if p.exists() and p.is_file():
            return p
    return None


def _pick_key(files: Sequence[str], candidates: Sequence[str]) -> Optional[str]:
    lowered = {f.lower(): f for f in files}
    for cand in candidates:
        if cand in files:
            return cand
        key = lowered.get(str(cand).lower())
        if key is not None:
            return key
    return None


def load_xy_npz(
    *,
    npz_path: Path,
    dataset: str,
    env_var: str,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    try:
        with np.load(npz_path, allow_pickle=False) as z:
            files = list(z.files)
            x_key = _pick_key(files, ("x", "x_seq", "states", "state", "x_gt"))
            y_key = _pick_key(files, ("y", "y_seq", "observations", "obs", "measurements", "z"))
            if x_key is None or y_key is None:
                raise DatasetMissingError(
                    dataset=dataset,
                    env_var=env_var,
                    message=(
                        f"Missing x/y keys in {npz_path}. Found keys={files}. "
                        "Expected x/y (or states/observations)."
                    ),
                )
            x = np.asarray(z[x_key], dtype=np.float64)
            y = np.asarray(z[y_key], dtype=np.float64)
            extras: Dict[str, np.ndarray] = {}
            reserved = {x_key, y_key}
            for key in files:
                if key in reserved:
                    continue
                extras[str(key)] = np.asarray(z[key])
    except DatasetMissingError:
        raise
    except Exception as exc:
        raise DatasetMissingError(
            dataset=dataset,
            env_var=env_var,
            message=f"Failed to parse npz file {npz_path}: {exc}",
        ) from exc

    if x.ndim not in (2, 3) or y.ndim not in (2, 3):
        raise DatasetMissingError(
            dataset=dataset,
            env_var=env_var,
            message=f"Expected x/y to be rank-2 or rank-3 arrays, got x.shape={x.shape}, y.shape={y.shape}",
        )
    return x, y, extras


def deterministic_windows(
    *,
    x_raw: np.ndarray,
    y_raw: np.ndarray,
    n_windows: int,
    window_t: int,
    stride: int,
    start_window: int = 0,
    dataset: str,
    env_var: str,
    context: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_windows = int(n_windows)
    window_t = int(window_t)
    stride = int(stride)
    start_window = int(start_window)
    if n_windows < 0:
        raise ValueError(f"n_windows must be >=0, got {n_windows}")
    if window_t <= 0:
        raise ValueError(f"window_t must be >0, got {window_t}")
    if stride <= 0:
        raise ValueError(f"stride must be >0, got {stride}")
    if start_window < 0:
        raise ValueError(f"start_window must be >=0, got {start_window}")

    x = np.asarray(x_raw, dtype=np.float64)
    y = np.asarray(y_raw, dtype=np.float64)
    if x.ndim != y.ndim:
        raise DatasetMissingError(
            dataset=dataset,
            env_var=env_var,
            message=f"{context}: x/y rank mismatch x.shape={x.shape}, y.shape={y.shape}",
        )

    if n_windows == 0:
        if x.ndim == 3:
            x_dim = int(x.shape[2])
            y_dim = int(y.shape[2])
        elif x.ndim == 2:
            x_dim = int(x.shape[1])
            y_dim = int(y.shape[1])
        else:
            raise DatasetMissingError(
                dataset=dataset,
                env_var=env_var,
                message=f"{context}: unsupported rank for zero-window extraction x.shape={x.shape}, y.shape={y.shape}",
            )
        x_empty = np.zeros((0, window_t, x_dim), dtype=np.float32)
        y_empty = np.zeros((0, window_t, y_dim), dtype=np.float32)
        starts = np.zeros((0,), dtype=np.int64)
        return x_empty, y_empty, starts

    if x.ndim == 3:
        if x.shape[0] != y.shape[0]:
            raise DatasetMissingError(
                dataset=dataset,
                env_var=env_var,
                message=f"{context}: x/y sequence-count mismatch x.shape={x.shape}, y.shape={y.shape}",
            )
        if x.shape[1] < window_t or y.shape[1] < window_t:
            raise DatasetMissingError(
                dataset=dataset,
                env_var=env_var,
                message=f"{context}: sequence length shorter than window_t={window_t}. x.shape={x.shape}, y.shape={y.shape}",
            )
        end = start_window + n_windows
        if end > int(x.shape[0]):
            raise DatasetMissingError(
                dataset=dataset,
                env_var=env_var,
                message=(
                    f"{context}: not enough pre-windowed sequences for requested split. "
                    f"needed={end}, available={x.shape[0]}"
                ),
            )
        x_out = np.asarray(x[start_window:end, :window_t, :], dtype=np.float32)
        y_out = np.asarray(y[start_window:end, :window_t, :], dtype=np.float32)
        starts = np.arange(start_window, end, dtype=np.int64)
        return x_out, y_out, starts

    if x.ndim != 2:
        raise DatasetMissingError(
            dataset=dataset,
            env_var=env_var,
            message=f"{context}: expected rank-2/3 arrays, got x.shape={x.shape}, y.shape={y.shape}",
        )
    if x.shape[0] != y.shape[0]:
        raise DatasetMissingError(
            dataset=dataset,
            env_var=env_var,
            message=f"{context}: x/y time-length mismatch x.shape={x.shape}, y.shape={y.shape}",
        )

    total_t = int(x.shape[0])
    max_start = total_t - window_t
    if max_start < 0:
        raise DatasetMissingError(
            dataset=dataset,
            env_var=env_var,
            message=f"{context}: total steps ({total_t}) shorter than requested window_t ({window_t})",
        )
    all_starts = np.arange(0, max_start + 1, stride, dtype=np.int64)
    end = start_window + n_windows
    if end > int(all_starts.shape[0]):
        raise DatasetMissingError(
            dataset=dataset,
            env_var=env_var,
            message=(
                f"{context}: not enough windows. requested end_index={end}, "
                f"available_windows={all_starts.shape[0]}, total_steps={total_t}, "
                f"window_t={window_t}, stride={stride}"
            ),
        )
    starts = np.asarray(all_starts[start_window:end], dtype=np.int64)
    x_out = np.stack([x[s : s + window_t, :] for s in starts], axis=0).astype(np.float32, copy=False)
    y_out = np.stack([y[s : s + window_t, :] for s in starts], axis=0).astype(np.float32, copy=False)
    return x_out, y_out, starts


def deep_merge(base: Mapping[str, Any], update: Mapping[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in update.items():
        if k in out and isinstance(out[k], Mapping) and isinstance(v, Mapping):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out
