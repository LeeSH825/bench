from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Dict, Mapping

import numpy as np

from .schema import missing_required_meta_paths


@dataclass(frozen=True)
class DeterminismFingerprint:
    x_first_k_sha256: str
    y_first_k_sha256: str
    meta_required_sha256: str
    k: int


def validate_artifacts(x: np.ndarray, y: np.ndarray, meta: Mapping[str, Any], *, strict: bool = True) -> None:
    if not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray):
        raise TypeError("x and y must be numpy arrays")
    if x.ndim != 3 or y.ndim != 3:
        raise ValueError(f"x/y must be rank-3 [N,T,D], got x={x.shape}, y={y.shape}")
    if x.shape[0] != y.shape[0] or x.shape[1] != y.shape[1]:
        raise ValueError(f"x/y must share N,T, got x={x.shape}, y={y.shape}")
    if strict and x.dtype != np.float32:
        raise TypeError(f"x must be float32 in strict mode, got {x.dtype}")
    if strict and y.dtype != np.float32:
        raise TypeError(f"y must be float32 in strict mode, got {y.dtype}")

    missing = missing_required_meta_paths(meta)
    if missing:
        raise ValueError(f"meta missing required TG0 keys: {missing}")

    dims = meta.get("dims", {})
    if not isinstance(dims, Mapping):
        raise TypeError("meta.dims must be a mapping")
    md_x = int(dims.get("x_dim"))
    md_y = int(dims.get("y_dim"))
    md_t = int(dims.get("T"))
    if x.shape[2] != md_x or y.shape[2] != md_y or x.shape[1] != md_t or y.shape[1] != md_t:
        raise ValueError(
            "meta dims mismatch with tensors: "
            f"meta(x_dim={md_x}, y_dim={md_y}, T={md_t}) vs x={x.shape}, y={y.shape}"
        )


def _sha256_first_k_values(arr: np.ndarray, k: int) -> str:
    flat = np.ascontiguousarray(arr.reshape(-1))
    used = min(int(k), int(flat.size))
    sample = np.ascontiguousarray(flat[:used].astype(np.float32, copy=False))
    return hashlib.sha256(sample.tobytes()).hexdigest()


def _required_meta_projection(meta: Mapping[str, Any]) -> Dict[str, Any]:
    keys = [
        "schema_version",
        "task_family",
        "dims",
        "splits",
        "ssm",
        "mismatch",
        "noise_schedule",
        "switching",
    ]
    return {k: meta.get(k) for k in keys}


def determinism_fingerprint(x: np.ndarray, y: np.ndarray, meta: Mapping[str, Any], *, k: int = 256) -> DeterminismFingerprint:
    proj = _required_meta_projection(meta)
    meta_txt = json.dumps(proj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return DeterminismFingerprint(
        x_first_k_sha256=_sha256_first_k_values(x, k),
        y_first_k_sha256=_sha256_first_k_values(y, k),
        meta_required_sha256=hashlib.sha256(meta_txt.encode("utf-8")).hexdigest(),
        k=int(k),
    )
