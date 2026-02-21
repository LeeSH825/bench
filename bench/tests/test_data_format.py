from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np


@dataclass
class DataFormatResult:
    ok: bool
    npz_path: Path
    note: str


def _find_one_test_npz(cache_root: Path, suite_name: str, task_id: str, seed: int) -> Optional[Path]:
    task_root = cache_root / suite_name / task_id
    if not task_root.exists():
        # best effort scan: <cache_root>/*/<task_id>
        found = list(cache_root.glob(f"*/{task_id}"))
        if found:
            task_root = found[0]
        else:
            return None

    scen_dirs = sorted(task_root.glob("scenario_*"))
    if not scen_dirs:
        return None
    npz = scen_dirs[0] / f"seed_{seed}" / "test.npz"
    return npz if npz.exists() else None


def run_data_format_check(cache_root: Path, suite_name: str, task_id: str, seed: int) -> DataFormatResult:
    npz_path = _find_one_test_npz(cache_root, suite_name, task_id, seed)
    if npz_path is None:
        return DataFormatResult(ok=False, npz_path=Path(""), note=f"no test.npz found under cache_root={cache_root}")

    try:
        d = np.load(npz_path, allow_pickle=True)
    except Exception as e:
        return DataFormatResult(ok=False, npz_path=npz_path, note=f"failed to load npz: {type(e).__name__}: {e}")

    # required keys
    required = ["x", "y", "meta_json"]
    for k in required:
        if k not in d:
            return DataFormatResult(ok=False, npz_path=npz_path, note=f"missing required key: {k}")

    x = d["x"]
    y = d["y"]

    if x.ndim != 3 or y.ndim != 3:
        return DataFormatResult(ok=False, npz_path=npz_path, note=f"x/y must be 3D [N,T,D]. got x={x.shape}, y={y.shape}")

    if x.shape[0] != y.shape[0] or x.shape[1] != y.shape[1]:
        return DataFormatResult(ok=False, npz_path=npz_path, note=f"x/y N,T mismatch. x={x.shape}, y={y.shape}")

    # meta_json parse
    try:
        meta = json.loads(d["meta_json"].item())
    except Exception as e:
        return DataFormatResult(ok=False, npz_path=npz_path, note=f"meta_json not parseable: {type(e).__name__}: {e}")

    # meta required hints
    for k in ("task_id", "seed", "x_dim", "y_dim", "T"):
        if k not in meta:
            # some implementations might nest; keep this as warning-level but still fail per spec
            return DataFormatResult(ok=False, npz_path=npz_path, note=f"meta_json missing key: {k}")

    # check shapes vs meta
    T = int(meta.get("T"))
    x_dim = int(meta.get("x_dim"))
    y_dim = int(meta.get("y_dim"))
    if x.shape[1] != T or y.shape[1] != T:
        return DataFormatResult(ok=False, npz_path=npz_path, note=f"T mismatch meta={T}, npz x.T={x.shape[1]}, y.T={y.shape[1]}")
    if x.shape[2] != x_dim or y.shape[2] != y_dim:
        return DataFormatResult(ok=False, npz_path=npz_path, note=f"dim mismatch meta x_dim={x_dim}, y_dim={y_dim} vs npz x={x.shape}, y={y.shape}")

    return DataFormatResult(ok=True, npz_path=npz_path, note="npz format OK (x,y,meta_json + NTD shapes)")
