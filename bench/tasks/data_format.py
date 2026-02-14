from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Optional, Dict, Literal
from pathlib import Path

import numpy as np



@dataclass(frozen=True)
class AdaptationBudget:
    """
    Budget for online adaptation (Budgeted track).

    Mirrors suite YAML runner.tracks[*].adaptation_budget keys.
    (SoT: /mnt/data/suite_basic.yaml, /mnt/data/suite_shift.yaml, /mnt/data/FAIRNESS.md)
    """
    max_updates: int = 0
    max_updates_per_step: int = 0
    allowed_after_t0_only: bool = False


@dataclass(frozen=True)
class SystemInfo:
    """
    Task-provided system model info.

    NOTE:
    - Different repos/models may assume different info access (F/H vs black-box).
    - The bench may store full info, but adapters can restrict exposure.
      (SoT: /mnt/data/FAIRNESS.md '모델이 사용할 수 있는 정보 범위')
    """
    system_type: str  # "linear" | "nonlinear" | "real"
    x_dim: Optional[int] = None
    y_dim: Optional[int] = None
    has_gt: bool = True
    provides_F: bool = False
    provides_H: bool = False
    provides_f: bool = False
    provides_h: bool = False
    provides_Q: bool = False
    provides_R: bool = False
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DatasetArtifacts:
    """
    Dataset artifacts handle.

    Step 2: define schema only.
    Step 4~: bench_generated will materialize actual files in a known format.

    Recommended on-disk convention (not implemented yet):
    - <root_dir>/
        meta.json
        train.npz
        val.npz
        test.npz

    Each split .npz may include:
    - y_seq: observation sequences
    - x_gt: ground truth (optional)
    - u_seq: control inputs (optional)
    - t0 / shift meta (shift tasks)
    """
    task_id: str
    scenario_id: str
    seed: int
    root_dir: str

    meta_path: str
    train_path: str
    val_path: str
    test_path: str

    has_gt: bool = True
    t0: Optional[int] = None  # shift tasks only


@dataclass(frozen=True)
class RunSpec:
    """
    Fully resolved run specification (one row in the execution plan).
    """
    suite_name: str
    suite_version: str
    task_id: str
    model_id: str
    track_id: str
    seed: int
    scenario_id: str

    # Optional pointers to resolved configs (kept as dicts to avoid tight coupling)
    task_cfg: dict[str, Any] = field(default_factory=dict)
    model_cfg: dict[str, Any] = field(default_factory=dict)
    runner_cfg: dict[str, Any] = field(default_factory=dict)

CANONICAL_LAYOUT_V0 = "NTD"  # N(seqs), T(time), D(dim). 저장/로더 기준 (D15)

@dataclass(frozen=True)
class DatasetSplitV0:
    path: Path
    split: Literal["train", "val", "test"]

@dataclass(frozen=True)
class DatasetArtifactsV0:
    format_version: str
    canonical_layout: str
    suite_name: str
    task_id: str
    scenario_id: str
    seed: int
    cache_dir: Path
    train: DatasetSplitV0
    val: DatasetSplitV0
    test: DatasetSplitV0
    meta_common: Dict[str, Any]

@dataclass(frozen=True)
class LoadedSplitV0:
    x: np.ndarray
    y: np.ndarray
    u: Optional[np.ndarray]
    F: Optional[np.ndarray]
    H: Optional[np.ndarray]
    meta: Dict[str, Any]

def dump_meta_json_v0(meta: Dict[str, Any]) -> np.ndarray:
    """
    np.savez에 넣기 위한 meta 직렬화.
    NOTE: meta는 결정론을 위해 timestamp 등 가변 필드를 넣지 않는 것을 권장.
    """
    s = json.dumps(meta, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return np.array(s, dtype=np.str_)

def load_meta_json_v0(meta_json: np.ndarray) -> Dict[str, Any]:
    s = str(meta_json.tolist()) if hasattr(meta_json, "tolist") else str(meta_json)
    return json.loads(s)

def save_npz_split_v0(
    *,
    path: Path,
    x: np.ndarray,
    y: np.ndarray,
    u: Optional[np.ndarray],
    F: Optional[np.ndarray],
    H: Optional[np.ndarray],
    meta: Dict[str, Any],
) -> None:
    """
    v0 포맷: npz keys
      - x: [N,T,x_dim] float32
      - y: [N,T,y_dim] float32
      - (optional) u: [N,T,u_dim] float32
      - (optional) F: [x_dim,x_dim] float32
      - (optional) H: [y_dim,x_dim] float32
      - meta_json: JSON string
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    #tmp = path.with_suffix(path.suffix + ".tmp")
    tmp = path.with_name(f"{path.stem}.tmp{path.suffix}")

    kwargs = {
        "x": x.astype(np.float32, copy=False),
        "y": y.astype(np.float32, copy=False),
        "meta_json": dump_meta_json_v0(meta),
    }
    if u is not None:
        kwargs["u"] = u.astype(np.float32, copy=False)
    if F is not None:
        kwargs["F"] = F.astype(np.float32, copy=False)
    if H is not None:
        kwargs["H"] = H.astype(np.float32, copy=False)

    np.savez_compressed(tmp, **kwargs)
    tmp.replace(path)

def load_npz_split_v0(path: Path) -> LoadedSplitV0:
    with np.load(path, allow_pickle=False) as z:
        x = z["x"].astype(np.float32, copy=False)
        y = z["y"].astype(np.float32, copy=False)
        u = z["u"].astype(np.float32, copy=False) if "u" in z.files else None
        F = z["F"].astype(np.float32, copy=False) if "F" in z.files else None
        H = z["H"].astype(np.float32, copy=False) if "H" in z.files else None
        meta = load_meta_json_v0(z["meta_json"])
    return LoadedSplitV0(x=x, y=y, u=u, F=F, H=H, meta=meta)
