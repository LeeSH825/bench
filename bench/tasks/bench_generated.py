from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .generators.linear import (
    build_linear_system_matrices_v0,
    generate_linear_gaussian_sequences_v0,
)
from .data_format import (
    CANONICAL_LAYOUT_V0,
    DatasetArtifactsV0,
    DatasetSplitV0,
    load_npz_split_v0,
    save_npz_split_v0,
)
from ..utils.seeding import stable_int_seed_v0, numpy_rng_v0
from ..utils.io import ensure_dir


def _repo_root_from_here() -> Path:
    # .../bench/bench/tasks/bench_generated.py -> parents[2] == repo root (.../bench)
    return Path(__file__).resolve().parents[2]


def default_cache_root() -> Path:
    env = os.environ.get("BENCH_DATA_CACHE", "").strip()
    if env:
        return Path(env).expanduser().resolve()
    return _repo_root_from_here() / "bench_data_cache"


def canonicalize_scenario_id(task_id: str, scenario_cfg: Dict[str, Any]) -> str:
    payload = {"task_id": task_id, "scenario": scenario_cfg}
    s = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    h = hashlib.sha1(s.encode("utf-8")).hexdigest()
    return h[:12]


def expand_scenarios_from_sweep(task_cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Supports flat sweep keys like:
      sweep:
        shift.post_shift.R_scale: [3.0, 10.0, 30.0]
    Returns list of nested dict scenario_cfg.
    """
    sweep = task_cfg.get("sweep") or {}
    if not sweep:
        return [{}]

    keys = list(sweep.keys())
    values_list: List[List[Any]] = [list(sweep[k]) for k in keys]

    scenarios: List[Dict[str, Any]] = []

    def set_deep(d: Dict[str, Any], dotted_key: str, val: Any) -> None:
        cur = d
        parts = dotted_key.split(".")
        for p in parts[:-1]:
            if p not in cur or not isinstance(cur[p], dict):
                cur[p] = {}
            cur = cur[p]
        cur[parts[-1]] = val

    def rec(i: int, base: Dict[str, Any]) -> None:
        if i == len(keys):
            scenarios.append(base)
            return
        k = keys[i]
        for v in values_list[i]:
            nxt = json.loads(json.dumps(base))
            set_deep(nxt, k, v)
            rec(i + 1, nxt)

    rec(0, {})
    return scenarios


def merge_scenario_overrides(base: Dict[str, Any], overrides: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not overrides:
        return base

    def deep_merge(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
        out = dict(a)
        for k, v in b.items():
            if k in out and isinstance(out[k], dict) and isinstance(v, dict):
                out[k] = deep_merge(out[k], v)
            else:
                out[k] = v
        return out

    return deep_merge(base, overrides)


def _task_cache_dir(cache_root: Path, suite_name: str, task_id: str, scenario_id: str, seed: int) -> Path:
    return cache_root / suite_name / task_id / f"scenario_{scenario_id}" / f"seed_{seed}"


def _maybe_generate_and_cache_one(
    *,
    cache_root: Path,
    suite_name: str,
    task_cfg: Dict[str, Any],
    task_id: str,
    scenario_cfg: Dict[str, Any],
    seed: int,
) -> DatasetArtifactsV0:
    """
    Create (train/val/test).npz if missing; otherwise load.
    Deterministic for (suite_name, task_id, scenario_cfg, seed).
    """
    scenario_id = canonicalize_scenario_id(task_id, scenario_cfg)
    out_dir = _task_cache_dir(cache_root, suite_name, task_id, scenario_id, seed)
    ensure_dir(out_dir)

    train_path = out_dir / "train.npz"
    val_path = out_dir / "val.npz"
    test_path = out_dir / "test.npz"

    if train_path.exists() and val_path.exists() and test_path.exists():
        meta = load_npz_split_v0(train_path).meta
        return DatasetArtifactsV0(
            format_version="0.1",
            canonical_layout=CANONICAL_LAYOUT_V0,
            suite_name=suite_name,
            task_id=task_id,
            scenario_id=scenario_id,
            seed=seed,
            cache_dir=out_dir,
            train=DatasetSplitV0(path=train_path, split="train"),
            val=DatasetSplitV0(path=val_path, split="val"),
            test=DatasetSplitV0(path=test_path, split="test"),
            meta_common={k: v for k, v in meta.items() if k != "split"},
        )

    # --- Generate deterministically ---
    x_dim = int(task_cfg["x_dim"])
    y_dim = int(task_cfg["y_dim"])
    T = int(task_cfg["sequence_length_T"])
    sizes = task_cfg["dataset_sizes"]
    n_train = int(sizes["N_train"])
    n_val = int(sizes["N_val"])
    n_test = int(sizes["N_test"])
    n_total = n_train + n_val + n_test

    obs = task_cfg.get("observation", {})
    H_spec = obs.get("H", "identity")

    noise = task_cfg["noise"]
    is_shift = ("pre_shift" in noise) and ("shift" in noise)

    if is_shift:
        q2 = float(noise["pre_shift"]["Q"]["q2"])
        r2 = float(noise["pre_shift"]["R"]["r2"])
        t0 = int(noise["shift"]["t0"])
        post = noise["shift"].get("post_shift", {})

        # applied R_scale: scenario overrides win
        r_scale = float(
            scenario_cfg.get("shift", {})
                       .get("post_shift", {})
                       .get("R_scale", post.get("R_scale", 1.0))
        )

        # post-shift distribution (MVP: gaussian only; dist shift task is enabled:false)
        obs_dist = post.get("obs_distribution", {"name": "gaussian"})
        dist_name = str(obs_dist.get("name", "gaussian"))
        dist_params = dict(obs_dist)
    else:
        q2 = float(noise["Q"]["q2"])
        r2 = float(noise["R"]["r2"])
        t0 = None
        r_scale = 1.0
        dist_name = "gaussian"
        dist_params = {"name": "gaussian"}

    # deterministic matrices for the system
    sys_seed = stable_int_seed_v0("system", suite_name, task_id, scenario_id, seed)
    rng_sys = numpy_rng_v0(sys_seed)
    F, H = build_linear_system_matrices_v0(rng=rng_sys, x_dim=x_dim, y_dim=y_dim, H_spec=str(H_spec))

    # deterministic sequence generation
    data_seed = stable_int_seed_v0("data", suite_name, task_id, scenario_id, seed)
    rng_data = numpy_rng_v0(data_seed)

    x_all, y_all = generate_linear_gaussian_sequences_v0(
        rng=rng_data,
        n_seq=n_total,
        T=T,
        F=F,
        H=H,
        q2=q2,
        r2=r2,
        t0_shift=t0,
        r_scale_post=r_scale,
        obs_dist_name=dist_name,
        obs_dist_params=dist_params,
    )

    # deterministic split permutation
    split_seed = stable_int_seed_v0("split", suite_name, task_id, scenario_id, seed)
    rng_split = numpy_rng_v0(split_seed)
    perm = rng_split.permutation(n_total)

    idx_train = perm[:n_train]
    idx_val = perm[n_train : n_train + n_val]
    idx_test = perm[n_train + n_val :]

    # meta: include scenario_cfg + applied shift params (R_scale must be readable)
    noise_meta = json.loads(json.dumps(noise))  # deep copy (json-safe)
    if is_shift:
        noise_meta.setdefault("shift", {})
        noise_meta["shift"]["t0"] = int(t0)
        noise_meta.setdefault("shift", {}).setdefault("post_shift", {})
        noise_meta["shift"]["post_shift"]["R_scale"] = float(r_scale)  # applied!
        noise_meta["shift"]["post_shift"]["obs_distribution"] = dict(dist_params)

    meta_common: Dict[str, Any] = {
        "format_version": "0.1",
        "canonical_layout": CANONICAL_LAYOUT_V0,
        "suite_name": suite_name,
        "task_id": task_id,
        "scenario_id": scenario_id,
        "scenario_cfg": scenario_cfg,
        "seed": int(seed),
        "x_dim": int(x_dim),
        "y_dim": int(y_dim),
        "T": int(T),
        "control_input_u": bool(task_cfg.get("control_input_u", False)),
        "ground_truth": dict(task_cfg.get("ground_truth", {"has_gt": True})),
        "observation": dict(obs),
        "noise": noise_meta,
    }

    # Save per split (npz). Store F/H (recommended for canonical checks).
    def save_split(path: Path, split: str, idx: np.ndarray) -> None:
        meta = dict(meta_common)
        meta["split"] = split
        save_npz_split_v0(
            path=path,
            x=x_all[idx].astype(np.float32),
            y=y_all[idx].astype(np.float32),
            u=None,
            F=F.astype(np.float32),
            H=H.astype(np.float32),
            meta=meta,
        )

    save_split(train_path, "train", idx_train)
    save_split(val_path, "val", idx_val)
    save_split(test_path, "test", idx_test)

    return DatasetArtifactsV0(
        format_version="0.1",
        canonical_layout=CANONICAL_LAYOUT_V0,
        suite_name=suite_name,
        task_id=task_id,
        scenario_id=scenario_id,
        seed=seed,
        cache_dir=out_dir,
        train=DatasetSplitV0(path=train_path, split="train"),
        val=DatasetSplitV0(path=val_path, split="val"),
        test=DatasetSplitV0(path=test_path, split="test"),
        meta_common=meta_common,
    )


@dataclass(frozen=True)
class LoaderCfgV0:
    batch_size: int = 32
    num_workers: int = 0
    pin_memory: bool = False
    shuffle_train: bool = True
    drop_last: bool = False


class BenchNpzDatasetV0(Dataset):
    """
    Loads a single split .npz with canonical layout:
      x: [N,T,x_dim], y: [N,T,y_dim] (float32)
    """
    def __init__(self, npz_path: Path):
        loaded = load_npz_split_v0(npz_path)
        self._x = loaded.x
        self._y = loaded.y
        self._u = loaded.u
        self.meta = loaded.meta
        self.path = npz_path

    def __len__(self) -> int:
        return int(self._y.shape[0])

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        x = torch.from_numpy(self._x[idx])
        y = torch.from_numpy(self._y[idx])
        if self._u is None:
            return {"x": x, "y": y}
        u = torch.from_numpy(self._u[idx])
        return {"x": x, "y": y, "u": u}


def _collate_with_meta_v0(samples: List[Dict[str, torch.Tensor]], meta: Dict[str, Any]) -> Dict[str, Any]:
    x = torch.stack([s["x"] for s in samples], dim=0)  # [B,T,x_dim]
    y = torch.stack([s["y"] for s in samples], dim=0)  # [B,T,y_dim]
    u = None
    if "u" in samples[0]:
        u = torch.stack([s["u"] for s in samples], dim=0)
    return {"x": x, "y": y, "u": u, "meta": meta}


def make_dataloaders_v0(
    artifacts: DatasetArtifactsV0,
    loader_cfg: Optional[LoaderCfgV0] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    cfg = loader_cfg or LoaderCfgV0()

    ds_train = BenchNpzDatasetV0(artifacts.train.path)
    ds_val = BenchNpzDatasetV0(artifacts.val.path)
    ds_test = BenchNpzDatasetV0(artifacts.test.path)

    train_loader = DataLoader(
        ds_train,
        batch_size=cfg.batch_size,
        shuffle=cfg.shuffle_train,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        drop_last=cfg.drop_last,
        collate_fn=lambda s: _collate_with_meta_v0(s, ds_train.meta),
    )
    val_loader = DataLoader(
        ds_val,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        drop_last=False,
        collate_fn=lambda s: _collate_with_meta_v0(s, ds_val.meta),
    )
    test_loader = DataLoader(
        ds_test,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        drop_last=False,
        collate_fn=lambda s: _collate_with_meta_v0(s, ds_test.meta),
    )
    return train_loader, val_loader, test_loader


def log_first_batch_v0(loader: DataLoader) -> Dict[str, Any]:
    batch = next(iter(loader))
    x = batch["x"]
    y = batch["y"]
    u = batch.get("u", None)
    info = {
        "x.shape": tuple(x.shape),
        "y.shape": tuple(y.shape),
        "x.dtype": str(x.dtype),
        "y.dtype": str(y.dtype),
        "x.device": str(x.device),
        "y.device": str(y.device),
        "u": None if u is None else {"shape": tuple(u.shape), "dtype": str(u.dtype), "device": str(u.device)},
    }
    return info


def prepare_bench_generated_v0(
    *,
    suite_name: str,
    task_cfg: Dict[str, Any],
    seed: int,
    cache_root: Optional[Path] = None,
    scenario_overrides: Optional[Dict[str, Any]] = None,
) -> List[DatasetArtifactsV0]:
    """
    Generate/cache datasets for (task_cfg + sweep grid) for one seed.
    Returns list of artifacts (one per scenario).
    """
    cache_root = cache_root or default_cache_root()
    task_id = str(task_cfg["task_id"])

    scenarios = expand_scenarios_from_sweep(task_cfg)
    out: List[DatasetArtifactsV0] = []
    for base_s in scenarios:
        scenario_cfg = merge_scenario_overrides(base_s, scenario_overrides)
        art = _maybe_generate_and_cache_one(
            cache_root=cache_root,
            suite_name=suite_name,
            task_cfg=task_cfg,
            task_id=task_id,
            scenario_cfg=scenario_cfg,
            seed=seed,
        )
        out.append(art)
    return out
