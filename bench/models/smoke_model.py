"""
Smoke test for Step 5: bench_generated NPZ -> adapter -> forward 1 batch

This module intentionally avoids importing third_party scripts directly.
It:
1) reads suite YAML to locate model repo path
2) loads a bench_generated NPZ from cache
3) creates adapter by model_id registry
4) calls predict once and prints output shape/dtype

Run:
  python -m bench.models.smoke_model \
    --suite-yaml /mnt/data/suite_shift.yaml \
    --task C_shift_Rscale_v0 \
    --model-id kalmannet_tsp \
    --seed 0 \
    --split test \
    --batch-size 8

If your cache root differs:
  export BENCH_DATA_CACHE=/home/xynus/bench_data_cache
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import yaml

from .registry import get_adapter_class


_LOG = logging.getLogger(__name__)


def _default_cache_root() -> Path:
    env = os.environ.get("BENCH_DATA_CACHE")
    if env:
        return Path(env).expanduser().resolve()
    # bench_data_cache next to bench root (best-effort)
    # This file: .../bench/bench/models/smoke_model.py -> bench root is parents[2]
    bench_root = Path(__file__).resolve().parents[2]
    return (bench_root / "bench_data_cache").resolve()


def _load_suite_yaml(path: Path) -> Dict[str, Any]:
    try:
        return yaml.safe_load(path.read_text())
    except Exception as e:
        raise RuntimeError(
            f"Failed to parse suite YAML: {path}\n"
            f"YAML error: {e}\n"
            "Tip: validate indentation in suite.description block scalar."
        ) from e


def _find_model_cfg(suite: Dict[str, Any], model_id: str) -> Dict[str, Any]:
    for m in suite.get("models", []) or []:
        if m.get("model_id") == model_id:
            return m
    raise KeyError(f"model_id={model_id} not found in suite.models")


def _pick_scenario_dir(task_dir: Path, scenario_id: Optional[str]) -> Path:
    if scenario_id:
        d = task_dir / f"scenario_{scenario_id}" if not str(scenario_id).startswith("scenario_") else task_dir / scenario_id
        if not d.exists():
            raise FileNotFoundError(f"Scenario dir not found: {d}")
        return d

    # pick first scenario_* sorted
    cands = sorted([p for p in task_dir.iterdir() if p.is_dir() and p.name.startswith("scenario_")])
    if not cands:
        raise FileNotFoundError(f"No scenario_* dirs under: {task_dir}")
    return cands[0]


def _load_npz(cache_root: Path, suite_name: str, task_id: str, seed: int, split: str, scenario_id: Optional[str]) -> Dict[str, Any]:
    task_dir = cache_root / suite_name / task_id
    if not task_dir.exists():
        raise FileNotFoundError(
            f"Task cache dir not found: {task_dir}\n"
            f"Tip: generate data first (smoke_data) or set BENCH_DATA_CACHE correctly."
        )

    scen_dir = _pick_scenario_dir(task_dir, scenario_id)
    seed_dir = scen_dir / f"seed_{seed}"
    p = seed_dir / f"{split}.npz"
    if not p.exists():
        raise FileNotFoundError(f"NPZ not found: {p}")

    d = np.load(p, allow_pickle=True)
    meta_json = d.get("meta_json", None)
    if meta_json is None:
        raise KeyError("NPZ missing meta_json")
    mj = meta_json.item() if hasattr(meta_json, "item") else meta_json
    if isinstance(mj, (bytes, np.bytes_)):
        mj = mj.decode("utf-8")
    meta = json.loads(mj)

    out = {
        "path": str(p),
        "x": d.get("x", None),   # [N,T,x]
        "y": d.get("y", None),   # [N,T,y]
        "F": d.get("F", None),   # optional
        "H": d.get("H", None),   # optional
        "meta": meta,
    }
    if out["y"] is None:
        raise KeyError("NPZ missing y")
    return out


def _make_batch(y: np.ndarray, batch_size: int) -> torch.Tensor:
    # y: [N,T,y_dim]
    N = y.shape[0]
    b = min(batch_size, N)
    yb = y[:b]
    return torch.as_tensor(yb, dtype=torch.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite-yaml", required=True, type=str)
    ap.add_argument("--task", required=True, type=str)
    ap.add_argument("--model-id", required=True, type=str)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--scenario-id", type=str, default=None)  # raw id or scenario_*
    args = ap.parse_args()

    suite_path = Path(args.suite_yaml).expanduser().resolve()
    suite = _load_suite_yaml(suite_path)
    suite_name = suite.get("suite", {}).get("name", None)
    if not suite_name:
        raise KeyError("suite.suite.name missing in YAML")

    # load model cfg (for repo path/device hints)
    model_cfg = _find_model_cfg(suite, args.model_id)

    cache_root = _default_cache_root()
    npz = _load_npz(cache_root, suite_name, args.task, args.seed, args.split, args.scenario_id)

    y = npz["y"]
    meta = npz["meta"]
    F = npz["F"]
    H = npz["H"]

    T = int(meta.get("T") or meta.get("sequence_length_T") or y.shape[1])
    x_dim = int(meta.get("x_dim") or (npz["x"].shape[-1] if npz["x"] is not None else H.shape[1] if H is not None else -1))
    y_dim = int(meta.get("y_dim") or y.shape[-1])

    y_batch = _make_batch(y, args.batch_size)

    # adapter instantiation
    AdapterCls = get_adapter_class(args.model_id)
    adapter = AdapterCls()

    # Build system_info for setup/predict
    system_info = {
        "F": F,
        "H": H,
        "T": T,
        "x_dim": x_dim,
        "y_dim": y_dim,
        "meta": meta,
    }

    print(f"[smoke_model] cache_root={cache_root}")
    print(f"[smoke_model] npz={npz['path']}")
    print(f"[smoke_model] model_id={args.model_id} repo_path={(model_cfg.get('repo') or {}).get('path')}")
    print(f"[smoke_model] y_batch shape={tuple(y_batch.shape)} (bench NTD)")

    adapter.setup(model_cfg, system_info)

    # predict once
    context = {"T": T, "meta": meta}
    x_hat = adapter.predict(y_batch, context=context, return_cov=False)

    if isinstance(x_hat, (tuple, list)):
        x_hat = x_hat[0]

    if not isinstance(x_hat, torch.Tensor):
        raise TypeError(f"Adapter returned non-tensor: {type(x_hat)}")

    print(f"[smoke_model] x_hat shape={tuple(x_hat.shape)} dtype={x_hat.dtype} device={x_hat.device}")

    # Optional quick MSE if x exists
    if npz["x"] is not None:
        xb = torch.as_tensor(npz["x"][: x_hat.shape[0]], dtype=torch.float32, device=x_hat.device)  # [B,T,x]
        if xb.shape == x_hat.shape:
            mse = torch.mean((xb - x_hat) ** 2).item()
            print(f"[smoke_model] quick_mse={mse:.6e} (for sanity only)")
        else:
            print(f"[smoke_model] quick_mse skipped (x shape={tuple(xb.shape)} != x_hat shape={tuple(x_hat.shape)})")


if __name__ == "__main__":
    main()

