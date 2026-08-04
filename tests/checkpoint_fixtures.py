"""Tiny deterministic fixtures shared by the checkpoint/resume tests.

Not a test module. Everything here is self-contained: no repository run, no
production registry, and no generated dataset is read, so these tests cannot
mutate tracked output (the V-008 hazard).
"""

from __future__ import annotations

import hashlib
import random
from typing import Any

import numpy as np
import torch

#: The certified envelope (ADR-CSR-002).
CPU_DETERMINISTIC = {
    "device": "cpu",
    "precision": "fp32",
    "num_workers": 0,
}


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def configure_threads() -> None:
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        # Only settable before any parallel work has started.
        pass


def tiny_fixture(model_id: str) -> tuple[dict[str, Any], dict[str, Any], np.ndarray, np.ndarray]:
    """A 6-sequence problem small enough to train in well under a second."""
    n_seq = 6
    n_step = 7 if model_id == "kalmannet_tsp" else 8
    x_dim = y_dim = 2

    t = np.arange(n_step, dtype=np.float32)
    y = np.empty((n_seq, n_step, y_dim), dtype=np.float32)
    x = np.zeros((n_seq, n_step, x_dim), dtype=np.float32)
    for b in range(n_seq):
        y[b, :, 0] = np.sin(np.float32(0.2) * t + np.float32(b) * np.float32(0.1))
        y[b, :, 1] = np.cos(np.float32(0.3) * t - np.float32(b) * np.float32(0.05))
    x[:, :, 0] = np.float32(0.1)
    x[:, :, 1] = np.float32(-0.2)

    if model_id == "kalmannet_tsp":
        matrices = {
            "F": np.asarray([[0.9, 0.1], [0.0, 0.8]], dtype=np.float32),
            "H": np.asarray([[1.0, 0.2], [-0.1, 1.0]], dtype=np.float32),
        }
        cfg: dict[str, Any] = {
            "model_id": model_id,
            "repo": {"path": "third_party/KalmanNet_TSP"},
        }
    else:
        matrices = {
            "F": np.asarray([[0.95, 0.1], [0.0, 0.9]], dtype=np.float32),
            "H": np.asarray([[1.0, 0.2], [-0.15, 0.8]], dtype=np.float32),
        }
        cfg = {
            "model_id": model_id,
            "repo": {"path": "third_party/Split_KalmanNet"},
            "estimator_class_path": "GSSFiltering.filtering.Split_KalmanNet_Filter",
            "input_layout": "BTD",
            "eval_init_from_gt": False,
        }

    system_info = {
        "x_dim": x_dim,
        "y_dim": y_dim,
        "T": n_step,
        **matrices,
        "Q": np.eye(x_dim, dtype=np.float32) * np.float32(1e-3),
        "R": np.eye(y_dim, dtype=np.float32) * np.float32(2e-3),
    }
    return cfg, system_info, x, y


def build_adapter(model_id: str, seed: int):
    """Construct and set up a real adapter against real third-party code."""
    from bench.models.kalmannet_tsp import KalmanNetTSPAdapter
    from bench.models.split_knet import SplitKNetAdapter

    configure_threads()
    seed_all(seed)
    cfg, system_info, x, y = tiny_fixture(model_id)
    adapter = KalmanNetTSPAdapter() if model_id == "kalmannet_tsp" else SplitKNetAdapter()
    adapter.setup(cfg, system_info, {"seed": seed, "deterministic": True, "device": "cpu"})
    val_batches = [{"x": x[:2].copy(), "y": y[:2].copy()}]
    adapter.begin_resumable_training(
        train_x=x, train_y=y, val_batches=val_batches, lr=1e-3, weight_decay=0.0
    )
    return adapter, x, y


def module_hash(module: torch.nn.Module) -> str:
    """Bitwise digest over a module's tensors."""
    digest = hashlib.sha256()
    for key, value in sorted(module.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def optimizer_hash(optimizer: torch.optim.Optimizer) -> str:
    digest = hashlib.sha256()
    state = optimizer.state_dict()
    digest.update(repr(state["param_groups"]).encode("utf-8"))
    for key in sorted(state["state"]):
        for name, value in sorted(state["state"][key].items()):
            digest.update(str(name).encode("utf-8"))
            if isinstance(value, torch.Tensor):
                digest.update(value.detach().cpu().contiguous().numpy().tobytes())
            else:
                digest.update(repr(value).encode("utf-8"))
    return digest.hexdigest()


def fingerprint(adapter, result) -> dict[str, Any]:
    """Everything compared between a continuous and a resumed run."""
    return {
        "weights": module_hash(adapter._ckpt_model_module()),
        "optimizer": optimizer_hash(adapter._ckpt_session.optimizer),
        "train_losses": list(result.progress.train_loss_history),
        "val_history": [dict(v) for v in result.progress.val_history],
        "updates": int(result.progress.global_update),
        "best_step": int(result.progress.best_step),
        "best_val": float(result.progress.best_val),
        "plan_position": int(result.progress.batch_plan_position),
    }
