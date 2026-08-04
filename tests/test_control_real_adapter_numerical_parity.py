"""CPU deterministic parity for control-plane observer and telemetry hooks.

This test intentionally exercises the real learned adapters and their real
third-party model paths.  The control-plane observer is swapped between a
true NullObserver and a recording observer; telemetry runs in its own sampler
thread and only reads process/system state.
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any
from unittest import mock

import numpy as np
import pytest
import torch

from bench.control.events.observer import NullObserver
from bench.control.telemetry import TelemetrySampler, default_collectors
from bench.models.kalmannet_tsp import KalmanNetTSPAdapter
from bench.models.split_knet import SplitKNetAdapter


torch.set_num_threads(1)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    # PyTorch permits this setting only before parallel work has started.
    pass


class RecordingObserver:
    def __init__(self) -> None:
        self.statuses: list[tuple[str, str | None]] = []
        self.metrics: list[tuple[str, int | None, float, str | None]] = []
        self.artifacts: list[tuple[str, str]] = []

    def status(self, state: str, *, phase: str | None = None, message: str | None = None, **payload: Any) -> None:
        del message, payload
        self.statuses.append((state, phase))

    def metric(
        self,
        name: str,
        value: float,
        *,
        step: int | None = None,
        step_type: str = "global_step",
        phase: str | None = None,
        unit: str | None = None,
        **payload: Any,
    ) -> None:
        del step_type, unit, payload
        self.metrics.append((name, step, float(value), phase))

    def log(self, message: str, *, level: str = "INFO", phase: str | None = None, **payload: Any) -> None:
        del message, level, phase, payload

    def artifact(self, *, kind: str, uri: str, **payload: Any) -> None:
        del payload
        self.artifacts.append((kind, uri))


def _fixture(model_id: str) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, np.ndarray]]]:
    n_seq, n_step, x_dim, y_dim = 3, (7 if model_id == "kalmannet_tsp" else 8), 2, 2
    t = np.arange(n_step, dtype=np.float32)
    y = np.empty((n_seq, n_step, y_dim), dtype=np.float32)
    for batch_idx in range(n_seq):
        y[batch_idx, :, 0] = np.sin(np.float32(0.2) * t + np.float32(batch_idx) * np.float32(0.1))
        y[batch_idx, :, 1] = np.cos(np.float32(0.3) * t - np.float32(batch_idx) * np.float32(0.05))
    x = np.zeros((n_seq, n_step, x_dim), dtype=np.float32)
    x[:, :, 0] = np.float32(0.1)
    x[:, :, 1] = np.float32(-0.2)
    matrices = {
        "F": np.asarray([[0.9, 0.1], [0.0, 0.8]], dtype=np.float32),
        "H": np.asarray([[1.0, 0.2], [-0.1, 1.0]], dtype=np.float32),
    }
    if model_id == "split_knet":
        matrices = {
            "F": np.asarray([[0.95, 0.1], [0.0, 0.9]], dtype=np.float32),
            "H": np.asarray([[1.0, 0.2], [-0.15, 0.8]], dtype=np.float32),
        }
    cfg: dict[str, Any] = {
        "model_id": model_id,
        "repo": {"path": "third_party/KalmanNet_TSP" if model_id == "kalmannet_tsp" else "third_party/Split_KalmanNet"},
    }
    if model_id == "split_knet":
        cfg.update(
            {
                "estimator_class_path": "GSSFiltering.filtering.Split_KalmanNet_Filter",
                "input_layout": "BTD",
                "eval_init_from_gt": False,
            }
        )
    info = {
        "x_dim": x_dim,
        "y_dim": y_dim,
        "T": n_step,
        **matrices,
        "Q": np.eye(x_dim, dtype=np.float32) * np.float32(1e-3),
        "R": np.eye(y_dim, dtype=np.float32) * np.float32(2e-3),
    }
    # Several batches force the real update loop to exercise its iterator and
    # validation path without depending on any repository run artifact.
    return cfg, info, [{"x": x.copy(), "y": y.copy()} for _ in range(3)]


def _tensor_hash(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(repr(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _array_hash(array: Any) -> str:
    tensor = array.detach().cpu().contiguous() if isinstance(array, torch.Tensor) else torch.as_tensor(array)
    return hashlib.sha256(tensor.numpy().tobytes()).hexdigest()


def _run_mode(model_id: str, observer_on: bool, telemetry_on: bool, root: Path) -> dict[str, Any]:
    seed = 17 if model_id == "kalmannet_tsp" else 23
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    cfg, system_info, batches = _fixture(model_id)
    adapter = KalmanNetTSPAdapter() if model_id == "kalmannet_tsp" else SplitKNetAdapter()
    adapter.setup(cfg, system_info, {"seed": seed, "deterministic": True, "device": "cpu"})
    initial = adapter.model.state_dict() if model_id == "kalmannet_tsp" else adapter._filter_obj.kf_net.state_dict()
    initial_hash = _tensor_hash(initial)
    recorded = RecordingObserver()
    samples: list[dict[str, Any]] = []
    observer = recorded if observer_on else NullObserver()
    sampler = TelemetrySampler(
        run_id=f"parity-{model_id}",
        collectors=default_collectors(pid=None, run_dir=root, device="cpu"),
        sink=lambda sample: samples.append(sample.as_dict()),
        interval_seconds=0.1,
    )
    try:
        with mock.patch(f"bench.models.{model_id}.active_observer", return_value=observer):
            if telemetry_on:
                sampler.start()
            result = adapter.train(
                batches,
                batches,
                budget={"train_max_updates": 3},
                ckpt_dir=root / "checkpoints",
            )
            if telemetry_on:
                # Ensure at least one real collector call even on a fast host.
                samples.append(sampler.sample_once().as_dict())
    finally:
        sampler.stop()
    final_state = adapter.model.state_dict() if model_id == "kalmannet_tsp" else adapter._filter_obj.kf_net.state_dict()
    prediction = adapter.eval(batches[:1])["x_hat"]
    train_state = json.loads(Path(result["train_state_path"]).read_text(encoding="utf-8"))
    return {
        "dataset_hash": hashlib.sha256(
            b"".join(batch[key].tobytes() for batch in batches for key in ("x", "y"))
        ).hexdigest(),
        "initial_hash": initial_hash,
        "final_hash": _tensor_hash(final_state),
        "prediction_hash": _array_hash(prediction),
        "updates_used": result["updates_used"],
        "best_step": result["best_step"],
        "last_train_loss": train_state["last_train_loss"],
        "val_history": train_state["val_history"],
        "observer_metrics": recorded.metrics,
        "observer_statuses": recorded.statuses,
        "telemetry_samples": samples,
    }


@pytest.mark.parametrize("model_id", ["kalmannet_tsp", "split_knet"])
def test_real_adapter_observer_and_telemetry_are_numerically_inert(tmp_path: Path, model_id: str) -> None:
    modes = {
        label: _run_mode(model_id, observer_on, telemetry_on, tmp_path / label)
        for label, observer_on, telemetry_on in (
            ("A_observer_off_telemetry_off", False, False),
            ("B_observer_on_telemetry_off", True, False),
            ("C_observer_off_telemetry_on", False, True),
            ("D_observer_on_telemetry_on", True, True),
        )
    }
    baseline = modes["A_observer_off_telemetry_off"]
    for label in ("B_observer_on_telemetry_off", "C_observer_off_telemetry_on", "D_observer_on_telemetry_on"):
        candidate = modes[label]
        for key in ("dataset_hash", "initial_hash", "final_hash", "prediction_hash", "updates_used", "best_step", "last_train_loss", "val_history"):
            assert candidate[key] == baseline[key], f"{model_id} {label} changed {key}"
    for label in ("B_observer_on_telemetry_off", "D_observer_on_telemetry_on"):
        assert modes[label]["observer_metrics"]
        assert modes[label]["observer_statuses"]
        metric_names = [item[0] for item in modes[label]["observer_metrics"]]
        assert "loss/train_total" in metric_names
        assert "loss/validation_total" in metric_names
    assert modes["A_observer_off_telemetry_off"]["observer_metrics"] == []
    assert modes["C_observer_off_telemetry_on"]["observer_metrics"] == []
    for label in ("C_observer_off_telemetry_on", "D_observer_on_telemetry_on"):
        assert modes[label]["telemetry_samples"]
        assert all(sample["run_id"] == f"parity-{model_id}" for sample in modes[label]["telemetry_samples"])
