from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import MethodType

import torch

from bench.models.me_split_knet import MESplitKNetV0Adapter
from bench.models.measurement_enhancer import MeasurementEnhancer
from bench.models.split_knet import SplitKNetAdapter


class _FakeNet:
    def eval(self) -> "_FakeNet":
        return self

    def train(self) -> "_FakeNet":
        return self


class _FakeFilter:
    def __init__(self) -> None:
        self.kf_net = _FakeNet()


def _minimal_split(tmp: Path, cls, *, x_dim: int = 9, y_dim: int = 6):
    adapter = cls()
    adapter.device = torch.device("cpu")
    adapter.dtype = torch.float32
    adapter._x_dim = x_dim
    adapter._y_dim = y_dim
    adapter._T_setup = 5
    adapter._filter_obj = _FakeFilter()
    adapter._run_ctx = {"track_id": "frozen", "init_id": "trained"}
    adapter._run_dir = tmp
    adapter._artifacts_dir = tmp / "artifacts"
    adapter._artifacts_dir.mkdir(parents=True, exist_ok=True)
    adapter._ledger_path = tmp / "budget_ledger.json"
    adapter.last_layout = "bench_BTD_to_repo_stepwise_colvec"
    if isinstance(adapter, MESplitKNetV0Adapter):
        adapter.enhancer = MeasurementEnhancer(y_dim, hidden_dim=8, num_layers=1, kernel_size=3)
        adapter.enhancer_pretrain_target = "imu_clean_y_seq"

    def _fake_forward(self, *, y_btd: torch.Tensor, x0_batch=None) -> torch.Tensor:
        _ = y_btd, x0_batch
        return torch.zeros(y_btd.shape[0], y_btd.shape[1], x_dim, dtype=torch.float32)

    adapter._forward_batch = MethodType(_fake_forward, adapter)
    return adapter


class BiasStateAdapterShapeTests(unittest.TestCase):
    def test_split_knet_returns_bias_state_shape(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            adapter = _minimal_split(Path(d), SplitKNetAdapter)
            out = adapter.eval([{"x": torch.zeros(2, 5, 9), "y": torch.zeros(2, 5, 6)}])
            self.assertEqual(tuple(out["x_hat"].shape), (2, 5, 9))
            self.assertEqual(adapter.adapt_updates_used, 0)

    def test_me_split_knet_returns_bias_state_shape_and_keeps_y_dim(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            adapter = _minimal_split(Path(d), MESplitKNetV0Adapter)
            batch = {
                "x": torch.zeros(2, 5, 9),
                "y": torch.randn(2, 5, 6),
                "imu_clean_y_seq": torch.zeros(2, 5, 6),
                "imu_error_seq": torch.zeros(2, 5, 6),
            }
            out = adapter.eval([batch])
            self.assertEqual(tuple(out["x_hat"].shape), (2, 5, 9))
            self.assertIn("delta_norm_mean", adapter.get_runtime_diagnostics())
            self.assertEqual(adapter.adapt(torch.zeros(1, 5, 6))["adapt_updates_used"], 0)


if __name__ == "__main__":
    unittest.main()
