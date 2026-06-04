from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import MethodType

import torch

from bench.models.me_split_knet import MESplitKNetV0Adapter
from bench.models.measurement_enhancer import MeasurementEnhancer


class _FakeNet:
    def eval(self) -> "_FakeNet":
        return self

    def train(self) -> "_FakeNet":
        return self


class _FakeFilter:
    def __init__(self) -> None:
        self.kf_net = _FakeNet()


def _minimal_adapter(tmp: Path) -> MESplitKNetV0Adapter:
    adapter = MESplitKNetV0Adapter()
    adapter.device = torch.device("cpu")
    adapter.dtype = torch.float32
    adapter._x_dim = 6
    adapter._y_dim = 6
    adapter._T_setup = 4
    adapter._filter_obj = _FakeFilter()
    adapter._run_ctx = {"track_id": "frozen", "init_id": "trained"}
    adapter._run_dir = tmp
    adapter._artifacts_dir = tmp / "artifacts"
    adapter._artifacts_dir.mkdir(parents=True, exist_ok=True)
    adapter._ledger_path = tmp / "budget_ledger.json"
    adapter.last_layout = "bench_BTD_to_repo_stepwise_colvec"
    adapter.enhancer = MeasurementEnhancer(6, hidden_dim=8, num_layers=1, kernel_size=3)

    def _fake_forward(self: MESplitKNetV0Adapter, *, y_btd: torch.Tensor, x0_batch=None) -> torch.Tensor:
        _ = x0_batch
        return torch.zeros(int(y_btd.shape[0]), int(y_btd.shape[1]), 6, dtype=y_btd.dtype, device=y_btd.device)

    adapter._forward_batch = MethodType(_fake_forward, adapter)
    return adapter


class MESplitKNetTests(unittest.TestCase):
    def test_adapter_output_shape(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            adapter = _minimal_adapter(Path(d))
            y = torch.randn(2, 4, 6)
            x_hat = adapter.predict(y)
            self.assertEqual(tuple(x_hat.shape), (2, 4, 6))
            diag = adapter.get_runtime_diagnostics()
            self.assertIn("delta_norm_mean", diag)

    def test_frozen_track_adapt_updates_zero(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            adapter = _minimal_adapter(Path(d))
            out = adapter.adapt(torch.zeros(1, 4, 6))
            self.assertEqual(out["adapt_updates_used"], 0)
            self.assertEqual(adapter.adapt_updates_used, 0)

    def test_no_metric_leakage_from_eval(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            adapter = _minimal_adapter(Path(d))
            x = torch.zeros(2, 4, 6)
            y = torch.randn(2, 4, 6)
            out = adapter.eval([(x, y)])
            self.assertEqual(tuple(out["x_hat"].shape), (2, 4, 6))
            self.assertNotIn("mse", out)
            self.assertNotIn("mse_db", out)

    def test_delta_ratio_clip_limits_applied_correction(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            adapter = _minimal_adapter(Path(d))
            adapter.delta_scale = 1.0
            adapter.delta_clip_ratio = 0.1
            self.assertIsNotNone(adapter.enhancer)
            adapter.enhancer.out.conv.bias.data.fill_(1.0)
            y = torch.ones(2, 5, 6)
            y_enh = adapter.transform_measurements(y, x_btd=torch.zeros_like(y), phase="eval")
            applied = y_enh - y
            ratio = torch.linalg.norm(applied, dim=2) / torch.linalg.norm(y, dim=2)
            self.assertLessEqual(float(ratio.max().item()), 0.100001)
            diag = adapter.get_runtime_diagnostics()
            self.assertLessEqual(float(diag["delta_to_raw_ratio_max"]), 0.100001)
            self.assertIn("innovation_collapse_ratio", diag)


if __name__ == "__main__":
    unittest.main()
