from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import MethodType

import torch

from bench.models.me_split_knet import MESplitKNetV0Adapter
from bench.models.measurement_enhancer import MeasurementEnhancer
from bench.models.split_knet import _LinearSystemModel


class _FakeNet:
    def eval(self) -> "_FakeNet":
        return self

    def train(self) -> "_FakeNet":
        return self


class _FakeFilter:
    def __init__(self) -> None:
        self.kf_net = _FakeNet()


def _minimal_me_adapter(tmp: Path, *, x_dim: int = 6, y_dim: int = 3) -> MESplitKNetV0Adapter:
    adapter = MESplitKNetV0Adapter()
    adapter.device = torch.device("cpu")
    adapter.dtype = torch.float32
    adapter._x_dim = x_dim
    adapter._y_dim = y_dim
    adapter._T_setup = 4
    adapter._filter_obj = _FakeFilter()
    adapter._run_ctx = {"track_id": "frozen", "init_id": "trained"}
    adapter._run_dir = tmp
    adapter._artifacts_dir = tmp / "artifacts"
    adapter._artifacts_dir.mkdir(parents=True, exist_ok=True)
    adapter._ledger_path = tmp / "budget_ledger.json"
    adapter.last_layout = "bench_BTD_to_repo_stepwise_colvec"
    adapter.enhancer = MeasurementEnhancer(y_dim, hidden_dim=8, num_layers=1, kernel_size=3)
    adapter.enhancer_pretrain_target = "none"

    def _fake_forward(self: MESplitKNetV0Adapter, *, y_btd: torch.Tensor, x0_batch=None) -> torch.Tensor:
        _ = y_btd, x0_batch
        return torch.zeros(2, 4, x_dim, dtype=torch.float32)

    adapter._forward_batch = MethodType(_fake_forward, adapter)
    return adapter


class BasiliskImuModelCompatTests(unittest.TestCase):
    def test_linear_system_model_supports_y_dim_not_equal_x_dim(self) -> None:
        f = torch.eye(6)
        h = torch.zeros(3, 6)
        h[:, 3:6] = torch.eye(3)
        model = _LinearSystemModel(
            F=f,
            H=h,
            cov_q=torch.eye(6),
            cov_r=torch.eye(3),
            init_state=torch.zeros(6, 1),
        )
        y = model.g(torch.ones(6, 1))
        self.assertEqual(tuple(y.shape), (3, 1))
        self.assertEqual(model.y_dim, 3)

    def test_me_split_enhancer_accepts_imu_y_dim(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            adapter = _minimal_me_adapter(Path(d), x_dim=6, y_dim=3)
            y = torch.randn(2, 4, 3)
            y_enh = adapter.transform_measurements(y, phase="eval")
            self.assertEqual(tuple(y_enh.shape), (2, 4, 3))
            diag = adapter.get_runtime_diagnostics()
            self.assertIn("delta_norm_mean", diag)

    def test_me_split_pretrain_can_skip_non_full_state_target(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            adapter = _minimal_me_adapter(Path(d), x_dim=6, y_dim=3)
            updates = adapter._pretrain_enhancer([], max_updates=5)
            self.assertEqual(updates, 0)
            self.assertEqual(adapter._enhancer_train_state["status"], "skipped")

    def test_frozen_adaptation_zero_for_imu_shape(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            adapter = _minimal_me_adapter(Path(d), x_dim=6, y_dim=3)
            out = adapter.adapt(torch.zeros(1, 4, 3))
            self.assertEqual(out["adapt_updates_used"], 0)
            self.assertEqual(adapter.adapt_updates_used, 0)


if __name__ == "__main__":
    unittest.main()

