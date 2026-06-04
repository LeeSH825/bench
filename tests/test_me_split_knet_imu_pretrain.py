from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from bench.models.me_split_knet import MESplitKNetV0Adapter
from bench.models.measurement_enhancer import MeasurementEnhancer


def _adapter(tmp: Path) -> MESplitKNetV0Adapter:
    adapter = MESplitKNetV0Adapter()
    adapter.device = torch.device("cpu")
    adapter.dtype = torch.float32
    adapter._x_dim = 6
    adapter._y_dim = 6
    adapter._T_setup = 5
    adapter._cfg = {
        "enhancer_lr": 1.0e-2,
        "enhancer_weight_decay": 0.0,
        "enhancer_max_grad_norm": 10.0,
    }
    adapter._run_ctx = {"track_id": "frozen", "init_id": "trained"}
    adapter._run_dir = tmp
    adapter._ledger_path = tmp / "budget_ledger.json"
    adapter.enhancer = MeasurementEnhancer(6, hidden_dim=8, num_layers=1, kernel_size=3)
    adapter.delta_scale = 0.25
    adapter.delta_clip_ratio = 1.0
    adapter.lambda_delta = 1.0e-3
    adapter.lambda_smooth = 1.0e-3
    adapter.w_imu_denoise = 1.0
    adapter.w_imu_corr = 0.5
    adapter.enhancer_pretrain_target = "imu_clean_y_seq"
    return adapter


def _batch() -> dict[str, torch.Tensor]:
    y = torch.zeros(2, 5, 6)
    x = torch.zeros(2, 5, 6)
    clean = torch.ones(2, 5, 6) * 0.1
    error = y - clean
    return {"x": x, "y": y, "imu_clean_y_seq": clean, "imu_error_seq": error}


class MESplitKNetImuPretrainTests(unittest.TestCase):
    def test_imu_clean_target_used_instead_of_x(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            adapter = _adapter(Path(d))
            updates = adapter._pretrain_enhancer([_batch()], max_updates=3)
            self.assertEqual(updates, 3)
            self.assertEqual(adapter._enhancer_train_state["target"], "imu_clean_y_seq")
            self.assertGreater(float(adapter._enhancer_train_state["history"][-1]["L_imu_denoise"]), 0.0)
            y_enh = adapter.transform_measurements(
                _batch()["y"],
                x_btd=_batch()["x"],
                batch=_batch(),
                phase="eval",
            )
            self.assertGreater(float(torch.linalg.norm(y_enh - _batch()["y"]).item()), 0.0)

    def test_imu_error_correction_loss_is_finite(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            adapter = _adapter(Path(d))
            adapter._pretrain_enhancer([_batch()], max_updates=1)
            hist = adapter._enhancer_train_state["history"][-1]
            self.assertTrue(torch.isfinite(torch.tensor(float(hist["L_imu_corr"]))).item())

    def test_no_fallback_to_x_when_imu_clean_missing(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            adapter = _adapter(Path(d))
            bad = {"x": torch.zeros(2, 5, 6), "y": torch.zeros(2, 5, 6)}
            with self.assertRaises(KeyError):
                adapter._pretrain_enhancer([bad], max_updates=1)

    def test_imu_diagnostics_finite(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            adapter = _adapter(Path(d))
            adapter._pretrain_enhancer([_batch()], max_updates=2)
            y = _batch()["y"]
            out = adapter.transform_measurements(y, x_btd=_batch()["x"], batch=_batch(), phase="eval")
            self.assertEqual(tuple(out.shape), tuple(y.shape))
            diag = adapter.get_runtime_diagnostics()
            for key in (
                "imu_y_raw_to_clean_mse",
                "imu_y_enh_to_clean_mse",
                "imu_mse_reduction",
                "imu_correction_alignment",
                "delta_to_imu_error_ratio_mean",
                "innovation_collapse_ratio",
            ):
                self.assertIn(key, diag)
                self.assertTrue(torch.isfinite(torch.tensor(float(diag[key]))).item(), key)
            self.assertEqual(adapter.adapt_updates_used, 0)


if __name__ == "__main__":
    unittest.main()
