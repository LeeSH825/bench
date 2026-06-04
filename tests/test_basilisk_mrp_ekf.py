from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from bench.models.basilisk_mrp_ekf import BasiliskMRPEKFAdapter, shadow_mrp


def _meta() -> dict:
    return {
        "task_family": "basilisk_adcs_v0",
        "x_dim": 6,
        "y_dim": 6,
        "T": 4,
        "noise": {
            "Q": {"q2": 1.0e-8},
            "R": {"r2": 1.0e-4},
        },
        "ssm": {
            "true": {
                "dt": 0.1,
                "inertia": [[10.0, 0.0, 0.0], [0.0, 8.0, 0.0], [0.0, 0.0, 6.0]],
                "disturbance_torque_B_Nm": [0.0, 0.0, 0.0],
            },
            "assumed": {
                "valid_for_oracle": False,
                "Q": (1.0e-8 * np.eye(6)).tolist(),
                "R": (1.0e-4 * np.eye(6)).tolist(),
            },
        },
    }


def _setup_adapter(tmp: Path) -> BasiliskMRPEKFAdapter:
    adapter = BasiliskMRPEKFAdapter()
    adapter.setup(
        {
            "model_id": "basilisk_mrp_ekf",
            "integration": "rk4",
            "fd_eps": 1.0e-5,
            "p0_source": "measurement_noise",
        },
        {"x_dim": 6, "y_dim": 6, "T": 4, "meta": _meta()},
        {"run_dir": str(tmp), "device": "cpu", "seed": 0, "track_id": "frozen", "init_id": "pretrained"},
    )
    return adapter


class BasiliskMRPEKFTests(unittest.TestCase):
    def test_mrp_shadow_set(self) -> None:
        sigma = torch.tensor([[2.0, 0.0, 0.0], [0.2, 0.1, 0.0]], dtype=torch.float32)
        out = shadow_mrp(sigma)
        self.assertLess(float(torch.linalg.norm(out[0])), 1.0)
        self.assertTrue(torch.allclose(out[1], sigma[1]))

    def test_propagation_finite_output(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            adapter = _setup_adapter(Path(d))
            x0 = torch.tensor([[0.05, -0.02, 0.01, 0.01, 0.02, -0.01]], dtype=torch.float32)
            x1 = adapter.propagate_discrete(x0)
            self.assertEqual(tuple(x1.shape), (1, 6))
            self.assertTrue(torch.isfinite(x1).all().item())

    def test_finite_difference_jacobian_shape(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            adapter = _setup_adapter(Path(d))
            x0 = torch.zeros(2, 6, dtype=torch.float32)
            jac = adapter.finite_difference_jacobian(x0)
            self.assertEqual(tuple(jac.shape), (2, 6, 6))
            self.assertTrue(torch.isfinite(jac).all().item())

    def test_ekf_one_step_update_finite(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            adapter = _setup_adapter(Path(d))
            y = torch.zeros(3, 4, 6, dtype=torch.float32)
            y[:, :, 3] = 0.01
            pred = adapter.predict(y)
            self.assertEqual(tuple(pred.shape), (3, 4, 6))
            self.assertTrue(torch.isfinite(pred).all().item())

    def test_no_ground_truth_leakage(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            adapter = _setup_adapter(Path(d))
            y = torch.randn(2, 4, 6, dtype=torch.float32) * 0.01
            x_a = torch.zeros_like(y)
            x_b = torch.ones_like(y) * 999.0
            out_a = adapter.eval([(x_a, y)])["x_hat"]
            out_b = adapter.eval([(x_b, y)])["x_hat"]
            self.assertTrue(torch.allclose(out_a, out_b))

    def test_db_invariant_and_frozen_adapt_zero(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            adapter = _setup_adapter(Path(d))
            y = torch.randn(2, 4, 6, dtype=torch.float32) * 0.01
            x = torch.zeros_like(y)
            x_hat = adapter.eval([(x, y)])["x_hat"]
            mse = float(torch.mean((x_hat - x) ** 2).item())
            mse_db = 10.0 * math.log10(max(mse, 1.0e-30))
            self.assertAlmostEqual(mse_db, 10.0 * math.log10(mse), places=12)
            self.assertEqual(adapter.adapt_updates_used, 0)
            ledger = Path(d) / "budget_ledger.json"
            self.assertTrue(ledger.exists())


if __name__ == "__main__":
    unittest.main()
