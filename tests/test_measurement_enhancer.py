from __future__ import annotations

import unittest

import torch

from bench.models.measurement_enhancer import MeasurementEnhancer, enhancement_regularization


class MeasurementEnhancerTests(unittest.TestCase):
    def test_shape(self) -> None:
        model = MeasurementEnhancer(6, hidden_dim=8, num_layers=2, kernel_size=3)
        y = torch.randn(2, 7, 6)
        delta = model(y)
        self.assertEqual(tuple(delta.shape), (2, 7, 6))

    def test_causality(self) -> None:
        torch.manual_seed(7)
        model = MeasurementEnhancer(6, hidden_dim=8, num_layers=2, kernel_size=3)
        for param in model.parameters():
            torch.nn.init.normal_(param, mean=0.0, std=0.05)
        model.eval()
        y = torch.randn(1, 8, 6)
        y_changed = y.clone()
        y_changed[:, 5:, :] += torch.randn_like(y_changed[:, 5:, :]) * 100.0
        out_a = model(y)
        out_b = model(y_changed)
        self.assertTrue(torch.allclose(out_a[:, :5, :], out_b[:, :5, :], atol=1.0e-6, rtol=1.0e-6))

    def test_zero_init_safety(self) -> None:
        model = MeasurementEnhancer(6, hidden_dim=8, num_layers=2, kernel_size=3)
        y = torch.randn(3, 5, 6)
        y_enh, delta = model.enhance(y, delta_scale=1.0)
        self.assertLess(float(torch.mean(torch.abs(y_enh - y)).item()), 1.0e-8)
        self.assertLess(float(torch.mean(torch.abs(delta)).item()), 1.0e-8)

    def test_regularization_finite(self) -> None:
        delta = torch.randn(2, 6, 6) * 0.01
        regs = enhancement_regularization(delta)
        self.assertTrue(torch.isfinite(regs["L_delta"]).item())
        self.assertTrue(torch.isfinite(regs["L_smooth"]).item())


if __name__ == "__main__":
    unittest.main()
