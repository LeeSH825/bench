from __future__ import annotations

import unittest

import numpy as np

from bench.diagnostics.imu_measurement_model_audit import (
    analytic_H_imu_bias,
    finite_difference_H_imu_bias,
    h_imu_bias,
)


class BasiliskImuBiasHHTests(unittest.TestCase):
    def test_h_imu_bias_shape_and_values(self) -> None:
        x = np.zeros((2, 4, 9), dtype=np.float32)
        x[..., 3:6] = np.array([0.01, -0.02, 0.03], dtype=np.float32)
        x[..., 6:9] = np.array([0.001, 0.002, -0.003], dtype=np.float32)
        y = h_imu_bias(x, dt=0.1)
        self.assertEqual(y.shape, (2, 4, 6))
        expected = np.array([0.011, -0.018, 0.027, 0.0011, -0.0018, 0.0027], dtype=np.float64)
        self.assertTrue(np.allclose(y[0, 0], expected))

    def test_H_shape_rank_and_finite_difference(self) -> None:
        h = analytic_H_imu_bias(0.1)
        x_t = np.array([0.1, 0.0, 0.0, 0.01, -0.02, 0.03, 0.001, 0.002, -0.003], dtype=np.float64)
        h_fd = finite_difference_H_imu_bias(x_t, dt=0.1)
        self.assertEqual(h.shape, (6, 9))
        self.assertEqual(int(np.linalg.matrix_rank(h)), 3)
        self.assertFalse(h.shape[0] == h.shape[1] and np.allclose(h, np.eye(h.shape[0])))
        self.assertLess(float(np.max(np.abs(h - h_fd))), 1.0e-8)


if __name__ == "__main__":
    unittest.main()
