from __future__ import annotations

import unittest

import numpy as np

from viz.analysis import decomposition


class VizDecompositionTest(unittest.TestCase):
    def test_bias_plus_noise_reconstructs_imu_error(self) -> None:
        rng = np.random.default_rng(21)
        bias = rng.normal(size=(4, 10, 6))
        noise = rng.normal(size=(4, 10, 6))
        imu_error = bias + noise
        residual = decomposition.decomposition_residual(bias, noise, imu_error)
        self.assertLess(float(np.max(np.abs(residual))), 1e-6)

    def test_contribution_fractions_sum_to_one(self) -> None:
        bias = np.ones((3, 4, 6), dtype=np.float64)
        noise = np.full((3, 4, 6), 2.0, dtype=np.float64)
        frac = decomposition.contribution_fractions(bias, noise)
        self.assertAlmostEqual(float(frac["deterministic"] + frac["stochastic"]), 1.0, places=12)
        self.assertGreater(float(frac["stochastic"]), float(frac["deterministic"]))


if __name__ == "__main__":
    unittest.main()
