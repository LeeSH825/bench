from __future__ import annotations

import unittest

import numpy as np

from viz.analysis import units


class VizUnitsTest(unittest.TestCase):
    def test_rad_s_to_deg_h_reference_value(self) -> None:
        got = float(units.rad_s_to_deg_h(1e-5))
        self.assertLess(abs(got - 2.0626) / 2.0626, 1e-4)

    def test_mrp_small_angle_factor(self) -> None:
        delta = 1e-4
        expected = (2 + 2) * delta * np.rad2deg(1.0)
        got = float(units.mrp_delta_to_deg(delta))
        self.assertLess(abs(got - expected) / expected, 1e-6)

    def test_mrp_exact_small_angle_error_warning_threshold(self) -> None:
        rel = units.mrp_small_angle_relative_error(np.array([0.01, 0.3]))
        self.assertLess(float(rel[0]), 1e-4)
        self.assertGreater(float(rel[1]), 0.01)

    def test_three_sigma_mrp_covariance_band(self) -> None:
        axis_sigma = 2.5e-5
        cov = np.eye(3, dtype=np.float64) * axis_sigma**2
        got = units.covariance_axis_band_deg(cov, "mrp")
        expected = 3 * (2 + 2) * axis_sigma * np.rad2deg(1.0)
        np.testing.assert_allclose(got, np.full((3,), expected), rtol=1e-12, atol=0.0)

    def test_covariance_space_mrp_and_rotation_vector_bands_differ_by_four(self) -> None:
        axis_sigma = 2.5e-5
        cov = np.eye(3, dtype=np.float64) * axis_sigma**2
        mrp = units.covariance_axis_band_deg(cov, "mrp")
        rotvec = units.covariance_axis_band_deg(cov, "rotation_vector_rad")
        np.testing.assert_allclose(mrp, (2 + 2) * rotvec, rtol=1e-12, atol=0.0)

    def test_unknown_covariance_space_raises(self) -> None:
        with self.assertRaises(ValueError):
            units.covariance_axis_band_deg(np.eye(3), None)


if __name__ == "__main__":
    unittest.main()
