from __future__ import annotations

import unittest

import numpy as np

from viz.analysis import attitude


class VizAttitudeTest(unittest.TestCase):
    def test_mrp_quaternion_roundtrip_random_inside_unit_ball(self) -> None:
        rng = np.random.default_rng(123)
        raw = rng.normal(size=(1000, 3))
        raw /= np.linalg.norm(raw, axis=1, keepdims=True)
        sigma = raw * rng.uniform(0.0, 0.9, size=(1000, 1))
        recovered = attitude.quat_to_mrp(attitude.mrp_to_quat(sigma))
        self.assertLess(float(np.max(np.abs(recovered - sigma))), 1e-10)

    def test_shadow_crossing_quaternion_sign_continuity(self) -> None:
        radii_one = np.concatenate(
            [
                np.linspace(0.8, 1.2, 40),
                np.linspace(1.2, 0.8, 40),
            ]
        )
        radii = np.tile(radii_one, 3)
        raw_sigma = np.stack([radii, np.zeros_like(radii), np.zeros_like(radii)], axis=1)
        stored_sigma = attitude.shadow_mrp(raw_sigma)
        q_raw = attitude.continuous_quat_sign(attitude.mrp_to_quat(raw_sigma))
        q_stored = attitude.mrp_to_quat_continuous(stored_sigma)
        dots = np.sum(q_stored[:-1] * q_stored[1:], axis=1)
        self.assertGreater(float(np.min(dots)), 0.0)
        raw_step = attitude.geodesic_angle_rad(q_raw[:-1], q_raw[1:])
        stored_step = attitude.geodesic_angle_rad(q_stored[:-1], q_stored[1:])
        np.testing.assert_allclose(stored_step, raw_step, rtol=1e-12, atol=1e-12)

    def test_quaternion_sign_ambiguity_has_zero_geodesic_error(self) -> None:
        q = attitude.quat_from_euler321(np.deg2rad(30.0), 0.0, np.deg2rad(45.0))
        self.assertAlmostEqual(float(attitude.geodesic_angle_rad(q, -q)), 0.0, places=14)

    def test_geodesic_matches_axis_error_norm_for_small_error(self) -> None:
        sigma_true = np.zeros((1, 3), dtype=np.float64)
        sigma_hat = np.array([[2e-5, -1e-5, 3e-5]], dtype=np.float64)
        geodesic = attitude.geodesic_angle_rad(
            attitude.mrp_to_quat(sigma_true),
            attitude.mrp_to_quat(sigma_hat),
        )
        axis_norm = np.linalg.norm(attitude.mrp_axis_error_rad(sigma_true, sigma_hat), axis=-1)
        np.testing.assert_allclose(axis_norm, geodesic, rtol=0.01, atol=0.0)

    def test_euler321_roundtrip_known_angles(self) -> None:
        expected = np.array([np.deg2rad(30.0), 0.0, np.deg2rad(45.0)])
        q = attitude.quat_from_euler321(*expected)
        got = attitude.euler321_from_quat(q)
        np.testing.assert_allclose(got, expected, rtol=0.0, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
