from __future__ import annotations

import unittest

import numpy as np

from viz.analysis import consistency, gain, regime


class VizConsistencyTest(unittest.TestCase):
    def test_nees_nis_and_chi_square_mean_bounds_for_tuned_filter(self) -> None:
        rng = np.random.default_rng(7)
        n_samples = 200
        n_time = 500
        n_state = 6
        n_meas = 3
        err = rng.normal(size=(n_samples, n_time, n_state))
        innov = rng.normal(size=(n_samples, n_time, n_meas))
        p = np.broadcast_to(np.eye(n_state), (n_samples, n_time, n_state, n_state))
        s = np.broadcast_to(np.eye(n_meas), (n_samples, n_time, n_meas, n_meas))

        nees_t = np.mean(consistency.nees(err, p), axis=0)
        nis_t = np.mean(consistency.nis(innov, s), axis=0)
        nees_bounds = consistency.chi2_mean_bounds(n_samples=n_samples, dim=n_state)
        nis_bounds = consistency.chi2_mean_bounds(n_samples=n_samples, dim=n_meas)

        self.assertGreater(float(np.mean(nees_t)), float(nees_bounds[0]))
        self.assertLess(float(np.mean(nees_t)), float(nees_bounds[1]))
        self.assertGreater(float(np.mean(nis_t)), float(nis_bounds[0]))
        self.assertLess(float(np.mean(nis_t)), float(nis_bounds[1]))

    def test_chi_square_bounds_use_n_run_mean_formula(self) -> None:
        n_samples = 20
        dim = 6
        bounds = consistency.chi2_mean_bounds(n_samples=n_samples, dim=dim)
        expected = consistency.chi2_ppf(np.array([0.005, 0.995]), n_samples * dim) / n_samples
        np.testing.assert_allclose(bounds, expected, rtol=0.0, atol=0.0)
        single_bounds = consistency.chi2_ppf(np.array([0.005, 0.995]), dim)
        self.assertLess(float(single_bounds[0]), float(bounds[0]))
        self.assertGreater(float(single_bounds[1]), float(bounds[1]))

        table = [
            (0.005, 3, 0.0717),
            (0.995, 3, 12.838),
            (0.005, 6, 0.6757),
            (0.995, 6, 18.548),
            (0.025, 10, 3.247),
            (0.975, 10, 20.483),
        ]
        for probability, dof, expected in table:
            with self.subTest(probability=probability, dof=dof):
                got = float(consistency.chi2_ppf(probability, dof))
                self.assertLess(abs(got - expected) / expected, 1e-3)

    def test_three_sigma_coverage(self) -> None:
        rng = np.random.default_rng(8)
        err = rng.normal(size=(200, 500, 6))
        p = np.broadcast_to(np.eye(6), (200, 500, 6, 6))
        coverage = consistency.three_sigma_coverage(err, p)
        self.assertLess(abs(float(coverage["overall_coverage"]) - 0.9973), 0.005)

    def test_innovation_whiteness_acf(self) -> None:
        rng = np.random.default_rng(9)
        n_time = 1000
        innov = rng.normal(size=(50, n_time, 1))
        acf = consistency.innovation_acf(innov, max_lag=10)
        bound = consistency.whiteness_bounds(n_time)
        self.assertTrue(bool(np.all(np.abs(acf[1:, 0]) < bound)))

    def test_mistuned_covariance_is_detected_below_nees_lower_bound(self) -> None:
        rng = np.random.default_rng(10)
        n_samples = 200
        n_time = 500
        n_state = 6
        err = rng.normal(size=(n_samples, n_time, n_state))
        p_bad = np.broadcast_to((2 + 2) * np.eye(n_state), (n_samples, n_time, n_state, n_state))
        nees_mean = float(np.mean(np.mean(consistency.nees(err, p_bad), axis=0)))
        lower = float(consistency.chi2_mean_bounds(n_samples=n_samples, dim=n_state)[0])
        self.assertLess(nees_mean, lower)

    def test_ensemble_sigma_matches_prediction_and_reports_sample_uncertainty(self) -> None:
        rng = np.random.default_rng(11)
        sigma_true = np.array([0.5, 1.2, 2.0], dtype=np.float64)
        err = rng.normal(scale=sigma_true, size=(2000, 5, 3))
        result = consistency.ensemble_sigma(err)
        emp = np.mean(result["emp_std"], axis=0)
        np.testing.assert_allclose(emp, sigma_true, rtol=0.05, atol=0.0)

        small = consistency.ensemble_sigma(err[:8])
        expected_rse = 1.0 / np.sqrt(2.0 * (8 - 1))
        self.assertEqual(small["n_samples"], 8)
        self.assertAlmostEqual(float(small["relative_standard_error"]), float(expected_rse), places=15)
        self.assertEqual(small["confidence_interval"].shape, (2, 5, 3))

    def test_regime_intervals_and_exponential_convergence(self) -> None:
        flags = np.array([False, True, True, False, True, True, True, False])
        self.assertEqual(
            regime.true_intervals(flags),
            [
                {"start": 1, "end": 3, "value": True},
                {"start": 4, "end": 7, "value": True},
            ],
        )
        tau = 2.0
        t = np.linspace(0.0, 10.0, 1001)
        signal = np.exp(-t / tau)
        got = regime.convergence_time(t, signal, threshold=np.exp(-2.0))
        self.assertLess(abs(got - (2.0 * tau)) / (2.0 * tau), 0.05)

    def test_gain_trajectory_norm_and_normalization(self) -> None:
        gain_arr = np.zeros((2, 3, 2, 2), dtype=np.float64)
        gain_arr[1, :, :, :] = np.eye(2)[None, :, :] * np.array([2.0, 1.0, 0.5])[:, None, None]
        out = gain.extract_gain_trajectory(gain_arr, traj_idx=1, normalize="initial")
        self.assertEqual(out["gain"].shape, (3, 2, 2))
        np.testing.assert_allclose(out["gain_norm"], np.array([1.0, 0.5, 0.25]))


if __name__ == "__main__":
    unittest.main()
