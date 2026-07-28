from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from bench.models.kalmannet_tsp import KalmanNetTSPAdapter
from viz.io.loader import load_run, load_trajectory
from viz.io.writer import write_viz_artifacts


REPO_ROOT = Path(__file__).resolve().parents[1]


def _fixture() -> tuple[dict, dict, np.ndarray, np.ndarray]:
    n_seq, n_step, x_dim, y_dim = 3, 7, 2, 2
    f = np.asarray([[0.9, 0.1], [0.0, 0.8]], dtype=np.float32)
    h = np.asarray([[1.0, 0.2], [-0.1, 1.0]], dtype=np.float32)
    t = np.arange(n_step, dtype=np.float32)
    y = np.empty((n_seq, n_step, y_dim), dtype=np.float32)
    for batch_idx in range(n_seq):
        y[batch_idx, :, 0] = np.sin(np.float32(0.2) * t + np.float32(batch_idx) * np.float32(0.1))
        y[batch_idx, :, 1] = np.cos(np.float32(0.3) * t - np.float32(batch_idx) * np.float32(0.05))
    x_true = np.zeros((n_seq, n_step, x_dim), dtype=np.float32)
    cfg = {
        "model_id": "kalmannet_tsp",
        "repo": {"path": "third_party/KalmanNet_TSP"},
        "batch_size": n_seq,
    }
    system_info = {
        "x_dim": x_dim,
        "y_dim": y_dim,
        "T": n_step,
        "F": f,
        "H": h,
        "Q": np.eye(x_dim, dtype=np.float32) * np.float32(1e-3),
        "R": np.eye(y_dim, dtype=np.float32) * np.float32(2e-3),
    }
    return cfg, system_info, x_true, y


def _adapter() -> KalmanNetTSPAdapter:
    cfg, system_info, _, _ = _fixture()
    adapter = KalmanNetTSPAdapter()
    adapter.setup(cfg, system_info, {"seed": 17, "deterministic": True, "device": "cpu"})
    return adapter


def _eval(adapter: KalmanNetTSPAdapter, x_true: np.ndarray, y: np.ndarray) -> dict:
    return adapter.eval([{"x": x_true, "y": y}])


class VizKalmanNetAdapterTest(unittest.TestCase):
    def test_opt_in_diagnostics_are_actual_update_gain_and_innovation(self) -> None:
        _, system_info, x_true, y = _fixture()
        adapter = _adapter()

        off = _eval(adapter, x_true, y)
        self.assertNotIn("diagnostics", off)
        self.assertNotIn("viz_diagnostics", adapter.get_runtime_diagnostics())

        adapter.set_viz_diagnostics_enabled(True)
        on = _eval(adapter, x_true, y)
        diagnostics = on["diagnostics"]
        self.assertEqual(set(diagnostics), {"innov", "gain"})
        self.assertNotIn("P", diagnostics)
        self.assertNotIn("S", diagnostics)
        self.assertEqual(diagnostics["innov"].shape, (3, 7, 2))
        self.assertEqual(diagnostics["gain"].shape, (3, 7, 2, 2))
        self.assertEqual(diagnostics["innov"].dtype, np.float32)
        self.assertEqual(diagnostics["gain"].dtype, np.float32)
        self.assertTrue(np.array_equal(off["x_hat"].numpy(), on["x_hat"].numpy()))

        f = np.asarray(system_info["F"], dtype=np.float32)
        h = np.asarray(system_info["H"], dtype=np.float32)
        x_hat = on["x_hat"].numpy()
        x_previous = np.zeros((x_hat.shape[0], x_hat.shape[2]), dtype=np.float32)
        for step in range(x_hat.shape[1]):
            x_prior = np.einsum("ij,bj->bi", f, x_previous)
            y_pred = np.einsum("ij,bj->bi", h, x_prior)
            expected_innov = y[:, step] - y_pred
            reconstructed = x_prior + np.einsum("bij,bj->bi", diagnostics["gain"][:, step], expected_innov)
            np.testing.assert_allclose(diagnostics["innov"][:, step], expected_innov, rtol=1e-6, atol=1e-6)
            np.testing.assert_allclose(x_hat[:, step], reconstructed, rtol=1e-6, atol=1e-6)
            x_previous = x_hat[:, step]

    def test_repeated_and_fresh_adapter_diagnostics_are_bitwise_equal(self) -> None:
        _, _, x_true, y = _fixture()
        first_adapter = _adapter()
        first_adapter.set_viz_diagnostics_enabled(True)
        first = _eval(first_adapter, x_true, y)
        second = _eval(first_adapter, x_true, y)

        fresh_adapter = _adapter()
        fresh_adapter.set_viz_diagnostics_enabled(True)
        fresh = _eval(fresh_adapter, x_true, y)
        for key in ("x_hat",):
            self.assertTrue(np.array_equal(first[key].numpy(), second[key].numpy()))
            self.assertTrue(np.array_equal(first[key].numpy(), fresh[key].numpy()))
        for key in ("gain", "innov"):
            self.assertTrue(np.array_equal(first["diagnostics"][key], second["diagnostics"][key]))
            self.assertTrue(np.array_equal(first["diagnostics"][key], fresh["diagnostics"][key]))

        first_adapter.set_viz_diagnostics_enabled(False)
        disabled_again = _eval(first_adapter, x_true, y)
        self.assertNotIn("diagnostics", disabled_again)
        self.assertNotIn("viz_diagnostics", first_adapter.get_runtime_diagnostics())

    def test_writer_loader_roundtrip_keeps_gain_only_capabilities(self) -> None:
        _, _, x_true, y = _fixture()
        adapter = _adapter()
        adapter.set_viz_diagnostics_enabled(True)
        result = _eval(adapter, x_true, y)
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            run_dir = Path(tmp) / "kalmannet_gain_only"
            write_viz_artifacts(
                run_dir=run_dir,
                repo_root=REPO_ROOT,
                suite_name="viz_kalmannet_test",
                task_id="linear_kalmannet_v0",
                task_family="linear_gaussian_v0",
                scenario_id="scenario_kalmannet",
                model_id="kalmannet_tsp",
                seed=17,
                track_id="frozen",
                init_id="pretrained",
                run_status="ok",
                time_s=np.arange(y.shape[1], dtype=np.float32),
                time_meta={"time_source": "fixture", "time_unit": "s", "dt_s": 1.0},
                x_true=x_true,
                y_obs=y,
                x_hat=result["x_hat"].numpy(),
                split_extras={},
                diagnostics=result["diagnostics"],
                adapter_meta=adapter.get_adapter_meta(),
            )
            run = load_run(run_dir)
            self.assertEqual(
                {key: run.meta["capabilities"][key] for key in ("covariance", "gain", "innovation", "innovation_cov")},
                {"covariance": False, "gain": True, "innovation": True, "innovation_cov": False},
            )
            traj = load_trajectory(run_dir, 0)
            self.assertIn("gain", traj)
            self.assertIn("innov", traj)
            self.assertNotIn("P", traj)
            self.assertNotIn("S", traj)
            self.assertEqual(traj["gain"].dtype, np.float16)
            self.assertEqual(traj["innov"].dtype, np.float32)
            self.assertEqual(traj["gain"].shape, (7, 2, 2))
            self.assertEqual(traj["innov"].shape, (7, 2))
            self.assertEqual(run.meta["adapter_meta"]["gain_semantics"], "learned_kalman_gain")


if __name__ == "__main__":
    unittest.main()
