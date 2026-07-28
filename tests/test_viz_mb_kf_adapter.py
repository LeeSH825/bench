from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from bench.models.mb_kf import ModelBasedKFAdapter
from viz.io.loader import load_run, load_trajectory
from viz.io.writer import write_viz_artifacts


REPO_ROOT = Path(__file__).resolve().parents[1]


def _fixture_matrices() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    f = np.array([[1.0, 0.2], [0.0, 1.0]], dtype=np.float32)
    h = np.array([[1.0, 0.0]], dtype=np.float32)
    q = np.diag([0.01, 0.02]).astype(np.float32)
    r = np.array([[0.04]], dtype=np.float32)
    return f, h, q, r


def _setup_adapter(tmp: Path) -> ModelBasedKFAdapter:
    f, h, q, r = _fixture_matrices()
    adapter = ModelBasedKFAdapter()
    adapter.setup(
        {
            "baseline_mode": "oracle",
            "outputs_covariance": False,
            "p0_scale": 0.5,
            "innovation_eps": 0.0,
        },
        {"x_dim": 2, "y_dim": 1, "T": 4, "F": f, "H": h, "Q": q, "R": r, "meta": {}},
        {"run_dir": str(tmp), "device": "cpu", "seed": 0, "track_id": "frozen", "init_id": "pretrained"},
    )
    return adapter


def _manual_kf(y: np.ndarray) -> dict[str, np.ndarray]:
    f, h, q, r = _fixture_matrices()
    x_post = np.zeros((2, 1), dtype=np.float64)
    p_post = np.eye(2, dtype=np.float64) * 0.5
    eye = np.eye(2, dtype=np.float64)
    xs = []
    ps = []
    innovs = []
    ss = []
    gains = []
    priors = []
    for t in range(y.shape[0]):
        x_prior = f.astype(np.float64) @ x_post
        p_prior = f.astype(np.float64) @ p_post @ f.astype(np.float64).T + q.astype(np.float64)
        innov = y[t].reshape(1, 1).astype(np.float64) - h.astype(np.float64) @ x_prior
        s = h.astype(np.float64) @ p_prior @ h.astype(np.float64).T + r.astype(np.float64)
        k = p_prior @ h.astype(np.float64).T @ np.linalg.inv(s)
        x_post = x_prior + k @ innov
        i_kh = eye - k @ h.astype(np.float64)
        p_post = i_kh @ p_prior @ i_kh.T + k @ r.astype(np.float64) @ k.T
        priors.append(x_prior[:, 0].copy())
        xs.append(x_post[:, 0].copy())
        ps.append(p_post.copy())
        innovs.append(innov[:, 0].copy())
        ss.append(s.copy())
        gains.append(k.copy())
    return {
        "x_prior": np.asarray(priors),
        "x_hat": np.asarray(xs),
        "P": np.asarray(ps),
        "innov": np.asarray(innovs),
        "S": np.asarray(ss),
        "gain": np.asarray(gains),
    }


class VizMbKfAdapterTest(unittest.TestCase):
    def test_diagnostics_match_independent_linear_kf_equations(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            adapter = _setup_adapter(Path(tmp))
            y = torch.tensor([[[0.30], [0.25], [0.40], [0.35]]], dtype=torch.float32)
            x = torch.zeros(1, 4, 2, dtype=torch.float32)
            off = adapter.eval([(x, y)])
            self.assertNotIn("diagnostics", off)
            adapter.set_viz_diagnostics_enabled(True)
            on = adapter.eval([(x, y)])
            np.testing.assert_array_equal(off["x_hat"].numpy(), on["x_hat"].numpy())

            manual = _manual_kf(y.numpy()[0])
            diag = on["diagnostics"]
            self.assertEqual(tuple(diag["P"].shape), (1, 4, 2, 2))
            self.assertEqual(tuple(diag["innov"].shape), (1, 4, 1))
            self.assertEqual(tuple(diag["S"].shape), (1, 4, 1, 1))
            self.assertEqual(tuple(diag["gain"].shape), (1, 4, 2, 1))
            self.assertEqual(diag["innov"].dtype, torch.float32)
            self.assertEqual(diag["P"].dtype, torch.float32)

            np.testing.assert_allclose(on["x_hat"].numpy()[0], manual["x_hat"], rtol=1e-6, atol=1e-6)
            np.testing.assert_allclose(diag["innov"].numpy()[0], manual["innov"], rtol=1e-6, atol=1e-6)
            np.testing.assert_allclose(diag["S"].numpy()[0], manual["S"], rtol=1e-6, atol=1e-6)
            np.testing.assert_allclose(diag["gain"].numpy()[0], manual["gain"], rtol=1e-6, atol=1e-6)
            np.testing.assert_allclose(diag["P"].numpy()[0], manual["P"], rtol=1e-6, atol=1e-6)

            x_from_saved_gain = manual["x_prior"] + np.einsum("tnm,tm->tn", diag["gain"].numpy()[0], diag["innov"].numpy()[0])
            np.testing.assert_allclose(x_from_saved_gain, on["x_hat"].numpy()[0], rtol=1e-6, atol=1e-6)
            p = diag["P"].numpy()[0]
            np.testing.assert_allclose(p, np.swapaxes(p, -1, -2), rtol=1e-6, atol=1e-6)
            self.assertGreaterEqual(float(np.min(np.linalg.eigvalsh(p.astype(np.float64)))), -1.0e-6)

    def test_writer_loader_contract_for_mb_kf_full_diagnostics(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            adapter = _setup_adapter(Path(tmp) / "adapter")
            adapter.set_viz_diagnostics_enabled(True)
            y = torch.tensor([[[0.30], [0.25], [0.40], [0.35]]], dtype=torch.float32)
            x = torch.zeros(1, 4, 2, dtype=torch.float32)
            out = adapter.eval([(x, y)])
            run_dir = Path(tmp) / "artifact"
            write_viz_artifacts(
                run_dir=run_dir,
                repo_root=REPO_ROOT,
                suite_name="viz_mb_kf_test",
                task_id="A_linear_viz_mb_kf_test_v0",
                task_family="linear_gaussian_v0",
                scenario_id="scenario_mb_kf",
                model_id="oracle_kf",
                seed=0,
                track_id="frozen",
                init_id="pretrained",
                run_status="ok",
                time_s=np.arange(4, dtype=np.float32),
                time_meta={"time_source": "test", "time_unit": "s", "dt_s": 1.0},
                x_true=x.numpy(),
                y_obs=y.numpy(),
                x_hat=out["x_hat"].numpy(),
                split_extras={},
                diagnostics=out["diagnostics"],
                adapter_meta=adapter.get_adapter_meta(),
            )
            run = load_run(run_dir)
            self.assertEqual(run.meta["model_id"], "oracle_kf")
            self.assertEqual(run.meta["state_spec"]["layout"][0]["kind"], "state")
            self.assertTrue(run.meta["capabilities"]["covariance"])
            self.assertTrue(run.meta["capabilities"]["innovation"])
            self.assertTrue(run.meta["capabilities"]["innovation_cov"])
            self.assertTrue(run.meta["capabilities"]["gain"])
            traj = load_trajectory(run_dir, 0)
            self.assertEqual(traj["P"].dtype, np.float16)
            self.assertEqual(traj["S"].dtype, np.float16)
            self.assertEqual(traj["gain"].dtype, np.float16)
            self.assertEqual(traj["innov"].dtype, np.float32)
            self.assertNotIn("q_true", traj)
            self.assertEqual(traj["P"].shape, (4, 2, 2))
            self.assertEqual(traj["S"].shape, (4, 1, 1))
            self.assertEqual(traj["gain"].shape, (4, 2, 1))


if __name__ == "__main__":
    unittest.main()
