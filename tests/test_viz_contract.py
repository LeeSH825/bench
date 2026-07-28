from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from viz.contract import ARTIFACT_VERSION, ContractError, UnsupportedArtifactVersion, deterministic_traj_index
from viz.io.loader import assert_overlay_compatible, load_run, load_runs, load_trajectory
from viz.io.writer import write_viz_artifacts


REPO_ROOT = Path(__file__).resolve().parents[1]


def _sample_payload(n_seq=5, n_step=7, dim=6):
    rng = np.random.default_rng(31)
    x_true = rng.normal(scale=0.01, size=(n_seq, n_step, dim)).astype(np.float32)
    x_hat = (x_true + rng.normal(scale=1e-3, size=(n_seq, n_step, dim))).astype(np.float32)
    y_obs = (x_true + rng.normal(scale=1e-3, size=(n_seq, n_step, dim))).astype(np.float32)
    p = np.broadcast_to(np.eye(dim, dtype=np.float32), (n_seq, n_step, dim, dim)).copy()
    s = np.broadcast_to(np.eye(dim, dtype=np.float32), (n_seq, n_step, dim, dim)).copy()
    gain = np.broadcast_to(np.eye(dim, dtype=np.float32) * 0.1, (n_seq, n_step, dim, dim)).copy()
    innov = (y_obs - x_hat).astype(np.float32)
    return x_true, x_hat, y_obs, {"P": p, "S": s, "gain": gain, "innov": innov}


def _write_run(root, name, *, task_family="basilisk_adcs_v0", model_id="basilisk_mrp_ekf", seed=0, diagnostics=None, n_seq=5):
    x_true, x_hat, y_obs, diag_default = _sample_payload(n_seq=n_seq)
    diag = diag_default if diagnostics is None else diagnostics
    return write_viz_artifacts(
        run_dir=Path(root) / name,
        repo_root=REPO_ROOT,
        suite_name="contract_suite",
        task_id=f"task_{name}",
        task_family=task_family,
        scenario_id="scenario_contract",
        model_id=model_id,
        seed=seed,
        track_id="frozen",
        init_id="pretrained",
        run_status="ok",
        time_s=np.arange(x_true.shape[1], dtype=np.float32) * 0.1,
        time_meta={"time_source": "test", "time_unit": "s", "dt_s": 0.1},
        x_true=x_true,
        y_obs=y_obs,
        x_hat=x_hat,
        split_extras={},
        diagnostics=diag,
        adapter_meta={"adapter_id": "test"},
    )


class VizContractTest(unittest.TestCase):
    def test_write_read_roundtrip_preserves_f32_and_f16_contract(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            x_true, x_hat, y_obs, diag = _sample_payload()
            write_viz_artifacts(
                run_dir=Path(tmp) / "run_a",
                repo_root=REPO_ROOT,
                suite_name="contract_suite",
                task_id="task_a",
                task_family="basilisk_adcs_v0",
                scenario_id="scenario_contract",
                model_id="basilisk_mrp_ekf",
                seed=0,
                track_id="frozen",
                init_id="pretrained",
                run_status="ok",
                time_s=np.arange(x_true.shape[1], dtype=np.float32) * 0.1,
                time_meta={"time_source": "test", "time_unit": "s", "dt_s": 0.1},
                x_true=x_true,
                y_obs=y_obs,
                x_hat=x_hat,
                split_extras={},
                diagnostics=diag,
                adapter_meta={"adapter_id": "test"},
            )
            run = load_run(Path(tmp) / "run_a")
            self.assertEqual(run.meta["artifact_version"], ARTIFACT_VERSION)
            self.assertEqual(run.meta["state_spec"]["covariance_space"], "mrp")
            traj = load_trajectory(Path(tmp) / "run_a", 0)
            np.testing.assert_array_equal(traj["x_true"], x_true[0])
            np.testing.assert_array_equal(traj["x_hat"], x_hat[0])
            np.testing.assert_allclose(traj["P"].astype(np.float32), diag["P"][0], rtol=1e-3, atol=1e-3)

    def test_v1_0_meta_defaults_covariance_space_to_mrp(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            _write_run(tmp, "legacy")
            meta_path = Path(tmp) / "legacy" / "meta.json"
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            meta["artifact_version"] = "1.0"
            del meta["state_spec"]["covariance_space"]
            meta_path.write_text(json.dumps(meta), encoding="utf-8")
            run = load_run(Path(tmp) / "legacy")
            self.assertEqual(run.meta["artifact_version"], "1.0")
            self.assertEqual(run.meta["state_spec"]["covariance_space"], "mrp")

    def test_unknown_covariance_space_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            _write_run(tmp, "bad_covariance_space")
            meta_path = Path(tmp) / "bad_covariance_space" / "meta.json"
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            meta["state_spec"]["covariance_space"] = "guessed_space"
            meta_path.write_text(json.dumps(meta), encoding="utf-8")
            with self.assertRaises(ContractError):
                load_run(Path(tmp) / "bad_covariance_space")

    def test_missing_covariance_is_capability_false_without_exception(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            _write_run(tmp, "missing_p", diagnostics={})
            run = load_run(Path(tmp) / "missing_p")
            self.assertFalse(run.meta["capabilities"]["covariance"])
            traj = load_trajectory(Path(tmp) / "missing_p", 0)
            self.assertNotIn("P", traj)

    def test_non_ekf_model_capabilities_follow_diagnostics_keys(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            _write_run(tmp, "oracle", task_family="linear_gaussian_v0", model_id="oracle_kf")
            run = load_run(Path(tmp) / "oracle")
            self.assertTrue(run.meta["capabilities"]["covariance"])
            self.assertTrue(run.meta["capabilities"]["innovation"])
            self.assertTrue(run.meta["capabilities"]["innovation_cov"])
            self.assertTrue(run.meta["capabilities"]["gain"])
            traj = load_trajectory(Path(tmp) / "oracle", 0)
            self.assertIn("P", traj)
            self.assertNotIn("q_true", traj)

    def test_capability_true_missing_npz_key_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            _write_run(tmp, "missing_p_claimed", diagnostics={})
            root = Path(tmp) / "missing_p_claimed"
            meta_path = root / "meta.json"
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            meta["capabilities"]["covariance"] = True
            meta_path.write_text(json.dumps(meta), encoding="utf-8")
            with self.assertRaises(ContractError):
                load_trajectory(root, 0)

    def test_gain_and_innovation_capability_true_without_keys_are_rejected(self) -> None:
        for capability in ("gain", "innovation"):
            with self.subTest(capability=capability), tempfile.TemporaryDirectory(dir="/tmp") as tmp:
                _write_run(tmp, f"missing_{capability}_claimed", diagnostics={})
                root = Path(tmp) / f"missing_{capability}_claimed"
                meta_path = root / "meta.json"
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                meta["capabilities"][capability] = True
                meta_path.write_text(json.dumps(meta), encoding="utf-8")
                with self.assertRaises(ContractError):
                    load_trajectory(root, 0)

    def test_capability_false_with_npz_key_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            _write_run(tmp, "p_present_claimed_false")
            root = Path(tmp) / "p_present_claimed_false"
            meta_path = root / "meta.json"
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            meta["capabilities"]["covariance"] = False
            meta_path.write_text(json.dumps(meta), encoding="utf-8")
            with self.assertRaises(ContractError):
                load_trajectory(root, 0)

    def test_unknown_artifact_version_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            _write_run(tmp, "bad_version")
            meta_path = Path(tmp) / "bad_version" / "meta.json"
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            meta["artifact_version"] = "9.9"
            meta_path.write_text(json.dumps(meta), encoding="utf-8")
            with self.assertRaises(UnsupportedArtifactVersion):
                load_run(Path(tmp) / "bad_version")

    def test_load_runs_returns_sorted_runs(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            dirs = []
            for seed in (2, 0, 1):
                _write_run(tmp, f"run_{seed}", seed=seed)
                dirs.append(Path(tmp) / f"run_{seed}")
            runs = load_runs(dirs)
            self.assertEqual([r.meta["seed"] for r in runs], [0, 1, 2])

    def test_formulation_mismatch_blocks_overlay(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            _write_run(tmp, "full", task_family="basilisk_adcs_v0")
            _write_run(tmp, "imu", task_family="basilisk_imu_adcs_v0")
            full = load_run(Path(tmp) / "full")
            imu = load_run(Path(tmp) / "imu")
            with self.assertRaises(ContractError):
                assert_overlay_compatible(full, imu)

    def test_traj_index_is_deterministic_for_same_n_and_k(self) -> None:
        self.assertEqual(deterministic_traj_index(32, 8), [0, 4, 8, 12, 16, 20, 24, 28])
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            _write_run(tmp, "seed0", seed=0, n_seq=32)
            _write_run(tmp, "seed1", seed=1, n_seq=32)
            a = load_run(Path(tmp) / "seed0").meta["traj_index"]
            b = load_run(Path(tmp) / "seed1").meta["traj_index"]
            self.assertEqual(a, b)

    def test_import_viz_has_no_plotting_side_effects(self) -> None:
        code = (
            "import sys; import viz; import viz.analysis.attitude; import viz.analysis.consistency; "
            "print('matplotlib', 'matplotlib' in sys.modules); "
            "print('streamlit', 'streamlit' in sys.modules); "
            "print('plotly', 'plotly' in sys.modules); "
            "print('scipy', 'scipy' in sys.modules)"
        )
        cp = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
        self.assertEqual(
            cp.stdout.strip().splitlines(),
            ["matplotlib False", "streamlit False", "plotly False", "scipy True"],
        )


if __name__ == "__main__":
    unittest.main()
