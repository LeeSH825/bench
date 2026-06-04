import tempfile
import unittest
from pathlib import Path

from bench.models.maml_knet import MAMLKNetAdapter, _build_maml_checkpoint_compatibility_report


class MAMLCheckpointCompatibilityTests(unittest.TestCase):
    def test_bundled_linear_checkpoint_detects_lorenz_mismatch(self) -> None:
        report = _build_maml_checkpoint_compatibility_report(
            ckpt_path=Path("/repo/MAML_model/linear/basenet.pt"),
            system_info={
                "x_dim": 3,
                "y_dim": 2,
                "T": 50,
                "task_id": "D_lorenz_partial_v0",
                "meta": {"task_family": "lorenz_v0"},
            },
            cfg={"inner_steps": 1, "update_step_test": 1},
            is_linear_net=True,
        )

        self.assertEqual(report["metadata_status"], "missing")
        self.assertEqual(report["compatibility_status"], "mismatch")
        self.assertFalse(report["official_allowed"])
        failed = {c["name"] for c in report["checks"] if c["status"] == "fail"}
        self.assertIn("state_dim", failed)
        self.assertIn("task_family", failed)

    def test_unknown_checkpoint_without_metadata_blocks_official_use(self) -> None:
        report = _build_maml_checkpoint_compatibility_report(
            ckpt_path=Path("/tmp/custom_maml_checkpoint.pt"),
            system_info={
                "x_dim": 3,
                "y_dim": 2,
                "T": 50,
                "task_id": "D_lorenz_partial_v0",
                "meta": {"task_family": "lorenz_v0"},
            },
            cfg={},
            is_linear_net=False,
        )

        self.assertEqual(report["metadata_status"], "missing")
        self.assertEqual(report["compatibility_status"], "unknown")
        self.assertFalse(report["official_allowed"])
        self.assertIn("metadata", report["official_block_reason"])

    def test_sidecar_metadata_can_make_checks_compatible_but_not_official(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            ckpt = Path(td) / "lorenz_basenet.pt"
            metadata = ckpt.with_suffix(".json")
            metadata.write_text(
                "{"
                '"task_family": "lorenz_v0",'
                '"state_dim": 3,'
                '"obs_dim": 2,'
                '"architecture_type": "nonlinear",'
                '"inner_loop_steps": 5'
                "}\n",
                encoding="utf-8",
            )
            report = _build_maml_checkpoint_compatibility_report(
                ckpt_path=ckpt,
                system_info={
                    "x_dim": 3,
                    "y_dim": 2,
                    "T": 50,
                    "task_id": "D_lorenz_partial_v0",
                    "meta": {"task_family": "lorenz_v0"},
                },
                cfg={"inner_steps": 5},
                is_linear_net=False,
            )

        self.assertEqual(report["metadata_status"], "present")
        self.assertEqual(report["compatibility_status"], "compatible")
        self.assertFalse(report["official_allowed"])
        self.assertIn("diagnostic-only", report["official_block_reason"])

    def test_setup_writes_run_dir_compatibility_report(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            adapter = MAMLKNetAdapter()
            adapter.setup(
                {
                    "repo": {"path": "third_party/MAML_KalmanNet"},
                    "is_linear_net": True,
                    "inner_steps": 1,
                },
                system_info={
                    "x_dim": 3,
                    "y_dim": 2,
                    "T": 16,
                    "task_id": "D_lorenz_partial_v0",
                    "meta": {"task_family": "lorenz_v0"},
                },
                run_ctx={"run_dir": td, "seed": 0, "device": "cpu", "track_id": "frozen", "init_id": "trained"},
            )
            report_path = Path(td) / "maml_checkpoint_compatibility.json"
            self.assertTrue(report_path.exists())
            text = report_path.read_text(encoding="utf-8")

        self.assertIn('"adapter_id": "maml_knet"', text)
        self.assertIn('"official_allowed": false', text)


if __name__ == "__main__":
    unittest.main()
