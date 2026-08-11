from __future__ import annotations

import io
import json
import os
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from bench.visualization.checkpoint_replay_adapters import (
    MOCK_CHECKPOINT_MODEL_ID,
    get_real_checkpoint_replay_model_ids,
    run_checkpoint_replay_adapter,
)
from bench.visualization.phase6b_checkpoint_replay import run_phase6b_replay
from bench.visualization.phase6c_replay_visualization import (
    run_phase6c_replay_visualization,
)
from bench.visualization.phase6e_checkpoint_package import (
    build_replay_checkpoint_package,
)
from bench.visualization.pred_artifact import load_pred_artifact
from bench.visualization.replay_suite_scenario import (
    REPLAY_SCENARIO_FILENAME,
    REPLAY_SCENARIO_META_FILENAME,
)


ENV_REAL_PACKAGE = "AI_ADCS_PHASE6G_REAL_PACKAGE"


class Phase6GKalmanNetReplayPathTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.replay_dir = self.root / "replay"
        self.replay_dir.mkdir()
        self.n_step = 5
        self.x_true = np.arange(
            self.n_step * 9,
            dtype=np.float32,
        ).reshape(1, self.n_step, 9)
        self.y_obs = self.x_true[..., :6] + np.float32(0.25)
        np.savez_compressed(
            self.replay_dir / REPLAY_SCENARIO_FILENAME,
            time_s=np.arange(self.n_step, dtype=np.float32) * 0.5,
            x_true=self.x_true,
            y_obs=self.y_obs,
            trajectory_id=np.array([0], dtype=np.int64),
        )
        self.meta = {
            "schema_version": "phase6a_replay_input_v1",
            "suite_name": "phase6g_test_suite",
            "task_id": "phase6g_test_task",
            "task_name": "Phase 6G test task",
            "seed": 0,
            "scenario_id": "scenario_phase6g",
            "state_dim": 9,
            "measurement_dim": 6,
            "state_schema": {
                "attitude": {"type": "mrp", "indices": [0, 1, 2]},
                "angular_rate": {
                    "type": "rad_s",
                    "indices": [3, 4, 5],
                },
                "gyro_bias": {
                    "type": "rad_s",
                    "indices": [6, 7, 8],
                    "optional": True,
                },
            },
            "observation": {
                "type": "partial",
                "observed_state": [0, 1, 2, 3, 4, 5],
            },
            "time": {
                "dt_s": 0.5,
                "sequence_length_T": self.n_step,
                "duration_s": 2.0,
            },
        }
        (self.replay_dir / REPLAY_SCENARIO_META_FILENAME).write_text(
            json.dumps(self.meta, indent=2) + "\n",
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _mock_package(self) -> Path:
        checkpoint = self.root / "mock_checkpoint.pt"
        torch.save(
            {
                "model_id": MOCK_CHECKPOINT_MODEL_ID,
                "gain": 1.0,
                "bias": 0.0,
            },
            checkpoint,
        )
        return build_replay_checkpoint_package(
            checkpoint=checkpoint,
            model_id=MOCK_CHECKPOINT_MODEL_ID,
            package_dir=self.root / "mock_package",
            state_dim=9,
            measurement_dim=6,
            observed_state=[0, 1, 2, 3, 4, 5],
            is_mock=True,
            not_for_benchmark_reporting=True,
        )

    def _legacy_5x5_package(self) -> Path:
        checkpoint = self.root / "legacy_5x5.pt"
        torch.save({"state_dict": {}}, checkpoint)
        return build_replay_checkpoint_package(
            checkpoint=checkpoint,
            model_id="kalmannet_tsp",
            package_dir=self.root / "legacy_5x5_package",
            state_dim=5,
            measurement_dim=5,
            observed_state=[0, 1, 2, 3, 4],
        )

    def _real_package(self) -> Path:
        package_env = os.environ.get(ENV_REAL_PACKAGE, "").strip()
        if not package_env:
            self.skipTest(
                f"set {ENV_REAL_PACKAGE} to a real Phase 6F/6G replay package to run the full replay integration"
            )
        package = Path(package_env).expanduser().resolve()
        if not package.is_dir():
            raise FileNotFoundError(
                f"{ENV_REAL_PACKAGE} must point to a package directory: {package}"
            )
        return package

    def test_mock_adapter_still_works(self) -> None:
        package = self._mock_package()
        result = run_checkpoint_replay_adapter(
            model_id=MOCK_CHECKPOINT_MODEL_ID,
            checkpoint=package / "checkpoint.pt",
            model_config=package / "replay_contract.json",
            y_obs=self.y_obs,
            replay_meta=self.meta,
        )
        self.assertEqual(result.x_hat.shape, (1, self.n_step, 9))
        self.assertTrue(result.metadata["is_mock_adapter"])

    def test_incompatible_package_is_rejected(self) -> None:
        package = self._legacy_5x5_package()
        with self.assertRaisesRegex(ValueError, "state_dim mismatch"):
            run_checkpoint_replay_adapter(
                model_id="kalmannet_tsp",
                checkpoint=package / "checkpoint.pt",
                model_config=package / "replay_contract.json",
                y_obs=self.y_obs,
                replay_meta=self.meta,
            )

    def test_real_replay_integration_if_package_env_set(self) -> None:
        if not os.environ.get(ENV_REAL_PACKAGE, "").strip():
            self.skipTest(
                f"set {ENV_REAL_PACKAGE} to a real package to run the end-to-end replay integration"
            )

        package = self._real_package()
        pred_run_dir = self.root / "pred_run"
        artifact, meta_path = run_phase6b_replay(
            self.replay_dir,
            out_dir=pred_run_dir,
            model_id="kalmannet_tsp",
            checkpoint=package / "checkpoint.pt",
            model_config=package / "replay_contract.json",
            device="cpu",
        )
        self.assertTrue(artifact.exists())
        self.assertTrue(meta_path.exists())
        pred = load_pred_artifact(artifact)
        self.assertEqual(pred["x_hat"].shape, (1, self.n_step, 9))
        self.assertTrue(np.isfinite(pred["x_hat"]).all())

        summary_dir = run_phase6c_replay_visualization(
            pred_run_dir,
            trajectory_id=0,
            position_source="dummy_circular_orbit",
            require_native_success=False,
        )
        summary = json.loads(
            (summary_dir / "artifacts" / "phase6c_replay_visualization_summary.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(summary["trajectory_id"], 0)
        self.assertEqual(summary["position_source"], "dummy_circular_orbit")
        self.assertEqual(summary["official_metrics_affected"], False)

    def test_real_adapter_registry_is_present(self) -> None:
        self.assertIn("kalmannet_tsp", get_real_checkpoint_replay_model_ids())


@dataclass
class Phase6GKalmanNetReplayPathResult:
    ok: bool
    skipped: bool
    note: str


def run_phase6g_kalmannet_replay_path_tests(
) -> Phase6GKalmanNetReplayPathResult:
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(
        Phase6GKalmanNetReplayPathTests
    )
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=1).run(suite)
    skipped = bool(result.skipped) and result.testsRun == len(result.skipped)
    return Phase6GKalmanNetReplayPathResult(
        ok=bool(result.wasSuccessful()),
        skipped=bool(skipped),
        note=(
            "Phase 6G KalmanNet replay-path tests passed"
            if result.wasSuccessful()
            else stream.getvalue().strip()
        ),
    )


if __name__ == "__main__":
    unittest.main()
