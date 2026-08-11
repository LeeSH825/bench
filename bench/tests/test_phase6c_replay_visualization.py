from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

import numpy as np

from bench.visualization.phase6c_replay_visualization import (
    PHASE6C_SUMMARY_FILENAME,
    main,
    run_phase6c_replay_visualization,
)
from bench.visualization.pred_artifact import save_pred_artifact
from bench.visualization.vizard_frame_checks import (
    generate_frame_check_fixtures,
)
from bench.visualization.vizard_native_bridge import (
    NATIVE_BRIDGE_LOG_FILENAME,
    NATIVE_BRIDGE_MANIFEST_FILENAME,
)
from bench.visualization.vizard_phase5c_review import (
    FRAME_CHECK_NATIVE_MANIFEST_FILENAME,
    REVIEW_MANIFEST_FILENAME,
    REVIEW_ZIP_FILENAME,
)


def _state_meta() -> dict:
    return {
        "phase": "phase6b_checkpoint_replay",
        "model_id": "replay_identity_baseline",
        "scenario_id": "scenario_phase6c",
        "suite_name": "phase6c_test_suite",
        "task_id": "phase6c_test_task",
        "seed": 0,
        "state_schema": {
            "attitude": {
                "type": "mrp",
                "name": "sigma_BN",
                "indices": [0, 1, 2],
            },
            "angular_rate": {
                "type": "rad_s",
                "name": "omega_BN_B",
                "indices": [3, 4, 5],
            },
            "gyro_bias": {
                "type": "rad_s",
                "name": "gyro_bias",
                "indices": [6, 7, 8],
                "optional": True,
            },
        },
        "attitude_convention": "MRP sigma_BN",
        "time_unit": "s",
    }


def _fake_native_bridge(
    run_dir: str | Path,
    **_: object,
) -> tuple[Path, Path]:
    root = Path(run_dir).resolve()
    basilisk_dir = root / "artifacts" / "vizard" / "basilisk"
    native_dir = basilisk_dir / "native"
    frame_dir = basilisk_dir / "frame_check"
    native_dir.mkdir(parents=True, exist_ok=True)
    generate_frame_check_fixtures(frame_dir)

    (native_dir / "basilisk_api_probe.json").write_text(
        json.dumps(
            {
                "schema_version": "basilisk_api_probe_v1",
                "basilisk_available": False,
                "basilisk_version": None,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    manifest_path = native_dir / NATIVE_BRIDGE_MANIFEST_FILENAME
    log_path = native_dir / NATIVE_BRIDGE_LOG_FILENAME
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "vizard_native_bridge_v1",
                "native_conversion_status": (
                    "not_attempted_basilisk_unavailable"
                ),
                "native_conversion_error": "test environment unavailable",
                "official_metrics_affected": False,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    log_path.write_text("test native bridge unavailable\n", encoding="utf-8")

    frame_native_dir = frame_dir / "native"
    frame_native_dir.mkdir(parents=True, exist_ok=True)
    playback_names = (
        "zero_attitude_vizard_playback.bin",
        "small_positive_yaw_vizard_playback.bin",
        "true_vs_estimated_offset_vizard_playback.bin",
    )
    for name in playback_names:
        (frame_native_dir / name).write_bytes(b"test playback")
    (frame_native_dir / FRAME_CHECK_NATIVE_MANIFEST_FILENAME).write_text(
        json.dumps(
            {
                "schema_version": "frame_check_native_manifest_v1",
                "all_successful": True,
                "num_fixtures": 3,
                "num_successful": 3,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return manifest_path, log_path


class Phase6CReplayVisualizationTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.run_dir = self.root / "prediction_run"
        self.artifacts = self.run_dir / "artifacts"
        self.n_step = 4
        time_s = np.arange(self.n_step, dtype=np.float64) * 0.5
        x_true = np.zeros((1, self.n_step, 9), dtype=np.float64)
        x_true[0, :, 0] = 0.001 * time_s
        x_true[0, :, 1] = -0.0005 * time_s
        x_true[0, :, 2] = 0.00075 * time_s
        x_true[0, :, 3:6] = np.array([0.001, -0.002, 0.003])
        x_true[0, :, 6:9] = np.array([0.0001, -0.0001, 0.0002])
        y_obs = x_true[..., :6] + 1.0e-5
        x_hat = x_true.copy()
        x_hat[..., :6] = y_obs
        save_pred_artifact(
            self.artifacts,
            time_s=time_s,
            x_true=x_true,
            y_obs=y_obs,
            x_hat=x_hat,
            trajectory_id=np.array([0], dtype=np.int64),
            meta=_state_meta(),
        )

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _run(self, **kwargs: object) -> Path:
        with patch(
            "bench.visualization.phase6c_replay_visualization."
            "run_vizard_native_bridge",
            side_effect=_fake_native_bridge,
        ):
            return run_phase6c_replay_visualization(
                self.run_dir,
                trajectory_id=0,
                position_source="fixed_origin",
                require_native_success=False,
                **kwargs,
            )

    def test_orchestrator_outputs_and_summary(self) -> None:
        summary_path = self._run()
        self.assertEqual(
            summary_path,
            self.artifacts / PHASE6C_SUMMARY_FILENAME,
        )
        required = (
            self.artifacts / "adcs_timeseries.csv",
            self.artifacts / "adcs_timeseries_meta.json",
            self.artifacts / "plots" / "adcs_plot_manifest.json",
            self.artifacts
            / "vizard"
            / "vizard_spacecraft_states.csv",
            self.artifacts / "vizard" / "vizard_export_manifest.json",
            self.artifacts
            / "vizard"
            / "basilisk"
            / "dataFileToViz_input.csv",
            self.artifacts
            / "vizard"
            / "basilisk"
            / "dataFileToViz_input_manifest.json",
            self.artifacts
            / "vizard"
            / "phase5c_review"
            / REVIEW_MANIFEST_FILENAME,
            self.artifacts
            / "vizard"
            / "phase5c_review"
            / REVIEW_ZIP_FILENAME,
        )
        self.assertTrue(all(path.exists() for path in required))
        self.assertTrue(any((self.artifacts / "plots").glob("*.png")))

        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        self.assertEqual(
            summary["schema_version"],
            "phase6c_replay_visualization_summary_v1",
        )
        self.assertEqual(summary["trajectory_id"], 0)
        self.assertEqual(summary["position_source"], "fixed_origin")
        self.assertEqual(summary["num_timestamps"], self.n_step)
        self.assertEqual(
            summary["time_range_s"],
            {"start": 0.0, "end": 1.5, "duration": 1.5},
        )
        self.assertIn("adcs_timeseries_csv", summary["generated_artifacts"])
        self.assertFalse(summary["official_metrics_affected"])
        self.assertEqual(
            summary["native_conversion_status"],
            "not_attempted_basilisk_unavailable",
        )
        self.assertEqual(summary["review_package_status"], "created")
        self.assertIn(
            "visualization pipeline validation only",
            summary["notes"],
        )

    def test_missing_prediction_and_metadata_raise(self) -> None:
        missing_run = self.root / "missing_run"
        with self.assertRaises(FileNotFoundError):
            run_phase6c_replay_visualization(missing_run)

        pred_path = self.artifacts / "preds_test.npz"
        saved_pred_path = self.artifacts / "preds_test.saved.npz"
        pred_path.replace(saved_pred_path)
        with self.assertRaisesRegex(FileNotFoundError, "prediction artifact"):
            run_phase6c_replay_visualization(self.run_dir)
        saved_pred_path.replace(pred_path)

        meta_path = self.artifacts / "preds_test_meta.json"
        meta_path.unlink()
        with self.assertRaisesRegex(
            FileNotFoundError,
            "prediction artifact metadata",
        ):
            run_phase6c_replay_visualization(self.run_dir)

    def test_invalid_trajectory_id_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "does not exist"):
            run_phase6c_replay_visualization(
                self.run_dir,
                trajectory_id=99,
                position_source="fixed_origin",
            )

    def test_native_non_strict_status_is_recorded(self) -> None:
        stale_playback = (
            self.artifacts
            / "vizard"
            / "basilisk"
            / "native"
            / "vizard_playback.bin"
        )
        stale_playback.parent.mkdir(parents=True, exist_ok=True)
        stale_playback.write_bytes(b"stale playback")
        summary = json.loads(
            self._run(create_zip=False).read_text(encoding="utf-8")
        )
        self.assertEqual(
            summary["native_conversion_status"],
            "not_attempted_basilisk_unavailable",
        )
        self.assertIsNone(
            summary["generated_artifacts"]["native_playback_bin"]
        )
        self.assertFalse(stale_playback.exists())
        self.assertIsNone(
            summary["generated_artifacts"]["review_bundle_zip"]
        )

    def test_cli_smoke(self) -> None:
        with patch(
            "bench.visualization.phase6c_replay_visualization."
            "run_vizard_native_bridge",
            side_effect=_fake_native_bridge,
        ):
            with redirect_stdout(io.StringIO()):
                result = main(
                    [
                        "--pred-run-dir",
                        str(self.run_dir),
                        "--trajectory-id",
                        "0",
                        "--position-source",
                        "fixed_origin",
                    ]
                )
        self.assertEqual(result, 0)
        self.assertTrue(
            (self.artifacts / PHASE6C_SUMMARY_FILENAME).exists()
        )


@dataclass
class Phase6CReplayVisualizationResult:
    ok: bool
    note: str


def run_phase6c_replay_visualization_tests(
) -> Phase6CReplayVisualizationResult:
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(
        Phase6CReplayVisualizationTests
    )
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=1).run(suite)
    return Phase6CReplayVisualizationResult(
        ok=bool(result.wasSuccessful()),
        note=(
            "Phase 6C replay visualization tests passed"
            if result.wasSuccessful()
            else stream.getvalue().strip()
        ),
    )


if __name__ == "__main__":
    unittest.main()
