from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from dataclasses import dataclass
from pathlib import Path

from bench.visualization.phase5c_demo import (
    DEMO_SUMMARY_FILENAME,
    main,
    run_phase5c_tiny_demo,
)


class Phase5CDemoTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _assert_demo_outputs(self, run_dir: Path) -> dict:
        artifacts = run_dir / "artifacts"
        summary_path = artifacts / DEMO_SUMMARY_FILENAME
        self.assertTrue(summary_path.exists())
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        self.assertEqual(
            summary["schema_version"],
            "phase5c_demo_summary_v1",
        )
        self.assertFalse(summary["official_metrics_affected"])
        self.assertEqual(
            summary["test_metric_summary"]["classification"],
            "demo_diagnostic_only_not_official",
        )

        required = [
            artifacts / "preds_test.npz",
            artifacts / "preds_test_meta.json",
            artifacts / "adcs_timeseries.csv",
            artifacts / "vizard" / "vizard_spacecraft_states.csv",
            artifacts
            / "vizard"
            / "basilisk"
            / "dataFileToViz_input.csv",
            artifacts
            / "vizard"
            / "basilisk"
            / "native"
            / "native_bridge_manifest.json",
            artifacts
            / "vizard"
            / "basilisk"
            / "frame_check"
            / "native"
            / "frame_check_native_manifest.json",
            artifacts
            / "vizard"
            / "phase5c_review"
            / "phase5c_review_manifest.json",
            artifacts
            / "vizard"
            / "phase5c_review"
            / "README_phase5c_review.md",
            artifacts
            / "vizard"
            / "phase5c_review"
            / "phase5c_review_bundle.zip",
        ]
        for path in required:
            self.assertTrue(path.exists(), str(path))
        self.assertTrue(any((artifacts / "plots").glob("*.png")))

        native_path = summary["artifact_paths"]["native_playback_bin"]
        if native_path is not None:
            self.assertTrue(Path(native_path).exists())
        self.assertIn(
            summary["native_conversion_status"],
            {
                "attempted_success",
                "attempted_failed",
                "not_attempted_basilisk_unavailable",
                "not_attempted_contract_unknown",
            },
        )
        return summary

    def test_tiny_demo_builds_full_artifact_chain(self) -> None:
        run_dir = run_phase5c_tiny_demo(
            out_root=self.root / "demo",
            device="cpu",
            seed=0,
            max_train_steps=1,
            require_native_success=False,
        )
        summary = self._assert_demo_outputs(run_dir)
        self.assertIn(
            "toy_bias_calibrated",
            summary["model_or_adapter_used"],
        )

    def test_cli_smoke(self) -> None:
        out_root = self.root / "cli"
        with redirect_stdout(io.StringIO()):
            result = main(
                [
                    "--out-root",
                    str(out_root),
                    "--device",
                    "cpu",
                    "--seed",
                    "0",
                    "--max-train-steps",
                    "1",
                ]
            )
        self.assertEqual(result, 0)
        run_dirs = list(out_root.glob("phase5c_tiny_demo_*"))
        self.assertEqual(len(run_dirs), 1)
        self._assert_demo_outputs(run_dirs[0])

    def test_non_cpu_device_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "CPU-only"):
            run_phase5c_tiny_demo(
                out_root=self.root / "bad",
                device="cuda",
                max_train_steps=1,
            )


@dataclass
class Phase5CDemoResult:
    ok: bool
    note: str


def run_phase5c_demo_tests() -> Phase5CDemoResult:
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(Phase5CDemoTests)
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=1).run(suite)
    return Phase5CDemoResult(
        ok=bool(result.wasSuccessful()),
        note=(
            "Phase 5C tiny demo tests passed"
            if result.wasSuccessful()
            else stream.getvalue().strip()
        ),
    )


if __name__ == "__main__":
    unittest.main()
