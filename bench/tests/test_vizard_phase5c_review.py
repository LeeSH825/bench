from __future__ import annotations

import io
import json
import tempfile
import unittest
import zipfile
from contextlib import redirect_stdout
from dataclasses import dataclass
from pathlib import Path

from bench.visualization.vizard_frame_checks import (
    generate_frame_check_fixtures,
)
from bench.visualization.vizard_phase5c_review import (
    REVIEW_MANIFEST_FILENAME,
    REVIEW_README_FILENAME,
    REVIEW_ZIP_FILENAME,
    VERIFICATION_REPORT_FILENAME,
    build_phase5c_review_package,
    main,
)


class VizardPhase5CReviewTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.run_dir = self.root / "run"
        self.artifacts = self.run_dir / "artifacts"
        self.vizard_dir = self.artifacts / "vizard"
        self.basilisk_dir = self.vizard_dir / "basilisk"
        self.native_dir = self.basilisk_dir / "native"
        self.plots_dir = self.artifacts / "plots"

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _write_representative_artifacts(
        self,
        *,
        include_playback: bool,
        include_frame_checks: bool = True,
    ) -> None:
        self.native_dir.mkdir(parents=True)
        self.plots_dir.mkdir(parents=True)
        (self.plots_dir / "rpy_true_vs_hat.png").write_bytes(b"png")
        (self.plots_dir / "adcs_plot_manifest.json").write_text(
            '{"schema_version":"adcs_plot_manifest_v1"}\n',
            encoding="utf-8",
        )
        (self.vizard_dir / "vizard_spacecraft_states.csv").write_text(
            "time_s,sc_name\n0,SC_true\n",
            encoding="utf-8",
        )
        (self.vizard_dir / "vizard_export_manifest.json").write_text(
            '{"schema_version":"vizard_export_v1"}\n',
            encoding="utf-8",
        )
        (self.basilisk_dir / "dataFileToViz_input.csv").write_text(
            "time_s,sc_name\n0,SC_true\n",
            encoding="utf-8",
        )
        (self.basilisk_dir / "dataFileToViz_input_manifest.json").write_text(
            '{"schema_version":"basilisk_vizard_offline_input_v1"}\n',
            encoding="utf-8",
        )
        (self.native_dir / "basilisk_api_probe.json").write_text(
            json.dumps(
                {
                    "schema_version": "basilisk_api_probe_v1",
                    "basilisk_version": "test",
                }
            )
            + "\n",
            encoding="utf-8",
        )
        (self.native_dir / "native_bridge_manifest.json").write_text(
            '{"schema_version":"vizard_native_bridge_v1"}\n',
            encoding="utf-8",
        )
        (self.native_dir / "native_bridge_log.txt").write_text(
            "test log\n",
            encoding="utf-8",
        )
        if include_playback:
            (self.native_dir / "vizard_playback.bin").write_bytes(b"playback")
        if include_frame_checks:
            generate_frame_check_fixtures(self.basilisk_dir / "frame_check")

    def test_build_review_package_and_zip(self) -> None:
        self._write_representative_artifacts(include_playback=True)
        manifest_path, readme_path = build_phase5c_review_package(
            self.run_dir,
            include_frame_checks=False,
        )
        output_dir = manifest_path.parent
        report_path = output_dir / VERIFICATION_REPORT_FILENAME
        zip_path = output_dir / REVIEW_ZIP_FILENAME
        self.assertTrue(readme_path.exists())
        self.assertTrue(report_path.exists())
        self.assertTrue(manifest_path.exists())
        self.assertTrue(zip_path.exists())

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(manifest["schema_version"], "phase5c_review_v1")
        self.assertTrue(manifest["native_playback_present"])
        self.assertTrue(manifest["plots_present"])
        self.assertFalse(manifest["official_metrics_affected"])
        self.assertIn(
            "native/vizard_playback.bin",
            manifest["included_artifacts"],
        )
        self.assertGreater(len(manifest["missing_optional_artifacts"]), 0)
        with zipfile.ZipFile(zip_path) as archive:
            names = set(archive.namelist())
        self.assertIn(REVIEW_README_FILENAME, names)
        self.assertIn(VERIFICATION_REPORT_FILENAME, names)
        self.assertIn("native/vizard_playback.bin", names)

    def test_missing_optional_playback_does_not_fail(self) -> None:
        self._write_representative_artifacts(
            include_playback=False,
            include_frame_checks=False,
        )
        manifest_path, _ = build_phase5c_review_package(
            self.run_dir,
            include_frame_checks=False,
        )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertFalse(manifest["native_playback_present"])
        self.assertTrue(
            any(
                "vizard_playback.bin" in item
                for item in manifest["missing_optional_artifacts"]
            )
        )

    def test_empty_run_dir_raises(self) -> None:
        self.run_dir.mkdir()
        with self.assertRaisesRegex(ValueError, "no meaningful"):
            build_phase5c_review_package(self.run_dir)

    def test_cli_smoke_without_zip(self) -> None:
        self._write_representative_artifacts(
            include_playback=False,
            include_frame_checks=False,
        )
        with redirect_stdout(io.StringIO()):
            result = main(
                [
                    "--run-dir",
                    str(self.run_dir),
                    "--no-frame-checks",
                    "--no-zip",
                ]
            )
        self.assertEqual(result, 0)
        output_dir = self.vizard_dir / "phase5c_review"
        self.assertTrue((output_dir / REVIEW_MANIFEST_FILENAME).exists())
        self.assertTrue((output_dir / REVIEW_README_FILENAME).exists())
        self.assertTrue((output_dir / VERIFICATION_REPORT_FILENAME).exists())
        self.assertFalse((output_dir / REVIEW_ZIP_FILENAME).exists())


@dataclass
class VizardPhase5CReviewResult:
    ok: bool
    note: str


def run_vizard_phase5c_review_tests() -> VizardPhase5CReviewResult:
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(
        VizardPhase5CReviewTests
    )
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=1).run(suite)
    return VizardPhase5CReviewResult(
        ok=bool(result.wasSuccessful()),
        note=(
            "Phase 5C Vizard review tests passed"
            if result.wasSuccessful()
            else stream.getvalue().strip()
        ),
    )


if __name__ == "__main__":
    unittest.main()
