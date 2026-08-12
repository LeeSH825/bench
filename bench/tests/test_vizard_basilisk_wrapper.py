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
import pandas as pd

from bench.visualization.vizard_basilisk_wrapper import (
    BASILISK_INPUT_FILENAME,
    BASILISK_INPUT_MANIFEST_FILENAME,
    MANUAL_CHECK_FILENAME,
    build_basilisk_vizard_offline_input,
    detect_basilisk_available,
    main,
)


def _phase4_frame(*, n_step: int = 5) -> pd.DataFrame:
    time_s = np.repeat(np.arange(n_step, dtype=np.float64) * 0.1, 2)
    sc_name = np.tile(["SC_true", "SC_estimated"], n_step)
    source = np.tile(["true", "estimated"], n_step)
    base = np.arange(2 * n_step, dtype=np.float64) * 0.001
    return pd.DataFrame(
        {
            "time_s": time_s,
            "traj_id": np.zeros(2 * n_step, dtype=np.int64),
            "sc_name": sc_name,
            "source": source,
            "r_BN_N_x_m": 7000.0e3 + base,
            "r_BN_N_y_m": base,
            "r_BN_N_z_m": np.zeros(2 * n_step),
            "v_BN_N_x_m_s": -base,
            "v_BN_N_y_m_s": np.full(2 * n_step, 8100.0),
            "v_BN_N_z_m_s": np.zeros(2 * n_step),
            "sigma_BN_1": base,
            "sigma_BN_2": base + 0.01,
            "sigma_BN_3": base + 0.02,
            "omega_BN_B_x_rad_s": base + 0.03,
            "omega_BN_B_y_rad_s": base + 0.04,
            "omega_BN_B_z_rad_s": base + 0.05,
        }
    )


class VizardBasiliskWrapperTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.run_dir = self.root / "run"
        self.vizard_dir = self.run_dir / "artifacts" / "vizard"
        self.vizard_dir.mkdir(parents=True)
        self.input_csv = self.vizard_dir / "vizard_spacecraft_states.csv"
        self.phase4_manifest = self.vizard_dir / "vizard_export_manifest.json"

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _write_input(
        self,
        frame: pd.DataFrame,
        *,
        manifest: bool = True,
    ) -> None:
        frame.to_csv(self.input_csv, index=False)
        if manifest:
            self.phase4_manifest.write_text(
                json.dumps(
                    {
                        "schema_version": "vizard_export_v1",
                        "position_source": "dummy_circular_orbit",
                    }
                )
                + "\n",
                encoding="utf-8",
            )

    def test_offline_input_generation_and_manifest(self) -> None:
        frame = _phase4_frame()
        self._write_input(frame)
        csv_path, manifest_path, readme_path = (
            build_basilisk_vizard_offline_input(self.run_dir)
        )
        self.assertTrue(csv_path.exists())
        self.assertTrue(manifest_path.exists())
        self.assertTrue(readme_path.exists())

        output = pd.read_csv(csv_path)
        self.assertEqual(len(output), len(frame))
        self.assertIn("attitude_type", output.columns)
        self.assertTrue(output["attitude_type"].eq("MRP").all())
        self.assertEqual(output["time_s"].tolist(), sorted(output["time_s"]))

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(
            manifest["schema_version"],
            "basilisk_vizard_offline_input_v1",
        )
        self.assertEqual(
            manifest["spacecraft_names"],
            ["SC_true", "SC_estimated"],
        )
        self.assertEqual(manifest["sources"], ["true", "estimated"])
        self.assertFalse(manifest["official_metrics_affected"])
        self.assertIn("basilisk_available", manifest)
        self.assertEqual(
            manifest["basilisk_conversion_status"],
            "not_attempted",
        )
        self.assertEqual(
            manifest["position_source"],
            "dummy_circular_orbit",
        )

        readme = readme_path.read_text(encoding="utf-8")
        self.assertIn("does not launch Basilisk", readme)
        self.assertIn("Final MRP sign and frame conventions", readme)

    def test_missing_required_column_raises(self) -> None:
        self._write_input(_phase4_frame().drop(columns=["sigma_BN_2"]))
        with self.assertRaisesRegex(ValueError, "missing required"):
            build_basilisk_vizard_offline_input(self.run_dir)

    def test_nonfinite_data_raises(self) -> None:
        frame = _phase4_frame()
        frame.loc[0, "omega_BN_B_z_rad_s"] = np.inf
        self._write_input(frame)
        with self.assertRaisesRegex(
            ValueError,
            "omega_BN_B_z_rad_s.*NaN or Inf",
        ):
            build_basilisk_vizard_offline_input(self.run_dir)

    def test_duplicate_timestamp_spacecraft_raises(self) -> None:
        frame = _phase4_frame()
        frame = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
        self._write_input(frame)
        with self.assertRaisesRegex(ValueError, "duplicate.*time_s, sc_name"):
            build_basilisk_vizard_offline_input(self.run_dir)

    def test_missing_input_csv_raises(self) -> None:
        with self.assertRaises(FileNotFoundError):
            build_basilisk_vizard_offline_input(self.run_dir)

    def test_direct_csv_cli_smoke(self) -> None:
        self._write_input(_phase4_frame(), manifest=False)
        output_dir = self.root / "direct_output"
        with redirect_stdout(io.StringIO()):
            result = main(
                [
                    "--input-csv",
                    str(self.input_csv),
                    "--out-dir",
                    str(output_dir),
                ]
            )
        self.assertEqual(result, 0)
        self.assertTrue((output_dir / BASILISK_INPUT_FILENAME).exists())
        self.assertTrue(
            (output_dir / BASILISK_INPUT_MANIFEST_FILENAME).exists()
        )
        self.assertTrue((output_dir / MANUAL_CHECK_FILENAME).exists())

    def test_run_dir_cli_smoke(self) -> None:
        self._write_input(_phase4_frame())
        with redirect_stdout(io.StringIO()):
            result = main(["--run-dir", str(self.run_dir)])
        self.assertEqual(result, 0)
        output_dir = self.vizard_dir / "basilisk"
        self.assertTrue((output_dir / BASILISK_INPUT_FILENAME).exists())
        self.assertTrue(
            (output_dir / BASILISK_INPUT_MANIFEST_FILENAME).exists()
        )
        self.assertTrue((output_dir / MANUAL_CHECK_FILENAME).exists())

    def test_require_basilisk_unavailable_raises(self) -> None:
        self._write_input(_phase4_frame())
        with patch(
            "bench.visualization.vizard_basilisk_wrapper._detect_basilisk",
            return_value=(False, None, "test import failure"),
        ):
            with self.assertRaisesRegex(RuntimeError, "Basilisk.*unavailable"):
                build_basilisk_vizard_offline_input(
                    self.run_dir,
                    require_basilisk=True,
                )

    def test_require_basilisk_matches_environment(self) -> None:
        self._write_input(_phase4_frame())
        if detect_basilisk_available():
            paths = build_basilisk_vizard_offline_input(
                self.run_dir,
                require_basilisk=True,
            )
            self.assertTrue(all(path.exists() for path in paths))
        else:
            with self.assertRaises(RuntimeError):
                build_basilisk_vizard_offline_input(
                    self.run_dir,
                    require_basilisk=True,
                )

    def test_unsupported_mode_raises(self) -> None:
        self._write_input(_phase4_frame())
        with self.assertRaisesRegex(ValueError, "unsupported mode"):
            build_basilisk_vizard_offline_input(
                self.run_dir,
                mode="stream",  # type: ignore[arg-type]
            )


@dataclass
class VizardBasiliskWrapperResult:
    ok: bool
    note: str


def run_vizard_basilisk_wrapper_tests() -> VizardBasiliskWrapperResult:
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(
        VizardBasiliskWrapperTests
    )
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=1).run(suite)
    return VizardBasiliskWrapperResult(
        ok=bool(result.wasSuccessful()),
        note=(
            "Basilisk/Vizard offline wrapper tests passed"
            if result.wasSuccessful()
            else stream.getvalue().strip()
        ),
    )


if __name__ == "__main__":
    unittest.main()
