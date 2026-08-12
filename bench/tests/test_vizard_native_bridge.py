from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from bench.visualization.vizard_frame_checks import (
    FRAME_CHECK_README_FILENAME,
    POSITIVE_YAW_FILENAME,
    TRUE_ESTIMATED_OFFSET_FILENAME,
    ZERO_ATTITUDE_FILENAME,
)
from bench.visualization.vizard_native_bridge import (
    NATIVE_BRIDGE_LOG_FILENAME,
    NATIVE_BRIDGE_MANIFEST_FILENAME,
    main,
    run_vizard_native_bridge,
)


def _native_input(*, n_step: int = 5) -> pd.DataFrame:
    times = np.repeat(np.arange(n_step, dtype=np.float64) * 0.1, 2)
    names = np.tile(["SC_true", "SC_estimated"], n_step)
    base = np.arange(2 * n_step, dtype=np.float64) * 0.001
    return pd.DataFrame(
        {
            "time_s": times,
            "sc_name": names,
            "r_BN_N_x_m": 7000.0e3 + base,
            "r_BN_N_y_m": base,
            "r_BN_N_z_m": np.zeros(2 * n_step),
            "v_BN_N_x_m_s": -base,
            "v_BN_N_y_m_s": np.full(2 * n_step, 8100.0),
            "v_BN_N_z_m_s": np.zeros(2 * n_step),
            "attitude_type": ["MRP"] * (2 * n_step),
            "sigma_BN_1": base,
            "sigma_BN_2": base + 0.01,
            "sigma_BN_3": base + 0.02,
            "omega_BN_B_x_rad_s": base + 0.03,
            "omega_BN_B_y_rad_s": base + 0.04,
            "omega_BN_B_z_rad_s": base + 0.05,
        }
    )


class VizardNativeBridgeTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.run_dir = self.root / "run"
        self.basilisk_dir = (
            self.run_dir / "artifacts" / "vizard" / "basilisk"
        )
        self.basilisk_dir.mkdir(parents=True)
        self.input_csv = self.basilisk_dir / "dataFileToViz_input.csv"

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _write(self, frame: pd.DataFrame) -> None:
        frame.to_csv(self.input_csv, index=False)

    def test_probe_only_bridge_and_frame_checks(self) -> None:
        self._write(_native_input())
        manifest_path, log_path = run_vizard_native_bridge(self.run_dir)
        self.assertTrue(manifest_path.exists())
        self.assertTrue(log_path.exists())
        self.assertTrue((manifest_path.parent / "basilisk_api_probe.json").exists())
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(
            manifest["native_conversion_status"],
            "not_attempted_probe_only",
        )

        frame_dir = self.basilisk_dir / "frame_check"
        for filename in (
            ZERO_ATTITUDE_FILENAME,
            POSITIVE_YAW_FILENAME,
            TRUE_ESTIMATED_OFFSET_FILENAME,
        ):
            path = frame_dir / filename
            self.assertTrue(path.exists())
            frame = pd.read_csv(path)
            self.assertTrue(frame["attitude_type"].eq("MRP").all())
        self.assertTrue((frame_dir / FRAME_CHECK_README_FILENAME).exists())

    def test_attempt_native_non_strict_records_result(self) -> None:
        self._write(_native_input())
        manifest_path, _ = run_vizard_native_bridge(
            self.run_dir,
            mode="attempt-native",
        )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertIn(
            manifest["native_conversion_status"],
            {
                "attempted_success",
                "attempted_failed",
                "not_attempted_basilisk_unavailable",
                "not_attempted_contract_unknown",
            },
        )

    def test_strict_native_behavior(self) -> None:
        self._write(_native_input())
        manifest_path, _ = run_vizard_native_bridge(
            self.run_dir,
            mode="attempt-native",
        )
        status = json.loads(
            manifest_path.read_text(encoding="utf-8")
        )["native_conversion_status"]
        if status == "attempted_success":
            strict_manifest, _ = run_vizard_native_bridge(
                self.run_dir,
                mode="attempt-native",
                require_native_success=True,
            )
            self.assertTrue(strict_manifest.exists())
        else:
            with self.assertRaises(RuntimeError):
                run_vizard_native_bridge(
                    self.run_dir,
                    mode="attempt-native",
                    require_native_success=True,
                )

    def test_missing_input_raises(self) -> None:
        with self.assertRaises(FileNotFoundError):
            run_vizard_native_bridge(self.run_dir)

    def test_missing_column_raises(self) -> None:
        self._write(_native_input().drop(columns=["sigma_BN_2"]))
        with self.assertRaisesRegex(ValueError, "missing required"):
            run_vizard_native_bridge(self.run_dir)

    def test_nonfinite_raises(self) -> None:
        frame = _native_input()
        frame.loc[0, "omega_BN_B_z_rad_s"] = np.inf
        self._write(frame)
        with self.assertRaisesRegex(
            ValueError,
            "omega_BN_B_z_rad_s.*NaN or Inf",
        ):
            run_vizard_native_bridge(self.run_dir)

    def test_duplicate_pair_raises(self) -> None:
        frame = _native_input()
        frame = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
        self._write(frame)
        with self.assertRaisesRegex(ValueError, "duplicate.*time_s, sc_name"):
            run_vizard_native_bridge(self.run_dir)

    def test_cli_run_dir_smoke(self) -> None:
        self._write(_native_input())
        with redirect_stdout(io.StringIO()):
            result = main(
                [
                    "--run-dir",
                    str(self.run_dir),
                    "--mode",
                    "probe-only",
                ]
            )
        self.assertEqual(result, 0)
        native_dir = self.basilisk_dir / "native"
        self.assertTrue((native_dir / NATIVE_BRIDGE_MANIFEST_FILENAME).exists())
        self.assertTrue((native_dir / NATIVE_BRIDGE_LOG_FILENAME).exists())

    def test_cli_direct_csv_smoke(self) -> None:
        self._write(_native_input())
        output_dir = self.root / "direct_native"
        with redirect_stdout(io.StringIO()):
            result = main(
                [
                    "--input-csv",
                    str(self.input_csv),
                    "--out-dir",
                    str(output_dir),
                    "--mode",
                    "probe-only",
                ]
            )
        self.assertEqual(result, 0)
        self.assertTrue(
            (output_dir / NATIVE_BRIDGE_MANIFEST_FILENAME).exists()
        )
        self.assertTrue((output_dir / NATIVE_BRIDGE_LOG_FILENAME).exists())


@dataclass
class VizardNativeBridgeResult:
    ok: bool
    note: str


def run_vizard_native_bridge_tests() -> VizardNativeBridgeResult:
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(
        VizardNativeBridgeTests
    )
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=1).run(suite)
    return VizardNativeBridgeResult(
        ok=bool(result.wasSuccessful()),
        note=(
            "Vizard native bridge tests passed"
            if result.wasSuccessful()
            else stream.getvalue().strip()
        ),
    )


if __name__ == "__main__":
    unittest.main()
