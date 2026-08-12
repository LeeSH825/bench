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

from bench.visualization.vizard_export import (
    VIZARD_MANIFEST_FILENAME,
    VIZARD_STATES_FILENAME,
    export_vizard_offline,
    main,
)


_POSITION_COLUMNS = ("r_BN_N_x_m", "r_BN_N_y_m", "r_BN_N_z_m")
_VELOCITY_COLUMNS = ("v_BN_N_x_m_s", "v_BN_N_y_m_s", "v_BN_N_z_m_s")


def _timeseries_frame(*, n_seq: int = 1, n_step: int = 5) -> pd.DataFrame:
    traj_id = np.repeat(np.arange(n_seq, dtype=np.int64), n_step)
    t_idx = np.tile(np.arange(n_step, dtype=np.int64), n_seq)
    time_s = t_idx.astype(np.float64) * 100.0
    base = 0.01 * (traj_id + t_idx)
    return pd.DataFrame(
        {
            "traj_id": traj_id,
            "t_idx": t_idx,
            "time_s": time_s,
            "sigma1_true": base,
            "sigma2_true": base + 0.01,
            "sigma3_true": base + 0.02,
            "sigma1_hat": base + 0.001,
            "sigma2_hat": base + 0.011,
            "sigma3_hat": base + 0.021,
            "omega_x_true_rad_s": base + 0.03,
            "omega_y_true_rad_s": base + 0.04,
            "omega_z_true_rad_s": base + 0.05,
            "omega_x_hat_rad_s": base + 0.031,
            "omega_y_hat_rad_s": base + 0.041,
            "omega_z_hat_rad_s": base + 0.051,
        }
    )


class VizardExportTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.run_dir = Path(self._tmp.name)
        self.artifacts = self.run_dir / "artifacts"
        self.artifacts.mkdir(parents=True)
        self.csv_path = self.artifacts / "adcs_timeseries.csv"
        self.meta_path = self.artifacts / "adcs_timeseries_meta.json"

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _write_inputs(self, frame: pd.DataFrame) -> None:
        frame.to_csv(self.csv_path, index=False)
        self.meta_path.write_text(
            json.dumps(
                {
                    "schema_version": "adcs_timeseries_v1",
                    "num_rows": int(len(frame)),
                    "time_unit": "s",
                }
            )
            + "\n",
            encoding="utf-8",
        )

    def test_fixed_origin_export(self) -> None:
        n_step = 5
        self._write_inputs(_timeseries_frame(n_step=n_step))
        csv_path, manifest_path = export_vizard_offline(
            self.run_dir,
            position_source="fixed_origin",
        )
        self.assertEqual(csv_path.name, VIZARD_STATES_FILENAME)
        self.assertEqual(manifest_path.name, VIZARD_MANIFEST_FILENAME)
        self.assertTrue(csv_path.exists())
        self.assertTrue(manifest_path.exists())

        output = pd.read_csv(csv_path)
        self.assertEqual(len(output), 2 * n_step)
        self.assertEqual(set(output["source"]), {"true", "estimated"})
        self.assertEqual(set(output["sc_name"]), {"SC_true", "SC_estimated"})
        self.assertTrue(
            np.equal(
                output[list(_POSITION_COLUMNS + _VELOCITY_COLUMNS)].to_numpy(),
                0.0,
            ).all()
        )
        self.assertEqual(
            output["source"].tolist(),
            ["true", "estimated"] * n_step,
        )

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(manifest["schema_version"], "vizard_export_v1")
        self.assertEqual(manifest["num_timestamps"], n_step)
        self.assertEqual(manifest["num_rows"], 2 * n_step)
        self.assertFalse(manifest["official_metrics_affected"])

    def test_dummy_circular_orbit_export(self) -> None:
        n_step = 5
        self._write_inputs(_timeseries_frame(n_step=n_step))
        csv_path, manifest_path = export_vizard_offline(
            self.run_dir,
            position_source="dummy_circular_orbit",
        )
        output = pd.read_csv(csv_path)
        orbit_values = output[
            list(_POSITION_COLUMNS + _VELOCITY_COLUMNS)
        ].to_numpy()
        self.assertEqual(len(output), 2 * n_step)
        self.assertTrue(np.isfinite(orbit_values).all())
        true_rows = output.loc[output["source"] == "true"]
        self.assertGreater(true_rows["r_BN_N_y_m"].nunique(), 1)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(manifest["position_source"], "dummy_circular_orbit")
        self.assertIn("Synthetic visualization-only", manifest["position_source_notes"])

    def test_missing_required_column_raises(self) -> None:
        frame = _timeseries_frame().drop(columns=["sigma2_hat"])
        self._write_inputs(frame)
        with self.assertRaisesRegex(ValueError, "missing required"):
            export_vizard_offline(self.run_dir)

    def test_trajectory_filtering(self) -> None:
        n_step = 5
        self._write_inputs(_timeseries_frame(n_seq=2, n_step=n_step))
        csv_path, manifest_path = export_vizard_offline(
            self.run_dir,
            trajectory_id=1,
        )
        output = pd.read_csv(csv_path)
        self.assertEqual(set(output["traj_id"]), {1})
        self.assertEqual(len(output), 2 * n_step)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(manifest["trajectory_id"], 1)

    def test_missing_trajectory_raises(self) -> None:
        self._write_inputs(_timeseries_frame())
        with self.assertRaisesRegex(ValueError, "does not exist"):
            export_vizard_offline(self.run_dir, trajectory_id=99)

    def test_nonfinite_input_raises(self) -> None:
        for value in (np.nan, np.inf):
            with self.subTest(value=value):
                frame = _timeseries_frame()
                frame.loc[0, "omega_z_hat_rad_s"] = value
                self._write_inputs(frame)
                with self.assertRaisesRegex(
                    ValueError,
                    "omega_z_hat_rad_s.*NaN or Inf",
                ):
                    export_vizard_offline(self.run_dir)

    def test_missing_input_artifact_and_unsupported_source_raise(self) -> None:
        with self.assertRaises(FileNotFoundError):
            export_vizard_offline(self.run_dir)

        self._write_inputs(_timeseries_frame())
        with self.assertRaisesRegex(ValueError, "unsupported position_source"):
            export_vizard_offline(
                self.run_dir,
                position_source="true_orbit",  # type: ignore[arg-type]
            )

    def test_cli_smoke(self) -> None:
        self._write_inputs(_timeseries_frame())
        with redirect_stdout(io.StringIO()):
            result = main(
                [
                    "--run-dir",
                    str(self.run_dir),
                    "--trajectory-id",
                    "0",
                    "--position-source",
                    "fixed_origin",
                ]
            )
        self.assertEqual(result, 0)
        output_dir = self.artifacts / "vizard"
        self.assertTrue((output_dir / VIZARD_STATES_FILENAME).exists())
        self.assertTrue((output_dir / VIZARD_MANIFEST_FILENAME).exists())


@dataclass
class VizardExportResult:
    ok: bool
    note: str


def run_vizard_export_tests() -> VizardExportResult:
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(VizardExportTests)
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=1).run(suite)
    return VizardExportResult(
        ok=bool(result.wasSuccessful()),
        note=(
            "Vizard offline export tests passed"
            if result.wasSuccessful()
            else stream.getvalue().strip()
        ),
    )


if __name__ == "__main__":
    unittest.main()
