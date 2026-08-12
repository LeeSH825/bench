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

from bench.visualization.adcs_plots import main, make_adcs_plots


_REQUIRED_PLOTS = {
    "rpy_true_vs_hat.png",
    "rpy_error.png",
    "omega_true_vs_hat.png",
    "omega_error_norm.png",
    "mrp_error_norm.png",
}
_BIAS_PLOTS = {
    "bias_true_vs_hat.png",
    "bias_error_norm.png",
}


def _timeseries_frame(*, include_bias: bool = True) -> pd.DataFrame:
    n_seq, n_step = 2, 5
    traj_id = np.repeat(np.arange(n_seq), n_step)
    t_idx = np.tile(np.arange(n_step), n_seq)
    time_s = t_idx.astype(np.float64) * 0.1
    base = 0.01 * (traj_id + t_idx)
    data = {
        "traj_id": traj_id,
        "t_idx": t_idx,
        "time_s": time_s,
        "roll_true_rad": base,
        "pitch_true_rad": base + 0.01,
        "yaw_true_rad": base + 0.02,
        "roll_hat_rad": base + 0.001,
        "pitch_hat_rad": base + 0.011,
        "yaw_hat_rad": base + 0.021,
        "roll_err_rad": np.full(n_seq * n_step, 0.001),
        "pitch_err_rad": np.full(n_seq * n_step, 0.001),
        "yaw_err_rad": np.full(n_seq * n_step, 0.001),
        "omega_x_true_rad_s": base + 0.03,
        "omega_y_true_rad_s": base + 0.04,
        "omega_z_true_rad_s": base + 0.05,
        "omega_x_hat_rad_s": base + 0.031,
        "omega_y_hat_rad_s": base + 0.041,
        "omega_z_hat_rad_s": base + 0.051,
        "omega_err_norm_rad_s": np.full(n_seq * n_step, np.sqrt(3.0) * 0.001),
        "mrp_err_norm": np.full(n_seq * n_step, np.sqrt(3.0) * 0.0005),
    }
    if include_bias:
        data.update(
            {
                "bias_x_true_rad_s": base + 0.001,
                "bias_y_true_rad_s": base + 0.002,
                "bias_z_true_rad_s": base + 0.003,
                "bias_x_hat_rad_s": base + 0.0011,
                "bias_y_hat_rad_s": base + 0.0021,
                "bias_z_hat_rad_s": base + 0.0031,
                "bias_err_norm_rad_s": np.full(
                    n_seq * n_step,
                    np.sqrt(3.0) * 0.0001,
                ),
            }
        )
    return pd.DataFrame(data)


class ADCSPlotTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.artifacts = self.root / "artifacts"
        self.artifacts.mkdir(parents=True)
        self.csv_path = self.artifacts / "adcs_timeseries.csv"

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _write_frame(self, frame: pd.DataFrame) -> Path:
        frame.to_csv(self.csv_path, index=False)
        return self.csv_path

    def test_complete_bias_columns_generate_all_plots_and_manifest(self) -> None:
        self._write_frame(_timeseries_frame(include_bias=True))
        plot_paths, manifest_path = make_adcs_plots(
            self.csv_path,
            trajectory_id=0,
            dpi=72,
        )
        names = {path.name for path in plot_paths}
        self.assertEqual(names, _REQUIRED_PLOTS | _BIAS_PLOTS)
        self.assertTrue(
            all(path.exists() and path.stat().st_size > 0 for path in plot_paths)
        )
        self.assertTrue(manifest_path.exists())

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(manifest["schema_version"], "adcs_plot_manifest_v1")
        self.assertEqual(manifest["trajectory_id"], 0)
        self.assertEqual(manifest["num_rows_plotted"], 5)
        self.assertEqual(
            {Path(path).name for path in manifest["generated_plots"]},
            names,
        )
        self.assertEqual(manifest["skipped_plots"], [])

    def test_absent_bias_columns_skip_bias_plots(self) -> None:
        self._write_frame(_timeseries_frame(include_bias=False))
        plot_paths, manifest_path = make_adcs_plots(self.csv_path, dpi=72)
        self.assertEqual({path.name for path in plot_paths}, _REQUIRED_PLOTS)
        self.assertTrue(all(path.exists() for path in plot_paths))
        self.assertTrue(
            all(not (manifest_path.parent / name).exists() for name in _BIAS_PLOTS)
        )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(len(manifest["skipped_plots"]), 2)
        self.assertTrue(
            all("absent" in item["reason"] for item in manifest["skipped_plots"])
        )

    def test_plot_bias_false_skips_complete_bias_columns(self) -> None:
        self._write_frame(_timeseries_frame(include_bias=True))
        plot_paths, manifest_path = make_adcs_plots(
            self.csv_path,
            plot_bias=False,
            dpi=72,
        )
        self.assertEqual({path.name for path in plot_paths}, _REQUIRED_PLOTS)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertTrue(
            all("disabled" in item["reason"] for item in manifest["skipped_plots"])
        )

    def test_partial_bias_columns_raise(self) -> None:
        frame = _timeseries_frame(include_bias=True).drop(
            columns=["bias_err_norm_rad_s"]
        )
        self._write_frame(frame)
        with self.assertRaisesRegex(ValueError, "partial gyro-bias"):
            make_adcs_plots(self.csv_path, dpi=72)

    def test_missing_required_column_raises(self) -> None:
        frame = _timeseries_frame().drop(columns=["roll_true_rad"])
        self._write_frame(frame)
        with self.assertRaisesRegex(ValueError, "missing required"):
            make_adcs_plots(self.csv_path, dpi=72)

    def test_nonfinite_plotted_column_raises(self) -> None:
        frame = _timeseries_frame()
        frame.loc[0, "yaw_hat_rad"] = np.inf
        self._write_frame(frame)
        with self.assertRaisesRegex(ValueError, "yaw_hat_rad.*NaN or Inf"):
            make_adcs_plots(self.csv_path, dpi=72)

    def test_missing_csv_and_unknown_trajectory_raise(self) -> None:
        with self.assertRaises(FileNotFoundError):
            make_adcs_plots(self.root / "missing.csv", dpi=72)

        self._write_frame(_timeseries_frame())
        with self.assertRaisesRegex(ValueError, "does not exist"):
            make_adcs_plots(self.csv_path, trajectory_id=99, dpi=72)

    def test_cli_run_dir_mode_generates_required_plots(self) -> None:
        self._write_frame(_timeseries_frame(include_bias=False))
        with redirect_stdout(io.StringIO()):
            result = main(
                [
                    "--run-dir",
                    str(self.root),
                    "--trajectory-id",
                    "1",
                    "--dpi",
                    "72",
                ]
            )
        self.assertEqual(result, 0)
        plots_dir = self.artifacts / "plots"
        self.assertTrue(
            all((plots_dir / filename).exists() for filename in _REQUIRED_PLOTS)
        )
        self.assertTrue((plots_dir / "adcs_plot_manifest.json").exists())


@dataclass
class ADCSPlotResult:
    ok: bool
    note: str


def run_adcs_plot_tests() -> ADCSPlotResult:
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(ADCSPlotTests)
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=1).run(suite)
    return ADCSPlotResult(
        ok=bool(result.wasSuccessful()),
        note=(
            "ADCS plot tests passed"
            if result.wasSuccessful()
            else stream.getvalue().strip()
        ),
    )


if __name__ == "__main__":
    unittest.main()
