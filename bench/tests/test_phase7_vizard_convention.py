from __future__ import annotations

import base64
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from bench.visualization.phase7_vizard_convention import (
    CANDIDATES_DIRNAME,
    PHASE7_LOCKED_FILENAME,
    PHASE7_MANIFEST_FILENAME,
    PHASE7_README_FILENAME,
    PHASE7_REPORT_FILENAME,
    PHASE7_TEMPLATE_FILENAME,
    build_phase7_vizard_convention_package,
    lock_vizard_convention,
    main,
)
from bench.visualization.phase6c_replay_visualization import (
    PHASE6C_SUMMARY_FILENAME,
)
from bench.visualization.vizard_convention import (
    SUPPORTED_VIZARD_CONVENTION_IDS,
    apply_vizard_convention_to_frame,
    build_vizard_convention,
    load_vizard_convention,
    save_vizard_convention,
)
from bench.visualization.vizard_export import (
    VIZARD_MANIFEST_FILENAME,
    VIZARD_STATES_FILENAME,
    export_vizard_offline,
)
from bench.visualization.pred_artifact import save_pred_artifact


_PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/wIAAgMBApV0mSAAAAAASUVORK5CYII="
)


def _timeseries_frame() -> pd.DataFrame:
    time_s = np.array([0.0, 1.0, 2.0], dtype=np.float64)
    traj_id = np.zeros_like(time_s, dtype=np.int64)
    t_idx = np.arange(time_s.size, dtype=np.int64)
    base = np.array([0.1, 0.2, 0.3], dtype=np.float64)
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


def _write_png(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_PNG_BYTES)


def _state_meta() -> dict[str, object]:
    return {
        "phase": "phase6b_checkpoint_replay",
        "model_id": "replay_identity_baseline",
        "scenario_id": "scenario_phase7",
        "suite_name": "phase7_test_suite",
        "task_id": "phase7_test_task",
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


def _prepare_source_run_dir(root: Path) -> Path:
    run_dir = root / "source_run"
    artifacts = run_dir / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)

    time_s = np.array([0.0, 1.0, 2.0], dtype=np.float64)
    x_true = np.zeros((1, time_s.size, 9), dtype=np.float64)
    x_true[0, :, 0] = np.array([0.1, 0.2, 0.3])
    x_true[0, :, 1] = np.array([0.01, 0.02, 0.03])
    x_true[0, :, 2] = np.array([0.001, 0.002, 0.003])
    x_true[0, :, 3:6] = np.array([0.01, -0.02, 0.03])
    x_true[0, :, 6:9] = np.array([0.001, -0.001, 0.002])
    y_obs = x_true[..., :6].copy()
    x_hat = x_true.copy()
    x_hat[..., :6] = y_obs
    save_pred_artifact(
        artifacts,
        time_s=time_s,
        x_true=x_true,
        y_obs=y_obs,
        x_hat=x_hat,
        trajectory_id=np.array([0], dtype=np.int64),
        meta=_state_meta(),
    )

    _timeseries_frame().to_csv(artifacts / "adcs_timeseries.csv", index=False)
    (artifacts / "adcs_timeseries_meta.json").write_text(
        json.dumps(
            {
                "schema_version": "adcs_timeseries_v1",
                "num_rows": 3,
                "trajectory_count": 1,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    export_vizard_offline(run_dir, position_source="dummy_circular_orbit")
    plots_dir = artifacts / "plots"
    for name in (
        "rpy_true_vs_hat.png",
        "rpy_error.png",
        "omega_true_vs_hat.png",
        "omega_error_norm.png",
        "mrp_error_norm.png",
    ):
        _write_png(plots_dir / name)
    (artifacts / "phase6c_replay_visualization_summary.json").write_text(
        json.dumps(
            {
                "schema_version": "phase6c_replay_visualization_summary_v1",
                "trajectory_id": 0,
                "position_source": "dummy_circular_orbit",
                "num_timestamps": 3,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return run_dir


def _fake_native_bridge(run_dir: str | Path, **_: object) -> tuple[Path, Path]:
    root = Path(run_dir).resolve()
    native_dir = root / "artifacts" / "vizard" / "basilisk" / "native"
    native_dir.mkdir(parents=True, exist_ok=True)
    (native_dir / "basilisk_api_probe.json").write_text(
        json.dumps(
            {
                "schema_version": "basilisk_api_probe_v1",
                "basilisk_available": True,
                "basilisk_version": "test",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    manifest_path = native_dir / "native_bridge_manifest.json"
    log_path = native_dir / "native_bridge_log.txt"
    (native_dir / "native_conversion_output_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "native_conversion_output_v1",
                "native_conversion_status": "attempted_success",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (native_dir / "vizard_playback.bin").write_bytes(b"playback")
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "vizard_native_bridge_v1",
                "native_conversion_status": "attempted_success",
                "native_conversion_attempted": True,
                "native_conversion_error": None,
                "official_metrics_affected": False,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    log_path.write_text("native bridge ok\n", encoding="utf-8")
    return manifest_path, log_path


class Phase7VizardConventionTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.run_dir = _prepare_source_run_dir(self.root)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_convention_transforms(self) -> None:
        frame = pd.DataFrame(
            {
                "sigma_BN_1": [1.0],
                "sigma_BN_2": [-2.0],
                "sigma_BN_3": [3.0],
                "omega_BN_B_x_rad_s": [4.0],
                "omega_BN_B_y_rad_s": [-5.0],
                "omega_BN_B_z_rad_s": [6.0],
            }
        )
        direct = apply_vizard_convention_to_frame(
            frame,
            build_vizard_convention("direct"),
        )
        self.assertTrue(direct.equals(frame))
        attitude_inverse = apply_vizard_convention_to_frame(
            frame,
            build_vizard_convention("attitude_inverse"),
        )
        self.assertTrue(
            np.allclose(
                attitude_inverse[["sigma_BN_1", "sigma_BN_2", "sigma_BN_3"]],
                -frame[["sigma_BN_1", "sigma_BN_2", "sigma_BN_3"]],
            )
        )
        self.assertTrue(
            np.allclose(
                attitude_inverse[[
                    "omega_BN_B_x_rad_s",
                    "omega_BN_B_y_rad_s",
                    "omega_BN_B_z_rad_s",
                ]],
                frame[[
                    "omega_BN_B_x_rad_s",
                    "omega_BN_B_y_rad_s",
                    "omega_BN_B_z_rad_s",
                ]],
            )
        )
        omega_negated = apply_vizard_convention_to_frame(
            frame,
            build_vizard_convention("omega_negated"),
        )
        self.assertTrue(
            np.allclose(
                omega_negated[["sigma_BN_1", "sigma_BN_2", "sigma_BN_3"]],
                frame[["sigma_BN_1", "sigma_BN_2", "sigma_BN_3"]],
            )
        )
        self.assertTrue(
            np.allclose(
                omega_negated[[
                    "omega_BN_B_x_rad_s",
                    "omega_BN_B_y_rad_s",
                    "omega_BN_B_z_rad_s",
                ]],
                -frame[[
                    "omega_BN_B_x_rad_s",
                    "omega_BN_B_y_rad_s",
                    "omega_BN_B_z_rad_s",
                ]],
            )
        )
        both = apply_vizard_convention_to_frame(
            frame,
            build_vizard_convention("attitude_inverse_omega_negated"),
        )
        self.assertTrue(
            np.allclose(
                both[["sigma_BN_1", "sigma_BN_2", "sigma_BN_3"]],
                -frame[["sigma_BN_1", "sigma_BN_2", "sigma_BN_3"]],
            )
        )
        self.assertTrue(
            np.allclose(
                both[[
                    "omega_BN_B_x_rad_s",
                    "omega_BN_B_y_rad_s",
                    "omega_BN_B_z_rad_s",
                ]],
                -frame[[
                    "omega_BN_B_x_rad_s",
                    "omega_BN_B_y_rad_s",
                    "omega_BN_B_z_rad_s",
                ]],
            )
        )

    def test_candidate_package_generation_and_locking(self) -> None:
        phase7_dir = self.root / "phase7"
        with patch(
            "bench.visualization.phase7_vizard_convention."
            "run_vizard_native_bridge",
            side_effect=_fake_native_bridge,
        ):
            manifest_path, readme_path = build_phase7_vizard_convention_package(
                self.run_dir,
                out_dir=phase7_dir,
                trajectory_id=0,
                position_source="dummy_circular_orbit",
                overwrite=True,
            )
        self.assertTrue(manifest_path.exists())
        self.assertTrue(readme_path.exists())
        self.assertTrue((phase7_dir / PHASE7_TEMPLATE_FILENAME).exists())
        self.assertTrue((phase7_dir / "plots" / "rpy_true_vs_hat.png").exists())
        self.assertTrue((phase7_dir / PHASE6C_SUMMARY_FILENAME).exists())

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(
            manifest["schema_version"],
            "phase7_vizard_convention_v1",
        )
        self.assertEqual(
            set(manifest["candidate_ids"]),
            set(SUPPORTED_VIZARD_CONVENTION_IDS),
        )
        self.assertEqual(manifest["trajectory_id"], 0)
        self.assertEqual(manifest["position_source"], "dummy_circular_orbit")
        self.assertFalse(manifest["official_metrics_affected"])

        for convention_id in SUPPORTED_VIZARD_CONVENTION_IDS:
            candidate_dir = phase7_dir / CANDIDATES_DIRNAME / convention_id
            self.assertTrue(candidate_dir.exists())
            candidate_manifest = json.loads(
                (candidate_dir / "candidate_manifest.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(candidate_manifest["convention_id"], convention_id)
            self.assertEqual(candidate_manifest["trajectory_id"], 0)
            self.assertEqual(candidate_manifest["spacecraft_count"], 2)
            self.assertEqual(candidate_manifest["time_start_s"], 0.0)
            self.assertEqual(candidate_manifest["time_end_s"], 2.0)
            self.assertEqual(candidate_manifest["native_conversion_status"], "attempted_success")
            csv = pd.read_csv(candidate_dir / VIZARD_STATES_FILENAME)
            self.assertEqual(set(csv["sc_name"]), {"SC_true", "SC_estimated"})
            self.assertEqual(set(csv["traj_id"]), {0})
            self.assertEqual(len(csv), 6)

        locked_path, report_path = lock_vizard_convention(
            phase7_dir,
            "direct",
            confirmed_by="manual_vizard_inspection",
            notes="Direct convention matched the expected motion.",
        )
        self.assertTrue(locked_path.exists())
        self.assertTrue(report_path.exists())
        locked = json.loads(locked_path.read_text(encoding="utf-8"))
        self.assertEqual(locked["schema_version"], "vizard_convention_v1")
        self.assertEqual(locked["convention_id"], "direct")
        self.assertEqual(locked["manual_confirmation_status"], "confirmed")
        self.assertEqual(locked["confirmed_by"], "manual_vizard_inspection")
        self.assertTrue(locked["confirmed_at_utc"])

        with self.assertRaises(ValueError):
            lock_vizard_convention(phase7_dir, "not-a-convention")

    def test_export_with_locked_convention_changes_signs(self) -> None:
        locked_path = save_vizard_convention(
            build_vizard_convention(
                "attitude_inverse",
                manual_confirmation_status="confirmed",
                confirmed_by="manual_vizard_inspection",
                source_run_dir=self.run_dir,
                notes=["test"],
            ),
            self.root / "attitude_inverse.json",
        )

        default_dir = self.root / "default_vizard"
        locked_dir = self.root / "locked_vizard"
        direct_dir = self.root / "direct_vizard"
        csv_default, _ = export_vizard_offline(
            self.run_dir,
            position_source="dummy_circular_orbit",
            out_dir=default_dir,
        )
        csv_locked, manifest_path = export_vizard_offline(
            self.run_dir,
            position_source="dummy_circular_orbit",
            out_dir=locked_dir,
            vizard_convention=locked_path,
        )
        default = pd.read_csv(csv_default)
        locked = pd.read_csv(csv_locked)
        self.assertTrue(
            np.allclose(
                locked[["sigma_BN_1", "sigma_BN_2", "sigma_BN_3"]],
                -default[["sigma_BN_1", "sigma_BN_2", "sigma_BN_3"]],
            )
        )
        self.assertTrue(
            np.allclose(
                locked[[
                    "omega_BN_B_x_rad_s",
                    "omega_BN_B_y_rad_s",
                    "omega_BN_B_z_rad_s",
                ]],
                default[[
                    "omega_BN_B_x_rad_s",
                    "omega_BN_B_y_rad_s",
                    "omega_BN_B_z_rad_s",
                ]],
            )
        )
        default_direct, _ = export_vizard_offline(
            self.run_dir,
            position_source="dummy_circular_orbit",
            out_dir=direct_dir,
            vizard_convention=build_vizard_convention("direct"),
        )
        pd.testing.assert_frame_equal(
            pd.read_csv(csv_default),
            pd.read_csv(default_direct),
        )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(manifest["convention_id"], "attitude_inverse")
        self.assertEqual(manifest["manual_confirmation_status"], "confirmed")
        self.assertEqual(manifest["attitude_mrp_mapping"], "sigma_BN_inverse")
        self.assertEqual(manifest["omega_mapping"], "omega_BN_B_direct")

    def test_cli_smoke_generation_and_lock(self) -> None:
        phase7_dir = self.root / "phase7_cli"
        with patch(
            "bench.visualization.phase7_vizard_convention."
            "run_vizard_native_bridge",
            side_effect=_fake_native_bridge,
        ):
            with redirect_stdout(io.StringIO()):
                result = main(
                    [
                        "--pred-run-dir",
                        str(self.run_dir),
                        "--out-dir",
                        str(phase7_dir),
                        "--trajectory-id",
                        "0",
                        "--position-source",
                        "dummy_circular_orbit",
                        "--overwrite",
                    ]
                )
        self.assertEqual(result, 0)
        self.assertTrue((phase7_dir / PHASE7_MANIFEST_FILENAME).exists())
        self.assertTrue((phase7_dir / PHASE7_README_FILENAME).exists())
        self.assertTrue((phase7_dir / PHASE7_TEMPLATE_FILENAME).exists())
        self.assertTrue((phase7_dir / CANDIDATES_DIRNAME).exists())

        with redirect_stdout(io.StringIO()):
            result = main(
                [
                    "--lock-convention",
                    "direct",
                    "--phase7-dir",
                    str(phase7_dir),
                    "--confirmed-by",
                    "manual_vizard_inspection",
                    "--notes",
                    "Direct convention locked.",
                ]
            )
        self.assertEqual(result, 0)
        self.assertTrue((phase7_dir / PHASE7_LOCKED_FILENAME).exists())
        self.assertTrue((phase7_dir / PHASE7_REPORT_FILENAME).exists())


def run_phase7_vizard_convention_tests() -> tuple[bool, str]:
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(
        Phase7VizardConventionTests
    )
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=1).run(suite)
    return bool(result.wasSuccessful()), (
        "Phase 7 Vizard convention tests passed"
        if result.wasSuccessful()
        else stream.getvalue().strip()
    )


if __name__ == "__main__":
    unittest.main()
