from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import yaml

from bench.visualization.checkpoint_contract_probe import (
    probe_checkpoint_contract,
)
from bench.visualization.phase6e_checkpoint_package import (
    validate_replay_checkpoint_package,
)
from bench.visualization.phase6f_kalmannet_export import (
    KALMANNET_MODEL_CONFIG_SCHEMA_VERSION,
    KALMANNET_SYSTEM_MODEL_SCHEMA_VERSION,
    export_kalmannet_tsp_replay_package,
    main,
)


class Phase6FKalmanNetExportTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.checkpoint = self.root / "source.pt"
        torch.save(
            {
                "state_dict": {},
                "model_class": "KalmanNetNN",
                "best_step": 5,
            },
            self.checkpoint,
        )
        self.model_config = self.root / "model_config.json"
        self.model_config.write_text(
            json.dumps(
                {
                    "schema_version": (
                        KALMANNET_MODEL_CONFIG_SCHEMA_VERSION
                    ),
                    "model_id": "kalmannet_tsp",
                    "state_dim": 9,
                    "measurement_dim": 6,
                    "input_layout": "NTD",
                    "output_layout": "NTD",
                    "repo": {"path": "third_party/KalmanNet_TSP"},
                    "in_mult_KNet": 5,
                    "out_mult_KNet": 40,
                    "normalization": {"enabled": False},
                    "hidden_state_initialization": {
                        "method": "zeros",
                    },
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        self.system_model = self.root / "system_model.json"
        self.system_model.write_text(
            json.dumps(
                {
                    "schema_version": (
                        KALMANNET_SYSTEM_MODEL_SCHEMA_VERSION
                    ),
                    "format": "linear_F_H",
                    "state_dim": 9,
                    "measurement_dim": 6,
                    "F": np.eye(9).tolist(),
                    "H": np.eye(9, dtype=float)[:6].tolist(),
                    "Q": (np.eye(9) * 1.0e-8).tolist(),
                    "R": (np.eye(6) * 1.0e-6).tolist(),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        self.training_summary = self.root / "training_summary.json"
        self.training_summary.write_text(
            json.dumps(
                {
                    "smoke_training": True,
                    "benchmark_reporting_recommended": False,
                    "updates_used": 5,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _export(self, package: Path, *, overwrite: bool = False) -> Path:
        return export_kalmannet_tsp_replay_package(
            checkpoint=self.checkpoint,
            package_dir=package,
            model_config=self.model_config,
            system_model=self.system_model,
            training_summary=self.training_summary,
            training_suite_name="phase6f_kalmannet_adcs_tiny",
            training_task_id="phase6f_adcs_bias_tiny_v0",
            training_seed=0,
            checkpoint_step=5,
            overwrite=overwrite,
        )

    def test_missing_checkpoint_raises(self) -> None:
        with self.assertRaises(FileNotFoundError):
            export_kalmannet_tsp_replay_package(
                checkpoint=self.root / "missing.pt",
                package_dir=self.root / "missing_package",
                model_config=self.model_config,
                system_model=self.system_model,
            )

    def test_export_builds_strict_9x6_package(self) -> None:
        package = self._export(self.root / "package")
        self.assertTrue((package / "checkpoint.pt").exists())
        self.assertTrue((package / "model_config.json").exists())
        self.assertTrue((package / "system_model.json").exists())
        self.assertTrue((package / "training_summary.json").exists())
        self.assertTrue((package / "replay_contract.json").exists())
        self.assertTrue(
            (package / "checkpoint_contract_probe.json").exists()
        )

        contract = validate_replay_checkpoint_package(
            package,
            expected_state_dim=9,
            expected_measurement_dim=6,
            expected_observed_state=[0, 1, 2, 3, 4, 5],
        )
        self.assertEqual(contract["model_id"], "kalmannet_tsp")
        self.assertEqual(
            contract["adapter_id"],
            "kalmannet_tsp_replay_adapter_v1",
        )
        self.assertTrue(contract["requires_system_model"])
        self.assertFalse(contract["requires_normalization"])
        self.assertFalse(contract["checkpoint_compatibility_verified"])
        self.assertTrue(contract["smoke_training"])
        warning_text = " ".join(str(item) for item in contract["warnings"])
        self.assertIn("adapter is structurally registered", warning_text)
        self.assertIn(
            "checkpoint/runtime compatibility remains unverified",
            warning_text,
        )
        self.assertNotIn("support remains disabled", warning_text)

        probe = probe_checkpoint_contract(
            package,
            model_id="kalmannet_tsp",
        )
        self.assertTrue(probe["replay_contract_present"])
        self.assertTrue(probe["replay_contract_valid"])
        self.assertTrue(probe["package_supported_for_phase6d"])
        self.assertIn(
            "explicit adapter is registered",
            probe["support_reason"],
        )

    def test_existing_package_without_overwrite_raises(self) -> None:
        package = self._export(self.root / "existing")
        with self.assertRaises(FileExistsError):
            self._export(package)

    def test_cli_export(self) -> None:
        package = self.root / "cli_package"
        with redirect_stdout(io.StringIO()):
            result = main(
                [
                    "--checkpoint",
                    str(self.checkpoint),
                    "--package-dir",
                    str(package),
                    "--model-config",
                    str(self.model_config),
                    "--system-model",
                    str(self.system_model),
                    "--training-summary",
                    str(self.training_summary),
                    "--training-suite-name",
                    "phase6f_kalmannet_adcs_tiny",
                    "--training-task-id",
                    "phase6f_adcs_bias_tiny_v0",
                    "--training-seed",
                    "0",
                    "--checkpoint-step",
                    "5",
                ]
            )
        self.assertEqual(result, 0)
        self.assertTrue((package / "replay_contract.json").exists())

    def test_phase6f_suite_has_required_contract(self) -> None:
        suite_path = (
            Path(__file__).resolve().parents[1]
            / "configs"
            / "suite_phase6f_kalmannet_adcs_tiny.yaml"
        )
        suite = yaml.safe_load(suite_path.read_text(encoding="utf-8"))
        task = suite["tasks"][0]
        self.assertEqual(task["system_type"], "adcs_replay")
        self.assertEqual(task["x_dim"], 9)
        self.assertEqual(task["y_dim"], 6)
        self.assertEqual(
            task["observation"]["observed_state"],
            [0, 1, 2, 3, 4, 5],
        )
        self.assertGreater(task["dataset_sizes"]["N_train"], 0)
        self.assertEqual(suite["models"][0]["model_id"], "kalmannet_tsp")


@dataclass
class Phase6FKalmanNetExportResult:
    ok: bool
    note: str


def run_phase6f_kalmannet_export_tests() -> Phase6FKalmanNetExportResult:
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(
        Phase6FKalmanNetExportTests
    )
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=1).run(suite)
    return Phase6FKalmanNetExportResult(
        ok=bool(result.wasSuccessful()),
        note=(
            "Phase 6F KalmanNet export tests passed"
            if result.wasSuccessful()
            else stream.getvalue().strip()
        ),
    )


if __name__ == "__main__":
    unittest.main()
