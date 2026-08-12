from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path

import torch
import yaml

from bench.visualization.checkpoint_contract_probe import (
    probe_checkpoint_contract,
)
from bench.visualization.phase6e_checkpoint_package import (
    validate_replay_checkpoint_package,
)
from bench.visualization.phase6g_kalmannet_export import (
    export_phase6g_kalmannet_tsp_package,
)


ENV_RUN_TINY_TRAIN = "AI_ADCS_PHASE6G_RUN_TINY_TRAIN"


class Phase6GExportProvenanceTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.run_dir = self.root / "run"
        checkpoints = self.run_dir / "checkpoints"
        checkpoints.mkdir(parents=True)

        suite_path = (
            Path(__file__).resolve().parents[1]
            / "configs"
            / "suite_phase6f_kalmannet_adcs_tiny.yaml"
        )
        suite = yaml.safe_load(suite_path.read_text(encoding="utf-8"))
        snapshot = {
            "suite": suite["suite"],
            "task": suite["tasks"][0],
            "model": suite["models"][0],
            "seed": 0,
        }
        (self.run_dir / "config_snapshot.yaml").write_text(
            yaml.safe_dump(snapshot, sort_keys=False),
            encoding="utf-8",
        )
        torch.save({"state_dict": {}}, checkpoints / "model.pt")
        self._write_json(checkpoints / "train_state.json", {"best_step": 4})
        self._write_json(self.run_dir / "metrics.json", {"status": "ok"})
        self._write_json(self.run_dir / "run_plan.json", {"init_id": "trained"})

    def tearDown(self) -> None:
        self._tmp.cleanup()

    @staticmethod
    def _write_json(path: Path, value: object) -> None:
        path.write_text(json.dumps(value) + "\n", encoding="utf-8")

    def test_nested_destination_and_top_level_best_step(self) -> None:
        destination = self.root / "missing" / "parents" / "package"
        exported = export_phase6g_kalmannet_tsp_package(
            source_run_dir=self.run_dir,
            package_dir=destination,
        )
        contract = validate_replay_checkpoint_package(exported)
        self.assertEqual(exported, destination.resolve())
        self.assertEqual(contract["checkpoint_step"], 4)

    def test_missing_required_provenance_is_rejected(self) -> None:
        (self.run_dir / "metrics.json").unlink()
        with self.assertRaisesRegex(FileNotFoundError, "required run provenance"):
            export_phase6g_kalmannet_tsp_package(
                source_run_dir=self.run_dir,
                package_dir=self.root / "package",
            )

    def test_malformed_train_state_is_rejected(self) -> None:
        (self.run_dir / "checkpoints" / "train_state.json").write_text(
            "not-json\n",
            encoding="utf-8",
        )
        with self.assertRaises(ValueError):
            export_phase6g_kalmannet_tsp_package(
                source_run_dir=self.run_dir,
                package_dir=self.root / "package",
            )

    def test_best_step_must_be_top_level_non_negative_integer(self) -> None:
        train_state = self.run_dir / "checkpoints" / "train_state.json"
        for invalid in (
            {},
            {"checkpoints": {"best_step": 4}},
            {"best_step": True},
            {"best_step": -1},
        ):
            with self.subTest(invalid=invalid):
                self._write_json(train_state, invalid)
                with self.assertRaisesRegex(ValueError, "top-level best_step"):
                    export_phase6g_kalmannet_tsp_package(
                        source_run_dir=self.run_dir,
                        package_dir=self.root / "package",
                    )


class Phase6GKalmanNetTinyTrainExportTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.bench_root = Path(__file__).resolve().parents[2]
        self.suite_yaml = (
            self.bench_root
            / "bench"
            / "configs"
            / "suite_phase6f_kalmannet_adcs_tiny.yaml"
        )

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _find_training_run_dir(self) -> Path:
        run_root = (
            self.bench_root
            / "runs"
            / "phase6f_kalmannet_adcs_tiny"
            / "phase6f_adcs_bias_tiny_v0"
            / "kalmannet_tsp"
            / "frozen"
            / "seed_0"
        )
        matches = sorted(run_root.glob("scenario_*"))
        if not matches:
            raise FileNotFoundError(
                f"no Phase 6G training run found under {run_root}"
            )
        for candidate in reversed(matches):
            if (candidate / "artifacts" / "preds_test.npz").exists():
                return candidate
        return matches[-1]

    def test_tiny_training_export_smoke(self) -> None:
        if os.environ.get(ENV_RUN_TINY_TRAIN, "").strip() not in {"1", "true", "TRUE", "yes", "YES"}:
            self.skipTest(
                f"set {ENV_RUN_TINY_TRAIN}=1 to run the tiny KalmanNet training bridge smoke"
            )

        cache_root = self.root / "cache"
        cache_root.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env["BENCH_DATA_CACHE"] = str(cache_root)
        env.setdefault("PYTHONHASHSEED", "0")

        cmd = [
            sys.executable,
            "-m",
            "bench.runners.run_suite",
            "--suite-yaml",
            str(self.suite_yaml),
            "--tasks",
            "phase6f_adcs_bias_tiny_v0",
            "--models",
            "kalmannet_tsp",
            "--seeds",
            "0",
            "--plans",
            "trained:frozen",
            "--device",
            "cpu",
        ]
        proc = subprocess.run(
            cmd,
            cwd=str(self.bench_root),
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                "tiny KalmanNet training run failed:\n"
                f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
            )

        run_dir = self._find_training_run_dir()
        self.assertTrue(run_dir.exists())
        self.assertTrue((run_dir / "config_snapshot.yaml").exists())
        self.assertTrue((run_dir / "artifacts" / "preds_test.npz").exists())

        package_dir = self.root / "package"
        exported = export_phase6g_kalmannet_tsp_package(
            source_run_dir=run_dir,
            package_dir=package_dir,
            overwrite=True,
        )
        self.assertEqual(exported, package_dir.resolve())
        self.assertTrue((exported / "checkpoint.pt").exists())
        self.assertTrue((exported / "replay_contract.json").exists())
        self.assertTrue((exported / "model_config.json").exists())
        self.assertTrue((exported / "system_model.json").exists())
        self.assertTrue((exported / "training_summary.json").exists())

        contract = validate_replay_checkpoint_package(
            exported,
            expected_state_dim=9,
            expected_measurement_dim=6,
            expected_observed_state=[0, 1, 2, 3, 4, 5],
        )
        self.assertEqual(contract["model_id"], "kalmannet_tsp")
        self.assertTrue(contract["smoke_training"])
        probe = probe_checkpoint_contract(
            exported,
            model_id="kalmannet_tsp",
        )
        self.assertTrue(probe["replay_contract_present"])
        self.assertTrue(probe["replay_contract_valid"])
        self.assertTrue(probe["supported_for_phase6d"])


@dataclass
class Phase6GKalmanNetTinyTrainExportResult:
    ok: bool
    skipped: bool
    note: str


def run_phase6g_kalmannet_tiny_train_export_tests(
) -> Phase6GKalmanNetTinyTrainExportResult:
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(
        Phase6GKalmanNetTinyTrainExportTests
    )
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=1).run(suite)
    skipped = bool(result.skipped) and result.testsRun == len(result.skipped)
    return Phase6GKalmanNetTinyTrainExportResult(
        ok=bool(result.wasSuccessful()),
        skipped=bool(skipped),
        note=(
            "Phase 6G tiny KalmanNet training/export smoke passed"
            if result.wasSuccessful()
            else stream.getvalue().strip()
        ),
    )


if __name__ == "__main__":
    unittest.main()
