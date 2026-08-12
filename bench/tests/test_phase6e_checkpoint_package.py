from __future__ import annotations

import io
import tempfile
import unittest
from contextlib import redirect_stdout
from dataclasses import dataclass
from pathlib import Path

import torch

from bench.visualization.phase6e_checkpoint_package import (
    build_replay_checkpoint_package,
    main,
    validate_replay_checkpoint_package,
)
from bench.visualization.replay_checkpoint_contract import (
    REPLAY_CHECKPOINT_CONTRACT_FILENAME,
)


class Phase6ECheckpointPackageTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.checkpoint = self.root / "source.pt"
        torch.save(
            {
                "model_id": "mock_checkpoint_adapter",
                "gain": 1.0,
                "bias": 0.0,
            },
            self.checkpoint,
        )

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _build(self, package: Path) -> Path:
        return build_replay_checkpoint_package(
            checkpoint=self.checkpoint,
            model_id="mock_checkpoint_adapter",
            package_dir=package,
            state_dim=9,
            measurement_dim=6,
            observed_state=[0, 1, 2, 3, 4, 5],
            is_mock=True,
            not_for_benchmark_reporting=True,
        )

    def test_build_and_validate_package(self) -> None:
        package = self._build(self.root / "package")
        self.assertTrue((package / "checkpoint.pt").exists())
        self.assertTrue(
            (package / REPLAY_CHECKPOINT_CONTRACT_FILENAME).exists()
        )
        validated = validate_replay_checkpoint_package(
            package,
            expected_state_dim=9,
            expected_measurement_dim=6,
            expected_observed_state=[0, 1, 2, 3, 4, 5],
        )
        self.assertEqual(validated["model_id"], "mock_checkpoint_adapter")
        self.assertTrue(validated["is_mock"])
        self.assertEqual(
            validated["_resolved_paths"]["checkpoint_path"],
            str((package / "checkpoint.pt").resolve()),
        )

    def test_existing_package_without_overwrite_raises(self) -> None:
        package = self._build(self.root / "existing")
        with self.assertRaises(FileExistsError):
            self._build(package)

    def test_cli_build_and_validate(self) -> None:
        package = self.root / "cli_package"
        with redirect_stdout(io.StringIO()):
            result = main(
                [
                    "--checkpoint",
                    str(self.checkpoint),
                    "--model-id",
                    "mock_checkpoint_adapter",
                    "--package-dir",
                    str(package),
                    "--state-dim",
                    "9",
                    "--measurement-dim",
                    "6",
                    "--observed-state",
                    "0,1,2,3,4,5",
                    "--is-mock",
                    "--not-for-benchmark-reporting",
                ]
            )
        self.assertEqual(result, 0)
        self.assertTrue((package / "checkpoint.pt").exists())

        with redirect_stdout(io.StringIO()):
            validate_result = main(
                [
                    "--validate-package",
                    str(package),
                    "--expected-state-dim",
                    "9",
                    "--expected-measurement-dim",
                    "6",
                    "--expected-observed-state",
                    "0,1,2,3,4,5",
                ]
            )
        self.assertEqual(validate_result, 0)

    def test_cli_overwrite(self) -> None:
        package = self._build(self.root / "overwrite")
        with redirect_stdout(io.StringIO()):
            result = main(
                [
                    "--checkpoint",
                    str(self.checkpoint),
                    "--model-id",
                    "mock_checkpoint_adapter",
                    "--package-dir",
                    str(package),
                    "--state-dim",
                    "9",
                    "--measurement-dim",
                    "6",
                    "--observed-state",
                    "0,1,2,3,4,5",
                    "--is-mock",
                    "--not-for-benchmark-reporting",
                    "--overwrite",
                ]
            )
        self.assertEqual(result, 0)


@dataclass
class Phase6ECheckpointPackageResult:
    ok: bool
    note: str


def run_phase6e_checkpoint_package_tests() -> Phase6ECheckpointPackageResult:
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(
        Phase6ECheckpointPackageTests
    )
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=1).run(suite)
    return Phase6ECheckpointPackageResult(
        ok=bool(result.wasSuccessful()),
        note=(
            "Phase 6E checkpoint package tests passed"
            if result.wasSuccessful()
            else stream.getvalue().strip()
        ),
    )


if __name__ == "__main__":
    unittest.main()
