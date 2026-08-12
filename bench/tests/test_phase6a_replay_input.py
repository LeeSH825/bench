from __future__ import annotations

import io
import json
import shutil
import tempfile
import unittest
from contextlib import redirect_stdout
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from bench.visualization.phase6a_replay_input import (
    SUITE_SNAPSHOT_FILENAME,
    TASK_SNAPSHOT_FILENAME,
    build_phase6a_replay_input,
    main,
)
from bench.visualization.replay_suite_scenario import (
    REPLAY_SCENARIO_FILENAME,
    REPLAY_SCENARIO_META_FILENAME,
)


def _short_suite_path() -> Path:
    return (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "suite_phase6a_replay_short.yaml"
    )


class Phase6AReplayInputTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.suite_path = self.root / "suite.yaml"
        shutil.copy2(_short_suite_path(), self.suite_path)
        self.task_id = "phase6a_adcs_bias_short_v0"

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _assert_outputs(self, output_dir: Path) -> None:
        npz_path = output_dir / REPLAY_SCENARIO_FILENAME
        meta_path = output_dir / REPLAY_SCENARIO_META_FILENAME
        self.assertTrue(npz_path.exists())
        self.assertTrue(meta_path.exists())
        self.assertTrue((output_dir / SUITE_SNAPSHOT_FILENAME).exists())
        self.assertTrue((output_dir / TASK_SNAPSHOT_FILENAME).exists())
        with np.load(npz_path, allow_pickle=False) as data:
            self.assertEqual(data["x_true"].shape, (1, 25, 9))
            self.assertEqual(data["y_obs"].shape, (1, 25, 6))
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        self.assertEqual(meta["task_id"], self.task_id)
        self.assertEqual(meta["source_suite_yaml"], str(self.suite_path.resolve()))

    def test_build_replay_input_and_snapshots(self) -> None:
        output_dir = self.root / "output"
        npz_path, meta_path = build_phase6a_replay_input(
            self.suite_path,
            task_id=self.task_id,
            seed=0,
            out_dir=output_dir,
        )
        self.assertEqual(npz_path, output_dir / REPLAY_SCENARIO_FILENAME)
        self.assertEqual(meta_path, output_dir / REPLAY_SCENARIO_META_FILENAME)
        self._assert_outputs(output_dir)

    def test_cli_smoke(self) -> None:
        output_dir = self.root / "cli_output"
        with redirect_stdout(io.StringIO()):
            result = main(
                [
                    "--suite-yaml",
                    str(self.suite_path),
                    "--task-id",
                    self.task_id,
                    "--seed",
                    "0",
                    "--out-dir",
                    str(output_dir),
                ]
            )
        self.assertEqual(result, 0)
        self._assert_outputs(output_dir)

    def test_missing_suite_and_unknown_task_raise(self) -> None:
        with self.assertRaises(FileNotFoundError):
            build_phase6a_replay_input(
                self.root / "missing.yaml",
                task_id=self.task_id,
                seed=0,
                out_dir=self.root / "missing_output",
            )
        with self.assertRaisesRegex(ValueError, "not found"):
            build_phase6a_replay_input(
                self.suite_path,
                task_id="unknown_task",
                seed=0,
                out_dir=self.root / "unknown_output",
            )


@dataclass
class Phase6AReplayInputResult:
    ok: bool
    note: str


def run_phase6a_replay_input_tests() -> Phase6AReplayInputResult:
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(
        Phase6AReplayInputTests
    )
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=1).run(suite)
    return Phase6AReplayInputResult(
        ok=bool(result.wasSuccessful()),
        note=(
            "Phase 6A replay input tests passed"
            if result.wasSuccessful()
            else stream.getvalue().strip()
        ),
    )


if __name__ == "__main__":
    unittest.main()
