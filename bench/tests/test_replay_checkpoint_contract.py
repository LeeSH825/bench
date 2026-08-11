from __future__ import annotations

import io
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path

from bench.visualization.replay_checkpoint_contract import (
    REPLAY_CHECKPOINT_CONTRACT_SCHEMA_VERSION,
    load_replay_checkpoint_contract,
    save_replay_checkpoint_contract,
    summarize_replay_checkpoint_contract,
    validate_replay_checkpoint_contract,
)


def _valid_contract() -> dict:
    return {
        "schema_version": REPLAY_CHECKPOINT_CONTRACT_SCHEMA_VERSION,
        "package_id": "replay_ckpt_test",
        "created_at_utc": "2026-06-13T00:00:00Z",
        "model_id": "mock_checkpoint_adapter",
        "adapter_id": "mock_checkpoint_adapter",
        "checkpoint_path": "checkpoint.pt",
        "model_config_path": None,
        "normalizer_path": None,
        "system_model_path": None,
        "training_summary_path": None,
        "state_dim": 9,
        "measurement_dim": 6,
        "observed_state": [0, 1, 2, 3, 4, 5],
        "input_layout": "NTD",
        "output_layout": "NTD",
        "time_layout": "time_s_T",
        "training_suite_name": "test_suite",
        "training_task_id": "test_task",
        "training_seed": 0,
        "training_run_dir": None,
        "checkpoint_step": 1,
        "checkpoint_metric": "mse",
        "checkpoint_metric_value": 0.1,
        "requires_system_model": False,
        "system_model_format": "none",
        "requires_normalization": False,
        "normalization_format": "none",
        "hidden_state_initialization": {
            "method": "zeros",
            "source": "replay_contract",
            "details": {},
        },
        "preprocessing": {
            "input_transform": "identity",
            "output_inverse_transform": "identity",
            "assumptions": [],
        },
        "compatibility": {
            "compatible_replay_schema_versions": [
                "phase6a_replay_input_v1"
            ],
            "compatible_state_schema": {"state_dim": 9},
            "compatible_observation_schema": {
                "observed_state": [0, 1, 2, 3, 4, 5]
            },
        },
        "is_mock": True,
        "not_for_benchmark_reporting": True,
        "warnings": ["test only"],
        "notes": "test contract",
    }


class ReplayCheckpointContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        (self.root / "checkpoint.pt").write_bytes(b"checkpoint")

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_valid_contract_loads_and_validates(self) -> None:
        path = save_replay_checkpoint_contract(
            _valid_contract(),
            self.root,
        )
        self.assertTrue(path.exists())
        loaded = load_replay_checkpoint_contract(self.root)
        self.assertEqual(loaded["state_dim"], 9)
        self.assertEqual(loaded["measurement_dim"], 6)
        self.assertEqual(
            loaded["_resolved_paths"]["checkpoint_path"],
            str((self.root / "checkpoint.pt").resolve()),
        )

    def test_missing_contract_raises(self) -> None:
        with self.assertRaises(FileNotFoundError):
            load_replay_checkpoint_contract(self.root)

    def test_duplicate_observed_state_raises(self) -> None:
        contract = _valid_contract()
        contract["observed_state"] = [0, 1, 2, 3, 4, 4]
        with self.assertRaisesRegex(ValueError, "duplicate"):
            validate_replay_checkpoint_contract(
                contract,
                package_dir=self.root,
            )

    def test_expected_state_dim_mismatch_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "state_dim mismatch"):
            validate_replay_checkpoint_contract(
                _valid_contract(),
                package_dir=self.root,
                expected_state_dim=8,
            )

    def test_required_system_model_missing_raises(self) -> None:
        contract = _valid_contract()
        contract["requires_system_model"] = True
        contract["system_model_format"] = "linear_F_H"
        with self.assertRaisesRegex(ValueError, "system_model_path"):
            validate_replay_checkpoint_contract(
                contract,
                package_dir=self.root,
            )

    def test_mock_contract_must_be_nonreporting(self) -> None:
        contract = _valid_contract()
        contract["not_for_benchmark_reporting"] = False
        with self.assertRaisesRegex(
            ValueError,
            "not_for_benchmark_reporting",
        ):
            validate_replay_checkpoint_contract(
                contract,
                package_dir=self.root,
            )

    def test_summary_contains_contract_identity(self) -> None:
        validated = validate_replay_checkpoint_contract(
            _valid_contract(),
            package_dir=self.root,
        )
        summary = summarize_replay_checkpoint_contract(validated)
        self.assertEqual(
            summary["schema_version"],
            REPLAY_CHECKPOINT_CONTRACT_SCHEMA_VERSION,
        )
        self.assertEqual(summary["model_id"], "mock_checkpoint_adapter")
        self.assertEqual(summary["observed_state"], [0, 1, 2, 3, 4, 5])
        self.assertTrue(summary["is_mock"])


@dataclass
class ReplayCheckpointContractResult:
    ok: bool
    note: str


def run_replay_checkpoint_contract_tests() -> ReplayCheckpointContractResult:
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(
        ReplayCheckpointContractTests
    )
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=1).run(suite)
    return ReplayCheckpointContractResult(
        ok=bool(result.wasSuccessful()),
        note=(
            "replay checkpoint contract tests passed"
            if result.wasSuccessful()
            else stream.getvalue().strip()
        ),
    )


if __name__ == "__main__":
    unittest.main()
