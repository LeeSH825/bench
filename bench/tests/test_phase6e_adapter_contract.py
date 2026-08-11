from __future__ import annotations

import io
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from bench.visualization.checkpoint_contract_probe import (
    probe_checkpoint_contract,
)
from bench.visualization.checkpoint_replay_adapters import (
    MOCK_CHECKPOINT_MODEL_ID,
    get_real_checkpoint_replay_model_ids,
    get_supported_checkpoint_replay_model_ids,
    get_test_checkpoint_replay_model_ids,
    run_checkpoint_replay_adapter,
)
from bench.visualization.phase6e_checkpoint_package import (
    build_replay_checkpoint_package,
)


class Phase6EAdapterContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.mock_checkpoint = self.root / "mock.pt"
        torch.save(
            {
                "model_id": MOCK_CHECKPOINT_MODEL_ID,
                "gain": 1.0,
                "bias": 0.0,
            },
            self.mock_checkpoint,
        )
        self.y_obs = np.arange(30, dtype=np.float32).reshape(1, 5, 6)
        self.replay_meta = {
            "schema_version": "phase6a_replay_input_v1",
            "state_dim": 9,
            "measurement_dim": 6,
            "observation": {
                "observed_state": [0, 1, 2, 3, 4, 5],
            },
        }

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _package(
        self,
        *,
        model_id: str = MOCK_CHECKPOINT_MODEL_ID,
        state_dim: int = 9,
        is_mock: bool = True,
    ) -> Path:
        source = self.mock_checkpoint
        if model_id != MOCK_CHECKPOINT_MODEL_ID:
            source = self.root / f"{model_id}.pt"
            torch.save({"state_dict": {}}, source)
        return build_replay_checkpoint_package(
            checkpoint=source,
            model_id=model_id,
            package_dir=self.root / f"package_{model_id}_{state_dim}",
            state_dim=state_dim,
            measurement_dim=6,
            observed_state=[0, 1, 2, 3, 4, 5],
            is_mock=is_mock,
            not_for_benchmark_reporting=is_mock,
        )

    def test_mock_adapter_with_valid_contract(self) -> None:
        package = self._package()
        result = run_checkpoint_replay_adapter(
            model_id=MOCK_CHECKPOINT_MODEL_ID,
            checkpoint=package / "checkpoint.pt",
            model_config=package / "replay_contract.json",
            y_obs=self.y_obs,
            replay_meta=self.replay_meta,
        )
        self.assertEqual(result.x_hat.shape, (1, 5, 9))
        self.assertIsNotNone(result.metadata["replay_contract_summary"])
        self.assertEqual(result.metadata["package_dir"], str(package))
        self.assertFalse(result.metadata["is_real_checkpoint_adapter"])

    def test_mock_adapter_incompatible_state_dim_raises(self) -> None:
        package = self._package(state_dim=8)
        with self.assertRaisesRegex(ValueError, "state_dim mismatch"):
            run_checkpoint_replay_adapter(
                model_id=MOCK_CHECKPOINT_MODEL_ID,
                checkpoint=package / "checkpoint.pt",
                model_config=package / "replay_contract.json",
                y_obs=self.y_obs,
                replay_meta=self.replay_meta,
            )

    def test_real_model_with_valid_contract_is_registered(self) -> None:
        package = self._package(
            model_id="kalmannet_tsp",
            is_mock=False,
        )
        probe = probe_checkpoint_contract(
            package,
            model_id="kalmannet_tsp",
        )
        self.assertTrue(probe["replay_contract_present"])
        self.assertTrue(probe["replay_contract_valid"])
        self.assertTrue(probe["supported_for_phase6d"])
        self.assertIn("explicit adapter is registered", probe["support_reason"])

    def test_probe_detects_package_without_claiming_real_support(self) -> None:
        mock_package = self._package()
        mock_probe = probe_checkpoint_contract(
            mock_package,
            model_id=MOCK_CHECKPOINT_MODEL_ID,
        )
        self.assertTrue(mock_probe["replay_contract_present"])
        self.assertTrue(mock_probe["replay_contract_valid"])
        self.assertTrue(mock_probe["package_supported_for_phase6d"])

        real_package = self._package(
            model_id="kalmannet_tsp",
            is_mock=False,
        )
        real_probe = probe_checkpoint_contract(
            real_package,
            model_id="kalmannet_tsp",
        )
        self.assertTrue(real_probe["replay_contract_present"])
        self.assertTrue(real_probe["supported_for_phase6d"])
        self.assertIn(
            "explicit adapter is registered",
            real_probe["support_reason"],
        )

    def test_registry_separates_real_and_test_ids(self) -> None:
        self.assertIn(
            "kalmannet_tsp",
            get_real_checkpoint_replay_model_ids(),
        )
        self.assertIn(
            "kalmannet_tsp",
            get_supported_checkpoint_replay_model_ids(),
        )
        self.assertIn(
            MOCK_CHECKPOINT_MODEL_ID,
            get_test_checkpoint_replay_model_ids(),
        )
        self.assertIn(
            MOCK_CHECKPOINT_MODEL_ID,
            get_supported_checkpoint_replay_model_ids(include_test=True),
        )


@dataclass
class Phase6EAdapterContractResult:
    ok: bool
    note: str


def run_phase6e_adapter_contract_tests() -> Phase6EAdapterContractResult:
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(
        Phase6EAdapterContractTests
    )
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=1).run(suite)
    return Phase6EAdapterContractResult(
        ok=bool(result.wasSuccessful()),
        note=(
            "Phase 6E adapter contract tests passed"
            if result.wasSuccessful()
            else stream.getvalue().strip()
        ),
    )


if __name__ == "__main__":
    unittest.main()
