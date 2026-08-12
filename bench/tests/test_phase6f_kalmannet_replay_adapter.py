from __future__ import annotations

import io
import json
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from bench.visualization.checkpoint_replay_adapters import (
    MOCK_CHECKPOINT_MODEL_ID,
    get_real_checkpoint_replay_model_ids,
    get_test_checkpoint_replay_model_ids,
    run_checkpoint_replay_adapter,
)
from bench.visualization.phase6e_checkpoint_package import (
    build_replay_checkpoint_package,
)
from bench.visualization.phase6f_kalmannet_export import (
    KALMANNET_MODEL_CONFIG_SCHEMA_VERSION,
    KALMANNET_SYSTEM_MODEL_SCHEMA_VERSION,
    export_kalmannet_tsp_replay_package,
)


class Phase6FKalmanNetReplayAdapterTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
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

    def _kalmannet_package(self) -> Path:
        checkpoint = self.root / "kalmannet.pt"
        torch.save(
            {"state_dict": {}, "model_class": "KalmanNetNN"},
            checkpoint,
        )
        model_config = self.root / "model_config.json"
        model_config.write_text(
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
                    "normalization": {"enabled": False},
                    "hidden_state_initialization": {"method": "zeros"},
                }
            ),
            encoding="utf-8",
        )
        system_model = self.root / "system_model.json"
        system_model.write_text(
            json.dumps(
                {
                    "schema_version": (
                        KALMANNET_SYSTEM_MODEL_SCHEMA_VERSION
                    ),
                    "format": "linear_F_H",
                    "state_dim": 9,
                    "measurement_dim": 6,
                    "F": np.eye(9).tolist(),
                    "H": np.eye(9)[:6].tolist(),
                    "Q": np.eye(9).tolist(),
                    "R": np.eye(6).tolist(),
                }
            ),
            encoding="utf-8",
        )
        return export_kalmannet_tsp_replay_package(
            checkpoint=checkpoint,
            package_dir=self.root / "kalmannet_package",
            model_config=model_config,
            system_model=system_model,
        )

    def test_real_kalmannet_registered_but_incomplete_package_fails_clearly(
        self,
    ) -> None:
        package = self._kalmannet_package()
        self.assertIn("kalmannet_tsp", get_real_checkpoint_replay_model_ids())
        with self.assertRaisesRegex(ValueError, "repo"):
            run_checkpoint_replay_adapter(
                model_id="kalmannet_tsp",
                checkpoint=package / "checkpoint.pt",
                model_config=package / "replay_contract.json",
                y_obs=self.y_obs,
                replay_meta=self.replay_meta,
            )

    def test_real_and_test_registry_remain_separate(self) -> None:
        self.assertIn("kalmannet_tsp", get_real_checkpoint_replay_model_ids())
        self.assertIn(
            MOCK_CHECKPOINT_MODEL_ID,
            get_test_checkpoint_replay_model_ids(),
        )

    def test_mock_adapter_still_works(self) -> None:
        checkpoint = self.root / "mock.pt"
        torch.save(
            {
                "model_id": MOCK_CHECKPOINT_MODEL_ID,
                "gain": 1.0,
                "bias": 0.0,
            },
            checkpoint,
        )
        package = build_replay_checkpoint_package(
            checkpoint=checkpoint,
            model_id=MOCK_CHECKPOINT_MODEL_ID,
            package_dir=self.root / "mock_package",
            state_dim=9,
            measurement_dim=6,
            observed_state=[0, 1, 2, 3, 4, 5],
            is_mock=True,
            not_for_benchmark_reporting=True,
        )
        result = run_checkpoint_replay_adapter(
            model_id=MOCK_CHECKPOINT_MODEL_ID,
            checkpoint=package / "checkpoint.pt",
            model_config=package / "replay_contract.json",
            y_obs=self.y_obs,
            replay_meta=self.replay_meta,
        )
        self.assertEqual(result.x_hat.shape, (1, 5, 9))
        self.assertTrue(result.metadata["is_mock_adapter"])

    def test_incompatible_5x5_contract_is_rejected_for_9x6_replay(
        self,
    ) -> None:
        checkpoint = self.root / "legacy_5x5.pt"
        torch.save({"state_dict": {}}, checkpoint)
        package = build_replay_checkpoint_package(
            checkpoint=checkpoint,
            model_id="kalmannet_tsp",
            package_dir=self.root / "legacy_5x5_package",
            state_dim=5,
            measurement_dim=5,
            observed_state=[0, 1, 2, 3, 4],
        )
        with self.assertRaisesRegex(ValueError, "state_dim mismatch"):
            run_checkpoint_replay_adapter(
                model_id="kalmannet_tsp",
                checkpoint=package / "checkpoint.pt",
                model_config=package / "replay_contract.json",
                y_obs=self.y_obs,
                replay_meta=self.replay_meta,
            )


@dataclass
class Phase6FKalmanNetReplayAdapterResult:
    ok: bool
    note: str


def run_phase6f_kalmannet_replay_adapter_tests(
) -> Phase6FKalmanNetReplayAdapterResult:
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(
        Phase6FKalmanNetReplayAdapterTests
    )
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=1).run(suite)
    return Phase6FKalmanNetReplayAdapterResult(
        ok=bool(result.wasSuccessful()),
        note=(
            "Phase 6F KalmanNet replay adapter tests passed"
            if result.wasSuccessful()
            else stream.getvalue().strip()
        ),
    )


if __name__ == "__main__":
    unittest.main()
