from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from dataclasses import dataclass
from pathlib import Path

import torch

from bench.visualization.checkpoint_contract_probe import (
    CHECKPOINT_CONTRACT_PROBE_FILENAME,
    main,
    probe_checkpoint_contract,
    save_checkpoint_contract_probe,
)


class CheckpointContractProbeTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _write_checkpoint(self, name: str = "checkpoint.pt") -> Path:
        path = self.root / name
        torch.save(
            {
                "model_id": "mock_checkpoint_adapter",
                "gain": 1.0,
                "bias": 0.0,
                "state_dict": {"weight": torch.ones(2, 2)},
                "epoch": 3,
            },
            path,
        )
        return path

    def test_missing_checkpoint_raises(self) -> None:
        with self.assertRaises(FileNotFoundError):
            probe_checkpoint_contract(self.root / "missing.pt")

    def test_malformed_checkpoint_raises(self) -> None:
        path = self.root / "malformed.pt"
        path.write_bytes(b"not a torch checkpoint")
        with self.assertRaisesRegex(ValueError, "failed to load checkpoint"):
            probe_checkpoint_contract(path)

    def test_simple_checkpoint_dict_is_probed(self) -> None:
        path = self._write_checkpoint()
        probe = probe_checkpoint_contract(
            path,
            model_id="mock_checkpoint_adapter",
        )
        self.assertEqual(
            probe["schema_version"],
            "checkpoint_contract_probe_v1",
        )
        self.assertEqual(probe["top_level_type"], "dict")
        self.assertIn("state_dict", probe["top_level_keys"])
        self.assertIn("state_dict", probe["state_dict_key_candidates"])
        self.assertGreater(probe["checkpoint_size_bytes"], 0)
        self.assertTrue(probe["supported_for_phase6d"])
        self.assertEqual(probe["inferred_model_family"], "phase6d_mock")

    def test_probe_json_write(self) -> None:
        probe = probe_checkpoint_contract(
            self._write_checkpoint(),
            model_id="mock_checkpoint_adapter",
        )
        path = save_checkpoint_contract_probe(probe, self.root / "probe")
        self.assertTrue(path.exists())
        loaded = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(loaded["top_level_keys"], probe["top_level_keys"])

    def test_cli_smoke(self) -> None:
        output_dir = self.root / "cli"
        with redirect_stdout(io.StringIO()):
            result = main(
                [
                    "--checkpoint",
                    str(self._write_checkpoint()),
                    "--model-id",
                    "mock_checkpoint_adapter",
                    "--out-dir",
                    str(output_dir),
                ]
            )
        self.assertEqual(result, 0)
        self.assertTrue(
            (output_dir / CHECKPOINT_CONTRACT_PROBE_FILENAME).exists()
        )


@dataclass
class CheckpointContractProbeResult:
    ok: bool
    note: str


def run_checkpoint_contract_probe_tests() -> CheckpointContractProbeResult:
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(
        CheckpointContractProbeTests
    )
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=1).run(suite)
    return CheckpointContractProbeResult(
        ok=bool(result.wasSuccessful()),
        note=(
            "checkpoint contract probe tests passed"
            if result.wasSuccessful()
            else stream.getvalue().strip()
        ),
    )


if __name__ == "__main__":
    unittest.main()
