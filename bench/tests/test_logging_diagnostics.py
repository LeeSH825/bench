from __future__ import annotations

import copy
import json
import logging
import os
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from bench.models import registry
from bench.models.base import ModelAdapter
from bench.runners.run_suite import _expand_sweep, load_suite_yaml, run_one
from bench.tasks.bench_generated import prepare_bench_generated_v0
from bench.utils.logging import clear_logging_context, configure_logging


def _reset_bench_logger() -> None:
    logger = logging.getLogger("bench")
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()
    clear_logging_context()


def _suite_path() -> Path:
    return Path(__file__).resolve().parents[1] / "configs" / "suite_kf_baseline_smoke.yaml"


class _WrongShapeAdapter(ModelAdapter):
    def __init__(self) -> None:
        self._x_dim = 0
        self._T = 0
        self.last_layout = "bench_bad_eval_BDT"
        self.last_class = "tests._WrongShapeAdapter"

    def setup(self, cfg: dict, system_info: Any, run_ctx: Optional[Dict[str, Any]] = None) -> None:
        self._x_dim = int(system_info["x_dim"])
        self._T = int(system_info["T"])

    def train(self, train_loader: Any, val_loader: Any, budget: Optional[Any] = None, ckpt_dir: Optional[Any] = None) -> Any:
        return {"status": "ok", "ckpt_path": None}

    def eval(self, test_loader: Any, ckpt_path: Optional[str] = None, track_cfg: Optional[dict] = None) -> Any:
        batches = []
        for batch in test_loader:
            y = batch["y"]
            bsz = int(y.shape[0])
            batches.append(torch.zeros((bsz, self._x_dim, self._T), dtype=torch.float32))
        return {"status": "ok", "x_hat": torch.cat(batches, dim=0)}

    def load(self, ckpt_path: str) -> None:
        return None

    def predict(
        self,
        y_seq: Any,
        u_seq: Optional[Any] = None,
        context: Optional[dict] = None,
        return_cov: bool = False,
    ) -> Any:
        y = torch.as_tensor(y_seq)
        return torch.zeros((int(y.shape[0]), self._x_dim, int(y.shape[1])), dtype=torch.float32)

    def adapt(
        self,
        y_seq: Any,
        u_seq: Optional[Any] = None,
        context: Optional[dict] = None,
        budget: Optional[Any] = None,
    ) -> None:
        return None

    def save(self, out_dir: str) -> None:
        return None


class LoggingDiagnosticsTests(unittest.TestCase):
    def setUp(self) -> None:
        _reset_bench_logger()
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self._tmp.name)

    def tearDown(self) -> None:
        _reset_bench_logger()
        self._tmp.cleanup()

    def _prepare_suite(self) -> Dict[str, Any]:
        suite = load_suite_yaml(_suite_path())
        suite = copy.deepcopy(suite)
        suite.setdefault("reporting", {})
        suite["reporting"]["output_dir_template"] = str(
            self.tmp_path / "runs" / "{task_id}" / "{model_id}" / "{track_id}" / "seed_{seed}" / "scenario_{scenario_id}"
        )
        cache_root = self.tmp_path / "cache"
        os.environ["BENCH_DATA_CACHE"] = str(cache_root)
        task = next(t for t in suite["tasks"] if t["task_id"] == "A_linear_kf_baseline_smoke_v0")
        prepare_bench_generated_v0(
            suite_name=str(suite["suite"]["name"]),
            task_cfg=task,
            seed=0,
            cache_root=cache_root,
            scenario_overrides={},
        )
        return suite

    def test_configure_logging_idempotent(self) -> None:
        configure_logging("DEBUG", run_dir=None, log_to_file=False)
        bench_logger = logging.getLogger("bench")
        self.assertEqual(len(bench_logger.handlers), 1)

        configure_logging("INFO", run_dir=None, log_to_file=False)
        self.assertEqual(len(bench_logger.handlers), 1)

        configure_logging("INFO", run_dir=self.tmp_path, log_to_file=True)
        self.assertEqual(len(bench_logger.handlers), 2)

        configure_logging("INFO", run_dir=self.tmp_path, log_to_file=True)
        self.assertEqual(len(bench_logger.handlers), 2)

        file_handlers = [h for h in bench_logger.handlers if isinstance(h, logging.FileHandler)]
        self.assertEqual(len(file_handlers), 1)
        self.assertEqual(Path(file_handlers[0].baseFilename), self.tmp_path / "logs" / "bench.log")

    def test_run_one_debug_writes_log_file_and_diagnostics(self) -> None:
        suite = self._prepare_suite()
        task = next(t for t in suite["tasks"] if t["task_id"] == "A_linear_kf_baseline_smoke_v0")
        model = next(m for m in suite["models"] if m["model_id"] == "oracle_kf")
        scenario_settings = (_expand_sweep(task.get("sweep")) or [{}])[0]

        res = run_one(
            suite=suite,
            task=task,
            model=model,
            scenario_settings=scenario_settings,
            seed=0,
            track_id="frozen",
            device_str="cpu",
            precision="fp32",
            init_id="pretrained",
            log_level="DEBUG",
            log_to_file=True,
            debug_every=1,
        )

        self.assertEqual(res["status"], "ok")
        run_dir = Path(str(res["run_dir"]))
        log_path = run_dir / "logs" / "bench.log"
        diag_stats = run_dir / "diagnostics" / "stats.json"
        diag_npz = run_dir / "diagnostics" / "first_batch_dump.npz"

        self.assertTrue(log_path.exists())
        log_text = log_path.read_text(encoding="utf-8")
        self.assertIn("Starting run", log_text)
        self.assertIn("Loaded test split shapes", log_text)
        self.assertIn("setup mode=oracle", log_text)

        self.assertTrue(diag_stats.exists())
        self.assertTrue(diag_npz.exists())
        stats_obj = json.loads(diag_stats.read_text(encoding="utf-8"))
        self.assertEqual(stats_obj["reason"], "debug")

    def test_run_one_shape_mismatch_reports_clear_error(self) -> None:
        suite = self._prepare_suite()
        task = next(t for t in suite["tasks"] if t["task_id"] == "A_linear_kf_baseline_smoke_v0")
        scenario_settings = (_expand_sweep(task.get("sweep")) or [{}])[0]
        bad_model = {"model_id": "wrong_shape_dummy"}

        original = registry._REGISTRY.get("wrong_shape_dummy")
        registry._REGISTRY["wrong_shape_dummy"] = _WrongShapeAdapter
        try:
            res = run_one(
                suite=suite,
                task=task,
                model=bad_model,
                scenario_settings=scenario_settings,
                seed=0,
                track_id="frozen",
                device_str="cpu",
                precision="fp32",
                init_id="pretrained",
                log_level="INFO",
                log_to_file=False,
                debug_every=0,
            )
        finally:
            if original is None:
                registry._REGISTRY.pop("wrong_shape_dummy", None)
            else:
                registry._REGISTRY["wrong_shape_dummy"] = original

        self.assertEqual(res["status"], "failed")
        self.assertEqual(res["failure_type"], "shape_mismatch")
        self.assertIn("expected x_hat [B,T,D]", str(res["error"]))
        self.assertIn("Fix the adapter permutation", str(res["error"]))


if __name__ == "__main__":
    unittest.main()
