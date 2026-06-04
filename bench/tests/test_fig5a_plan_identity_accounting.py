from __future__ import annotations

import copy
import json
import math
import os
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from bench.models import registry
from bench.models.base import ModelAdapter
from bench.reports.aggregate import RunRecord, aggregate_by_seed
from bench.reports.plots import _build_fig5a_series
from bench.runners.run_suite import _expand_sweep, load_suite_yaml, run_one
from bench.tasks.bench_generated import prepare_bench_generated_v0


def _suite_path() -> Path:
    return Path(__file__).resolve().parents[1] / "configs" / "suite_kf_baseline_smoke.yaml"


def _fig_record(
    *,
    model_id: str = "kalmannet_tsp",
    init_id: str = "trained",
    track_id: str = "frozen",
    mse: float = 1.0,
    mse_db: float = 0.0,
) -> RunRecord:
    return RunRecord(
        suite="fig5a_unit",
        task_id="F5a_unit",
        scenario_id=f"{model_id}_{init_id}_{track_id}",
        seed=0,
        model_id=model_id,
        init_id=init_id,
        track_id=track_id,
        status="ok",
        run_dir=Path("."),
        mse=mse,
        rmse=math.sqrt(max(mse, 0.0)),
        mse_db=mse_db,
        x_dim=2,
        y_dim=2,
        T=50,
        q2=1.0,
        r2=1.0,
        inv_r2_db=0.0,
    )


class _AliasLedgerTrainAdapter(ModelAdapter):
    def __init__(self) -> None:
        self.run_dir: Optional[Path] = None
        self.x_dim = 0
        self.T = 0
        self.fake_train_updates = 0
        self.train_updates_used = 0
        self.adapt_updates_used = 0
        self.last_layout = "test_BTD"
        self.last_class = "tests._AliasLedgerTrainAdapter"

    def setup(self, cfg: dict, system_info: Any, run_ctx: Optional[Dict[str, Any]] = None) -> None:
        run_ctx = run_ctx or {}
        self.run_dir = Path(str(run_ctx["run_dir"]))
        self.x_dim = int(system_info["x_dim"])
        self.T = int(system_info["T"])
        self.fake_train_updates = int(cfg.get("fake_train_updates", 1))

    def train(
        self,
        train_loader: Any,
        val_loader: Any,
        budget: Optional[Any] = None,
        ckpt_dir: Optional[Any] = None,
    ) -> Any:
        if self.run_dir is None:
            raise RuntimeError("setup must run first")
        budget = dict(budget or {})
        updates = int(self.fake_train_updates)
        self.train_updates_used = updates
        ledger_path = self.run_dir / "budget_ledger.json"
        ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
        ledger.update(
            {
                "train_updates_used": updates,
                "train_outer_updates_used": 0,
                "train_inner_updates_used": 0,
                "adapt_updates_used": 0,
                "train_max_updates": int(budget.get("train_max_updates", 0)),
                "train_skipped": False,
            }
        )
        ledger_path.write_text(json.dumps(ledger, indent=2), encoding="utf-8")
        out = Path(str(ckpt_dir or (self.run_dir / "checkpoints")))
        out.mkdir(parents=True, exist_ok=True)
        ckpt_path = out / "model.pt"
        torch.save({"state_dict": {}}, ckpt_path)
        (out / "train_state.json").write_text(
            json.dumps({"status": "ok", "updates_used": updates}, indent=2),
            encoding="utf-8",
        )
        return {"status": "ok", "ckpt_path": str(ckpt_path), "updates_used": updates}

    def eval(self, test_loader: Any, ckpt_path: Optional[str] = None, track_cfg: Optional[dict] = None) -> Any:
        preds = []
        for batch in test_loader:
            preds.append(torch.as_tensor(batch["x"], dtype=torch.float32))
        return {"status": "ok", "x_hat": torch.cat(preds, dim=0)}

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
        return torch.zeros((int(y.shape[0]), int(y.shape[1]), self.x_dim), dtype=torch.float32)

    def adapt(
        self,
        y_seq: Any,
        u_seq: Optional[Any] = None,
        context: Optional[dict] = None,
        budget: Optional[Any] = None,
    ) -> None:
        self.adapt_updates_used = 0

    def save(self, out_dir: str) -> None:
        return None


class Fig5aPlanIdentityAndAccountingTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_fig5a_series_splits_plan_identity(self) -> None:
        records = [
            _fig_record(init_id="trained", mse=1.0, mse_db=0.0),
            _fig_record(init_id="untrained", mse=1.0e35, mse_db=350.0),
        ]
        series = _build_fig5a_series(records=records)
        labels = {str(s["label"]) for s in series}

        self.assertEqual(len(series), 2)
        self.assertIn("kalmannet_tsp | trained:frozen | 2x2, T=50", labels)
        self.assertIn("kalmannet_tsp | untrained:frozen | 2x2, T=50", labels)
        high_db_labels = [str(s["label"]) for s in series if max(s["ys"]) > 100.0]
        self.assertEqual(high_db_labels, ["kalmannet_tsp | untrained:frozen | 2x2, T=50"])

    def test_fig5a_official_filter_keeps_only_official_plans(self) -> None:
        records = [
            _fig_record(model_id="mb_kf_oracle", init_id="pretrained", track_id="frozen"),
            _fig_record(model_id="kalmannet_tsp", init_id="trained", track_id="frozen"),
            _fig_record(model_id="split_knet", init_id="trained", track_id="frozen"),
            _fig_record(model_id="adaptive_knet", init_id="trained", track_id="frozen"),
            _fig_record(model_id="maml_knet", init_id="trained", track_id="frozen"),
            _fig_record(model_id="kalmannet_tsp", init_id="untrained", track_id="frozen"),
        ]
        series = _build_fig5a_series(records=records, official_plans_only=True)
        labels = {str(s["label"]) for s in series}

        self.assertEqual(len(series), 3)
        self.assertIn("mb_kf_oracle | pretrained:frozen | 2x2, T=50", labels)
        self.assertIn("kalmannet_tsp | trained:frozen | 2x2, T=50", labels)
        self.assertIn("split_knet | trained:frozen | 2x2, T=50", labels)
        self.assertNotIn("adaptive_knet | trained:frozen | 2x2, T=50", labels)
        self.assertNotIn("maml_knet | trained:frozen | 2x2, T=50", labels)
        self.assertNotIn("kalmannet_tsp | untrained:frozen | 2x2, T=50", labels)

    def test_metric_db_invariant_for_one_seed_aggregate(self) -> None:
        mse = 2.0e35
        mse_db = 10.0 * math.log10(mse)
        rec = _fig_record(mse=mse, mse_db=mse_db)
        row = aggregate_by_seed([rec])[0]

        self.assertAlmostEqual(float(rec.mse_db), 10.0 * math.log10(float(rec.mse)), places=12)
        self.assertAlmostEqual(float(row["mse_db_mean"]), 10.0 * math.log10(float(row["mse_mean"])), places=12)

    def _prepare_suite(self, *, fake_train_updates: int, train_max_updates: int) -> tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        suite = copy.deepcopy(load_suite_yaml(_suite_path()))
        suite.setdefault("reporting", {})
        suite["reporting"]["output_dir_template"] = str(
            self.tmp_path
            / "runs"
            / "{task_id}"
            / "{model_id}"
            / "{track_id}"
            / "seed_{seed}"
            / "scenario_{scenario_id}"
            / "init_{init_id}"
        )
        suite.setdefault("runner", {}).setdefault("budget", {})
        suite["runner"]["budget"]["train_max_updates"] = int(train_max_updates)
        suite["runner"]["budget"]["train_batch_size"] = 4
        suite["runner"]["budget"]["eval_batch_size"] = 4
        suite["runner"]["tracks"] = [{"track_id": "frozen", "adaptation_enabled": False}]

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
        model = {"model_id": "alias_ledger_dummy", "fake_train_updates": int(fake_train_updates)}
        scenario_settings = (_expand_sweep(task.get("sweep")) or [{}])[0]
        return suite, task, model, scenario_settings

    def test_trained_alias_update_accounting_commits_metrics(self) -> None:
        suite, task, model, scenario_settings = self._prepare_suite(fake_train_updates=2, train_max_updates=2)
        original = registry._REGISTRY.get("alias_ledger_dummy")
        registry._REGISTRY["alias_ledger_dummy"] = _AliasLedgerTrainAdapter
        try:
            res = run_one(
                suite=suite,
                task=task,
                model=model,
                scenario_settings=scenario_settings,
                seed=0,
                track_id="frozen",
                device_str="cpu",
                precision="fp32",
                init_id="trained",
                log_level="INFO",
                log_to_file=False,
                debug_every=0,
            )
        finally:
            if original is None:
                registry._REGISTRY.pop("alias_ledger_dummy", None)
            else:
                registry._REGISTRY["alias_ledger_dummy"] = original

        self.assertEqual(res["status"], "ok")
        run_dir = Path(str(res["run_dir"]))
        self.assertTrue((run_dir / "metrics.json").exists())
        ledger = json.loads((run_dir / "budget_ledger.json").read_text(encoding="utf-8"))
        self.assertEqual(int(ledger["train_updates_used"]), 2)
        self.assertEqual(int(ledger["train_outer_updates_used"]), 2)
        self.assertTrue(bool(ledger.get("train_update_accounting_normalized_from_alias")))

    def test_trained_alias_update_accounting_still_enforces_budget_cap(self) -> None:
        suite, task, model, scenario_settings = self._prepare_suite(fake_train_updates=3, train_max_updates=2)
        original = registry._REGISTRY.get("alias_ledger_dummy")
        registry._REGISTRY["alias_ledger_dummy"] = _AliasLedgerTrainAdapter
        try:
            res = run_one(
                suite=suite,
                task=task,
                model=model,
                scenario_settings=scenario_settings,
                seed=0,
                track_id="frozen",
                device_str="cpu",
                precision="fp32",
                init_id="trained",
                log_level="INFO",
                log_to_file=False,
                debug_every=0,
            )
        finally:
            if original is None:
                registry._REGISTRY.pop("alias_ledger_dummy", None)
            else:
                registry._REGISTRY["alias_ledger_dummy"] = original

        self.assertEqual(res["status"], "failed")
        self.assertEqual(res["failure_type"], "budget_overflow")
        run_dir = Path(str(res["run_dir"]))
        self.assertFalse((run_dir / "metrics.json").exists())


if __name__ == "__main__":
    unittest.main()
