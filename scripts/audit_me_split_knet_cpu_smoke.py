#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List


ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = ROOT / "runs" / "basilisk_me_split_smoke"
OUT = ROOT / "reports" / "me_split_knet_cpu_smoke_audit.csv"


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return obj if isinstance(obj, dict) else {}


def _stats_value(stats: Dict[str, Any], key: str, stat: str = "mean") -> float:
    obj = stats.get("adapter_runtime_stats", {}).get(f"adapter_runtime.{key}")
    if isinstance(obj, dict) and stat in obj:
        try:
            return float(obj[stat])
        except Exception:
            return float("nan")
    return float("nan")


def _metric_float(metrics: Dict[str, Any], key: str) -> float:
    for obj in (metrics, metrics.get("accuracy", {})):
        if isinstance(obj, dict) and key in obj:
            try:
                return float(obj[key])
            except Exception:
                return float("nan")
    return float("nan")


def main() -> int:
    rows: List[Dict[str, Any]] = []
    for metrics_path in sorted(RUN_ROOT.glob("**/metrics.json")):
        run_dir = metrics_path.parent
        metrics = _read_json(metrics_path)
        ledger = _read_json(run_dir / "budget_ledger.json")
        stats = _read_json(run_dir / "diagnostics" / "stats.json")
        mse = _metric_float(metrics, "mse")
        mse_db = _metric_float(metrics, "mse_db")
        expected_db = 10.0 * math.log10(max(mse, 1.0e-300)) if math.isfinite(mse) and mse > 0 else float("nan")
        rows.append(
            {
                "run_dir": str(run_dir),
                "model_id": str(metrics.get("model_id", "")),
                "plan": f"{metrics.get('init_id', 'trained')}:{metrics.get('track_id', 'frozen')}",
                "mse": mse,
                "mse_db": mse_db,
                "expected_mse_db": expected_db,
                "db_abs_err": abs(mse_db - expected_db) if math.isfinite(mse_db) and math.isfinite(expected_db) else float("nan"),
                "failure_json": (run_dir / "failure.json").exists(),
                "adapt_updates_used": int(ledger.get("adapt_updates_used", -1)),
                "train_updates_used": int(ledger.get("train_updates_used", 0) or 0),
                "enhancer_updates_used": int(ledger.get("enhancer_updates_used", 0) or 0),
                "split_updates_used": int(ledger.get("split_updates_used", 0) or 0),
                "diagnostics_present": (run_dir / "diagnostics" / "stats.json").exists(),
                "residual_norm": (stats.get("residual_stats") or {}).get("norm"),
                "delta_norm_mean": _stats_value(stats, "delta_norm_mean", "mean"),
                "delta_norm_max": _stats_value(stats, "delta_norm_max", "max"),
            }
        )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as f:
        if rows:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps({"rows": len(rows), "audit_csv": str(OUT)}, indent=2))
    return 0 if rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
