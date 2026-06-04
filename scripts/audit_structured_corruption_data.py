#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bench.tasks.data_format import load_npz_split_v0




def _cache_root() -> Path:
    env = os.environ.get("BENCH_DATA_CACHE", "").strip()
    if env:
        return Path(env).expanduser().resolve()
    return ROOT / "bench_data_cache"


def _finite(arr: np.ndarray) -> bool:
    return bool(np.isfinite(arr).all())


def _row_for(path: Path, suite_name: str, task_id: str, scenario_id: str, seed: int, split: str) -> Dict[str, Any]:
    loaded = load_npz_split_v0(path)
    meta = loaded.meta
    corruption = meta.get("corruption", {}) if isinstance(meta.get("corruption"), dict) else {}
    stats = corruption.get("stats", {}) if isinstance(corruption.get("stats"), dict) else {}
    extras = loaded.extras
    return {
        "suite_name": suite_name,
        "task_id": task_id,
        "scenario_id": scenario_id,
        "seed": seed,
        "split": split,
        "profile_id": corruption.get("profile_id", ""),
        "corruption_enabled": bool(corruption.get("enabled", False)),
        "x_shape": list(loaded.x.shape),
        "y_shape": list(loaded.y.shape),
        "x_dtype": str(loaded.x.dtype),
        "y_dtype": str(loaded.y.dtype),
        "finite": _finite(loaded.x) and _finite(loaded.y),
        "y_dim": int(loaded.y.shape[2]) if loaded.y.ndim == 3 else -1,
        "y_clean_exists": "y_clean_seq" in extras,
        "corruption_total_exists": "corruption_total_seq" in extras,
        "outlier_mask_exists": "corruption_outlier_mask_seq" in extras,
        "total_corruption_norm_mean": float(stats.get("total_corruption_norm_mean", float("nan"))),
        "total_corruption_to_clean_ratio": float(stats.get("total_corruption_to_clean_ratio", float("nan"))),
        "outlier_rate_observed": float(stats.get("outlier_rate_observed", float("nan"))),
        "fake_marker_present": bool(meta.get("fake_basilisk_data", False) or meta.get("synthetic_fallback", False)),
        "meta_task_family": meta.get("task_family", ""),
        "path": str(path),
    }


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite-name", required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--seed", type=int, action="append", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    rows: List[Dict[str, Any]] = []
    base = _cache_root() / args.suite_name / args.task_id
    for scenario_dir in sorted(base.glob("scenario_*")):
        scenario_id = scenario_dir.name.replace("scenario_", "", 1)
        for seed in args.seed:
            seed_dir = scenario_dir / f"seed_{seed}"
            for split in ("train", "val", "test"):
                path = seed_dir / f"{split}.npz"
                if path.exists():
                    rows.append(_row_for(path, args.suite_name, args.task_id, scenario_id, seed, split))
                else:
                    rows.append(
                        {
                            "suite_name": args.suite_name,
                            "task_id": args.task_id,
                            "scenario_id": scenario_id,
                            "seed": seed,
                            "split": split,
                            "profile_id": "",
                            "corruption_enabled": False,
                            "x_shape": [],
                            "y_shape": [],
                            "x_dtype": "",
                            "y_dtype": "",
                            "finite": False,
                            "y_dim": -1,
                            "y_clean_exists": False,
                            "corruption_total_exists": False,
                            "outlier_mask_exists": False,
                            "total_corruption_norm_mean": float("nan"),
                            "total_corruption_to_clean_ratio": float("nan"),
                            "outlier_rate_observed": float("nan"),
                            "fake_marker_present": False,
                            "meta_task_family": "",
                            "path": str(path),
                        }
                    )

    _write_csv(Path(args.out), rows)
    finite_ok = all(bool(r["finite"]) for r in rows)
    fake_count = sum(1 for r in rows if bool(r["fake_marker_present"]))
    ratios = [
        float(r["total_corruption_to_clean_ratio"])
        for r in rows
        if str(r["split"]) == "train" and math.isfinite(float(r["total_corruption_to_clean_ratio"]))
    ]
    print(
        json.dumps(
            {
                "rows": len(rows),
                "finite_ok": finite_ok,
                "fake_marker_count": fake_count,
                "train_corruption_ratios": ratios,
                "out": str(Path(args.out).resolve()),
            },
            indent=2,
        )
    )
    return 0 if rows and finite_ok and fake_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
