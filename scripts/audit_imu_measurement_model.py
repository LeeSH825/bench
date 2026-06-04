from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bench.diagnostics.imu_measurement_model_audit import run_audit


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit Basilisk IMU h/H measurement-model semantics.")
    parser.add_argument("--cache-root", default="/home/dss-pc-05/bench/bench_data_cache")
    parser.add_argument("--suite-name", default="gpu_basilisk_imu_pilot_pretrained_enhancer")
    parser.add_argument("--task-id", default="Basilisk_IMU_ADCS_pilot_v0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--reports-dir", default=str(ROOT / "reports"))
    parser.add_argument("--plots-dir", default=str(ROOT / "plots"))
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    result = run_audit(
        root=ROOT,
        cache_root=Path(args.cache_root),
        suite_name=str(args.suite_name),
        task_id=str(args.task_id),
        seed=int(args.seed),
        reports_dir=Path(args.reports_dir),
        plots_dir=Path(args.plots_dir),
        write_plots=not bool(args.no_plots),
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
