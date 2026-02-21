from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List

from .test_seed_repro import run_seed_repro
from .test_data_format import run_data_format_check
from .test_adapter_smoke import run_adapter_smoke
from .test_runner_smoke import run_runner_smoke
from .test_report_smoke import run_report_smoke


def _bench_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _default_suite_shift() -> Path:
    return _bench_root() / "bench" / "configs" / "suite_shift.yaml"


def _default_cache_root() -> Path:
    env = os.environ.get("BENCH_DATA_CACHE")
    if env:
        return Path(env).expanduser().resolve()
    return (_bench_root() / "bench_data_cache").resolve()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=str, default="cpu", help="cpu (default) or cuda")
    ap.add_argument("--keep-going", action="store_true", help="continue even if a step fails")
    ap.add_argument("--suite-yaml", type=str, default=None, help="default: bench/configs/suite_shift.yaml")
    ap.add_argument("--task-id", type=str, default="C_shift_Rscale_v0", help="MVP task")
    ap.add_argument("--model-id", type=str, default="kalmannet_tsp", help="MVP model")
    ap.add_argument("--seed", type=int, default=0, help="default seed")
    args = ap.parse_args()

    device = args.device
    suite_yaml = Path(args.suite_yaml).expanduser().resolve() if args.suite_yaml else _default_suite_shift()
    task_id = args.task_id
    model_id = args.model_id
    seed = int(args.seed)

    bench_root = _bench_root()
    cache_root = _default_cache_root()
    runs_root = bench_root / "runs"
    reports_dir = bench_root / "reports"

    def step(name: str, ok: bool, note: str) -> None:
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {name}: {note}")
        if (not ok) and (not args.keep_going):
            sys.exit(1)

    print("=== Bench Manual Test Suite (Step 8) ===")
    print("bench_root:", bench_root)
    print("suite_yaml:", suite_yaml)
    print("task_id:", task_id, "model_id:", model_id, "seed:", seed, "device:", device)
    print("cache_root:", cache_root)
    print("runs_root:", runs_root)
    print("reports_dir:", reports_dir)
    print("=======================================")

    # 1) seed reproducibility: smoke_data twice
    r1 = run_seed_repro(suite_yaml=suite_yaml, task_id=task_id, seed=seed, device=device, cache_root=cache_root)
    step("test_seed_repro", r1.ok, f"{r1.note} (npz={r1.npz_path})")

    # 2) data format: npz keys + NTD shapes
    # suite_name inferred from YAML path: shift/basic. default shift.
    suite_name = "shift"
    df = run_data_format_check(cache_root=cache_root, suite_name=suite_name, task_id=task_id, seed=seed)
    step("test_data_format", df.ok, df.note)

    # 3) adapter smoke: forward one batch (prefer cpu by default)
    ad = run_adapter_smoke(suite_yaml=suite_yaml, task_id=task_id, model_id=model_id, seed=seed, device=device)
    step("test_adapter_smoke", ad.ok, ad.note)

    # 4) runner smoke: run_dir + artifacts
    rr = run_runner_smoke(suite_yaml=suite_yaml, task_id=task_id, model_id=model_id, seed=seed, device=device, track="frozen")
    step("test_runner_smoke", rr.ok, f"{rr.note} (run_dir={rr.run_dir})")

    # 5) report smoke: make_report produces csv + plot
    rep = run_report_smoke(suite_yaml=suite_yaml, suite_name=suite_name, runs_root=runs_root, out_dir=reports_dir)
    step("test_report_smoke", rep.ok, rep.note)

    print("=== ALL TESTS PASSED ===")


if __name__ == "__main__":
    main()
