from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .test_determinism_gpu_seed_stable import run_determinism_gpu_seed_stable
from .test_gpu_all_models_device_smoke import run_gpu_all_models_device_smoke


def _bench_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _default_suite_plan_matrix_smoke() -> Path:
    return _bench_root() / "bench" / "configs" / "suite_plan_matrix_smoke.yaml"


def _default_suite_gpu_models_smoke() -> Path:
    return _bench_root() / "bench" / "configs" / "suite_gpu_models_smoke.yaml"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite-yaml", type=str, default=None, help="default: bench/configs/suite_plan_matrix_smoke.yaml")
    ap.add_argument(
        "--all-models-suite-yaml",
        type=str,
        default=None,
        help="default: bench/configs/suite_gpu_models_smoke.yaml",
    )
    args = ap.parse_args()

    suite_yaml = Path(args.suite_yaml).expanduser().resolve() if args.suite_yaml else _default_suite_plan_matrix_smoke()
    all_models_suite_yaml = (
        Path(args.all_models_suite_yaml).expanduser().resolve()
        if args.all_models_suite_yaml
        else _default_suite_gpu_models_smoke()
    )

    overall_ok = True

    res = run_determinism_gpu_seed_stable(suite_yaml=suite_yaml)
    if res.skipped:
        print(f"[SKIP] test_determinism_gpu_seed_stable: {res.note}")
    elif not res.ok:
        print(f"[FAIL] test_determinism_gpu_seed_stable: {res.note} (run_dir={res.run_dir})")
        overall_ok = False
    else:
        print(f"[PASS] test_determinism_gpu_seed_stable: {res.note} (run_dir={res.run_dir})")

    res_all = run_gpu_all_models_device_smoke(suite_yaml=all_models_suite_yaml)
    if res_all.skipped:
        print(f"[SKIP] test_gpu_all_models_device_smoke: {res_all.note}")
    elif not res_all.ok:
        print(f"[FAIL] test_gpu_all_models_device_smoke: {res_all.note} (run_dir={res_all.run_dir})")
        overall_ok = False
    else:
        print(f"[PASS] test_gpu_all_models_device_smoke: {res_all.note} (run_dir={res_all.run_dir})")

    if not overall_ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
