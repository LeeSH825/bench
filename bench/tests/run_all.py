from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List

from .test_seed_repro import run_seed_repro
from .test_data_format import run_data_format_check
from .test_generator_contract_tg0 import run_generator_contract_tg0_smoke
from .test_noise_schedule_tg1 import run_noise_schedule_tg1
from .test_linear_mismatch_tg2 import run_linear_mismatch_tg2
from .test_ucm_tg3 import run_ucm_tg3
from .test_sine_poly_tg4 import run_sine_poly_tg4
from .test_lorenz_tg5 import run_lorenz_tg5
from .test_switching_tg6 import run_switching_tg6
from .test_datasets_tg7 import run_dataset_loaders_tg7
from .test_adapter_smoke import run_adapter_smoke
from .test_runner_smoke import run_runner_smoke
from .test_report_smoke import run_report_smoke
from .test_train_smoke_plan import run_train_smoke_route_b
from .test_adapt_smoke_plan import run_adapt_smoke_route_b
from .test_cache_invariance import run_cache_invariance
from .test_plan_matrix_minimal import run_plan_matrix_minimal
from .test_report_schema_guardrails import run_report_schema_guardrails
from .test_fig5a_report_smoke import run_fig5a_report_smoke
from .test_determinism_cpu_seed_stable import run_determinism_cpu_seed_stable
from .test_determinism_gpu_seed_stable import run_determinism_gpu_seed_stable
from .test_artifact_invariants_smoke import run_artifact_invariants_smoke
from .test_failure_type_compat import run_failure_type_compat
from .test_split_train_smoke_plan import run_split_train_smoke_route_b
from .test_maml_train_smoke_plan import run_maml_train_smoke_route_b
from .test_kf_baseline_smoke_plan import run_kf_baseline_smoke_plan


def _bench_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _default_suite_shift() -> Path:
    return _bench_root() / "bench" / "configs" / "suite_shift.yaml"


def _default_cache_root() -> Path:
    env = os.environ.get("BENCH_DATA_CACHE")
    if env:
        return Path(env).expanduser().resolve()
    return (_bench_root() / "bench_data_cache").resolve()


def _default_suite_train_smoke() -> Path:
    return _bench_root() / "bench" / "configs" / "suite_train_smoke.yaml"


def _default_suite_split_train_smoke() -> Path:
    return _bench_root() / "bench" / "configs" / "suite_split_train_smoke.yaml"


def _default_suite_maml_train_smoke() -> Path:
    return _bench_root() / "bench" / "configs" / "suite_maml_train_smoke.yaml"


def _default_suite_adapt_smoke() -> Path:
    return _bench_root() / "bench" / "configs" / "suite_adapt_smoke.yaml"


def _default_suite_adapt_smoke_overflow() -> Path:
    return _bench_root() / "bench" / "configs" / "suite_adapt_smoke_overflow.yaml"


def _default_suite_cache_smoke() -> Path:
    return _bench_root() / "bench" / "configs" / "suite_cache_smoke.yaml"


def _default_suite_plan_matrix_smoke() -> Path:
    return _bench_root() / "bench" / "configs" / "suite_plan_matrix_smoke.yaml"


def _default_suite_kf_baseline_smoke() -> Path:
    return _bench_root() / "bench" / "configs" / "suite_kf_baseline_smoke.yaml"


def _default_suite_fig5a_overlay_smoke() -> Path:
    return _bench_root() / "bench" / "configs" / "suite_fig5a_overlay_smoke.yaml"


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

    def step(name: str, ok: bool, note: str, skipped: bool = False) -> None:
        status = "SKIP" if skipped else ("PASS" if ok else "FAIL")
        print(f"[{status}] {name}: {note}")
        if (not ok) and (not skipped) and (not args.keep_going):
            sys.exit(1)

    print("=== Bench Manual Test Suite (Step 9) ===")
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

    # 2b) TG0 generator contract schema + determinism smoke
    tg0 = run_generator_contract_tg0_smoke()
    step("test_generator_contract_tg0", tg0.ok, f"{tg0.note} (npz={tg0.npz_path})")

    # 2c) TG1 shared noise_schedule utility smoke
    tg1 = run_noise_schedule_tg1()
    step("test_noise_schedule_tg1", tg1.ok, f"{tg1.note} (npz={tg1.npz_path})")

    # 2d) TG2 linear mismatch (rotated true-vs-assumed F/H) smoke
    tg2 = run_linear_mismatch_tg2()
    step("test_linear_mismatch_tg2", tg2.ok, f"{tg2.note} (npz={tg2.npz_path})")

    # 2e) TG3 UCM generator smoke (linear/nonlinear + optional task_set/task_key)
    tg3 = run_ucm_tg3()
    step("test_ucm_tg3", tg3.ok, f"{tg3.note} (npz={tg3.npz_path})")

    # 2f) TG4 synthetic nonlinear sine/poly generator smoke
    tg4 = run_sine_poly_tg4()
    step("test_sine_poly_tg4", tg4.ok, f"{tg4.note} (npz={tg4.npz_path})")

    # 2g) TG5 lorenz family generator smoke
    tg5 = run_lorenz_tg5()
    step("test_lorenz_tg5", tg5.ok, f"{tg5.note} (npz={tg5.npz_path})")

    # 2h) TG6 switching dynamics generator smoke
    tg6 = run_switching_tg6()
    step("test_switching_tg6", tg6.ok, f"{tg6.note} (npz={tg6.npz_path})")

    # 2i) TG7 dataset loader scaffolding diagnostics (CI-safe skip when datasets absent)
    tg7 = run_dataset_loaders_tg7()
    step("test_datasets_tg7", tg7.ok, tg7.note, skipped=bool(tg7.skipped))

    # 3) adapter smoke: forward one batch (prefer cpu by default)
    ad = run_adapter_smoke(suite_yaml=suite_yaml, task_id=task_id, model_id=model_id, seed=seed, device=device)
    step("test_adapter_smoke", ad.ok, ad.note)

    # 4) runner smoke: run_dir + artifacts
    rr = run_runner_smoke(suite_yaml=suite_yaml, task_id=task_id, model_id=model_id, seed=seed, device=device, track="frozen")
    step("test_runner_smoke", rr.ok, f"{rr.note} (run_dir={rr.run_dir})")

    # 5) report smoke: make_report produces csv + plot
    rep = run_report_smoke(suite_yaml=suite_yaml, suite_name=suite_name, runs_root=runs_root, out_dir=reports_dir)
    step("test_report_smoke", rep.ok, rep.note)

    # 6) Route-B tiny train smoke (trained,frozen + reproducibility)
    suite_train = _default_suite_train_smoke()
    tr = run_train_smoke_route_b(suite_yaml=suite_train)
    step("test_train_smoke_plan", tr.ok, f"{tr.note} (run_dir={tr.run_dir})")

    # 7) Route-B tiny split-knet train smoke (trained,frozen + reproducibility)
    suite_split_train = _default_suite_split_train_smoke()
    tr_split = run_split_train_smoke_route_b(suite_yaml=suite_split_train)
    step("test_split_train_smoke_plan", tr_split.ok, f"{tr_split.note} (run_dir={tr_split.run_dir})")

    # 8) Route-B tiny MAML-KNet train smoke (trained,frozen + reproducibility)
    suite_maml_train = _default_suite_maml_train_smoke()
    tr_maml = run_maml_train_smoke_route_b(suite_yaml=suite_maml_train)
    step("test_maml_train_smoke_plan", tr_maml.ok, f"{tr_maml.note} (run_dir={tr_maml.run_dir})")

    # 9) KF baseline smoke (pretrained,frozen)
    suite_kf = _default_suite_kf_baseline_smoke()
    kf_smoke = run_kf_baseline_smoke_plan(suite_yaml=suite_kf)
    step("test_kf_baseline_smoke_plan", kf_smoke.ok, f"{kf_smoke.note} (run_dir={kf_smoke.run_dir})")

    # 10) Route-B tiny adapt smoke (trained,frozen + trained,budgeted + overflow path)
    suite_adapt = _default_suite_adapt_smoke()
    suite_adapt_overflow = _default_suite_adapt_smoke_overflow()
    adp = run_adapt_smoke_route_b(
        suite_yaml=suite_adapt,
        suite_overflow_yaml=suite_adapt_overflow,
    )
    step("test_adapt_smoke_plan", adp.ok, f"{adp.note} (run_dir={adp.run_dir})")

    # 11) model_cache_dir invariance smoke (cache miss -> hit + metric invariance)
    suite_cache = _default_suite_cache_smoke()
    cache_res = run_cache_invariance(suite_yaml=suite_cache)
    step("test_cache_invariance", cache_res.ok, f"{cache_res.note} (run_dir={cache_res.run_dir})")

    # 12) explicit plan-matrix smoke
    suite_plan = _default_suite_plan_matrix_smoke()
    pm = run_plan_matrix_minimal(suite_yaml=suite_plan)
    step("test_plan_matrix_minimal", pm.ok, f"{pm.note} (run_dir={pm.run_dir})")

    # 13) S7 report schema guardrails (required S6 tables/columns/plots)
    rep_guard = run_report_schema_guardrails(suite_yaml=suite_plan)
    step("test_report_schema_guardrails", rep_guard.ok, f"{rep_guard.note} (out_dir={rep_guard.out_dir})")

    # 14) Fig5a report smoke (KNet vs oracle_kf overlay outputs)
    suite_fig5a_smoke = _default_suite_fig5a_overlay_smoke()
    fig5a_smoke = run_fig5a_report_smoke(suite_yaml=suite_fig5a_smoke)
    step("test_fig5a_report_smoke", fig5a_smoke.ok, f"{fig5a_smoke.note} (out_dir={fig5a_smoke.out_dir})")

    # 15) CPU determinism policy smoke for stability metrics
    det = run_determinism_cpu_seed_stable(suite_yaml=suite_plan)
    step("test_determinism_cpu_seed_stable", det.ok, f"{det.note} (run_dir={det.run_dir})")

    # 16) GPU determinism smoke (skips cleanly if no CUDA)
    gpu_det = run_determinism_gpu_seed_stable(suite_yaml=suite_plan)
    step(
        "test_determinism_gpu_seed_stable",
        gpu_det.ok,
        f"{gpu_det.note} (run_dir={gpu_det.run_dir})",
        skipped=bool(gpu_det.skipped),
    )

    # 17) run_dir artifact invariants (success + failed run structures)
    inv = run_artifact_invariants_smoke()
    step("test_artifact_invariants_smoke", inv.ok, inv.note)

    # 18) legacy failure.json compatibility mapping (category -> failure_type on read)
    compat = run_failure_type_compat()
    step("test_failure_type_compat", compat.ok, compat.note)

    print("=== ALL TESTS PASSED ===")


if __name__ == "__main__":
    main()
