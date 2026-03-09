from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import yaml

from bench.utils.logging import configure_logging, get_logger

from .bench_generated import (
    default_cache_root,
    make_dataloaders_v0,
    log_first_batch_v0,
    prepare_bench_generated_v0,
    LoaderCfgV0,
)


logger = get_logger(__name__)
from .data_format import load_npz_split_v0
from .generator.datasets.common import DatasetMissingError
from .generators.linear import (
    kalmannet_tsp_F_linear_canonical,
    kalmannet_tsp_H_reverse_canonical,
)


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _find_task_cfg(suite: Dict[str, Any], task_id: str) -> Dict[str, Any]:
    for t in suite.get("tasks", []):
        if str(t.get("task_id")) == task_id:
            return t
    raise KeyError(f"task_id not found in suite: {task_id}")


def _canonical_match_report(F: np.ndarray, H: np.ndarray) -> Tuple[bool, bool, float, float]:
    n = int(F.shape[0])
    F_ref = kalmannet_tsp_F_linear_canonical(n)
    H_ref = kalmannet_tsp_H_reverse_canonical(n)
    f_err = float(np.max(np.abs(F.astype(np.float64) - F_ref)))
    h_err = float(np.max(np.abs(H.astype(np.float64) - H_ref)))
    return (f_err <= 1e-6, h_err <= 1e-6, f_err, h_err)


def _variance_ratio_report(x: np.ndarray, y: np.ndarray, H: np.ndarray, t0: int) -> float:
    # residual v = y - Hx
    y_clean = np.matmul(x.astype(np.float64), H.astype(np.float64).T)
    v = y.astype(np.float64) - y_clean
    var_pre = float(np.var(v[:, :t0, :]))
    var_post = float(np.var(v[:, t0:, :]))
    return float(var_post / (var_pre + 1e-12))


def main() -> int:
    p = argparse.ArgumentParser(description="bench_generated v0 smoke check (generate -> cache -> load -> shape)")
    p.add_argument("--suite-yaml", type=str, required=True, help="Path to suite YAML (e.g., /mnt/data/suite_shift.yaml)")
    p.add_argument("--task", type=str, required=True, help="task_id (e.g., C_shift_Rscale_v0)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cache-root", type=str, default="", help="Override BENCH_DATA_CACHE (optional)")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--pin-memory", action="store_true")
    p.add_argument("--no-shuffle", action="store_true")
    p.add_argument("--sweep-all", action="store_true", help="Generate all scenarios in task sweep grid")
    p.add_argument(
        "--log-level",
        type=str,
        choices=("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"),
        default="INFO",
    )
    p.add_argument("--log-to-file", action="store_true")
    p.add_argument("--log-file", type=str, default=None)
    args = p.parse_args()
    configure_logging(
        str(args.log_level),
        run_dir=None,
        log_to_file=bool(args.log_file or args.log_to_file),
        log_file=(Path(str(args.log_file)) if args.log_file else None),
    )

    suite_path = Path(args.suite_yaml).expanduser().resolve()
    suite = _load_yaml(suite_path)
    logger.info("smoke_data suite_yaml=%s task=%s seed=%s", suite_path, args.task, args.seed)

    suite_name = str(suite["suite"]["name"])
    task_cfg = _find_task_cfg(suite, args.task)

    # enabled check (D11)
    enabled = bool(task_cfg.get("enabled", True))
    if not enabled:
        print(f"[SKIP] task.enabled=false for {args.task}. (D11) Use enabled:true to run.")
        return 2

    cache_root = Path(args.cache_root).expanduser().resolve() if args.cache_root else default_cache_root()

    # For smoke: default is baseline scenario, unless --sweep-all
    if args.sweep_all:
        scenario_overrides: Optional[Dict[str, Any]] = None
    else:
        scenario_overrides = {}

    try:
        arts = prepare_bench_generated_v0(
            suite_name=suite_name,
            task_cfg=task_cfg,
            seed=int(args.seed),
            cache_root=cache_root,
            scenario_overrides=scenario_overrides,
        )
    except DatasetMissingError as exc:
        print(f"[DATA MISSING] {exc}")
        return 3

    print(f"[OK] prepared {len(arts)} scenario(s) at cache_root={cache_root}")
    for art in arts:
        print(f"  - scenario_id={art.scenario_id} cache_dir={art.cache_dir}")

        # Load one split npz to verify meta + F/H + variance ratio
        loaded = load_npz_split_v0(art.train.path)
        meta = loaded.meta

        # --- Meta echo: shift diagnostics only for shift tasks ---
        t0 = None
        r_scale = None
        try:
            noise_meta = meta.get("noise", {}) if isinstance(meta, dict) else {}
            shift_meta = noise_meta.get("shift", {}) if isinstance(noise_meta, dict) else {}
            if isinstance(shift_meta, dict) and ("t0" in shift_meta):
                t0 = int(shift_meta["t0"])
                post_shift = shift_meta.get("post_shift", {}) if isinstance(shift_meta.get("post_shift", {}), dict) else {}
                if "R_scale" in post_shift:
                    r_scale = float(post_shift["R_scale"])
                elif "R_scale(applied)" in post_shift:
                    r_scale = float(post_shift["R_scale(applied)"])
                print(f"    meta.noise.shift.t0 = {t0}")
                if r_scale is not None:
                    print(f"    meta.noise.shift.post_shift.R_scale(applied) = {r_scale}")
            else:
                # Non-shift task: show base R.r2 if present, without warning noise.
                base_r2 = None
                if isinstance(noise_meta, dict):
                    r_meta = noise_meta.get("R", {})
                    if isinstance(r_meta, dict) and ("r2" in r_meta):
                        base_r2 = float(r_meta["r2"])
                if base_r2 is not None:
                    print(f"    meta.noise.R.r2 = {base_r2}")
        except Exception as e:
            print(f"    [WARN] failed to parse meta noise diagnostics: {e}")

        # --- Canonical F/H match check (for canonical_inverse tasks) ---
        if loaded.F is not None and loaded.H is not None:
            F_ok, H_ok, f_err, h_err = _canonical_match_report(loaded.F, loaded.H)
            print(f"    canonical F match: {F_ok} (max_abs_err={f_err:.3e})")
            print(f"    canonical H match: {H_ok} (max_abs_err={h_err:.3e})")
        else:
            print("    [WARN] NPZ missing F/H; cannot verify canonical mapping.")

        # --- Empirical variance ratio check ---
        if (t0 is not None) and (loaded.H is not None):
            ratio = _variance_ratio_report(loaded.x, loaded.y, loaded.H, t0=int(t0))
            if r_scale is None:
                print(f"    residual var ratio(post/pre) ≈ {ratio:.3f}")
            else:
                print(f"    residual var ratio(post/pre) ≈ {ratio:.3f} (meta R_scale={r_scale})")

        # DataLoader batch shape check
        train_loader, val_loader, test_loader = make_dataloaders_v0(
            art,
            loader_cfg=LoaderCfgV0(
                batch_size=int(args.batch_size),
                num_workers=int(args.num_workers),
                pin_memory=bool(args.pin_memory),
                shuffle_train=not bool(args.no_shuffle),
            ),
        )
        info = log_first_batch_v0(train_loader)
        print(f"    first train batch: {info}")

    print("[DONE] smoke check complete")

    # --- Cache invalidation guide (important after generator changes) ---
    print("[CACHE NOTE] F/H mapping changed -> delete old cache for this task and regenerate.")
    print("            e.g.) rm -rf bench_data_cache/shift/C_shift_Rscale_v0/")
    print("            then rerun the same smoke_data command.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
