from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple


@dataclass
class AdaptSmokeResult:
    ok: bool
    run_dir: Path
    note: str


def _bench_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _run(cmd: List[str], cwd: Path, env: Dict[str, str]) -> Tuple[int, str]:
    cp = subprocess.run(cmd, cwd=str(cwd), env=env, capture_output=True, text=True)
    out = (cp.stdout or "") + (cp.stderr or "")
    return cp.returncode, out


def _find_run_dir(
    bench_root: Path,
    suite_name: str,
    task_id: str,
    model_id: str,
    track: str,
) -> Path:
    root = bench_root / "runs" / suite_name / task_id / model_id / track / "seed_0"
    cands = sorted(root.glob("scenario_*"))
    if not cands:
        return Path("")
    return cands[0]


def _read_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_per_step(raw: object) -> Dict[int, int]:
    out: Dict[int, int] = {}
    if isinstance(raw, dict):
        items = raw.items()
    elif isinstance(raw, list):
        items = enumerate(raw)
    else:
        return out
    for k, v in items:
        try:
            out[int(k)] = int(v)
        except Exception:
            continue
    return out


def run_adapt_smoke_route_b(
    suite_yaml: Path,
    suite_overflow_yaml: Path,
) -> AdaptSmokeResult:
    bench_root = _bench_root()
    suite_yaml = suite_yaml.expanduser().resolve()
    suite_overflow_yaml = suite_overflow_yaml.expanduser().resolve()

    task_id = "C_shift_adapt_smoke_v0"
    model_id = "adaptive_knet"

    env = os.environ.copy()
    env["BENCH_DATA_CACHE"] = str((bench_root / "bench_data_cache").resolve())

    cmd_data = [
        sys.executable,
        "-m",
        "bench.tasks.smoke_data",
        "--suite-yaml",
        str(suite_yaml),
        "--task",
        task_id,
        "--seed",
        "0",
    ]
    rc_data, out_data = _run(cmd_data, bench_root, env)
    if rc_data != 0:
        return AdaptSmokeResult(ok=False, run_dir=Path(""), note=f"smoke_data failed:\n{out_data}")

    cmd_frozen = [
        sys.executable,
        "-m",
        "bench.runners.run_suite",
        "--suite-yaml",
        str(suite_yaml),
        "--tasks",
        task_id,
        "--models",
        model_id,
        "--seeds",
        "0",
        "--track",
        "frozen",
        "--init-id",
        "trained",
        "--device",
        "cpu",
    ]
    rc_frozen, out_frozen = _run(cmd_frozen, bench_root, env)
    if rc_frozen != 0:
        return AdaptSmokeResult(ok=False, run_dir=Path(""), note=f"frozen run failed:\n{out_frozen}")

    run_dir_frozen = _find_run_dir(
        bench_root=bench_root,
        suite_name="adapt_smoke",
        task_id=task_id,
        model_id=model_id,
        track="frozen",
    )
    if not run_dir_frozen.exists():
        return AdaptSmokeResult(ok=False, run_dir=Path(""), note="frozen run_dir not found")

    required_frozen = [
        "run_plan.json",
        "budget_ledger.json",
        "checkpoints/model.pt",
        "checkpoints/train_state.json",
        "metrics.json",
        "metrics_step.csv",
        "timing.csv",
    ]
    missing_frozen = [p for p in required_frozen if not (run_dir_frozen / p).exists()]
    if missing_frozen:
        return AdaptSmokeResult(
            ok=False,
            run_dir=run_dir_frozen,
            note=f"frozen run missing required artifacts: {missing_frozen}",
        )

    ledger_frozen = _read_json(run_dir_frozen / "budget_ledger.json")
    if int(ledger_frozen.get("adapt_updates_used", -1)) != 0:
        return AdaptSmokeResult(
            ok=False,
            run_dir=run_dir_frozen,
            note=f"frozen track violation: adapt_updates_used={ledger_frozen.get('adapt_updates_used')}",
        )

    cmd_budgeted = [
        sys.executable,
        "-m",
        "bench.runners.run_suite",
        "--suite-yaml",
        str(suite_yaml),
        "--tasks",
        task_id,
        "--models",
        model_id,
        "--seeds",
        "0",
        "--track",
        "budgeted",
        "--init-id",
        "trained",
        "--device",
        "cpu",
    ]
    rc_budgeted, out_budgeted = _run(cmd_budgeted, bench_root, env)
    if rc_budgeted != 0:
        return AdaptSmokeResult(ok=False, run_dir=run_dir_frozen, note=f"budgeted run failed:\n{out_budgeted}")

    run_dir_budgeted = _find_run_dir(
        bench_root=bench_root,
        suite_name="adapt_smoke",
        task_id=task_id,
        model_id=model_id,
        track="budgeted",
    )
    if not run_dir_budgeted.exists():
        return AdaptSmokeResult(ok=False, run_dir=Path(""), note="budgeted run_dir not found")

    required_budgeted = [
        "run_plan.json",
        "budget_ledger.json",
        "checkpoints/model.pt",
        "checkpoints/train_state.json",
        "metrics.json",
        "metrics_step.csv",
        "timing.csv",
    ]
    missing_budgeted = [p for p in required_budgeted if not (run_dir_budgeted / p).exists()]
    if missing_budgeted:
        return AdaptSmokeResult(
            ok=False,
            run_dir=run_dir_budgeted,
            note=f"budgeted run missing required artifacts: {missing_budgeted}",
        )

    ledger_budgeted = _read_json(run_dir_budgeted / "budget_ledger.json")
    metrics_budgeted = _read_json(run_dir_budgeted / "metrics.json")
    if str(metrics_budgeted.get("status", "")).lower() != "ok":
        return AdaptSmokeResult(
            ok=False,
            run_dir=run_dir_budgeted,
            note=f"budgeted metrics status is not ok: {metrics_budgeted.get('status')}",
        )

    adapt_updates_used = int(ledger_budgeted.get("adapt_updates_used", -1))
    if adapt_updates_used <= 0:
        return AdaptSmokeResult(
            ok=False,
            run_dir=run_dir_budgeted,
            note=f"budgeted track expected adaptation updates > 0, got {adapt_updates_used}",
        )
    if adapt_updates_used > 200:
        return AdaptSmokeResult(
            ok=False,
            run_dir=run_dir_budgeted,
            note=f"budgeted track violation: adapt_updates_used={adapt_updates_used} (>200)",
        )

    per_step = _parse_per_step(ledger_budgeted.get("adapt_updates_per_step", {}))
    if per_step and max(per_step.values()) > 1:
        return AdaptSmokeResult(
            ok=False,
            run_dir=run_dir_budgeted,
            note=f"budgeted track violation: max per-step updates={max(per_step.values())} (>1)",
        )

    allowed_after_t0_only = bool(ledger_budgeted.get("allowed_after_t0_only", True))
    t0 = ledger_budgeted.get("adapt_t0", metrics_budgeted.get("t0_used"))
    try:
        t0_i = int(t0) if t0 is not None else None
    except Exception:
        t0_i = None
    if allowed_after_t0_only and t0_i is not None:
        pre_t0 = [t for t, c in per_step.items() if t < t0_i and c > 0]
        if pre_t0:
            return AdaptSmokeResult(
                ok=False,
                run_dir=run_dir_budgeted,
                note=f"budgeted track violation: updates before t0 (t0={t0_i}, first={min(pre_t0)})",
            )

    cmd_data_overflow = [
        sys.executable,
        "-m",
        "bench.tasks.smoke_data",
        "--suite-yaml",
        str(suite_overflow_yaml),
        "--task",
        task_id,
        "--seed",
        "0",
    ]
    rc_data_overflow, out_data_overflow = _run(cmd_data_overflow, bench_root, env)
    if rc_data_overflow != 0:
        return AdaptSmokeResult(
            ok=False,
            run_dir=run_dir_budgeted,
            note=f"overflow smoke_data failed:\n{out_data_overflow}",
        )

    cmd_overflow = [
        sys.executable,
        "-m",
        "bench.runners.run_suite",
        "--suite-yaml",
        str(suite_overflow_yaml),
        "--tasks",
        task_id,
        "--models",
        model_id,
        "--seeds",
        "0",
        "--track",
        "budgeted",
        "--init-id",
        "trained",
        "--device",
        "cpu",
    ]
    rc_overflow, out_overflow = _run(cmd_overflow, bench_root, env)
    if rc_overflow != 0:
        return AdaptSmokeResult(
            ok=False,
            run_dir=run_dir_budgeted,
            note=f"overflow run command failed unexpectedly:\n{out_overflow}",
        )

    run_dir_overflow = _find_run_dir(
        bench_root=bench_root,
        suite_name="adapt_smoke_overflow",
        task_id=task_id,
        model_id=model_id,
        track="budgeted",
    )
    if not run_dir_overflow.exists():
        return AdaptSmokeResult(ok=False, run_dir=run_dir_budgeted, note="overflow run_dir not found")
    if not (run_dir_overflow / "failure.json").exists():
        return AdaptSmokeResult(
            ok=False,
            run_dir=run_dir_overflow,
            note="overflow expected failure.json, but it is missing",
        )

    failure = _read_json(run_dir_overflow / "failure.json")
    failure_type = str(failure.get("failure_type", ""))
    if failure_type != "budget_overflow":
        return AdaptSmokeResult(
            ok=False,
            run_dir=run_dir_overflow,
            note=f"overflow expected failure_type=budget_overflow, got {failure_type}",
        )
    if "category" in failure:
        return AdaptSmokeResult(
            ok=False,
            run_dir=run_dir_overflow,
            note="overflow failure.json must not contain legacy 'category' key",
        )

    return AdaptSmokeResult(
        ok=True,
        run_dir=run_dir_budgeted,
        note="adaptive_knet trained,frozen and trained,budgeted passed; overflow path produced budget_overflow",
    )
