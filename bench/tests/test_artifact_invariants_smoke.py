from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass
class ArtifactInvariantsResult:
    ok: bool
    note: str


def _bench_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _find_first(base: Path, pattern: str) -> Path:
    cands = sorted(base.glob(pattern))
    if not cands:
        return Path("")
    return cands[0]


def run_artifact_invariants_smoke() -> ArtifactInvariantsResult:
    bench_root = _bench_root()

    # Success-run invariant: trained run_dir should have required files.
    train_base = (
        bench_root
        / "runs"
        / "train_smoke"
        / "A_linear_train_smoke_v0"
        / "kalmannet_tsp"
        / "frozen"
        / "seed_0"
    )
    run_dir_ok = _find_first(train_base, "scenario_*")
    if not run_dir_ok.exists():
        return ArtifactInvariantsResult(ok=False, note="missing train_smoke run_dir for invariant checks")

    required_ok: List[str] = [
        "run_plan.json",
        "budget_ledger.json",
        "metrics.json",
        "metrics_step.csv",
        "timing.csv",
        "checkpoints/model.pt",
    ]
    missing_ok = [p for p in required_ok if not (run_dir_ok / p).exists()]
    if missing_ok:
        return ArtifactInvariantsResult(
            ok=False,
            note=f"successful run missing required artifacts: {missing_ok} (run_dir={run_dir_ok})",
        )

    # Failed-run invariant: overflow failure run has failure.json with canonical fields.
    fail_base = (
        bench_root
        / "runs"
        / "adapt_smoke_overflow"
        / "C_shift_adapt_smoke_v0"
        / "adaptive_knet"
        / "budgeted"
        / "seed_0"
    )
    run_dir_fail = _find_first(fail_base, "scenario_*")
    if not run_dir_fail.exists():
        return ArtifactInvariantsResult(ok=False, note="missing adapt_smoke_overflow run_dir for invariant checks")

    failure_path = run_dir_fail / "failure.json"
    if not failure_path.exists():
        return ArtifactInvariantsResult(ok=False, note=f"failed run missing failure.json ({run_dir_fail})")
    failure = json.loads(failure_path.read_text(encoding="utf-8"))
    failure_type = str(failure.get("failure_type", "")).strip()
    failure_stage = str(failure.get("failure_stage") or failure.get("phase") or "").strip()
    if not failure_type:
        return ArtifactInvariantsResult(ok=False, note=f"failure.json missing failure_type ({failure_path})")
    if not failure_stage:
        return ArtifactInvariantsResult(ok=False, note=f"failure.json missing failure_stage/phase ({failure_path})")
    if "category" in failure:
        return ArtifactInvariantsResult(ok=False, note=f"failure.json should not contain legacy category ({failure_path})")

    return ArtifactInvariantsResult(
        ok=True,
        note="artifact invariants passed for successful and failed run_dirs",
    )
