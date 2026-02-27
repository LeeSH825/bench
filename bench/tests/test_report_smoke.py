from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class ReportSmokeResult:
    ok: bool
    note: str


def _bench_root() -> Path:
    return Path(__file__).resolve().parents[2]


def run_report_smoke(
    suite_yaml: Path,
    suite_name: str,
    runs_root: Path,
    out_dir: Path,
) -> ReportSmokeResult:
    bench_root = _bench_root()

    cmd = [
        sys.executable,
        "-m",
        "bench.reports.make_report",
        "--suite-yaml",
        str(suite_yaml),
        "--runs-root",
        str(runs_root),
        "--out-dir",
        str(out_dir),
        "--input-scope",
        "all_runs",
    ]
    cp = subprocess.run(cmd, cwd=str(bench_root), capture_output=True, text=True)
    out = (cp.stdout or "") + (cp.stderr or "")
    if cp.returncode != 0:
        return ReportSmokeResult(ok=False, note=f"make_report failed.\n{out}")

    summary = out_dir / f"summary_{suite_name}.csv"
    agg = out_dir / f"aggregate_{suite_name}.csv"
    if not summary.exists() or not agg.exists():
        return ReportSmokeResult(ok=False, note=f"missing report csv(s): {summary.exists()=}, {agg.exists()=}\n{out}")

    # For shift suite, expect at least one plot file (best-effort)
    if suite_name == "shift":
        plots = list(out_dir.glob("shift_recovery_*.png"))
        if not plots:
            return ReportSmokeResult(ok=False, note=f"shift suite expected at least one shift_recovery_*.png, none found.\n{out}")

    return ReportSmokeResult(ok=True, note="report smoke OK (summary/aggregate + plot present)")
