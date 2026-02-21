from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional

from .aggregate import (
    RunRecord,
    aggregate_by_seed,
    aggregate_to_latex_tabular,
    expected_plan_from_suite,
    load_yaml,
    merge_records_with_expected,
    scan_runs,
    write_aggregate_csv,
    write_summary_csv,
)
from .plots import plot_shift_recovery_curves, plot_severity_sweep


def _bench_root() -> Path:
    # .../bench/bench/reports/make_report.py -> parents[2] == .../bench
    return Path(__file__).resolve().parents[2]


def _suite_name(suite: Dict[str, Any]) -> str:
    return str((suite.get("suite", {}) or {}).get("name", "unknown"))


def _enabled(obj: Dict[str, Any], default: bool = True) -> bool:
    if "enabled" not in obj:
        return default
    return bool(obj["enabled"])


def _task_shift_t0(task: Dict[str, Any]) -> Optional[int]:
    try:
        t0 = (task.get("noise", {}) or {}).get("shift", {}).get("t0", None)
        return int(t0) if t0 is not None else None
    except Exception:
        return None


def _filter_records(records: List[RunRecord], suite_name: str, task_id: Optional[str] = None) -> List[RunRecord]:
    out = [r for r in records if r.suite == suite_name]
    if task_id:
        out = [r for r in out if r.task_id == task_id]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite-yaml", type=str, required=True)
    ap.add_argument("--runs-root", type=str, default=None, help="default: <bench_root>/runs")
    ap.add_argument("--out-dir", type=str, default=None, help="default: <bench_root>/reports")
    ap.add_argument("--dry-run", action="store_true", help="print plan only; do not write files")
    ap.add_argument("--latex", action="store_true", help="also write summary_<suite>.tex (aggregate table)")
    args = ap.parse_args()

    suite_path = Path(args.suite_yaml).expanduser().resolve()
    suite = load_yaml(suite_path)
    suite_name = _suite_name(suite)

    bench_root = _bench_root()
    runs_root = Path(args.runs_root).expanduser().resolve() if args.runs_root else (bench_root / "runs")
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else (bench_root / "reports")

    # output filenames (consistent)
    summary_csv = out_dir / f"summary_{suite_name}.csv"
    aggregate_csv = out_dir / f"aggregate_{suite_name}.csv"
    latex_tex = out_dir / f"summary_{suite_name}.tex"

    # scan existing run dirs
    scanned = scan_runs(runs_root=runs_root, suite_name=suite_name)

    # include expected missing combos based on suite enabled_policy (D11)
    expected = expected_plan_from_suite(suite)
    records = merge_records_with_expected(scanned, expected)

    # aggregation
    agg_rows = aggregate_by_seed(records)

    # dry run print
    if args.dry_run:
        print("[dry-run] suite:", suite_name)
        print("[dry-run] runs_root:", runs_root)
        print("[dry-run] out_dir:", out_dir)
        print("[dry-run] scanned run_dirs:", len(scanned))
        print("[dry-run] total rows (with expected missing):", len(records))
        print("[dry-run] will write:", summary_csv.name, aggregate_csv.name)
        if args.latex:
            print("[dry-run] will write:", latex_tex.name)

        # plots plan (shift suite)
        if suite_name == "shift":
            tasks = suite.get("tasks", []) or []
            for t in tasks:
                if not _enabled(t, True):
                    continue
                tid = t.get("task_id")
                if not tid:
                    continue
                print("[dry-run] will plot:", f"shift_recovery_{tid}.png")
                print("[dry-run] will plot:", f"severity_sweep_{tid}_R_scale.png")
        return

    # write tables
    write_summary_csv(records, summary_csv)
    write_aggregate_csv(agg_rows, aggregate_csv)

    # optional latex export (aggregate table)
    if args.latex:
        tex = aggregate_to_latex_tabular(
            agg_rows,
            caption=f"Aggregate results ({suite_name})",
            label=f"tab:summary_{suite_name}",
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        latex_tex.write_text(tex, encoding="utf-8")

    # plots (shift suite 핵심)
    if suite_name == "shift":
        tasks = suite.get("tasks", []) or []
        for t in tasks:
            if not _enabled(t, True):
                continue
            task_id = t.get("task_id")
            if not task_id:
                continue

            t0 = _task_shift_t0(t)

            task_records = _filter_records(records, suite_name=suite_name, task_id=str(task_id))

            # 1) shift recovery curve (mse_t)
            out_png = out_dir / f"shift_recovery_{task_id}.png"
            plot_shift_recovery_curves(
                task_id=str(task_id),
                records=task_records,
                out_path=out_png,
                t0=t0,
                metric="mse_t",
            )

            # 2) mismatch severity sweep (R_scale if present)
            out_png2 = out_dir / f"severity_sweep_{task_id}_R_scale.png"
            plot_severity_sweep(
                task_id=str(task_id),
                records=task_records,
                out_path=out_png2,
                severity_key="shift.post_shift.R_scale",
            )

    print(f"[make_report] done. wrote: {summary_csv} and {aggregate_csv}")
    if args.latex:
        print(f"[make_report] wrote: {latex_tex}")


if __name__ == "__main__":
    main()
