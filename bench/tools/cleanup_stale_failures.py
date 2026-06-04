from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List


@dataclass(frozen=True)
class StaleFailure:
    run_dir: Path
    failure_path: Path
    metrics_path: Path
    destination: Path
    reason: str


def _iter_stale_failures(runs_root: Path) -> Iterable[StaleFailure]:
    for failure_path in sorted(runs_root.rglob("failure.json")):
        if "stale" in failure_path.parts:
            continue
        run_dir = failure_path.parent
        metrics_path = run_dir / "metrics.json"
        if not metrics_path.exists():
            continue
        failure_mtime = failure_path.stat().st_mtime
        metrics_mtime = metrics_path.stat().st_mtime
        if metrics_mtime < failure_mtime:
            continue
        destination = run_dir / "stale" / "failure.json"
        reason = "metrics_newer_than_failure"
        if metrics_mtime == failure_mtime:
            reason = "metrics_same_timestamp_as_failure"
        yield StaleFailure(
            run_dir=run_dir,
            failure_path=failure_path,
            metrics_path=metrics_path,
            destination=destination,
            reason=reason,
        )


def _unique_destination(path: Path) -> Path:
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    parent = path.parent
    idx = 1
    while True:
        candidate = parent / f"{stem}.{idx}{suffix}"
        if not candidate.exists():
            return candidate
        idx += 1


def _write_log(log_csv: Path, rows: List[dict]) -> None:
    log_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "action",
        "reason",
        "run_dir",
        "failure_path",
        "metrics_path",
        "destination",
    ]
    with log_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Move stale failure.json artifacts aside when a successful metrics.json "
            "exists and is at least as new as the failure artifact."
        )
    )
    parser.add_argument("--runs-root", default="runs", help="Root directory to scan.")
    parser.add_argument(
        "--log-csv",
        default="reports/cleanup_stale_failures_log.csv",
        help="CSV audit log path to write.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply moves. Default is dry-run and only writes the audit log.",
    )
    args = parser.parse_args()

    runs_root = Path(args.runs_root).expanduser().resolve()
    log_csv = Path(args.log_csv).expanduser().resolve()
    stale = list(_iter_stale_failures(runs_root))
    rows: List[dict] = []
    moved = 0
    for item in stale:
        destination = _unique_destination(item.destination)
        action = "move" if args.apply else "dry_run_move"
        rows.append(
            {
                "action": action,
                "reason": item.reason,
                "run_dir": str(item.run_dir),
                "failure_path": str(item.failure_path),
                "metrics_path": str(item.metrics_path),
                "destination": str(destination),
            }
        )
        if args.apply:
            destination.parent.mkdir(parents=True, exist_ok=True)
            item.failure_path.rename(destination)
            moved += 1

    _write_log(log_csv, rows)
    mode = "apply" if args.apply else "dry-run"
    print(
        f"[cleanup_stale_failures] mode={mode} scanned={runs_root} "
        f"stale_candidates={len(stale)} moved={moved} log_csv={log_csv}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
