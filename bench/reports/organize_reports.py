from __future__ import annotations

import argparse
import re
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

try:
    import yaml  # type: ignore
except Exception:
    yaml = None


_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_TIME_RE = re.compile(r"^\d{6}$")

_KINDS = {"tables", "plots", "misc"}
_TABLE_PREFIXES = (
    "summary_",
    "aggregate_",
    "plan_compare_",
    "failure_by_plan_",
    "ops_by_plan_",
    "fig5a_points_",
)
_PLOT_PREFIXES = (
    "ops_tradeoff_",
    "fig5a_overlay_mse_db_vs_inv_r2_db_",
    "fig5a_mse_db_vs_inv_r2_db_",
)


@dataclass
class MoveItem:
    src: Path
    dst: Path
    suite: str
    kind: str


@dataclass
class OrganizeResult:
    moved: int
    skipped: int
    suites: int
    notes: List[str]


def _bench_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_yaml(path: Path) -> Dict[str, object]:
    if yaml is None:
        return {}
    try:
        obj = yaml.safe_load(path.read_text(encoding="utf-8"))
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def _build_task_to_suite(configs_dir: Path) -> Dict[str, Set[str]]:
    task_to_suite: Dict[str, Set[str]] = {}
    for y in sorted(configs_dir.glob("suite*.yaml")):
        obj = _load_yaml(y)
        suite_name = str(((obj.get("suite") or {}) if isinstance(obj.get("suite"), dict) else {}).get("name", y.stem))
        tasks = obj.get("tasks")
        if not isinstance(tasks, list):
            continue
        for t in tasks:
            if not isinstance(t, dict):
                continue
            tid = t.get("task_id")
            if not tid:
                continue
            task_to_suite.setdefault(str(tid), set()).add(suite_name)
    return task_to_suite


def _is_new_layout_path(rel: Path) -> bool:
    parts = rel.parts
    if len(parts) >= 5 and _DATE_RE.match(parts[1]) and _TIME_RE.match(parts[2]) and parts[3] in _KINDS:
        return True
    if len(parts) >= 4 and parts[1] == "latest" and parts[2] in _KINDS:
        return True
    return False


def _infer_task_from_plot_name(name: str, task_ids: List[str]) -> Optional[str]:
    if name.startswith("shift_recovery_") and name.endswith(".png"):
        return name[len("shift_recovery_") : -len(".png")]

    if name.startswith("severity_sweep_") and name.endswith("_R_scale.png"):
        return name[len("severity_sweep_") : -len("_R_scale.png")]

    for pref in ("track_compare_", "budget_curve_", "mse_db_by_model_"):
        if name.startswith(pref) and name.endswith(".png"):
            body = name[len(pref) : -len(".png")]
            for tid in task_ids:
                if body == tid or body.startswith(tid + "_"):
                    return tid
    return None


def _infer_suite_from_name(name: str, task_to_suite: Dict[str, Set[str]], task_ids_longest: List[str]) -> Optional[str]:
    for pref in _TABLE_PREFIXES:
        if name.startswith(pref) and (name.endswith(".csv") or name.endswith(".tex")):
            s = name[len(pref) :]
            if "." in s:
                s = s.rsplit(".", 1)[0]
            if s:
                return s

    for pref in _PLOT_PREFIXES:
        if name.startswith(pref) and name.endswith(".png"):
            s = name[len(pref) : -len(".png")]
            if s:
                return s

    task = _infer_task_from_plot_name(name, task_ids_longest)
    if task is None:
        return None
    suites = sorted(task_to_suite.get(task, set()))
    if len(suites) == 1:
        return suites[0]
    if len(suites) > 1:
        return suites[0]
    return None


def _infer_kind(name: str) -> str:
    lower = name.lower()
    if lower.endswith(".png"):
        return "plots"
    if lower.endswith(".csv"):
        return "tables"
    if lower.endswith(".tex"):
        return "misc"
    return "misc"


def _unique_dst(dst: Path) -> Path:
    if not dst.exists():
        return dst
    stem = dst.stem
    suffix = dst.suffix
    i = 1
    while True:
        cand = dst.with_name(f"{stem}.dup{i}{suffix}")
        if not cand.exists():
            return cand
        i += 1


def _build_move_plan(reports_dir: Path, task_to_suite: Dict[str, Set[str]]) -> Tuple[List[MoveItem], int]:
    task_ids_longest = sorted(task_to_suite.keys(), key=lambda s: len(s), reverse=True)
    moves: List[MoveItem] = []
    skipped = 0

    files = [p for p in reports_dir.rglob("*") if p.is_file()]
    for src in sorted(files):
        rel = src.relative_to(reports_dir)

        if _is_new_layout_path(rel):
            skipped += 1
            continue

        suite_from_old = None
        kind_from_old = None
        parts = rel.parts
        if len(parts) >= 3 and parts[1] in _KINDS:
            suite_from_old = parts[0]
            kind_from_old = parts[1]

        suite = suite_from_old
        if suite is None:
            suite = _infer_suite_from_name(src.name, task_to_suite, task_ids_longest)
        if not suite:
            suite = "_unmapped"

        kind = kind_from_old or _infer_kind(src.name)
        if kind not in _KINDS:
            kind = "misc"

        dt = datetime.fromtimestamp(src.stat().st_mtime).astimezone()
        day = dt.strftime("%Y-%m-%d")
        tms = dt.strftime("%H%M%S")
        dst = reports_dir / suite / day / tms / kind / src.name
        if src.resolve() == dst.resolve():
            skipped += 1
            continue
        dst = _unique_dst(dst)
        moves.append(MoveItem(src=src, dst=dst, suite=suite, kind=kind))

    return moves, skipped


def _remove_empty_dirs(root: Path) -> None:
    for p in sorted([d for d in root.rglob("*") if d.is_dir()], key=lambda x: len(x.parts), reverse=True):
        try:
            if not any(p.iterdir()):
                p.rmdir()
        except Exception:
            continue


def organize_reports(*, reports_dir: Path, configs_dir: Path, dry_run: bool, copy_mode: bool) -> OrganizeResult:
    task_to_suite = _build_task_to_suite(configs_dir)
    move_plan, skipped = _build_move_plan(reports_dir, task_to_suite)

    notes: List[str] = []
    if dry_run:
        for m in move_plan[:50]:
            notes.append(f"[dry-run] {m.src} -> {m.dst}")
        if len(move_plan) > 50:
            notes.append(f"[dry-run] ... and {len(move_plan)-50} more")
        suites = len({m.suite for m in move_plan})
        return OrganizeResult(moved=0, skipped=skipped, suites=suites, notes=notes)

    moved = 0
    for m in move_plan:
        m.dst.parent.mkdir(parents=True, exist_ok=True)
        if copy_mode:
            shutil.copy2(m.src, m.dst)
        else:
            shutil.move(str(m.src), str(m.dst))
        moved += 1

    _remove_empty_dirs(reports_dir)
    suites = len({m.suite for m in move_plan})
    notes.append(f"organized {moved} file(s) across {suites} suite bucket(s)")
    if copy_mode:
        notes.append("mode=copy (source files retained)")
    else:
        notes.append("mode=move (source files relocated)")

    return OrganizeResult(moved=moved, skipped=skipped, suites=suites, notes=notes)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reports-dir", type=str, default=None, help="default: <bench_root>/reports")
    ap.add_argument("--configs-dir", type=str, default=None, help="default: <bench_root>/bench/configs")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--copy", action="store_true", help="copy files instead of moving")
    args = ap.parse_args()

    root = _bench_root()
    reports_dir = Path(args.reports_dir).expanduser().resolve() if args.reports_dir else (root / "reports")
    configs_dir = Path(args.configs_dir).expanduser().resolve() if args.configs_dir else (root / "bench" / "configs")

    res = organize_reports(
        reports_dir=reports_dir,
        configs_dir=configs_dir,
        dry_run=bool(args.dry_run),
        copy_mode=bool(args.copy),
    )

    print(f"reports_dir={reports_dir}")
    print(f"moved={res.moved} skipped={res.skipped} suites={res.suites}")
    for n in res.notes:
        print(n)


if __name__ == "__main__":
    main()
