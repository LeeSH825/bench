from __future__ import annotations

import argparse
import math
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from bench.utils.logging import configure_logging, get_logger

from .aggregate import (
    RunRecord,
    aggregate_by_seed,
    aggregate_to_latex_tabular,
    build_plan_comparison_rows,
    expected_plan_from_suite,
    load_yaml,
    merge_records_with_expected,
    scan_runs,
    summarize_failures_by_plan,
    summarize_ops_by_plan,
    write_aggregate_csv,
    write_rows_csv,
    write_summary_csv,
)


logger = get_logger(__name__)
from .plots import (
    plot_fig5a_mse_vs_inv_r2,
    plot_budget_curve,
    plot_mse_db_by_model,
    plot_ops_tradeoff,
    plot_severity_sweep,
    plot_shift_recovery_curves,
    plot_track_comparison,
)


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


def _suite_task_ids(suite: Dict[str, Any], records: List[RunRecord]) -> List[str]:
    ids: List[str] = []
    for t in (suite.get("tasks", []) or []):
        if not _enabled(t, True):
            continue
        tid = t.get("task_id")
        if tid:
            ids.append(str(tid))
    if ids:
        return ids
    return sorted({r.task_id for r in records})


def _infer_baseline_mode(model_id: str) -> str:
    mid = str(model_id).strip().lower()
    if mid in {"oracle_kf", "mb_kf_oracle"}:
        return "oracle"
    if mid in {"nominal_kf", "mb_kf_nominal"}:
        return "nominal"
    if mid == "oracle_shift_kf":
        return "oracle_shift"
    return ""


def _build_fig5a_point_rows(records: List[RunRecord]) -> tuple[List[Dict[str, Any]], List[str]]:
    fields = [
        "model_id",
        "baseline_mode",
        "is_oracle_kf",
        "init_id",
        "track_id",
        "task_id",
        "scenario_id",
        "seed",
        "x_dim",
        "y_dim",
        "T",
        "q2",
        "r2",
        "inv_r2_db",
        "mse_db",
        "mse",
        "rmse",
        "run_dir",
    ]
    rows: List[Dict[str, Any]] = []
    for r in records:
        if r.status != "ok":
            continue
        if r.mse_db is None:
            continue
        row = {
            "model_id": str(r.model_id),
            "baseline_mode": _infer_baseline_mode(r.model_id),
            "is_oracle_kf": bool(str(r.model_id).strip().lower() in {"oracle_kf", "mb_kf_oracle"}),
            "init_id": str(r.init_id),
            "track_id": str(r.track_id),
            "task_id": str(r.task_id),
            "scenario_id": str(r.scenario_id),
            "seed": int(r.seed),
            "x_dim": ("" if r.x_dim is None else int(r.x_dim)),
            "y_dim": ("" if r.y_dim is None else int(r.y_dim)),
            "T": ("" if r.T is None else int(r.T)),
            "q2": ("" if r.q2 is None else float(r.q2)),
            "r2": ("" if r.r2 is None else float(r.r2)),
            "inv_r2_db": ("" if r.inv_r2_db is None else float(r.inv_r2_db)),
            "mse_db": float(r.mse_db),
            "mse": ("" if r.mse is None else float(r.mse)),
            "rmse": ("" if r.rmse is None else float(r.rmse)),
            "run_dir": ("" if str(r.run_dir) == "." else str(r.run_dir)),
        }
        rows.append(row)
    rows.sort(
        key=lambda rr: (
            str(rr["model_id"]),
            str(rr["task_id"]),
            str(rr["scenario_id"]),
            int(rr["seed"]),
        )
    )
    return rows, fields


def _copy_if_exists(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def _resolve_report_stamp(stamp: Optional[str]) -> tuple[str, str]:
    if stamp is None or str(stamp).strip() == "":
        now = datetime.now().astimezone()
        return now.strftime("%Y-%m-%d"), now.strftime("%H%M%S")

    s = str(stamp).strip()
    fmts = (
        "%Y%m%d-%H%M%S",
        "%Y-%m-%d_%H%M%S",
        "%Y-%m-%dT%H%M%S",
        "%Y-%m-%d %H%M%S",
    )
    for fmt in fmts:
        try:
            dt = datetime.strptime(s, fmt)
            return dt.strftime("%Y-%m-%d"), dt.strftime("%H%M%S")
        except Exception:
            continue
    raise ValueError(
        "Invalid --report-stamp format. Use one of: "
        "YYYYMMDD-HHMMSS, YYYY-MM-DD_HHMMSS, YYYY-MM-DDTHHMMSS, YYYY-MM-DD HHMMSS"
    )


def _organize_outputs_by_suite(
    *,
    out_dir: Path,
    suite_name: str,
    day_stamp: str,
    time_stamp: str,
    table_files: List[Path],
    plot_files: List[Path],
    misc_files: Optional[List[Path]] = None,
) -> Dict[str, List[Path]]:
    suite_dir = out_dir / str(suite_name)
    stamp_dir = suite_dir / str(day_stamp) / str(time_stamp)
    latest_dir = suite_dir / "latest"
    tables_dir = stamp_dir / "tables"
    plots_dir = stamp_dir / "plots"
    misc_dir = stamp_dir / "misc"
    latest_tables_dir = latest_dir / "tables"
    latest_plots_dir = latest_dir / "plots"
    latest_misc_dir = latest_dir / "misc"

    copied: Dict[str, List[Path]] = {"tables": [], "plots": [], "misc": []}
    if latest_dir.exists():
        shutil.rmtree(latest_dir)

    seen_tables = set()
    for src in table_files:
        src_res = src.resolve()
        if src_res in seen_tables:
            continue
        seen_tables.add(src_res)
        dst = tables_dir / src.name
        if _copy_if_exists(src, dst):
            copied["tables"].append(dst)
            _copy_if_exists(src, latest_tables_dir / src.name)

    seen_plots = set()
    for src in plot_files:
        src_res = src.resolve()
        if src_res in seen_plots:
            continue
        seen_plots.add(src_res)
        dst = plots_dir / src.name
        if _copy_if_exists(src, dst):
            copied["plots"].append(dst)
            _copy_if_exists(src, latest_plots_dir / src.name)

    seen_misc = set()
    for src in (misc_files or []):
        src_res = src.resolve()
        if src_res in seen_misc:
            continue
        seen_misc.add(src_res)
        dst = misc_dir / src.name
        if _copy_if_exists(src, dst):
            copied["misc"].append(dst)
            _copy_if_exists(src, latest_misc_dir / src.name)

    return copied


def main() -> None:
    severity_x_choices = (
        "severity_key",
        "mse_mean",
        "mse_db_mean",
        "rmse_mean",
        "inv_r2_db",
        "severity_r_scale_mean",
    )
    severity_y_choices = (
        "mse_mean",
        "mse_db_mean",
        "rmse_mean",
        "severity_key",
        "severity_r_scale_mean",
    )
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite-yaml", type=str, required=True)
    ap.add_argument("--runs-root", type=str, default=None, help="default: <bench_root>/runs")
    ap.add_argument("--out-dir", type=str, default=None, help="default: <bench_root>/reports")
    ap.add_argument(
        "--input-scope",
        type=str,
        choices=("latest_manifest", "all_runs"),
        default="latest_manifest",
        help="run discovery scope. default: latest_manifest",
    )
    ap.add_argument("--dry-run", action="store_true", help="print plan only; do not write files")
    ap.add_argument("--latex", action="store_true", help="also write summary_<suite>.tex (aggregate table)")
    # S6 additive report views
    ap.add_argument(
        "--group-by",
        type=str,
        default="init_id,track_id",
        help="group keys for failure/ops views (comma-separated). default: init_id,track_id",
    )
    ap.add_argument(
        "--include-ops",
        action="store_true",
        help="write ops summary table and ops tradeoff plot",
    )
    ap.add_argument(
        "--budget-curves",
        action="store_true",
        help="write per-task budget curve plot (adapt_updates_used vs metric)",
    )
    ap.add_argument(
        "--plan-views",
        action="store_true",
        help="write plan comparison and failure-by-plan tables, plus track comparison plots",
    )
    ap.add_argument(
        "--fig5a-plot",
        action="store_true",
        help="write Fig5(a)-style plot: mse_db vs inv_r2_db grouped by (model_id,x_dim,y_dim,T)",
    )
    ap.add_argument(
        "--severity-x-field",
        type=str,
        choices=severity_x_choices,
        default="severity_key",
        help="x field for severity sweep plot (default: severity_key)",
    )
    ap.add_argument(
        "--severity-y-field",
        type=str,
        choices=severity_y_choices,
        default="mse_mean",
        help="y field for severity sweep plot (default: mse_mean)",
    )
    ap.add_argument(
        "--organize-by-suite",
        dest="organize_by_suite",
        action="store_true",
        default=True,
        help="also mirror outputs under reports/<suite>/tables and reports/<suite>/plots (default: on)",
    )
    ap.add_argument(
        "--no-organize-by-suite",
        dest="organize_by_suite",
        action="store_false",
        help="disable suite-folder mirror and keep only flat reports/<files>",
    )
    ap.add_argument(
        "--report-stamp",
        type=str,
        default=None,
        help="optional fixed stamp for organized folders (e.g., 20260223-091500).",
    )
    ap.add_argument(
        "--log-level",
        type=str,
        choices=("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"),
        default="INFO",
    )
    ap.add_argument("--log-to-file", action="store_true")
    ap.add_argument("--log-file", type=str, default=None)
    args = ap.parse_args()
    configure_logging(
        str(args.log_level),
        run_dir=None,
        log_to_file=bool(args.log_file or args.log_to_file),
        log_file=(Path(str(args.log_file)) if args.log_file else None),
    )

    suite_path = Path(args.suite_yaml).expanduser().resolve()
    suite = load_yaml(suite_path)
    suite_name = _suite_name(suite)
    logger.info("make_report suite=%s input_scope=%s", suite_name, args.input_scope)

    bench_root = _bench_root()
    runs_root = Path(args.runs_root).expanduser().resolve() if args.runs_root else (bench_root / "runs")
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else (bench_root / "reports")
    day_stamp, time_stamp = _resolve_report_stamp(args.report_stamp)

    # baseline outputs (backward compatible)
    summary_csv = out_dir / f"summary_{suite_name}.csv"
    aggregate_csv = out_dir / f"aggregate_{suite_name}.csv"
    latex_tex = out_dir / f"summary_{suite_name}.tex"

    # S6 additive outputs
    plan_compare_csv = out_dir / f"plan_compare_{suite_name}.csv"
    failure_by_plan_csv = out_dir / f"failure_by_plan_{suite_name}.csv"
    ops_by_plan_csv = out_dir / f"ops_by_plan_{suite_name}.csv"
    fig5a_points_csv = out_dir / f"fig5a_points_{suite_name}.csv"
    fig5a_overlay_path = out_dir / f"fig5a_overlay_mse_db_vs_inv_r2_db_{suite_name}.png"
    fig5a_legacy_path = out_dir / f"fig5a_mse_db_vs_inv_r2_db_{suite_name}.png"

    scanned = scan_runs(
        runs_root=runs_root,
        suite_name=suite_name,
        input_scope=str(args.input_scope),
    )
    if args.input_scope == "latest_manifest" and len(scanned) == 0:
        print(
            f"[make_report] note: no runs discovered via latest manifest for suite={suite_name}. "
            "Use --input-scope all_runs to include historical scan."
        )
    expected = expected_plan_from_suite(suite)
    records = merge_records_with_expected(scanned, expected)
    agg_rows = aggregate_by_seed(records)
    suspicious = [
        r for r in records
        if r.status == "ok"
        and r.mse_db is not None
        and ((not math.isfinite(float(r.mse_db))) or float(r.mse_db) > 100.0)
    ]
    for rec in suspicious[:10]:
        logger.warning(
            "Suspicious mse_db in report model=%s task=%s scenario=%s seed=%s init=%s track=%s mse_db=%s run_dir=%s",
            rec.model_id,
            rec.task_id,
            rec.scenario_id,
            rec.seed,
            rec.init_id,
            rec.track_id,
            rec.mse_db,
            rec.run_dir,
        )

    if args.dry_run:
        print("[dry-run] suite:", suite_name)
        print("[dry-run] runs_root:", runs_root)
        print("[dry-run] input_scope:", args.input_scope)
        print("[dry-run] out_dir:", out_dir)
        print("[dry-run] scanned run_dirs:", len(scanned))
        print("[dry-run] total rows (with expected missing):", len(records))
        print("[dry-run] will write:", summary_csv.name, aggregate_csv.name)
        if args.latex:
            print("[dry-run] will write:", latex_tex.name)
        if args.plan_views:
            print("[dry-run] will write:", plan_compare_csv.name, failure_by_plan_csv.name)
        if args.include_ops:
            print("[dry-run] will write:", ops_by_plan_csv.name)
        if args.fig5a_plot:
            print("[dry-run] will write:", f"fig5a_points_{suite_name}.csv")
            print("[dry-run] will plot:", f"fig5a_overlay_mse_db_vs_inv_r2_db_{suite_name}.png")
            print("[dry-run] will plot:", f"fig5a_mse_db_vs_inv_r2_db_{suite_name}.png")
        if args.organize_by_suite:
            print("[dry-run] will mirror to:", out_dir / suite_name / day_stamp / time_stamp / "tables")
            print("[dry-run] will mirror to:", out_dir / suite_name / day_stamp / time_stamp / "plots")
            print("[dry-run] will mirror latest to:", out_dir / suite_name / "latest")

        for task_id in _suite_task_ids(suite, records):
            if suite_name == "shift":
                print("[dry-run] will plot:", f"shift_recovery_{task_id}.png")
                if (
                    str(args.severity_x_field) == "severity_key"
                    and str(args.severity_y_field) == "mse_mean"
                ):
                    print("[dry-run] will plot:", f"severity_sweep_{task_id}_R_scale.png")
                else:
                    print(
                        "[dry-run] will plot:",
                        f"severity_sweep_{task_id}_{args.severity_x_field}_vs_{args.severity_y_field}.png",
                    )
            if args.plan_views:
                print("[dry-run] will plot:", f"track_compare_{task_id}_<metric>.png")
                print("[dry-run] will plot:", f"mse_db_by_model_{task_id}.png")
            if args.budget_curves:
                print("[dry-run] will plot:", f"budget_curve_{task_id}_<metric>.png")
        if args.include_ops:
            print("[dry-run] will plot:", f"ops_tradeoff_{suite_name}.png")
        return

    # baseline tables
    write_summary_csv(records, summary_csv)
    write_aggregate_csv(agg_rows, aggregate_csv)
    written_tables: List[Path] = [summary_csv, aggregate_csv]
    written_plots: List[Path] = []
    written_misc: List[Path] = []

    if args.latex:
        tex = aggregate_to_latex_tabular(
            agg_rows,
            caption=f"Aggregate results ({suite_name})",
            label=f"tab:summary_{suite_name}",
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        latex_tex.write_text(tex, encoding="utf-8")
        written_misc.append(latex_tex)

    # baseline shift plots (preserved)
    if suite_name == "shift":
        for t in (suite.get("tasks", []) or []):
            if not _enabled(t, True):
                continue
            task_id = t.get("task_id")
            if not task_id:
                continue
            t0 = _task_shift_t0(t)
            task_records = _filter_records(records, suite_name=suite_name, task_id=str(task_id))

            out_png = out_dir / f"shift_recovery_{task_id}.png"
            plot_shift_recovery_curves(
                task_id=str(task_id),
                records=task_records,
                out_path=out_png,
                t0=t0,
                metric="mse_t",
            )
            written_plots.append(out_png)

            if (
                str(args.severity_x_field) == "severity_key"
                and str(args.severity_y_field) == "mse_mean"
            ):
                out_png2 = out_dir / f"severity_sweep_{task_id}_R_scale.png"
            else:
                out_png2 = out_dir / (
                    f"severity_sweep_{task_id}_{args.severity_x_field}_vs_{args.severity_y_field}.png"
                )
            plot_severity_sweep(
                task_id=str(task_id),
                records=task_records,
                out_path=out_png2,
                severity_key="shift.post_shift.R_scale",
                x_field=str(args.severity_x_field),
                y_field=str(args.severity_y_field),
            )
            written_plots.append(out_png2)

    # S6 views (additive; flag-gated)
    if args.plan_views:
        plan_rows, plan_fields = build_plan_comparison_rows(records)
        write_rows_csv(plan_rows, plan_compare_csv, plan_fields)
        written_tables.append(plan_compare_csv)

        fail_rows, fail_fields = summarize_failures_by_plan(records, group_by=args.group_by)
        write_rows_csv(fail_rows, failure_by_plan_csv, fail_fields)
        written_tables.append(failure_by_plan_csv)

        for task_id in _suite_task_ids(suite, records):
            task_records = _filter_records(records, suite_name=suite_name, task_id=str(task_id))
            tmp_path = out_dir / f"track_compare_{task_id}_metric.png"
            metric_key = plot_track_comparison(task_id=str(task_id), records=task_records, out_path=tmp_path)
            if metric_key and tmp_path.exists():
                final_path = out_dir / f"track_compare_{task_id}_{metric_key}.png"
                tmp_path.rename(final_path)
                written_plots.append(final_path)

            mse_model_plot = out_dir / f"mse_db_by_model_{task_id}.png"
            if plot_mse_db_by_model(task_id=str(task_id), records=task_records, out_path=mse_model_plot):
                written_plots.append(mse_model_plot)

    if args.budget_curves:
        for task_id in _suite_task_ids(suite, records):
            task_records = _filter_records(records, suite_name=suite_name, task_id=str(task_id))
            tmp_path = out_dir / f"budget_curve_{task_id}_metric.png"
            metric_key = plot_budget_curve(task_id=str(task_id), records=task_records, out_path=tmp_path)
            if metric_key and tmp_path.exists():
                final_path = out_dir / f"budget_curve_{task_id}_{metric_key}.png"
                tmp_path.rename(final_path)
                written_plots.append(final_path)

    if args.include_ops:
        ops_rows, ops_fields = summarize_ops_by_plan(records, group_by=args.group_by)
        write_rows_csv(ops_rows, ops_by_plan_csv, ops_fields)
        written_tables.append(ops_by_plan_csv)
        ops_plot = out_dir / f"ops_tradeoff_{suite_name}.png"
        plot_ops_tradeoff(
            suite_name=suite_name,
            records=records,
            out_path=ops_plot,
        )
        written_plots.append(ops_plot)

    if args.fig5a_plot or str(suite_name).lower().startswith("fig5a"):
        fig5a_rows, fig5a_fields = _build_fig5a_point_rows(records)
        write_rows_csv(fig5a_rows, fig5a_points_csv, fig5a_fields)
        written_tables.append(fig5a_points_csv)
        print(f"[make_report] wrote: {fig5a_points_csv}")

        wrote_overlay = plot_fig5a_mse_vs_inv_r2(records=records, out_path=fig5a_overlay_path)
        wrote_legacy = plot_fig5a_mse_vs_inv_r2(records=records, out_path=fig5a_legacy_path)

        if wrote_overlay:
            written_plots.append(fig5a_overlay_path)
            print(f"[make_report] wrote: {fig5a_overlay_path}")
        else:
            print(f"[make_report] note: fig5a overlay skipped (insufficient inv_r2_db/mse_db data)")

        if wrote_legacy:
            written_plots.append(fig5a_legacy_path)
            print(f"[make_report] wrote: {fig5a_legacy_path}")

    if args.organize_by_suite:
        mirrored = _organize_outputs_by_suite(
            out_dir=out_dir,
            suite_name=suite_name,
            day_stamp=day_stamp,
            time_stamp=time_stamp,
            table_files=written_tables,
            plot_files=written_plots,
            misc_files=written_misc,
        )
        print(
            "[make_report] organized outputs:",
            f"{out_dir / suite_name / day_stamp / time_stamp} "
            f"(tables={len(mirrored['tables'])}, plots={len(mirrored['plots'])}, misc={len(mirrored['misc'])})",
        )

    print(f"[make_report] done. wrote: {summary_csv} and {aggregate_csv}")
    if args.plan_views:
        print(f"[make_report] wrote: {plan_compare_csv} and {failure_by_plan_csv}")
    if args.include_ops:
        print(f"[make_report] wrote: {ops_by_plan_csv}")
        print(f"[make_report] wrote: {out_dir / f'ops_tradeoff_{suite_name}.png'}")
    if args.latex:
        print(f"[make_report] wrote: {latex_tex}")


if __name__ == "__main__":
    main()
