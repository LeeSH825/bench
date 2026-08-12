"""Control-plane command line interface.

    python -m bench.control.cli launch --suite bench/configs/x.yaml --model kalmannet_tsp ...
    python -m bench.control.cli launch-synthetic --updates 50
    python -m bench.control.cli list
    python -m bench.control.cli show <run_id>
    python -m bench.control.cli import-legacy
    python -m bench.control.cli reconcile

CLI launching is always available. The HTTP API and dashboard are read-only by
default; explicit local write mode enables their guarded actions and the New
Run page. CLI and UI launches share the same
:class:`~bench.control.process.manager.WorkerManager` path.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional

import yaml

from .config.compatibility import draft_from_suite
from .config.resolver import resolve_run_spec
from .config.schema import (
    DatasetSection,
    ExperimentSection,
    ProvenanceSection,
    RunSpecDraft,
    RuntimeSection,
    SystemSection,
    TelemetrySection,
    TrainingSection,
)
from .capabilities import implementation_id_for
from .identity import ExperimentId, ImplementationId, ModelId
from .legacy import import_legacy_runs
from .paths import control_root
from .process.manager import WorkerManager
from .provenance import repository_provenance
from .registry.sqlite import open_registry


def _registry(args: argparse.Namespace):
    return open_registry(args.control_root)


def _manager(args: argparse.Namespace) -> WorkerManager:
    return WorkerManager(_registry(args), control_root_path=args.control_root)


def _find(items: Any, key: str, value: str) -> Optional[dict[str, Any]]:
    for item in items or []:
        if str(item.get(key)) == str(value):
            return dict(item)
    return None


def cmd_launch(args: argparse.Namespace) -> int:
    """Launch one suite entry as a supervised control-plane run."""
    suite_path = Path(args.suite).expanduser().resolve()
    suite = yaml.safe_load(suite_path.read_text(encoding="utf-8"))

    tasks = suite.get("tasks") or []
    models = suite.get("models") or []
    task = _find(tasks, "task_id", args.task) if args.task else (tasks[0] if tasks else None)
    model = _find(models, "model_id", args.model) if args.model else (models[0] if models else None)
    if task is None:
        print(f"error: task_id {args.task!r} not found in {suite_path}", file=sys.stderr)
        return 2
    if model is None:
        print(f"error: model_id {args.model!r} not found in {suite_path}", file=sys.stderr)
        return 2

    draft = draft_from_suite(
        suite,
        task=task,
        model=model,
        seed=int(args.seed),
        track_id=str(args.track),
        init_id=str(args.init),
        experiment_name=args.experiment_name,
        device=args.device,
        precision=args.precision,
        provenance=repository_provenance(),
        telemetry=TelemetrySection(enabled=not args.no_telemetry, interval_seconds=args.telemetry_interval),
    )
    draft = _with_executor(draft, args.executor)
    spec = resolve_run_spec(draft)

    manager = _manager(args)
    location = manager.prepare_run(spec)
    if args.dry_run:
        print(json.dumps({"run_id": spec.run_id.value, "run_dir": str(location.root), "launched": False}, indent=2))
        return 0
    result = manager.launch(spec, location=location)
    print(json.dumps(result.as_dict(), indent=2))
    return 0


def cmd_launch_synthetic(args: argparse.Namespace) -> int:
    """Launch a tiny synthetic run — no dataset, no torch, seconds to finish.

    This is what the operator quickstart and the smoke tests use.
    """
    model_id = str(args.model)
    draft = RunSpecDraft(
        experiment=ExperimentSection(
            experiment_id=args.experiment_id or ExperimentId.new().value,
            name=args.experiment_name or "synthetic-demo",
            description="Synthetic control-plane demonstration run.",
            tags=("synthetic", "demo"),
        ),
        model_id=ModelId(model_id),
        implementation_id=ImplementationId(implementation_id_for(model_id)),
        system=SystemSection(task_id="synthetic_task", scenario_id="synthetic_scenario", state_dim=2, observation_dim=2),
        dataset=DatasetSection(dataset_id="synthetic_dataset", split_seed=int(args.seed)),
        training=TrainingSection(
            enabled=True,
            max_updates=int(args.updates),
            batch_size=4,
            validation_interval_updates=max(1, int(args.updates) // 5),
        ),
        runtime=RuntimeSection(device=args.device, seed=int(args.seed)),
        telemetry=TelemetrySection(enabled=not args.no_telemetry, interval_seconds=args.telemetry_interval),
        provenance=repository_provenance(),
        bench_context={"executor": "synthetic", "track_id": "frozen"},
    )
    spec = resolve_run_spec(draft)
    manager = _manager(args)
    location = manager.prepare_run(spec)
    extra = ["--fail-at-step", str(args.fail_at_step)] if args.fail_at_step else []
    if extra:
        # The failure-injection flag is a worker CLI argument, so it is threaded
        # through the environment rather than the manager's fixed argv.
        result = manager.launch(spec, location=location, extra_env={"BENCH_CONTROL_FAIL_AT_STEP": str(args.fail_at_step)})
    else:
        result = manager.launch(spec, location=location)
    print(json.dumps(result.as_dict(), indent=2))
    return 0


def _with_executor(draft: RunSpecDraft, executor: str) -> RunSpecDraft:
    from dataclasses import replace

    context = dict(draft.bench_context)
    context["executor"] = executor
    return replace(draft, bench_context=context)


def cmd_list(args: argparse.Namespace) -> int:
    registry = _registry(args)
    records = registry.list_runs(limit=args.limit, include_legacy=args.include_legacy, active_only=args.active)
    if args.json:
        print(json.dumps([record.as_dict() for record in records], indent=2))
        return 0
    if not records:
        print("(no runs)")
        return 0
    print(f"{'RUN ID':38} {'STATE':10} {'MODEL':22} {'STEP':>7}  {'LEGACY':6} TASK")
    for record in records:
        print(
            f"{record.run_id:38} {record.state.value:10} {record.model_id[:22]:22} "
            f"{record.global_step:7}  {'yes' if record.legacy else 'no':6} {record.task_id[:40]}"
        )
    return 0


def cmd_show(args: argparse.Namespace) -> int:
    registry = _registry(args)
    record = registry.get_run(args.run_id)
    if record is None:
        print(f"error: unknown run {args.run_id}", file=sys.stderr)
        return 2
    manager = _manager(args)
    document = {
        **record.as_dict(),
        "worker": manager.describe_worker(args.run_id),
        "transitions": registry.list_transitions(args.run_id),
    }
    print(json.dumps(document, indent=2, default=str))
    return 0


def cmd_import_legacy(args: argparse.Namespace) -> int:
    registry = _registry(args)
    report = import_legacy_runs(registry, root=Path(args.root) if args.root else None, limit=args.limit)
    print(json.dumps({k: v for k, v in report.as_dict().items() if k != "run_ids"}, indent=2))
    return 0


def cmd_reconcile(args: argparse.Namespace) -> int:
    manager = _manager(args)
    candidates = manager.find_orphan_candidates(heartbeat_timeout_seconds=args.heartbeat_timeout)
    if args.dry_run:
        print(json.dumps([candidate.as_dict() for candidate in candidates], indent=2))
        return 0
    marked = manager.reconcile(heartbeat_timeout_seconds=args.heartbeat_timeout)
    print(json.dumps({"candidates": len(candidates), "marked_orphaned": [c.run_id for c in marked]}, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="bench.control.cli", description=__doc__)
    parser.add_argument("--control-root", default=None, help="Control-plane root (default: $BENCH_CONTROL_ROOT or <repo>/control)")
    sub = parser.add_subparsers(dest="command", required=True)

    launch = sub.add_parser("launch", help="Launch a suite entry as a supervised run")
    launch.add_argument("--suite", required=True)
    launch.add_argument("--task", default=None)
    launch.add_argument("--model", default=None)
    launch.add_argument("--seed", type=int, default=0)
    launch.add_argument("--track", default="frozen")
    launch.add_argument("--init", default="trained", choices=["untrained", "trained", "pretrained", "loaded"])
    launch.add_argument("--device", default=None)
    launch.add_argument("--precision", default=None)
    launch.add_argument("--experiment-name", default=None)
    launch.add_argument("--executor", default="suite", choices=["suite", "synthetic"])
    launch.add_argument("--telemetry-interval", type=float, default=2.0)
    launch.add_argument("--no-telemetry", action="store_true")
    launch.add_argument("--dry-run", action="store_true", help="Allocate and register but do not start a worker")
    launch.set_defaults(func=cmd_launch)

    synthetic = sub.add_parser("launch-synthetic", help="Launch a tiny synthetic demo run")
    synthetic.add_argument("--model", default="kalmannet_tsp")
    synthetic.add_argument("--updates", type=int, default=40)
    synthetic.add_argument("--seed", type=int, default=0)
    synthetic.add_argument("--device", default="cpu")
    synthetic.add_argument("--experiment-id", default=None)
    synthetic.add_argument("--experiment-name", default=None)
    synthetic.add_argument("--telemetry-interval", type=float, default=1.0)
    synthetic.add_argument("--no-telemetry", action="store_true")
    synthetic.add_argument("--fail-at-step", type=int, default=None, help="Inject an ordinary failure at this step")
    synthetic.set_defaults(func=cmd_launch_synthetic)

    listing = sub.add_parser("list", help="List runs")
    listing.add_argument("--limit", type=int, default=50)
    listing.add_argument("--include-legacy", action="store_true", default=True)
    listing.add_argument("--no-include-legacy", dest="include_legacy", action="store_false")
    listing.add_argument("--active", action="store_true")
    listing.add_argument("--json", action="store_true")
    listing.set_defaults(func=cmd_list)

    show = sub.add_parser("show", help="Show one run in detail")
    show.add_argument("run_id")
    show.set_defaults(func=cmd_show)

    importer = sub.add_parser("import-legacy", help="Import existing runs/ directories read-only")
    importer.add_argument("--root", default=None)
    importer.add_argument("--limit", type=int, default=None)
    importer.set_defaults(func=cmd_import_legacy)

    reconcile = sub.add_parser("reconcile", help="Detect and classify vanished workers")
    reconcile.add_argument("--heartbeat-timeout", type=float, default=90.0)
    reconcile.add_argument("--dry-run", action="store_true")
    reconcile.set_defaults(func=cmd_reconcile)

    # -- checkpoint / stop / resume ------------------------------------------
    # These commands remain available directly in the CLI. The API/dashboard
    # expose guarded counterparts only when explicit local write mode is on.
    checkpoints = sub.add_parser("checkpoints", help="Inspect the checkpoint catalog")
    checkpoint_sub = checkpoints.add_subparsers(dest="checkpoint_command", required=True)

    ck_list = checkpoint_sub.add_parser("list", help="List checkpoints for a run")
    ck_list.add_argument("--run-id", required=True)
    ck_list.set_defaults(func=cmd_checkpoints_list)

    ck_show = checkpoint_sub.add_parser("show", help="Show one checkpoint manifest")
    ck_show.add_argument("--checkpoint-id", required=True)
    ck_show.set_defaults(func=cmd_checkpoints_show)

    ck_validate = checkpoint_sub.add_parser(
        "validate", help="Verify a checkpoint's digest and structure"
    )
    ck_validate.add_argument("--checkpoint-id", required=True)
    ck_validate.set_defaults(func=cmd_checkpoints_validate)

    ck_reconcile = checkpoint_sub.add_parser(
        "reconcile", help="Reconcile on-disk checkpoint packages against the catalog"
    )
    ck_reconcile.add_argument("--run-id", required=True)
    ck_reconcile.set_defaults(func=cmd_checkpoints_reconcile)

    stop = sub.add_parser("stop", help="Request a graceful stop (persistent, idempotent)")
    stop.add_argument("--run-id", required=True)
    stop.add_argument(
        "--idempotency-key",
        default=None,
        help="Reusing a key makes a retry the same logical request; defaults to a new UUID",
    )
    stop.set_defaults(func=cmd_stop)

    resume = sub.add_parser("resume", help="Validate and describe a resume from a checkpoint")
    resume.add_argument("--checkpoint-id", required=True)
    # Plan-only stays the default: `resume` never had a side effect and must
    # not silently acquire one (continuation §6.5).
    resume.add_argument(
        "--launch",
        action="store_true",
        help="Actually allocate an immutable child run and launch its worker",
    )
    resume.add_argument(
        "--idempotency-key",
        default=None,
        help="Required with --launch; reusing a key makes a retry the same logical request",
    )
    resume.add_argument("--expected-parent-state-version", type=int, default=None)
    resume.set_defaults(func=cmd_resume)

    reconcile_actions = sub.add_parser(
        "reconcile-actions", help="Finish or fail open resume actions after a restart"
    )
    reconcile_actions.set_defaults(func=cmd_reconcile_actions)

    return parser


def _checkpoint_service(args: argparse.Namespace, run_record: Any):
    from .checkpoints import CheckpointService
    from .paths import control_root

    return CheckpointService(
        Path(run_record.run_dir),
        registry=_registry(args),
        control_root=control_root(getattr(args, "control_root", None)),
    )


def cmd_checkpoints_list(args: argparse.Namespace) -> int:
    registry = _registry(args)
    rows = registry.list_checkpoints(args.run_id)
    print(json.dumps({"run_id": args.run_id, "count": len(rows), "checkpoints": rows},
                     indent=2, default=str))
    return 0


def cmd_checkpoints_show(args: argparse.Namespace) -> int:
    registry = _registry(args)
    row = registry.get_checkpoint(args.checkpoint_id)
    if row is None:
        print(f"error: unknown checkpoint {args.checkpoint_id}", file=sys.stderr)
        return 2
    document = dict(row)
    manifest_json = document.pop("manifest_json", "") or ""
    if manifest_json:
        try:
            document["manifest"] = json.loads(manifest_json)
        except json.JSONDecodeError:
            document["manifest"] = None
    print(json.dumps(document, indent=2, default=str))
    return 0


def cmd_checkpoints_validate(args: argparse.Namespace) -> int:
    registry = _registry(args)
    row = registry.get_checkpoint(args.checkpoint_id)
    if row is None:
        print(f"error: unknown checkpoint {args.checkpoint_id}", file=sys.stderr)
        return 2
    run = registry.get_run(str(row["run_id"]))
    if run is None:
        print(f"error: checkpoint references unknown run {row['run_id']}", file=sys.stderr)
        return 2
    service = _checkpoint_service(args, run)
    report = service.validate(args.checkpoint_id, expected_sha256=row.get("payload_sha256"))
    registry.set_checkpoint_validation(
        args.checkpoint_id, status=str(report.status), detail="; ".join(report.errors) or None
    )
    print(json.dumps(report.as_dict(), indent=2, default=str))
    return 0 if report.valid else 1


def cmd_checkpoints_reconcile(args: argparse.Namespace) -> int:
    from .checkpoints import reconcile_run_checkpoints

    registry = _registry(args)
    run = registry.get_run(args.run_id)
    if run is None:
        print(f"error: unknown run {args.run_id}", file=sys.stderr)
        return 2
    report = reconcile_run_checkpoints(
        run_id=args.run_id, run_dir=Path(run.run_dir), registry=registry
    )
    print(json.dumps(report.as_dict(), indent=2, default=str))
    return 0


def cmd_stop(args: argparse.Namespace) -> int:
    """Record a durable stop request.

    This does not signal anything and does not wait: the worker polls for its
    own outstanding action and honours it at the next safe boundary, which is
    why a stop survives this process exiting (ADR-CSR-011).
    """
    import uuid as _uuid

    registry = _registry(args)
    run = registry.get_run(args.run_id)
    if run is None:
        print(f"error: unknown run {args.run_id}", file=sys.stderr)
        return 2

    key = args.idempotency_key or str(_uuid.uuid4())
    action = registry.request_action(
        run_id=args.run_id, action="stop", idempotency_key=key, requested_by="cli"
    )
    print(json.dumps(
        {
            "action_id": action["action_id"],
            "run_id": action["run_id"],
            "status": action["status"],
            "idempotency_key": key,
            "note": (
                "Request persisted. The worker completes the stop at its next "
                "checkpoint-safe boundary; this command does not wait."
            ),
        },
        indent=2, default=str,
    ))
    return 0


def _resume_coordinator(args: argparse.Namespace):
    from .checkpoints.resume_coordinator import ResumeCoordinator
    from .paths import control_root

    return ResumeCoordinator(
        registry=_registry(args),
        manager=_manager(args),
        control_root=control_root(getattr(args, "control_root", None)),
    )


def cmd_reconcile_actions(args: argparse.Namespace) -> int:
    """Restart recovery for resume actions left open by a crash."""
    outcomes = _resume_coordinator(args).reconcile_open_actions()
    print(json.dumps(
        {"reconciled": len(outcomes), "actions": [o.as_dict() for o in outcomes]},
        indent=2, default=str,
    ))
    return 0


def cmd_resume(args: argparse.Namespace) -> int:
    """Validate a resume, and with --launch actually run the child.

    Without --launch this is plan-only and has no side effect, exactly as
    before. Launching is explicit because it allocates a run and starts a
    process.
    """
    from .checkpoints.lifecycle import plan_resume

    registry = _registry(args)
    row = registry.get_checkpoint(args.checkpoint_id)
    if row is None:
        print(f"error: unknown checkpoint {args.checkpoint_id}", file=sys.stderr)
        return 2
    run = registry.get_run(str(row["run_id"]))
    if run is None:
        print(f"error: checkpoint references unknown run {row['run_id']}", file=sys.stderr)
        return 2

    if getattr(args, "launch", False):
        from .checkpoints.resume_coordinator import ResumeConflict, ResumeRejected

        if not args.idempotency_key:
            print(
                json.dumps({
                    "launched": False,
                    "reason": "--idempotency-key is required with --launch so a retried "
                              "invocation cannot create a second child run",
                }, indent=2),
                file=sys.stderr,
            )
            return 2
        try:
            outcome = _resume_coordinator(args).request_resume(
                checkpoint_id=args.checkpoint_id,
                idempotency_key=args.idempotency_key,
                expected_parent_state_version=args.expected_parent_state_version,
            )
        except ResumeConflict as exc:
            print(json.dumps({"launched": False, "conflict": str(exc)}, indent=2),
                  file=sys.stderr)
            return 3
        except ResumeRejected as exc:
            print(json.dumps(
                {"launched": False, "reason": str(exc), "reason_codes": exc.reason_codes},
                indent=2), file=sys.stderr)
            return 4
        print(json.dumps({"launched": True, **outcome.as_dict()}, indent=2, default=str))
        return 0 if outcome.state != "FAILED" else 1

    service = _checkpoint_service(args, run)
    try:
        plan = plan_resume(
            checkpoint_id=args.checkpoint_id, registry=registry, service=service
        )
    except Exception as exc:
        print(json.dumps({"resumable": False, "reason": f"{type(exc).__name__}: {exc}"},
                         indent=2), file=sys.stderr)
        return 1

    print(json.dumps(
        {
            "resumable": True,
            "checkpoint_id": plan.checkpoint_id,
            "resume_boundary": plan.manifest.resume_boundary,
            "cursor": plan.cursor.as_dict(),
            "identity": plan.identity,
            "lineage": plan.lineage(),
            "certification": plan.manifest.certification,
        },
        indent=2, default=str,
    ))
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
