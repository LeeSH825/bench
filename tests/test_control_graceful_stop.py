"""Graceful stop: state ordering, idempotency, API independence, lineage.

The invariant under test: a run reaches ``INTERRUPTED`` only after an interrupt
checkpoint is durable *and* validated. Every failure path must land somewhere
that does not claim resumable state exists.
"""

from __future__ import annotations

import json
import subprocess
import sys
import uuid
from pathlib import Path

import pytest

from bench.control.checkpoints import (
    BatchPlan,
    CheckpointKind,
    CheckpointService,
    TrainingCursor,
    capture_rng,
)
from bench.control.checkpoints.lifecycle import (
    EXIT_CHECKPOINT_FAILED,
    EXIT_INTERRUPTED,
    plan_resume,
    settle_graceful_stop,
)
from bench.control.checkpoints.stop import StopCoordinator
from bench.control.checkpoints.training import TrainingProgress, TrainingSchedule
from bench.control.registry.schema import ExperimentRecord, RunRecord, RunState
from bench.control.registry.sqlite import SqliteRegistry, utc_now
from tests.checkpoint_fixtures import build_adapter, fingerprint

REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL = "kalmannet_tsp"
SEED = 17


def _registry_with_run(tmp_path: Path, run_id: str = "run-1") -> SqliteRegistry:
    registry = SqliteRegistry(tmp_path / "registry.sqlite3")
    experiment = ExperimentRecord(experiment_id=str(uuid.uuid4()), name="t", created_at=utc_now())
    registry.upsert_experiment(experiment)
    registry.create_run(
        RunRecord(
            run_id=run_id,
            experiment_id=experiment.experiment_id,
            state=RunState.CREATED,
            state_version=0,
            created_at=utc_now(),
            updated_at=utc_now(),
            model_id=MODEL,
            implementation_id=f"{MODEL}_v1",
            variant_id="sha256:variant",
            run_dir=str(tmp_path / "run"),
        )
    )
    for state in (RunState.VALIDATING, RunState.QUEUED, RunState.STARTING, RunState.RUNNING):
        registry.transition(run_id, to_state=state, actor="test", reason="setup")
    return registry


def _stop_at(tmp_path: Path, model_id: str, interrupt_at: int, total: int = 6):
    """Train, trip the stop coordinator, settle the stop. Returns the pieces."""
    registry = _registry_with_run(tmp_path)
    service = CheckpointService(tmp_path / "run", registry=registry, control_root=tmp_path)
    plan = BatchPlan(dataset_length=6, batch_size=2, seed=SEED)

    action = registry.request_action(run_id="run-1", action="stop", idempotency_key="k1")
    coordinator = StopCoordinator(
        run_id="run-1", registry=registry, install_signal_handlers=False
    )

    adapter, _, _ = build_adapter(model_id, SEED)
    progress = TrainingProgress()
    result = adapter.resumable_train(
        plan=plan,
        schedule=TrainingSchedule(max_updates=total, eval_interval=2),
        progress=progress,
        stop_requested=lambda: progress.global_update >= interrupt_at and coordinator.poll(),
    )
    assert result.interrupted

    outcome = settle_graceful_stop(
        run_id="run-1",
        registry=registry,
        service=service,
        cursor=result.cursor,
        adapter=adapter,
        rng=capture_rng(),
        identity={"model_id": model_id, "implementation_id": f"{model_id}_v1"},
        action_id=coordinator.decision.action_id,
        batch_plan=plan,
        progress=result.progress.as_dict(),
    )
    return registry, service, outcome, action, result


# -- state ordering ----------------------------------------------------------


def test_stop_follows_the_required_state_ordering(tmp_path: Path) -> None:
    registry, service, outcome, _, _ = _stop_at(tmp_path, MODEL, 3)

    assert outcome.state is RunState.INTERRUPTED
    assert outcome.exit_code == EXIT_INTERRUPTED
    assert outcome.checkpoint_id

    states = [t["to_state"] for t in registry.list_transitions("run-1")]
    tail = states[states.index("RUNNING") :]
    assert tail == ["RUNNING", "STOP_REQUESTED", "CHECKPOINTING", "INTERRUPTED"]

    run = registry.get_run("run-1")
    assert run.state is RunState.INTERRUPTED
    assert run.exit_code == EXIT_INTERRUPTED


def test_interrupt_checkpoint_is_valid_before_terminal_state(tmp_path: Path) -> None:
    registry, service, outcome, _, result = _stop_at(tmp_path, MODEL, 3)

    report = service.validate(outcome.checkpoint_id)
    assert report.valid, report.errors
    row = registry.get_checkpoint(outcome.checkpoint_id)
    assert row["kind"] == "interrupt"
    assert row["validation_status"] == "VALID"
    assert row["global_step"] == 3

    # The checkpoint row exists at or before the INTERRUPTED transition.
    assert registry.get_run("run-1").latest_checkpoint_id == outcome.checkpoint_id


def test_checkpoint_write_failure_fails_the_run_instead_of_interrupting(tmp_path: Path) -> None:
    """A stop that cannot persist state must not look resumable."""
    registry = _registry_with_run(tmp_path)
    service = CheckpointService(tmp_path / "run", registry=registry, control_root=tmp_path)

    class _Boom(RuntimeError):
        pass

    def exploding_save(**_kwargs):
        raise _Boom("disk full")

    service.save = exploding_save  # type: ignore[assignment]

    adapter, _, _ = build_adapter(MODEL, SEED)
    adapter.resumable_train(
        plan=BatchPlan(dataset_length=6, batch_size=2, seed=SEED),
        schedule=TrainingSchedule(max_updates=2, eval_interval=2),
    )
    action = registry.request_action(run_id="run-1", action="stop", idempotency_key="k1")

    outcome = settle_graceful_stop(
        run_id="run-1",
        registry=registry,
        service=service,
        cursor=TrainingCursor(global_update=2, batch_plan_position=2),
        adapter=adapter,
        rng=capture_rng(),
        identity={"model_id": MODEL, "implementation_id": f"{MODEL}_v1"},
        action_id=action["action_id"],
    )

    assert outcome.state is RunState.FAILED
    assert outcome.exit_code == EXIT_CHECKPOINT_FAILED
    assert registry.get_run("run-1").state is RunState.FAILED
    assert registry.get_action(action["action_id"])["status"] == "FAILED"
    assert registry.list_checkpoints("run-1") == []


# -- idempotency -------------------------------------------------------------


def test_repeated_stop_requests_are_one_logical_action(tmp_path: Path) -> None:
    registry = _registry_with_run(tmp_path)
    first = registry.request_action(run_id="run-1", action="stop", idempotency_key="same")
    second = registry.request_action(run_id="run-1", action="stop", idempotency_key="same")
    third = registry.request_action(run_id="run-1", action="stop", idempotency_key="same")

    assert first["action_id"] == second["action_id"] == third["action_id"]
    assert len(registry.list_actions("run-1")) == 1


def test_repeated_stop_requests_produce_one_interrupt_checkpoint(tmp_path: Path) -> None:
    registry, service, outcome, _, _ = _stop_at(tmp_path, MODEL, 3)
    # Ask again after the run already settled.
    registry.request_action(run_id="run-1", action="stop", idempotency_key="k1")
    interrupts = [c for c in registry.list_checkpoints("run-1") if c["kind"] == "interrupt"]
    assert len(interrupts) == 1
    assert interrupts[0]["checkpoint_id"] == outcome.checkpoint_id


def test_completed_action_is_no_longer_open(tmp_path: Path) -> None:
    registry, _, outcome, action, _ = _stop_at(tmp_path, MODEL, 3)
    settled = registry.get_action(action["action_id"])
    assert settled["status"] == "COMPLETED"
    assert settled["acknowledged_at"] is not None
    assert settled["result_checkpoint_id"] == outcome.checkpoint_id
    assert registry.open_action("run-1") is None


# -- signal handler discipline ----------------------------------------------


def test_signal_path_only_sets_a_flag(tmp_path: Path) -> None:
    registry = _registry_with_run(tmp_path)
    coordinator = StopCoordinator(
        run_id="run-1", registry=registry, install_signal_handlers=False
    )
    assert coordinator.poll() is False
    coordinator.signal_stop()
    assert coordinator.poll() is True
    assert coordinator.decision.source == "signal"
    # No state was written by the flag itself.
    assert registry.get_run("run-1").state is RunState.RUNNING
    assert registry.list_checkpoints("run-1") == []


def test_registry_request_is_acknowledged_when_seen(tmp_path: Path) -> None:
    registry = _registry_with_run(tmp_path)
    action = registry.request_action(run_id="run-1", action="stop", idempotency_key="k")
    assert registry.get_action(action["action_id"])["status"] == "REQUESTED"

    coordinator = StopCoordinator(
        run_id="run-1", registry=registry, install_signal_handlers=False
    )
    assert coordinator.poll() is True
    assert registry.get_action(action["action_id"])["status"] == "ACKNOWLEDGED"
    assert coordinator.decision.action_id == action["action_id"]


# -- API independence --------------------------------------------------------

_CHILD_STOP = r"""
import json, sys
sys.path.insert(0, {repo!r})
from pathlib import Path
from bench.control.checkpoints import BatchPlan, CheckpointService, capture_rng
from bench.control.checkpoints.lifecycle import settle_graceful_stop
from bench.control.checkpoints.stop import StopCoordinator
from bench.control.checkpoints.training import TrainingProgress, TrainingSchedule
from bench.control.registry.sqlite import SqliteRegistry
from tests.checkpoint_fixtures import build_adapter

root = Path(sys.argv[1])
registry = SqliteRegistry(root / "registry.sqlite3")
service = CheckpointService(root / "run", registry=registry, control_root=root)
plan = BatchPlan(dataset_length=6, batch_size=2, seed=17)
coordinator = StopCoordinator(run_id="run-1", registry=registry, install_signal_handlers=False)

adapter, _, _ = build_adapter("kalmannet_tsp", 17)
progress = TrainingProgress()
result = adapter.resumable_train(
    plan=plan, schedule=TrainingSchedule(max_updates=6, eval_interval=2),
    progress=progress, stop_requested=coordinator)
outcome = settle_graceful_stop(
    run_id="run-1", registry=registry, service=service, cursor=result.cursor,
    adapter=adapter, rng=capture_rng(),
    identity={{"model_id": "kalmannet_tsp", "implementation_id": "kalmannet_tsp_v1"}},
    action_id=coordinator.decision.action_id, batch_plan=plan,
    progress=result.progress.as_dict())
print("__OUTCOME__" + json.dumps(outcome.as_dict()))
"""


def test_stop_completes_with_no_api_process_involved(tmp_path: Path) -> None:
    """The stop request is a registry row, so nothing needs to still be running.

    The request is written here and the worker runs in a separate process with
    no API, no server, and no live requester.
    """
    registry = _registry_with_run(tmp_path)
    registry.request_action(run_id="run-1", action="stop", idempotency_key="detached")
    registry.close()

    script = _CHILD_STOP.format(repo=str(REPO_ROOT))
    completed = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path)],
        cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=900,
    )
    assert completed.returncode == 0, f"{completed.stdout}\n{completed.stderr}"
    marker = [l for l in completed.stdout.splitlines() if l.startswith("__OUTCOME__")]
    assert marker, completed.stdout + completed.stderr
    outcome = json.loads(marker[0][len("__OUTCOME__"):])

    assert outcome["state"] == "INTERRUPTED"
    assert outcome["exit_code"] == EXIT_INTERRUPTED

    reopened = SqliteRegistry(tmp_path / "registry.sqlite3")
    assert reopened.get_run("run-1").state is RunState.INTERRUPTED
    assert reopened.open_action("run-1") is None
    interrupts = [c for c in reopened.list_checkpoints("run-1") if c["kind"] == "interrupt"]
    assert len(interrupts) == 1


# -- resume lineage ----------------------------------------------------------


def test_resume_plan_carries_full_lineage_and_inherits_variant(tmp_path: Path) -> None:
    registry, service, outcome, _, _ = _stop_at(tmp_path, MODEL, 3)

    plan = plan_resume(
        checkpoint_id=outcome.checkpoint_id, registry=registry, service=service
    )
    lineage = plan.lineage()
    assert lineage["parent_run_id"] == "run-1"
    assert lineage["resumed_from_run_id"] == "run-1"
    assert lineage["resumed_from_checkpoint_id"] == outcome.checkpoint_id
    assert plan.cursor.global_update == 3
    # Exact resume is execution lineage, not a new variant (ADR-CSR-004).
    assert plan.identity["model_id"] == MODEL


def test_resume_leaves_the_parent_run_untouched(tmp_path: Path) -> None:
    registry, service, outcome, _, _ = _stop_at(tmp_path, MODEL, 3)

    before_run = registry.get_run("run-1")
    before_ckpts = registry.list_checkpoints("run-1")
    before_files = sorted(p.name for p in (tmp_path / "run" / "checkpoints").rglob("*"))

    plan_resume(checkpoint_id=outcome.checkpoint_id, registry=registry, service=service)

    after_run = registry.get_run("run-1")
    assert after_run.state is before_run.state
    assert after_run.state_version == before_run.state_version
    assert after_run.exit_code == before_run.exit_code
    assert registry.list_checkpoints("run-1") == before_ckpts
    assert sorted(p.name for p in (tmp_path / "run" / "checkpoints").rglob("*")) == before_files


def test_resume_from_a_running_run_is_refused(tmp_path: Path) -> None:
    registry = _registry_with_run(tmp_path, run_id="run-1")
    service = CheckpointService(tmp_path / "run", registry=registry, control_root=tmp_path)
    adapter, _, _ = build_adapter(MODEL, SEED)
    adapter.resumable_train(
        plan=BatchPlan(dataset_length=6, batch_size=2, seed=SEED),
        schedule=TrainingSchedule(max_updates=2, eval_interval=2),
    )
    saved = service.save(
        run_id="run-1",
        kind=CheckpointKind.PERIODIC,
        cursor=TrainingCursor(global_update=2, batch_plan_position=2),
        adapter_state=adapter.capture_training_state(TrainingCursor(global_update=2)),
        rng=capture_rng(),
        identity={"model_id": MODEL, "implementation_id": f"{MODEL}_v1"},
    )
    with pytest.raises(ValueError, match="no longer executing"):
        plan_resume(checkpoint_id=saved.checkpoint_id, registry=registry, service=service)


def test_resumed_child_reaches_the_same_result_as_continuous(tmp_path: Path) -> None:
    """End to end: stop, plan a resume, restore, finish — bitwise identical."""
    from bench.control.checkpoints import restore_rng

    adapter, _, _ = build_adapter(MODEL, SEED)
    continuous = fingerprint(
        adapter,
        adapter.resumable_train(
            plan=BatchPlan(dataset_length=6, batch_size=2, seed=SEED),
            schedule=TrainingSchedule(max_updates=6, eval_interval=2),
        ),
    )

    registry, service, outcome, _, result = _stop_at(tmp_path, MODEL, 3)
    resume = plan_resume(
        checkpoint_id=outcome.checkpoint_id, registry=registry, service=service
    )
    _manifest, cursor, state, rng, payload = service.load(outcome.checkpoint_id)

    child, _, _ = build_adapter(MODEL, SEED + 999)
    child.restore_training_state(state)
    restore_rng(rng)
    child_result = child.resumable_train(
        plan=BatchPlan.from_dict(payload["batch_plan"]),
        schedule=TrainingSchedule(max_updates=6, eval_interval=2),
        progress=TrainingProgress.from_dict(state.extra_state["progress"]),
    )
    assert fingerprint(child, child_result) == continuous
    assert resume.cursor.global_update == 3
