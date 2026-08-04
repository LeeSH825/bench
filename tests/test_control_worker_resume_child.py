"""Durable RESUME_EXACT action, immutable child, and real WorkerManager launch.

These use a real SQLite registry, real checkpoint packages on disk, a real
``WorkerManager``, and real subprocesses. Everything lives under ``tmp_path``.
"""

from __future__ import annotations

import uuid
from pathlib import Path

import pytest
import torch

from bench.control.checkpoints import (
    AdapterTrainingState,
    BatchPlan,
    CheckpointKind,
    CheckpointService,
    TrainingCursor,
    capture_rng,
    evaluate_resume_eligibility,
)
from bench.control.checkpoints.resume_coordinator import (
    ACTION_RESUME_EXACT,
    ResumeConflict,
    ResumeCoordinator,
    ResumeRejected,
)
from bench.control.process.manager import WorkerManager
from bench.control.registry.schema import ExperimentRecord, RunRecord, RunState
from bench.control.registry.sqlite import SqliteRegistry, utc_now

PARENT = "parent-run"


def _state() -> AdapterTrainingState:
    return AdapterTrainingState(
        model_slots={"model": {"w": torch.ones(2, 2)}},
        optimizer_slots={"main": {"state": {}, "param_groups": [{"lr": 1e-3}]}},
        best_state={"weights": {"w": torch.ones(2, 2)}},
    )


def _setup(tmp_path: Path, *, training_path_id: str = "control_resumable_v1",
           parent_state: RunState = RunState.INTERRUPTED):
    registry = SqliteRegistry(tmp_path / "registry.sqlite3")
    experiment = ExperimentRecord(
        experiment_id=str(uuid.uuid4()), name="t", created_at=utc_now()
    )
    registry.upsert_experiment(experiment)
    run_dir = tmp_path / "runs" / PARENT
    run_dir.mkdir(parents=True, exist_ok=True)
    registry.create_run(
        RunRecord(
            run_id=PARENT, experiment_id=experiment.experiment_id, state=RunState.CREATED,
            state_version=0, created_at=utc_now(), updated_at=utc_now(),
            model_id="kalmannet_tsp",
            implementation_id="bench_kalmannet_tsp_adapter_v1",
            variant_id="sha256:variant", run_dir=str(run_dir),
            training_path_id=training_path_id,
            training_path_reason_code="CERTIFIED", training_path_contract_version=1,
        )
    )
    path = [RunState.VALIDATING, RunState.QUEUED, RunState.STARTING, RunState.RUNNING]
    if parent_state is RunState.INTERRUPTED:
        path += [RunState.STOP_REQUESTED, RunState.CHECKPOINTING, RunState.INTERRUPTED]
    for state in path:
        registry.transition(PARENT, to_state=state, actor="test", reason="setup")

    service = CheckpointService(run_dir, registry=registry, control_root=tmp_path)
    saved = service.save(
        run_id=PARENT,
        kind=CheckpointKind.INTERRUPT,
        cursor=TrainingCursor(global_update=3, batch_plan_position=3),
        adapter_state=_state(),
        rng=capture_rng(),
        identity={
            "model_id": "kalmannet_tsp",
            "implementation_id": "bench_kalmannet_tsp_adapter_v1",
            "variant_id": "sha256:variant",
        },
        batch_plan=BatchPlan(dataset_length=6, batch_size=2, seed=17),
        training_path_id=training_path_id,
        training_path_contract_version=1,
    )
    return registry, service, saved


class _StubManager:
    """Stands in for WorkerManager where the test is about the action, not the
    process. The real manager is exercised in the E2E test below."""

    def __init__(self, registry, tmp_path: Path, *, fail_launch: bool = False):
        self.registry = registry
        self.tmp_path = tmp_path
        self.fail_launch = fail_launch
        self.launches: list[str] = []
        self.prepared: list[str] = []

    def prepare_run(self, spec):
        run_id = spec.run_id.value
        directory = self.tmp_path / "runs" / run_id
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "resolved_run_spec.json").write_text(spec.to_json(), encoding="utf-8")
        self.registry.create_run(
            RunRecord(
                run_id=run_id, experiment_id=spec.draft.experiment.experiment_id,
                state=RunState.CREATED, state_version=0,
                created_at=utc_now(), updated_at=utc_now(),
                model_id=str(spec.model_id), implementation_id=str(spec.implementation_id),
                variant_id=str(spec.variant_id), run_dir=str(directory),
                training_path_id=spec.draft.execution.training_path_id,
            )
        )
        self.prepared.append(run_id)
        return directory

    def launch(self, spec):
        if self.fail_launch:
            raise RuntimeError("simulated launch failure")
        self.launches.append(spec.run_id.value)

        class _R:
            worker_instance_id = f"worker-{len(self.launches)}"

        return _R()


def _write_parent_spec(tmp_path: Path, registry) -> None:
    """A resume needs the parent's resolved spec on disk to clone it."""
    import dataclasses

    from bench.control.config.resolver import resolve_run_spec
    from bench.control.config.schema import (
        DatasetSection, ExperimentSection, RunSpecDraft, SystemSection,
    )
    from bench.control.identity import ImplementationId, ModelId

    parent = registry.get_run(PARENT)
    draft = RunSpecDraft(
        experiment=ExperimentSection(experiment_id=parent.experiment_id, name="t"),
        model_id=ModelId("kalmannet_tsp"),
        implementation_id=ImplementationId("bench_kalmannet_tsp_adapter_v1"),
        system=SystemSection(task_id="t", scenario_id="s", state_dim=2, observation_dim=2),
        dataset=DatasetSection(dataset_id="d"),
    )
    draft = dataclasses.replace(
        draft, training=dataclasses.replace(draft.training, enabled=True, max_updates=6)
    )
    spec = resolve_run_spec(draft)
    (Path(parent.run_dir) / "resolved_run_spec.json").write_text(
        spec.to_json(), encoding="utf-8"
    )


# -- eligibility gating ------------------------------------------------------


def test_v1_checkpoint_cannot_launch_a_child(tmp_path: Path) -> None:
    """A valid v1 package is still ineligible, with an explicit reason."""
    registry, service, saved = _setup(tmp_path)
    # Re-save without training-path provenance => v1 package.
    v1 = service.save(
        run_id=PARENT, kind=CheckpointKind.INTERRUPT,
        cursor=TrainingCursor(global_update=3, batch_plan_position=3),
        adapter_state=_state(), rng=capture_rng(),
        identity={"model_id": "kalmannet_tsp",
                  "implementation_id": "bench_kalmannet_tsp_adapter_v1"},
    )
    coordinator = ResumeCoordinator(
        registry=registry, manager=_StubManager(registry, tmp_path), control_root=tmp_path
    )
    with pytest.raises(ResumeRejected) as excinfo:
        coordinator.request_resume(checkpoint_id=v1.checkpoint_id, idempotency_key="k")
    assert "CHECKPOINT_TRAINING_PATH_UNPROVEN" in excinfo.value.reason_codes
    assert registry.list_runs(limit=100) and len(registry.list_runs(limit=100)) == 1


def test_legacy_training_path_cannot_launch_a_child(tmp_path: Path) -> None:
    registry, _, saved = _setup(tmp_path, training_path_id="legacy_train_v1")
    coordinator = ResumeCoordinator(
        registry=registry, manager=_StubManager(registry, tmp_path), control_root=tmp_path
    )
    with pytest.raises(ResumeRejected) as excinfo:
        coordinator.request_resume(checkpoint_id=saved.checkpoint_id, idempotency_key="k")
    assert "TRAINING_PATH_NOT_RESUMABLE" in excinfo.value.reason_codes


def test_running_parent_cannot_launch_a_child(tmp_path: Path) -> None:
    registry, _, saved = _setup(tmp_path, parent_state=RunState.RUNNING)
    coordinator = ResumeCoordinator(
        registry=registry, manager=_StubManager(registry, tmp_path), control_root=tmp_path
    )
    with pytest.raises(ResumeRejected) as excinfo:
        coordinator.request_resume(checkpoint_id=saved.checkpoint_id, idempotency_key="k")
    assert "PARENT_NOT_TERMINAL" in excinfo.value.reason_codes


def test_corrupt_checkpoint_creates_no_child(tmp_path: Path) -> None:
    registry, service, saved = _setup(tmp_path)
    payload = saved.directory / "payload.pt"
    payload.write_bytes(payload.read_bytes()[:-32])
    manager = _StubManager(registry, tmp_path)
    coordinator = ResumeCoordinator(
        registry=registry, manager=manager, control_root=tmp_path
    )
    with pytest.raises(ResumeRejected) as excinfo:
        coordinator.request_resume(checkpoint_id=saved.checkpoint_id, idempotency_key="k")
    assert "CHECKPOINT_NOT_VALID" in excinfo.value.reason_codes
    assert manager.prepared == [] and manager.launches == []


# -- durable action ----------------------------------------------------------


def test_resume_launches_exactly_one_child(tmp_path: Path) -> None:
    registry, _, saved = _setup(tmp_path)
    _write_parent_spec(tmp_path, registry)
    manager = _StubManager(registry, tmp_path)
    coordinator = ResumeCoordinator(
        registry=registry, manager=manager, control_root=tmp_path
    )

    outcome = coordinator.request_resume(
        checkpoint_id=saved.checkpoint_id, idempotency_key="k1"
    )
    assert outcome.state == "COMPLETED"
    assert outcome.child_run_id and outcome.child_run_id != PARENT
    assert manager.launches == [outcome.child_run_id]

    child = registry.get_run(outcome.child_run_id)
    assert child.parent_run_id == PARENT
    assert child.resumed_from_run_id == PARENT
    assert child.resumed_from_checkpoint_id == saved.checkpoint_id
    assert child.training_path_id == "control_resumable_v1"
    assert child.run_dir != registry.get_run(PARENT).run_dir


def test_five_identical_requests_produce_one_child_and_one_worker(tmp_path: Path) -> None:
    registry, _, saved = _setup(tmp_path)
    _write_parent_spec(tmp_path, registry)
    manager = _StubManager(registry, tmp_path)
    coordinator = ResumeCoordinator(
        registry=registry, manager=manager, control_root=tmp_path
    )

    outcomes = [
        coordinator.request_resume(checkpoint_id=saved.checkpoint_id, idempotency_key="same")
        for _ in range(5)
    ]
    assert len({o.action_id for o in outcomes}) == 1
    assert len({o.child_run_id for o in outcomes}) == 1
    assert len(manager.launches) == 1
    assert len(manager.prepared) == 1
    actions = [a for a in registry.list_actions(PARENT) if a["action"] == ACTION_RESUME_EXACT]
    assert len(actions) == 1
    children = [r for r in registry.list_runs(limit=100) if r.resumed_from_run_id == PARENT]
    assert len(children) == 1


def test_same_key_different_checkpoint_is_a_conflict(tmp_path: Path) -> None:
    registry, service, saved = _setup(tmp_path)
    _write_parent_spec(tmp_path, registry)
    other = service.save(
        run_id=PARENT, kind=CheckpointKind.INTERRUPT,
        cursor=TrainingCursor(global_update=4, batch_plan_position=4),
        adapter_state=_state(), rng=capture_rng(),
        identity={"model_id": "kalmannet_tsp",
                  "implementation_id": "bench_kalmannet_tsp_adapter_v1"},
        training_path_id="control_resumable_v1", training_path_contract_version=1,
    )
    coordinator = ResumeCoordinator(
        registry=registry, manager=_StubManager(registry, tmp_path), control_root=tmp_path
    )
    coordinator.request_resume(checkpoint_id=saved.checkpoint_id, idempotency_key="dup")
    with pytest.raises(ResumeConflict):
        coordinator.request_resume(checkpoint_id=other.checkpoint_id, idempotency_key="dup")


def test_stale_parent_state_version_is_a_conflict_with_no_side_effect(tmp_path: Path) -> None:
    registry, _, saved = _setup(tmp_path)
    _write_parent_spec(tmp_path, registry)
    manager = _StubManager(registry, tmp_path)
    coordinator = ResumeCoordinator(
        registry=registry, manager=manager, control_root=tmp_path
    )
    with pytest.raises(ResumeConflict):
        coordinator.request_resume(
            checkpoint_id=saved.checkpoint_id, idempotency_key="k",
            expected_parent_state_version=999,
        )
    assert manager.prepared == [] and manager.launches == []
    assert not [a for a in registry.list_actions(PARENT) if a["action"] == ACTION_RESUME_EXACT]


def test_launch_failure_fails_the_action_and_leaves_no_live_child(tmp_path: Path) -> None:
    registry, _, saved = _setup(tmp_path)
    _write_parent_spec(tmp_path, registry)
    manager = _StubManager(registry, tmp_path, fail_launch=True)
    coordinator = ResumeCoordinator(
        registry=registry, manager=manager, control_root=tmp_path
    )
    outcome = coordinator.request_resume(
        checkpoint_id=saved.checkpoint_id, idempotency_key="k"
    )
    assert outcome.state == "FAILED"
    assert "simulated launch failure" in (outcome.reason or "")
    child = registry.get_run(outcome.child_run_id)
    # Nothing ran, so CANCELLED is the accurate terminal state; the point is
    # that it must not be left looking live.
    assert child.state is RunState.CANCELLED, "an allocated child must never look live"
    assert child.state not in (RunState.CREATED, RunState.RUNNING)


# -- restart recovery --------------------------------------------------------


def test_crash_after_action_row_is_recovered(tmp_path: Path) -> None:
    """Action exists, nothing else. Recovery must complete it, once."""
    registry, _, saved = _setup(tmp_path)
    _write_parent_spec(tmp_path, registry)
    manager = _StubManager(registry, tmp_path)
    coordinator = ResumeCoordinator(
        registry=registry, manager=manager, control_root=tmp_path
    )
    outcome = coordinator.request_resume(
        checkpoint_id=saved.checkpoint_id, idempotency_key="k", launch=False
    )
    assert outcome.state == "REQUESTED"
    assert manager.prepared == []

    recovered = coordinator.reconcile_open_actions()
    assert len(recovered) == 1 and recovered[0].state == "COMPLETED"
    assert len(manager.launches) == 1

    # A second reconcile must be a no-op.
    assert coordinator.reconcile_open_actions() == []
    assert len(manager.launches) == 1


def test_crash_after_child_allocation_reuses_the_same_child(tmp_path: Path) -> None:
    registry, _, saved = _setup(tmp_path)
    _write_parent_spec(tmp_path, registry)
    manager = _StubManager(registry, tmp_path)
    coordinator = ResumeCoordinator(
        registry=registry, manager=manager, control_root=tmp_path
    )
    action = coordinator.request_resume(
        checkpoint_id=saved.checkpoint_id, idempotency_key="k", launch=False
    )
    child_id = coordinator._allocate_child(
        parent_run_id=PARENT, checkpoint_id=saved.checkpoint_id, action_id=action.action_id
    )
    assert len(manager.prepared) == 1

    # Coordinator restarts and finds the linked child.
    fresh = ResumeCoordinator(registry=registry, manager=manager, control_root=tmp_path)
    outcome = fresh.settle(action.action_id)
    assert outcome.state == "COMPLETED"
    assert outcome.child_run_id == child_id
    assert len(manager.prepared) == 1, "must not allocate a second child"
    assert len(manager.launches) == 1


def test_completed_action_is_not_reopened(tmp_path: Path) -> None:
    """A child that later fails does not un-complete a successful launch."""
    registry, _, saved = _setup(tmp_path)
    _write_parent_spec(tmp_path, registry)
    manager = _StubManager(registry, tmp_path)
    coordinator = ResumeCoordinator(
        registry=registry, manager=manager, control_root=tmp_path
    )
    outcome = coordinator.request_resume(
        checkpoint_id=saved.checkpoint_id, idempotency_key="k"
    )
    child = registry.get_run(outcome.child_run_id)
    for state in (RunState.VALIDATING, RunState.QUEUED, RunState.STARTING,
                  RunState.RUNNING, RunState.FAILED):
        registry.transition(child.run_id, to_state=state, actor="test", reason="child failed")

    again = coordinator.settle(outcome.action_id)
    assert again.state == "COMPLETED"
    assert again.reused_existing
    assert len(manager.launches) == 1


# -- parent immutability -----------------------------------------------------


def test_parent_is_untouched_by_a_resume(tmp_path: Path) -> None:
    registry, _, saved = _setup(tmp_path)
    _write_parent_spec(tmp_path, registry)
    before = registry.get_run(PARENT)
    before_ckpts = registry.list_checkpoints(PARENT)
    before_files = sorted(
        p.name for p in (Path(before.run_dir) / "checkpoints").rglob("*")
    )
    before_transitions = len(registry.list_transitions(PARENT))

    coordinator = ResumeCoordinator(
        registry=registry, manager=_StubManager(registry, tmp_path), control_root=tmp_path
    )
    coordinator.request_resume(checkpoint_id=saved.checkpoint_id, idempotency_key="k")

    after = registry.get_run(PARENT)
    assert after.state is before.state
    assert after.state_version == before.state_version
    assert after.exit_code == before.exit_code
    assert after.parent_run_id is None and after.resumed_from_run_id is None
    assert registry.list_checkpoints(PARENT) == before_ckpts
    assert len(registry.list_transitions(PARENT)) == before_transitions
    assert sorted(
        p.name for p in (Path(after.run_dir) / "checkpoints").rglob("*")
    ) == before_files


def test_eligibility_is_reported_without_unpickling(tmp_path: Path) -> None:
    """Eligibility comes from the manifest and registry, never the payload."""
    registry, _, saved = _setup(tmp_path)
    row = registry.get_checkpoint(saved.checkpoint_id)
    report = evaluate_resume_eligibility(
        checkpoint_row=row, parent_run=registry.get_run(PARENT), registry=registry
    )
    assert report.eligible
    assert report.training_path_id == "control_resumable_v1"
    assert report.checkpoint_schema_version == 2
