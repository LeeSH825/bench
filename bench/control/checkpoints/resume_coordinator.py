"""Durable RESUME_EXACT action and immutable child launch (gates B1 + C).

The action row is the unit of durability, not the process. Every step writes
its progress to SQLite before doing the next irreversible thing, so a crash at
any point leaves a row the reconciler can adjudicate rather than an orphaned
child or a duplicate worker.

Ordering, and why:

    action row            -- so a retry finds the same logical request
    → ACKNOWLEDGED        -- coordinator ownership, before any allocation
    → child allocation    -- recorded on the action *before* launching
    → WorkerManager.launch
    → COMPLETED           -- only once a worker actually exists

`COMPLETED` means "exactly one child worker was launched", never "the child
finished training" (ADR-WC-016). A child that later fails is a child run
state, and does not retroactively fail a launch action that did succeed.
"""

from __future__ import annotations

import dataclasses
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from ..registry.schema import RunState
from ..registry.sqlite import (
    ACTION_ACKNOWLEDGED,
    ACTION_COMPLETED,
    ACTION_FAILED,
    ACTION_REQUESTED,
)
from .eligibility import evaluate_resume_eligibility
from .schema import CheckpointError, MANIFEST_FILENAME
from .validation import validate_package

logger = logging.getLogger(__name__)

#: The action type this module owns.
ACTION_RESUME_EXACT = "RESUME_EXACT"


class ResumeConflict(RuntimeError):
    """Stale or contradictory request. No side effect has been performed."""


class ResumeRejected(RuntimeError):
    """Eligibility failed. Carries machine-readable reason codes."""

    def __init__(self, message: str, reason_codes: Optional[list[str]] = None):
        super().__init__(message)
        self.reason_codes = list(reason_codes or [])


@dataclass
class ResumeOutcome:
    action_id: str
    state: str
    child_run_id: Optional[str] = None
    worker_instance_id: Optional[str] = None
    reason: Optional[str] = None
    reason_codes: list[str] = field(default_factory=list)
    reused_existing: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "action_id": self.action_id,
            "action_type": ACTION_RESUME_EXACT,
            "state": self.state,
            "child_run_id": self.child_run_id,
            "worker_instance_id": self.worker_instance_id,
            "reason": self.reason,
            "reason_codes": list(self.reason_codes),
            "reused_existing": bool(self.reused_existing),
        }


class ResumeCoordinator:
    """Owns the RESUME_EXACT lifecycle. Independent of any HTTP process."""

    def __init__(self, *, registry: Any, manager: Any, control_root: Optional[Path] = None):
        self.registry = registry
        self.manager = manager
        self.control_root = Path(control_root) if control_root is not None else None

    # -- public ---------------------------------------------------------------

    def request_resume(
        self,
        *,
        checkpoint_id: str,
        idempotency_key: str,
        expected_parent_state_version: Optional[int] = None,
        requested_by: str = "cli",
        launch: bool = True,
    ) -> ResumeOutcome:
        """Validate, record a durable action, allocate a child, and launch it."""
        row = self.registry.get_checkpoint(checkpoint_id)
        if row is None:
            raise ResumeRejected(f"unknown checkpoint {checkpoint_id!r}", ["UNKNOWN_CHECKPOINT"])
        parent_run_id = str(row["run_id"])
        parent = self.registry.get_run(parent_run_id)
        if parent is None:
            raise ResumeRejected(
                f"checkpoint references unknown run {parent_run_id!r}", ["UNKNOWN_PARENT"]
            )

        payload = {
            "checkpoint_id": checkpoint_id,
            "parent_run_id": parent_run_id,
            "expected_parent_state_version": expected_parent_state_version,
        }

        existing = self.registry.get_action_by_key(idempotency_key)
        if existing is not None:
            return self._resolve_existing(existing, payload)

        # Optimistic concurrency, before anything durable is written.
        if expected_parent_state_version is not None and int(
            expected_parent_state_version
        ) != int(parent.state_version):
            raise ResumeConflict(
                f"parent {parent_run_id} is at state_version {parent.state_version}, "
                f"request expected {expected_parent_state_version}"
            )

        self._assert_eligible(checkpoint_id=checkpoint_id, row=row, parent=parent)

        action = self.registry.request_action(
            run_id=parent_run_id,
            action=ACTION_RESUME_EXACT,
            idempotency_key=idempotency_key,
            requested_by=requested_by,
            parameters=payload,
            expected_state_version=expected_parent_state_version,
        )
        action_id = str(action["action_id"])

        if not launch:
            return ResumeOutcome(action_id=action_id, state=str(action["status"]))

        return self.settle(action_id)

    def settle(self, action_id: str) -> ResumeOutcome:
        """Drive an open action to COMPLETED or FAILED. Safe to re-run."""
        action = self.registry.get_action(action_id)
        if action is None:
            raise ResumeRejected(f"unknown action {action_id!r}", ["UNKNOWN_ACTION"])
        if str(action["status"]) in (ACTION_COMPLETED, ACTION_FAILED):
            return self._outcome_from(action, reused_existing=True)

        parameters = _params(action)
        checkpoint_id = str(parameters.get("checkpoint_id"))
        parent_run_id = str(action["run_id"])

        self.registry.acknowledge_action(action_id)

        # A child may already exist from a crashed earlier attempt. Adopt it
        # rather than allocating a second one.
        child_run_id = action.get("result_child_run_id") or self._find_existing_child(
            parent_run_id=parent_run_id, checkpoint_id=checkpoint_id, action_id=action_id
        )

        try:
            if child_run_id is None:
                child_run_id = self._allocate_child(
                    parent_run_id=parent_run_id, checkpoint_id=checkpoint_id, action_id=action_id
                )
            worker_id = self._launch_child(child_run_id=child_run_id, action_id=action_id)
        except Exception as exc:
            reason = f"{type(exc).__name__}: {exc}"
            self.registry.fail_action(action_id, reason=reason)
            if child_run_id:
                self._mark_child_failed(child_run_id, reason)
            return ResumeOutcome(
                action_id=action_id, state=ACTION_FAILED,
                child_run_id=child_run_id, reason=reason,
            )

        self.registry.complete_resume_action(
            action_id, child_run_id=child_run_id, worker_instance_id=worker_id
        )
        return ResumeOutcome(
            action_id=action_id, state=ACTION_COMPLETED,
            child_run_id=child_run_id, worker_instance_id=worker_id,
        )

    def reconcile_open_actions(self) -> list[ResumeOutcome]:
        """Restart recovery: finish or fail every open RESUME_EXACT action.

        Called on coordinator/CLI start. A worker launched before the crash is
        detected by its registry row and adopted rather than launched twice.
        """
        outcomes: list[ResumeOutcome] = []
        for action in self.registry.list_open_actions(action=ACTION_RESUME_EXACT):
            try:
                outcomes.append(self.settle(str(action["action_id"])))
            except Exception as exc:  # pragma: no cover - defensive
                self.registry.fail_action(
                    str(action["action_id"]), reason=f"{type(exc).__name__}: {exc}"
                )
        return outcomes

    # -- internals ------------------------------------------------------------

    def _resolve_existing(self, action: dict[str, Any], payload: dict[str, Any]) -> ResumeOutcome:
        """Same key: same action, or a conflict if the payload disagrees."""
        stored = _params(action)
        for key in ("checkpoint_id", "parent_run_id"):
            if str(stored.get(key)) != str(payload.get(key)):
                raise ResumeConflict(
                    f"idempotency key already used for a different request "
                    f"({key}: {stored.get(key)!r} != {payload.get(key)!r})"
                )
        if str(action["status"]) in (ACTION_REQUESTED, ACTION_ACKNOWLEDGED):
            return self.settle(str(action["action_id"]))
        return self._outcome_from(action, reused_existing=True)

    def _outcome_from(self, action: dict[str, Any], *, reused_existing: bool) -> ResumeOutcome:
        return ResumeOutcome(
            action_id=str(action["action_id"]),
            state=str(action["status"]),
            child_run_id=action.get("result_child_run_id"),
            worker_instance_id=action.get("result_worker_instance_id"),
            reason=action.get("failure_reason"),
            reused_existing=reused_existing,
        )

    def _assert_eligible(self, *, checkpoint_id: str, row: dict[str, Any], parent: Any) -> None:
        parent_dir = Path(str(parent.run_dir))
        report = validate_package(parent_dir / "checkpoints" / checkpoint_id)
        manifest = report.manifest
        eligibility = evaluate_resume_eligibility(
            manifest=manifest,
            checkpoint_row=row,
            parent_run=parent,
            validation_ok=report.valid,
            registry=self.registry,
        )
        if not eligibility.eligible:
            raise ResumeRejected(
                "checkpoint is not eligible to launch a resumed child: "
                + "; ".join(eligibility.messages),
                eligibility.reason_codes,
            )

    def _find_existing_child(
        self, *, parent_run_id: str, checkpoint_id: str, action_id: str
    ) -> Optional[str]:
        for candidate in self.registry.list_runs(limit=1000):
            if (
                candidate.resumed_from_run_id == parent_run_id
                and candidate.resumed_from_checkpoint_id == checkpoint_id
            ):
                return candidate.run_id
        return None

    def _allocate_child(
        self, *, parent_run_id: str, checkpoint_id: str, action_id: str
    ) -> str:
        """Create the immutable child run. Parent is never touched."""
        from ..config.resolver import resolved_from_json

        parent = self.registry.get_run(parent_run_id)
        parent_dir = Path(str(parent.run_dir))
        spec_path = parent_dir / "resolved_run_spec.json"
        if not spec_path.exists():
            raise CheckpointError(f"parent run spec not found at {spec_path}")
        parent_spec = resolved_from_json(spec_path.read_text(encoding="utf-8"))

        # Child inherits every structural field, including the training path
        # and variant, and differs only in identity/lineage (ADR-WC-007).
        child_spec = dataclasses.replace(
            parent_spec,
            run_id=type(parent_spec.run_id).new(),
        )
        location = self.manager.prepare_run(child_spec)
        child_run_id = child_spec.run_id.value

        self.registry.set_run_lineage(
            child_run_id,
            parent_run_id=parent_run_id,
            resumed_from_run_id=parent_run_id,
            resumed_from_checkpoint_id=checkpoint_id,
        )
        # Link the child to the action before launching, so a crash between
        # here and launch is recoverable without allocating a second child.
        self.registry.link_action_child(action_id, child_run_id=child_run_id)
        _ = location
        return child_run_id

    def _launch_child(self, *, child_run_id: str, action_id: str) -> Optional[str]:
        from ..config.resolver import resolved_from_json

        child = self.registry.get_run(child_run_id)
        if child is None:
            raise CheckpointError(f"child run {child_run_id} vanished before launch")

        # Already running from a pre-crash launch? Adopt, never relaunch.
        worker = self.registry.worker_for_run(child_run_id)
        if worker is not None and child.state in (
            RunState.STARTING, RunState.RUNNING, RunState.COMPLETED,
            RunState.FAILED, RunState.INTERRUPTED,
        ):
            return worker.worker_instance_id

        spec_path = Path(str(child.run_dir)) / "resolved_run_spec.json"
        child_spec = resolved_from_json(spec_path.read_text(encoding="utf-8"))
        result = self.manager.launch(child_spec)
        return getattr(result, "worker_instance_id", None)

    def _mark_child_failed(self, child_run_id: str, reason: str) -> None:
        """Never leave an allocated child looking live.

        A child that was allocated but never started is ``CANCELLED``, not
        ``FAILED``: nothing ran, so calling it a failure would misreport what
        happened (and ``CREATED -> FAILED`` is not a legal transition anyway).
        Once it has entered the running path, ``FAILED`` is correct.
        """
        child = self.registry.get_run(child_run_id)
        if child is None or child.state in (
            RunState.COMPLETED, RunState.FAILED, RunState.CANCELLED
        ):
            return
        never_started = child.state in (
            RunState.CREATED, RunState.VALIDATING, RunState.QUEUED
        )
        target = RunState.CANCELLED if never_started else RunState.FAILED
        try:
            self.registry.transition(
                child_run_id,
                to_state=target,
                actor="resume-coordinator",
                reason=f"resume launch failed: {reason}",
                fields={"exit_code": 51, "terminal_reason": "resume_launch_failed"},
            )
        except Exception:  # pragma: no cover - defensive
            logger.warning(
                "could not mark child %s terminal after launch failure",
                child_run_id, exc_info=True,
            )


def _params(action: dict[str, Any]) -> dict[str, Any]:
    raw = action.get("parameters_json") or "{}"
    try:
        return json.loads(raw)
    except (TypeError, json.JSONDecodeError):  # pragma: no cover
        return {}
