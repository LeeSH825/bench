"""Durable LAUNCH_RUN action.

Reuses the existing ``run_actions`` table and ``WorkerManager``; there is no
launch queue and no second database. The ordering mirrors the resume
coordinator, for the same reason: every step is durable before the next
irreversible one, so a crash leaves a row the reconciler can adjudicate rather
than an orphaned run or a duplicate worker.

    action row → ACKNOWLEDGED → run allocation (recorded on the action)
    → WorkerManager.launch → COMPLETED

`COMPLETED` means exactly one run was allocated and one worker launched — never
that the benchmark finished. That is the run's own state.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional

from .provenance import repository_provenance
from .registry.schema import RunState
from .registry.sqlite import ACTION_COMPLETED, ACTION_FAILED

logger = logging.getLogger(__name__)

ACTION_LAUNCH_RUN = "LAUNCH_RUN"

#: Files written into every GUI-launched run for provenance.
ORIGINAL_PRESET_FILE = "original_preset.yaml"
SUBMITTED_DRAFT_FILE = "submitted_draft.yaml"
VALIDATION_FILE = "config_validation.json"
LAUNCH_REQUEST_FILE = "launch_request.json"


class LaunchConflict(RuntimeError):
    """Stale or contradictory request. Nothing was allocated."""


class LaunchRejected(RuntimeError):
    """The config or envelope is not launchable. Carries reason codes."""

    def __init__(self, message: str, reason_codes: Optional[list[str]] = None):
        super().__init__(message)
        self.reason_codes = list(reason_codes or [])


@dataclass
class LaunchOutcome:
    action_id: str
    state: str
    run_id: Optional[str] = None
    worker_instance_id: Optional[str] = None
    reason: Optional[str] = None
    reason_codes: list[str] = field(default_factory=list)
    reused_existing: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "action_id": self.action_id, "action_type": ACTION_LAUNCH_RUN,
            "state": self.state, "run_id": self.run_id,
            "worker_instance_id": self.worker_instance_id,
            "reason": self.reason, "reason_codes": list(self.reason_codes),
            "idempotency_reused": bool(self.reused_existing),
        }


class LaunchCoordinator:
    """Owns the LAUNCH_RUN lifecycle. Independent of any HTTP process."""

    def __init__(self, *, registry: Any, manager: Any, control_root: Optional[Path] = None):
        self.registry = registry
        self.manager = manager
        self.control_root = Path(control_root) if control_root is not None else None

    def request_launch(
        self,
        *,
        preset_id: str,
        preset_digest: str,
        idempotency_key: str,
        task_id: Optional[str] = None,
        model_id: Optional[str] = None,
        init_id: Optional[str] = None,
        overrides: Optional[Mapping[str, Any]] = None,
        expected_structural_config_hash: Optional[str] = None,
        expected_operational_config_hash: Optional[str] = None,
        requested_by: str = "gui",
        launch: bool = True,
    ) -> LaunchOutcome:
        from .config.gui_service import parse_submitted_yaml, validate_config
        from .config.presets import PresetCatalog, PresetError

        payload = {
            "preset_id": preset_id, "preset_digest": preset_digest,
            "task_id": task_id, "model_id": model_id, "init_id": init_id,
            "overrides": dict(overrides or {}),
            "expected_structural_config_hash": expected_structural_config_hash,
        }

        existing = self.registry.get_action_by_key(idempotency_key)
        if existing is not None:
            return self._resolve_existing(existing, payload)

        catalog = PresetCatalog()
        try:
            entry, preset_text = catalog.get(preset_id)
        except PresetError as exc:
            raise LaunchRejected(str(exc), ["UNKNOWN_PRESET"]) from exc

        # Preset drift: the file changed between preview and launch.
        if preset_digest and entry.content_digest != preset_digest:
            raise LaunchConflict(
                f"preset {preset_id} changed since preview "
                f"(digest {entry.content_digest[:19]}… != {preset_digest[:19]}…); revalidate")

        # Capture repository provenance now: a GUI-launched run must carry the
        # same git/submodule/environment fingerprint the CLI records, otherwise
        # its results are not attributable to a repository state.
        result = validate_config(
            suite_document=parse_submitted_yaml(preset_text),
            task_id=task_id, model_id=model_id, overrides=overrides,
            init_id=init_id, registry=self.registry,
            provenance=repository_provenance(),
        )
        if not result.valid:
            raise LaunchRejected(
                "configuration is not valid: "
                + "; ".join(i.message for i in result.issues if i.severity == "error"),
                [i.code for i in result.issues if i.severity == "error"])

        if not result.launch_eligibility.get("eligible"):
            raise LaunchRejected(
                str(result.launch_eligibility.get("reason") or "not launchable"),
                [str(result.launch_eligibility.get("reason_code") or "NOT_ELIGIBLE")])

        # Preview/launch hash agreement: if the resolved config moved between
        # the preview the operator approved and now, they are not launching
        # what they reviewed.
        if (expected_structural_config_hash
                and expected_structural_config_hash != result.structural_config_hash):
            raise LaunchConflict(
                "resolved structural_config_hash changed since preview "
                f"({result.structural_config_hash} != {expected_structural_config_hash}); "
                "revalidate before launching")
        if (expected_operational_config_hash
                and expected_operational_config_hash != result.operational_config_hash):
            raise LaunchConflict(
                "resolved operational_config_hash changed since preview; revalidate")

        action = self.registry.request_action(
            run_id=None, action=ACTION_LAUNCH_RUN, idempotency_key=idempotency_key,
            requested_by=requested_by, parameters=payload,
        )
        action_id = str(action["action_id"])
        if not launch:
            return LaunchOutcome(action_id=action_id, state=str(action["status"]))
        return self.settle(action_id, preset_text=preset_text, validation=result)

    def settle(self, action_id: str, *, preset_text: Optional[str] = None,
               validation: Any = None) -> LaunchOutcome:
        """Drive an open launch action to COMPLETED or FAILED. Re-runnable."""
        action = self.registry.get_action(action_id)
        if action is None:
            raise LaunchRejected(f"unknown action {action_id!r}", ["UNKNOWN_ACTION"])
        if str(action["status"]) in (ACTION_COMPLETED, ACTION_FAILED):
            return self._outcome_from(action, reused_existing=True)

        params = _params(action)
        self.registry.acknowledge_action(action_id)

        run_id = action.get("result_child_run_id")
        try:
            if run_id is None:
                run_id = self._allocate(action_id, params, preset_text, validation)
            worker_id = self._launch(run_id)
        except Exception as exc:
            reason = f"{type(exc).__name__}: {exc}"
            self.registry.fail_action(action_id, reason=reason)
            if run_id:
                self._mark_not_live(run_id, reason)
            return LaunchOutcome(action_id=action_id, state=ACTION_FAILED,
                                 run_id=run_id, reason=reason)

        self.registry.complete_resume_action(
            action_id, child_run_id=run_id, worker_instance_id=worker_id)
        return LaunchOutcome(action_id=action_id, state=ACTION_COMPLETED,
                             run_id=run_id, worker_instance_id=worker_id)

    def reconcile_open_actions(self) -> list[LaunchOutcome]:
        """Restart recovery. Adopts an allocated run rather than making another."""
        outcomes: list[LaunchOutcome] = []
        for action in self.registry.list_open_actions(action=ACTION_LAUNCH_RUN):
            try:
                outcomes.append(self.settle(str(action["action_id"])))
            except Exception as exc:  # pragma: no cover - defensive
                self.registry.fail_action(str(action["action_id"]),
                                          reason=f"{type(exc).__name__}: {exc}")
        return outcomes

    # -- internals ----------------------------------------------------------

    def _resolve_existing(self, action: dict[str, Any],
                          payload: Mapping[str, Any]) -> LaunchOutcome:
        stored = _params(action)
        if str(action.get("action")) != ACTION_LAUNCH_RUN:
            raise LaunchConflict("this Idempotency-Key was used for a different action type")
        for key in ("preset_id", "task_id", "model_id"):
            if str(stored.get(key)) != str(payload.get(key)):
                raise LaunchConflict(
                    f"idempotency key already used for a different request "
                    f"({key}: {stored.get(key)!r} != {payload.get(key)!r})")
        if dict(stored.get("overrides") or {}) != dict(payload.get("overrides") or {}):
            raise LaunchConflict(
                "idempotency key already used with different configuration overrides")
        if str(action["status"]) in (ACTION_COMPLETED, ACTION_FAILED):
            return self._outcome_from(action, reused_existing=True)
        return self.settle(str(action["action_id"]))

    def _outcome_from(self, action: dict[str, Any], *, reused_existing: bool) -> LaunchOutcome:
        return LaunchOutcome(
            action_id=str(action["action_id"]), state=str(action["status"]),
            run_id=action.get("result_child_run_id"),
            worker_instance_id=action.get("result_worker_instance_id"),
            reason=action.get("failure_reason"), reused_existing=reused_existing)

    def _allocate(self, action_id: str, params: Mapping[str, Any],
                  preset_text: Optional[str], validation: Any) -> str:
        """Allocate the immutable run and write its provenance snapshots."""
        import yaml

        from .config.gui_service import parse_submitted_yaml, validate_config
        from .config.presets import PresetCatalog
        from .config.resolver import resolved_from_dict

        catalog = PresetCatalog()
        entry, text = catalog.get(str(params["preset_id"]))
        preset_text = preset_text or text
        if validation is None or not getattr(validation, "valid", False):
            validation = validate_config(
                suite_document=parse_submitted_yaml(preset_text),
                task_id=params.get("task_id"), model_id=params.get("model_id"),
                overrides=params.get("overrides"), init_id=params.get("init_id"),
                registry=self.registry, provenance=repository_provenance())
            if not validation.valid:
                raise LaunchRejected("configuration became invalid before allocation")

        spec = resolved_from_dict(validation.resolved_run_spec)
        location = self.manager.prepare_run(spec)
        run_id = spec.run_id.value

        # Provenance: what the operator started from, what they submitted, and
        # what it resolved to. The idempotency key is deliberately not stored.
        root = Path(location.root) if hasattr(location, "root") else Path(str(location))
        try:
            (root / ORIGINAL_PRESET_FILE).write_text(preset_text, encoding="utf-8")
            (root / SUBMITTED_DRAFT_FILE).write_text(
                validation.canonical_yaml or "", encoding="utf-8")
            (root / VALIDATION_FILE).write_text(
                json.dumps(validation.as_dict(), indent=2, default=str), encoding="utf-8")
            (root / LAUNCH_REQUEST_FILE).write_text(json.dumps({
                "launch_source": "gui",
                "preset_id": entry.preset_id,
                "preset_digest": entry.content_digest,
                "preset_relative_path": entry.relative_path,
                "task_id": params.get("task_id"), "model_id": params.get("model_id"),
                "init_id": params.get("init_id"),
                "overrides": dict(params.get("overrides") or {}),
                "structural_config_hash": validation.structural_config_hash,
                "operational_config_hash": validation.operational_config_hash,
                "variant_id": validation.variant_id,
                "training_path_id": validation.training_path_id,
                "action_id": action_id,
            }, indent=2, default=str), encoding="utf-8")
        except Exception:
            logger.warning("failed to write launch provenance snapshots", exc_info=True)

        self.registry.link_action_child(action_id, child_run_id=run_id)
        return run_id

    def _launch(self, run_id: str) -> Optional[str]:
        from .config.resolver import resolved_from_json

        record = self.registry.get_run(run_id)
        if record is None:
            raise RuntimeError(f"allocated run {run_id} vanished before launch")
        worker = self.registry.worker_for_run(run_id)
        if worker is not None and record.state in (
            RunState.STARTING, RunState.RUNNING, RunState.COMPLETED,
            RunState.FAILED, RunState.INTERRUPTED,
        ):
            return worker.worker_instance_id  # adopt a pre-crash launch
        spec_path = Path(str(record.run_dir)) / "resolved_run_spec.json"
        spec = resolved_from_json(spec_path.read_text(encoding="utf-8"))
        result = self.manager.launch(spec)
        return getattr(result, "worker_instance_id", None)

    def _mark_not_live(self, run_id: str, reason: str) -> None:
        """An allocated run that never started must not look live."""
        record = self.registry.get_run(run_id)
        if record is None or record.state in (
            RunState.COMPLETED, RunState.FAILED, RunState.CANCELLED
        ):
            return
        never_started = record.state in (
            RunState.CREATED, RunState.VALIDATING, RunState.QUEUED)
        target = RunState.CANCELLED if never_started else RunState.FAILED
        try:
            self.registry.transition(
                run_id, to_state=target, actor="launch-coordinator",
                reason=f"launch failed: {reason}",
                fields={"exit_code": 52, "terminal_reason": "launch_failed"})
        except Exception:  # pragma: no cover
            logger.warning("could not mark run %s terminal after launch failure",
                           run_id, exc_info=True)


def _params(action: Mapping[str, Any]) -> dict[str, Any]:
    try:
        return json.loads(action.get("parameters_json") or "{}")
    except (TypeError, json.JSONDecodeError):  # pragma: no cover
        return {}
