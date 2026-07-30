"""Read-only importer for the pre-existing `runs/` tree.

Hard guarantee: **this module never writes to, moves, or deletes anything under
the legacy run tree.** It opens files for reading and records what it found in
the control registry. That is the whole contract (design doc 03 §16, doc 05 §5.2).

What it produces is a *synthetic* run record with ``legacy = 1``:

* ``run_id`` is a **deterministic** UUIDv5 over the run's absolute path, so
  re-importing is idempotent instead of duplicating rows. (Control-plane runs use
  UUIDv7 — allocated, not derived. The two never collide, and the version digit
  makes which is which visible in the id itself.)
* status is decided **best-effort** with an explicit confidence level. A legacy
  directory has no state machine, so "COMPLETED" here means "the artifacts look
  like a finished run", not "a worker recorded a terminal transition".
* checkpoints found on disk are recorded but **never** marked
  ``exact_resume_certified``. A legacy `model.pt` holds weights and a couple of
  counters — loading it is a warm start, not a resume (DND-003).
"""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Mapping, Optional, Sequence

import yaml

from ..capabilities import implementation_id_for
from ..identity import ImplementationId, InitId, ModelId, compute_variant_id
from ..paths import legacy_runs_root
from ..registry.schema import ExperimentRecord, RunRecord, RunState
from ..registry.sqlite import SqliteRegistry, utc_now

#: Namespace for deterministic legacy run ids. Fixed forever: changing it would
#: re-identify every previously imported legacy run.
LEGACY_UUID_NAMESPACE = uuid.UUID("6f9c1f6a-6f2f-5f8e-9c3a-9e2f1b7d4a55")

#: Files that mark a directory as a legacy run.
MARKER_FILES = ("run_plan.json", "metrics.json", "meta.json", "failure.json")

#: Confidence vocabulary for a best-effort status decision.
CONFIDENCE_HIGH = "high"
CONFIDENCE_MEDIUM = "medium"
CONFIDENCE_LOW = "low"
CONFIDENCE_UNKNOWN = "unknown"


def legacy_path_hash(path: Path) -> str:
    """Stable hash of an absolute legacy path."""
    return hashlib.sha256(str(Path(path).resolve()).encode("utf-8")).hexdigest()


def legacy_run_id(path: Path) -> str:
    """Deterministic synthetic run id for a legacy directory."""
    return str(uuid.uuid5(LEGACY_UUID_NAMESPACE, str(Path(path).resolve())))


@dataclass(frozen=True)
class LegacyRunCandidate:
    """What could be recovered from one legacy run directory."""

    path: Path
    model_id: str
    task_id: str
    scenario_id: str
    track_id: str
    init_id: str
    suite_name: str
    seed: int
    state: RunState
    status_confidence: str
    status_reason: str
    created_at: Optional[str] = None
    metrics: Mapping[str, Any] = field(default_factory=dict)
    checkpoint_paths: tuple[str, ...] = ()
    has_viz_meta: bool = False
    error_summary: Optional[str] = None
    unknown_fields: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "model_id": self.model_id,
            "task_id": self.task_id,
            "scenario_id": self.scenario_id,
            "track_id": self.track_id,
            "init_id": self.init_id,
            "suite_name": self.suite_name,
            "seed": self.seed,
            "state": self.state.value,
            "status_confidence": self.status_confidence,
            "status_reason": self.status_reason,
            "created_at": self.created_at,
            "metrics": dict(self.metrics),
            "checkpoint_paths": list(self.checkpoint_paths),
            "has_viz_meta": self.has_viz_meta,
            "error_summary": self.error_summary,
            "unknown_fields": list(self.unknown_fields),
        }


@dataclass(frozen=True)
class LegacyImportReport:
    scanned: int
    imported: int
    skipped: int
    already_present: int
    errors: tuple[str, ...] = ()
    run_ids: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "scanned": self.scanned,
            "imported": self.imported,
            "skipped": self.skipped,
            "already_present": self.already_present,
            "errors": list(self.errors),
            "run_ids": list(self.run_ids),
        }


def _read_json(path: Path) -> Optional[dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            document = json.load(handle)
        return document if isinstance(document, dict) else None
    except Exception:
        return None


def _read_yaml(path: Path) -> Optional[dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            document = yaml.safe_load(handle)
        return document if isinstance(document, dict) else None
    except Exception:
        return None


def discover_legacy_runs(root: Optional[Path] = None, *, limit: Optional[int] = None) -> Iterator[Path]:
    """Yield legacy run directories under *root*.

    A directory qualifies if it contains any of :data:`MARKER_FILES`. Each
    directory is yielded **once**, however many markers it holds — a run with
    both `run_plan.json` and `metrics.json` is one run, not two. The
    ``_model_cache`` and quarantine trees are skipped: they are caches and
    disabled artifacts, not runs.
    """
    base = Path(root) if root is not None else legacy_runs_root()
    if not base.exists():
        return
    emitted: set[Path] = set()
    for marker in sorted(base.rglob("*")):
        if not marker.is_file() or marker.name not in MARKER_FILES:
            continue
        directory = marker.parent
        if directory in emitted:
            continue
        parts = set(directory.parts)
        if "_model_cache" in parts or any(part.startswith("_quarantine") for part in parts):
            continue
        emitted.add(directory)
        yield directory
        if limit is not None and len(emitted) >= limit:
            return


def _decide_status(
    *,
    metrics: Optional[Mapping[str, Any]],
    failure: Optional[Mapping[str, Any]],
    run_plan: Optional[Mapping[str, Any]],
    has_viz_meta: bool,
) -> tuple[RunState, str, str, Optional[str]]:
    """Best-effort status decision, with confidence and an explicit reason.

    Returns ``(state, confidence, reason, error_summary)``.
    """
    if failure is not None:
        message = str(failure.get("message") or failure.get("error") or "").strip()
        status = str(failure.get("status") or "failed")
        return (
            RunState.FAILED,
            CONFIDENCE_HIGH,
            f"failure.json present with status={status!r}",
            message[:2000] or status,
        )
    if metrics is not None:
        status = str(metrics.get("status") or "").lower()
        if status in ("ok", "pass", "success", "completed"):
            return (
                RunState.COMPLETED,
                CONFIDENCE_HIGH,
                f"metrics.json present with status={status!r}",
                None,
            )
        if status:
            return (
                RunState.FAILED,
                CONFIDENCE_MEDIUM,
                f"metrics.json present with non-ok status={status!r}",
                status,
            )
        return (
            RunState.COMPLETED,
            CONFIDENCE_MEDIUM,
            "metrics.json present but carries no explicit status field",
            None,
        )
    if has_viz_meta:
        return (
            RunState.COMPLETED,
            CONFIDENCE_MEDIUM,
            "visualization meta.json present but no metrics.json; the run produced "
            "artifacts but its completion was not recorded",
            None,
        )
    if run_plan is not None:
        return (
            RunState.ORPHANED,
            CONFIDENCE_LOW,
            "run_plan.json present but neither metrics.json nor failure.json; the run "
            "was planned and its outcome is unknown",
            None,
        )
    return (
        RunState.ORPHANED,
        CONFIDENCE_UNKNOWN,
        "no recognizable outcome artifact was found",
        None,
    )


def inspect_legacy_run(directory: Path) -> Optional[LegacyRunCandidate]:
    """Read one legacy run directory. Returns ``None`` if nothing is recoverable."""
    directory = Path(directory)
    run_plan = _read_json(directory / "run_plan.json")
    metrics = _read_json(directory / "metrics.json")
    failure = _read_json(directory / "failure.json")
    viz_meta = _read_json(directory / "meta.json")
    config_snapshot = _read_yaml(directory / "config_snapshot.yaml")

    sources: list[Mapping[str, Any]] = [
        source for source in (run_plan, metrics, viz_meta) if source is not None
    ]
    if not sources and config_snapshot is None:
        return None

    def pick(key: str, default: Any = None) -> Any:
        for source in sources:
            value = source.get(key)
            if value not in (None, ""):
                return value
        return default

    unknown: list[str] = []

    model_id = pick("model_id")
    if not model_id:
        unknown.append("model_id")
        model_id = "unknown"
    task_id = pick("task_id") or pick("task")
    if not task_id:
        unknown.append("task_id")
        task_id = "unknown"
    scenario_id = pick("scenario_id")
    if not scenario_id:
        unknown.append("scenario_id")
        scenario_id = "unknown"
    track_id = pick("track_id")
    if not track_id:
        unknown.append("track_id")
        track_id = "unknown"
    init_id = pick("init_id")
    if not init_id:
        unknown.append("init_id")
        init_id = "unknown"
    suite_name = pick("suite_name") or pick("suite")
    if not suite_name:
        unknown.append("suite_name")
        suite_name = "unknown"
    seed_value = pick("seed", 0)
    try:
        seed = int(seed_value)
    except (TypeError, ValueError):
        unknown.append("seed")
        seed = 0

    state, confidence, reason, error_summary = _decide_status(
        metrics=metrics, failure=failure, run_plan=run_plan, has_viz_meta=viz_meta is not None
    )

    checkpoints = tuple(
        str(path.relative_to(directory))
        for path in sorted((directory / "checkpoints").glob("*.pt"))
    ) if (directory / "checkpoints").is_dir() else ()

    created_at = None
    if viz_meta is not None:
        created_at = viz_meta.get("created_at")
    if not created_at:
        try:
            from datetime import datetime, timezone

            created_at = (
                datetime.fromtimestamp(directory.stat().st_mtime, tz=timezone.utc)
                .isoformat(timespec="milliseconds")
                .replace("+00:00", "Z")
            )
        except Exception:
            created_at = None

    accuracy = dict((metrics or {}).get("accuracy") or {})

    return LegacyRunCandidate(
        path=directory,
        model_id=str(model_id),
        task_id=str(task_id),
        scenario_id=str(scenario_id),
        track_id=str(track_id),
        init_id=str(init_id),
        suite_name=str(suite_name),
        seed=seed,
        state=state,
        status_confidence=confidence,
        status_reason=reason,
        created_at=created_at,
        metrics=accuracy,
        checkpoint_paths=checkpoints,
        has_viz_meta=viz_meta is not None,
        error_summary=error_summary,
        unknown_fields=tuple(unknown),
    )


def _legacy_experiment_id(suite_name: str) -> str:
    """One synthetic experiment per legacy suite, deterministic by name."""
    return str(uuid.uuid5(LEGACY_UUID_NAMESPACE, f"experiment:{suite_name}"))


def import_legacy_runs(
    registry: SqliteRegistry,
    *,
    root: Optional[Path] = None,
    limit: Optional[int] = None,
    directories: Optional[Sequence[Path]] = None,
) -> LegacyImportReport:
    """Import legacy runs into the registry as read-only records.

    Idempotent: importing the same tree twice updates nothing and creates no
    duplicates, because the synthetic run id is derived from the path.
    """
    scanned = 0
    imported = 0
    skipped = 0
    already = 0
    errors: list[str] = []
    run_ids: list[str] = []

    candidates_iter: Iterator[Path]
    if directories is not None:
        candidates_iter = iter(Path(item) for item in directories)
    else:
        candidates_iter = discover_legacy_runs(root, limit=limit)

    for directory in candidates_iter:
        scanned += 1
        try:
            candidate = inspect_legacy_run(directory)
        except Exception as exc:
            errors.append(f"{directory}: {type(exc).__name__}: {exc}")
            continue
        if candidate is None:
            skipped += 1
            continue

        run_id = legacy_run_id(candidate.path)
        path_hash = legacy_path_hash(candidate.path)
        if registry.get_run(run_id) is not None:
            already += 1
            run_ids.append(run_id)
            continue

        experiment_id = _legacy_experiment_id(candidate.suite_name)
        registry.upsert_experiment(
            ExperimentRecord(
                experiment_id=experiment_id,
                name=f"[legacy] {candidate.suite_name}",
                description=(
                    "Synthetic experiment grouping imported legacy runs. These records are "
                    "read-only projections of directories under runs/; the control plane "
                    "never modifies them."
                ),
                tags=("legacy", candidate.suite_name),
            )
        )

        implementation_id = implementation_id_for(candidate.model_id)
        try:
            model_identity = ModelId(candidate.model_id)
        except ValueError:
            model_identity = ModelId("unknown")
        init_mode = candidate.init_id if candidate.init_id in (
            "untrained",
            "trained",
            "pretrained",
            "loaded",
        ) else "unknown"
        variant_id = compute_variant_id(
            model_id=model_identity,
            implementation_id=ImplementationId(implementation_id),
            init=InitId(mode=init_mode),
            architecture_fingerprint=None,
            # Legacy runs have no canonical structural hash — the control plane
            # never computed one for them. Leaving it None (rather than inventing
            # one) keeps legacy variant ids from colliding with new-run ids.
            structural_config_hash=None,
        )

        created = candidate.created_at or utc_now()
        try:
            registry.create_run(
                RunRecord(
                    run_id=run_id,
                    experiment_id=experiment_id,
                    state=RunState.CREATED,
                    state_version=0,
                    created_at=created,
                    updated_at=created,
                    model_id=candidate.model_id,
                    implementation_id=implementation_id,
                    init_id=candidate.init_id,
                    variant_id=variant_id.value,
                    task_id=candidate.task_id,
                    scenario_id=candidate.scenario_id,
                    seed=candidate.seed,
                    run_dir=str(candidate.path),
                    legacy=True,
                    status_confidence=candidate.status_confidence,
                )
            )
            # Walk the imported record to its observed state through legal
            # transitions, so the transition log explains how it got there
            # rather than the row simply appearing in a terminal state.
            _apply_legacy_state(registry, run_id, candidate)
            registry.record_legacy_mapping(
                run_id=run_id,
                legacy_path=str(candidate.path),
                legacy_path_hash=path_hash,
                meta=candidate.as_dict(),
                status_confidence=candidate.status_confidence,
            )
            imported += 1
            run_ids.append(run_id)
        except Exception as exc:
            errors.append(f"{directory}: {type(exc).__name__}: {exc}")

    return LegacyImportReport(
        scanned=scanned,
        imported=imported,
        skipped=skipped,
        already_present=already,
        errors=tuple(errors),
        run_ids=tuple(run_ids),
    )


def _apply_legacy_state(
    registry: SqliteRegistry, run_id: str, candidate: LegacyRunCandidate
) -> None:
    """Drive an imported run to its observed state via legal transitions."""
    path_to_state = {
        RunState.COMPLETED: (RunState.VALIDATING, RunState.QUEUED, RunState.STARTING, RunState.RUNNING, RunState.COMPLETED),
        RunState.FAILED: (RunState.VALIDATING, RunState.QUEUED, RunState.STARTING, RunState.RUNNING, RunState.FAILED),
        RunState.ORPHANED: (RunState.VALIDATING, RunState.QUEUED, RunState.STARTING, RunState.ORPHANED),
    }
    sequence = path_to_state.get(candidate.state, ())
    record = registry.get_run(run_id)
    if record is None:
        return
    for index, state in enumerate(sequence):
        is_last = index == len(sequence) - 1
        record = registry.transition(
            run_id,
            to_state=state,
            expected_state_version=record.state_version,
            actor="legacy-importer",
            reason=(candidate.status_reason if is_last else "legacy import reconstruction"),
            fields=(
                {
                    "terminal_reason": "legacy_import",
                    "error_summary": candidate.error_summary,
                    "status_confidence": candidate.status_confidence,
                }
                if is_last
                else None
            ),
        )
