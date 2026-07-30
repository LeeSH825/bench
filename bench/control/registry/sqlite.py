"""SQLite implementation of the run registry.

Operational rules (design doc 03 §12.2):

* **WAL journal mode** so readers (API, dashboard) never block the writer
  (worker heartbeats, state transitions).
* **foreign keys on** — they are off by default in SQLite, per connection.
* **busy timeout** so concurrent writers retry instead of raising immediately.
* **short transactions** — every public method here opens and closes its own
  transaction. There is no long-lived read transaction, which is what would
  otherwise pin the WAL and starve the writer.
* **optimistic concurrency** via ``state_version``: a caller passes the version
  it read, and the UPDATE only applies if the row still has it.

Threading/process model: one :class:`SqliteRegistry` per process. A connection
is created per thread (``threading.local``), because a SQLite connection is not
safe to share across threads by default.
"""

from __future__ import annotations

import json
import os
import shutil
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping, Optional, Sequence

from ..identity import uuid7
from ..paths import ensure_dir, registry_path
from . import migrations as migration_module
from .schema import (
    REGISTRY_SCHEMA_VERSION,
    ArtifactRecord,
    ExperimentRecord,
    RunRecord,
    RunState,
    WorkerRecord,
    validate_transition,
)

#: Default seconds a writer waits for a lock before raising ``database is locked``.
DEFAULT_BUSY_TIMEOUT_SECONDS = 15.0


class RegistryError(RuntimeError):
    """Base class for registry failures."""


class ConcurrencyError(RegistryError):
    """Raised when an optimistic-concurrency guard rejects a write.

    The caller read ``state_version = N``, but the row has moved on. The correct
    response is to re-read and decide again — never to retry blindly.
    """


class SchemaVersionError(RegistryError):
    """Raised when the database schema is newer than this code understands."""


def utc_now() -> str:
    """Timestamp string used everywhere in the registry (RFC 3339, UTC, ms)."""
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def backup_database(path: Path) -> Optional[Path]:
    """Copy *path* aside before a migration. Returns the backup path.

    Required by design doc 05 §5.3 (migration rollback): keep the previous
    database file so a failed migration can be rolled back by restoring it.
    """
    if not path.exists():
        return None
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup = path.with_name(f"{path.name}.backup-{stamp}")
    shutil.copy2(path, backup)
    return backup


class SqliteRegistry:
    """Authoritative run-state store."""

    def __init__(
        self,
        path: str | os.PathLike[str],
        *,
        busy_timeout_seconds: float = DEFAULT_BUSY_TIMEOUT_SECONDS,
        migrate: bool = True,
    ):
        self.path = Path(path).expanduser().resolve()
        self.busy_timeout_seconds = float(busy_timeout_seconds)
        self._local = threading.local()
        ensure_dir(self.path.parent)
        if migrate:
            self.migrate()
        else:
            self._check_version()

    # -- connection handling -------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            str(self.path),
            timeout=self.busy_timeout_seconds,
            isolation_level=None,  # explicit transaction control
            check_same_thread=False,
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute(f"PRAGMA busy_timeout={int(self.busy_timeout_seconds * 1000)}")
        # NORMAL is the right durability/throughput tradeoff under WAL: a commit
        # survives process crash, and only a host power loss can lose the most
        # recent transactions. Heartbeats at 1 Hz do not warrant FULL.
        connection.execute("PRAGMA synchronous=NORMAL")
        return connection

    @property
    def connection(self) -> sqlite3.Connection:
        existing = getattr(self._local, "connection", None)
        if existing is None:
            existing = self._connect()
            self._local.connection = existing
        return existing

    def close(self) -> None:
        existing = getattr(self._local, "connection", None)
        if existing is not None:
            existing.close()
            self._local.connection = None

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """Short IMMEDIATE transaction.

        ``BEGIN IMMEDIATE`` takes the write lock up front, which converts a
        would-be mid-transaction ``SQLITE_BUSY`` (that SQLite cannot retry for
        us) into an up-front wait governed by ``busy_timeout``.
        """
        connection = self.connection
        connection.execute("BEGIN IMMEDIATE")
        try:
            yield connection
        except BaseException:
            connection.execute("ROLLBACK")
            raise
        else:
            connection.execute("COMMIT")

    # -- migration -----------------------------------------------------------

    def _current_version(self) -> int:
        row = self.connection.execute("PRAGMA user_version").fetchone()
        return int(row[0]) if row else 0

    def _check_version(self) -> None:
        version = self._current_version()
        if version > REGISTRY_SCHEMA_VERSION:
            raise SchemaVersionError(
                f"registry at {self.path} has schema version {version}, but this build "
                f"understands at most {REGISTRY_SCHEMA_VERSION}. Refusing to open: "
                "operating on a newer schema risks silent data loss."
            )

    def migrate(self) -> list[int]:
        """Apply pending migrations. Returns the versions applied."""
        self._check_version()
        current = self._current_version()
        pending = list(migration_module.pending(current))
        if not pending:
            return []
        if current > 0:
            backup_database(self.path)
        applied: list[int] = []
        for migration in pending:
            with self.transaction() as connection:
                for statement in migration.statements:
                    connection.execute(statement)
                connection.execute(
                    "INSERT OR REPLACE INTO schema_migrations(version, name, applied_at) "
                    "VALUES (?, ?, ?)",
                    (migration.version, migration.name, utc_now()),
                )
                # PRAGMA user_version does not accept a bound parameter.
                connection.execute(f"PRAGMA user_version={int(migration.version)}")
            applied.append(migration.version)
        return applied

    @property
    def schema_version(self) -> int:
        return self._current_version()

    # -- experiments ---------------------------------------------------------

    def upsert_experiment(self, record: ExperimentRecord) -> ExperimentRecord:
        created_at = record.created_at or utc_now()
        with self.transaction() as connection:
            connection.execute(
                """
                INSERT INTO experiments(experiment_id, name, description, tags_json, created_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(experiment_id) DO UPDATE SET
                    name = excluded.name,
                    description = excluded.description,
                    tags_json = excluded.tags_json
                """,
                (
                    record.experiment_id,
                    record.name,
                    record.description,
                    json.dumps(list(record.tags)),
                    created_at,
                ),
            )
        return ExperimentRecord(
            experiment_id=record.experiment_id,
            name=record.name,
            description=record.description,
            tags=tuple(record.tags),
            created_at=created_at,
        )

    def get_experiment(self, experiment_id: str) -> Optional[ExperimentRecord]:
        row = self.connection.execute(
            "SELECT * FROM experiments WHERE experiment_id = ?", (experiment_id,)
        ).fetchone()
        if row is None:
            return None
        return ExperimentRecord(
            experiment_id=row["experiment_id"],
            name=row["name"],
            description=row["description"],
            tags=tuple(json.loads(row["tags_json"])),
            created_at=row["created_at"],
        )

    def list_experiments(self) -> list[ExperimentRecord]:
        rows = self.connection.execute(
            "SELECT * FROM experiments ORDER BY created_at DESC"
        ).fetchall()
        return [
            ExperimentRecord(
                experiment_id=row["experiment_id"],
                name=row["name"],
                description=row["description"],
                tags=tuple(json.loads(row["tags_json"])),
                created_at=row["created_at"],
            )
            for row in rows
        ]

    # -- runs ----------------------------------------------------------------

    def create_run(self, record: RunRecord) -> RunRecord:
        """Insert a new run in state ``CREATED``.

        Fails loudly on duplicate ``run_id`` — a duplicate means an id was
        reused, which must never happen silently.
        """
        now = record.created_at or utc_now()
        with self.transaction() as connection:
            connection.execute(
                """
                INSERT INTO runs (
                    run_id, experiment_id, state, state_version, created_at, updated_at,
                    model_id, implementation_id, init_id, variant_id, task_id, scenario_id,
                    seed, device, run_dir, structural_config_hash, operational_config_hash,
                    resolved_spec_hash, legacy, status_confidence, parent_run_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.run_id,
                    record.experiment_id,
                    record.state.value,
                    int(record.state_version),
                    now,
                    now,
                    record.model_id,
                    record.implementation_id,
                    record.init_id,
                    record.variant_id,
                    record.task_id,
                    record.scenario_id,
                    int(record.seed),
                    record.device,
                    record.run_dir,
                    record.structural_config_hash,
                    record.operational_config_hash,
                    record.resolved_spec_hash,
                    1 if record.legacy else 0,
                    record.status_confidence,
                    record.parent_run_id,
                ),
            )
            connection.execute(
                """
                INSERT INTO run_state_transitions(run_id, from_state, to_state, state_version, at, actor, reason)
                VALUES (?, NULL, ?, ?, ?, ?, ?)
                """,
                (record.run_id, record.state.value, int(record.state_version), now, "control-plane", "run created"),
            )
        got = self.get_run(record.run_id)
        assert got is not None  # just inserted
        return got

    def get_run(self, run_id: str) -> Optional[RunRecord]:
        row = self.connection.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        return _row_to_run(row) if row is not None else None

    def list_runs(
        self,
        *,
        state: Optional[RunState | str] = None,
        experiment_id: Optional[str] = None,
        model_id: Optional[str] = None,
        variant_id: Optional[str] = None,
        include_legacy: bool = True,
        active_only: bool = False,
        limit: int = 200,
        offset: int = 0,
    ) -> list[RunRecord]:
        """Indexed run listing.

        ``limit`` is always applied: an unbounded run table query is the classic
        way to make a dashboard unusable once the registry has 100k rows
        (design doc 06 §12).
        """
        clauses: list[str] = []
        params: list[Any] = []
        if state is not None:
            clauses.append("state = ?")
            params.append(state.value if isinstance(state, RunState) else str(state))
        if experiment_id is not None:
            clauses.append("experiment_id = ?")
            params.append(experiment_id)
        if model_id is not None:
            clauses.append("model_id = ?")
            params.append(model_id)
        if variant_id is not None:
            clauses.append("variant_id = ?")
            params.append(variant_id)
        if not include_legacy:
            clauses.append("legacy = 0")
        if active_only:
            clauses.append("state NOT IN ('COMPLETED','FAILED','CANCELLED')")
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        rows = self.connection.execute(
            f"SELECT * FROM runs {where} ORDER BY created_at DESC, run_id DESC LIMIT ? OFFSET ?",
            (*params, int(max(1, limit)), int(max(0, offset))),
        ).fetchall()
        return [_row_to_run(row) for row in rows]

    def count_runs(self, *, include_legacy: bool = True) -> int:
        sql = "SELECT COUNT(*) FROM runs" + ("" if include_legacy else " WHERE legacy = 0")
        return int(self.connection.execute(sql).fetchone()[0])

    def transition(
        self,
        run_id: str,
        *,
        to_state: RunState,
        expected_state_version: Optional[int] = None,
        actor: str = "worker",
        reason: Optional[str] = None,
        fields: Optional[Mapping[str, Any]] = None,
    ) -> RunRecord:
        """Move a run to *to_state*, validating the transition and version.

        The whole check-and-set happens inside one IMMEDIATE transaction, so two
        concurrent transitions cannot both observe the same ``state_version``.

        *fields* may carry additional column updates (pid, phase, exit_code, …)
        applied atomically with the transition.
        """
        with self.transaction() as connection:
            row = connection.execute(
                "SELECT state, state_version FROM runs WHERE run_id = ?", (run_id,)
            ).fetchone()
            if row is None:
                raise RegistryError(f"unknown run_id {run_id!r}")
            current_state = RunState(row["state"])
            current_version = int(row["state_version"])
            if expected_state_version is not None and expected_state_version != current_version:
                raise ConcurrencyError(
                    f"run {run_id} is at state_version {current_version}, caller expected "
                    f"{expected_state_version}; re-read the run and decide again"
                )
            validate_transition(current_state, to_state)

            now = utc_now()
            new_version = current_version + 1
            assignments = {
                "state": to_state.value,
                "state_version": new_version,
                "updated_at": now,
            }
            if to_state is RunState.RUNNING and not _has_value(connection, run_id, "started_at"):
                assignments["started_at"] = now
            if to_state in (RunState.COMPLETED, RunState.FAILED, RunState.CANCELLED):
                assignments["ended_at"] = now
            for key, value in dict(fields or {}).items():
                if key not in _RUN_UPDATABLE_COLUMNS:
                    raise RegistryError(f"column {key!r} is not updatable through transition()")
                assignments[key] = value

            columns = ", ".join(f"{key} = ?" for key in assignments)
            connection.execute(
                f"UPDATE runs SET {columns} WHERE run_id = ? AND state_version = ?",
                (*assignments.values(), run_id, current_version),
            )
            connection.execute(
                """
                INSERT INTO run_state_transitions(run_id, from_state, to_state, state_version, at, actor, reason)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (run_id, current_state.value, to_state.value, new_version, now, actor, reason),
            )
            # The worker row is registered before the child can atomically move
            # its run from STARTING to RUNNING. Mirror that observation here so
            # a restarted API does not report a live run as worker STARTING.
            if to_state is RunState.RUNNING:
                connection.execute(
                    "UPDATE workers SET state = ?, last_heartbeat_at = COALESCE(last_heartbeat_at, ?) "
                    "WHERE run_id = ? AND state = 'STARTING'",
                    (RunState.RUNNING.value, now, run_id),
                )
        updated = self.get_run(run_id)
        assert updated is not None
        return updated

    def update_progress(
        self,
        run_id: str,
        *,
        phase: Optional[str] = None,
        subphase: Optional[str] = None,
        global_step: Optional[int] = None,
        epoch: Optional[int] = None,
        batch_cursor: Optional[int] = None,
        last_event_id: Optional[int] = None,
    ) -> None:
        """Update progress columns without touching ``state`` or ``state_version``.

        Progress is not a state transition: bumping ``state_version`` on every
        training step would make optimistic concurrency useless for its actual
        purpose (guarding control actions).
        """
        assignments: dict[str, Any] = {"updated_at": utc_now()}
        for key, value in (
            ("phase", phase),
            ("subphase", subphase),
            ("global_step", global_step),
            ("epoch", epoch),
            ("batch_cursor", batch_cursor),
            ("last_event_id", last_event_id),
        ):
            if value is not None:
                assignments[key] = value
        columns = ", ".join(f"{key} = ?" for key in assignments)
        with self.transaction() as connection:
            connection.execute(
                f"UPDATE runs SET {columns} WHERE run_id = ?", (*assignments.values(), run_id)
            )

    def record_heartbeat(self, run_id: str, *, worker_instance_id: Optional[str] = None) -> str:
        """Record a liveness heartbeat for *run_id*. Returns the timestamp used."""
        now = utc_now()
        with self.transaction() as connection:
            connection.execute(
                "UPDATE runs SET heartbeat_at = ?, updated_at = ? WHERE run_id = ?",
                (now, now, run_id),
            )
            if worker_instance_id:
                connection.execute(
                    "UPDATE workers SET last_heartbeat_at = ? WHERE worker_instance_id = ?",
                    (now, worker_instance_id),
                )
        return now

    # -- workers -------------------------------------------------------------

    def register_worker(self, record: WorkerRecord) -> WorkerRecord:
        with self.transaction() as connection:
            connection.execute(
                """
                INSERT INTO workers (
                    worker_instance_id, run_id, host, pid, process_group_id,
                    process_start_time, worker_token, started_at, last_heartbeat_at, state
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.worker_instance_id,
                    record.run_id,
                    record.host,
                    int(record.pid),
                    int(record.process_group_id),
                    float(record.process_start_time),
                    record.worker_token,
                    record.started_at,
                    record.last_heartbeat_at,
                    record.state,
                ),
            )
            connection.execute(
                "UPDATE runs SET worker_instance_id = ?, pid = ?, process_group_id = ?, host = ? "
                "WHERE run_id = ?",
                (
                    record.worker_instance_id,
                    int(record.pid),
                    int(record.process_group_id),
                    record.host,
                    record.run_id,
                ),
            )
        return record

    def finish_worker(self, worker_instance_id: str, *, state: str, exit_code: Optional[int]) -> None:
        with self.transaction() as connection:
            connection.execute(
                "UPDATE workers SET state = ?, exit_code = ? WHERE worker_instance_id = ?",
                (state, exit_code, worker_instance_id),
            )

    def get_worker(self, worker_instance_id: str) -> Optional[WorkerRecord]:
        row = self.connection.execute(
            "SELECT * FROM workers WHERE worker_instance_id = ?", (worker_instance_id,)
        ).fetchone()
        return _row_to_worker(row) if row is not None else None

    def worker_for_run(self, run_id: str) -> Optional[WorkerRecord]:
        row = self.connection.execute(
            "SELECT * FROM workers WHERE run_id = ? ORDER BY started_at DESC LIMIT 1", (run_id,)
        ).fetchone()
        return _row_to_worker(row) if row is not None else None

    def list_workers(self, *, active_only: bool = False) -> list[WorkerRecord]:
        sql = "SELECT * FROM workers"
        if active_only:
            sql += " WHERE state NOT IN ('EXITED','ORPHANED')"
        sql += " ORDER BY started_at DESC"
        return [_row_to_worker(row) for row in self.connection.execute(sql).fetchall()]

    # -- artifacts / checkpoints --------------------------------------------

    def record_artifact(self, record: ArtifactRecord) -> ArtifactRecord:
        created_at = record.created_at or utc_now()
        with self.transaction() as connection:
            connection.execute(
                """
                INSERT INTO artifacts (
                    artifact_id, run_id, kind, uri, sha256, bytes, media_type,
                    created_at, complete, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(artifact_id) DO UPDATE SET
                    sha256 = excluded.sha256,
                    bytes = excluded.bytes,
                    complete = excluded.complete,
                    metadata_json = excluded.metadata_json
                """,
                (
                    record.artifact_id,
                    record.run_id,
                    record.kind,
                    record.uri,
                    record.sha256,
                    int(record.bytes_),
                    record.media_type,
                    created_at,
                    1 if record.complete else 0,
                    json.dumps(dict(record.metadata)),
                ),
            )
        return record

    def list_artifacts(self, run_id: str) -> list[ArtifactRecord]:
        rows = self.connection.execute(
            "SELECT * FROM artifacts WHERE run_id = ? ORDER BY created_at", (run_id,)
        ).fetchall()
        return [
            ArtifactRecord(
                artifact_id=row["artifact_id"],
                run_id=row["run_id"],
                kind=row["kind"],
                uri=row["uri"],
                sha256=row["sha256"],
                bytes_=int(row["bytes"]),
                media_type=row["media_type"],
                created_at=row["created_at"],
                complete=bool(row["complete"]),
                metadata=json.loads(row["metadata_json"]),
            )
            for row in rows
        ]

    def list_checkpoints(self, run_id: str) -> list[dict[str, Any]]:
        """List checkpoint catalog rows for a run.

        This tranche never *writes* checkpoint rows — checkpoint v1 is Phase 2
        work. The table and this reader exist so the dashboard can already show
        "no checkpoints registered" honestly instead of hiding the concept.
        """
        rows = self.connection.execute(
            "SELECT * FROM checkpoints WHERE run_id = ? ORDER BY created_at", (run_id,)
        ).fetchall()
        return [dict(row) for row in rows]

    # -- state transitions log ----------------------------------------------

    def list_transitions(self, run_id: str) -> list[dict[str, Any]]:
        rows = self.connection.execute(
            "SELECT * FROM run_state_transitions WHERE run_id = ? ORDER BY id", (run_id,)
        ).fetchall()
        return [dict(row) for row in rows]

    # -- legacy mapping ------------------------------------------------------

    def record_legacy_mapping(
        self,
        *,
        run_id: str,
        legacy_path: str,
        legacy_path_hash: str,
        meta: Mapping[str, Any],
        status_confidence: str,
    ) -> None:
        with self.transaction() as connection:
            connection.execute(
                """
                INSERT INTO legacy_run_mappings (
                    run_id, legacy_path, legacy_path_hash, imported_at, meta_json, status_confidence
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id) DO UPDATE SET
                    meta_json = excluded.meta_json,
                    status_confidence = excluded.status_confidence
                """,
                (
                    run_id,
                    legacy_path,
                    legacy_path_hash,
                    utc_now(),
                    json.dumps(dict(meta)),
                    status_confidence,
                ),
            )

    def legacy_run_for_path_hash(self, legacy_path_hash: str) -> Optional[str]:
        row = self.connection.execute(
            "SELECT run_id FROM legacy_run_mappings WHERE legacy_path_hash = ?", (legacy_path_hash,)
        ).fetchone()
        return str(row["run_id"]) if row is not None else None

    def legacy_mapping(self, run_id: str) -> Optional[dict[str, Any]]:
        row = self.connection.execute(
            "SELECT * FROM legacy_run_mappings WHERE run_id = ?", (run_id,)
        ).fetchone()
        return dict(row) if row is not None else None

    # -- capabilities cache --------------------------------------------------

    def record_capabilities(self, documents: Sequence[Mapping[str, Any]]) -> None:
        now = utc_now()
        with self.transaction() as connection:
            for document in documents:
                connection.execute(
                    """
                    INSERT INTO model_capabilities(model_id, implementation_id, document_json, recorded_at)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(model_id, implementation_id) DO UPDATE SET
                        document_json = excluded.document_json,
                        recorded_at = excluded.recorded_at
                    """,
                    (
                        str(document.get("model_id")),
                        str(document.get("implementation_id")),
                        json.dumps(dict(document)),
                        now,
                    ),
                )

    # -- GPU leases ----------------------------------------------------------

    def acquire_gpu_lease(self, *, device_index: int, run_id: str, device_uuid: Optional[str] = None) -> Optional[str]:
        """Acquire an exclusive lease on *device_index*.

        Returns the lease id, or ``None`` if the device is already held. The
        exclusion is enforced by a partial unique index, so two concurrent
        processes cannot both succeed.
        """
        lease_id = uuid7()
        try:
            with self.transaction() as connection:
                connection.execute(
                    """
                    INSERT INTO gpu_leases(lease_id, device_index, device_uuid, run_id, acquired_at, state)
                    VALUES (?, ?, ?, ?, ?, 'HELD')
                    """,
                    (lease_id, int(device_index), device_uuid, run_id, utc_now()),
                )
                connection.execute(
                    "UPDATE runs SET gpu_lease_id = ? WHERE run_id = ?", (lease_id, run_id)
                )
        except sqlite3.IntegrityError:
            return None
        return lease_id

    def release_gpu_lease(self, lease_id: str) -> None:
        with self.transaction() as connection:
            connection.execute(
                "UPDATE gpu_leases SET state = 'RELEASED', released_at = ? WHERE lease_id = ?",
                (utc_now(), lease_id),
            )

    def active_gpu_leases(self) -> list[dict[str, Any]]:
        rows = self.connection.execute(
            "SELECT * FROM gpu_leases WHERE state = 'HELD' ORDER BY acquired_at"
        ).fetchall()
        return [dict(row) for row in rows]


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

#: Columns a state transition is allowed to set alongside the state change.
_RUN_UPDATABLE_COLUMNS = frozenset(
    {
        "started_at",
        "ended_at",
        "host",
        "pid",
        "process_group_id",
        "heartbeat_at",
        "worker_instance_id",
        "gpu_lease_id",
        "device",
        "phase",
        "subphase",
        "global_step",
        "epoch",
        "batch_cursor",
        "last_event_id",
        "latest_checkpoint_id",
        "best_checkpoint_id",
        "exit_code",
        "terminal_reason",
        "error_summary",
        "status_confidence",
    }
)


def _has_value(connection: sqlite3.Connection, run_id: str, column: str) -> bool:
    row = connection.execute(f"SELECT {column} FROM runs WHERE run_id = ?", (run_id,)).fetchone()
    return bool(row is not None and row[0])


def _row_to_run(row: sqlite3.Row) -> RunRecord:
    return RunRecord(
        run_id=row["run_id"],
        experiment_id=row["experiment_id"],
        state=RunState(row["state"]),
        state_version=int(row["state_version"]),
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        model_id=row["model_id"],
        implementation_id=row["implementation_id"],
        init_id=row["init_id"],
        variant_id=row["variant_id"],
        task_id=row["task_id"],
        scenario_id=row["scenario_id"],
        seed=int(row["seed"]),
        started_at=row["started_at"],
        ended_at=row["ended_at"],
        host=row["host"],
        pid=row["pid"],
        process_group_id=row["process_group_id"],
        heartbeat_at=row["heartbeat_at"],
        worker_instance_id=row["worker_instance_id"],
        gpu_lease_id=row["gpu_lease_id"],
        device=row["device"],
        phase=row["phase"],
        subphase=row["subphase"],
        global_step=int(row["global_step"]),
        epoch=int(row["epoch"]),
        batch_cursor=int(row["batch_cursor"]),
        last_event_id=int(row["last_event_id"]),
        latest_checkpoint_id=row["latest_checkpoint_id"],
        best_checkpoint_id=row["best_checkpoint_id"],
        parent_run_id=row["parent_run_id"],
        resumed_from_run_id=row["resumed_from_run_id"],
        resumed_from_checkpoint_id=row["resumed_from_checkpoint_id"],
        exit_code=row["exit_code"],
        terminal_reason=row["terminal_reason"],
        error_summary=row["error_summary"],
        run_dir=row["run_dir"],
        structural_config_hash=row["structural_config_hash"],
        operational_config_hash=row["operational_config_hash"],
        resolved_spec_hash=row["resolved_spec_hash"],
        legacy=bool(row["legacy"]),
        status_confidence=row["status_confidence"],
    )


def _row_to_worker(row: sqlite3.Row) -> WorkerRecord:
    return WorkerRecord(
        worker_instance_id=row["worker_instance_id"],
        run_id=row["run_id"],
        host=row["host"],
        pid=int(row["pid"]),
        process_group_id=int(row["process_group_id"]),
        process_start_time=float(row["process_start_time"]),
        worker_token=row["worker_token"],
        started_at=row["started_at"],
        last_heartbeat_at=row["last_heartbeat_at"],
        state=row["state"],
        exit_code=row["exit_code"],
    )


def open_registry(root: Optional[str | os.PathLike[str]] = None, **kwargs: Any) -> SqliteRegistry:
    """Open (creating if needed) the registry under the control root."""
    return SqliteRegistry(registry_path(root), **kwargs)
