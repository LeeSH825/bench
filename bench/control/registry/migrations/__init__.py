"""Forward-only registry migrations.

Rules (design doc 03 §12.2, §5.3 of doc 05):

* migrations are **forward-only** and numbered from 1
* each migration is applied inside one transaction and recorded in
  ``schema_migrations``
* a database whose ``user_version`` is *newer* than this code understands is
  rejected rather than opened — silently operating on a future schema is worse
  than refusing
* the caller takes a file backup before migrating an existing database
  (:func:`bench.control.registry.sqlite.backup_database`)

To add a migration: append a new ``Migration`` to :data:`MIGRATIONS` with the
next version number. Never edit an already-released migration body — that
produces databases that disagree about what version 1 means.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class Migration:
    version: int
    name: str
    statements: tuple[str, ...]


_M0001_STATEMENTS: tuple[str, ...] = (
    """
    CREATE TABLE IF NOT EXISTS schema_migrations (
        version     INTEGER PRIMARY KEY,
        name        TEXT NOT NULL,
        applied_at  TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS experiments (
        experiment_id TEXT PRIMARY KEY,
        name          TEXT NOT NULL,
        description   TEXT NOT NULL DEFAULT '',
        tags_json     TEXT NOT NULL DEFAULT '[]',
        created_at    TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS runs (
        run_id                     TEXT PRIMARY KEY,
        experiment_id              TEXT NOT NULL REFERENCES experiments(experiment_id),
        state                      TEXT NOT NULL,
        state_version              INTEGER NOT NULL DEFAULT 0,
        created_at                 TEXT NOT NULL,
        updated_at                 TEXT NOT NULL,
        started_at                 TEXT,
        ended_at                   TEXT,
        host                       TEXT,
        pid                        INTEGER,
        process_group_id           INTEGER,
        heartbeat_at               TEXT,
        worker_instance_id         TEXT,
        gpu_lease_id               TEXT,
        device                     TEXT,
        phase                      TEXT,
        subphase                   TEXT,
        global_step                INTEGER NOT NULL DEFAULT 0,
        epoch                      INTEGER NOT NULL DEFAULT 0,
        batch_cursor               INTEGER NOT NULL DEFAULT 0,
        last_event_id              INTEGER NOT NULL DEFAULT 0,
        latest_checkpoint_id       TEXT,
        best_checkpoint_id         TEXT,
        parent_run_id              TEXT,
        resumed_from_run_id        TEXT,
        resumed_from_checkpoint_id TEXT,
        exit_code                  INTEGER,
        terminal_reason            TEXT,
        error_summary              TEXT,
        model_id                   TEXT NOT NULL DEFAULT '',
        implementation_id          TEXT NOT NULL DEFAULT '',
        init_id                    TEXT NOT NULL DEFAULT '',
        variant_id                 TEXT NOT NULL DEFAULT '',
        task_id                    TEXT NOT NULL DEFAULT '',
        scenario_id                TEXT NOT NULL DEFAULT '',
        seed                       INTEGER NOT NULL DEFAULT 0,
        run_dir                    TEXT NOT NULL DEFAULT '',
        structural_config_hash     TEXT NOT NULL DEFAULT '',
        operational_config_hash    TEXT NOT NULL DEFAULT '',
        resolved_spec_hash         TEXT NOT NULL DEFAULT '',
        legacy                     INTEGER NOT NULL DEFAULT 0,
        status_confidence          TEXT
    )
    """,
    # Index set from design doc 03 §12.1.
    "CREATE INDEX IF NOT EXISTS idx_runs_state_updated ON runs(state, updated_at DESC)",
    "CREATE INDEX IF NOT EXISTS idx_runs_experiment_created ON runs(experiment_id, created_at DESC)",
    "CREATE INDEX IF NOT EXISTS idx_runs_identity ON runs(model_id, implementation_id, init_id)",
    "CREATE INDEX IF NOT EXISTS idx_runs_variant ON runs(variant_id)",
    "CREATE INDEX IF NOT EXISTS idx_runs_parent ON runs(parent_run_id)",
    "CREATE INDEX IF NOT EXISTS idx_runs_heartbeat ON runs(heartbeat_at)",
    "CREATE INDEX IF NOT EXISTS idx_runs_device ON runs(device)",
    "CREATE INDEX IF NOT EXISTS idx_runs_legacy ON runs(legacy)",
    """
    CREATE TABLE IF NOT EXISTS run_state_transitions (
        id            INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id        TEXT NOT NULL REFERENCES runs(run_id),
        from_state    TEXT,
        to_state      TEXT NOT NULL,
        state_version INTEGER NOT NULL,
        at            TEXT NOT NULL,
        actor         TEXT NOT NULL DEFAULT 'system',
        reason        TEXT
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_transitions_run ON run_state_transitions(run_id, id)",
    """
    CREATE TABLE IF NOT EXISTS run_actions (
        action_id              TEXT PRIMARY KEY,
        run_id                 TEXT NOT NULL REFERENCES runs(run_id),
        action                 TEXT NOT NULL,
        requested_at           TEXT NOT NULL,
        requested_by           TEXT NOT NULL DEFAULT 'local-user',
        expected_state_version INTEGER,
        parameters_json        TEXT NOT NULL DEFAULT '{}',
        status                 TEXT NOT NULL DEFAULT 'PENDING',
        handled_at             TEXT,
        result_json            TEXT,
        idempotency_key        TEXT
    )
    """,
    "CREATE UNIQUE INDEX IF NOT EXISTS idx_actions_idempotency ON run_actions(idempotency_key) WHERE idempotency_key IS NOT NULL",
    "CREATE INDEX IF NOT EXISTS idx_actions_run ON run_actions(run_id, requested_at)",
    """
    CREATE TABLE IF NOT EXISTS workers (
        worker_instance_id TEXT PRIMARY KEY,
        run_id             TEXT NOT NULL REFERENCES runs(run_id),
        host               TEXT NOT NULL,
        pid                INTEGER NOT NULL,
        process_group_id   INTEGER NOT NULL,
        process_start_time REAL NOT NULL,
        worker_token       TEXT NOT NULL,
        started_at         TEXT NOT NULL,
        last_heartbeat_at  TEXT,
        state              TEXT NOT NULL DEFAULT 'STARTING',
        exit_code          INTEGER
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_workers_run ON workers(run_id)",
    "CREATE INDEX IF NOT EXISTS idx_workers_heartbeat ON workers(last_heartbeat_at)",
    """
    CREATE TABLE IF NOT EXISTS gpu_leases (
        lease_id     TEXT PRIMARY KEY,
        device_index INTEGER NOT NULL,
        device_uuid  TEXT,
        run_id       TEXT NOT NULL REFERENCES runs(run_id),
        acquired_at  TEXT NOT NULL,
        released_at  TEXT,
        state        TEXT NOT NULL DEFAULT 'HELD'
    )
    """,
    # At most one HELD lease per device — this is the DB-level guarantee behind
    # acceptance P-07 (no two trainable runs share a GPU).
    "CREATE UNIQUE INDEX IF NOT EXISTS idx_gpu_lease_active ON gpu_leases(device_index) WHERE state = 'HELD'",
    """
    CREATE TABLE IF NOT EXISTS checkpoints (
        checkpoint_id  TEXT PRIMARY KEY,
        run_id         TEXT NOT NULL REFERENCES runs(run_id),
        kind           TEXT NOT NULL,
        created_at     TEXT NOT NULL,
        phase          TEXT,
        global_step    INTEGER NOT NULL DEFAULT 0,
        payload_uri    TEXT NOT NULL,
        payload_sha256 TEXT,
        payload_bytes  INTEGER NOT NULL DEFAULT 0,
        manifest_json  TEXT NOT NULL DEFAULT '{}',
        event_cursor   INTEGER NOT NULL DEFAULT 0,
        complete       INTEGER NOT NULL DEFAULT 0,
        exact_resume_certified INTEGER NOT NULL DEFAULT 0
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_checkpoints_run ON checkpoints(run_id, created_at)",
    """
    CREATE TABLE IF NOT EXISTS artifacts (
        artifact_id   TEXT PRIMARY KEY,
        run_id        TEXT NOT NULL REFERENCES runs(run_id),
        kind          TEXT NOT NULL,
        uri           TEXT NOT NULL,
        sha256        TEXT,
        bytes         INTEGER NOT NULL DEFAULT 0,
        media_type    TEXT NOT NULL DEFAULT 'application/octet-stream',
        created_at    TEXT NOT NULL,
        complete      INTEGER NOT NULL DEFAULT 1,
        metadata_json TEXT NOT NULL DEFAULT '{}'
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_artifacts_run ON artifacts(run_id, created_at)",
    """
    CREATE TABLE IF NOT EXISTS model_capabilities (
        model_id          TEXT NOT NULL,
        implementation_id TEXT NOT NULL,
        document_json     TEXT NOT NULL,
        recorded_at       TEXT NOT NULL,
        PRIMARY KEY (model_id, implementation_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS legacy_run_mappings (
        run_id           TEXT PRIMARY KEY REFERENCES runs(run_id),
        legacy_path      TEXT NOT NULL,
        legacy_path_hash TEXT NOT NULL,
        imported_at      TEXT NOT NULL,
        meta_json        TEXT NOT NULL DEFAULT '{}',
        status_confidence TEXT NOT NULL DEFAULT 'unknown'
    )
    """,
    # One synthetic run per legacy directory: re-importing is idempotent rather
    # than duplicating the run list.
    "CREATE UNIQUE INDEX IF NOT EXISTS idx_legacy_path ON legacy_run_mappings(legacy_path_hash)",
)


MIGRATIONS: tuple[Migration, ...] = (
    Migration(version=1, name="initial_control_plane_schema", statements=_M0001_STATEMENTS),
)


def latest_version() -> int:
    return max(migration.version for migration in MIGRATIONS)


def pending(current_version: int) -> Sequence[Migration]:
    return tuple(m for m in MIGRATIONS if m.version > current_version)
