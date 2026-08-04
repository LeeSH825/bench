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


# Migration 2 turns the schema-only checkpoint and action tables from
# migration 1 into live lifecycle tables. Forward-only, additive: every
# statement is an ADD COLUMN or a new index, so an existing registry keeps all
# of its rows and nothing is rewritten.
#
# SQLite has no "ADD COLUMN IF NOT EXISTS", and re-running a migration is not
# possible anyway because schema_migrations gates it, so the ADD COLUMNs are
# plain. Defaults are chosen so pre-existing rows (there are none in practice —
# this tranche is the first writer) read as UNVERIFIED rather than as VALID.
_M0002_STATEMENTS: tuple[str, ...] = (
    # -- checkpoints: validation, compatibility keys, lineage ----------------
    "ALTER TABLE checkpoints ADD COLUMN validation_status TEXT NOT NULL DEFAULT 'UNVERIFIED'",
    "ALTER TABLE checkpoints ADD COLUMN validation_detail TEXT",
    "ALTER TABLE checkpoints ADD COLUMN validated_at TEXT",
    "ALTER TABLE checkpoints ADD COLUMN resume_boundary TEXT",
    "ALTER TABLE checkpoints ADD COLUMN structural_config_hash TEXT NOT NULL DEFAULT ''",
    "ALTER TABLE checkpoints ADD COLUMN dataset_fingerprint TEXT NOT NULL DEFAULT ''",
    "ALTER TABLE checkpoints ADD COLUMN implementation_id TEXT NOT NULL DEFAULT ''",
    "ALTER TABLE checkpoints ADD COLUMN variant_id TEXT NOT NULL DEFAULT ''",
    "ALTER TABLE checkpoints ADD COLUMN certification_key TEXT NOT NULL DEFAULT ''",
    "CREATE INDEX IF NOT EXISTS idx_checkpoints_kind ON checkpoints(run_id, kind, global_step DESC)",
    "CREATE INDEX IF NOT EXISTS idx_checkpoints_validation ON checkpoints(validation_status)",
    # -- run_actions: the persistent stop-request lifecycle -------------------
    # Migration 1 shipped `status` and `handled_at`; the acknowledge/complete
    # split and the resulting checkpoint pointer are new.
    "ALTER TABLE run_actions ADD COLUMN acknowledged_at TEXT",
    "ALTER TABLE run_actions ADD COLUMN completed_at TEXT",
    "ALTER TABLE run_actions ADD COLUMN failure_reason TEXT",
    "ALTER TABLE run_actions ADD COLUMN result_checkpoint_id TEXT",
    "ALTER TABLE run_actions ADD COLUMN state_version INTEGER NOT NULL DEFAULT 0",
    "CREATE INDEX IF NOT EXISTS idx_actions_pending ON run_actions(run_id, status)",
    # -- certified exact-resume envelopes ------------------------------------
    # A certification is a row, not a boolean on a model name (ADR-CSR-013).
    """
    CREATE TABLE IF NOT EXISTS exact_resume_certifications (
        certification_key          TEXT PRIMARY KEY,
        model_id                   TEXT NOT NULL,
        implementation_id          TEXT NOT NULL,
        checkpoint_schema_version  INTEGER NOT NULL,
        resume_boundary            TEXT NOT NULL,
        precision                  TEXT NOT NULL,
        device_class               TEXT NOT NULL,
        num_workers                INTEGER NOT NULL,
        training_mode              TEXT NOT NULL,
        certified                  INTEGER NOT NULL DEFAULT 0,
        evidence_uri               TEXT,
        certified_at               TEXT,
        notes                      TEXT
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_certifications_model ON exact_resume_certifications(model_id, certified)",
)


# Migration 3 persists the training-path decision on the run row and extends
# the certification key with it (ADR-WC-004, continuation gate B0).
#
# The default is deliberately 'legacy_train_v1': a row written before this
# migration has no evidence that it used the resumable loop, and inventing that
# evidence would let an old run claim exact-resume eligibility it never earned.
# Promotion is only ever forward, by a new run resolving to the certified path.
_M0003_STATEMENTS: tuple[str, ...] = (
    "ALTER TABLE runs ADD COLUMN training_path_id TEXT NOT NULL DEFAULT 'legacy_train_v1'",
    "ALTER TABLE runs ADD COLUMN training_path_reason_code TEXT",
    "ALTER TABLE runs ADD COLUMN training_path_contract_version INTEGER NOT NULL DEFAULT 0",
    "CREATE INDEX IF NOT EXISTS idx_runs_training_path ON runs(training_path_id)",
    # Checkpoint rows record the path the package proves, so eligibility can be
    # answered from the catalog without opening the payload.
    "ALTER TABLE checkpoints ADD COLUMN training_path_id TEXT",
    "ALTER TABLE checkpoints ADD COLUMN checkpoint_schema_version INTEGER NOT NULL DEFAULT 1",
    "ALTER TABLE exact_resume_certifications ADD COLUMN training_path_id TEXT NOT NULL DEFAULT 'legacy_train_v1'",
    # A resume action points at the child it launched. Recorded before the
    # launch so a crash mid-launch is recoverable without a second child.
    "ALTER TABLE run_actions ADD COLUMN result_child_run_id TEXT",
    "ALTER TABLE run_actions ADD COLUMN result_worker_instance_id TEXT",
    "CREATE INDEX IF NOT EXISTS idx_actions_child ON run_actions(result_child_run_id)",
)


# Migration 4 makes run_actions.run_id nullable.
#
# Why this is necessary rather than convenient: a durable LAUNCH_RUN action has
# to be recorded *before* the run it will allocate, so that a retry with the
# same idempotency key finds the existing intent instead of allocating a second
# run. With run_id NOT NULL the action could only be written after allocation,
# which reopens exactly the duplicate-run window the idempotency contract
# forbids.
#
# Alternatives considered and rejected: allocating first and writing the action
# afterwards (same duplicate window), and a separate launch table (forbidden —
# the durable action infrastructure is meant to be reused, not duplicated).
#
# SQLite cannot drop a NOT NULL constraint in place, so this is the standard
# table rebuild: create, copy every existing row, drop, rename, recreate
# indexes. Existing rows keep their run_id; only the constraint relaxes.
_M0004_STATEMENTS: tuple[str, ...] = (
    """
    CREATE TABLE run_actions_new (
        action_id                 TEXT PRIMARY KEY,
        run_id                    TEXT REFERENCES runs(run_id),
        action                    TEXT NOT NULL,
        requested_at              TEXT NOT NULL,
        requested_by              TEXT NOT NULL DEFAULT 'local-user',
        expected_state_version    INTEGER,
        parameters_json           TEXT NOT NULL DEFAULT '{}',
        status                    TEXT NOT NULL DEFAULT 'PENDING',
        handled_at                TEXT,
        result_json               TEXT,
        idempotency_key           TEXT,
        acknowledged_at           TEXT,
        completed_at              TEXT,
        failure_reason            TEXT,
        result_checkpoint_id      TEXT,
        state_version             INTEGER NOT NULL DEFAULT 0,
        result_child_run_id       TEXT,
        result_worker_instance_id TEXT
    )
    """,
    """
    INSERT INTO run_actions_new (
        action_id, run_id, action, requested_at, requested_by,
        expected_state_version, parameters_json, status, handled_at, result_json,
        idempotency_key, acknowledged_at, completed_at, failure_reason,
        result_checkpoint_id, state_version, result_child_run_id,
        result_worker_instance_id
    )
    SELECT
        action_id, run_id, action, requested_at, requested_by,
        expected_state_version, parameters_json, status, handled_at, result_json,
        idempotency_key, acknowledged_at, completed_at, failure_reason,
        result_checkpoint_id, state_version, result_child_run_id,
        result_worker_instance_id
    FROM run_actions
    """,
    "DROP TABLE run_actions",
    "ALTER TABLE run_actions_new RENAME TO run_actions",
    "CREATE UNIQUE INDEX IF NOT EXISTS idx_actions_idempotency ON run_actions(idempotency_key) WHERE idempotency_key IS NOT NULL",
    "CREATE INDEX IF NOT EXISTS idx_actions_run ON run_actions(run_id, requested_at)",
    "CREATE INDEX IF NOT EXISTS idx_actions_pending ON run_actions(run_id, status)",
    "CREATE INDEX IF NOT EXISTS idx_actions_child ON run_actions(result_child_run_id)",
)


MIGRATIONS: tuple[Migration, ...] = (
    Migration(version=1, name="initial_control_plane_schema", statements=_M0001_STATEMENTS),
    Migration(version=2, name="checkpoint_v1_and_stop_actions", statements=_M0002_STATEMENTS),
    Migration(version=3, name="training_path_persistence", statements=_M0003_STATEMENTS),
    Migration(version=4, name="nullable_action_run_id_for_launch", statements=_M0004_STATEMENTS),
)


def latest_version() -> int:
    return max(migration.version for migration in MIGRATIONS)


def pending(current_version: int) -> Sequence[Migration]:
    return tuple(m for m in MIGRATIONS if m.version > current_version)
