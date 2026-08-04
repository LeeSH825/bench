"""Checkpoint service: publish, catalog, validate, and restore.

Publication order is the contract (ADR-CSR-008): bytes become durable before
the catalog learns the checkpoint exists. Every failure mode therefore
degrades to "files the reconciler can adjudicate", never to "a registry row
promising state that was never written".
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

from ..canonical import canonical_json
from .atomic import fsync_dir, write_bytes_durably, write_via_callback_durably
from .batchplan import BatchPlan
from .compatibility import check_manifest_compatibility
from .payload import (
    build_payload,
    component_inventory,
    payload_to_state,
    read_payload,
    write_payload,
)
from .schema import (
    CHECKPOINT_SCHEMA_VERSION,
    MANIFEST_FILENAME,
    PAYLOAD_FILENAME,
    RESUME_BOUNDARY_OPTIMIZER_UPDATE,
    AdapterTrainingState,
    CheckpointCapabilities,
    CheckpointError,
    CheckpointKind,
    CheckpointManifest,
    RngState,
    TrainingCursor,
    ValidationStatus,
)
from .validation import validate_package


def new_checkpoint_id() -> str:
    return str(uuid.uuid7()) if hasattr(uuid, "uuid7") else str(uuid.uuid4())


@dataclass
class SaveResult:
    checkpoint_id: str
    directory: Path
    manifest: CheckpointManifest
    payload_sha256: str
    payload_bytes: int


class CheckpointService:
    """Owns the checkpoint directory layout for one run.

    Layout::

        <run_dir>/checkpoints/<checkpoint_id>/manifest.json
        <run_dir>/checkpoints/<checkpoint_id>/payload.pt

    One directory per checkpoint means publication is two renames into a fresh
    directory, and an existing checkpoint is never overwritten.
    """

    def __init__(
        self,
        run_dir: Path,
        *,
        registry: Optional[Any] = None,
        event_writer: Optional[Any] = None,
        control_root: Optional[Path] = None,
    ) -> None:
        self.run_dir = Path(run_dir)
        self.root = self.run_dir / "checkpoints"
        self.registry = registry
        self.event_writer = event_writer
        self.control_root = Path(control_root) if control_root is not None else None

    # -- paths ---------------------------------------------------------------

    def directory_for(self, checkpoint_id: str) -> Path:
        return self.root / checkpoint_id

    def _assert_trusted(self, path: Path) -> None:
        """Refuse to load a package from outside the approved control root.

        Payloads are pickle-based, so provenance of the *path* is part of the
        security boundary, not just the digest (ADR-CSR-009).
        """
        if self.control_root is None:
            return
        try:
            path.resolve().relative_to(self.control_root.resolve())
        except ValueError as exc:
            raise CheckpointError(
                f"refusing to load checkpoint from {path}: outside the approved control root "
                f"{self.control_root}. Checkpoint payloads are trusted-local artifacts only."
            ) from exc

    # -- publish -------------------------------------------------------------

    def save(
        self,
        *,
        run_id: str,
        kind: CheckpointKind,
        cursor: TrainingCursor,
        adapter_state: AdapterTrainingState,
        rng: RngState,
        identity: dict[str, Any],
        structural_config_hash: str = "",
        dataset_fingerprint: str = "",
        batch_plan: Optional[BatchPlan] = None,
        resolved_run_spec: Optional[dict[str, Any]] = None,
        capabilities: Optional[CheckpointCapabilities] = None,
        certification: Optional[dict[str, Any]] = None,
        training_path_id: Optional[str] = None,
        training_path_contract_version: Optional[int] = None,
        lineage: Optional[dict[str, Any]] = None,
        provenance: Optional[dict[str, Any]] = None,
        checkpoint_id: Optional[str] = None,
        fault_hook: Optional[Callable[[str, Path], None]] = None,
    ) -> SaveResult:
        """Publish a checkpoint atomically, then catalog it."""
        from ..registry.sqlite import utc_now

        checkpoint_id = checkpoint_id or new_checkpoint_id()
        directory = self.directory_for(checkpoint_id)
        if directory.exists() and (directory / MANIFEST_FILENAME).exists():
            raise CheckpointError(
                f"checkpoint {checkpoint_id} already published; checkpoints are immutable"
            )
        directory.mkdir(parents=True, exist_ok=True)

        payload = build_payload(
            cursor=cursor,
            adapter_state=adapter_state,
            rng=rng,
            batch_plan=batch_plan.as_dict() if batch_plan is not None else {},
            resolved_run_spec=resolved_run_spec,
        )

        # 1-6: payload temp write, fsync, digest, atomic replace.
        payload_path, digest, size = write_via_callback_durably(
            directory / PAYLOAD_FILENAME,
            lambda temp: write_payload(temp, payload),
            fault_hook=fault_hook,
        )

        lineage = dict(lineage or {})
        # The schema version is *derived*, not imposed: a package that can
        # prove its training path is v2, one that cannot is v1 and keeps
        # exactly its old meaning. This is what stops v1 being silently
        # redefined, and what stops a v2 package existing without the proof
        # that makes it a v2 package (continuation §5.2).
        schema_version = (
            CHECKPOINT_SCHEMA_VERSION if training_path_id else 1
        )
        manifest = CheckpointManifest(
            schema_version=schema_version,
            checkpoint_id=checkpoint_id,
            run_id=run_id,
            kind=kind,
            created_at=utc_now(),
            model_id=str(identity.get("model_id", "")),
            implementation_id=str(identity.get("implementation_id", "")),
            variant_id=str(identity.get("variant_id", "")),
            phase=cursor.phase,
            subphase=cursor.subphase,
            resume_boundary=RESUME_BOUNDARY_OPTIMIZER_UPDATE,
            cursor=cursor.as_dict(),
            component_inventory=component_inventory(adapter_state),
            structural_config_hash=structural_config_hash,
            dataset_fingerprint=dataset_fingerprint,
            git_revision=(provenance or {}).get("git_revision"),
            submodule_revisions=dict((provenance or {}).get("submodule_revisions", {})),
            payload_uri=PAYLOAD_FILENAME,
            payload_bytes=size,
            payload_sha256=digest,
            parent_run_id=lineage.get("parent_run_id"),
            resumed_from_run_id=lineage.get("resumed_from_run_id"),
            resumed_from_checkpoint_id=lineage.get("resumed_from_checkpoint_id"),
            training_path_id=training_path_id,
            training_path_contract_version=training_path_contract_version,
            certification=dict(certification or {}),
            capabilities=capabilities.as_dict() if capabilities is not None else {},
        )

        if int(manifest.schema_version) >= 2 and not manifest.training_path_id:
            raise CheckpointError(  # pragma: no cover - guarded by derivation above
                "checkpoint schema v2 requires training_path_id"
            )

        # 4-8: manifest temp write, fsync, atomic replace, directory fsync.
        _hook(fault_hook, "before_manifest_write", directory)
        write_bytes_durably(
            directory / MANIFEST_FILENAME,
            (canonical_json(manifest.as_dict()) + "\n").encode("utf-8"),
        )
        fsync_dir(directory)
        _hook(fault_hook, "after_manifest_rename", directory)

        # 9: registry transaction. Only now does the checkpoint become visible.
        if self.registry is not None:
            self.registry.record_checkpoint(
                checkpoint_id=checkpoint_id,
                run_id=run_id,
                kind=str(kind),
                created_at=manifest.created_at,
                phase=cursor.phase,
                global_step=int(cursor.global_update),
                payload_uri=str(payload_path),
                payload_sha256=digest,
                payload_bytes=size,
                manifest_json=canonical_json(manifest.as_dict()),
                complete=True,
                validation_status=str(ValidationStatus.VALID),
                resume_boundary=RESUME_BOUNDARY_OPTIMIZER_UPDATE,
                structural_config_hash=structural_config_hash,
                dataset_fingerprint=dataset_fingerprint,
                implementation_id=manifest.implementation_id,
                variant_id=manifest.variant_id,
                certification_key=str((certification or {}).get("key", "")),
                training_path_id=manifest.training_path_id,
                checkpoint_schema_version=int(manifest.schema_version),
            )
        _hook(fault_hook, "after_registry_insert", directory)

        # 10: event journal.
        if self.event_writer is not None:
            self.event_writer.checkpoint(
                checkpoint_id=checkpoint_id,
                uri=str(payload_path),
                kind=str(kind),
                global_step=int(cursor.global_update),
                sha256=digest,
                bytes=size,
            )

        return SaveResult(
            checkpoint_id=checkpoint_id,
            directory=directory,
            manifest=manifest,
            payload_sha256=digest,
            payload_bytes=size,
        )

    # -- read ----------------------------------------------------------------

    def validate(self, checkpoint_id: str, *, expected_sha256: Optional[str] = None):
        return validate_package(self.directory_for(checkpoint_id), expected_sha256=expected_sha256)

    def load(
        self,
        checkpoint_id: str,
        *,
        expected: Optional[dict[str, Any]] = None,
    ) -> tuple[CheckpointManifest, TrainingCursor, AdapterTrainingState, RngState, dict[str, Any]]:
        """Validate, gate on compatibility, then load.

        The order matters: an invalid or incompatible checkpoint is never
        unpickled.
        """
        directory = self.directory_for(checkpoint_id)
        self._assert_trusted(directory)

        report = validate_package(directory)
        report.raise_if_invalid()
        manifest = report.manifest
        assert manifest is not None

        if expected:
            check_manifest_compatibility(manifest, expected=expected).raise_if_incompatible()

        payload = read_payload(directory / (manifest.payload_uri or PAYLOAD_FILENAME))
        cursor, state, rng = payload_to_state(payload)
        return manifest, cursor, state, rng, payload

    def list_on_disk(self) -> list[str]:
        """Checkpoint ids present on disk with a manifest. Temp files ignored."""
        if not self.root.exists():
            return []
        return sorted(
            entry.name
            for entry in self.root.iterdir()
            if entry.is_dir() and (entry / MANIFEST_FILENAME).exists()
        )


def _hook(hook: Optional[Callable[[str, Path], None]], name: str, path: Path) -> None:
    if hook is not None:
        hook(name, path)
