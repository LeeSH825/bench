"""Crash reconciliation for checkpoint packages.

Because publication makes bytes durable before cataloguing them, a crash
produces one of a small set of knowable states. This module names each one
rather than guessing, and it never silently deletes a researcher's bytes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from .atomic import list_temp_files
from .schema import MANIFEST_FILENAME, ValidationStatus
from .validation import validate_package


@dataclass
class ReconciliationFinding:
    checkpoint_id: str
    kind: str
    detail: str
    action: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "checkpoint_id": self.checkpoint_id,
            "kind": self.kind,
            "detail": self.detail,
            "action": self.action,
        }


@dataclass
class ReconciliationReport:
    run_id: str = ""
    catalogued: list[str] = field(default_factory=list)
    quarantined: list[str] = field(default_factory=list)
    invalidated: list[str] = field(default_factory=list)
    temp_leftovers: list[str] = field(default_factory=list)
    findings: list[ReconciliationFinding] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "catalogued": list(self.catalogued),
            "quarantined": list(self.quarantined),
            "invalidated": list(self.invalidated),
            "temp_leftovers": list(self.temp_leftovers),
            "findings": [f.as_dict() for f in self.findings],
        }


def reconcile_run_checkpoints(
    *,
    run_id: str,
    run_dir: Path,
    registry: Optional[Any] = None,
    adopt_orphan_packages: bool = True,
) -> ReconciliationReport:
    """Reconcile on-disk checkpoint packages against the catalog.

    Three cases, each handled explicitly:

    * **files, no row** — the crash happened between publication and the
      registry transaction. The bytes are complete and digest-verified, so the
      package is adopted (or quarantined if it does not validate).
    * **row, no valid files** — the catalog promises state that is not there.
      Marked INVALID; never silently dropped, because the row is evidence.
    * **temp leftovers** — an interrupted write. Reported, never counted as a
      checkpoint, and not deleted here: a temp file may belong to a live writer.
    """
    report = ReconciliationReport(run_id=run_id)
    root = Path(run_dir) / "checkpoints"

    rows = {}
    if registry is not None:
        for row in registry.list_checkpoints(run_id):
            rows[str(row.get("checkpoint_id"))] = row

    on_disk = []
    if root.exists():
        on_disk = [e.name for e in root.iterdir() if e.is_dir() and (e / MANIFEST_FILENAME).exists()]

    for temp in list_temp_files(root):
        report.temp_leftovers.append(str(temp))
        report.findings.append(
            ReconciliationFinding(
                checkpoint_id="",
                kind="temp_leftover",
                detail=f"interrupted write: {temp.name}",
                action="reported; not deleted (may belong to a live writer)",
            )
        )

    for checkpoint_id in sorted(on_disk):
        result = validate_package(root / checkpoint_id)
        if checkpoint_id in rows:
            if not result.valid:
                report.invalidated.append(checkpoint_id)
                if registry is not None:
                    registry.set_checkpoint_validation(
                        checkpoint_id, status=str(ValidationStatus.INVALID),
                        detail="; ".join(result.errors),
                    )
                report.findings.append(
                    ReconciliationFinding(
                        checkpoint_id=checkpoint_id,
                        kind="catalogued_but_invalid",
                        detail="; ".join(result.errors),
                        action="marked INVALID",
                    )
                )
            continue

        # Files with no row.
        if result.valid and adopt_orphan_packages and registry is not None and result.manifest:
            manifest = result.manifest
            registry.record_checkpoint(
                checkpoint_id=checkpoint_id,
                run_id=run_id,
                kind=str(manifest.kind),
                created_at=manifest.created_at,
                phase=manifest.phase,
                global_step=int((manifest.cursor or {}).get("global_update", 0)),
                payload_uri=str(root / checkpoint_id / manifest.payload_uri),
                payload_sha256=result.payload_sha256 or "",
                payload_bytes=int(result.payload_bytes or 0),
                manifest_json="",
                complete=True,
                validation_status=str(ValidationStatus.VALID),
                resume_boundary=manifest.resume_boundary,
                structural_config_hash=manifest.structural_config_hash,
                dataset_fingerprint=manifest.dataset_fingerprint,
                implementation_id=manifest.implementation_id,
                variant_id=manifest.variant_id,
                certification_key=str((manifest.certification or {}).get("key", "")),
            )
            report.catalogued.append(checkpoint_id)
            report.findings.append(
                ReconciliationFinding(
                    checkpoint_id=checkpoint_id,
                    kind="adopted_orphan_package",
                    detail="complete package with no registry row; digest verified",
                    action="catalogued",
                )
            )
        else:
            report.quarantined.append(checkpoint_id)
            report.findings.append(
                ReconciliationFinding(
                    checkpoint_id=checkpoint_id,
                    kind="quarantined",
                    detail="; ".join(result.errors) or "no registry row",
                    action="quarantined for operator adjudication",
                )
            )

    # Rows whose package is gone entirely.
    for checkpoint_id, row in rows.items():
        if checkpoint_id in on_disk:
            continue
        report.invalidated.append(checkpoint_id)
        if registry is not None:
            registry.set_checkpoint_validation(
                checkpoint_id, status=str(ValidationStatus.INVALID),
                detail="payload/manifest missing on disk",
            )
        report.findings.append(
            ReconciliationFinding(
                checkpoint_id=checkpoint_id,
                kind="row_without_package",
                detail="catalogued checkpoint has no package on disk",
                action="marked INVALID",
            )
        )

    return report
