"""Checkpoint package validation.

A checkpoint is only ever loaded after this module says its bytes are the bytes
that were catalogued. Digest first, structure second, load last.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from .atomic import TEMP_SUFFIX, sha256_file
from .schema import (
    CHECKPOINT_SCHEMA_VERSION,
    SUPPORTED_CHECKPOINT_SCHEMA_VERSIONS,
    MANIFEST_FILENAME,
    PAYLOAD_FILENAME,
    CheckpointManifest,
    CheckpointValidationError,
    ValidationStatus,
)


@dataclass
class ValidationReport:
    status: ValidationStatus
    checkpoint_id: str = ""
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    manifest: Optional[CheckpointManifest] = None
    payload_sha256: Optional[str] = None
    payload_bytes: Optional[int] = None

    @property
    def valid(self) -> bool:
        return self.status is ValidationStatus.VALID

    def raise_if_invalid(self) -> None:
        if not self.valid:
            raise CheckpointValidationError(
                f"checkpoint {self.checkpoint_id or '<unknown>'} is {self.status}: "
                + "; ".join(self.errors)
            )

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": str(self.status),
            "checkpoint_id": self.checkpoint_id,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "payload_sha256": self.payload_sha256,
            "payload_bytes": self.payload_bytes,
        }


def read_manifest(path: Path) -> CheckpointManifest:
    data = json.loads(path.read_text(encoding="utf-8"))
    return CheckpointManifest.from_dict(data)


def validate_package(directory: Path, *, expected_sha256: Optional[str] = None) -> ValidationReport:
    """Validate a checkpoint directory containing a manifest and payload.

    This never imports torch and never unpickles: it is safe to run against an
    untrusted-looking directory precisely because it only reads JSON and hashes
    bytes. Loading happens elsewhere, after this returns VALID.
    """
    errors: list[str] = []
    warnings: list[str] = []
    manifest_path = directory / MANIFEST_FILENAME

    if not directory.exists():
        return ValidationReport(status=ValidationStatus.INVALID, errors=[f"missing directory {directory}"])

    leftovers = [p.name for p in directory.glob(f"*{TEMP_SUFFIX}")]
    if leftovers:
        # Not fatal: a temp file is an incomplete write, never a checkpoint.
        warnings.append(f"leftover temp files ignored: {sorted(leftovers)}")

    if not manifest_path.exists():
        return ValidationReport(
            status=ValidationStatus.INVALID,
            errors=[f"missing {MANIFEST_FILENAME}"],
            warnings=warnings,
        )

    try:
        manifest = read_manifest(manifest_path)
    except Exception as exc:
        return ValidationReport(
            status=ValidationStatus.INVALID,
            errors=[f"unreadable manifest: {type(exc).__name__}: {exc}"],
            warnings=warnings,
        )

    checkpoint_id = manifest.checkpoint_id

    # v1 packages stay readable: their meaning is not retroactively redefined.
    # An unknown or future version is refused rather than guessed at.
    if int(manifest.schema_version) not in SUPPORTED_CHECKPOINT_SCHEMA_VERSIONS:
        errors.append(
            f"manifest schema_version={manifest.schema_version} is not readable by this "
            f"build (supported: {list(SUPPORTED_CHECKPOINT_SCHEMA_VERSIONS)}); "
            "silent migration is not performed"
        )
    elif int(manifest.schema_version) >= 2 and not manifest.training_path_id:
        errors.append("schema v2 manifest is missing the required training_path_id")

    payload_path = directory / (manifest.payload_uri or PAYLOAD_FILENAME)
    if not payload_path.exists():
        errors.append(f"missing payload {payload_path.name}")
        return ValidationReport(
            status=ValidationStatus.INVALID,
            checkpoint_id=checkpoint_id,
            errors=errors,
            warnings=warnings,
            manifest=manifest,
        )

    actual_size = payload_path.stat().st_size
    actual_digest = sha256_file(payload_path)

    if manifest.payload_sha256 and actual_digest != manifest.payload_sha256:
        errors.append(
            f"payload digest mismatch: manifest={manifest.payload_sha256[:16]}… "
            f"actual={actual_digest[:16]}…"
        )
    if expected_sha256 and actual_digest != expected_sha256:
        errors.append("payload digest does not match the registry row")
    if manifest.payload_bytes and int(manifest.payload_bytes) != actual_size:
        errors.append(f"payload size mismatch: manifest={manifest.payload_bytes} actual={actual_size}")

    inventory = manifest.component_inventory or {}
    if not inventory.get("model_slots"):
        errors.append("manifest declares no model slots")
    if not inventory.get("optimizer_slots"):
        errors.append("manifest declares no optimizer slots")

    status = ValidationStatus.VALID if not errors else ValidationStatus.INVALID
    return ValidationReport(
        status=status,
        checkpoint_id=checkpoint_id,
        errors=errors,
        warnings=warnings,
        manifest=manifest,
        payload_sha256=actual_digest,
        payload_bytes=actual_size,
    )
