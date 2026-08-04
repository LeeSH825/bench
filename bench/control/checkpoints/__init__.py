"""Checkpoint v1: durable, typed, resumable training state.

Import surface for the rest of the control plane. Nothing here imports a UI
framework; the dashboard reads checkpoints through the read-only API like any
other client.
"""

from __future__ import annotations

from .batchplan import BatchPlan, dataset_fingerprint
from .certification import (
    CERTIFIED,
    NOT_CERTIFIED,
    CertificationRecord,
    certification_matrix,
    is_certified,
    seed_certifications,
)
from .eligibility import (
    EligibilityReport,
    evaluate_resume_eligibility,
)
from .compatibility import (
    ALLOWED_OPERATIONAL_OVERRIDES,
    CompatibilityReport,
    certification_key_for,
    check_certification,
    check_manifest_compatibility,
)
from .payload import capture_rng, restore_rng
from .reconciliation import ReconciliationReport, reconcile_run_checkpoints
from .schema import (
    CHECKPOINT_SCHEMA_VERSION,
    SUPPORTED_CHECKPOINT_SCHEMA_VERSIONS,
    WRITE_CONTROL_MIN_SCHEMA_VERSION,
    EligibilityReason,
    MANIFEST_FILENAME,
    PAYLOAD_FILENAME,
    RESUME_BOUNDARY_OPTIMIZER_UPDATE,
    AdapterTrainingState,
    CertificationKey,
    CheckpointCapabilities,
    CheckpointCompatibilityError,
    CheckpointError,
    CheckpointKind,
    CheckpointManifest,
    CheckpointUnsupportedError,
    CheckpointValidationError,
    RngState,
    TrainingCursor,
    ValidationStatus,
)
from .service import CheckpointService, SaveResult, new_checkpoint_id
from .validation import ValidationReport, validate_package

__all__ = [
    "ALLOWED_OPERATIONAL_OVERRIDES",
    "CERTIFIED",
    "NOT_CERTIFIED",
    "CertificationRecord",
    "certification_matrix",
    "is_certified",
    "seed_certifications",
    "AdapterTrainingState",
    "BatchPlan",
    "CHECKPOINT_SCHEMA_VERSION",
    "SUPPORTED_CHECKPOINT_SCHEMA_VERSIONS",
    "WRITE_CONTROL_MIN_SCHEMA_VERSION",
    "EligibilityReason",
    "CertificationKey",
    "CheckpointCapabilities",
    "CheckpointCompatibilityError",
    "CheckpointError",
    "CheckpointKind",
    "CheckpointManifest",
    "CheckpointService",
    "CheckpointUnsupportedError",
    "CheckpointValidationError",
    "CompatibilityReport",
    "EligibilityReport",
    "evaluate_resume_eligibility",
    "MANIFEST_FILENAME",
    "PAYLOAD_FILENAME",
    "RESUME_BOUNDARY_OPTIMIZER_UPDATE",
    "ReconciliationReport",
    "RngState",
    "SaveResult",
    "TrainingCursor",
    "ValidationReport",
    "ValidationStatus",
    "capture_rng",
    "certification_key_for",
    "check_certification",
    "check_manifest_compatibility",
    "dataset_fingerprint",
    "new_checkpoint_id",
    "reconcile_run_checkpoints",
    "restore_rng",
    "validate_package",
]
