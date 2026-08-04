"""Exact-resume compatibility gate.

Everything here answers one question: may this checkpoint be resumed into this
configuration *without changing what the numbers mean*? A mismatch is refused
rather than coerced, because a silently-tolerated mismatch produces a run whose
results look continuous and are not (ADR-CSR §4).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from .schema import (
    CHECKPOINT_SCHEMA_VERSION,
    RESUME_BOUNDARY_OPTIMIZER_UPDATE,
    CertificationKey,
    CheckpointCompatibilityError,
    CheckpointManifest,
)

#: Must match exactly for an exact resume (ADR-CSR §4.1).
STRICT_KEYS = (
    "model_id",
    "implementation_id",
    "structural_config_hash",
    "dataset_fingerprint",
)

#: May differ without changing the numerical meaning (ADR-CSR §4.2).
#: Learning rate is deliberately absent: changing it is a forked continuation,
#: not a resume.
ALLOWED_OPERATIONAL_OVERRIDES = frozenset(
    {"log_level", "telemetry_interval", "api_endpoint", "poll_interval", "run_dir"}
)


@dataclass
class CompatibilityReport:
    compatible: bool
    mismatches: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def raise_if_incompatible(self) -> None:
        if not self.compatible:
            raise CheckpointCompatibilityError(
                "checkpoint is not compatible with the requested resume configuration: "
                + "; ".join(self.mismatches)
            )


def check_manifest_compatibility(
    manifest: CheckpointManifest,
    *,
    expected: dict[str, Any],
) -> CompatibilityReport:
    """Compare a manifest against the configuration a resume would run under."""
    mismatches: list[str] = []
    warnings: list[str] = []

    if int(manifest.schema_version) != CHECKPOINT_SCHEMA_VERSION:
        mismatches.append(
            f"schema_version {manifest.schema_version} != {CHECKPOINT_SCHEMA_VERSION}"
        )

    if manifest.resume_boundary != RESUME_BOUNDARY_OPTIMIZER_UPDATE:
        mismatches.append(
            f"resume_boundary {manifest.resume_boundary!r} is not certified; "
            f"only {RESUME_BOUNDARY_OPTIMIZER_UPDATE!r} is supported"
        )

    for key in STRICT_KEYS:
        if key not in expected:
            continue
        want = expected[key]
        got = getattr(manifest, key, None)
        if want in (None, ""):
            continue
        if str(got) != str(want):
            mismatches.append(f"{key}: checkpoint={got!r} requested={want!r}")

    inventory = manifest.component_inventory or {}
    if not inventory.get("model_slots"):
        mismatches.append("payload declares no model slots")
    if not inventory.get("optimizer_slots"):
        mismatches.append(
            "payload declares no optimizer slots; restoring weights without optimizer "
            "state is a warm start, not an exact resume"
        )

    return CompatibilityReport(compatible=not mismatches, mismatches=mismatches, warnings=warnings)


def check_certification(
    manifest: CheckpointManifest,
    *,
    certified: dict[str, Any],
    requested: Optional[dict[str, Any]] = None,
) -> CompatibilityReport:
    """Verify the run's execution envelope is inside the certified envelope.

    A CPU/fp32/single-worker certification says nothing about GPU or AMP or a
    multi-worker loader, and must not be generalised to them (DND-CSR-008).
    """
    mismatches: list[str] = []
    requested = dict(requested or {})

    for key, allowed in certified.items():
        if key not in requested:
            continue
        value = requested[key]
        if isinstance(allowed, (list, tuple, set, frozenset)):
            if value not in allowed:
                mismatches.append(f"{key}={value!r} outside certified {sorted(allowed)!r}")
        elif str(value) != str(allowed):
            mismatches.append(f"{key}={value!r} is not certified (certified: {allowed!r})")

    return CompatibilityReport(compatible=not mismatches, mismatches=mismatches)


def certification_key_for(
    *,
    model_id: str,
    implementation_id: str,
    precision: str = "fp32",
    device_class: str = "cpu",
    num_workers: int = 0,
    training_mode: str = "supervised_single_optimizer",
    training_path_id: str = "control_resumable_v1",
) -> CertificationKey:
    return CertificationKey(
        model_id=model_id,
        implementation_id=implementation_id,
        checkpoint_schema_version=CHECKPOINT_SCHEMA_VERSION,
        resume_boundary=RESUME_BOUNDARY_OPTIMIZER_UPDATE,
        precision=precision,
        device_class=device_class,
        num_workers=num_workers,
        training_mode=training_mode,
        training_path_id=training_path_id,
    )
