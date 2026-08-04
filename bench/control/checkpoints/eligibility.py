"""Write-control resume eligibility.

Three questions that are deliberately *not* the same question:

1. Is the checkpoint a **valid artifact**? (digest, schema, inventory)
2. Can it be **restored** at adapter/service level?
3. May it **launch a write-control child run**?

A schema v1 package can answer yes to the first two and no to the third,
because it predates the training-path contract and cannot prove which loop
produced it. Saying so with a reason code is the whole point — silently
treating it as eligible would let a legacy run acquire an exact-resume claim it
never earned (continuation §5.2).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from ..registry.schema import RunState
from .schema import (
    WRITE_CONTROL_MIN_SCHEMA_VERSION,
    CheckpointKind,
    CheckpointManifest,
    EligibilityReason,
    ELIGIBILITY_MESSAGES,
)

#: The only training path that may launch a resumed child.
RESUMABLE_PATH_ID = "control_resumable_v1"


@dataclass
class EligibilityReport:
    """Machine-readable answer to question 3."""

    eligible: bool
    reason_codes: list[str] = field(default_factory=list)
    messages: list[str] = field(default_factory=list)
    certification_id: Optional[str] = None
    training_path_id: Optional[str] = None
    checkpoint_schema_version: Optional[int] = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "eligible": bool(self.eligible),
            "reason_codes": list(self.reason_codes),
            "messages": list(self.messages),
            "certification_id": self.certification_id,
            "training_path_id": self.training_path_id,
            "checkpoint_schema_version": self.checkpoint_schema_version,
        }

    def add(self, reason: EligibilityReason) -> None:
        self.eligible = False
        self.reason_codes.append(str(reason))
        self.messages.append(ELIGIBILITY_MESSAGES.get(reason, str(reason)))


def evaluate_resume_eligibility(
    *,
    manifest: Optional[CheckpointManifest] = None,
    checkpoint_row: Optional[dict[str, Any]] = None,
    parent_run: Optional[Any] = None,
    validation_ok: bool = True,
    registry: Any = None,
    require_interrupt_kind: bool = True,
) -> EligibilityReport:
    """Decide whether this checkpoint may launch a write-control child.

    Answers from the manifest and the registry only — the payload is never
    unpickled to decide eligibility.
    """
    from .certification import is_certified
    from .compatibility import certification_key_for

    report = EligibilityReport(eligible=True)

    if not validation_ok:
        report.add(EligibilityReason.CHECKPOINT_NOT_VALID)
        return report

    schema_version = None
    training_path = None
    kind = None
    model_id = implementation_id = ""

    if manifest is not None:
        schema_version = int(manifest.schema_version)
        training_path = manifest.training_path_id
        kind = str(manifest.kind)
        model_id = manifest.model_id
        implementation_id = manifest.implementation_id
    elif checkpoint_row is not None:
        schema_version = int(checkpoint_row.get("checkpoint_schema_version") or 1)
        training_path = checkpoint_row.get("training_path_id")
        kind = str(checkpoint_row.get("kind") or "")
        model_id = str(checkpoint_row.get("model_id") or "")
        implementation_id = str(checkpoint_row.get("implementation_id") or "")

    report.checkpoint_schema_version = schema_version
    report.training_path_id = training_path

    # A pre-contract package: valid, restorable, but unprovable.
    if schema_version is not None and schema_version < WRITE_CONTROL_MIN_SCHEMA_VERSION:
        report.add(EligibilityReason.CHECKPOINT_SCHEMA_NOT_WRITE_CONTROL_CERTIFIED)
        report.add(EligibilityReason.CHECKPOINT_TRAINING_PATH_UNPROVEN)
        return report

    if not training_path:
        report.add(EligibilityReason.CHECKPOINT_TRAINING_PATH_UNPROVEN)
        return report

    if str(training_path) != RESUMABLE_PATH_ID:
        report.add(EligibilityReason.TRAINING_PATH_NOT_RESUMABLE)

    if require_interrupt_kind and kind and kind != str(CheckpointKind.INTERRUPT):
        report.add(EligibilityReason.WRONG_CHECKPOINT_KIND)

    if parent_run is not None:
        # The parent must have stopped. INTERRUPTED is the expected case;
        # other non-executing states are allowed by the service layer but the
        # UI-facing MVP targets interrupt checkpoints (ADR-WC-012).
        executing = {
            RunState.CREATED, RunState.VALIDATING, RunState.QUEUED,
            RunState.STARTING, RunState.RUNNING, RunState.STOP_REQUESTED,
            RunState.CHECKPOINTING,
        }
        if getattr(parent_run, "state", None) in executing:
            report.add(EligibilityReason.PARENT_NOT_TERMINAL)

        parent_path = getattr(parent_run, "training_path_id", None)
        if parent_path and str(parent_path) != str(training_path):
            report.add(EligibilityReason.TRAINING_PATH_MISMATCH)

    if model_id and implementation_id:
        training_mode = (
            "supervised_single_optimizer_split_deviation"
            if model_id == "split_knet"
            else "supervised_single_optimizer"
        )
        certified = is_certified(
            model_id=model_id,
            implementation_id=implementation_id,
            precision="fp32",
            device_class="cpu",
            num_workers=0,
            training_mode=training_mode,
            registry=registry,
        )
        if certified:
            report.certification_id = certification_key_for(
                model_id=model_id,
                implementation_id=implementation_id,
                precision="fp32",
                device_class="cpu",
                num_workers=0,
                training_mode=training_mode,
            ).as_key()
        else:
            report.add(EligibilityReason.UNCERTIFIED_TUPLE)

    return report
