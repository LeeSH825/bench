"""Checkpoint v1 typed schema.

This is deliberately *not* the existing ``model.pt``. ``model.pt`` is a
weight-only artifact: loading it is a warm start, because it carries no
optimizer, no RNG, and no cursor. A checkpoint v1 package carries everything
needed to continue training at a certified boundary, and says so in a typed
manifest rather than leaving it to be guessed from a filename (ADR-CSR-001,
DND-CSR-001, DND-CSR-002).
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field, asdict
from typing import Any, Optional

#: The schema version this build *writes*. v2 adds the training-path proof
#: required for write-control child launch (continuation gate B0).
CHECKPOINT_SCHEMA_VERSION = 2

#: Versions this build can read. v1 packages stay readable and restorable
#: exactly as before — their meaning is not retroactively redefined
#: (DND-CSR-001). Anything outside this set is refused, never migrated.
SUPPORTED_CHECKPOINT_SCHEMA_VERSIONS = (1, 2)

#: The first schema version that can prove which training loop produced it.
#: A v1 package may be a perfectly valid artifact and still be ineligible to
#: launch a write-control child, because it cannot make that proof.
WRITE_CONTROL_MIN_SCHEMA_VERSION = 2

#: The only resume boundary this tranche certifies (ADR-CSR-001).
RESUME_BOUNDARY_OPTIMIZER_UPDATE = "optimizer_update"

MANIFEST_FILENAME = "manifest.json"
PAYLOAD_FILENAME = "payload.pt"


class CheckpointKind(str, enum.Enum):
    """Why a checkpoint exists. A typed field, not a filename convention."""

    PERIODIC = "periodic"
    BEST = "best"
    INTERRUPT = "interrupt"
    FINAL = "final"

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.value


class ValidationStatus(str, enum.Enum):
    """Result of validating a checkpoint package on disk."""

    #: Digest, schema, and inventory all check out.
    VALID = "VALID"
    #: Registered but the payload/manifest is missing or fails its digest.
    INVALID = "INVALID"
    #: Found on disk with no registry row; awaiting operator adjudication.
    QUARANTINED = "QUARANTINED"
    #: Not yet checked.
    UNVERIFIED = "UNVERIFIED"

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.value


class EligibilityReason(str, enum.Enum):
    """Why a checkpoint may or may not launch a write-control child."""

    ELIGIBLE = "ELIGIBLE"
    #: Valid, restorable, but predates the training-path contract.
    CHECKPOINT_TRAINING_PATH_UNPROVEN = "CHECKPOINT_TRAINING_PATH_UNPROVEN"
    CHECKPOINT_SCHEMA_NOT_WRITE_CONTROL_CERTIFIED = (
        "CHECKPOINT_SCHEMA_NOT_WRITE_CONTROL_CERTIFIED"
    )
    CHECKPOINT_NOT_VALID = "CHECKPOINT_NOT_VALID"
    TRAINING_PATH_NOT_RESUMABLE = "TRAINING_PATH_NOT_RESUMABLE"
    TRAINING_PATH_MISMATCH = "TRAINING_PATH_MISMATCH"
    PARENT_NOT_TERMINAL = "PARENT_NOT_TERMINAL"
    UNCERTIFIED_TUPLE = "UNCERTIFIED_TUPLE"
    WRONG_CHECKPOINT_KIND = "WRONG_CHECKPOINT_KIND"
    NO_REMAINING_UPDATES = "NO_REMAINING_UPDATES"

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.value


ELIGIBILITY_MESSAGES = {
    EligibilityReason.CHECKPOINT_TRAINING_PATH_UNPROVEN: (
        "This checkpoint predates the training-path contract, so it cannot prove which "
        "training loop produced it. It remains a valid artifact and can still be "
        "inspected and restored, but it cannot launch a resumed child run."
    ),
    EligibilityReason.CHECKPOINT_SCHEMA_NOT_WRITE_CONTROL_CERTIFIED: (
        "Checkpoint schema v1 is not certified for write-control child launch; "
        "schema v2 or later is required."
    ),
    EligibilityReason.CHECKPOINT_NOT_VALID: "The checkpoint failed validation.",
    EligibilityReason.TRAINING_PATH_NOT_RESUMABLE: (
        "Exact resume requires training_path_id=control_resumable_v1."
    ),
    EligibilityReason.TRAINING_PATH_MISMATCH: (
        "The manifest, payload, and registry disagree about the training path."
    ),
    EligibilityReason.PARENT_NOT_TERMINAL: (
        "The parent run is still executing; resume requires a run that has stopped."
    ),
    EligibilityReason.UNCERTIFIED_TUPLE: (
        "No exact-resume certification exists for this tuple."
    ),
    EligibilityReason.WRONG_CHECKPOINT_KIND: (
        "Only interrupt checkpoints can launch a resumed child in this build."
    ),
    EligibilityReason.NO_REMAINING_UPDATES: (
        "The checkpoint is already at the run's update budget; nothing to resume."
    ),
}


class CheckpointError(RuntimeError):
    """Base class for typed checkpoint failures."""


class CheckpointValidationError(CheckpointError):
    """Payload/manifest failed digest, schema, or inventory validation."""


class CheckpointCompatibilityError(CheckpointError):
    """Checkpoint cannot be resumed into the requested configuration."""


class CheckpointUnsupportedError(CheckpointError):
    """Adapter or configuration is outside the certified envelope."""


@dataclass(frozen=True)
class TrainingCursor:
    """Where training is, expressed at the certified resume boundary.

    ``global_update`` is the number of *completed* optimizer updates, so it is
    the same counter the observer emits as ``step``. ``batch_plan_position`` is
    the index into the deterministic batch plan of the batch that will be
    consumed *next*.
    """

    global_update: int = 0
    epoch: int = 0
    batch_plan_position: int = 0
    batch_plan_id: str = ""
    phase: str = "train"
    subphase: Optional[str] = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TrainingCursor":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in dict(data or {}).items() if k in known})


@dataclass(frozen=True)
class CheckpointCapabilities:
    """What an adapter can actually round-trip.

    A blanket ``supports_exact_resume=True`` is exactly the research-integrity
    hazard this type exists to prevent (ADR-CSR-013, DND-CSR-007). The
    certification envelope travels with the claim.
    """

    supports_exact_resume: bool = False
    resume_boundary: Optional[str] = None
    model_slots: tuple[str, ...] = ()
    optimizer_slots: tuple[str, ...] = ()
    scheduler_slots: tuple[str, ...] = ()
    has_grad_scaler: bool = False
    #: Conditional state that exists in this lifecycle and must be captured for
    #: the exact-resume claim to hold (ADR-CSR §3.2).
    required_conditional_state: tuple[str, ...] = ()
    certified_device: str = "cpu"
    certified_precision: str = "fp32"
    certified_num_workers: int = 0
    certified_training_mode: str = "supervised_single_optimizer"
    notes: Optional[str] = None

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        for key in ("model_slots", "optimizer_slots", "scheduler_slots", "required_conditional_state"):
            data[key] = list(data[key])
        return data


@dataclass
class AdapterTrainingState:
    """Everything an adapter must hand over to be resumable.

    Slots are *named*, never positional: the payload reader must not have to
    infer how many state dicts there are (§8).
    """

    model_slots: dict[str, Any] = field(default_factory=dict)
    optimizer_slots: dict[str, Any] = field(default_factory=dict)
    scheduler_slots: dict[str, Any] = field(default_factory=dict)
    grad_scaler: Optional[Any] = None
    #: best metric / best step / best weights, and the early-stop counters.
    best_state: dict[str, Any] = field(default_factory=dict)
    #: validation cadence and history.
    validation_state: dict[str, Any] = field(default_factory=dict)
    #: adapter-specific extras that are part of the numerical path.
    extra_state: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RngState:
    """Captured RNG for every stream that can influence the numerical path."""

    python_state: Any = None
    numpy_state: Any = None
    torch_cpu_state: Any = None
    torch_cuda_states: Optional[list[Any]] = None


@dataclass(frozen=True)
class CertificationKey:
    """The tuple an exact-resume claim is scoped to (ADR-CSR-013)."""

    model_id: str
    implementation_id: str
    checkpoint_schema_version: int
    resume_boundary: str
    precision: str
    device_class: str
    num_workers: int
    training_mode: str
    #: Which training loop the certification was earned on. A certification is
    #: meaningless without it: the same model on the legacy loop is a different
    #: numerical population (continuation §5.3).
    training_path_id: str = "control_resumable_v1"

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def as_key(self) -> str:
        return "|".join(
            str(part)
            for part in (
                self.model_id,
                self.implementation_id,
                self.checkpoint_schema_version,
                self.resume_boundary,
                self.precision,
                self.device_class,
                self.num_workers,
                self.training_mode,
                self.training_path_id,
            )
        )


@dataclass
class CheckpointManifest:
    """The typed description published beside a payload.

    Everything a reader needs to decide "may I resume this, into this config?"
    lives here, so the decision never depends on reading the payload first.
    """

    schema_version: int
    checkpoint_id: str
    run_id: str
    kind: CheckpointKind
    created_at: str

    # identity
    model_id: str = ""
    implementation_id: str = ""
    variant_id: str = ""

    # position
    phase: str = "train"
    subphase: Optional[str] = None
    resume_boundary: str = RESUME_BOUNDARY_OPTIMIZER_UPDATE
    cursor: dict[str, Any] = field(default_factory=dict)

    # inventory: what slots the payload actually contains
    component_inventory: dict[str, list[str]] = field(default_factory=dict)

    # compatibility keys
    structural_config_hash: str = ""
    dataset_fingerprint: str = ""

    # provenance
    git_revision: Optional[str] = None
    submodule_revisions: dict[str, str] = field(default_factory=dict)

    # payload
    payload_uri: str = PAYLOAD_FILENAME
    payload_bytes: int = 0
    payload_sha256: str = ""

    # lineage
    parent_run_id: Optional[str] = None
    resumed_from_run_id: Optional[str] = None
    resumed_from_checkpoint_id: Optional[str] = None

    # training path (schema v2+). None on a v1 package, which is exactly what
    # makes such a package ineligible for write-control child launch.
    training_path_id: Optional[str] = None
    training_path_contract_version: Optional[int] = None

    # certification
    certification: dict[str, Any] = field(default_factory=dict)
    capabilities: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["kind"] = str(self.kind)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CheckpointManifest":
        payload = dict(data or {})
        payload["kind"] = CheckpointKind(str(payload.get("kind", "periodic")))
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in payload.items() if k in known})
