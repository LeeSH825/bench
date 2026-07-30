"""Typed run-configuration schema.

Implemented with stdlib dataclasses rather than Pydantic. Design doc 03 asks for
"Pydantic v2 **or an equivalent typed validation layer**"; dataclasses are used
here for one concrete reason: the worker process must be able to load and
validate its own run spec with **zero** third-party imports beyond what the
training code already needs. Pydantic lives only in the API layer, where FastAPI
requires it anyway. This also enforces DND-009 structurally — nothing in this
module can import a web framework.

Scope honesty (MVP)
-------------------

This schema does **not** model every field of every adapter. It models the
execution contract the control plane needs: identity, dataset, training budget,
optimizer, initialization, runtime, telemetry, artifacts. Everything else is
preserved verbatim in :attr:`RunSpecDraft.model_config_extra` /
:attr:`RunSpecDraft.task_config_extra` and reported through
:attr:`RunSpecDraft.unsupported_fields`.

Unsupported fields are **never silently dropped** — the policy in
:class:`UnknownKeyPolicy` decides whether they are captured with a warning or
rejected outright, and the captured list is carried into the resolved spec and
surfaced by the API.
"""

from __future__ import annotations

import enum
import json
from dataclasses import dataclass, field, replace
from typing import Any, Mapping, Optional, Sequence

from ..canonical import canonical_json, content_hash
from ..identity import (
    ExperimentId,
    ImplementationId,
    InitId,
    ModelId,
    RunId,
    VariantId,
    describe_identity,
    init_from_mapping,
)

#: Schema version of RunSpecDraft/ResolvedRunSpec. Bump on any change to the
#: set or meaning of fields. Readers reject strictly-newer versions.
CONFIG_SCHEMA_VERSION = 1

#: Precisions the control plane understands. `fp32` is the only one certified
#: for numerical reproducibility in this tranche; the others are accepted so
#: that a run can be *described*, and are flagged in capabilities.
PRECISIONS = ("fp32", "fp64", "fp16", "bf16", "amp_fp16", "amp_bf16")

#: Optimizer names the control plane can resolve. Adapters may support more;
#: an unrecognized name is a validation error rather than a silent pass-through
#: because the optimizer class is a *structural* identity input.
OPTIMIZERS = ("adam", "adamw", "sgd", "rmsprop", "none")

#: Training/evaluation phases used for event `phase` tagging.
PHASES = ("setup", "train", "validation", "test", "adapt", "report")


def _jsonable(value: Any) -> Any:
    """Coerce arbitrary YAML-loaded content into canonical-JSON-safe types.

    Suite YAML can contain dates, tuples, and numpy scalars, none of which
    :func:`bench.control.canonical.canonical_json` accepts. Rather than reject the
    whole spec — or silently drop the original config — those values are
    stringified through ``json.dumps(default=str)``.
    """
    if value is None:
        return None
    try:
        return json.loads(json.dumps(value, default=str))
    except (TypeError, ValueError):
        return {"__unserializable__": str(type(value).__name__)}


class UnknownKeyPolicy(str, enum.Enum):
    """What to do with configuration keys the schema does not model.

    ``CAPTURE`` (default) keeps the value, records its path, and lets the run
    proceed — appropriate for the many adapter-specific knobs in the existing
    suite YAML. ``ERROR`` rejects them — appropriate for GUI-authored configs
    where a typo must not become a silently ignored setting.

    There is deliberately no ``DROP`` policy.
    """

    CAPTURE = "capture"
    ERROR = "error"


@dataclass(frozen=True)
class ValidationIssue:
    """One field-level validation problem.

    ``path`` is a dotted document path (``training.batch_size``) so a form UI
    can attach the message to the right control without string matching.
    """

    path: str
    code: str
    message: str

    def as_dict(self) -> dict[str, Any]:
        return {"path": self.path, "code": self.code, "message": self.message}


class ConfigValidationError(ValueError):
    """Aggregate of :class:`ValidationIssue` values.

    Carries *all* issues, not just the first, so a form can render every error
    in one pass.
    """

    def __init__(self, issues: Sequence[ValidationIssue]):
        self.issues = list(issues)
        detail = "; ".join(f"{issue.path}: {issue.message}" for issue in self.issues)
        super().__init__(f"run spec validation failed ({len(self.issues)} issue(s)): {detail}")

    def as_dict(self) -> dict[str, Any]:
        return {"issues": [issue.as_dict() for issue in self.issues]}


# --------------------------------------------------------------------------- #
# Section value objects
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ExperimentSection:
    experiment_id: str
    name: str
    description: str = ""
    tags: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "name": self.name,
            "description": self.description,
            "tags": list(self.tags),
        }


@dataclass(frozen=True)
class SystemSection:
    """The dynamical system / task under estimation."""

    task_id: str
    scenario_id: str
    state_dim: int
    observation_dim: int
    sequence_length: Optional[int] = None
    scenario_config: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "scenario_id": self.scenario_id,
            "state_dim": self.state_dim,
            "observation_dim": self.observation_dim,
            "sequence_length": self.sequence_length,
            "scenario_config": dict(self.scenario_config),
        }


@dataclass(frozen=True)
class DatasetSection:
    dataset_id: str
    fingerprint: Optional[str] = None
    train_uri: Optional[str] = None
    val_uri: Optional[str] = None
    test_uri: Optional[str] = None
    split_seed: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "fingerprint": self.fingerprint,
            "train_uri": self.train_uri,
            "val_uri": self.val_uri,
            "test_uri": self.test_uri,
            "split_seed": self.split_seed,
        }


@dataclass(frozen=True)
class TrainingSection:
    enabled: bool = True
    max_updates: int = 0
    batch_size: int = 1
    gradient_accumulation_steps: int = 1
    validation_interval_updates: int = 0
    early_stopping: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "max_updates": self.max_updates,
            "batch_size": self.batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "validation_interval_updates": self.validation_interval_updates,
            "early_stopping": dict(self.early_stopping),
        }


@dataclass(frozen=True)
class OptimizerSection:
    name: str = "adam"
    learning_rate: float = 1e-3
    weight_decay: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
        }


@dataclass(frozen=True)
class InitializationSection:
    mode: str = "untrained"
    checkpoint_id: Optional[str] = None
    checkpoint_uri: Optional[str] = None
    source_checkpoint_hash: Optional[str] = None
    source_run_id: Optional[str] = None

    def as_init_id(self) -> InitId:
        return init_from_mapping(
            {
                "mode": self.mode,
                "source_checkpoint_hash": self.source_checkpoint_hash,
                "source_run_id": self.source_run_id,
            }
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "checkpoint_id": self.checkpoint_id,
            "checkpoint_uri": self.checkpoint_uri,
            "source_checkpoint_hash": self.source_checkpoint_hash,
            "source_run_id": self.source_run_id,
        }


@dataclass(frozen=True)
class ResumeSection:
    """Resume intent.

    Present in the schema so that Phase 2 can populate it without a migration,
    but this tranche **only** accepts ``mode="none"``. Any other value is a
    validation error: declaring a capability the code does not have is exactly
    the failure mode DND-003/R-05 warn about.
    """

    mode: str = "none"
    checkpoint_id: Optional[str] = None
    allowed_overrides: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "checkpoint_id": self.checkpoint_id,
            "allowed_overrides": list(self.allowed_overrides),
        }


@dataclass(frozen=True)
class RuntimeSection:
    device: str = "cpu"
    precision: str = "fp32"
    deterministic: bool = True
    seed: int = 0
    num_workers: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "device": self.device,
            "precision": self.precision,
            "deterministic": self.deterministic,
            "seed": self.seed,
            "num_workers": self.num_workers,
        }


@dataclass(frozen=True)
class TelemetrySection:
    enabled: bool = True
    interval_seconds: float = 2.0

    def as_dict(self) -> dict[str, Any]:
        return {"enabled": self.enabled, "interval_seconds": self.interval_seconds}


@dataclass(frozen=True)
class ArtifactsSection:
    save_predictions: bool = False
    emit_visualization: bool = False
    checkpoint_policy: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "save_predictions": self.save_predictions,
            "emit_visualization": self.emit_visualization,
            "checkpoint_policy": dict(self.checkpoint_policy),
        }


@dataclass(frozen=True)
class ProvenanceSection:
    git_commit: Optional[str] = None
    git_dirty: Optional[bool] = None
    submodule_revisions: Mapping[str, Any] = field(default_factory=dict)
    environment_fingerprint: Optional[str] = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "git_commit": self.git_commit,
            "git_dirty": self.git_dirty,
            "submodule_revisions": dict(self.submodule_revisions),
            "environment_fingerprint": self.environment_fingerprint,
        }


# --------------------------------------------------------------------------- #
# Draft (mutable intent) and Resolved (immutable execution contract)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class RunSpecDraft:
    """A validated but not-yet-resolved run specification.

    A draft has no ``run_id`` and no hashes: it describes *what* to run, not
    *which execution*. :func:`bench.control.config.resolver.resolve_run_spec`
    turns it into a :class:`ResolvedRunSpec` by allocating identity.
    """

    experiment: ExperimentSection
    model_id: ModelId
    implementation_id: ImplementationId
    system: SystemSection
    dataset: DatasetSection
    training: TrainingSection = field(default_factory=TrainingSection)
    optimizer: OptimizerSection = field(default_factory=OptimizerSection)
    initialization: InitializationSection = field(default_factory=InitializationSection)
    resume: ResumeSection = field(default_factory=ResumeSection)
    runtime: RuntimeSection = field(default_factory=RuntimeSection)
    telemetry: TelemetrySection = field(default_factory=TelemetrySection)
    artifacts: ArtifactsSection = field(default_factory=ArtifactsSection)
    provenance: ProvenanceSection = field(default_factory=ProvenanceSection)

    #: Bench-specific execution context preserved verbatim (track_id, plan,
    #: adapter dotted path, …). Carried into the run spec so the worker can
    #: reconstruct an equivalent legacy invocation.
    bench_context: Mapping[str, Any] = field(default_factory=dict)
    #: Adapter-specific model config not modelled by this schema.
    model_config_extra: Mapping[str, Any] = field(default_factory=dict)
    #: Task/generator config not modelled by this schema.
    task_config_extra: Mapping[str, Any] = field(default_factory=dict)
    #: Dotted paths of keys that the schema does not model, for UI display.
    unsupported_fields: tuple[str, ...] = ()
    #: The untouched source document this draft was built from (design doc 03:
    #: "original config attachment 보존").
    original_config: Optional[Mapping[str, Any]] = None
    architecture_fingerprint: Optional[str] = None

    def with_provenance(self, provenance: ProvenanceSection) -> "RunSpecDraft":
        return replace(self, provenance=provenance)


@dataclass(frozen=True)
class ResolvedRunSpec:
    """Immutable execution contract shared by CLI launch and (future) GUI launch.

    Written to ``resolved_run_spec.json`` in the run directory and handed to the
    worker process. Everything needed to execute and to identify the run is
    here; the worker never re-reads the suite YAML.
    """

    schema_version: int
    run_id: RunId
    variant_id: VariantId
    draft: RunSpecDraft
    structural_config_hash: str
    operational_config_hash: str
    resolved_spec_hash: str

    # -- identity conveniences ------------------------------------------------

    @property
    def experiment_id(self) -> ExperimentId:
        return ExperimentId(self.draft.experiment.experiment_id)

    @property
    def model_id(self) -> ModelId:
        return self.draft.model_id

    @property
    def implementation_id(self) -> ImplementationId:
        return self.draft.implementation_id

    @property
    def init_id(self) -> InitId:
        return self.draft.initialization.as_init_id()

    # -- serialization --------------------------------------------------------

    def as_dict(self) -> dict[str, Any]:
        """Full JSON-serializable document (design doc 03 §4 shape)."""
        draft = self.draft
        return {
            "schema_version": self.schema_version,
            "experiment": draft.experiment.as_dict(),
            "identity": {
                **describe_identity(
                    model_id=draft.model_id,
                    implementation_id=draft.implementation_id,
                    init=self.init_id,
                    variant_id=self.variant_id,
                ),
                "run_id": self.run_id.value,
                "architecture_fingerprint": draft.architecture_fingerprint,
            },
            "system": draft.system.as_dict(),
            "dataset": draft.dataset.as_dict(),
            "training": draft.training.as_dict(),
            "optimizer": draft.optimizer.as_dict(),
            "initialization": draft.initialization.as_dict(),
            "resume": draft.resume.as_dict(),
            "runtime": draft.runtime.as_dict(),
            "telemetry": draft.telemetry.as_dict(),
            "artifacts": draft.artifacts.as_dict(),
            "provenance": draft.provenance.as_dict(),
            "bench_context": dict(draft.bench_context),
            "model_config_extra": dict(draft.model_config_extra),
            "task_config_extra": dict(draft.task_config_extra),
            "unsupported_fields": list(draft.unsupported_fields),
            # The untouched source document travels *with* the spec. The worker
            # reads only resolved_run_spec.json, so anything omitted here is
            # invisible to it — dropping this is what made the suite executor
            # unable to reconstruct its task/model entries.
            "original_config": _jsonable(draft.original_config),
            "hashes": {
                "structural_config_hash": self.structural_config_hash,
                "operational_config_hash": self.operational_config_hash,
                "resolved_spec_hash": self.resolved_spec_hash,
            },
        }

    def to_json(self) -> str:
        """Canonical JSON text (stable key order, no incidental whitespace)."""
        return canonical_json(self.as_dict())

    def summary(self) -> dict[str, Any]:
        """Compact projection for run tables and API list responses."""
        return {
            "run_id": self.run_id.value,
            "experiment_id": self.draft.experiment.experiment_id,
            "experiment_name": self.draft.experiment.name,
            "model_id": self.draft.model_id.value,
            "implementation_id": self.draft.implementation_id.value,
            "init_id": self.init_id.mode,
            "variant_id": self.variant_id.value,
            "variant_id_short": self.variant_id.short,
            "task_id": self.draft.system.task_id,
            "scenario_id": self.draft.system.scenario_id,
            "seed": self.draft.runtime.seed,
            "device": self.draft.runtime.device,
            "training_enabled": self.draft.training.enabled,
            "max_updates": self.draft.training.max_updates,
        }


# --------------------------------------------------------------------------- #
# Structural / operational partition
# --------------------------------------------------------------------------- #

#: Fields that change *what is computed*. A difference here means two runs are
#: not the same experiment and an exact resume must be refused
#: (design doc 03 §4.1).
STRUCTURAL_SECTIONS = (
    "model_identity",
    "system",
    "dataset",
    "training_structure",
    "optimizer",
    "initialization",
    "precision_semantics",
    "model_config_extra",
    "task_config_extra",
)

#: Fields that change *how a run is observed or labelled*. Overriding these on
#: resume is permitted (with an audit record); they never enter the structural
#: hash.
OPERATIONAL_SECTIONS = (
    "experiment_presentation",
    "telemetry",
    "artifacts",
    "runtime_placement",
)


def structural_document(draft: RunSpecDraft) -> dict[str, Any]:
    """The document hashed into ``structural_config_hash``.

    Deliberate exclusions and why:

    * ``runtime.seed`` — a different seed is the *same* experiment; averaging
      over seeds is the whole point of the benchmark.
    * ``runtime.device`` / ``num_workers`` — placement, not semantics.
    * ``telemetry`` / ``artifacts`` — observation, not computation.
    * ``experiment`` name/description/tags — labels.

    Deliberate inclusions:

    * ``runtime.precision`` and ``runtime.deterministic`` — these do change
      numerics, so they are structural even though they look operational.
    * ``model_config_extra`` / ``task_config_extra`` — unmodelled adapter knobs
      absolutely change results, so they are hashed verbatim. This is why they
      are captured rather than dropped.
    """
    return {
        "schema_version": CONFIG_SCHEMA_VERSION,
        "model_identity": {
            "model_id": draft.model_id.value,
            "implementation_id": draft.implementation_id.value,
            "architecture_fingerprint": draft.architecture_fingerprint,
        },
        "system": draft.system.as_dict(),
        "dataset": {
            "dataset_id": draft.dataset.dataset_id,
            "fingerprint": draft.dataset.fingerprint,
            "split_seed": draft.dataset.split_seed,
        },
        "training_structure": {
            "enabled": draft.training.enabled,
            "max_updates": draft.training.max_updates,
            "batch_size": draft.training.batch_size,
            "gradient_accumulation_steps": draft.training.gradient_accumulation_steps,
            "validation_interval_updates": draft.training.validation_interval_updates,
            "early_stopping": dict(draft.training.early_stopping),
        },
        "optimizer": draft.optimizer.as_dict(),
        "initialization": draft.initialization.as_dict(),
        "precision_semantics": {
            "precision": draft.runtime.precision,
            "deterministic": draft.runtime.deterministic,
        },
        "model_config_extra": dict(draft.model_config_extra),
        "task_config_extra": dict(draft.task_config_extra),
    }


def operational_document(draft: RunSpecDraft) -> dict[str, Any]:
    """The document hashed into ``operational_config_hash``."""
    return {
        "schema_version": CONFIG_SCHEMA_VERSION,
        "experiment_presentation": {
            "name": draft.experiment.name,
            "description": draft.experiment.description,
            "tags": list(draft.experiment.tags),
        },
        "telemetry": draft.telemetry.as_dict(),
        "artifacts": draft.artifacts.as_dict(),
        "runtime_placement": {
            "device": draft.runtime.device,
            "num_workers": draft.runtime.num_workers,
            "seed": draft.runtime.seed,
        },
    }


def structural_config_hash(draft: RunSpecDraft) -> str:
    return content_hash(structural_document(draft))


def operational_config_hash(draft: RunSpecDraft) -> str:
    return content_hash(operational_document(draft))
