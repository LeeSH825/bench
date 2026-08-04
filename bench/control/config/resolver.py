"""Validation and resolution of run specifications.

:func:`validate_draft` performs field-level validation and returns *every*
issue found, so a form can render them all at once.

:func:`resolve_run_spec` allocates a fresh :class:`~bench.control.identity.RunId`
and computes the identity hashes. Calling it twice with the same draft yields
two different run ids and the same variant/structural hashes — which is exactly
the invariant acceptance test C-02 checks.
"""

from __future__ import annotations

import json
from dataclasses import replace
from typing import Any, Mapping, Optional, Sequence

from ..canonical import content_hash
from ..identity import (
    ExperimentId,
    ImplementationId,
    InitId,
    ModelId,
    RunId,
    VariantId,
    compute_variant_id,
)
from .schema import (
    CONFIG_SCHEMA_VERSION,
    OPTIMIZERS,
    PRECISIONS,
    ArtifactsSection,
    ConfigValidationError,
    DatasetSection,
    ExecutionSection,
    ExperimentSection,
    InitializationSection,
    OptimizerSection,
    ProvenanceSection,
    ResolvedRunSpec,
    ResumeSection,
    RunSpecDraft,
    RuntimeSection,
    SystemSection,
    TelemetrySection,
    TrainingSection,
    ValidationIssue,
    operational_config_hash,
    structural_config_hash,
)
from ..identity import INIT_MODES


def validate_draft(draft: RunSpecDraft) -> list[ValidationIssue]:
    """Return all field-level validation issues for *draft* (empty == valid)."""
    issues: list[ValidationIssue] = []

    def bad(path: str, code: str, message: str) -> None:
        issues.append(ValidationIssue(path=path, code=code, message=message))

    # -- experiment ---------------------------------------------------------
    try:
        ExperimentId(draft.experiment.experiment_id)
    except ValueError as exc:
        bad("experiment.experiment_id", "invalid_id", str(exc))
    if not str(draft.experiment.name).strip():
        bad("experiment.name", "required", "experiment name must not be empty")

    # -- system -------------------------------------------------------------
    if not str(draft.system.task_id).strip():
        bad("system.task_id", "required", "task_id must not be empty")
    if draft.system.state_dim <= 0:
        bad("system.state_dim", "range", f"state_dim must be > 0, got {draft.system.state_dim}")
    if draft.system.observation_dim <= 0:
        bad(
            "system.observation_dim",
            "range",
            f"observation_dim must be > 0, got {draft.system.observation_dim}",
        )
    if draft.system.sequence_length is not None and draft.system.sequence_length <= 0:
        bad(
            "system.sequence_length",
            "range",
            f"sequence_length must be > 0 when set, got {draft.system.sequence_length}",
        )

    # -- dataset ------------------------------------------------------------
    if not str(draft.dataset.dataset_id).strip():
        bad("dataset.dataset_id", "required", "dataset_id must not be empty")

    # -- training -----------------------------------------------------------
    training = draft.training
    if training.batch_size <= 0:
        bad("training.batch_size", "range", f"batch_size must be > 0, got {training.batch_size}")
    if training.gradient_accumulation_steps <= 0:
        bad(
            "training.gradient_accumulation_steps",
            "range",
            f"must be > 0, got {training.gradient_accumulation_steps}",
        )
    if training.max_updates < 0:
        bad("training.max_updates", "range", f"must be >= 0, got {training.max_updates}")
    if training.validation_interval_updates < 0:
        bad(
            "training.validation_interval_updates",
            "range",
            f"must be >= 0, got {training.validation_interval_updates}",
        )
    # cross-field: training enabled with a zero budget silently does nothing.
    if training.enabled and training.max_updates == 0:
        bad(
            "training.max_updates",
            "cross_field",
            "training.enabled is true but max_updates is 0; either disable training "
            "or set a positive update budget",
        )

    # -- optimizer ----------------------------------------------------------
    if draft.optimizer.name not in OPTIMIZERS:
        bad(
            "optimizer.name",
            "enum",
            f"optimizer must be one of {list(OPTIMIZERS)}, got {draft.optimizer.name!r}",
        )
    if draft.optimizer.learning_rate <= 0 and draft.optimizer.name != "none":
        bad(
            "optimizer.learning_rate",
            "range",
            f"learning_rate must be > 0, got {draft.optimizer.learning_rate}",
        )
    if draft.optimizer.weight_decay < 0:
        bad(
            "optimizer.weight_decay",
            "range",
            f"weight_decay must be >= 0, got {draft.optimizer.weight_decay}",
        )
    # cross-field: a trainable run needs a real optimizer.
    if training.enabled and draft.optimizer.name == "none":
        bad(
            "optimizer.name",
            "cross_field",
            "training.enabled is true but optimizer is 'none'",
        )

    # -- initialization -----------------------------------------------------
    if draft.initialization.mode not in INIT_MODES:
        bad(
            "initialization.mode",
            "enum",
            f"init mode must be one of {list(INIT_MODES)}, got {draft.initialization.mode!r}",
        )
    if draft.initialization.mode in ("pretrained", "loaded") and not (
        draft.initialization.checkpoint_uri or draft.initialization.checkpoint_id
    ):
        bad(
            "initialization.checkpoint_uri",
            "cross_field",
            f"init mode {draft.initialization.mode!r} requires a checkpoint_uri or checkpoint_id",
        )

    # -- resume (schema present, capability absent in this tranche) ---------
    if draft.resume.mode != "none":
        bad(
            "resume.mode",
            "unsupported_capability",
            "exact resume is not implemented in this tranche; only resume.mode='none' "
            "is accepted. The field exists so Phase 2 can enable it without a schema "
            "migration — it is not a working feature.",
        )

    # -- runtime ------------------------------------------------------------
    if draft.runtime.precision not in PRECISIONS:
        bad(
            "runtime.precision",
            "enum",
            f"precision must be one of {list(PRECISIONS)}, got {draft.runtime.precision!r}",
        )
    if draft.runtime.num_workers < 0:
        bad("runtime.num_workers", "range", f"must be >= 0, got {draft.runtime.num_workers}")
    device = str(draft.runtime.device)
    if not (device == "cpu" or device == "cuda" or device.startswith("cuda:") or device == "mps"):
        bad(
            "runtime.device",
            "format",
            f"device must be 'cpu', 'mps', 'cuda' or 'cuda:<index>', got {device!r}",
        )

    # -- telemetry ----------------------------------------------------------
    if draft.telemetry.interval_seconds <= 0:
        bad(
            "telemetry.interval_seconds",
            "range",
            f"must be > 0, got {draft.telemetry.interval_seconds}",
        )

    return issues


def _apply_training_path(draft: RunSpecDraft, *, registry: Any = None) -> RunSpecDraft:
    """Fill in :class:`ExecutionSection` from the certification tuple."""
    from ..training_path import select_training_path
    from ..capabilities import capabilities_for

    capability = capabilities_for(str(draft.model_id))
    decision = select_training_path(
        model_id=str(draft.model_id),
        implementation_id=str(draft.implementation_id),
        training_enabled=bool(draft.training.enabled),
        device=draft.runtime.device,
        precision=draft.runtime.precision,
        num_workers=int(draft.runtime.num_workers),
        gradient_accumulation_steps=int(draft.training.gradient_accumulation_steps or 1),
        trainable=bool(getattr(capability, "trainable", True)),
        registry=registry,
    )
    return replace(
        draft,
        execution=ExecutionSection(
            training_path_id=str(decision.training_path_id),
            training_path_reason_codes=tuple(str(c) for c in decision.reason_codes),
            certification_id=decision.certification_id,
        ),
    )


def resolve_run_spec(
    draft: RunSpecDraft,
    *,
    run_id: Optional[RunId] = None,
    registry: Any = None,
) -> ResolvedRunSpec:
    """Validate *draft* and resolve it into an immutable :class:`ResolvedRunSpec`.

    A fresh :class:`RunId` is allocated unless one is supplied explicitly
    (the worker re-hydrating a spec from disk supplies the stored id).

    Raises :class:`ConfigValidationError` carrying every issue found.
    """
    issues = validate_draft(draft)
    if issues:
        raise ConfigValidationError(issues)

    # The training path is decided exactly once, here, from the full
    # certification tuple — never re-derived in the worker from the model name
    # (ADR-WC-001/005). It feeds the structural hash, so it must be settled
    # before any hashing happens.
    draft = _apply_training_path(draft, registry=registry)

    variant_id = compute_variant_id(
        model_id=draft.model_id,
        implementation_id=draft.implementation_id,
        init=draft.initialization.as_init_id(),
        architecture_fingerprint=draft.architecture_fingerprint,
        structural_config_hash=structural_config_hash(draft),
    )
    resolved_run_id = run_id or RunId.new()
    struct_hash = structural_config_hash(draft)
    oper_hash = operational_config_hash(draft)

    spec = ResolvedRunSpec(
        schema_version=CONFIG_SCHEMA_VERSION,
        run_id=resolved_run_id,
        variant_id=variant_id,
        draft=draft,
        structural_config_hash=struct_hash,
        operational_config_hash=oper_hash,
        # Placeholder; replaced below once the full document exists.
        resolved_spec_hash="sha256:" + "0" * 64,
    )
    document = spec.as_dict()
    document["hashes"]["resolved_spec_hash"] = None
    return ResolvedRunSpec(
        schema_version=spec.schema_version,
        run_id=spec.run_id,
        variant_id=spec.variant_id,
        draft=spec.draft,
        structural_config_hash=struct_hash,
        operational_config_hash=oper_hash,
        resolved_spec_hash=content_hash(document),
    )


# --------------------------------------------------------------------------- #
# Deserialization (round-trip)
# --------------------------------------------------------------------------- #


def _tuple_of_str(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    return tuple(str(item) for item in value)


def draft_from_dict(document: Mapping[str, Any]) -> RunSpecDraft:
    """Rebuild a :class:`RunSpecDraft` from a resolved-spec document.

    Used by the worker (which reads ``resolved_run_spec.json``) and by
    round-trip tests. Rejects strictly-newer schema versions rather than
    guessing (design doc 03 §18).
    """
    version = int(document.get("schema_version", 0))
    if version > CONFIG_SCHEMA_VERSION:
        raise ConfigValidationError(
            [
                ValidationIssue(
                    path="schema_version",
                    code="unsupported_version",
                    message=(
                        f"document schema_version={version} is newer than supported "
                        f"version {CONFIG_SCHEMA_VERSION}; refusing to guess"
                    ),
                )
            ]
        )

    experiment = document.get("experiment", {})
    identity = document.get("identity", {})
    system = document.get("system", {})
    dataset = document.get("dataset", {})
    training = document.get("training", {})
    optimizer = document.get("optimizer", {})
    initialization = document.get("initialization", {})
    resume = document.get("resume", {})
    execution = document.get("execution", {})
    runtime = document.get("runtime", {})
    telemetry = document.get("telemetry", {})
    artifacts = document.get("artifacts", {})
    provenance = document.get("provenance", {})

    return RunSpecDraft(
        experiment=ExperimentSection(
            experiment_id=str(experiment.get("experiment_id")),
            name=str(experiment.get("name", "")),
            description=str(experiment.get("description", "")),
            tags=_tuple_of_str(experiment.get("tags")),
        ),
        model_id=ModelId(str(identity.get("model_id"))),
        implementation_id=ImplementationId(str(identity.get("implementation_id"))),
        system=SystemSection(
            task_id=str(system.get("task_id", "")),
            scenario_id=str(system.get("scenario_id", "")),
            state_dim=int(system.get("state_dim", 0)),
            observation_dim=int(system.get("observation_dim", 0)),
            sequence_length=(
                int(system["sequence_length"])
                if system.get("sequence_length") is not None
                else None
            ),
            scenario_config=dict(system.get("scenario_config", {}) or {}),
        ),
        dataset=DatasetSection(
            dataset_id=str(dataset.get("dataset_id", "")),
            fingerprint=dataset.get("fingerprint"),
            train_uri=dataset.get("train_uri"),
            val_uri=dataset.get("val_uri"),
            test_uri=dataset.get("test_uri"),
            split_seed=int(dataset.get("split_seed", 0)),
        ),
        training=TrainingSection(
            enabled=bool(training.get("enabled", True)),
            max_updates=int(training.get("max_updates", 0)),
            batch_size=int(training.get("batch_size", 1)),
            gradient_accumulation_steps=int(training.get("gradient_accumulation_steps", 1)),
            validation_interval_updates=int(training.get("validation_interval_updates", 0)),
            early_stopping=dict(training.get("early_stopping", {}) or {}),
        ),
        optimizer=OptimizerSection(
            name=str(optimizer.get("name", "adam")),
            learning_rate=float(optimizer.get("learning_rate", 1e-3)),
            weight_decay=float(optimizer.get("weight_decay", 0.0)),
        ),
        initialization=InitializationSection(
            mode=str(initialization.get("mode", "untrained")),
            checkpoint_id=initialization.get("checkpoint_id"),
            checkpoint_uri=initialization.get("checkpoint_uri"),
            source_checkpoint_hash=initialization.get("source_checkpoint_hash"),
            source_run_id=initialization.get("source_run_id"),
        ),
        resume=ResumeSection(
            mode=str(resume.get("mode", "none")),
            checkpoint_id=resume.get("checkpoint_id"),
            allowed_overrides=_tuple_of_str(resume.get("allowed_overrides")),
        ),
        execution=ExecutionSection(
            # Absent field => this spec predates the contract => legacy.
            # Never promoted to resumable retroactively (ADR-WC-002).
            training_path_id=str(execution.get("training_path_id", "legacy_train_v1")),
            training_path_reason_codes=_tuple_of_str(
                execution.get("training_path_reason_codes")
            ),
            certification_id=execution.get("certification_id"),
        ),
        runtime=RuntimeSection(
            device=str(runtime.get("device", "cpu")),
            precision=str(runtime.get("precision", "fp32")),
            deterministic=bool(runtime.get("deterministic", True)),
            seed=int(runtime.get("seed", 0)),
            num_workers=int(runtime.get("num_workers", 0)),
        ),
        telemetry=TelemetrySection(
            enabled=bool(telemetry.get("enabled", True)),
            interval_seconds=float(telemetry.get("interval_seconds", 2.0)),
        ),
        artifacts=ArtifactsSection(
            save_predictions=bool(artifacts.get("save_predictions", False)),
            emit_visualization=bool(artifacts.get("emit_visualization", False)),
            checkpoint_policy=dict(artifacts.get("checkpoint_policy", {}) or {}),
        ),
        provenance=ProvenanceSection(
            git_commit=provenance.get("git_commit"),
            git_dirty=provenance.get("git_dirty"),
            submodule_revisions=dict(provenance.get("submodule_revisions", {}) or {}),
            environment_fingerprint=provenance.get("environment_fingerprint"),
        ),
        bench_context=dict(document.get("bench_context", {}) or {}),
        model_config_extra=dict(document.get("model_config_extra", {}) or {}),
        task_config_extra=dict(document.get("task_config_extra", {}) or {}),
        unsupported_fields=_tuple_of_str(document.get("unsupported_fields")),
        original_config=document.get("original_config"),
        architecture_fingerprint=identity.get("architecture_fingerprint"),
    )


def resolved_from_dict(document: Mapping[str, Any]) -> ResolvedRunSpec:
    """Rebuild a :class:`ResolvedRunSpec` verbatim, preserving its stored ids.

    Unlike :func:`resolve_run_spec` this does **not** allocate a new run id and
    does not recompute hashes — it reproduces exactly what was persisted, so a
    tampered file is detectable by comparing :meth:`ResolvedRunSpec.to_json`.
    """
    draft = draft_from_dict(document)
    identity = document.get("identity", {})
    hashes = document.get("hashes", {})
    return ResolvedRunSpec(
        schema_version=int(document.get("schema_version", CONFIG_SCHEMA_VERSION)),
        run_id=RunId(str(identity.get("run_id"))),
        variant_id=VariantId(str(identity.get("variant_id"))),
        draft=draft,
        structural_config_hash=str(hashes.get("structural_config_hash")),
        operational_config_hash=str(hashes.get("operational_config_hash")),
        resolved_spec_hash=str(hashes.get("resolved_spec_hash")),
    )


def resolved_from_json(text: str) -> ResolvedRunSpec:
    return resolved_from_dict(json.loads(text))
