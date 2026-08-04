"""Adapter from the existing bench suite YAML to the typed control-plane schema.

The suite YAML format (`bench/configs/*.yaml`) is **not** changed by this
tranche and `bench/runners/run_suite.py` keeps parsing it exactly as before.
This module is a one-way projection: suite dict → :class:`RunSpecDraft`. It
exists so that the control plane can describe, identify, and observe a run that
the existing CLI could equally well have launched.

Supported-field policy
----------------------

The suite format is large, loosely validated, and adapter-specific
(see audit §7.1). Rather than pretend to model all of it, this module declares
which keys it *interprets* and preserves everything else verbatim:

* interpreted task keys → :const:`TASK_SUPPORTED_KEYS`
* interpreted model keys → :const:`MODEL_SUPPORTED_KEYS`
* interpreted runner keys → :const:`RUNNER_SUPPORTED_KEYS`

Anything outside those sets is copied into ``task_config_extra`` /
``model_config_extra`` (so it still participates in the structural hash — an
uninterpreted key can absolutely change results) and its dotted path is listed
in ``unsupported_fields`` so the UI can show exactly what the control plane did
not understand. Nothing is dropped (design doc 06, C-05).
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Optional, Sequence

from ..capabilities import implementation_id_for
from ..identity import ExperimentId, ImplementationId, InitId, ModelId
from .schema import (
    ArtifactsSection,
    DatasetSection,
    ExperimentSection,
    InitializationSection,
    OptimizerSection,
    ProvenanceSection,
    ResumeSection,
    RunSpecDraft,
    RuntimeSection,
    SystemSection,
    TelemetrySection,
    TrainingSection,
    UnknownKeyPolicy,
    ConfigValidationError,
    ValidationIssue,
)

#: Task keys this module interprets into typed fields.
TASK_SUPPORTED_KEYS = frozenset(
    {
        "task_id",
        "task_name",
        "x_dim",
        "y_dim",
        "sequence_length_T",
        "dataset_sizes",
        "enabled",
    }
)

#: Model keys this module interprets into typed fields.
MODEL_SUPPORTED_KEYS = frozenset(
    {
        "model_id",
        "display_name",
        "batch_size",
        "lr",
        "weight_decay",
        "val_eval_interval_updates",
        "enabled",
    }
)

#: Runner keys this module interprets into typed fields.
RUNNER_SUPPORTED_KEYS = frozenset(
    {
        "device",
        "precision",
        "deterministic",
        "budget",
        "early_stopping",
        "tracks",
        "enabled_policy",
        "data_mode",
        "model_cache_dir",
    }
)


def _unsupported_paths(
    section: Mapping[str, Any], supported: Iterable[str], prefix: str
) -> list[str]:
    known = set(supported)
    return sorted(f"{prefix}.{key}" for key in section if key not in known)


def _extra(section: Mapping[str, Any], supported: Iterable[str]) -> dict[str, Any]:
    known = set(supported)
    return {key: value for key, value in section.items() if key not in known}


def _scenario_id_for(task: Mapping[str, Any], scenario_settings: Mapping[str, Any]) -> str:
    """Compute the scenario id exactly the way ``run_suite`` does.

    Delegates to the runner's own private helpers rather than reimplementing the
    canonicalization, so that a control-plane run and a CLI run of the same
    config agree on ``scenario_id``. The import is deferred because
    ``run_suite`` imports torch, and the API/dashboard processes must stay light.

    If the runner cannot be imported (e.g. torch missing in a read-only API
    container), fall back to the task id so identity is still deterministic and
    the degradation is visible rather than silent.
    """
    try:
        from bench.runners.run_suite import (  # type: ignore[attr-defined]
            _build_scenario_cfg_basis,
            _canonicalize_scenario_id,
        )
    except Exception:
        return f"unresolved_{task.get('task_id', 'unknown')}"
    basis = _build_scenario_cfg_basis(dict(task), dict(scenario_settings))
    return str(_canonicalize_scenario_id(str(task.get("task_id")), basis))


def _resolve_device(runner: Mapping[str, Any], override: Optional[str]) -> str:
    device = str(override or runner.get("device") or "cpu").strip()
    return device or "cpu"


def _training_from(
    runner: Mapping[str, Any],
    model: Mapping[str, Any],
    *,
    init_id: str,
) -> TrainingSection:
    """Map runner budget + model knobs onto the typed training section.

    Mirrors `run_suite`'s precedence: the model's own ``batch_size`` wins over
    the runner-level ``budget.train_batch_size`` when both are present.

    ``training.enabled`` follows the runner's own semantics: only
    ``init_id == "trained"`` triggers a training phase (audit §6). ``pretrained``
    / ``loaded`` / ``untrained`` are evaluation-only plans.
    """
    budget = dict(runner.get("budget") or {})
    max_updates = int(budget.get("train_max_updates", 0) or 0)
    batch_size = int(model.get("batch_size") or budget.get("train_batch_size") or 1)
    enabled = str(init_id).lower() == "trained"
    return TrainingSection(
        enabled=enabled,
        max_updates=max_updates if enabled else 0,
        batch_size=max(1, batch_size),
        gradient_accumulation_steps=1,
        validation_interval_updates=int(model.get("val_eval_interval_updates", 0) or 0),
        early_stopping=dict(runner.get("early_stopping") or {}),
    )


def _optimizer_from(model: Mapping[str, Any], *, trainable: bool) -> OptimizerSection:
    """Map model knobs onto the optimizer section.

    Every trainable adapter in the audit uses Adam; the suite YAML has no
    optimizer selector, so ``adam`` is the honest mapping rather than a guess.
    Non-trainable model-based filters get ``none``.
    """
    if not trainable:
        return OptimizerSection(name="none", learning_rate=1.0, weight_decay=0.0)
    return OptimizerSection(
        name="adam",
        learning_rate=float(model.get("lr", 1e-3) or 1e-3),
        weight_decay=float(model.get("weight_decay", 0.0) or 0.0),
    )


def draft_from_suite(
    suite: Mapping[str, Any],
    *,
    task: Mapping[str, Any],
    model: Mapping[str, Any],
    seed: int,
    track_id: str,
    init_id: str = "untrained",
    scenario_settings: Optional[Mapping[str, Any]] = None,
    experiment_id: Optional[str] = None,
    experiment_name: Optional[str] = None,
    device: Optional[str] = None,
    precision: Optional[str] = None,
    unknown_key_policy: UnknownKeyPolicy = UnknownKeyPolicy.CAPTURE,
    provenance: Optional[ProvenanceSection] = None,
    telemetry: Optional[TelemetrySection] = None,
) -> RunSpecDraft:
    """Project one ``(task, model, seed, track, init)`` combination onto a draft.

    The arguments mirror ``bench.runners.run_suite.run_one``'s signature on
    purpose: anything the CLI can launch, this can describe.
    """
    scenario_settings = dict(scenario_settings or {})
    runner = dict(suite.get("runner") or {})
    suite_meta = dict(suite.get("suite") or {})

    model_id_raw = str(model.get("model_id") or "")
    if not model_id_raw:
        raise ConfigValidationError(
            [ValidationIssue(path="models[].model_id", code="required", message="model_id is required")]
        )

    implementation_id = implementation_id_for(model_id_raw)
    from ..capabilities import capabilities_for

    capability = capabilities_for(model_id_raw)

    unsupported: list[str] = []
    unsupported += _unsupported_paths(task, TASK_SUPPORTED_KEYS, "task")
    unsupported += _unsupported_paths(model, MODEL_SUPPORTED_KEYS, "model")
    unsupported += _unsupported_paths(runner, RUNNER_SUPPORTED_KEYS, "runner")

    if unsupported and unknown_key_policy is UnknownKeyPolicy.ERROR:
        raise ConfigValidationError(
            [
                ValidationIssue(
                    path=path,
                    code="unknown_key",
                    message=(
                        "key is not modelled by the control-plane schema and the active "
                        "policy is ERROR; use UnknownKeyPolicy.CAPTURE to preserve it"
                    ),
                )
                for path in unsupported
            ]
        )

    scenario_id = _scenario_id_for(task, scenario_settings)
    dataset_sizes = dict(task.get("dataset_sizes") or {})
    suite_name = str(suite_meta.get("name") or "unnamed_suite")

    system = SystemSection(
        task_id=str(task.get("task_id") or ""),
        scenario_id=scenario_id,
        state_dim=int(task.get("x_dim") or 0),
        observation_dim=int(task.get("y_dim") or 0),
        sequence_length=(
            int(task["sequence_length_T"]) if task.get("sequence_length_T") is not None else None
        ),
        scenario_config=dict(scenario_settings),
    )

    dataset = DatasetSection(
        # The bench data cache is keyed by (suite, task, scenario, seed); that
        # tuple *is* the dataset identity for a generated task.
        dataset_id=f"{suite_name}/{system.task_id}/{scenario_id}/seed_{int(seed)}",
        fingerprint=None,
        train_uri=None,
        val_uri=None,
        test_uri=None,
        split_seed=int(seed),
    )

    init_mode = str(init_id).strip().lower()
    initialization = InitializationSection(mode=init_mode if init_mode else "untrained")

    draft = RunSpecDraft(
        experiment=ExperimentSection(
            experiment_id=experiment_id or ExperimentId.new().value,
            name=experiment_name or suite_name,
            description=str(suite_meta.get("description") or "").strip(),
            tags=(suite_name, str(track_id), init_mode),
        ),
        model_id=ModelId(model_id_raw),
        implementation_id=ImplementationId(implementation_id),
        system=system,
        dataset=dataset,
        training=_training_from(runner, model, init_id=init_mode),
        optimizer=_optimizer_from(model, trainable=capability.trainable),
        initialization=initialization,
        resume=ResumeSection(mode="none"),
        runtime=RuntimeSection(
            device=_resolve_device(runner, device),
            precision=str(precision or runner.get("precision") or "fp32"),
            deterministic=bool(runner.get("deterministic", True)),
            seed=int(seed),
            num_workers=0,
        ),
        telemetry=telemetry or TelemetrySection(),
        artifacts=ArtifactsSection(
            save_predictions=False,
            emit_visualization=False,
            checkpoint_policy={},
        ),
        provenance=provenance or ProvenanceSection(),
        bench_context={
            "suite_name": suite_name,
            "suite_version": str(suite_meta.get("version") or ""),
            "track_id": str(track_id),
            "plan_id": f"{init_mode}__{track_id}",
            "data_mode": str(runner.get("data_mode") or ""),
            "display_name": str(model.get("display_name") or model_id_raw),
        },
        model_config_extra=_extra(model, MODEL_SUPPORTED_KEYS),
        task_config_extra={
            **_extra(task, TASK_SUPPORTED_KEYS),
            "dataset_sizes": dataset_sizes,
        },
        unsupported_fields=tuple(unsupported),
        original_config=dict(suite),
        architecture_fingerprint=None,
    )
    return draft


def drafts_from_suite(
    suite: Mapping[str, Any],
    *,
    init_id: str = "untrained",
    experiment_id: Optional[str] = None,
    seeds: Optional[Sequence[int]] = None,
    unknown_key_policy: UnknownKeyPolicy = UnknownKeyPolicy.CAPTURE,
) -> list[RunSpecDraft]:
    """Expand a whole suite into drafts, honouring the suite's `enabled` policy.

    Mirrors ``bench.runners.orchestrate.plan_runs`` semantics (D11: a missing
    ``enabled`` key defaults to enabled), so the control plane's expansion of a
    suite matches the runner's plan.
    """
    runner = dict(suite.get("runner") or {})
    policy = dict(runner.get("enabled_policy") or {})
    task_default = bool(policy.get("task_default", True))
    model_default = bool(policy.get("model_default", True))
    skip_if_disabled = bool(policy.get("skip_if_disabled", True))

    tracks = list(runner.get("tracks") or [{"track_id": "frozen", "adaptation_enabled": False}])
    seed_values = list(seeds if seeds is not None else suite.get("seeds") or [0])
    experiment = experiment_id or ExperimentId.new().value

    drafts: list[RunSpecDraft] = []
    for task in suite.get("tasks") or []:
        if skip_if_disabled and not bool(task.get("enabled", task_default)):
            continue
        for model in suite.get("models") or []:
            if skip_if_disabled and not bool(model.get("enabled", model_default)):
                continue
            for track in tracks:
                for seed in seed_values:
                    drafts.append(
                        draft_from_suite(
                            suite,
                            task=task,
                            model=model,
                            seed=int(seed),
                            track_id=str(track.get("track_id", "frozen")),
                            init_id=init_id,
                            experiment_id=experiment,
                            unknown_key_policy=unknown_key_policy,
                        )
                    )
    return drafts
