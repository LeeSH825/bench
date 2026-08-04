"""Validation and preview for GUI-submitted configs.

One path only:

    preset YAML → safe parse → overrides → typed RunSpecDraft
    → validate → ResolvedRunSpec → canonical hashes

The GUI and the CLI resolve through the *same* ``draft_from_suite`` and
``resolve_run_spec``. There is deliberately no GUI-specific parser or
defaulting, because a second implementation is how a form quietly produces a
different experiment than the command line.

Nothing here allocates a run, touches the registry, or writes to disk.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field, replace
from typing import Any, Mapping, Optional

import yaml

from .compatibility import draft_from_suite
from .descriptor import supported_paths
from .presets import PresetError, safe_load_preset_text
from .resolver import resolve_run_spec
from .schema import ConfigValidationError, PRECISIONS, RunSpecDraft

VALIDATION_SCHEMA_VERSION = 1

#: A submitted document is small. This bounds the request before parsing.
MAX_SUBMITTED_BYTES = 512 * 1024


@dataclass
class ValidationIssueOut:
    path: str
    code: str
    message: str
    severity: str = "error"

    def as_dict(self) -> dict[str, Any]:
        return {"path": self.path, "code": self.code,
                "message": self.message, "severity": self.severity}


@dataclass
class ValidationResult:
    valid: bool
    issues: list[ValidationIssueOut] = field(default_factory=list)
    unsupported_fields: list[str] = field(default_factory=list)
    resolved_run_spec: Optional[dict[str, Any]] = None
    canonical_yaml: Optional[str] = None
    structural_config_hash: Optional[str] = None
    operational_config_hash: Optional[str] = None
    variant_id: Optional[str] = None
    training_path_id: Optional[str] = None
    implementation_id: Optional[str] = None
    launch_eligibility: dict[str, Any] = field(default_factory=dict)
    diff: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": VALIDATION_SCHEMA_VERSION,
            "valid": bool(self.valid),
            "issues": [i.as_dict() for i in self.issues],
            "unsupported_fields": list(self.unsupported_fields),
            "resolved_run_spec": self.resolved_run_spec,
            "canonical_yaml": self.canonical_yaml,
            "structural_config_hash": self.structural_config_hash,
            "operational_config_hash": self.operational_config_hash,
            "variant_id": self.variant_id,
            "training_path_id": self.training_path_id,
            "implementation_id": self.implementation_id,
            "launch_eligibility": dict(self.launch_eligibility),
            "diff": dict(self.diff),
        }


def _issue(path: str, code: str, message: str, severity: str = "error") -> ValidationIssueOut:
    return ValidationIssueOut(path=path, code=code, message=message, severity=severity)


def _set_by_path(document: dict[str, Any], path: str, value: Any) -> None:
    parts = path.split(".")
    node = document
    for part in parts[:-1]:
        node = node.setdefault(part, {})
        if not isinstance(node, dict):
            raise PresetError(f"cannot set {path}: {part} is not a mapping")
    node[parts[-1]] = value


def _get_by_path(document: Mapping[str, Any], path: str) -> Any:
    node: Any = document
    for part in path.split("."):
        if not isinstance(node, Mapping) or part not in node:
            return None
        node = node[part]
    return node


def apply_overrides(suite: Mapping[str, Any], overrides: Mapping[str, Any]) -> dict[str, Any]:
    """Apply GUI field overrides onto a parsed preset document.

    Only descriptor-declared paths are accepted. An override for anything else
    is refused rather than silently written, so the form cannot become a way
    to inject arbitrary config.
    """
    document = copy.deepcopy(dict(suite))
    allowed = supported_paths()
    unknown = [p for p in overrides if p not in allowed]
    if unknown:
        raise PresetError(
            "unsupported override path(s): " + ", ".join(sorted(unknown))
            + ". Only fields declared by the schema descriptor may be overridden.")
    runner = document.setdefault("runner", {})
    for path, value in overrides.items():
        # Runtime/training/optimizer overrides live under the suite's runner
        # block, which is what draft_from_suite reads.
        if path.startswith("runtime."):
            key = path.split(".", 1)[1]
            runner[{"seed": "seed"}.get(key, key)] = value
        elif path.startswith("training."):
            key = path.split(".", 1)[1]
            budget = runner.setdefault("budget", {})
            mapping = {"max_updates": "train_max_updates",
                       "batch_size": "train_batch_size"}
            if key in mapping:
                budget[mapping[key]] = value
            elif key == "enabled":
                runner["training_enabled"] = value
            else:
                budget[key] = value
        elif path.startswith("optimizer."):
            key = path.split(".", 1)[1]
            for model in document.get("models") or []:
                if isinstance(model, dict):
                    model[{"learning_rate": "lr"}.get(key, key)] = value
        elif path.startswith("telemetry.") or path.startswith("artifacts."):
            document.setdefault("_gui", {})[path] = value
        elif path == "experiment.description":
            document.setdefault("suite", {})["description"] = value
        elif path == "initialization.mode":
            document.setdefault("_gui", {})["init_id"] = value
        else:
            _set_by_path(document, path, value)
    return document


def _unsupported_from_draft(draft: RunSpecDraft) -> list[str]:
    return sorted(str(f) for f in (draft.unsupported_fields or ()))


def validate_config(
    *,
    suite_document: Mapping[str, Any],
    task_id: Optional[str] = None,
    model_id: Optional[str] = None,
    overrides: Optional[Mapping[str, Any]] = None,
    seed: int = 0,
    track_id: str = "frozen",
    init_id: Optional[str] = None,
    baseline_document: Optional[Mapping[str, Any]] = None,
    registry: Any = None,
    provenance: Any = None,
    executor: str = "suite",
) -> ValidationResult:
    """Validate and resolve a draft. Never mutates anything.

    ``provenance`` is left out of a preview on purpose: capturing it shells out
    to Git, and a preview may be recomputed on every keystroke. The launch path
    passes a freshly captured section, so what is stored with a run is the repo
    state at launch — not at preview. Provenance is not an input to the
    structural or operational hash, so this does not move run identity.
    """
    from ..training_path import TrainingPathId

    try:
        document = apply_overrides(suite_document, dict(overrides or {}))
    except PresetError as exc:
        return ValidationResult(False, [_issue("", "UNSUPPORTED_OVERRIDE", str(exc))])

    gui = dict(document.pop("_gui", {}) or {})
    resolved_init = init_id or str(gui.get("init_id") or "untrained")

    tasks = document.get("tasks") or []
    models = document.get("models") or []
    task = next((t for t in tasks if str(t.get("task_id")) == str(task_id)), None) \
        if task_id else (tasks[0] if tasks else None)
    model = next((m for m in models if str(m.get("model_id")) == str(model_id)), None) \
        if model_id else (models[0] if models else None)
    if task is None:
        return ValidationResult(False, [_issue("system.task_id", "UNKNOWN_TASK",
                                               f"task_id {task_id!r} is not in this preset")])
    if model is None:
        return ValidationResult(False, [_issue("model_id", "UNKNOWN_MODEL",
                                               f"model_id {model_id!r} is not in this preset")])

    runner = document.get("runner") or {}
    device = str(runner.get("device", "cpu"))
    precision = str(runner.get("precision", "fp32"))
    if precision not in PRECISIONS:
        return ValidationResult(False, [_issue("runtime.precision", "INVALID_PRECISION",
                                               f"unknown precision {precision!r}")])

    try:
        draft = draft_from_suite(
            document, task=task, model=model, seed=int(runner.get("seed", seed)),
            track_id=track_id, init_id=resolved_init,
            device=device, precision=precision, provenance=provenance,
        )
        # The CLI stamps the executor explicitly; a GUI-launched run must record
        # the same thing rather than relying on the worker's default, or the two
        # stored specs describe different executions of the same config.
        draft = replace(draft, bench_context={**dict(draft.bench_context),
                                              "executor": executor})
    except ConfigValidationError as exc:
        return ValidationResult(False, [
            _issue(getattr(i, "path", ""), getattr(i, "code", "INVALID"),
                   getattr(i, "message", str(i))) for i in exc.issues])
    except Exception as exc:
        return ValidationResult(False, [_issue("", "DRAFT_ERROR", str(exc))])

    try:
        spec = resolve_run_spec(draft, registry=registry)
    except ConfigValidationError as exc:
        return ValidationResult(
            False,
            [_issue(getattr(i, "path", ""), getattr(i, "code", "INVALID"),
                    getattr(i, "message", str(i))) for i in exc.issues],
            unsupported_fields=_unsupported_from_draft(draft))
    except Exception as exc:
        return ValidationResult(False, [_issue("", "RESOLVE_ERROR", str(exc))])

    document_out = spec.as_dict()
    path_id = spec.draft.execution.training_path_id
    eligibility = _launch_eligibility(spec)

    result = ValidationResult(
        valid=True,
        unsupported_fields=_unsupported_from_draft(draft),
        resolved_run_spec=document_out,
        canonical_yaml=yaml.safe_dump(document, sort_keys=True, allow_unicode=True),
        structural_config_hash=spec.structural_config_hash,
        operational_config_hash=spec.operational_config_hash,
        variant_id=spec.variant_id.value,
        training_path_id=path_id,
        implementation_id=spec.implementation_id.value,
        launch_eligibility=eligibility,
    )
    if baseline_document is not None:
        result.diff = compute_diff(baseline_document, document, spec, task_id=task_id,
                                   model_id=model_id, init_id=resolved_init, registry=registry)
    if result.unsupported_fields:
        result.issues.append(_issue(
            "", "UNSUPPORTED_FIELDS_PRESERVED",
            "Some keys are not modelled by the config schema. They are preserved "
            "verbatim in the raw YAML but the GUI does not manage them: "
            + ", ".join(result.unsupported_fields[:10]),
            severity="warning"))
    return result


def _launch_eligibility(spec: Any) -> dict[str, Any]:
    """Whether this resolved spec may be launched from the GUI."""
    from ..capabilities import capabilities_for
    from ..training_path import RESUMABLE_MODEL_IDS, TrainingPathId

    model_id = spec.model_id.value
    path_id = str(spec.draft.execution.training_path_id)
    try:
        capability = capabilities_for(model_id)
        trainable = bool(getattr(capability, "trainable", True))
    except Exception:
        trainable = True

    if model_id in RESUMABLE_MODEL_IDS:
        if path_id != str(TrainingPathId.CONTROL_RESUMABLE_V1):
            codes = list(spec.draft.execution.training_path_reason_codes) or ["UNCERTIFIED"]
            return {"eligible": False, "reason_code": codes[0],
                    "reason": ("This configuration resolves to the legacy training path, "
                               "which is not certified for GUI launch with Stop/Resume."),
                    "stop_resume_available": False}
        return {"eligible": True, "reason_code": None, "reason": None,
                "stop_resume_available": True}

    if not trainable:
        # Model-based baselines: launchable, but no learning lifecycle, so no
        # Stop/Resume is offered.
        return {"eligible": True, "reason_code": None,
                "reason": ("Model-based baseline: it runs evaluation only, so Stop and "
                           "Resume are not offered."),
                "stop_resume_available": False}

    return {"eligible": False, "reason_code": "ADAPTER_NOT_GUI_LAUNCH_CERTIFIED",
            "reason": (f"GUI launch is not certified for {model_id}. Certified models are "
                       "kalmannet_tsp and split_knet, plus the model-based baselines."),
            "stop_resume_available": False}


def compute_diff(baseline: Mapping[str, Any], edited: Mapping[str, Any],
                 spec: Any, *, task_id: Optional[str], model_id: Optional[str],
                 init_id: str, registry: Any = None) -> dict[str, Any]:
    """Field-level diff plus whether identity actually moved.

    The comparison is made on the **resolved** specs, not the raw suite
    documents. Descriptor paths like ``training.max_updates`` describe the
    typed draft; the same value lives somewhere else entirely in the suite YAML
    (``runner.budget.train_max_updates``). Diffing the raw documents therefore
    reported "nothing changed" for edits that genuinely moved the structural
    hash — which is exactly the reassurance a review step must not give.
    """
    from .descriptor import field_by_path, supported_paths

    baseline_result = None
    try:
        baseline_result = validate_config(
            suite_document=baseline, task_id=task_id, model_id=model_id,
            init_id=init_id, registry=registry)
    except Exception:
        baseline_result = None

    before_spec = (baseline_result.resolved_run_spec
                   if baseline_result and baseline_result.valid else None) or {}
    after_spec = spec.as_dict()

    changed: list[dict[str, Any]] = []
    for path in sorted(supported_paths()):
        before, after = _get_by_path(before_spec, path), _get_by_path(after_spec, path)
        if before == after:
            continue
        descriptor = field_by_path(path)
        changed.append({
            "path": path, "before": before, "after": after,
            "label": descriptor.label if descriptor else path,
            "classification": descriptor.classification if descriptor else "operational",
        })

    return {
        "changed_fields": changed,
        "structural_changed": (
            baseline_result.structural_config_hash != spec.structural_config_hash
            if baseline_result and baseline_result.valid else None),
        "operational_changed": (
            baseline_result.operational_config_hash != spec.operational_config_hash
            if baseline_result and baseline_result.valid else None),
        "variant_changed": (
            baseline_result.variant_id != spec.variant_id.value
            if baseline_result and baseline_result.valid else None),
        "baseline_structural_config_hash": (
            baseline_result.structural_config_hash if baseline_result else None),
        "baseline_variant_id": baseline_result.variant_id if baseline_result else None,
    }


def parse_submitted_yaml(text: str) -> Mapping[str, Any]:
    """Parse user-edited YAML with the same defensive limits as a preset."""
    if len(text.encode("utf-8")) > MAX_SUBMITTED_BYTES:
        raise PresetError(f"submitted config exceeds {MAX_SUBMITTED_BYTES} bytes")
    return safe_load_preset_text(text)
