"""Canonical training-path selection (ADR-WC-001 … ADR-WC-005).

A certified control-plane run uses :meth:`resumable_train` from update 0. That
is decided **once**, at resolve/launch time, recorded as structural provenance,
and then simply executed by the worker. There is no user-facing toggle, and
there is no fallback: a run that resolves to the resumable path and cannot run
it fails rather than quietly training down the legacy loop (ADR-WC-003).

Why decide once: the alternative — deciding per-run inside the worker from the
model name — is how a run ends up half-certified, where Stop/Resume is offered
for a loop that was never actually resumable.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

#: Bumped only when the *meaning* of a path id changes. Not a schema version.
TRAINING_PATH_CONTRACT_VERSION = 1


class TrainingPathId(str, enum.Enum):
    """Concrete training entry points. Persisted, never re-derived."""

    #: Certified control-plane path: resumable_train() from update 0.
    CONTROL_RESUMABLE_V1 = "control_resumable_v1"
    #: The existing adapter train() loop. Unchanged by this tranche.
    LEGACY_TRAIN_V1 = "legacy_train_v1"
    #: Evaluation-only or model-based baselines with no learning lifecycle.
    NOT_APPLICABLE = "not_applicable"

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.value


#: The only models whose adapters implement the resumable contract.
RESUMABLE_MODEL_IDS = frozenset({"kalmannet_tsp", "split_knet"})


class PathReasonCode(str, enum.Enum):
    """Machine-readable reasons, so the UI never invents its own wording."""

    CERTIFIED = "CERTIFIED"
    NOT_TRAINABLE = "NOT_TRAINABLE"
    TRAINING_DISABLED = "TRAINING_DISABLED"
    MODEL_NOT_RESUMABLE = "MODEL_NOT_RESUMABLE"
    UNCERTIFIED_DEVICE = "UNCERTIFIED_DEVICE"
    UNCERTIFIED_PRECISION = "UNCERTIFIED_PRECISION"
    UNCERTIFIED_NUM_WORKERS = "UNCERTIFIED_NUM_WORKERS"
    UNCERTIFIED_GRAD_ACCUM = "UNCERTIFIED_GRAD_ACCUM"
    NO_CERTIFICATION_ROW = "NO_CERTIFICATION_ROW"
    LEGACY_SPEC_NO_PATH_FIELD = "LEGACY_SPEC_NO_PATH_FIELD"

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.value


_MESSAGES = {
    PathReasonCode.CERTIFIED: "Certified for exact resume; uses the resumable control path.",
    PathReasonCode.NOT_TRAINABLE: "This model has no learning lifecycle.",
    PathReasonCode.TRAINING_DISABLED: "Training is disabled for this run.",
    PathReasonCode.MODEL_NOT_RESUMABLE: (
        "Exact resume is implemented only for kalmannet_tsp and split_knet."
    ),
    PathReasonCode.UNCERTIFIED_DEVICE: (
        "Exact resume is certified only for CPU; this run requests a different device."
    ),
    PathReasonCode.UNCERTIFIED_PRECISION: (
        "Exact resume is certified only for fp32."
    ),
    PathReasonCode.UNCERTIFIED_NUM_WORKERS: (
        "Exact resume is certified only for num_workers=0."
    ),
    PathReasonCode.UNCERTIFIED_GRAD_ACCUM: (
        "Exact resume is certified only without gradient accumulation."
    ),
    PathReasonCode.NO_CERTIFICATION_ROW: (
        "No exact-resume certification exists for this model/implementation tuple."
    ),
    PathReasonCode.LEGACY_SPEC_NO_PATH_FIELD: (
        "This run predates the training-path contract and is treated as legacy."
    ),
}


def message_for(code: PathReasonCode) -> str:
    return _MESSAGES.get(code, str(code))


@dataclass(frozen=True)
class TrainingPathDecision:
    """Why a run got the path it got. Stored, not recomputed."""

    training_path_id: TrainingPathId
    reason_codes: tuple[PathReasonCode, ...] = ()
    certification_id: Optional[str] = None

    @property
    def is_resumable(self) -> bool:
        return self.training_path_id is TrainingPathId.CONTROL_RESUMABLE_V1

    @property
    def messages(self) -> list[str]:
        return [message_for(code) for code in self.reason_codes]

    def as_dict(self) -> dict[str, Any]:
        return {
            "training_path_id": str(self.training_path_id),
            "reason_codes": [str(c) for c in self.reason_codes],
            "messages": self.messages,
            "certification_id": self.certification_id,
            "contract_version": TRAINING_PATH_CONTRACT_VERSION,
        }


def select_training_path(
    *,
    model_id: str,
    implementation_id: str,
    training_enabled: bool,
    device: str,
    precision: str,
    num_workers: int,
    gradient_accumulation_steps: int = 1,
    trainable: bool = True,
    registry: Any = None,
) -> TrainingPathDecision:
    """Decide the training path from the *full* certification tuple.

    Called once, at resolve time. Never called again from the worker, and never
    re-derived from the model name (ADR-WC-005).
    """
    from .checkpoints.certification import is_certified
    from .checkpoints.compatibility import certification_key_for

    if not trainable:
        return TrainingPathDecision(
            TrainingPathId.NOT_APPLICABLE, (PathReasonCode.NOT_TRAINABLE,)
        )
    if not training_enabled:
        return TrainingPathDecision(
            TrainingPathId.NOT_APPLICABLE, (PathReasonCode.TRAINING_DISABLED,)
        )

    reasons: list[PathReasonCode] = []
    if str(model_id) not in RESUMABLE_MODEL_IDS:
        reasons.append(PathReasonCode.MODEL_NOT_RESUMABLE)
    if str(device).lower() != "cpu":
        reasons.append(PathReasonCode.UNCERTIFIED_DEVICE)
    if str(precision).lower() != "fp32":
        reasons.append(PathReasonCode.UNCERTIFIED_PRECISION)
    if int(num_workers) != 0:
        reasons.append(PathReasonCode.UNCERTIFIED_NUM_WORKERS)
    if int(gradient_accumulation_steps or 1) != 1:
        reasons.append(PathReasonCode.UNCERTIFIED_GRAD_ACCUM)

    if reasons:
        return TrainingPathDecision(TrainingPathId.LEGACY_TRAIN_V1, tuple(reasons))

    training_mode = _training_mode_for(model_id)
    certified = is_certified(
        model_id=str(model_id),
        implementation_id=str(implementation_id),
        precision="fp32",
        device_class="cpu",
        num_workers=0,
        training_mode=training_mode,
        registry=registry,
    )
    if not certified:
        return TrainingPathDecision(
            TrainingPathId.LEGACY_TRAIN_V1, (PathReasonCode.NO_CERTIFICATION_ROW,)
        )

    key = certification_key_for(
        model_id=str(model_id),
        implementation_id=str(implementation_id),
        precision="fp32",
        device_class="cpu",
        num_workers=0,
        training_mode=training_mode,
    ).as_key()
    return TrainingPathDecision(
        TrainingPathId.CONTROL_RESUMABLE_V1, (PathReasonCode.CERTIFIED,), certification_id=key
    )


def _training_mode_for(model_id: str) -> str:
    """The certified training mode for a model.

    Split is recorded as a deviation because its adapter uses a single
    optimizer slot rather than the paper's alternating scheme; that is part of
    the certification key, not a footnote.
    """
    if str(model_id) == "split_knet":
        return "supervised_single_optimizer_split_deviation"
    return "supervised_single_optimizer"


def path_from_spec(document: Optional[Mapping[str, Any]]) -> TrainingPathId:
    """Read the persisted path from a resolved spec.

    A spec with no field predates the contract and is **legacy**. It is never
    reinterpreted as resumable after the fact (ADR-WC-002).
    """
    if not document:
        return TrainingPathId.LEGACY_TRAIN_V1
    raw = document.get("training_path_id")
    if raw is None:
        execution = document.get("execution") or {}
        raw = execution.get("training_path_id") if isinstance(execution, Mapping) else None
    if raw is None:
        return TrainingPathId.LEGACY_TRAIN_V1
    try:
        return TrainingPathId(str(raw))
    except ValueError:
        return TrainingPathId.LEGACY_TRAIN_V1
