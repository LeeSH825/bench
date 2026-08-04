"""Declared adapter capabilities.

The UI must never infer what a model supports from its name (design doc 03 §3).
This module is the single declaration point, and every value here is derived
from the code audit in ``docs/benchmark_gui_current_state_audit.md`` §6 and §10
— not from optimism.

Two rules are enforced by construction:

1. ``supports_exact_resume`` is ``False`` for **every** entry. The audit's verdict
   for every trainable adapter is "weight load supported; partial resume not
   wired; exact resume unsupported". It may only be flipped for a specific
   ``implementation_id`` once an E-01 continuous-vs-resumed parity test passes.
2. ``paper_fidelity_status`` is independent of ``trainable``. An adapter that
   runs is not thereby paper-faithful (DND-013).

``event_instrumentation`` is added by this tranche and records how much of the
run is visible as structured events. ``"phase"`` means the runner emits phase and
final-metric events but the adapter's inner training loop does not stream
per-update losses. That is a coverage statement, not a defect to hide — and it
is *never* patched over by parsing stdout (DND-006).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Mapping, Optional

#: Schema version of the capability document.
CAPABILITY_SCHEMA_VERSION = 1

#: Paper-fidelity vocabulary.
#: ``verified``       — a parity test against the published procedure passes.
#: ``partial``        — runs, but a known deviation from the paper is documented.
#: ``unverified``     — executes; no fidelity claim has been tested either way.
#: ``not_applicable`` — no learned procedure to be faithful to.
FIDELITY_STATUSES = ("verified", "partial", "unverified", "not_applicable")

#: How much structured-event coverage an adapter has.
#: ``none``  — no events beyond registry state transitions.
#: ``phase`` — runner-level phase boundaries + final metrics only.
#: ``step``  — per-update training/validation metrics streamed live.
INSTRUMENTATION_LEVELS = ("none", "phase", "step")


@dataclass(frozen=True)
class AdapterCapabilities:
    """Capability declaration for one ``(model_id, implementation_id)`` pair."""

    model_id: str
    implementation_id: str
    display_name: str
    trainable: bool
    supports_evaluation: bool = True
    supports_warm_start: bool = False
    supports_graceful_stop: bool = False
    supports_checkpoint: bool = False
    supports_exact_resume: bool = False
    resume_boundary: Optional[str] = None
    training_phases: tuple[str, ...] = ("test",)
    optimizer_slots: tuple[str, ...] = ()
    scheduler_slots: tuple[str, ...] = ()
    amp_supported: bool = False
    online_adaptation_supported: bool = False
    paper_fidelity_status: str = "unverified"
    paper_fidelity_note: str = ""
    event_instrumentation: str = "none"
    instrumentation_note: str = ""
    schema_version: int = CAPABILITY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.paper_fidelity_status not in FIDELITY_STATUSES:
            raise ValueError(
                f"paper_fidelity_status must be one of {FIDELITY_STATUSES}, "
                f"got {self.paper_fidelity_status!r}"
            )
        if self.event_instrumentation not in INSTRUMENTATION_LEVELS:
            raise ValueError(
                f"event_instrumentation must be one of {INSTRUMENTATION_LEVELS}, "
                f"got {self.event_instrumentation!r}"
            )
        if self.supports_exact_resume and not self.resume_boundary:
            raise ValueError(
                "supports_exact_resume=True requires an explicit resume_boundary; "
                "an uncertified resume claim is a research-integrity hazard (R-05)"
            )

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "model_id": self.model_id,
            "implementation_id": self.implementation_id,
            "display_name": self.display_name,
            "trainable": self.trainable,
            "supports_evaluation": self.supports_evaluation,
            "supports_warm_start": self.supports_warm_start,
            "supports_graceful_stop": self.supports_graceful_stop,
            "supports_checkpoint": self.supports_checkpoint,
            "supports_exact_resume": self.supports_exact_resume,
            "resume_boundary": self.resume_boundary,
            "training_phases": list(self.training_phases),
            "optimizer_slots": list(self.optimizer_slots),
            "scheduler_slots": list(self.scheduler_slots),
            "amp_supported": self.amp_supported,
            "online_adaptation_supported": self.online_adaptation_supported,
            "paper_fidelity_status": self.paper_fidelity_status,
            "paper_fidelity_note": self.paper_fidelity_note,
            "event_instrumentation": self.event_instrumentation,
            "instrumentation_note": self.instrumentation_note,
        }


_TRAIN_PHASES = ("train", "validation", "test")

#: Capability table keyed by ``model_id``.
#:
#: ``implementation_id`` carries an explicit ``_v1`` suffix: certification is
#: granted per implementation version, so the identity must change whenever the
#: adapter's numerical behaviour changes.
_CAPABILITIES: dict[str, AdapterCapabilities] = {
    "kalmannet_tsp": AdapterCapabilities(
        model_id="kalmannet_tsp",
        implementation_id="bench_kalmannet_tsp_adapter_v1",
        display_name="KalmanNet (TSP)",
        trainable=True,
        supports_warm_start=True,
        supports_checkpoint=True,
        training_phases=_TRAIN_PHASES,
        optimizer_slots=("main",),
        paper_fidelity_status="unverified",
        paper_fidelity_note=(
            "Supervised MSE training with a single Adam optimizer and an update-budget "
            "loop. Executes against the upstream KalmanNet_TSP architecture, but no "
            "numerical parity test against the published training procedure has been run."
        ),
        event_instrumentation="step",
        instrumentation_note="Per-update train loss and periodic validation metrics are emitted.",
    ),
    "split_knet": AdapterCapabilities(
        model_id="split_knet",
        implementation_id="bench_split_adapter_v1",
        display_name="Split-KalmanNet",
        trainable=True,
        supports_warm_start=True,
        supports_checkpoint=True,
        training_phases=_TRAIN_PHASES,
        optimizer_slots=("main",),
        paper_fidelity_status="partial",
        paper_fidelity_note=(
            "The adapter exposes one combined network with a single Adam optimizer. "
            "The paper's alternating optimization of the two split heads is NOT "
            "implemented here. Comparisons against published Split-KalmanNet numbers "
            "must state this deviation."
        ),
        event_instrumentation="step",
        instrumentation_note="Per-update train loss and periodic validation metrics are emitted.",
    ),
    "adaptive_knet": AdapterCapabilities(
        model_id="adaptive_knet",
        implementation_id="bench_adaptive_knet_adapter_v1",
        display_name="Adaptive-KNet (ICASSP24)",
        trainable=True,
        supports_warm_start=True,
        supports_checkpoint=True,
        training_phases=("train", "validation", "adapt", "test"),
        optimizer_slots=("main", "adapt"),
        online_adaptation_supported=True,
        paper_fidelity_status="unverified",
        paper_fidelity_note=(
            "Supervised training plus optional budgeted online self-supervised "
            "adaptation with a separately constructed Adam. The adapted in-memory "
            "state is not persisted by the runner after the adapt phase."
        ),
        event_instrumentation="phase",
        instrumentation_note=(
            "Runner phase boundaries and final metrics only. The adapt-phase optimizer "
            "is created inside the adapter and its per-update losses are not exposed "
            "through the observer contract yet."
        ),
    ),
    "maml_knet": AdapterCapabilities(
        model_id="maml_knet",
        implementation_id="bench_maml_knet_adapter_v1",
        display_name="MAML-KalmanNet",
        trainable=True,
        supports_warm_start=True,
        supports_checkpoint=True,
        training_phases=_TRAIN_PHASES,
        optimizer_slots=("outer", "inner"),
        paper_fidelity_status="unverified",
        paper_fidelity_note=(
            "Meta outer-loop Adam with per-task inner SGD. Inner/outer/task cursors are "
            "absent from the checkpoint, so even partial resume is impossible today."
        ),
        event_instrumentation="phase",
        instrumentation_note=(
            "Runner phase boundaries and final metrics only. Inner/outer loop metrics "
            "would need a two-level step vocabulary that this tranche does not define."
        ),
    ),
    "me_split_knet_v0": AdapterCapabilities(
        model_id="me_split_knet_v0",
        implementation_id="bench_me_split_adapter_v0",
        display_name="Measurement-Enhanced Split-KalmanNet",
        trainable=True,
        supports_warm_start=True,
        supports_checkpoint=True,
        training_phases=("train", "validation", "test"),
        optimizer_slots=("enhancer", "split"),
        paper_fidelity_status="not_applicable",
        paper_fidelity_note=(
            "Project extension, not a reproduction of a published method. Two sequential "
            "phases (enhancer Adam, then the inherited Split Adam) share one checkpoint."
        ),
        event_instrumentation="phase",
        instrumentation_note="Runner phase boundaries and final metrics only.",
    ),
    "mb_kf": AdapterCapabilities(
        model_id="mb_kf",
        implementation_id="bench_mb_kf_adapter_v1",
        display_name="Model-Based Kalman Filter",
        trainable=False,
        supports_warm_start=False,
        supports_checkpoint=True,
        training_phases=("test",),
        optimizer_slots=(),
        paper_fidelity_status="not_applicable",
        paper_fidelity_note=(
            "Classical model-based filter with no learning lifecycle. 'train' writes a "
            "zero-update state; resume is not a meaningful concept here."
        ),
        event_instrumentation="step",
        instrumentation_note=(
            "Emits phase boundaries and final evaluation metrics. There is no training "
            "loop to stream, so 'step' coverage is complete for this adapter."
        ),
    ),
    "basilisk_mrp_ekf": AdapterCapabilities(
        model_id="basilisk_mrp_ekf",
        implementation_id="bench_basilisk_mrp_ekf_adapter_v1",
        display_name="Basilisk MRP EKF",
        trainable=False,
        supports_checkpoint=True,
        training_phases=("test",),
        paper_fidelity_status="not_applicable",
        paper_fidelity_note="Diagnostic model-based EKF; no learning lifecycle.",
        event_instrumentation="phase",
        instrumentation_note="Runner phase boundaries and final metrics only.",
    ),
}

#: ``model_id`` aliases that share one adapter implementation.
#:
#: ``bench/models/registry.py`` maps 20 model ids onto 10 adapter classes. The
#: aliases genuinely differ in *configuration* (dataset scale, regularization,
#: gradient clipping), which is captured by ``structural_config_hash`` — but they
#: share an implementation, so they must share an ``implementation_id``.
_ALIASES: dict[str, str] = {
    "me_split_knet_v0_ds100": "me_split_knet_v0",
    "me_split_knet_v0_ds025": "me_split_knet_v0",
    "me_split_knet_v0_ds010": "me_split_knet_v0",
    "me_split_knet_v0_small": "me_split_knet_v0",
    "me_split_knet_v0_regstrong": "me_split_knet_v0",
    "me_split_knet_v0_clip025": "me_split_knet_v0",
    "oracle_kf": "mb_kf",
    "nominal_kf": "mb_kf",
    "oracle_shift_kf": "mb_kf",
    "mb_kf_oracle": "mb_kf",
    "mb_kf_nominal": "mb_kf",
    # Spiking / SNN variants are registered adapters but were not covered by the
    # audit's lifecycle matrix. They are declared explicitly below rather than
    # left to a name-based guess.
    "spike_split_knet": "split_knet",
    "g1_snn_split_knet": "split_knet",
    "spike_ra_knet": "kalmannet_tsp",
}

#: Fallback used when a model_id is registered in bench but has no declaration
#: here. Deliberately pessimistic: nothing is claimed.
UNKNOWN_CAPABILITIES_NOTE = (
    "No capability declaration exists for this model_id. All capabilities are "
    "reported as unsupported and paper fidelity as unverified until one is added "
    "to bench/control/capabilities.py."
)


def capabilities_for(model_id: str) -> AdapterCapabilities:
    """Return the declared capabilities for *model_id*.

    Unknown ids get a conservative all-false declaration rather than an
    exception, so that a newly registered adapter shows up in the UI as
    "undeclared" instead of breaking the run list.
    """
    key = _ALIASES.get(model_id, model_id)
    declared = _CAPABILITIES.get(key)
    if declared is None:
        return AdapterCapabilities(
            model_id=model_id,
            implementation_id=f"undeclared_{model_id}",
            display_name=model_id,
            trainable=False,
            supports_evaluation=False,
            paper_fidelity_status="unverified",
            paper_fidelity_note=UNKNOWN_CAPABILITIES_NOTE,
            event_instrumentation="none",
            instrumentation_note=UNKNOWN_CAPABILITIES_NOTE,
        )
    if key != model_id:
        # Preserve the caller's model_id while keeping the shared implementation
        # identity — the alias is a different configuration of the same code.
        return replace(declared, model_id=model_id)
    return declared


def implementation_id_for(model_id: str) -> str:
    """Canonical implementation id for *model_id*."""
    return capabilities_for(model_id).implementation_id


def all_capabilities() -> list[dict[str, Any]]:
    """Every declared capability document, sorted by model id.

    Includes alias ids so the UI can show the full selectable set.
    """
    ids = sorted(set(_CAPABILITIES) | set(_ALIASES))
    return [capabilities_for(model_id).as_dict() for model_id in ids]


def capability_index() -> Mapping[str, AdapterCapabilities]:
    """Mapping of every known model id to its capabilities."""
    ids = sorted(set(_CAPABILITIES) | set(_ALIASES))
    return {model_id: capabilities_for(model_id) for model_id in ids}
