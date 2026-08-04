"""Exact-resume certification records.

``supports_exact_resume`` is not a property of a model name. It is a property
of a *(model, implementation, schema, boundary, precision, device, loader,
training mode)* tuple, and it is only true where a parity test has actually
been run (ADR-CSR-013, DND-CSR-007, DND-CSR-008).

This module holds the certifications this tranche earned by execution, and the
explicit non-certifications for everything else. A configuration that is not
listed is uncertified — the default answer is "no", not "probably".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from .compatibility import certification_key_for
from .schema import CHECKPOINT_SCHEMA_VERSION, RESUME_BOUNDARY_OPTIMIZER_UPDATE

#: Evidence for the certified rows below.
CERTIFICATION_EVIDENCE = "tests/test_control_exact_resume_certification.py"


@dataclass(frozen=True)
class CertificationRecord:
    model_id: str
    implementation_id: str
    certified: bool
    precision: str = "fp32"
    device_class: str = "cpu"
    num_workers: int = 0
    training_mode: str = "supervised_single_optimizer"
    training_path_id: str = "control_resumable_v1"
    evidence_uri: Optional[str] = None
    notes: Optional[str] = None

    @property
    def key(self) -> str:
        return certification_key_for(
            model_id=self.model_id,
            implementation_id=self.implementation_id,
            precision=self.precision,
            device_class=self.device_class,
            num_workers=self.num_workers,
            training_mode=self.training_mode,
            training_path_id=self.training_path_id,
        ).as_key()

    def as_dict(self) -> dict[str, Any]:
        return {
            "certification_key": self.key,
            "model_id": self.model_id,
            "implementation_id": self.implementation_id,
            "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
            "resume_boundary": RESUME_BOUNDARY_OPTIMIZER_UPDATE,
            "precision": self.precision,
            "device_class": self.device_class,
            "num_workers": self.num_workers,
            "training_mode": self.training_mode,
            "training_path_id": self.training_path_id,
            "certified": self.certified,
            "evidence_uri": self.evidence_uri,
            "notes": self.notes,
        }


#: Certified by fresh-process bitwise parity in this tranche.
CERTIFIED: tuple[CertificationRecord, ...] = (
    CertificationRecord(
        model_id="kalmannet_tsp",
        implementation_id="bench_kalmannet_tsp_adapter_v1",
        certified=True,
        evidence_uri=CERTIFICATION_EVIDENCE,
        notes=(
            "Continuous vs interrupted+resumed training is bitwise identical on weights, "
            "optimizer state, per-update loss sequence, validation history, and best state."
        ),
    ),
    CertificationRecord(
        model_id="split_knet",
        implementation_id="bench_split_adapter_v1",
        certified=True,
        training_mode="supervised_single_optimizer_split_deviation",
        evidence_uri=CERTIFICATION_EVIDENCE,
        notes=(
            "Same parity evidence as kalmannet_tsp. Requires capturing the GRU initial "
            "hidden state (hn1_init/hn2_init), which lives outside state_dict(). This "
            "adapter uses one optimizer slot, not the paper's alternating optimization; "
            "exact-resume certification is a statement about this implementation, not "
            "about paper fidelity."
        ),
    ),
)

#: Explicitly *not* certified. Listed so the answer is recorded rather than absent.
NOT_CERTIFIED: tuple[CertificationRecord, ...] = (
    CertificationRecord(
        model_id="adaptive_knet",
        implementation_id="bench_adaptive_knet_adapter_v1",
        certified=False,
        training_path_id="legacy_train_v1",
        notes="Adapt-phase lifecycle is not modelled by this checkpoint schema (ADR-CSR-006).",
    ),
    CertificationRecord(
        model_id="maml_knet",
        implementation_id="bench_maml_knet_adapter_v1",
        certified=False,
        training_path_id="legacy_train_v1",
        notes="Meta inner/outer-loop cursor is not captured (ADR-CSR-006).",
    ),
    CertificationRecord(
        model_id="me_split_knet_v0",
        implementation_id="bench_me_split_adapter_v1",
        certified=False,
        training_path_id="legacy_train_v1",
        notes="Measurement-enhancer lifecycle is uncertified (ADR-CSR-006).",
    ),
    CertificationRecord(
        model_id="mb_kf",
        implementation_id="bench_mb_kf_adapter_v1",
        certified=False,
        training_path_id="not_applicable",
        notes="Model-based filter has no learning lifecycle; resume is not meaningful.",
    ),
)

#: Envelopes that are uncertified for *every* model, whatever the model says.
UNCERTIFIED_ENVELOPES = {
    "device_class": ["cuda", "gpu", "mps"],
    "precision": ["fp16", "bf16", "amp"],
    "num_workers": "any value other than 0",
}


def seed_certifications(registry: Any) -> int:
    """Write the certification matrix into a registry. Idempotent."""
    count = 0
    for record in (*CERTIFIED, *NOT_CERTIFIED):
        registry.upsert_certification(
            certification_key=record.key,
            model_id=record.model_id,
            implementation_id=record.implementation_id,
            checkpoint_schema_version=CHECKPOINT_SCHEMA_VERSION,
            resume_boundary=RESUME_BOUNDARY_OPTIMIZER_UPDATE,
            precision=record.precision,
            device_class=record.device_class,
            num_workers=record.num_workers,
            training_mode=record.training_mode,
            training_path_id=record.training_path_id,
            certified=record.certified,
            evidence_uri=record.evidence_uri,
            notes=record.notes,
        )
        count += 1
    return count


def is_certified(
    *,
    model_id: str,
    implementation_id: str,
    precision: str = "fp32",
    device_class: str = "cpu",
    num_workers: int = 0,
    training_mode: str = "supervised_single_optimizer",
    training_path_id: str = "control_resumable_v1",
    registry: Any = None,
) -> bool:
    """Exact lookup. An unlisted combination is uncertified, not assumed."""
    key = certification_key_for(
        model_id=model_id,
        implementation_id=implementation_id,
        precision=precision,
        device_class=device_class,
        num_workers=num_workers,
        training_mode=training_mode,
        training_path_id=training_path_id,
    ).as_key()

    if registry is not None:
        row = registry.get_certification(key)
        return bool(row and row["certified"])

    return any(record.key == key and record.certified for record in CERTIFIED)


def certification_matrix() -> list[dict[str, Any]]:
    """The full matrix, certified and not, for docs and read-only API display."""
    return [record.as_dict() for record in (*CERTIFIED, *NOT_CERTIFIED)]
