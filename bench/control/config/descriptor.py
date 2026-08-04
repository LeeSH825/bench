"""GUI schema descriptor, derived from the existing typed config.

The GUI does not get its own schema. This module describes the *same*
dataclasses the resolver already uses, so a form field and a CLI field cannot
drift apart. If a field is not here, the GUI does not claim to support it —
it is preserved in the raw YAML and surfaced as unsupported instead.

Structural vs operational matters: structural fields feed
``structural_config_hash`` and therefore ``variant_id``, so editing one makes
a genuinely different experiment. Operational fields do not.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from .schema import OPTIMIZERS, PRECISIONS

DESCRIPTOR_SCHEMA_VERSION = 1

STRUCTURAL = "structural"
OPERATIONAL = "operational"
IDENTITY = "identity"


@dataclass(frozen=True)
class FieldDescriptor:
    path: str
    type: str
    label: str
    help: str = ""
    default: Any = None
    required: bool = False
    enum: Optional[tuple[Any, ...]] = None
    minimum: Optional[float] = None
    maximum: Optional[float] = None
    read_only: bool = False
    classification: str = OPERATIONAL
    #: Rendered only when this predicate over the draft holds.
    visible_when: Optional[str] = None
    #: Capability the field depends on, if any.
    capability: Optional[str] = None
    is_path: bool = False
    sensitive: bool = False
    group: str = "experiment"

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": self.path, "type": self.type, "label": self.label,
            "help": self.help, "default": self.default, "required": self.required,
            "enum": list(self.enum) if self.enum else None,
            "minimum": self.minimum, "maximum": self.maximum,
            "read_only": self.read_only, "classification": self.classification,
            "visible_when": self.visible_when, "capability": self.capability,
            "is_path": self.is_path, "sensitive": self.sensitive, "group": self.group,
        }


#: Certified launch envelope. Offering anything else in the form would
#: advertise a capability the backend refuses.
CERTIFIED_DEVICES = ("cpu",)
CERTIFIED_PRECISIONS = ("fp32",)

_FIELDS: tuple[FieldDescriptor, ...] = (
    # -- experiment ---------------------------------------------------------
    FieldDescriptor("experiment.name", "string", "Experiment name",
                    "Inherited from the preset; shown for context.",
                    required=True, classification=OPERATIONAL, group="experiment"),
    FieldDescriptor("experiment.description", "string", "Run label / description",
                    "Free-text label for this run. Operational: it does not change identity.",
                    classification=OPERATIONAL, group="experiment"),
    FieldDescriptor("runtime.seed", "integer", "Seed",
                    "Random seed. Operational placement, but it does change the batch plan.",
                    default=0, minimum=0, maximum=2**31 - 1,
                    classification=OPERATIONAL, group="experiment"),
    FieldDescriptor("system.task_id", "string", "Task",
                    "Task from the preset. Structural.",
                    required=True, read_only=True, classification=STRUCTURAL,
                    group="experiment"),
    FieldDescriptor("model_id", "string", "Model",
                    "Model from the preset. Structural, and decides launch eligibility.",
                    required=True, read_only=True, classification=IDENTITY,
                    group="experiment"),
    FieldDescriptor("initialization.mode", "enum", "Initialization / plan",
                    "untrained runs evaluation only; trained runs the training plan.",
                    enum=("untrained", "trained"), classification=IDENTITY,
                    group="experiment"),

    # -- training (only meaningful for a trainable model on a trained plan) --
    FieldDescriptor("training.enabled", "boolean", "Training enabled",
                    "Disabled for model-based baselines, which have no learning lifecycle.",
                    default=True, classification=STRUCTURAL,
                    visible_when="model_trainable", capability="trainable", group="training"),
    FieldDescriptor("training.max_updates", "integer", "Max optimizer updates",
                    "Total completed optimizer updates. Structural: it changes the result.",
                    default=0, minimum=0, maximum=10_000_000,
                    classification=STRUCTURAL, visible_when="model_trainable",
                    capability="trainable", group="training"),
    FieldDescriptor("training.batch_size", "integer", "Batch size",
                    "Structural: batch size changes the batch plan and the numbers.",
                    default=1, minimum=1, maximum=4096,
                    classification=STRUCTURAL, visible_when="model_trainable",
                    capability="trainable", group="training"),
    FieldDescriptor("training.validation_interval_updates", "integer",
                    "Validation interval (updates)",
                    "How often validation runs, in completed updates.",
                    default=0, minimum=0, maximum=1_000_000,
                    classification=STRUCTURAL, visible_when="model_trainable",
                    capability="trainable", group="training"),
    FieldDescriptor("training.gradient_accumulation_steps", "integer",
                    "Gradient accumulation steps",
                    "Only 1 is certified for exact resume; other values fall back to the "
                    "legacy training path.",
                    default=1, minimum=1, maximum=1,
                    classification=STRUCTURAL, visible_when="model_trainable",
                    capability="trainable", group="training"),
    FieldDescriptor("optimizer.learning_rate", "number", "Learning rate",
                    "Structural: it is part of the experiment definition.",
                    default=1e-3, minimum=0.0, maximum=1.0,
                    classification=STRUCTURAL, visible_when="model_trainable",
                    capability="trainable", group="training"),
    FieldDescriptor("optimizer.weight_decay", "number", "Weight decay",
                    default=0.0, minimum=0.0, maximum=1.0,
                    classification=STRUCTURAL, visible_when="model_trainable",
                    capability="trainable", group="training"),
    FieldDescriptor("optimizer.name", "enum", "Optimizer",
                    "Structural identity input.",
                    default="adam", enum=tuple(sorted(OPTIMIZERS)),
                    classification=STRUCTURAL, visible_when="model_trainable",
                    capability="trainable", group="training"),

    # -- runtime ------------------------------------------------------------
    FieldDescriptor("runtime.device", "enum", "Device",
                    "Only CPU is certified for exact resume in this build.",
                    default="cpu", enum=CERTIFIED_DEVICES,
                    classification=OPERATIONAL, group="runtime"),
    FieldDescriptor("runtime.precision", "enum", "Precision",
                    "Only fp32 is certified. Precision is structural: it changes semantics.",
                    default="fp32", enum=CERTIFIED_PRECISIONS,
                    classification=STRUCTURAL, group="runtime"),
    FieldDescriptor("runtime.num_workers", "integer", "DataLoader workers",
                    "Only 0 is certified for exact resume.",
                    default=0, minimum=0, maximum=0,
                    classification=OPERATIONAL, group="runtime"),
    FieldDescriptor("runtime.deterministic", "boolean", "Deterministic algorithms",
                    "Structural: it changes numerical semantics.",
                    default=True, classification=STRUCTURAL, group="runtime"),
    FieldDescriptor("telemetry.enabled", "boolean", "Telemetry enabled",
                    default=True, classification=OPERATIONAL, group="runtime"),
    FieldDescriptor("telemetry.interval_seconds", "number", "Telemetry interval (s)",
                    default=5.0, minimum=0.5, maximum=3600.0,
                    classification=OPERATIONAL, group="runtime"),

    # -- output -------------------------------------------------------------
    FieldDescriptor("artifacts.emit_viz_artifacts", "boolean",
                    "Emit visualization artifacts",
                    "Writes prediction/trajectory artifacts for the Run Inspector.",
                    default=False, classification=OPERATIONAL, group="output"),
)

GROUPS = (
    {"id": "experiment", "label": "Experiment",
     "help": "What is being run. Most of this comes from the preset."},
    {"id": "training", "label": "Training",
     "help": "Budget and optimization. Hidden for models with no learning lifecycle."},
    {"id": "runtime", "label": "Runtime",
     "help": "Placement and observation. The certified envelope is CPU / fp32 / 0 workers."},
    {"id": "output", "label": "Output",
     "help": "What the run writes besides metrics."},
)


def descriptor_document() -> dict[str, Any]:
    """The machine-readable descriptor the GUI renders."""
    from .schema import CONFIG_SCHEMA_VERSION

    return {
        "schema_version": DESCRIPTOR_SCHEMA_VERSION,
        "config_schema_version": CONFIG_SCHEMA_VERSION,
        "groups": list(GROUPS),
        "fields": [f.as_dict() for f in _FIELDS],
        "classifications": {
            "structural": "Feeds structural_config_hash and variant_id; changing it "
                          "makes a different experiment.",
            "operational": "Does not change run identity.",
            "identity": "Part of the model/run identity itself.",
        },
        "certified_envelope": {
            "device": list(CERTIFIED_DEVICES),
            "precision": list(CERTIFIED_PRECISIONS),
            "num_workers": 0,
            "gradient_accumulation_steps": 1,
            "training_path_id": "control_resumable_v1",
        },
        "visibility_predicates": {
            "model_trainable": "True when the selected model has a learning lifecycle.",
        },
        "supported_field_paths": [f.path for f in _FIELDS],
    }


def field_by_path(path: str) -> Optional[FieldDescriptor]:
    for descriptor in _FIELDS:
        if descriptor.path == path:
            return descriptor
    return None


def supported_paths() -> frozenset[str]:
    return frozenset(f.path for f in _FIELDS)
