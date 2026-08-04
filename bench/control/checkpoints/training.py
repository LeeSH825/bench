"""Resumable training driver.

This owns the update loop, the cursor, the checkpoint cadence, and the stop
handshake. The *numerics* stay in the adapters: the driver only ever calls
``train_step`` and ``validate`` hooks, so the forward/backward path that was
certified numerically inert in the previous tranche is not reimplemented here.

The loop body is a deliberate mirror of the adapters' own ``train()`` ordering:

    zero_grad → forward → loss → backward → clip → step
    → updates_used += 1
    → scheduled validation, best/early-stop update
    → checkpoint-safe boundary

The safe boundary is *after* the validation/early-stop block, so a checkpoint
never lands between an update and the validation that update was supposed to
trigger (ADR-CSR-001).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Protocol

import numpy as np

from .batchplan import BatchPlan
from .schema import (
    AdapterTrainingState,
    CheckpointCapabilities,
    CheckpointKind,
    TrainingCursor,
)


class CheckpointableTrainingAdapter(Protocol):
    """What the driver requires of an adapter (§8)."""

    def checkpoint_capabilities(self) -> CheckpointCapabilities: ...

    def capture_training_state(self, cursor: TrainingCursor) -> AdapterTrainingState: ...

    def restore_training_state(self, state: AdapterTrainingState) -> TrainingCursor: ...

    def checkpoint_safe(self) -> bool: ...


@dataclass
class TrainingHooks:
    """Adapter-supplied numerical body.

    ``train_step`` must perform exactly one optimizer update and return the
    scalar training loss. ``validate`` must run the adapter's own validation
    path and return the scalar validation loss.
    """

    train_step: Callable[[np.ndarray], float]
    validate: Callable[[], float]
    snapshot_best: Callable[[], dict[str, Any]]
    restore_best: Callable[[dict[str, Any]], None]


@dataclass
class TrainingSchedule:
    """Cadence knobs, read from the adapter config exactly as ``train()`` does."""

    max_updates: int
    eval_interval: int = 1
    patience_evals: int = 0
    min_delta: float = 0.0
    checkpoint_interval: int = 0


@dataclass
class TrainingProgress:
    """Mutable training position and early-stop bookkeeping.

    Everything here is checkpointed; anything not here cannot survive a resume.
    """

    global_update: int = 0
    batch_plan_position: int = 0
    best_step: int = 0
    best_val: float = float("inf")
    no_improve_evals: int = 0
    last_train_loss: Optional[float] = None
    train_loss_history: list[float] = field(default_factory=list)
    val_history: list[dict[str, float]] = field(default_factory=list)
    stopped_early: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "global_update": int(self.global_update),
            "batch_plan_position": int(self.batch_plan_position),
            "best_step": int(self.best_step),
            "best_val": float(self.best_val),
            "no_improve_evals": int(self.no_improve_evals),
            "last_train_loss": self.last_train_loss,
            "train_loss_history": list(self.train_loss_history),
            "val_history": [dict(v) for v in self.val_history],
            "stopped_early": bool(self.stopped_early),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TrainingProgress":
        data = dict(data or {})
        return cls(
            global_update=int(data.get("global_update", 0)),
            batch_plan_position=int(data.get("batch_plan_position", 0)),
            best_step=int(data.get("best_step", 0)),
            best_val=float(data.get("best_val", float("inf"))),
            no_improve_evals=int(data.get("no_improve_evals", 0)),
            last_train_loss=data.get("last_train_loss"),
            train_loss_history=list(data.get("train_loss_history", [])),
            val_history=[dict(v) for v in data.get("val_history", [])],
            stopped_early=bool(data.get("stopped_early", False)),
        )


class StopRequested(Exception):
    """Raised internally to unwind to the interrupt-checkpoint path."""

    def __init__(self, progress: TrainingProgress):
        super().__init__("stop requested at a safe boundary")
        self.progress = progress


@dataclass
class TrainingResult:
    progress: TrainingProgress
    interrupted: bool = False
    cursor: Optional[TrainingCursor] = None


class ResumableTrainer:
    """Drives training over a :class:`BatchPlan` with checkpointable cursors."""

    def __init__(
        self,
        *,
        hooks: TrainingHooks,
        schedule: TrainingSchedule,
        plan: BatchPlan,
        observer: Optional[Any] = None,
        stop_requested: Optional[Callable[[], bool]] = None,
        on_checkpoint: Optional[Callable[[CheckpointKind, TrainingProgress], None]] = None,
    ) -> None:
        self.hooks = hooks
        self.schedule = schedule
        self.plan = plan
        self.observer = observer
        self.stop_requested = stop_requested or (lambda: False)
        self.on_checkpoint = on_checkpoint

    def cursor_for(self, progress: TrainingProgress) -> TrainingCursor:
        return TrainingCursor(
            global_update=progress.global_update,
            epoch=progress.batch_plan_position // max(1, self.plan.batches_per_epoch),
            batch_plan_position=progress.batch_plan_position,
            batch_plan_id=self.plan.plan_id,
            phase="train",
        )

    def run(self, progress: Optional[TrainingProgress] = None) -> TrainingResult:
        """Train until the update budget, an early stop, or a stop request."""
        progress = progress or TrainingProgress()
        schedule = self.schedule
        batches = self.plan.iter_from(progress.batch_plan_position)

        while progress.global_update < schedule.max_updates:
            position, indices = next(batches)

            loss = self.hooks.train_step(indices)
            progress.global_update += 1
            progress.batch_plan_position = position + 1
            progress.last_train_loss = float(loss)
            progress.train_loss_history.append(float(loss))
            self._metric("loss/train_total", float(loss), progress.global_update, "train")

            should_eval = (
                progress.global_update % max(1, schedule.eval_interval) == 0
                or progress.global_update == schedule.max_updates
            )
            if should_eval:
                val_loss = float(self.hooks.validate())
                progress.val_history.append(
                    {"step": float(progress.global_update), "val_mse": float(val_loss)}
                )
                self._metric(
                    "loss/validation_total", val_loss, progress.global_update, "validation"
                )
                if (progress.best_val - val_loss) > schedule.min_delta:
                    progress.best_val = val_loss
                    progress.best_step = int(progress.global_update)
                    self.hooks.snapshot_best()
                    progress.no_improve_evals = 0
                else:
                    progress.no_improve_evals += 1

                if (
                    schedule.patience_evals > 0
                    and progress.no_improve_evals >= schedule.patience_evals
                ):
                    progress.stopped_early = True
                    break

            # Checkpoint-safe boundary: the update and any validation it was
            # due to trigger are both complete.
            if (
                schedule.checkpoint_interval > 0
                and progress.global_update % schedule.checkpoint_interval == 0
                and self.on_checkpoint is not None
            ):
                self.on_checkpoint(CheckpointKind.PERIODIC, progress)

            if self.stop_requested():
                # Unwind to the caller, which writes the interrupt checkpoint
                # *before* any terminal state is recorded (DND-CSR-004).
                return TrainingResult(
                    progress=progress, interrupted=True, cursor=self.cursor_for(progress)
                )

        return TrainingResult(
            progress=progress, interrupted=False, cursor=self.cursor_for(progress)
        )

    def _metric(self, name: str, value: float, step: int, phase: str) -> None:
        if self.observer is None:
            return
        self.observer.metric(name, value, step=int(step), phase=phase, unit="mse")
