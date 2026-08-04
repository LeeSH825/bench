"""Checkpointable-training support shared by the certified adapters.

This is deliberately *additive*: it introduces a second, opt-in training entry
point (:meth:`CheckpointableAdapterMixin.resumable_train`) and leaves each
adapter's existing ``train()`` untouched. The previous tranche certified that
observer and telemetry are numerically inert on ``train()``; nothing here may
disturb that, so nothing here is called unless a run explicitly asks for
checkpointable training.

Each adapter supplies three small hooks — which module holds the weights, how
to compute a training loss for one batch, and how to compute validation loss —
and inherits the optimizer/best-state/RNG plumbing.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import torch

from bench.control.checkpoints.batchplan import BatchPlan
from bench.control.checkpoints.schema import (
    AdapterTrainingState,
    CheckpointCapabilities,
    CheckpointUnsupportedError,
    TrainingCursor,
)
from bench.control.checkpoints.training import (
    ResumableTrainer,
    TrainingHooks,
    TrainingProgress,
    TrainingResult,
    TrainingSchedule,
)

#: Slot names are part of the checkpoint contract: a reader must never have to
#: infer how many state dicts a payload holds (§8).
MODEL_SLOT = "model"
OPTIMIZER_SLOT = "main"


class _Session:
    """Live training objects for one resumable training call."""

    def __init__(self, optimizer: torch.optim.Optimizer, loss_fn: Any, params: list[torch.Tensor]):
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.params = params
        self.best_state: dict[str, torch.Tensor] = {}
        self.max_grad_norm: Optional[float] = None
        self.train_x: Optional[np.ndarray] = None
        self.train_y: Optional[np.ndarray] = None
        self.val_batches: list[dict[str, np.ndarray]] = []


class CheckpointableAdapterMixin:
    """Adds capture/restore and a resumable training loop to an adapter."""

    _ckpt_session: Optional[_Session] = None

    # -- hooks each adapter must provide ------------------------------------

    def _ckpt_model_module(self) -> torch.nn.Module:
        raise NotImplementedError

    def _ckpt_batch_loss(self, x: torch.Tensor, y: torch.Tensor, loss_fn: Any) -> torch.Tensor:
        raise NotImplementedError

    def _ckpt_validation_loss(self, val_batches: list[dict[str, np.ndarray]], loss_fn: Any) -> float:
        raise NotImplementedError

    def _ckpt_implementation_id(self) -> str:
        return getattr(self, "implementation_id", "") or ""

    def _ckpt_extra_state(self) -> dict[str, Any]:
        """Adapter state that is part of the numerical path but not in ``state_dict()``.

        Default: nothing. Overridden where the underlying third-party module
        keeps numerically-significant tensors as plain attributes rather than
        registered buffers — in which case they are invisible to
        ``state_dict()`` and would silently not survive a resume.
        """
        return {}

    def _ckpt_restore_extra(self, extra: dict[str, Any]) -> None:
        """Inverse of :meth:`_ckpt_extra_state`."""
        return None

    # -- capability ----------------------------------------------------------

    def checkpoint_capabilities(self) -> CheckpointCapabilities:
        """What this adapter can round-trip, and under what envelope.

        Both certified adapters run a single Adam over one module, with no
        scheduler and no GradScaler, and re-initialise recurrent hidden state
        per batch — so there is no persistent hidden state to carry across an
        update boundary. That last fact is what makes the optimizer-update
        boundary sufficient here rather than merely convenient.
        """
        return CheckpointCapabilities(
            supports_exact_resume=True,
            resume_boundary="optimizer_update",
            model_slots=(MODEL_SLOT,),
            optimizer_slots=(OPTIMIZER_SLOT,),
            scheduler_slots=(),
            has_grad_scaler=False,
            required_conditional_state=(),
            certified_device="cpu",
            certified_precision="fp32",
            certified_num_workers=0,
            certified_training_mode="supervised_single_optimizer",
            notes=(
                "Recurrent hidden state is re-initialised per batch, so no persistent "
                "hidden state crosses an update boundary."
            ),
        )

    def checkpoint_safe(self) -> bool:
        """True only between completed updates, with no live gradients."""
        session = self._ckpt_session
        if session is None:
            return False
        return all(p.grad is None or torch.all(torch.isfinite(p.grad)) for p in session.params)

    # -- capture / restore ---------------------------------------------------

    def capture_training_state(self, cursor: TrainingCursor) -> AdapterTrainingState:
        session = self._require_session()
        module = self._ckpt_model_module()
        return AdapterTrainingState(
            model_slots={MODEL_SLOT: _cpu_state_dict(module)},
            optimizer_slots={OPTIMIZER_SLOT: session.optimizer.state_dict()},
            scheduler_slots={},
            grad_scaler=None,
            best_state={
                "weights": {k: v.clone() for k, v in session.best_state.items()},
            },
            validation_state={},
            extra_state={"cursor": cursor.as_dict(), "adapter": self._ckpt_extra_state()},
        )

    def restore_training_state(self, state: AdapterTrainingState) -> TrainingCursor:
        session = self._require_session()
        module = self._ckpt_model_module()

        if MODEL_SLOT not in state.model_slots:
            raise CheckpointUnsupportedError(
                f"checkpoint has no {MODEL_SLOT!r} model slot; slots present: "
                f"{sorted(state.model_slots)}"
            )
        if OPTIMIZER_SLOT not in state.optimizer_slots:
            raise CheckpointUnsupportedError(
                f"checkpoint has no {OPTIMIZER_SLOT!r} optimizer slot; restoring weights "
                "without optimizer state would be a warm start, not an exact resume"
            )

        module.load_state_dict(state.model_slots[MODEL_SLOT], strict=True)
        session.optimizer.load_state_dict(state.optimizer_slots[OPTIMIZER_SLOT])
        _assert_optimizer_devices(session.optimizer, module)

        best = dict(state.best_state or {})
        session.best_state = {k: v.clone() for k, v in dict(best.get("weights") or {}).items()}
        extra = dict(state.extra_state or {})
        self._ckpt_restore_extra(dict(extra.get("adapter") or {}))
        return TrainingCursor.from_dict(extra.get("cursor", {}))

    # -- resumable training --------------------------------------------------

    def begin_resumable_training(
        self,
        *,
        train_x: np.ndarray,
        train_y: np.ndarray,
        val_batches: list[dict[str, np.ndarray]],
        lr: float,
        weight_decay: float = 0.0,
        max_grad_norm: Optional[float] = None,
    ) -> None:
        """Create the optimizer/loss for a resumable training call."""
        module = self._ckpt_model_module()
        params = [p for p in module.parameters() if p.requires_grad]
        if not params:
            raise CheckpointUnsupportedError("adapter has no trainable parameters")
        optimizer = torch.optim.Adam(params, lr=float(lr), weight_decay=float(weight_decay))
        session = _Session(optimizer, torch.nn.MSELoss(reduction="mean"), params)
        session.max_grad_norm = max_grad_norm
        session.train_x = np.ascontiguousarray(train_x)
        session.train_y = np.ascontiguousarray(train_y)
        session.val_batches = list(val_batches)
        session.best_state = _cpu_state_dict(module)
        self._ckpt_session = session
        module.train()

    def resumable_train(
        self,
        *,
        plan: BatchPlan,
        schedule: TrainingSchedule,
        progress: Optional[TrainingProgress] = None,
        observer: Optional[Any] = None,
        stop_requested: Optional[Any] = None,
        on_checkpoint: Optional[Any] = None,
    ) -> TrainingResult:
        """Run the resumable loop. Call :meth:`begin_resumable_training` first."""
        self._require_session()
        trainer = ResumableTrainer(
            hooks=TrainingHooks(
                train_step=self._ckpt_step,
                validate=self._ckpt_validate,
                snapshot_best=self._ckpt_snapshot_best,
                restore_best=self._ckpt_restore_best,
            ),
            schedule=schedule,
            plan=plan,
            observer=observer,
            stop_requested=stop_requested,
            on_checkpoint=on_checkpoint,
        )
        return trainer.run(progress)

    def finish_resumable_training(self) -> None:
        """Load the best weights back, mirroring ``train()``'s final step."""
        session = self._require_session()
        if session.best_state:
            self._ckpt_model_module().load_state_dict(session.best_state, strict=True)

    # -- internals -----------------------------------------------------------

    def _require_session(self) -> _Session:
        if self._ckpt_session is None:
            raise CheckpointUnsupportedError(
                "no resumable training session; call begin_resumable_training() first"
            )
        return self._ckpt_session

    def _ckpt_step(self, indices: np.ndarray) -> float:
        session = self._require_session()
        x = torch.as_tensor(session.train_x[indices], dtype=self.dtype, device=self.device)
        y = torch.as_tensor(session.train_y[indices], dtype=self.dtype, device=self.device)

        session.optimizer.zero_grad(set_to_none=True)
        loss = self._ckpt_batch_loss(x, y, session.loss_fn)
        if not torch.isfinite(loss):
            raise FloatingPointError("train_nan: non-finite training loss")
        loss.backward()
        if session.max_grad_norm is not None and session.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(session.params, max_norm=float(session.max_grad_norm))
        session.optimizer.step()
        return float(loss.detach().item())

    def _ckpt_validate(self) -> float:
        session = self._require_session()
        return float(self._ckpt_validation_loss(session.val_batches, session.loss_fn))

    def _ckpt_snapshot_best(self) -> dict[str, Any]:
        session = self._require_session()
        session.best_state = _cpu_state_dict(self._ckpt_model_module())
        return session.best_state

    def _ckpt_restore_best(self, state: dict[str, Any]) -> None:
        if state:
            self._ckpt_model_module().load_state_dict(state, strict=True)


def _cpu_state_dict(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}


def _assert_optimizer_devices(optimizer: torch.optim.Optimizer, module: torch.nn.Module) -> None:
    """Optimizer state must live on the same device as the parameters.

    ``load_state_dict`` maps optimizer state onto the *current* parameters, so a
    mismatch here means the checkpoint was captured under a different device
    layout than the one being resumed into.
    """
    target = next((p.device for p in module.parameters()), torch.device("cpu"))
    for state in optimizer.state.values():
        for value in state.values():
            if isinstance(value, torch.Tensor) and value.device != target:
                raise CheckpointUnsupportedError(
                    f"optimizer state on {value.device} but parameters on {target}; "
                    "cross-device resume is not certified"
                )
