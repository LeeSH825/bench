from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch

from .measurement_enhancer import MeasurementEnhancer, enhancement_regularization, enhancer_diagnostics
from .split_knet import SplitKNetAdapter, _extract_batch_xy, _to_tensor, _write_json


class MESplitKNetV0Adapter(SplitKNetAdapter):
    """
    Measurement-Enhanced Split-KalmanNet v0.

    This is a wrapper prototype: a causal residual measurement enhancer is
    trained first, frozen, and then the unchanged Split-KalmanNet filter is
    trained/evaluated on y_enh = y_raw + delta_scale * E(y_raw).
    """

    def __init__(self) -> None:
        super().__init__()
        self.enhancer: Optional[MeasurementEnhancer] = None
        self.delta_scale: float = 1.0
        self.delta_clip_ratio: Optional[float] = None
        self.delta_clip_abs: Optional[float] = None
        self.lambda_delta: float = 1e-3
        self.lambda_smooth: float = 1e-3
        self.lambda_identity: float = 0.0
        self.w_imu_denoise: float = 1.0
        self.w_imu_corr: float = 0.5
        self.enhancer_updates_used: int = 0
        self.split_updates_used: int = 0
        self.train_outer_updates_used: int = 0
        self.train_inner_updates_used: int = 0
        self.enhancer_pretrain_target: str = "x"
        self._enhancer_diag: Dict[str, Any] = {}
        self._enhancer_arrays: Dict[str, Any] = {}
        self._enhancer_train_state: Dict[str, Any] = {}

    def setup(
        self,
        cfg: Dict[str, Any],
        system_info: Optional[Dict[str, Any]] = None,
        run_ctx: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().setup(cfg=cfg, system_info=system_info, run_ctx=run_ctx)
        if self._y_dim is None:
            raise RuntimeError("SplitKNet setup did not initialize y_dim.")
        self.delta_scale = float(cfg.get("delta_scale", 1.0))
        self.delta_clip_ratio = self._optional_float(cfg.get("delta_clip_ratio", None))
        self.delta_clip_abs = self._optional_float(cfg.get("delta_clip_abs", None))
        self.lambda_delta = float(cfg.get("lambda_delta", 1e-3))
        self.lambda_smooth = float(cfg.get("lambda_smooth", 1e-3))
        self.lambda_identity = float(cfg.get("lambda_identity", 0.0))
        self.w_imu_denoise = float(cfg.get("w_imu_denoise", 1.0))
        self.w_imu_corr = float(cfg.get("w_imu_corr", 0.5))
        target_default = "imu_clean_y_seq" if bool(cfg.get("imu_pretrain_enabled", False)) else "x"
        self.enhancer_pretrain_target = str(cfg.get("enhancer_pretrain_target", target_default)).strip().lower() or target_default
        self.enhancer = MeasurementEnhancer(
            int(self._y_dim),
            hidden_dim=int(cfg.get("enhancer_hidden_dim", 64)),
            num_layers=int(cfg.get("enhancer_num_layers", 2)),
            kernel_size=int(cfg.get("enhancer_kernel_size", 3)),
            dropout=float(cfg.get("enhancer_dropout", 0.0)),
            output_dim=int(cfg.get("enhancer_output_dim", self._y_dim)),
        ).to(device=self.device, dtype=self.dtype)
        self.enhancer.eval()

    def train(
        self,
        train_dl: Any,
        val_dl: Any,
        budget: Optional[Dict[str, Any]] = None,
        ckpt_dir: Optional[Union[str, Path]] = None,
    ) -> Dict[str, Any]:
        if self.enhancer is None:
            raise RuntimeError("setup() must be called before train().")
        budget = dict(budget or {})
        max_updates = int(budget.get("train_max_updates", 0))
        if max_updates <= 0:
            raise ValueError("train_max_updates must be > 0 for init_id=trained.")

        requested_enhancer_updates = int(
            self._cfg.get("enhancer_pretrain_updates", min(100, max(0, max_updates // 4)))
        )
        if self.enhancer_pretrain_target in {"none", "off", "disabled", "skip"}:
            requested_enhancer_updates = 0
        enhancer_updates = min(max(0, requested_enhancer_updates), max(0, max_updates - 1))
        split_budget = dict(budget)
        split_budget["train_max_updates"] = int(max_updates - enhancer_updates)
        if int(split_budget["train_max_updates"]) <= 0:
            raise ValueError("ME-Split-KNet v0 requires at least one Split-KalmanNet training update.")

        out_ckpt_dir = Path(ckpt_dir).expanduser().resolve() if ckpt_dir is not None else self._ckpt_dir
        if out_ckpt_dir is None:
            raise ValueError("ckpt_dir is required when adapter has no run_dir.")
        out_ckpt_dir.mkdir(parents=True, exist_ok=True)

        self.enhancer_updates_used = self._pretrain_enhancer(train_dl, max_updates=enhancer_updates)
        self._set_enhancer_trainable(False)

        base_result = super().train(train_dl=train_dl, val_dl=val_dl, budget=split_budget, ckpt_dir=out_ckpt_dir)
        self.split_updates_used = int(base_result.get("updates_used", getattr(self, "train_updates_used", 0)) or 0)
        total_updates = int(self.enhancer_updates_used + self.split_updates_used)
        self.train_updates_used = total_updates
        self.train_outer_updates_used = total_updates
        self.train_inner_updates_used = 0
        self._update_me_ledger(train_max_updates=max_updates)
        self._write_me_train_state(
            out_ckpt_dir=out_ckpt_dir,
            max_updates=max_updates,
            base_result=base_result,
        )
        save_result = self.save(out_ckpt_dir)
        base_result.update(
            {
                "ckpt_path": save_result["ckpt_path"],
                "updates_used": total_updates,
                "enhancer_updates_used": int(self.enhancer_updates_used),
                "split_updates_used": int(self.split_updates_used),
            }
        )
        return base_result

    def transform_measurements(
        self,
        y_btd: torch.Tensor,
        *,
        x_btd: Optional[torch.Tensor] = None,
        batch: Optional[Dict[str, Any]] = None,
        phase: str = "eval",
    ) -> torch.Tensor:
        _ = phase
        if self.enhancer is None:
            return y_btd
        y_enh, applied_delta, raw_delta = self._enhance_with_safety(y_btd)
        y_clean_imu = self._batch_extra_tensor(batch, "measurement_clean_y_seq", like=y_btd)
        if y_clean_imu is None:
            y_clean_imu = self._batch_extra_tensor(batch, "imu_clean_y_seq", like=y_btd)
        imu_error = self._batch_extra_tensor(batch, "measurement_error_seq", like=y_btd)
        if imu_error is None:
            imu_error = self._batch_extra_tensor(batch, "imu_error_seq", like=y_btd)
        self._enhancer_diag = enhancer_diagnostics(
            y_raw_btd=y_btd,
            y_enh_btd=y_enh,
            delta_btd=applied_delta,
            raw_delta_btd=raw_delta,
            x_ref_btd=None if y_clean_imu is not None else x_btd,
            y_clean_imu_btd=y_clean_imu,
            imu_error_btd=imu_error,
        )
        self._enhancer_arrays = {
            "y_raw_btd": y_btd.detach().cpu().numpy(),
            "y_enh_btd": y_enh.detach().cpu().numpy(),
            "delta_applied_btd": applied_delta.detach().cpu().numpy(),
            "delta_raw_btd": raw_delta.detach().cpu().numpy(),
        }
        if x_btd is not None:
            self._enhancer_arrays["x_ref_btd"] = x_btd.detach().cpu().numpy()
        if y_clean_imu is not None:
            self._enhancer_arrays["imu_clean_y_btd"] = y_clean_imu.detach().cpu().numpy()
        if imu_error is not None:
            self._enhancer_arrays["imu_error_btd"] = imu_error.detach().cpu().numpy()
        return y_enh

    def measurement_extra_loss(
        self,
        *,
        x_btd: torch.Tensor,
        y_raw_btd: torch.Tensor,
        y_model_btd: torch.Tensor,
        batch: Optional[Dict[str, Any]] = None,
        phase: str = "train",
    ) -> torch.Tensor:
        _ = x_btd, y_model_btd, batch, phase
        return y_raw_btd.new_zeros(())

    def load(self, ckpt_path: str) -> None:
        if self._filter_obj is None:
            raise RuntimeError("setup() must be called before load().")
        state = torch.load(ckpt_path, map_location=self.device)
        base_state = state
        if isinstance(state, dict) and "base_state_dict" in state:
            base_state = state["base_state_dict"]
            if self.enhancer is not None and "enhancer_state_dict" in state:
                self.enhancer.load_state_dict(state["enhancer_state_dict"], strict=True)
        elif isinstance(state, dict) and "state_dict" in state:
            base_state = state["state_dict"]
        self._filter_obj.kf_net.load_state_dict(base_state, strict=True)
        self._filter_obj.kf_net.to(self.device)
        self._filter_obj.kf_net.eval()
        if self.enhancer is not None:
            self.enhancer.to(device=self.device, dtype=self.dtype)
            self.enhancer.eval()

    def save(self, out_dir: Union[str, Path]) -> Dict[str, Any]:
        if self._filter_obj is None or self.enhancer is None:
            raise RuntimeError("setup() must be called before save().")
        out = Path(out_dir).expanduser().resolve()
        out.mkdir(parents=True, exist_ok=True)
        ckpt_path = out / "model.pt"
        torch.save(
            {
                "adapter_id": "me_split_knet_v0",
                "base_adapter": "split_knet",
                "base_state_dict": self._filter_obj.kf_net.state_dict(),
                "enhancer_state_dict": self.enhancer.state_dict(),
                "enhancer_config": self.enhancer.config_dict(),
                "delta_scale": float(self.delta_scale),
                "delta_clip_ratio": self.delta_clip_ratio,
                "delta_clip_abs": self.delta_clip_abs,
                "enhancer_updates_used": int(self.enhancer_updates_used),
                "split_updates_used": int(self.split_updates_used),
                "train_updates_used": int(self.train_updates_used),
            },
            ckpt_path,
        )
        self._saved_ckpt_path = ckpt_path
        return {"ckpt_path": str(ckpt_path)}

    def get_adapter_meta(self) -> Dict[str, Any]:
        meta = super().get_adapter_meta()
        meta.update(
            {
                "adapter_id": "me_split_knet_v0",
                "adapter_version": "me_split_v0_lite_wrapper",
                "base_adapter": "split_knet",
                "measurement_enhancer": self.enhancer.config_dict() if self.enhancer is not None else None,
                "delta_scale": float(self.delta_scale),
                "delta_clip_ratio": self.delta_clip_ratio,
                "delta_clip_abs": self.delta_clip_abs,
                "training_strategy": {
                    "stage_a": (
                        "skipped"
                        if self.enhancer_pretrain_target in {"none", "off", "disabled", "skip"}
                        else f"pretrain causal residual enhancer against target={self.enhancer_pretrain_target}"
                    ),
                    "stage_b": "freeze enhancer and train unchanged Split-KalmanNet on enhanced measurements",
                    "joint_finetune": False,
                    "budget_semantics": "train_max_updates is the total optimizer.step cap across enhancer and Split-KalmanNet",
                    "enhancer_pretrain_target": self.enhancer_pretrain_target,
                },
                "regularization": {
                    "lambda_delta": float(self.lambda_delta),
                    "lambda_smooth": float(self.lambda_smooth),
                    "lambda_identity": float(self.lambda_identity),
                    "w_imu_denoise": float(self.w_imu_denoise),
                    "w_imu_corr": float(self.w_imu_corr),
                },
            }
        )
        return meta

    def get_runtime_diagnostics(self) -> Dict[str, Any]:
        diag = super().get_runtime_diagnostics()
        diag.update(dict(self._enhancer_diag))
        diag.update(dict(self._enhancer_arrays))
        diag.update(
            {
                "enhancer_updates_used": int(self.enhancer_updates_used),
                "split_updates_used": int(self.split_updates_used),
            }
        )
        return diag

    def _pretrain_enhancer(self, train_dl: Any, *, max_updates: int) -> int:
        if self.enhancer is None:
            raise RuntimeError("setup() must be called before enhancer pretraining.")
        if max_updates <= 0:
            self._enhancer_train_state = {"status": "skipped", "updates_used": 0}
            return 0
        if self.enhancer_pretrain_target in {"none", "off", "disabled", "skip"}:
            self._enhancer_train_state = {
                "status": "skipped",
                "reason": "enhancer_pretrain_target disabled",
                "updates_used": 0,
            }
            return 0
        imu_targets = {"imu_clean_y_seq", "imu_clean_y", "imu", "measurement_clean_y_seq", "measurement_clean_y"}
        use_imu_target = self.enhancer_pretrain_target in imu_targets
        if self.enhancer_pretrain_target != "x" and not use_imu_target:
            raise NotImplementedError(
                "ME-Split-KNet v0 supports enhancer_pretrain_target='x', 'imu_clean_y_seq', "
                "'measurement_clean_y_seq', or 'none'. "
                f"Got {self.enhancer_pretrain_target!r}."
            )
        if self.enhancer_pretrain_target == "x" and self._x_dim != self._y_dim:
            raise ValueError(
                "ME-Split-KNet v0 enhancer pretraining requires y_dim == x_dim because x is used as clean target."
            )

        self._set_enhancer_trainable(True)
        self.enhancer.train()
        lr = float(self._cfg.get("enhancer_lr", self._cfg.get("lr", 1e-3)))
        wd = float(self._cfg.get("enhancer_weight_decay", 0.0))
        max_grad_norm = self._cfg.get(
            "enhancer_gradient_clip_norm",
            self._cfg.get("enhancer_max_grad_norm", self._cfg.get("gradient_clip_norm", self._cfg.get("max_grad_norm", 10.0))),
        )
        max_grad_norm_f = float(max_grad_norm) if max_grad_norm is not None else None
        optimizer = torch.optim.Adam(self.enhancer.parameters(), lr=lr, weight_decay=wd)
        mse = torch.nn.MSELoss(reduction="mean")

        updates = 0
        last_loss = None
        history: List[Dict[str, float]] = []
        while updates < max_updates:
            progressed = False
            for batch in train_dl:
                if updates >= max_updates:
                    break
                x_raw, y_raw = _extract_batch_xy(batch)
                x = _to_tensor(x_raw, device=self.device, dtype=self.dtype)
                y = _to_tensor(y_raw, device=self.device, dtype=self.dtype)
                if x.ndim != 3 or y.ndim != 3:
                    raise ValueError(f"shape_mismatch: expected rank-3 x,y; got x={tuple(x.shape)} y={tuple(y.shape)}")
                if self.enhancer_pretrain_target == "x" and x.shape != y.shape:
                    raise ValueError(
                        "ME-Split-KNet v0 enhancer pretraining requires same x/y shape; "
                        f"got x={tuple(x.shape)} y={tuple(y.shape)}"
                    )
                if use_imu_target:
                    target_key = (
                        "measurement_clean_y_seq"
                        if self.enhancer_pretrain_target in {"measurement_clean_y_seq", "measurement_clean_y"}
                        else "imu_clean_y_seq"
                    )
                    error_key = "measurement_error_seq" if target_key == "measurement_clean_y_seq" else "imu_error_seq"
                    y_clean = self._batch_extra_tensor(batch, target_key, like=y)
                    if y_clean is None:
                        raise KeyError(
                            f"ME-Split-KNet IMU enhancer pretraining requires batch extra {target_key!r}. "
                            "Refusing to fall back to x because IMU y and state x have different semantics."
                        )
                    imu_error = self._batch_extra_tensor(batch, error_key, like=y)
                    measurement_mask = self._batch_extra_tensor(batch, "measurement_mask_seq", like=y)
                else:
                    y_clean = x
                    imu_error = None
                    measurement_mask = None

                optimizer.zero_grad(set_to_none=True)
                y_enh, applied_delta, raw_delta = self._enhance_with_safety(y)
                regs = enhancement_regularization(raw_delta)
                denoise_loss = self._masked_mse(y_enh, y_clean, measurement_mask) if measurement_mask is not None else mse(y_enh, y_clean)
                corr_loss = y.new_zeros(())
                if imu_error is not None:
                    corr_loss = (
                        self._masked_mse(applied_delta, -imu_error, measurement_mask)
                        if measurement_mask is not None
                        else mse(applied_delta, -imu_error)
                    )
                target_loss = (
                    float(self.w_imu_denoise) * denoise_loss + float(self.w_imu_corr) * corr_loss
                    if use_imu_target
                    else denoise_loss
                )
                loss = (
                    target_loss
                    + float(self.lambda_delta) * regs["L_delta"]
                    + float(self.lambda_smooth) * regs["L_smooth"]
                    + float(self.lambda_identity) * mse(y_enh, y)
                )
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"train_nan: non-finite enhancer loss at update={updates}")
                loss.backward()
                if max_grad_norm_f is not None and max_grad_norm_f > 0:
                    torch.nn.utils.clip_grad_norm_(self.enhancer.parameters(), max_norm=max_grad_norm_f)
                optimizer.step()
                updates += 1
                progressed = True
                last_loss = float(loss.detach().cpu().item())
                if updates == 1 or updates == max_updates or (updates % max(1, min(25, max_updates)) == 0):
                    history.append(
                        {
                            "step": float(updates),
                            "loss": float(last_loss),
                            "L_target": float(target_loss.detach().cpu().item()),
                            "L_imu_denoise": float(denoise_loss.detach().cpu().item()) if use_imu_target else float("nan"),
                            "L_imu_corr": float(corr_loss.detach().cpu().item()) if imu_error is not None else float("nan"),
                            "L_delta": float(regs["L_delta"].detach().cpu().item()),
                            "L_smooth": float(regs["L_smooth"].detach().cpu().item()),
                        }
                    )
            if not progressed:
                break

        self.enhancer.eval()
        self._enhancer_train_state = {
            "status": "ok",
            "updates_used": int(updates),
            "requested_updates": int(max_updates),
            "last_loss": last_loss,
            "target": self.enhancer_pretrain_target,
            "w_imu_denoise": float(self.w_imu_denoise),
            "w_imu_corr": float(self.w_imu_corr),
            "history": history[-20:],
        }
        return int(updates)

    def _batch_extra_tensor(self, batch: Optional[Dict[str, Any]], key: str, *, like: torch.Tensor) -> Optional[torch.Tensor]:
        if not isinstance(batch, dict) or key not in batch:
            return None
        value = batch[key]
        tensor = _to_tensor(value, device=like.device, dtype=like.dtype)
        if tuple(tensor.shape) != tuple(like.shape):
            raise ValueError(f"shape_mismatch: batch extra {key!r} expected {tuple(like.shape)}, got {tuple(tensor.shape)}")
        return tensor

    @staticmethod
    def _masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        if mask is None:
            return torch.nn.functional.mse_loss(pred, target, reduction="mean")
        weight = mask.to(device=pred.device, dtype=pred.dtype)
        denom = weight.sum().clamp_min(torch.finfo(pred.dtype).eps)
        return torch.sum(((pred - target) ** 2) * weight) / denom

    def _enhance_with_safety(self, y_btd: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.enhancer is None:
            zero = torch.zeros_like(y_btd)
            return y_btd, zero, zero
        raw_delta = self.enhancer(y_btd)
        clipped_delta = self._clip_delta(y_raw_btd=y_btd, delta_btd=raw_delta)
        applied_delta = float(self.delta_scale) * clipped_delta
        return y_btd + applied_delta, applied_delta, raw_delta

    def _clip_delta(self, *, y_raw_btd: torch.Tensor, delta_btd: torch.Tensor) -> torch.Tensor:
        out = delta_btd
        if self.delta_clip_abs is not None:
            out = torch.clamp(out, min=-float(self.delta_clip_abs), max=float(self.delta_clip_abs))
        if self.delta_clip_ratio is not None:
            eps = torch.finfo(y_raw_btd.dtype).eps
            raw_norm = torch.linalg.norm(y_raw_btd, dim=2, keepdim=True).clamp_min(float(eps))
            max_norm = float(self.delta_clip_ratio) * raw_norm
            delta_norm = torch.linalg.norm(out, dim=2, keepdim=True).clamp_min(float(eps))
            scale = torch.minimum(torch.ones_like(delta_norm), max_norm / delta_norm)
            out = out * scale
        return out

    @staticmethod
    def _optional_float(value: Any) -> Optional[float]:
        if value is None:
            return None
        if isinstance(value, str) and value.strip().lower() in {"", "none", "null", "nan"}:
            return None
        return float(value)

    def _set_enhancer_trainable(self, trainable: bool) -> None:
        if self.enhancer is None:
            return
        for p in self.enhancer.parameters():
            p.requires_grad_(bool(trainable))
        if trainable:
            self.enhancer.train()
        else:
            self.enhancer.eval()

    def _update_me_ledger(self, *, train_max_updates: int) -> None:
        if self._ledger_path is None:
            return
        current: Dict[str, Any] = {}
        if self._ledger_path.exists():
            try:
                current = json.loads(self._ledger_path.read_text(encoding="utf-8"))
                if not isinstance(current, dict):
                    current = {}
            except Exception:
                current = {}
        current.update(
            {
                "train_updates_used": int(self.train_updates_used),
                "train_outer_updates_used": int(self.train_outer_updates_used),
                "train_inner_updates_used": 0,
                "enhancer_updates_used": int(self.enhancer_updates_used),
                "split_updates_used": int(self.split_updates_used),
                "adapt_updates_used": int(self.adapt_updates_used),
                "train_max_updates": int(train_max_updates),
                "track_id": self._run_ctx.get("track_id"),
                "init_id": self._run_ctx.get("init_id"),
            }
        )
        _write_json(self._ledger_path, current)

    def _write_me_train_state(
        self,
        *,
        out_ckpt_dir: Path,
        max_updates: int,
        base_result: Dict[str, Any],
    ) -> None:
        train_state_path = out_ckpt_dir / "train_state.json"
        state: Dict[str, Any] = {}
        if train_state_path.exists():
            try:
                loaded = json.loads(train_state_path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    state = loaded
            except Exception:
                state = {}
        state.update(
            {
                "status": "ok",
                "updates_used": int(self.train_updates_used),
                "max_updates": int(max_updates),
                "enhancer_updates_used": int(self.enhancer_updates_used),
                "split_updates_used": int(self.split_updates_used),
                "split_train_result": {
                    k: v
                    for k, v in base_result.items()
                    if k not in {"ckpt_path", "train_state_path"}
                },
                "measurement_enhancer": self._enhancer_train_state,
            }
        )
        _write_json(train_state_path, state)
        self._train_state_path = train_state_path
