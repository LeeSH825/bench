"""Legacy Euclidean 6D-observation SpikeRA benchmark adapter.

This is not the current right-local Phase 2 estimator: gyro is part of the
six-dimensional observation here, not a propagation/process input. Event
labels are training-loss and post-hoc diagnostic metadata only; deployable
forward inference receives causal innovation-derived tensors.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

import numpy as np
import torch
from torch import nn

from bench.metrics.adcs_event import attitude_error_deg
from bench.utils.logging import get_logger

from .spike_split_knet import RecurrentLIFCell
from .split_knet import SplitKNetAdapter


logger = get_logger(__name__)


class SNNReliabilityAdapter(nn.Module):
    """Recurrent LIF adapter producing a scalar measurement reliability gate."""

    def __init__(
        self,
        input_dim: int,
        *,
        hidden_dim: int = 32,
        beta: float = 0.9,
        threshold: float = 1.0,
        surrogate_alpha: float = 10.0,
        max_suppression: float = 0.5,
        initial_alpha: float = 0.99,
        detach_recurrent_state: bool = False,
    ) -> None:
        super().__init__()
        if not 0.0 < float(max_suppression) < 1.0:
            raise ValueError("max_suppression must be in (0, 1)")
        min_alpha = 1.0 - float(max_suppression)
        if not min_alpha < float(initial_alpha) < 1.0:
            raise ValueError(
                f"initial_alpha must be in ({min_alpha}, 1) for "
                f"max_suppression={max_suppression}"
            )

        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.max_suppression = float(max_suppression)
        self.initial_alpha = float(initial_alpha)
        self.detach_recurrent_state = bool(detach_recurrent_state)
        self.cell = RecurrentLIFCell(
            self.input_dim,
            self.hidden_dim,
            beta=beta,
            threshold=threshold,
            surrogate_alpha=surrogate_alpha,
        )
        self.decoder = nn.Linear(2 * self.hidden_dim, 1)
        self._fanout = 1
        self._last_alpha = torch.empty(0)
        self._trace_enabled = False
        self._trace_sequences: List[List[Dict[str, float]]] = []
        self._trace_current: Optional[List[Dict[str, float]]] = None
        self.reset_stats()
        self._initialize_decoder()

    def _initialize_decoder(self) -> None:
        nn.init.zeros_(self.decoder.weight)
        suppression_fraction = (1.0 - self.initial_alpha) / self.max_suppression
        raw_bias = math.log(suppression_fraction / (1.0 - suppression_fraction))
        nn.init.constant_(self.decoder.bias, raw_bias)

    def reset_state(
        self,
        batch_size: int = 1,
        *,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        self.cell.reset_state(batch_size, device=device, dtype=dtype)
        if self._trace_enabled:
            self._trace_current = []
            self._trace_sequences.append(self._trace_current)

    def reset_stats(self) -> None:
        self._spike_count = 0.0
        self._spike_numel = 0
        self._alpha_sum = 0.0
        self._alpha_numel = 0
        self._alpha_min = float("inf")
        self._alpha_max = float("-inf")

    def get_stats(self) -> Dict[str, Union[int, float]]:
        spike_rate = (
            self._spike_count / float(self._spike_numel)
            if self._spike_numel > 0
            else 0.0
        )
        mean_alpha = (
            self._alpha_sum / float(self._alpha_numel)
            if self._alpha_numel > 0
            else float("nan")
        )
        return {
            "spike_count": float(self._spike_count),
            "spike_numel": int(self._spike_numel),
            "avg_spike_rate": float(spike_rate),
            "active_ops_proxy": float(self._spike_count * self._fanout),
            "fanout": int(self._fanout),
            "mean_alpha": float(mean_alpha),
            "min_alpha": (
                float(self._alpha_min) if self._alpha_numel > 0 else float("nan")
            ),
            "max_alpha": (
                float(self._alpha_max) if self._alpha_numel > 0 else float("nan")
            ),
            "mean_suppression": (
                float(1.0 - mean_alpha) if self._alpha_numel > 0 else float("nan")
            ),
        }

    def reset_trace(self) -> None:
        self._trace_sequences = []
        self._trace_current = None

    def set_trace_enabled(self, enabled: bool) -> None:
        self._trace_enabled = bool(enabled)
        if not self._trace_enabled:
            self._trace_current = None

    def get_trace(self) -> List[List[Dict[str, float]]]:
        return [[dict(row) for row in sequence] for sequence in self._trace_sequences]

    def forward(
        self,
        reliability_input: torch.Tensor,
        *,
        innovation: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.detach_recurrent_state:
            self.cell.detach_state()
        membrane, spike = self.cell(reliability_input)
        decoder_input = torch.cat((membrane, spike), dim=-1)
        raw_gate = self.decoder(decoder_input)
        alpha = 1.0 - self.max_suppression * torch.sigmoid(raw_gate)
        self._last_alpha = alpha

        spike_detached = spike.detach()
        alpha_detached = alpha.detach()
        self._spike_count += float(spike_detached.sum().cpu().item())
        self._spike_numel += int(spike_detached.numel())
        self._alpha_sum += float(alpha_detached.sum().cpu().item())
        self._alpha_numel += int(alpha_detached.numel())
        self._alpha_min = min(
            self._alpha_min,
            float(alpha_detached.min().cpu().item()),
        )
        self._alpha_max = max(
            self._alpha_max,
            float(alpha_detached.max().cpu().item()),
        )
        if self._trace_enabled:
            if self._trace_current is None:
                self._trace_current = []
                self._trace_sequences.append(self._trace_current)
            innovation_tensor = reliability_input if innovation is None else innovation
            self._trace_current.append(
                {
                    "alpha": float(alpha_detached.mean().cpu().item()),
                    "spike_rate": float(spike_detached.float().mean().cpu().item()),
                    "spike_count": float(spike_detached.sum().cpu().item()),
                    "spike_numel": float(spike_detached.numel()),
                    "innovation_norm": float(
                        torch.linalg.norm(innovation_tensor.detach().float()).cpu().item()
                    ),
                }
            )
        return alpha


class SpikeRAKNet(nn.Module):
    """Original Split-KalmanNet G1/G2 with an SNN gate applied to Sk."""

    def __init__(
        self,
        original_net: nn.Module,
        *,
        hidden_dim: int = 32,
        beta: float = 0.9,
        threshold: float = 1.0,
        surrogate_alpha: float = 10.0,
        max_suppression: float = 0.5,
        initial_alpha: float = 0.99,
        detach_recurrent_state: bool = False,
    ) -> None:
        super().__init__()
        required = (
            "x_dim",
            "y_dim",
            "input_dim_1",
            "input_dim_2",
            "gru_input_dim",
            "gru_hidden_dim",
            "gru_n_layer",
            "batch_size",
            "seq_len_input",
            "l1",
            "GRU1",
            "l2",
            "l3",
            "GRU2",
            "l4",
            "hn1_init",
            "hn2_init",
        )
        missing = [name for name in required if not hasattr(original_net, name)]
        if missing:
            raise TypeError(f"Split-KalmanNet network is missing fields: {missing}")

        self.x_dim = int(original_net.x_dim)
        self.y_dim = int(original_net.y_dim)
        self.input_dim_1 = int(original_net.input_dim_1)
        self.input_dim_2 = int(original_net.input_dim_2)
        self.gru_input_dim = int(original_net.gru_input_dim)
        self.gru_hidden_dim = int(original_net.gru_hidden_dim)
        self.gru_n_layer = int(original_net.gru_n_layer)
        self.batch_size = int(original_net.batch_size)
        self.seq_len_input = int(original_net.seq_len_input)

        # Reuse both original Split-KalmanNet branches without copying modules.
        self.l1 = original_net.l1
        self.GRU1 = original_net.GRU1
        self.l2 = original_net.l2
        self.l3 = original_net.l3
        self.GRU2 = original_net.GRU2
        self.l4 = original_net.l4
        self.register_buffer(
            "hn1_init",
            original_net.hn1_init.detach().clone(),
            persistent=False,
        )
        self.register_buffer(
            "hn2_init",
            original_net.hn2_init.detach().clone(),
            persistent=False,
        )
        self.hn1 = self.hn1_init.detach().clone()
        self.hn2 = self.hn2_init.detach().clone()

        self.reliability_adapter = SNNReliabilityAdapter(
            self.input_dim_2,
            hidden_dim=hidden_dim,
            beta=beta,
            threshold=threshold,
            surrogate_alpha=surrogate_alpha,
            max_suppression=max_suppression,
            initial_alpha=initial_alpha,
            detach_recurrent_state=detach_recurrent_state,
        )

    def initialize_hidden(self) -> None:
        parameter = next(self.parameters())
        self.hn1 = self.hn1_init.detach().clone().to(
            device=parameter.device,
            dtype=parameter.dtype,
        )
        self.hn2 = self.hn2_init.detach().clone().to(
            device=parameter.device,
            dtype=parameter.dtype,
        )
        self.reliability_adapter.reset_state(
            self.batch_size,
            device=parameter.device,
            dtype=parameter.dtype,
        )

    def reset_spike_ra_stats(self) -> None:
        self.reliability_adapter.reset_stats()

    def get_spike_ra_stats(self) -> Dict[str, Union[int, float]]:
        return self.reliability_adapter.get_stats()

    def reset_spike_ra_trace(self) -> None:
        self.reliability_adapter.reset_trace()

    def set_spike_ra_trace_enabled(self, enabled: bool) -> None:
        self.reliability_adapter.set_trace_enabled(enabled)

    def get_spike_ra_trace(self) -> List[List[Dict[str, float]]]:
        return self.reliability_adapter.get_trace()

    def forward(
        self,
        state_inno: torch.Tensor,
        observation_inno: torch.Tensor,
        diff_state: torch.Tensor,
        diff_obs: torch.Tensor,
        linearization_error: torch.Tensor,
        jacobian: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        input1 = torch.cat(
            (state_inno, diff_state, linearization_error, jacobian),
            dim=0,
        ).reshape(-1)
        input2 = torch.cat(
            (observation_inno, diff_obs, linearization_error, jacobian),
            dim=0,
        ).reshape(-1)

        l1_out = self.l1(input1)
        gru1_input = torch.zeros(
            self.seq_len_input,
            self.batch_size,
            self.gru_input_dim,
            device=l1_out.device,
            dtype=l1_out.dtype,
        )
        gru1_input[0, 0, :] = l1_out
        gru1_out, self.hn1 = self.GRU1(gru1_input, self.hn1)
        pk_base = self.l2(gru1_out).reshape(self.x_dim, self.x_dim)

        l3_out = self.l3(input2)
        gru2_input = torch.zeros(
            self.seq_len_input,
            self.batch_size,
            self.gru_input_dim,
            device=l3_out.device,
            dtype=l3_out.dtype,
        )
        gru2_input[0, 0, :] = l3_out
        gru2_out, self.hn2 = self.GRU2(gru2_input, self.hn2)
        sk_base = self.l4(gru2_out).reshape(self.y_dim, self.y_dim)

        alpha = self.reliability_adapter(
            input2,
            innovation=observation_inno,
        ).reshape(())
        return pk_base, alpha * sk_base


class SpikeRAKNetAdapter(SplitKNetAdapter):
    """Legacy Split-KalmanNet adapter with a recurrent LIF reliability gate."""

    def __init__(self) -> None:
        super().__init__()
        self._spike_ra_cfg: Dict[str, Any] = {}
        self._reliability_gate_metrics: Dict[str, Any] = {}
        self._gate_trace_path: Optional[Path] = None
        self._adapter_only_train = False
        self._freeze_base = False
        self._event_loss_lambda = 0.0
        self._base_checkpoint_path: Optional[Path] = None
        self._base_checkpoint_report: Dict[str, Any] = {}
        self._frozen_module_names: List[str] = []
        self._trainable_parameter_names: List[str] = []
        self._stage2_train_summary: Dict[str, Any] = {}

    def setup(
        self,
        cfg: Dict[str, Any],
        system_info: Optional[Dict[str, Any]] = None,
        run_ctx: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().setup(cfg, system_info=system_info, run_ctx=run_ctx)
        self._spike_ra_cfg = {
            "snn_hidden_dim": int(cfg.get("snn_hidden_dim", 32)),
            "lif_beta": float(cfg.get("lif_beta", 0.9)),
            "lif_threshold": float(cfg.get("lif_threshold", 1.0)),
            "surrogate_alpha": float(cfg.get("surrogate_alpha", 10.0)),
            "ra_max_suppression": float(cfg.get("ra_max_suppression", 0.5)),
            "ra_initial_alpha": float(cfg.get("ra_initial_alpha", 0.99)),
            "detach_recurrent_state": bool(
                cfg.get("detach_recurrent_state", False)
            ),
        }
        self._adapter_only_train = bool(cfg.get("adapter_only_train", False))
        self._freeze_base = bool(
            cfg.get("freeze_base", self._adapter_only_train)
        )
        self._event_loss_lambda = float(cfg.get("event_loss_lambda", 0.0))
        if self._event_loss_lambda < 0.0:
            raise ValueError("event_loss_lambda must be >= 0")
        if self._adapter_only_train:
            self._cfg["lr"] = float(cfg.get("adapter_lr", cfg.get("lr", 1e-4)))
        if self._filter_obj is None or not hasattr(self._filter_obj, "kf_net"):
            raise RuntimeError("Split-KalmanNet setup did not produce filter_obj.kf_net")

        original_net = self._filter_obj.kf_net
        spike_ra_net = SpikeRAKNet(
            original_net,
            hidden_dim=self._spike_ra_cfg["snn_hidden_dim"],
            beta=self._spike_ra_cfg["lif_beta"],
            threshold=self._spike_ra_cfg["lif_threshold"],
            surrogate_alpha=self._spike_ra_cfg["surrogate_alpha"],
            max_suppression=self._spike_ra_cfg["ra_max_suppression"],
            initial_alpha=self._spike_ra_cfg["ra_initial_alpha"],
            detach_recurrent_state=self._spike_ra_cfg[
                "detach_recurrent_state"
            ],
        )
        spike_ra_net.to(device=self.device, dtype=self.dtype)
        spike_ra_net.initialize_hidden()
        self._filter_obj.kf_net = spike_ra_net

        checkpoint_raw = cfg.get(
            "init_from_split_knet_checkpoint",
            cfg.get("split_knet_checkpoint"),
        )
        if checkpoint_raw:
            self._initialize_base_from_split_checkpoint(checkpoint_raw)
        if self._freeze_base:
            self._freeze_base_modules()
        self._trainable_parameter_names = [
            name
            for name, parameter in spike_ra_net.named_parameters()
            if parameter.requires_grad
        ]
        if self._adapter_only_train and not self._trainable_parameter_names:
            raise RuntimeError("adapter_only_train produced no trainable parameters")
        logger.info(
            "SpikeRA training mode=%s base_checkpoint=%s frozen_modules=%s "
            "trainable_parameters=%s",
            "adapter_only" if self._adapter_only_train else "joint",
            self._base_checkpoint_path,
            self._frozen_module_names,
            self._trainable_parameter_names,
        )

    @staticmethod
    def _resolve_checkpoint_path(path_value: Union[str, Path]) -> Path:
        path = Path(path_value).expanduser()
        if not path.is_absolute():
            path = Path(__file__).resolve().parents[2] / path
        return path.resolve()

    def _initialize_base_from_split_checkpoint(
        self,
        checkpoint_path: Union[str, Path],
    ) -> None:
        net = self._spike_ra_net()
        if net is None:
            raise RuntimeError("SpikeRA network is unavailable for base initialization")
        path = self._resolve_checkpoint_path(checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(f"split_knet checkpoint not found: {path}")

        checkpoint = torch.load(path, map_location=self.device)
        state = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        if not isinstance(state, Mapping):
            raise TypeError(f"invalid split_knet checkpoint state at {path}")

        base_prefixes = ("l1.", "GRU1.", "l2.", "l3.", "GRU2.", "l4.")
        base_state = {
            str(key): value
            for key, value in state.items()
            if str(key).startswith(base_prefixes)
        }
        if not base_state:
            raise RuntimeError(f"no Split-KalmanNet G1/G2 weights found in {path}")

        incompatible = net.load_state_dict(base_state, strict=False)
        unexpected = list(incompatible.unexpected_keys)
        missing = list(incompatible.missing_keys)
        expected_adapter_missing = [
            key for key in missing if key.startswith("reliability_adapter.")
        ]
        unexpected_missing = [
            key for key in missing if not key.startswith("reliability_adapter.")
        ]
        if unexpected or unexpected_missing:
            raise RuntimeError(
                "split_knet checkpoint is incompatible with SpikeRA base: "
                f"unexpected={unexpected} missing_base={unexpected_missing}"
            )

        loaded_groups = [
            group
            for group in ("l1", "GRU1", "l2", "l3", "GRU2", "l4")
            if any(key.startswith(f"{group}.") for key in base_state)
        ]
        if loaded_groups != ["l1", "GRU1", "l2", "l3", "GRU2", "l4"]:
            raise RuntimeError(
                f"split_knet checkpoint did not cover all base groups: {loaded_groups}"
            )

        self._base_checkpoint_path = path
        self._base_checkpoint_report = {
            "path": str(path),
            "loaded_groups": loaded_groups,
            "loaded_tensor_count": int(len(base_state)),
            "missing_adapter_keys": expected_adapter_missing,
            "unexpected_keys": unexpected,
        }
        logger.info(
            "Initialized SpikeRA base from Split-KalmanNet checkpoint=%s groups=%s "
            "tensor_count=%s missing_adapter_keys=%s",
            path,
            loaded_groups,
            len(base_state),
            len(expected_adapter_missing),
        )

    def _freeze_base_modules(self) -> None:
        net = self._spike_ra_net()
        if net is None:
            raise RuntimeError("SpikeRA network is unavailable for freezing")
        self._frozen_module_names = ["l1", "GRU1", "l2", "l3", "GRU2", "l4"]
        for module_name in self._frozen_module_names:
            getattr(net, module_name).requires_grad_(False)
        net.reliability_adapter.requires_grad_(True)

    def state_estimation_loss(
        self,
        *,
        pred_btd: torch.Tensor,
        x_btd: torch.Tensor,
        batch: Optional[Dict[str, Any]],
        phase: str,
        loss_fn: torch.nn.Module,
    ) -> torch.Tensor:
        # Event flags weight an offline supervised training objective only.
        # They are never accepted by SpikeRAKNet.forward or predict().
        if self._event_loss_lambda <= 0.0:
            return super().state_estimation_loss(
                pred_btd=pred_btd,
                x_btd=x_btd,
                batch=batch,
                phase=phase,
                loss_fn=loss_fn,
            )
        if not isinstance(batch, Mapping) or "event_flag_seq" not in batch:
            raise ValueError(
                "event_loss_lambda requires event_flag_seq in train and validation batches"
            )

        event_flag = torch.as_tensor(
            batch["event_flag_seq"],
            device=pred_btd.device,
            dtype=pred_btd.dtype,
        )
        if event_flag.ndim == 3 and event_flag.shape[-1] == 1:
            event_flag = event_flag[..., 0]
        if event_flag.ndim != 2 or event_flag.shape != pred_btd.shape[:2]:
            raise ValueError(
                "event_flag_seq shape must match [B,T]; "
                f"got {tuple(event_flag.shape)} for pred={tuple(pred_btd.shape)}"
            )

        if x_btd.shape[1] > 1:
            pred_used = pred_btd[:, 1:, :]
            truth_used = x_btd[:, 1:, :]
            event_used = event_flag[:, 1:]
        else:
            pred_used = pred_btd
            truth_used = x_btd
            event_used = event_flag
        squared_error = torch.mean((pred_used - truth_used) ** 2, dim=-1)
        weights = 1.0 + self._event_loss_lambda * event_used
        return torch.sum(weights * squared_error) / torch.clamp(
            torch.sum(weights),
            min=1.0,
        )

    def train(
        self,
        train_dl: Any,
        val_dl: Any,
        budget: Optional[Dict[str, Any]] = None,
        ckpt_dir: Optional[Union[str, Path]] = None,
    ) -> Dict[str, Any]:
        net = self._spike_ra_net()
        if net is None:
            raise RuntimeError("SpikeRA network is unavailable for training")

        base_named_parameters = [
            (name, parameter)
            for name, parameter in net.named_parameters()
            if not name.startswith("reliability_adapter.")
        ]
        base_before = {
            name: parameter.detach().cpu().clone()
            for name, parameter in base_named_parameters
        }
        result = super().train(
            train_dl=train_dl,
            val_dl=val_dl,
            budget=budget,
            ckpt_dir=ckpt_dir,
        )

        base_grad_present = [
            name
            for name, parameter in base_named_parameters
            if parameter.grad is not None
        ]
        base_changed = [
            name
            for name, parameter in base_named_parameters
            if not torch.equal(base_before[name], parameter.detach().cpu())
        ]
        adapter_grad_nonzero_final = any(
            parameter.grad is not None
            and bool(torch.isfinite(parameter.grad).all())
            and bool(torch.count_nonzero(parameter.grad.detach()).item())
            for parameter in net.reliability_adapter.parameters()
        )

        grad_norms = [
            float(row["grad_norm_total"])
            for row in self._train_diag_history
            if row.get("phase") == "train"
            and row.get("grad_norm_total") is not None
        ]
        finite_grad_norms = [value for value in grad_norms if math.isfinite(value)]
        adapter_grad_nonzero = any(
            float(row.get("max_abs_grad") or 0.0) > 0.0
            for row in self._train_diag_history
            if row.get("phase") == "train"
        )
        self._stage2_train_summary = {
            "training_mode": (
                "adapter_only_frozen_base"
                if self._adapter_only_train
                else "joint"
            ),
            "base_checkpoint_path": (
                str(self._base_checkpoint_path)
                if self._base_checkpoint_path is not None
                else None
            ),
            "frozen_modules": list(self._frozen_module_names),
            "trainable_parameter_names": list(self._trainable_parameter_names),
            "event_loss_lambda": float(self._event_loss_lambda),
            "event_sample_total_weight": float(1.0 + self._event_loss_lambda),
            "adapter_lr": float(self._cfg.get("lr", 0.0)),
            "base_grad_present": base_grad_present,
            "base_parameters_changed": base_changed,
            "adapter_grad_nonzero": bool(adapter_grad_nonzero),
            "adapter_grad_nonzero_final": bool(adapter_grad_nonzero_final),
            "clipped_updates": int(self._clip_applied_count),
            "grad_norm_first": grad_norms[0] if grad_norms else None,
            "grad_norm_final": grad_norms[-1] if grad_norms else None,
            "grad_norm_max": max(grad_norms) if grad_norms else None,
            "grad_norm_median": (
                float(np.median(finite_grad_norms))
                if finite_grad_norms
                else None
            ),
            "gradient_observation_count": int(len(grad_norms)),
            "base_checkpoint_report": dict(self._base_checkpoint_report),
        }
        if self._adapter_only_train:
            if base_grad_present:
                raise RuntimeError(
                    f"frozen SpikeRA base parameters received gradients: {base_grad_present}"
                )
            if base_changed:
                raise RuntimeError(
                    f"frozen SpikeRA base parameters changed: {base_changed}"
                )
            if not adapter_grad_nonzero:
                raise RuntimeError("SpikeRA adapter gradients were zero after training")

        train_state_path_raw = result.get("train_state_path")
        if train_state_path_raw:
            train_state_path = Path(str(train_state_path_raw)).expanduser().resolve()
            train_state = json.loads(train_state_path.read_text(encoding="utf-8"))
            train_state["spike_ra_stage2"] = dict(self._stage2_train_summary)
            train_state_path.write_text(
                json.dumps(train_state, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
        return result

    def reset_spike_ra_stats(self) -> None:
        net = getattr(self._filter_obj, "kf_net", None)
        if hasattr(net, "reset_spike_ra_stats"):
            net.reset_spike_ra_stats()

    def get_spike_ra_stats(self) -> Dict[str, Union[int, float]]:
        net = getattr(self._filter_obj, "kf_net", None)
        if hasattr(net, "get_spike_ra_stats"):
            return dict(net.get_spike_ra_stats())
        return {
            "spike_count": 0.0,
            "spike_numel": 0,
            "avg_spike_rate": 0.0,
            "active_ops_proxy": 0.0,
            "fanout": 0,
            "mean_alpha": float("nan"),
            "min_alpha": float("nan"),
            "max_alpha": float("nan"),
            "mean_suppression": float("nan"),
        }

    def _spike_ra_net(self) -> Optional[SpikeRAKNet]:
        net = getattr(self._filter_obj, "kf_net", None)
        return net if isinstance(net, SpikeRAKNet) else None

    def eval(
        self,
        test_dl: Any,
        ckpt_path: Optional[Union[str, Path]] = None,
        track_cfg: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        self.reset_spike_ra_stats()
        self._reliability_gate_metrics = {}
        self._gate_trace_path = None
        net = self._spike_ra_net()
        if net is not None:
            net.reset_spike_ra_trace()
            net.set_spike_ra_trace_enabled(True)
        try:
            return super().eval(test_dl, ckpt_path=ckpt_path, track_cfg=track_cfg)
        finally:
            if net is not None:
                net.set_spike_ra_trace_enabled(False)

    @staticmethod
    def _event_flag_nt(
        event_flag: np.ndarray,
        *,
        n_seq: int,
        n_step: int,
    ) -> np.ndarray:
        flag = np.asarray(event_flag)
        if flag.ndim == 3 and flag.shape == (n_seq, n_step, 1):
            flag = flag[..., 0]
        elif flag.ndim != 2 or flag.shape != (n_seq, n_step):
            raise ValueError(
                f"event_flag_seq must have shape [N,T,1] or [N,T], got {flag.shape}"
            )
        return np.asarray(flag > 0.5, dtype=bool)

    @staticmethod
    def _masked_mean(values: np.ndarray, mask: np.ndarray) -> Optional[float]:
        selected = np.asarray(values, dtype=np.float64)[mask]
        if selected.size == 0:
            return None
        return float(np.mean(selected))

    def finalize_evaluation_diagnostics(
        self,
        *,
        run_dir: Union[str, Path],
        split_extras: Optional[Mapping[str, np.ndarray]],
        x_true: np.ndarray,
        x_pred: np.ndarray,
    ) -> Dict[str, Any]:
        net = self._spike_ra_net()
        trace = net.get_spike_ra_trace() if net is not None else []
        truth = np.asarray(x_true)
        pred = np.asarray(x_pred)
        n_seq, n_step = int(truth.shape[0]), int(truth.shape[1])
        expected_trace_steps = max(0, n_step - 1)
        if len(trace) != n_seq or any(
            len(sequence) != expected_trace_steps for sequence in trace
        ):
            raise ValueError(
                "SpikeRA trace shape mismatch: "
                f"sequences={len(trace)} expected={n_seq}, "
                f"steps={[len(sequence) for sequence in trace[:3]]} "
                f"expected_steps={expected_trace_steps}"
            )

        alpha = np.full((n_seq, n_step), np.nan, dtype=np.float64)
        spike_rate = np.full((n_seq, n_step), np.nan, dtype=np.float64)
        innovation_norm = np.full((n_seq, n_step), np.nan, dtype=np.float64)
        for sequence_index, sequence in enumerate(trace):
            for t, row in enumerate(sequence, start=1):
                alpha[sequence_index, t] = float(row["alpha"])
                spike_rate[sequence_index, t] = float(row["spike_rate"])
                innovation_norm[sequence_index, t] = float(row["innovation_norm"])

        extras = split_extras if isinstance(split_extras, Mapping) else {}
        event_flag_raw = extras.get("event_flag_seq")
        event_flag = (
            self._event_flag_nt(event_flag_raw, n_seq=n_seq, n_step=n_step)
            if event_flag_raw is not None
            else np.zeros((n_seq, n_step), dtype=bool)
        )
        attitude_error = attitude_error_deg(truth, pred)
        valid = np.isfinite(alpha)

        trace_path = (
            Path(run_dir).expanduser().resolve()
            / "diagnostics"
            / "spike_ra_gate_trace.csv"
        )
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        with trace_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "trajectory_index",
                    "t",
                    "alpha",
                    "suppression",
                    "spike_rate",
                    "innovation_norm",
                    "event_flag",
                    "attitude_error_deg",
                ]
            )
            for sequence_index in range(n_seq):
                for t in range(1, n_step):
                    writer.writerow(
                        [
                            sequence_index,
                            t,
                            float(alpha[sequence_index, t]),
                            float(1.0 - alpha[sequence_index, t]),
                            float(spike_rate[sequence_index, t]),
                            float(innovation_norm[sequence_index, t]),
                            int(event_flag[sequence_index, t]),
                            float(attitude_error[sequence_index, t]),
                        ]
                    )
        self._gate_trace_path = trace_path

        if event_flag_raw is None:
            self._reliability_gate_metrics = {
                "event_flag_available": False,
                "event_mean_alpha": None,
                "non_event_mean_alpha": None,
                "event_mean_suppression": None,
                "non_event_mean_suppression": None,
                "event_avg_spike_rate": None,
                "non_event_avg_spike_rate": None,
                "trace_path": str(trace_path),
                "note": "event_flag_seq unavailable; event/non-event metrics omitted",
            }
        else:
            event_mask = valid & event_flag
            non_event_mask = valid & ~event_flag
            event_alpha = self._masked_mean(alpha, event_mask)
            non_event_alpha = self._masked_mean(alpha, non_event_mask)
            self._reliability_gate_metrics = {
                "event_flag_available": True,
                "event_mean_alpha": event_alpha,
                "non_event_mean_alpha": non_event_alpha,
                "event_mean_suppression": (
                    None if event_alpha is None else float(1.0 - event_alpha)
                ),
                "non_event_mean_suppression": (
                    None if non_event_alpha is None else float(1.0 - non_event_alpha)
                ),
                "event_avg_spike_rate": self._masked_mean(spike_rate, event_mask),
                "non_event_avg_spike_rate": self._masked_mean(
                    spike_rate,
                    non_event_mask,
                ),
                "trace_path": str(trace_path),
                "event_flag_source": "test.npz:event_flag_seq",
            }
        return {
            "spike_ra_gate_trace": str(trace_path),
            "reliability_gate": dict(self._reliability_gate_metrics),
        }

    def get_extra_metrics(self) -> Dict[str, Any]:
        stats = self.get_spike_ra_stats()
        metrics = {
            "spike_ra": {
                "avg_spike_rate": float(stats["avg_spike_rate"]),
                "spike_count": int(round(float(stats["spike_count"]))),
                "spike_numel": int(stats["spike_numel"]),
                "active_ops_proxy": float(stats["active_ops_proxy"]),
                "fanout": int(stats["fanout"]),
                "mean_alpha": float(stats["mean_alpha"]),
                "min_alpha": float(stats["min_alpha"]),
                "max_alpha": float(stats["max_alpha"]),
                "mean_suppression": float(stats["mean_suppression"]),
                "collection_scope": "evaluation_only",
                "proxy_note": (
                    "active_ops_proxy is a neural activity proxy, not hardware energy"
                ),
            }
        }
        if self._reliability_gate_metrics:
            metrics["reliability_gate"] = dict(self._reliability_gate_metrics)
        if self._stage2_train_summary:
            metrics["spike_ra_training"] = dict(self._stage2_train_summary)
        return metrics

    def get_runtime_diagnostics(self) -> Dict[str, Any]:
        diagnostics = super().get_runtime_diagnostics()
        diagnostics["spike_ra"] = self.get_spike_ra_stats()
        if self._gate_trace_path is not None:
            diagnostics["spike_ra_gate_trace"] = str(self._gate_trace_path)
        return diagnostics

    def get_adapter_meta(self) -> Dict[str, Any]:
        meta = super().get_adapter_meta()
        meta.update(
            {
                "adapter_id": "spike_ra_knet",
                "adapter_version": "spike_ra_knet_reliability_lif_v1_stage2",
                "method_name": "SpikeRA-KalmanNet",
                "implementation_stage": "p0_reliability_adapter",
                "base_adapter": "split_knet",
                "g1_implementation": "original_split_knet_l1_gru1_l2",
                "g2_implementation": "original_split_knet_l3_gru2_l4",
                "reliability_adapter": "bench_recurrent_lif_scalar_sk_gate",
                "spike_ra_config": dict(self._spike_ra_cfg),
                "training_mode": (
                    "adapter_only_frozen_base"
                    if self._adapter_only_train
                    else "joint"
                ),
                "freeze_base": bool(self._freeze_base),
                "base_checkpoint_path": (
                    str(self._base_checkpoint_path)
                    if self._base_checkpoint_path is not None
                    else None
                ),
                "base_checkpoint_report": dict(self._base_checkpoint_report),
                "frozen_modules": list(self._frozen_module_names),
                "trainable_parameter_names": list(self._trainable_parameter_names),
                "event_loss_lambda": float(self._event_loss_lambda),
                "active_ops_proxy_is_hardware_energy": False,
            }
        )
        return meta
