"""Legacy Euclidean 6D-observation Spike-Split benchmark adapter.

The third-party Split-KalmanNet filter and its G1 branch are reused unchanged.
Only the innovation-covariance branch is replaced by bench-owned code.
This module is structurally portable but is not current right-local Phase 2
evidence. Its gyro channels are observations, and its deployable forward path
accepts innovation-derived tensors rather than truth or event labels.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import torch
from torch import nn
from torch.nn import functional as F

from .split_knet import SplitKNetAdapter


class SurrogateSpikeFn(torch.autograd.Function):
    """Binary threshold in forward, fast-sigmoid surrogate derivative backward."""

    @staticmethod
    def forward(ctx: Any, membrane_delta: torch.Tensor, alpha: float) -> torch.Tensor:
        ctx.save_for_backward(membrane_delta)
        ctx.alpha = float(alpha)
        return (membrane_delta >= 0.0).to(dtype=membrane_delta.dtype)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        (membrane_delta,) = ctx.saved_tensors
        alpha = float(ctx.alpha)
        scale = alpha / torch.square(1.0 + alpha * torch.abs(membrane_delta))
        return grad_output * scale, None


def surrogate_spike(membrane_delta: torch.Tensor, alpha: float = 10.0) -> torch.Tensor:
    return SurrogateSpikeFn.apply(membrane_delta, float(alpha))


class RecurrentLIFCell(nn.Module):
    """Minimal recurrent leaky-integrate-and-fire cell with hard reset."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        *,
        beta: float = 0.9,
        threshold: float = 1.0,
        surrogate_alpha: float = 10.0,
    ) -> None:
        super().__init__()
        if input_dim <= 0 or hidden_dim <= 0:
            raise ValueError("input_dim and hidden_dim must be positive")
        if not 0.0 <= float(beta) < 1.0:
            raise ValueError("lif beta must be in [0, 1)")
        if float(threshold) <= 0.0:
            raise ValueError("lif threshold must be positive")
        if float(surrogate_alpha) <= 0.0:
            raise ValueError("surrogate_alpha must be positive")

        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.beta = float(beta)
        self.threshold = float(threshold)
        self.surrogate_alpha = float(surrogate_alpha)
        self.input_linear = nn.Linear(self.input_dim, self.hidden_dim)
        self.recurrent_linear = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.register_buffer("_membrane", torch.empty(0), persistent=False)
        self.register_buffer("_spike", torch.empty(0), persistent=False)

    def reset_state(
        self,
        batch_size: int = 1,
        *,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        parameter = next(self.parameters())
        state_device = parameter.device if device is None else torch.device(device)
        state_dtype = parameter.dtype if dtype is None else dtype
        shape = (int(batch_size), self.hidden_dim)
        self._membrane = torch.zeros(shape, device=state_device, dtype=state_dtype)
        self._spike = torch.zeros(shape, device=state_device, dtype=state_dtype)

    def detach_state(self) -> None:
        """Truncate recurrent autograd history without changing LIF state values."""
        self._membrane = self._membrane.detach()
        self._spike = self._spike.detach()

    def forward(self, input_t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        squeeze_batch = input_t.ndim == 1
        if squeeze_batch:
            input_t = input_t.unsqueeze(0)
        if input_t.ndim != 2 or input_t.shape[-1] != self.input_dim:
            raise ValueError(
                f"expected LIF input [B,{self.input_dim}] or [{self.input_dim}], "
                f"got {tuple(input_t.shape)}"
            )

        batch_size = int(input_t.shape[0])
        if (
            self._membrane.shape != (batch_size, self.hidden_dim)
            or self._membrane.device != input_t.device
            or self._membrane.dtype != input_t.dtype
        ):
            self.reset_state(batch_size, device=input_t.device, dtype=input_t.dtype)

        current = self.input_linear(input_t) + self.recurrent_linear(self._spike)
        membrane = self.beta * self._membrane + current
        spike = surrogate_spike(membrane - self.threshold, self.surrogate_alpha)
        membrane = membrane - spike * self.threshold
        self._membrane = membrane
        self._spike = spike

        if squeeze_batch:
            return membrane.squeeze(0), spike.squeeze(0)
        return membrane, spike


class G2SNNBranch(nn.Module):
    """Recurrent LIF G2 branch producing a positive diagonal Sk."""

    def __init__(
        self,
        input_dim: int,
        y_dim: int,
        *,
        hidden_dim: int = 32,
        beta: float = 0.9,
        threshold: float = 1.0,
        surrogate_alpha: float = 10.0,
        output_eps: float = 1.0e-4,
        initial_diag: float = 0.1,
    ) -> None:
        super().__init__()
        if y_dim <= 0:
            raise ValueError("y_dim must be positive")
        if output_eps <= 0.0:
            raise ValueError("output_eps must be positive")
        if initial_diag <= output_eps:
            raise ValueError("initial_diag must exceed output_eps")

        self.input_dim = int(input_dim)
        self.y_dim = int(y_dim)
        self.hidden_dim = int(hidden_dim)
        self.output_eps = float(output_eps)
        self.cell = RecurrentLIFCell(
            self.input_dim,
            self.hidden_dim,
            beta=beta,
            threshold=threshold,
            surrogate_alpha=surrogate_alpha,
        )
        self.decoder = nn.Linear(2 * self.hidden_dim, self.y_dim)
        self._fanout = int(self.y_dim)
        self._spike_count = 0.0
        self._spike_numel = 0
        self._initialize_decoder(initial_diag=float(initial_diag))

    def _initialize_decoder(self, *, initial_diag: float) -> None:
        nn.init.normal_(self.decoder.weight, mean=0.0, std=0.01)
        target = max(float(initial_diag) - self.output_eps, 1.0e-8)
        inverse_softplus = math.log(math.expm1(target))
        nn.init.constant_(self.decoder.bias, inverse_softplus)

    def reset_state(
        self,
        batch_size: int = 1,
        *,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        self.cell.reset_state(batch_size, device=device, dtype=dtype)

    def reset_spike_stats(self) -> None:
        self._spike_count = 0.0
        self._spike_numel = 0

    def get_spike_stats(self) -> Dict[str, Union[int, float]]:
        spike_rate = (
            float(self._spike_count) / float(self._spike_numel)
            if self._spike_numel > 0
            else 0.0
        )
        return {
            "spike_count": float(self._spike_count),
            "spike_numel": int(self._spike_numel),
            "avg_spike_rate": float(spike_rate),
            "active_ops_proxy": float(self._spike_count * self._fanout),
            "fanout": int(self._fanout),
        }

    def forward(self, g2_input: torch.Tensor) -> torch.Tensor:
        membrane, spike = self.cell(g2_input)
        self._spike_count += float(spike.detach().sum().cpu().item())
        self._spike_numel += int(spike.numel())
        decoder_input = torch.cat((membrane, spike), dim=-1)
        sk_diag = F.softplus(self.decoder(decoder_input)) + self.output_eps
        return torch.diag_embed(sk_diag)


class G1SNNBranch(G2SNNBranch):
    """Recurrent LIF G1 branch producing a positive diagonal Pk."""

    def __init__(
        self,
        input_dim: int,
        x_dim: int,
        *,
        hidden_dim: int = 32,
        beta: float = 0.9,
        threshold: float = 1.0,
        surrogate_alpha: float = 10.0,
        output_eps: float = 1.0e-4,
        initial_diag: float = 0.1,
    ) -> None:
        super().__init__(
            input_dim,
            x_dim,
            hidden_dim=hidden_dim,
            beta=beta,
            threshold=threshold,
            surrogate_alpha=surrogate_alpha,
            output_eps=output_eps,
            initial_diag=initial_diag,
        )
        self.x_dim = int(x_dim)


def resolve_snn_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "snn_hidden_dim": int(cfg.get("snn_hidden_dim", 32)),
        "lif_beta": float(cfg.get("lif_beta", 0.9)),
        "lif_threshold": float(cfg.get("lif_threshold", 1.0)),
        "surrogate_alpha": float(cfg.get("surrogate_alpha", 10.0)),
        "snn_output_eps": float(cfg.get("snn_output_eps", 1.0e-4)),
        "snn_initial_diag": float(cfg.get("snn_initial_diag", 0.1)),
    }


class SpikeSplitKNetG2SNN(nn.Module):
    """Split-KalmanNet network preserving G1 and replacing only G2."""

    def __init__(
        self,
        original_net: nn.Module,
        *,
        hidden_dim: int = 32,
        beta: float = 0.9,
        threshold: float = 1.0,
        surrogate_alpha: float = 10.0,
        output_eps: float = 1.0e-4,
        initial_diag: float = 0.1,
    ) -> None:
        super().__init__()
        required = (
            "x_dim",
            "y_dim",
            "input_dim_2",
            "l1",
            "GRU1",
            "l2",
            "hn1_init",
        )
        missing = [name for name in required if not hasattr(original_net, name)]
        if missing:
            raise TypeError(f"Split-KalmanNet network is missing G1 contract fields: {missing}")

        self.x_dim = int(original_net.x_dim)
        self.y_dim = int(original_net.y_dim)
        self.input_dim_2 = int(original_net.input_dim_2)
        self.gru_input_dim = int(original_net.gru_input_dim)
        self.gru_hidden_dim = int(original_net.gru_hidden_dim)
        self.gru_n_layer = int(original_net.gru_n_layer)
        self.batch_size = int(original_net.batch_size)
        self.seq_len_input = int(original_net.seq_len_input)

        # These are the original G1 module objects, not copies.
        self.l1 = original_net.l1
        self.GRU1 = original_net.GRU1
        self.l2 = original_net.l2
        self.register_buffer(
            "hn1_init",
            original_net.hn1_init.detach().clone(),
            persistent=False,
        )
        self.hn1 = self.hn1_init.detach().clone()

        self.g2_snn = G2SNNBranch(
            self.input_dim_2,
            self.y_dim,
            hidden_dim=hidden_dim,
            beta=beta,
            threshold=threshold,
            surrogate_alpha=surrogate_alpha,
            output_eps=output_eps,
            initial_diag=initial_diag,
        )

    def initialize_hidden(self) -> None:
        parameter = next(self.parameters())
        self.hn1 = self.hn1_init.detach().clone().to(
            device=parameter.device,
            dtype=parameter.dtype,
        )
        self.g2_snn.reset_state(
            self.batch_size,
            device=parameter.device,
            dtype=parameter.dtype,
        )

    def reset_spike_stats(self) -> None:
        self.g2_snn.reset_spike_stats()

    def get_spike_stats(self) -> Dict[str, Union[int, float]]:
        return self.g2_snn.get_spike_stats()

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
        gru_input = l1_out.reshape(self.seq_len_input, self.batch_size, self.gru_input_dim)
        gru_out, self.hn1 = self.GRU1(gru_input, self.hn1)
        pk = self.l2(gru_out).reshape(self.x_dim, self.x_dim)
        sk = self.g2_snn(input2)
        return pk, sk


class SpikeSplitKNetAdapter(SplitKNetAdapter):
    """Split-KalmanNet adapter with an optional bench-owned G2-SNN branch."""

    def __init__(self) -> None:
        super().__init__()
        self._snn_enabled = True
        self._snn_target = "g2"
        self._snn_cfg: Dict[str, Any] = {}

    def setup(
        self,
        cfg: Dict[str, Any],
        system_info: Optional[Dict[str, Any]] = None,
        run_ctx: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().setup(cfg, system_info=system_info, run_ctx=run_ctx)
        self._snn_enabled = bool(cfg.get("snn_enabled", True))
        self._snn_target = str(cfg.get("snn_target", "g2")).lower()
        self._snn_cfg = resolve_snn_config(cfg)
        if not self._snn_enabled:
            return
        if self._snn_target != "g2":
            raise ValueError(
                f"Spike-Split P0 supports only snn_target='g2', got {self._snn_target!r}"
            )
        if self._filter_obj is None or not hasattr(self._filter_obj, "kf_net"):
            raise RuntimeError("Split-KalmanNet setup did not produce filter_obj.kf_net")

        original_net = self._filter_obj.kf_net
        spike_net = SpikeSplitKNetG2SNN(
            original_net,
            hidden_dim=self._snn_cfg["snn_hidden_dim"],
            beta=self._snn_cfg["lif_beta"],
            threshold=self._snn_cfg["lif_threshold"],
            surrogate_alpha=self._snn_cfg["surrogate_alpha"],
            output_eps=self._snn_cfg["snn_output_eps"],
            initial_diag=self._snn_cfg["snn_initial_diag"],
        )
        spike_net.to(device=self.device, dtype=self.dtype)
        spike_net.initialize_hidden()
        self._filter_obj.kf_net = spike_net

    def reset_spike_stats(self) -> None:
        net = getattr(self._filter_obj, "kf_net", None)
        if self._snn_enabled and hasattr(net, "reset_spike_stats"):
            net.reset_spike_stats()

    def get_spike_stats(self) -> Dict[str, Union[int, float]]:
        net = getattr(self._filter_obj, "kf_net", None)
        if self._snn_enabled and hasattr(net, "get_spike_stats"):
            return dict(net.get_spike_stats())
        return {
            "spike_count": 0.0,
            "spike_numel": 0,
            "avg_spike_rate": 0.0,
            "active_ops_proxy": 0.0,
            "fanout": 0,
        }

    def eval(
        self,
        test_dl: Any,
        ckpt_path: Optional[Union[str, Path]] = None,
        track_cfg: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        self.reset_spike_stats()
        return super().eval(test_dl, ckpt_path=ckpt_path, track_cfg=track_cfg)

    def get_runtime_diagnostics(self) -> Dict[str, Any]:
        diagnostics = super().get_runtime_diagnostics()
        diagnostics["spike_stats"] = self.get_spike_stats()
        return diagnostics

    def get_extra_metrics(self) -> Dict[str, Any]:
        if not self._snn_enabled:
            return {}
        stats = self.get_spike_stats()
        return {
            "spike_activity": {
                "avg_spike_rate": float(stats["avg_spike_rate"]),
                "spike_count": int(round(float(stats["spike_count"]))),
                "spike_numel": int(stats["spike_numel"]),
                "active_ops_proxy": float(stats["active_ops_proxy"]),
                "fanout": int(stats["fanout"]),
                "collection_scope": "evaluation_only",
                "proxy_note": (
                    "active_ops_proxy is a neural activity proxy, not hardware energy"
                ),
            }
        }

    def get_adapter_meta(self) -> Dict[str, Any]:
        meta = super().get_adapter_meta()
        meta.update(
            {
                "adapter_id": "spike_split_knet",
                "adapter_version": (
                    "spike_split_knet_g2_snn_v0"
                    if self._snn_enabled
                    else "spike_split_knet_passthrough_v0"
                ),
                "implementation_stage": (
                    "p0_g2_snn" if self._snn_enabled else "p0_passthrough"
                ),
                "base_adapter": "split_knet",
                "snn_enabled": bool(self._snn_enabled),
                "snn_target": self._snn_target,
                "g1_implementation": "original_split_knet_l1_gru1_l2",
                "g2_implementation": (
                    "bench_recurrent_lif_positive_diagonal"
                    if self._snn_enabled
                    else "original_split_knet_l3_gru2_l4"
                ),
                "snn_config": dict(self._snn_cfg),
                "spike_stats": self.get_spike_stats(),
                "active_ops_proxy_is_hardware_energy": False,
            }
        )
        return meta
