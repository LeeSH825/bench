"""Legacy Euclidean G1-SNN ablation for the 6D-observation benchmark.

It does not implement the current right-local gyro-process-input Phase 2
interface and must not be cited as evidence for that architecture.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
from torch import nn

from .spike_split_knet import (
    G1SNNBranch,
    SpikeSplitKNetAdapter,
    resolve_snn_config,
)
from .split_knet import SplitKNetAdapter


class G1SNNSplitKNet(nn.Module):
    """Split-KalmanNet network replacing G1 while preserving original G2."""

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
            "input_dim_1",
            "l3",
            "GRU2",
            "l4",
            "hn2_init",
        )
        missing = [name for name in required if not hasattr(original_net, name)]
        if missing:
            raise TypeError(f"Split-KalmanNet network is missing G2 contract fields: {missing}")

        self.x_dim = int(original_net.x_dim)
        self.y_dim = int(original_net.y_dim)
        self.input_dim_1 = int(original_net.input_dim_1)
        self.gru_input_dim = int(original_net.gru_input_dim)
        self.gru_hidden_dim = int(original_net.gru_hidden_dim)
        self.gru_n_layer = int(original_net.gru_n_layer)
        self.batch_size = int(original_net.batch_size)
        self.seq_len_input = int(original_net.seq_len_input)

        self.g1_snn = G1SNNBranch(
            self.input_dim_1,
            self.x_dim,
            hidden_dim=hidden_dim,
            beta=beta,
            threshold=threshold,
            surrogate_alpha=surrogate_alpha,
            output_eps=output_eps,
            initial_diag=initial_diag,
        )

        # These are the original G2 module objects, not copies.
        self.l3 = original_net.l3
        self.GRU2 = original_net.GRU2
        self.l4 = original_net.l4
        self.register_buffer(
            "hn2_init",
            original_net.hn2_init.detach().clone(),
            persistent=False,
        )
        self.hn2 = self.hn2_init.detach().clone()

    def initialize_hidden(self) -> None:
        parameter = next(self.parameters())
        self.g1_snn.reset_state(
            self.batch_size,
            device=parameter.device,
            dtype=parameter.dtype,
        )
        self.hn2 = self.hn2_init.detach().clone().to(
            device=parameter.device,
            dtype=parameter.dtype,
        )

    def reset_spike_stats(self) -> None:
        self.g1_snn.reset_spike_stats()

    def get_spike_stats(self) -> Dict[str, Any]:
        return self.g1_snn.get_spike_stats()

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

        pk = self.g1_snn(input1)
        l3_out = self.l3(input2)
        gru_input = l3_out.reshape(self.seq_len_input, self.batch_size, self.gru_input_dim)
        gru_out, self.hn2 = self.GRU2(gru_input, self.hn2)
        sk = self.l4(gru_out).reshape(self.y_dim, self.y_dim)
        return pk, sk


class G1SNNSplitKNetAdapter(SpikeSplitKNetAdapter):
    """Ablation adapter using G1-SNN with the original Split-KalmanNet G2."""

    def __init__(self) -> None:
        super().__init__()
        self._snn_target = "g1"

    def setup(
        self,
        cfg: Dict[str, Any],
        system_info: Optional[Dict[str, Any]] = None,
        run_ctx: Optional[Dict[str, Any]] = None,
    ) -> None:
        SplitKNetAdapter.setup(self, cfg, system_info=system_info, run_ctx=run_ctx)
        self._snn_enabled = bool(cfg.get("snn_enabled", True))
        self._snn_target = str(cfg.get("snn_target", "g1")).lower()
        self._snn_cfg = resolve_snn_config(cfg)
        if not self._snn_enabled:
            return
        if self._snn_target != "g1":
            raise ValueError(
                f"G1-SNN ablation supports only snn_target='g1', got {self._snn_target!r}"
            )
        if self._filter_obj is None or not hasattr(self._filter_obj, "kf_net"):
            raise RuntimeError("Split-KalmanNet setup did not produce filter_obj.kf_net")

        original_net = self._filter_obj.kf_net
        g1_snn_net = G1SNNSplitKNet(
            original_net,
            hidden_dim=self._snn_cfg["snn_hidden_dim"],
            beta=self._snn_cfg["lif_beta"],
            threshold=self._snn_cfg["lif_threshold"],
            surrogate_alpha=self._snn_cfg["surrogate_alpha"],
            output_eps=self._snn_cfg["snn_output_eps"],
            initial_diag=self._snn_cfg["snn_initial_diag"],
        )
        g1_snn_net.to(device=self.device, dtype=self.dtype)
        g1_snn_net.initialize_hidden()
        self._filter_obj.kf_net = g1_snn_net

    def get_adapter_meta(self) -> Dict[str, Any]:
        meta = super().get_adapter_meta()
        meta.update(
            {
                "adapter_id": "g1_snn_split_knet",
                "adapter_version": (
                    "g1_snn_split_knet_g1_snn_v0"
                    if self._snn_enabled
                    else "g1_snn_split_knet_passthrough_v0"
                ),
                "implementation_stage": (
                    "p1_g1_snn_ablation" if self._snn_enabled else "p1_passthrough"
                ),
                "snn_target": "g1",
                "g1_implementation": (
                    "bench_recurrent_lif_positive_diagonal"
                    if self._snn_enabled
                    else "original_split_knet_l1_gru1_l2"
                ),
                "g2_implementation": "original_split_knet_l3_gru2_l4",
            }
        )
        return meta
