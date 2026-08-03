from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch import nn

from bench.models.registry import get_adapter_class, list_model_ids
from bench.models.spike_ra_knet import (
    SNNReliabilityAdapter,
    SpikeRAKNet,
    SpikeRAKNetAdapter,
)
from bench.models.split_knet import SplitKNetAdapter


@dataclass
class SpikeRAResult:
    ok: bool
    skipped: bool
    note: str


class _TinyOriginalSplitNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.x_dim = 2
        self.y_dim = 3
        self.input_dim_1 = 2 * self.x_dim + self.y_dim + self.x_dim * self.y_dim
        self.input_dim_2 = 3 * self.y_dim + self.x_dim * self.y_dim
        self.gru_input_dim = 7
        self.gru_hidden_dim = 5
        self.gru_n_layer = 1
        self.batch_size = 1
        self.seq_len_input = 1
        self.l1 = nn.Sequential(nn.Linear(self.input_dim_1, 7), nn.ReLU())
        self.GRU1 = nn.GRU(7, 5, 1)
        self.l2 = nn.Linear(5, self.x_dim * self.x_dim)
        self.l3 = nn.Sequential(nn.Linear(self.input_dim_2, 7), nn.ReLU())
        self.GRU2 = nn.GRU(7, 5, 1)
        self.l4 = nn.Linear(5, self.y_dim * self.y_dim)
        self.hn1_init = torch.zeros(1, 1, 5)
        self.hn2_init = torch.zeros(1, 1, 5)


def _bench_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _run(cmd: List[str], *, cwd: Path, env: Dict[str, str]) -> Tuple[int, str]:
    completed = subprocess.run(cmd, cwd=str(cwd), env=env, capture_output=True, text=True)
    return completed.returncode, (completed.stdout or "") + (completed.stderr or "")


def _base_forward(
    net: _TinyOriginalSplitNet,
    inputs: Tuple[torch.Tensor, ...],
) -> Tuple[torch.Tensor, torch.Tensor]:
    state_inno, observation_inno, diff_state, diff_obs, linearization_error, jacobian = inputs
    input1 = torch.cat((state_inno, diff_state, linearization_error, jacobian), dim=0).reshape(-1)
    input2 = torch.cat((observation_inno, diff_obs, linearization_error, jacobian), dim=0).reshape(-1)
    gru1_input = net.l1(input1).reshape(1, 1, net.gru_input_dim)
    gru1_out, _ = net.GRU1(gru1_input, net.hn1_init.clone())
    pk = net.l2(gru1_out).reshape(net.x_dim, net.x_dim)
    gru2_input = net.l3(input2).reshape(1, 1, net.gru_input_dim)
    gru2_out, _ = net.GRU2(gru2_input, net.hn2_init.clone())
    sk = net.l4(gru2_out).reshape(net.y_dim, net.y_dim)
    return pk, sk


def _module_checks() -> None:
    torch.manual_seed(0)
    reliability = SNNReliabilityAdapter(input_dim=15, hidden_dim=8, initial_alpha=0.99)
    reliability.reset_stats()
    reliability.reset_trace()
    reliability.set_trace_enabled(True)
    reliability.reset_state()
    gate = reliability(
        torch.randn(15, requires_grad=True),
        innovation=torch.tensor([3.0, 4.0]),
    )
    reliability.set_trace_enabled(False)
    if gate.shape != (1,) or not bool(torch.isfinite(gate).all()):
        raise AssertionError(f"invalid reliability gate: shape={tuple(gate.shape)} gate={gate}")
    if not 0.98 <= float(gate.detach().item()) <= 1.0:
        raise AssertionError(f"initial gate is not close to identity: {gate.item()}")
    gate.sum().backward()
    for name, parameter in reliability.named_parameters():
        if parameter.grad is None or not bool(torch.isfinite(parameter.grad).all()):
            raise AssertionError(f"missing/non-finite reliability gradient: {name}")
    trace = reliability.get_trace()
    if len(trace) != 1 or len(trace[0]) != 1:
        raise AssertionError(f"invalid reliability trace shape: {trace}")
    if not np.isclose(float(trace[0][0]["innovation_norm"]), 5.0):
        raise AssertionError(f"innovation norm was not traced correctly: {trace}")

    original = _TinyOriginalSplitNet()
    baseline = copy.deepcopy(original)
    wrapper = SpikeRAKNet(original, hidden_dim=8, initial_alpha=0.99)
    for name in ("l1", "GRU1", "l2", "l3", "GRU2", "l4"):
        if getattr(wrapper, name) is not getattr(original, name):
            raise AssertionError(f"SpikeRA wrapper did not reuse original module: {name}")

    inputs = (
        torch.randn(2, 1),
        torch.randn(3, 1),
        torch.randn(2, 1),
        torch.randn(3, 1),
        torch.randn(3, 1),
        torch.randn(6, 1),
    )
    pk_base, sk_base = _base_forward(baseline, inputs)
    wrapper.initialize_hidden()
    wrapper.reset_spike_ra_stats()
    pk, sk = wrapper(*inputs)
    alpha = wrapper.reliability_adapter._last_alpha.reshape(())
    if pk.shape != pk_base.shape or sk.shape != sk_base.shape:
        raise AssertionError(
            f"SpikeRA shape mismatch: Pk={tuple(pk.shape)} Sk={tuple(sk.shape)}"
        )
    if not bool(torch.isfinite(pk).all() and torch.isfinite(sk).all()):
        raise AssertionError("SpikeRA forward produced non-finite values")
    if not bool(torch.allclose(pk, pk_base, atol=1e-7, rtol=1e-6)):
        raise AssertionError("SpikeRA changed the original G1 output")
    if not bool(torch.allclose(sk, alpha * sk_base, atol=1e-7, rtol=1e-6)):
        raise AssertionError("SpikeRA Sk does not equal alpha * Sk_base")
    if float(alpha.detach().item()) < 0.98:
        raise AssertionError(f"SpikeRA initial alpha is too far from identity: {alpha.item()}")
    relative_change = float(
        (
            torch.linalg.norm(sk - sk_base)
            / torch.clamp(torch.linalg.norm(sk_base), min=1e-8)
        )
        .detach()
        .item()
    )
    if relative_change > 0.03:
        raise AssertionError(f"SpikeRA initialization changed Sk too much: {relative_change}")

    with tempfile.TemporaryDirectory(prefix="spike_ra_stage2_") as tmp:
        checkpoint_path = Path(tmp) / "split_model.pt"
        source = _TinyOriginalSplitNet()
        with torch.no_grad():
            for parameter in source.parameters():
                parameter.add_(0.25)
        torch.save({"state_dict": source.state_dict()}, checkpoint_path)

        target_original = _TinyOriginalSplitNet()
        target = SpikeRAKNet(target_original, hidden_dim=8, initial_alpha=0.99)
        adapter = SpikeRAKNetAdapter()
        adapter.device = torch.device("cpu")
        adapter._filter_obj = SimpleNamespace(kf_net=target)
        adapter._initialize_base_from_split_checkpoint(checkpoint_path)
        adapter._freeze_base_modules()

        for module_name in ("l1", "GRU1", "l2", "l3", "GRU2", "l4"):
            source_state = getattr(source, module_name).state_dict()
            target_state = getattr(target, module_name).state_dict()
            for key in source_state:
                if not torch.equal(source_state[key], target_state[key]):
                    raise AssertionError(
                        f"split checkpoint did not initialize {module_name}.{key}"
                    )
            if any(parameter.requires_grad for parameter in getattr(target, module_name).parameters()):
                raise AssertionError(f"base module was not frozen: {module_name}")
        if not all(
            parameter.requires_grad
            for parameter in target.reliability_adapter.parameters()
        ):
            raise AssertionError("reliability adapter was unexpectedly frozen")

        target.initialize_hidden()
        _, sk_stage2 = target(*inputs)
        sk_stage2.square().mean().backward()
        if any(
            parameter.grad is not None
            for module_name in ("l1", "GRU1", "l2", "l3", "GRU2", "l4")
            for parameter in getattr(target, module_name).parameters()
        ):
            raise AssertionError("frozen base received gradients")
        if not any(
            parameter.grad is not None
            and bool(torch.count_nonzero(parameter.grad).item())
            for parameter in target.reliability_adapter.parameters()
        ):
            raise AssertionError("adapter gradients were zero")

        adapter._event_loss_lambda = 4.0
        pred = torch.tensor([[[0.0], [2.0], [3.0]]])
        truth = torch.zeros_like(pred)
        batch = {"event_flag_seq": torch.tensor([[[0.0], [0.0], [1.0]]])}
        weighted = adapter.state_estimation_loss(
            pred_btd=pred,
            x_btd=truth,
            batch=batch,
            phase="train",
            loss_fn=nn.MSELoss(),
        )
        expected = torch.tensor((4.0 + 5.0 * 9.0) / 6.0)
        if not torch.allclose(weighted, expected):
            raise AssertionError(
                f"event-weighted loss mismatch: got={weighted} expected={expected}"
            )


def run_spike_ra_knet_tests() -> SpikeRAResult:
    try:
        _module_checks()
    except Exception as exc:
        return SpikeRAResult(False, False, f"SpikeRA module checks failed: {type(exc).__name__}: {exc}")

    if "spike_ra_knet" not in list_model_ids():
        return SpikeRAResult(False, False, "spike_ra_knet is missing from the registry")
    if get_adapter_class("spike_ra_knet") is not SpikeRAKNetAdapter:
        return SpikeRAResult(False, False, "spike_ra_knet registry class mismatch")
    if not issubclass(SpikeRAKNetAdapter, SplitKNetAdapter):
        return SpikeRAResult(False, False, "SpikeRAKNetAdapter does not reuse SplitKNetAdapter")

    bench_root = _bench_root()
    suite_yaml = bench_root / "bench" / "configs" / "suite_basilisk_spike_ra_smoke.yaml"
    task_id = "Basilisk_IMU_ADCS_event_spike_ra_smoke_v0"
    env = os.environ.copy()
    env["BENCH_DATA_CACHE"] = str((bench_root / "bench_data_cache").resolve())

    data_cmd = [
        sys.executable,
        "-m",
        "bench.tasks.smoke_data",
        "--suite-yaml",
        str(suite_yaml),
        "--task",
        task_id,
        "--seed",
        "0",
    ]
    data_rc, data_output = _run(data_cmd, cwd=bench_root, env=env)
    if data_rc != 0:
        if "Basilisk generator unavailable" in data_output or "DatasetMissingError" in data_output:
            return SpikeRAResult(True, True, "Basilisk unavailable; SpikeRA smoke skipped")
        return SpikeRAResult(False, False, f"SpikeRA dataset generation failed:\n{data_output}")

    run_cmd = [
        sys.executable,
        "-m",
        "bench.runners.run_suite",
        "--suite-yaml",
        str(suite_yaml),
        "--device",
        "cpu",
        "--init-id",
        "trained",
        "--track",
        "frozen",
    ]
    run_rc, run_output = _run(run_cmd, cwd=bench_root, env=env)
    if run_rc != 0:
        return SpikeRAResult(False, False, f"SpikeRA smoke run failed:\n{run_output}")

    run_root = (
        bench_root
        / "runs"
        / "basilisk_spike_ra_smoke"
        / task_id
    )
    metrics: Dict[str, Dict[str, object]] = {}
    for model_id in ("split_knet", "spike_ra_knet"):
        candidates = sorted((run_root / model_id / "frozen" / "seed_0").glob("scenario_*"))
        if not candidates:
            return SpikeRAResult(False, False, f"missing smoke run directory for {model_id}")
        run_dir = candidates[0]
        for rel in ("checkpoints/model.pt", "metrics.json", "timing.csv"):
            if not (run_dir / rel).exists():
                return SpikeRAResult(False, False, f"{model_id} missing {rel}")
        metrics[model_id] = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))

    if "spike_ra" in metrics["split_knet"]:
        return SpikeRAResult(False, False, "split_knet unexpectedly contains SpikeRA metrics")
    spike_ra = metrics["spike_ra_knet"].get("spike_ra")
    if not isinstance(spike_ra, dict):
        return SpikeRAResult(False, False, f"SpikeRA metrics missing: {spike_ra}")
    required = (
        "avg_spike_rate",
        "spike_count",
        "spike_numel",
        "active_ops_proxy",
        "mean_alpha",
        "min_alpha",
        "max_alpha",
        "mean_suppression",
    )
    for key in required:
        if key not in spike_ra or not np.isfinite(float(spike_ra[key])):
            return SpikeRAResult(False, False, f"invalid SpikeRA metric {key}: {spike_ra}")
    if not 0.0 < float(spike_ra["min_alpha"]) <= float(spike_ra["max_alpha"]) <= 1.0:
        return SpikeRAResult(False, False, f"invalid alpha range: {spike_ra}")
    if str(spike_ra.get("collection_scope")) != "evaluation_only":
        return SpikeRAResult(False, False, f"invalid collection scope: {spike_ra}")
    if "not hardware energy" not in str(spike_ra.get("proxy_note", "")).lower():
        return SpikeRAResult(False, False, f"missing proxy warning: {spike_ra}")

    reliability_gate = metrics["spike_ra_knet"].get("reliability_gate")
    if not isinstance(reliability_gate, dict):
        return SpikeRAResult(False, False, "SpikeRA reliability_gate metrics missing")
    for key in (
        "event_mean_alpha",
        "non_event_mean_alpha",
        "event_mean_suppression",
        "non_event_mean_suppression",
        "event_avg_spike_rate",
        "non_event_avg_spike_rate",
    ):
        if key not in reliability_gate or not np.isfinite(float(reliability_gate[key])):
            return SpikeRAResult(
                False,
                False,
                f"invalid reliability gate metric {key}: {reliability_gate}",
            )
    trace_path = Path(str(reliability_gate.get("trace_path", "")))
    if not trace_path.exists():
        return SpikeRAResult(False, False, f"SpikeRA gate trace missing: {trace_path}")
    header = trace_path.read_text(encoding="utf-8").splitlines()[0].split(",")
    for column in ("t", "alpha", "suppression", "spike_rate", "event_flag"):
        if column not in header:
            return SpikeRAResult(False, False, f"gate trace missing column {column}: {header}")

    return SpikeRAResult(
        True,
        False,
        "SpikeRA module, trace, event segmentation, smoke, and metrics checks passed",
    )


if __name__ == "__main__":
    result = run_spike_ra_knet_tests()
    print(result.note)
    raise SystemExit(0 if result.ok else 1)
