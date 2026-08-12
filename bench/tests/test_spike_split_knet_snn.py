from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch import nn

from bench.models.g1_snn_split_knet import (
    G1SNNSplitKNet,
    G1SNNSplitKNetAdapter,
)
from bench.models.registry import get_adapter_class, list_model_ids
from bench.models.spike_split_knet import (
    G1SNNBranch,
    G2SNNBranch,
    SpikeSplitKNetAdapter,
    SpikeSplitKNetG2SNN,
)
from bench.models.split_knet import SplitKNetAdapter
from bench.runners.run_suite import _sanitize_extra_metric_value
from bench.tasks.generator.datasets.common import DatasetMissingError


@dataclass
class SpikeSplitSNNResult:
    ok: bool
    skipped: bool
    note: str
    split_run_dir: Path
    spike_run_dir: Path
    g1_snn_run_dir: Path
    metric_differences: Dict[str, float]


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
        self.l1 = nn.Sequential(
            nn.Linear(self.input_dim_1, 7),
            nn.ReLU(),
        )
        self.GRU1 = nn.GRU(7, 5, 1)
        self.l2 = nn.Linear(5, self.x_dim * self.x_dim)
        self.hn1_init = torch.zeros(1, 1, 5)
        self.l3 = nn.Sequential(
            nn.Linear(self.input_dim_2, 7),
            nn.ReLU(),
        )
        self.GRU2 = nn.GRU(7, 5, 1)
        self.l4 = nn.Linear(5, self.y_dim * self.y_dim)
        self.hn2_init = torch.zeros(1, 1, 5)


def _bench_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _run(cmd: List[str], *, cwd: Path, env: Dict[str, str]) -> Tuple[int, str]:
    completed = subprocess.run(cmd, cwd=str(cwd), env=env, capture_output=True, text=True)
    return completed.returncode, (completed.stdout or "") + (completed.stderr or "")


def _find_run_dir(bench_root: Path, model_id: str) -> Path:
    root = (
        bench_root
        / "runs"
        / "basilisk_spike_split_smoke"
        / "Basilisk_IMU_ADCS_event_spike_split_smoke_v0"
        / model_id
        / "frozen"
        / "seed_0"
    )
    candidates = sorted(root.glob("scenario_*"))
    return candidates[0] if candidates else Path("")


def _read_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _failure(note: str) -> SpikeSplitSNNResult:
    return SpikeSplitSNNResult(
        ok=False,
        skipped=False,
        note=note,
        split_run_dir=Path(""),
        spike_run_dir=Path(""),
        g1_snn_run_dir=Path(""),
        metric_differences={},
    )


def _run_module_checks() -> None:
    sanitized = _sanitize_extra_metric_value(
        {
            "finite": np.float32(0.25),
            "nonfinite": float("nan"),
            "count": torch.tensor(3),
        }
    )
    if sanitized != {"finite": 0.25, "nonfinite": None, "count": 3}:
        raise AssertionError(f"extra metric JSON sanitization mismatch: {sanitized}")

    torch.manual_seed(0)
    branch = G2SNNBranch(
        input_dim=9,
        y_dim=3,
        hidden_dim=8,
        beta=0.9,
        threshold=1.0,
        surrogate_alpha=10.0,
    )
    with torch.no_grad():
        branch.cell.input_linear.weight.zero_()
        branch.cell.input_linear.bias.fill_(2.0)
        branch.cell.recurrent_linear.weight.zero_()
    branch.reset_state(batch_size=4)
    branch.reset_spike_stats()
    input_batch = torch.randn(4, 9, requires_grad=True)
    sk_first = branch(input_batch)
    sk_second = branch(input_batch)

    if sk_first.shape != (4, 3, 3):
        raise AssertionError(f"unexpected G2-SNN output shape: {tuple(sk_first.shape)}")
    if not bool(torch.isfinite(sk_first).all() and torch.isfinite(sk_second).all()):
        raise AssertionError("G2-SNN output contains non-finite values")
    diagonal = torch.diagonal(sk_second, dim1=-2, dim2=-1)
    off_diagonal = sk_second - torch.diag_embed(diagonal)
    if not bool((diagonal > 0.0).all()):
        raise AssertionError("G2-SNN diagonal is not strictly positive")
    if not bool(torch.equal(off_diagonal, torch.zeros_like(off_diagonal))):
        raise AssertionError("G2-SNN output contains nonzero off-diagonal entries")

    loss = sk_first.square().mean() + sk_second.square().mean()
    loss.backward()
    if input_batch.grad is None or not bool(torch.isfinite(input_batch.grad).all()):
        raise AssertionError("G2-SNN backward did not produce finite input gradients")
    for name, parameter in branch.named_parameters():
        if parameter.grad is None or not bool(torch.isfinite(parameter.grad).all()):
            raise AssertionError(f"G2-SNN parameter gradient missing/non-finite: {name}")

    stats = branch.get_spike_stats()
    expected_numel = 2 * 4 * 8
    if int(stats["spike_numel"]) != expected_numel:
        raise AssertionError(f"unexpected spike_numel: {stats}")
    if float(stats["spike_count"]) <= 0.0:
        raise AssertionError(f"forced-spike test recorded no spikes: {stats}")
    if not 0.0 <= float(stats["avg_spike_rate"]) <= 1.0:
        raise AssertionError(f"invalid average spike rate: {stats}")
    if not np.isclose(
        float(stats["active_ops_proxy"]),
        float(stats["spike_count"]) * 3.0,
    ):
        raise AssertionError(f"active_ops_proxy mismatch: {stats}")
    branch.reset_spike_stats()
    reset_stats = branch.get_spike_stats()
    if float(reset_stats["spike_count"]) != 0.0 or int(reset_stats["spike_numel"]) != 0:
        raise AssertionError(f"spike stats did not reset: {reset_stats}")

    original = _TinyOriginalSplitNet()
    wrapper = SpikeSplitKNetG2SNN(original, hidden_dim=8)
    if wrapper.l1 is not original.l1 or wrapper.GRU1 is not original.GRU1 or wrapper.l2 is not original.l2:
        raise AssertionError("Spike-Split wrapper copied or replaced original G1 modules")
    wrapper.initialize_hidden()
    pk, sk = wrapper(
        torch.randn(2, 1),
        torch.randn(3, 1),
        torch.randn(2, 1),
        torch.randn(3, 1),
        torch.randn(3, 1),
        torch.randn(6, 1),
    )
    if pk.shape != (2, 2) or sk.shape != (3, 3):
        raise AssertionError(f"wrapper contract mismatch: Pk={tuple(pk.shape)} Sk={tuple(sk.shape)}")
    if not bool(torch.isfinite(pk).all() and torch.isfinite(sk).all()):
        raise AssertionError("wrapper forward produced non-finite Pk or Sk")

    g1_branch = G1SNNBranch(input_dim=11, x_dim=2, hidden_dim=8)
    g1_input = torch.randn(4, 11, requires_grad=True)
    pk_batch = g1_branch(g1_input)
    pk_diag = torch.diagonal(pk_batch, dim1=-2, dim2=-1)
    pk_off_diag = pk_batch - torch.diag_embed(pk_diag)
    if pk_batch.shape != (4, 2, 2):
        raise AssertionError(f"unexpected G1-SNN output shape: {tuple(pk_batch.shape)}")
    if not bool(torch.isfinite(pk_batch).all() and (pk_diag > 0.0).all()):
        raise AssertionError("G1-SNN output is non-finite or not strictly positive diagonal")
    if not bool(torch.equal(pk_off_diag, torch.zeros_like(pk_off_diag))):
        raise AssertionError("G1-SNN output contains nonzero off-diagonal entries")
    pk_batch.square().mean().backward()
    if g1_input.grad is None or not bool(torch.isfinite(g1_input.grad).all()):
        raise AssertionError("G1-SNN backward did not produce finite input gradients")
    for name, parameter in g1_branch.named_parameters():
        if parameter.grad is None or not bool(torch.isfinite(parameter.grad).all()):
            raise AssertionError(f"G1-SNN parameter gradient missing/non-finite: {name}")

    g1_original = _TinyOriginalSplitNet()
    g1_wrapper = G1SNNSplitKNet(g1_original, hidden_dim=8)
    if (
        g1_wrapper.l3 is not g1_original.l3
        or g1_wrapper.GRU2 is not g1_original.GRU2
        or g1_wrapper.l4 is not g1_original.l4
    ):
        raise AssertionError("G1-SNN wrapper copied or replaced original G2 modules")
    if hasattr(g1_wrapper, "l1") or hasattr(g1_wrapper, "GRU1") or hasattr(g1_wrapper, "l2"):
        raise AssertionError("G1-SNN wrapper still exposes original G1 modules")
    g1_wrapper.initialize_hidden()
    state_inno = torch.randn(2, 1, requires_grad=True)
    observation_inno = torch.randn(3, 1, requires_grad=True)
    diff_state = torch.randn(2, 1, requires_grad=True)
    diff_obs = torch.randn(3, 1, requires_grad=True)
    linearization_error = torch.randn(3, 1, requires_grad=True)
    jacobian = torch.randn(6, 1, requires_grad=True)
    pk, sk = g1_wrapper(
        state_inno,
        observation_inno,
        diff_state,
        diff_obs,
        linearization_error,
        jacobian,
    )
    pk_diag = torch.diagonal(pk)
    if pk.shape != (2, 2) or sk.shape != (3, 3):
        raise AssertionError(f"G1-SNN wrapper contract mismatch: Pk={tuple(pk.shape)} Sk={tuple(sk.shape)}")
    if not bool(torch.isfinite(pk).all() and torch.isfinite(sk).all() and (pk_diag > 0.0).all()):
        raise AssertionError("G1-SNN wrapper produced invalid Pk or Sk")
    if not bool(torch.equal(pk - torch.diag_embed(pk_diag), torch.zeros_like(pk))):
        raise AssertionError("G1-SNN wrapper Pk contains nonzero off-diagonal entries")
    (pk.square().mean() + sk.square().mean()).backward()
    for name, parameter in g1_wrapper.named_parameters():
        if parameter.grad is None or not bool(torch.isfinite(parameter.grad).all()):
            raise AssertionError(f"G1-SNN wrapper parameter gradient missing/non-finite: {name}")


def run_spike_split_snn_ablation_tests() -> SpikeSplitSNNResult:
    try:
        _run_module_checks()
    except Exception as exc:
        return _failure(f"G2-SNN module checks failed: {type(exc).__name__}: {exc}")

    if "spike_split_knet" not in list_model_ids():
        return _failure("spike_split_knet is missing from the model registry")
    adapter_class = get_adapter_class("spike_split_knet")
    if adapter_class is not SpikeSplitKNetAdapter:
        return _failure(f"registry returned unexpected class: {adapter_class}")
    if not issubclass(SpikeSplitKNetAdapter, SplitKNetAdapter):
        return _failure("SpikeSplitKNetAdapter does not reuse SplitKNetAdapter")
    if get_adapter_class("split_knet") is not SplitKNetAdapter:
        return _failure("split_knet baseline registry entry changed")
    if "g1_snn_split_knet" not in list_model_ids():
        return _failure("g1_snn_split_knet is missing from the model registry")
    if get_adapter_class("g1_snn_split_knet") is not G1SNNSplitKNetAdapter:
        return _failure("g1_snn_split_knet registry entry returned an unexpected class")

    adapter_meta = SpikeSplitKNetAdapter().get_adapter_meta()
    expected_meta = {
        "adapter_id": "spike_split_knet",
        "implementation_stage": "p0_g2_snn",
        "base_adapter": "split_knet",
        "snn_enabled": True,
        "snn_target": "g2",
        "g1_implementation": "original_split_knet_l1_gru1_l2",
        "g2_implementation": "bench_recurrent_lif_positive_diagonal",
    }
    if any(adapter_meta.get(key) != value for key, value in expected_meta.items()):
        return _failure(f"G2-SNN adapter metadata mismatch: {adapter_meta}")
    disabled_adapter = SpikeSplitKNetAdapter()
    disabled_adapter._snn_enabled = False
    if disabled_adapter.get_extra_metrics() != {}:
        return _failure("pass-through mode unexpectedly exposes spike_activity metrics")
    g1_adapter_meta = G1SNNSplitKNetAdapter().get_adapter_meta()
    expected_g1_meta = {
        "adapter_id": "g1_snn_split_knet",
        "implementation_stage": "p1_g1_snn_ablation",
        "base_adapter": "split_knet",
        "snn_enabled": True,
        "snn_target": "g1",
        "g1_implementation": "bench_recurrent_lif_positive_diagonal",
        "g2_implementation": "original_split_knet_l3_gru2_l4",
    }
    if any(g1_adapter_meta.get(key) != value for key, value in expected_g1_meta.items()):
        return _failure(f"G1-SNN adapter metadata mismatch: {g1_adapter_meta}")

    bench_root = _bench_root()
    suite_yaml = bench_root / "bench" / "configs" / "suite_basilisk_spike_split_smoke.yaml"
    task_id = "Basilisk_IMU_ADCS_event_spike_split_smoke_v0"
    shutil.rmtree(bench_root / "runs" / "basilisk_spike_split_smoke", ignore_errors=True)
    summary_csv = bench_root / "reports" / "summary_basilisk_spike_split_smoke.csv"
    if summary_csv.exists():
        summary_csv.unlink()

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
        if any(
            marker in data_output
            for marker in (
                "Basilisk generator unavailable",
                "DatasetMissingError",
                "AVS Basilisk is required for task_family=",
            )
        ):
            return SpikeSplitSNNResult(
                ok=True,
                skipped=True,
                note=f"Basilisk unavailable; SNN ablation smoke skipped: {data_output.strip()}",
                split_run_dir=Path(""),
                spike_run_dir=Path(""),
                g1_snn_run_dir=Path(""),
                metric_differences={},
            )
        return _failure(f"event dataset smoke generation failed:\n{data_output}")

    run_cmd = [
        sys.executable,
        "-m",
        "bench.runners.run_suite",
        "--suite-yaml",
        str(suite_yaml),
        "--tasks",
        task_id,
        "--models",
        "split_knet",
        "spike_split_knet",
        "g1_snn_split_knet",
        "--seeds",
        "0",
        "--track",
        "frozen",
        "--init-id",
        "trained",
        "--device",
        "cpu",
    ]
    run_rc, run_output = _run(run_cmd, cwd=bench_root, env=env)
    if run_rc != 0:
        return _failure(f"Spike-Split G2-SNN run failed:\n{run_output}")

    run_dirs = {
        model_id: _find_run_dir(bench_root, model_id)
        for model_id in ("split_knet", "spike_split_knet", "g1_snn_split_knet")
    }
    required_artifacts = (
        "run_plan.json",
        "budget_ledger.json",
        "checkpoints/model.pt",
        "checkpoints/train_state.json",
        "metrics.json",
        "metrics_step.csv",
        "timing.csv",
    )
    metrics: Dict[str, Dict[str, object]] = {}
    for model_id, run_dir in run_dirs.items():
        if not run_dir.exists():
            return _failure(f"run directory missing for {model_id}")
        missing = [name for name in required_artifacts if not (run_dir / name).exists()]
        if missing:
            return _failure(f"{model_id} missing required artifacts: {missing}")

        ledger = _read_json(run_dir / "budget_ledger.json")
        train_updates = int(ledger.get("train_updates_used", 0))
        if train_updates <= 0 or train_updates > int(ledger.get("train_max_updates", 0)):
            return _failure(f"{model_id} invalid train update ledger: {ledger}")

        model_metrics = _read_json(run_dir / "metrics.json")
        if model_metrics.get("status") != "ok":
            return _failure(f"{model_id} metrics status is not ok: {model_metrics}")
        event_metrics = model_metrics.get("adcs_event")
        if not isinstance(event_metrics, dict) or int(event_metrics.get("event_sample_count", 0)) <= 0:
            return _failure(f"{model_id} event metrics missing or empty: {event_metrics}")
        metrics[model_id] = model_metrics

    if "spike_activity" in metrics["split_knet"]:
        return _failure("split_knet baseline unexpectedly contains spike_activity metrics")
    required_spike_fields = (
        "avg_spike_rate",
        "spike_count",
        "spike_numel",
        "active_ops_proxy",
        "fanout",
    )
    for model_id in ("spike_split_knet", "g1_snn_split_knet"):
        spike_activity = metrics[model_id].get("spike_activity")
        if not isinstance(spike_activity, dict):
            return _failure(f"{model_id} metrics missing spike_activity: {spike_activity}")
        for key in required_spike_fields:
            if key not in spike_activity:
                return _failure(f"{model_id} spike_activity missing {key}: {spike_activity}")
            if not np.isfinite(float(spike_activity[key])):
                return _failure(
                    f"{model_id} spike_activity field is non-finite: {key}={spike_activity[key]}"
                )
        if not 0.0 <= float(spike_activity["avg_spike_rate"]) <= 1.0:
            return _failure(f"{model_id} spike rate is invalid: {spike_activity}")
        if int(spike_activity["spike_numel"]) <= 0:
            return _failure(f"{model_id} spike_numel is not positive: {spike_activity}")
        if str(spike_activity.get("collection_scope")) != "evaluation_only":
            return _failure(f"{model_id} collection scope is unclear: {spike_activity}")
        if "not hardware energy" not in str(spike_activity.get("proxy_note", "")).lower():
            return _failure(f"{model_id} proxy note is insufficient: {spike_activity}")

    split_meta = metrics["split_knet"].get("adapter_meta")
    spike_meta = metrics["spike_split_knet"].get("adapter_meta")
    if not isinstance(split_meta, dict) or split_meta.get("adapter_id") != "split_knet":
        return _failure(f"split_knet adapter identity changed: {split_meta}")
    if split_meta.get("snn_enabled") is not None:
        return _failure(f"split_knet baseline unexpectedly exposes SNN metadata: {split_meta}")
    if not isinstance(spike_meta, dict) or any(
        spike_meta.get(key) != value for key, value in expected_meta.items()
    ):
        return _failure(f"spike_split_knet adapter identity mismatch: {spike_meta}")
    g1_meta = metrics["g1_snn_split_knet"].get("adapter_meta")
    if not isinstance(g1_meta, dict) or any(
        g1_meta.get(key) != value for key, value in expected_g1_meta.items()
    ):
        return _failure(f"g1_snn_split_knet adapter identity mismatch: {g1_meta}")

    spike_stats = spike_meta.get("spike_stats")
    if not isinstance(spike_stats, dict) or int(spike_stats.get("spike_numel", 0)) <= 0:
        return _failure(f"spike statistics missing from metrics metadata: {spike_stats}")
    spike_rate = float(spike_stats.get("avg_spike_rate", float("nan")))
    if not np.isfinite(spike_rate) or not 0.0 <= spike_rate <= 1.0:
        return _failure(f"invalid spike rate in metrics metadata: {spike_stats}")

    split_checkpoint = torch.load(
        run_dirs["split_knet"] / "checkpoints" / "model.pt",
        map_location="cpu",
    )
    spike_checkpoint = torch.load(
        run_dirs["spike_split_knet"] / "checkpoints" / "model.pt",
        map_location="cpu",
    )
    g1_checkpoint = torch.load(
        run_dirs["g1_snn_split_knet"] / "checkpoints" / "model.pt",
        map_location="cpu",
    )
    split_keys = set(split_checkpoint["state_dict"])
    spike_keys = set(spike_checkpoint["state_dict"])
    g1_keys = set(g1_checkpoint["state_dict"])
    if not any(key.startswith("l3.") for key in split_keys) or not any(
        key.startswith("GRU2.") for key in split_keys
    ):
        return _failure("split_knet baseline checkpoint no longer contains original G2")
    if not any(key.startswith("g2_snn.") for key in spike_keys):
        return _failure("Spike-Split checkpoint does not contain G2-SNN parameters")
    if any(
        key.startswith(prefix)
        for key in spike_keys
        for prefix in ("l3.", "GRU2.", "l4.")
    ):
        return _failure("Spike-Split checkpoint still contains original G2 parameters")
    if not any(key.startswith("g1_snn.") for key in g1_keys):
        return _failure("G1-SNN checkpoint does not contain G1-SNN parameters")
    if not all(any(key.startswith(prefix) for key in g1_keys) for prefix in ("l3.", "GRU2.", "l4.")):
        return _failure("G1-SNN checkpoint does not preserve all original G2 modules")
    if any(
        key.startswith(prefix)
        for key in g1_keys
        for prefix in ("l1.", "GRU1.", "l2.")
    ):
        return _failure("G1-SNN checkpoint still contains original G1 parameters")

    split_accuracy = metrics["split_knet"].get("accuracy")
    spike_accuracy = metrics["spike_split_knet"].get("accuracy")
    if not isinstance(split_accuracy, dict) or not isinstance(spike_accuracy, dict):
        return _failure("accuracy metrics missing from one or both runs")
    metric_differences = {
        key: abs(float(split_accuracy[key]) - float(spike_accuracy[key]))
        for key in ("mse", "rmse", "mse_db")
    }
    if not all(np.isfinite(value) for value in metric_differences.values()):
        return _failure(f"non-finite metric difference: {metric_differences}")

    return SpikeSplitSNNResult(
        ok=True,
        skipped=False,
        note=(
            "Split/G2-SNN/G1-SNN module, registry, checkpoint, spike-stat, and CPU "
            f"pipeline checks passed; accuracy metric differences={metric_differences}"
        ),
        split_run_dir=run_dirs["split_knet"],
        spike_run_dir=run_dirs["spike_split_knet"],
        g1_snn_run_dir=run_dirs["g1_snn_split_knet"],
        metric_differences=metric_differences,
    )


if __name__ == "__main__":
    try:
        result = run_spike_split_snn_ablation_tests()
    except DatasetMissingError as exc:
        print(f"[SKIP] {exc}")
        raise SystemExit(0)
    status = "SKIP" if result.skipped else ("PASS" if result.ok else "FAIL")
    print(f"[{status}] {result.note}")
    raise SystemExit(0 if result.ok else 1)
