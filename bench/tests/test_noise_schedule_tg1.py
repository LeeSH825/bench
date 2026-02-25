from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import yaml

from bench.tasks.bench_generated import prepare_bench_generated_v0
from bench.tasks.data_format import load_npz_split_v0
from bench.tasks.generator.contract import make_split_cfg, make_task_cfg
from bench.tasks.generator.noise_schedule import build_noise_schedule
from bench.tasks.generator.schema import enforce_meta_v1
from bench.tasks.generator.validate import validate_artifacts


@dataclass
class NoiseScheduleTG1Result:
    ok: bool
    note: str
    npz_path: Path


def _bench_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _default_suite_tg1_smoke() -> Path:
    return _bench_root() / "bench" / "configs" / "suite_tg1_smoke.yaml"


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _find_task_cfg(suite: Dict[str, Any], task_id: str) -> Dict[str, Any]:
    for t in suite.get("tasks", []) or []:
        if str(t.get("task_id")) == str(task_id):
            return dict(t)
    raise KeyError(f"task_id not found in suite: {task_id}")


def _check(cond: bool, msg: str) -> Optional[str]:
    return None if cond else msg


def run_noise_schedule_tg1(suite_yaml: Optional[Path] = None) -> NoiseScheduleTG1Result:
    # 1) determinism for stochastic per_step_jump
    det_cfg = {
        "enabled": True,
        "kind": "per_step_jump",
        "params": {
            "base_q2": 1.0e-3,
            "base_r2": 1.0e-2,
            "p_jump": 0.4,
            "jump_mult_dist": {"kind": "discrete", "values": [0.5, 1.0, 2.0, 4.0]},
        },
        "sow_hat": {"enabled": True, "mode": "mul_linear", "sigma_mul": 0.15},
    }
    a1, _ = build_noise_schedule(det_cfg, T=32, seed=7)
    a2, _ = build_noise_schedule(det_cfg, T=32, seed=7)
    for k in ("q2_t", "r2_t", "SoW_t", "SoW_dB_t", "SoW_hat_t"):
        if k in a1 or k in a2:
            if not np.array_equal(a1.get(k), a2.get(k)):
                return NoiseScheduleTG1Result(ok=False, note=f"determinism failed for key={k}", npz_path=Path(""))

    # 2) invariants SoW and dB
    sow_ref = np.asarray(a1["q2_t"], dtype=np.float64) / np.asarray(a1["r2_t"], dtype=np.float64)
    if not np.allclose(np.asarray(a1["SoW_t"], dtype=np.float64), sow_ref, rtol=1e-6, atol=1e-8):
        return NoiseScheduleTG1Result(ok=False, note="SoW_t invariant failed", npz_path=Path(""))
    sow_db_ref = 10.0 * np.log10(np.maximum(sow_ref, 1e-12))
    if not np.allclose(np.asarray(a1["SoW_dB_t"], dtype=np.float64), sow_db_ref, rtol=1e-6, atol=1e-6):
        return NoiseScheduleTG1Result(ok=False, note="SoW_dB_t invariant failed", npz_path=Path(""))

    # 3) split_eq20 exact arrays for small T
    train_cfg = {
        "enabled": True,
        "kind": "split_eq20",
        "params": {"phase": "train", "t0": 20, "q2_base": 1.0e-3},
    }
    arr_train, _ = build_noise_schedule(train_cfg, T=15, seed=0)
    expected_db_train = np.asarray([(t // 2 + 20) % 50 for t in range(15)], dtype=np.float64)
    expected_r2_train = np.power(10.0, expected_db_train / 10.0)
    if not np.allclose(np.asarray(arr_train["r2_t"], dtype=np.float64), expected_r2_train, rtol=1e-6, atol=1e-6):
        return NoiseScheduleTG1Result(ok=False, note="split_eq20 train expected array mismatch", npz_path=Path(""))

    test_cfg = {
        "enabled": True,
        "kind": "split_eq20",
        "params": {"phase": "test", "t0": 20, "q2_base": 1.0e-3},
    }
    arr_test, _ = build_noise_schedule(test_cfg, T=25, seed=0)
    expected_db_test = np.asarray([(t // 10 + 30) % 50 for t in range(25)], dtype=np.float64)
    expected_r2_test = np.power(10.0, expected_db_test / 10.0)
    if not np.allclose(np.asarray(arr_test["r2_t"], dtype=np.float64), expected_r2_test, rtol=1e-6, atol=1e-6):
        return NoiseScheduleTG1Result(ok=False, note="split_eq20 test expected array mismatch", npz_path=Path(""))

    # 4) integration smoke: NPZ extras + meta noise_schedule
    suite_path = suite_yaml.expanduser().resolve() if suite_yaml is not None else _default_suite_tg1_smoke()
    suite = _load_yaml(suite_path)
    suite_name = str((suite.get("suite", {}) or {}).get("name", "tg1_smoke"))
    task_cfg = _find_task_cfg(suite, task_id="TG1_noise_schedule_smoke_v0")

    with tempfile.TemporaryDirectory(prefix="tg1_smoke_") as tmp:
        cache_root = Path(tmp).resolve()
        arts = prepare_bench_generated_v0(
            suite_name=suite_name,
            task_cfg=task_cfg,
            seed=0,
            cache_root=cache_root,
            scenario_overrides={},
        )
        if not arts:
            return NoiseScheduleTG1Result(ok=False, note="prepare_bench_generated_v0 returned no artifacts", npz_path=Path(""))
        npz_path = arts[0].train.path
        split = load_npz_split_v0(npz_path)

        for req in ("q2_t", "r2_t", "SoW_t", "SoW_hat_t"):
            miss = _check(req in split.extras, f"integration smoke missing NPZ extra '{req}'")
            if miss:
                return NoiseScheduleTG1Result(ok=False, note=miss, npz_path=npz_path)

        ns_meta = split.meta.get("noise_schedule", {})
        if not isinstance(ns_meta, dict):
            return NoiseScheduleTG1Result(ok=False, note="meta.noise_schedule missing or not dict", npz_path=npz_path)
        if not bool(ns_meta.get("enabled", False)):
            return NoiseScheduleTG1Result(ok=False, note="meta.noise_schedule.enabled is not true", npz_path=npz_path)
        if str(ns_meta.get("kind")) != "split_eq20":
            return NoiseScheduleTG1Result(
                ok=False,
                note=f"meta.noise_schedule.kind mismatch: {ns_meta.get('kind')}",
                npz_path=npz_path,
            )

        task_obj = make_task_cfg(task_cfg, scenario_cfg={})
        split_obj = make_split_cfg(task_cfg)
        meta_v1 = enforce_meta_v1(
            split.meta,
            task_cfg=task_obj,
            split_cfg=split_obj,
            x=split.x,
            y=split.y,
            extras=split.extras,
        )
        validate_artifacts(split.x, split.y, meta_v1, strict=True)

    return NoiseScheduleTG1Result(
        ok=True,
        note="TG1 noise_schedule checks passed (determinism, invariants, split_eq20, integration smoke)",
        npz_path=npz_path,
    )
