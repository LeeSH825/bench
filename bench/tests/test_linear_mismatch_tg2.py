from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import yaml

from bench.tasks.bench_generated import prepare_bench_generated_v0
from bench.tasks.data_format import load_npz_split_v0
from bench.tasks.generator.contract import make_split_cfg, make_task_cfg
from bench.tasks.generator.schema import enforce_meta_v1
from bench.tasks.generator.validate import determinism_fingerprint, validate_artifacts


@dataclass
class LinearMismatchTG2Result:
    ok: bool
    note: str
    npz_path: Path


def _bench_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _default_suite_tg2_smoke() -> Path:
    return _bench_root() / "bench" / "configs" / "suite_linear_mismatch_smoke.yaml"


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _find_task_cfg(suite: Dict[str, Any], task_id: str) -> Dict[str, Any]:
    for t in suite.get("tasks", []) or []:
        if str(t.get("task_id")) == str(task_id):
            return dict(t)
    raise KeyError(f"task_id not found in suite: {task_id}")


def _gen_split(
    *,
    cache_root: Path,
    suite_name: str,
    task_cfg: Dict[str, Any],
    seed: int,
) -> Tuple[Path, Dict[str, Any], np.ndarray, np.ndarray]:
    arts = prepare_bench_generated_v0(
        suite_name=suite_name,
        task_cfg=task_cfg,
        seed=int(seed),
        cache_root=cache_root,
        scenario_overrides={},
    )
    if not arts:
        raise RuntimeError("prepare_bench_generated_v0 returned no artifacts")
    npz_path = arts[0].train.path
    split = load_npz_split_v0(npz_path)

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
    return npz_path, meta_v1, split.x, split.y


def _matrix(meta: Dict[str, Any], which: str, key: str) -> np.ndarray:
    ssm = meta.get("ssm", {})
    if not isinstance(ssm, dict):
        raise KeyError("meta.ssm missing")
    block = ssm.get(which, {})
    if not isinstance(block, dict):
        raise KeyError(f"meta.ssm.{which} missing")
    raw = block.get(key)
    arr = np.asarray(raw, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"meta.ssm.{which}.{key} must be rank-2 matrix, got shape={arr.shape}")
    return arr


def _check_meta_for_mode(meta: Dict[str, Any], *, expected_kind: str, expected_alpha: float) -> Optional[str]:
    mm = meta.get("mismatch", {})
    if not isinstance(mm, dict):
        return "meta.mismatch missing or not dict"
    if not bool(mm.get("enabled", False)):
        return "meta.mismatch.enabled is not true"
    kind = str(mm.get("kind", ""))
    if kind != expected_kind:
        return f"meta.mismatch.kind mismatch: expected={expected_kind}, got={kind}"

    params = mm.get("params", {})
    if not isinstance(params, dict):
        return "meta.mismatch.params missing or not dict"
    try:
        alpha = float(params.get("alpha_true_deg"))
    except Exception:
        return "meta.mismatch.params.alpha_true_deg missing/non-numeric"
    if not np.isclose(alpha, float(expected_alpha), rtol=0.0, atol=1e-9):
        return f"meta mismatch alpha_true_deg mismatch: expected={expected_alpha}, got={alpha}"

    f_true = _matrix(meta, "true", "F")
    f_assumed = _matrix(meta, "assumed", "F")
    h_true = _matrix(meta, "true", "H")
    h_assumed = _matrix(meta, "assumed", "H")

    if expected_kind == "F_rotation":
        if np.allclose(f_true, f_assumed, rtol=1e-8, atol=1e-10):
            return "F_rotation expected F_true != F_assumed"
        if not np.allclose(h_true, h_assumed, rtol=1e-8, atol=1e-10):
            return "F_rotation expected H_true == H_assumed"
    elif expected_kind == "H_rotation":
        if np.allclose(h_true, h_assumed, rtol=1e-8, atol=1e-10):
            return "H_rotation expected H_true != H_assumed"
        if not np.allclose(f_true, f_assumed, rtol=1e-8, atol=1e-10):
            return "H_rotation expected F_true == F_assumed"
    else:
        return f"unsupported expected_kind in test: {expected_kind}"

    ns = meta.get("noise_schedule", {})
    if not isinstance(ns, dict):
        return "meta.noise_schedule missing or not dict"
    if bool(ns.get("enabled", True)):
        return "meta.noise_schedule.enabled expected false for TG2 smoke"
    return None


def run_linear_mismatch_tg2(suite_yaml: Optional[Path] = None) -> LinearMismatchTG2Result:
    suite_path = suite_yaml.expanduser().resolve() if suite_yaml is not None else _default_suite_tg2_smoke()
    suite = _load_yaml(suite_path)
    suite_name = str((suite.get("suite", {}) or {}).get("name", "linear_mismatch_smoke"))
    seed = 0

    task_f = _find_task_cfg(suite, task_id="TG2_F_rotation_smoke_v0")
    task_h = _find_task_cfg(suite, task_id="TG2_H_rotation_smoke_v0")

    with tempfile.TemporaryDirectory(prefix="tg2_smoke_a_") as tmp_a, tempfile.TemporaryDirectory(prefix="tg2_smoke_b_") as tmp_b:
        npz_a, meta_a, x_a, y_a = _gen_split(
            cache_root=Path(tmp_a).resolve(),
            suite_name=suite_name,
            task_cfg=task_f,
            seed=seed,
        )
        npz_b, meta_b, x_b, y_b = _gen_split(
            cache_root=Path(tmp_b).resolve(),
            suite_name=suite_name,
            task_cfg=task_f,
            seed=seed,
        )
        fp_a = determinism_fingerprint(x_a, y_a, meta_a, k=128)
        fp_b = determinism_fingerprint(x_b, y_b, meta_b, k=128)
        if fp_a.x_first_k_sha256 != fp_b.x_first_k_sha256 or fp_a.y_first_k_sha256 != fp_b.y_first_k_sha256:
            return LinearMismatchTG2Result(
                ok=False,
                note=(
                    "determinism mismatch for TG2 F_rotation fixed seed; "
                    f"x_hash_a={fp_a.x_first_k_sha256} x_hash_b={fp_b.x_first_k_sha256} "
                    f"y_hash_a={fp_a.y_first_k_sha256} y_hash_b={fp_b.y_first_k_sha256}"
                ),
                npz_path=npz_a,
            )
        if fp_a.meta_required_sha256 != fp_b.meta_required_sha256:
            return LinearMismatchTG2Result(
                ok=False,
                note=(
                    "required meta hash mismatch for TG2 F_rotation fixed seed; "
                    f"meta_a={fp_a.meta_required_sha256} meta_b={fp_b.meta_required_sha256}"
                ),
                npz_path=npz_a,
            )

    with tempfile.TemporaryDirectory(prefix="tg2_meta_") as tmp:
        cache_root = Path(tmp).resolve()
        npz_f, meta_f, _, _ = _gen_split(cache_root=cache_root, suite_name=suite_name, task_cfg=task_f, seed=seed)
        err = _check_meta_for_mode(meta_f, expected_kind="F_rotation", expected_alpha=10.0)
        if err is not None:
            return LinearMismatchTG2Result(ok=False, note=err, npz_path=npz_f)

        npz_h, meta_h, _, _ = _gen_split(cache_root=cache_root, suite_name=suite_name, task_cfg=task_h, seed=seed)
        err = _check_meta_for_mode(meta_h, expected_kind="H_rotation", expected_alpha=10.0)
        if err is not None:
            return LinearMismatchTG2Result(ok=False, note=err, npz_path=npz_h)

    return LinearMismatchTG2Result(
        ok=True,
        note="TG2 linear_mismatch checks passed (determinism + true/assumed meta assertions for F/H rotations)",
        npz_path=npz_a,
    )
