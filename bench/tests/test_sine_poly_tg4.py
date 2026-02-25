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
class SinePolyTG4Result:
    ok: bool
    note: str
    npz_path: Path


def _bench_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _default_suite_tg4_smoke() -> Path:
    return _bench_root() / "bench" / "configs" / "suite_sine_poly_smoke.yaml"


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _find_task_cfg(suite: Dict[str, Any], task_id: str) -> Dict[str, Any]:
    for t in suite.get("tasks", []) or []:
        if str(t.get("task_id")) == str(task_id):
            return dict(t)
    raise KeyError(f"task_id not found in suite: {task_id}")


def _generate_train_split(
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


def _params(meta: Dict[str, Any], which: str) -> Dict[str, Any]:
    ssm = meta.get("ssm", {})
    if not isinstance(ssm, dict):
        raise KeyError("meta.ssm missing")
    block = ssm.get(which, {})
    if not isinstance(block, dict):
        raise KeyError(f"meta.ssm.{which} missing")
    params = block.get("params", {})
    if not isinstance(params, dict):
        raise KeyError(f"meta.ssm.{which}.params missing")
    return params


def _as_vec(params: Dict[str, Any], name: str, dim: int) -> np.ndarray:
    raw = params.get(name)
    arr = np.asarray(raw, dtype=np.float64)
    if arr.shape != (dim,):
        raise ValueError(f"{name} params must have shape ({dim},), got {arr.shape}")
    return arr


def _check_common(meta: Dict[str, Any], *, expect_mismatch_enabled: bool, expect_kind: str) -> Optional[str]:
    ssm = meta.get("ssm", {})
    if not isinstance(ssm, dict):
        return "meta.ssm missing or not dict"
    true_block = ssm.get("true", {})
    assumed_block = ssm.get("assumed", {})
    if not isinstance(true_block, dict) or not isinstance(assumed_block, dict):
        return "meta.ssm.true/assumed missing"
    if str(true_block.get("type", "")) != "sine_poly":
        return f"meta.ssm.true.type expected sine_poly, got {true_block.get('type')}"
    if str(assumed_block.get("type", "")) != "sine_poly":
        return f"meta.ssm.assumed.type expected sine_poly, got {assumed_block.get('type')}"

    mm = meta.get("mismatch", {})
    if not isinstance(mm, dict):
        return "meta.mismatch missing or not dict"
    if bool(mm.get("enabled", False)) != bool(expect_mismatch_enabled):
        return f"meta.mismatch.enabled mismatch expected={expect_mismatch_enabled}, got={mm.get('enabled')}"
    if str(mm.get("kind", "")) != str(expect_kind):
        return f"meta.mismatch.kind mismatch expected={expect_kind}, got={mm.get('kind')}"

    ns = meta.get("noise_schedule", {})
    if not isinstance(ns, dict) or bool(ns.get("enabled", True)):
        return "meta.noise_schedule.enabled expected false for sine_poly_v0"

    return None


def run_sine_poly_tg4(suite_yaml: Optional[Path] = None) -> SinePolyTG4Result:
    suite_path = suite_yaml.expanduser().resolve() if suite_yaml is not None else _default_suite_tg4_smoke()
    suite = _load_yaml(suite_path)
    suite_name = str((suite.get("suite", {}) or {}).get("name", "sine_poly_smoke"))
    seed = 0

    full_task = _find_task_cfg(suite, task_id="TG4_sine_poly_fullinfo_smoke_v0")
    partial_task = _find_task_cfg(suite, task_id="TG4_sine_poly_partialinfo_smoke_v0")

    # 1) determinism on full-info
    with tempfile.TemporaryDirectory(prefix="tg4_det_a_") as tmp_a, tempfile.TemporaryDirectory(prefix="tg4_det_b_") as tmp_b:
        npz_a, meta_a, x_a, y_a = _generate_train_split(
            cache_root=Path(tmp_a).resolve(),
            suite_name=suite_name,
            task_cfg=full_task,
            seed=seed,
        )
        _, meta_b, x_b, y_b = _generate_train_split(
            cache_root=Path(tmp_b).resolve(),
            suite_name=suite_name,
            task_cfg=full_task,
            seed=seed,
        )
        fp_a = determinism_fingerprint(x_a, y_a, meta_a, k=128)
        fp_b = determinism_fingerprint(x_b, y_b, meta_b, k=128)
        if fp_a.x_first_k_sha256 != fp_b.x_first_k_sha256 or fp_a.y_first_k_sha256 != fp_b.y_first_k_sha256:
            return SinePolyTG4Result(
                ok=False,
                note=(
                    "sine_poly_v0 determinism mismatch for fixed seed; "
                    f"x_hash_a={fp_a.x_first_k_sha256} x_hash_b={fp_b.x_first_k_sha256} "
                    f"y_hash_a={fp_a.y_first_k_sha256} y_hash_b={fp_b.y_first_k_sha256}"
                ),
                npz_path=npz_a,
            )
        if fp_a.meta_required_sha256 != fp_b.meta_required_sha256:
            return SinePolyTG4Result(
                ok=False,
                note=(
                    "sine_poly_v0 required meta hash mismatch for fixed seed; "
                    f"meta_a={fp_a.meta_required_sha256} meta_b={fp_b.meta_required_sha256}"
                ),
                npz_path=npz_a,
            )

    # 2) full-info metadata
    with tempfile.TemporaryDirectory(prefix="tg4_meta_") as tmp:
        cache_root = Path(tmp).resolve()
        npz_full, meta_full, x_full, y_full = _generate_train_split(
            cache_root=cache_root,
            suite_name=suite_name,
            task_cfg=full_task,
            seed=seed,
        )
        if x_full.ndim != 3 or y_full.ndim != 3:
            return SinePolyTG4Result(ok=False, note=f"full-info shape rank mismatch x={x_full.shape} y={y_full.shape}", npz_path=npz_full)
        err = _check_common(meta_full, expect_mismatch_enabled=False, expect_kind="none")
        if err is not None:
            return SinePolyTG4Result(ok=False, note=err, npz_path=npz_full)

        dim = int(meta_full.get("dims", {}).get("x_dim", 0))
        if dim <= 0:
            return SinePolyTG4Result(ok=False, note="meta.dims.x_dim invalid", npz_path=npz_full)
        true_p = _params(meta_full, "true")
        assumed_p = _params(meta_full, "assumed")
        for name in ("alpha", "beta", "phi", "delta", "a", "b", "c"):
            tv = _as_vec(true_p, name, dim=dim)
            av = _as_vec(assumed_p, name, dim=dim)
            if not np.allclose(tv, av, rtol=1e-8, atol=1e-10):
                return SinePolyTG4Result(ok=False, note=f"full-info expected true==assumed for {name}", npz_path=npz_full)

        # 3) partial-info metadata
        npz_partial, meta_partial, _, _ = _generate_train_split(
            cache_root=cache_root,
            suite_name=suite_name,
            task_cfg=partial_task,
            seed=seed,
        )
        err = _check_common(meta_partial, expect_mismatch_enabled=True, expect_kind="param_perturb")
        if err is not None:
            return SinePolyTG4Result(ok=False, note=err, npz_path=npz_partial)

        dim_p = int(meta_partial.get("dims", {}).get("x_dim", 0))
        t_params = _params(meta_partial, "true")
        a_params = _params(meta_partial, "assumed")
        alpha_true = _as_vec(t_params, "alpha", dim=dim_p)
        alpha_assumed = _as_vec(a_params, "alpha", dim=dim_p)
        beta_true = _as_vec(t_params, "beta", dim=dim_p)
        beta_assumed = _as_vec(a_params, "beta", dim=dim_p)
        if np.allclose(alpha_true, alpha_assumed, rtol=1e-8, atol=1e-10):
            return SinePolyTG4Result(ok=False, note="partial-info expected alpha true!=assumed", npz_path=npz_partial)
        if np.allclose(beta_true, beta_assumed, rtol=1e-8, atol=1e-10):
            return SinePolyTG4Result(ok=False, note="partial-info expected beta true!=assumed", npz_path=npz_partial)

        mm_params = meta_partial.get("mismatch", {}).get("params", {})
        if not isinstance(mm_params, dict):
            return SinePolyTG4Result(ok=False, note="partial-info mismatch.params missing", npz_path=npz_partial)
        if "which" not in mm_params:
            return SinePolyTG4Result(ok=False, note="partial-info mismatch.params.which missing", npz_path=npz_partial)

    return SinePolyTG4Result(
        ok=True,
        note="TG4 sine_poly checks passed (determinism + full/partial-info meta assertions)",
        npz_path=npz_partial,
    )
