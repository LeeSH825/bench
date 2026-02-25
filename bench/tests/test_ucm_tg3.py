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
class UCMTG3Result:
    ok: bool
    note: str
    npz_path: Path


def _bench_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _default_suite_tg3_smoke() -> Path:
    return _bench_root() / "bench" / "configs" / "suite_ucm_smoke.yaml"


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
) -> Tuple[Path, Dict[str, Any], np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
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
    return npz_path, meta_v1, split.x, split.y, split.extras


def _check_common_meta(meta: Dict[str, Any], expected_obs_mode: str) -> Optional[str]:
    ssm = meta.get("ssm", {})
    if not isinstance(ssm, dict):
        return "meta.ssm missing or not dict"
    ssm_true = ssm.get("true", {})
    ssm_assumed = ssm.get("assumed", {})
    if not isinstance(ssm_true, dict) or not isinstance(ssm_assumed, dict):
        return "meta.ssm.true/assumed missing"

    true_params = ssm_true.get("params", {})
    if not isinstance(true_params, dict):
        return "meta.ssm.true.params missing"
    obs_mode = str(true_params.get("obs_mode", ""))
    if obs_mode != expected_obs_mode:
        return f"obs_mode mismatch: expected={expected_obs_mode}, got={obs_mode}"

    if expected_obs_mode == "nonlinear":
        units = true_params.get("measurement_units", {})
        if not isinstance(units, dict) or str(units.get("angle", "")) != "radians":
            return "nonlinear mode must record angle units in radians"

    mm = meta.get("mismatch", {})
    if not isinstance(mm, dict) or bool(mm.get("enabled", True)):
        return "meta.mismatch.enabled expected false for ucm_v0"

    ns = meta.get("noise_schedule", {})
    if not isinstance(ns, dict) or bool(ns.get("enabled", True)):
        return "meta.noise_schedule.enabled expected false for ucm_v0"

    sw = meta.get("switching", {})
    if not isinstance(sw, dict) or bool(sw.get("enabled", True)):
        return "meta.switching.enabled expected false for ucm_v0"

    return None


def run_ucm_tg3(suite_yaml: Optional[Path] = None) -> UCMTG3Result:
    suite_path = suite_yaml.expanduser().resolve() if suite_yaml is not None else _default_suite_tg3_smoke()
    suite = _load_yaml(suite_path)
    suite_name = str((suite.get("suite", {}) or {}).get("name", "ucm_smoke"))
    seed = 0

    task_linear = _find_task_cfg(suite, task_id="TG3_ucm_linear_smoke_v0")
    task_nonlinear = _find_task_cfg(suite, task_id="TG3_ucm_nonlinear_smoke_v0")
    task_taskset = _find_task_cfg(suite, task_id="TG3_ucm_taskset_smoke_v0")

    # 1) determinism check for linear mode
    with tempfile.TemporaryDirectory(prefix="tg3_det_a_") as tmp_a, tempfile.TemporaryDirectory(prefix="tg3_det_b_") as tmp_b:
        npz_a, meta_a, x_a, y_a, _ = _generate_train_split(
            cache_root=Path(tmp_a).resolve(),
            suite_name=suite_name,
            task_cfg=task_linear,
            seed=seed,
        )
        _, meta_b, x_b, y_b, _ = _generate_train_split(
            cache_root=Path(tmp_b).resolve(),
            suite_name=suite_name,
            task_cfg=task_linear,
            seed=seed,
        )
        fp_a = determinism_fingerprint(x_a, y_a, meta_a, k=128)
        fp_b = determinism_fingerprint(x_b, y_b, meta_b, k=128)
        if fp_a.x_first_k_sha256 != fp_b.x_first_k_sha256 or fp_a.y_first_k_sha256 != fp_b.y_first_k_sha256:
            return UCMTG3Result(
                ok=False,
                note=(
                    "ucm_v0 linear determinism mismatch for fixed seed; "
                    f"x_hash_a={fp_a.x_first_k_sha256} x_hash_b={fp_b.x_first_k_sha256} "
                    f"y_hash_a={fp_a.y_first_k_sha256} y_hash_b={fp_b.y_first_k_sha256}"
                ),
                npz_path=npz_a,
            )
        if fp_a.meta_required_sha256 != fp_b.meta_required_sha256:
            return UCMTG3Result(
                ok=False,
                note=(
                    "ucm_v0 linear required meta hash mismatch for fixed seed; "
                    f"meta_a={fp_a.meta_required_sha256} meta_b={fp_b.meta_required_sha256}"
                ),
                npz_path=npz_a,
            )

    # 2) linear + nonlinear schema/meta checks
    with tempfile.TemporaryDirectory(prefix="tg3_meta_") as tmp:
        cache_root = Path(tmp).resolve()

        npz_linear, meta_linear, x_linear, y_linear, _ = _generate_train_split(
            cache_root=cache_root,
            suite_name=suite_name,
            task_cfg=task_linear,
            seed=seed,
        )
        if x_linear.ndim != 3 or y_linear.ndim != 3:
            return UCMTG3Result(ok=False, note=f"linear split rank mismatch x={x_linear.shape} y={y_linear.shape}", npz_path=npz_linear)
        err = _check_common_meta(meta_linear, expected_obs_mode="linear")
        if err is not None:
            return UCMTG3Result(ok=False, note=err, npz_path=npz_linear)

        npz_nl, meta_nl, x_nl, y_nl, _ = _generate_train_split(
            cache_root=cache_root,
            suite_name=suite_name,
            task_cfg=task_nonlinear,
            seed=seed,
        )
        if x_nl.ndim != 3 or y_nl.ndim != 3:
            return UCMTG3Result(ok=False, note=f"nonlinear split rank mismatch x={x_nl.shape} y={y_nl.shape}", npz_path=npz_nl)
        err = _check_common_meta(meta_nl, expected_obs_mode="nonlinear")
        if err is not None:
            return UCMTG3Result(ok=False, note=err, npz_path=npz_nl)

    # 3) task_set extras checks (task_key shape/determinism)
    with tempfile.TemporaryDirectory(prefix="tg3_taskset_a_") as tmp_a, tempfile.TemporaryDirectory(prefix="tg3_taskset_b_") as tmp_b:
        npz_ta, meta_ta, _, _, extras_a = _generate_train_split(
            cache_root=Path(tmp_a).resolve(),
            suite_name=suite_name,
            task_cfg=task_taskset,
            seed=seed,
        )
        _, meta_tb, _, _, extras_b = _generate_train_split(
            cache_root=Path(tmp_b).resolve(),
            suite_name=suite_name,
            task_cfg=task_taskset,
            seed=seed,
        )
        task_set_meta = meta_ta.get("task_set", {})
        if not isinstance(task_set_meta, dict) or not bool(task_set_meta.get("enabled", False)):
            return UCMTG3Result(ok=False, note="task_set enabled meta block missing for task_set smoke", npz_path=npz_ta)
        if "task_key" not in extras_a:
            return UCMTG3Result(ok=False, note="task_set smoke missing extras['task_key']", npz_path=npz_ta)
        task_key_a = np.asarray(extras_a["task_key"], dtype=np.float32)
        task_key_b = np.asarray(extras_b.get("task_key"), dtype=np.float32)
        n_train = int(task_taskset.get("dataset_sizes", {}).get("N_train", 0))
        if task_key_a.shape != (n_train, 2):
            return UCMTG3Result(
                ok=False,
                note=f"task_key shape mismatch: expected ({n_train},2), got {task_key_a.shape}",
                npz_path=npz_ta,
            )
        if not np.array_equal(task_key_a, task_key_b):
            return UCMTG3Result(ok=False, note="task_key determinism mismatch for fixed seed", npz_path=npz_ta)
        if not isinstance(meta_tb.get("task_set"), dict):
            return UCMTG3Result(ok=False, note="task_set meta missing in second deterministic run", npz_path=npz_ta)

    return UCMTG3Result(
        ok=True,
        note="TG3 ucm checks passed (linear/nonlinear schema + determinism + task_key task_set support)",
        npz_path=npz_ta,
    )
