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
class LorenzTG5Result:
    ok: bool
    note: str
    npz_path: Path


def _bench_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _default_suite_tg5_smoke() -> Path:
    return _bench_root() / "bench" / "configs" / "suite_lorenz_smoke.yaml"


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _find_task_cfg(suite: Dict[str, Any], task_id: str) -> Dict[str, Any]:
    for t in suite.get("tasks", []) or []:
        if str(t.get("task_id")) == str(task_id):
            return dict(t)
    raise KeyError(f"task_id not found in suite: {task_id}")


def _generate_split(
    *,
    cache_root: Path,
    suite_name: str,
    task_cfg: Dict[str, Any],
    seed: int,
    split_name: str = "train",
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
    art = arts[0]
    if split_name == "train":
        npz_path = art.train.path
    elif split_name == "val":
        npz_path = art.val.path
    elif split_name == "test":
        npz_path = art.test.path
    else:
        raise ValueError(f"unsupported split_name={split_name}")

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


def _contains_kind(kind_field: Any, expected: str) -> bool:
    if isinstance(kind_field, list):
        return str(expected) in [str(v) for v in kind_field]
    return str(kind_field) == str(expected)


def run_lorenz_tg5(suite_yaml: Optional[Path] = None) -> LorenzTG5Result:
    suite_path = suite_yaml.expanduser().resolve() if suite_yaml is not None else _default_suite_tg5_smoke()
    suite = _load_yaml(suite_path)
    suite_name = str((suite.get("suite", {}) or {}).get("name", "lorenz_smoke"))
    seed = 0

    full_task = _find_task_cfg(suite, task_id="TG5_lorenz_fullinfo_smoke_v0")
    mismatch_task = _find_task_cfg(suite, task_id="TG5_lorenz_Jmismatch_smoke_v0")

    # 1) determinism for full-info task
    with tempfile.TemporaryDirectory(prefix="tg5_det_a_") as tmp_a, tempfile.TemporaryDirectory(prefix="tg5_det_b_") as tmp_b:
        npz_a, meta_a, x_a, y_a = _generate_split(
            cache_root=Path(tmp_a).resolve(),
            suite_name=suite_name,
            task_cfg=full_task,
            seed=seed,
            split_name="train",
        )
        _, meta_b, x_b, y_b = _generate_split(
            cache_root=Path(tmp_b).resolve(),
            suite_name=suite_name,
            task_cfg=full_task,
            seed=seed,
            split_name="train",
        )
        fp_a = determinism_fingerprint(x_a, y_a, meta_a, k=128)
        fp_b = determinism_fingerprint(x_b, y_b, meta_b, k=128)
        if fp_a.x_first_k_sha256 != fp_b.x_first_k_sha256 or fp_a.y_first_k_sha256 != fp_b.y_first_k_sha256:
            return LorenzTG5Result(
                ok=False,
                note=(
                    "lorenz_v0 determinism mismatch for fixed seed; "
                    f"x_hash_a={fp_a.x_first_k_sha256} x_hash_b={fp_b.x_first_k_sha256} "
                    f"y_hash_a={fp_a.y_first_k_sha256} y_hash_b={fp_b.y_first_k_sha256}"
                ),
                npz_path=npz_a,
            )
        if fp_a.meta_required_sha256 != fp_b.meta_required_sha256:
            return LorenzTG5Result(
                ok=False,
                note=(
                    "lorenz_v0 required meta hash mismatch for fixed seed; "
                    f"meta_a={fp_a.meta_required_sha256} meta_b={fp_b.meta_required_sha256}"
                ),
                npz_path=npz_a,
            )

    # 2) mismatch metadata checks
    with tempfile.TemporaryDirectory(prefix="tg5_meta_") as tmp:
        cache_root = Path(tmp).resolve()
        npz_train, meta_train, x_train, y_train = _generate_split(
            cache_root=cache_root,
            suite_name=suite_name,
            task_cfg=mismatch_task,
            seed=seed,
            split_name="train",
        )
        if x_train.ndim != 3 or y_train.ndim != 3:
            return LorenzTG5Result(ok=False, note=f"train rank mismatch x={x_train.shape} y={y_train.shape}", npz_path=npz_train)
        if x_train.dtype != np.float32 or y_train.dtype != np.float32:
            return LorenzTG5Result(ok=False, note=f"train dtype mismatch x={x_train.dtype} y={y_train.dtype}", npz_path=npz_train)

        ssm = meta_train.get("ssm", {})
        if not isinstance(ssm, dict):
            return LorenzTG5Result(ok=False, note="meta.ssm missing", npz_path=npz_train)
        true_block = ssm.get("true")
        assumed_block = ssm.get("assumed")
        if not isinstance(true_block, dict) or not isinstance(assumed_block, dict):
            return LorenzTG5Result(ok=False, note="meta.ssm.true/assumed missing", npz_path=npz_train)

        mm = meta_train.get("mismatch", {})
        if not isinstance(mm, dict):
            return LorenzTG5Result(ok=False, note="meta.mismatch missing", npz_path=npz_train)
        if not bool(mm.get("enabled", False)):
            return LorenzTG5Result(ok=False, note="meta.mismatch.enabled expected true for J-mismatch task", npz_path=npz_train)
        if not _contains_kind(mm.get("kind"), "lorenz_J"):
            return LorenzTG5Result(ok=False, note=f"mismatch.kind missing lorenz_J: {mm.get('kind')}", npz_path=npz_train)

        params_true = true_block.get("params", {})
        params_assumed = assumed_block.get("params", {})
        if not isinstance(params_true, dict) or not isinstance(params_assumed, dict):
            return LorenzTG5Result(ok=False, note="meta.ssm.true/assumed.params missing", npz_path=npz_train)
        if int(params_true.get("J", -1)) == int(params_assumed.get("J", -1)):
            return LorenzTG5Result(ok=False, note="expected J_true != J_assumed for J-mismatch task", npz_path=npz_train)

        # Split-specific T support check.
        npz_test, _meta_test, x_test, y_test = _generate_split(
            cache_root=cache_root,
            suite_name=suite_name,
            task_cfg=mismatch_task,
            seed=seed,
            split_name="test",
        )
        if int(x_test.shape[1]) != 80 or int(y_test.shape[1]) != 80:
            return LorenzTG5Result(
                ok=False,
                note=f"expected test T=80 from suite config, got x={x_test.shape} y={y_test.shape}",
                npz_path=npz_test,
            )

    return LorenzTG5Result(
        ok=True,
        note="TG5 lorenz checks passed (determinism + mismatch/meta assertions + split-specific T support)",
        npz_path=npz_train,
    )
