from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from bench.tasks.bench_generated import prepare_bench_generated_v0
from bench.tasks.data_format import load_npz_split_v0
from bench.tasks.generator.contract import make_split_cfg, make_task_cfg
from bench.tasks.generator.schema import enforce_meta_v1
from bench.tasks.generator.validate import determinism_fingerprint, validate_artifacts


@dataclass
class GeneratorContractTG0Result:
    ok: bool
    note: str
    npz_path: Path


def _bench_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _default_suite_tg0_smoke() -> Path:
    return _bench_root() / "bench" / "configs" / "suite_tg0_smoke.yaml"


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _find_task_cfg(suite: Dict[str, Any], task_id: str) -> Dict[str, Any]:
    for t in suite.get("tasks", []) or []:
        if str(t.get("task_id")) == str(task_id):
            return dict(t)
    raise KeyError(f"task_id not found in suite: {task_id}")


def _generate_and_fingerprint(*, cache_root: Path, suite_name: str, task_cfg: Dict[str, Any], seed: int) -> tuple[Path, str, str, str]:
    arts = prepare_bench_generated_v0(
        suite_name=suite_name,
        task_cfg=task_cfg,
        seed=int(seed),
        cache_root=cache_root,
        scenario_overrides={},
    )
    if not arts:
        raise RuntimeError("prepare_bench_generated_v0 returned no artifacts")
    split = load_npz_split_v0(arts[0].train.path)
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
    fp = determinism_fingerprint(split.x, split.y, meta_v1, k=128)
    return (
        arts[0].train.path,
        fp.x_first_k_sha256,
        fp.y_first_k_sha256,
        fp.meta_required_sha256,
    )


def run_generator_contract_tg0_smoke(suite_yaml: Optional[Path] = None) -> GeneratorContractTG0Result:
    suite_path = suite_yaml.expanduser().resolve() if suite_yaml is not None else _default_suite_tg0_smoke()
    suite = _load_yaml(suite_path)
    suite_name = str((suite.get("suite", {}) or {}).get("name", "tg0_smoke"))
    task_cfg = _find_task_cfg(suite, task_id="TG0_linear_smoke_v0")
    seed = 0

    with tempfile.TemporaryDirectory(prefix="tg0_smoke_a_") as tmp_a, tempfile.TemporaryDirectory(prefix="tg0_smoke_b_") as tmp_b:
        npz_a, xh_a, yh_a, mh_a = _generate_and_fingerprint(
            cache_root=Path(tmp_a).resolve(),
            suite_name=suite_name,
            task_cfg=task_cfg,
            seed=seed,
        )
        npz_b, xh_b, yh_b, mh_b = _generate_and_fingerprint(
            cache_root=Path(tmp_b).resolve(),
            suite_name=suite_name,
            task_cfg=task_cfg,
            seed=seed,
        )

    if xh_a != xh_b or yh_a != yh_b:
        return GeneratorContractTG0Result(
            ok=False,
            note=(
                "determinism mismatch for fixed seed; "
                f"x_hash_a={xh_a} x_hash_b={xh_b} y_hash_a={yh_a} y_hash_b={yh_b}"
            ),
            npz_path=npz_a,
        )
    if mh_a != mh_b:
        return GeneratorContractTG0Result(
            ok=False,
            note=f"required meta keys are unstable across same-seed runs: a={mh_a} b={mh_b}",
            npz_path=npz_a,
        )

    return GeneratorContractTG0Result(
        ok=True,
        note="TG0 contract smoke passed (schema validation + same-seed determinism hash match)",
        npz_path=npz_a,
    )
