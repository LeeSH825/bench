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
class SwitchingTG6Result:
    ok: bool
    note: str
    npz_path: Path


def _bench_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _default_suite_tg6_smoke() -> Path:
    return _bench_root() / "bench" / "configs" / "suite_switching_smoke.yaml"


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


def _extract_true_models(meta: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    ssm = meta.get("ssm", {})
    if not isinstance(ssm, dict):
        raise KeyError("meta.ssm missing")
    true = ssm.get("true", {})
    if not isinstance(true, dict):
        raise KeyError("meta.ssm.true missing")
    models = true.get("models", {})
    if not isinstance(models, dict):
        raise KeyError("meta.ssm.true.models missing")
    model_a = models.get("A", {})
    model_b = models.get("B", {})
    if not isinstance(model_a, dict) or not isinstance(model_b, dict):
        raise KeyError("meta.ssm.true.models.A/B missing")
    f_a = np.asarray(model_a.get("F"), dtype=np.float64)
    f_b = np.asarray(model_b.get("F"), dtype=np.float64)
    if f_a.ndim != 2 or f_b.ndim != 2:
        raise ValueError("meta ssm true F matrices must be rank-2")
    return f_a, f_b


def _switch_occurs_stat(
    *,
    x: np.ndarray,
    f_a: np.ndarray,
    f_b: np.ndarray,
    t_change_seq: np.ndarray,
) -> Tuple[float, float, float, float]:
    x_prev = np.asarray(x[:, :-1, :], dtype=np.float64)
    x_next = np.asarray(x[:, 1:, :], dtype=np.float64)
    pred_a = x_prev @ f_a.T
    pred_b = x_prev @ f_b.T
    err_a = np.linalg.norm(x_next - pred_a, axis=2)
    err_b = np.linalg.norm(x_next - pred_b, axis=2)

    n_seq, t_trans = err_a.shape
    t_idx = np.arange(t_trans, dtype=np.int64)
    tc = np.asarray(t_change_seq, dtype=np.int64)
    if tc.shape != (n_seq,):
        raise ValueError(f"t_change_seq shape mismatch: expected ({n_seq},), got {tc.shape}")

    pre_vals_a = []
    pre_vals_b = []
    post_vals_a = []
    post_vals_b = []
    for i in range(n_seq):
        pre_mask = t_idx < int(tc[i])
        post_mask = t_idx >= int(tc[i])
        if np.any(pre_mask):
            pre_vals_a.append(err_a[i, pre_mask])
            pre_vals_b.append(err_b[i, pre_mask])
        if np.any(post_mask):
            post_vals_a.append(err_a[i, post_mask])
            post_vals_b.append(err_b[i, post_mask])

    if not pre_vals_a or not post_vals_a:
        raise ValueError("pre/post switch transition windows are empty; choose valid t_change in [1, T-2]")

    mean_pre_a = float(np.mean(np.concatenate(pre_vals_a)))
    mean_pre_b = float(np.mean(np.concatenate(pre_vals_b)))
    mean_post_a = float(np.mean(np.concatenate(post_vals_a)))
    mean_post_b = float(np.mean(np.concatenate(post_vals_b)))
    return mean_pre_a, mean_pre_b, mean_post_a, mean_post_b


def run_switching_tg6(suite_yaml: Optional[Path] = None) -> SwitchingTG6Result:
    suite_path = suite_yaml.expanduser().resolve() if suite_yaml is not None else _default_suite_tg6_smoke()
    suite = _load_yaml(suite_path)
    suite_name = str((suite.get("suite", {}) or {}).get("name", "switching_smoke"))
    seed = 0
    task_cfg = _find_task_cfg(suite, task_id="TG6_switching_linear_smoke_v0")

    # 1) determinism
    with tempfile.TemporaryDirectory(prefix="tg6_det_a_") as tmp_a, tempfile.TemporaryDirectory(prefix="tg6_det_b_") as tmp_b:
        npz_a, meta_a, x_a, y_a, extras_a = _generate_train_split(
            cache_root=Path(tmp_a).resolve(),
            suite_name=suite_name,
            task_cfg=task_cfg,
            seed=seed,
        )
        _, meta_b, x_b, y_b, extras_b = _generate_train_split(
            cache_root=Path(tmp_b).resolve(),
            suite_name=suite_name,
            task_cfg=task_cfg,
            seed=seed,
        )
        fp_a = determinism_fingerprint(x_a, y_a, meta_a, k=128)
        fp_b = determinism_fingerprint(x_b, y_b, meta_b, k=128)
        if fp_a.x_first_k_sha256 != fp_b.x_first_k_sha256 or fp_a.y_first_k_sha256 != fp_b.y_first_k_sha256:
            return SwitchingTG6Result(
                ok=False,
                note=(
                    "switching_dynamics_v0 determinism mismatch for fixed seed; "
                    f"x_hash_a={fp_a.x_first_k_sha256} x_hash_b={fp_b.x_first_k_sha256} "
                    f"y_hash_a={fp_a.y_first_k_sha256} y_hash_b={fp_b.y_first_k_sha256}"
                ),
                npz_path=npz_a,
            )
        if fp_a.meta_required_sha256 != fp_b.meta_required_sha256:
            return SwitchingTG6Result(
                ok=False,
                note=(
                    "switching_dynamics_v0 required meta hash mismatch for fixed seed; "
                    f"meta_a={fp_a.meta_required_sha256} meta_b={fp_b.meta_required_sha256}"
                ),
                npz_path=npz_a,
            )
        tca = np.asarray(extras_a.get("t_change_seq"), dtype=np.float32)
        tcb = np.asarray(extras_b.get("t_change_seq"), dtype=np.float32)
        if tca.shape != tcb.shape or not np.array_equal(tca, tcb):
            return SwitchingTG6Result(ok=False, note="t_change_seq determinism mismatch", npz_path=npz_a)

    # 2) metadata + switch-dynamics check
    with tempfile.TemporaryDirectory(prefix="tg6_meta_") as tmp:
        npz_path, meta, x, y, extras = _generate_train_split(
            cache_root=Path(tmp).resolve(),
            suite_name=suite_name,
            task_cfg=task_cfg,
            seed=seed,
        )
        if x.ndim != 3 or y.ndim != 3:
            return SwitchingTG6Result(ok=False, note=f"x/y rank mismatch x={x.shape} y={y.shape}", npz_path=npz_path)

        sw = meta.get("switching", {})
        if not isinstance(sw, dict):
            return SwitchingTG6Result(ok=False, note="meta.switching missing or not dict", npz_path=npz_path)
        if not bool(sw.get("enabled", False)):
            return SwitchingTG6Result(ok=False, note="meta.switching.enabled is not true", npz_path=npz_path)
        models = sw.get("models", [])
        if not isinstance(models, list) or "A" not in models or "B" not in models:
            return SwitchingTG6Result(ok=False, note="meta.switching.models must include A and B", npz_path=npz_path)
        if "t_change" not in sw:
            return SwitchingTG6Result(ok=False, note="meta.switching.t_change missing", npz_path=npz_path)

        ssm = meta.get("ssm", {})
        if not isinstance(ssm, dict):
            return SwitchingTG6Result(ok=False, note="meta.ssm missing", npz_path=npz_path)
        if not isinstance(ssm.get("true"), dict) or not isinstance(ssm.get("assumed"), dict):
            return SwitchingTG6Result(ok=False, note="meta.ssm.true/assumed missing", npz_path=npz_path)

        mm = meta.get("mismatch", {})
        if not isinstance(mm, dict):
            return SwitchingTG6Result(ok=False, note="meta.mismatch missing", npz_path=npz_path)
        kind = mm.get("kind")
        if isinstance(kind, list):
            has_kind = "switching_dynamics" in [str(k) for k in kind]
        else:
            has_kind = str(kind) == "switching_dynamics"
        if not has_kind:
            return SwitchingTG6Result(ok=False, note=f"mismatch.kind missing switching_dynamics: {kind}", npz_path=npz_path)

        if "t_change_seq" not in extras:
            return SwitchingTG6Result(ok=False, note="extras.t_change_seq missing", npz_path=npz_path)
        t_change_seq = np.asarray(extras["t_change_seq"], dtype=np.int64)

        f_a, f_b = _extract_true_models(meta)
        try:
            mean_pre_a, mean_pre_b, mean_post_a, mean_post_b = _switch_occurs_stat(
                x=x,
                f_a=f_a,
                f_b=f_b,
                t_change_seq=t_change_seq,
            )
        except Exception as exc:
            return SwitchingTG6Result(ok=False, note=f"switch occurs stat failed: {exc}", npz_path=npz_path)
        if not (mean_pre_a < mean_pre_b):
            return SwitchingTG6Result(
                ok=False,
                note=(
                    "switch signal failed in pre-window: expected A residual < B residual, "
                    f"got pre_A={mean_pre_a:.6e}, pre_B={mean_pre_b:.6e}"
                ),
                npz_path=npz_path,
            )
        if not (mean_post_b < mean_post_a):
            return SwitchingTG6Result(
                ok=False,
                note=(
                    "switch signal failed in post-window: expected B residual < A residual, "
                    f"got post_A={mean_post_a:.6e}, post_B={mean_post_b:.6e}"
                ),
                npz_path=npz_path,
            )

    return SwitchingTG6Result(
        ok=True,
        note="TG6 switching checks passed (determinism + meta + switch dynamics signal)",
        npz_path=npz_path,
    )
