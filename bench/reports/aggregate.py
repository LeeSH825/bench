from __future__ import annotations

import copy
import csv
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    import yaml  # type: ignore
except Exception:
    yaml = None


# -----------------------------
# Data structures
# -----------------------------
@dataclass
class RunRecord:
    suite: str
    task_id: str
    scenario_id: str
    seed: int
    model_id: str
    init_id: str
    track_id: str
    status: str  # ok / failed / missing / missing_data / ...
    run_dir: Path

    # scalar metrics (None if missing/failed)
    mse: Optional[float] = None
    rmse: Optional[float] = None
    mse_db: Optional[float] = None
    timing_ms_per_step: Optional[float] = None
    recovery_k: Optional[float] = None
    nll_value: Optional[float] = None  # policy: NA if no cov

    # system/noise metadata (for figure-style analysis)
    x_dim: Optional[int] = None
    y_dim: Optional[int] = None
    T: Optional[int] = None
    q2: Optional[float] = None
    r2: Optional[float] = None
    inv_r2_db: Optional[float] = None

    # shift metadata
    t0_used: Optional[int] = None
    severity_r_scale: Optional[float] = None

    # failure metadata
    failure_type: Optional[str] = None
    failure_stage: Optional[str] = None
    error: Optional[str] = None

    # budget/ledger
    train_updates_used: Optional[int] = None
    adapt_updates_used: Optional[int] = None
    adapt_updates_per_step_max: Optional[int] = None
    train_max_updates: Optional[int] = None
    adapt_max_updates: Optional[int] = None
    adapt_max_updates_per_step: Optional[int] = None

    # cache flags
    cache_enabled: Optional[bool] = None
    cache_hit: Optional[bool] = None
    train_skipped: Optional[bool] = None
    cache_key: Optional[str] = None

    # ops timings
    train_time_s: Optional[float] = None
    eval_time_s: Optional[float] = None
    adapt_time_s: Optional[float] = None
    total_time_s: Optional[float] = None


# -----------------------------
# Helpers: load / dump
# -----------------------------
def load_yaml(path: Path) -> Dict[str, Any]:
    if yaml is None:
        raise RuntimeError("PyYAML is required. Please install pyyaml.")
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default


def safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        if isinstance(x, str) and x.strip().lower() in ("na", "nan", ""):
            return None
        return float(x)
    except Exception:
        return None


def safe_bool(x: Any, default: Optional[bool] = None) -> Optional[bool]:
    if x is None:
        return default
    if isinstance(x, bool):
        return x
    if isinstance(x, (int, float)):
        return bool(x)
    if isinstance(x, str):
        v = x.strip().lower()
        if v in ("1", "true", "yes", "y", "on"):
            return True
        if v in ("0", "false", "no", "n", "off"):
            return False
    return default


def _first_non_none(vals: Iterable[Any]) -> Any:
    for v in vals:
        if v is not None:
            return v
    return None


def _first_float(vals: Iterable[Any]) -> Optional[float]:
    for v in vals:
        fv = safe_float(v)
        if fv is not None:
            return fv
    return None


def _first_int(vals: Iterable[Any]) -> Optional[int]:
    for v in vals:
        if v is None:
            continue
        try:
            return int(v)
        except Exception:
            continue
    return None


def _dotted_get(obj: Any, dotted_key: str) -> Any:
    cur = obj
    for part in dotted_key.split("."):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(part)
    return cur


def _normalize_per_step_updates(x: Any) -> Dict[int, int]:
    out: Dict[int, int] = {}
    if not isinstance(x, dict):
        return out
    for k, v in x.items():
        try:
            out[int(k)] = int(v)
        except Exception:
            continue
    return out


# -----------------------------
# Scenario utilities (for completeness checks)
# -----------------------------
def _dotted_set(d: Dict[str, Any], dotted_key: str, value: Any) -> None:
    parts = dotted_key.split(".")
    cur: Dict[str, Any] = d
    for p in parts[:-1]:
        if p not in cur or not isinstance(cur[p], dict):
            cur[p] = {}
        cur = cur[p]  # type: ignore[assignment]
    cur[parts[-1]] = value


def build_scenario_cfg_basis(task: Dict[str, Any], scenario_settings: Dict[str, Any]) -> Dict[str, Any]:
    """
    Basis used for stable scenario_id:
      scenario_cfg_basis = deep copy of task.noise + sweep overrides applied.
    """
    noise = copy.deepcopy(task.get("noise", {}) or {})
    for k, v in (scenario_settings or {}).items():
        if k.startswith("noise."):
            _dotted_set(noise, k[len("noise."):], v)
        else:
            _dotted_set(noise, k, v)
    return noise


def canonicalize_scenario_id(task_id: str, scenario_cfg_basis: Dict[str, Any]) -> str:
    """
    Prefer bench.tasks.bench_generated.canonicalize_scenario_id if available.
    Fallback: sha1(json(payload))[:12].
    """
    try:
        from bench.tasks.bench_generated import canonicalize_scenario_id as _bg  # type: ignore
        return str(_bg(task_id, scenario_cfg_basis))
    except Exception:
        payload = {"task_id": task_id, "scenario": scenario_cfg_basis}
        s = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        return hashlib.sha1(s.encode("utf-8")).hexdigest()[:12]


def expand_sweep(sweep: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not sweep:
        return [{}]
    keys = sorted(sweep.keys())
    vals: List[List[Any]] = []
    for k in keys:
        v = sweep[k]
        vals.append(v if isinstance(v, list) else [v])

    out: List[Dict[str, Any]] = []

    def rec(i: int, cur: Dict[str, Any]) -> None:
        if i >= len(keys):
            out.append(dict(cur))
            return
        k = keys[i]
        for vv in vals[i]:
            cur[k] = vv
            rec(i + 1, cur)
        cur.pop(k, None)

    rec(0, {})
    return out


# -----------------------------
# run_dir scanning
# -----------------------------
def _read_config_snapshot(run_dir: Path) -> Optional[Dict[str, Any]]:
    p = run_dir / "config_snapshot.yaml"
    if not p.exists():
        return None
    try:
        return load_yaml(p)
    except Exception:
        return None


def _read_failure(run_dir: Path) -> Optional[Dict[str, Any]]:
    p = run_dir / "failure.json"
    if not p.exists():
        return None
    try:
        obj = load_json(p)
        # Backward compatibility for legacy artifacts.
        if "failure_type" not in obj and "category" in obj:
            obj["failure_type"] = obj.get("category")
        if "message" not in obj and "error" in obj:
            obj["message"] = obj.get("error")
        return obj
    except Exception:
        return None


def _read_metrics(run_dir: Path) -> Optional[Dict[str, Any]]:
    p = run_dir / "metrics.json"
    if not p.exists():
        return None
    try:
        return load_json(p)
    except Exception:
        return None


def _read_run_plan(run_dir: Path) -> Optional[Dict[str, Any]]:
    p = run_dir / "run_plan.json"
    if not p.exists():
        return None
    try:
        return load_json(p)
    except Exception:
        return None


def _read_budget_ledger(run_dir: Path) -> Optional[Dict[str, Any]]:
    p = run_dir / "budget_ledger.json"
    if not p.exists():
        return None
    try:
        return load_json(p)
    except Exception:
        return None


def _read_timing_csv_eval_s(run_dir: Path) -> Optional[float]:
    p = run_dir / "timing.csv"
    if not p.exists():
        return None
    try:
        total_ms = 0.0
        with p.open("r", encoding="utf-8") as f:
            r = csv.DictReader(f)
            for row in r:
                total_ms += float(row.get("ms_predict_whole_seq", "0"))
        return total_ms / 1000.0
    except Exception:
        return None


def _infer_keys_from_path(
    run_dir: Path,
) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str], Optional[int], Optional[str], Optional[str]]:
    """
    Best-effort inference for default template:
      runs/{suite}/{task}/{model}/{track}/seed_{seed}/scenario_{scenario}/init_{init_id}
    and legacy/basic variants without scenario/init segments.
    """
    parts = run_dir.parts
    try:
        i = parts.index("runs")
    except ValueError:
        return None, None, None, None, None, None, None

    suite = parts[i + 1] if i + 1 < len(parts) else None
    task = parts[i + 2] if i + 2 < len(parts) else None
    model = parts[i + 3] if i + 3 < len(parts) else None
    track = parts[i + 4] if i + 4 < len(parts) else None

    seed = None
    scenario = None
    init_id = None
    for p in parts[i + 1:]:
        if p.startswith("seed_"):
            seed = safe_int(p.replace("seed_", ""), default=0)
        if p.startswith("scenario_"):
            scenario = p.replace("scenario_", "")
        if p.startswith("init_"):
            init_id = p.replace("init_", "")

    return suite, task, scenario, model, seed, track, init_id


def _extract_r_scale(
    metrics_obj: Optional[Dict[str, Any]],
    cfg_obj: Optional[Dict[str, Any]],
) -> Optional[float]:
    candidates: List[Any] = []
    if metrics_obj:
        candidates.extend(
            [
                _dotted_get(metrics_obj.get("scenario_cfg_basis", {}), "shift.post_shift.R_scale"),
                _dotted_get(metrics_obj.get("scenario_settings", {}), "shift.post_shift.R_scale"),
                (metrics_obj.get("scenario_settings", {}) or {}).get("shift.post_shift.R_scale"),
            ]
        )
    if cfg_obj:
        candidates.extend(
            [
                _dotted_get(cfg_obj.get("scenario_cfg_basis", {}), "shift.post_shift.R_scale"),
                _dotted_get(cfg_obj.get("scenario_settings", {}), "shift.post_shift.R_scale"),
                (cfg_obj.get("scenario_settings", {}) or {}).get("shift.post_shift.R_scale"),
            ]
        )
    return _first_float(candidates)


def _extract_q2_r2(
    metrics_obj: Optional[Dict[str, Any]],
    cfg_obj: Optional[Dict[str, Any]],
) -> Tuple[Optional[float], Optional[float]]:
    q2_candidates: List[Any] = []
    r2_candidates: List[Any] = []
    if metrics_obj:
        q2_candidates.extend(
            [
                _dotted_get(metrics_obj.get("scenario_cfg_basis", {}), "Q.q2"),
                _dotted_get(metrics_obj.get("scenario_cfg_basis", {}), "pre_shift.Q.q2"),
                _dotted_get(metrics_obj.get("scenario_settings", {}), "noise.Q.q2"),
                _dotted_get(metrics_obj.get("scenario_settings", {}), "noise.pre_shift.Q.q2"),
                (metrics_obj.get("scenario_settings", {}) or {}).get("noise.Q.q2"),
                (metrics_obj.get("scenario_settings", {}) or {}).get("noise.pre_shift.Q.q2"),
            ]
        )
        r2_candidates.extend(
            [
                _dotted_get(metrics_obj.get("scenario_cfg_basis", {}), "R.r2"),
                _dotted_get(metrics_obj.get("scenario_cfg_basis", {}), "pre_shift.R.r2"),
                _dotted_get(metrics_obj.get("scenario_settings", {}), "noise.R.r2"),
                _dotted_get(metrics_obj.get("scenario_settings", {}), "noise.pre_shift.R.r2"),
                (metrics_obj.get("scenario_settings", {}) or {}).get("noise.R.r2"),
                (metrics_obj.get("scenario_settings", {}) or {}).get("noise.pre_shift.R.r2"),
            ]
        )
    if cfg_obj:
        q2_candidates.extend(
            [
                _dotted_get(cfg_obj.get("task", {}), "noise.Q.q2"),
                _dotted_get(cfg_obj.get("task", {}), "noise.pre_shift.Q.q2"),
                _dotted_get(cfg_obj.get("scenario_cfg_basis", {}), "Q.q2"),
                _dotted_get(cfg_obj.get("scenario_cfg_basis", {}), "pre_shift.Q.q2"),
            ]
        )
        r2_candidates.extend(
            [
                _dotted_get(cfg_obj.get("task", {}), "noise.R.r2"),
                _dotted_get(cfg_obj.get("task", {}), "noise.pre_shift.R.r2"),
                _dotted_get(cfg_obj.get("scenario_cfg_basis", {}), "R.r2"),
                _dotted_get(cfg_obj.get("scenario_cfg_basis", {}), "pre_shift.R.r2"),
            ]
        )
    return _first_float(q2_candidates), _first_float(r2_candidates)


def scan_runs(runs_root: Path, suite_name: Optional[str] = None) -> List[RunRecord]:
    """
    Scan run_dir folders by looking for config_snapshot.yaml / metrics.json / failure.json.
    """
    base = runs_root
    if suite_name:
        cand = runs_root / suite_name
        base = cand if cand.exists() else runs_root

    candidates: List[Path] = []
    for p in base.rglob("metrics.json"):
        candidates.append(p.parent)
    for p in base.rglob("failure.json"):
        candidates.append(p.parent)
    for p in base.rglob("config_snapshot.yaml"):
        candidates.append(p.parent)

    uniq = sorted(set(candidates))
    records: List[RunRecord] = []

    for rd in uniq:
        cfg = _read_config_snapshot(rd)
        met = _read_metrics(rd)
        fail = _read_failure(rd)
        run_plan_file = _read_run_plan(rd)
        ledger_file = _read_budget_ledger(rd)

        suite = None
        task_id = None
        scenario_id = None
        seed = None
        model_id = None
        track_id = None
        init_id = None

        if cfg:
            suite = (cfg.get("suite", {}) or {}).get("name") or cfg.get("suite", {}).get("suite", {}).get("name")
            task_id = (cfg.get("task", {}) or {}).get("task_id")
            model_id = (cfg.get("model", {}) or {}).get("model_id")
            track_id = cfg.get("track_id") or cfg.get("track")
            init_id = cfg.get("init_id")
            seed = cfg.get("seed")
            scenario_id = cfg.get("scenario_id") or cfg.get("scenario")

        met_run_plan = (met or {}).get("run_plan", {}) if met else {}
        met_budgets = (met or {}).get("budgets", {}) if met else {}
        run_plan = met_run_plan if isinstance(met_run_plan, dict) and met_run_plan else (run_plan_file or {})
        ledger = met_budgets if isinstance(met_budgets, dict) and met_budgets else (ledger_file or {})

        if isinstance(run_plan, dict):
            suite = suite or run_plan.get("suite_name") or run_plan.get("suite")
            task_id = task_id or run_plan.get("task_id")
            model_id = model_id or run_plan.get("model_id")
            track_id = track_id or run_plan.get("track_id") or run_plan.get("track")
            init_id = init_id or run_plan.get("init_id")
            seed = seed if seed is not None else run_plan.get("seed")
            scenario_id = scenario_id or run_plan.get("scenario_id")

        if met and (suite is None):
            suite = met.get("suite_name") or met.get("suite")
        if met and (task_id is None):
            task_id = met.get("task_id")
        if met and (model_id is None):
            model_id = met.get("model_id")
        if met and (track_id is None):
            track_id = met.get("track_id") or met.get("track")
        if met and (init_id is None):
            init_id = (met.get("run_plan", {}) or {}).get("init_id")
        if met and (seed is None):
            seed = met.get("seed")
        if met and (scenario_id is None):
            scenario_id = met.get("scenario_id")

        fail_ctx = (fail or {}).get("context", {}) if fail else {}
        if fail and (suite is None):
            suite = fail.get("suite_name") or fail.get("suite") or fail_ctx.get("suite_name")
        if fail and (task_id is None):
            task_id = fail.get("task_id") or fail_ctx.get("task_id")
        if fail and (model_id is None):
            model_id = fail.get("model_id") or fail_ctx.get("model_id")
        if fail and (track_id is None):
            track_id = fail.get("track_id") or fail.get("track") or fail_ctx.get("track_id")
        if fail and (init_id is None):
            init_id = fail.get("init_id") or fail_ctx.get("init_id")
        if fail and (seed is None):
            seed = fail.get("seed") or fail_ctx.get("seed")
        if fail and (scenario_id is None):
            scenario_id = fail.get("scenario_id") or fail_ctx.get("scenario_id")

        if suite is None or task_id is None or model_id is None or track_id is None or seed is None:
            suite2, task2, scen2, model2, seed2, track2, init2 = _infer_keys_from_path(rd)
            suite = suite or suite2
            task_id = task_id or task2
            scenario_id = scenario_id or scen2
            model_id = model_id or model2
            track_id = track_id or track2
            seed = seed if seed is not None else seed2
            init_id = init_id or init2

        suite = suite or "unknown"
        task_id = task_id or "unknown"
        model_id = model_id or "unknown"
        track_id = track_id or "unknown"
        init_id = init_id or "unknown"
        seed = safe_int(seed, 0)
        scenario_id = scenario_id or "na"

        status = "missing"
        if met:
            status = met.get("status", "ok") or "ok"
        elif fail:
            status = fail.get("status", "failed") or "failed"

        rec = RunRecord(
            suite=str(suite),
            task_id=str(task_id),
            scenario_id=str(scenario_id),
            seed=int(seed),
            model_id=str(model_id),
            init_id=str(init_id),
            track_id=str(track_id),
            status=str(status),
            run_dir=rd,
        )

        if met:
            acc = met.get("accuracy", {}) or {}
            tim = met.get("timing", {}) or {}
            nll = met.get("nll", {}) or {}
            recov = met.get("shift_recovery", {}) or {}
            dims = met.get("dims", {}) or {}

            rec.mse = safe_float(acc.get("mse"))
            rec.rmse = safe_float(acc.get("rmse"))
            rec.mse_db = safe_float(acc.get("mse_db"))
            rec.timing_ms_per_step = safe_float(tim.get("timing_ms_per_step"))
            rec.nll_value = safe_float(nll.get("value"))
            if isinstance(recov, dict):
                rec.recovery_k = safe_float(recov.get("recovery_k"))
            rec.x_dim = _first_int([dims.get("x_dim"), _dotted_get(cfg, "task.x_dim"), met.get("x_dim")])
            rec.y_dim = _first_int([dims.get("y_dim"), _dotted_get(cfg, "task.y_dim"), met.get("y_dim")])
            rec.T = _first_int([dims.get("T"), _dotted_get(cfg, "task.sequence_length_T"), met.get("T")])
            q2, r2 = _extract_q2_r2(met, cfg)
            rec.q2 = q2
            rec.r2 = r2
            if rec.r2 is not None and float(rec.r2) > 0.0:
                rec.inv_r2_db = float(-10.0 * math.log10(float(rec.r2)))
            rec.t0_used = _first_int(
                [
                    met.get("t0_used"),
                    recov.get("t0") if isinstance(recov, dict) else None,
                    _dotted_get(run_plan, "shift.t0"),
                    (ledger or {}).get("adapt_t0"),
                ]
            )
            rec.severity_r_scale = _extract_r_scale(met, cfg)

            rec.train_time_s = _first_float(
                [
                    tim.get("train_time_s"),
                    (ledger or {}).get("train_time_s"),
                    _dotted_get(run_plan, "timing.train_time_s"),
                ]
            )
            rec.eval_time_s = _first_float(
                [
                    tim.get("eval_time_s"),
                    (ledger or {}).get("eval_time_s"),
                    _dotted_get(run_plan, "timing.eval_time_s"),
                ]
            )
            rec.adapt_time_s = _first_float(
                [
                    tim.get("adapt_time_s"),
                    (ledger or {}).get("adapt_time_s"),
                    _dotted_get(run_plan, "timing.adapt_time_s"),
                ]
            )
            rec.total_time_s = _first_float(
                [
                    tim.get("total_time_s"),
                    (ledger or {}).get("total_time_s"),
                    _dotted_get(run_plan, "timing.total_time_s"),
                ]
            )

            # Fallback: derive eval wall-time from timing.csv (sum batch predict times).
            if rec.eval_time_s is None:
                rec.eval_time_s = _read_timing_csv_eval_s(rd)
            if rec.total_time_s is None:
                parts = [v for v in [rec.train_time_s, rec.eval_time_s, rec.adapt_time_s] if v is not None]
                if parts:
                    rec.total_time_s = float(sum(parts))

            rec.train_updates_used = _first_int([(ledger or {}).get("train_updates_used")])
            rec.adapt_updates_used = _first_int([(ledger or {}).get("adapt_updates_used")])
            rec.train_max_updates = _first_int(
                [
                    (ledger or {}).get("train_max_updates"),
                    _dotted_get(run_plan, "budgets.train_max_updates"),
                ]
            )
            rec.adapt_max_updates = _first_int(
                [
                    (ledger or {}).get("adapt_max_updates"),
                    _dotted_get(run_plan, "budgets.adapt_max_updates"),
                ]
            )
            rec.adapt_max_updates_per_step = _first_int(
                [
                    (ledger or {}).get("adapt_max_updates_per_step"),
                    _dotted_get(run_plan, "budgets.adapt_max_updates_per_step"),
                ]
            )
            per_step = _normalize_per_step_updates((ledger or {}).get("adapt_updates_per_step", {}))
            rec.adapt_updates_per_step_max = max(per_step.values()) if per_step else 0

            rec.cache_enabled = _first_non_none(
                [
                    safe_bool((ledger or {}).get("cache_enabled"), default=None),
                    safe_bool(_dotted_get(run_plan, "cache.enabled"), default=None),
                ]
            )
            rec.cache_hit = _first_non_none(
                [
                    safe_bool((ledger or {}).get("cache_hit"), default=None),
                    safe_bool(_dotted_get(run_plan, "cache.cache_hit"), default=None),
                ]
            )
            rec.train_skipped = _first_non_none(
                [
                    safe_bool((ledger or {}).get("train_skipped"), default=None),
                    safe_bool(_dotted_get(run_plan, "cache.train_skipped"), default=None),
                ]
            )
            ck = _first_non_none([(ledger or {}).get("cache_key"), _dotted_get(run_plan, "cache.cache_key")])
            rec.cache_key = None if ck in (None, "") else str(ck)

            # Dims fallback for missing t0/severity values.
            if rec.severity_r_scale is None:
                rec.severity_r_scale = _extract_r_scale(None, cfg)
            if rec.t0_used is None:
                rec.t0_used = _first_int(
                    [
                        _dotted_get(run_plan, "shift.t0"),
                        _dotted_get(cfg, "task.noise.shift.t0"),
                        _dotted_get(cfg, "scenario_cfg_basis.shift.t0"),
                    ]
                )
            _ = dims  # keep for future extensions without lint churn

        if fail:
            failure_type = fail.get("failure_type")
            if failure_type is not None:
                rec.failure_type = str(failure_type)
            stage = fail.get("failure_stage") or fail.get("phase")
            if stage not in (None, ""):
                rec.failure_stage = str(stage)
            rec.error = str(fail.get("message") or fail.get("error") or fail.get("hint") or "")

            # Carry cache/init/track context for failed runs when available.
            if rec.cache_enabled is None:
                rec.cache_enabled = safe_bool((ledger or {}).get("cache_enabled"), default=None)
            if rec.cache_hit is None:
                rec.cache_hit = safe_bool((ledger or {}).get("cache_hit"), default=None)
            if rec.train_skipped is None:
                rec.train_skipped = safe_bool((ledger or {}).get("train_skipped"), default=None)

        records.append(rec)

    if suite_name:
        records = [r for r in records if r.suite == suite_name]
    return records


# -----------------------------
# Completeness: expected runs from suite
# -----------------------------
def _enabled(obj: Dict[str, Any], default: bool = True) -> bool:
    if "enabled" not in obj:
        return default
    return bool(obj["enabled"])


def _parse_plan_token(token: str) -> Tuple[Optional[str], Optional[str]]:
    t = str(token).strip()
    if not t:
        return None, None
    if ":" in t:
        init_id, track_id = t.split(":", 1)
    elif "," in t:
        init_id, track_id = t.split(",", 1)
    else:
        return None, None
    return str(init_id).strip(), str(track_id).strip()


def expected_plan_from_suite(suite: Dict[str, Any]) -> List[Tuple[str, str, str, int, str, Optional[str], str, Dict[str, Any]]]:
    """
    Returns expected tuples:
      (suite_name, task_id, scenario_id, seed, model_id, init_id_or_none, track_id, scenario_settings)

    If suite YAML declares runner.plans, expected rows are plan-aware.
    Otherwise init_id is None (wildcard) and only track-level completeness is checked.
    """
    suite_name = (suite.get("suite", {}) or {}).get("name", "unknown")

    runner = suite.get("runner", {}) or {}
    enabled_policy = runner.get("enabled_policy", {}) or {}
    skip_if_disabled = bool(enabled_policy.get("skip_if_disabled", True))
    task_default = bool(enabled_policy.get("task_default", True))
    model_default = bool(enabled_policy.get("model_default", True))

    seeds = suite.get("seeds", []) or []
    raw_plans = runner.get("plans", []) or []
    plan_pairs: List[Tuple[str, str]] = []
    if isinstance(raw_plans, list):
        for p in raw_plans:
            init_id, track_id = _parse_plan_token(str(p))
            if init_id and track_id:
                plan_pairs.append((init_id, track_id))

    tracks = [t.get("track_id") for t in (runner.get("tracks", []) or []) if t.get("track_id")] or ["frozen"]

    tasks = suite.get("tasks", []) or []
    models = suite.get("models", []) or []

    plan: List[Tuple[str, str, str, int, str, Optional[str], str, Dict[str, Any]]] = []
    for task in tasks:
        if skip_if_disabled and not _enabled(task, task_default):
            continue
        task_id = task.get("task_id")
        if not task_id:
            continue
        scenario_list = expand_sweep(task.get("sweep"))
        for scen_settings in scenario_list:
            basis = build_scenario_cfg_basis(task, scen_settings)
            scenario_id = canonicalize_scenario_id(str(task_id), basis)
            for seed in seeds:
                for model in models:
                    if skip_if_disabled and not _enabled(model, model_default):
                        continue
                    model_id = model.get("model_id")
                    if not model_id:
                        continue
                    if plan_pairs:
                        for init_id, track_id in plan_pairs:
                            plan.append(
                                (
                                    suite_name,
                                    str(task_id),
                                    str(scenario_id),
                                    int(seed),
                                    str(model_id),
                                    str(init_id),
                                    str(track_id),
                                    scen_settings,
                                )
                            )
                    else:
                        for track_id in tracks:
                            plan.append(
                                (
                                    suite_name,
                                    str(task_id),
                                    str(scenario_id),
                                    int(seed),
                                    str(model_id),
                                    None,  # wildcard init axis
                                    str(track_id),
                                    scen_settings,
                                )
                            )
    return plan


def merge_records_with_expected(
    scanned: List[RunRecord],
    expected: List[Tuple[str, str, str, int, str, Optional[str], str, Dict[str, Any]]],
) -> List[RunRecord]:
    """
    Ensures 'missing' rows exist for expected combinations not present in scanned results.
    """
    idx_exact: Dict[Tuple[str, str, str, int, str, str, str], RunRecord] = {}
    idx_wild: Dict[Tuple[str, str, str, int, str, str], bool] = {}
    for r in scanned:
        k_exact = (r.suite, r.task_id, r.scenario_id, r.seed, r.model_id, r.init_id, r.track_id)
        idx_exact[k_exact] = r
        k_wild = (r.suite, r.task_id, r.scenario_id, r.seed, r.model_id, r.track_id)
        idx_wild[k_wild] = True

    out = list(scanned)
    for (suite, task_id, scenario_id, seed, model_id, init_id, track_id, _scen_settings) in expected:
        if init_id is None:
            present = idx_wild.get((suite, task_id, scenario_id, seed, model_id, track_id), False)
        else:
            present = (suite, task_id, scenario_id, seed, model_id, init_id, track_id) in idx_exact
        if present:
            continue
        out.append(
            RunRecord(
                suite=suite,
                task_id=task_id,
                scenario_id=scenario_id,
                seed=seed,
                model_id=model_id,
                init_id=(str(init_id) if init_id else "unknown"),
                track_id=track_id,
                status="missing",
                run_dir=Path(""),
            )
        )

    out.sort(
        key=lambda r: (
            r.suite,
            r.task_id,
            r.model_id,
            r.init_id,
            r.track_id,
            r.scenario_id,
            r.seed,
        )
    )
    return out


# -----------------------------
# CSV outputs
# -----------------------------
SUMMARY_FIELDS = [
    "suite",
    "task_id",
    "scenario_id",
    "model_id",
    "seed",
    "init_id",
    "track_id",
    "track",
    "status",
    "failure_type",
    "failure_stage",
    "mse",
    "mse_db",
    "rmse",
    "x_dim",
    "y_dim",
    "T",
    "q2",
    "r2",
    "inv_r2_db",
    "nll",
    "recovery_k",
    "t0",
    "severity_r_scale",
    "timing_ms_per_step",
    "train_time_s",
    "eval_time_s",
    "adapt_time_s",
    "total_time_s",
    "train_updates_used",
    "adapt_updates_used",
    "adapt_updates_per_step_max",
    "train_max_updates",
    "adapt_max_updates",
    "adapt_max_updates_per_step",
    "cache_enabled",
    "cache_hit",
    "train_skipped",
    "cache_key",
    "run_dir",
    "error",
]


def write_summary_csv(records: List[RunRecord], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        w.writeheader()
        for r in records:
            w.writerow(
                {
                    "suite": r.suite,
                    "task_id": r.task_id,
                    "scenario_id": r.scenario_id,
                    "model_id": r.model_id,
                    "seed": r.seed,
                    "init_id": r.init_id,
                    "track_id": r.track_id,
                    "track": r.track_id,  # backward-compatible alias
                    "status": r.status,
                    "failure_type": "" if r.failure_type is None else r.failure_type,
                    "failure_stage": "" if r.failure_stage is None else r.failure_stage,
                    "mse": "" if r.mse is None else r.mse,
                    "mse_db": "" if r.mse_db is None else r.mse_db,
                    "rmse": "" if r.rmse is None else r.rmse,
                    "x_dim": "" if r.x_dim is None else r.x_dim,
                    "y_dim": "" if r.y_dim is None else r.y_dim,
                    "T": "" if r.T is None else r.T,
                    "q2": "" if r.q2 is None else r.q2,
                    "r2": "" if r.r2 is None else r.r2,
                    "inv_r2_db": "" if r.inv_r2_db is None else r.inv_r2_db,
                    "nll": "" if r.nll_value is None else r.nll_value,
                    "recovery_k": "" if r.recovery_k is None else r.recovery_k,
                    "t0": "" if r.t0_used is None else r.t0_used,
                    "severity_r_scale": "" if r.severity_r_scale is None else r.severity_r_scale,
                    "timing_ms_per_step": "" if r.timing_ms_per_step is None else r.timing_ms_per_step,
                    "train_time_s": "" if r.train_time_s is None else r.train_time_s,
                    "eval_time_s": "" if r.eval_time_s is None else r.eval_time_s,
                    "adapt_time_s": "" if r.adapt_time_s is None else r.adapt_time_s,
                    "total_time_s": "" if r.total_time_s is None else r.total_time_s,
                    "train_updates_used": "" if r.train_updates_used is None else r.train_updates_used,
                    "adapt_updates_used": "" if r.adapt_updates_used is None else r.adapt_updates_used,
                    "adapt_updates_per_step_max": (
                        "" if r.adapt_updates_per_step_max is None else r.adapt_updates_per_step_max
                    ),
                    "train_max_updates": "" if r.train_max_updates is None else r.train_max_updates,
                    "adapt_max_updates": "" if r.adapt_max_updates is None else r.adapt_max_updates,
                    "adapt_max_updates_per_step": (
                        "" if r.adapt_max_updates_per_step is None else r.adapt_max_updates_per_step
                    ),
                    "cache_enabled": "" if r.cache_enabled is None else bool(r.cache_enabled),
                    "cache_hit": "" if r.cache_hit is None else bool(r.cache_hit),
                    "train_skipped": "" if r.train_skipped is None else bool(r.train_skipped),
                    "cache_key": "" if r.cache_key is None else r.cache_key,
                    "run_dir": "" if str(r.run_dir) == "." else str(r.run_dir),
                    "error": "" if r.error is None else r.error,
                }
            )


# -----------------------------
# Aggregation statistics
# -----------------------------
def _mean(xs: List[float]) -> float:
    return sum(xs) / float(len(xs))


def _std(xs: List[float]) -> float:
    if len(xs) < 2:
        return 0.0
    mu = _mean(xs)
    return math.sqrt(sum((x - mu) ** 2 for x in xs) / float(len(xs) - 1))


def _sem(xs: List[float]) -> float:
    if len(xs) == 0:
        return float("nan")
    return _std(xs) / math.sqrt(float(len(xs)))


def _ci95(xs: List[float]) -> float:
    return 1.96 * _sem(xs)


def _median(xs: List[float]) -> float:
    if not xs:
        return float("nan")
    ys = sorted(xs)
    n = len(ys)
    mid = n // 2
    if n % 2 == 1:
        return ys[mid]
    return (ys[mid - 1] + ys[mid]) / 2.0


AGG_FIELDS = [
    "suite",
    "task_id",
    "scenario_id",
    "model_id",
    "init_id",
    "track_id",
    "track",
    "n_success",
    "n_total",
    "fail_count",
    "fail_rate",
    "failure_type",
    "x_dim",
    "y_dim",
    "T",
    "q2",
    "r2",
    "inv_r2_db",
    "mse_mean",
    "mse_std",
    "mse_sem",
    "mse_ci95",
    "mse_db_mean",
    "mse_db_std",
    "mse_db_sem",
    "mse_db_ci95",
    "rmse_mean",
    "rmse_std",
    "rmse_sem",
    "rmse_ci95",
    "timing_ms_per_step_mean",
    "timing_ms_per_step_std",
    "timing_ms_per_step_sem",
    "timing_ms_per_step_ci95",
    "recovery_k_mean",
    "recovery_k_std",
    "recovery_k_sem",
    "recovery_k_ci95",
    "nll_mean",
    "nll_std",
    "nll_sem",
    "nll_ci95",
    "train_time_s_mean",
    "train_time_s_median",
    "eval_time_s_mean",
    "eval_time_s_median",
    "adapt_time_s_mean",
    "adapt_time_s_median",
    "total_time_s_mean",
    "total_time_s_median",
    "train_updates_used_mean",
    "train_updates_used_median",
    "adapt_updates_used_mean",
    "adapt_updates_used_median",
    "adapt_updates_per_step_max_mean",
    "adapt_updates_per_step_max_median",
    "cache_enabled_rate",
    "cache_hit_rate",
    "train_skipped_rate",
    "severity_r_scale_mean",
    "t0_mean",
]


def aggregate_by_seed(records: List[RunRecord]) -> List[Dict[str, Any]]:
    """
    Group by (suite, task_id, scenario_id, model_id, init_id, track_id), aggregate across seeds.
    """
    groups: Dict[Tuple[str, str, str, str, str, str], List[RunRecord]] = {}
    for r in records:
        k = (r.suite, r.task_id, r.scenario_id, r.model_id, r.init_id, r.track_id)
        groups.setdefault(k, []).append(r)

    out: List[Dict[str, Any]] = []
    for (suite, task_id, scenario_id, model_id, init_id, track_id), rs in sorted(groups.items()):
        n_total = len(rs)
        fails = [r for r in rs if r.status != "ok"]
        n_fail = len(fails)
        n_success = n_total - n_fail
        fail_types = sorted({str(r.failure_type) for r in fails if r.failure_type})
        if len(fail_types) == 0:
            failure_type = ""
        elif len(fail_types) == 1:
            failure_type = fail_types[0]
        else:
            failure_type = "|".join(fail_types)

        def collect_ok(getter) -> List[float]:
            xs: List[float] = []
            for r in rs:
                if r.status != "ok":
                    continue
                v = getter(r)
                if v is None:
                    continue
                xs.append(float(v))
            return xs

        def collect_any(getter) -> List[float]:
            xs: List[float] = []
            for r in rs:
                if r.status == "missing":
                    continue
                v = getter(r)
                if v is None:
                    continue
                xs.append(float(v))
            return xs

        mse = collect_ok(lambda r: r.mse)
        mse_db = collect_ok(lambda r: r.mse_db)
        rmse = collect_ok(lambda r: r.rmse)
        timing = collect_ok(lambda r: r.timing_ms_per_step)
        recovery = collect_ok(lambda r: r.recovery_k)
        nll = collect_ok(lambda r: r.nll_value)

        train_time = collect_any(lambda r: r.train_time_s)
        eval_time = collect_any(lambda r: r.eval_time_s)
        adapt_time = collect_any(lambda r: r.adapt_time_s)
        total_time = collect_any(lambda r: r.total_time_s)

        train_updates = collect_any(lambda r: r.train_updates_used)
        adapt_updates = collect_any(lambda r: r.adapt_updates_used)
        adapt_per_step = collect_any(lambda r: r.adapt_updates_per_step_max)
        sev = collect_any(lambda r: r.severity_r_scale)
        t0_vals = collect_any(lambda r: r.t0_used)
        xdim_vals = collect_any(lambda r: r.x_dim)
        ydim_vals = collect_any(lambda r: r.y_dim)
        t_vals = collect_any(lambda r: r.T)
        q2_vals = collect_any(lambda r: r.q2)
        r2_vals = collect_any(lambda r: r.r2)
        inv_r2_vals = collect_any(lambda r: r.inv_r2_db)

        def _stable_scalar(xs: List[float], cast_int: bool = False) -> Any:
            if not xs:
                return ""
            x0 = float(xs[0])
            if all(abs(float(x) - x0) <= 1e-12 for x in xs):
                return int(round(x0)) if cast_int else x0
            x_mean = _mean(xs)
            return int(round(x_mean)) if cast_int else x_mean

        def pack_stats(prefix: str, xs: List[float]) -> Dict[str, Any]:
            if len(xs) == 0:
                return {
                    f"{prefix}_mean": "",
                    f"{prefix}_std": "",
                    f"{prefix}_sem": "",
                    f"{prefix}_ci95": "",
                }
            return {
                f"{prefix}_mean": _mean(xs),
                f"{prefix}_std": _std(xs),
                f"{prefix}_sem": _sem(xs),
                f"{prefix}_ci95": _ci95(xs),
            }

        def pack_mean_median(prefix: str, xs: List[float]) -> Dict[str, Any]:
            if len(xs) == 0:
                return {f"{prefix}_mean": "", f"{prefix}_median": ""}
            return {f"{prefix}_mean": _mean(xs), f"{prefix}_median": _median(xs)}

        cache_enabled_true = sum(1 for r in rs if r.cache_enabled is True)
        cache_enabled_den = len([r for r in rs if r.cache_enabled is not None])
        cache_enabled_rate = (
            (float(cache_enabled_true) / float(cache_enabled_den))
            if cache_enabled_den > 0
            else ""
        )
        cache_hit_true = sum(1 for r in rs if r.cache_hit is True)
        train_skipped_true = sum(1 for r in rs if r.train_skipped is True)
        cache_hit_rate = (
            (float(cache_hit_true) / float(cache_enabled_true))
            if cache_enabled_true > 0
            else ""
        )
        train_skipped_rate = (
            (float(train_skipped_true) / float(cache_enabled_true))
            if cache_enabled_true > 0
            else ""
        )

        row: Dict[str, Any] = {
            "suite": suite,
            "task_id": task_id,
            "scenario_id": scenario_id,
            "model_id": model_id,
            "init_id": init_id,
            "track_id": track_id,
            "track": track_id,  # backward-compatible alias
            "n_success": n_success,
            "n_total": n_total,
            "fail_count": n_fail,
            "fail_rate": (float(n_fail) / float(n_total)) if n_total > 0 else 0.0,
            "failure_type": failure_type,
            "x_dim": _stable_scalar(xdim_vals, cast_int=True),
            "y_dim": _stable_scalar(ydim_vals, cast_int=True),
            "T": _stable_scalar(t_vals, cast_int=True),
            "q2": _stable_scalar(q2_vals, cast_int=False),
            "r2": _stable_scalar(r2_vals, cast_int=False),
            "inv_r2_db": _stable_scalar(inv_r2_vals, cast_int=False),
            "cache_enabled_rate": cache_enabled_rate,
            "cache_hit_rate": cache_hit_rate,
            "train_skipped_rate": train_skipped_rate,
            "severity_r_scale_mean": (_mean(sev) if sev else ""),
            "t0_mean": (_mean(t0_vals) if t0_vals else ""),
        }
        row.update(pack_stats("mse", mse))
        row.update(pack_stats("mse_db", mse_db))
        row.update(pack_stats("rmse", rmse))
        row.update(pack_stats("timing_ms_per_step", timing))
        row.update(pack_stats("recovery_k", recovery))
        row.update(pack_stats("nll", nll))
        row.update(pack_mean_median("train_time_s", train_time))
        row.update(pack_mean_median("eval_time_s", eval_time))
        row.update(pack_mean_median("adapt_time_s", adapt_time))
        row.update(pack_mean_median("total_time_s", total_time))
        row.update(pack_mean_median("train_updates_used", train_updates))
        row.update(pack_mean_median("adapt_updates_used", adapt_updates))
        row.update(pack_mean_median("adapt_updates_per_step_max", adapt_per_step))

        out.append(row)

    return out


def write_aggregate_csv(rows: List[Dict[str, Any]], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=AGG_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in AGG_FIELDS})


def write_rows_csv(rows: List[Dict[str, Any]], out_csv: Path, fieldnames: List[str]) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fieldnames})


# -----------------------------
# S6 views: plan/failure/ops tables
# -----------------------------
def _plan_sort_key(plan: Tuple[str, str]) -> Tuple[int, int, str, str]:
    init_order = {"untrained": 0, "pretrained": 1, "trained": 2}
    track_order = {"frozen": 0, "budgeted": 1}
    init_id, track_id = plan
    return (
        init_order.get(init_id, 99),
        track_order.get(track_id, 99),
        init_id,
        track_id,
    )


def build_plan_comparison_rows(records: List[RunRecord]) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    Wide comparison table:
      key=(suite, task_id, scenario_id, seed, model_id)
      columns per observed plan (init_id, track_id).
    """
    active = [r for r in records if r.status != "missing"]
    plans = sorted({(r.init_id, r.track_id) for r in active}, key=_plan_sort_key)
    plan_ids = [f"{init_id}__{track_id}" for init_id, track_id in plans]

    groups: Dict[Tuple[str, str, str, int, str], List[RunRecord]] = {}
    for r in active:
        k = (r.suite, r.task_id, r.scenario_id, r.seed, r.model_id)
        groups.setdefault(k, []).append(r)

    rows: List[Dict[str, Any]] = []
    base_fields = ["suite", "task_id", "scenario_id", "seed", "model_id"]
    suffixes = [
        "status",
        "failure_type",
        "failure_stage",
        "mse_db",
        "recovery_k",
        "train_updates_used",
        "adapt_updates_used",
        "total_time_s",
        "cache_hit",
    ]
    fields = list(base_fields)
    for pid in plan_ids:
        for sfx in suffixes:
            fields.append(f"{sfx}__{pid}")
    fields.extend(
        [
            "delta_mse_db_trained_budgeted_minus_frozen",
            "delta_recovery_k_trained_budgeted_minus_frozen",
        ]
    )

    for (suite, task_id, scenario_id, seed, model_id), rs in sorted(groups.items()):
        row: Dict[str, Any] = {
            "suite": suite,
            "task_id": task_id,
            "scenario_id": scenario_id,
            "seed": seed,
            "model_id": model_id,
        }
        by_plan: Dict[Tuple[str, str], RunRecord] = {(r.init_id, r.track_id): r for r in rs}
        for (init_id, track_id), pid in zip(plans, plan_ids):
            rr = by_plan.get((init_id, track_id))
            if rr is None:
                continue
            row[f"status__{pid}"] = rr.status
            row[f"failure_type__{pid}"] = "" if rr.failure_type is None else rr.failure_type
            row[f"failure_stage__{pid}"] = "" if rr.failure_stage is None else rr.failure_stage
            row[f"mse_db__{pid}"] = "" if rr.mse_db is None else rr.mse_db
            row[f"recovery_k__{pid}"] = "" if rr.recovery_k is None else rr.recovery_k
            row[f"train_updates_used__{pid}"] = "" if rr.train_updates_used is None else rr.train_updates_used
            row[f"adapt_updates_used__{pid}"] = "" if rr.adapt_updates_used is None else rr.adapt_updates_used
            row[f"total_time_s__{pid}"] = "" if rr.total_time_s is None else rr.total_time_s
            row[f"cache_hit__{pid}"] = "" if rr.cache_hit is None else rr.cache_hit

        tf = by_plan.get(("trained", "frozen"))
        tb = by_plan.get(("trained", "budgeted"))
        if tf and tb and tf.mse_db is not None and tb.mse_db is not None:
            row["delta_mse_db_trained_budgeted_minus_frozen"] = float(tb.mse_db) - float(tf.mse_db)
        else:
            row["delta_mse_db_trained_budgeted_minus_frozen"] = ""
        if tf and tb and tf.recovery_k is not None and tb.recovery_k is not None:
            row["delta_recovery_k_trained_budgeted_minus_frozen"] = float(tb.recovery_k) - float(tf.recovery_k)
        else:
            row["delta_recovery_k_trained_budgeted_minus_frozen"] = ""

        rows.append(row)

    return rows, fields


def _parse_group_by(group_by: str) -> List[str]:
    out: List[str] = []
    for p in str(group_by or "").split(","):
        x = p.strip()
        if x:
            out.append(x)
    return out


def summarize_failures_by_plan(records: List[RunRecord], group_by: str = "init_id,track_id") -> Tuple[List[Dict[str, Any]], List[str]]:
    records = [r for r in records if r.status != "missing"]
    dims = ["suite", "task_id", "model_id"]
    for gb in _parse_group_by(group_by):
        if gb not in dims:
            dims.append(gb)
    if "init_id" not in dims:
        dims.append("init_id")
    if "track_id" not in dims:
        dims.append("track_id")

    grouped: Dict[Tuple[Any, ...], List[RunRecord]] = {}
    for r in records:
        k = tuple(getattr(r, d) if hasattr(r, d) else None for d in dims)
        grouped.setdefault(k, []).append(r)

    fields = dims + ["failure_type", "fail_count", "n_total", "fail_rate"]
    rows: List[Dict[str, Any]] = []
    for key, rs in sorted(grouped.items()):
        n_total = len(rs)
        fail_rs = [r for r in rs if r.status != "ok"]
        if not fail_rs:
            row = {d: v for d, v in zip(dims, key)}
            row.update({"failure_type": "", "fail_count": 0, "n_total": n_total, "fail_rate": 0.0})
            rows.append(row)
            continue

        counts: Dict[str, int] = {}
        for r in fail_rs:
            ft = r.failure_type or "unknown"
            counts[ft] = counts.get(ft, 0) + 1
        for ft, cnt in sorted(counts.items()):
            row = {d: v for d, v in zip(dims, key)}
            row.update(
                {
                    "failure_type": ft,
                    "fail_count": cnt,
                    "n_total": n_total,
                    "fail_rate": float(cnt) / float(n_total) if n_total > 0 else 0.0,
                }
            )
            rows.append(row)

    return rows, fields


def summarize_ops_by_plan(records: List[RunRecord], group_by: str = "init_id,track_id") -> Tuple[List[Dict[str, Any]], List[str]]:
    records = [r for r in records if r.status != "missing"]
    dims = ["suite", "task_id", "model_id"]
    for gb in _parse_group_by(group_by):
        if gb not in dims:
            dims.append(gb)
    if "init_id" not in dims:
        dims.append("init_id")
    if "track_id" not in dims:
        dims.append("track_id")

    grouped: Dict[Tuple[Any, ...], List[RunRecord]] = {}
    for r in records:
        k = tuple(getattr(r, d) if hasattr(r, d) else None for d in dims)
        grouped.setdefault(k, []).append(r)

    fields = dims + [
        "n_total",
        "n_success",
        "train_time_s_mean",
        "train_time_s_median",
        "eval_time_s_mean",
        "eval_time_s_median",
        "adapt_time_s_mean",
        "adapt_time_s_median",
        "total_time_s_mean",
        "total_time_s_median",
        "train_updates_used_mean",
        "train_updates_used_median",
        "adapt_updates_used_mean",
        "adapt_updates_used_median",
        "adapt_updates_per_step_max_mean",
        "adapt_updates_per_step_max_median",
        "cache_enabled_rate",
        "cache_hit_rate",
        "train_skipped_rate",
    ]

    def _vals(rs: List[RunRecord], getter) -> List[float]:
        xs: List[float] = []
        for r in rs:
            if r.status != "ok":
                continue
            v = getter(r)
            if v is None:
                continue
            xs.append(float(v))
        return xs

    def _pack(prefix: str, xs: List[float]) -> Dict[str, Any]:
        if not xs:
            return {f"{prefix}_mean": "", f"{prefix}_median": ""}
        return {f"{prefix}_mean": _mean(xs), f"{prefix}_median": _median(xs)}

    rows: List[Dict[str, Any]] = []
    for key, rs in sorted(grouped.items()):
        ok_rs = [r for r in rs if r.status == "ok"]
        n_total = len(rs)
        n_success = len(ok_rs)
        enabled_true = sum(1 for r in rs if r.cache_enabled is True)
        enabled_den = len([r for r in rs if r.cache_enabled is not None])
        hit_true = sum(1 for r in rs if r.cache_hit is True)
        skipped_true = sum(1 for r in rs if r.train_skipped is True)

        row: Dict[str, Any] = {d: v for d, v in zip(dims, key)}
        row["n_total"] = n_total
        row["n_success"] = n_success
        row.update(_pack("train_time_s", _vals(rs, lambda r: r.train_time_s)))
        row.update(_pack("eval_time_s", _vals(rs, lambda r: r.eval_time_s)))
        row.update(_pack("adapt_time_s", _vals(rs, lambda r: r.adapt_time_s)))
        row.update(_pack("total_time_s", _vals(rs, lambda r: r.total_time_s)))
        row.update(_pack("train_updates_used", _vals(rs, lambda r: r.train_updates_used)))
        row.update(_pack("adapt_updates_used", _vals(rs, lambda r: r.adapt_updates_used)))
        row.update(_pack("adapt_updates_per_step_max", _vals(rs, lambda r: r.adapt_updates_per_step_max)))
        row["cache_enabled_rate"] = (
            (float(enabled_true) / float(enabled_den)) if enabled_den > 0 else ""
        )
        row["cache_hit_rate"] = (
            (float(hit_true) / float(enabled_true)) if enabled_true > 0 else ""
        )
        row["train_skipped_rate"] = (
            (float(skipped_true) / float(enabled_true)) if enabled_true > 0 else ""
        )
        rows.append(row)

    return rows, fields


def aggregate_to_latex_tabular(rows: List[Dict[str, Any]], caption: str = "", label: str = "") -> str:
    """
    Simple LaTeX exporter for aggregate table.
    """
    cols = [
        ("model_id", "Model"),
        ("task_id", "Task"),
        ("scenario_id", "Scenario"),
        ("init_id", "Init"),
        ("track_id", "Track"),
        ("mse_mean", "MSE"),
        ("rmse_mean", "RMSE"),
        ("mse_db_mean", "MSE(dB)"),
        ("eval_time_s_mean", "eval(s)"),
        ("recovery_k_mean", "recovery_k"),
        ("fail_rate", "fail_rate"),
    ]

    def fmt(v: Any) -> str:
        if v is None or v == "":
            return "--"
        try:
            x = float(v)
            if abs(x) >= 1000:
                return f"{x:.1f}"
            if abs(x) >= 1:
                return f"{x:.4f}"
            return f"{x:.6f}"
        except Exception:
            return str(v)

    lines: List[str] = []
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    if caption:
        lines.append(f"\\caption{{{caption}}}")
    if label:
        lines.append(f"\\label{{{label}}}")

    lines.append("\\begin{tabular}{" + "l" * len(cols) + "}")
    lines.append("\\hline")
    lines.append(" & ".join(h for _, h in cols) + " \\\\")
    lines.append("\\hline")
    for r in rows:
        lines.append(" & ".join(fmt(r.get(k, "")) for k, _ in cols) + " \\\\")
    lines.append("\\hline")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    return "\n".join(lines) + "\n"
