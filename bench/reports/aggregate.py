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

    # extra
    t0_used: Optional[int] = None
    error: Optional[str] = None


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
    # cartesian product
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
        return load_json(p)
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


def _infer_keys_from_path(run_dir: Path) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str], Optional[int], Optional[str]]:
    """
    Best-effort inference for default template:
      runs/{suite}/{task}/{model}/{track}/seed_{seed}/scenario_{scenario}
    and basic template:
      runs/{suite}/{task}/{model}/{track}/seed_{seed}
    """
    parts = run_dir.parts
    # find ".../runs/<suite>/..."
    try:
        i = parts.index("runs")
    except ValueError:
        return None, None, None, None, None, None

    suite = parts[i + 1] if i + 1 < len(parts) else None
    task = parts[i + 2] if i + 2 < len(parts) else None
    model = parts[i + 3] if i + 3 < len(parts) else None
    track = parts[i + 4] if i + 4 < len(parts) else None

    seed = None
    scenario = None
    for p in parts[i + 1:]:
        if p.startswith("seed_"):
            seed = safe_int(p.replace("seed_", ""), default=0)
        if p.startswith("scenario_"):
            scenario = p.replace("scenario_", "")

    return suite, task, scenario, model, seed, track


def scan_runs(runs_root: Path, suite_name: Optional[str] = None) -> List[RunRecord]:
    """
    Scan run_dir folders by looking for config_snapshot.yaml / metrics.json / failure.json.
    """
    base = runs_root
    if suite_name:
        cand = runs_root / suite_name
        base = cand if cand.exists() else runs_root

    candidates: List[Path] = []
    # include dirs that contain metrics.json OR failure.json OR config_snapshot.yaml
    for p in base.rglob("metrics.json"):
        candidates.append(p.parent)
    for p in base.rglob("failure.json"):
        candidates.append(p.parent)
    for p in base.rglob("config_snapshot.yaml"):
        candidates.append(p.parent)

    # dedupe
    uniq = sorted(set(candidates))

    records: List[RunRecord] = []
    for rd in uniq:
        cfg = _read_config_snapshot(rd)
        met = _read_metrics(rd)
        fail = _read_failure(rd)

        suite = None
        task_id = None
        scenario_id = None
        seed = None
        model_id = None
        track_id = None

        if cfg:
            suite = (cfg.get("suite", {}) or {}).get("name") or cfg.get("suite", {}).get("suite", {}).get("name")
            task_id = (cfg.get("task", {}) or {}).get("task_id")
            model_id = (cfg.get("model", {}) or {}).get("model_id")
            track_id = cfg.get("track_id") or cfg.get("track", cfg.get("track_id"))
            seed = cfg.get("seed")
            scenario_id = cfg.get("scenario_id") or cfg.get("scenario", cfg.get("scenario_id"))

        if met and (suite is None):
            suite = met.get("suite_name") or met.get("suite")
        if met and (task_id is None):
            task_id = met.get("task_id")
        if met and (model_id is None):
            model_id = met.get("model_id")
        if met and (track_id is None):
            track_id = met.get("track_id") or met.get("track")
        if met and (seed is None):
            seed = met.get("seed")
        if met and (scenario_id is None):
            scenario_id = met.get("scenario_id")

        if fail and (suite is None):
            suite = fail.get("suite_name") or fail.get("suite")
        if fail and (task_id is None):
            task_id = fail.get("task_id")
        if fail and (model_id is None):
            model_id = fail.get("model_id")
        if fail and (track_id is None):
            track_id = fail.get("track_id") or fail.get("track")
        if fail and (seed is None):
            seed = fail.get("seed")
        if fail and (scenario_id is None):
            scenario_id = fail.get("scenario_id")

        if suite is None or task_id is None or model_id is None or track_id is None or seed is None:
            suite2, task2, scen2, model2, seed2, track2 = _infer_keys_from_path(rd)
            suite = suite or suite2
            task_id = task_id or task2
            scenario_id = scenario_id or scen2
            model_id = model_id or model2
            track_id = track_id or track2
            seed = seed if seed is not None else seed2

        # final defaults
        suite = suite or "unknown"
        task_id = task_id or "unknown"
        model_id = model_id or "unknown"
        track_id = track_id or "unknown"
        seed = safe_int(seed, 0)
        scenario_id = scenario_id or "na"

        # determine status + metrics
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
            track_id=str(track_id),
            status=str(status),
            run_dir=rd,
        )

        if met:
            acc = met.get("accuracy", {}) or {}
            tim = met.get("timing", {}) or {}
            nll = met.get("nll", {}) or {}
            recov = met.get("shift_recovery", {}) or {}

            rec.mse = safe_float(acc.get("mse"))
            rec.rmse = safe_float(acc.get("rmse"))
            rec.mse_db = safe_float(acc.get("mse_db"))
            rec.timing_ms_per_step = safe_float(tim.get("timing_ms_per_step"))
            rec.nll_value = safe_float(nll.get("value"))
            # recovery_k may be missing for non-shift tasks
            if isinstance(recov, dict):
                rec.recovery_k = safe_float(recov.get("recovery_k"))
            rec.t0_used = safe_int(met.get("t0_used"), default=0) if met.get("t0_used") is not None else None

        if fail:
            rec.error = str(fail.get("error") or fail.get("hint") or "")

        records.append(rec)

    # filter by suite if requested
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


def expected_plan_from_suite(suite: Dict[str, Any]) -> List[Tuple[str, str, str, int, str, str, Dict[str, Any]]]:
    """
    Returns expected tuples:
      (suite_name, task_id, scenario_id, seed, model_id, track_id, scenario_settings)
    Includes enabled_policy D11: skip disabled tasks/models if skip_if_disabled==true.
    """
    suite_name = (suite.get("suite", {}) or {}).get("name", "unknown")

    runner = suite.get("runner", {}) or {}
    enabled_policy = runner.get("enabled_policy", {}) or {}
    skip_if_disabled = bool(enabled_policy.get("skip_if_disabled", True))
    task_default = bool(enabled_policy.get("task_default", True))
    model_default = bool(enabled_policy.get("model_default", True))

    seeds = suite.get("seeds", []) or []
    tracks = [t.get("track_id") for t in (runner.get("tracks", []) or [])] or ["frozen"]

    tasks = suite.get("tasks", []) or []
    models = suite.get("models", []) or []

    plan: List[Tuple[str, str, str, int, str, str, Dict[str, Any]]] = []
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
                    for track_id in tracks:
                        plan.append((suite_name, str(task_id), str(scenario_id), int(seed), str(model_id), str(track_id), scen_settings))
    return plan


def merge_records_with_expected(
    scanned: List[RunRecord],
    expected: List[Tuple[str, str, str, int, str, str, Dict[str, Any]]],
) -> List[RunRecord]:
    """
    Ensures 'missing' rows exist for expected combinations not present in scanned results.
    """
    idx: Dict[Tuple[str, str, str, int, str, str], RunRecord] = {}
    for r in scanned:
        k = (r.suite, r.task_id, r.scenario_id, r.seed, r.model_id, r.track_id)
        idx[k] = r

    out = list(scanned)
    for (suite, task_id, scenario_id, seed, model_id, track_id, _scen_settings) in expected:
        k = (suite, task_id, scenario_id, seed, model_id, track_id)
        if k not in idx:
            out.append(
                RunRecord(
                    suite=suite,
                    task_id=task_id,
                    scenario_id=scenario_id,
                    seed=seed,
                    model_id=model_id,
                    track_id=track_id,
                    status="missing",
                    run_dir=Path(""),
                )
            )

    # stable sort
    out.sort(key=lambda r: (r.suite, r.task_id, r.model_id, r.track_id, r.scenario_id, r.seed))
    return out


# -----------------------------
# CSV outputs
# -----------------------------
SUMMARY_FIELDS = [
    "suite",
    "task_id",
    "scenario_id",
    "model_id",
    "track",
    "seed",
    "status",
    "mse",
    "mse_db",
    "rmse",
    "timing_ms_per_step",
    "recovery_k",
    "nll",
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
                    "track": r.track_id,
                    "seed": r.seed,
                    "status": r.status,
                    "mse": "" if r.mse is None else r.mse,
                    "mse_db": "" if r.mse_db is None else r.mse_db,
                    "rmse": "" if r.rmse is None else r.rmse,
                    "timing_ms_per_step": "" if r.timing_ms_per_step is None else r.timing_ms_per_step,
                    "recovery_k": "" if r.recovery_k is None else r.recovery_k,
                    "nll": "" if r.nll_value is None else r.nll_value,
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
    # Normal approximation (good enough for MVP). If n is tiny, treat as indicative.
    return 1.96 * _sem(xs)


AGG_FIELDS = [
    "suite",
    "task_id",
    "scenario_id",
    "model_id",
    "track",
    "n_success",
    "n_total",
    "fail_count",
    "fail_rate",
    # metrics mean/std/sem/ci
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
]


def aggregate_by_seed(records: List[RunRecord]) -> List[Dict[str, Any]]:
    """
    Group by (suite, task_id, scenario_id, model_id, track_id), aggregate across seeds.
    - Failures/missing are counted in fail_count/fail_rate.
    - Metric statistics use only successful records with that metric present.
    """
    groups: Dict[Tuple[str, str, str, str, str], List[RunRecord]] = {}
    for r in records:
        k = (r.suite, r.task_id, r.scenario_id, r.model_id, r.track_id)
        groups.setdefault(k, []).append(r)

    out: List[Dict[str, Any]] = []
    for (suite, task_id, scenario_id, model_id, track_id), rs in sorted(groups.items()):
        n_total = len(rs)
        fails = [r for r in rs if r.status != "ok"]
        n_fail = len(fails)
        n_success = n_total - n_fail

        def collect(getter) -> List[float]:
            xs: List[float] = []
            for r in rs:
                if r.status != "ok":
                    continue
                v = getter(r)
                if v is None:
                    continue
                xs.append(float(v))
            return xs

        mse = collect(lambda r: r.mse)
        mse_db = collect(lambda r: r.mse_db)
        rmse = collect(lambda r: r.rmse)
        timing = collect(lambda r: r.timing_ms_per_step)
        recovery = collect(lambda r: r.recovery_k)
        nll = collect(lambda r: r.nll_value)

        def pack(prefix: str, xs: List[float]) -> Dict[str, Any]:
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

        row: Dict[str, Any] = {
            "suite": suite,
            "task_id": task_id,
            "scenario_id": scenario_id,
            "model_id": model_id,
            "track": track_id,
            "n_success": n_success,
            "n_total": n_total,
            "fail_count": n_fail,
            "fail_rate": (float(n_fail) / float(n_total)) if n_total > 0 else 0.0,
        }
        row.update(pack("mse", mse))
        row.update(pack("mse_db", mse_db))
        row.update(pack("rmse", rmse))
        row.update(pack("timing_ms_per_step", timing))
        row.update(pack("recovery_k", recovery))
        row.update(pack("nll", nll))
        out.append(row)

    return out


def write_aggregate_csv(rows: List[Dict[str, Any]], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=AGG_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in AGG_FIELDS})


def aggregate_to_latex_tabular(rows: List[Dict[str, Any]], caption: str = "", label: str = "") -> str:
    """
    Simple LaTeX exporter for aggregate table (MVP).
    """
    cols = [
        ("model_id", "Model"),
        ("task_id", "Task"),
        ("scenario_id", "Scenario"),
        ("track", "Track"),
        ("mse_mean", "MSE"),
        ("rmse_mean", "RMSE"),
        ("mse_db_mean", "MSE(dB)"),
        ("timing_ms_per_step_mean", "ms/step"),
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
