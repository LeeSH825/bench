from __future__ import annotations

import os
import subprocess
import time
from dataclasses import asdict, dataclass
from typing import Any

from ..tasks.data_format import RunSpec
from ..utils.io import load_yaml


def _is_enabled(obj_cfg: dict, default: bool = True) -> bool:
    """
    D11: enabled 키가 없으면 기본 true, enabled:false면 스킵.
    (SoT: /mnt/data/DECISIONS.md D11, suite YAML runner.enabled_policy)
    """
    if "enabled" not in obj_cfg:
        return default
    return bool(obj_cfg["enabled"])


def plan_runs(suite_yaml_path: str) -> list[RunSpec]:
    """
    Create an execution plan WITHOUT running anything.

    Step 3 scope:
    - Parse suite YAML.
    - Apply enabled_policy-based skipping (D11) when constructing plan.
    - Expand tracks and seeds.
    - scenario expansion (sweep) is NOT fully implemented here (Step 4+).
      For now, scenario_id="default" only.

    Returns:
        A list of RunSpec objects.
    """
    suite = load_yaml(suite_yaml_path)

    suite_name = suite["suite"]["name"]
    suite_version = str(suite["suite"]["version"])
    seeds = suite.get("seeds", [])
    tasks = suite.get("tasks", [])
    models = suite.get("models", [])
    runner_cfg = suite.get("runner", {})

    enabled_policy = runner_cfg.get("enabled_policy", {})
    task_default = bool(enabled_policy.get("task_default", True))
    model_default = bool(enabled_policy.get("model_default", True))
    skip_if_disabled = bool(enabled_policy.get("skip_if_disabled", True))

    tracks = runner_cfg.get("tracks", [])
    if not tracks:
        tracks = [{"track_id": "frozen", "adaptation_enabled": False}]

    plan: list[RunSpec] = []

    for task_cfg in tasks:
        if skip_if_disabled and (not _is_enabled(task_cfg, default=task_default)):
            continue

        for model_cfg in models:
            if skip_if_disabled and (not _is_enabled(model_cfg, default=model_default)):
                continue

            for track in tracks:
                track_id = track["track_id"]
                for seed in seeds:
                    scenario_id = "default"  # Step 4+: sweep expansion

                    plan.append(
                        RunSpec(
                            suite_name=suite_name,
                            suite_version=suite_version,
                            task_id=task_cfg["task_id"],
                            model_id=model_cfg["model_id"],
                            track_id=track_id,
                            seed=int(seed),
                            scenario_id=scenario_id,
                            task_cfg=dict(task_cfg),
                            model_cfg=dict(model_cfg),
                            runner_cfg=dict(runner_cfg),
                        )
                    )
    return plan


@dataclass(frozen=True)
class SubprocessResult:
    returncode: int
    stdout: str
    stderr: str
    elapsed_s: float


def run_in_subprocess(
    cmd: list[str],
    cwd: str | None = None,
    env: dict[str, str] | None = None,
    timeout_s: int | None = None,
) -> SubprocessResult:
    """
    Execute a command in a subprocess and capture stdout/stderr.

    Step 3 scope:
    - This is an execution primitive for future fallback mode.
    - It does not implement training/eval loops itself.

    Intended use (Step 6+):
    - Call third_party main scripts safely
    - Always capture stdout/stderr to run_dir logs
    """
    t0 = time.time()
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)

    p = subprocess.Popen(
        cmd,
        cwd=cwd,
        env=merged_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        out, err = p.communicate(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        p.kill()
        out, err = p.communicate()
        err = (err or "") + "\n[TIMEOUT]\n"
    elapsed = time.time() - t0
    return SubprocessResult(returncode=int(p.returncode), stdout=out or "", stderr=err or "", elapsed_s=float(elapsed))


def run_in_docker(
    image: str,
    cmd: list[str],
    mounts: list[tuple[str, str]] | None = None,
    gpus: str | None = None,
    workdir: str | None = None,
    env: dict[str, str] | None = None,
) -> None:
    """
    Placeholder for model-specific container execution (fallback).

    Step 3 scope:
    - Provide signature + documentation only.
    - Actual orchestration & result collection is implemented in Step 6+.

    Conventions (planned):
    - mounts: list of (host_path, container_path)
    - gpus: e.g. "all" or "device=0"
    - runner ensures run_dir is mounted to a shared host location.
    """
    raise NotImplementedError("Step 3 scaffold: docker execution is implemented in Step 6+.")


def run_plan(*args: Any, **kwargs: Any) -> None:
    """
    Placeholder for Step 4+.

    Responsibilities (future):
    - data generation (bench_generated)
    - adapter import-mode execution (default)
    - fallback subprocess/docker execution
    - env/git snapshot + metrics artifacts writing
    """
    raise NotImplementedError("Step 3 scaffold: execution is implemented in Step 4+.")


def dump_plan_jsonable(plan: list[RunSpec]) -> list[dict]:
    """Convert RunSpec list to JSON-serializable dicts (debug/inspection)."""
    return [asdict(p) for p in plan]

