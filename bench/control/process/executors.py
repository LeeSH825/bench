"""What a worker actually executes.

Two executors ship in this tranche.

:class:`SyntheticExecutor`
    A tiny, dependency-free workload that emits the full event vocabulary. It is
    what the tests and the operator demo use — per the mandate to prefer
    "tiny/synthetic smoke run" over long training. It is not a fake success: it
    can be told to fail, and it produces real artifacts.

:class:`SuiteExecutor`
    Runs a real bench suite entry by calling
    ``bench.runners.run_suite.run_one`` — the existing, unmodified runner.

**Output containment.** ``run_one`` derives its output directory from
``reporting.output_dir_template`` and would otherwise write to the shared
deterministic path under ``runs/`` — the exact overwrite hazard of R-01/DND-004.
:class:`SuiteExecutor` therefore rewrites that template (and the model cache
directory) to absolute paths *inside the immutable control run directory* before
calling it. The suite dict is deep-copied first, so nothing on disk is modified
and the original YAML is untouched.
"""

from __future__ import annotations

import copy
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Protocol

from ..allocation import RunLocation, atomic_write_text
from ..config.schema import ResolvedRunSpec
from ..events.observer import RunObserver
from ..events.schema import (
    METRIC_TEST_MSE,
    METRIC_TRAIN_LOSS,
    METRIC_VALIDATION_LOSS,
)


class ExecutionError(RuntimeError):
    """Raised when the workload fails in an ordinary (recorded) way."""


class Executor(Protocol):
    """Runs the workload described by a resolved run spec."""

    name: str

    def execute(
        self, spec: ResolvedRunSpec, *, location: RunLocation, observer: RunObserver
    ) -> Mapping[str, Any]:
        """Run the workload and return a JSON-serializable result summary."""


@dataclass
class SyntheticExecutor:
    """Deterministic synthetic workload for smoke tests and demos.

    Emits phase status events, per-update training loss, periodic validation
    loss, and a final test metric — i.e. exactly the minimum metric set the
    contract requires — without importing torch or touching any dataset.

    ``fail_at_step`` injects an ordinary failure so the FAILED path and the
    traceback artifact can be tested for real rather than asserted about.
    """

    name: str = "synthetic"
    step_sleep_seconds: float = 0.0
    fail_at_step: Optional[int] = None

    def execute(
        self, spec: ResolvedRunSpec, *, location: RunLocation, observer: RunObserver
    ) -> Mapping[str, Any]:
        training = spec.draft.training
        total_updates = max(1, int(training.max_updates)) if training.enabled else 0
        validation_interval = max(1, int(training.validation_interval_updates or 5))
        seed = int(spec.draft.runtime.seed)

        observer.status("PHASE_START", phase="setup", message="synthetic executor starting")
        observer.log(
            f"synthetic workload: {total_updates} updates, seed {seed}", phase="setup"
        )

        best_validation = math.inf
        if total_updates:
            observer.status("PHASE_START", phase="train")
            for step in range(1, total_updates + 1):
                if self.fail_at_step is not None and step >= int(self.fail_at_step):
                    raise ExecutionError(
                        f"synthetic failure injected at optimizer step {step}"
                    )
                # A smooth, seed-dependent decay — deterministic, so the same
                # spec always produces the same curve.
                loss = (1.0 + (seed % 7) * 0.05) / (1.0 + 0.35 * step)
                observer.metric(
                    METRIC_TRAIN_LOSS, loss, step=step, phase="train", unit="mse"
                )
                if step % validation_interval == 0:
                    validation = loss * 1.08
                    best_validation = min(best_validation, validation)
                    observer.metric(
                        METRIC_VALIDATION_LOSS,
                        validation,
                        step=step,
                        phase="validation",
                        unit="mse",
                    )
                if self.step_sleep_seconds > 0:
                    time.sleep(self.step_sleep_seconds)
            observer.status("PHASE_END", phase="train")

        observer.status("PHASE_START", phase="test")
        test_mse = (best_validation if math.isfinite(best_validation) else 1.0) * 1.02
        observer.metric(METRIC_TEST_MSE, test_mse, step=total_updates, phase="test", unit="mse")
        observer.status("PHASE_END", phase="test")

        result = {
            "executor": self.name,
            "updates": total_updates,
            "test_mse": test_mse,
            "best_validation_loss": (
                best_validation if math.isfinite(best_validation) else None
            ),
        }
        metrics_path = location.artifacts_dir / "metrics.json"
        atomic_write_text(
            metrics_path, json.dumps(result, indent=2, sort_keys=True), tmp_dir=location.tmp_dir
        )
        observer.artifact(
            kind="metrics",
            uri=str(metrics_path.relative_to(location.root)),
            bytes_=metrics_path.stat().st_size,
        )
        return result


@dataclass
class SuiteExecutor:
    """Executes a real bench suite entry via the existing runner.

    The suite document, task, and model are recovered from the run spec's
    preserved ``original_config`` — which is why preserving it is not optional.
    """

    name: str = "suite"

    def execute(
        self, spec: ResolvedRunSpec, *, location: RunLocation, observer: RunObserver
    ) -> Mapping[str, Any]:
        original = spec.draft.original_config
        if not original:
            raise ExecutionError(
                "the resolved run spec carries no original_config, so the bench suite "
                "entry cannot be reconstructed. Use the synthetic executor, or resolve "
                "the spec through bench.control.config.compatibility."
            )

        suite = copy.deepcopy(dict(original))
        task = _find_by_id(suite.get("tasks") or [], "task_id", spec.draft.system.task_id)
        if task is None:
            raise ExecutionError(
                f"task_id {spec.draft.system.task_id!r} is not present in the attached suite config"
            )
        model = _find_by_id(suite.get("models") or [], "model_id", spec.model_id.value)
        if model is None:
            raise ExecutionError(
                f"model_id {spec.model_id.value!r} is not present in the attached suite config"
            )

        # Contain every runner output inside this run's immutable directory.
        legacy_root = location.root / "legacy"
        legacy_root.mkdir(parents=True, exist_ok=True)
        reporting = dict(suite.get("reporting") or {})
        reporting["output_dir_template"] = str(legacy_root)
        suite["reporting"] = reporting
        runner = dict(suite.get("runner") or {})
        # Redirect the shared train cache too: it lives under `runs/_model_cache`
        # by default, and this tranche writes nothing into the legacy run tree.
        runner["model_cache_dir"] = str(location.root / "model_cache")
        suite["runner"] = runner

        observer.status("PHASE_START", phase="setup", message="invoking bench.runners.run_suite.run_one")

        from bench.runners.run_suite import run_one  # deferred: imports torch

        started = time.time()
        result = run_one(
            suite=suite,
            task=task,
            model=model,
            scenario_settings=dict(spec.draft.system.scenario_config or {}),
            seed=int(spec.draft.runtime.seed),
            track_id=str(spec.draft.bench_context.get("track_id") or "frozen"),
            device_str=str(spec.draft.runtime.device),
            precision=str(spec.draft.runtime.precision),
            init_id=str(spec.init_id.mode),
        )
        elapsed = time.time() - started

        summary = {
            "executor": self.name,
            "elapsed_seconds": elapsed,
            "runner_result": _jsonable(result),
        }
        _emit_runner_metrics(result, observer=observer)
        summary_path = location.artifacts_dir / "runner_result.json"
        atomic_write_text(
            summary_path, json.dumps(summary, indent=2, default=str), tmp_dir=location.tmp_dir
        )
        observer.artifact(
            kind="metrics",
            uri=str(summary_path.relative_to(location.root)),
            bytes_=summary_path.stat().st_size,
        )
        return summary


def _find_by_id(items: Any, key: str, value: str) -> Optional[dict[str, Any]]:
    for item in items or []:
        if isinstance(item, Mapping) and str(item.get(key)) == str(value):
            return dict(item)
    return None


def _jsonable(value: Any) -> Any:
    try:
        json.dumps(value)
        return value
    except (TypeError, ValueError):
        return json.loads(json.dumps(value, default=str))


def _emit_runner_metrics(result: Any, *, observer: RunObserver) -> None:
    """Emit final metrics from the runner's own result mapping.

    Only well-known numeric keys are lifted, and each is emitted under the
    canonical event namespace. This reads the runner's **return value** — a
    structured object — not its stdout (DND-006).
    """
    if not isinstance(result, Mapping):
        return
    interesting = {
        "mse": "metric/test_mse",
        "mse_db": "metric/test_mse_db",
        "rmse": "metric/test_rmse",
        "nll": "metric/test_nll",
        "timing_ms_per_step": "latency/update_ms",
    }
    for source_key, metric_name in interesting.items():
        value = result.get(source_key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if math.isfinite(float(value)):
                observer.metric(metric_name, float(value), phase="test", step_type="global_step")


#: Executors selectable from the worker CLI.
EXECUTORS: dict[str, Any] = {
    "synthetic": SyntheticExecutor,
    "suite": SuiteExecutor,
}


def build_executor(name: str, **kwargs: Any) -> Executor:
    factory = EXECUTORS.get(str(name))
    if factory is None:
        raise ExecutionError(
            f"unknown executor {name!r}; available: {sorted(EXECUTORS)}"
        )
    return factory(**kwargs)  # type: ignore[no-any-return]
