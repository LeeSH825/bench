"""GUI/API and CLI must describe — and produce — the same experiment.

The GUI is a second way to start work that already had one. If the two paths
resolve differently, every number produced through the browser is quietly
incomparable with the ones in the paper. So this pins both halves:

* **config parity** — the resolved spec, hashes, variant, training path,
  dataset identity and budget must be identical, modulo the fields that are
  per-run identity by definition;
* **numerical parity** — a tiny deterministic run launched each way must
  produce the same metrics. That half is expensive, so it is opt-in via
  ``BENCH_PARITY_E2E=1`` and is exercised as a release gate, not on every
  pytest invocation.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Optional

import pytest

from bench.control.config.gui_service import validate_config
from bench.control.config.presets import PresetCatalog, safe_load_preset_text
from bench.control.provenance import repository_provenance

REPO_ROOT = Path(__file__).resolve().parents[1]

#: Fields that *must* differ: they identify this run, not this configuration.
PER_RUN_IDENTITY = {
    "experiment.experiment_id",
    "identity.run_id",
    "hashes.resolved_spec_hash",  # derived from the two above
}

#: Not a config property. ``environment_document`` only reports torch facts when
#: torch happens to be imported in the *capturing* process, so a CLI subprocess
#: and an in-process API can fingerprint the same machine differently. Excluded
#: here and recorded in known_limitations rather than silently asserted away.
ENVIRONMENT_DEPENDENT = {"provenance.environment_fingerprint"}

CASES = [
    ("bench/configs/suite_train_smoke.yaml", "kalmannet_tsp", "trained"),
    ("bench/configs/suite_split_train_smoke.yaml", "split_knet", "trained"),
    ("bench/configs/suite_all_simple_tiny.yaml", "mb_kf_oracle", "untrained"),
]


def _flatten(document: Any, prefix: str = "") -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in (document or {}).items():
        path = f"{prefix}{key}"
        if isinstance(value, dict):
            flat.update(_flatten(value, path + "."))
        else:
            flat[path] = value
    return flat


def _cli_spec(tmp_path: Path, suite: str, model: str, init: str) -> dict[str, Any]:
    """Allocate through the CLI and read back the spec it persisted."""
    env = dict(os.environ, BENCH_CONTROL_ROOT=str(tmp_path))
    completed = subprocess.run(
        [sys.executable, "-m", "bench.control.cli", "launch",
         "--suite", str(REPO_ROOT / suite), "--model", model,
         "--init", init, "--dry-run"],
        cwd=str(REPO_ROOT), env=env, capture_output=True, text=True, timeout=900)
    assert completed.returncode == 0, completed.stderr
    run_dir = Path(json.loads(completed.stdout)["run_dir"])
    return json.loads((run_dir / "resolved_run_spec.json").read_text(encoding="utf-8"))


def _gui_spec(suite: str, model: str, init: str) -> Any:
    text = (REPO_ROOT / suite).read_text(encoding="utf-8")
    result = validate_config(suite_document=safe_load_preset_text(text),
                             model_id=model, init_id=init,
                             provenance=repository_provenance())
    assert result.valid, [i.as_dict() for i in result.issues]
    return result


@pytest.mark.parametrize("suite,model,init", CASES,
                         ids=[c[1] for c in CASES])
def test_gui_and_cli_resolve_to_the_same_spec(tmp_path, suite, model, init) -> None:
    cli = _flatten(_cli_spec(tmp_path, suite, model, init))
    gui = _flatten(_gui_spec(suite, model, init).resolved_run_spec)

    assert set(cli) == set(gui), "the two paths produced different spec shapes"
    allowed = PER_RUN_IDENTITY | ENVIRONMENT_DEPENDENT
    differing = {k for k in cli if cli[k] != gui[k]}
    assert differing <= allowed, {
        k: (cli[k], gui[k]) for k in sorted(differing - allowed)}
    # And the identity fields really are present, i.e. the assertion above is
    # not vacuous because the keys vanished.
    assert PER_RUN_IDENTITY <= set(cli)


@pytest.mark.parametrize("suite,model,init", CASES, ids=[c[1] for c in CASES])
def test_identity_and_budget_match_field_by_field(tmp_path, suite, model, init) -> None:
    """Spelled out explicitly, so a shape change cannot hide a drift."""
    cli = _cli_spec(tmp_path, suite, model, init)
    result = _gui_spec(suite, model, init)
    gui = result.resolved_run_spec

    hashes_cli, hashes_gui = cli["hashes"], gui["hashes"]
    assert hashes_cli["structural_config_hash"] == hashes_gui["structural_config_hash"]
    assert hashes_cli["operational_config_hash"] == hashes_gui["operational_config_hash"]
    assert cli["dataset"] == gui["dataset"], "dataset identity drifted"
    assert cli["identity"]["variant_id"] == gui["identity"]["variant_id"]
    assert cli["identity"]["implementation_id"] == gui["identity"]["implementation_id"]
    assert cli["execution"]["training_path_id"] == gui["execution"]["training_path_id"]
    assert cli["execution"]["certification_id"] == gui["execution"]["certification_id"]
    assert cli["training"] == gui["training"]
    assert cli["optimizer"] == gui["optimizer"]
    assert cli["runtime"] == gui["runtime"]
    assert cli["system"] == gui["system"]
    assert cli["initialization"] == gui["initialization"]
    assert cli["bench_context"] == gui["bench_context"]

    # The API surfaces the same values it resolved.
    assert result.structural_config_hash == hashes_cli["structural_config_hash"]
    assert result.variant_id == cli["identity"]["variant_id"]
    assert result.training_path_id == cli["execution"]["training_path_id"]


def test_provenance_is_captured_on_the_launch_path(tmp_path, monkeypatch) -> None:
    """A preview skips Git; an allocated run must not."""
    from bench.control.launch_coordinator import LaunchCoordinator
    from bench.control.registry.sqlite import SqliteRegistry
    from tests.test_control_launch_api import StubManager

    preview = validate_config(
        suite_document=safe_load_preset_text(
            (REPO_ROOT / CASES[0][0]).read_text(encoding="utf-8")),
        model_id="kalmannet_tsp", init_id="trained")
    assert preview.valid
    assert preview.resolved_run_spec["provenance"]["git_commit"] is None

    monkeypatch.setenv("BENCH_CONTROL_ROOT", str(tmp_path))
    registry = SqliteRegistry(tmp_path / "registry.sqlite3")
    manager = StubManager(registry, control_root_path=str(tmp_path))
    catalog = PresetCatalog()
    entry = next(e for e in catalog.list()
                 if e.relative_path == CASES[0][0])
    outcome = LaunchCoordinator(registry=registry, manager=manager,
                                control_root=tmp_path).request_launch(
        preset_id=entry.preset_id, preset_digest=entry.content_digest,
        idempotency_key="prov", model_id="kalmannet_tsp", init_id="trained")
    spec = json.loads((Path(registry.get_run(outcome.run_id).run_dir)
                       / "resolved_run_spec.json").read_text(encoding="utf-8"))
    assert spec["provenance"]["git_commit"], "GUI-launched run has no git provenance"
    assert spec["provenance"]["environment_fingerprint"]
    assert spec["provenance"]["submodule_revisions"]
    assert spec["bench_context"]["executor"] == "suite"

    # Provenance is not an identity input, so capturing it did not move the run.
    assert spec["hashes"]["structural_config_hash"] == preview.structural_config_hash
    assert spec["identity"]["variant_id"] == preview.variant_id


# -- numerical parity (opt-in) ----------------------------------------------

PARITY_E2E = os.environ.get("BENCH_PARITY_E2E") == "1"
requires_e2e = pytest.mark.skipif(
    not PARITY_E2E, reason="set BENCH_PARITY_E2E=1 to run the real training gate")


def _wait_terminal(registry, run_id: str, timeout: float = 1800.0) -> Any:
    deadline = time.time() + timeout
    while time.time() < deadline:
        record = registry.get_run(run_id)
        if record is not None and str(record.state) in (
                "RunState.COMPLETED", "COMPLETED", "RunState.FAILED", "FAILED"):
            return record
        if record is not None and getattr(record.state, "name", "") in ("COMPLETED", "FAILED"):
            return record
        time.sleep(2.0)
    raise AssertionError(f"run {run_id} did not finish within {timeout}s")


#: Wall-clock measurements. They differ between any two executions of the same
#: code and say nothing about numerical agreement.
WALL_CLOCK_KEYS = frozenset({
    "elapsed_seconds", "duration_seconds", "wall_clock_seconds", "timing_ms_per_step",
    "started_at", "finished_at", "timestamp", "train_seconds", "eval_seconds",
})


def _metrics(run_dir: Path) -> dict[str, Any]:
    """Runner metrics with wall-clock and run-local paths normalised away.

    Every run lives in its own immutable directory, so absolute paths must
    differ; they are rewritten to a token rather than dropped, which keeps the
    *shape* of the artifact contract under comparison.
    """
    payload = json.loads((run_dir / "artifacts" / "runner_result.json").read_text())

    def _clean(value: Any) -> Any:
        if isinstance(value, dict):
            return {k: _clean(v) for k, v in value.items() if k not in WALL_CLOCK_KEYS}
        if isinstance(value, list):
            return [_clean(v) for v in value]
        if isinstance(value, str):
            return value.replace(str(run_dir), "<run_dir>")
        return value

    return _clean(dict(payload.get("runner_result") or {}))


@requires_e2e
@pytest.mark.parametrize("suite,model,init", CASES[:2], ids=["knet", "split"])
def test_gui_and_cli_produce_identical_numbers(tmp_path, suite, model, init) -> None:
    """Same config, two launch paths, one set of numbers."""
    from bench.control.launch_coordinator import LaunchCoordinator
    from bench.control.process.manager import WorkerManager
    from bench.control.registry.sqlite import SqliteRegistry

    env = dict(os.environ, BENCH_CONTROL_ROOT=str(tmp_path))
    completed = subprocess.run(
        [sys.executable, "-m", "bench.control.cli", "launch",
         "--suite", str(REPO_ROOT / suite), "--model", model, "--init", init],
        cwd=str(REPO_ROOT), env=env, capture_output=True, text=True, timeout=3600)
    assert completed.returncode == 0, completed.stderr
    cli_run_id = json.loads(completed.stdout)["run_id"]

    registry = SqliteRegistry(tmp_path / "registry.sqlite3")
    manager = WorkerManager(registry, control_root_path=str(tmp_path),
                            repo_root=REPO_ROOT)
    entry = next(e for e in PresetCatalog().list() if e.relative_path == suite)
    outcome = LaunchCoordinator(registry=registry, manager=manager,
                                control_root=tmp_path).request_launch(
        preset_id=entry.preset_id, preset_digest=entry.content_digest,
        idempotency_key=f"parity-{model}", model_id=model, init_id=init)
    assert outcome.run_id

    cli_record = _wait_terminal(registry, cli_run_id)
    gui_record = _wait_terminal(registry, outcome.run_id)
    assert getattr(cli_record.state, "name", str(cli_record.state)) == "COMPLETED"
    assert getattr(gui_record.state, "name", str(gui_record.state)) == "COMPLETED"

    cli_metrics = _metrics(Path(cli_record.run_dir))
    gui_metrics = _metrics(Path(gui_record.run_dir))
    assert cli_metrics == gui_metrics, (
        f"numerical drift between CLI and GUI launch for {model}")
    assert cli_metrics, "no metrics were compared"


@requires_e2e
def test_model_based_baseline_launches_and_completes(tmp_path) -> None:
    """A filter with no learning lifecycle is launchable, just not resumable."""
    from bench.control.launch_coordinator import LaunchCoordinator
    from bench.control.process.manager import WorkerManager
    from bench.control.registry.sqlite import SqliteRegistry

    suite, model, init = CASES[2]
    result = _gui_spec(suite, model, init)
    assert result.training_path_id == "not_applicable"
    assert result.launch_eligibility["eligible"] is True
    assert result.launch_eligibility["stop_resume_available"] is False

    os.environ["BENCH_CONTROL_ROOT"] = str(tmp_path)
    registry = SqliteRegistry(tmp_path / "registry.sqlite3")
    manager = WorkerManager(registry, control_root_path=str(tmp_path), repo_root=REPO_ROOT)
    entry = next(e for e in PresetCatalog().list() if e.relative_path == suite)
    outcome = LaunchCoordinator(registry=registry, manager=manager,
                                control_root=tmp_path).request_launch(
        preset_id=entry.preset_id, preset_digest=entry.content_digest,
        idempotency_key="baseline-e2e", model_id=model, init_id=init)
    record = _wait_terminal(registry, outcome.run_id)
    assert getattr(record.state, "name", str(record.state)) == "COMPLETED", record.state
    assert record.exit_code == 0
    assert _metrics(Path(record.run_dir))
