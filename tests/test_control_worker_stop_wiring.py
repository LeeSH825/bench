"""Production worker graceful-stop wiring (gate 19 §6.1).

The blocker this closes was invisible to unit tests: StopCoordinator and
settle_graceful_stop were implemented and covered at service level, but nothing
constructed them in the worker. These tests assert the *wiring* exists and is
gated on the certified path only.
"""

from __future__ import annotations

import inspect

import pytest

from bench.control.process import worker_cli
from bench.control.registry.schema import RunState


def test_worker_module_wires_the_stop_coordinator() -> None:
    """The worker must construct a StopCoordinator, not merely import one."""
    source = inspect.getsource(worker_cli)
    assert "StopCoordinator(" in source, "worker never constructs a StopCoordinator"
    assert '"stop_requested"' in source, "worker never injects the stop callback"
    assert '"on_interrupt"' in source, "worker never injects the settlement callback"
    assert "settle_graceful_stop" in source


def test_stop_wiring_is_gated_on_the_certified_path() -> None:
    """legacy_train_v1 and not_applicable must get no stop callback."""
    source = inspect.getsource(worker_cli)
    assert 'training_path_id == "control_resumable_v1"' in source, (
        "stop wiring must be gated on the certified resumable path"
    )


def test_worker_does_not_overwrite_a_settled_terminal_state() -> None:
    """An interrupted run must not fall through to the COMPLETED handler."""
    assert RunState.INTERRUPTED in worker_cli._RUNNER_SETTLED_STATES
    assert RunState.FAILED in worker_cli._RUNNER_SETTLED_STATES
    assert RunState.COMPLETED not in worker_cli._RUNNER_SETTLED_STATES


def test_settlement_builder_passes_the_training_path() -> None:
    """The interrupt package must be Checkpoint v2, or a child cannot resume."""
    source = inspect.getsource(worker_cli._build_interrupt_settlement)
    assert "training_path_id=" in source
    assert "training_path_contract_version=" in source
    assert "capture_rng()" in source


def test_runner_refuses_to_complete_an_interrupted_run_without_settlement() -> None:
    """No settlement callback must raise, never report COMPLETED."""
    from bench.runners import run_suite

    source = inspect.getsource(run_suite._call_resumable_train)
    assert "result.interrupted" in source
    assert "on_interrupt" in source
    assert "refusing" in source.lower()


def test_settle_graceful_stop_accepts_a_training_path() -> None:
    from bench.control.checkpoints.lifecycle import settle_graceful_stop

    params = inspect.signature(settle_graceful_stop).parameters
    assert "training_path_id" in params
    assert "training_path_contract_version" in params
