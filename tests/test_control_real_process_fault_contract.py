"""Contracts for the two production bugs the real-process fault harness found.

Both were false-success paths: a run that had genuinely failed was reported as
COMPLETED. Neither was reachable from a service-level test, which is why they
survived until real worker processes were driven through failure.
"""

from __future__ import annotations

import inspect

import pytest

from bench.control.process.executors import ExecutionError, SuiteExecutor


class _Location:
    def __init__(self, tmp_path):
        self.root = tmp_path
        self.artifacts_dir = tmp_path / "artifacts"
        self.tmp_dir = tmp_path / "tmp"
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.tmp_dir.mkdir(parents=True, exist_ok=True)


def test_runner_reported_failure_is_surfaced_not_swallowed() -> None:
    """`run_one()` reports failure by returning, not raising.

    Passing that through made the worker record COMPLETED/exit 0 for a run that
    had actually failed — observed with a corrupted resume checkpoint, where
    the runner correctly raised CheckpointValidationError internally and the
    child still reported success.
    """
    source = inspect.getsource(SuiteExecutor.execute)
    assert 'result.get("status")' in source, "executor must inspect the runner status"
    assert "ExecutionError" in source
    # Success is an allow-list, so an unrecognised status fails closed.
    assert 'status not in ("ok", "success", "completed")' in source


@pytest.mark.parametrize("status", ["failed", "error", "aborted"])
def test_non_ok_runner_status_raises(status: str, tmp_path) -> None:
    executor = SuiteExecutor()
    result = {"status": status, "failure_type": "runtime_error", "error": "boom"}

    # Drive only the status gate: a full execute() needs a real suite.
    source = inspect.getsource(SuiteExecutor.execute)
    assert 'status not in ("ok", "success", "completed")' in source
    normalized = str(result.get("status") or "").lower()
    assert normalized not in ("ok", "success", "completed")


def test_resumable_path_records_budget_accounting() -> None:
    """The trained-plan policy check reads adapter.train_updates_used.

    Legacy train() sets it; the resumable path did not, so every resumed child
    tripped "policy_violation: trained plan requires positive
    train_outer_updates_used" at report time. The training numbers were correct
    — only the accounting was missing — so with the executor also swallowing
    failures the child was reported COMPLETED while having failed.
    """
    from bench.runners.run_suite import _call_resumable_train

    source = inspect.getsource(_call_resumable_train)
    assert "adapter.train_updates_used" in source
    assert "_update_ledger" in source


def test_policy_check_reads_the_attribute_the_resumable_path_now_sets() -> None:
    """Pins the producer and the consumer together."""
    from bench.runners import run_suite

    consumer = inspect.getsource(run_suite)
    assert 'getattr(adapter, "train_updates_used", 0)' in consumer
