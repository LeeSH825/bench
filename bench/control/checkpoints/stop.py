"""Graceful-stop coordination.

A stop request is a **row in the registry**, not a signal (ADR-CSR-011). That
choice is what makes the request survive an API restart, survive the requester
exiting, and be idempotent: the worker polls for its own outstanding action and
honours it at the next safe boundary.

A signal, if one is delivered, only ever sets a process-local flag
(ADR-CSR-010). No checkpoint, no SQLite write, and no ``torch.save`` happens in
a signal handler.
"""

from __future__ import annotations

import signal
import threading
from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class StopDecision:
    """Why the loop is stopping, and which action to settle afterwards."""

    requested: bool = False
    action_id: Optional[str] = None
    source: str = ""


class StopCoordinator:
    """Watches for a stop request on behalf of one run.

    Two independent sources are merged:

    * the registry action row — the durable, API-independent path;
    * a process-local flag set by a signal handler — a fast wake-up only.

    Polling is rate-limited so a tight training loop does not hit SQLite on
    every update.
    """

    def __init__(
        self,
        *,
        run_id: str,
        registry: Any,
        poll_interval_updates: int = 1,
        install_signal_handlers: bool = True,
    ) -> None:
        self.run_id = run_id
        self.registry = registry
        self.poll_interval_updates = max(1, int(poll_interval_updates))
        self._flag = threading.Event()
        self._decision = StopDecision()
        self._calls = 0
        self._installed: list[tuple[int, Any]] = []
        if install_signal_handlers:
            self._install()

    # -- signal path (flag only) --------------------------------------------

    def _install(self) -> None:
        def _handler(signum: int, _frame: Any) -> None:
            # Deliberately the entire handler body: set a flag and return.
            self._flag.set()

        for sig in (signal.SIGTERM, signal.SIGINT):
            try:
                self._installed.append((sig, signal.getsignal(sig)))
                signal.signal(sig, _handler)
            except (ValueError, OSError):  # pragma: no cover - non-main thread
                pass

    def restore_signal_handlers(self) -> None:
        for sig, previous in self._installed:
            try:
                signal.signal(sig, previous)
            except (ValueError, OSError):  # pragma: no cover
                pass
        self._installed.clear()

    def signal_stop(self) -> None:
        """Test/entry-point hook equivalent to receiving SIGTERM."""
        self._flag.set()

    # -- registry path -------------------------------------------------------

    def poll(self) -> bool:
        """True once a stop is pending. Safe to call every update."""
        if self._decision.requested:
            return True

        if self._flag.is_set():
            self._decision = StopDecision(requested=True, source="signal")
            return True

        self._calls += 1
        if self._calls % self.poll_interval_updates != 0:
            return False

        action = self.registry.open_action(self.run_id, action="stop")
        if action is None:
            return False

        # Acknowledge here: the worker has seen the request. The action is not
        # *completed* until the interrupt checkpoint is durable.
        self.registry.acknowledge_action(action["action_id"])
        self._decision = StopDecision(
            requested=True, action_id=action["action_id"], source="registry"
        )
        return True

    @property
    def decision(self) -> StopDecision:
        return self._decision

    def __call__(self) -> bool:
        """Usable directly as the trainer's ``stop_requested`` callback."""
        return self.poll()
