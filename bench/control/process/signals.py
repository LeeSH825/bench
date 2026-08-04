"""Worker exit codes, signal handling, and process-group helpers.

Exit-code contract from design doc 03 §14. The contract exists so the manager
can distinguish "the training raised" from "the worker itself is broken" from
"something outside killed us" — but the exit code is **advisory**. The
authoritative outcome is the registry terminal transition the worker recorded
before exiting (design doc 03 §14, final line). A process killed by SIGKILL
records nothing and gets no exit code at all, which is exactly why abrupt death
becomes ORPHANED rather than FAILED.
"""

from __future__ import annotations

import enum
import os
import signal
import threading
from typing import Any, Callable, Optional


class ExitCode(enum.IntEnum):
    """Worker process exit codes."""

    COMPLETED = 0
    INTERRUPTED_WITH_CHECKPOINT = 10
    CANCELLED_BEFORE_EXECUTION = 20
    VALIDATION_ERROR = 30
    EXECUTION_FAILURE = 40
    CHECKPOINT_WRITE_FAILURE = 50
    WORKER_PROTOCOL_FAILURE = 60
    EXTERNAL_TERMINATION = 70


#: Human-readable explanation per exit code, surfaced in the dashboard.
EXIT_CODE_DESCRIPTIONS: dict[int, str] = {
    ExitCode.COMPLETED: "Run completed normally.",
    ExitCode.INTERRUPTED_WITH_CHECKPOINT: (
        "Run was interrupted at a safe boundary with a valid checkpoint. "
        "Not produced in this tranche (graceful stop is not enabled)."
    ),
    ExitCode.CANCELLED_BEFORE_EXECUTION: "Run was cancelled before execution began.",
    ExitCode.VALIDATION_ERROR: "Run specification failed validation or was incompatible.",
    ExitCode.EXECUTION_FAILURE: "Training or evaluation raised an exception.",
    ExitCode.CHECKPOINT_WRITE_FAILURE: "A checkpoint could not be written.",
    ExitCode.WORKER_PROTOCOL_FAILURE: (
        "The worker itself failed (registry unreachable, spec unreadable, journal unwritable)."
    ),
    ExitCode.EXTERNAL_TERMINATION: "The worker observed an external termination signal.",
}


def describe_exit_code(code: Optional[int]) -> str:
    """Explain an exit code, including signal deaths (negative codes)."""
    if code is None:
        return "No exit code recorded — the process did not report an exit status."
    if code < 0:
        try:
            name = signal.Signals(-code).name
        except ValueError:
            name = f"signal {-code}"
        return f"Process was killed by {name}. No orderly shutdown occurred."
    return EXIT_CODE_DESCRIPTIONS.get(int(code), f"Unrecognized exit code {code}.")


class TerminationRequest(BaseException):
    """Raised inside the worker when a termination signal arrives.

    Derives from :class:`BaseException`, not :class:`Exception`, so that a
    ``except Exception`` in adapter or training code cannot accidentally swallow
    a shutdown request.
    """

    def __init__(self, signal_number: int):
        self.signal_number = signal_number
        try:
            name = signal.Signals(signal_number).name
        except ValueError:
            name = str(signal_number)
        super().__init__(f"termination requested by {name}")


class SignalHandler:
    """Records SIGINT/SIGTERM so the worker can exit deliberately.

    The handler only sets a flag and invokes an optional callback; it does not
    do the shutdown work itself. Doing real work in a signal handler is how you
    get half-written files.

    This tranche does **not** implement graceful stop: on a signal the worker
    records a terminal FAILED/CANCELLED transition and exits. It does not
    checkpoint, and it does not claim to.
    """

    def __init__(self, *, on_signal: Optional[Callable[[int], None]] = None):
        self._on_signal = on_signal
        self._received: Optional[int] = None
        self._event = threading.Event()
        self._previous: dict[int, Any] = {}

    @property
    def received(self) -> Optional[int]:
        return self._received

    @property
    def triggered(self) -> bool:
        return self._event.is_set()

    def wait(self, timeout: Optional[float] = None) -> bool:
        return self._event.wait(timeout)

    def _handle(self, signal_number: int, _frame: Any) -> None:
        if self._received is None:
            self._received = int(signal_number)
        self._event.set()
        if self._on_signal is not None:
            try:
                self._on_signal(int(signal_number))
            except Exception:
                pass

    def install(self) -> "SignalHandler":
        for signal_number in (signal.SIGINT, signal.SIGTERM):
            try:
                self._previous[signal_number] = signal.getsignal(signal_number)
                signal.signal(signal_number, self._handle)
            except (ValueError, OSError):
                # Not on the main thread, or the platform disallows it.
                continue
        return self

    def restore(self) -> None:
        for signal_number, handler in self._previous.items():
            try:
                signal.signal(signal_number, handler)
            except (ValueError, OSError):
                continue
        self._previous.clear()

    def __enter__(self) -> "SignalHandler":
        return self.install()

    def __exit__(self, *exc_info: Any) -> None:
        self.restore()


# --------------------------------------------------------------------------- #
# process group helpers
# --------------------------------------------------------------------------- #


def start_new_session() -> None:
    """``preexec_fn`` that detaches the child into its own session.

    Two consequences, both required:

    * the worker gets a **new process group**, so DataLoader workers and any
      subprocess it spawns are in one killable unit (acceptance P-02);
    * the worker leaves the launching terminal's session, so a Ctrl-C in the
      shell that started the API does not propagate to running training
      (acceptance P-01).
    """
    os.setsid()


def process_group_of(pid: int) -> Optional[int]:
    """Process group id of *pid*, or ``None`` if it cannot be read."""
    try:
        return os.getpgid(pid)
    except (ProcessLookupError, PermissionError, OSError):
        return None


def signal_process_group(pgid: int, signal_number: int) -> bool:
    """Send *signal_number* to an entire process group.

    Returns ``True`` if the signal was delivered. Used for force-termination in
    a later tranche; exposed here so the group semantics live in one place.
    Guards against ``pgid <= 1``: ``killpg(0, …)`` would signal *our own* group,
    which would take down the manager.
    """
    if pgid <= 1:
        return False
    try:
        os.killpg(pgid, signal_number)
        return True
    except (ProcessLookupError, PermissionError, OSError):
        return False
