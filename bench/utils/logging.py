from __future__ import annotations

import contextlib
import contextvars
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Iterator, Optional


_LOGGER_NAME = "bench"
_CONSOLE_HANDLER_NAME = "bench_console"
_FILE_HANDLER_NAME = "bench_file"
_RUN_CONTEXT: contextvars.ContextVar[Dict[str, str]] = contextvars.ContextVar("bench_log_context", default={})


class _RunContextFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        ctx = _RUN_CONTEXT.get({})
        if ctx:
            parts = []
            for key in ("task_id", "scenario_id", "model_id", "track_id", "init_id", "seed"):
                val = ctx.get(key)
                if val not in (None, ""):
                    parts.append(f"{key}={val}")
            record.run_context = f"[{' '.join(parts)}] " if parts else ""
        else:
            record.run_context = ""
        return True


def _normalize_level(level: str | int) -> int:
    if isinstance(level, int):
        return int(level)
    name = str(level).strip().upper()
    value = getattr(logging, name, None)
    if not isinstance(value, int):
        raise ValueError(f"Invalid log level: {level}")
    return int(value)


def _build_formatter() -> logging.Formatter:
    return logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(run_context)s%(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def _iter_named_handlers(logger: logging.Logger):
    for handler in logger.handlers:
        if getattr(handler, "_bench_handler_name", None):
            yield handler


def _resolve_log_file(run_dir: Optional[Path], log_file: Optional[Path], log_to_file: bool) -> Optional[Path]:
    if log_file is not None:
        p = Path(log_file).expanduser()
        if not p.is_absolute():
            p = (run_dir / p) if run_dir is not None else Path.cwd() / p
        return p.resolve()
    if not log_to_file or run_dir is None:
        return None
    return (run_dir / "logs" / "bench.log").resolve()


def configure_logging(
    level: str,
    *,
    run_dir: Optional[Path],
    log_to_file: bool,
    quiet: bool = False,
    log_file: Optional[Path] = None,
) -> None:
    """
    Configure the shared `bench` logger.

    Safe to call multiple times. Console/file handlers are updated in place and
    duplicate handlers are not added.
    """
    logger = logging.getLogger(_LOGGER_NAME)
    logger.setLevel(_normalize_level(level))
    logger.propagate = False

    formatter = _build_formatter()
    context_filter = _RunContextFilter()

    console_handler: Optional[logging.Handler] = None
    file_handler: Optional[logging.Handler] = None
    for handler in list(_iter_named_handlers(logger)):
        handler.setFormatter(formatter)
        if not any(isinstance(flt, _RunContextFilter) for flt in handler.filters):
            handler.addFilter(context_filter)
        if getattr(handler, "_bench_handler_name", None) == _CONSOLE_HANDLER_NAME:
            console_handler = handler
        elif getattr(handler, "_bench_handler_name", None) == _FILE_HANDLER_NAME:
            file_handler = handler

    if quiet:
        if console_handler is not None:
            logger.removeHandler(console_handler)
            console_handler.close()
            console_handler = None
    elif console_handler is None:
        console_handler = logging.StreamHandler(stream=sys.stderr)
        console_handler._bench_handler_name = _CONSOLE_HANDLER_NAME  # type: ignore[attr-defined]
        console_handler.setFormatter(formatter)
        console_handler.addFilter(context_filter)
        logger.addHandler(console_handler)

    resolved_log_file = _resolve_log_file(run_dir=run_dir, log_file=log_file, log_to_file=log_to_file)
    if resolved_log_file is None:
        if file_handler is not None:
            logger.removeHandler(file_handler)
            file_handler.close()
        return

    resolved_log_file.parent.mkdir(parents=True, exist_ok=True)
    if file_handler is not None:
        existing_path = getattr(file_handler, "_bench_log_path", None)
        if existing_path == str(resolved_log_file):
            file_handler.setFormatter(formatter)
            return
        logger.removeHandler(file_handler)
        file_handler.close()

    new_file_handler = logging.FileHandler(str(resolved_log_file), mode="a", encoding="utf-8")
    new_file_handler._bench_handler_name = _FILE_HANDLER_NAME  # type: ignore[attr-defined]
    new_file_handler._bench_log_path = str(resolved_log_file)  # type: ignore[attr-defined]
    new_file_handler.setFormatter(formatter)
    new_file_handler.addFilter(context_filter)
    logger.addHandler(new_file_handler)


def get_logger(name: str = _LOGGER_NAME) -> logging.Logger:
    if name == _LOGGER_NAME or name.startswith(f"{_LOGGER_NAME}."):
        return logging.getLogger(name)
    return logging.getLogger(f"{_LOGGER_NAME}.{name}")


def get_effective_level(name: str = _LOGGER_NAME) -> int:
    return int(get_logger(name).getEffectiveLevel())


def is_debug_enabled(name: str = _LOGGER_NAME) -> bool:
    return get_effective_level(name) <= logging.DEBUG


def set_logging_context(**fields: object) -> contextvars.Token[Dict[str, str]]:
    current = dict(_RUN_CONTEXT.get({}))
    for key, value in fields.items():
        if value is None:
            current.pop(str(key), None)
        else:
            current[str(key)] = str(value)
    return _RUN_CONTEXT.set(current)


def clear_logging_context() -> None:
    _RUN_CONTEXT.set({})


@contextlib.contextmanager
def logging_context(**fields: object) -> Iterator[None]:
    token = set_logging_context(**fields)
    try:
        yield
    finally:
        _RUN_CONTEXT.reset(token)


def env_log_level(default: str = "INFO") -> str:
    return str(os.environ.get("BENCH_LOG_LEVEL", default)).strip().upper()
