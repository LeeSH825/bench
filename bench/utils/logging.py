from __future__ import annotations

import logging
from typing import Optional


def get_logger(name: str = "bench", level: int = logging.INFO) -> logging.Logger:
    """
    Standard logger factory.

    Step 2: simple console logger.
    Step 4~: runner will also tee stdout/stderr into run_dir logs.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(level)
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False
    return logger

