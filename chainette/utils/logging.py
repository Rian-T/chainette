from __future__ import annotations
"""Centralised structured logger for Chainette (<40 LOC).

Usage:
    from chainette.utils.logging import log
    log.info("message", extra={"step": step_id})

It wraps Python's stdlib logging with RichHandler for pretty console output and
a default JSON formatter for file handlers if needed.
"""
from logging import Logger, getLogger, INFO, DEBUG, WARNING, ERROR, basicConfig
from rich.logging import RichHandler

__all__ = ["log", "get"]

# --------------------------------------------------------------------------- #
_LEVEL_MAP = {
    "info": INFO,
    "debug": DEBUG,
    "warning": WARNING,
    "error": ERROR,
}

# Configure root once with Rich pretty handler
basicConfig(
    level=INFO,
    format="%(message)s",
    datefmt="%H:%M:%S",
    handlers=[RichHandler(rich_tracebacks=True, markup=True)],
)

log: Logger = getLogger("chainette")


def get(level: str = "info") -> Logger:  # noqa: D401
    """Return a logger with *level* (str)."""
    lvl = _LEVEL_MAP.get(level.lower(), INFO)
    lg = getLogger("chainette")
    lg.setLevel(lvl)
    return lg
