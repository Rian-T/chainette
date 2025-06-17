from __future__ import annotations
"""Lightweight manager that spawns / reuses external engine processes.

Currently supports vLLM Serve (`backend == "vllm_api"`). For other back-ends
no-op. The implementation is intentionally minimal (<80 LOC).
"""

import os
import subprocess
import time
from typing import Optional

from .registry import EngineConfig

__all__ = ["ensure_running", "maybe_stop"]

# Simple map cfg.name -> process handle to avoid duplicates
_PROCESSES: dict[str, subprocess.Popen] = {}


def _default_port(index: int) -> int:
    """Return default port 8000+index to avoid collisions."""
    return 8000 + index


def ensure_running(cfg: EngineConfig) -> str:
    """Ensure an engine process for *cfg* is running, return base URL.

    Accepts any object exposing `.backend`, `.endpoint`, `.model`, `.name`.
    For test stubs that don't have these attributes we short-circuit.
    """
    if not hasattr(cfg, "backend"):
        return ""

    if cfg.backend != "vllm_api":
        return cfg.endpoint or ""

    # If endpoint already provided → assume external server exists.
    if cfg.endpoint:
        return cfg.endpoint

    # Already spawned?
    if cfg.name in _PROCESSES and _PROCESSES[cfg.name].poll() is None:
        port = cfg.port or _default_port(list(_PROCESSES.keys()).index(cfg.name))
        cfg.endpoint = f"http://localhost:{port}/v1"
        return cfg.endpoint

    # Spawn new vllm serve process
    try:
        import vllm  # noqa: F401  # verify extra installed
    except ModuleNotFoundError as e:  # pragma: no cover
        raise RuntimeError("vllm extra not installed – cannot auto-spawn server") from e

    port = cfg.port or _default_port(len(_PROCESSES))
    cmd = [
        os.environ.get("PYTHON", "python"),
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        cfg.model,
        "--port",
        str(port),
        *cfg.extra_serve_flags,
    ]

    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)  # noqa: S603,S607
    _PROCESSES[cfg.name] = proc
    cfg.process = proc  # save handle in config
    cfg.endpoint = f"http://localhost:{port}/v1"

    # Wait briefly for server to boot (simple sleep – can be improved with health check)
    time.sleep(2)
    return cfg.endpoint


def maybe_stop(cfg: EngineConfig):
    """Terminate process if Chainette spawned it and `lazy` is True.

    Safely returns if *cfg* is not a full EngineConfig (e.g. SimpleNamespace in unit tests).
    """
    if not hasattr(cfg, "name"):
        return

    proc: Optional[subprocess.Popen] = _PROCESSES.get(cfg.name)
    if proc is None:
        return
    if cfg.lazy is False:
        return  # persistent → caller manages lifecycle
    if proc.poll() is None:  # running
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
    _PROCESSES.pop(cfg.name, None)
    cfg.process = None 