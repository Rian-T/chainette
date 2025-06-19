from __future__ import annotations
"""Lightweight manager that spawns / reuses external engine processes.

Currently supports vLLM Serve (`backend == "vllm_api"`). For other back-ends
no-op. The implementation is intentionally minimal (<80 LOC).
"""

import os
import subprocess
import time
from typing import Optional, Sequence, Union, TYPE_CHECKING

import httpx

if TYPE_CHECKING:
    from .registry import EngineConfig

__all__ = ["ensure_running", "maybe_stop"]

# Simple map cfg.name -> process handle to avoid duplicates
_PROCESSES: dict[str, subprocess.Popen] = {}


def _default_port(index: int) -> int:
    """Return default port 8000+index to avoid collisions."""
    return 8000 + index


def _build_serve_cmd(cfg: 'EngineConfig', port: int) -> list[str]:  # noqa: D401
    """Assemble the `vllm serve` command for *cfg* on *port*."""
    cmd = [
        "vllm",
        "serve",
        cfg.model,
        "--port",
        str(port),
    ]

    # Auto-map common fields → CLI flags
    if cfg.dtype:
        cmd += ["--dtype", str(cfg.dtype)]
    if cfg.gpu_memory_utilization is not None:
        cmd += ["--gpu-memory-utilization", str(cfg.gpu_memory_utilization)]
    if cfg.max_model_len is not None:
        cmd += ["--max-model-len", str(cfg.max_model_len)]
    if cfg.tensor_parallel_size is not None:
        cmd += ["--tensor-parallel-size", str(cfg.tensor_parallel_size)]

    # Reasoning support
    if cfg.enable_reasoning:
        cmd.append("--enable-reasoning")
    if cfg.reasoning_parser:
        cmd += ["--reasoning-parser", cfg.reasoning_parser]

    # Extra flags provided by the user take precedence
    cmd.extend(cfg.extra_serve_flags or [])
    return cmd


def _wait_until_healthy(cfg: 'EngineConfig', proc: subprocess.Popen):
    """Poll the /health endpoint until it returns 200 or timeout."""
    if not cfg.endpoint:
        return

    deadline = time.time() + cfg.startup_timeout
    wait = 1.0  # initial wait time for backoff

    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"vLLM server process for '{cfg.name}' exited prematurely.")

        try:
            # vLLM's OpenAI-compatible server uses /v1/health
            health_url = cfg.endpoint.replace("/v1", "") + "/health"
            r = httpx.get(health_url, timeout=2.0)
            if r.status_code == 200:
                return  # Server is ready
        except httpx.RequestError:
            pass  # Ignore connection errors, server is likely starting up

        time.sleep(wait)
        wait = min(wait * 2, 16)  # exponential backoff up to 16s

    # If loop finishes, timeout was reached
    raise RuntimeError(
        f"vLLM server '{cfg.name}' did not become healthy in {cfg.startup_timeout}s."
    )


def ensure_running(cfg: 'EngineConfig') -> str:
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
    cmd = _build_serve_cmd(cfg, port)

    # Environment variables (CUDA device, user extras)
    env = {**os.environ, **(cfg.extra_env or {})}
    gpu_ids = getattr(cfg, "gpu_ids", None)
    if gpu_ids is not None:
        if isinstance(gpu_ids, (list, tuple)):
            gpu_ids = ",".join(map(str, gpu_ids))
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_ids)

    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env, preexec_fn=os.setsid)  # noqa: S603,S607
    _PROCESSES[cfg.name] = proc
    cfg.process = proc  # save handle in config
    cfg.endpoint = f"http://localhost:{port}/v1"

    _wait_until_healthy(cfg, proc)
    return cfg.endpoint


def maybe_stop(cfg: 'EngineConfig'):
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