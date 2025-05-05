from __future__ import annotations

"""chainette.engine.runtime
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Run‑time management of vLLM server processes.

This module is deliberately *simple*—just enough to support the
`warmup`, `run`, and `kill` CLI commands while remaining easy to read.
It relies on :pymod:`chainette.engine.registry` for static configuration.

High‑level responsibilities
===========================
1. **Spawn** a vLLM server given an :class:`EngineConfig`.
2. **Reuse** an already‑running server when its configuration hash matches.
3. **Persist** lightweight metadata in `~/.cache/chainette/engines/<name>.json`.
4. **Gracefully kill** servers on demand.

Backend assumptions:
* vLLM is installed and available as the `python -m vllm.entrypoints.api_server` module.
* The HTTP API is available at `/v1/models` for health checks.

No other sophisticated orchestration (e.g. load balancing) is attempted.
"""

from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

try:
    import requests  # noqa: WPS433 (3rd‑party is fine here)
except ModuleNotFoundError as exc:  # pragma: no cover – only raised in dev envs
    raise ImportError("The 'requests' package is required for chainette.engine.runtime") from exc

from chainette.engine.registry import EngineConfig, get_engine_config

__all__ = [
    "LiveEngine",
    "spawn_engine",
    "kill_engine",
    "kill_all_engines",
]

# ---------------------------------------------------------------------------
# Basics – cache directory helpers
# ---------------------------------------------------------------------------


_CACHE_DIR = Path.home() / ".cache" / "chainette" / "engines"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _cache_path(name: str) -> Path:  # noqa: D401
    """Return the JSON cache path for *name*."""

    return _CACHE_DIR / f"{name}.json"


# ---------------------------------------------------------------------------
# Helpers – ports, pids, health check
# ---------------------------------------------------------------------------


def _find_free_port(start: int = 8000, max_tries: int = 100) -> int:  # noqa: WPS432
    """Return an unused TCP port starting from *start*.

    We iterate at most *max_tries* ports.  This is adequate for local
    workflows and keeps code short.
    """

    for port in range(start, start + max_tries):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:  # noqa: WPS442
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:  # port in use
                continue
    raise RuntimeError("No free port found for vLLM server")


def _pid_alive(pid: int) -> bool:  # noqa: D401
    """Return ``True`` if *pid* is running and we can signal it."""

    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _health_check(port: int, timeout: float = 2.0) -> bool:
    """Ping `/v1/models` to verify server readiness."""

    try:
        resp = requests.get(f"http://127.0.0.1:{port}/v1/models", timeout=timeout)
        return resp.status_code == 200
    except requests.RequestException:
        return False


# ---------------------------------------------------------------------------
# In‑memory structure for live engines
# ---------------------------------------------------------------------------


@dataclass
class LiveEngine:  # noqa: D401
    """Runtime information about a launched engine."""

    name: str
    config: EngineConfig
    pid: int
    port: int

    def is_healthy(self) -> bool:  # noqa: D401
        return _pid_alive(self.pid) and _health_check(self.port)

    # ------------------------------------------------------
    # serialisation helpers
    # ------------------------------------------------------

    def to_dict(self) -> Dict[str, object]:
        return {
            "pid": self.pid,
            "port": self.port,
            "cfg_hash": self.config.cfg_hash(),
            "devices": self.config.devices,
            "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }

    @staticmethod
    def from_cache(name: str, cfg: EngineConfig, data: Dict[str, object]) -> "LiveEngine":  # noqa: D401
        return LiveEngine(
            name=name,
            config=cfg,
            pid=int(data["pid"]),
            port=int(data["port"]),
        )


# ---------------------------------------------------------------------------
# Core – spawn / reuse logic
# ---------------------------------------------------------------------------


_LIVE: Dict[str, LiveEngine] = {}


def spawn_engine(cfg: EngineConfig, *, force: bool = False, wait: float = 60.0) -> LiveEngine:  # noqa: WPS231
    """Ensure a vLLM server matching *cfg* is running.

    If a live server with the same *name* **and** matching `cfg_hash` exists,
    we reuse it.  Otherwise we launch a new process.

    Parameters
    ----------
    cfg:
        The engine configuration.
    force:
        When ``True`` we kill any existing server with the same *name* and
        start fresh, regardless of its config hash.
    wait:
        Seconds to wait for the health‑check to pass.
    """

    # 1) attempt cache reuse --------------------------------------------------
    cache_file = _cache_path(cfg.name)
    if cache_file.exists() and not force:
        data = json.loads(cache_file.read_text())
        if cfg.cfg_hash() == data.get("cfg_hash") and _pid_alive(int(data["pid"])):
            live = LiveEngine.from_cache(cfg.name, cfg, data)
            if _health_check(live.port):
                _LIVE[cfg.name] = live
                return live  # reuse

    # 2) if here → need fresh process ----------------------------------------
    if cache_file.exists():
        try:
            _terminate_pid(int(json.loads(cache_file.read_text())["pid"]))
        except Exception:  # noqa: WPS430 – best effort
            pass
        cache_file.unlink(missing_ok=True)

    port = _find_free_port()
    cmd = _build_vllm_command(cfg, port)

    process = subprocess.Popen(  # noqa: WPS316
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    # 3) poll for readiness ---------------------------------------------------
    deadline = time.time() + wait
    while time.time() < deadline:
        if _health_check(port):
            break
        if process.poll() is not None:  # crashed
            raise RuntimeError(f"vLLM server for '{cfg.name}' exited early. Check stderr:\n{process.stderr.read()}")
        time.sleep(0.5)
    else:
        process.kill()
        raise TimeoutError(f"Timed out waiting for vLLM server '{cfg.name}' to become ready")

    live = LiveEngine(cfg.name, cfg, process.pid, port)
    cache_file.write_text(json.dumps(live.to_dict(), indent=2))
    _LIVE[cfg.name] = live
    return live


# ---------------------------------------------------------------------------
# Kill helpers
# ---------------------------------------------------------------------------


def kill_engine(name: str, *, timeout: float = 5.0) -> None:
    """Terminate the engine named *name* if running."""

    cache_file = _cache_path(name)
    if not cache_file.exists():
        return

    data = json.loads(cache_file.read_text())
    pid = int(data["pid"])
    _terminate_pid(pid, timeout=timeout)
    cache_file.unlink(missing_ok=True)
    _LIVE.pop(name, None)


def kill_all_engines() -> None:
    """Terminate *all* cached engines."""

    for cache in list(_CACHE_DIR.glob("*.json")):
        name = cache.stem
        kill_engine(name)


# ---------------------------------------------------------------------------
# Internal utilities
# ---------------------------------------------------------------------------


def _terminate_pid(pid: int, *, timeout: float = 5.0) -> None:  # noqa: D401
    """Gracefully terminate a process, fallback to SIGKILL."""

    if not _pid_alive(pid):
        return

    try:
        os.kill(pid, signal.SIGTERM)
        deadline = time.time() + timeout
        while time.time() < deadline:
            if not _pid_alive(pid):
                return
            time.sleep(0.1)
        os.kill(pid, signal.SIGKILL)
    except OSError:
        pass  # already dead


def _build_vllm_command(cfg: EngineConfig, port: int) -> List[str]:  # noqa: D401
    """Translate *cfg* into a `python -m vllm.entrypoints.api_server` command."""

    cmd: List[str] = [
        sys.executable,
        "-m",
        "vllm.entrypoints.api_server",
        "--model",
        cfg.model,
        "--port",
        str(port),
        "--dtype",
        cfg.dtype,
        "--gpu-memory-utilization",
        str(cfg.gpu_memory_utilization),
        "--gpu-ids",
        ",".join(str(i) for i in cfg.devices),
    ]

    if cfg.enable_reasoning:
        cmd += ["--enable-reasoning"]
        if cfg.reasoning_parser:
            cmd += ["--reasoning-parser", cfg.reasoning_parser]

    if len(cfg.devices) > 1:
        cmd += ["--tensor-parallel-size", str(len(cfg.devices))]

    # extra kwargs (truthy only)
    for k, v in cfg.extra.items():
        flag = f"--{k.replace('_', '-')}"
        if isinstance(v, bool):
            if v:
                cmd.append(flag)
        else:
            cmd += [flag, str(v)]

    return cmd
