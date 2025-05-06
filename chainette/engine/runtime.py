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

import json
import logging
import os
import signal
import socket
import subprocess
import sys
import threading
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

_LOGS_DIR = Path.home() / ".cache" / "chainette" / "logs"
_LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Set up basic logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("chainette.engine")


def _get_engine_logger(name: str) -> logging.Logger:
    """Create a logger for the specified engine with file handler."""
    log_file = _LOGS_DIR / f"{name}.log"
    engine_logger = logging.getLogger(f"chainette.engine.{name}")
    engine_logger.setLevel(logging.INFO)  # Ensure logger level is set

    # Prevent propagation to root logger (which might log to console)
    engine_logger.propagate = False

    # Remove existing handlers to avoid duplicates
    for handler in engine_logger.handlers[:]:
        engine_logger.removeHandler(handler)
        handler.close()  # Close handler before removing

    # Add file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        "%Y-%m-%d %H:%M:%S"
    ))
    engine_logger.addHandler(file_handler)
    return engine_logger


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
_SPAWN_LOCK = threading.Lock()  # Global lock for synchronizing port allocation and process spawning


def spawn_engine(cfg: EngineConfig, *, force: bool = False, wait: float = 3600.0) -> LiveEngine:  # noqa: WPS231
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

    logger.info(f"Ensuring engine '{cfg.name}' is running (force={force})")

    # Check _LIVE first, outside the main lock for a quick path if already in memory and healthy.
    # This requires _LIVE access to be somewhat atomic or for is_healthy to be robust.
    # For simplicity and stronger consistency, all checks and modifications related to
    # _LIVE and cache files will be under the _SPAWN_LOCK.
    
    with _SPAWN_LOCK:
        # Check in-memory _LIVE state first
        if cfg.name in _LIVE and not force:
            live_engine_in_memory = _LIVE[cfg.name]
            if live_engine_in_memory.config.cfg_hash() == cfg.cfg_hash() and live_engine_in_memory.is_healthy():
                logger.info(f"Reusing in-memory live and healthy engine '{cfg.name}' (PID: {live_engine_in_memory.pid}, Port: {live_engine_in_memory.port}).")
                return live_engine_in_memory

        # 1) attempt cache reuse from disk --------------------------------------------------
        cache_file = _cache_path(cfg.name)
        if cache_file.exists() and not force:
            try:
                data = json.loads(cache_file.read_text())
                if cfg.cfg_hash() == data.get("cfg_hash") and _pid_alive(int(data["pid"])):
                    live = LiveEngine.from_cache(cfg.name, cfg, data)
                    if _health_check(live.port): # Final health check before reuse
                        _LIVE[cfg.name] = live # Update _LIVE cache
                        logger.info(f"Reusing disk-cached, verified, and healthy engine '{cfg.name}' (PID: {live.pid}, Port: {live.port}).")
                        return live  # reuse
            except (json.JSONDecodeError, ValueError) as e:
                logger.warning(f"Error reading or parsing cache file for '{cfg.name}', will attempt to start fresh: {e}")
                # Proceed to terminate and start fresh if cache is corrupted

        # 2) if here → need fresh process ----------------------------------------
        # Terminate existing process if any (from cache or if force=True)
        existing_pid_to_terminate = None
        if cache_file.exists():
            try:
                data = json.loads(cache_file.read_text())
                existing_pid_to_terminate = int(data["pid"])
            except (json.JSONDecodeError, ValueError, KeyError) as e:
                logger.warning(f"Could not determine PID from cache file for '{cfg.name}' for termination: {e}")
            
            if existing_pid_to_terminate and _pid_alive(existing_pid_to_terminate):
                logger.info(f"Terminating existing process {existing_pid_to_terminate} for engine '{cfg.name}' (due to force, mismatch, or unhealthiness).")
                _terminate_pid(existing_pid_to_terminate)
            elif existing_pid_to_terminate:
                logger.info(f"Existing process for engine '{cfg.name}' (PID: {existing_pid_to_terminate} from cache) was not alive.")
            
            cache_file.unlink(missing_ok=True) # Clean up old cache file

        # Also remove from _LIVE if it was there but not reused (e.g. stale, unhealthy, or force)
        if cfg.name in _LIVE:
            stale_live_engine = _LIVE.pop(cfg.name)
            if stale_live_engine.pid != existing_pid_to_terminate and _pid_alive(stale_live_engine.pid):
                # This case might happen if _LIVE had a PID different from cache, and cache was preferred for termination.
                logger.info(f"Terminating stale in-memory tracked engine '{cfg.name}' (PID: {stale_live_engine.pid}) as well.")
                _terminate_pid(stale_live_engine.pid)


        port = _find_free_port()
        cmd, env_vars = _build_vllm_command(cfg, port)

        logger.info(f"Starting vLLM server for '{cfg.name}' on port {port}. Logs: {_LOGS_DIR}/{cfg.name}.log")

        engine_logger = _get_engine_logger(cfg.name)
        log_file_path = _LOGS_DIR / f"{cfg.name}.log"

        try:
            with open(log_file_path, 'a') as log_f:
                process_env = os.environ.copy()
                process_env.update(env_vars)
                if env_vars:
                    env_vars_str = ", ".join(f"{k}={v}" for k, v in env_vars.items())
                    logger.info(f"Using environment variables: {env_vars_str}")
                
                process_handle = subprocess.Popen(
                    cmd,
                    stdout=log_f,
                    stderr=log_f,
                    text=True,
                    env=process_env
                )
        except Exception as e:
            logger.error(f"Failed to start vLLM process for '{cfg.name}': {e}")
            raise

        logger.info(f"vLLM server process for '{cfg.name}' started with PID {process_handle.pid}")
        # The lock is held until Popen returns.

    # 3) poll for readiness (outside the main spawn lock for this part) ------
    # The 'process' variable from the locked section is now 'process_handle'
    logger.info(f"Waiting for vLLM server '{cfg.name}' on port {port} to become ready (timeout: {wait}s)...")
    deadline = time.time() + wait
    while time.time() < deadline:
        if _health_check(port):
            logger.info(f"vLLM server '{cfg.name}' is ready on port {port}")
            break
        if process_handle.poll() is not None:  # crashed
            logger.error(f"vLLM server for '{cfg.name}' exited early with code {process_handle.returncode}. Check logs: {log_file_path}")
            raise RuntimeError(f"vLLM server for '{cfg.name}' exited early. Check logs in {log_file_path}")
        time.sleep(0.5)
    else:
        process_handle.kill()
        logger.error(f"Timed out waiting for vLLM server '{cfg.name}' to become ready. Check logs: {log_file_path}")
        raise TimeoutError(f"Timed out waiting for vLLM server '{cfg.name}' to become ready")

    live = LiveEngine(cfg.name, cfg, process_handle.pid, port)
    
    # Safely update shared resources (_LIVE and cache file) after successful start and health check
    with _SPAWN_LOCK:
        cache_file = _cache_path(cfg.name) # Re-evaluate cache_path, though it's deterministic
        cache_file.write_text(json.dumps(live.to_dict(), indent=2))
        _LIVE[cfg.name] = live
    return live


# ---------------------------------------------------------------------------
# Kill helpers
# ---------------------------------------------------------------------------


def kill_engine(name: str, *, timeout: float = 5.0) -> None:
    """Terminate the engine named *name* if running."""
    with _SPAWN_LOCK: # Protect access to _LIVE and cache file
        cache_file = _cache_path(name)
        pid_to_kill = None
        
        if name in _LIVE:
            pid_to_kill = _LIVE[name].pid
            logger.info(f"Found engine '{name}' in _LIVE (PID: {pid_to_kill}). Preparing to terminate.")

        if cache_file.exists():
            try:
                data = json.loads(cache_file.read_text())
                pid_from_cache = int(data["pid"])
                if pid_to_kill is None:
                    pid_to_kill = pid_from_cache
                    logger.info(f"Found engine '{name}' in cache (PID: {pid_from_cache}). Preparing to terminate.")
                elif pid_to_kill != pid_from_cache:
                    logger.warning(f"PID for '{name}' in _LIVE ({pid_to_kill}) differs from cache ({pid_from_cache}). Will terminate PID from _LIVE if alive, then PID from cache if different and alive.")
                    if _pid_alive(pid_to_kill):
                         _terminate_pid(pid_to_kill, timeout=timeout)
                    pid_to_kill = pid_from_cache # Prioritize cache for the main termination attempt if different
                
                cache_file.unlink(missing_ok=True) # Remove cache file
            except (json.JSONDecodeError, ValueError, KeyError) as e:
                logger.warning(f"Error reading cache file for engine '{name}' during kill: {e}. Will still attempt to kill if PID known from _LIVE.")
                if cache_file.exists(): # Attempt to remove corrupted cache file
                    cache_file.unlink(missing_ok=True)
        
        if pid_to_kill and _pid_alive(pid_to_kill):
            logger.info(f"Terminating engine '{name}' (PID: {pid_to_kill}).")
            _terminate_pid(pid_to_kill, timeout=timeout)
        elif pid_to_kill:
            logger.info(f"Engine '{name}' (PID: {pid_to_kill}) was already terminated or PID was invalid.")
        else:
            logger.info(f"Engine '{name}' not found in _LIVE or cache for termination.")
            
        _LIVE.pop(name, None) # Remove from _LIVE cache


def kill_all_engines() -> None:
    """Terminate *all* cached engines."""
    # No need to acquire lock for list(_CACHE_DIR.glob("*.json")) itself
    # kill_engine will acquire the lock for each individual engine.
    cached_engine_names = [cache.stem for cache in list(_CACHE_DIR.glob("*.json"))]
    
    # Also consider engines that might only be in _LIVE (e.g., if cache was manually deleted)
    # Use a copy of keys for safe iteration if kill_engine modifies _LIVE
    live_engine_names = []
    with _SPAWN_LOCK: # Protect reading _LIVE keys
        live_engine_names = list(_LIVE.keys())

    all_names_to_kill = set(cached_engine_names + live_engine_names)
    
    if not all_names_to_kill:
        logger.info("No engines found in cache or live memory to kill.")
        return

    logger.info(f"Attempting to kill all ({len(all_names_to_kill)}) known engines: {', '.join(all_names_to_kill)}")
    for name in all_names_to_kill:
        kill_engine(name) # kill_engine handles its own locking


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


def _build_vllm_command(cfg: EngineConfig, port: int) -> tuple[List[str], Dict[str, str]]:  # noqa: D401
    """Translate *cfg* into a `vllm serve` command and environment variables.
    
    Returns:
        A tuple of (command_list, environment_variables)
    """

    # Build the vllm command portion
    vllm_cmd: List[str] = [
        "vllm",
        "serve",
        cfg.model,
        "--port",
        str(port),
        "--dtype",
        cfg.dtype,
        "--gpu-memory-utilization",
        str(cfg.gpu_memory_utilization),
        "--max-model-len",
        str(cfg.max_model_len) if cfg.max_model_len else "4096",
    ]

    if cfg.enable_reasoning:
        vllm_cmd += ["--enable-reasoning"]
    if cfg.reasoning_parser:
        vllm_cmd += ["--reasoning-parser", cfg.reasoning_parser]

    if len(cfg.devices) > 1:
        vllm_cmd += ["--tensor-parallel-size", str(len(cfg.devices))]

    # extra kwargs (truthy only)
    for k, v in cfg.extra.items():
        flag = f"--{k.replace('_', '-')}"
        if isinstance(v, bool):
            if v:
                vllm_cmd.append(flag)
        else:
            vllm_cmd += [flag, str(v)]

    # Create environment variables dict instead of prefixing the command
    env_vars = {}
    if cfg.devices:
        devices_str = ",".join(map(str, cfg.devices))
        env_vars["CUDA_VISIBLE_DEVICES"] = devices_str

    # print the command for debugging
    logger.debug(f"vllm command: {' '.join(vllm_cmd)}")
    logger.debug(f"Environment variables: {env_vars}")
    
    return vllm_cmd, env_vars
