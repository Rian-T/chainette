from __future__ import annotations

"""Runtime execution functionality for Chainette engines."""

from typing import Dict, Any, Optional
import asyncio

__all__ = ["execute_with_timeout", "EngineTimeout"]


class EngineTimeout(Exception):
    """Exception raised when an engine call times out."""
    pass


async def execute_with_timeout(coro, timeout_seconds: float):
    """Execute a coroutine with a timeout.
    
    Returns the result of the coroutine, or raises EngineTimeout if it times out.
    """
    try:
        return await asyncio.wait_for(coro, timeout=timeout_seconds)
    except asyncio.TimeoutError:
        raise EngineTimeout(f"Engine call timed out after {timeout_seconds} seconds")