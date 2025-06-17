from __future__ import annotations
"""Backward-compat shim – import from new engine_pool."""
from .engine_pool import ENGINE_POOL, EnginePool  # re-export

__all__ = ["ENGINE_POOL", "EnginePool"] 