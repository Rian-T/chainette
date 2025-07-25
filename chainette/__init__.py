"""Chainette: tiny, type-safe LLM pipeline composition.

Main components:
* `Step`: Single LLM task with input/output models
* `Chain`: Sequence of steps to execute
* `Branch`: Named sequence that can be used in parallel
* `apply`: Inject Python functions into your chain
"""

# Version info
__version__ = "0.1.0"

# Core components
from chainette.core.step import Step, SamplingParams, GuidedDecodingParams
from chainette.core.chain import Chain
from chainette.core.branch import Branch
from chainette.core.node import Node
from chainette.core.apply import ApplyNode, apply, Apply

# Engine registry functions
from chainette.engine.registry import (
    EngineConfig,
    register_engine,
    get_engine_config,
    load_engines_from_yaml,
)

# Utility re-exports
from chainette.utils.templates import render
from chainette.utils.ids import snake_case, new_run_id

# Export all important symbols
__all__ = [
    # Core classes
    "Step",
    "Chain",
    "Branch",
    "Node",
    "ApplyNode",
    "Apply",
    
    # Functions
    "apply",
    "register_engine",
    "get_engine_config",
    "load_engines_from_yaml",
    "render",
    "snake_case",
    "new_run_id",
    
    # vLLM re-exports
    "SamplingParams",
    "GuidedDecodingParams",
    
    # Config classes
    "EngineConfig",
]

# --------------------------------------------------------------------------- #
# Silence noisy third-party INFO logs (e.g. vllm import banner) before they run
# --------------------------------------------------------------------------- #
import logging as _logging

_logging.basicConfig(level=_logging.ERROR)
