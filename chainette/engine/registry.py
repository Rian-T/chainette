from __future__ import annotations

"""Simple engine registry for Chainette.

Currently only supports vLLM engines, but is designed to be easily extended.
"""

import asyncio  # Add this import
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

from vllm import LLM

__all__ = [
    "EngineConfig",
    "register_engine",
    "get_engine_config",
    "load_engines_from_yaml",
]


_REGISTRY: Dict[str, "EngineConfig"] = {}


def _is_vllm_model(model: str) -> bool:
    """Very naive detection whether *model* should be loaded with vLLM."""
    return True  # For now we assume everything is vLLM – keeps the code tiny.


@dataclass
class EngineConfig:  # noqa: D101 – self-documenting via fields
    name: str
    model: str
    dtype: Optional[str] = None
    gpu_memory_utilization: Optional[float] = None
    max_model_len: Optional[int] = None
    tensor_parallel_size: Optional[int] = None

    # Reasoning
    enable_reasoning: bool = False
    reasoning_parser: Optional[str] = None

    # Engine backend: "vllm" (default) or "ollama"
    backend: str = "vllm"

    # Additional, engine-specific kwargs
    extra: Dict[str, Any] = field(default_factory=dict)

    # Internal cache for the instantiated engine object
    _engine: Optional[LLM] = field(init=False, default=None, repr=False)

    # HTTP-specific fields
    endpoint: Optional[str] = None
    api_key: Optional[str] = None

    # -------------------------------------------------- #
    # Public helpers
    # -------------------------------------------------- #

    @property
    def engine(self):
        """Return the instantiated engine (lazy-loaded once)."""
        if self._engine is None:
            if self.backend == "vllm":
                self._engine = self._create_vllm_engine()
            elif self.backend == "ollama":
                self._engine = self._create_ollama_engine()
            else:
                raise ValueError(f"Unsupported backend '{self.backend}'.")
        return self._engine

    def release_engine(self):
        """Release the cached engine instance to free resources."""
        if self._engine is not None:
            engine_to_release = self._engine
            self._engine = None  # Remove reference from this config object.

            # Try to explicitly shut down the vLLM engine's components
            if hasattr(engine_to_release, 'llm_engine'):
                if hasattr(engine_to_release.llm_engine, 'model_executor'):
                    del engine_to_release.llm_engine.model_executor
                del engine_to_release.llm_engine
            
            del engine_to_release
        else:
            pass # No active engine instance to release

    # -------------------------------------------------- #
    # Private helpers
    # -------------------------------------------------- #

    def _create_vllm_engine(self):
        """Instantiate a vLLM LLM object from this config."""
        kwargs: Dict[str, Any] = {
            "model": self.model,
        }
        if self.dtype:
            kwargs["dtype"] = self.dtype
        if self.gpu_memory_utilization is not None:
            kwargs["gpu_memory_utilization"] = self.gpu_memory_utilization
        if self.max_model_len is not None:
            kwargs["max_model_len"] = self.max_model_len
        if self.tensor_parallel_size is not None:
            kwargs["tensor_parallel_size"] = self.tensor_parallel_size
        if self.enable_reasoning:
            kwargs["enable_reasoning"] = True
            if self.reasoning_parser:
                kwargs["reasoning_parser"] = self.reasoning_parser

        # Merge in extra (last so callers can override anything)
        kwargs.update(self.extra)

        return LLM(**kwargs)

    def _create_ollama_engine(self):
        """Instantiate an Ollama engine wrapper matching vLLM interface."""
        try:
            from chainette.engine.ollama_client import OllamaLLM
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                "Ollama backend requested but 'ollama' package is not installed.\n"
                "Install with: pip install chainette[ollama] or poetry add --optional ollama"
            ) from e

        return OllamaLLM(model=self.model)

    # -------------------------------------------------- #

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable representation (metadata only)."""
        d = self.__dict__.copy()
        d.pop("_engine", None)
        return d


# --------------------------------------------------------------------------- #
# Registry helpers
# --------------------------------------------------------------------------- #

def register_engine(name: str, **kwargs):  # noqa: D401 – simple factory
    """Register a new engine configuration.

    The keyword arguments map to :class:`EngineConfig` fields. Unknown keys are
    stored in *extra* so we remain forward-compatible.
    """
    known_fields = {f.name for f in EngineConfig.__dataclass_fields__.values()}  # type: ignore[arg-type]
    cfg_kwargs = {k: v for k, v in kwargs.items() if k in known_fields}
    extra = {k: v for k, v in kwargs.items() if k not in known_fields}
    cfg_kwargs["extra"] = extra
    cfg_kwargs["name"] = name

    # Default backend fallback
    if "backend" not in cfg_kwargs:
        cfg_kwargs["backend"] = "vllm"

    cfg = EngineConfig(**cfg_kwargs)
    _REGISTRY[name] = cfg
    return cfg


def get_engine_config(name: str) -> EngineConfig:
    if name not in _REGISTRY:
        raise KeyError(f"Engine '{name}' is not registered.")
    return _REGISTRY[name]


def load_engines_from_yaml(path: str | bytes | Any):  # pragma: no cover – rarely used
    """Load multiple engine configs from a YAML file."""
    import yaml

    data = yaml.safe_load(Path(path).read_text())  # type: ignore[arg-type]
    for item in data:
        register_engine(**item)