from __future__ import annotations

"""Simple engine registry for Chainette.

Currently only supports vLLM engines, but is designed to be easily extended.
"""

import asyncio  # Add this import
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

_Any = Any  # simple alias for forward references without importing heavy libs

# Public exports for `import *`
__all__ = [
    "EngineConfig",
    "register_engine",
    "get_engine_config",
    "load_engines_from_yaml",
]

# Global in-memory store of engine configurations.
_REGISTRY: Dict[str, "EngineConfig"] = {}

# `_is_vllm_model` used to decide whether to instantiate an in-process vLLM.
# (Removed)


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

    # Engine backend: "vllm_api", "openai", "ollama_api", "ollama" (legacy)
    backend: str = "vllm_api"

    # Process-lifecycle / serve flags (Phase D)
    lazy: bool = True  # if False -> spawn at chain start & keep until end
    port: Optional[int] = None  # preferred port when spawning a vllm serve
    extra_serve_flags: list[str] = field(default_factory=list)
    extra_env: Dict[str, str] = field(default_factory=dict)

    # Additional, engine-specific kwargs
    extra: Dict[str, Any] = field(default_factory=dict)

    # Internal cache for the instantiated engine object
    _engine: Optional[_Any] = field(init=False, default=None, repr=False)

    # Handle to subprocess when Chainette spawns an external server
    process: Optional[Any] = field(default=None, repr=False, compare=False)

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
            if self.backend == "ollama":
                self._engine = self._create_ollama_engine()
            elif self.backend == "vllm_api":
                from chainette.engine.http_client import VLLMClient
                self._engine = VLLMClient(self.endpoint, self.model)
            elif self.backend == "openai":
                from chainette.engine.http_client import OpenAIClient
                self._engine = OpenAIClient(self.endpoint, self.api_key, self.model)
            elif self.backend == "ollama_api":
                from chainette.engine.http_client import OllamaHTTPClient
                self._engine = OllamaHTTPClient(self.endpoint, self.model)
            else:
                raise ValueError(f"Unsupported backend '{self.backend}'.")

            # ----- UI notification ------------------------------------------------
            try:
                from chainette.utils.logging import console  # noqa: WPS433

                console.print(
                    f"[green]🚀 Started engine '[bold]{self.name}[/]' ({self.backend})[/]"
                )
            except Exception:  # pragma: no cover
                pass
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

            # ----- UI notification ------------------------------------------------
            try:
                from chainette.utils.logging import console  # noqa: WPS433

                console.print(
                    f"[red]🗑️ Released engine '[bold]{self.name}[/]' ({self.backend})[/]"
                )
            except Exception:
                pass
        else:
            pass # No active engine instance to release

    # -------------------------------------------------- #
    # Private helpers
    # -------------------------------------------------- #

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

    # Default backend fallback -> HTTP vLLM
    if "backend" not in cfg_kwargs:
        cfg_kwargs["backend"] = "vllm_api"

    # Deprecation shim for legacy configs
    if cfg_kwargs.get("backend") == "vllm_local":
        raise ValueError("backend 'vllm_local' is no longer supported. Use 'vllm_api' instead.")

    cfg = EngineConfig(**cfg_kwargs)

    # Warn if reasoning requested but backend lacks support
    if cfg.enable_reasoning and cfg.backend != "vllm_api":
        import warnings
        warnings.warn(
            f"enable_reasoning is not supported for backend '{cfg.backend}'. The flag will be ignored.",
            UserWarning,
        )
        cfg.enable_reasoning = False

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