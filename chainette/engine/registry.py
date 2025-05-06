from __future__ import annotations

"""chainette.engine.registry
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Central registry for *static* LLM engine configurations.

Public API
==========
* **`EngineConfig`** – validated data model for engine settings.
* **`register_engine()`** – store or update an engine configuration.
* **`get_engine_config()`** – fetch a configuration by name.
* **`load_engines_from_yaml()`** – bulk‑load configurations from a YAML file.

This module defines *configuration only*; actual server lifecycle
(start / reuse / shutdown) is handled in :pymod:`chainette.engine.runtime`.
"""

from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Optional

import yaml
from pydantic import BaseModel, Field, PositiveFloat, field_validator

__all__ = [
    "EngineConfig",
    "register_engine",
    "get_engine_config",
    "load_engines_from_yaml",
]

# ---------------------------------------------------------------------------
# Pydantic model
# ---------------------------------------------------------------------------


class EngineConfig(BaseModel):
    """Validated static description of an LLM back‑end.

    Only fields that *actually affect* the vLLM command line (or future
    back‑ends) are included.  Simplicity is paramount.
    """

    name: str = Field(..., description="Unique identifier to reference this engine")
    model: str = Field(..., description="Model name or HF repository path")

    # Hardware / memory tuning ------------------------------------------------
    dtype: str = Field("auto", description="torch dtype string accepted by vLLM")
    gpu_memory_utilization: PositiveFloat = Field(
        0.9, ge=0, le=1, description="Fraction of GPU RAM to allocate"
    )
    devices: List[int] = Field(
        default_factory=lambda: [0],
        description="CUDA device indices for this engine (tensor parallel if >1)",
    )
    max_model_len: Optional[int] = Field(
        None, description="Maximum sequence length the model can handle"
    )

    # Reasoning & other toggles ----------------------------------------------
    enable_reasoning: bool = False
    reasoning_parser: Optional[str] = None
    lazy: bool = False  # defer spawn until first use

    extra: Dict[str, Any] = Field(
        default_factory=dict,
        description="Backend‑specific kwargs passed verbatim to the Engine",
    )

    # ---------------------------------------------------------------------
    # derived / validated fields
    # ---------------------------------------------------------------------

    @field_validator("reasoning_parser")
    @classmethod
    def check_reasoning_parser(cls, v, values):  # noqa: ANN001,N805
        if v is not None and values.data.get("enable_reasoning") is False:
            raise ValueError("reasoning_parser set but enable_reasoning is False")
        return v

    def cfg_hash(self) -> str:
        """Deterministic hash used to detect restart needs.

        We ignore *name* because you can re‑register the same engine under a
        different name without forcing a restart.
        """

        import hashlib, json  # local import keeps global namespace tidy

        payload = {
            k: getattr(self, k)
            for k in (
                "model",
                "dtype",
                "gpu_memory_utilization",
                "devices",
                "enable_reasoning",
                "reasoning_parser",
                "extra",
            )
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, EngineConfig):
            return NotImplemented
        return (
            self.name == other.name and
            self.model == other.model and
            sorted(self.devices) == sorted(other.devices) and
            self.lazy == other.lazy and
            self.dtype == other.dtype and
            self.max_model_len == other.max_model_len and
            self.gpu_memory_utilization == other.gpu_memory_utilization and
            self.enable_reasoning == other.enable_reasoning and
            self.reasoning_parser == other.reasoning_parser and
            self.extra == other.extra
        )

    def __hash__(self) -> int:
        return hash((
            self.name,
            self.model,
            tuple(sorted(self.devices)),
            self.lazy,
            self.dtype,
            self.max_model_len,
            self.gpu_memory_utilization,
            self.enable_reasoning,
            self.reasoning_parser,
            tuple(sorted(self.extra.items()))
        ))


# ---------------------------------------------------------------------------
# Global registry
# ---------------------------------------------------------------------------


_REGISTRY: MutableMapping[str, EngineConfig] = {}


def register_engine(**kwargs: Any) -> EngineConfig:  # noqa: D401, ANN001
    """Register or update an :class:`EngineConfig`.

    Returns the validated config instance so the caller can re‑use it.
    """

    config = EngineConfig(**kwargs)
    _REGISTRY[config.name] = config
    return config


def get_engine_config(name: str) -> EngineConfig:
    """Fetch a config by name, raising `KeyError` if missing."""

    try:
        return _REGISTRY[name]
    except KeyError as exc:  # re‑raise with nicer message
        raise KeyError(f"Engine '{name}' is not registered. Did you call register_engine()?") from exc


# ---------------------------------------------------------------------------
# YAML helper
# ---------------------------------------------------------------------------


def load_engines_from_yaml(path: str | Path) -> List[EngineConfig]:
    """Load multiple engine configs from a YAML file.

    YAML schema:
    ```yaml
    engines:
      - name: qwen_r1
        model: deepseek‑ai/DeepSeek‑R1‑Distill‑Qwen‑7B
        dtype: bfloat16
        gpu_memory_utilization: 0.45
        enable_reasoning: true
        reasoning_parser: deepseek_r1
        devices: [0, 1]
        lazy: true
    ```
    """

    path = Path(path)
    data = yaml.safe_load(path.read_text())

    if not data or "engines" not in data:
        raise ValueError("YAML file must contain a top‑level 'engines' key")

    configs: List[EngineConfig] = []
    for raw in data["engines"]:
        cfg = register_engine(**raw)
        configs.append(cfg)
    return configs
