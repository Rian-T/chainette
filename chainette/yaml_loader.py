from __future__ import annotations
"""Minimal YAML → Chain loader (≈50 LOC).

A declarative alternative to the operator DSL.  Example YAML:

```yaml
name: Demo Chain
batch_size: 2
steps:
  - qa_step             # symbol in provided *globals*
  - filter_node
  - - fr_branch         # list → parallel
    - es_branch
```

Usage:
    from chainette.yaml_loader import load_chain
    chain = load_chain(path="my.yml", symbols=globals())

This prototype intentionally avoids introspection: the YAML is just a tree of
strings/lists which are looked-up in *symbols* (or via import if they contain a
module path like "pkg.mod:obj").
"""
from importlib import import_module
from pathlib import Path
from typing import Any, Dict, List, Union

import yaml

from chainette.core.chain import Chain
from chainette.core.branch import Branch
from chainette.core.node import Node

# Optional jsonschema validation
try:
    from jsonschema import validate as _js_validate  # type: ignore
except ImportError:  # pragma: no cover
    _js_validate = None

__all__ = ["load_chain"]

StepLike = Union[Node, List[Branch]]


# --------------------------------------------------------------------------- #

def _resolve(name: str, symbols: Dict[str, Any]):  # noqa: D401
    """Return python object for *name* (look in *symbols* then import)."""
    if name in symbols:
        return symbols[name]
    if ":" in name:  # module:path style
        mod_name, attr = name.split(":", 1)
        mod = import_module(mod_name)
        return getattr(mod, attr)
    raise KeyError(f"Symbol '{name}' not found in symbols nor importable")


def _build_steps(data: Any, symbols: Dict[str, Any]) -> StepLike:  # noqa: D401
    if isinstance(data, list):
        # Could be parallel list or nested
        if data and isinstance(data[0], list):  # nested lists – recurse per item
            return [_build_steps(item, symbols) for item in data]  # type: ignore[list-item]
        # list of names = parallel branches
        return [_resolve(item, symbols) for item in data]  # type: ignore[list-item]
    # single step/branch name
    return _resolve(data, symbols)


def load_chain(path: str | Path, symbols: Dict[str, Any]) -> Chain:  # noqa: D401
    """Load YAML file at *path* into a Chain."""
    data = yaml.safe_load(Path(path).read_text())
    if _js_validate is not None:
        _js_validate(instance=data, schema=_SCHEMA)
    name = data.get("name", "YAML Chain")
    batch_size = data.get("batch_size", 1)
    steps_raw = data["steps"]
    steps: List[StepLike] = []
    for item in steps_raw:
        steps.append(_build_steps(item, symbols))
    return Chain(name=name, steps=steps, batch_size=batch_size)


# --------------------------------------------------------------------------- #
# Minimal JSON Schema for YAML files
# --------------------------------------------------------------------------- #

_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "required": ["steps"],
    "properties": {
        "name": {"type": "string"},
        "batch_size": {"type": "integer", "minimum": 1},
        "steps": {
            "type": "array",
            "items": {
                "anyOf": [
                    {"type": "string"},
                    {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 1,
                    },
                ]
            },
        },
    },
} 