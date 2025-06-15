from __future__ import annotations
"""Context builder for Jinja template rendering (≤50 LOC)."""
from typing import Any, Dict
from pydantic import BaseModel

__all__ = ["build_context"]

def build_context(item_history: Dict[str, Any]) -> Dict[str, Any]:  # noqa: D401
    """Flatten nested `item_history` into a template-friendly dict.

    Example:
        {"qa": QAOut(answer="yes")}
        → {"qa": QAOut(...), "qa.answer": "yes"}
    """
    ctx: Dict[str, Any] = {}
    for key, value in item_history.items():
        if isinstance(value, BaseModel):
            d = value.model_dump()
            for field_name, field_val in d.items():
                ctx[f"{key}.{field_name}"] = field_val
            ctx[key] = value
        else:
            ctx[key] = value
    return ctx 