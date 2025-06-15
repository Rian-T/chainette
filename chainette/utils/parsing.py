from __future__ import annotations
"""Utility helpers to parse LLM completion JSON -> pydantic models.
Keeps Step implementation small.
"""
from typing import Any, Optional, Tuple, Type
import json
from pydantic import BaseModel

__all__ = ["parse_llm_json"]


def _extract_reasoning(completion) -> Optional[str]:  # noqa: D401
    """Return reasoning text if present in *completion* object."""
    for attr in ("reasoning", "reasoning_content"):
        if hasattr(completion, attr):
            return str(getattr(completion, attr))
    return None


def parse_llm_json(
    output_model: Type[BaseModel],
    llm_output: Any,
    *,
    engine_name: str,
    step_id: str,
) -> Tuple[BaseModel, Optional[str]]:  # noqa: D401
    """Validate first completion against *output_model* JSON schema.

    Returns (parsed_object, reasoning_string | None).
    Raises ValueError on any validation/parsing error.
    """
    first = llm_output.outputs[0]
    text = first.text.strip()

    reasoning = _extract_reasoning(first)

    try:
        data = json.loads(text)
        obj = output_model.model_validate(data)
        return obj, reasoning
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Invalid JSON output from '{engine_name}' for step '{step_id}': {text}. Error: {e}"
        ) from e
    except Exception as exc:
        raise ValueError(
            f"Failed to validate output for step '{step_id}': {exc}\nModel: {engine_name}\nOutput: {text}"
        ) from exc 