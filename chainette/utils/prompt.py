from __future__ import annotations
"""Prompt rendering helpers (≤80 LOC)."""
from typing import Any, Dict, List
from transformers import AutoTokenizer
from .templates import render
from .context import build_context
from ..engine.registry import get_engine_config

__all__ = ["build_prompt"]


def build_prompt(step, item_history: Dict[str, Any]) -> str:  # noqa: D401
    """Build final prompt string for *step* and *item_history*."""
    ctx = build_context(item_history)

    messages: List[Dict[str, str]] = []
    if step.system_prompt:
        messages.append({"role": "system", "content": render(step.system_prompt, ctx)})
    messages.append({"role": "user", "content": render(step.user_prompt, ctx)})

    if not messages:
        return ""

    cfg = get_engine_config(step.engine_name)
    if getattr(cfg, "backend", "vllm") == "ollama":
        prompt_lines = [f"## {m['role']}\n{m['content']}" for m in messages]
        prompt_lines.append("## assistant\n")
        return "\n\n".join(prompt_lines)

    # vLLM path
    if step.tokenizer is None:
        step.tokenizer = AutoTokenizer.from_pretrained(cfg.model)
    return step.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) 