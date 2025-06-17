from __future__ import annotations
"""Thin HTTP clients for remote LLM APIs (OpenAI, vLLM-serve, etc.).

Each client mimics the minimal interface expected by :pyclass:`chainette.core.step.Step`:

    engine.generate(prompts: list[str], sampling_params: SamplingParams) -> list[LLMOutput]

where *LLMOutput* only needs to expose ``outputs[0].text`` (optionally ``reasoning_content``).

Keeping these wrappers tiny allows painless addition of new back-ends.
"""

from typing import List, Any, Optional
from dataclasses import dataclass
import os

# Local fallback if the OpenAI package is missing – we raise at runtime.
try:
    import openai  # type: ignore
except ModuleNotFoundError:  # pragma: no cover – handled later
    openai = None  # type: ignore

__all__ = [
    "BaseHTTPClient",
    "OpenAIClient",
]


class _SimpleCompletion:  # noqa: D401 – internal struct
    """Minimal completion wrapper compatible with parse_llm_json."""

    def __init__(self, text: str):
        self.text = text
        # Keep interface parity with vLLM
        self.reasoning_content: Optional[str] = None


class _RequestOutput:  # noqa: D401 – mimic vllm.RequestOutput
    def __init__(self, text: str):
        self.outputs = [_SimpleCompletion(text)]


@dataclass(slots=True)
class BaseHTTPClient:  # noqa: D101 – base contract
    endpoint: str | None
    api_key: str | None
    model: str

    # ------------------------------------------------------------------ #
    def _build_headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}

    # ------------------------------------------------------------------ #
    def generate(self, *, prompts: List[str], sampling_params: Any | None = None):  # noqa: D401
        """Subclasses must implement."""
        raise NotImplementedError


# --------------------------------------------------------------------------- #
# OpenAI – official Python SDK (sync)                                          #
# --------------------------------------------------------------------------- #


class OpenAIClient(BaseHTTPClient):  # noqa: D101
    def __init__(self, endpoint: str | None, api_key: str | None, model: str):
        if openai is None:
            raise ModuleNotFoundError(
                "The 'openai' package is required for backend 'openai'.\n"
                "Install with: pip install openai"
            )
        super().__init__(endpoint, api_key or os.getenv("OPENAI_API_KEY"), model)
        # Build a lazily-instantiated client – avoids import cost if unused.
        self._client = openai.OpenAI(base_url=self.endpoint, api_key=self.api_key)  # type: ignore[arg-type]

    # ------------------------------------------------------------------ #
    def generate(self, *, prompts: List[str], sampling_params: Any | None = None):  # noqa: D401
        """Return list of objects exposing .outputs[0].text.

        We send each *prompt* as a single-message chat completion requesting
        **JSON response format**.  The content is returned verbatim so that
        Chainette's existing JSON→Pydantic validation continues to work.
        """
        temperature: float | None = None
        if sampling_params is not None and hasattr(sampling_params, "temperature"):
            temperature = sampling_params.temperature  # type: ignore[attr-defined]

        out: List[_RequestOutput] = []
        for prompt in prompts:
            resp = self._client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                response_format={"type": "json_object"},
            )
            text = resp.choices[0].message.content or ""
            out.append(_RequestOutput(text))
        return out 