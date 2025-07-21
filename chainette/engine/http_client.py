from __future__ import annotations
"""Thin HTTP clients for remote LLM APIs (OpenAI, vLLM-serve, etc.).

Each client mimics the minimal interface expected by :pyclass:`chainette.core.step.Step`:

    engine.generate(prompts: list[str], sampling_params: SamplingParams) -> list[LLMOutput]

where *LLMOutput* only needs to expose ``outputs[0].text`` (optionally ``reasoning_content``).

Keeping these wrappers tiny allows painless addition of new back-ends.
"""

from typing import List, Any, Optional, Sequence, Type
from dataclasses import dataclass
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pydantic import BaseModel
import openai

try:
    import ollama
except ModuleNotFoundError:
    ollama = None

__all__ = [
    "BaseHTTPClient",
    "OpenAIClient",
    "VLLMClient",
    "OllamaHTTPClient",
]


class _SimpleCompletion:
    """Minimal completion wrapper compatible with parse_llm_json."""

    def __init__(self, text: str):
        self.text = text
        self.reasoning_content: Optional[str] = None


class _RequestOutput:
    """Mimic vllm.RequestOutput."""
    
    def __init__(self, text: str):
        self.outputs = [_SimpleCompletion(text)]


@dataclass(slots=True)
class BaseHTTPClient:
    """Base contract for HTTP clients."""
    
    endpoint: str | None
    api_key: str | None
    model: str

    def generate(
        self,
        *,
        prompts: List[str],
        output_model: Type[BaseModel],
        sampling_params: Any | None = None,
        step_id: str | None = None,
    ):
        """Generate outputs for *prompts* enforcing guided JSON schema."""
        temperature = getattr(sampling_params, "temperature", None) if sampling_params else None
        
        msgs: List[Sequence[dict]] = [p if isinstance(p, list) else [{"role": "user", "content": p}] for p in prompts]
        results: List[str] = [None] * len(msgs)

        def _run(idx: int, m: Sequence[dict]):
            txt = self._send_chat(messages=list(m), temperature=temperature, output_model=output_model)
            return idx, txt

        with ThreadPoolExecutor(max_workers=min(8, len(msgs))) as pool:
            futures = [pool.submit(_run, i, m) for i, m in enumerate(msgs)]
            for fut in as_completed(futures):
                idx, txt = fut.result()
                results[idx] = txt

        return [_RequestOutput(txt) for txt in results]

    def _send_chat(self, *, messages: List[dict], temperature: float | None, output_model: Type[BaseModel]):
        """Backend-specific implementation – must return response text."""
        raise NotImplementedError


class OpenAIClient(BaseHTTPClient):
    """OpenAI API client."""
    
    def __init__(self, endpoint: str | None, api_key: str | None, model: str):
        super().__init__(endpoint, api_key or os.getenv("OPENAI_API_KEY"), model)
        self._client = openai.OpenAI(base_url=self.endpoint, api_key=self.api_key)

    def _send_chat(self, *, messages: List[dict], temperature: float | None, output_model: Type[BaseModel]):
        resp = self._client.responses.parse(
            model=self.model,
            input=messages,
            temperature=temperature,
            text_format=output_model,
            timeout=120,
        )
        return resp.output_parsed


class VLLMClient(BaseHTTPClient):
    """Client for vLLM's OpenAI-compatible HTTP server using OpenAI SDK."""

    def __init__(self, endpoint: str | None, model: str):
        super().__init__(endpoint or "http://localhost:8000/v1", None, model)
        self._client = openai.OpenAI(base_url=self.endpoint, api_key="dummy-key")

    def _send_chat(self, *, messages: List[dict], temperature: float | None, output_model: Type[BaseModel]):
        json_schema = output_model.model_json_schema()
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            extra_body={"guided_json": json_schema},
            timeout=120,
        )
        return resp.choices[0].message.content


class OllamaHTTPClient(BaseHTTPClient):
    """Client using Ollama Python package."""

    def __init__(self, endpoint: str | None, model: str):
        if ollama is None:
            raise ModuleNotFoundError(
                "The 'ollama' package is required for backend 'ollama_api'.\n"
                "Install with: pip install ollama"
            )
        super().__init__(endpoint, None, model)
        self._client = ollama.Client(host=self.endpoint)

    def _send_chat(self, *, messages: List[dict], temperature: float | None, output_model: Type[BaseModel]):
        json_schema = output_model.model_json_schema()
        resp = self._client.chat(
            model=self.model,
            messages=messages,
            stream=False,
            format="json",
            options={"temperature": temperature},
        )
        # The official ollama client returns a dict, not an object.
        content = resp.get("message", {}).get("content", "")
        return content 