from __future__ import annotations
"""Thin HTTP clients for remote LLM APIs (OpenAI, vLLM-serve, etc.).

Each client mimics the minimal interface expected by :pyclass:`chainette.core.step.Step`:

    engine.generate(prompts: list[str], sampling_params: SamplingParams) -> list[LLMOutput]

where *LLMOutput* only needs to expose ``outputs[0].text`` (optionally ``reasoning_content``).

Keeping these wrappers tiny allows painless addition of new back-ends.
"""

from typing import List, Any, Optional, Sequence
from dataclasses import dataclass
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

# Local fallback if the OpenAI package is missing – we raise at runtime.
try:
    import openai  # type: ignore
except ModuleNotFoundError:  # pragma: no cover – handled later
    openai = None  # type: ignore

__all__ = [
    "BaseHTTPClient",
    "OpenAIClient",
    "VLLMClient",
    "OllamaHTTPClient",
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
    def generate(self, *, prompts: List[str], sampling_params: Any | None = None, step_id: str | None = None):  # noqa: D401
        """Shared generate that delegates to :meth:`_send_chat` per prompt."""
        out: List[_RequestOutput] = []
        temperature: float | None = None
        if sampling_params is not None and hasattr(sampling_params, "temperature"):
            temperature = sampling_params.temperature  # type: ignore[attr-defined]

        # Build message payloads first to preserve order
        msgs: List[Sequence[dict]] = [p if isinstance(p, list) else [{"role": "user", "content": p}] for p in prompts]

        # Fire requests concurrently – small thread pool is enough because network-bound
        results: List[Optional[str]] = [None] * len(msgs)

        def _run(idx: int, m: Sequence[dict]):
            txt = self._send_chat(messages=list(m), temperature=temperature)
            if step_id is not None:
                try:
                    from chainette.utils.logging import update_step_badge  # noqa: WPS433

                    update_step_badge(step_id, advance=1)
                except Exception:
                    pass
            return idx, txt

        with ThreadPoolExecutor(max_workers=min(8, len(msgs))) as pool:
            futures = [pool.submit(_run, i, m) for i, m in enumerate(msgs)]
            for fut in as_completed(futures):
                try:
                    idx, txt = fut.result()
                except Exception as exc:  # noqa: BLE001
                    # Safe-guard: timeout or network issue – skip this prompt
                    print(f"[WARN] Engine request failed or timed out: {exc}. Skipping item.")
                    # leave result as None which will propagate as empty string
                    continue
                results[idx] = txt

        out = [_RequestOutput(txt or "") for txt in results]
        return out

    # ------------------------------------------------------------------ #
    def _send_chat(self, *, messages: List[dict], temperature: float | None):  # noqa: D401
        """Backend-specific implementation – must return response text."""
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
        self._client = openai.OpenAI(base_url=self.endpoint, api_key=self.api_key)  # type: ignore[arg-type]

    def _send_chat(self, *, messages: List[dict], temperature: float | None):  # noqa: D401
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            response_format={"type": "json_object"},
            timeout=60*2,  # seconds – abort long-running requests
        )
        return resp.choices[0].message.content or ""


# --------------------------------------------------------------------------- #
# vLLM Serve – OpenAI compatible endpoint                                    #
# --------------------------------------------------------------------------- #


class VLLMClient(BaseHTTPClient):  # noqa: D101
    """Client for vLLM's OpenAI-compatible HTTP server."""

    def __init__(self, endpoint: str | None, model: str):
        super().__init__(endpoint or "http://localhost:8000/v1", None, model)
        import httpx

        self._client = httpx.Client(base_url=self.endpoint, timeout=60)

    def _send_chat(self, *, messages: List[dict], temperature: float | None):  # noqa: D401
        import httpx

        payload = {
            "model": self.model,
            "messages": messages,
            "response_format": {"type": "json_object"},
        }
        if temperature is not None:
            payload["temperature"] = temperature

        resp: httpx.Response = self._client.post("/chat/completions", json=payload)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]


# --------------------------------------------------------------------------- #
# Ollama REST API client                                                      #
# --------------------------------------------------------------------------- #


class OllamaHTTPClient(BaseHTTPClient):  # noqa: D101
    """Client hitting Ollama's `/api/chat` endpoint.

    Requires Ollama daemon running locally (default `http://localhost:11434`).
    """

    def __init__(self, endpoint: str | None, model: str):
        super().__init__(endpoint or "http://localhost:11434", None, model)
        import httpx

        self._client = httpx.Client(base_url=self.endpoint, timeout=60)

    def _send_chat(self, *, messages: List[dict], temperature: float | None):  # noqa: D401
        import httpx

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
        }
        if temperature is not None:
            payload["options"] = {"temperature": temperature}

        resp: httpx.Response = self._client.post("/api/chat", json=payload)
        resp.raise_for_status()
        data = resp.json()
        return (
            data.get("message", {}).get("content")
            if isinstance(data, dict)
            else data["message"]["content"]
        ) 