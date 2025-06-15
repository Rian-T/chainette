from __future__ import annotations
"""Ollama backend wrapper to mimic vLLM LLM interface.

This class provides a minimal `.generate` method compatible with
`chainette.core.step.Step`, returning objects that expose the same
attributes used in `_parse_output` (in particular `outputs[0].text`).
It relies on the official `ollama` Python package which communicates with
an Ollama server running locally (or via the `OLLAMA_HOST` env var).

Limitations:
- Guided decoding is *not* enforced by Ollama – instead, Chainette still
  relies on prompt-based JSON instructions. Validation continues to be
  done via Pydantic in `Step._parse_output`.
- Sampling parameters are ignored except for `temperature`, if provided.
- Streaming is not yet supported.
"""

from typing import List, Any, Optional

# The `ollama` import is optional; we defer the import error until runtime
try:
    import ollama  # type: ignore
except ModuleNotFoundError:  # pragma: no cover – handled by registry
    ollama = None  # type: ignore

__all__ = ["OllamaLLM"]


class _SimpleCompletion:  # noqa: D401 – small internal struct
    def __init__(self, text: str):
        self.text = text
        # vLLM may add `.reasoning` – keep placeholder for interface parity
        self.reasoning_content: Optional[str] = None


class _RequestOutput:  # noqa: D401 – mimic vllm.RequestOutput
    def __init__(self, text: str):
        self.outputs = [_SimpleCompletion(text)]


class OllamaLLM:  # noqa: D101
    def __init__(self, model: str):
        if ollama is None:
            raise ModuleNotFoundError(
                "The 'ollama' package is required for the Ollama backend.\n"
                "Install with: pip install chainette[ollama] or poetry install --with ollama"
            )
        self.model = model

    # ------------------------------------------------------------------ #
    # Public API expected by Step
    # ------------------------------------------------------------------ #

    def generate(self, *, prompts: List[str], sampling_params: Any | None = None):  # noqa: D401
        """Generate completions for *prompts*.

        Args:
            prompts: List of prompt strings. Each prompt is sent as a
                separate request to the Ollama server.
            sampling_params: vLLM `SamplingParams` instance (optional). Only
                `temperature` is currently mapped.

        Returns:
            List of objects exposing `.outputs[0].text` similar to vLLM.
        """
        responses: List[_RequestOutput] = []

        # Map basic sampling options
        options: dict[str, Any] = {}
        if sampling_params is not None:
            if hasattr(sampling_params, "temperature") and sampling_params.temperature is not None:
                options["temperature"] = sampling_params.temperature
            # Map top_p if present (Ollama supports it via options)
            if hasattr(sampling_params, "top_p") and sampling_params.top_p is not None:
                options["top_p"] = sampling_params.top_p

        for prompt in prompts:
            # Extract JSON schema if guided decoding is enabled
            schema = None
            if (
                sampling_params is not None
                and hasattr(sampling_params, "guided_decoding")
                and sampling_params.guided_decoding is not None
                and hasattr(sampling_params.guided_decoding, "json")
            ):
                schema = sampling_params.guided_decoding.json  # type: ignore[attr-defined]

            # Use chat endpoint to leverage role separation + structured outputs
            # Fallback to generate if chat not available
            try:
                resp = ollama.chat(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    stream=False,
                    format=schema if schema is not None else None,
                    options=options if options else None,
                )
                # `ollama.chat` may return dict or object depending on version.
                if isinstance(resp, dict):
                    text = resp.get("message", {}).get("content", resp.get("response", ""))
                else:
                    # Newer versions return an object with `.message.content`
                    text = getattr(resp, "message", getattr(resp, "response", None))
                    if text and hasattr(text, "content"):
                        text = text.content  # type: ignore[assignment]
                    if not isinstance(text, str):
                        text = str(text)
            except AttributeError:
                # chat() not present – fall back to generate()
                resp = ollama.generate(
                    model=self.model,
                    prompt=prompt,
                    stream=False,
                    format=schema if schema is not None else None,
                    options=options if options else None,
                )
                text = resp.get("response", "")

            responses.append(_RequestOutput(text))

        return responses 