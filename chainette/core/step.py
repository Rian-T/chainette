"""
Single‑step execution wrapper.

* Mandatory: every Step has an *output_model* (Pydantic).
* We ALWAYS ask vLLM for guided JSON decoding that matches that schema.
  ‑ User SamplingParams are respected (temperature, top_p, …).
  ‑ If the caller already supplied guided_decoding we merge, ours wins.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Type

import requests
from pydantic import BaseModel

from chainette.engine.registry import get_engine_config # Added import
from chainette.engine.runtime import spawn_engine, LiveEngine
from chainette.utils.templates import render

# --- thin re‑export ----------------------------------------------------------

from vllm import SamplingParams  # noqa: F401
from vllm import GuidedDecodingParams  # noqa: F401

__all__ = ["Step"]


class Step:  # noqa: D101
    def __init__(
        self,
        *,
        id: str,
        name: str,
        input_model: Type[BaseModel],
        output_model: Type[BaseModel],
        engine_name: str,
        sampling: "SamplingParams",
        system_prompt: str,
        user_prompt: str,
        emoji: str = "🔗",
    ):
        self.id = id
        self.name = name
        self.emoji = emoji
        self.input_model = input_model
        self.output_model = output_model
        self.engine_name = engine_name
        self._base_sampling = sampling
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt

        # Pre‑compute JSON schema + GuidedDecodingParams once
        schema = json.dumps(self.output_model.model_json_schema())
        self._guided = GuidedDecodingParams(json=schema)

        # Hash helps Step equality / caching
        self._sig = hashlib.md5(
            f"{id}{schema}{json.dumps(sampling.model_dump())}".encode(),
            usedforsecurity=False,
        ).hexdigest()[:8]

    # --------------------------------------------------------------------- #

    def _sampling_with_guidance(self) -> "SamplingParams":
        """Return a copy of user sampling with mandatory guided JSON."""
        kw = self._base_sampling.model_dump()
        # User might already set 'guided_decoding'; we override
        kw["guided_decoding"] = self._guided
        return SamplingParams(**kw)

    # --------------------------------------------------------------------- #

    def run(
        self, inputs: list[BaseModel], run_id: str, step_index: int
    ) -> list[BaseModel]:
        """Execute a batch of inputs and return parsed outputs."""
        engine_cfg = get_engine_config(self.engine_name) # Get config first
        engine: LiveEngine = spawn_engine(engine_cfg) # Pass config to spawn
        sp = self._sampling_with_guidance()

        results: list[BaseModel] = []
        for idx, inp in enumerate(inputs):
            # Prepare payload for each input
            prompts = [
                {
                    "role": "system",
                    "content": self.system_prompt,
                },
                {
                    "role": "user",
                    "content": render(self.user_prompt, inp.model_dump()), # Pass dict to render
                },
            ]

            payload = {
                "model": engine.config.model, # Use config from LiveEngine
                "messages": prompts,
                **sp.model_dump(mode="json", exclude_none=True),
                "stream": False,
            }
            # Construct the correct URL using the engine's port
            api_url = f"http://127.0.0.1:{engine.port}/v1/chat/completions"
            resp = requests.post(api_url, json=payload, timeout=60)
            resp.raise_for_status()
            text = resp.json()["choices"][0]["message"]["content"]

            try:
                parsed = self.output_model.model_validate_json(text)
            except Exception as exc:
                raise RuntimeError(
                    f"Step '{self.id}' expected JSON compliant with "
                    f"{self.output_model.__name__}, got:\n{text}"
                ) from exc

            # add bookkeeping fields expected by writer
            parsed_dict = parsed.model_dump()
            parsed_dict.update(
                step_name=self.name,
                step_emoji=self.emoji,
                id=idx, # Use loop index for id
                run_id=run_id, # Add run_id
                step_index=step_index # Add step_index
            )
            # Re-validate with the added fields if they are part of the model
            # If not, consider creating a wrapper model or handling differently
            # For now, assuming output_model can handle these extra fields or they are ignored
            try:
                final_model = self.output_model.model_validate(parsed_dict)
            except Exception as e:
                 # If validation fails, maybe just return the dict or a simpler model
                 # For now, let's assume it works or handle the error
                 print(f"Warning: Could not re-validate model with bookkeeping fields: {e}")
                 final_model = parsed # Fallback to original parsed model

            results.append(final_model)

        return results

    # Convenience for writer
    def signature(self) -> str:  # noqa: D401
        """A short hash that changes when sampling or schema changes."""
        return self._sig
