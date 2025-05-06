from __future__ import annotations

"""chainette.core.step – single LLM invocation node.

This final version ensures **mandatory guided JSON decoding** derived from
``output_model.model_json_schema()`` and gracefully handles either the
lightweight wrapper *or* vLLM's native :class:`SamplingParams` object.
"""

import hashlib
import json
from typing import Any, List, Sequence, Type

import requests
from pydantic import BaseModel
from vllm import SamplingParams  # type: ignore
from vllm.sampling_params import GuidedDecodingParams  # type: ignore

from chainette.engine.registry import get_engine_config
from chainette.engine.runtime import spawn_engine
from chainette.utils.templates import render

from rich.console import Console

__all__ = ["Step"]
class Step:  # noqa: D101
    """Declarative definition and execution of a single model call."""

    def __init__(
        self,
        *,
        id: str,
        name: str,
        input_model: Type[BaseModel],
        output_model: Type[BaseModel],
        engine_name: str,
        sampling: SamplingParams | Any,
        system_prompt: str,
        user_prompt: str,
        emoji: str = "🔗",
    ) -> None:  # noqa: D401, ANN001
        self.id = id
        self.name = name
        self.emoji = emoji
        self.input_model = input_model
        self.output_model = output_model
        self.engine_name = engine_name
        self._base_sampling = sampling
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt

        # ------------------------------------------------------------------
        # Pre‑compute guided‑decoding schema and deterministic signature
        # ------------------------------------------------------------------

        schema_json = json.dumps(self.output_model.model_json_schema())
        self._guided = GuidedDecodingParams(json=schema_json)

        sampling_blob = (
            sampling.as_dict()  # our thin wrapper
            if hasattr(sampling, "as_dict")
            else getattr(sampling, "model_dump", lambda: vars(sampling))()
        )
        self._sig = hashlib.md5(
            f"{id}{schema_json}{json.dumps(sampling_blob, default=str)}".encode(),
            usedforsecurity=False,
        ).hexdigest()[:8]

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _sampling_with_guidance(self) -> SamplingParams:  # noqa: D401
        if hasattr(self._base_sampling, "model_dump"):
            d = self._base_sampling.model_dump()
        elif hasattr(self._base_sampling, "as_dict"):
            d = self._base_sampling.as_dict()
        else:
            d = dict(vars(self._base_sampling))
        d["guided_decoding"] = self._guided
        return SamplingParams(**d)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, inputs: Sequence[BaseModel], *, run_id: str, step_index: int) -> List[BaseModel]:  # noqa: D401,WPS231
        console = Console()

        if not inputs:
            return []

        cfg = get_engine_config(self.engine_name)
        engine = spawn_engine(cfg)
        sp = self._sampling_with_guidance()

        url = f"http://127.0.0.1:{engine.port}/v1/chat/completions"
        headers = {"Content-Type": "application/json"}

        outputs: List[BaseModel] = []

        for row_id, inp in enumerate(inputs):
            ctx = inp.model_dump()
            sys_msg = render(self.system_prompt, ctx) if self.system_prompt else ""
            usr_msg = render(self.user_prompt, ctx)

            messages = []
            if sys_msg:
                messages.append({"role": "system", "content": sys_msg})
            messages.append({"role": "user", "content": usr_msg})

            if hasattr(sp, "model_dump"):
                sp_dict = sp.model_dump(exclude_none=True)
            elif hasattr(sp, "as_dict"):
                sp_dict = sp.as_dict()
            else:
                sp_dict = {k: v for k, v in vars(sp).items() if v is not None}

            body = {
                "model": cfg.model,
                "messages": messages,
                **sp_dict,
                "stream": False,
                "guided_json": self._guided.json,
            }

            try:
                resp = requests.post(url, json=body, headers=headers, timeout=90)
                resp.raise_for_status()
                text = resp.json()["choices"][0]["message"]["content"]
            except requests.RequestException as e:
                raise RuntimeError(f"API request failed for step '{self.id}': {e}") from e

            try:
                parsed = self.output_model.model_validate_json(text)
            except Exception as exc:  # noqa: BLE001
                raise RuntimeError(
                    f"Guided decoding failed for step '{self.id}'. Raw reply: {text[:200]}…"
                ) from exc

            outputs.append(parsed)

        return outputs

    # convenience -----------------------------------------------------------

    def signature(self) -> str:  # noqa: D401
        return self._sig
