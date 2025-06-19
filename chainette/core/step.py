from __future__ import annotations

"""Chainette Step implementation.

A *Step* is a single LLM invocation that converts a list of *input_model*
objects into a list of *output_model* objects.
"""

from typing import List, Tuple, Type, Any, Optional, Dict

from pydantic import BaseModel
from transformers import AutoTokenizer

from chainette.engine.registry import get_engine_config
from chainette.core.node import Node
from chainette.io.writer import RunWriter
from chainette.utils.json_schema import generate_json_output_prompt

try:
    # Prefer the real vLLM classes when available.
    from vllm import SamplingParams  # type: ignore
    from vllm.sampling_params import GuidedDecodingParams  # type: ignore
except (ModuleNotFoundError, ImportError):  # pragma: no cover – fallback when vllm extra not installed
    from dataclasses import dataclass, field
    from typing import List as _List, Optional as _Optional, Dict as _Dict, Any as _Any

    @dataclass
    class GuidedDecodingParams:  # noqa: D401
        """Minimal stub providing the *json* attribute only."""

        json: _Dict[str, _Any] | None = None

    @dataclass
    class SamplingParams:  # noqa: D401
        """Lightweight replacement exposing the attrs Chainette relies on."""

        temperature: float | None = None
        top_p: _Optional[float] = None
        max_tokens: _Optional[int] = None
        stop: _List[str] = field(default_factory=list)
        guided_decoding: 'GuidedDecodingParams | None' = None

# Additional imports that depend on the presence of SamplingParams
from chainette.utils.prompt import build_prompt
from chainette.utils.parsing import parse_llm_json

__all__ = [
    "Step",
    "SamplingParams",
    "GuidedDecodingParams",
]


class Step(Node):
    def __init__(
        self,
        *,
        id: str,
        name: str,
        input_model: Type[BaseModel] | None = None,
        output_model: Type[BaseModel],
        engine_name: str,
        sampling: SamplingParams,
        system_prompt: str = "",
        user_prompt: str = "{{input}}",
        emoji: str | None = None,
        output_format_instruction: str | None = None,
        yield_output: bool = True,
    ) -> None:
        self.id = id
        self.name = name
        self.input_model: Type[BaseModel] | None = input_model
        self.output_model = output_model
        self.engine_name = engine_name
        self.sampling = sampling

        self.tokenizer = None
        self.max_model_len = None

        _original_system_prompt = system_prompt.strip()

        if output_format_instruction is None:
            json_instruction_string = generate_json_output_prompt(self.output_model)
        else:
            json_instruction_string = output_format_instruction.strip()

        if _original_system_prompt and json_instruction_string:
            self.system_prompt = f"{_original_system_prompt}\n\n{json_instruction_string}"
        elif json_instruction_string:
            self.system_prompt = json_instruction_string
        else:
            self.system_prompt = _original_system_prompt

        self.user_prompt = user_prompt.strip()
        self.emoji = emoji or ""
        self.yield_output = yield_output

        # Internal counter for progress tracking across batches
        self._completed_items: int = 0

        # No direct JSON schema here – downstream HTTP clients will derive it from *output_model*.

    def execute(
        self,
        inputs: List[BaseModel],
        item_histories: List[Dict[str, Any]],
        writer: RunWriter | None = None,
        debug: bool = False,
        batch_size: int = 0,  # Kept for API-compat but now ignored – batching handled by Executor
    ) -> Tuple[List[BaseModel], List[Dict[str, Any]]]:
        if len(inputs) != len(item_histories):
            raise ValueError("Mismatch between number of inputs and item_histories in Step.execute")

        # Lazily fetch tokenizer once
        cfg = get_engine_config(self.engine_name)
        backend = getattr(cfg, "backend", "vllm")
        if backend not in ("ollama", "ollama_api", "openai", "vllm_api") and self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(cfg.model)

        if not inputs:
            return [], []

        prompts = [build_prompt(self, hist) for hist in item_histories]

        if debug and prompts:
            print(f"STEP {self.id}: generating {len(prompts)} prompt(s). Sample prompt:\n{prompts[0][:500]}…")

        from chainette.engine.broker import EngineBroker

        with EngineBroker.acquire(self.engine_name) as eng:
            raw_outputs = eng.generate(
                prompts=prompts,
                output_model=self.output_model,
                sampling_params=self.sampling,
                step_id=self.id,
            )

        parsed_outputs: List[BaseModel] = []
        parsed_records_for_writer: List[Any] = []
        new_histories: List[Dict[str, Any]] = []

        for hist, raw in zip(item_histories, raw_outputs):
            try:
                parsed, reasoning = parse_llm_json(
                    self.output_model,
                    raw,
                    engine_name=self.engine_name,
                    step_id=self.id,
                )
                parsed_outputs.append(parsed)

                h = hist.copy()
                if self.yield_output:
                    h[self.id] = parsed
                if reasoning:
                    h[f"{self.id}_reasoning"] = reasoning
                new_histories.append(h)

                # Attach row_id for alignment if present in history
                row_id = hist.get("row_id")
                record = parsed.model_dump(mode="json")
                if row_id is not None:
                    record["row_id"] = row_id
                parsed_records_for_writer.append(record)

                # progress now handled in engine via advance updates
            except ValueError as e:
                print(f"Warning: skipping item in step '{self.id}' due to parse error: {e}")
                new_histories.append(hist.copy())

        if writer is not None and self.yield_output and parsed_records_for_writer:
            writer.write_step(self.id, parsed_records_for_writer)

        return parsed_outputs, new_histories
