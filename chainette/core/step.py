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

from vllm import SamplingParams
from vllm.sampling_params import GuidedDecodingParams
from chainette.utils.prompt import build_prompt
from chainette.utils.parsing import parse_llm_json
from chainette.engine.pool import ENGINE_POOL

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
        input_model: Type[BaseModel],
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
        self.input_model = input_model
        self.output_model = output_model
        self.engine_name = engine_name
        self.sampling = sampling

        self.tokenizer = None
        self.max_model_len = None
        self.engine = None

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

        json_schema = self.output_model.model_json_schema()
        guided_params = GuidedDecodingParams(json=json_schema)
        self.sampling.guided_decoding = guided_params

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

        # Lazily fetch engine & tokenizer (if needed)
        cfg = get_engine_config(self.engine_name)
        if self.engine is None:
            self.engine = ENGINE_POOL.acquire(self.engine_name)
            if getattr(cfg, "backend", "vllm") != "ollama" and self.tokenizer is None:
                self.tokenizer = AutoTokenizer.from_pretrained(cfg.model)

        if not inputs:
            return [], []

        prompts = [build_prompt(self, hist) for hist in item_histories]

        if debug and prompts:
            print(f"STEP {self.id}: generating {len(prompts)} prompt(s). Sample prompt:\n{prompts[0][:500]}…")

        raw_outputs = self.engine.generate(prompts=prompts, sampling_params=self.sampling)

        parsed_outputs: List[BaseModel] = []
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
            except ValueError as e:
                print(f"Warning: skipping item in step '{self.id}' due to parse error: {e}")
                new_histories.append(hist.copy())

        if writer is not None and self.yield_output and parsed_outputs:
            writer.write_step(self.id, parsed_outputs)

        return parsed_outputs, new_histories
