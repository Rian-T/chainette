from __future__ import annotations

"""Chainette Step implementation.

A *Step* is a single LLM invocation that converts a list of *input_model*
objects into a list of *output_model* objects.
"""

from typing import List, Tuple, Type, Any, Optional, Dict
import json

from pydantic import BaseModel
from transformers import AutoTokenizer

from chainette.utils.templates import render
from chainette.engine.registry import get_engine_config
from chainette.core.node import Node
from chainette.io.writer import RunWriter
from chainette.utils.json_schema import generate_json_output_prompt

from vllm import SamplingParams
from vllm.sampling_params import GuidedDecodingParams

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

    def _build_prompt(self, item_history: Dict[str, Any]) -> str:
        rendering_context = {}
        for key, value in item_history.items():
            if isinstance(value, BaseModel): # Check if Pydantic model
                model_dict = value.model_dump()
                for field_name, field_val in model_dict.items():
                    rendering_context[f"{key}.{field_name}"] = field_val
                rendering_context[key] = value # Keep original model object as well
            else:
                rendering_context[key] = value # For non-model items like 'chain_input' or 'step_id.reasoning_content'

        messages = []

        rendered_system_prompt = render(self.system_prompt, rendering_context) if self.system_prompt else ""
        if rendered_system_prompt:
            messages.append({"role": "system", "content": rendered_system_prompt})

        rendered_user_prompt = render(self.user_prompt, rendering_context)
        messages.append({"role": "user", "content": rendered_user_prompt})

        if not messages:
            return ""
            
        # Ensure tokenizer is initialized
        if self.tokenizer is None:
            cfg = get_engine_config(self.engine_name)
            self.tokenizer = AutoTokenizer.from_pretrained(cfg.model)

        # If using Ollama backend, return simple concatenated prompt (Ollama understands role tags)
        cfg = get_engine_config(self.engine_name)
        if getattr(cfg, "backend", "vllm") == "ollama":
            prompt_lines = []
            for m in messages:
                role = m["role"]
                content = m["content"]
                prompt_lines.append(f"## {role}\n{content}")
            prompt_lines.append("## assistant\n")  # Indicate generation start
            return "\n\n".join(prompt_lines)

        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    def _parse_output(self, llm_output: Any) -> Tuple[BaseModel, Optional[str]]:
        first_completion = llm_output.outputs[0]
        text = first_completion.text.strip()
        print(f"DEBUG: Raw LLM-output text: {text}")
        reasoning_content: Optional[str] = None
        if hasattr(first_completion, "reasoning"):
            reasoning_content = str(first_completion.reasoning)
        elif hasattr(first_completion, "reasoning_content"):
            reasoning_content = str(first_completion.reasoning_content)

        try:
            data = json.loads(text)
            obj = self.output_model.model_validate(data)
            return obj, reasoning_content
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON output from '{self.engine_name}' for step '{self.id}': {text}. Error: {e}")
        except Exception as exc:
            raise ValueError(f"Failed to validate output for step '{self.id}': {exc}\nModel: {self.engine_name}\nOutput: {text}")

    def execute(
        self,
        inputs: List[BaseModel],
        item_histories: List[Dict[str, Any]],
        writer: RunWriter | None = None,
        debug: bool = False,
        batch_size: int = 0, # Default to 0 means no batching or handle full list
    ) -> Tuple[List[BaseModel], List[Dict[str, Any]]]:
        if len(inputs) != len(item_histories):
            raise ValueError("Mismatch between number of inputs and item_histories in Step.execute")

        cfg = get_engine_config(self.engine_name)
        if self.engine is None:
            self.engine = cfg.engine
            # Re-initialize tokenizer whenever engine is initialized
            self.tokenizer = AutoTokenizer.from_pretrained(cfg.model)

        all_parsed_outputs: List[BaseModel] = []
        all_new_item_histories: List[Dict[str, Any]] = []

        # Determine actual batch_size: if 0 or less, or larger than inputs, process all at once
        effective_batch_size = batch_size if batch_size > 0 and batch_size < len(inputs) else len(inputs)
        if effective_batch_size == 0 and len(inputs) > 0: # handle case where inputs might be empty
             effective_batch_size = len(inputs)
        elif len(inputs) == 0:
            return [], []


        for i in range(0, len(inputs), effective_batch_size):
            batch_inputs = inputs[i:i + effective_batch_size]
            batch_item_histories = item_histories[i:i + effective_batch_size]

            prompts = [self._build_prompt(hist) for hist in batch_item_histories]

            if debug:
                print(f"DEBUG: Step '{self.id}' processing batch {i // effective_batch_size + 1} with {len(batch_inputs)} items.")
                # Optionally print one prompt from the batch for brevity
                if prompts:
                    print(f"DEBUG: Sample prompt from batch: {prompts[0][:500]}...") # Print first 500 chars

            outputs_raw_batch = self.engine.generate(prompts=prompts, sampling_params=self.sampling)

            batch_parsed_outputs: List[BaseModel] = []
            batch_new_item_histories: List[Dict[str, Any]] = []

            for j, raw_out in enumerate(outputs_raw_batch):
                original_item_history = batch_item_histories[j]
                try:
                    parsed_output, reasoning = self._parse_output(raw_out)
                    batch_parsed_outputs.append(parsed_output)
                    
                    # Update history for this item
                    updated_history = original_item_history.copy()
                    if self.yield_output:
                        updated_history[self.id] = parsed_output
                    if reasoning:
                        updated_history[f"{self.id}_reasoning"] = reasoning
                    batch_new_item_histories.append(updated_history)

                except ValueError as e:
                    print(f"Warning: Skipping item due to parsing/validation error in step '{self.id}': {e}")
                    # Append original history if output is skipped, or decide on error handling
                    batch_new_item_histories.append(original_item_history.copy()) 
                    # Optionally, append a placeholder or error object to batch_parsed_outputs
                    # For now, we just skip adding to batch_parsed_outputs for this item

            all_parsed_outputs.extend(batch_parsed_outputs)
            all_new_item_histories.extend(batch_new_item_histories)

            if writer is not None and self.yield_output and batch_parsed_outputs:
                # Write only the successfully parsed outputs of the current batch
                writer.write_step(self.id, batch_parsed_outputs)
                if debug:
                    print(f"DEBUG: Step '{self.id}' wrote {len(batch_parsed_outputs)} outputs for batch {i // effective_batch_size + 1} to writer.")
        
        # The writer.finalize() will be called at the end of the Chain.run()
        return all_parsed_outputs, all_new_item_histories
