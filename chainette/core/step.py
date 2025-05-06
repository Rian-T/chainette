from __future__ import annotations

"""chainette.core.step – single LLM invocation node.

This final version ensures **mandatory guided JSON decoding** derived from
``output_model.model_json_schema()`` and gracefully handles either the
lightweight wrapper *or* vLLM's native :class:`SamplingParams` object.
"""

import hashlib
import json
from typing import Any, List, Sequence, Type, Dict, Optional, Tuple
import concurrent.futures
import time

import requests
from pydantic import BaseModel
from vllm import SamplingParams  # type: ignore
from vllm.sampling_params import GuidedDecodingParams  # type: ignore

from chainette.engine.registry import get_engine_config
from chainette.engine.runtime import spawn_engine
from chainette.utils.templates import render

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn, TaskID

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
    
    def _prepare_request_body(self, inp: BaseModel, cfg, sp_dict: Dict) -> Dict:
        """Prepare the request body for a single input."""
        ctx = inp.model_dump()
        sys_msg = render(self.system_prompt, ctx) if self.system_prompt else ""
        usr_msg = render(self.user_prompt, ctx)

        messages = []
        if sys_msg:
            messages.append({"role": "system", "content": sys_msg})
        messages.append({"role": "user", "content": usr_msg})

        return {
            "model": cfg.model,
            "messages": messages,
            **sp_dict,
            "stream": False,
            "guided_json": self._guided.json,
        }
    
    def _make_api_request(self, url: str, body: Dict, headers: Dict, timeout: int = 1200) -> Tuple[Optional[BaseModel], Optional[Exception]]:
        """Make a single API request and handle parsing."""
        try:
            resp = requests.post(url, json=body, headers=headers, timeout=timeout)
            resp.raise_for_status()
            text = resp.json()["choices"][0]["message"]["content"]
            parsed = self.output_model.model_validate_json(text)
            return parsed, None
        except requests.RequestException as e:
            return None, e
        except Exception as exc:
            return None, exc

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self, 
        inputs: Sequence[BaseModel], 
        *, 
        run_id: str, 
        step_index: int, 
        batch_size: int = None,
        main_progress: Optional[Progress] = None,
        parent_description_prefix: str = ""
    ) -> List[BaseModel]:  # noqa: D401,WPS231
        """Run the step on input data with batched processing.
        
        Parameters
        ----------
        inputs : Sequence[BaseModel]
            The input records to process
        run_id : str
            Unique run identifier
        step_index : int or str
            Step index for tracking
        batch_size : int, optional
            Batch size for processing. If None, gets batch size from chain context.
        main_progress : Optional[Progress], optional
            An existing Rich Progress instance to use for displaying progress.
        parent_description_prefix : str, optional
            A prefix for the description of tasks added to main_progress.
            
        Returns
        -------
        List[BaseModel]
            List of output models
        """
        console = Console()

        if not inputs:
            return []

        # Determine the batch size
        # If not explicitly provided, use context from thread-local or default to 16
        if batch_size is None:
            from threading import current_thread
            thread_locals = getattr(current_thread(), "__dict__", {})
            chain_context = thread_locals.get("chain_context", {})
            batch_size = chain_context.get("batch_size", 16)
        
        batch_size = max(1, batch_size)  # Ensure at least 1

        cfg = get_engine_config(self.engine_name)
        engine = spawn_engine(cfg)
        sp = self._sampling_with_guidance()

        url = f"http://127.0.0.1:{engine.port}/v1/chat/completions"
        headers = {"Content-Type": "application/json"}

        # Convert sampling params to dict
        if hasattr(sp, "model_dump"):
            sp_dict = sp.model_dump(exclude_none=True)
        elif hasattr(sp, "as_dict"):
            sp_dict = sp.as_dict()
        else:
            sp_dict = {k: v for k, v in vars(sp).items() if v is not None}

        outputs = []
        
        # Create batches
        batches = [list(inputs[i:i + batch_size]) for i in range(0, len(inputs), batch_size)]

        if not main_progress:
            # Fallback to creating its own Progress instance if none is provided
            console_for_step = Console()
            with Progress(
                SpinnerColumn(),
                TextColumn("[blue]{task.description}"),
                BarColumn(bar_width=None),
                TaskProgressColumn(),
                TimeElapsedColumn(),
                console=console_for_step,
                transient=True,
            ) as progress_instance:
                return self._process_batches(
                    inputs, batches, cfg, sp_dict, url, headers, progress_instance, 
                    parent_description_prefix or f"[cyan]Step '{self.name}'"
                )
        else:
            # Use the provided main_progress instance
            return self._process_batches(
                inputs, batches, cfg, sp_dict, url, headers, main_progress, 
                parent_description_prefix or f"[cyan]Step '{self.name}'"
            )

    def _process_batches(
        self,
        inputs: Sequence[BaseModel],
        batches: List[List[BaseModel]],
        cfg: Any, # EngineConfig
        sp_dict: Dict,
        url: str,
        headers: Dict,
        progress_instance: Progress,
        description_prefix: str
    ) -> List[BaseModel]:
        """Helper method to process batches and update progress."""
        outputs = []
        console = Console() # For error printing if needed, separate from progress

        # Add a single task for the entire step's item processing
        # This task will show total items for the step and advance per item.
        # Its description will be updated to show current batch info.
        step_item_task_id: Optional[TaskID] = None
        if progress_instance: # Ensure progress_instance is not None
            step_item_task_id = progress_instance.add_task(
                f"{description_prefix}: Preparing...",
                total=len(inputs)
            )

        for batch_idx, batch in enumerate(batches):
            if progress_instance and step_item_task_id is not None:
                progress_instance.update(
                    step_item_task_id,
                    description=f"{description_prefix}: Batch {batch_idx+1}/{len(batches)} ({len(batch)} items)"
                )
            
            request_bodies = [self._prepare_request_body(inp, cfg, sp_dict) for inp in batch]
            batch_outputs_list = [None] * len(batch) # Use list for ordered results
            batch_errors_info = []

            with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(batch), 32)) as executor:
                future_to_idx_map = {
                    executor.submit(self._make_api_request, url, body, headers): original_idx 
                    for original_idx, body in enumerate(request_bodies)
                }
                
                for future in concurrent.futures.as_completed(future_to_idx_map):
                    original_idx = future_to_idx_map[future]
                    result, error = future.result()
                    
                    if error:
                        batch_errors_info.append((original_idx, error))
                    else:
                        batch_outputs_list[original_idx] = result
                    
                    if progress_instance and step_item_task_id is not None:
                        progress_instance.update(step_item_task_id, advance=1)
            
            if batch_errors_info:
                # Log errors (consider using progress_instance.log or console.print)
                # For now, just print to console to avoid interfering with progress display too much
                error_summary_msg = f"[red]❌ {description_prefix}: {len(batch_errors_info)} errors in batch {batch_idx+1}."
                # To prevent flooding, print summary or use progress.log if available and desired
                if progress_instance:
                    progress_instance.log(error_summary_msg)
                else:
                    console.print(error_summary_msg)


            # Extend outputs with successfully processed items, maintaining order
            for out_item in batch_outputs_list:
                if out_item is not None: # Only add successful items
                    outputs.append(out_item)
        
        if progress_instance and step_item_task_id is not None:
            progress_instance.remove_task(step_item_task_id)
            
        return outputs

    # convenience -----------------------------------------------------------

    def signature(self) -> str:  # noqa: D401
        return self._sig
