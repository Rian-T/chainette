# 11 – Tool Use Demo: Calculator & Weather
# Demonstrates how an LLM can decide which tool to invoke (calculator or weather)
# and Chainette integrates the resulting tool call via an Apply node.

from __future__ import annotations

import os
from typing import List, Literal

from pydantic import BaseModel, Field

from chainette import Chain, Step, ApplyNode, register_engine
from chainette.core.step import SamplingParams

# --------------------------------------------------------------------------- #
# Data models
# --------------------------------------------------------------------------- #

class Question(BaseModel):
    """User question that may require a calculation or weather lookup."""

    question: str = Field(..., description="Natural language user query")


class ToolChoice(BaseModel):
    """LLM output specifying which tool to call and its arguments."""

    tool: Literal["calculator", "weather"] = Field(..., description="Selected tool")
    # For calculator
    expression: str | None = Field(None, description="Math expression to evaluate, e.g. '3 * 4 + 2'")
    # For weather
    location: str | None = Field(None, description="City or place name for weather lookup")


class Answer(BaseModel):
    """Final answer returned to the user after tool execution."""

    answer: str

# Rebuild ToolChoice to resolve postponed annotations (Literal)
ToolChoice.model_rebuild()

# --------------------------------------------------------------------------- #
# Register OpenAI engine – requires OPENAI_API_KEY in env.
# --------------------------------------------------------------------------- #

OPENAI_KEY = os.getenv("OPENAI_API_KEY", "")
if not OPENAI_KEY:
    raise RuntimeError(
        "OPENAI_API_KEY environment variable not set – required for examples."
    )

register_engine(
    "openai_default",
    backend="openai",
    model="gpt-4.1-mini",
    api_key=OPENAI_KEY,
)

# --------------------------------------------------------------------------- #
# Step 1 – LLM selects the appropriate tool and arguments.
# --------------------------------------------------------------------------- #

tool_selector = Step(
    id="choose_tool",
    name="Tool Selector",
    engine_name="openai_default",
    input_model=Question,
    output_model=ToolChoice,
    sampling=SamplingParams(temperature=0.0),
    system_prompt=(
        "You are an AI assistant with access to two tools: \n"
        "1. calculator – evaluate basic math expressions (use Python syntax).\n"
        "2. weather – returns today's weather for a location.\n\n"
        "Read the user question and reply with JSON specifying the tool to use and its arguments."
    ),
    user_prompt="{{chain_input.question}}",
)

# --------------------------------------------------------------------------- #
# Step 2 – Apply node executes the selected tool.
# --------------------------------------------------------------------------- #

def _execute_tool(choice: ToolChoice) -> List[Answer]:
    """Execute calculator or weather based on *choice*."""

    if choice.tool == "calculator":
        if not choice.expression:
            return [Answer(answer="[error] No expression provided")]  # pragma: no cover

        try:
            # Very limited safe eval: allow digits, operators, whitespace.
            allowed_chars = set("0123456789+-*/(). %")
            if not set(choice.expression).issubset(allowed_chars):
                raise ValueError("invalid characters in expression")
            result = eval(choice.expression, {"__builtins__": {}}, {})  # noqa: S307
            return [Answer(answer=str(result))]
        except Exception as e:  # noqa: BLE001
            return [Answer(answer=f"[error] {e}")]

    elif choice.tool == "weather":
        loc = choice.location or "unknown"
        # Stubbed weather response.
        return [Answer(answer=f"It's sunny in {loc} ☀️ (stubbed)")]

    else:
        return [Answer(answer="[error] Unknown tool")]


execute_tool_node: ApplyNode[ToolChoice, Answer] = ApplyNode(
    fn=_execute_tool,
    id="execute_tool",
    name="Execute Tool",
    input_model=ToolChoice,
    output_model=Answer,
)

# --------------------------------------------------------------------------- #
# Chain definition
# --------------------------------------------------------------------------- #

tool_use_chain = Chain(
    name="Tool Use Demo",
    steps=[tool_selector, execute_tool_node],
) 