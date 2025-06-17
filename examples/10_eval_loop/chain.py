"""10 – Open-Ended Task Grading Demo (LLM-as-a-Judge).

Pipeline:
1. **Worker** (`gpt-4.1-nano`) generates an answer to an open-ended prompt.
2. **Judge**  (`gpt-4.1-mini`) evaluates that answer against the prompt and
   returns an integer **score 1-10** based on depth, accuracy, clarity, and
   creativity.

This showcases a more ambiguous evaluation where no single reference answer
exists and the judge must apply holistic criteria.
"""

import os

from pydantic import BaseModel, Field

from chainette import Chain, Step, register_engine
from chainette.core.step import SamplingParams

# --------------------------------------------------------------------------- #
# Engine registration (requires OPENAI_API_KEY)
# --------------------------------------------------------------------------- #

OPENAI_KEY = os.getenv("OPENAI_API_KEY", "")
if not OPENAI_KEY:
    raise RuntimeError("OPENAI_API_KEY environment variable not set – required for examples.")

# Worker – tiny, cheaper model
register_engine(
    "openai_worker",
    backend="openai",
    model="gpt-4.1-nano",
    api_key=OPENAI_KEY,
)

# Judge – stronger model
register_engine(
    "openai_judge",
    backend="openai",
    model="gpt-4.1",
    api_key=OPENAI_KEY,
)

# --------------------------------------------------------------------------- #
# Data models – open-ended task with 1-10 grading
# --------------------------------------------------------------------------- #

class Prompt(BaseModel):
    prompt: str = Field(..., description="Open-ended task/question for the assistant")

class Response(BaseModel):
    response: str

class Grade(BaseModel):
    score: float = Field(..., ge=1, le=10, description="Score from 1 to 10")

# --------------------------------------------------------------------------- #
# Step 1 – Worker prediction
# --------------------------------------------------------------------------- #

worker_step = Step(
    id="predict",
    name="Worker QA",
    input_model=Prompt,
    output_model=Response,
    engine_name="openai_worker",
    sampling=SamplingParams(temperature=0.3),
    system_prompt="You are a knowledgeable assistant. Answer the question concisely.",
    user_prompt="{{chain_input.prompt}}",
)

# --------------------------------------------------------------------------- #
# Step 2 – Judge evaluation
# --------------------------------------------------------------------------- #

judge_step = Step(
    id="grade",
    name="Judge Score",
    input_model=Prompt,  # includes reference answer
    output_model=Grade,
    engine_name="openai_judge",
    sampling=SamplingParams(temperature=0.0),
    system_prompt=(
        "You are an expert judge. Determine if the worker's answer correctly answers the question. "
        "Consider semantic equivalence; minor wording differences are acceptable. Be severe in your grading."
        "We are in a prestigious university. The student's answer should be precise."
        "Grade from 1 to 10 like a professor."
    ),
    user_prompt=(
        "Question: {{chain_input.prompt}}\n"\
        "Worker answer: {{predict.response}}"
    ),
)

# --------------------------------------------------------------------------- #
# Chain definition
# --------------------------------------------------------------------------- #

eval_chain = Chain(name="Eval Loop Demo", steps=[worker_step, judge_step]) 