"""Steps for YAML chatbot demo."""

import os
from pydantic import BaseModel, Field
from chainette import Step, register_engine
from chainette.core.step import SamplingParams

OPENAI_KEY = os.getenv("OPENAI_API_KEY", "")
if not OPENAI_KEY:
    raise RuntimeError("OPENAI_API_KEY not set")

register_engine("openai_default", backend="openai", model="gpt-4.1-nano", api_key=OPENAI_KEY)

class Question(BaseModel):
    question: str = Field(..., description="Customer question")

class Answer(BaseModel):
    answer: str

answer_step = Step(
    id="answer",
    name="Customer Support Answer",
    input_model=Question,
    output_model=Answer,
    engine_name="openai_default",
    sampling=SamplingParams(temperature=0.2),
    system_prompt="You are a helpful customer support agent. Provide concise, friendly answers.",
    user_prompt="{{chain_input.question}}",
) 