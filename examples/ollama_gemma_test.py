from __future__ import annotations

"""Example chain to test Gemma 1B via Ollama backend."""

from pydantic import BaseModel
from chainette import Step, Chain, SamplingParams, register_engine

# Register Gemma engine (name as per Ollama repo: gemma3:1b etc.)
register_engine(
    name="gemma_ollama",
    backend="ollama",
    model="gemma3:1b",  # Adjust to correct model tag in Ollama
)

class Question(BaseModel):
    text: str

class Answer(BaseModel):
    answer: str

qa_step = Step(
    id="qa",
    name="Gemma QA (Ollama)",
    input_model=Question,
    output_model=Answer,
    engine_name="gemma_ollama",
    sampling=SamplingParams(temperature=0.2),
    system_prompt="Answer the user's question with a short factual answer.",
    user_prompt="{{chain_input.text}}",
)

qa_chain = Chain(name="Gemma QA", steps=[qa_step]) 