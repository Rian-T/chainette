from __future__ import annotations

"""Simple Chainette example to validate the Ollama backend.

Run with:
    poetry install --with ollama
    ollama run qwen2.5-instruct   # pull model if not present (name may vary)
    chainette run examples.ollama_qwen_test:qa_chain -i inputs.jsonl --output_dir run_qwen

Where 'inputs.jsonl' contains one JSON line per `Question` model, e.g.
    {"text": "What is the capital of France?"}
"""

from pydantic import BaseModel
from chainette import Step, Chain, SamplingParams, register_engine

# 1. Register Ollama engine (ensure Ollama server is running locally)
register_engine(
    name="qwen_ollama",
    backend="ollama",
    model="qwen2.5-instruct",  # Ollama model name
)

# 2. Define schemas
class Question(BaseModel):
    text: str

class Answer(BaseModel):
    answer: str

# 3. Single QA step
qa_step = Step(
    id="qa",
    name="Question Answering (Qwen via Ollama)",
    input_model=Question,
    output_model=Answer,
    engine_name="qwen_ollama",
    sampling=SamplingParams(temperature=0.2),
    system_prompt="Answer the user's question briefly and accurately.",
    user_prompt="{{text}}",
)

# 4. Build chain
qa_chain = Chain(name="Qwen QA", steps=[qa_step]) 