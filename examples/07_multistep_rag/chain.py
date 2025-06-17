"""07 – Multi-step RAG demo.
Retrieval (ApplyNode) ➜ Answering Step (OpenAI).
"""
import os
from pydantic import BaseModel, Field
from chainette import Chain, Step, register_engine
from chainette.core.apply import ApplyNode
from chainette.core.step import SamplingParams
import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).parent))  # ensure directory importable
import retriever

OPENAI_KEY = os.getenv("OPENAI_API_KEY", "")
if not OPENAI_KEY:
    raise RuntimeError("OPENAI_API_KEY not set")

register_engine("openai_mini", backend="openai", model="gpt-4.1-mini", api_key=OPENAI_KEY)

class Answer(BaseModel):
    answer: str

Query = retriever.Query
Context = retriever.Context
retrieve = retriever.retrieve

retrieve_step = ApplyNode(retrieve, id="retrieve", input_model=Query)

qa_step = Step(
    id="answer",
    name="RAG Answer",
    input_model=Context,
    output_model=Answer,
    engine_name="openai_mini",
    sampling=SamplingParams(temperature=0.2),
    system_prompt="Answer the user question using the provided context.",
    user_prompt="Question: {{chain_input.query}}\nContext: {{retrieve.context}}",
)

rag_chain = Chain(name="RAG Demo", steps=[retrieve_step, qa_step]) 