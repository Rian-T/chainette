"""Naive in-memory retriever for demo purposes."""
from typing import List
from pydantic import BaseModel

_DOCS = {
    "llm": "LLM stands for Large Language Model, a neural network trained on vast text corpora.",
    "rag": "Retrieval-Augmented Generation (RAG) combines information retrieval with text generation.",
}

class Query(BaseModel):
    query: str

class Context(BaseModel):
    context: str

def retrieve(q: Query) -> List[Context]:
    key = "rag" if "rag" in q.query.lower() else "llm"
    return [Context(context=_DOCS[key])] 