from __future__ import annotations
"""Comprehensive example chain to exercise Branch, Apply, multiple Steps.
Run:
    poetry run chainette run examples/ollama_gemma_features.py full_chain inputs.jsonl run_features
Assumes Ollama daemon running with `gemma3:1b` pulled.
"""
from typing import List
from pydantic import BaseModel
from chainette import (
    Step,
    Chain,
    SamplingParams,
    register_engine,
    Branch,
    apply,
)

# ------------------------------------------------------------------ #
# Engine registration (shared for all steps)
# ------------------------------------------------------------------ #
register_engine(name="gemma_ollama", backend="ollama", model="gemma3:1b")

# ------------------------------------------------------------------ #
# Models
# ------------------------------------------------------------------ #
class Question(BaseModel):
    text: str

class QAOut(BaseModel):
    answer: str
    confidence: float

class Translation(BaseModel):
    translated: str

class Summary(BaseModel):
    summary: str

# ------------------------------------------------------------------ #
# Steps
# ------------------------------------------------------------------ #
qa_step = Step(
    id="qa",
    name="Gemma QA",
    input_model=Question,
    output_model=QAOut,
    engine_name="gemma_ollama",
    sampling=SamplingParams(temperature=0.2),
    system_prompt="Answer the question. Provide a confidence score between 0 and 1.",
    user_prompt="{{chain_input.text}}",
)

# Apply filter – keep only high confidence answers (>0.7)

def filter_high_confidence(item: QAOut) -> List[QAOut]:
    return [item] if item.confidence >= 0.7 else []

filter_node = apply(filter_high_confidence, input_model=QAOut, output_model=QAOut)

# Translation Steps (parallel branches)
fr_translate = Step(
    id="fr_trans",
    name="Translate to French",
    input_model=QAOut,
    output_model=Translation,
    engine_name="gemma_ollama",
    sampling=SamplingParams(temperature=0.2),
    system_prompt="You are a professional translator. Return JSON with key 'translated'.",
    user_prompt="Translate into French the sentence: 'The answer to the question \"{{chain_input.text}}\" is {{qa.answer}}.'",
)

es_translate = Step(
    id="es_trans",
    name="Translate to Spanish",
    input_model=QAOut,
    output_model=Translation,
    engine_name="gemma_ollama",
    sampling=SamplingParams(temperature=0.2),
    system_prompt="You are a professional translator. Return JSON with key 'translated'.",
    user_prompt="Translate into Spanish the sentence: 'The answer to the question \"{{chain_input.text}}\" is {{qa.answer}}.'",
)

translation_branch_fr = Branch(name="fr_branch", steps=[fr_translate]).join("fr")
translation_branch_es = Branch(name="es_branch", steps=[es_translate]).join("es")

# Aggregator step consumes joined translations
class DualSummary(BaseModel):
    summary: str

aggregator_step = Step(
    id="agg",
    name="Aggregator",
    input_model=Translation,  # dummy, we will feed same QAOut but rely on histories
    output_model=DualSummary,
    engine_name="gemma_ollama",
    sampling=SamplingParams(temperature=0.0),
    system_prompt="Return JSON with key 'summary'.",
    user_prompt="Compare French: {{fr.answer}}\nSpanish: {{es.answer}}",
)

# ------------------------------------------------------------------ #
# Printer node – prints final summary to terminal while preserving chain type safety
# ------------------------------------------------------------------ #

def _print_result(item: DualSummary):  # noqa: D401 – simple side effect
    print("\n============ RESULT ============")
    print(item.summary)
    print("===============================\n")
    return [item]

printer_node = apply(_print_result, input_model=DualSummary, output_model=DualSummary, name="print_result")

# ------------------------------------------------------------------ #
# Chain definition
# ------------------------------------------------------------------ #
full_chain = Chain(
    name="Gemma Feature Demo",
    steps=[
        qa_step,
        filter_node,
        [translation_branch_fr, translation_branch_es],  # Parallel branches with join
        aggregator_step,
        printer_node,
    ],
) 