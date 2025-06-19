"""12 – Flagship Enterprise Pipeline Demo (Ollama backend).

Runs the flagship incident-management pipeline using a local Ollama server. Make sure
`ollama serve` is running and the `gemma3:4b` model is pulled (Ollama will pull it
automatically on first use).
"""

from __future__ import annotations

import re
from typing import List

from pydantic import BaseModel, Field

from chainette import Chain, Step, Branch, ApplyNode, register_engine
from chainette.core.step import SamplingParams

# --------------------------------------------------------------------------- #
# Engine registration – local Ollama daemon
# --------------------------------------------------------------------------- #

register_engine(
    "ollama_4b",
    backend="ollama_api",
    model="gemma3:4b",
)

# --------------------------------------------------------------------------- #
# Data models
# --------------------------------------------------------------------------- #


class Incident(BaseModel):
    description: str = Field(..., description="Raw incident description from engineer")


class Retrieval(BaseModel):
    docs: List[str]


class Summary(BaseModel):
    summary: str
    actions: str


class Translation(BaseModel):
    language: str
    text: str


class Grade(BaseModel):
    score: int = Field(..., ge=1, le=10)

# Resolve forward references when using postponed evaluation of annotations.
Retrieval.model_rebuild()

# --------------------------------------------------------------------------- #
# Step 1 – Retrieval (ApplyNode)
# --------------------------------------------------------------------------- #

_KB = [
    "Service X outage due to misconfigured load balancer – mitigated by rolling back config.",
    "High latency in API Y caused by database connection pool exhaustion.",
    "Data pipeline failure triggered by malformed CSV input and insufficient validation.",
    "Intermittent auth errors traced to expired OAuth tokens cached by edge nodes.",
]


def _retrieve(inc: Incident) -> List[Retrieval]:
    desc_words = set(re.findall(r"[a-zA-Z]+", inc.description.lower()))
    scored = []
    for doc in _KB:
        overlap = len(desc_words.intersection(doc.lower().split()))
        scored.append((overlap, doc))
    top_docs = [d for _, d in sorted(scored, reverse=True)[:2]]
    return [Retrieval(docs=top_docs)]


retrieve_node = ApplyNode(
    fn=_retrieve,
    id="retrieve",
    name="Retrieve Similar Incidents",
    input_model=Incident,
    output_model=Retrieval,
)

# --------------------------------------------------------------------------- #
# Step 2 – Summarise & Recommend
# --------------------------------------------------------------------------- #

summarise_step = Step(
    id="summarise",
    name="Summarise & Recommend",
    input_model=Retrieval,
    output_model=Summary,
    engine_name="ollama_4b",
    sampling=SamplingParams(temperature=0.2),
    system_prompt=(
        "You are an SRE assistant. Draft a concise executive summary (<=150 words) "
        "and 3 actionable next steps to resolve the incident."
    ),
    user_prompt=(
        "Incident description:\n{{chain_input.description}}\n\n"
        "Relevant past incidents:\n{{retrieve.docs}}"
    ),
)

# --------------------------------------------------------------------------- #
# Step 3 – Parallel translations (ES & FR)
# --------------------------------------------------------------------------- #

def _make_translation_step(lang_code: str, language_label: str) -> Step:
    return Step(
        id=f"translate_{lang_code}",
        name=f"Translate to {language_label}",
        input_model=Summary,
        output_model=Translation,
        engine_name="ollama_4b",
        sampling=SamplingParams(temperature=0.0),
        system_prompt=f"Translate the following summary and actions into {language_label}. Respond only with the translated text.",
        user_prompt="{{summarise.summary}}\n\nActions:\n{{summarise.actions}}",
    )

branch_es = Branch("es", [_make_translation_step("es", "Spanish")])
branch_fr = Branch("fr", [_make_translation_step("fr", "French")])

# --------------------------------------------------------------------------- #
# Step 4 – Quality grading
# --------------------------------------------------------------------------- #

grade_step = Step(
    id="grade",
    name="Grade Summary Quality",
    input_model=Summary,
    output_model=Grade,
    engine_name="ollama_4b",
    sampling=SamplingParams(temperature=0.0),
    system_prompt=(
        "You are a senior incident commander. Score the provided summary (1-10) based on: "
        "clarity, completeness, and actionability. Be strict; 8+ means exec-ready."
    ),
    user_prompt="Summary:\n{{summarise.summary}}\n\nActions:\n{{summarise.actions}}",
)

# --------------------------------------------------------------------------- #
# Chain definition
# --------------------------------------------------------------------------- #

flagship_chain_ollama = Chain(
    name="Flagship Incident Pipeline (Ollama)",
    steps=[
        retrieve_node,
        summarise_step,
        [branch_es, branch_fr],  # parallel translations
        grade_step,
    ],
) 