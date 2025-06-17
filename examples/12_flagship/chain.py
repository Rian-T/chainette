"""12 – Flagship Enterprise Pipeline Demo.

A realistic, multi-step pipeline mimicking an enterprise incident-management
workflow:

1. **Retrieval** – Pull similar past incidents / KB articles to provide
   context (simple heuristic retrieval in this demo).
2. **Summarise & Recommend** – LLM drafts an executive summary and actionable
   next steps.
3. **Parallel Translations** – Spanish and French versions of the summary.
4. **Quality Grading** – Senior LLM rates the English summary on a 1-10 scale
   for clarity & completeness.

The example showcases branching, history templating, Apply nodes, and LLM
judging ‑ all using the OpenAI backend.
"""

from __future__ import annotations

import os
import re
from typing import List

from pydantic import BaseModel, Field

from chainette import Chain, Step, Branch, ApplyNode, register_engine
from chainette.core.step import SamplingParams

# --------------------------------------------------------------------------- #
# Engine registration (requires OPENAI_API_KEY)
# --------------------------------------------------------------------------- #

OPENAI_KEY = os.getenv("OPENAI_API_KEY", "")
if not OPENAI_KEY:
    raise RuntimeError("OPENAI_API_KEY environment variable not set – required for examples.")

register_engine(
    "openai_default",
    backend="openai",
    model="gpt-4.1-mini",
    api_key=OPENAI_KEY,
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
    engine_name="openai_default",
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
        engine_name="openai_default",
        sampling=SamplingParams(temperature=0.0),
        system_prompt=f"Translate the following summary and actions into {language_label}. Respond only with the translated text.",
        user_prompt="{{summarise.summary}}\n\nActions:\n{{summarise.actions}}",
    )

branch_es = Branch("es", [ _make_translation_step("es", "Spanish") ])
branch_fr = Branch("fr", [ _make_translation_step("fr", "French") ])

# --------------------------------------------------------------------------- #
# Step 4 – Quality grading
# --------------------------------------------------------------------------- #

grade_step = Step(
    id="grade",
    name="Grade Summary Quality",
    input_model=Summary,
    output_model=Grade,
    engine_name="openai_default",
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

flagship_chain = Chain(
    name="Flagship Incident Pipeline",
    steps=[
        retrieve_node,
        summarise_step,
        [branch_es, branch_fr],  # parallel translations
        grade_step,
    ],
) 