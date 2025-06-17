"""03 – Translate to FR/ES/DE in parallel, then summarise.

Uses real OpenAI calls.  Demonstrates Branch + Join + multi-engine feature.
Translators use `gpt-4.1-nano`; summariser uses `gpt-4.1-mini`.
"""

import os

from pydantic import BaseModel, Field

from chainette import Chain, Branch, Step, register_engine
from chainette.core.step import SamplingParams

# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #

class Sentence(BaseModel):
    text: str = Field(..., description="Original English sentence")

class Translation(BaseModel):
    text: str

class Summary(BaseModel):
    summary: str

# --------------------------------------------------------------------------- #
# Engine registration
# --------------------------------------------------------------------------- #

OPENAI_KEY = os.getenv("OPENAI_API_KEY", "")
if not OPENAI_KEY:
    raise RuntimeError("OPENAI_API_KEY environment variable not set.")

register_engine("openai_nano", backend="openai", model="gpt-4.1-nano", api_key=OPENAI_KEY)
register_engine("openai_mini", backend="openai", model="gpt-4.1-mini", api_key=OPENAI_KEY)

# --------------------------------------------------------------------------- #
# Translation Steps
# --------------------------------------------------------------------------- #

fr_step = Step(
    id="fr",
    name="Translate to FR",
    input_model=Sentence,
    output_model=Translation,
    engine_name="openai_nano",
    sampling=SamplingParams(temperature=0.3),
    system_prompt="Translate the sentence to French.",
    user_prompt="{{chain_input.text}}",
)

es_step = Step(
    id="es",
    name="Translate to ES",
    input_model=Sentence,
    output_model=Translation,
    engine_name="openai_nano",
    sampling=SamplingParams(temperature=0.3),
    system_prompt="Translate the sentence to Spanish.",
    user_prompt="{{chain_input.text}}",
)

de_step = Step(
    id="de",
    name="Translate to DE",
    input_model=Sentence,
    output_model=Translation,
    engine_name="openai_nano",
    sampling=SamplingParams(temperature=0.3),
    system_prompt="Translate the sentence to German.",
    user_prompt="{{chain_input.text}}",
)

fr_branch = Branch(name="fr_branch", steps=[fr_step]).join("fr")
es_branch = Branch(name="es_branch", steps=[es_step]).join("es")
de_branch = Branch(name="de_branch", steps=[de_step]).join("de")

# --------------------------------------------------------------------------- #
# Summary Step (uses mini model)
# --------------------------------------------------------------------------- #

sum_step = Step(
    id="summary",
    name="Summarise",
    input_model=Sentence,  # not used but required
    output_model=Summary,
    engine_name="openai_mini",
    sampling=SamplingParams(temperature=0.0),
    system_prompt="Provide a concise English summary of the sentence using the provided translations as reference.",
    user_prompt=(
        "French: {{fr.text}}\nSpanish: {{es.text}}\nGerman: {{de.text}}"
    ),
)

# --------------------------------------------------------------------------- #
# Chain
# --------------------------------------------------------------------------- #

translate_chain = Chain(
    name="Translate & Summarise Demo",
    steps=[[fr_branch, es_branch, de_branch], sum_step],
) 