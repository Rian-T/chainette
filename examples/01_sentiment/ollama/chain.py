"""01 – Sentiment classification demo (Ollama backend).

Runs against a local Ollama daemon exposing the HTTP chat endpoint.
Ensure you have Ollama installed and `ollama serve` running.
"""
from pydantic import BaseModel, Field
from chainette import Chain, Step, register_engine
from chainette.core.step import SamplingParams

# --------------------------------------------------------------------------- #
# Data models
# --------------------------------------------------------------------------- #


class Review(BaseModel):
    review: str = Field(..., description="User review text")


class Sentiment(BaseModel):
    sentiment: str  # positive | negative | neutral


# --------------------------------------------------------------------------- #
# Register engine (local Ollama HTTP API)
# --------------------------------------------------------------------------- #

register_engine(
    "ollama_1b",
    backend="ollama_api",
    model="gemma3:1b",
)

register_engine(
    "ollama_4b",
    backend="ollama_api",
    model="gemma3:4b",
)

sentiment_step_1b = Step(
    id="sentiment_1b",
    name="Sentiment (1B)",
    engine_name="ollama_1b",
    input_model=Review,
    output_model=Sentiment,
    sampling=SamplingParams(temperature=0.0),
    system_prompt="Classify the sentiment of the given customer review as positive, neutral, or negative.",
    user_prompt="{{chain_input.review}}",
)

sentiment_step_4b = Step(
    id="sentiment_4b",
    name="Sentiment (4B)",
    engine_name="ollama_4b",
    input_model=Review,
    output_model=Sentiment,
    sampling=SamplingParams(temperature=0.0),
    system_prompt="Classify the sentiment of the given customer review as positive, neutral, or negative.",
    user_prompt="{{chain_input.review}}",
)


# --------------------------------------------------------------------------- #
# Chain
# --------------------------------------------------------------------------- #

sentiment_chain = Chain(
    name="Sentiment Demo (Ollama Multi-Model)",
    steps=[sentiment_step_1b, sentiment_step_4b],
) 