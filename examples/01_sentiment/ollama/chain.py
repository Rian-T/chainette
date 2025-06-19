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
    "ollama_default",
    backend="ollama_api",
    model="phi3:mini",  # Replace with your preferred Ollama model tag
)

sentiment_step = Step(
    id="sentiment",
    name="Sentiment Classifier",
    engine_name="ollama_default",
    input_model=Review,
    output_model=Sentiment,
    sampling=SamplingParams(temperature=0.0),
    system_prompt="Classify the sentiment of the given customer review as positive, neutral, or negative.",
    user_prompt="{{chain_input.review}}",
)

# --------------------------------------------------------------------------- #
# Chain
# --------------------------------------------------------------------------- #

sentiment_chain = Chain(name="Sentiment Demo (Ollama)", steps=[sentiment_step]) 