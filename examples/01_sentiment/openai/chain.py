"""01 – Sentiment classification demo (OpenAI backend).

This is a copy of the original 01_sentiment example using the OpenAI Chat API.
"""
from pydantic import BaseModel, Field
from chainette import Chain, Step, register_engine
from chainette.core.step import SamplingParams
import os

# --------------------------------------------------------------------------- #
# Data models
# --------------------------------------------------------------------------- #


class Review(BaseModel):
    review: str = Field(..., description="User review text")


class Sentiment(BaseModel):
    sentiment: str  # positive | negative | neutral


# --------------------------------------------------------------------------- #
# Register engine (expects OPENAI_API_KEY in env)
# --------------------------------------------------------------------------- #

OPENAI_KEY = os.getenv("OPENAI_API_KEY", "")
if not OPENAI_KEY:
    raise RuntimeError(
        "OPENAI_API_KEY environment variable not set – required for OpenAI example."
    )

register_engine(
    "openai_default",
    backend="openai",
    model="gpt-4.1-mini",
    api_key=OPENAI_KEY,
)

sentiment_step = Step(
    id="sentiment",
    name="Sentiment Classifier",
    engine_name="openai_default",
    input_model=Review,
    output_model=Sentiment,
    sampling=SamplingParams(temperature=0.0),
    system_prompt="Classify the sentiment of the given customer review as positive, neutral, or negative.",
    user_prompt="{{chain_input.review}}",
)

# --------------------------------------------------------------------------- #
# Chain
# --------------------------------------------------------------------------- #

sentiment_chain = Chain(name="Sentiment Demo (OpenAI)", steps=[sentiment_step]) 