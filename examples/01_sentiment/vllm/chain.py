"""01 – Sentiment classification demo (vLLM HTTP backend).

Assumes a vLLM server is running with the OpenAI-compatible endpoint, e.g.:

    python -m vllm.entrypoints.openai.api_server --model mistralai/Mistral-7B-Instruct-v0.2

Default endpoint http://localhost:8000/v1 is used if none provided.
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
# Register engine (vLLM OpenAI-compatible HTTP server)
# --------------------------------------------------------------------------- #

register_engine(
    "vllm_default",
    backend="vllm_api",
    model="mistralai/Mistral-7B-Instruct-v0.2",  # Replace with the model loaded by your vLLM server
    #endpoint="http://localhost:8000/v1",
)

sentiment_step = Step(
    id="sentiment",
    name="Sentiment Classifier",
    engine_name="vllm_default",
    input_model=Review,
    output_model=Sentiment,
    sampling=SamplingParams(temperature=0.0),
    system_prompt="Classify the sentiment of the given customer review as positive, neutral, or negative.",
    user_prompt="{{chain_input.review}}",
)

# --------------------------------------------------------------------------- #
# Chain
# --------------------------------------------------------------------------- #

sentiment_chain = Chain(name="Sentiment Demo (vLLM)", steps=[sentiment_step]) 