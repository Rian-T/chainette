from pydantic import BaseModel, Field

from chainette import (
    Step,
    Chain,
    Branch,
    SamplingParams,
    register_engine,
)


class RawDesc(BaseModel):
    """Incoming noisy e‑commerce text."""

    text: str = Field(..., description="messy product description")


class Attr(BaseModel):
    """Clean, structured product attributes."""

    brand: str
    model: str
    price_eur: float


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

register_engine(
    name="llama3",
    model="NousResearch/Meta-Llama-3-8B-Instruct",
    dtype="bfloat16",
    gpu_memory_utilization=0.6,
    lazy=True,
)

# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------

extract = Step(
    id="extract",
    name="Extract attributes",
    input_model=RawDesc,
    output_model=Attr,
    engine_name="llama3",
    sampling=SamplingParams(temperature=0),
    system_prompt=(
        "Identify and return the product's brand, model name and price in euros."
    ),
    user_prompt="{{text}}",
)

fr = Step(
    id="fr",
    name="French version",
    input_model=Attr,
    output_model=Attr,
    engine_name="llama3",
    sampling=SamplingParams(temperature=0.3),
    system_prompt=(
        "Translate brand and model to French if appropriate and keep the price value unchanged."
    ),
    user_prompt="{{brand}} {{model}} coûte {{price_eur}} €",
)

es = Step(
    id="es",
    name="Spanish version",
    input_model=Attr,
    output_model=Attr,
    engine_name="llama3",
    sampling=SamplingParams(temperature=0.3),
    system_prompt=(
        "Translate brand and model to Spanish if appropriate and keep the price value unchanged."
    ),
    user_prompt="El {{brand}} {{model}} cuesta {{price_eur}} €",
)

# ---------------------------------------------------------------------------
# Chain
# ---------------------------------------------------------------------------

chain = Chain(
    name="Extract & Translate",
    steps=[
        extract,
        [
            Branch(name="fr", steps=[fr]),
            Branch(name="es", steps=[es]),
        ],
    ],
    batch_size=4,
)


if __name__ == "__main__":
    chain.run([RawDesc(text="🔥 Apple iPhone 15 Pro, 128 Go, 999 €")])