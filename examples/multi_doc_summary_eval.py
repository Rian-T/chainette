from pydantic import BaseModel

from chainette import Step, Chain, SamplingParams, register_engine


# ---------------------------------------------------------------------------
# Engine registration
# ---------------------------------------------------------------------------

register_engine(
    name="gemma",
    model="google/gemma-3-4b-it",
    dtype="bfloat16",
    gpu_memory_utilization=0.7,
    lazy=True,
)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class Article(BaseModel):
    title: str
    body: str


class Summary(BaseModel):
    summary: str


class Score(BaseModel):
    score: int


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------

summ = Step(
    id="summ",
    name="Summarise",
    input_model=Article,
    output_model=Summary,
    engine_name="gemma",
    sampling=SamplingParams(temperature=0.4, max_tokens=200),
    system_prompt="Write a concise, factual summary (1‑2 sentences).",
    user_prompt="{{body}}",
)

eval_ = Step(
    id="eval",
    name="Judge",
    input_model=Summary,
    output_model=Score,
    engine_name="gemma",
    sampling=SamplingParams(temperature=0),
    system_prompt=(
        "Evaluate the summary's accuracy and completeness on a scale of 1 (poor) to 5 (excellent). "
        "Respond only with the integer score."
    ),
    user_prompt="{{summary}}",
)


# ---------------------------------------------------------------------------
# Chain
# ---------------------------------------------------------------------------

chain = Chain(name="Summ + Eval", steps=[summ, eval_], batch_size=8)


if __name__ == "__main__":
    docs = [Article(title="Example", body="Long article text here …")]
    chain.run(docs)