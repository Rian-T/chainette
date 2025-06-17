from __future__ import annotations

"""Clinical Insights QA example using OpenAI back-ends.

This is a trimmed-down variant of `clinical_insights_qa.py` that:

1. Uses OpenAI Chat models (HTTP) instead of local vLLM.
   • Extraction, QA generation, translations  → `gpt-4.1`.
   • Reasoned answering                       → `gpt-4o-mini` (short name `o4-mini`).
2. Streams the source dataset and processes only the first **10** examples.
3. Writes results to a local `outputs/` directory next to this script.

To run:
    export OPENAI_API_KEY=…
    poetry run python examples/clinical_insights_qa_openai.py

The script exits with code 0 if everything went fine and prints the final
output directory path.
"""

import os
from pathlib import Path
from enum import Enum
from typing import List

from datasets import load_dataset
from pydantic import BaseModel, Field

from chainette import (
    Step,
    Chain,
    Apply,
    Branch,
    register_engine,
    SamplingParams,
)

# --------------------------------------------------------------------------- #
# Engine registration (OpenAI HTTP)                                           #
# --------------------------------------------------------------------------- #

# Common kwargs for both engines
_openai_kwargs = {
    "backend": "openai",
    "endpoint": "https://api.openai.com/v1",
}

register_engine(
    name="gpt4_1",  # extraction / QA / translations
    model="gpt-4.1",
    **_openai_kwargs,
)

register_engine(
    name="o4_mini",  # reasoning-heavy answer step
    model="gpt-4o-mini",  # OpenAI naming convention
    **_openai_kwargs,
)

# --------------------------------------------------------------------------- #
# Schemas                                                                     #
# --------------------------------------------------------------------------- #

class Article(BaseModel):
    text: str = Field(..., description="The full text of the scientific article")


class ExtractedPair(BaseModel):
    clinical_case: str = Field(..., description="The extracted clinical case passage")
    insights: str = Field(..., description="The extracted insights passage corresponding to the clinical case")


class ExtractedPairList(BaseModel):
    extracted_pairs: List[ExtractedPair] = Field(
        ..., description="A list of clinical case and insights pairs"
    )


class ReasoningType(str, Enum):
    INDUCTIVE = "Inductive"
    DEDUCTIVE = "Deductive"
    ABDUCTIVE = "Abductive"


class QAItemTask(BaseModel):
    question: str = Field(..., description="The generated question")
    reasoning_type: ReasoningType = Field(
        ..., description="The type of reasoning required to answer the question"
    )


class QAItemAnalysis(BaseModel):
    reasoning_steps: List[str] = Field(
        ..., description="Steps explaining how to arrive at the answer"
    )


class QAItemSolution(BaseModel):
    answer: str = Field(..., description="The concise answer to the question")
    answer_span: str = Field(..., description="The exact span from Insights that proves the answer")
    confidence: float = Field(..., description="Confidence score for the answer")


class QAItem(BaseModel):
    task: QAItemTask
    analysis: QAItemAnalysis
    solution: QAItemSolution


class Answer(BaseModel):
    question: str = Field(..., description="The question being answered")
    thinking: str = Field(..., description="The reasoning process for answering the question")
    answer: str = Field(..., description="The answer to the question, based on the clinical case")


class TranslatedQA(BaseModel):
    clinical_case: str = Field(..., description="The translated clinical case text.")
    question: str = Field(..., description="The translated question.")
    answer: str = Field(..., description="The translated answer text.")
    reasoning_content: str = Field(..., description="The translated reasoning content.")


# --------------------------------------------------------------------------- #
# Prompt templates (identical to original example)                            #
# --------------------------------------------------------------------------- #

EXTRACT_SYSTEM_PROMPT = """
You are given a scientific article as plain text. Your task is to extract every "Clinical Case" passage together with its matching "Insights" passage, ensuring each Insights refers only to that specific Clinical Case.

Instructions:
1.  Identify Pairs: Locate all passages describing a single patient's case and the corresponding passages that interpret or reflect exclusively on that specific case.
2.  Extract Verbatim: Copy the exact text for each "clinical_case" and its matching "insights".
3.  Return Empty If None Found: If the article contains no clear clinical cases or insights pairs, return an empty list of extracted pairs.

Definitions:
- Clinical Case: A detailed description of a single patient's medical condition, symptoms, and history etc.
- Insights: Interpretations, analyses, or conclusions drawn from the clinical case.
"""

QA_SYSTEM_PROMPT = """
You are given a single Clinical case and its matching Insights. Your job is to generate one challenging QA item.

Task Overview:
- Quiz-taker sees: Only the Clinical Case.
- Grader uses: Both the Clinical Case and the Insights.
- Goal: Create a genuinely hard question that demands significant reasoning and integration of information, solvable using only the Clinical Case text (potentially combined with general medical knowledge). Crucially, verifying the answer's correctness must be trivial for the grader by simply checking the provided Answer span within the Insights.

Instructions:
1.  Generate a Question that is hard, non-trivial, and requires integrating multiple details from the Clinical Case.
2.  Determine the Reasoning type required to answer the question. It must be one of: Inductive, Deductive, or Abductive.
3.  Provide Reasoning steps (from YOUR perspective) explaining how to logically arrive at the answer using only the Clinical Case.
4.  State the correct, concise Answer.
5.  Extract the exact Answer span from the Insights that proves the answer is correct.
6.  Assign a Confidence score (0-1) indicating how confident you are that the answer is correct.
Ensure genuine reasoning, not simple fact lookup.
"""

REASONING_ANSWER_SYSTEM_PROMPT = """\
You are given a Clinical Case and a Question.
Your task is to:
1. Answer the question based *only* on the information present in the Clinical Case.
2. Please think deeply before giving your answer. The reasoning should clearly explain how the answer was derived from the clinical case.
"""

TRANSLATE_SYSTEM_PROMPT_TEMPLATE = """\
You are a professional translator. Your task is to translate ALL fields of the provided clinical case information into {language}.
This includes:
- The 'clinical_case' text.
- The 'question' text.
- The 'answer' text.
- The 'reasoning_content' text (this refers to the reasoning provided by the previous step, if available).
Ensure all parts are accurately and fluently translated into {language} and follow the required JSON output format.
"""

TRANSLATE_USER_PROMPT_TEMPLATE = """\
Please translate the following content into {language}. Ensure all fields are translated.

Original Clinical Case:
{{flatten_extracted_pairs.clinical_case}}

Original Question:
{{reasoned_answering.question}}

Original Answer:
{{reasoned_answering.answer}}

Original Reasoning Content:
{{reasoned_answering.thinking}}
"""

# --------------------------------------------------------------------------- #
# Helper Apply function                                                       #
# --------------------------------------------------------------------------- #

def flatten_extracted_pairs(pair_list_container: ExtractedPairList, on_field: str = "extracted_pairs"):
    """Return every `ExtractedPair` contained in *pair_list_container*."""
    items = getattr(pair_list_container, on_field, [])
    return list(items) if isinstance(items, list) else []

flatten_extracted_pairs_node = Apply(
    fn=flatten_extracted_pairs,
    id="flatten_extracted_pairs",
    input_model=ExtractedPairList,
    output_model=ExtractedPair,
)

# --------------------------------------------------------------------------- #
# Step definitions                                                            #
# --------------------------------------------------------------------------- #

extract_step = Step(
    input_model=Article,
    output_model=ExtractedPairList,
    engine_name="gpt4_1",
    id="clinical_case_extraction",
    name="Clinical Case Extraction",
    emoji="📄",
    sampling=SamplingParams(temperature=0.0, max_tokens=4096),
    system_prompt=EXTRACT_SYSTEM_PROMPT,
    user_prompt="{{chain_input.text}}",
    yield_output=False,
)

qa_step = Step(
    input_model=ExtractedPair,
    output_model=QAItem,
    engine_name="gpt4_1",
    id="question_answering",
    name="Question Answering Generation",
    emoji="❓",
    sampling=SamplingParams(temperature=0.1, max_tokens=4096),
    system_prompt=QA_SYSTEM_PROMPT,
    user_prompt="Clinical case:\n{{flatten_extracted_pairs.clinical_case}}\n\nInsights:\n{{flatten_extracted_pairs.insights}}",
)

reasoning_answer_step = Step(
    input_model=QAItem,
    output_model=Answer,
    engine_name="o4_mini",
    id="reasoned_answering",
    name="Reasoned Answering of Generated Question",
    emoji="💡",
    sampling=SamplingParams(temperature=0.1, max_tokens=2048),
    system_prompt=REASONING_ANSWER_SYSTEM_PROMPT,
    user_prompt="Clinical Case:\n{{flatten_extracted_pairs.clinical_case}}\n\nQuestion:\n{{question_answering.task.question}}",
)


translate_fr_step = Step(
    input_model=QAItem,
    output_model=TranslatedQA,
    engine_name="gpt4_1",
    id="translate_to_french",
    name="Translate to French",
    emoji="🇫🇷",
    sampling=SamplingParams(temperature=0.1, max_tokens=2048),
    system_prompt=TRANSLATE_SYSTEM_PROMPT_TEMPLATE.format(language="French"),
    user_prompt=TRANSLATE_USER_PROMPT_TEMPLATE,
)

translate_de_step = Step(
    input_model=QAItem,
    output_model=TranslatedQA,
    engine_name="gpt4_1",
    id="translate_to_german",
    name="Translate to German",
    emoji="🇩🇪",
    sampling=SamplingParams(temperature=0.1, max_tokens=2048),
    system_prompt=TRANSLATE_SYSTEM_PROMPT_TEMPLATE.format(language="German"),
    user_prompt=TRANSLATE_USER_PROMPT_TEMPLATE,
)

translate_es_step = Step(
    input_model=Answer,
    output_model=TranslatedQA,
    engine_name="gpt4_1",
    id="translate_to_spanish",
    name="Translate to Spanish",
    emoji="🇪🇸",
    sampling=SamplingParams(temperature=0.1, max_tokens=2048),
    system_prompt=TRANSLATE_SYSTEM_PROMPT_TEMPLATE.format(language="Spanish"),
    user_prompt=TRANSLATE_USER_PROMPT_TEMPLATE,
)

# --------------------------------------------------------------------------- #
# Chain definition                                                            #
# --------------------------------------------------------------------------- #

clinical_qa_chain = Chain(
    name="Clinical Insights QA (OpenAI)",
    emoji="⚕️🧠",
    steps=[
        extract_step,
        flatten_extracted_pairs_node,
        qa_step,
        reasoning_answer_step,
        Branch(name="translate_fr", steps=[translate_fr_step]),
        Branch(name="translate_de", steps=[translate_de_step]),
        Branch(name="translate_es", steps=[translate_es_step]),
    ],
    batch_size=10,
)

# --------------------------------------------------------------------------- #
# Main execution                                                              #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError("OPENAI_API_KEY environment variable is not set.")

    print("Starting Clinical Insights QA Chainette (OpenAI) example…")

    # 1. Stream dataset and take first 10 suitable articles ------------------ #
    ds_iter = load_dataset("rntc/edu3-clinical", split="train", streaming=True)

    articles: List[Article] = []
    for item in ds_iter:
        # Rough: 1 token ≈ 4 characters
        if len(item["article_text"]) <= 64000:
            articles.append(Article(text=item["article_text"]))
        if len(articles) == 10:
            break

    print(f"Loaded {len(articles)} articles for processing …")

    # 2. Run chain ----------------------------------------------------------- #
    output_dir = Path(__file__).parent / "outputs" / "clinical_qa_openai"
    output_dir.mkdir(parents=True, exist_ok=True)

    result_ds = clinical_qa_chain.run(
        articles,
        output_dir=str(output_dir),
        fmt="jsonl",
        generate_flattened_output=True,
        max_lines_per_file=100,
    )

    print("Chain execution completed. Results written to:", output_dir)

    # Sanity check: ensure each split contains <= 10 items ------------------- #
    for name, dataset in result_ds.items():
        assert len(dataset) <= 10, f"Dataset '{name}' has unexpected size {len(dataset)}"
    print("All output dataset splits have <= 10 items – sanity check passed.")

    print("✅ Example finished successfully.") 