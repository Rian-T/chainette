import os
from datasets import load_dataset, Dataset, concatenate_datasets
from pydantic import BaseModel, Field, field_validator
from enum import Enum
from chainette import Step, Chain, Apply, register_engine, SamplingParams, Branch
import glob

# Schemas
class Article(BaseModel):
    text: str = Field(..., description="The full text of the scientific article")

class ExtractedPair(BaseModel):
    clinical_case: str = Field(..., description="The extracted clinical case passage")
    insights: str = Field(..., description="The extracted insights passage corresponding to the clinical case")

class ExtractedPairList(BaseModel):
    extracted_pairs: list[ExtractedPair] = Field(..., description="A list of clinical case and insights pairs")

class ReasoningType(str, Enum):
    INDUCTIVE = "Inductive"
    DEDUCTIVE = "Deductive"
    ABDUCTIVE = "Abductive"

class QuestionDifficulty(str, Enum):
    EASY = "Easy"
    MEDIUM = "Medium"
    HARD = "Hard"

class QAItemTask(BaseModel):
    question: str = Field(..., description="The generated question")
    reasoning_type: ReasoningType = Field(..., description="The type of reasoning required to answer the question (Inductive, Deductive, or Abductive)")

class QAItemAnalysis(BaseModel):
    reasoning_steps: list[str] = Field(..., description="Steps explaining how to arrive at the answer")

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

# Renamed and simplified from TranslatedReasoningQA
class TranslatedQA(BaseModel):
    clinical_case: str = Field(..., description="The translated clinical case text.")
    question: str = Field(..., description="The translated question.")
    answer: str = Field(..., description="The translated answer text.")
    reasoning_content: str = Field(..., description="The translated reasoning content.")

# Engine Registration
register_engine(
    name="llama3_3_70b_instruct",
    model="meta-llama/Llama-3.3-70B-Instruct",
    dtype="bfloat16",
    gpu_memory_utilization=0.95,
    enable_reasoning=False,
    max_model_len=32768,
    tensor_parallel_size=4,
)

register_engine(
    name="deepseek_r1_distill_llama_70b_reasoning",
    model="deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
    dtype="bfloat16",
    gpu_memory_utilization=0.95,
    enable_reasoning=True,
    reasoning_parser="deepseek_r1",
    max_model_len=16384,
    tensor_parallel_size=4,
)

# Prompt Templates
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

# Steps
extract_step = Step(
    input_model=Article,
    output_model=ExtractedPairList,
    engine_name="llama3_3_70b_instruct",
    id="clinical_case_extraction",
    name="Clinical Case Extraction",
    emoji="📄",
    sampling=SamplingParams(temperature=0.0, max_tokens=8192),
    system_prompt=EXTRACT_SYSTEM_PROMPT,
    user_prompt="{{chain_input.text}}",
    yield_output=False,
)

qa_step = Step(
    input_model=ExtractedPair,
    output_model=QAItem,
    engine_name="llama3_3_70b_instruct",
    id="question_answering",
    name="Question Answering Generation",
    emoji="❓",
    sampling=SamplingParams(temperature=0.1, max_tokens=8192),
    system_prompt=QA_SYSTEM_PROMPT,
    user_prompt="Clinical case:\n{{flatten_extracted_pairs.clinical_case}}\n\nInsights:\n{{flatten_extracted_pairs.insights}}",
)

reasoning_answer_step = Step(
    input_model=QAItem,
    output_model=Answer,
    engine_name="deepseek_r1_distill_llama_70b_reasoning",
    id="reasoned_answering",
    name="Reasoned Answering of Generated Question",
    emoji="💡",
    sampling=SamplingParams(temperature=0.1, max_tokens=4096),
    system_prompt=REASONING_ANSWER_SYSTEM_PROMPT,
    user_prompt="Clinical Case:\\n{{flatten_extracted_pairs.clinical_case}}\\n\\nQuestion:\\n{{question_answering.task.question}}",
)

translate_fr_step = Step(
    input_model=QAItem,
    output_model=TranslatedQA,
    engine_name="llama3_3_70b_instruct",
    id="translate_to_french",
    name="Translate to French",
    emoji="🇫🇷",
    sampling=SamplingParams(temperature=0.1, max_tokens=4096),
    system_prompt=TRANSLATE_SYSTEM_PROMPT_TEMPLATE.format(language="French"),
    user_prompt=TRANSLATE_USER_PROMPT_TEMPLATE,
)

translate_de_step = Step(
    input_model=QAItem,
    output_model=TranslatedQA,
    engine_name="llama3_3_70b_instruct",
    id="translate_to_german",
    name="Translate to German",
    emoji="🇩🇪",
    sampling=SamplingParams(temperature=0.1, max_tokens=4096),
    system_prompt=TRANSLATE_SYSTEM_PROMPT_TEMPLATE.format(language="German"),
    user_prompt=TRANSLATE_USER_PROMPT_TEMPLATE,
)

translate_es_step = Step(
    input_model=Answer,
    output_model=TranslatedQA,
    engine_name="llama3_3_70b_instruct",
    id="translate_to_spanish",
    name="Translate to Spanish",
    emoji="🇪🇸",
    sampling=SamplingParams(temperature=0.1, max_tokens=4096),
    system_prompt=TRANSLATE_SYSTEM_PROMPT_TEMPLATE.format(language="Spanish"),
    user_prompt=TRANSLATE_USER_PROMPT_TEMPLATE,
)

# Custom Flattening Function
def flatten_extracted_pairs(pair_list_container: ExtractedPairList, on_field: str = "extracted_pairs") -> list[ExtractedPair]:
    # This function now processes one ExtractedPairList container at a time.
    # It returns a list of individual ExtractedPair items found within that container.
    individual_pairs = []
    if hasattr(pair_list_container, on_field):
        items = getattr(pair_list_container, on_field)
        if isinstance(items, list):
            individual_pairs.extend(items)
    return individual_pairs

# Chain Definition
clinical_qa_chain = Chain(
    name="Clinical Insights QA and Translation",
    emoji="⚕️🌍",
    steps=[
        extract_step,
        Apply(fn=flatten_extracted_pairs, id="flatten_extracted_pairs", input_model=ExtractedPairList, output_model=ExtractedPair),
        qa_step,
        reasoning_answer_step,
        Branch(name="translate_fr", steps=[translate_fr_step]),
        Branch(name="translate_de", steps=[translate_de_step]),
        Branch(name="translate_es", steps=[translate_es_step]),
    ],
    batch_size=1000,
)

if __name__ == "__main__":
    print("Starting Clinical Insights Extraction and QA Chainette example...")

    # Load dataset
    data_dir = "rntc/edu3-clinical"
    full_dataset = load_dataset(data_dir, split="train")

    # Select 500,000 samples
    sample_size = min(500000, len(full_dataset))
    sample_data = list(full_dataset.select(range(sample_size)))
    
    # Filter out articles longer than approximately 16K tokens (using rough approximation)
    filtered_articles = []
    for item in sample_data:
        # Rough approximation: 1 token ≈ 4 characters
        if len(item["article_text"]) <= 64000:  # 16K tokens * 4 chars/token
            filtered_articles.append(Article(text=item["article_text"]))
    
    print(f"Loaded {len(filtered_articles)} articles for processing (filtered from {len(sample_data)} total).")

    # Process in batches of 1000
    batch_size = 1000
    for i in range(0, len(filtered_articles), batch_size):
        batch_start = i
        batch_end = min(i + batch_size, len(filtered_articles))
        batch_articles = filtered_articles[batch_start:batch_end]
        
        print(f"Processing batch {i//batch_size + 1}: articles {batch_start} to {batch_end-1}")
        
        # Incremental output directory
        output_dir = f"/scratch/rtouchen/clinical_insight/output_clinical_qa_batch_{i//batch_size + 1:04d}"
        
        result_dataset_dict = clinical_qa_chain.run(
            batch_articles,
            output_dir=output_dir,
            fmt="jsonl",
            generate_flattened_output=True,
            max_lines_per_file=1000,
        )
        
        print(f"Batch {i//batch_size + 1} completed. Output saved to {output_dir}")
    
    print("Chain execution finished.")
    print("Example finished.")
