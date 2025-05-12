import os
from datasets import load_dataset, Dataset, concatenate_datasets
from pydantic import BaseModel, Field
from chainette import Step, Chain, apply, register_engine, SamplingParams, Branch
import glob

# Schemas
class Article(BaseModel):
    text: str = Field(..., description="The full text of the scientific article")

class ExtractedPair(BaseModel):
    clinical_case: str = Field(..., description="The extracted clinical case passage")
    insights: str = Field(..., description="The extracted insights passage corresponding to the clinical case")

class ExtractedPairList(BaseModel):
    extracted_pairs: list[ExtractedPair] = Field(..., description="A list of clinical case and insights pairs")

class QAItemTask(BaseModel):
    question: str = Field(..., description="The generated question")

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
    clinical_case_context_for_task: str = Field(..., description="The verbatim clinical case text that this QA item is based on.")

class ReasonedAnswer(BaseModel):
    clinical_case: str = Field(..., description="The clinical case text used to derive the answer")
    question: str = Field(..., description="The question being answered")
    answer: str = Field(..., description="The answer to the question, based on the clinical case")
    reasoning_content: str = Field(..., description="The step-by-step reasoning provided by the model to arrive at the answer")

# Engine Registration
register_engine(
    name="qwen2.5_32b_instruct",
    model="/lustre/fsn1/projects/rech/rua/uvb79kr/meta-llama--Llama-3.3-70B-Instruct",
    dtype="bfloat16",
    gpu_memory_utilization=0.9,
    enable_reasoning=False,
    lazy=True,
    max_model_len=16384,
    devices=[0,1,2,3,4,5,6,7],
)

register_engine(
    name="qwen3_30b_a3b_reasoning",
    model="/lustre/fsn1/projects/rech/rua/uvb79kr/deepseek-ai--DeepSeek-R1-Distill-Llama-70B",
    dtype="bfloat16",
    gpu_memory_utilization=0.9,
    enable_reasoning=True,
    reasoning_parser="deepseek_r1",
    lazy=True,
    max_model_len=16384,
    devices=[0,1,2,3,4,5,6,7],
)

# Prompt Templates
EXTRACT_SYSTEM_PROMPT = """
You are given a scientific article as plain text. Your task is to extract every "Clinical Case" passage together with its matching "Insights" passage, ensuring each Insights refers only to that specific Clinical Case.

**Instructions:**
1.  **Identify Pairs:** Locate all passages describing a single patient's case and the corresponding passages that interpret or reflect *exclusively* on that specific case.
2.  **Extract Verbatim:** Copy the exact text for each "clinical_case" and its matching "insights".
"""

QA_SYSTEM_PROMPT = """
You are given a single Clinical case and its matching Insights. Your job is to generate one challenging QA item formatted as a JSON object.

**Task Overview:**
- **Quiz-taker sees:** Only the Clinical Case.
- **Grader uses:** Both the Clinical Case and the Insights.
- **Goal:** Create a genuinely **hard** question that demands significant **reasoning** and integration of information, solvable using **only the Clinical Case** text (potentially combined with general medical knowledge). Crucially, verifying the answer's correctness must be **trivial** for the grader by simply checking the provided **Answer span** within the Insights.

**Instructions:**
1.  Generate a **Question** that is hard, non-trivial, and requires integrating multiple details from the Clinical Case.
2.  Provide **Reasoning steps** (from YOUR perspective) explaining how to logically arrive at the answer using *only* the Clinical Case.
3.  State the correct, concise **Answer**.
4.  Extract the exact **Answer span** from the Insights that proves the answer is correct.
5.  Ensure genuine reasoning, not simple fact lookup.
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
- The 'reasoning_content' text.
Ensure all parts are accurately and fluently translated into {language}.
"""

TRANSLATE_USER_PROMPT_TEMPLATE = """\
Please translate the following content into {language}. Ensure all fields are translated.

Clinical Case:
{{clinical_case}}

Question:
{{question}}

Answer:
{{answer}}

Reasoning Content:
{{reasoning_content}}
"""

# Steps
extract_step = Step(
    input_model=Article,
    output_model=ExtractedPairList,
    engine_name="qwen2.5_32b_instruct",
    id="clinical_case_extraction",
    name="Clinical Case Extraction",
    emoji="📄",
    sampling=SamplingParams(temperature=0.0, max_tokens=8192),
    system_prompt=EXTRACT_SYSTEM_PROMPT,
    user_prompt="{{text}}",
)

qa_step = Step(
    input_model=ExtractedPair,
    output_model=QAItem,
    engine_name="qwen2.5_32b_instruct",
    id="question_answering",
    name="Question Answering Generation",
    emoji="❓",
    sampling=SamplingParams(temperature=0.1, max_tokens=8192),
    system_prompt=QA_SYSTEM_PROMPT,
    user_prompt="Clinical case:\\n{{clinical_case}}\\n\\nInsights:\\n{{insights}}",
)

reasoning_answer_step = Step(
    input_model=QAItem,
    output_model=ReasonedAnswer,
    engine_name="qwen3_30b_a3b_reasoning",
    id="reasoned_answering",
    name="Reasoned Answering of Generated Question",
    emoji="💡",
    sampling=SamplingParams(temperature=0.1, max_tokens=4096),
    system_prompt=REASONING_ANSWER_SYSTEM_PROMPT,
    user_prompt="Clinical Case:\\n{{clinical_case_context_for_task}}\\n\\nQuestion:\\n{{task.question}}",
)

translate_fr_step = Step(
    input_model=ReasonedAnswer,
    output_model=ReasonedAnswer,
    engine_name="qwen2.5_32b_instruct",
    id="translate_to_french",
    name="Translate to French",
    emoji="🇫🇷",
    sampling=SamplingParams(temperature=0.1, max_tokens=4096),
    system_prompt=TRANSLATE_SYSTEM_PROMPT_TEMPLATE.format(language="French"),
    user_prompt=TRANSLATE_USER_PROMPT_TEMPLATE.format(language="French"),
)

translate_de_step = Step(
    input_model=ReasonedAnswer,
    output_model=ReasonedAnswer,
    engine_name="qwen2.5_32b_instruct",
    id="translate_to_german",
    name="Translate to German",
    emoji="🇩🇪",
    sampling=SamplingParams(temperature=0.1, max_tokens=4096),
    system_prompt=TRANSLATE_SYSTEM_PROMPT_TEMPLATE.format(language="German"),
    user_prompt=TRANSLATE_USER_PROMPT_TEMPLATE.format(language="German"),
)

translate_es_step = Step(
    input_model=ReasonedAnswer,
    output_model=ReasonedAnswer,
    engine_name="qwen2.5_32b_instruct",
    id="translate_to_spanish",
    name="Translate to Spanish",
    emoji="🇪🇸",
    sampling=SamplingParams(temperature=0.1, max_tokens=4096),
    system_prompt=TRANSLATE_SYSTEM_PROMPT_TEMPLATE.format(language="Spanish"),
    user_prompt=TRANSLATE_USER_PROMPT_TEMPLATE.format(language="Spanish"),
)

# Custom Flattening Function
def flatten_extracted_pairs(step_outputs: list[ExtractedPairList], on_field: str = "extracted_pairs", **kwargs) -> list[ExtractedPair]:
    all_individual_pairs = []
    for pair_list_container in step_outputs:
        if hasattr(pair_list_container, on_field):
            all_individual_pairs.extend(getattr(pair_list_container, on_field))
    return all_individual_pairs

# Chain Definition
clinical_qa_chain = Chain(
    name="Clinical Insights Extraction and QA with Translations",
    emoji="⚕️🌍",
    steps=[
        extract_step,
        apply(flatten_extracted_pairs, on_field="extracted_pairs", to=ExtractedPair),
        qa_step,
        reasoning_answer_step,
        [
            Branch(name="translate_fr", steps=[translate_fr_step]),
            Branch(name="translate_de", steps=[translate_de_step]),
            Branch(name="translate_es", steps=[translate_es_step]),
        ],
    ],
    batch_size=5,
)

if __name__ == "__main__":
    print("Starting Clinical Insights Extraction and QA Chainette example...")

    try:
        # Define the directory containing parquet files
        data_dir = "/lustre/fsn1/projects/rech/rua/uvb79kr/biomed-augment/rntc--edu3-clinical/data/"
        
        # Get all parquet files in the directory
        parquet_files = glob.glob(os.path.join(data_dir, "train-*.parquet"))
        print(f"Found {len(parquet_files)} parquet files.")
        
        # Load and concatenate all datasets
        datasets = [Dataset.from_parquet(file_path) for file_path in parquet_files]
        full_dataset = concatenate_datasets(datasets)
        
        # Select 100 samples
        sample_size = 10
        sample_size = min(sample_size, len(full_dataset))
        sample_data = list(full_dataset.select(range(sample_size)))
        
        input_articles = [Article(text=item["article_text"]) for item in sample_data]
        print(f"Loaded {len(input_articles)} articles for processing.")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        input_articles = [
            Article(text="Fallback Article 1: Clinical Case: A 50-year-old male presented with chest pain. Insights: The pain was cardiac in origin."),
            Article(text="Fallback Article 2: Clinical Case: A 30-year-old female reported persistent headaches. Insights: Headaches were diagnosed as migraines. Further investigation showed stress as a trigger.")
        ]

    if not input_articles:
        print("No input data to process. Exiting.")
    else:
        try:
            print("Running the chainette chain...")
            result_dataset_dict = clinical_qa_chain.run(
                input_articles,
                output_dir="output_clinical_qa",
                fmt="jsonl",
                generate_flattened_output=True,
                max_lines_per_file=100,
            )
            print("Chain execution finished.")
            if result_dataset_dict:
                print(f"Result dataset: {type(result_dataset_dict).__name__}")
        except Exception as e:
            print(f"Error during chainette run: {e}")

    print("Example finished.")
