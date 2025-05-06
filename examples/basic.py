from pydantic import BaseModel
from chainette import Step, Chain, register_engine, SamplingParams

class Question(BaseModel): q: str
class Answer(BaseModel): a: str

register_engine(
    name="llama3", model="meta-llama/Llama-3.2-3B-Instruct",
    dtype="bfloat16", gpu_memory_utilization=0.9, lazy=True, max_model_len=2048
)

ask = Step(
    id="ask", name="Ask",
    input_model=Question, output_model=Answer, engine_name="llama3",
    system_prompt="Answer precisely:",
    user_prompt="{{q}}",
    sampling=SamplingParams(temperature=0.2, max_tokens=4096)
)

chain = Chain(name="Q&A", steps=[ask])
out = chain.run([Question(q="Capital of France?")])
print(out["ask"][0])