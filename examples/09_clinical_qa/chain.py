"""09 – Clinical QA demo.

1. Extract key symptoms from a clinical note (Step, nano model).
2. Suggest a likely diagnosis (Step, mini model).
3. Pose a follow-up question for the patient (Step, mini model).
"""
import os
from pydantic import BaseModel, Field
from chainette import Step, Chain, register_engine
from chainette.core.step import SamplingParams

OPENAI_KEY = os.getenv("OPENAI_API_KEY", "")
if not OPENAI_KEY:
    raise RuntimeError("OPENAI_API_KEY not set")

register_engine("openai_nano", backend="openai", model="gpt-4.1-nano", api_key=OPENAI_KEY)
register_engine("openai_mini", backend="openai", model="gpt-4.1-mini", api_key=OPENAI_KEY)

class Note(BaseModel):
    note: str = Field(..., description="Short clinician note")

class Symptoms(BaseModel):
    symptoms: list[str]

class Diagnosis(BaseModel):
    diagnosis: str

class FollowUp(BaseModel):
    question: str

symptom_step = Step(
    id="symptoms",
    name="Extract Symptoms",
    input_model=Note,
    output_model=Symptoms,
    engine_name="openai_nano",
    sampling=SamplingParams(temperature=0.0),
    system_prompt="List the main patient symptoms mentioned in the note.",
    user_prompt="{{chain_input.note}}",
)

diagnosis_step = Step(
    id="diagnosis",
    name="Suggest Diagnosis",
    input_model=Symptoms,
    output_model=Diagnosis,
    engine_name="openai_mini",
    sampling=SamplingParams(temperature=0.0),
    system_prompt="Provide the most likely diagnosis based on the symptoms.",
    user_prompt="Symptoms: {{symptoms.symptoms}}",
)

follow_step = Step(
    id="followup",
    name="Follow-up Question",
    input_model=Diagnosis,
    output_model=FollowUp,
    engine_name="openai_mini",
    sampling=SamplingParams(temperature=0.2),
    system_prompt="Ask one concise follow-up question to clarify the diagnosis with the patient.",
    user_prompt="Diagnosis: {{diagnosis.diagnosis}}",
)

clinical_chain = Chain(name="Clinical QA Demo", steps=[symptom_step, diagnosis_step, follow_step]) 