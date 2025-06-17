from datasets import load_dataset
from pydantic import BaseModel, Field
from chainette import Step, Chain, register_engine, SamplingParams

# Schemas
class ClinicalCase(BaseModel):
    text: str = Field(..., description="The clinical case description")

class ClinicalNote(BaseModel):
    note_text: str = Field(..., description="The clinical case rewritten as a physician's note, including sections like HPI, PMH, Exam, Assessment, and Plan if applicable based on the input.")
    primary_diagnosis: str = Field(..., description="The primary diagnosis or most likely condition")
    specialty: str = Field(..., description="The medical specialty most relevant to this case (e.g., Cardiology, Neurology, Emergency Medicine)")
    urgency_level: str = Field(..., description="Clinical urgency level: Low, Medium, High, or Critical")
    keywords: list[str] = Field(..., description="List of keywords or phrases relevant to the case")
    medications_mentioned: list[str] = Field(..., description="List of medications mentioned or implied")
    symptoms_mentioned: list[str] = Field(..., description="List of symptoms or clinical findings mentioned or implied")
    diagnostic_tests_mentioned: list[str] = Field(..., description="List of diagnostic tests or procedures mentioned or implied")
    estimated_patient_age_range: str = Field(..., description="Estimated age range of the patient (e.g., '20-30', '50-60', '52', 'Pediatric', 'Geriatric')")

# Engine Registration
register_engine(
    name="medgemma",
    model="google/medgemma-27b-text-it",
    dtype="bfloat16",
    gpu_memory_utilization=0.9,
    enable_reasoning=False,
    max_model_len=8192,
    tensor_parallel_size=1,
)

# Prompt Templates
CLINICAL_NOTE_SYSTEM_PROMPT = """\
Vous êtes un médecin hospitalier français rédigeant une note clinique dans le dossier patient.

Rédigez dans le style authentique des notes cliniques hospitalières françaises :
- Style télégraphique et concis
- Abréviations médicales françaises courantes
- Terminologie médicale française standard
- Présentation structurée selon les sections cliniques pertinentes
- Ton professionnel et factuel

Adaptez la structure et le contenu selon les informations disponibles dans le cas clinique fourni.
Maintenez toutes les informations médicales importantes du cas original.

IMPORTANT : Ne pas utiliser de placeholders ou d'éléments génériques. Créez une note réaliste avant tout, avec des détails médicaux précis et cohérents basés uniquement sur les informations du cas présenté. Si certaines informations ne sont pas disponibles, ne les inventez pas et ne laissez pas d'espaces à compléter.
"""
# CLINICAL_NOTE_SYSTEM_PROMPT = """\
# Vous êtes un médecin hospitalier français rédigeant une note clinique dans le dossier patient.

# Rédigez dans le style authentique des notes cliniques hospitalières françaises :
# - Style télégraphique et concis
# - Abréviations médicales françaises courantes
# - Terminologie médicale française standard
# - Présentation structurée selon les sections cliniques pertinentes
# - Ton professionnel et factuel

# Adaptez la structure et le contenu selon les informations disponibles dans le cas clinique fourni.
# Maintenez toutes les informations médicales importantes du cas original.

# IMPORTANT : Ne pas utiliser de placeholders ou d'éléments génériques. Créez une note réaliste avant tout, avec des détails médicaux précis et cohérents basés uniquement sur les informations du cas présenté. Si certaines informations ne sont pas disponibles, ne les inventez pas et ne laissez pas d'espaces à compléter.
# """

# Steps
clinical_note_generation_step = Step(
    input_model=ClinicalCase,
    output_model=ClinicalNote,
    engine_name="medgemma",
    id="clinical_note_generation",
    name="Clinical Note Generation",
    emoji="📝",
    sampling=SamplingParams(temperature=0.6, max_tokens=4096),
    system_prompt=CLINICAL_NOTE_SYSTEM_PROMPT,
    user_prompt="Transformez ce cas clinique en note hospitalière française :\n\n{{chain_input.text}}",
)

# Chain Definition
clinical_note_chain = Chain(
    name="Clinical Case to Note Transformation",
    emoji="📝⚕️",
    steps=[
        clinical_note_generation_step,
    ],
    batch_size=10,
)

if __name__ == "__main__":
    print("Starting Clinical Case to Note Transformation example...")

    # Load the French translated clinical cases dataset
    data_dir = "rntc/open-clinical-cases-pubmed-comet"
    full_dataset = load_dataset(data_dir, split="fr_translated")
    
    # Select 10 samples
    sample_size = 100000
    sample_data = list(full_dataset.select(range(sample_size)))
    
    # Use the 'text' column as clinical cases
    input_cases = [ClinicalCase(text=item["text"]) for item in sample_data]
    print(f"Loaded {len(input_cases)} clinical cases for processing.")

    print("Running the clinical note generation chain...")
    result_dataset_dict = clinical_note_chain.run(
        input_cases,
        output_dir="/scratch/rtouchen/output_clinical_notes",
        fmt="jsonl",
        generate_flattened_output=True,
        max_lines_per_file=1000,
        debug=True
    )
    print("Chain execution finished.")
