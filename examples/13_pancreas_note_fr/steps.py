"""Pipeline steps for the Pancreas Cancer French Clinical Note Generator."""

from chainette import Step, SamplingParams

from models import ClinicalCase, ClassifiedCase, ClinicalNote, ExtractedInfo, SpanInfo


# --------------------------------------------------------------------------- #
# Step 1 – Classification
# --------------------------------------------------------------------------- #

classify_step = Step(
    id="classify",
    name="Pancreas Cancer Classifier",
    input_model=ClinicalCase,
    output_model=ClassifiedCase,
    engine_name="openai_mini",
    sampling=SamplingParams(temperature=0.0),
    system_prompt=(
        "You are a medical language model. Classify the following French clinical case as one of three categories: "
        "'pancreas' (pancreatic cancer), 'lymphoma' (lymphatic cancer), or 'other'. "
    ),
    user_prompt="{{chain_input.text}}",
)


# --------------------------------------------------------------------------- #
# Step 2 – Reformulation
# --------------------------------------------------------------------------- #

reformulate_step = Step(
    id="note",
    name="Reformulate into French Hospital Note",
    input_model=ClassifiedCase,
    output_model=ClinicalNote,
    engine_name="gpt-4.1",
    sampling=SamplingParams(temperature=0.3),
    system_prompt=(
        "You are a senior French clinician writing an internal EHR note. Reformulate the provided text into a concise, formal "
        "clinical note that would appear in a real French hospital record. Retain all medically relevant details, dates, findings, "
        "and patient data. Do NOT invent new information. Use realistic sections like 'Motif de consultation', 'Antécédents', "
        "'Examen clinique', 'Hypothèses diagnostiques', 'Plan'."
    ),
    user_prompt="{{chain_input.text}}",
)


# --------------------------------------------------------------------------- #
# Step 3 – Information extraction
# --------------------------------------------------------------------------- #

extract_step = Step(
    id="extract",
    name="Extract Structured Info",
    input_model=ClinicalNote,
    output_model=ExtractedInfo,
    engine_name="o4-mini",
    sampling=SamplingParams(temperature=1.0),
    system_prompt=(
        "Vous êtes un assistant médical. À partir de la note clinique française suivante, remplissez un objet JSON "
        "contenant uniquement les champs prévus. Laissez la valeur null si l'information n'est pas présente. "
    ),
    user_prompt="{{note.note}}",
)


# --------------------------------------------------------------------------- #
# Step 4 – Span localisation
# --------------------------------------------------------------------------- #

span_step = Step(
    id="spans",
    name="Locate Info Spans",
    input_model=ClinicalNote,
    output_model=SpanInfo,
    engine_name="openai_mini",
    sampling=SamplingParams(temperature=0.0),
    system_prompt=(
        "Pour chaque variable d'information clinique (sexe, âge, dates, etc.), "
        "renvoyez l'extrait exact (span) du texte sans modification de la note qui contient cette information. "
        "(ou null si absent)."
    ),
    user_prompt="{{note.note}}",
) 