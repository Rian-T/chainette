from pydantic import BaseModel

from chainette import Step, Chain, SamplingParams, register_engine


# ---------------------------------------------------------------------------
# Engine registration
# ---------------------------------------------------------------------------

register_engine(
    name="gemma",
    model="meta-llama/Llama-3.2-3B-Instruct",
    dtype="float16",
    gpu_memory_utilization=0.95,
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
    sampling=SamplingParams(temperature=0.4, max_tokens=2048),
    system_prompt="Write a concise, factual summary (1‑2 sentences).",
    user_prompt="{{body}}",
)

eval_ = Step(
    id="eval",
    name="Judge",
    input_model=Summary,
    output_model=Score,
    engine_name="gemma",
    sampling=SamplingParams(temperature=0, max_tokens=2048),
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
    docs = [
        Article(
            title="New COVID-19 Variant Discovered",
            body="Scientists have identified a new COVID-19 variant in South Africa, designated as B.1.529 or 'Omicron' by the World Health Organization. Early studies suggest it may be more transmissible than previous variants but appears to cause milder symptoms in most patients. The variant contains over 30 mutations to the spike protein, which has raised concerns among virologists worldwide. Researchers are working to determine if existing vaccines provide adequate protection against this new strain, with preliminary laboratory studies showing reduced neutralization by antibodies from previous infections or vaccinations. Health authorities recommend continued mask usage and social distancing measures while more data is collected. Several countries have implemented travel restrictions to southern African nations in response to the emergence of this variant. Pharmaceutical companies have already begun work on updated vaccine formulations that could better target this specific variant if necessary. Public health officials emphasize that basic preventive measures remain effective regardless of the variant. Additionally, advanced genomic surveillance systems are being deployed globally to track the spread and evolution of this new variant as it moves across international borders. The WHO has convened emergency meetings with health officials from affected regions to coordinate a global response strategy."
        ),
        Article(
            title="SpaceX Successfully Launches Satellite Constellation",
            body="SpaceX completed its largest satellite deployment mission to date, successfully launching 60 Starlink satellites into orbit as part of its ambitious global internet coverage initiative. The Falcon 9 rocket lifted off from Cape Canaveral at 3:22 PM EST under perfect weather conditions with visibility extending for miles along the Florida coastline. This marks the company's fifteenth Starlink mission overall, bringing the total number of satellites in the constellation to over 800, approximately halfway to the 1,440 needed for initial global coverage. The first stage booster successfully landed on the 'Just Read the Instructions' drone ship in the Atlantic Ocean, achieving its sixth successful recovery and demonstrating SpaceX's commitment to reusable rocket technology. Each satellite is equipped with four powerful phased array antennas and a single solar array, designed to provide high-speed internet to users on the ground, particularly in remote or underserved areas. The Starlink satellites were deployed at an initial altitude of 280 kilometers before using their ion thrusters to reach their operational orbit of 550 kilometers. SpaceX engineers have implemented design changes to these newer satellites to reduce their brightness and mitigate concerns from astronomers about interference with astronomical observations. The company has already begun limited beta testing of the Starlink service in northern United States and Canada, with users reporting download speeds between 50 to 150 Mbps. Elon Musk, SpaceX's founder, has stated that the Starlink constellation could eventually grow to include as many as 42,000 satellites, revolutionizing global internet infrastructure and providing connectivity to billions of people who currently lack reliable internet access."
        )
    ]
    results = chain.run(docs)
    print(results)