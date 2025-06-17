"""02 – Financial metric extractor.

Offline heuristic implementation that extracts
`company`, `metric` (revenue, profit, loss), and `value` from a short earnings
sentence.  Swap to an LLM `Step` by uncommenting the bottom block.
"""

import re
from typing import List

from pydantic import BaseModel, Field

from chainette import Chain
from chainette.core.apply import ApplyNode

# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #

class Sentence(BaseModel):
    text: str = Field(..., description="Earnings sentence")

class Metric(BaseModel):
    company: str
    metric: str
    value: str

# --------------------------------------------------------------------------- #
# Heuristic extractor
# --------------------------------------------------------------------------- #

_METRIC_KWS = {
    "revenue": ["revenue", "sales"],
    "profit": ["profit", "income"],
    "loss": ["loss", "deficit"],
}

def _extract(sent: Sentence) -> List[Metric]:  # noqa: D401
    txt = sent.text
    # Company: first capitalised words before metric keyword
    company_match = re.match(r"([A-Z][A-Za-z0-9 &]+?) (?:reported|posted|announced)", txt)
    company = company_match.group(1) if company_match else "Unknown"

    metric_key = "unknown"
    for key, kws in _METRIC_KWS.items():
        if any(k in txt.lower() for k in kws):
            metric_key = key
            break

    value_match = re.search(r"(\$?[0-9\.]+[MB%]?)", txt)
    value = value_match.group(1) if value_match else "?"

    return [Metric(company=company, metric=metric_key, value=value)]

extract_step = ApplyNode(_extract, id="extract", input_model=Sentence)

# --------------------------------------------------------------------------- #
# Chain
# --------------------------------------------------------------------------- #

fin_metrics_chain = Chain(name="Financial Metrics Demo", steps=[extract_step])

# --------------------------------------------------------------------------- #
# Optional LLM extractor (commented)
# --------------------------------------------------------------------------- #
# from chainette import Step, register_engine
# register_engine("openai_default", backend="openai", model="gpt-4.1-mini")
# extract_step = Step(
#     id="extract",
#     input_model=Sentence,
#     output_model=Metric,
#     engine_name="openai_default",
#     system_prompt="Extract the company name, financial metric (revenue, profit, loss), and value from the sentence.",
#     user_prompt="{{chain_input.text}}",
# ) 