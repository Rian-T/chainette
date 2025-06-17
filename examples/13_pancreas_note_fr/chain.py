"""13 – Pancreas Cancer French Clinical Note Generator.

Pipeline:
1. Load 100 items from HuggingFace dataset `rntc/open-clinical-cases-pubmed-comet`,
   split `fr_translated` (done via helper script in this folder).
2. Classify whether case concerns pancreatic cancer (Step – LLM `o4-mini`).
3. Filter to positive cases only (ApplyNode).
4. Reformulate text into realistic French hospital clinical note (Step – LLM).
5. Extract structured information (Step – LLM).
6. Locate text spans for extracted information (Step – LLM).
7. Upload results to HuggingFace Hub (Custom Node).
"""

from chainette import Chain

from config import setup_engines
from steps import classify_step, reformulate_step, extract_step, span_step
from nodes import filter_node, upload_node


def create_pancreas_chain() -> Chain:
    """Create and configure the pancreas cancer pipeline chain."""
    
    # Setup engines
    setup_engines()
    
    # Create the chain
    pancreas_chain = Chain(
        name="Pancreas Cancer FR Note",
        steps=[
            classify_step,
            filter_node,
            reformulate_step,
            extract_step,
            span_step,
            upload_node,
        ],
        batch_size=100,
    )
    
    return pancreas_chain


# Create the chain instance for backwards compatibility
pancreas_chain = create_pancreas_chain()