"""Custom nodes for the Pancreas Cancer French Clinical Note Generator pipeline."""

from typing import List

from datasets import Dataset

from chainette import ApplyNode
from chainette.io.writer import RunWriter

from models import ClassifiedCase, SpanInfo


def _filter_relevant(case: ClassifiedCase) -> List[ClassifiedCase]:
    """Filter function to keep only pancreas and lymphoma cases."""
    return [case] if case.category in ("pancreas", "lymphoma") else []


# Filter node to keep only relevant cases
filter_node = ApplyNode(
    fn=_filter_relevant,
    id="filter",
    name="Filter Other Cases",
    input_model=ClassifiedCase,
    output_model=ClassifiedCase,
)


class UploadNode(ApplyNode):
    """Final node: build clean HF row and push to hub."""

    def __init__(self):
        # Minimal init without fn – we override execute anyway
        self.id = "upload"
        self.name = "Upload to HF"
        self.input_model = SpanInfo  # for inspector
        self.output_model = SpanInfo

    def execute(
        self,
        inputs: List[SpanInfo],
        item_histories: List[dict],
        writer: RunWriter | None = None,
        debug: bool = False,
        batch_size: int = 0,
    ) -> tuple[List[SpanInfo], List[dict]]:
        """Execute the upload to HuggingFace Hub."""
        repo_id = "rntc/clinical-reformulate"

        # Build rows --------------------------------------------------------
        rows = []
        for span_rec, hist in zip(inputs, item_histories):
            # Get original case & note from history
            clinical_case = getattr(hist.get("classify"), "text", None) or getattr(hist.get("filter"), "text", "")
            clinical_note = getattr(hist.get("note"), "note", "")

            # Variables string (non-null)
            extract_rec = hist.get("extract")
            variables_lines: list[str] = []
            if extract_rec is not None:
                for key, value in extract_rec.model_dump().items():
                    if value is not None:
                        variables_lines.append(f"{key}: {value}")
            variables = "\n\n".join(variables_lines)

            # Variable locations list of tuples
            var_locs: list[list[str | None]] = []
            span_dict = span_rec.model_dump()
            if extract_rec is not None:
                for key, value in extract_rec.model_dump().items():
                    if value is None:
                        continue
                    span = span_dict.get(f"{key}_span")
                    var_locs.append([str(key)+": "+str(value), span])

            rows.append(
                {
                    "clinical_case": clinical_case,
                    "clinical_note": clinical_note,
                    "variables": variables,
                    "variables_locations": var_locs,
                }
            )

        # Push single Dataset to HF (append=True behaviour)
        try:
            ds = Dataset.from_list(rows)
            ds.push_to_hub(repo_id, split="train")
        except Exception as exc:
            print(f"[WARN] Failed to push to HF Hub: {exc}")

        # Histories unchanged, but attach upload record for traceability
        new_histories = []
        for hist, row in zip(item_histories, rows):
            h = hist.copy()
            h[self.id] = row
            new_histories.append(h)

        if writer is not None:
            writer.add_node_to_graph({"id": self.id, "name": self.name, "type": "Apply"})
            writer.write_step(self.id, rows)

        return inputs, new_histories


# Create upload node instance
upload_node = UploadNode() 