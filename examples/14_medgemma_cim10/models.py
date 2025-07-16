from pydantic import BaseModel, Field
from typing import List, Optional


class CIM10Input(BaseModel):
    """Input model for CIM-10 medical codes and descriptions."""
    
    code: str = Field(description="CIM-10 medical code (e.g., F02.00)")
    label: str = Field(description="Medical condition label in French")
    description: Optional[str] = Field(default="", description="Additional description")
    note: Optional[str] = Field(default="", description="Additional notes")
    inclusion_notes: List[str] = Field(default_factory=list, description="Inclusion criteria")
    exclusion_notes: List[str] = Field(default_factory=list, description="Exclusion criteria")
    parent_label: Optional[str] = Field(default="", description="Parent category label")
    full_description: Optional[str] = Field(default="", description="Complete formatted description")


class SyntheticTextbook(BaseModel):
    """Output model for textbook-quality synthetic medical text."""
    
    title: str = Field(description="Educational title for the medical condition")
    content: str = Field(
        description="Comprehensive textbook-style explanation (multiple paragraphs) covering "
        "definition, pathophysiology, clinical presentation, diagnosis, and treatment"
    )
    key_points: List[str] = Field(
        description="3-5 key learning points summarizing the condition"
    )
    clinical_context: str = Field(
        description="Real-world clinical scenarios and applications"
    )