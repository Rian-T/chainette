from chainette import Step
from .models import CIM10Input, SyntheticTextbook


def create_synthetic_textbook_step() -> Step[CIM10Input, SyntheticTextbook]:
    """
    Generate comprehensive textbook-quality medical content from CIM-10 codes.
    
    This step uses MedGemma-27B to transform structured medical codes into
    educational content suitable for medical training, following the phi-style
    approach of creating high-quality synthetic training data.
    """
    
    prompt_template = """You are a medical education expert creating comprehensive textbook content.

Given this CIM-10 medical code and information:
- Code: {{code}}
- Condition: {{label}}
- Parent Category: {{parent_label}}
- Full Description: {{full_description}}
{% if inclusion_notes %}
- Inclusion Notes: {% for note in inclusion_notes %}• {{note}}{% endfor %}
{% endif %}
{% if exclusion_notes %}
- Exclusion Notes: {% for note in exclusion_notes %}• {{note}}{% endfor %}
{% endif %}

Create comprehensive textbook-quality educational content about this medical condition. Your response should be:

1. **Academically rigorous** - Use proper medical terminology and evidence-based information
2. **Pedagogically structured** - Organize information logically for learning
3. **Clinically relevant** - Include practical applications and real-world context
4. **Comprehensive** - Cover definition, pathophysiology, presentation, diagnosis, and management

Focus on creating content that would be valuable for medical education and training, similar to high-quality medical textbooks. Write in clear, professional medical language that balances technical accuracy with educational clarity.

Provide your response in the exact JSON format requested, ensuring all fields are properly filled with substantial, educational content."""

    return Step(
        name="generate_synthetic_textbook",
        prompt_template=prompt_template,
        engine="medgemma",
        output_schema=SyntheticTextbook
    )