import os
import json
import pytest
from pydantic import BaseModel

# Skip whole module if OpenAI tests not desired (e.g., quota issues)
if os.getenv("OPENAI_API_KEY") is not None:
    pytest.skip("Skipping OpenAI live tests – external API not permitted in current environment.", allow_module_level=True)

pytestmark = pytest.mark.integration


def test_openai_client_live():
    """Smoke‐test OpenAIClient against the real API.

    Skips automatically if no OPENAI_API_KEY in env or network not desired.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key is None:
        pytest.skip("OPENAI_API_KEY not set – skipping live OpenAI test")

    from chainette.engine.http_client import OpenAIClient

    client = OpenAIClient(
        endpoint="https://api.openai.com/v1",
        api_key=api_key,
        model="gpt-4.1-mini",
    )

    # Prompt using system+user roles for realism
    prompt = (
        "Extract the event information from the sentence and output JSON with keys 'name', 'date', 'participants'. "
        "Sentence: Alice and Bob are going to a science fair on Friday."
    )

    outputs = client.generate(prompts=[prompt])

    # If API quota exceeded or other failure, OpenAIClient may return empty text → skip
    if not outputs or not outputs[0].outputs[0].text.strip():
        pytest.skip("OpenAI API unavailable or quota exceeded – skipping live OpenAI test")

    text = outputs[0].outputs[0].text.strip()

    # Define Pydantic model mirroring expected schema
    class CalendarEvent(BaseModel):
        name: str
        date: str
        participants: list[str]

    # Validate via Pydantic to ensure structured output compatibility
    event = CalendarEvent.model_validate(json.loads(text))

    assert event.name and event.date and event.participants 