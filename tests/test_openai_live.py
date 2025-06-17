import os
import json
import pytest

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

    prompt = (
        "Return a JSON object with keys 'name', 'date', 'participants' describing the following event: "
        "Alice and Bob are going to a science fair on Friday."
    )

    outputs = client.generate(prompts=[prompt])
    assert outputs, "No output returned from OpenAI"

    text = outputs[0].outputs[0].text.strip()
    data = json.loads(text)  # should be valid JSON

    # Basic sanity of keys
    assert set(data.keys()) >= {"name", "date", "participants"} 