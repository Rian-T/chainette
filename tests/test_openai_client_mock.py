import json
import pytest
from chainette.engine.http_client import OpenAIClient

respx = pytest.importorskip("respx")
import httpx  # noqa: E402


@pytest.mark.parametrize("temperature", [None, 0.2])
@respx.mock
def test_openai_client_mock(temperature):
    """OpenAIClient should POST correct payload and parse response text."""
    api_key = "test_key"
    client = OpenAIClient(endpoint="https://api.openai.com/v1", api_key=api_key, model="gpt-4.1-mini")

    # Mock endpoint
    route = respx.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "id": "cmpl-1",
                "object": "chat.completion",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "{\"hello\": \"world\"}"},
                        "finish_reason": "stop",
                    }
                ],
            },
        )
    )

    # Perform call
    prompts = ["Tell me something"]
    outputs = client.generate(prompts=prompts, sampling_params=type("S", (), {"temperature": temperature})())

    # Route should have been called once
    assert route.called, "OpenAI endpoint not called"

    # Validate request payload
    sent = json.loads(route.calls[0].request.content)
    assert sent["model"] == "gpt-4.1-mini"
    assert sent["messages"][0]["content"] == prompts[0]
    assert sent["response_format"] == {"type": "json_object"}
    if temperature is not None:
        assert sent["temperature"] == temperature

    # Validate parsed output
    assert outputs[0].outputs[0].text == '{"hello": "world"}' 