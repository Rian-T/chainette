from chainette.utils.prompt import build_prompt
from chainette.engine.registry import register_engine


class DummyStep:  # noqa: D101
    def __init__(self):
        self.system_prompt = "You are a bot."
        self.user_prompt = "Hello {{name}}"
        self.engine_name = "dummy_ollama"
        self.tokenizer = None


def test_build_prompt_with_ollama_backend():
    # Register dummy engine with 'ollama' backend to avoid tokenizer
    register_engine(name="dummy_ollama", model="irrelevant", backend="ollama")

    step = DummyStep()
    prompt = build_prompt(step, {"name": "Alice"})

    assert prompt.startswith("## system")
    assert "Alice" in prompt
    assert prompt.endswith("## assistant\n") 