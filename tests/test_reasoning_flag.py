import warnings
from chainette.engine.registry import register_engine


def test_reasoning_flag_warns():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        cfg = register_engine("gpt_reason", model="gpt-4.1-mini", backend="openai", enable_reasoning=True)
        assert cfg.enable_reasoning is False
        assert any("not supported" in str(wi.message) for wi in w) 