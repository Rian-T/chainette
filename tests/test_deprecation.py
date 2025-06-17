import warnings

from chainette.engine.registry import register_engine, get_engine_config

def test_vllm_local_deprecation_warning():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        cfg = register_engine("legacy", model="dummy", backend="vllm_local")
        assert cfg.backend == "vllm_api"
        assert cfg.endpoint == "http://localhost:8000/v1"
        assert any("vllm_local" in str(wn.message) for wn in w), "Deprecation warning not emitted"
    # Ensure registry returns updated config
    cfg2 = get_engine_config("legacy")
    assert cfg2 is cfg 