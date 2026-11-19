import os

import pytest

from reasoning_reliability.model import HuggingFaceEngine


@pytest.mark.integration
@pytest.mark.skipif(os.getenv("RUN_REAL_MODEL_TEST") != "1", reason="set RUN_REAL_MODEL_TEST=1 to download a model")
def test_real_small_model_inference():
    engine = HuggingFaceEngine(
        {"model_id": "Qwen/Qwen3-0.6B", "device": "auto", "precision": "auto", "native_thinking": False,
         "hub_auth": "anonymous"},
        {"max_new_tokens": 8, "do_sample": False, "timeout_seconds": 120},
    )
    result = engine.generate("Return only FINAL_ANSWER: 4")
    assert result.error is None and result.output_tokens > 0
