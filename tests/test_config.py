from reasoning_reliability.utils.config import load_config, merge_config


def test_config_is_reproducible():
    config = load_config("configs/smoke.yaml")
    assert config["benchmark"]["seed"] == 42
    assert merge_config(config, {"benchmark": {"seed": 7}})["benchmark"]["seed"] == 7

