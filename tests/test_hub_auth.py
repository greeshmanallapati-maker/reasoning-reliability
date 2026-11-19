from __future__ import annotations

import os

import pytest

from reasoning_reliability.model.inference import configure_hub_auth


def test_invalid_hub_auth_rejected_before_model_load():
    with pytest.raises(ValueError, match="hub_auth"):
        configure_hub_auth("invalid")


def test_anonymous_hub_auth_sets_documented_environment(monkeypatch):
    # Stop after the environment behavior under test, before any model download.
    monkeypatch.delenv("HF_HUB_DISABLE_IMPLICIT_TOKEN", raising=False)
    configure_hub_auth("anonymous")
    assert os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] == "1"
