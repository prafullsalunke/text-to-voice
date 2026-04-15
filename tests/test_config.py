import os
import importlib
import pytest


def test_default_model_id():
    import config
    importlib.reload(config)
    assert config.settings.model_id == "openbmb/VoxCPM2"


def test_default_port():
    import config
    importlib.reload(config)
    assert config.settings.port == 8000


def test_default_text_max_length():
    import config
    importlib.reload(config)
    assert config.settings.text_max_length == 500


def test_env_override_model_id(monkeypatch):
    monkeypatch.setenv("MODEL_ID", "openbmb/VoxCPM1.5")
    import config
    importlib.reload(config)
    assert config.settings.model_id == "openbmb/VoxCPM1.5"


def test_env_override_port(monkeypatch):
    monkeypatch.setenv("PORT", "9000")
    import config
    importlib.reload(config)
    assert config.settings.port == 9000
