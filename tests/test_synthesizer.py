from unittest.mock import MagicMock, patch
import numpy as np
import pytest
from synthesizer import Synthesizer


# --- build_text ---

def test_build_text_no_description():
    s = Synthesizer("openbmb/VoxCPM2")
    assert s.build_text("Hello world", None) == "Hello world"


def test_build_text_with_description():
    s = Synthesizer("openbmb/VoxCPM2")
    result = s.build_text("Hello world", "Young woman, gentle voice")
    assert result == "(Young woman, gentle voice)Hello world"


def test_build_text_empty_description_treated_as_none():
    s = Synthesizer("openbmb/VoxCPM2")
    assert s.build_text("Hello", "") == "Hello"


# --- readiness ---

def test_is_not_ready_before_load():
    s = Synthesizer("openbmb/VoxCPM2")
    assert s.is_ready is False


def test_is_ready_after_model_injected(ready_synthesizer):
    assert ready_synthesizer.is_ready is True


def test_device_before_load():
    s = Synthesizer("openbmb/VoxCPM2")
    assert s.device == "not loaded"


def test_device_after_load(ready_synthesizer):
    assert ready_synthesizer.device == "cuda:0"


# --- load ---

def test_load_calls_from_pretrained():
    s = Synthesizer("openbmb/VoxCPM2")
    with patch("synthesizer.VoxCPM") as mock_cls:
        mock_cls.from_pretrained.return_value = MagicMock()
        s.load()
    mock_cls.from_pretrained.assert_called_once_with(
        "openbmb/VoxCPM2", load_denoiser=False
    )
    assert s.is_ready is True


# --- generate ---

def test_generate_returns_wav_bytes(ready_synthesizer):
    result = ready_synthesizer.generate("Hello world")
    assert isinstance(result, bytes)
    # WAV files start with RIFF magic bytes
    assert result[:4] == b"RIFF"


def test_generate_without_description_passes_plain_text(ready_synthesizer, mock_voxcpm_model):
    ready_synthesizer.generate("Hello world")
    mock_voxcpm_model.generate.assert_called_once_with(
        text="Hello world",
        cfg_value=2.0,
        inference_timesteps=10,
    )


def test_generate_with_description_prepends_it(ready_synthesizer, mock_voxcpm_model):
    ready_synthesizer.generate("Hello", voice_description="Gentle voice")
    mock_voxcpm_model.generate.assert_called_once_with(
        text="(Gentle voice)Hello",
        cfg_value=2.0,
        inference_timesteps=10,
    )


def test_generate_passes_custom_cfg_and_steps(ready_synthesizer, mock_voxcpm_model):
    ready_synthesizer.generate("Hello", cfg_value=3.5, inference_timesteps=20)
    mock_voxcpm_model.generate.assert_called_once_with(
        text="Hello",
        cfg_value=3.5,
        inference_timesteps=20,
    )
