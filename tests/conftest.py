import numpy as np
import pytest
from unittest.mock import MagicMock


@pytest.fixture
def mock_voxcpm_model():
    """A MagicMock that mimics a loaded VoxCPM model instance."""
    model = MagicMock()
    model.tts_model.sample_rate = 48000
    # Return 0.1s of silence at 48kHz
    model.generate.return_value = np.zeros(4800, dtype=np.float32)
    return model


@pytest.fixture
def ready_synthesizer(mock_voxcpm_model):
    """A Synthesizer with its model already injected (no GPU needed)."""
    from synthesizer import Synthesizer

    s = Synthesizer("openbmb/VoxCPM2")
    s.model = mock_voxcpm_model
    s._device = "cuda:0"
    return s
