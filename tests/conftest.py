import sys
import io
from unittest.mock import MagicMock

# Prevent ImportError when voxcpm / torch / soundfile are not installed in the test env
for _mod in ("voxcpm", "torch"):
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

# Create a proper soundfile mock with a working write function
if "soundfile" not in sys.modules:
    sf_mock = MagicMock()
    def mock_sf_write(file, data, samplerate, format=None):
        """Mock soundfile.write that actually writes WAV header + data."""
        file.write(b"RIFF")
        file.write(b"\x00\x00\x00\x00")  # Placeholder for file size
        file.write(b"WAVE")
    sf_mock.write = mock_sf_write
    sys.modules["soundfile"] = sf_mock

import numpy as np
import pytest


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
