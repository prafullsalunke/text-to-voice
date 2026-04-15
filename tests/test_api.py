from unittest.mock import MagicMock, patch
import pytest
from fastapi.testclient import TestClient


def _make_synth(is_ready: bool = True) -> MagicMock:
    synth = MagicMock()
    synth.is_ready = is_ready
    synth.device = "cuda:0"
    synth.vram_used_gb = 7.8
    synth.generate.return_value = b"RIFF" + b"\x00" * 44
    return synth


def test_health_returns_ok():
    with patch("main.synthesizer", _make_synth()):
        from main import app
        r = TestClient(app).get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model"] == "openbmb/VoxCPM2"
    assert body["device"] == "cuda:0"
    assert "vram_used_gb" in body


def test_health_returns_503_when_model_not_ready():
    with patch("main.synthesizer", _make_synth(is_ready=False)):
        from main import app
        r = TestClient(app).get("/health")
    assert r.status_code == 503
    assert r.json()["detail"] == "Model not ready"
