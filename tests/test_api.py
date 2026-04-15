from unittest.mock import MagicMock, patch
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


def test_synthesize_returns_wav_bytes():
    with patch("main.synthesizer", _make_synth()):
        from main import app
        r = TestClient(app).post("/synthesize", json={"text": "Hello world"})
    assert r.status_code == 200
    assert "audio/wav" in r.headers["content-type"]
    assert len(r.content) > 0


def test_synthesize_rejects_empty_text():
    with patch("main.synthesizer", _make_synth()):
        from main import app
        r = TestClient(app).post("/synthesize", json={"text": ""})
    assert r.status_code == 422


def test_synthesize_rejects_missing_text():
    with patch("main.synthesizer", _make_synth()):
        from main import app
        r = TestClient(app).post("/synthesize", json={})
    assert r.status_code == 422


def test_synthesize_rejects_text_over_500_chars():
    with patch("main.synthesizer", _make_synth()):
        from main import app
        r = TestClient(app).post("/synthesize", json={"text": "x" * 501})
    assert r.status_code == 400
    assert "too long" in r.json()["detail"]


def test_synthesize_returns_503_when_model_not_ready():
    with patch("main.synthesizer", _make_synth(is_ready=False)):
        from main import app
        r = TestClient(app).post("/synthesize", json={"text": "Hello"})
    assert r.status_code == 503


def test_synthesize_passes_voice_description_to_synthesizer():
    synth = _make_synth()
    with patch("main.synthesizer", synth):
        from main import app
        TestClient(app).post("/synthesize", json={
            "text": "Hello",
            "voice_description": "Gentle voice",
        })
    synth.generate.assert_called_once_with(
        text="Hello",
        voice_description="Gentle voice",
        cfg_value=2.0,
        inference_timesteps=10,
    )


def test_synthesize_uses_default_params_when_omitted():
    synth = _make_synth()
    with patch("main.synthesizer", synth):
        from main import app
        TestClient(app).post("/synthesize", json={"text": "Hello"})
    synth.generate.assert_called_once_with(
        text="Hello",
        voice_description=None,
        cfg_value=2.0,
        inference_timesteps=10,
    )


def test_synthesize_forwards_custom_cfg_and_steps():
    synth = _make_synth()
    with patch("main.synthesizer", synth):
        from main import app
        TestClient(app).post("/synthesize", json={
            "text": "Hello",
            "cfg_value": 3.5,
            "inference_timesteps": 20,
        })
    synth.generate.assert_called_once_with(
        text="Hello",
        voice_description=None,
        cfg_value=3.5,
        inference_timesteps=20,
    )


def test_synthesize_returns_500_on_inference_error():
    synth = _make_synth()
    synth.generate.side_effect = RuntimeError("CUDA OOM")
    with patch("main.synthesizer", synth):
        from main import app
        r = TestClient(app).post("/synthesize", json={"text": "Hello"})
    assert r.status_code == 500
    assert r.json()["detail"] == "Synthesis failed"
