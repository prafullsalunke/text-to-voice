# TTS API Design — VoxCPM2 Backend

**Date:** 2026-04-16
**Status:** Approved

---

## Overview

A synchronous REST API that wraps VoxCPM2 to provide text-to-speech and voice design capabilities for a content creation frontend (voiceovers for videos, slides, and scripts). The API runs locally on a GPU-equipped machine, with the model resident in GPU memory for low-latency responses.

---

## Architecture

```
Frontend (browser)
      │
      │ HTTP POST (text + optional voice description)
      ▼
┌─────────────────────────────────┐
│  FastAPI app  (main.py)         │
│                                 │
│  startup: load VoxCPM2 → GPU   │
│  POST /synthesize               │
│  GET  /health                   │
└────────────┬────────────────────┘
             │ model.generate(...)
             ▼
      VoxCPM2 (GPU, ~8GB VRAM)
             │
             ▼
      WAV bytes → HTTP response
```

**Key decisions:**
- Model loaded once at startup, stays resident in GPU memory between requests
- Every request is blocking — GPU runs, returns WAV bytes, response sent
- Audio returned as raw WAV binary (`Content-Type: audio/wav`); frontend creates a blob URL and plays directly — no intermediate file storage
- CORS enabled for local dev and production domains

---

## API Endpoints

### `POST /synthesize`

Unified endpoint for both basic TTS and voice design. Voice design is TTS with a prepended description — no need for separate routes.

**Request body (JSON):**
```json
{
  "text": "Welcome to the future of content creation.",
  "voice_description": "Middle-aged man, warm and authoritative",
  "language": "en",
  "cfg_value": 2.0,
  "inference_timesteps": 10
}
```

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `text` | string | yes | — | Text to synthesize. Max 500 characters. |
| `voice_description` | string | no | `null` | Natural language voice style (e.g. "Young woman, gentle voice"). When provided, prepended to text as `(description)text`. |
| `language` | string | no | `"en"` | BCP-47 language code. VoxCPM2 supports 30 languages. |
| `cfg_value` | float | no | `2.0` | Classifier-free guidance scale. Controls style adherence. |
| `inference_timesteps` | int | no | `10` | Speed/quality trade-off. Lower = faster, higher = better quality. |

**Response:**
```
HTTP 200
Content-Type: audio/wav
Body: raw WAV bytes (48kHz, mono)
```

When `voice_description` is provided, text sent to VoxCPM is formatted as:
```
(Middle-aged man, warm and authoritative)Welcome to the future of content creation.
```

---

### `GET /health`

**Response (JSON):**
```json
{
  "status": "ok",
  "model": "openbmb/VoxCPM2",
  "device": "cuda:0",
  "vram_used_gb": 7.8
}
```

Returns `503` with `{"status": "loading"}` if the model has not finished loading at startup.

---

## Error Handling

| Scenario | HTTP status | Response body |
|---|---|---|
| Missing or empty `text` | 422 | Pydantic validation error (auto) |
| `text` exceeds 500 characters | 400 | `{"detail": "Text too long, max 500 characters"}` |
| Model not loaded yet | 503 | `{"detail": "Model not ready"}` |
| GPU OOM / inference crash | 500 | `{"detail": "Synthesis failed"}` |

Long scripts should be chunked on the frontend and sent as multiple requests — keeps the API simple and response times predictable.

---

## CORS Configuration

Allowed origins:
- `http://localhost:*` (all local dev ports)
- `*.zeliontech.in`
- `prafulls.me`

---

## Project Structure

```
voxcpm/
├── main.py           # FastAPI app, model lifecycle, endpoints
├── config.py         # Settings: model name, port, CORS origins, text limits
├── synthesizer.py    # VoxCPM2 wrapper: load(), generate() → WAV bytes
├── requirements.txt  # voxcpm, fastapi, uvicorn, soundfile, torch
└── .env              # Optional overrides (model variant, port)
```

**Responsibilities:**
- `synthesizer.py` — owns all VoxCPM interaction. Swap model here without touching API layer.
- `config.py` — centralizes all tuneable values (port, limits, CORS, model ID).
- `main.py` — routing and request/response logic only.

---

## Non-Goals

- Voice cloning (not in scope for this version)
- Authentication / API keys (local deployment, single user)
- Async job queue (synchronous is sufficient for the use case)
- Audio formats other than WAV
- Persistent audio file storage

---

## Hardware Requirements

- Python 3.10–3.12
- PyTorch >= 2.5.0
- CUDA >= 12.0
- ~8GB VRAM (VoxCPM2)
