# VoxCPM TTS API

A local REST API that wraps [VoxCPM2](https://github.com/OpenBMB/VoxCPM) to convert text into studio-quality speech. Supports plain text-to-speech and voice design — where a natural language description shapes the voice without needing a reference recording.

Built with FastAPI. Returns raw WAV audio that any frontend can play directly.

---

## How it works

VoxCPM2 is a 2-billion parameter tokenizer-free TTS model that generates continuous speech representations directly from text. It supports 30 languages and runs on CUDA (NVIDIA), MPS (Apple Silicon), or CPU.

This API wraps it with two endpoints:

- **`POST /synthesize`** — submit text (and an optional voice description) and receive a WAV file
- **`GET /health`** — check whether the model is loaded and how much memory it is using

Voice design works by prepending a description to the text in VoxCPM's format:

```
(Middle-aged man, warm and authoritative)Welcome to the show.
```

No reference audio required — the model synthesises a matching voice from the description alone.

---

## System requirements

| Hardware | Minimum | Notes |
|---|---|---|
| RAM | 8 GB | Model weights stay in memory between requests |
| GPU VRAM | 8 GB (NVIDIA) / 6 GB (Apple) | Optional — falls back to CPU |
| Disk | 5 GB | HuggingFace model cache |

| Software | Version |
|---|---|
| Python | 3.10 – 3.14 |
| PyTorch | ≥ 2.5.0 |
| CUDA | ≥ 12.0 (NVIDIA only) |
| macOS | Apple Silicon (M1+) supported via MPS |
| Linux/Windows | Supported via CUDA or CPU |

CPU-only works but is slow — expect 15–30 s per generation versus 2–5 s on a GPU.

---

## Installation

### 1. Install VoxCPM

VoxCPM pulls in PyTorch. The recommended way is via [pipx](https://pipx.pypa.io) so its heavy dependencies stay isolated:

```bash
pipx install voxcpm
```

Verify it works:

```bash
voxcpm design --text "Hello world" --output test.wav
```

> **NVIDIA GPU users:** Install the CUDA-enabled PyTorch build first, then voxcpm:
> ```bash
> pip install torch --index-url https://download.pytorch.org/whl/cu121
> pip install voxcpm
> ```

### 2. Clone this repo

```bash
git clone https://github.com/prafullsalunke/text-to-voice.git
cd text-to-voice
```

### 3. Install API dependencies

Install into the same Python environment that has VoxCPM (the pipx venv):

```bash
/Users/$USER/.local/pipx/venvs/voxcpm/bin/pip install \
  fastapi "uvicorn[standard]" pydantic-settings soundfile
```

Or if you prefer a standalone venv:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
# then also: pip install voxcpm (+ torch if on NVIDIA)
```

### 4. Configure (optional)

Copy `.env.example` to `.env` and override any defaults:

```bash
cp .env.example .env
```

```env
MODEL_ID=openbmb/VoxCPM2    # or openbmb/VoxCPM1.5 for a lighter model
PORT=8000
TEXT_MAX_LENGTH=500
```

---

## Running the server

```bash
/Users/$USER/.local/pipx/venvs/voxcpm/bin/uvicorn main:app --port 8000
```

Or if using a standalone venv:

```bash
source venv/bin/activate
uvicorn main:app --port 8000
```

On first start the model weights (~4 GB) are downloaded from HuggingFace and cached in `~/.cache/huggingface`. Subsequent starts load from cache in ~10–30 s.

When you see:

```
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:8000
```

the server is ready.

---

## API reference

### `GET /health`

Check whether the model is loaded.

**Response `200 OK`:**
```json
{
  "status": "ok",
  "model": "openbmb/VoxCPM2",
  "device": "mps:0",
  "vram_used_gb": 5.6
}
```

**Response `503 Service Unavailable`** (model still loading):
```json
{ "detail": "Model not ready" }
```

---

### `POST /synthesize`

Convert text to speech and return a WAV file.

**Request body (JSON):**

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `text` | string | yes | — | Text to speak. Max 500 characters. |
| `voice_description` | string | no | `null` | Natural language voice style, e.g. `"Young woman, gentle voice"`. Omit for default voice. |
| `cfg_value` | float | no | `2.0` | Style adherence strength. Higher = closer to description. |
| `inference_timesteps` | int | no | `10` | Quality vs speed trade-off. Higher = better quality, slower. |

**Response `200 OK`:**
```
Content-Type: audio/wav
Body: raw WAV bytes (48 kHz)
```

**Error responses:**

| Status | Condition |
|---|---|
| `400` | Text exceeds 500 characters |
| `422` | Missing or empty `text` field |
| `500` | Inference failed (e.g. out of memory) |
| `503` | Model not loaded yet |

---

**Examples:**

Basic TTS:
```bash
curl -X POST http://localhost:8000/synthesize \
  -H "Content-Type: application/json" \
  -d '{"text": "Welcome to the future of content creation."}' \
  --output output.wav
```

Voice design:
```bash
curl -X POST http://localhost:8000/synthesize \
  -H "Content-Type: application/json" \
  -d '{
    "text": "This is your voiceover.",
    "voice_description": "Middle-aged man, warm and authoritative"
  }' \
  --output output.wav
```

Play on macOS:
```bash
afplay output.wav
```

Play on Linux:
```bash
aplay output.wav
```

JavaScript (browser):
```js
const res = await fetch('http://localhost:8000/synthesize', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ text: 'Hello world', voice_description: 'Calm narrator' }),
});
const blob = await res.blob();
const url = URL.createObjectURL(blob);
new Audio(url).play();
```

---

## Exposing via Cloudflare Tunnel

To make the API reachable from a remote frontend (e.g. `prafulls.me` or `zeliontech.in`) without opening firewall ports, use [cloudflared](https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/).

### Quick tunnel (no account needed)

Creates a temporary public HTTPS URL that forwards to your local server:

```bash
cloudflared tunnel --url http://localhost:8000
```

Output:
```
Your quick Tunnel has been created! Visit it at:
https://random-name-here.trycloudflare.com
```

Use that URL in your frontend instead of `localhost:8000`. The tunnel stays alive as long as the process runs. The URL changes every time you restart.

### Named tunnel (persistent URL, requires Cloudflare account)

A named tunnel gives you a stable subdomain (e.g. `tts.zeliontech.in`).

**1. Authenticate:**
```bash
cloudflared tunnel login
```

**2. Create the tunnel:**
```bash
cloudflared tunnel create tts-api
```

**3. Create a config file** at `~/.cloudflared/config.yml`:
```yaml
tunnel: tts-api
credentials-file: /Users/$USER/.cloudflared/<tunnel-id>.json

ingress:
  - hostname: tts.zeliontech.in
    service: http://localhost:8000
  - service: http_status:404
```

**4. Add a DNS record:**
```bash
cloudflared tunnel route dns tts-api tts.zeliontech.in
```

**5. Run the tunnel:**
```bash
cloudflared tunnel run tts-api
```

Your API is now available at `https://tts.zeliontech.in`.

**To run tunnel and server together**, add both to a process manager or run them in separate terminal tabs.

---

## Running tests

```bash
source venv/bin/activate   # or use the pipx venv python
pytest tests/ -v
```

All 31 tests run without a GPU — VoxCPM is mocked at the test layer.

---

## CORS

The server allows requests from:

- `http://localhost` (any port) — local frontend development
- `*.zeliontech.in` — production subdomains
- `prafulls.me` — production domain

To allow additional origins, update `allow_origins` and `allow_origin_regex` in `main.py`.
