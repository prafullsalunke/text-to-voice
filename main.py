import asyncio
import json
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import jwt
from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field

from config import settings
from synthesizer import Synthesizer

# --- Auth ---

_bearer = HTTPBearer(auto_error=False)


def require_api_token(
    creds: HTTPAuthorizationCredentials | None = Depends(_bearer),
):
    if settings.api_token is None:
        return  # auth disabled — no public key configured
    if creds is None:
        raise HTTPException(status_code=401, detail="Missing Bearer token")
    try:
        jwt.decode(creds.credentials, settings.api_token, algorithms=["RS256"])
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token has expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token signature")


# --- Voice presets ---

def load_voices() -> dict:
    path = Path("voices.json")
    if path.exists():
        return json.loads(path.read_text())
    return {}


def resolve_voice(preset: dict | str) -> tuple[str | None, str | None]:
    """Return (description, reference_wav_path) from a preset entry."""
    if isinstance(preset, str):
        return preset, None
    return preset.get("description"), preset.get("reference")


# --- Job queue ---

jobs: dict[str, dict] = {}
_queue: asyncio.Queue = asyncio.Queue()
synthesizer = Synthesizer(model_id=settings.model_id)


async def _worker():
    audio_dir = Path(settings.audio_dir)
    audio_dir.mkdir(parents=True, exist_ok=True)
    while True:
        job_id, text, voice_description, reference_wav_path, cfg_value, inference_timesteps = await _queue.get()
        jobs[job_id]["status"] = "processing"
        try:
            wav_bytes = await asyncio.to_thread(
                synthesizer.generate,
                text=text,
                voice_description=voice_description,
                reference_wav_path=reference_wav_path,
                cfg_value=cfg_value,
                inference_timesteps=inference_timesteps,
            )
            path = audio_dir / f"{job_id}.wav"
            path.write_bytes(wav_bytes)
            jobs[job_id]["status"] = "done"
            jobs[job_id]["audio_path"] = str(path)
        except Exception as e:
            jobs[job_id]["status"] = "failed"
            jobs[job_id]["error"] = str(e)
        finally:
            _queue.task_done()


@asynccontextmanager
async def lifespan(app: FastAPI):
    synthesizer.load()
    worker_task = asyncio.create_task(_worker())
    yield
    worker_task.cancel()


app = FastAPI(title="VoxCPM TTS API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://prafulls.me",
        "https://www.prafulls.me",
    ],
    allow_origin_regex=(
        r"(https?://localhost(:\d+)?|https://[a-zA-Z0-9-]+\.zeliontech\.in)"
    ),
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization"],
)


# --- Endpoints ---

@app.get("/health", dependencies=[Depends(require_api_token)])
def health():
    if not synthesizer.is_ready:
        raise HTTPException(status_code=503, detail="Model not ready")
    return {
        "status": "ok",
        "model": settings.model_id,
        "device": synthesizer.device,
        "vram_used_gb": synthesizer.vram_used_gb,
        "queue_depth": _queue.qsize(),
    }


@app.get("/voices", dependencies=[Depends(require_api_token)])
def list_voices():
    return load_voices()


class SynthesizeRequest(BaseModel):
    text: str = Field(..., min_length=1)
    voice: Optional[str] = None             # preset name from voices.json
    voice_description: Optional[str] = None  # raw description (overrides voice)
    cfg_value: float = 2.0
    inference_timesteps: int = 10


@app.post("/synthesize", status_code=202, dependencies=[Depends(require_api_token)])
async def synthesize(req: SynthesizeRequest):
    if not synthesizer.is_ready:
        raise HTTPException(status_code=503, detail="Model not ready")
    if len(req.text) > settings.text_max_length:
        raise HTTPException(
            status_code=400,
            detail=f"Text too long, max {settings.text_max_length} characters",
        )

    # Resolve voice — raw description takes precedence over preset name
    # Falls back to "Alex" preset if neither voice nor voice_description is provided
    voice_description = req.voice_description
    reference_wav_path = None
    voice_name = req.voice if req.voice is not None else (None if voice_description else "Alex")
    if voice_name is not None:
        voices = load_voices()
        if voice_name not in voices:
            raise HTTPException(status_code=400, detail=f"Unknown voice preset: '{voice_name}'")
        preset_desc, preset_ref = resolve_voice(voices[voice_name])
        if voice_description is None:
            voice_description = preset_desc
        reference_wav_path = preset_ref

    job_id = str(uuid.uuid4())
    jobs[job_id] = {"status": "queued"}
    await _queue.put((job_id, req.text, voice_description, reference_wav_path, req.cfg_value, req.inference_timesteps))
    return {"job_id": job_id, "status": "queued"}


@app.get("/jobs/{job_id}", dependencies=[Depends(require_api_token)])
def get_job(job_id: str):
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return {"job_id": job_id, "status": job["status"], "error": job.get("error")}


@app.get("/jobs/{job_id}/audio", dependencies=[Depends(require_api_token)])
def get_job_audio(job_id: str):
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if job["status"] != "done":
        raise HTTPException(status_code=425, detail=f"Job is {job['status']}")
    return FileResponse(job["audio_path"], media_type="audio/wav", filename=f"{job_id}.wav")
