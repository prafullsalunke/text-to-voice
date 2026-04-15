from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel, Field

from config import settings
from synthesizer import Synthesizer

synthesizer = Synthesizer(model_id=settings.model_id)


@asynccontextmanager
async def lifespan(app: FastAPI):
    synthesizer.load()
    yield


app = FastAPI(title="VoxCPM TTS API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://prafulls.me", "http://prafulls.me"],
    allow_origin_regex=(
        r"(https?://localhost(:\d+)?|https?://[a-zA-Z0-9-]+\.zeliontech\.in)"
    ),
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


@app.get("/health")
def health():
    if not synthesizer.is_ready:
        raise HTTPException(status_code=503, detail="Model not ready")
    return {
        "status": "ok",
        "model": settings.model_id,
        "device": synthesizer.device,
        "vram_used_gb": synthesizer.vram_used_gb,
    }


class SynthesizeRequest(BaseModel):
    text: str = Field(..., min_length=1)
    voice_description: Optional[str] = None
    language: str = "en"
    cfg_value: float = 2.0
    inference_timesteps: int = 10


@app.post("/synthesize")
def synthesize(req: SynthesizeRequest):
    if not synthesizer.is_ready:
        raise HTTPException(status_code=503, detail="Model not ready")
    if len(req.text) > settings.text_max_length:
        raise HTTPException(
            status_code=400,
            detail=f"Text too long, max {settings.text_max_length} characters",
        )
    try:
        wav_bytes = synthesizer.generate(
            text=req.text,
            voice_description=req.voice_description,
            cfg_value=req.cfg_value,
            inference_timesteps=req.inference_timesteps,
        )
    except Exception:
        raise HTTPException(status_code=500, detail="Synthesis failed")
    return Response(content=wav_bytes, media_type="audio/wav")
