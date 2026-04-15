from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

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
