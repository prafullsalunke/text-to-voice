from contextlib import asynccontextmanager
from typing import Optional

import jwt
from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field

from config import settings
from synthesizer import Synthesizer

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

synthesizer = Synthesizer(model_id=settings.model_id)


@asynccontextmanager
async def lifespan(app: FastAPI):
    synthesizer.load()
    yield


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


@app.get("/health", dependencies=[Depends(require_api_token)])
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
    cfg_value: float = 2.0
    inference_timesteps: int = 10


@app.post("/synthesize", dependencies=[Depends(require_api_token)])
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
