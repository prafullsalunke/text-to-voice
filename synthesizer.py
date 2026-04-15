import io

import numpy as np
import soundfile as sf
from voxcpm import VoxCPM


class Synthesizer:
    def __init__(self, model_id: str) -> None:
        self.model_id = model_id
        self.model = None
        self._device: str | None = None

    def load(self) -> None:
        self.model = VoxCPM.from_pretrained(self.model_id, load_denoiser=False)
        self._device = str(next(self.model.parameters()).device)

    @property
    def is_ready(self) -> bool:
        return self.model is not None

    @property
    def device(self) -> str:
        return self._device or "not loaded"

    @property
    def vram_used_gb(self) -> float:
        if not self.is_ready:
            return 0.0
        import torch
        return round(torch.cuda.memory_allocated() / 1e9, 1)

    def build_text(self, text: str, voice_description: str | None) -> str:
        if voice_description:
            return f"({voice_description}){text}"
        return text

    def generate(
        self,
        text: str,
        voice_description: str | None = None,
        cfg_value: float = 2.0,
        inference_timesteps: int = 10,
    ) -> bytes:
        if self.model is None:
            raise RuntimeError("Synthesizer is not loaded; call load() first")
        full_text = self.build_text(text, voice_description)
        wav: np.ndarray = self.model.generate(
            text=full_text,
            cfg_value=cfg_value,
            inference_timesteps=inference_timesteps,
        )
        buf = io.BytesIO()
        sf.write(buf, wav, self.model.tts_model.sample_rate, format="WAV")
        buf.seek(0)
        return buf.read()
