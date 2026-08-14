"""NovaStudio MusicGen on Modal — a real HTTP endpoint, no push/poll/download.

Deploy once (needs a Modal account; $30/month free credits = ~50 T4-hours):
    pip install modal
    modal deploy generate.py          # run from daw/ai-worker/modal/

The endpoint is warm-loaded (model resident in a container) and scales to zero,
so calls cost only the seconds of GPU actually used. Result is streamed back
as a ZIP of WAVs (generated.wav + optional stems).

Call it:
    POST https://<user>--novastudio-generate.modal.run/generate
    {"prompt": "lofi beat", "seconds": 8, "mode": "mix"}

Server wiring (daw/server/ai/generate.py): set MODAL_ENDPOINT to the deploy URL
and AI_ENGINE=modal, or set MODAL_ENDPOINT and it becomes the preferred lane.
"""
import io
import subprocess
import tempfile
import zipfile
from pathlib import Path

import modal

GPU = "T4"
MODEL = "facebook/musicgen-medium"
STEMS = ["drums", "bass", "other", "vocals"]

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", "transformers", "soundfile", "numpy", "demucs")
)
volume = modal.Volume.from_name("novastudio-models", create_if_missing=True)
app = modal.App("novastudio-generate")


@app.cls(
    image=image,
    gpu=GPU,
    volumes={"/models": volume},
    timeout=1800,
    container_idle_timeout=300,
    allow_concurrent_inputs=4,
)
class Generator:
    @modal.enter()
    def load(self):
        import torch
        from transformers import AutoProcessor, MusicgenForConditionalGeneration

        self.torch = torch
        self.processor = AutoProcessor.from_pretrained(MODEL, cache_dir="/models")
        self.model = (
            MusicgenForConditionalGeneration.from_pretrained(
                MODEL, cache_dir="/models", torch_dtype=torch.float16
            )
            .to("cuda")
            .eval()
        )

    @modal.web_endpoint(method="POST", label="generate")
    def generate(self, req: dict):
        import numpy as np
        import soundfile as sf

        prompt = str(req.get("prompt") or "").strip()
        if not prompt:
            return _err("no prompt")
        seconds = min(max(float(req.get("seconds", 8)), 3), 30)
        mode = req.get("mode", "mix") if req.get("mode", "mix") in ("mix", "beat") else "mix"

        model = str(req.get("model") or MODEL).strip()
        if model != MODEL:
            import torch
            from transformers import AutoProcessor, MusicgenForConditionalGeneration

            self.processor = AutoProcessor.from_pretrained(model, cache_dir="/models")
            self.model = (
                MusicgenForConditionalGeneration.from_pretrained(model, cache_dir="/models", torch_dtype=torch.float16)
                .to("cuda")
                .eval()
            )

        enc = getattr(self.model.config, "audio_encoder", None)
        frame_rate = getattr(enc, "frame_rate", 50)
        sr = getattr(enc, "sampling_rate", 32000)
        max_new = max(64, int(seconds * frame_rate))

        with self.torch.no_grad():
            inputs = self.processor(text=[prompt], padding=True, return_tensors="pt").to("cuda")
            audio = self.model.generate(**inputs, max_new_tokens=max_new)

        arr = audio[0].float().cpu().numpy()
        if arr.ndim == 1:
            arr = np.stack([arr, arr], axis=1)
        elif arr.ndim == 2 and arr.shape[0] == 1:
            arr = np.stack([arr[0], arr[0]], axis=1)
        else:
            arr = arr.T

        buf = io.BytesIO()
        sf.write(buf, arr, sr, subtype="PCM_24", format="WAV")
        files = {"generated.wav": buf.getvalue()}

        if mode == "beat":
            with tempfile.TemporaryDirectory() as tmp:
                mix_path = f"{tmp}/generated.wav"
                sf.write(mix_path, arr, sr, subtype="PCM_24")
                proc = subprocess.run(
                    ["python", "-m", "demucs", "-n", "htdemucs", "-d", "cuda",
                     "--out", tmp, "--float32", mix_path],
                    capture_output=True, text=True,
                )
                if proc.returncode == 0:
                    for p in Path(tmp).rglob("*.wav"):
                        if p.stem in STEMS:
                            files[f"{p.stem}.wav"] = p.read_bytes()

        zipbuf = io.BytesIO()
        with zipfile.ZipFile(zipbuf, "w", zipfile.ZIP_DEFLATED) as z:
            for name, data in files.items():
                z.writestr(name, data)
        return modal.WebResponse(status_code=200, content_type="application/zip", content=zipbuf.getvalue())


def _err(msg):
    return modal.WebResponse(status_code=400, content_type="application/json",
                             content=msg.encode())
