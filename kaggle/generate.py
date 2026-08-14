import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

MODEL = "__MODEL__"
PROMPT = "__PROMPT__"
SECONDS = __SECONDS__
MODE = "__MODE__"
STEMS = ["drums", "bass", "other", "vocals"]


def ensure_deps():
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--quiet", "transformers", "soundfile", "demucs"],
        check=True,
    )


def main():
    import numpy as np
    import soundfile as sf
    import torch
    from transformers import AutoProcessor, MusicgenForConditionalGeneration

    work = Path("/kaggle/working")
    work.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}", flush=True)

    processor = AutoProcessor.from_pretrained(MODEL)
    dtype = torch.float16 if device == "cuda" else torch.float32
    model = (
        MusicgenForConditionalGeneration.from_pretrained(MODEL, torch_dtype=dtype, low_cpu_mem_usage=True)
        .to(device)
        .eval()
    )

    enc = getattr(model.config, "audio_encoder", None)
    frame_rate = getattr(enc, "frame_rate", 50)
    sr = getattr(enc, "sampling_rate", 32000)
    max_new = max(64, int(max(3.0, float(SECONDS)) * frame_rate))

    print(f"generating {SECONDS}s ...", flush=True)
    with torch.no_grad():
        inputs = processor(text=[PROMPT], padding=True, return_tensors="pt")
        if device == "cuda":
            inputs = {k: v.to(device) for k, v in inputs.items()}
        audio = model.generate(**inputs, max_new_tokens=max_new)

    arr = audio[0].float().cpu().numpy()
    if arr.ndim == 1:
        arr = np.stack([arr, arr], axis=1)
    elif arr.ndim == 2 and arr.shape[0] == 1:
        arr = np.stack([arr[0], arr[0]], axis=1)
    else:
        arr = arr.T
    sf.write(work / "generated.wav", arr, sr, subtype="PCM_24")
    print("wrote generated.wav", flush=True)

    if MODE == "beat":
        print("splitting stems (demucs, gpu)...", flush=True)
        with tempfile.TemporaryDirectory() as tmp:
            proc = subprocess.run(
                [sys.executable, "-m", "demucs", "-n", "htdemucs", "-d", device,
                 "--out", tmp, "--float32", str(work / "generated.wav")],
                capture_output=True, text=True,
            )
            if proc.returncode != 0:
                raise RuntimeError(f"demucs failed: {proc.stderr[-2000:]}")
            found = 0
            for p in sorted(Path(tmp).rglob("*.wav")):
                if p.stem in STEMS:
                    shutil.move(str(p), str(work / f"{p.stem}.wav"))
                    print(f"stem {p.stem}.wav ready", flush=True)
                    found += 1
            if found == 0:
                raise RuntimeError("demucs produced no recognizable stems")
    print("DONE", flush=True)


if __name__ == "__main__":
    ensure_deps()
    main()
