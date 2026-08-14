import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from transformers import AutoProcessor, MusicgenForConditionalGeneration

MODELS = {
    "small": "facebook/musicgen-small",
    "medium": "facebook/musicgen-medium",
    "stereo-medium": "facebook/musicgen-stereo-medium",
    "large": "facebook/musicgen-large",
}
STEMS = ["drums", "bass", "other", "vocals"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prompt")
    ap.add_argument("seconds", type=float, default=8.0)
    ap.add_argument("-o", "--out", default="generated.wav")
    ap.add_argument(
        "--mode",
        choices=["mix", "beat"],
        default="mix",
        help="mix = one WAV; beat = full mix + demucs stems (drums/bass/other/vocals)",
    )
    ap.add_argument(
        "--model",
        choices=list(MODELS),
        default="small",
        help="musicgen checkpoint (default: small — pick medium for quality on 16GB runners)",
    )
    args = ap.parse_args()

    import soundfile as sf

    model_name = MODELS[args.model]

    outdir = Path(args.out).parent if str(Path(args.out).parent) else Path(".")
    outdir.mkdir(parents=True, exist_ok=True)
    mix_path = outdir / "generated.wav"

    print(f"loading {model_name} ...", flush=True)
    proc = AutoProcessor.from_pretrained(model_name)
    model = MusicgenForConditionalGeneration.from_pretrained(model_name)
    model.eval()

    enc = getattr(model.config, "audio_encoder", None)
    frame_rate = getattr(enc, "frame_rate", 50)
    sr = getattr(enc, "sampling_rate", 32000)
    max_new = max(64, int(max(3.0, args.seconds) * frame_rate))

    print(f"generating {args.seconds}s full mix ...", flush=True)
    with torch.no_grad():
        audio = model.generate(
            **proc(text=[args.prompt], padding=True, return_tensors="pt"),
            max_new_tokens=max_new,
        )
    arr = audio[0].cpu().float().numpy()
    if arr.ndim == 1:
        arr = np.stack([arr, arr], axis=1)
    elif arr.ndim == 2 and arr.shape[0] == 1:
        arr = np.stack([arr[0], arr[0]], axis=1)
    else:
        arr = arr.T

    sf.write(mix_path, arr, sr, subtype="PCM_24")
    print(f"wrote {mix_path} ({args.seconds}s @ {sr}Hz)", flush=True)

    if args.mode == "beat":
        print("splitting stems with demucs htdemucs ...", flush=True)
        with tempfile.TemporaryDirectory() as tmp:
            cmd = [
                sys.executable, "-m", "demucs",
                "-n", "htdemucs",
                "-d", "cpu",
                "--out", tmp,
                "--float32",
                str(mix_path),
            ]
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
            if proc.returncode != 0:
                raise RuntimeError(f"demucs failed: {proc.stderr[-2000:]}")
            import shutil

            found = 0
            for p in sorted(Path(tmp).rglob("*.wav")):
                stem_name = p.stem  # demucs writes <stem>.wav files
                if stem_name in STEMS:
                    shutil.move(str(p), str(outdir / f"{stem_name}.wav"))
                    print(f"stem {stem_name}.wav ready", flush=True)
                    found += 1
            if found == 0:
                raise RuntimeError("demucs produced no recognizable stems")
    print("done", flush=True)


if __name__ == "__main__":
    main()
