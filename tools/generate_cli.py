import argparse

import numpy as np
import torch
from transformers import AutoProcessor, MusicgenForConditionalGeneration

MODEL = "facebook/musicgen-small"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prompt")
    ap.add_argument("seconds", type=float, default=8.0)
    ap.add_argument("-o", "--out", default="generated.wav")
    args = ap.parse_args()

    import soundfile as sf

    print(f"loading {MODEL} ...", flush=True)
    proc = AutoProcessor.from_pretrained(MODEL)
    model = MusicgenForConditionalGeneration.from_pretrained(MODEL)
    model.eval()

    enc = getattr(model.config, "audio_encoder", None)
    frame_rate = getattr(enc, "frame_rate", 50)
    sr = getattr(enc, "sampling_rate", 32000)
    max_new = max(64, int(max(3.0, args.seconds) * frame_rate))

    print(f"generating {args.seconds}s ...", flush=True)
    with torch.no_grad():
        audio = model.generate(
            **proc(text=[args.prompt], padding=True, return_tensors="pt"),
            max_new_tokens=max_new,
        )
    arr = audio[0].cpu().numpy()
    if arr.ndim == 1:
        arr = np.stack([arr, arr], axis=1)
    elif arr.ndim == 2 and arr.shape[0] == 1:
        arr = np.stack([arr[0], arr[0]], axis=1)
    else:
        arr = arr.T

    sf.write(args.out, arr, sr, subtype="PCM_24")
    print(f"wrote {args.out} ({args.seconds}s @ {sr}Hz)", flush=True)


if __name__ == "__main__":
    main()
