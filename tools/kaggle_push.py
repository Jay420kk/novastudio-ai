#!/usr/bin/env python3
"""Push a NovaStudio MusicGen job to a Kaggle GPU kernel and download the result.

Requires the `kaggle` CLI (`pip install kaggle`) with ~/.kaggle/kaggle.json
(username + API key). Every job pushes a fresh private script kernel
(enable_gpu, enable_internet) so jobs can run concurrently, then polls the
status and downloads /kaggle/working into --out.

Usage:
  python kaggle_push.py --prompt "lo-fi beat" --seconds 8 --mode mix -o /tmp/dl
"""
import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path

KAGGLE_DIR = Path(__file__).resolve().parent.parent / "kaggle"
STEMS = ["drums.wav", "bass.wav", "other.wav", "vocals.wav"]
MODELS = {
    "small": "facebook/musicgen-small",
    "medium": "facebook/musicgen-medium",
    "stereo-medium": "facebook/musicgen-stereo-medium",
    "large": "facebook/musicgen-large",
}


def _kaggle_cli():
    """Locate the kaggle CLI: on PATH, or in sys.executable's dir (venv bin)."""
    which = shutil.which("kaggle")
    if which:
        return "kaggle"
    venv_bin = Path(sys.executable).parent / "kaggle"
    if venv_bin.is_file():
        return str(venv_bin)
    raise RuntimeError(
        "kaggle CLI not found. Install it into this venv: "
        f"`{sys.executable} -m pip install kaggle`"
    )


def _say(msg):
    print(msg, flush=True)


def _progress(frac, msg):
    # machine-readable progress line the server parses
    print(f"PROGRESS\t{frac:.3f}\t{msg}", flush=True)


def _read_username():
    cfg = Path.home() / ".kaggle" / "kaggle.json"
    if not cfg.is_file():
        raise RuntimeError(
            "~/.kaggle/kaggle.json not found — create it with your Kaggle "
            "username and API key (kaggle.com/settings -> API)"
        )
    try:
        data = json.loads(cfg.read_text())
        return data.get("username")
    except Exception as e:
        raise RuntimeError(f"cannot read ~/.kaggle/kaggle.json: {e}")


def build_kernel_dir(username, slug, prompt, seconds, mode, model, title):
    tmp = Path(tempfile.mkdtemp(prefix="kaggle-kernel-"))
    script_src = KAGGLE_DIR / "generate.py"
    script = script_src.read_text()
    script = script.replace("__MODEL__", model)
    script = script.replace("__PROMPT__", prompt)
    script = script.replace("__SECONDS__", str(int(seconds)))
    script = script.replace("__MODE__", mode)
    (tmp / "generate.py").write_text(script)

    meta = (KAGGLE_DIR / "kernel-metadata.json").read_text()
    meta = meta.replace("__OWNER__", username)
    meta = meta.replace("__SLUG__", slug)
    meta = meta.replace("__TITLE__", title)
    (tmp / "kernel-metadata.json").write_text(meta)
    return tmp


def status_of(username, slug, cli):
    out = subprocess.run(
        [cli, "kernels", "status", f"{username}/{slug}"],
        capture_output=True, text=True, check=False,
    )
    text = out.stdout + out.stderr
    m = re.search(r"\b(complete|running|queued|error|canceled|created)\b", text, re.I)
    if m:
        return m.group(1).lower()
    return "unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--seconds", type=int, default=8)
    ap.add_argument("--mode", choices=["mix", "beat"], default="mix")
    ap.add_argument("--model", choices=["small", "medium", "stereo-medium", "large"], default="medium")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--title", default="NovaStudio generate")
    args = ap.parse_args()

    username = _read_username()
    slug = f"novastudio-gen-{uuid.uuid4().hex[:10]}"
    model = MODELS[args.model]
    title = f"{args.title} {slug}"

    _say(f"username={username} slug={slug} model={model}")
    _progress(0.02, "building Kaggle kernel...")
    kdir = build_kernel_dir(username, slug, args.prompt, args.seconds, args.mode, model, title)
    cli = _kaggle_cli()
    try:
        _progress(0.05, "pushing kernel (private GPU script)...")
        subprocess.run(
            [cli, "kernels", "push", "-p", str(kdir)],
            check=True, capture_output=True, text=True,
        )

        _progress(0.1, "queued — waiting for GPU slot...")
        last = None
        deadline = time.time() + 60 * 60  # 1h hard cap (Kaggle runs max ~12h)
        while time.time() < deadline:
            st = status_of(username, slug, cli)
            if st != last:
                _say(f"kernel status: {st}")
                last = st
            if st == "complete":
                break
            if st in ("error", "canceled"):
                raise RuntimeError(f"Kaggle kernel failed with status {st}")
            time.sleep(10)

        if last != "complete":
            raise RuntimeError("Kaggle kernel did not finish within 1 hour")

        _progress(0.9, "downloading output...")
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [cli, "kernels", "output", f"{username}/{slug}", "-p", str(out)],
            check=True, capture_output=True, text=True,
        )
    finally:
        shutil.rmtree(kdir, ignore_errors=True)

    names = ["generated.wav"] + (STEMS if args.mode == "beat" else [])
    found = []
    for name in names:
        if (out / name).is_file():
            found.append(name)
    _say(f"files: {found}")
    _progress(1.0, "done")


if __name__ == "__main__":
    main()
