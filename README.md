# NovaStudio AI worker

Generates music with Meta's MusicGen (open source, MIT) and returns the WAV.
Three remote engines, all free (see [Free compute summary](#free-compute-summary)):

| Engine | Hardware | Model | Notes |
|--------|----------|-------|-------|
| **Modal** (best) | serverless T4 | `musicgen-medium`/`stereo-medium` | real HTTP endpoint, $30/mo free credits (~50 T4-hrs), warm-loaded, scale-to-zero |
| **Kaggle** | GPU (T4/P100, up to 2×T4) | `musicgen-medium` | ~30 GPU-hr/week free; private script kernel per job |
| **GitHub Actions** | 4-core/16GB CPU | `small`/`medium` | unlimited minutes on a public repo; 10GB model cache |

## Directory layout

- `.github/workflows/generate.yml` — Actions workflow (dispatch via `workflow_dispatch`)
- `tools/generate_cli.py` — CPU runner script (`--model` picks the checkpoint)
- `tools/kaggle_push.py` — Kaggle lane: push kernel, poll, download output
- `kaggle/generate.py` + `kaggle/kernel-metadata.json` — Kaggle notebook template (GPU, internet on)
- `modal/generate.py` — Modal app: deploy once, call as a plain HTTP endpoint

## Modes

- `mix` (default): one full-mix WAV (`generated.wav`)
- `beat`: full mix **plus** stems split with Demucs (`drums.wav`, `bass.wav`, `other.wav`, `vocals.wav`) — like BandLab SongStarter

## Modal setup (one time)

```bash
pip install modal
modal setup                     # browser login; free Starter = $30/mo credits
cd ai-worker/modal
modal deploy generate.py        # prints the endpoint URL
# then start the DAW server with:
#   MODAL_ENDPOINT=https://<you>--novastudio-generate.modal.run
# or use AI_ENGINE=modal to force it
```

## Kaggle setup (one time)

```bash
pip install kaggle
mkdir -p ~/.kaggle
# paste your username + API key from https://www.kaggle.com/settings -> API
# into ~/.kaggle/kaggle.json, then:
chmod 600 ~/.kaggle/kaggle.json
```

## Trigger manually

```bash
# Modal (needs deploy above)
curl -X POST "$MODAL_ENDPOINT/generate" -H 'Content-Type: application/json' \
  -d '{"prompt":"lofi hip hop drums, 90 bpm","seconds":8,"mode":"mix"}' -o /tmp/out.zip

# Kaggle GPU lane (needs kaggle.json above)
python tools/kaggle_push.py --prompt "lofi hip hop drums, 90 bpm" --seconds 8 -o /tmp/dl

# GitHub Actions CPU lane
gh workflow run generate.yml -f prompt="trap beat, dark, 140 bpm" -f seconds=8 -f mode=beat
gh run download <run-id> -n generated
```

The DAW server (`daw/server/ai/generate.py`) picks Modal first when
`MODAL_ENDPOINT` is set, then Kaggle, then GitHub Actions.
Override with `AI_ENGINE=modal|kaggle|github`.
Local CPU generation is disabled — if no remote lane is configured the job
fails loudly instead of running on slow local hardware.

## Free compute summary (research, 2026-08)

Verified free GPU/storage options — use what you need, skip the rest:

- **Modal** — $30/mo recurring credits, no card, ~50 T4-hrs/mo. Serverless HTTP
  endpoints; the DAW calls it like any API. Best developer ergonomics.
- **Kaggle** — 30 GPU-hr/wk guaranteed, P100 or 2×T4, 12h sessions. Reliable quota.
- **Google Colab** — 15–30 GPU-hr/wk T4 (throttled). Official CLI (`colab run --gpu T4`).
- **Lightning AI** — 80 GPU-hr/mo (phone verification), persistent VS Code workspace.
- **GitHub Actions** — unlimited on public repos (4-core/16GB CPU); 2nd account
  doubles concurrency (`GH_REPO=owner/a,owner/b` round-robins).
- **Cloudflare R2** — 10GB free object storage, $0 egress, S3-compatible.
- **TTS** — edge-tts / Kokoro / Piper: unlimited free neural voices, local/self-hosted.

