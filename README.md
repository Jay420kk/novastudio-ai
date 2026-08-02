# NovaStudio AI worker

Free 16GB GitHub Actions runner that generates music with Meta's MusicGen
(open source, MIT) and returns the WAV as a workflow artifact.

- **Workflow**: `.github/workflows/generate.yml` (triggered via `workflow_dispatch`)
- **Script**: `tools/generate_cli.py`
- **Model**: `facebook/musicgen-small` (cached in the repo's free Actions cache)

## Modes

- `mix` (default): one full-mix WAV (`generated.wav`)
- `beat`: full mix **plus** stems split with Demucs (`drums.wav`, `bass.wav`, `other.wav`, `vocals.wav`) — like BandLab SongStarter

## Trigger manually

```bash
# single mix
gh workflow run generate.yml -f prompt="lofi hip hop drums, 90 bpm" -f seconds=8
# full beat with stems
gh workflow run generate.yml -f prompt="trap beat, dark, 140 bpm" -f seconds=8 -f mode=beat
gh run download <run-id> -n generated
```
