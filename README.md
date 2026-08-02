# NovaStudio AI worker

Free 16GB GitHub Actions runner that generates music with Meta's MusicGen
(open source, MIT) and returns the WAV as a workflow artifact.

- **Workflow**: `.github/workflows/generate.yml` (triggered via `workflow_dispatch`)
- **Script**: `tools/generate_cli.py`
- **Model**: `facebook/musicgen-small` (cached in the repo's free Actions cache)

## Trigger manually

```bash
gh workflow run generate.yml -f prompt="lofi hip hop drums, 90 bpm" -f seconds=8
gh run download <run-id> -n generated
```
