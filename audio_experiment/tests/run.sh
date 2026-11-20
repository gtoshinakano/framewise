#!/usr/bin/env bash
# Unit tests of the audio experiment, CPU only, in a throwaway container of the audio image (numpy, soundfile, torch there).
cd "$(dirname "$0")/../.."
exec docker run --rm --entrypoint python -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONPATH=/work -e MPLCONFIGDIR=/tmp/mpl \
  -v "$PWD/audio_experiment:/work:ro" -v "$PWD/backend/app:/app/app:ro" -w /work framewise-audio-rocm:local \
  -m unittest discover -s tests -t . "$@"
