#!/bin/sh
# Throwaway: run backend tests inside the API image with httpx installed.
# Usage: scripts/run-tests-docker.sh <test-module-or-file>...
# Remove after verification.
set -e
cd "$(dirname "$0")/.."
exec docker run --rm --network host \
  -v "$PWD":/repo -w /repo \
  -e PYTHONPATH=/repo/backend \
  -e DATA_DIR=/tmp/data \
  video-analyzer:local \
  sh -c "pip install --quiet --no-cache-dir httpx==0.28.1 && python -m unittest -v $*"
