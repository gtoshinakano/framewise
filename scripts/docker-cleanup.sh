#!/usr/bin/env bash
# Keep only the latest Docker build cache and drop images replaced by rebuilds.
# Never touches volumes, so framewise_data and framewise_models are safe.
set -euo pipefail
KEEP_CACHE="${KEEP_CACHE:-8GB}"
docker image prune -f                                 # dangling images left behind by rebuilds
docker builder prune -f --keep-storage "$KEEP_CACHE"  # newest cache stays, older layers go
docker container prune -f                             # stopped containers only
docker system df
