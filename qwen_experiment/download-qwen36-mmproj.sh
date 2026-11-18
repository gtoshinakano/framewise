#!/usr/bin/env bash
# Download ONLY the vision projector for the Qwen 3.6 GGUF the app already has.
#
# scripts/download-qwen36.sh fetches the text-only Q4_K_M GGUF; it ships no mmproj, so
# Qwen 3.6 cannot see images. This companion script drops the matching BF16 projector
# next to the existing text model so llama-server can run Qwen 3.6 as a vision model.
# Nothing else is downloaded and no app file is changed.
#
# The source repo is gated, so it needs HF_TOKEN (the value the app already uses).
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
model_dir="${MODEL_DIR:-$project_dir/.models/qwen3.6-35b}"
file='mmproj-Qwen3.6-35B-A3B-BF16.gguf'
expected_sha='360c746ddf38fb67c06f83a0a88742492aa231e59e335cd980e6302f59ea6089'
url='https://huggingface.co/glm-ai/Qwen3.6-35B-A3B-GGUF-Q4_K_M/resolve/main/mmproj-Qwen3.6-35B-A3B-BF16.gguf?download=true'

mkdir -p "$model_dir"
if [[ -f "$model_dir/$file" ]]; then
  printf '%s  %s\n' "$expected_sha" "$model_dir/$file" | sha256sum -c - && exit 0
fi

# The repo is gated: send the token, resume, verify, then rename.
if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "HF_TOKEN is not set (the source repo requires authentication)." >&2
  exit 1
fi
curl --fail --location --retry 5 --retry-delay 5 --continue-at - \
  --header "Authorization: Bearer $HF_TOKEN" \
  --output "$model_dir/$file.partial" "$url"
printf '%s  %s\n' "$expected_sha" "$model_dir/$file.partial" | sha256sum -c -
mv "$model_dir/$file.partial" "$model_dir/$file"
echo "Downloaded $model_dir/$file"
