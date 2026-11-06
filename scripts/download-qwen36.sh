#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
model_dir="${TEXT_MODEL_DIR:-$project_dir/.models/qwen3.6-35b}"
model_name='Qwen3.6-35B-A3B-Q4_K_M.gguf'
expected_sha='671e47e0ec53c665d048b98c3ecbfd5236b5ca9c3e02ed19fc8f81f7b85140c7'
revision='7878dc9a905c7c9379e5e8d6a1aa9affdd6839c5'
mkdir -p "$model_dir"
if [[ -f "$model_dir/$model_name" ]]; then
  printf '%s  %s\n' "$expected_sha" "$model_dir/$model_name" | sha256sum -c -
  exit 0
fi
curl --fail --location --retry 5 --retry-delay 5 --continue-at - \
  --output "$model_dir/$model_name.partial" \
  "https://huggingface.co/ggml-org/Qwen3.6-35B-A3B-GGUF/resolve/$revision/$model_name?download=true"
printf '%s  %s\n' "$expected_sha" "$model_dir/$model_name.partial" | sha256sum -c -
mv "$model_dir/$model_name.partial" "$model_dir/$model_name"
