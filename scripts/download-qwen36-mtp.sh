#!/usr/bin/env bash
# Qwen3.6-35B-A3B with multi-token-prediction heads (for llama.cpp speculative decoding).
# Kept beside, not over, the plain Q4_K_M in .models/qwen3.6-35b.
set -euo pipefail
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
model_dir="${TEXT_MTP_MODEL_DIR:-$project_dir/.models/qwen3.6-35b-mtp}"
model_name='Qwen3.6-35B-A3B-UD-Q4_K_M.gguf'
expected_sha='0b21525e972670ed59e1812e170b27c26355381f0656ecc4e25617ece7dac58b'
revision='5bc3e238d916f48a861bac2f8a1990a0e9b7e98d'
mkdir -p "$model_dir"
if [[ -f "$model_dir/$model_name" ]]; then
  printf '%s  %s\n' "$expected_sha" "$model_dir/$model_name" | sha256sum -c -
  exit 0
fi
curl --fail --location --retry 5 --retry-delay 5 --continue-at - \
  --output "$model_dir/$model_name.partial" \
  "https://huggingface.co/unsloth/Qwen3.6-35B-A3B-MTP-GGUF/resolve/$revision/$model_name?download=true"
printf '%s  %s\n' "$expected_sha" "$model_dir/$model_name.partial" | sha256sum -c -
mv "$model_dir/$model_name.partial" "$model_dir/$model_name"
