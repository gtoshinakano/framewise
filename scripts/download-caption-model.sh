#!/usr/bin/env bash
# Download the pinned local frame-caption model once; resume interrupted downloads.
set -euo pipefail
revision=89644892e4d85e24eaac8bacfd4f463576704203
model_dir=${CAPTION_MODEL_DIR:-"$(cd "$(dirname "$0")/.." && pwd)/.models/qwen3-vl-2b"}
mkdir -p "$model_dir"
exec 9>"$model_dir/.download.lock"
flock 9
fetch() {
  local name=$1 checksum=${2:-} target="$model_dir/$1"
  if [[ -f "$target" ]]; then
    if [[ -z "$checksum" ]] || printf '%s  %s\n' "$checksum" "$target" | sha256sum -c --status; then
      printf 'Using cached %s\n' "$name"
      return
    fi
    printf 'Checksum mismatch for %s; remove it after investigation and rerun.\n' "$target" >&2
    return 1
  fi
  curl --fail --location --retry 5 --retry-delay 5 --continue-at - --output "$target.partial" "https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct/resolve/$revision/$name"
  if [[ -n "$checksum" ]] && ! printf '%s  %s\n' "$checksum" "$target.partial" | sha256sum -c --status; then
    printf 'Checksum mismatch for %s; retained partial file for inspection.\n' "$name" >&2
    return 1
  fi
  mv "$target.partial" "$target"
}
fetch model.safetensors 7de1838c87a5349b016c26a1c3f7d2bc400a3d485f95ef39a7059ffd734977a0
for name in README.md chat_template.json config.json generation_config.json merges.txt preprocessor_config.json tokenizer.json tokenizer_config.json video_preprocessor_config.json vocab.json; do fetch "$name"; done
printf 'Qwen3-VL caption model ready at %s\n' "$model_dir"
