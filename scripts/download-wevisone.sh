#!/usr/bin/env bash
# Download the pinned release once, resuming interrupted files and checking the large LFS blobs.
set -euo pipefail
revision=6b44a45873b4bbc7dca383922504430dc60eca23
model_dir=${WEVISONE_MODEL_DIR:-"$(cd "$(dirname "$0")/.." && pwd)/.models/wevisone-4b"}
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
  curl --fail --location --retry 5 --retry-delay 5 --continue-at - --output "$target.partial" "https://huggingface.co/tencent/WeVisOne-VL-4B/resolve/$revision/$name"
  if [[ -n "$checksum" ]] && ! printf '%s  %s\n' "$checksum" "$target.partial" | sha256sum -c --status; then
    printf 'Checksum mismatch for %s; retained partial file for inspection.\n' "$name" >&2
    return 1
  fi
  mv "$target.partial" "$target"
}
fetch model.safetensors 002c3ac0783af7c86dc28b282c84dd4bcd0c29dcb3b5e0d584492fe3212acb4a
fetch mask_decoder.pt 8528460481bd09d0780c53559b08194c7abbd29dff0d51783b619158acbe7b46
fetch tokenizer.json 3bb172429b952469b51c64783f52fa01c26d1b30f1f31a4693e385a43f9cfc5e
for name in LICENSE README.md chat_template.jinja config.json generation_config.json processor_config.json tokenizer_config.json; do fetch "$name"; done
printf 'WeVisOne-VL 4B ready at %s\n' "$model_dir"
