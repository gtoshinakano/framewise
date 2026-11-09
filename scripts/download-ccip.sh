#!/usr/bin/env bash
# Fetch the deepghs CCIP anime character-similarity models (ONNX, openrail)
# used by bulk character recognition. Files are pinned to one Hugging Face
# revision and checked by SHA-256 before they are moved into place.
set -euo pipefail
root_dir=$(cd "$(dirname "$0")/.." && pwd)
ccip_dir=${CCIP_MODEL_DIR:-"$root_dir/.models/ccip"}
revision=eb2acdd29af1703388d3d0c04221add322bc9110
base="https://huggingface.co/deepghs/ccip_onnx/resolve/$revision"
# variant/file sha256 (LFS files only; small JSON files are pinned by revision)
files=(
  "ccip-caformer_b36-24/model_feat.onnx c1e7333a55c2ad9e03cd340c635e96c9c1d86f0836fa7eae65720e7c9c94ee51"
  "ccip-caformer_b36-24/model_metrics.onnx 816f291e385e854896276ff283dc83eac4966e5f51ab10260fee50de3ca9b8e3"
  "ccip-caformer_b36-24/metrics.json -"
  "ccip-caformer_b36-24/cluster.json -"
  "ccip-caformer-24-randaug-pruned/model_feat.onnx 4ea118d16496274f4f6e08d3afc768cc592389e8f7f32f8732ce2215c228ac5f"
  "ccip-caformer-24-randaug-pruned/model_metrics.onnx 7e4646fdfbe369ad485b227ec47ac4092c98b8deef3af44e19fdbd8ba7bc25c1"
  "ccip-caformer-24-randaug-pruned/metrics.json -"
  "ccip-caformer-24-randaug-pruned/cluster.json -"
)
mkdir -p "$ccip_dir"
exec 9>"$ccip_dir/.download.lock"
flock 9
for entry in "${files[@]}"; do
  read -r path sha <<<"$entry"
  target="$ccip_dir/$path"
  mkdir -p "$(dirname "$target")"
  if [[ -s "$target" ]]; then
    if [[ "$sha" != - ]] && ! printf '%s  %s\n' "$sha" "$target" | sha256sum --check --status; then
      printf 'CCIP checksum mismatch: %s; inspect it before replacing it.\n' "$target" >&2
      exit 1
    fi
    continue
  fi
  curl --fail --location --retry 5 --retry-delay 5 --continue-at - --output "$target.partial" "$base/$path"
  if [[ "$sha" != - ]]; then
    printf '%s  %s\n' "$sha" "$target.partial" | sha256sum --check --status || {
      printf 'CCIP download differs from the pinned version: %s\n' "$path" >&2
      exit 1
    }
  fi
  mv "$target.partial" "$target"
done
printf '%s\n' "$revision" >"$ccip_dir/REVISION"
printf 'CCIP ready at %s (revision %s)\n' "$ccip_dir" "$revision"
