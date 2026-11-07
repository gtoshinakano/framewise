#!/usr/bin/env bash
# Fetch only the models used by the optional bulk anime detection workflow.
set -euo pipefail
root_dir=$(cd "$(dirname "$0")/.." && pwd)
aniref_dir=${ANIREF_MODEL_DIR:-"$root_dir/.models/aniref"}
reid_dir=${ANIME_REID_MODEL_DIR:-"$root_dir/.models/anime-reid"}
aniref_file="$aniref_dir/AniRef40000-s-epoch40.pt"
aniref_sha=0b811109555bb6e13d0c39791aa23d7f48f841163656ced1d42386ed8350fd6c
reid_graph_sha=a3a421923436f72bc91faaff69542434173ab2415138b38fdff8d1520e5c74cb
reid_data_sha=eb070c3a8b4944429947485c44f667b632aecf1301b095c53d1ca223c4803b2e
mkdir -p "$aniref_dir" "$reid_dir"
exec 9>"$aniref_dir/.download.lock"
flock 9
if [[ ! -f "$aniref_file" ]]; then
  curl --fail --location --retry 5 --retry-delay 5 --continue-at - \
    --output "$aniref_file.partial" \
    'https://github.com/SoulflareRC/AniRef-yolov8/releases/download/model/AniRef40000-s-epoch40.pt'
  printf '%s  %s\n' "$aniref_sha" "$aniref_file.partial" | sha256sum --check --status
  mv "$aniref_file.partial" "$aniref_file"
elif ! printf '%s  %s\n' "$aniref_sha" "$aniref_file" | sha256sum --check --status; then
  printf 'AniRef checksum mismatch: %s\n' "$aniref_file" >&2
  exit 1
fi
printf 'AniRef ready at %s\n' "$aniref_file"
if [[ -s "$reid_dir/dino_v3.onnx" && -s "$reid_dir/dino_v3.onnx.data" ]]; then
  printf '%s  %s\n%s  %s\n' "$reid_graph_sha" "$reid_dir/dino_v3.onnx" "$reid_data_sha" "$reid_dir/dino_v3.onnx.data" | sha256sum --check --status || {
    printf 'Re-ID checksum mismatch; inspect the existing files before replacing them.\n' >&2
    exit 1
  }
  printf 'Using existing Re-ID model at %s\n' "$reid_dir/dino_v3.onnx"
  exit 0
fi
if ! command -v kaggle >/dev/null; then
  printf 'Install the Kaggle CLI and authenticate to obtain the upstream Re-ID notebook output.\n' >&2
  exit 1
fi
# The upstream project distributes its adapted DINOv3 through Kaggle. Download
# into a staging directory so an interrupted transfer is never treated as ready.
staging=$(mktemp -d "$reid_dir/.download.XXXXXX")
trap 'rm -rf "$staging"' EXIT
kaggle kernels output muneeburrehman98/dinov3-finetune-anime -p "$staging"
printf '%s  %s\n%s  %s\n' "$reid_graph_sha" "$staging/dino_v3.onnx" "$reid_data_sha" "$staging/dino_v3.onnx.data" | sha256sum --check --status || {
  printf 'Re-ID output is incomplete or differs from the pinned version.\n' >&2
  exit 1
}
for name in dino_v3.onnx dino_v3.onnx.data; do
  if [[ ! -s "$staging/$name" ]]; then
    printf 'The Kaggle output did not contain %s at its root.\n' "$name" >&2
    exit 1
  fi
  if [[ -f "$reid_dir/$name" ]] && ! cmp -s "$staging/$name" "$reid_dir/$name"; then
    printf 'Existing %s differs from the download; inspect it before replacing it.\n' "$reid_dir/$name" >&2
    exit 1
  fi
done
for name in dino_v3.onnx dino_v3.onnx.data; do
  [[ -f "$reid_dir/$name" ]] || mv "$staging/$name" "$reid_dir/$name"
done
printf 'DINOv3 ready at %s (SHA-256 below)\n' "$reid_dir/dino_v3.onnx"
sha256sum "$reid_dir/dino_v3.onnx" "$reid_dir/dino_v3.onnx.data"
