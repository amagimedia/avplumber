#!/bin/sh
# Exercise the pinned raw-source path and the codec/compositor path in the mixer image.
set -eu
here=$(dirname "$0")
scratch=$(mktemp -d)
trap 'rm -rf "$scratch"' EXIT
python3 "$here/smoke_raw_direct_input.py"
ffmpeg -hide_banner -loglevel error -f lavfi -i testsrc2=size=320x192:rate=25 \
    -frames:v 50 -c:v hevc_nvenc -g 1 -bf 0 -preset p1 "$scratch/clip.mp4"
python3 "$here/nvdec/compositor_decode.py" --input "$scratch/clip.mp4" \
    --width 320 --height 192 --frames 50 --rounds 1
