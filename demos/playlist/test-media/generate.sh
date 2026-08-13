#!/usr/bin/env bash
set -euo pipefail

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)

if ! command -v ffmpeg >/dev/null 2>&1; then
    echo "ffmpeg is required" >&2
    exit 1
fi
if ! command -v ffprobe >/dev/null 2>&1; then
    echo "ffprobe is required" >&2
    exit 1
fi

generate_clip() {
    pattern=$1
    clip_id=$2
    output=$3
    temporary="${output}.tmp"

    ffmpeg -hide_banner -loglevel error -y \
        -f lavfi -i "${pattern}=size=1920x1080:rate=30" \
        -vf "drawtext=font='DejaVu Sans Mono':text='${clip_id}':fontcolor=white:fontsize=64:borderw=4:bordercolor=black:x=60:y=60,drawtext=font='DejaVu Sans Mono':text='FRAME %{n}   PTS %{pts\\:hms}':fontcolor=white:fontsize=52:borderw=4:bordercolor=black:x=60:y=h-th-60" \
        -frames:v 300 -an \
        -c:v libx264 -preset veryfast -crf 20 -pix_fmt yuv420p \
        -g 30 -keyint_min 30 -sc_threshold 0 -movflags +faststart \
        -metadata title="${clip_id}" \
        -f mp4 "$temporary"
    mv -- "$temporary" "$output"
}

generate_clip testsrc2 "CLIP 01 TESTSRC2" "$script_dir/01-testsrc2.mp4"
generate_clip smptebars "CLIP 02 SMPTE" "$script_dir/02-smpte.mp4"
generate_clip smptehdbars "CLIP 03 SMPTE HD" "$script_dir/03-smpte-hd.mp4"
generate_clip rgbtestsrc "CLIP 04 RGB" "$script_dir/04-rgb.mp4"
generate_clip yuvtestsrc "CLIP 05 YUV" "$script_dir/05-yuv.mp4"

for clip in "$script_dir"/*.mp4; do
    properties=$(ffprobe -v error -select_streams v:0 \
        -show_entries stream=codec_name,width,height,r_frame_rate,nb_frames,duration \
        -of csv=p=0 "$clip")
    case "$properties" in
        h264,1920,1080,30/1,10.000000,300|h264,1920,1080,30/1,300,10.000000)
            ;;
        *)
            echo "unexpected fixture properties for $clip: $properties" >&2
            exit 1
            ;;
    esac
done
