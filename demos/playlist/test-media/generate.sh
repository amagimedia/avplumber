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

# Machine-readable frame code (see tests/frame_codes.py): a 512x24 strip at
# (64,0) with 32 cells of 16 px; top row = bits of
#   0xA<<28 | clip<<24 | frame<<8 | checksum,  bottom row inverted.
frame_code_filter() {
    clip=$1
    code="(10*pow(2,28)+${clip}*pow(2,24)+N*256+mod(${clip}*37+floor(N/256)+mod(N,256),256))"
    bit="mod(floor(${code}/pow(2,31-floor(X/16))),2)"
    echo "split[bg][strip];[strip]crop=512:24:64:0,geq=lum='if(lt(Y,12),if(${bit},240,16),if(${bit},16,240))':cb=128:cr=128[code];[bg][code]overlay=64:0"
}

generate_clip() {
    pattern=$1
    clip_id=$2
    output=$3
    clip_index=$4
    temporary="${output}.tmp"

    ffmpeg -hide_banner -loglevel error -y \
        -f lavfi -i "${pattern}=size=1920x1080:rate=30" \
        -vf "format=yuv420p,drawtext=font='DejaVu Sans Mono':text='${clip_id}':fontcolor=white:fontsize=64:borderw=4:bordercolor=black:x=60:y=60,drawtext=font='DejaVu Sans Mono':text='FRAME %{n}   PTS %{pts\\:hms}':fontcolor=white:fontsize=52:borderw=4:bordercolor=black:x=60:y=h-th-60,$(frame_code_filter "$clip_index")" \
        -frames:v 300 -an \
        -c:v libx264 -preset veryfast -crf 20 -pix_fmt yuv420p \
        -g 30 -keyint_min 30 -bf 0 -sc_threshold 0 -movflags +faststart \
        -metadata title="${clip_id}" \
        -f mp4 "$temporary"
    mv -- "$temporary" "$output"
}

generate_clip testsrc2 "CLIP 01 TESTSRC2" "$script_dir/01-testsrc2.mp4" 1
generate_clip smptebars "CLIP 02 SMPTE" "$script_dir/02-smpte.mp4" 2
generate_clip smptehdbars "CLIP 03 SMPTE HD" "$script_dir/03-smpte-hd.mp4" 3
generate_clip rgbtestsrc "CLIP 04 RGB" "$script_dir/04-rgb.mp4" 4
generate_clip yuvtestsrc "CLIP 05 YUV" "$script_dir/05-yuv.mp4" 5

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
