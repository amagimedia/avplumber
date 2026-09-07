#!/usr/bin/env bash
# Compose the demo recording: program video (left) beside the captured TUI
# (right) on a 1600x900 canvas at 30 fps, plus a poster JPEG.
#
#   compose_recording.sh <program.mp4> <capture-dir> <out.mp4> [offset_seconds]
#
# <capture-dir> holds tui/NNNNN.jpg from capture_tui.cjs.  offset_seconds trims
# the program so both start together (program started earlier than the capture:
# positive offset).  The program is never resampled in time: -r 30 matches its
# native rate, so every program frame appears exactly once in the composite.
set -euo pipefail
program=$1; capture=$2; out=$3; offset=${4:-0}
ffmpeg -hide_banner -loglevel error -y \
    -ss "$offset" -i "$program" \
    -framerate 30 -i "$capture/tui/%05d.jpg" \
    -filter_complex "[0:v]scale=1000:-2:flags=lanczos,pad=1000:900:0:(oh-ih)/2:color=#080d17[left];[1:v]scale=600:900:flags=lanczos[right];[left][right]hstack=inputs=2,format=yuv420p" \
    -r 30 -c:v libx264 -preset slow -crf 20 -movflags +faststart -an -shortest "$out"
ffmpeg -hide_banner -loglevel error -y -ss 1 -i "$out" -frames:v 1 -q:v 3 "${out%.mp4}.jpg"
ffprobe -v error -select_streams v:0 -show_entries stream=width,height,r_frame_rate,nb_frames,duration -of csv=p=0 "$out"
