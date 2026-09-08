#!/usr/bin/env bash
# Generates a test video for examples/freezedetect_stats.avplumber.
#
# Pattern: interleaved MOVING (testsrc) and FROZEN (held frame) segments:
#   M  F  M  F  M  F  M   (N_EVENTS frozen segments between N_EVENTS+1 moving)
#
# Frozen segments are > 2s so they trigger freezedetect=noise=0.001:duration=2,
# and are encoded ALL-INTRA so every held frame decodes bit-identically (SD=0,
# well under the strict noise=0.001 threshold). Moving segments use testsrc,
# whose counter/pattern advance every frame, so freezedetect does not fire there.
#
# A 1kHz sine tone is muxed in so the example's audio path (dec_audio -> null_sink)
# is exercised too.
#
# Usage:
#   ./make_freeze_test_video.sh [out.mp4] [W] [H] [FPS] [move_s] [freeze_s] [n_events]
# Defaults: freezedetect_test.mp4 640 360 25 4 3 3
set -euo pipefail

OUT="${1:-freezedetect_test.mp4}"
W="${2:-640}"; H="${3:-360}"; FPS="${4:-25}"
MOVE_S="${5:-4}"     # moving segment duration (s)
FREEZE_S="${6:-3}"   # frozen segment duration (s); must be > 2 for duration=2
N_EVENTS="${7:-3}"   # number of freeze events

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
FREEZE_PNG="$TMP/freeze.png"

# 1. One static frame from testsrc (held frame for every frozen segment).
ffmpeg -y -hide_banner -loglevel error -f lavfi \
  -i "testsrc=size=${W}x${H}:rate=${FPS}" -frames:v 1 -update 1 "$FREEZE_PNG"

# 2. Build interleaved inputs: M F M F ... M  (2*N_EVENTS+1 streams).
args=()
filters=""
labels=""
i=0
for e in $(seq 0 "$N_EVENTS"); do
  args+=(-f lavfi -i "testsrc=size=${W}x${H}:rate=${FPS}:duration=${MOVE_S}")
  filters+="[$i]format=yuv420p,fps=${FPS},setpts=PTS-STARTPTS[m$i];"
  labels+="[m$i]"; i=$((i+1))
  if [ "$e" -lt "$N_EVENTS" ]; then
    args+=(-loop 1 -framerate "$FPS" -t "$FREEZE_S" -i "$FREEZE_PNG")
    filters+="[$i]format=yuv420p,fps=${FPS},setpts=PTS-STARTPTS[f$i];"
    labels+="[f$i]"; i=$((i+1))
  fi
done
n=$((2*N_EVENTS+1))
filters+="${labels}concat=n=${n}:v=1:a=0[vout]"
SINE_IDX=$n   # sine is the last input, 0-based index = number of video inputs

# 3. All-intra encode (frozen frames decode bit-identically) + sine audio.
TOTAL_S=$(awk "BEGIN{printf \"%g\", ($MOVE_S*($N_EVENTS+1)) + ($FREEZE_S*$N_EVENTS)}")
ffmpeg -y -hide_banner -loglevel error "${args[@]}" \
  -f lavfi -i "sine=frequency=1000:sample_rate=44100:duration=${TOTAL_S}" \
  -filter_complex "$filters" -map "[vout]" -map "${SINE_IDX}:a:0" \
  -c:v libx264 -pix_fmt yuv420p -g 1 -bf 0 -crf 18 \
  -x264-params keyint=1:min-keyint=1:scenecut=0 \
  -c:a aac -b:a 64k -movflags +faststart "$OUT"

echo "wrote $OUT"
ffprobe -v error -select_streams v:0 -show_entries stream=width,height,r_frame_rate,nb_frames,duration,pix_fmt -of default=nw=1 "$OUT"
ffprobe -v error -select_streams a:0 -show_entries stream=codec_name,sample_rate,duration -of default=nw=1 "$OUT"
echo "total duration: ${TOTAL_S}s  (freeze events: ${N_EVENTS} x ${FREEZE_S}s, moving: $((N_EVENTS+1)) x ${MOVE_S}s)"
