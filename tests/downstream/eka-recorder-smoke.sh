#!/usr/bin/env bash
# Standalone eka-recorder e2e smoke: brings up the Kafka broker it needs, checks the
# models and the simulation input, runs the recorder GCS harness on a given image and
# prints PASS/FAIL with the verified numbers.
#
#   ROOT=<harness dir> GCS_PREFIX=gs://<bucket>/<prefix> eka-recorder-smoke.sh IMAGE [LABEL]
#   KEEP_BROKER=1  leave the redpanda container running afterwards (default: remove it)
#   Run as root or with `sudo KEEP_BROKER=1 bash eka-recorder-smoke.sh ...` (plain sudo drops env).
#
# Downstream check for the eka-recorder / recorder-ai product on a candidate avplumber build:
# the recorder image is built from the recorder repo with avplumber's python module and FFmpeg
# libraries injected (see the recorder repo's audit Dockerfile). Host-specific locations are
# variables so the script can move between hosts. Exit 0 only on PASS.
set -uo pipefail

IMAGE=${1:?recorder image (built from the recorder repo with this avplumber injected)}
LABEL=${2:-smoke}
ROOT=${ROOT:?harness dir: recorder repo checkout, input excerpt, models, outputs}
REPO_DIR=${REPO_DIR:-$ROOT/repo}                        # contains ai-recorder/scripts/run_e2e_gcs_test.sh
SOURCE_FILE=${SOURCE_FILE:-$ROOT/simulation-40s.ts}     # 40 s stream-copy excerpt, looped by file ingest
MODELS_DIR=${MODELS_DIR:-$ROOT/models}                  # basketball/{yolo,tracknet_v2,salient-viewport,scoreboard}/*.plan
OUTPUT_BASE_DIR=${OUTPUT_BASE_DIR:-$ROOT/output}
GCS_PREFIX=${GCS_PREFIX:?gs://<bucket>/<prefix> the harness uploads recordings to}
BROKER=${BROKER:-recorder-smoke-kafka}
BROKER_IMAGE=${BROKER_IMAGE:-docker.redpanda.com/redpandadata/redpanda:v24.3.1}
KAFKA_PORT=${KAFKA_PORT:-29092}                         # the recorder resolves kafka -> 127.0.0.1:29092
KEEP_BROKER=${KEEP_BROKER:-0}
RUN_SEC=${RUN_SEC:-150}
E2E_WINDOW_SEC=${E2E_WINDOW_SEC:-110}

log() { echo "[smoke] $*"; }
die() { echo "[smoke] FAIL: $*" >&2; exit 1; }

# --- preflight -------------------------------------------------------------
docker image inspect "$IMAGE" >/dev/null 2>&1 || die "image $IMAGE not found"
[ -f "$SOURCE_FILE" ] || die "simulation input missing: $SOURCE_FILE"
[ -f "$REPO_DIR/ai-recorder/scripts/run_e2e_gcs_test.sh" ] || die "harness missing under $REPO_DIR"
for m in yolo tracknet_v2 salient-viewport scoreboard; do
    ls "$MODELS_DIR/basketball/$m/"*.plan >/dev/null 2>&1 || die "models missing: $MODELS_DIR/basketball/$m"
done
gcloud storage ls "$GCS_PREFIX/" >/dev/null 2>&1 || die "no GCS access to $GCS_PREFIX"

# --- broker ----------------------------------------------------------------
# Redpanda's default schema-registry/pandaproxy/rpc ports (8081/8082/33145) may be taken
# by other containers on a shared host, so they are moved; only the Kafka port matters.
if ! (echo > "/dev/tcp/127.0.0.1/$KAFKA_PORT") 2>/dev/null; then
    docker rm -f "$BROKER" >/dev/null 2>&1 || true
    docker run -d --name "$BROKER" --network host "$BROKER_IMAGE" redpanda start \
        --overprovisioned --smp 1 --memory 512M --reserve-memory 0M --node-id 0 --check=false \
        --kafka-addr "plaintext://0.0.0.0:$KAFKA_PORT" --advertise-kafka-addr "plaintext://127.0.0.1:$KAFKA_PORT" \
        --schema-registry-addr 0.0.0.0:28081 --pandaproxy-addr 0.0.0.0:28082 --advertise-pandaproxy-addr 127.0.0.1:28082 \
        --rpc-addr 0.0.0.0:33146 --advertise-rpc-addr 127.0.0.1:33146 >/dev/null || die "cannot start $BROKER"
    for i in $(seq 1 30); do (echo > "/dev/tcp/127.0.0.1/$KAFKA_PORT") 2>/dev/null && break; sleep 1; done
    (echo > "/dev/tcp/127.0.0.1/$KAFKA_PORT") 2>/dev/null || { docker logs --tail 5 "$BROKER"; die "broker not listening on $KAFKA_PORT"; }
    log "broker $BROKER up on $KAFKA_PORT"
else
    log "port $KAFKA_PORT already accepting connections; reusing"
fi

# --- harness: file ingest, AI on, camera motion, no Janus ---
export IMAGE REPO_DIR SOURCE_FILE MODELS_DIR OUTPUT_BASE_DIR REBUILD_IMAGE=0 KEEP_LOCAL=1
export E2E_RECORDING_ID="${LABEL}-smoke-$(date -u +%Y%m%d-%H%M%S)"
export CONTAINER_NAME="recorder-${LABEL}-smoke" MEDIAMTX_CONTAINER="recorder-${LABEL}-unused-mediamtx"
export E2E_INGEST=file E2E_SPORT=basketball E2E_QUALITIES=fhd,hd,hi E2E_OCR=1 E2E_NO_AI=0
export E2E_CAMERA_MOTION=1 E2E_WEBUI=0 E2E_JANUS=0 E2E_HLS_DEBUG=1 E2E_HLS_DEBUG_UPLOAD=1
export RECORDER_METADATA_KAFKA_ENABLED=true RECORDER_METADATA_JSONL=1
export RECORDER_SRC_DIR="$REPO_DIR/ai-recorder/recorder"
export E2E_WINDOW_SEC RUN_SEC MODELS_MOUNT_MODE=rw
log "recording $E2E_RECORDING_ID on $IMAGE (${RUN_SEC}s run, ${E2E_WINDOW_SEC}s window)"
HLOG="$OUTPUT_BASE_DIR/$E2E_RECORDING_ID.harness.log"; mkdir -p "$OUTPUT_BASE_DIR"
bash "$REPO_DIR/ai-recorder/scripts/run_e2e_gcs_test.sh" > "$HLOG" 2>&1
STATUS=$?

# --- numbers ---------------------------------------------------------------
RUN="$OUTPUT_BASE_DIR/$E2E_RECORDING_ID"
SEGMENTS=$(find "$RUN/recordings" -name '*.ts' 2>/dev/null | wc -l)
AVPLOG="$GCS_PREFIX/$E2E_RECORDING_ID/logs/avplumber.log"
PANICS=$(gcloud storage cat "$AVPLOG" 2>/dev/null | grep -ci panic)
RESTARTS=$(gcloud storage cat "$AVPLOG" 2>/dev/null | grep -c "initiated group auto-restart")
META=$(gcloud storage cat "$GCS_PREFIX/$E2E_RECORDING_ID/metadata/inference.jsonl" 2>/dev/null | python3 -c '
import sys, json
pts = [json.loads(l)["frame_pts"] for l in sys.stdin if l.strip()]
print(f"{len(pts)} records spanning {max(pts) - min(pts):.1f} s of frame_pts" if pts else "0 records")')
VERDICT=$(grep -oE "full-flow .* (PASSED|FAILED)" "$HLOG" | tail -1)

echo "----------------------------------------------------------------"
echo "recording:     $E2E_RECORDING_ID"
echo "image:         $IMAGE"
echo "harness:       exit=$STATUS  ${VERDICT:-no verdict line}"
echo "hls segments:  $SEGMENTS (all qualities)"
echo "inference:     ${META:-unavailable}"
echo "avplumber:     panics=${PANICS:-?} input-group restarts=${RESTARTS:-?} (file ingest loops at EOF)"
echo "harness log:   $HLOG"
echo "----------------------------------------------------------------"
[ "$KEEP_BROKER" = "1" ] || docker rm -f "$BROKER" >/dev/null 2>&1
if [ "$STATUS" = "0" ] && [ "${PANICS:-1}" = "0" ]; then echo "RESULT: PASS"; exit 0; else echo "RESULT: FAIL"; exit 1; fi
