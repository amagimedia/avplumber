# Docker Compose Runner

This directory owns shared demo orchestration. It keeps Janus, web UI, DMA
browser, and AVPlumber runners in one stack.

The generic `runner` service runs any mounted `.avplumber` script. Application
and sport-specific stacks belong in their respective downstream repositories.

## Layout

- `docker-compose.yml` - shared support services and selectable runners.
- `run-demo.sh` - small wrapper around `docker compose`.
- `.env.example` - all common knobs in one place.
- `demos/*.env.example` - starting points for common demo choices.
- `scripts/run-avplumber-script.sh` - entrypoint for a custom `.avplumber` graph.
- `images/` - Dockerfiles for the shared support services.

## Custom Script

Copy and edit an env file:

```sh
cp docker-compose/demos/custom-script.env.example docker-compose/demos/custom-script.env
```

Set host paths in the env file:

- `AVP_SCRIPT_FILE` is the host `.avplumber` script to run.
- `AVP_MEDIA_DIR` is mounted read-only at `/media`.
- `AVP_MODELS_DIR` is mounted read-only at `/models`.
- `AVP_ARTIFACT_DIR` is mounted read-write at `/artifacts`.

Run it:

```sh
docker-compose/run-demo.sh --env docker-compose/demos/custom-script.env --profile script up
```

## Support Services Only

```sh
docker-compose/run-demo.sh --env docker-compose/.env up -d
```

The web UI defaults to `http://127.0.0.1:22222`, Janus HTTP to
`http://127.0.0.1:8088/janus`, and the preview server to
`http://127.0.0.1:8080`.

The preview footer supports [cut-to-encoded-output measurements](../doc/mixer_cut_latency.md)
beside WebRTC RTT. Enable the optional native probe and supply a deployment-local
`metrics.json`; the viewer never changes the mixer itself.

Playback FPS, receiver buffer residence and the browser's dropped-frame counter are
shown separately from cut latency and network RTT. Unsupported counters display
`—`. The preview also keeps up to 300 diagnostic samples per connection in
memory, available at `/receiver-stats`, to correlate a visible stall with
packet loss, decode time or presentation gaps. It stores counters only, for at
most 16 connections, and removes a connection five minutes after its last report.
Hidden-tab and paused-video intervals are marked and excluded from the local
presentation-gap detector. A new connection or decoder counter reset starts a
new measurement period; compare counters within that period.

## RTP burst headroom

A detailed mixer grid can produce much larger keyframes than a fullscreen
source at the same average bitrate. If the whole picture periodically freezes,
check Janus's UDP sockets with `sudo ss -uanmp`: increasing `d` counters in
`skmem` indicate packets dropped at that socket, before WebRTC delivery.
A smooth encoder frame rate does not rule out this packet loss.

On a Linux demo host, allow 4 MiB socket buffers before starting Janus:

```sh
sudo sysctl -w net.core.rmem_max=4194304 net.core.wmem_max=4194304
sudo sysctl -w net.core.rmem_default=4194304 net.core.wmem_default=4194304
```

Keep any existing larger limits. These are host-wide defaults for newly created
sockets; existing Janus mountpoints and viewer connections need to be recreated
to pick them up. The settings last until reboot unless added to the host's
sysctl configuration. This stack uses host networking, so they belong on the
host rather than in Compose service `sysctls`.

This adds capacity for packet bursts, not a fixed playout delay, and does not
change encoded image quality. If packet loss persists downstream, measure frame
sizes and receiver loss before reducing bitrate or the NVENC VBV budget: an
overly small keyframe budget can make text and details in small tiles unreadable.
