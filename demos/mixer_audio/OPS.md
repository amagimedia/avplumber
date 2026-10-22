# mixer_audio — ops cheat sheet

Practical host layout, ports, and copy-paste commands so a later chat does
not have to rediscover them. Product docs stay in [`README.md`](README.md).

## Host layout (AWS Fedora demo host)

| Item | Path / value |
|------|----------------|
| Workspace root | `/home/fedora/audio-video-mixer/` |
| avplumber checkout | `/home/fedora/audio-video-mixer/avplumber` |
| SSH | `ssh -i ~/.ssh/<your-key>.pem fedora@<HOST_IP>` |
| Current IP | ask the team / check the ops channel — it changes whenever the instance is replaced |

Test video **and** audio are fully synthetic — `prepare_demo.py` generates
them on first run via FFmpeg (`lavfi testsrc2` / `sine=frequency=...` muxed
to h264/AAC, cached under `/media/assets/synthetic_v6_*`). Nothing needs to
be copied onto the host by hand.

Compose is run from the `avplumber` checkout:

```sh
cd /home/fedora/audio-video-mixer/avplumber
COMPOSE=docker-compose/mixer_audio/compose.yaml
```

## URLs / ports

| Port | Service |
|------|---------|
| 7681 | mixer web UI (`pyplumber.mixer.gui.web`, operator cut/fade/scenes) |
| 7777 | mixer control (avplumber commands) |
| 8088 | Janus HTTP API |
| 5004 | video RTP → Janus |
| 5002 / 5003 | Opus RTP / RTCP → Janus |

Compose project: `avplumber-mixer`. Service: `mixer`
(container `avplumber-mixer-mixer-1`). Image: `avplumber-mixer:local`, built
from `docker-compose/mixer/Dockerfile.fedora44` with build context `../..`.
No CI/registry — the image is always rebuilt directly on the host.

## Rebuild and redeploy

```sh
ssh -i ~/.ssh/<your-key>.pem fedora@<HOST_IP>
cd /home/fedora/audio-video-mixer/avplumber
docker compose -f docker-compose/mixer_audio/compose.yaml build mixer
docker compose -f docker-compose/mixer_audio/compose.yaml up -d --force-recreate mixer
docker logs avplumber-mixer-mixer-1 --tail 100
```

## Known live fixes already in tree

1. **Startup `mixer.cut` races node construction, leaving audio permanently
   silent (fixed 2026-10-16).** `MixerGraphBuilder._emit_mixer_commands()`
   used to fire a synthetic `mixer.cut` synchronously inside `build()`, to
   deliver one `transition_ended` to `audio_router_node` and raise
   `AudioMixer`'s gains off their all-zero construction default. But the
   compositor/`audio_mixer` node objects are created lazily, only once their
   group's `startNodes()` runs — strictly after `build()` returns. The cut
   always raced ahead and failed with `mixer: set mixer_comp_b.composition
   failed: node has no runtime controls`, visible in `docker logs` right at
   startup. Video was unaffected (PGM video routing is set directly from
   `mixer.init`), but audio stayed muted until a human triggered a manual
   scene cut from the web UI. Fixed by moving the gain-raising cut to
   `MixerApplication._start()`, after `OUTPUT_GROUP.startNodes()` and
   `_wait_for_node(audio_router_node)` — see
   `pyplumber/mixer/graph.py::raise_initial_audio_gains` and
   `pyplumber/mixer/application.py::MixerApplication._start`.

   Diagnostic: if audio is silent on a fresh container start but comes back
   after a manual cut via the web UI, suspect this exact race recurring
   (e.g. after reverting the fix, or in a from-scratch rebuild of an older
   image). Check `docker logs <container> | grep "mixer.cut"` for a startup
   failure immediately followed, much later, by a successful manual one.

2. **Source clips were only 2 seconds long, so audio sounded like a
   constantly-restarting loop (fixed 2026-10-16).** `demo.example.json`'s
   `generation.seconds` defaulted to `2` — fine for GPU capacity testing, bad
   for actually listening to the demo. Bumped the template to `300` (5
   minutes) and patched the live `/media/demo.json` on the host directly
   (editing the template alone has no effect once a host already has a
   persisted `/media/demo.json` — `SetupRuntime.resume()` reads that file
   as-is on every restart and never re-derives it from the template unless
   it's missing). After `docker restart avplumber-mixer-mixer-1`,
   `prepare_demo` regenerated fresh assets under
   `/media/assets/synthetic_v6_*_300s/` (confirmed via `ffprobe`: `Duration:
   00:05:00.00`); the stale `*_2s/` cache directory is harmless leftover
   (root-owned, `fedora` can't `rm` it, not worth `sudo` for 2.6 MB).

   Ruled out before landing on this: the periodic `[resample_audio_N]
   output PTSes too large, dropping output, 256 samples` log line (every
   ~2.00s) looked like it might be a loop-boundary artifact, since it
   matched the old 2-second clip length exactly. It is **not** — after
   regenerating 300-second assets, the same message still fires every ~2s
   from the first few seconds of playback, long before any 300s loop point
   could occur. It's a small, constant, pre-existing audio-clock-rate drift
   (~5.3 ms dropped roughly every 2s, independent of content length or
   looping) — a separate, minor cosmetic issue, not investigated further
   here.

3. **Janus never relayed the audio track to WebRTC viewers at all, even
   with fixes #1/#2 live (fixed 2026-10-16).**
   `docker-compose/dmabuf/compose.yaml`'s shared `janus` service hardcodes
   `JANUS_AUDIO_ENABLED: "false"` (a literal string, not a `${VAR:-false}`
   default — an env var override from outside has no effect). Its
   entrypoint (`docker-compose/images/janus/janus-entrypoint.sh`)
   `sed`s that straight into `janus.plugin.streaming.jcfg`'s
   `avplumber-program` mountpoint's `audio = __JANUS_AUDIO_ENABLED__`. With
   it `false`, Janus never advertises or forwards the Opus audio m-line to
   any WebRTC subscriber — regardless of whether avplumber's own
   `audio_mixer`/Janus-audio-encode pipeline is correctly producing and
   sending Opus RTP to port 5002/5003 (it was, the whole time; this is a
   separate stage downstream of everything fixes #1 and #2 touch).
   `docker-compose/mixer/compose.yaml` (video-only) correctly leaves this
   off; `docker-compose/mixer_audio/compose.yaml`'s own `janus:` service
   just never overrode it. Fixed by adding, next to the existing
   `janus-preview:` environment override in the same file:
   ```yaml
     janus:
       extends: {file: ../../docker-compose/dmabuf/compose.yaml, service: janus}
       environment:
         JANUS_AUDIO_ENABLED: "true"
   ```
   Only the `janus` service needs recreating for this — no image rebuild,
   the substitution happens in the entrypoint at container start:
   ```sh
   docker compose -f docker-compose/mixer_audio/compose.yaml up -d --force-recreate janus
   ```
   Verify: `docker exec <janus container> grep -A1 "^avplumber-program" \
   /opt/janus-avp/etc/janus/janus.plugin.streaming.jcfg` should show
   `audio = true` a few lines down.
