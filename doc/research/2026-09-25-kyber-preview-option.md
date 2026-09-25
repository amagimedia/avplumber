# Preview delivery latency: current RTP→Janus→WebRTC path vs a Kyber (QUIC) option

Status: investigation, 2026-09-25. Nothing in the live setup was changed.

## Measured budget of the current path (1080x1920p60 H.264 program rendition)

Sender side, captured on the loopback between the mixer and Janus (7 s, 420 frames):

| Stage | Measured |
|---|---|
| Frame emission interval | median 16.6 ms, p95 20.4 ms, max 30.8 ms |
| P-frame send burst (UDP pacer) | median 0.2 ms, p95 5.5 ms |
| I-frame send burst (pacer at 3x bitrate) | 11–24 ms, once per second plus PLI-triggered |

Receiver side, one Chrome viewer over the internet (preview receiver stats, 300 samples):

| Metric | Value |
|---|---|
| RTT | 38 ms (one way ≈ 19 ms) |
| Jitter buffer residence | median 16 ms, min 9, p90 28, max 34 |
| Decode | 2 ms |
| Session loss | 134 packets lost, 50 NACKs, 4 PLIs, 7 freezes |

Pipeline constants: compositor playout buffer is two output frames by default
(`canvas.latency_ms`, 33 ms at 60 fps). Snapshot and ForceFPS nodes pass frames
through without holding them in live mode. NVENC runs with `tune=ull`,
`delay=0`, no B-frames.

Estimated mixer-tick-to-glass: 33 playout + ~5 encode + 0–24 pacing + 19 network
+ 16 jitter buffer + 2 decode + ~8 vsync ≈ 85–110 ms. Only the 19 ms network
share is outside our control; no transport change reduces the RTT.

## Cheap improvements inside the current stack (do these first)

1. Browser: set `jitterBufferTarget = 0` (fallback `playoutDelayHint = 0`) on
   the video RTCRtpReceiver in the preview page. Removes most of the 9–34 ms
   jitter-buffer residence. One line, per-viewer, reversible.
2. Keyframes: periodic IDR every second is redundant with PLI feedback and
   Janus NACK retransmission. Lengthen `interval_sec` / `g` (5–10 s) or use
   NVENC intra refresh. Removes the 11–24 ms per-second pacing bursts that also
   inflate Chrome's jitter estimate.
3. Mixer: try `canvas.latency_ms` = one frame (16.7 ms) on the aux bus first;
   watch playout repeat/overflow counters before applying to program.

Expected total gain: 20–45 ms, more than a Kyber migration would give.

## What Kyber is, from its sources (gitlab.com/kyber)

- Data plane `kyproto` over QUIC (native, quinn/wtransport) or WebTransport
  (browser). Video protocols: `reliable` (one stream), `gopstream` (one QUIC
  stream per GOP, old GOPs abandoned), `unreliable` (datagrams, reorder with
  timeout), `unreliable_fec` (datagrams + RaptorQ repair).
- No NACK and no in-band keyframe request. Loss is handled by QUIC
  retransmission (reliable/gopstream, costs an RTT) or FEC + timeout.
  Default `intra_refresh = true` so the decoder recovers without IDRs.
- Encoder control exists on the server side only: controller → avserver
  commands `VideoForceIdr` and `VideoSetBitrate`.
- Ingest: the streamer (avserver) hands encoded access units to kymux over a
  local TCP IPC (`kycom`): 12-byte header (`codec` fourcc `h264`/`h265`/`av1`,
  or media: pts, is_key, is_config, size) + payload, first a codec packet and
  an extradata config packet. Exactly what our `janus_repeat_headers` edge
  carries, so avplumber can feed Kyber without decode/re-encode or screen grab.
- Web client: Chrome only (WebTransport + WebCodecs, `video_buffer` default
  0 ms). Firefox experimental, Safari shows no video. HEVC only where Chrome
  has it. Our current preview plays HEVC/HDR in Safari.
- One QUIC connection per viewer session; no server-side fan-out like Janus.
- License AGPLv3 or commercial.

## Expected latency effect of Kyber

Replaces Janus + Chrome's adaptive jitter buffer with WebTransport + WebCodecs
at 0 ms buffering. Realistic saving is the jitter-buffer residence (≈16 ms
median, ≤34 ms) and part of the RTP pacing. It does not change the 38 ms RTT,
the 33 ms playout buffer, encode, decode or vsync. Net: 15–30 ms, comparable to
item 1 above. Its real differentiator is FEC on lossy links (our session showed
134 lost packets / 50 NACKs), which Janus streaming mountpoints do not offer.

## Migration plan as option 2 (program + aux preview), if pursued

Keep the RTP→Janus path untouched; add Kyber as an additional rendition target.

1. `pyplumber/mixer/janus.py`-style builder `build_kyber_output`: same
   ForceFPS → ForceKeyFrame → EncVideo → dump_extra chain, terminated by a new
   avplumber node `kycom_output` that writes kycom AVPackets over TCP
   (codec packet, extradata config packet, then media packets with pts in the
   stream time base and `is_key`). Config: `renditions[].target = "kyber"`.
2. Bridge service (Rust, ~kyproto + kynet-wtransport + kycom crates): accepts
   the avplumber TCP feed, accepts WebTransport sessions, registers a video
   endpoint per viewer with `unreliable_fec` (lossy links) or `gopstream`,
   and fans out packets. Kyber's own controller/avserver are not needed; they
   assume they own screen capture and encoding.
3. Keyframes: bridge requests an IDR from avplumber on viewer join (reuse the
   `janus_force_keyframe` trigger, min-interval guarded) instead of PLI.
   Enable NVENC intra refresh for the Kyber rendition so loss recovery does
   not depend on IDRs.
4. Viewer page: either kyber-web's wasm client (needs its controller/auth
   flow) or a small WebTransport + WebCodecs page mirroring the current
   preview footer stats. Chrome only; keep the Janus page for Safari/HEVC.
5. Measure with the same receiver-stats pipeline before deciding to switch
   program preview; aux/multiview first.

Effort: node + builder ~1–2 days; bridge + page ~1 week; certificates and
WebTransport `serverCertificateHashes` handling for non-public hosts extra.

## Local-box deployment (GUI and CUDA mixer on the same machine, no internet)

The WebRTC stack works offline: Janus needs only `JANUS_HOST_IP` set to the
LAN or loopback address, the preview already uses no STUN/TURN
(`iceServers: []`), and `ice_enforce_list` pins the interface. RTT drops to
≈0, so the same page lands at ≈60–80 ms tick-to-glass with the tweaks above.
It keeps one code path for cloud and local. Going lower on a local box means
bypassing the browser transport entirely (direct GPU presentation), which is a
different GUI, not a transport change.

## Live trial and rollback (2026-09-25)

All three changes were applied together and the mixer restarted through the
setup manager (20 s). A fresh H.264 viewer then measured jitter-buffer
residence median 12 ms / max 14 ms (before 16 / 34) at 59.8 fps, IDRs every
5 s, GPU load unchanged. The HDR (HEVC) preview became unusable at the same
time: fps swinging 0–70, buffer up to 220 ms, hundreds of NACKs. Everything
was rolled back and the mixer restarted again; after a page reload the viewer
was steady at 60 fps.

Most likely cause: viewers whose Janus session survived a mixer restart see
the RTP sequence numbers reset and go into a NACK/PLI storm; the tab has to
be reloaded after any mixer restart. The zero-buffer hint may additionally
hurt slow HEVC decoders, which drop late frames and lose references.

Current state:
- Preview page: the receiver hint is opt-in with `?lowlat=1`, off by default,
  so it can be A/B tested per tab without touching the mixer.
- `JanusVideoConfig.keyframe_interval_sec` default 3 s: applied alone on the
  live demo (restart plus tab reload), both renditions verified at one IDR per
  3.0 s and 60 fps, accepted by the user.
- `?lowlat=1` tested: about 4–5 ms less buffer on a clean link, no cushion on
  a bad one; left opt-in and not recommended by default.
- `canvas.latency_ms` is back to the default two frames on the live recipe;
  the one-frame trial is still open (restart, tab reload, watch compositor
  repeat counters).

## Playout buffer trial: round-up alignment + 25 ms (2026-09-25 evening)

Browser paint timing sampled from the DMA-BUF service (16 windows, 72k paints
over 75 s): p99 lateness +4.3 ms, p99.9 +7.3 ms, worst +9 ms, none beyond
12 ms. With nearest-tick numbering a browser can additionally sit up to 8.3 ms
behind its assigned tick, so the one-frame buffer had negative worst-case
margin while two frames had ~14 ms.

Change: `smooth_timestamps` gained `round_up`, used by the browser chain, so
the first paint and any resync are numbered to the next canvas tick instead of
the nearest. The compositor now publishes playout counters (repeats, discards,
overflow, missed deadlines, per input) in its `status` object, readable with
`node.object.get <compositor> status`, no debug logging needed.

Live result with `canvas.latency_ms = 25`: over 13,020 frames per program
compositor at steady state, 0 repeats, 0 missed deadlines, 5 discards (drift
housekeeping from browsers at 60.02 fps). Viewers at 60 fps, 10–12 ms jitter
buffer, no loss. Program latency 8.3 ms lower than the two-frame default with
the same worst-case margin. Startup still shows ~60 repeats in the first 90 s
while browsers reconnect and prewarm fills.
