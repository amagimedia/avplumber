# Cut-to-encoded-output measurements

The optional mixer probe measures elapsed monotonic time from receipt of a
complete `mixer.cut` command by AVP to enqueue of the first matching encoded
frame at the configured encoder's output. TCP control-queue and command-lock
waiting are included. For in-process commands, timing begins when the command
line is parsed. Scheduled cuts include the requested scheduling delay.

This is not command acknowledgment time, output PTS age, capture-to-output
pipeline delay, RTP transmission time, or browser display latency. The endpoint
is explicitly reported as `encoder_output`; subsequent bitstream filtering,
muxing, Janus, network transit, jitter buffering and rendering are excluded.
WebRTC RTT is a separate round trip and must not be added as a one-way delay.

After the mixer selector and encoder have been created, enable the probe once:

```text
mixer.measurements {"mixer":"mixer","encoder":"janus_encoder"}
mixer.status mixer
```

The Python mixer demo can enable it at every startup with
`--cut-latency-encoder janus_encoder` (or the name of its recording encoder).

The configuration is opt-in. It binds observers to existing nodes, without
rewiring, flushing, restarting, or changing prewarm or routing behavior.
`mixer.status` contains `cut_latency` with independent `direct` and `previewed`
samples. Each has `id`, `scene`, `state`, and `ms` (null until measured).
`encoded_pts` identifies the matching access unit in the encoder's time base.
Each category also has `recent`, its last three successfully measured cuts in
completion order. This bounded history is maintained in AVP, so cuts between
viewer polls are retained. Failed or interrupted measurements do not enter it;
it resets when the probe is recreated, including on mixer process restart.
Previewed means the target was already loaded in PVW at CUT receipt; earlier
preview preparation is excluded. Loading PVW is not proof it was fully warmed.

The selector marks target frames only after the cut is armed. A private AVFrame
property copy shares the existing pixel buffers; it does not download or copy
CUDA pixels. The encoder correlates that marker with packet PTS, including
buffered or reordered encoder output. Old pictures, fallback frames, stale cut
generations and mismatched PTS cannot complete a measurement. Interruption,
failure, supersession, duplicate/non-monotonic input PTS or a 30-second timeout
leave the affected sample unavailable. Timing records are bounded. Intermediate
nodes must preserve AVFrame metadata; paths that discard the marker remain
unavailable rather than estimating a value. The probe follows scene selection,
not optical image differences: identical-looking scenes can still be measured.

## Preview page

The preview footer shows AVP Direct · median 3, AVP Previewed · median 3 and
WebRTC RTT side by side, outside the video. All use integer milliseconds and
green through 200 ms, orange through 400 ms, then red. Each cut value is the
median of up to the last three successfully measured cuts in that category,
not continuously recomputed pipeline latency. One result is shown immediately;
with two, their midpoint is shown; from three onward the window rolls. Existing
history remains visible during a pending or failed cut. No measurements or
disconnected data is gray `—`; hover lists
the contributing cuts and explains the latest state. Browser reloads do not
reset AVP's history. Older probes without history cannot display a median.

Supply a deployment-local `metrics.json` alongside the preview's `index.html`:

```json
{
  "ws_url": "ws://127.0.0.1:22222/ws",
  "instance_id": "127.0.0.1:7777",
  "mixer": "mixer"
}
```

Use the WebUI bridge address reachable **from the browser**, the registered
instance ID, and `wss://` when the preview is HTTPS. Keep deployment addresses
and source lists out of the public repository. The existing bridge is a control
endpoint, not an authenticated read-only API; retain its deployment's access
controls. The measurement client sends only `mixer.status`, once per second,
with one request in flight and timeout/reconnect handling. WS and browser
delivery delay do not enter the AVP measurement. It never enables the probe or
sends cuts automatically. An older AVP build displays “Mixer not instrumented”
on hover, with no fabricated values and no effect on video playback.

Tests:

```sh
python -m pytest tests/test_cut_latency.py
node docker-compose/images/preview/tests/cut-latency.test.mjs
node docker-compose/images/preview/tests/rtt.test.mjs
```

Media integration must additionally be verified on a CUDA/NVENC host, checking
the identity of decoded encoded-output frames for both direct and previewed
cuts. Installing a new native binary in a running demo requires a separately
agreed rollout; serving this viewer does not upgrade the mixer.
