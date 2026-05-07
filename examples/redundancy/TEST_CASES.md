# Fabric Redundancy Test Cases

This file records the redundancy experiments run on the Fedora host against the
jittered sync-bar SRT inputs. The current useful output path is:

```text
rtmp://test-streamer-s3dev.aws-dev.intranet/stream_test/test
```

The jittered source generator sends ordered MPEG-TS bytes with roughly 1000 ms
mean delay and +/-1000 ms jitter while preserving media PTS.

## 1. Active-Active, Two URLs, Jittered

Graphs:

```text
examples/redundancy/h264_intra/active_active_sync_bars_rtmp/consumer_selector_rtmp.avplumber
examples/redundancy/h264_intra/active_active_sync_bars_rtmp/producer_a.avplumber
examples/redundancy/h264_intra/active_active_sync_bars_rtmp/producer_b.avplumber
```

Switch sequence:

```text
A active -> kill A -> B selected
restart A -> kill B -> A selected
```

Result:

```text
output: /home/jp/out_active_active_jitter_2switch.mp4
duration: 33.993s
packets: 850
max packet delta: 0.040s
bad gaps >60ms: 0
```

Important behavior:

```text
Both producers send full media.
The consumer does not decode both streams.
redundancy_selector emits one selected packet per normalized frame identity.
Bandwidth is roughly doubled before selection.
```

## 2. Active-Standby, Same URL

Graphs:

```text
examples/redundancy/h264_intra/active_standby_sync_bars_rtmp/consumer_rtmp.avplumber
examples/redundancy/h264_intra/active_standby_sync_bars_rtmp/producer_a.avplumber
temporary producer_b using the first URL
```

Switch sequence:

```text
A active -> kill A -> B repairs/promotes
restart A -> kill B -> A takes over
```

Result:

```text
output: /home/jp/out_active_standby_same_url_2switch.mp4
duration: 33.993s
packets: 850
max packet delta: 0.040s
bad gaps >60ms: 0
```

Important behavior:

```text
A sends media.
B sends status only.
The standby status payload is zero bytes, so steady-state media bandwidth is not doubled.
```

## 3. Active-Standby, Two URLs, Timeline Normalized

Graphs:

```text
examples/redundancy/h264_intra/active_standby_sync_bars_rtmp/consumer_rtmp.avplumber
examples/redundancy/h264_intra/active_standby_sync_bars_rtmp/producer_a.avplumber
examples/redundancy/h264_intra/active_standby_sync_bars_rtmp/producer_b.avplumber
```

Switch sequence:

```text
A active -> kill A -> B repairs/promotes
restart A -> kill B -> A takes over
```

Result:

```text
output: /home/jp/out_active_standby_two_url_normalized_2switch.mp4
duration: 33.993s
packets: 850
max packet delta: 0.040s
bad gaps >60ms: 0
```

Important behavior:

```text
fabric_source learns standby timeline offset.
Slots are keyed by normalized frame ID.
REPAIR/PROMOTE commands are translated back to standby raw frame IDs.
```

Example proof log:

```text
fabric_source learned standby offset replica_id=2 offset=183600
fabric_source control REPAIR normalized_pts=179353920 standby_raw_pts=179170320
fabric_sink repair sent pts=179170320 replica_id=2
```

## 4. Restart/Election, Asymmetric Rejoin

Attempted sequence:

```text
A active -> kill A -> B promotes
restart A as active sender
kill B -> try to use A
restart B as standby
kill A -> try to use B
```

Result:

```text
output: /home/jp/out_active_standby_restart_election_frag.mp4
result: failed quality check
max packet delta: 12.760s
bad gaps >60ms: 2
```

Reason:

```text
Restarted A came back as an active media sender, not as a controlled standby.
When B died, A's live input was ahead of the consumer playout point.
The source had to skip forward to A's current frame identity.
```

Conclusion:

```text
For restart/election, both replicas must be able to rejoin as controlled standby
instances with repair cache and a control port.
```

## 5. Restart/Election, Symmetric Standby Rejoin

Implementation added:

```text
fabric_sink generation=0 now creates a per-process session generation.
fabric_source tracks generation changes per replica.
fabric_source supports standby_control_ports, mapping replica ID to control port.
fabric_source demotes a restarted active owner back to preferred active ownership.
fabric_source can issue REPAIR/PROMOTE to either replica.
```

Test shape:

```text
A starts as active on port 5556.
B starts as standby on port 5557, control port 6556.
After A dies, A is restarted from a temporary standby graph with control port 6555.
B dies, A repairs/promotes.
B is restarted as standby.
A dies again, B repairs/promotes.
```

Proof logs:

```text
fabric_source control REPAIR replica_id=2 normalized_pts=277644720 raw_pts=277471920
fabric_source elected active replica_id=2 previous_active_replica_id=1

fabric_source control PROMOTE replica_id=1 normalized_pts=279376320 raw_pts=279376320
fabric_sink promoted from pts=279376320 replica_id=1

fabric_source active replica restarted as new generation; returning active owner to preferred replica restarted_replica_id=2 preferred_replica_id=1

fabric_source control PROMOTE replica_id=2 normalized_pts=281003520 raw_pts=280819920
fabric_sink promoted from pts=280819920 replica_id=2
```

Recorder note:

```text
The symmetric restart/election control flow passed in logs, but the local MP4
recording only captured a short valid fragmented segment:

output: /home/jp/out_active_standby_restart_election_symmetric.mp4
duration: 5.600s
packets: 140
max packet delta: 0.040s
bad gaps >60ms: 0

This MP4 is not sufficient proof of the full multi-switch scenario. The next
recording pass should capture to MPEG-TS or FLV first, then remux to MP4 after
the process exits cleanly.
```

## 6. Restart/Election, Symmetric Standby Rejoin, MPEG-TS Recording

The same symmetric standby-rejoin sequence was rerun with direct MPEG-TS
recording to avoid MP4 finalization problems.

Output:

```text
/home/jp/out_active_standby_restart_election.ts
```

Recorded file:

```text
duration: 5.360s
packets: 134
max packet delta: 0.040s
bad gaps >60ms: 0
```

This proves the TS container was valid, but it still did not capture the full
multi-switch scenario. The fabric logs show the run stopped producing media
after B rejoined:

```text
fabric_source active replica restarted as new generation; returning active owner to preferred replica restarted_replica_id=2 preferred_replica_id=1
fabric_source generation changed replica_id=2 generation=1778092029722511578
fabric_source skipped missing frame slot from_normalized_pts=296670720 to_normalized_pts=8589934592
fabric_source throughput mbps=0 ... total_bytes=9566408
```

Interpretation:

```text
The source accepted a bad/extreme frame identity from a restarted standby and
advanced next_emit_frame_ to it. After that, B was sending status-only frames
and no media was emitted.
```

Next fix:

```text
Reject or quarantine invalid pts values before they can update
frame_slots_, highest_seen_frame_, or next_emit_frame_.

A new generation should not be eligible until its frame identity is sane
relative to the current playout window and learned offset.
```

## 7. Restart/Election Fix, Clean Log Validation

Status: archived only. The source changes from this section were reverted after
the packet-copy H.264 failover test proved unreliable. Keep this section as a
record of what was tried, not as the current implementation.

Implemented in `fabric_source`:

```text
per-replica generation reset
per-replica control ports
current active owner instead of fixed preferred active owner
offset learning for restarted replicas, including preferred replica A
normalized frame history for offset learning
implausible frame-id guard before frame_slots_/highest_seen_frame_/next_emit_frame_
10 minute default sanity window, still rejecting huge corrupt jumps
```

Clean log-only run on Fedora:

```text
A active, B standby
kill A
B repair/promote/elected active
restart A as standby
A learns offset and becomes eligible
kill B
source promotes A
no "dropped implausible frame id" after widening the guard
```

Key log lines:

```text
fabric_source elected active replica_id=2 previous_active_replica_id=1 normalized_pts=413116320
fabric_source generation changed replica_id=1 generation=1778093299895158970
fabric_source learned standby offset replica_id=1 offset=-10800
fabric_source replica eligible replica_id=1 generation=1778093299895158970 offset=-10800
fabric_source control PROMOTE replica_id=1 normalized_pts=414653520 raw_pts=414664320
```

Earlier TS recording before the final sanity-window adjustment:

```text
/home/jp/out_active_standby_restart_clean.ts
packets: 169
duration: 6.720s
max packet delta: 0.040s
bad gaps >60ms: 0
```

Remaining verification:

```text
Capture a longer TS after the final 10 minute sanity-window change and inspect
the visible A/B overlays through both switch directions. The code path now
passes the log-level restart/election checks.
```

## 8. Packet-Copy H.264 Failover Failure

The later restart/election test was different from the earlier working MP4
proof. It added a harder sequence:

```text
A active, B standby
kill A
B becomes active
restart A as standby with a new generation
kill B
A becomes active again
```

The output was still packet-copy H.264: the consumer muxed packets selected
from two independent producer encoders as if they were one continuous H.264
elementary stream.

Observed files:

```text
working short MP4:
  /home/jp/out_active_standby_working_switch.mp4
  source: /home/jp/out_active_standby_two_url_normalized_2switch.mp4
  packets: 850
  duration: 33.960s
  max packet delta: 0.040s
  bad gaps >60ms: 0

broken TS packet-copy capture:
  /home/jp/out_active_standby_final_direct.ts
```

The broken TS had acceptable packet timing, but ffmpeg reported H.264 decode
corruption:

```text
error while decoding MB 15 2, bytestream -42
corrupt decoded frame
```

Interpretation:

```text
The failure was not just TS container timing. The downstream decoder saw a
single H.264 stream whose packets came from different encoder instances. Even
with g=1, forced-idr=1, repeat_pps=1, and all-intra NVENC settings, switching
between independent encoders can violate decoder expectations at the access
unit / SPS / PPS / IDR boundary.
```

Implications for the MVP:

```text
Do not use the later restart/election code as proof of packet-copy H.264
redundancy. It was reverted.

For packet-copy H.264 redundancy, the source must switch only on a clean IDR
access unit and must ensure the first emitted packet after switch carries the
required decoder headers, especially SPS/PPS/AUD. It may need to drop packets
from the new replica until that boundary.

If that remains unreliable, the safer architecture is to do redundancy before
the final encoder: transport raw/I-frame/image-like records or decode selected
packets at the consumer and have exactly one final encoder. That is not the
same as proving packet-copy encoded-stream failover.
```

## Current Recommendation

Use the last working active-standby example as the baseline:

```text
/home/jp/out_active_standby_working_switch.mp4
```

The restart/rejoin/election experiment should remain archived until the H.264
handoff rule is implemented explicitly.

## 10. NVJPEG Active-Standby, Restart/Rejoin, MPEG-TS Proof

Current NVJPEG transport proof, after the `fabric_source` refactor and stop /
interrupt interface fix.

Graphs:

```text
examples/redundancy/nvjpeg/active_standby_mpegts/consumer_decode_h264_mpegts.avplumber
examples/redundancy/nvjpeg/active_standby_mpegts/producer_a_active.avplumber
examples/redundancy/nvjpeg/active_standby_mpegts/producer_b_standby.avplumber
examples/redundancy/nvjpeg/active_standby_mpegts/producer_a_standby.avplumber
```

Switch sequence:

```text
A starts active.
B starts standby.
kill A -> B repairs/promotes and becomes active.
restart A as controlled standby.
kill B -> A repairs/promotes and becomes active.
```

Output:

```text
/home/jp/out_nvjpeg_complicated_restart_rejoin.ts
remote source: /home/fedora/out_nvjpeg_complicated_restart_rejoin.ts
duration: 53.480s
size: 40M
bitrate: 6195774 bps
video packets: 1337
max packet delta: 0.040s
bad gaps >60ms: 0
ffmpeg decode warnings: none
```

Proof log excerpts:

```text
fabric_source elected active replica_id=2 previous_active_replica_id=1 normalized_pts=1086078720
fabric_source skipped active status-only frame slot replica_id=2 normalized_pts=1086082320
fabric_source generation changed replica_id=1 generation=1778100770414938351
fabric_source replica eligible replica_id=1 generation=1778100770414938351 offset=0
fabric_source elected active replica_id=1 previous_active_replica_id=2 normalized_pts=1088415120
fabric_source skipped active status-only frame slot replica_id=1 normalized_pts=1088418720
```

Shutdown caveat:

```text
The selected media path worked, but the final consumer shutdown aborted after
MJPEG_CUVID_Selected stopped, and producer processes did not exit on SIGINT.
They required SIGKILL cleanup. This is a node shutdown/interface issue, not a
media-continuity failure in the captured TS.
```

More edge-case scenarios worth testing:

```text
1. Standby joins late, then active dies before standby reaches eligibility.
2. Standby sends status with a valid generation but bad/extreme pts.
3. Standby repair cache is too small for the requested playout point.
4. Active pauses without process death, then resumes after standby promotion.
5. Both replicas are alive but one has bursty 500-1500 ms network jitter.
6. Control UDP packets are dropped or duplicated: REPAIR lost, PROMOTE lost,
   PROMOTE delivered twice.
7. Active restarts with the same replica_id while old TCP connection/packets
   are still draining.
8. Source/consumer restarts while both producers keep running.
9. Backpressure stalls the active fabric_sink long enough for false promotion.
10. Codec handoff stress: H.264 packet-copy switching only on IDR/SPS/PPS/AUD
    boundaries, compared against the current safer JPEG-decode-reencode path.
```

## 9. Codec Transport Smoke Tests

These are point-to-point fabric checks, not redundancy failover proofs. They
verify that the fabric packet path can carry both current encoded payloads and
that the receiver can hand them to GPU decoders.

JPEG over fabric TCP:

```text
examples/redundancy/nvjpeg/fabric_tcp/producer.avplumber
examples/redundancy/nvjpeg/fabric_tcp/consumer_decode_null.avplumber
```

Fedora proof run:

```text
nvjpeg_enc quality=95
fabric_sink payload_prefix=ff d8 ff e0 ... JFIF
fabric_sink throughput: about 145-148 Mbps, 30 fps
NVJPEG_ENCODER_LATENCY_SUMMARY:
  min 1.219 ms
  avg 1.551 ms
  p95 2.114 ms
  max 2.470 ms
fabric_source received JPEG packets
decoder selected/opened: mjpeg_cuvid
consumer stats: pix_fmt=cuda, frame_num=398, about 30 fps, no dropped frames
```

H.264 intra over fabric TCP:

```text
examples/redundancy/h264_intra/fabric_tcp/producer.avplumber
examples/redundancy/h264_intra/fabric_tcp/consumer_decode_null.avplumber
```

Fedora proof run:

```text
fabric_sink payload_prefix=00 00 00 01 09 10 ... 00 00 00 01 67
fabric_sink throughput: about 13 Mbps, 30 fps at 1280x720
fabric_source received H.264 packets
decoder selected/opened: h264_cuvid
consumer stats: pix_fmt=cuda, frame_num=390, about 30 fps, no dropped frames
```

Note:

```text
The fabric_source nodes do not yet implement graceful stop, so test shutdown
logs include "Stopping node ... failed". The processes were killed after the
proof window; this did not affect media receive/decode evidence.
```

## 10. Active-Active NVJPEG, Two Switches, MPEG-TS Proof

Graphs:

```text
examples/redundancy/nvjpeg/active_active_mpegts/producer_a.avplumber
examples/redundancy/nvjpeg/active_active_mpegts/producer_b.avplumber
examples/redundancy/nvjpeg/active_active_mpegts/consumer_decode_h264_mpegts.avplumber
```

Switch sequence:

```text
A and B send NVJPEG payloads over fabric TCP.
consumer selects one JPEG timeline.
consumer decodes selected JPEG with mjpeg_cuvid.
consumer encodes one continuous H.264 proof stream to MPEG-TS.

kill A -> B selected
restart A -> kill B -> A selected
```

Output copied locally:

```text
/home/jp/out_nvjpeg_active_active_2switch_h264.ts
/home/jp/out_nvjpeg_active_active_2switch_h264_trim.ts
```

Result:

```text
full file:
  duration: 36.520s
  size: 23,592,960 bytes
  stream: H.264 640x360, yuv420p, MPEG-TS
  packets: 911
  max packet delta: 0.120s
  bad gaps >60ms: 1

trimmed after initial alignment:
  duration: 31.440s
  size: 20,354,572 bytes
  packets: 786
  max packet delta: 0.040s
  bad gaps >60ms: 0
```

Important detail:

```text
Direct packet-copy MJPEG into MPEG-TS did produce a .ts file, but ffprobe saw
the stream as bin_data/private data. For a viewable proof, the consumer must
decode selected JPEG and encode one normal H.264 stream into TS.
```

## 11. Active-Standby NVJPEG, Restart/Rejoin, MPEG-TS Proof

Graphs:

```text
examples/redundancy/nvjpeg/active_standby_mpegts/producer_a_active.avplumber
examples/redundancy/nvjpeg/active_standby_mpegts/producer_b_standby.avplumber
examples/redundancy/nvjpeg/active_standby_mpegts/producer_a_standby.avplumber
examples/redundancy/nvjpeg/active_standby_mpegts/consumer_decode_h264_mpegts.avplumber
```

Switch sequence:

```text
A starts active and sends full NVJPEG payloads.
B starts standby and sends status metadata only.
kill A -> B repairs the requested frame and promotes to full media.
restart A as standby.
kill B -> A repairs/promotes and resumes full media.
consumer decodes selected JPEG with mjpeg_cuvid and writes H.264 MPEG-TS.
```

Output copied locally:

```text
/home/jp/out_nvjpeg_active_standby_restart_rejoin.ts
```

Result:

```text
duration: 40.960s
size: 31M
stream: H.264 640x360, yuv420p, MPEG-TS, 25 fps
bitrate: about 6.2 Mbps
max packet delta: 0.040s
bad gaps >60ms: 0
```

Selector fix discovered by this test:

```text
After promotion, fabric_source could elect the standby but then block on
stale status-only slots from the newly active replica. The selector now sends
a repair request for that active status-only frame and skips it if no payload
arrives, allowing the first promoted media frame to continue the timeline.
```

## 12. Active-Standby H.264 Intra, Restart/Rejoin, MPEG-TS Proof

Graphs:

```text
examples/redundancy/h264_intra/active_standby_mpegts/producer_a_active.avplumber
examples/redundancy/h264_intra/active_standby_mpegts/producer_b_standby.avplumber
examples/redundancy/h264_intra/active_standby_mpegts/producer_a_standby.avplumber
examples/redundancy/h264_intra/active_standby_mpegts/consumer_mpegts.avplumber
```

Switch sequence:

```text
A starts active and sends H.264 intra payloads.
B starts standby and sends status metadata only.
kill A -> B repairs/promotes.
restart A as standby.
kill B -> A repairs/promotes.
consumer packet-relays selected H.264 into MPEG-TS.
```

Output copied locally:

```text
/home/jp/out_h264_active_standby_restart_rejoin.ts
```

Result:

```text
duration: 40.920s
size: 9.0M
stream: H.264 640x360, MPEG-TS
bitrate: about 1.85 Mbps
max packet delta: 0.120s
bad gaps >60ms: 2
```

Interpretation:

```text
The fabric active-standby repair path now survives the same restart/rejoin
scenario for H.264 too. Compared with NVJPEG, the packet timeline still shows
two 120 ms gaps at the switches, so H.264 is functional but not as clean as
the NVJPEG proof for this failover mode.
```

## 13. Encoder Latency, NBA Source, H.264 Intra QP8 vs NVJPEG Q99

Historical measurement note from the redundancy spike. The standalone latency
probe graphs are intentionally not part of this clean fabric branch.

Common source:

```text
/home/fedora/nba.mp4
1920x1080
30000/1001 fps
NVDEC CUDA/NV12 before encode
null sink after encode
```

H.264 intra QP8:

```text
encoder: h264_nvenc
settings: all-IDR, QP8, p1, ull, no B-frames, no lookahead
samples: 300 after 30 warmup frames
encode_ms_min: 5.870
encode_ms_avg: 7.027
encode_ms_max: 7.351
observed encoded bitrate after warmup: about 207 Mbps
```

NVJPEG Q99:

```text
encoder: nvjpeg_enc
settings: quality 99, optimized_huffman=true
samples: 300 after 30 warmup frames
encode_ms_min: 1.196
encode_ms_avg: 1.288
encode_ms_p95: 1.332
encode_ms_max: 1.348
avg_mbit: 231.688
```

Interpretation:

```text
At near-lossless settings on the NBA source, NVJPEG Q99 encoded about 5.5x
faster than H.264 intra QP8 in this probe, but used slightly more payload
bitrate. H.264 intra remains much more container/player friendly; NVJPEG is
lower latency and simpler for frame-granular failover.
```

## 14. Active-Active NVJPEG, Two SRT Inputs, No Extra Jitter

Graphs:

```text
examples/redundancy/nvjpeg/active_active_mpegts/producer_a.avplumber
examples/redundancy/nvjpeg/active_active_mpegts/producer_b.avplumber
examples/redundancy/nvjpeg/active_active_mpegts/consumer_decode_h264_mpegts.avplumber
```

Inputs:

```text
A: srt://ingest-1-qa.tellyo.com:9000?streamid=output/live/sync164bdf34f
B: srt://ingest-1-qa.tellyo.com:9000?streamid=output/live/sync2e446f12e
```

Output copied locally:

```text
/home/jp/out_active_active_simul_start.ts
```

Result:

```text
duration: 33.200s
size: 21M
bitrate: about 5.18 Mbps
video packets: 830
max packet delta: 0.040s
bad gaps >60ms: 0
ffmpeg decode warnings: none
```

Observed switch:

```text
last A frames around switch: SYNC 1 090, SYNC 1 091, +0s
first B frames after switch: SYNC 2 044, SYNC 2 045, +2s
```

Interpretation:

```text
The packet timeline is clean, but the visual switch is not frame-equivalent.
B's first raw pts was 180000 ticks behind A, exactly 2 seconds at
90 kHz. The selector learned a wallclock offset and emitted continuous PTS,
but it selected B content that did not represent the same source frame.

This is not a transport loss or MPEG-TS/H.264 decode problem. It proves that
active-active redundancy needs a shared frame identity contract, not only a
receiver-wallclock PTS offset. For a true frame-perfect switch, the two
producers must either ingest the same frame identity or a pre-fabric node must
rewrite pts from a trusted shared source identity.
```

Low-cost follow-up:

```text
Add a strict identity mode to redundancy_selector/fabric_source:
- wallclock mode keeps today's behavior and can bridge fixed source PTS skew.
- strict frame-id mode requires the standby to have the exact next pts.
- mismatched standby content should freeze/drop by policy instead of silently
  switching to a visually different frame.
```

## 15. Active-Active NVJPEG, Strict Frame Identity

Change:

```text
redundancy_selector alignment_mode: strict_frame_identity
```

Output copied locally:

```text
/home/jp/out_active_active_strict_identity.ts
/home/jp/out_active_active_strict_identity_2switch.ts
```

Single-switch result:

```text
duration: 24.320s
size: 15M
bitrate: about 5.17 Mbps
video packets: 608
max packet delta: 0.040s
bad gaps >60ms: 0
ffmpeg decode warnings: none
```

Double-switch result:

```text
sequence: A active -> kill A -> B -> restart A -> kill B -> A
duration: 49.400s
size: 31M
bitrate: about 5.18 Mbps
video packets: 1235
max packet delta: 0.040s
bad gaps >60ms: 0
ffmpeg decode warnings: none
```

Observed switch:

```text
before strict mode: A SYNC 1 091 -> B SYNC 2 044
after strict mode, one switch: A SYNC 1 675 -> B SYNC 2 676
after strict mode, double switch:
  A -> B: A SYNC 1 436 -> B SYNC 2 437
  B -> A: B SYNC 2 699 -> A SYNC 1 700
```

Interpretation:

```text
Strict mode fixes the visible frame jump for the offset SRT pair by refusing
to normalize B with receiver wallclock. While A is active, B frames with older
raw pts values are dropped as stale. After A is killed, the selector
waits until B reaches the exact next pts and then selects B.

In a live output this can appear as a freeze equal to the inter-source skew.
In the MPEG-TS proof file, emitted packet PTS remains continuous because the
file only contains the selected frames after B reaches the required identity.

The double-switch run exposed and fixed two implementation issues:
- initial bogus `pts=0` packets could poison `frame_step` and block
  strict playout; the selector now waits for a plausible frame step before
  starting strict output.
- `fabric_packet_ingress` needed stop/interrupt support so consumer shutdown
  can finalize the MPEG-TS proof cleanly.
```
