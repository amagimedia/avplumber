# Fabric Redundancy Demos

This directory is split by payload codec:

```text
examples/redundancy/h264_intra/
examples/redundancy/nvjpeg/
examples/redundancy/hevc_lossless/
examples/redundancy/jpeg2000_lossless/
examples/redundancy/prores4444/
```

Each codec folder keeps its own active-active, active-standby, and
point-to-point fabric demos where available.

## H.264 Intra

RTMP demos:

```text
examples/redundancy/h264_intra/active_active_sync_bars_rtmp/
examples/redundancy/h264_intra/active_standby_sync_bars_rtmp/
```

MPEG-TS restart/rejoin proof:

```text
examples/redundancy/h264_intra/active_standby_mpegts/
```

Point-to-point smoke test:

```text
examples/redundancy/h264_intra/fabric_tcp/
```

## NVJPEG

MPEG-TS restart/rejoin proofs:

```text
examples/redundancy/nvjpeg/active_active_mpegts/
examples/redundancy/nvjpeg/active_standby_mpegts/
```

Point-to-point smoke test:

```text
examples/redundancy/nvjpeg/fabric_tcp/
```

## Lossless HEVC Intra

Every frame intra and bit-exact, encoded by NVENC in constant-QP mode at qp=0.

Point-to-point smoke test:

```text
examples/redundancy/hevc_lossless/fabric_tcp/
```

Point-to-point key/fill smoke test (`alpha_src`/`alpha_dst`, two decoders, no GPU alpha):

```text
examples/redundancy/hevc_lossless/fabric_tcp_alpha/
```

## Lossless JPEG 2000

Bit-exact 4:4:4 on the GPU through nvJPEG2000, so it needs a build with
`HAVE_CUDA=1 HAVE_NVJPEG2K=1`. The `nvjpeg2k_enc` node can also emit HTJ2K
(`high_throughput`), which is the variant fast enough for video rates; see
`doc/NODES.md` for the decoder-compatibility caveat.

Point-to-point smoke test:

```text
examples/redundancy/jpeg2000_lossless/fabric_tcp/
```

## ProRes 4444 (single-stream alpha)

ProRes 4444 carries alpha inside the payload, so one stream is the whole picture - no key/fill
pair, no merge step, and the stock libavcodec `prores` decoder returns the alpha plane.
`codec` accepts `prores` / `prores4444`, which travel as one wire code (`CODEC_PRORES`).
`prores_ks` encodes on the CPU (`{"profile": "4444"}` over a `yuva444p10le` input);
[ProresNV](https://github.com/siml1169/ProresNV)'s CUDA `prores_nv` produces the same bitstream
and can replace it later without a protocol change.

Point-to-point smoke test, which writes the received payloads to a MOV so the alpha plane can be
checked offline:

```text
examples/redundancy/prores4444/fabric_tcp/
```

Measured on the QA host, 640x360p30, 180 frames: the sink sent 180 frames (~95.6 MB, ~124 Mb/s of
a noise-heavy alpha-ramp test pattern, an upper bound; real footage is much smaller), the consumer
decoded/wrote 180 of 180 with 0 dropped, and the alpha ramp the producer stamped arrived intact
(0..65407 across the frame). ProRes is several times the bytes of lossless key/fill and is
near-lossless rather than bit-exact, so pick it when a single alpha-bearing stream matters more
than bandwidth or exactness. Decoding is CPU only - no GPU ProRes decoder exists.

## FFV1 (lossless, CPU, single-stream alpha)

`codec` accepts `ffv1`, which travels as its own wire code (6) and maps to libavcodec's
`AV_CODEC_ID_FFV1`. The encoder is CPU only and lossless, and it carries alpha in the payload -
`yuva444p`, `yuva422p`, `yuva420p` and their deeper variants are all supported - so one stream is
the whole picture, like ProRes 4444 and unlike key/fill. Nothing decodes it on the GPU.

The encoder must be told to write a self-describing bitstream:

```text
node.add { "type": "enc_video", "src": "v_frames", "dst": "v_enc", "group": "tx", "name": "Ffv1Enc", "codec": "ffv1", "options": { "level": 1, "g": 1 } }
```

`level: 1` is not optional. From FFV1 bitstream version 2 on - libavcodec's default - the
configuration record (coder, quantisation tables, bit depth) is written once into
`AVCodecContext.extradata` instead of the frames, and the fabric wire carries media packets only
with `fabric_source` stripping extradata, so the consumer rejects every frame
(`read_quant_table error`, then an invalid-data error from the decoder). Version 1 repeats the
record in-band on every keyframe, which costs a little size per frame and rules out slice threading
and the options that need version 2 or 3, in exchange for packets that decode on their own.
`slicecrc` is one of those: it raises the version to 3 without an error, so it must stay off with
`level: 1` or the stream goes back to carrying extradata.

`g: 1` (or `0`) makes every frame a keyframe, as with every other fabric codec: version 1 writes
its configuration only on keyframes, and `fabric_sink` requires keyframes by default.

The consumer's `dec_video` has to name the decoder - `"codec_map": { "ffv1": "ffv1" }` or
`"codec": "ffv1"` - like the HEVC and H.264 fabric consumers do with their `codec_map`. A fabric
stream is codec-only metadata, so the node has no demuxed stream to resolve a default decoder from
and fails at creation with "Action impossible with given codec context direction".

The sink rejects a stream whose encoder parameters report `level` 2 or more, or carry extradata,
but it only sees them when the producer graph also feeds a muxer (`IEncoder::setOutput` is what
populates them), so the options above are the contract.

Measured on the same QA host as the tables above, 900 frames of the same 1080p camera footage, CPU
decode and CPU encode, the consumer decoding on loopback over the `tcp` provider, so both processes
share the host's four cores:

| format | KiB/frame | ratio | fps | Mb/s |
| --- | --- | --- | --- | --- |
| 8-bit 4:4:4 | 1039 | 5.85:1 | 8.0 | 68.5 |
| 8-bit 4:2:0 | 747 | 4.07:1 | 12.4 | 75.7 |

Level 1 costs nothing in bytes: the same frames encoded at level 3 come out at 1035.5 KiB/frame
(4:4:4) and 745.6 (4:2:0), so level 1 was 0.3-0.4% smaller here - inside the noise either way. It
costs encode time instead: a standalone `ffmpeg` encode of 900 4:4:4 frames takes 80.6 s at level 1
against 77.4 s at level 3 (52.2 s against 49.1 s at 4:2:0), and the fabric pipeline runs at ~72% of
that standalone rate because the consumer's decode shares the cores. On their own, the stages reach
23.0 fps decode-only and 11.2 fps decode+encode at 4:4:4 (34.8 and 17.2 at 4:2:0).

That is not a real-time 25p path at 1080p on four cores. Every other row above assumes NVENC or
NVDEC; FFV1 level 1 is the CPU-only lossless option, at 8-12 fps fit for offline or low-rate use, or
for hosts with more cores. A 640x360 `yuva444p` alpha ramp round trip carried its 180 frames in one
stream at 27.1 KiB/frame and the received alpha plane was the ramp the producer stamped (0..254).

## Alpha (key/fill)

A frame can travel as two records in the same fabric session: the fill (the picture) and the alpha
(its key). Both carry the same `stream_id`, `replica_id`, `generation`, PTS, timebase and geometry,
and the media header's `plane` field says which one a record is, so a consumer pairs them by PTS.
A producer without `alpha_src` sends `plane=fill` only, as it did before the plane field existed.

Producer graph:

```text
video -> fill encoder -> fabric_sink.src
alpha -> key encoder  -> fabric_sink.alpha_src
```

Alpha is a normal encoded stream - encode the key plane as `gray` (or whatever the consumer's
decoder expects) and hand the packets to `alpha_src`. `alpha_codec` defaults to `codec` and
`alpha_pixel_format` to `gray`; `alpha_real_pixel_format` describes the software layout behind a
`cuda` alpha plane. The key takes the fill's geometry, and an alpha edge that advertises a
different size fails the node.

Consumer graph, one chain per plane:

```text
fabric_source.dst       -> selector -> fabric_unpack -> decoder -> fill frames
fabric_source.alpha_dst -> selector -> fabric_unpack -> decoder -> key frames
```

`alpha_dst` is optional on `fabric_source`, and each edge carries its own stream metadata
(`alpha_codec`, `alpha_pixel_format`) so `fabric_unpack` describes the key correctly. A record with
`plane=alpha` arriving at a `fabric_source` without `alpha_dst` is an error, not a dropped packet.
Each plane needs its own `redundancy_selector`: a selector that sees both planes on its inputs
throws. Recombining decoded fill and key frames into a `yuva*` frame is a compositing step that has
no node in this tree yet.

Both halves of a demo run keep running after the source ends: the group stays alive, and a
`fabric_source` waiting for the next record does not notice that its peer process exited. Run the
pair under `timeout` (or stop the groups from the control port) and compare the summary counters,
as the smoke tests do.

A `fabric_source` can end its own session when the producer goes quiet: set `idle_timeout_ms` and
the node sends EOF on every output after that many milliseconds without a record, so the decoders
and the rest of the consumer graph finish normally. The timer starts at the first record, so a
consumer started before its producer still waits. Leaving `idle_timeout_ms` at its default `0`
keeps the receiver waiting forever, which is what a consumer needs to survive a producer restart -
the same port accepts the returning producer without any reconnect step. A provider that reports
the peer's connection loss is no longer fatal: `fabric_source` logs `peer closed` and keeps its
endpoint, so either choice works. That report is best effort: the `tcp` provider does not surface a
clean or idle close on an RDM endpoint, and it does not enable TCP keepalive for RDM peer
connections, so a half-open producer stays invisible (see "Receiver Lifetime" in
`doc/specs/fabric-sink.md`). Record cadence is the liveness signal instead: `idle_timeout_ms` for
the source, and the selector's `active_timeout_ms` for failover. A standby stays eligible only while
its input keeps producing status records, so keep that input flowing. If the new producer starts
before the old one's records have drained, give it a new `generation` (or leave `generation: 0`) so
the selector resets instead of dropping the new frames.

The consumer process only exits when its group stops, so pair `idle_timeout_ms` with a group policy
that stops on a finished node (`auto_restart: "panic"` on the nodes, as the non-alpha consumer
does); otherwise the finished chains leave the process running and the outer `timeout` still
applies. `fabric_unpack` and `redundancy_selector` forward the EOF marker to their outputs, so a
chain that ends on the source's EOF ends all the way to the null sink.

NVENC can encode an HEVC alpha layer (base layer plus an auxiliary layer in one access unit), but no
FFmpeg release exposes it: `libavcodec/nvenc.c` has no alpha-layer support in 7.1, 8.1, 9.1 or current
master, so `enc_video` cannot enable it. A direct NVENC encoder plus a consumer-side splitter that
separates the layer-1 NALs and rewrites their layer ids would be needed, and the feature only accepts
8-bit NV12/ARGB/ABGR input. That is why key/fill is the alpha path in this tree.

## Codec Mode Performance

Measured on an RTX 4000 Ada, driver 615.71.09, nvJPEG2000 0.12.0, through the producer graph
these examples use (NVDEC -> `scale_cuda` -> encoder), 1800 frames of 1080p camera footage,
every frame intra. `fps` is end-to-end pipeline throughput, so it is what the host sustains
rather than an encoder-only figure; rows marked `~` were limited by decode of the test source
and are floors. "NVDEC" is whether the consumer can decode the result on the GPU.

### Lossless

| `codec` | wire | format | KiB/frame | ratio | fps | Mb/s p50 | p59.94 | p60 | NVDEC |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `hevc_intra` | 3 | 8-bit 4:2:0 | 974 | 3.12:1 | 222 | 399 | 478 | 479 | yes |
| `h264_intra` | 1 | 8-bit 4:2:0 | 999 | 3.04:1 | 352 | 409 | 490 | 491 | yes |
| `jpeg2000` HT | 4 | 8-bit 4:4:4 | 1102 | 5.51:1 | 577 | 451 | 541 | 542 | `nvjpeg2k_dec` |
| `jpeg2000` Part 1 | 4 | 8-bit 4:4:4 | 970 | 6.26:1 | 60 | 397 | 476 | 477 | `nvjpeg2k_dec` |
| `hevc_intra` | 3 | 8-bit 4:4:4 | 1358 | 4.47:1 | 152 | 556 | 667 | 668 | yes |
| `h264_intra` | 1 | 8-bit 4:4:4 | 1453 | 4.18:1 | 255 | 595 | 713 | 714 | **no** |
| `hevc_intra` | 3 | 10-bit 4:2:0 | 1662 | 2.28:1 | 152~ | 681 | 816 | 817 | yes |
| `hevc_intra` | 3 | 10-bit 4:4:4 | 2420 | 3.14:1 | 98 | 991 | 1188 | 1190 | yes |
| `jpeg2000` Part 1 | 4 | 10-bit 4:4:4 | 4158 | 1.83:1 | 24 | 1703 | 2042 | 2044 | `nvjpeg2k_dec` |
| `jpeg2000` HT | 4 | 10-bit 4:4:4 | 5445 | 1.39:1 | 92~ | 2230 | 2674 | 2676 | `nvjpeg2k_dec` |

`jpeg` has no lossless mode: nvJPEG provides lossless JPEG decoding only, so `nvjpeg_enc`
rejects `lossless: true`. Lossless `h264_intra` is bit-exact in both chroma formats, but NVENC
signals High 4:4:4 Predictive even for 4:2:0 content, and NVDEC cannot decode its 4:4:4 output at
all - so lossless 4:4:4 over H.264 is a GPU one-way street. Ratios are against packed raw at the
content's own depth. Bit-exactness was verified by hash, all three planes, across
`h264_intra`, `hevc_intra` and both JPEG 2000 coders.

### Lossy, matched quality

Bitrate at equal median luma PSNR against a verified-lossless reference of the same encoder
input. Luma only, so these numbers do not capture 4:4:4's chroma advantage and cannot be compared
across chroma formats.

| `codec` | wire | format | KiB/f @44 dB | @48 dB | Mb/s p60 @48 dB | fps |
| --- | --- | --- | --- | --- | --- | --- |
| `hevc_intra` | 3 | 8-bit 4:2:0 | 77 | 142 | 70 | 414 |
| `h264_intra` | 1 | 8-bit 4:2:0 | 92 | 174 | 86 | 346 |
| `hevc_intra` | 3 | 8-bit 4:4:4 | 68 | 140 | 69 | 347 |
| `h264_intra` | 1 | 8-bit 4:4:4 | 113 | 204 | 100 | 340 |
| `jpeg2000` Part 1 | 4 | 8-bit 4:4:4 | 125 | 224 | 110 | 39 |
| `jpeg` (nvJPEG) | 2 | 8-bit 4:4:4 | 196 | 282 | 138 | 724~ |
| `hevc_intra` | 3 | 10-bit 4:2:0 | 69 | 144 | 71 | 368-419 |

`jpeg2000` lossy is Part 1 only - nvJPEG2000 has no target-PSNR rate control for HT codestreams.

### Why `hevc_intra` is the default

`codec` defaults to `hevc_intra`. It is the smallest at matched quality in every lossy row above,
it wins lossless 4:2:0 on size and lossless 10-bit 4:4:4 outright, it is the only mode that
decodes on NVDEC in every format tested, and it covers 8-bit and 10-bit from one setting.

Reasons to override it:

- `jpeg2000` with `high_throughput` for lossless 8-bit 4:4:4 - 19% smaller than `hevc_intra`
  and 3.8x the throughput, the one case where it wins both axes. Part 1 compresses best of all
  (6.26:1) but at 60 fps cannot hold 1080p59.94 or p60.
- `h264_intra` where decoder reach matters more than 6% of bitrate, or where its extra
  throughput headroom is needed. Not for lossless 4:4:4 (see above).
- `jpeg` when throughput dominates and bitrate does not; it is 2-3x `hevc_intra` at equal luma
  PSNR.
- `ffv1` when no GPU encoder is available and bit-exact lossless (including alpha in one stream) is
  wanted; keep it at bitstream `level: 1`, see "FFV1" above.

Both `fabric_sink` and `fabric_source` default to `hevc_intra`, so they stay consistent when
neither sets `codec`. A producer and consumer that disagree are caught at the consumer: the wire
codec is authoritative and an unknown code is rejected rather than decoded as the configured
default.

See `TEST_CASES.md` for the latest measured failover results.
See `EDGE_CASES.md` for the next failure scenarios to test.
