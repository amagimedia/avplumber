# Fabric Redundancy Demos

This directory is split by payload codec:

```text
examples/redundancy/h264_intra/
examples/redundancy/nvjpeg/
examples/redundancy/hevc_lossless/
examples/redundancy/jpeg2000_lossless/
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

## Lossless JPEG 2000

Bit-exact 4:4:4 on the GPU through nvJPEG2000, so it needs a build with
`HAVE_CUDA=1 HAVE_NVJPEG2K=1`. The `nvjpeg2k_enc` node can also emit HTJ2K
(`high_throughput`), which is the variant fast enough for video rates; see
`doc/NODES.md` for the decoder-compatibility caveat.

Point-to-point smoke test:

```text
examples/redundancy/jpeg2000_lossless/fabric_tcp/
```

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

Both `fabric_sink` and `fabric_source` default to `hevc_intra`, so they stay consistent when
neither sets `codec`. A producer and consumer that disagree are caught at the consumer: the wire
codec is authoritative and an unknown code is rejected rather than decoded as the configured
default.

See `TEST_CASES.md` for the latest measured failover results.
See `EDGE_CASES.md` for the next failure scenarios to test.
