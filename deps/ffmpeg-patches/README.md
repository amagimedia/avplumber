# FFmpeg Patch Stack

This directory contains the public FFmpeg patch stack used by avplumber's CUDA
composition and media-input workflows.

## Base

- Upstream repository: `https://github.com/FFmpeg/FFmpeg`
- Upstream tag: `n7.1.5`
- Upstream commit: `3a0867c2bfda4a4d4309ca1a8cbdc6175e67f587`
- Expected patched tree: `52361f7251069ef74fbb41460e6e1b65d6f9947c`

## Series

The seven patches are ordered by filename and grouped by feature rather than by
the chronology of incomplete ports and follow-up fixes:

1. `0001-swscale-aarch64-argb-yuva420p.patch` — AArch64 fast color conversion.
2. `0002-avfilter-cuda-composition-suite.patch` — CUDA pad, convert, crop,
   overlay, overlay-many, scale edge handling, transitions, and procedural
   wipes.
3. `0003-avfilter-npp-cuda13-compat.patch` — CUDA 13 NPP compatibility.
4. `0004-avcodec-nvdec-intra.patch` — NVDEC intra-only stream handling.
5. `0005-avformat-rtp-rfc4175.patch` — RFC 4175 4:2:0 and incomplete-frame
   handling.
6. `0006-avdevice-v4l2-compat.patch` — V4L2 timestamp compatibility.
7. `0007-avdevice-ndi-v5.patch` — NDI v5 device registration and documentation.

Each patch message lists the original exported FFmpeg commits it replaces.

The old FFmpeg `af_whisper` port is intentionally absent. Speech-to-text belongs
in an AVPlumber node and is not part of this FFmpeg variant.

## Apply

```bash
git clone --branch n7.1.5 --depth 1 \
  https://github.com/FFmpeg/FFmpeg clean-ffmpeg
git -C clean-ffmpeg config user.name "patch application"
git -C clean-ffmpeg config user.email "patch-application@local"
git -C clean-ffmpeg am /path/to/avplumber/deps/ffmpeg-patches/*.patch
```

## Verify

Run the verifier with any FFmpeg Git checkout that contains the documented base
commit. It creates and removes an isolated temporary worktree; it does not alter
the checkout's active branch:

```bash
deps/ffmpeg-patches/verify.sh /path/to/FFmpeg
```

Verification succeeds only when all seven patches apply and produce the exact
expected Git tree. Runtime validation is provided by `demos/cuda-overlay` and
`demos/mixer`, whose Dockerfiles build this series against FFmpeg `n7.1.5`.
