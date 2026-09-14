# FFmpeg Patch Stack

This directory contains the public FFmpeg patch stack used by avplumber's CUDA
composition and media-input workflows.

## Base

- Upstream repository: `https://github.com/FFmpeg/FFmpeg`
- Upstream tag: `n7.1.5`
- Upstream commit: `3a0867c2bfda4a4d4309ca1a8cbdc6175e67f587`
- Expected patched tree: `f4eebb6a255d1f4b76eb795885102d1cfea8cd08`
  (`52361f7251069ef74fbb41460e6e1b65d6f9947c` for the 0001-0007 subset)

## Series

The eight patches are ordered by filename and grouped by feature rather than by
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
8. `0008-avformat-libmxl-demuxer-muxer.patch` — [MXL](https://github.com/dmf-mxl/mxl)
   demuxer, muxer, URI parser, monitoring/diag interface, and
   `--enable-libmxl` configure glue. Squashed from 32 upstream commits on
   `cbcrc/FFmpeg` branch `dmf-mxl/master` (pinned at `5c5d593`); the MXL
   SDK is pinned to `v1.1.0-beta-1`. Consumed by `demos/mxl/Dockerfile`.

Each patch message lists the original exported FFmpeg commits it replaces.

### Regenerating `0008-avformat-libmxl-demuxer-muxer.patch`

Producing the patch (Docker only — works on Linux and macOS hosts):

```bash
docker build -f deps/ffmpeg-patches/Dockerfile.mkpatch \
    -t avplumber-mxl-mkpatch:local deps/ffmpeg-patches
docker run --rm \
    -v "$PWD/deps/ffmpeg-patches:/out" \
    -v "$PWD/deps/ffmpeg-patches:/patches:ro" \
    avplumber-mxl-mkpatch:local
```

On success the container writes:

* `deps/ffmpeg-patches/0008-avformat-libmxl-demuxer-muxer.patch`
* `deps/ffmpeg-patches/expected-tree.txt` — paste this hash into the
  "Expected patched tree" line above.

Then run the verifier to confirm the tree matches:

```bash
deps/ffmpeg-patches/verify.sh /path/to/any/FFmpeg
```

If the cherry-pick hits conflicts (usually `configure`,
`libavformat/allformats.c`, or `libavformat/Makefile`), the container
exits with instructions. Re-run it with `--entrypoint bash` to finish
by hand:

```bash
docker run --rm -it \
    -v "$PWD/deps/ffmpeg-patches:/out" \
    -v "$PWD/deps/ffmpeg-patches:/patches:ro" \
    --entrypoint bash avplumber-mxl-mkpatch:local
# inside:  /usr/local/bin/mkpatch-0008-mxl
# fix conflicts under /tmp/ffmpeg-mxl-build/ffmpeg, then:
#   git cherry-pick --continue
#   /usr/local/bin/mkpatch-finish
```

Once landed, remove this "Pending" subsection and add the patch to the
numbered list. The `demos/mxl/Dockerfile` startup check
(`ffmpeg -demuxers | grep mxl`) will then pass and the demo becomes
runnable end-to-end.

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
