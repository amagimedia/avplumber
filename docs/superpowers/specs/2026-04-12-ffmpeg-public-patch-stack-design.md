# FFmpeg Public Patch Stack And `/usr/local` Validation

## Goal

Produce a public FFmpeg patch stack in the avplumber repository that reproduces the full remote FFmpeg branch delta on top of upstream FFmpeg tag `n7.1.3`, then validate that patched FFmpeg builds and installs cleanly to `/usr/local` on the SSH machine with feature parity to the current `/apps/ffmpeg` install.

This phase ends with FFmpeg only. It does not include Docker work or avplumber runtime testing.

## Scope

In scope:

- export the full ordered commit delta from remote branch `n7.1.3-tellyodev` relative to upstream tag `n7.1.3`
- store the exported patches in this repo under `deps/ffmpeg-patches/`
- document the patch base and application workflow in `deps/ffmpeg-patches/README.md`
- keep the current remote experiment tree at `/home/fedora/ffmpeg` untouched
- create and use a separate validation checkout at `/home/fedora/clean-ffmpeg`
- apply all exported patches to a clean public FFmpeg `n7.1.3` checkout
- build and install the patched FFmpeg to `/usr/local`
- verify that the `/usr/local` FFmpeg build exposes the same custom filters and other required custom features as the existing `/apps/ffmpeg` build

Out of scope:

- writing any Dockerfile
- changing existing `Dockerfile` or `Dockerfile.fedora`
- switching avplumber itself to `/usr/local`
- running avplumber examples as part of this phase

## Constraints

- all patches must apply cleanly to a fresh public FFmpeg `n7.1.3` checkout
- `/home/fedora/ffmpeg` remains the known-good reference tree and must not be modified during validation
- the deliverable is incomplete if the `/usr/local` build is missing any custom FFmpeg feature relied on by the current `/apps/ffmpeg`
- after FFmpeg is complete, stop and ask the user whether to proceed with avplumber testing

## Source Of Truth

Remote source branch:

- repository checkout: `/home/fedora/ffmpeg`
- branch: `n7.1.3-tellyodev`
- upstream base tag: `n7.1.3`

Export target in avplumber repo:

- `deps/ffmpeg-patches/`

Recommended contents:

- numbered patch series generated from the remote FFmpeg branch
- `README.md` with:
  - upstream base tag `n7.1.3`
  - source branch name `n7.1.3-tellyodev`
  - patch application instructions
  - validation directory `/home/fedora/clean-ffmpeg`
  - install prefix `/usr/local`

## Recommended Approach

Preserve the original commit series rather than squashing.

Why:

- keeps intent and change boundaries visible
- makes future review and rebasing easier
- matches the future public build flow of `clone public FFmpeg -> apply patch series -> build`

## Validation Checkout

Use a clean test tree on the SSH machine:

- checkout path: `/home/fedora/clean-ffmpeg`
- source: public FFmpeg at tag `n7.1.3`

Validation steps:

1. create a clean public FFmpeg checkout at `/home/fedora/clean-ffmpeg`
2. apply the full patch series from `deps/ffmpeg-patches/`
3. configure FFmpeg with prefix `/usr/local`
4. build and install FFmpeg
5. compare the resulting `/usr/local` FFmpeg feature surface against `/apps/ffmpeg`

## FFmpeg Build Contract

Use the exact host configure line supplied by the user, with `FFMPEG_PREFIX=/usr/local` during validation:

```bash
PKG_CONFIG_PATH="/usr/local/lib/pkgconfig:${FFMPEG_PREFIX}/lib/pkgconfig" ./configure --prefix="${FFMPEG_PREFIX}" --libdir="${FFMPEG_PREFIX}/lib" --incdir="${FFMPEG_PREFIX}/include" --pkgconfigdir="${FFMPEG_PREFIX}/lib/pkgconfig" --enable-gpl --enable-nonfree --extra-ldflags=-lm --extra-ldflags=-ldl --enable-pthreads --extra-libs=-lpthread --enable-encoders --enable-decoders --enable-avfilter --enable-muxers --enable-demuxers --enable-parsers --enable-x86asm --disable-debug --disable-ffplay --enable-ffprobe --disable-libx264 --disable-libx265 --disable-libaom --disable-libvpx --extra-cflags=-I/usr/local/cuda-13.0/targets/x86_64-linux/include --extra-ldflags=-L/usr/local/cuda-13.0/targets/x86_64-linux/lib --enable-ffnvcodec --enable-cuda-nvcc --enable-cuda --disable-libvpl --enable-cuvid --enable-nvdec --enable-nvenc --disable-libsrt --disable-libvmaf --enable-shared --disable-doc --disable-htmlpages --disable-manpages --disable-podpages --disable-txtpages --enable-libwebp --enable-gnutls --enable-bsf=scte35ptsadjust --enable-bsf=h264_mp4toannexb --enable-bsf=h264_redundant_pps --enable-bsf=extract_extradata --enable-bsf=aac_adtstoasc --enable-protocol=file --enable-protocol=udp --enable-protocol=pipe --enable-protocol=tls --enable-protocol=http --enable-protocol=https --enable-protocol=rtmp --enable-protocol=rtmps --enable-protocol=tcp --enable-protocol=fd --enable-protocol=crypto --enable-protocol=tee --enable-protocol=concat --enable-protocol=hls --enable-protocol=rtp --enable-protocol=rtsp --enable-muxer=flv --enable-muxer=rtp_mpegts --enable-muxer=mpegts --enable-muxer=mp4 --enable-muxer=matroska --enable-demuxer=mpegts --enable-demuxer=mov --enable-demuxer=matroska --enable-demuxer=flv --enable-demuxer=aac --enable-filter=aformat --enable-filter=aresample --enable-encoder=aac --enable-encoder=ac3 --enable-encoder=mp2 --enable-encoder=mp3 --enable-encoder=opus --enable-decoder=aac --enable-decoder=h264 --enable-decoder=h265 --enable-decoder=ac3 --enable-decoder=mp2 --enable-decoder=mp3 --enable-decoder=opus --enable-parser=aac --enable-parser=h264 --enable-parser=hevc --enable-parser=h265 --enable-parser=ac3 --enable-parser=mp3 --enable-parser=mp2 --enable-parser=opus --nvcc=/usr/local/cuda-13.0/bin/nvcc
```

This configure contract is part of the deliverable. Any later Docker work must reproduce it rather than invent a new FFmpeg configuration.

## Feature Parity Checks

The `/usr/local` install must be checked against `/apps/ffmpeg` for at least:

- custom filters
- enabled protocols
- enabled bitstream filters
- build configuration summary
- any custom CUDA or neural-net-adjacent functionality required by the current avplumber neural-net workflow

If parity is missing, FFmpeg is not done.

## Failure Handling

- If patch export omits commits or numbering is unstable, regenerate the series before validation.
- If patch application fails on clean public `n7.1.3`, fix the patch set before attempting any build.
- If the patched build installs but does not match `/apps/ffmpeg` feature surface, continue working on FFmpeg only.
- Do not start Docker work until the `/usr/local` validation result is accepted.

## Deliverables

- `deps/ffmpeg-patches/` full ordered patch series
- `deps/ffmpeg-patches/README.md`
- validated patched FFmpeg installed to `/usr/local` on the SSH host
- feature parity evidence comparing `/usr/local` to `/apps/ffmpeg`
- explicit pause for user confirmation before any avplumber testing

## Self-Review

- scope is limited to FFmpeg patch export and `/usr/local` validation
- the private remote experiment tree is explicitly protected
- the base tag, source branch, validation path, and install prefix are explicit
- the exact configure line is captured verbatim
- Docker and avplumber testing are explicitly deferred
