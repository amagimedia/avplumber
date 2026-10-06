# FFmpeg patch series

`apply.sh` selects the series from the upstream checkout. Both series include opt-in pinned CUDA upload and GPU v210 unpacking. The experimental `9/` series targets the exact development revision
`98e92563a3b60dbf6d370fd3491d7f896398e4c1` (`master-98e92563`), with libavcodec 63,
libavfilter 12 and libavutil 61. It does **not** apply to release n9.0.2; that
release also lacks the opaque CUDA array decode path being evaluated here.
Pin the commit, not the moving `master` branch.

```bash
deps/ffmpeg/apply.sh <ffmpeg-checkout>
deps/ffmpeg/verify.sh master-98e92563 <ffmpeg-repo>
```

Build the experimental revision with nv-codec-headers **n13.1.15.0**
(`0a6fba9a2820628b8103464f4c8753ee05838baa`) to enable its SDK 13.1 CUARRAY path.
The existing Docker defaults remain FFmpeg 8.1 / headers 13.0.19.0. A revision
hash needs an explicit fetch/checkout; `git clone --branch` accepts a branch
or tag, not a commit hash. Rebuild avcpp, the AVP binary and Python module
against the selected library ABI, and keep runtime library selection consistent.
The Fedora mixer Dockerfile accepts `--build-arg FFMPEG_TAG=master-98e92563`
to fetch both pinned revisions; the default remains `n8.1`.

## Development-series differences

All seven custom CUDA filters remain: `convert_cuda`, `crop_cuda`,
`overlay_many_cuda`, `pad_cuda`, `transition_cuda` (including 10-bit and dip),
`tonemap_cuda` and `band_blur_cuda`. The RFC 4175, V4L2, optional NDI registration
and AArch64 patches remain too. Changes from the 8.x series:

- NPP compatibility is omitted because upstream removed NPP filter support.
  `--enable-libnpp` now only warns and does nothing. This is a feature removal
  for consumers using NPP; the mixer demo already disables it.
- CUDA frame formats are checked generically upstream, including YUVA444P,
  so the old format-list addition is redundant.
- Upstream `scale_cuda` has a new antialiasing/filter path and already clips
  and rounds scaled samples. Keep that implementation; port explicit texture
  edge clamping and the stable Lanczos coefficient calculation without the
  old duplicate sample-conversion helpers. Pixel-reference and timing tests
  must cover the changed upstream scaler.
- Registration context and the AArch64 helper insertion point changed.
  The retained CUVID intra-only patch still recognizes the existing
  `gop_size=100000` convention to set `ulIntraDecodeOnly`, with the existing
  decoder-initialization logging.
- Ordinary NVDEC zero-copy CUARRAY output honors an explicit
  `options.extra_hw_frames` budget, already included by FFmpeg's frame-context
  setup, instead of reserving another 16 surfaces. An unspecified budget retains
  upstream's default; explicit pools above the registration limit fail before
  allocating arrays. Linear CUDA, CPU and copying CUARRAY modes are unchanged.
  Three extra frames are the opt-in source default, in addition to codec and
  FFmpeg working surfaces, not a full-mixer sizing recommendation. The measured
  192-input / 20-output graph used eight extras (17 surfaces for its HEVC
  clips); 28 outputs needed twelve extras (21 surfaces). Twelve total surfaces
  exhausted during 20-output startup, and seventeen exhausted with 28 outputs.
  Size a fixed pool for the
  complete downstream graph. The FFmpeg CLI adds its own queue allowance, so
  CLI pool counts are not directly comparable to an AVP decoder node.
- Internally allocated zero-copy CUARRAY pools use a private nonblocking CUDA
  stream per decoder, sharing the parent CUDA context. The output frames expose
  that producer stream through their device context so consumers can order
  reads with events. This path requires `cuvidDecodePictureAsync`; it does not
  fall back to synchronous decoding. Caller-supplied initialized frame pools
  retain their existing stream, and linear CUDA decoding is unchanged.

CUARRAY is a separate opt-in pixel format. Linear `AV_PIX_FMT_CUDA` and CPU
decoding remain upstream options. The experimental series permits `scale_cuda`
to forward matching-size, matching-format CUARRAY frames with passthrough enabled.
Array resizing or format conversion fails explicitly; the scaling kernels are
unchanged. Identity passthrough returns before loading unused scaling kernels.
The `pad_cuda`, `crop_cuda` and `transition_cuda` extensions consume NV12/P010/P210
arrays and produce ordinary linear CUDA output. Pad and transition sample array
planes; crop copies its rectangle directly into the final output. Mixed
array/linear transition inputs and different producer streams in the same CUDA
context are supported without an intermediate image copy. Completion fences
keep source frames alive through GPU reads. `tonemap_cuda` likewise reads arrays
in its color/depth/chroma conversion kernel and writes the final linear output,
without an intermediate input image. Declared identity conversions with no format
request preserve opaque frames; auto mode negotiates stable linear output and
retains zero-copy identity only for linear input. Known matching source colors
should bypass the conversion filter to preserve opaque storage. Other custom
filters retain their existing linear-input contracts; array support is not
enabled globally.
Upstream supplies a GPU-to-GPU
`av_hwframe_transfer_data()` path into a linear CUDA frame; its CUDA context
does not supply the map callbacks needed by `hwmap`. Direct array consumption
needs an explicit compatible consumer and lifetime handling. Do not label
array handles as linear device pointers or enable opaque output globally.

`verify.sh` proves exact patch application only. Compilation, custom-filter
pixel tests, CPU/linear-NVDEC regressions and resource/latency comparisons on an
NVIDIA host are separate acceptance gates; patch application is not evidence
of a performance improvement.

The [L4 capacity validation](../../doc/research/2026-10-03-cuarray-capacity.md)
records the matched 20-output comparison and 28-output capacity result, including
fixed-pool sizing failures and the zero-copy boundary.

## FFmpeg 8.x

One ordered series in `8/` applies to upstream **n8.0 and n8.1** from a single
copy; there are no per-version directories. `8/bases.env` pins each base
commit and the exact patched tree; `patch_count` guards against stray files.

```bash
deps/ffmpeg/apply.sh  <ffmpeg-checkout>          # git am 8/*.patch + configure fix-up
deps/ffmpeg/verify.sh <n8.0|n8.1> <ffmpeg-repo>  # isolated worktree, whole series, tree check
```

The mixer Dockerfiles call `apply.sh`; `FFMPEG_TAG` defaults to
`n8.1` and may be set to `n8.0`. avcpp and avplumber must be rebuilt against
the selected FFmpeg libraries.

## How one series serves both bases

The composition-suite filters are new files, so they carry no base-tree
context; the remaining patches are generated with minimal (`-U1`) context so
they apply where 8.0 and 8.1 differ only cosmetically. The single genuine
8.0/8.1 difference is a libnpp `configure` check that 8.1 added and that fails
on CUDA 13 (the legacy `nppiYCbCr420_8u_P2P3R` symbol is gone). `apply.sh`
rewrites that check to probe the stream-context API; on 8.0, which has no such
check, the rewrite is a no-op. Everything else in the NPP patch (`npp_compat.h`
and the filter changes) is base-independent.

## The series

1. **swscale aarch64 ARGB→YUVA420P** — uses `SwsInternal`/`opts` and the
   unscaled callback signature; algorithms unchanged.
2. **CUDA composition suite** — `convert_cuda`, `crop_cuda`, `overlay_many_cuda`,
   `pad_cuda` (intentionally replaces upstream's), `transition_cuda`, plus the
   scaling-edge/overlay-context fixes and YUVA444P CUDA frames. Upstream 8.x
   `scale_cuda` (with its expanded pixel formats) is kept.
3. **NPP CUDA 13 compatibility** — stream-context helper and filter changes.
4. **NVDEC intra-only handling.**
5. **RFC 4175 RTP frame handling.**
6. **V4L2 source timestamps.**
7. **NDI v5 registration** — registers the optional integration only; no SDK
   or device implementation is supplied, and NDI stays disabled in demo builds.
8. **10-bit CUDA transitions** — YUV420P10/422P10/444P10, P010 and P210 (plus
   8-bit 4:2:2/4:4:4) in a word-sample `transition_cuda` kernel.
9. **`tonemap_cuda`** — SDR BT.709 / HLG / PQ conversion on CUDA semiplanar
   frames (NV12/P010 4:2:0, NV16/P210 4:2:2, resampled in the same pass) in
   both directions: display-light conversion with configurable SDR
   white and HDR peak, HDR-to-SDR operators with a knee parameter, automatic
   per-frame contract resolution (untagged frames are BT.709 SDR), zero-copy
   identity frames and fixed NV12/P010 output storage.
10. **`band_blur_cuda`** — configurable vertical-band blur and luma gradient
    on NV12 CUDA frames; pixels outside the band are unchanged. Carries forward
    the filter from `1876208` and its FFmpeg 8.1 adaptation in `6dcda46` without
    changing its kernel or option defaults. The same patch applies to 8.0 and 8.1.
11. **`transition_cuda` dip** — mode `dip` fades main to a solid colour over
    alpha 0–0.5 and the colour to overlay over 0.5–1; the runtime `color`
    option takes `Y:Cb:Cr` as 8-bit limited-range codes (fractions allowed),
    which the filter scales to each frame's depth and range. A dip sample
    reads only the picture still visible, none at 0.5; fade and wipe modes
    are unchanged.

## FFmpeg 8 notes

- `AVFrame.pkt_pos` is gone: `transition_cuda`'s legacy `pos` expression
  variable evaluates to `NAN`; `crop_cuda` guards it by API version.
- The `C` command-support marker left `-filters` output; the mixer Dockerfile
  checks the transition's runtime `mode` and `color` options and the `dip`
  mode in filter help instead.
- FFmpeg 8 validates CUDA input formats at filter init, so the AVP filter node
  attaches `hw_frames_ctx` / `hw_device_ctx` between allocation and
  initialisation (segmented graph parser); the metadata-driven crop node does
  the same.
- When reusing a build tree with different GL flags, rebuild
  `deps/cuda_loader/cuda_drvapi_dynlink.o`; a loader built without GL lacks the
  EGL function-pointer variables, and linking `libcuda` directly to paper over
  that crashes during DMA-BUF import.

## Validation

Compilation, linking and filter registration are the acceptance criteria for
the series itself; runtime behaviour is covered by the demo and `tests/cuda`
suites. Validated on a T4 with FFmpeg 8.1: the 8-bit mixer at 30/60 fps with
video and DMA-BUF browser inputs, wipe-cache loads, cut/fade/wipe, NVENC to a
WebRTC browser; the former overlay demo’s 45-case pixel-reference matrix (now retired); and
the live recorder surviving SRT disconnects (`ignore_eof` on the pre-sentinel
format nodes, opt-in). NPP, NDI and AArch64 paths are not compiled in the demo
images.

For the band-blur pixel smoke on an NVIDIA host (requires NumPy):

```bash
python3 tests/cuda/smoke_band_blur.py /path/to/patched/ffmpeg
```

This synthetic-input test intentionally uploads/downloads frames to compare
identity, band boundaries, luma gradients, neutral chroma and blur-radius
endpoints. For a GPU-native decode/crop/filter/encode check, use
`tests/cuda/smoke_crop_filter_chain.py --band-blur` with a bounded video fixture.

## Pinned CPU-to-CUDA upload

The final patch in each series extends `hwupload_cuda` with `pinned=1` for
software NV12/P010 frames. It reuses the supplied filter CUDA device, copies
active pixels into reusable pinned host staging, and uploads on a private
nonblocking stream. The default transfer path is unchanged. An event orders
writes into recycled output frames after readers on the device stream, and
stream synchronization completes each output before it is forwarded. This
preserves the custom packet uploader's synchronization contract; moving it
into FFmpeg does not by itself eliminate driver locks or CPU waits.

`hwupload_cuda=v210_width=<picture-width>` accepts rawvideo/gray frames whose
width is the packed v210 byte stride. It uploads the packed bytes and unpacks
them on the GPU into P210, without CPU v210 decoding. This mode implies pinned
staging and requires NVCC or CUDA LLVM at build time; ordinary pinned upload
does not require a CUDA compiler. Set color and chroma tags with `setparams`.
The mixer uses this mode for v210 sources and `pinned=1` for pinned NV12/P010
sources.

On an NVIDIA host, `tests/cuda/smoke_nv12_input.py` checks both ordinary and
pinned uploads, looping timestamps, and exact v210 samples including padded
rows. `tests/cuda/benchmark_raw_upload.py` compares pinned and pageable filter uploads
with matched input counts, pacing and downstream GPU reads. CPU comparisons
must also retain the same output frame rate.
