# FFmpeg 8.1 compatibility series

Base: upstream `n8.1`, commit `9047fa1b084f76b1b4d065af2d743df1b40dfb56`.
The exact patched tree is recorded in `base.env`.

This is the 8.1 adaptation of the seven features in `../7.1.5`, not a new
pixel-format or mixer pipeline design.

## Adaptations

1. **AArch64 ARGB conversion:** use `SwsInternal`, `opts` fields and the updated
   unscaled callback signature. Retain the existing conversion algorithms.
2. **CUDA composition:** use `FFFilter` registrations for `convert_cuda`,
   `crop_cuda`, `overlay_many_cuda`, `pad_cuda` and `transition_cuda`.
   The custom `pad_cuda` implementation intentionally replaces the upstream
   implementation in this series; switching padding semantics is out of scope.
   Keep upstream 8.1 `scale_cuda`, including its expanded pixel formats, and
   carry the existing scaling-edge and overlay-context fixes. Retain YUVA444P
   CUDA frame support. Use upstream's existing compute-75 compiler fallback.
3. **NPP CUDA 13 compatibility:** carry the stream-context helper and filter
   changes; probe the stream-context API in configure, since NPP 13 removes
   the legacy API checked by upstream. The mixer demo build keeps NPP disabled.
4. **NVDEC intra-only handling:** move the existing initialization hunk to its
   corresponding 8.1 location.
5. **RFC 4175:** carry the existing frame handling patch.
6. **V4L2:** carry the existing source-timestamp patch.
7. **NDI v5 registration:** rebase configuration and documentation. As in the
   7.1.5 series, this patch only registers the optional integration; it does not
   supply the NDI device implementation files or SDK. NDI remains disabled in
   the demo build and is not covered by its compile check.

FFmpeg 8 removed `AVFrame.pkt_pos`. The legacy `transition_cuda` expression
variable `pos` therefore evaluates to `NAN` (unavailable). `crop_cuda` already
guards that legacy variable by FFmpeg API version. Time/frame-based expressions
and the demo transition configuration do not use packet byte positions.

FFmpeg 8 also removed the `C` command-support marker from `-filters` output.
The mixer Dockerfile checks the transition's runtime-capable `mode` option in
filter help instead; this check works with both series.

The AVP filter node sets the buffer source's `hw_frames_ctx` before initializing
the filter. FFmpeg 8.1 validates CUDA input formats during initialization;
setting the context after `avfilter_graph_create_filter` is too late.
The metadata-driven CUDA crop node builds its own graph and follows the same
allocate -> attach frames context -> initialize sequence. Updating only the
generic filter node does not cover crop, portrait, or two-box output paths.
Filters that request a hardware device also receive it before initialization.
The AVP node uses FFmpeg's segmented graph parser to attach `hw_device_ctx`
between filter allocation and initialization; this is needed by `hwupload`
when preloading alpha wipes. These APIs are also available in FFmpeg 7.1.5,
so the same AVP filter source supports both versions without a version fork.
The binaries and Python modules must still be built separately for each
FFmpeg ABI. The 7.1.5 patch series and default Docker build version are unchanged.

## Hardware acceleration gains and limits

Compared with upstream n7.1.5, the n8.1 CUDA/NVIDIA path makes these features
available to applications that select the corresponding formats and codecs:

| Capability | Gain | Requirement / current coverage |
| --- | --- | --- |
| H.264 10-bit NVDEC/NVENC | Hardware High10 decode and encode | Blackwell GPU, SDK 13 headers and compatible driver; not tested on Blackwell. |
| H.264 / HEVC 4:2:2 NVDEC/NVENC | Hardware paths for higher chroma resolution, including 10-bit 4:2:2 | Blackwell GPU, SDK 13 headers and compatible driver; not provided by a T4 or L4 upgrade to FFmpeg alone. |
| CUDA scaling formats | Adds planar 4:2:2, NV16, P210/P216 and planar 10-bit 4:2:0/4:2:2/4:4:4 to upstream `scale_cuda` | CUDA format/scaling support is distinct from hardware codec support. P010 10-bit 4:2:0 already existed in 7.1.5. |
| Existing HEVC Main10 | Remains available on supporting GPUs | Not a new 8.1 capability; this PR does not qualify an end-to-end Main10 graph. |
| Existing custom CUDA composition and CUDA 13 NPP | Keeps the seven-patch suite buildable and usable with the new FFmpeg API | Tested 8-bit paths; custom padding/overlays/inference do not become 10-bit or 4:2:2 automatically. |

The recorder and mixer remain configured for 8-bit NV12/4:2:0. A 10-bit or 4:2:2
end-to-end product pipeline still needs compatible decode, filter, composition,
inference and encode stages, plus matching frame metadata. This update does not
add HDR tone mapping or qualify HDR metadata preservation. T4 testing cannot
establish Blackwell codec support or throughput gains.

SDK-dependent NVENC options are compiled conditionally. The mixer demo still
pins `NV_CODEC_HEADERS_TAG=n12.1.14.0`; a Blackwell build must select SDK 13-era
headers and a matching driver as well as `FFMPEG_TAG=n8.1`.

Sources: [NVIDIA SDK 13 release notes](https://docs.nvidia.com/video-technologies/video-codec-sdk/13.0/read-me/index.html),
[FFmpeg n8.1 H.264 NVENC profiles](https://github.com/FFmpeg/FFmpeg/blob/n8.1/libavcodec/nvenc_h264.c),
[FFmpeg n8.1 CUDA scaler](https://github.com/FFmpeg/FFmpeg/blob/n8.1/libavfilter/vf_scale_cuda.c),
[FFmpeg n7.1.5 CUDA scaler](https://github.com/FFmpeg/FFmpeg/blob/n7.1.5/libavfilter/vf_scale_cuda.c).

## Apply and verify

```bash
git clone --branch n8.1 --depth 1 https://github.com/FFmpeg/FFmpeg <build-path>
git -C <build-path> am <avplumber-path>/deps/ffmpeg/8.1/*.patch
deps/ffmpeg/8.1/verify.sh <path-to-FFmpeg>
```

Compilation/linking and filter registration are the acceptance criteria for this
port. They do not establish runtime correctness, performance, or support for
uncompiled optional NPP, NDI or AArch64 paths. Do not deploy over an existing
demo until separate runtime validation is completed.

Validated on 2026-09-15 in an isolated x86-64 CUDA development container:

- Both ordered patch series reproduce their pinned trees.
- FFmpeg 8.1 builds; all seven composition/scaling filters are registered.
- Pinned avcpp `31de3f4f937ed3bb30d083275e5e76192dfc9cb3` builds unchanged.
- avplumber binary and Python module build with CUDA/NVCC/DRM/GL enabled,
  FRUC/neural/TensorRT disabled.
- No GPU media graph or running demo was changed by this compile check.

Subsequent GPU runtime checks covered the 8-bit mixer at 30 and 60 fps, 41 prewarmed
scenes, video and DMA-BUF browser inputs, all five wipe-cache loads, cut/fade/wipe
commands, and NVENC output received by a WebRTC browser. The same filter source
also compiled against FFmpeg 7.1.5 and passed CUDA scaling and alpha-upload tests.

When reusing a build tree with different GL feature flags, rebuild
`deps/cuda_loader/cuda_drvapi_dynlink.o` with the new flags. A loader compiled
without GL lacks the EGL function-pointer variables. Linking `libcuda` directly
to satisfy those missing symbols is incorrect: it supplies functions where AVP
expects variables and crashes during DMA-BUF import. The validated mixer module
uses the GL-enabled dynamic loader, without direct `libcuda` linkage.

The current avcpp pin additionally backports custom-IO allocation/cleanup fixes
and CMake link-list handling. These retain the existing wrapper API; they are
maintenance fixes, not requirements for FFmpeg 8.1 compilation or a v3 migration.

## Full reframer and composition checks

The 2026-09-15 T4 check with the reframer's eight-patch FFmpeg 8.1 runtime,
CUDA 13/NPP, TensorRT and legacy float TrackNet covered native 1080p25 input,
H=20 camera-pan planning, Player 360p, salient detection, frame classification,
15 Hz scoreboard OCR, DMA-BUF browser overlays, portrait/square crops and eight
HLS video renditions. All 2,502 measured frames reached every pre-NVENC branch;
the latency collector reported no incomplete frames or dropped packets. All
eight finalized renditions were 25 fps and 100.2 seconds long.

This validates functionality, not steady low-latency performance: processing
had catch-up bursts, with post-NVDEC-to-pre-NVENC latency of 1.095 s median,
3.635 s p95 and 4.359 s maximum. GPU utilization was 54% median and peak device
memory was 2,558 MiB. Native 60 fps remains unqualified.

The independent `demos/cuda-overlay` pixel-reference matrix passed all 45 cases
on FFmpeg 8.1: 1-15 overlays in 420/420, 420/444 and 444/444 combinations,
including a 641-pixel-wide canvas. Every compared YUV sample matched.

`tests/cuda/smoke_crop_filter_chain.py` exercises the AVP crop node together
with CUDA padding, scaling, format conversion and NVENC.
Use `--scaler scale_npp` to cover NPP and `--band-blur` when the reframer's
optional `band_blur_cuda` patch is installed. CPU decoding is only the final
encoded-output assertion, not a transfer inside the CUDA processing chain.

## Live recorder EOF regression

A live SRT disconnect can finish the input group and propagate EOF into the
permanent pre-sentinel format declaration. `ignore_eof=true` on
`fake_video_format` / `fake_audio_metadata` keeps those nodes accepting frames
across reconnection. This is opt-in; default finite-graph EOF still propagates.

The FFmpeg 8.1 T4 recorder check survived two SRT disconnects with one recorder
generation. Its 1,600 consecutive 25 fps metadata records matched Kafka and GCS
JSONL; all primary HLS outputs contained 64 finalized one-second segments.
Native audio/video reconnect tests preserve their decoded frame timestamp
sequences, while default finite-EOF tests still finish. The unpatched image
fails the live-EOF regression. Full-recorder finite-VOD completion remains
separate work; the recorder still applies its live restart policy to file input.
