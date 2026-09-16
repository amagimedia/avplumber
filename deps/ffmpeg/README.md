# FFmpeg patch series (FFmpeg 8.x)

One ordered series in `8/` applies to upstream **n8.0 and n8.1** from a single
copy; there are no per-version directories. `8/bases.env` pins each base
commit and the exact patched tree; `patch_count` guards against stray files.

```bash
deps/ffmpeg/apply.sh  <ffmpeg-checkout>          # git am 8/*.patch + configure fix-up
deps/ffmpeg/verify.sh <n8.0|n8.1> <ffmpeg-repo>  # isolated worktree, whole series, tree check
```

The demo Dockerfiles (`demos/mixer`, `demos/cuda-overlay`,
`demos/dmabuf-browser/consumer`) call `apply.sh`; `FFMPEG_TAG` defaults to
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
8. **10-bit CUDA transitions** — P210 and planar 10-bit frame handling.
9. **CUDA HDR-to-SDR tone mapping** — HLG/PQ input, selectable operators and
   BT.709 output, with fast-math transfer functions.
10. **CUDA source color conversion** — explicit SDR/HLG/PQ input and output
    transfers, display-light and gamut conversion, configurable SDR white,
    and output color metadata; the legacy tone-mapping interface is retained.
11. **P010 CUDA transitions** — accepts 10-bit 4:2:0 in the existing word-sample
    kernel, allowing NVDEC P010 frames to stay P010 through the mixer and NVENC.

12. **Automatic CUDA color conversion** — resolves complete SDR/HLG/PQ frame
    metadata on every frame, rejects missing or unsupported contracts, supports
    fixed NV12/P010 output storage and preserves matching frames without a copy.

## FFmpeg 8 notes

- `AVFrame.pkt_pos` is gone: `transition_cuda`'s legacy `pos` expression
  variable evaluates to `NAN`; `crop_cuda` guards it by API version.
- The `C` command-support marker left `-filters` output; the mixer Dockerfile
  checks the transition's `mode` option in filter help instead.
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
WebRTC browser; the `demos/cuda-overlay` 45-case pixel-reference matrix; and
the live recorder surviving SRT disconnects (`ignore_eof` on the pre-sentinel
format nodes, opt-in). NPP, NDI and AArch64 paths are not compiled in the demo
images.
