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
8. **10-bit CUDA transitions** — YUV420P10/422P10/444P10, P010 and P210 (plus
   8-bit 4:2:2/4:4:4) in a word-sample `transition_cuda` kernel.
9. **`tonemap_cuda`** — SDR BT.709 / HLG / PQ conversion on CUDA semiplanar
   frames (NV12/P010 4:2:0, NV16/P210 4:2:2, resampled in the same pass) in
   both directions: display-light conversion with configurable SDR
   white and HDR peak, HDR-to-SDR operators with a knee parameter, automatic
   per-frame contract resolution (untagged frames are BT.709 SDR), zero-copy
   identity frames and fixed NV12/P010 output storage.
10. **libmxl demuxer and muxer** — [MXL](https://github.com/dmf-mxl/mxl)
    shared-memory flows: demuxer, muxer, URI parser, JSON/diagnostic helpers,
    FATE coverage and `--enable-libmxl` glue. Squashed from `cbcrc/FFmpeg`
    branch `dmf-mxl/8.1` (pinned at `9eddb90`), which forks from the same
    n8.1 commit `8/bases.env` pins. Consumed by `demos/mixer/Dockerfile` for
    `demos/mxl`; the MXL SDK itself is pinned in that Dockerfile.

### Regenerating `8/0010-avformat-libmxl-demuxer-muxer.patch`

Docker only, and no compiler needed — the container just replays git history
(works on Linux and macOS hosts):

```bash
docker build -f deps/ffmpeg/Dockerfile.mkpatch \
    -t avplumber-mxl-mkpatch:local deps/ffmpeg
docker run --rm \
    -v "$PWD/deps/ffmpeg/8:/out" \
    -v "$PWD/deps/ffmpeg/8:/patches:ro" \
    avplumber-mxl-mkpatch:local
```

Set `MXL_REMOTE_REF`/`MXL_PIN` (and `FFMPEG_TAG`) to move to a newer fork
branch, e.g. `dmf-mxl/9.0` for a future base. Afterwards refresh
`8/bases.env`: `verify.sh` prints the actual tree per base when the pinned
one no longer matches, and `patch_count` must match the file count.

Cherry-pick conflicts stop the container with instructions; re-run it with
`--entrypoint bash` and finish by hand (`git cherry-pick --continue`, then
`/usr/local/bin/mkpatch-finish`).
>>>>>>> fcd616d (Add the MXL demuxer/muxer patch to the FFmpeg 8.x series)

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
