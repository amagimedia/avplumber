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
  FRUC/neural/TensorRT disabled. EGL/CUDA binary linking uses the toolkit's
  driver stub; the module import check also uses that link-only stub.
- No GPU media graph or running demo was changed by this compile check.

The current avcpp pin additionally backports custom-IO allocation/cleanup fixes
and CMake link-list handling. These retain the existing wrapper API; they are
maintenance fixes, not requirements for FFmpeg 8.1 compilation or a v3 migration.
