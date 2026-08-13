# Chromium offscreen DMA-BUF capture: history and NVIDIA gap

Research date: 2026-08-13. This note uses Chromium and Electron changes,
source, and documentation as primary sources. Statements labelled **inference**
connect those upstream facts to this repository's implementation.

## Short version

Chromium acquired the machinery for capturing compositor output into a shared
RGBA GPU texture in 2024. That made Electron's `useSharedTexture` path possible,
including a Linux `NativePixmap` handle whose plane FD is usually a DMA-BUF.
It did not solve which kind of buffer Linux should allocate.

On NVIDIA, Chromium's usual CPU-mappable capture allocation asks GBM for a
linear buffer. That allocation cannot also be used as the render target on the
affected NVIDIA drivers. Chromium's 2025 NVIDIA change added an internal
consumer choice for an exportable native handle without CPU access, selecting
`SCANOUT` instead of `SCANOUT_CPU_READ_WRITE`. The choice did not become a Chrome
flag, and Electron 41 and 43 still select the old mappable mode.

There is a second, important gap. A still-unmerged August 2026 Chromium change
says the capturer itself only entered its GPU texture/blit path for the mappable
preference. Therefore changing Electron to select the native-handle preference
is not sufficient by itself for Chromium 146 or 150. A source-built solution
must carry both the Electron consumer selection and that Chromium follow-up.

The repository's GBM shim takes a different route. It leaves Electron on the
known-working mappable texture-capture path, but intercepts GBM allocation calls
below Chromium, clears `GBM_BO_USE_LINEAR`, and adds
`GBM_BO_USE_SCANOUT | GBM_BO_USE_RENDERING`. This is why the shim works with a
stock Electron build, but it is deliberately a narrow NVIDIA workaround rather
than a portable Chromium policy.

## 1. The 2024 foundation: shared RGBA textures

Reito OvO's two foundational changes were generic native-texture work, not an
NVIDIA-specific DMA-BUF API:

- [CL 5276423](https://chromium-review.googlesource.com/c/chromium/src/+/5276423),
  landed 2024-02-15 as
  [commit `5e5912eb`](https://chromium.googlesource.com/chromium/src/+/5e5912eb4ee6d0986a4b098d7109c62267b3fd0e)
  (`main@{#1261327}`). It added RGBA blitting into an existing native texture
  and tests for native-texture and blit output. Its commit message explicitly
  identifies CL 5265077 as its consumer.
- [CL 5265077](https://chromium-review.googlesource.com/c/chromium/src/+/5265077),
  landed 2024-03-05 as
  [commit `b99a8c87`](https://chromium.googlesource.com/chromium/src/+/b99a8c87a20c5d462ec887c5ac2f970cbe58abc7)
  (`main@{#1268507}`). It made the frame pools format-aware and connected ARGB
  capture to GPU-memory-buffer/native-texture output. Its rationale explicitly
  names CEF offscreen rendering, Electron, and OBS as users of accelerated
  shared textures.

Electron documents the platform handle produced by that abstraction. On Linux,
an [`OffscreenSharedTexture`](https://github.com/electron/electron/blob/v43.4.0/docs/api/structures/offscreen-shared-texture.md)
contains a
[`SharedTextureHandle.nativePixmap`](https://github.com/electron/electron/blob/v43.4.0/docs/api/structures/shared-texture-handle.md):
per-plane stride, offset, size, and an FD described as "usually dmabuf," plus the
GBM modifier needed by EGL. Thus "native texture" is the cross-platform capture
concept; DMA-BUF is the usual Linux backing handle.

### Tests and pixel-format corrections

The same author followed the initial work with tests and format corrections.
The most relevant sequence is:

| Date (UTC) | Chromium change | What it established |
| --- | --- | --- |
| 2024-02-29 | [CL 5311840](https://chromium-review.googlesource.com/c/chromium/src/+/5311840), [commit `b923fdd8`](https://chromium.googlesource.com/chromium/src/+/b923fdd8c3580438139b1a33197ef632ffe74bb8) | Parameterized capturer tests and default + ARGB coverage. |
| 2024-03-01 | [CL 5311841](https://chromium-review.googlesource.com/c/chromium/src/+/5311841), [commit `c5e96f88`](https://chromium.googlesource.com/chromium/src/+/c5e96f88500f15ebe0bd2783afb20987bb641ec1) | GPU-buffer preference + NV12 coverage. |
| 2024-03-06 | [CL 5310884](https://chromium-review.googlesource.com/c/chromium/src/+/5310884), [commit `f2d7b0c4`](https://chromium.googlesource.com/chromium/src/+/f2d7b0c426872d06bd40ca6bd9156d2a93ea6b0d) | Parameterized renderable-pool tests and ARGB coverage. |
| 2024-04-06 | [CL 5418235](https://chromium-review.googlesource.com/c/chromium/src/+/5418235), [commit `c2d790e5`](https://chromium.googlesource.com/chromium/src/+/c2d790e55570b2a5bada724f60ad326ef91748e3) | Clarified ARGB/RGBA/BGRA terminology and platform-dependent storage. |
| 2024-04-13 | [CL 5421371](https://chromium-review.googlesource.com/c/chromium/src/+/5421371), [commit `4ac051a4`](https://chromium.googlesource.com/chromium/src/+/4ac051a42f3f764737fc8064d759d4e4e5839308) | Corrected ARGB/ABGR to BGRA/RGBA buffer-format selection and tests. |
| 2024-04-17 | [CL 5446634](https://chromium-review.googlesource.com/c/chromium/src/+/5446634), [commit `37a215f0`](https://chromium.googlesource.com/chromium/src/+/37a215f02c719c04a1244a81b6a1babe432aef63) | Made BGRA the default GPU-buffer storage for requested ARGB. |

[Reito's complete Chromium change history](https://chromium-review.googlesource.com/q/author:reito@chromium.org+or+author:carolwolfking@gmail.com)
also includes two useful Linux/NVIDIA-adjacent changes: [CL
5338184](https://chromium-review.googlesource.com/c/chromium/src/+/5338184)
fixed modifier-import validation using a wrong 1x1 size in a case attributed to
NVIDIA driver behavior, and [CL
5348599](https://chromium-review.googlesource.com/c/chromium/src/+/5348599)
made NativePixmap GPU tests exercise the Ozone Wayland path and included an
NVIDIA trybot. [CL
5423057](https://chromium-review.googlesource.com/c/chromium/src/+/5423057)
later corrected unsupported RGBA NativePixmap backing selection on X11.

## 2. The NVIDIA allocation conflict

[Chromium CL 6681354](https://chromium-review.googlesource.com/c/chromium/src/+/6681354),
submitted by an NVIDIA engineer, landed 2025-08-06 as
[commit `a531c83a`](https://chromium.googlesource.com/chromium/src/+/a531c83a9bbb552fa13ceeaf73d0a19d1203cc12)
(`main@{#1497315}`). Its commit message and code describe the conflict:

1. Linux capture buffers normally use
   `gfx::BufferUsage::SCANOUT_CPU_READ_WRITE`.
2. That usage implies `GBM_BO_USE_LINEAR`.
3. On the affected NVIDIA GBM path, the linear request prevents the buffer from
   being used for GPU rendering.
4. `gfx::BufferUsage::SCANOUT` omits the linear/CPU-mappable requirement while
   retaining the rendering, scanout, and texturing usages needed for GPU
   interop.

The landed design did not hard-code NVIDIA vendor detection. It added the
internal Mojo enum
[`kPreferSharedImageWithNativeHandle`](https://chromium.googlesource.com/chromium/src/+/a531c83a9bbb552fa13ceeaf73d0a19d1203cc12/services/viz/privileged/mojom/compositing/frame_sink_video_capture.mojom#94),
then propagated that consumer preference to
[`requires_cpu_access=false`](https://chromium.googlesource.com/chromium/src/+/a531c83a9bbb552fa13ceeaf73d0a19d1203cc12/components/viz/service/frame_sinks/video_capture/gpu_memory_buffer_video_frame_pool.cc#80).
Only the non-CPU-access Linux branch chooses
[`BufferUsage::SCANOUT`](https://chromium.googlesource.com/chromium/src/+/a531c83a9bbb552fa13ceeaf73d0a19d1203cc12/media/video/renderable_gpu_memory_buffer_video_frame_pool.cc#195);
the default behavior remains CPU-mappable.

This consumer-level design was intentional. An early patch set selected
`SCANOUT` from an NVIDIA vendor ID, but review converged on describing what the
caller needs instead. The review explains that Chrome needs buffers it can map
for its hardware/software encoder mix, while a CEF-style GPU-only consumer can
ask for an exportable native handle without mapping. See the
[consumer-intent proposal](https://chromium-review.googlesource.com/c/chromium/src/+/6681354/comments/988937b9_66a79096)
and [mapping/encoder rationale](https://chromium-review.googlesource.com/c/chromium/src/+/6681354/comments/a3b73dd3_2e9b8341).

## 3. What was lost between Chromium and Electron

There are two independent missing links.

### Electron never selects the new consumer preference

Electron publicly exposes `webPreferences.offscreen.useSharedTexture` and
pixel-format selection, but not Chromium's mappable-versus-native-handle choice
([Electron 43 API](https://github.com/electron/electron/blob/v43.4.0/docs/api/structures/web-preferences.md#L84-L98)).
Its C++ OSR consumer decides the internal preference:

- [Electron 41.3.0](https://github.com/electron/electron/blob/v41.3.0/shell/browser/osr/osr_video_consumer.cc#L78-L88),
  whose [DEPS pin Chromium 146.0.7680.188](https://github.com/electron/electron/blob/v41.3.0/DEPS#L3-L6),
  selects `kPreferMappableSharedImage` whenever `useSharedTexture` is true.
- [Electron 43.4.0](https://github.com/electron/electron/blob/v43.4.0/shell/browser/osr/osr_video_consumer.cc#L80-L90),
  whose [DEPS pin Chromium 150.0.7871.224](https://github.com/electron/electron/blob/v43.4.0/DEPS#L3-L6),
  makes the same selection.

**Verified fact:** CL 6681354 added an internal C++/Mojo consumer preference,
not a command-line flag or web API. Electron 41 and 43 do not select it.

**Inference:** these public APIs and fixed OSR call sites provide no supported
runtime way for an Electron 41/43 user to select the native-handle preference.
The Chromium review also says Chrome's own capture use remains mappable. This
is the precise sense in which the NVIDIA fix was "lost in translation": the
allocation capability exists upstream, but the Electron consumer cannot ask
for it.

### The native-handle preference did not reach texture capture

There is a deeper Chromium gap. [CL
8220427](https://chromium-review.googlesource.com/c/chromium/src/+/8220427),
created 2026-08-07 and still **NEW** at patch set 3 on 2026-08-13, states that
`kPreferSharedImageWithNativeHandle` already allocates the intended
non-CPU-mappable `SCANOUT` buffer, but `FrameSinkVideoCapturerImpl` still:

- enables GPU texture/blit capture only for `kPreferMappableSharedImage`; and
- checks for that mappable preference when accepting ARGB texture results.

The proposed change admits the native-handle preference to those paths and
sets `populates_mappable_shared_image=false` on its blit request.

**Inference from the current CL and source:** changing only Electron's OSR
consumer from `kPreferMappableSharedImage` to
`kPreferSharedImageWithNativeHandle` is not a complete source fix for Chromium
146 or 150. A rebuild must also carry the Chromium portion of CL 8220427 (or an
equivalent landed successor). This conclusion supersedes the earlier assumption
that CL 6681354 was complete end-to-end. The demo now bundles that production
change as `chromium-native-handle-texture-capture.patch` alongside its Electron
selector patch.

## 4. How this repository solves it

### Stock Electron: the GBM shim

[`gbm_linear_shim.c`](../../deps/dma-browser/native/gbm-linear-shim/gbm_linear_shim.c)
interposes the four GBM allocation entry points in this demo that carry usage
flags:

- `gbm_bo_create`;
- `gbm_bo_create_with_modifiers2`;
- `gbm_surface_create`; and
- `gbm_surface_create_with_modifiers2`.

For every intercepted call while `GBM_LINEAR_SHIM` is enabled, it computes:

```text
new_flags = (old_flags & ~GBM_BO_USE_LINEAR)
          | GBM_BO_USE_SCANOUT
          | GBM_BO_USE_RENDERING
```

Adding the last two flags is controlled by `GBM_LINEAR_SHIM_ADD_SCANOUT`; both
shim switches default to enabled once the library is preloaded. The shim then
forwards the allocation to the real GBM function through `RTLD_NEXT`. It does
not change Chromium's feature list, the supplied modifier list, the captured
pixel format, or video decoding.

Why this avoids both missing links:

1. Stock Electron remains on `kPreferMappableSharedImage`, so Chromium 146/150
   still enters their established shared-texture/blit path.
2. At the final GBM allocation boundary, the shim removes the linear request
   that conflicts with NVIDIA rendering and adds the renderable/scanout uses.
3. Electron exposes the resulting NativePixmap FD and modifier; avplumber
   imports that allocation through EGL rather than CPU-mapping it.

This is a pragmatic policy override, not a general fix. Its risks are visible
in the code:

- `LD_PRELOAD` applies the interposition process-wide, not only to Electron's
  OSR capture pool. Any intercepted allocation that genuinely needs a linear,
  CPU-mappable buffer receives different semantics.
- The C shim itself does not inspect the GPU vendor. The surrounding launcher
  and documented invocation must restrict it to the intended NVIDIA process;
  preloading it on Intel or AMD can alter an otherwise valid allocation policy.
- Because both environment switches default to true, merely preloading the
  library enables the rewrite unless `GBM_LINEAR_SHIM=0` is set.
- The output may be tiled/block-linear. Every consumer must respect the DRM
  modifier; treating the FD as linear memory produces corrupted output.
- The `.so` must be compiled for the target host architecture and linked ABI.

[`GBM_LINEAR_SHIM_LOG=1`](../../deps/dma-browser/native/gbm-linear-shim/gbm_linear_shim.c)
logs old and new flag values, which is the direct way to verify the rewrite.
The normal [`bin/run.sh`](../../deps/dma-browser/bin/run.sh) does not preload
the shim automatically.

### Recompiled Electron: current repository state

The bundled
[`electron-offscreen-native-handle.patch`](../../demos/dmabuf-browser/chromium/electron-offscreen-native-handle.patch)
adds a disabled-by-default Electron feature. When enabled for a
`useSharedTexture` window, it changes only Electron's OSR consumer from
`kPreferMappableSharedImage` to `kPreferSharedImageWithNativeHandle`. The
[`build-electron.sh`](../../demos/dmabuf-browser/chromium/build-electron.sh)
script intentionally applies that Electron patch while requiring the embedded
Chromium checkout to remain clean.

**Verified repository fact after the documentation fix:** the build script
applies both `chromium-native-handle-texture-capture.patch`, based on CL
8220427, and `electron-offscreen-native-handle.patch`. Both patches pass
`git apply --check` against Electron 41.3.0 / Chromium 146.0.7680.188 and
Electron 43.4.0 / Chromium 150.0.7871.224.

The combined build remains runtime-gated and vendor-neutral in its binary:
Electron selects the native-handle preference only when the demo feature is
enabled, while the launcher may enable that
feature only after detecting NVIDIA.

## README narrative distilled

For an external reader, the shortest accurate story is:

1. Reito OvO's 2024 Chromium work made compositor capture available as shared
   RGBA native textures; on Linux, Electron exposes those as NativePixmap plane
   FDs, usually DMA-BUFs.
2. Chromium originally allocated those capture textures as CPU-mappable. That
   implies a linear GBM buffer, which the affected NVIDIA path cannot render
   into.
3. NVIDIA's 2025 Chromium CL added an internal consumer preference for an
   exportable, non-CPU-mappable native handle, but Electron never selected it,
   and the Chromium texture-capture path still needs the unmerged 2026
   follow-up.
4. The tested stock-Electron route keeps the working capture path and uses the
   GBM shim to remove `LINEAR` and add `SCANOUT | RENDERING` at allocation time.
   It is effective but process-wide and NVIDIA-specific in its intended use.
5. The clean source route is a combined Chromium + Electron patch, exposed as
   a disabled-by-default runtime feature and enabled only by the NVIDIA-aware
   launcher.
