# Chromium native-handle DMA-BUF capture: upstream status

Research snapshot: 2026-08-17.

## Short answer

No, not by itself.

[Chromium CL 8220427](https://chromium-review.googlesource.com/c/chromium/src/+/8220427)
is the upstream equivalent of this demo's Chromium texture-capture patch. Once
it lands in the Chromium revision embedded by an application, that Chromium
patch can go away. The CL does not, however, make Chrome or Electron request
the native-handle allocation, expose a user flag, or add a public API.

Stock Electron still calls `FrameSinkVideoCapturer::Start()` with
`kPreferMappableSharedImage` when `offscreen.useSharedTexture` is enabled, both
in [Electron 43](https://github.com/electron/electron/blob/v43.4.0/shell/browser/osr/osr_video_consumer.cc#L79-L87)
and on [Electron main at the researched revision](https://github.com/electron/electron/blob/ee0811dc0ef42ec01af55e890ce375735be95eb2/shell/browser/osr/osr_video_consumer.cc#L79-L88).
Therefore an official, no-shim route also needs an upstream Electron change
that selects `kPreferSharedImageWithNativeHandle`, followed by an Electron
release containing both changes. There is no such Electron PR at this time.

Standalone Chrome is not a substitute. This is an internal privileged Mojo
capture preference, not a web API or Chrome policy, and Chrome has no API that
hands a page or end user DMA-BUF file descriptors.

In other words:

- **Today:** use the GBM allocation shim with stock Electron, or use the two
  local source patches.
- **After only CL 8220427 lands:** the local Chromium patch can be removed from
  a sufficiently new build, but stock Electron still needs the shim because it
  still asks for a CPU-mappable allocation.
- **After Chromium and Electron both land their parts:** an official Electron
  binary can avoid both the Chromium patch and the allocation shim. The
  application still needs Electron OSR setup and, when sending the DMA-BUF to
  avplumber, real FD-passing and lifetime management.

## What is already upstream, and what CL 8220427 adds

The allocation half landed in August 2025 as
[CL 6681354 / commit `a531c83a`](https://chromium.googlesource.com/chromium/src/+/a531c83a9bbb552fa13ceeaf73d0a19d1203cc12).
It introduced the consumer preference
[`kPreferSharedImageWithNativeHandle`](https://chromium.googlesource.com/chromium/src/+/c63860d47b10c40d033684aa448fd4509ebba99e/services/viz/privileged/mojom/compositing/frame_sink_video_capture.mojom#93),
which means that the consumer wants an exportable native handle and does not
require CPU mapping. On Linux the preference makes the capture pool use
[`gfx::BufferUsage::SCANOUT`](https://chromium.googlesource.com/chromium/src/+/c63860d47b10c40d033684aa448fd4509ebba99e/media/video/renderable_mappable_shared_image_video_frame_pool.cc#198)
instead of `SCANOUT_CPU_READ_WRITE`. Chromium maps those to the following GBM
flags:

| Chromium usage | GBM flags relevant here |
|---|---|
| `SCANOUT` | `RENDERING | SCANOUT | TEXTURING` |
| `SCANOUT_CPU_READ_WRITE` | `LINEAR | SCANOUT | TEXTURING` |

The mapping is in
[`ui/gfx/linux/gbm_util.cc`](https://chromium.googlesource.com/chromium/src/+/c63860d47b10c40d033684aa448fd4509ebba99e/ui/gfx/linux/gbm_util.cc#12).
The original review deliberately made this a consumer choice. An attempted
global switch broke Intel and NVIDIA GPU tests, and a reviewer noted that
Chrome needs mappable frames for its mixture of hardware and software video
encoders; see the discussion on
[CL 6681354](https://chromium-review.googlesource.com/c/chromium/src/+/6681354?tab=comments).

That change left a hole: the capture pool can allocate the non-mappable
SharedImage, but `FrameSinkVideoCapturerImpl` still chooses the GPU
SharedImage/blit result only for `kPreferMappableSharedImage`. The native-handle
preference therefore follows the system-memory result path instead of
populating the allocated texture. The current pre-CL condition is visible in
[`FrameSinkVideoCapturerImpl::CaptureFrame()`](https://chromium.googlesource.com/chromium/src/+/c63860d47b10c40d033684aa448fd4509ebba99e/components/viz/service/frame_sinks/video_capture/frame_sink_video_capturer_impl.cc#1122).

CL 8220427 closes that hole by:

1. treating both SharedImage preferences as GPU texture results;
2. setting `BlitRequest::populates_mappable_shared_image` only for the actually
   mappable preference; and
3. accepting either preference when processing the texture result.

It changes only the Viz implementation, its test, and `AUTHORS`. It does not
add a base feature, switch, policy, environment variable, vendor test, Chrome
UI, Electron code, or public JavaScript API. The capture preference remains an
argument supplied by the privileged native consumer to
[`FrameSinkVideoCapturer::Start()`](https://chromium.googlesource.com/chromium/src/+/c63860d47b10c40d033684aa448fd4509ebba99e/services/viz/privileged/mojom/compositing/frame_sink_video_capture.mojom#208).

### Important format limitation in the current patch set

The current patch is safe to treat as **ARGB-only** for the native-handle
preference. `CaptureFrame()`'s new condition mentions NV12 and RGBAF16, but
[`Start()` still CHECKs](https://chromium.googlesource.com/chromium/src/+/c63860d47b10c40d033684aa448fd4509ebba99e/components/viz/service/frame_sinks/video_capture/frame_sink_video_capturer_impl.cc#454)
that both formats use `kPreferMappableSharedImage`. Patch set 3 adds a
native-handle test only for ARGB. An Electron follow-up must therefore either:

- select the native-handle preference only for ARGB and keep the mappable path
  for `nv12` and `rgbaf16`; or
- first extend Chromium's format contract and tests for those formats.

Blindly replacing Electron's preference for every `useSharedTexture` pixel
format would turn the NV12/RGBAF16 cases into process-terminating `CHECK`
failures. This demo uses the ARGB path, whose delivered Linux texture can be
BGRA or RGBA as described by the
[`SetFormat()` contract](https://chromium.googlesource.com/chromium/src/+/c63860d47b10c40d033684aa448fd4509ebba99e/services/viz/privileged/mojom/compositing/frame_sink_video_capture.mojom#120).

## Current review and release status

As of the snapshot time, CL 8220427 is **not landed**: Gerrit reports status
`NEW`, patch set 3, two Code-Review +1 votes, CQ 0, one unresolved naming
comment, and no submitted commit. Its parent is
[Chromium 153.0.7995.0](https://chromium.googlesource.com/chromium/src/+/c63860d47b10c40d033684aa448fd4509ebba99e/chrome/VERSION),
but that does not establish a minimum released version. M153's branch date is
also 2026-08-17, so whether the CL reaches M153 depends on landing/branching or
a backport; otherwise the first containing milestone will be later. Chrome's
official schedule lists M153 stable for 2026-09-08.
[Chrome release schedule](https://developer.chrome.com/blog/chrome-two-week-release)

No minimum Electron version can yet be named. Electron 44 targets Chromium
M152 and therefore cannot naturally contain this M153-based change. The next
normal release train, Electron 45, targets Chromium M156, but it will only solve
the problem if an Electron selector change also lands. Electron lists 45.0.0
stable for 2026-10-20; prerelease dependency versions and dates are explicitly
estimates.
[Electron release schedule](https://releases.electronjs.org/schedule)

The practical minimum will be:

1. the first Chromium commit/milestone that actually contains CL 8220427; and
2. the first Electron build that both embeds that commit and changes its OSR
   consumer to request the native-handle preference for a supported format.

Until both commit hashes are present, version-number guessing is unsafe.

## User configuration versus native application work

### What an end user can configure

CL 8220427 itself needs **no command-line feature flag and defines no policy**.
There is no Chrome flag that selects `kPreferSharedImageWithNativeHandle`.

The local flag

```text
--enable-features=RenderableMappableSharedImageForceScanout
```

is defined by this demo's Electron patch, not Chromium or upstream Electron.
It is ignored by stock binaries because no upstream code registers or reads
that feature. It remains useful only with the locally patched Electron build.

Ozone/GL switches solve a different problem: selecting a working NVIDIA
rendering backend. For this headless Wayland demo the tested family is:

```text
--enable-gpu
--ozone-platform=wayland
--use-gl=angle
--use-angle=gl-egl
--disable-vulkan
```

The demo also uses deployment-oriented switches such as
`--ignore-gpu-blocklist`, `--disable-hardware-overlays`, and
`--disable-accelerated-video-decode`. None changes the capture buffer
preference and none replaces the Electron native change. `--no-sandbox` is a
container choice, not a requirement introduced by this CL.

### What the Electron application must do

Once an official Electron release contains both upstream changes, the app must
still:

1. create the window with
   `webPreferences.offscreen = { useSharedTexture: true }`;
2. keep GPU acceleration enabled and choose a working Ozone/GL backend for its
   deployment;
3. consume the `paint` event's `OffscreenSharedTexture` in the main process;
4. import or transmit the complete native-pixmap description; and
5. call `texture.release()` only after downstream GPU use is complete.

Electron documents the Linux handle as one or more planes containing
FD/stride/offset/size plus a DRM modifier.
[Electron `SharedTextureHandle`](https://www.electronjs.org/docs/latest/api/structures/shared-texture-handle)
Electron also warns that only a limited number of textures can exist and that
the texture must be released promptly after use.
[Electron `OffscreenSharedTexture`](https://www.electronjs.org/docs/latest/api/structures/offscreen-shared-texture)

That JS configuration cannot itself choose the native-handle allocation today.
Electron native code owns that choice. A good upstream Electron API would make
the GPU-only/no-CPU-access intent explicit rather than silently changing all
existing `useSharedTexture` callers; existing callers may rely on mappability,
and the native-handle path currently supports only ARGB.

### What remains necessary for this avplumber topology

Removing the GBM allocation shim does not remove the native FD transport. A
DMA-BUF descriptor number is local to a process. Sending it to the external
avplumber process still requires `SCM_RIGHTS` or an equivalent native IPC
mechanism. Electron's documentation likewise says that the texture cannot be
directly passed to another process and its lifetime must remain managed by the
main process.

The demo's `fdpass` addon and acknowledgment protocol are therefore still
needed (or must be replaced with an equivalent), even on a future stock
Electron. They are not the problematic GBM allocation shim.

## What “GPU-only DMA-BUF” does and does not guarantee

`kPreferSharedImageWithNativeHandle` is vendor-neutral and is a preference for
an exportable native handle that **may not be CPU mappable**. On Linux it removes
the CPU-linear allocation requirement by selecting `SCANOUT`; it does not
promise that the storage is physically device-local, that `mmap()` must fail,
or that only NVIDIA can import it. NVIDIA's GBM implementation is then free to
choose its renderable tiled/block-linear layout and export the corresponding
modifier.

A DMA-BUF FD alone does not describe pixels. A correct interchange contract
must include:

- the DRM fourcc/pixel format;
- width and height;
- the DRM format modifier; and
- an FD, stride, and offset for every plane.

The Linux kernel's DMA-BUF interchange documentation requires exactly this
metadata and warns that even an apparent format/modifier intersection does not
guarantee a particular importer will accept the buffer.
[Linux DMA-BUF allocation and exchange](https://docs.kernel.org/userspace-api/dma-buf-alloc-exchange.html)

NVIDIA modifiers are tiled and vendor/device specific. A CPU row-by-row read or
a DRM path that discards the modifier can produce corrupt imagery. Import
through EGL with
[`EGL_EXT_image_dma_buf_import_modifiers`](https://registry.khronos.org/EGL/extensions/EXT/EGL_EXT_image_dma_buf_import_modifiers.txt),
or another API that explicitly supports the exact fourcc/modifier/plane tuple.
Cross-GPU and cross-driver imports are not guaranteed; the reliable path is an
importer using a compatible NVIDIA graphics stack on the intended GPU.

Electron exposes a Linux `NativePixmapHandle`, not a CUDA memory handle. This
demo correctly converts the plane metadata to an EGLImage and then registers
that EGLImage with CUDA. NVIDIA documents
[`cudaGraphicsEGLRegisterImage()`](https://docs.nvidia.com/cuda/archive/8.0/cuda-runtime-api/group__CUDART__EGL.html)
as the API that registers an EGL image for CUDA access. A DRM consumer instead
imports each DMA-BUF FD into its own DRM/GEM handle; subsystem-native handles
are not portable, while DMA-BUF FDs are the interchange object.

The demo currently transports only plane 0 and restricts capture to 32-bit
BGRA/RGBA, which is valid for its single-plane ARGB path. It must be extended
before adopting a multi-plane format such as NV12, independently of the
Chromium/Electron changes.

## Synchronization and release

Chromium orders the internal capture blit with its GPU `SyncToken`; CL 8220427
does not export a new synchronization object. Electron's Linux public handle
contains the image planes and modifier but no Chromium `SyncToken` or explicit
sync-file FD.

DMA-BUF supports implicit synchronization through reservation fences. An
implicit-sync EGL/OpenGL importer normally waits on prior producer work.
Explicit-sync APIs such as Vulkan require the application to bridge fences
correctly, for example with `DMA_BUF_IOCTL_EXPORT_SYNC_FILE` and
`DMA_BUF_IOCTL_IMPORT_SYNC_FILE`; the kernel documentation explicitly
distinguishes these cases.
[Linux DMA-BUF synchronization](https://docs.kernel.org/driver-api/dma-buf.html)

Read readiness is only half of the contract. The producer must not recycle and
overwrite a captured allocation while the consumer is still reading it. Keep
the `OffscreenSharedTexture` alive until the downstream GPU operation has
completed, then call `release()`. This demo's frame-number acknowledgment does
that: it holds the Electron texture until avplumber's CUDA completion releases
the last downstream reference. Merely duplicating the DMA-BUF FD keeps the
allocation object alive, but does not by itself stop Chromium from reusing its
contents.

## NVIDIA runtime prerequisites

The Chromium CL does not set a minimum NVIDIA driver version. It assumes that
the surrounding EGL/GBM/DRM stack can allocate, export, and import the chosen
native pixmap.

For NVIDIA's GBM path, the official driver README requires DRM KMS, Mesa
`libgbm.so.1` 21.2 or newer, and (for Wayland) `egl-wayland` 1.1.8 or newer.
It also describes the NVIDIA GBM backend and GBM EGL external-platform library.
[NVIDIA GBM and Wayland requirements](https://download.nvidia.com/XFree86/Linux-x86_64/580.173.02/README/gbm.html)

The host/container must expose the DRM render node and matching NVIDIA graphics
userspace: EGL/GLVND, the NVIDIA GBM backend, and its external-platform files.
Compute-only container capabilities are insufficient. NVIDIA's container
toolkit documents `graphics` as the capability required for OpenGL/Vulkan and
notes that the default capability set is only `utility,compute`.
[NVIDIA Container Toolkit driver capabilities](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/docker-specialized.html#driver-capabilities)

No CL can compensate for loading Mesa or distro FFmpeg/EGL libraries in one
part of the pipeline and incompatible NVIDIA libraries in another, choosing
the wrong render node, or omitting modifier support.

## Other related and upcoming work

### Concrete and relevant

- [Chromium issue 428022619](https://issues.chromium.org/issues/428022619) is
  the tracking bug. Its linked code resources are the already-merged allocation
  CL 6681354 and open CL 8220427.
- [Chromium CL 6681354](https://chromium-review.googlesource.com/c/chromium/src/+/6681354)
  is the completed allocation/API half.
- [Chromium CL 8220427](https://chromium-review.googlesource.com/c/chromium/src/+/8220427)
  is the only open Chromium CL found by bug number, preference name, GBM-linear,
  native-pixmap, and frame-sink-capture searches. Gerrit's related-changes and
  submitted-together endpoints list no dependencies.
- [Electron issue #52618](https://github.com/electron/electron/issues/52618)
  is an open, reproducible NVIDIA `useSharedTexture` failure report on Electron
  43.2.0. It describes the same `SCANOUT_CPU_READ_WRITE`/`GBM_BO_USE_LINEAR`
  failure. It has no milestone, assignee, maintainer design response, or linked
  fix PR at the snapshot time.

### Foundations, not upcoming fixes

- [Electron PR #42953](https://github.com/electron/electron/pull/42953) is the
  merged GPU shared-texture OSR feature that exposes Linux NativePixmap
  metadata. It is the API this demo already uses; it does not select the new
  Chromium native-handle preference.
- The earlier Chromium capture-texture work cited by this demo established the
  shared-texture path, but does not fix NVIDIA's allocation choice.

### Not a solution to this allocation problem

- Electron issue #49247 concerned paint events not firing in an NVIDIA/Wayland
  setup and is closed. It is a different symptom from receiving no usable
  shared texture.
- Electron's `sharedTexture` import API imports external textures *into*
  Electron. It does not change the allocation used when exporting OSR capture
  frames.
- Vulkan/ANGLE/Ozone changes can improve backend compatibility, but no active
  change found in those areas selects
  `kPreferSharedImageWithNativeHandle` for Electron OSR.

Searches of Chromium Gerrit and Electron's public issue/PR index found no
active Electron selector PR and no second Chromium CL that would make the
route flag-only. This is a point-in-time result, so it should be rechecked when
CL 8220427 lands or Electron issue #52618 receives maintainer activity.

## Recommended upstream end state

The robust no-shim design is:

1. land CL 8220427 with the format mismatch resolved or explicitly documented;
2. add an upstream Electron option that expresses “exportable native handle,
   no CPU access required,” initially for ARGB;
3. have Electron map that option to
   `kPreferSharedImageWithNativeHandle` and retain the current mappable default
   for compatibility;
4. ship an Electron release containing both commit hashes; and
5. keep the application's existing modifier-aware EGL import, CUDA completion,
   FD-passing, and release acknowledgment.

That removes the broad `LD_PRELOAD` allocation rewrite and the custom Chromium
build without silently changing unrelated GPU allocations or Electron callers.
