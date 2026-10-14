# Alternative: zero-copy export with a fixed, preallocated pool

This is an opt-in Chromium/Electron alternative to copy capture. It is not
enabled by the browser service's defaults. Chromium renders into exportable
buffers; AVP may still copy each incoming frame into its own CUDA storage.
That consumer choice is independent of Chromium's capture mode.

The `0001` patches preserve the last exporter implementation, including native
handles, acquire fences, transparent root capture, bounded leases, FIFO reuse,
and skipping a paint when the pool is exhausted. The Chromium `0002` patch
makes the allocation budget exactly **11**, shared by the exporter and capturer.
The separate, optional `0003` patch makes that budget **N**, selected at startup
with **11** as the default. Both variants preserve the same contract:

- All N backing buffers are allocated during the initial `Reshape`, before
  the first paint. A partial allocation failure fails initialization.
- `BeginPaint` only takes an existing free buffer. It cannot allocate, free,
  grow the pool, or allocate a scratch surface when exhausted.
- Returned buffers retain their allocation and enter the FIFO. Idle time,
  capture pressure and `DiscardBackbuffer` do not replace them.
- Incompatible buffers held across a reshape still occupy slots. Their actual
  release permits replacement, with at most N old plus new allocations.
  Size, format, alpha or color-space changes require consumer warm-up again.
- Context loss requires recovery; it cannot trigger allocations on the lost
  context. Normal matching-buffer releases do not touch GL state.

With `0003` applied and built, select the size at process launch with
`--enable-features=AvpZeroCopyOffscreenExport:buffers/6,RenderableMappableSharedImageForceScanout`.
Use `buffers/9` or `buffers/11` for those sizes; omitting the parameter uses 11.
The shared setting is clamped to 2..16. No recompile is needed to change it;
restart the browser workers and repeat consumer warm-up. A running window
never grows or shrinks its pool in response to load. N counts all export
buffers, including rendering and pending returns, with no extra spares.

## Applying and building

Target: Electron **44.5.1**, Chromium **152.0.7977.130**, after Electron's
normal Chromium patches have been applied. The recovered source revisions are
Chromium `1eca9e10915d9c2126a749946658a7b5ee44ea5a` and Electron
`19c601167d4ae75989938416c18ed81eb21c6020`.

Use a separate checkout. Apply this series instead of the copy-capture pool
patch; the two implementations modify overlapping capture code.
From Chromium's `src`, with an absolute path to this directory:

```sh
git apply --check <patch-dir>/chromium/0001-zero-copy-export.patch
git apply <patch-dir>/chromium/0001-zero-copy-export.patch
git apply --check <patch-dir>/chromium/0002-preallocated-pool.patch
git apply <patch-dir>/chromium/0002-preallocated-pool.patch
# Optional: allow startup selection of the pool size.
git apply --check <patch-dir>/chromium/0003-configurable-pool.patch
git apply <patch-dir>/chromium/0003-configurable-pool.patch
git -C electron apply --check <patch-dir>/electron/0001-zero-copy-export.patch
git -C electron apply <patch-dir>/electron/0001-zero-copy-export.patch
git -C electron apply --check <patch-dir>/electron/0002-shared-texture-invalidate.patch
git -C electron apply <patch-dir>/electron/0002-shared-texture-invalidate.patch
autoninja -C out/Release electron electron:electron_dist_zip
```

The Electron refresh patch requests a fresh shared-texture capture without
restarting the capturer, allowing static pages to cycle the complete FIFO.
Enable `AvpZeroCopyOffscreenExport` and
`RenderableMappableSharedImageForceScanout` only in the alternative build.
The exporter requires Linux/Ozone with Ganesh GL and working native-pixmap
allocation. Retain the NVIDIA GBM shim where the runtime requires it.
Set the content size explicitly after creating an offscreen window and verify
the delivered `codedSize`: native window creation can clamp a requested 1080p
window to the Wayland display's smaller work area.

## Consumer contract and readiness

Preallocation is not consumer registration. Before going on air, the consumer
must observe and register all **N distinct backing buffers per window**, then
verify another rotation introduces no new buffers. Use static invalidation to
drive that warm-up. A timeout must fail readiness, not accept a partial pool.
Keep the import cache at least N per window with idle expiry disabled.
The delivery queue and mixer deadline are separate bounds; neither changes
the producer's allocation count.

The exporter can deliver an unfinished GPU frame with an `acquireFenceFd`.
Wait for that fence before sampling or copying, and acknowledge the frame only
after GPU use is complete. With AVP's CUDA-copy ingestion, release the Chromium
lease when that copy completes; the mixer retains only AVP-owned storage.
Drop the input reference before a potentially blocking output-queue write.
Enqueuing the copy is not completion. This shortens the producer lease without
reducing the fixed pool size.

The default copy-capture service no longer forwards
this fence protocol: installing this archive alone does not make that service
a compatible zero-copy consumer. Do not advertise fixed-pool readiness
until capture-mode selection, fence handling and the readiness gate
are wired to this alternative.

## Validation

Run `fixed-pool-probe.cjs` with the alternative Electron on the Linux GPU host:

```sh
<electron-dist>/electron --no-sandbox <patch-dir>/fixed-pool-probe.cjs
# These overrides require the optional 0003 patch in the binary.
AVP_PROBE_POOL_SIZE=6 <electron-dist>/electron --no-sandbox <patch-dir>/fixed-pool-probe.cjs
AVP_PROBE_POOL_SIZE=9 <electron-dist>/electron --no-sandbox <patch-dir>/fixed-pool-probe.cjs
```

The probe checks static warm-up, repeated FIFO reuse, exhaustion while every
buffer is held, recovery after releasing one, batched returns, and close with
outstanding leases. It identifies DMA-BUF allocations by device and inode,
not by reusable file-descriptor numbers. The default run requires exactly 11
allocations; the environment override tests a different startup size using
the same binary.

Also inspect the producer log: the full-size `pool preallocated: N` event
must precede the first paint, and the allocation/free counts must stay fixed
during pressure and recovery. These producer tests do not establish mixer
performance or replace a consumer import/fence test and short scene-transition
test before deployment.

The fixed-11 build passed the 1920×1080 producer probe: 69 delivered frames,
11 distinct DMA-BUFs, no bitmap fallback, and successful exhaustion and return
recovery. The optional `0003` startup-size patch is application-checked but has
not yet been compiled or GPU-tested. The fixed-11 binary ignores `buffers/N`;
build once with `0003` before using startup size changes without recompilation.
