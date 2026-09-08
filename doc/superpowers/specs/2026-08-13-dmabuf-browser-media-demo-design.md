# DMA-BUF Browser Documentation Media Demo

## Goal

Add a representative screenshot and a ten-second 1920x1080@60 MP4 to the
`dmabuf-browser` README. The media must demonstrate the live 16-input grid made
from two Electron processes with eight windows each.

## Capture path

The recording must use the production GPU path rather than recording the Janus
preview:

```text
16 DMA-BUF inputs
-> cached EGLImage/CUDA resources
-> egl_image_cuda_overlay
-> CUDA RGB0 output frame
-> h264_nvenc
-> encoded H.264 packets
-> file muxer
```

There is no `hwdownload`, CPU pixel copy, screen capture, Janus, RTP, or WebRTC
in this path. Only compressed H.264 packets leave GPU processing for the file
muxer.

## Graph interface

Add `OUTPUT_FORMAT` and `OUTPUT_URL` environment overrides to the scaling demo.
Their defaults retain the current Janus RTP behavior exactly. A finite
documentation run selects MPEG-TS output and `TEST_DURATION_SEC=10` because
MPEG-TS remains usable when the diagnostic process takes its existing fast
finite-run exit. Remux the H.264 stream to MP4 with stream copy; do not decode
or re-encode it.

This is a demo-graph change only. Do not modify graph management, the control
protocol, the compositor, the DMA-BUF receiver, or the NVENC chain.

## Timestamp ownership

Do not insert `smooth_timestamps` into the 16 production input branches. Each
browser uses the same host monotonic clock, and each branch independently
delivers its latest DMA-BUF to `egl_image_cuda_overlay`. The compositor owns the
single program clock: it renders on a steady 60 Hz schedule and emits consecutive
PTS values in a `1/60` time base. This is already the smoothing boundary and
allows one late input to reuse its previous image without stalling the other
inputs.

The legacy CUDA diagnostic path retains its existing timestamp normalization
and output `smooth_timestamps` node. Adding sixteen independent pacing nodes to
the production EGL path would add queues and undermine the independent-input
design.

## Documentation assets

Commit the final MP4 and a PNG extracted from its midpoint under
`demos/dmabuf-browser/docs/`. In the scaling-test section, show the PNG as a
clickable link to the MP4 and label the exact configuration: 16 independent
1920x1080@60 browser inputs, two Electron workers, cached DMA-BUF/EGL/CUDA
interop, one CUDA compositor, and one NVENC output.

## Validation

- Before recording, run the existing per-input `mpdecimate` diagnostic for all
  16 sources. Compare cumulative reports after the initial ten-second warmup so
  startup does not contaminate the result. Report the post-warmup input rate,
  unique-frame rate, and duplicates for every source rather than only an
  aggregate.
- In the production EGL/CUDA path, take browser status snapshots over a fixed
  post-warmup interval and confirm every input advances at 60 fps with no new
  browser drops. Confirm the compositor advances by 600 frames per ten seconds,
  with no additional missed deadline after warmup.
- Confirm the compositor owns the `1/60` program time base and consecutive PTS;
  do not claim that per-input `smooth_timestamps` nodes exist in this path.
- Confirm the direct-file run uses `COMPOSITOR_BACKEND=egl_cuda`, 16 sources,
  two browser workers with eight windows each, and no Janus output.
- Use `ffprobe` to verify H.264, 1920x1080, `60/1`, approximately ten seconds,
  and approximately 600 frames.
- Confirm the MP4 is a stream-copy remux of the captured H.264 packets.
- Inspect the midpoint PNG and sampled video frames for the complete 4x4 grid,
  correct aspect ratio, and visible motion. Include both assets and the measured
  post-warmup validation in the README.
- Restart the normal Janus composer after recording and confirm 16 live windows
  and a healthy preview endpoint.
