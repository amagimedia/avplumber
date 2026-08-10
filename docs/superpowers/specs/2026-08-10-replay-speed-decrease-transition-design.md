# Replay Speed-Decrease Transition Design

Date: 2026-08-10

Related design: `docs/superpowers/specs/2026-08-10-replay-vod-player-design.md`

## Problem

The single-slot replay player emits visibly skipped frames after changing from
2x to 1x. A post-realtime probe reproduced the failure in 12 of 12 transitions:

```text
45, 47, 49, 49, 51, 51, ...
```

At accelerated playback, `input_rec` intentionally skips all-intra packets and
`force_fps` selects frames to retain a 25 fps output. The speed command updates
the input skip policy correctly, but frames discarded or already executing in
the decoder and downstream event loop cannot be reconstructed by timestamp
rescaling.

An exact-frame pause and seek did not eliminate the transition: decoder and
downstream work selected at the old cadence still arrived after the seek.

## Scope

Coordinate configured playback-speed changes that cross from above 100 percent
to 100 percent or below while playback is active. Preserve the existing direct
behavior for speed increases, zero-speed pause, and speed changes made while
already paused. Scrubbing behavior is unchanged in this fix.

Do not change framework nodes, graph management, pybind, or the control
protocol.

## Considered Approaches

1. **Backpressure before `force_fps` — selected.** A second existing `pause`
   node stops new frames between `speed_video` and `force_fps` without blocking
   its worker. Frames already selected at 2x drain normally; the speed change
   then resets `force_fps` before the gate resumes.
2. **Exact-frame pause, seek, and resume.** This replayed the requested frame,
   but stale decoder work still produced duplicates and gaps after resume.
3. **Flush or change generic speed-node behavior.** An immediate flush raced
   in-flight decoder/event-loop work, while a framework-wide barrier would be
   risky and disproportionate for this small replay demo.

The player does not attach `input_rec` to the speed team. All source frames are
decoded, `speed_video` scales their timestamps, and `force_fps` performs the
2x selection. This costs roughly 50 decoded frames per second for a 25 fps
source at 2x, but keeps consecutive source frames available at the boundary.

## Transition

The controller uses its observed-frame seam and a condition variable. For an
active transition from above 100 percent to 100 percent or below:

1. Request `pause replay_transition now` on the gate before `force_fps`.
2. Wait until the post-realtime observation sequence is quiet for two nominal
   frame periods, bounded by the slot control timeout.
3. Send the new signed `speed.set`; this retimes queued frames and resets
   `force_fps` while the gate remains closed.
4. Request a Janus IDR, then `resume replay_transition`.

The controller's condition waits release its state lock so the position probe
can publish observations. Only one controller operation executes the transition
at a time.

If the output does not become quiet or a command fails, the operation raises an
actionable error and reopens the transition gate.

## Testing

The existing confirmed seams remain in use:

- A controller test drives the public speed operation and verifies the literal
  command order: close gate, signed speed, IDR, reopen gate. The failure case
  verifies that the gate reopens.
- Direct speed changes while paused and speed increases retain their existing
  command sequences.
- The NVIDIA-host `--exercise-v2` path records every observation callback after
  a 2x to 1x transition. The next ten source frames must advance by exactly one
  without a jump.
- The complete replay Python suite and all existing v2 GPU exercises must pass.

No timing assertion includes NVENC, Janus, WebRTC, or display latency.
