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

Reducing queue capacity, disabling the input optimization, and issuing an
immediate zero-offset seek did not eliminate the transition. The latter two
reduced parts of the transient but still allowed old selections to re-enter
after apparently correct frames.

## Scope

Coordinate configured playback-speed changes that cross from above 100 percent
to 100 percent or below while playback is active. Preserve the existing direct
behavior for speed increases, zero-speed pause, and speed changes made while
already paused. Scrubbing behavior is unchanged in this fix.

Do not change `input_rec`, decoders, `speed_video`, `force_fps`, graph
management, pybind, or the control protocol.

## Considered Approaches

1. **Coordinated pause, exact-frame seek, and resume — selected.** This is the
   only approach that stops displaying old selections and reconstructs the
   source sequence from the all-intra seek table. It adds a short intentional
   freeze while the transition completes.
2. **Disable `input_rec` fast-frame skipping.** This raises NVDEC work to 50 fps
   at 2x and shortened the observed transient to about 200 ms, but three old
   selected frames still escaped downstream.
3. **Flush or change generic speed-node behavior.** An immediate flush raced
   in-flight decoder/event-loop work, while a framework-wide barrier would be
   risky and disproportionate for this small replay demo.

## Transition

The controller uses its existing observed-frame seam and a condition variable.
For an active transition from above 100 percent to 100 percent or below:

1. Record whether playback was active and request `pause <pause-team> now`.
2. Wait until the post-realtime observation sequence is quiet for two nominal
   frame periods, bounded by the slot control timeout.
3. Capture the last observed absolute `frame_no`.
4. Send the new signed `speed.set` before seeking, so `input_rec` updates its
   fast-frame skip policy before reading from the new position.
5. Send `seek <sync-team> frame <frame_no>` and wait for a new observation of
   that exact frame while paused.
6. Resume only when playback was active before the transition.

The controller's condition waits release its state lock so the position probe
can publish observations. Only one controller operation executes the transition
at a time. The existing command wrapper forces a Janus IDR for the absolute
seek.

If the pause does not become quiet or the exact seek frame is not observed
within the configured timeout, the operation raises an actionable error and
leaves playback paused. It must not resume from an uncertain position.

## Testing

The existing confirmed seams remain in use:

- A controller test drives the public speed operation, publishes deterministic
  observations from a command adapter, and verifies the literal command order:
  pause, signed speed, absolute frame seek, resume.
- Direct speed changes while paused and speed increases retain their existing
  command sequences.
- The NVIDIA-host `--exercise-v2` path samples observed frame numbers after a
  2x to 1x transition. Once the coordinated operation returns, the next ten
  changing source frames must advance by exactly one without a jump.
- The complete replay Python suite and all existing v2 GPU exercises must pass.

No timing assertion includes NVENC, Janus, WebRTC, or display latency.
