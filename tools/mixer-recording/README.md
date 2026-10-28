# Mixer demo recording

The [demo page](https://amagimedia.github.io/avplumber/demos/mixer/docs/) replays a 20-second
host capture in the matching mixer UI. Both encoded outputs are recorded at Janus's RTP input,
before Internet transport. The T-bar follows sampled mixer status, including a halfway hold,
completion and a return stroke. This is a reconstruction from recorded video and state, not a
browser screen recording or a measurement of WebRTC latency.

1. **Capture** on the host running the 192-input, 256-scene, 24-output 1080p30 show:
   `python3 capture_on_instance.py > capture.json`. It changes scenes, keys and the lever, then
   restores Program, Preview and keys. Avoid operating the show during the capture. Copy the
   two `.mjr` files it names out of Janus as `pgm.mjr` and `preview.mjr`, and save the matching
   control page (`/`) and `/outputs.js` as `ui.html` and `outputs.js`.
2. **Process** after capture, with python3, numpy and ffmpeg, in the directory holding those files:
   `python3 process.py`. It recognizes H.264/H.265, reconstructs access units, discards an incomplete
   final access unit, reports RTP timestamp steps and packet gaps, extracts the requested frames,
   and measures ticker motion. The Program MP4 preserves the encoder bitstream, with preroll
   and tail. Review `report.json`; constant RTP cadence alone does not prove that source frames
   or compositors met every deadline.
3. **Render** in a Playwright container in that directory: `node render.cjs`. Each frame uses
   recorded pictures, the latest sampled status and the matching UI, with an upright T-bar.
   `render-check.json` records the drawn grip and sampled position for verification.
4. **Encode** `ui-frames/%03d.png` at the captured frame rate, inspect the transition and hold,
   and extract a poster and scene stills. Do this after capture so offline processing does not
   compete with the live show during measurement.

`capture.json` contains the rate, frame count and timeline used by both downstream tools. Full
state and recording paths are private; publish only a sanitized configuration and timing summary.
Keep deadline/drop counters scoped to their measured interval. See
[publishing](../../.github/pages/README.md) for release assets and Pages updates.
