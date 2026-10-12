# Mixer demo recording

How the ten-second recording on the [demo page](https://amagimedia.github.io/avplumber/demos/mixer/docs/)
is made. The outputs are recorded on the instance, so the result shows the mixer's own cadence and
not a network path or a screen recorder. See [publishing](../../.github/pages/README.md) for where
the results go.

1. **Capture**, on the host that runs the mixer stack:
   `python3 capture_on_instance.py > capture.json`. Copy the two `.mjr` files it names out of the
   Janus container as `pgm.mjr` and `preview.mjr`, and save the control page (`/`) and
   `/outputs.js` as `ui.html` and `outputs.js`.
2. **Process**, anywhere with python3, numpy and ffmpeg, in the directory holding those files:
   `python3 process.py`. It rebuilds both elementary streams from the RTP packets, checks that the
   RTP timestamps step by exactly one frame with no packet gap, extracts the ten seconds as
   frames, finds the frames the cuts land on and measures the ticker's step between frames. It
   writes `report.json` and the Program bitstream as an MP4 without re-encoding.
3. **Render**, in a Playwright container in the same directory: `node render.cjs`. Each output
   frame is the real control page drawn with the recorded mixer state and the two recorded
   pictures of that instant, written to `ui-frames/`.
4. **Encode** `ui-frames/%03d.png` at the show's rate with an H.264 encoder, take the poster and the
   scene stills from the frames, and record the figures of `report.json` in the capture JSON.

`process.py` and `render.cjs` carry the frame rate and the scene sequence at the top; keep them in
step with `capture_on_instance.py`.
