# Mixer demo recording

The current [demo page](https://amagimedia.github.io/avplumber/demos/mixer/docs/) uses a
screen recording of the **actual live UI**, including both WebRTC viewers and the T-bar.
Record the controls and pictures together; do not reconstruct the UI from sampled state.

## Live UI capture

`capture_ui.cjs` runs in a dedicated Linux desktop with a 1920×1080 X display, a window
manager, Playwright, Chrome and FFmpeg with x11grab/libx264. The browser must support both
outputs' H.264 profiles. Software-only Chrome may play Program but reject the High-profile
AUX viewer; verify hardware decoding before recording. This setup belongs to the capture
browser and does not require rebuilding or restarting the mixer.

```sh
MIXER_UI_URL=http://127.0.0.1:7681/ DISPLAY=:99 node capture_ui.cjs ui-capture
```

Optional environment variables: `CHROME_EXECUTABLE`, `CHROME_ARGS` (a JSON array), and
`MIXER_HTTP_USER` / `MIXER_HTTP_PASSWORD` for a protected proxy. Keep credentials out of Git.

The script selects the two viewers and upright T-bar using browser preferences, checks
that both streams decode, then operates scene buttons and the T-bar with real mouse input.
It records 20 seconds at 30 fps with cuts, a halfway hold and a return stroke. It restores
Program, Preview and keys afterward if the show has not changed. A setup revision change
aborts the capture and leaves the new deployment alone. Avoid other show controls during
the short recording window.

Review `live-ui.mp4` and the before/after playback counters in `capture.json`. Full state and
URLs in that report are private: publish only a sanitized capture summary. Verify frame
count, rate, dimensions, both moving viewers and the actual T-bar fade before replacing the
[release assets and Pages links](../../.github/pages/README.md).

## Earlier host-output recordings

`capture_on_instance.py`, `process.py` and `render.cjs` implement the older workflow: record
Program and Program/Preview RTP inputs at Janus before Internet transport, extract pictures,
and replay sampled state in the control page. That workflow preserves the native Program
bitstream but **does not capture live UI operation**. It remains available for reproducing the
historical recordings; use `capture_ui.cjs` for the current documentation video.
