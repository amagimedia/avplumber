# Mixer demo recording

The [demo page](https://amagimedia.github.io/avplumber/demos/mixer/docs/) shows 20 seconds
of the develop mixer with two small viewers, an upright T-bar and the performance bar.
Native pictures are captured on the host **before Janus**; the actual control page replays
recorded controls and measured statistics offline at 30 fps. The UI viewport is 1328×734
CSS pixels at 2× Retina scale, yielding a 2656×1468 video. Use macOS Chrome to reproduce
the reference's system fonts.

## Capture on the instance

Use a dedicated Playwright/Chrome environment and the regular 192-input, 1080p30 show.
Start a packet capture before running the control recorder, and stop it afterward:

```sh
sudo tcpdump -i lo -U -s 0 -B 8192 -w native.pcap \
  'udp and (dst port 5004 or dst port 5008)'
```

In another terminal, from the capture directory:

```sh
MIXER_UI_URL=http://127.0.0.1:7681/ node <repo>/tools/mixer-recording/capture_ui.cjs .
```

The ports are the existing Program and Program/Preview encoder destinations. Adjust the
filter and decoder's port mapping together if the deployment differs. The parser expects
an Ethernet/IPv4 pcap, as produced by Linux loopback tcpdump. Set `MIXER_HTTP_USER` and
`MIXER_HTTP_PASSWORD` if authentication is required; never store credentials in Git.

The recorder blocks WebRTC playback in its own browser to avoid capture-time decode load.
It operates the real scene buttons and T-bar with mouse input, records the control trace
and state, and restores Program, Preview and keys only if the setup revision is unchanged.
It records a fade to 50%, a hold and completion, a return stroke and a cut with all keys on.
Stop tcpdump with SIGINT after the recorder exits, then release the host. Full capture
state contains private paths and URLs; publish only sanitized metadata.

## Process and render offline

Copy `native.pcap`, `capture.json`, and the exact deployed UI's `index.html` and `outputs.js`
to a local working directory, naming the HTML `ui.html`. Install Python numpy, FFmpeg,
Node Playwright and Chrome. Run from that directory:

```sh
python3 <repo>/tools/mixer-recording/process.py
node <repo>/tools/mixer-recording/render.cjs
ffmpeg -framerate 30 -i ui-retina/%03d.jpg -frames:v 600 \
  -c:v libx264 -preset fast -crf 16 -pix_fmt yuv420p -movflags +faststart demo.mp4
```

`CHROME_CDP_URL` optionally connects the renderer to an existing dedicated Chrome instance.
The renderer uses the real page's layout, labels and controls, inserts native frames beneath
the viewer overlays, and replays measured CPU/GPU statistics. It does not invent performance
numbers or WebRTC latency. The viewer footer identifies host capture. Timing is explicit per
frame, so slow offline rendering cannot introduce playback stalls.

Check `report.json`: both streams must have zero RTP packet gaps and continuous 3000-tick
steps at 30 fps. The ticker's 599 frame-to-frame displacements should all be −4 pixels for
this show. Also inspect the halfway hold, return stroke, complete performance bar and final
MP4 dimensions, duration and frame count. This recording is not a new capacity benchmark.
Publish the MP4 as a [release asset](../../.github/pages/README.md), not a Git blob, and label
the UI and metrics as replayed in the public capture report.

`capture_on_instance.py` remains the historical ten-second Janus-input recorder. Its older
capture schema is not the control trace consumed by the current renderer.
