# Running demo snapshot

[`config.demo.json`](../config.demo.json) captures the deployed show inspected
on 2026-09-11. It preserves source IDs and order, all scene items, rectangles,
crops and layer order, the wipe library, control defaults and rendition settings.
Only browser URLs and source/wipe filesystem locations are replaced by
placeholders. The smaller [`config.example.json`](../config.example.json) remains
a configuration tutorial, not the deployed show.

## Workload

| Component | Configuration |
| --- | --- |
| Canvas | 1080×1920, 30 fps |
| Browser sources | Seven distinct 1920×1080 pages, requested at 30 fps |
| Video sources | Eight numbered 640×360 H.264 clips at 60 fps; one 1920×1080 Big Buck Bunny clip |
| Scenes | 16 fullscreen + 8 two-box + 4 four-box + 2 eight-box + 1 sixteen-box + 10 custom = 41 |
| Output | One Janus rendition, 1080×1920 at 30 fps, H.264 NVENC baseline/P7, 2 700 kbit/s |
| Wipes | Ribbons, Colour bars, Fedora, Dip black, Dip white |
| Wipe cache budget | 1 536 MiB, supplied on the command line; this is a ceiling, not measured allocation |
| Control defaults | Direct on, fade selected, 0.5-second fades, Ribbons as default wipe |

The source clips are re-timed to the canvas rate. `grid_16_page_0` contains each
of the sixteen unique sources exactly once. The twin/PiP scenes reuse source
frames through aliases, not additional decoders or browser windows.

`initial_scene` is `fullscreen_0`. Selecting the sixteen-box scene during a run
does not rewrite that startup setting. Direct mode means a scene pick goes to
Program using the selected transition; with these defaults that is a **fade**,
not a cut.

## Use the snapshot

Copy the JSON to your deployment configuration and replace every `https://<host>/…`
browser URL and `<path>/…` media location. Browser locations must remain distinct.
Use paths as seen by the mixer process, and keep the input sizes above when
comparing layouts or performance. The fixed crop rectangles assume those sizes.
Media and browser-page contents are not bundled; replacing them reproduces the
show structure, not necessarily its pixels or GPU load.

This mixed browser/video show needs the CUDA **and DRM/GL** build from the
[DMA-BUF stack](../../dmabuf-browser/README.md), a shared browser socket directory,
and a browser service that allows at least seven 1920×1080 windows. The video-only
mixer Dockerfile disables DRM/GL and is not sufficient for this configuration.
Keep the graph web UI backend running with registration enabled.

With those services available on a separate NVIDIA deployment, the mixer command
is equivalent to:

```sh
python3 demos/mixer/mixer.py --config '<path>/mixer.json' \
  --janus-output --janus-host 127.0.0.1 --janus-video-port 5004 \
  --janus-video-bitrate-kbps 3000 --remote-control-port 7777 \
  --dmabuf-rest http://127.0.0.1:9009 --dmabuf-socket-dir /tmp/dma-page \
  --wipe-cache-mb 1536 --webui-url http://127.0.0.1:22222
```

The JSON rendition's 2 700 kbit/s takes precedence over the 3 000 kbit/s CLI
fallback. Start the browser control surface separately:

```sh
python3 demos/mixer/webui.py --host 127.0.0.1 --port 7777 --http-port 7681
```

These commands are for a separate deployment, not a second process sharing the
live demo's ports, browser window IDs or sockets. No live container, image or
mount was changed to publish this snapshot.

## Code and prewarming

The deployed mixer entry point, graph builder, startup prewarm helpers, input
chains and Janus output code matched the public source at `b13ba74`. All native
source files still tracked in that revision matched the inspected remote build
tree; the deployed native module also matched that build's binary artifact.
This is a source audit, not a reproducible-image claim: the running container uses
copied/bind-mounted files and does not identify a complete deployed Git revision.

Some deployed control helpers use the older module locations; the public code
already consolidates the same helpers in `avpmixer.control`. Those older copies,
stale test assertions and intentionally removed proprietary nodes are not restored
as part of this snapshot.

Startup prewarming initializes inputs, compositor slots and transition processing;
wipe clips are also cached. It does **not** continuously render all 41 scenes or
retain a ready-to-cut picture for every scene. A direct cut to a newly loaded
scene still resets that slot's frame history. Taking an already prepared Preview
is a different path.

The separate A→B experiment retaining input histories is **not enabled in this
demo or implemented as a production mixer feature**. Its latency results and the
estimated memory cost of extending it to all sources must not be presented as
performance of this deployed configuration.
