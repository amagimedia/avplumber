# GStreamer CUDA comparison

This is the corrected Python harness used for the 129-input L4 qualification:
105 HEVC decoders, 16 resident raw NV12 clips, and eight live Chromium DMA-BUF
pages. The rendering graph and feeder logic are retained; local paths are now
command-line options. It uses stock GStreamer 1.28 CUDA/NVIDIA elements, with
Python `appsrc` feeders for raw memory and the browser capture protocol.

AVPlumber is a full switcher; this harness is a static composition graph.
129 is a validated operating point, not a proven GStreamer maximum.

## Requirements and workload

Use a Linux NVIDIA host with GStreamer 1.28, Python GObject introspection
(`Gst`, `GstCuda`, `GstAllocators`, `GstVideo`), CUDA driver libraries, and the
`nvcodec`, `opengl`, `app`, `proxy`, `rawparse`, `videoparsersbad` and RTP plugins.
The graph requires `nvh265dec`, `cudaupload`, `cudacompositor` and `nvh264enc`.
Keep this on a dedicated test environment: a DMA-BUF page socket must have only
one consuming mixer. The script does not stop or restart other services.

Prepare the [mixer demo](../README.md) and retain its show JSON and original
media. All source paths in the JSON must resolve inside the benchmark process.
The tested raw clips contain 60 complete 1920×1080 NV12 frames each; use these
original files, not a single-frame substitute. The pinned feeder reads each
whole clip into `cuMemHostAlloc` memory in GStreamer's CUDA context and wraps it
in GstBuffers through the public API. Every selected raw source is fed even
when `--light` samples instrumentation.

For each `video` source, extract its HEVC elementary stream without re-encoding:

```sh
ffmpeg -i INPUT -map 0:v:0 -c:v copy -bsf:v hevc_mp4toannexb -f hevc OUTPUT.hevc
```

Create a manifest with one entry per video source, using the source IDs from
the show and paths visible to the benchmark:

```json
{"sources": [{"id": "video_000", "es": "/media/video_000.hevc"}]}
```

Run the demo's Chromium capture workers separately with their DMA-BUF sockets
available in `--socket-dir`. The show needs `random_200` / `random_201`, its AUX
and thumbnail layouts, seven ordinary browser sources and `dsk_bug_right`.
`--browsers 7 --dsk bug_right` selects **eight total pages**, not nine.
For latency measurement, that key must render the timestamp barcode described
in `rtp_latency.py`; an ordinary graphic cannot produce latency samples.

## Validated invocation

```sh
GST_GL_PLATFORM=egl GST_GL_WINDOW=surfaceless python3 bench.py \
  --show /media/mixer.demo.json --sources /media/sources.json \
  --socket-dir /tmp/dma-page \
  --light --browser-source dmabuf --split-gl \
  --video 105 --raw 16 --raw-source pinned \
  --browsers 7 --dsk bug_right --aux 20 --thumbs 3 \
  --seconds 50 --warmup 20 --rtp-port 5004 --latency-ms 33
```

The `RAW_MANIFEST` record identifies the loaded clips and frame hashes. The
`SUMMARY` record reports every browser's frame rate, every raw feeder's push
rate and output rates. Program and browser inputs must hold nominal 30 fps
(counting-window tolerance: 29.9 fps); the three thumbnail banks run at 15 fps.
A raw feeder push is not proof that no downstream frame was dropped.
`--light` disables queue-drop counting: its zero drop counters are not measured
zero drops. Do not use `--dmabuf-cache` for this result; that experimental path
stalled in the tests. `--browser-source gl` uses stand-ins and is not equivalent
to real Chromium pages.

## Browser-to-RTP latency

After warm-up, capture 20 seconds of program RTP on the host:

```sh
sudo timeout -s INT 20 tcpdump -i lo -s 0 -n -w gst.pcap udp dst port 5004
python3 rtp_latency.py gst.pcap
```

The analyzer compares the timestamp painted in the browser with the kernel
arrival time of the last RTP packet of each encoded frame, using the same host
clock. Its default barcode position is x=1584, y=32; adjust flags to match the
show. This is browser-to-output latency, **not cut latency**. Use median and
p99 in comparisons and require both below 200 ms.

The retained 129-input run produced 85.4 ms median / 93.8 ms p99, program
30 fps, every browser 29.979–29.999 fps, and all 16 raw feeders at least
29.999 fps. The separate frozen AVPlumber reference carried 192 inputs at
70.8 ms median / 86.3 ms p99. These are unequal workloads, short captures and
different browser sessions on the same hardware, not a controlled same-session
overhead comparison. No replacement CPU, thread or VRAM measurements accompany
this corrected run.
