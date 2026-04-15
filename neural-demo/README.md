# Neural Demo

Requires a Linux host with an NVIDIA GPU, NVIDIA driver `r580+`, NVIDIA
Container Toolkit, and `docker --gpus all`.

## Quick Start

Build the image:

```bash
docker build -f neural-demo/Dockerfile -t avplumber-neural-demo:latest .
```

Run the default live demo:

```bash
neural-demo/run-neural-demo.sh \
  --example tracker \
  --mode live
```

Use `tracker` when you want to see the inference overlays and understand what
the underlying models are detecting and tracking.

Use `tracker-cropped` when you want the alternative reframed output:

```bash
neural-demo/run-neural-demo.sh \
  --example tracker-cropped \
  --mode live
```

Use `tracker_compositor` when you want the regular drawn tracker output plus a
looped picture-in-picture `nba.mp4` feed in the upper-left corner:

```bash
neural-demo/run-neural-demo.sh \
  --example tracker_compositor \
  --mode live
```

If you do not provide custom input or output, the demo uses:

- looped input MP4:
  `https://tellyo-docker-dev-images.s3.eu-west-1.amazonaws.com/neural-demo-models/bbl.mp4`
- backup input MP4:
  `https://tellyo-docker-dev-images.s3.eu-west-1.amazonaws.com/neural-demo-models/nba.mp4`
- default model archive:
  `https://tellyo-docker-dev-images.s3.eu-west-1.amazonaws.com/neural-demo-models/models.tar.gz`
- default live RTMP output:
  `rtmp://ingest-1.tellyo.com/external/nabai2026920514b2`

That default `bbl.mp4` is the preselected demo source and works well with the
bundled inference model set.
Keep `nba.mp4` as a public fallback input URL if you want the previous demo
source.
Those demo models are open source and were borrowed from Roboflow.
An NVIDIA GPU is mandatory for this demo. This setup was tested on an AWS
`g4dn.2xlarge` instance with a Tesla T4 70W, which is a minimum sensible
option for this workflow, running about `30 fps` in live mode and about
`78 fps` in VOD mode.

## Most Useful Commands

VOD in, VOD out:

```bash
neural-demo/run-neural-demo.sh \
  --example tracker \
  --mode vod \
  --output /path/to/output/tracker.ts
```

VOD in, live out with a custom local MP4:

```bash
neural-demo/run-neural-demo.sh \
  --example tracker \
  --mode live \
  --input /path/to/input.mp4 \
  --output rtmp://your-server/app/stream
```

VOD in, live out with SRT:

```bash
neural-demo/run-neural-demo.sh \
  --example tracker-cropped \
  --mode live \
  --input /path/to/input.mp4 \
  --output srt://dest.example:9001?mode=caller
```

## Defaults And Options

- `--example tracker` draws the inference overlays so you can understand how
  the models behave.
- `--example tracker-cropped` uses the same model stack but outputs the cropped
  reframed version instead.
- `--example tracker_compositor` uses the drawn tracker output as the main
  canvas and adds a looped `nba.mp4` picture-in-picture feed in the upper-left
  with padding. This example is intended for `live` mode.
- `--mode live` means looped MP4 input plus live RTMP or SRT output.
- `--mode vod` means finite VOD input plus file output.
- In `live` mode, if `--input` is omitted, the demo loops the public `bbl.mp4`.
- `nba.mp4` remains available as the documented fallback URL.
- In `live` mode, if `--output` is omitted, the demo streams to the default
  public RTMP endpoint above.
- In `vod` mode, `--output` is required.
- The helper prints the final `docker run` command before executing it.
- The helper supports `--dry-run`.

## Models

The runtime downloads a model tarball and normalizes it into `/home/tensorrt`.
The demo model tarball is open source and borrowed from Roboflow.
It requires these three plan files:

- `ball_960x544.plan`
- `court-segmentation_960x544.plan`
- `basketball-players-full_960x544.plan`

Use a different model archive if needed:

```bash
neural-demo/run-neural-demo.sh \
  --example tracker \
  --mode vod \
  --output /path/to/output/tracker.ts \
  --models-tar-url https://example.com/other-models.tgz
```

## Raw Docker

Default live demo without the helper:

```bash
docker run --rm --gpus all \
  -e AVP_EXAMPLE=tracker \
  -e AVP_MODE=live \
  avplumber-neural-demo:latest
```

VOD output without the helper:

```bash
docker run --rm --gpus all \
  -v /absolute/path/output:/run/avp/output \
  -e AVP_EXAMPLE=tracker \
  -e AVP_MODE=vod \
  -e AVP_OUTPUT=/run/avp/output/tracker.ts \
  avplumber-neural-demo:latest
```

## Notes

- The image builds public upstream FFmpeg `n7.1.3` plus
  `deps/ffmpeg-patches/`.
- TensorRT comes from the minimal public archive used by this demo:
  `https://tellyo-docker-dev-images.s3.eu-west-1.amazonaws.com/neural-demo-models/tensorrt-minimal-10.15.1.29.tar.gz`
- The image does not install NVIDIA drivers. The host must provide them.
