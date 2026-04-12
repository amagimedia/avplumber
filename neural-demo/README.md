# Neural Demo

This directory contains a public Fedora 42 Docker flow for the neural avplumber
demo.

The image is built from:

- public upstream FFmpeg tag `n7.1.3`
- local patch stack from `deps/ffmpeg-patches/`
- avplumber built against that patched FFmpeg in `/usr/local`

The image does not install NVIDIA drivers. The host must provide the driver and
GPU runtime integration.

TensorRT for this image is pulled from a minimal public archive containing only
the headers and runtime libraries needed by this demo:

- `https://tellyo-docker-dev-images.s3.eu-west-1.amazonaws.com/neural-demo-models/tensorrt-minimal-10.15.1.29.tar.gz`

## Host Prerequisites

- Linux host with NVIDIA GPU
- NVIDIA driver branch `r580+`
- NVIDIA Container Toolkit or equivalent OCI GPU integration
- `docker` with `--gpus all` support

Validated reference host:

- Fedora 42
- NVIDIA driver `580.126.18`
- `nvidia-smi` reports `CUDA Version: 13.0`

## Required Model Plans

The runtime downloads a model tarball and normalizes it into `/home/tensorrt`.
It requires these three plan files:

- `ball_960x544.plan`, about `20.3 MiB`
- `court-segmentation_960x544.plan`, about `22.2 MiB`
- `basketball-players-full_960x544.plan`, about `20.2 MiB`

Total size is about `62.7 MiB`.

Default model tarball URL used by the helper and image:

- `https://tellyo-docker-dev-images.s3.eu-west-1.amazonaws.com/neural-demo-models/models.tar.gz`

You can override it with `--models-tar-url` or by setting `AVP_MODELS_TAR_URL`
directly in a raw `docker run` command.

## Build

```bash
docker build -f neural-demo/Dockerfile -t avplumber-neural-demo:latest .
```

## Helper Script

The recommended interface is:

```bash
neural-demo/run-neural-demo.sh --help
```

The helper prints the final `docker run` command before it executes it.

## VOD Examples

Tracker:

```bash
neural-demo/run-neural-demo.sh \
  --example tracker \
  --mode vod \
  --input /path/to/input.mp4 \
  --output /path/to/output/tracker.ts
```

Tracker cropped:

```bash
neural-demo/run-neural-demo.sh \
  --example tracker-cropped \
  --mode vod \
  --input /path/to/input.mp4 \
  --output /path/to/output/tracker-cropped.ts
```

## Live Examples

RTMP:

```bash
neural-demo/run-neural-demo.sh \
  --example tracker \
  --mode live \
  --input rtmp://source.example/live/in \
  --output rtmp://dest.example/live/out
```

SRT:

```bash
neural-demo/run-neural-demo.sh \
  --example tracker-cropped \
  --mode live \
  --input srt://source.example:9000?mode=caller \
  --output srt://dest.example:9001?mode=caller
```

Override the model archive URL when needed:

```bash
neural-demo/run-neural-demo.sh \
  --example tracker \
  --mode vod \
  --input /path/to/input.mp4 \
  --output /path/to/output/tracker.ts \
  --models-tar-url https://example.com/other-models.tgz
```

## Raw Docker Example

VOD tracker example without the helper:

```bash
docker run --rm --gpus all \
  -v /absolute/path/input.mp4:/run/avp/input/input.mp4:ro \
  -v /absolute/path/output:/run/avp/output \
  -e AVP_EXAMPLE=tracker \
  -e AVP_MODE=vod \
  -e AVP_INPUT=/run/avp/input/input.mp4 \
  -e AVP_OUTPUT=/run/avp/output/tracker.ts \
  avplumber-neural-demo:latest
```

## Notes

- The helper supports `--dry-run`.
- VOD input files are mounted read-only.
- VOD outputs and sidecar debug files are written into the host output
  directory you pass.
- Live mode picks protocol handling from the URL schemes you provide.
- For live troubleshooting, you can add Docker flags with repeated
  `--docker-extra ...` options.
- Docker bridge networking is the default. Use host networking only if your
  environment needs it.
