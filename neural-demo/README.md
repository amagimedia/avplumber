# Neural Demo

Requires a Linux host with an NVIDIA GPU, NVIDIA driver `r580+`, NVIDIA
Container Toolkit, and `docker --gpus all`.

## Quick Start

Build the image:

```bash
neural-demo/build-neural-demo-image.sh \
  --models-onnx /path/to/models_onnx.tgz \
  --tensorrt /path/to/tensorrt-minimal-with-tools.tgz
```

The ONNX and TensorRT `.tgz` files are host-side inputs. They do not need to be
in this repository. The helper mounts the ONNX archive into the plan-generation
container, stages Docker build inputs under the temporary ignored
`neural-demo/.docker-build-inputs/` directory, writes generated plans and
TensorRT timing data to the ignored persistent cache under
`neural-demo/.model-build-cache/`, stages the cached plans for the final Docker
build, and then builds the runtime image from that staged plan tree.

Input MP4 files are not bundled with the image. The runner mounts the path
passed with `--input` into `/run/avp/input/`.

Run the default live demo (provide your own MP4):

```bash
neural-demo/run-neural-demo.sh \
  --example tracker \
  --mode live \
  --input /path/to/input.mp4
```

Use `tracker` when you want to see the inference overlays and understand what
the underlying models are detecting and tracking.

Use `metadata` for the full basketball analytics demo. This graph adds player
masks, torso masks, feet masks, team classification, possession and shot
metadata, the tactical court panel, and JSON/NDJSON sidecar dumps:

```bash
neural-demo/run-neural-demo.sh \
  --example metadata \
  --mode live \
  --input /path/to/input.mp4 \
  --output rtmp://your-server/app/stream \
  --artifacts-dir /path/to/artifacts
```

Use `tracker-cropped` when you want the alternative reframed output:

```bash
neural-demo/run-neural-demo.sh \
  --example tracker-cropped \
  --mode live \
  --input /path/to/input.mp4
```

Use `tracker_compositor` when you want the regular drawn tracker output plus a
looped picture-in-picture feed in the upper-left corner. The PiP source must be
supplied separately:

```bash
neural-demo/run-neural-demo.sh \
  --example tracker_compositor \
  --mode live \
  --input /path/to/input.mp4 \
  --pip-input /path/to/pip.mp4
```

If you do not pass `--output`, the demo streams to:

- default live output: `http://test-streamer-s3dev.aws-dev.intranet/steam_test/test`

The image build converts portable ONNX models into TensorRT plans for the
current GPU and bakes those plans into `/home/tensorrt` in the runtime image.
An NVIDIA GPU is mandatory for this demo. This setup was tested on an AWS
`g4dn.2xlarge` instance with a Tesla T4 70W. The no-OCR metadata live graph
held `25 fps` on that GPU with a 25 fps 1080p input. Treat OCR and other
per-frame additions as target-GPU-specific performance work.

## Most Useful Commands

VOD in, VOD out:

```bash
neural-demo/run-neural-demo.sh \
  --example tracker \
  --mode vod \
  --input /path/to/input.mp4 \
  --output /path/to/output/tracker.ts
```

VOD in, multi-quality HLS out:

```bash
neural-demo/run-neural-demo.sh \
  --example tracker \
  --mode hls \
  --input /path/to/input.mp4 \
  --output /path/to/output/hls
```

VOD in, live out with a custom local MP4:

```bash
neural-demo/run-neural-demo.sh \
  --example metadata \
  --mode live \
  --input /path/to/input.mp4 \
  --output rtmp://your-server/app/stream \
  --artifacts-dir /path/to/artifacts
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
- `--example metadata` is the full live basketball analytics graph. It includes
  player segmentation, torso and feet masks, torso-based team classification,
  possession and shot event metadata, metadata dumps, and the tactical court
  panel driven by court/pose homography. The committed live template does not
  run scoreboard OCR because it can push the T4 live graph below realtime.
- `--example tracker-cropped` uses the same model stack but outputs the cropped
  reframed version instead.
- `--example tracker_compositor` uses the drawn tracker output as the main
  canvas and adds a looped picture-in-picture feed (provided via
  `--pip-input`) in the upper-left with padding. This example is intended
  for `live` mode.
- `--mode live` means looped MP4 input plus live RTMP or SRT output.
- `--mode vod` means finite VOD input plus file output.
- `--mode hls` means finite VOD input plus a video-only HLS ladder under the
  output directory. It creates `index.m3u8`, plus `med`, `hd`, and `fhd`
  rendition playlists with 1-second keyframe-aligned segments.
- `--input` is required.
- `--pip-input` is required when `--example tracker_compositor`.
- `--artifacts-dir` persists sidecar files from live graphs, especially
  `metadata`. In `hls` mode it is mounted separately from the HLS output
  directory.
- In `live` mode, if `--output` is omitted, the demo streams to the default
  output URL above.
- In `vod` mode, `--output` is required.
- In `hls` mode, `--output` is required and must be a directory. HLS mode is
  available for `tracker` and `tracker-cropped`.
- The build helper prints the Docker commands and model cache path before
  executing them.
- The helper supports `--dry-run`.

## Models

The runtime image contains TensorRT `.plan` files. Plans are GPU-specific, so
the build helper takes a portable ONNX archive, runs `trtexec` with `--gpus all`,
and then copies the generated `/home/tensorrt` tree into the final image.

Plan generation is cached. The cache key includes the GPU architecture, ONNX
archive hash, TensorRT archive hash, and the model-build script hash. The first
build for a key generates plans and a shared TensorRT timing cache; later builds
reuse existing `.plan` files and skip TensorRT profiling. Plans are written
through temporary files before they enter the cache, so an interrupted first
build cannot leave a partial `.plan` that is later treated as valid. Use
`--rebuild-models` to overwrite a populated cache, `--cache-dir` to put the
cache somewhere else, and `--gpu-arch sm75` to bypass auto-detection when
needed.

The ONNX archive should contain:

- `ball_960x544.onnx`
- `basketball-players-full_960x544.onnx`
- `court-segmentation_960x544.onnx`
- `player-seg/player-seg_960x544.onnx`
- `pose-small/pose-small_960x544.onnx`
- `court-pose-4/court-pose.onnx`

Scoreboard OCR is optional. If the ONNX archive also contains
`en-ppocr-v4-rec/en_PP-OCRv3_rec.onnx` and
`en-ppocr-v4-rec/en_dict.txt`, the builder creates
`en-ppocr-v4-rec/en_PP-OCRv3_rec_48x320.plan` with a fixed
`x:1x3x48x320` TensorRT profile and copies the dictionary into the final image.
Only enable OCR in a custom graph after measuring the target GPU; in the
1080p live metadata graph it was enough to cause RTMP drops on a T4.

The final image must contain these generated files:

- `ball_960x544.plan`
- `court-segmentation_960x544.plan`
- `basketball-players-full_960x544.plan`
- `player-seg/player-seg_960x544.plan`
- `pose-small/pose-small.plan`
- `pose-small/pose-small_960x544.plan`
- `court-pose-4/court-pose.plan`

The TensorRT archive path is required and is not hardcoded in the Dockerfile.
The archive should contain:

- `*/include/*`
- `*/bin/trtexec`
- `*/lib/libnvinfer.so*`
- `*/lib/libnvinfer_plugin.so*`
- `*/lib/libnvonnxparser.so*`
- `*/lib/libnvinfer_builder_resource_smXX.so*` for the target GPU architecture

`trtexec` and the ONNX parser are used only while building plans. The final
runtime image still copies only `libnvinfer.so*`, `libnvinfer_plugin.so*`, and
the generated model files. The builder resource library is required only while
generating plans; a T4 needs `sm75`.

Create that archive from a full TensorRT install:

```bash
neural-demo/package-tensorrt-bundle.sh \
  --source /opt/tensorrt \
  --builder-resource sm75 \
  --output /path/to/tensorrt-minimal-with-tools.tgz
```

Use multiple `--builder-resource` flags, or `--builder-resource all`, only when
one TensorRT archive must support more than one GPU architecture.

Build with a custom image name:

```bash
neural-demo/build-neural-demo-image.sh \
  --models-onnx /path/to/models_onnx.tgz \
  --tensorrt /path/to/tensorrt-minimal-with-tools.tgz \
  --image registry.example/avplumber-neural-demo:t4
```

Force regeneration of cached plans:

```bash
neural-demo/build-neural-demo-image.sh \
  --models-onnx /path/to/models_onnx.tgz \
  --tensorrt /path/to/tensorrt-minimal-with-tools.tgz \
  --rebuild-models
```

## Raw Docker

Live demo without the helper:

```bash
docker run --rm --gpus all \
  -v /absolute/path/input.mp4:/run/avp/input/input.mp4:ro \
  -v /absolute/path/artifacts:/run/avp/output \
  -e AVP_EXAMPLE=metadata \
  -e AVP_MODE=live \
  -e AVP_INPUT=/run/avp/input/input.mp4 \
  -e AVP_OUTPUT=rtmp://your-server/app/stream \
  -e AVP_ARTIFACT_DIR=/run/avp/output \
  avplumber-neural-demo:latest
```

VOD output without the helper:

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

HLS output without the helper:

```bash
docker run --rm --gpus all \
  -v /absolute/path/input.mp4:/run/avp/input/input.mp4:ro \
  -v /absolute/path/hls:/run/avp/output \
  -e AVP_EXAMPLE=tracker \
  -e AVP_MODE=hls \
  -e AVP_INPUT=/run/avp/input/input.mp4 \
  -e AVP_OUTPUT=/run/avp/output \
  avplumber-neural-demo:latest
```

## Notes

- The image builds public upstream FFmpeg `n7.1.3` plus
  `deps/ffmpeg-patches/`.
- TensorRT comes from the local archive passed with `--tensorrt`.
- TensorRT plan generation requires `trtexec` and `libnvonnxparser.so*`, but
  the final runtime image does not copy those files.
- The image does not install NVIDIA drivers. The host must provide them.
