# Playlist Regression Container

Date: 2026-08-12

## Purpose

Provide a playlist-specific container image whose five deterministic regression
clips are immutable image contents. Recreating a container from the image must
not depend on temporary checkouts, host media, or generating clips at startup.

This image packages the existing playlist harness. It does not define a second
AVPlumber build, alter framework behavior, or add the fixtures to the general
AVPlumber runtime image.

## Build contract

The Docker build context is `demos/playlist/`. The Dockerfile has three stages:

1. An Alpine 3.21 fixture stage installs Bash, FFmpeg, and DejaVu fonts. It runs
   `test-media/generate.sh`, which produces and validates the five H.264,
   1920x1080, 30 fps, 300-frame, ten-second clips.
2. A Python dependency stage installs the pinned, pure-Python playlist
   requirements into a standalone target directory.
3. The final stage derives from the required `AVP_BASE_IMAGE` build argument.
   That base image provides a CUDA-enabled `pyplumber` matching its AVPlumber
   runtime and Python 3. The final image copies the dependency directory,
   playlist application, and generated MP4s under
   `/opt/avplumber/demos/playlist/`.

The final stage prepends the copied pure-Python dependency directory to the
base image's `PYTHONPATH`, verifies that `pyplumber` is discoverable without
loading the GPU binding, imports Textual, and checks exactly five fixture files.
It neither requires package-manager/root access to the base nor installs or
rebuilds AVPlumber, CUDA, FFmpeg, or TensorRT.

The build fails if FFmpeg lacks the required test sources, drawtext support, or
H.264 encoder; if fixture validation fails; or if the final base does not expose
the required Python runtime and `pyplumber` module.

## Image contents and execution

The runtime working directory is `/opt/avplumber/demos/playlist`. The image
entrypoint runs `python3 player.py`. Default arguments target Janus at
`127.0.0.1:5004/5005` and write framework logs to `/tmp/playlist-demo.log`;
callers may override those arguments.

A live container is run interactively with host networking and NVIDIA GPU
access. The default media directory remains `test-media` next to `player.py`, so
no media mount or machine-specific path is needed. Containers recreated from
the same image see byte-identical clips. Rebuilding the image recreates clips
deterministically from the committed generator.

## Build context hygiene

A playlist-local `.dockerignore` excludes locally generated MP4s, temporary
files, Python caches, pytest caches, and unrelated mockup artifacts. Thus the
fixture stage is the only source of MP4 files in the image, and a developer's
working directory cannot silently replace them.

## Verification

Acceptance requires:

- the fixture stage builds independently and its generator validation passes;
- the final image builds from a real CUDA-enabled Python AVPlumber base;
- an image-level smoke check sees exactly five fixture files, imports Textual,
  and finds `pyplumber`; the real binding import runs on the GPU host;
- a dry-run container starts without host media;
- on the remote NVIDIA/Janus host, the container reaches `JANUS ALIVE`, the
  decoded 1920x1080 WebRTC preview advances, and the existing live playlist
  regression passes;
- deleting and recreating the container from the same image preserves the five
  fixture hashes.

No private hostnames, addresses, credentials, temporary paths, or base-image
names are committed. README examples use placeholders.
