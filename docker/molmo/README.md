# Molmo Docker Smoke Image

This image reuses an existing AVP runtime image for patched FFmpeg, FFmpeg
libraries, and the built `pyplumber` extension. It only adds the Python GPU
stack needed by the Molmo/vLLM prototype plus CUDA NVRTC headers/libraries for
CuPy startup compilation.

Build from the repository root on an x86 NVIDIA host:

```bash
docker build \
  -f docker/molmo/Dockerfile \
  --build-arg AVP_BASE_IMAGE=fanpulse-recorder:latest \
  -t avplumber-molmo-test:latest \
  .
```

Run import/CUDA/kernel smoke tests:

```bash
docker run --rm --gpus all --ipc=host avplumber-molmo-test:latest
```

Run the same via compose:

```bash
cd docker-compose
docker compose --profile molmo build molmo-smoke
docker compose --profile molmo run --rm molmo-smoke
```

Run the mock graph smoke with mounted media:

```bash
cd docker-compose
MOLMO_SMOKE_INPUT=/path/to/input.mp4 \
AVP_MOLMO_RUN_GRAPH=1 \
docker compose --profile molmo run --rm molmo-smoke
```

The graph defaults to `AVP_MOLMO_BACKEND=mock` and writes
`/artifacts/molmo-smoke.ts`. That validates CUDA decode, `scale_cuda`,
CuPy preprocessing, async node initialization, metadata overlays, and NVENC
without requiring Molmo weights or the vLLM tensor-runner hook.
