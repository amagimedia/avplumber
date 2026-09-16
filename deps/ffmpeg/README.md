# FFmpeg patch series

Each version directory contains a complete series for one exact upstream tag.
Do not apply the 7.1.5 series and then the 8.1 series to the same checkout.

| Directory | Upstream tag | Purpose |
| --- | --- | --- |
| `7.1.5/` | `n7.1.5` | Existing default; patch contents preserved unchanged. |
| `8.1/` | `n8.1` | Compatibility port with 8-bit CUDA mixer runtime checks; see its README for coverage. |

The mixer, CUDA-overlay and DMA-BUF CUDA consumer Dockerfiles select the series
using their existing `FFMPEG_TAG` argument. Their default remains `n7.1.5`.
An isolated 8.1 mixer build can be requested from the repository root with:

```bash
docker build --build-arg FFMPEG_TAG=n8.1 \
  -f demos/mixer/Dockerfile -t avplumber-mixer:ffmpeg8.1 .
```

Build on an NVIDIA development host, with the required submodules populated.
This command creates a separate image; it does not replace a running container.
avcpp and avplumber must be rebuilt against the selected FFmpeg libraries.
Keep the current avcpp revision unless a verified compatibility issue requires
a change. The mixer graph remains 8-bit NV12; this port does not add MXL or
10-bit composition. AVPlumber's [v210 upload node](../../doc/v210_to_cuda.md)
can feed raw 10-bit 4:2:2 into the 8.1 CUDA frame/scaling path, independently
of the mixer and without another FFmpeg patch.

Each version's `base.env` pins the upstream commit, patched tree and patch count.
Verify either series without changing the source checkout's branch:

```bash
deps/ffmpeg/7.1.5/verify.sh <path-to-FFmpeg>
deps/ffmpeg/8.1/verify.sh <path-to-FFmpeg>
```

The supplied checkout must contain the corresponding upstream commit. The
shared verifier uses an isolated worktree and checks the entire ordered series,
not just individual patches.
