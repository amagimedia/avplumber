# cuda-overlay — deterministic `overlay_many_cuda` validation

Build the repository's FFmpeg patch stack on public FFmpeg `n7.1.5`, run the
patched `overlay_many_cuda` through a purpose-built PyPlumber graph, and compare
every output plane against an independent CPU reference.

The demo generates a labeled color-bar base plus 15 RGBA PNG overlays. The
overlays contain alpha ramps, exact alpha edge values, overlapping regions, and
markers on the final row and odd final column. It tests one through 15 overlays
so the filter's complete `inputs=2` through `inputs=16` range is covered. The
requested visual sweep is overlays 2 through 15; the one-overlay cases retain
the minimum API boundary. Labels identify filter input indices: `L0` is the
base, and `L1` through `L15` are the overlay inputs.

## Covered CUDA paths

- `yuv420p` main with `yuva420p` overlays (`420_420`)
- `yuv420p` main with `yuva444p` overlays (`420_444`)
- `yuv444p` main with `yuva444p` overlays (`444_444`)

`yuv444p` with `yuva420p` is not a supported filter combination and is not
presented as a fourth mode. The 641x360 canvas exercises a partial horizontal
4:2:0 chroma block in all 45 cases. Odd height is intentionally excluded because
FFmpeg n7.1.5's plain CUDA upload/download path does not preserve its final
4:2:0 chroma row, before `overlay_many_cuda` is involved.

## Requirements

- Docker with Compose
- NVIDIA Container Toolkit
- an NVIDIA GPU with compute capability 7.0 or newer
- a driver capable of running CUDA 11.7 applications

The image deliberately compiles FFmpeg with CUDA Toolkit 11.7, the minimum
supported version. It does not use the host CUDA toolkit. The demo build omits
the unrelated `nvjpeg_enc` node because its NV12 encoder API requires a newer
CUDA toolkit; all FFmpeg CUDA and NVCC support remains enabled.

## Run

From the repository root:

```bash
./demos/cuda-overlay/run.sh
```

To build without starting the matrix, run `./demos/cuda-overlay/build.sh`.

For a smaller diagnostic sweep:

```bash
./demos/cuda-overlay/run.sh --counts 1,2,8,15
```

For a native 1080p maximum-input run:

```bash
./demos/cuda-overlay/run.sh --width 1920 --height 1080 --counts 15
```

Explicit dimensions apply to fixture generation, CUDA composition, the CPU
reference, PNG conversion, and the report. The default remains 641x360 so the
normal regression matrix retains its odd-width edge coverage.

The command builds the image, generates fixtures, starts a fresh PyPlumber
process per mode/count pair, and returns nonzero if any graph or exact comparison
fails. There is no CPU fallback.

## Artifacts

Each run creates a timestamped directory below `demos/cuda-overlay/artifacts/`
containing:

- `assets/png/`: the base and 15 source overlay PNGs;
- `assets/raw/`: the exact planar input fixtures used by both implementations;
- `assets/input-streams/`: two-frame repeats of each fixture used to give every
  filter input deterministic PTS ordering before end-of-stream;
- `gpu-raw/`: frames produced by `overlay_many_cuda`;
- `reference-raw/`: frames produced by the CPU oracle;
- `result-png/`: GPU results converted for inspection after comparison;
- `contact-sheets/`: one source sheet and one sheet per supported mode;
- `logs/`: one PyPlumber log per case;
- `report.json`: environment, patch hashes, per-plane deltas, and final status.

Correctness is decided from the raw Y, U, and V planes. PNG conversion is only
for display and cannot turn a mismatch into a pass.

## Graph boundary

The graph reads deterministic software frames, uploads each source once through
a shared instance-scoped CUDA device inside one multi-input filter graph,
composes on CUDA, then downloads once to inspect the result. Those transfers are
explicit test boundaries; there is no upload/download round trip inside the
CUDA composition path, and this graph is not a production-pipeline template.
