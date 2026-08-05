# CUDA Overlay Many Demo

## Goal

Add a self-contained `cuda-overlay-demo/` that builds the repository's patched
FFmpeg `n7.1.5` and PyPlumber integration, generates deterministic color-bar
fixtures, and exercises `overlay_many_cuda` on an NVIDIA GPU.

The demo is both a visual sample and a correctness harness. It produces PNG
contact sheets for inspection, but judges correctness from raw YUV plane data
against an independent CPU reference. It targets CUDA Toolkit 11.7 so the
container build verifies the filter's documented minimum toolkit version. The
runtime still requires compute capability 7.0 or newer.

This directory is independent of the in-progress `dmabuf-demo/`. It follows the
same idea of keeping a runnable example, graph, container definition, and
documentation together, but it does not share files or services with that
demo. It does not add a browser, Wayland compositor, Janus, or a live network
output because none of those components contribute to CUDA overlay validation.

## Test Surface

The main matrix covers every supported `overlay_many_cuda` software-format
combination:

1. `yuv420p` main with `yuva420p` overlays;
2. `yuv420p` main with `yuva444p` overlays;
3. `yuv444p` main with `yuva444p` overlays.

The requested visual sweep uses 2 through 15 overlay images, corresponding to
3 through 16 total filter inputs. One additional minimum-boundary case per
format uses one overlay, so the complete matrix covers `inputs=2` through
`inputs=16`: 15 input counts across three format combinations, or 45 cases.

`yuv444p` main with `yuva420p` overlays is not included because the filter does
not support that pairing. The demo must report the three supported modes
explicitly rather than implying that all permutations of 4:2:0 and 4:4:4 are
valid.

All fixtures use a 641x361 canvas. The odd width and height exercise ceiling
chroma dimensions and partial 2x2 edge blocks in every matrix case instead of
relegating those paths to a rarely run optional test.

## Directory Shape

The implementation will add:

```text
cuda-overlay-demo/
|-- README.md
|-- Dockerfile
|-- compose.yaml
|-- run.sh
|-- graph/
|   `-- overlay_case.py
|-- tools/
|   |-- generate_assets.py
|   |-- cpu_reference.py
|   |-- run_matrix.py
|   `-- make_contact_sheets.py
`-- artifacts/                 # generated and ignored, except .gitignore
```

The Compose build context is the repository root so the image can copy the
FFmpeg patch stack, avplumber sources, and PyPlumber sources. All paths stored
in tracked files are repository-relative or generic container paths. Hostnames,
SSH keys, test-instance addresses, and host-specific CUDA paths are never
stored in the demo.

## Fixture Generation

`generate_assets.py` creates one opaque PNG base and 15 RGBA PNG overlays. The
base is a labeled color-bar and checker pattern that makes luma, chroma, and
edge corruption visible. Each overlay contains:

- a unique colored bar and numeric layer label;
- a shared central region that overlaps all earlier layers and reveals z-order;
- horizontal and vertical alpha ramps;
- exact alpha samples including 0, 1, 127, 128, 254, and 255;
- one-pixel markers on the final row and column.

The source PNGs are retained under the generated artifacts for direct viewing.
PNG encodes RGBA pixels, not subsampled planar YUVA formats, so it is not itself
a 4:2:0 or 4:4:4 fixture. A deterministic preparation step uses the same built
FFmpeg to create lossless, one-frame planar fixtures for `yuv420p`, `yuv444p`,
`yuva420p`, and `yuva444p`. The graph consumes these planar fixtures. Raw plane
copies are also retained as the inputs to the CPU oracle, avoiding ambiguity
from RGB/YUV conversion or PNG decoding during comparison.

Assets are generated rather than committed as binary blobs. A fixed algorithm,
dimensions, labels, and palette make every run reproducible.

## PyPlumber Graph

`graph/overlay_case.py` builds one graph for one mode and one input count. The
matrix runner starts it in a fresh process for each case. This deliberately
trades some process startup time for bounded GPU memory, isolated logs, and a
clear association between a failure and its graph configuration.

For every source, the graph reads one lossless planar frame, decodes it in
software, and uploads it once at the input boundary with the appropriate
software format. The ordered CUDA edges feed a single `FilterVideo` node whose
graph is `overlay_many_cuda=inputs=N`. The main frame is source zero; overlay
sources one through N-1 are connected in ascending layer order.

After composition, the graph downloads the result once at the validation
boundary and writes one raw YUV frame. There is no CPU/GPU round trip between
upload, composition, and download. The boundary transfers are intentional in
this correctness demo and are not proposed as production pipeline behavior.

The graph accepts only explicit mode, input count, fixture directory, output
path, dimensions, and timeout arguments. It validates that total inputs are in
the filter's 2 through 16 range and that the selected mode is one of the three
supported combinations. It terminates successfully only after receiving the
exact expected output byte count. Timeout, graph failure, short output, excess
output, or unsupported format is an error.

## CPU Reference and Comparison

`cpu_reference.py` reads the prepared planar fixtures and implements the
kernel's arithmetic directly. It does not call FFmpeg's software `overlay`
filter, because that would make agreement depend on another filter's rounding
and chroma policy rather than the documented CUDA contract.

Every blend step uses exact rounded division by 255:

```text
sum = alpha * foreground + (255 - alpha) * background
rounded = sum + 128
result = (rounded + (rounded >> 8)) >> 8
```

The reference rounds back to eight bits after each layer. For `yuva420p`, the
chroma alpha is the rounded average of only the valid luma alpha samples in the
corresponding 2x2 block. For `yuva444p` onto `yuv420p`, it composes all layers
at each valid full-resolution chroma position before averaging the final
values. Odd right and bottom blocks use only valid samples.

The primary acceptance criterion is byte-for-byte equality for each output Y,
U, and V plane. A mismatch fails the case. The report additionally records the
mismatch count, maximum absolute difference, and mean absolute difference per
plane to make a failure diagnosable; those diagnostic values do not weaken the
exact acceptance criterion.

After comparison, `make_contact_sheets.py` converts the raw GPU outputs to PNG
for human inspection. It creates one sheet per format mode, ordered by overlay
count, plus a sheet of the source layers. PNG conversion occurs after raw-plane
comparison and therefore cannot hide or introduce a passing result.

## Container and Run Interface

The Dockerfile uses a CUDA 11.7 development image and builds:

- FFmpeg `n7.1.5` with the repository patch series and CUDA/NVCC support;
- the avplumber Python module with the CUDA and NVCC/PTX features required by
  the graph;
- the small Python dependencies used for fixture generation, planar reference
  calculations, reports, and contact sheets.

The final image checks the CUDA compiler version and confirms that FFmpeg lists
`overlay_many_cuda` before it is accepted as built. The runtime entrypoint also
records FFmpeg version, filter availability, CUDA compiler version, GPU model,
driver version, and compute capability in the report.

`compose.yaml` defines one GPU test service with the generated artifacts bind
mounted to `cuda-overlay-demo/artifacts/`. A multi-service live stack would add
coordination without improving this test, so generation, graph execution,
comparison, and reporting run sequentially inside the test service. `run.sh`
is a thin convenience wrapper around the corresponding Compose build and run
commands; it contains no machine-specific configuration.

The normal user flow is:

```text
./cuda-overlay-demo/run.sh
```

The command returns nonzero if image generation, graph construction, any GPU
case, any exact comparison, or report generation fails. Successful output
includes a machine-readable JSON summary, per-case logs and raw frames, source
PNGs, result PNGs, and contact sheets.

## Error Handling and Reproducibility

The matrix runner creates a fresh case directory and log for every mode/count
pair. It continues after an individual case failure so the final report shows
the complete failure surface, then exits nonzero if any case failed. A build or
fixture-generation failure stops immediately because later cases would not be
meaningful.

Generated files include their dimensions, format mode, overlay count, patch
series identity, and tool versions in the JSON metadata. Test ordering is
stable. No random values are used unless a future option supplies and records
an explicit seed.

Container execution requires NVIDIA Container Toolkit GPU access. If no GPU is
visible, compute capability is below 7.0, or the driver cannot run the CUDA
11.7-built code, the demo fails with a direct prerequisite error rather than
falling back to software composition.

## Validation

Before reporting the demo complete:

- build the image from a clean repository context;
- verify the image uses CUDA Toolkit 11.7;
- verify patched FFmpeg exposes `overlay_many_cuda`;
- run all 45 matrix cases on the configured NVIDIA T4 host;
- require exact raw-plane agreement for all cases;
- inspect the three mode contact sheets for layer order, transparency, and odd
  edge corruption;
- run targeted local tests for the pure-Python asset and CPU-reference logic;
- confirm tracked demo files contain no private host, credential, or
  machine-specific paths;
- confirm the existing `dmabuf-demo/` and unrelated dirty files are unchanged.

CUDA build and execution are performed on the configured remote NVIDIA host,
not on a local machine without `nvidia-smi`.

## Non-goals

This demo does not benchmark throughput, prove production zero-copy ingest or
egress, exercise unsupported pixel-format pairs, add positioning or scaling to
`overlay_many_cuda`, test more than 15 overlays, provide a live browser preview,
or modify the graph framework. It does not replace FFmpeg FATE coverage; it is
a reproducible integration and visual-validation harness for the patch stack.
