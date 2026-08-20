# Generic CUDA Vertical-Band Blur

## Goal

Replace the scorebug-specific `bottom_blur_cuda` FFmpeg patch with a reusable
`band_blur_cuda` filter. The filter must apply independently configurable blur
and luma gradients to any vertical band, including bands feathered from either
the top or bottom edge, while retaining NV12 CUDA zero-copy processing.

## Filter Contract

The filter accepts CUDA hardware frames whose software format is NV12 and
produces another CUDA NV12 frame. It copies pixels outside the selected band
unchanged and processes only the selected band using persistent CUDA scratch
buffers.

The public options are:

| Option | Range | Default | Meaning |
|---|---:|---:|---|
| `y_start` | 0.0–0.95 | 0.70 | normalized top coordinate of the band |
| `y_end` | 0.05–1.0 | 1.0 | normalized exclusive lower coordinate of the band |
| `blur_start` | 0.0–1.0 | 0.0 | blurred-image mix at `y_start` |
| `blur_end` | 0.0–1.0 | 1.0 | blurred-image mix at `y_end` |
| `luma_start` | 0.0–2.0 | 1.0 | luma multiplier at `y_start` |
| `luma_end` | 0.0–2.0 | 0.86 | luma multiplier at `y_end` |
| `radius` | 1–12 | 6 | half-resolution separable box-blur radius |
| `gradient` | `linear`, `smoothstep` | `smoothstep` | interpolation curve |
| `gradient_power` | 0.1–10.0 | 1.0 | bias applied after the selected curve |

Configuration fails when `y_end` is not greater than `y_start`. Frame-derived
pixel coordinates are clamped and aligned for NV12 chroma. Endpoint values are
evaluated at the first and last processed rows so the requested gradient does
not depend on resolution.

The defaults reproduce the current bottom-scorebug cleanup. A top-edge cleanup
uses `y_start=0`, `y_end=0.30`, descending blur, and ascending luma:

```text
band_blur_cuda=y_start=0:y_end=.30:
  blur_start=1:blur_end=0:luma_start=.86:luma_end=1
```

## Gradient and Composition

For each row in the band, normalize its position to `u` in `[0, 1]`. Apply the
selected curve, then its configurable power:

```text
linear:     t = u
smoothstep: t = u*u*(3 - 2*u)
biased:     g = pow(t, gradient_power)
```

Interpolate blur mix and luma independently:

```text
blur_mix = mix(blur_start, blur_end, g)
luma     = mix(luma_start, luma_end, g)
output_y = mix(sharp_y, blurred_y, blur_mix) * luma
output_uv = mix(sharp_uv, blurred_uv, blur_mix)
```

The byte conversion saturates after luma multiplication. Chroma is blurred but
is not multiplied by luma.

## CUDA Implementation

Rename the FFmpeg filter, C/CUDA sources, generated PTX symbol, configuration
gate, and registration symbol to `band_blur_cuda`. Allocate downsampled scratch
for `y_start..y_end` only. The existing downsample and separable blur passes
remain unchanged apart from generic band coordinates.

Pass the endpoint values, gradient enum, and power to the composite kernels.
The option-dependent math is constant per launch and adds no host/device copy.
All scratch allocations remain persistent across frames and are rebuilt only
when output geometry changes.

The specialized `bottom_blur_cuda` name is removed rather than retained as an
alias. It has not landed on AVPlumber `develop`; the only consumer is updated
atomically with this patch.

## Consumer Integration

Update `sports-live-reframer` to construct `band_blur_cuda` before the browser
overlay. Its scoreboard-overlay configuration exposes all spatial, blur, luma,
and curve options with the defaults above. Rename the existing bottom-specific
configuration and CLI flags instead of retaining misleading names.

The lower-third validation continues to use the bottom defaults. A focused
top-edge invocation proves the generic direction without changing the product
layout.

## Verification

- verify the FFmpeg patch series applies cleanly;
- build patched FFmpeg and confirm `ffmpeg -filters` lists
  `band_blur_cuda` and not `bottom_blur_cuda`;
- test option serialization and invalid coordinate rejection;
- test graph ordering before CUDA overlay composition;
- render deterministic top and bottom gradient frames and inspect boundary,
  midpoint, and endpoint pixels;
- rerun the existing 9:16 lower-third end-to-end sample; and
- benchmark both directions on the T4, retaining the existing 608x1080 p95
  target below 1 ms and confirming no host-frame transfer.

## Out of Scope

- horizontal or arbitrary two-dimensional masks;
- RGB or YUV formats other than CUDA NV12;
- multiple disjoint bands in one filter instance; and
- temporal blur or motion-aware cleanup.
