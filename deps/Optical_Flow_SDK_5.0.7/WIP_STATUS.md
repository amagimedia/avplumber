# WIP: NVOF camera-motion (Phase 2)

Foundation only. Dense-CUDA NVOF headers vendored + standalone probe verified
on Tesla T4 (`nvCreateOpticalFlowCuda` OK; engine from driver via `--gpus all`).

TODO: author the `CudaCameraMotion` node (NVOF dense flow on decoded nv12 ->
`.cu` masked-median reduce -> per-frame `{tx, nvof_cost}` metadata), add a
`HAVE_NVOF` Makefile gate, and wire it into the Sports-Reframing-Service
analysis graph downstream of the detection JoinMetadata.

No SDK bundle / FRUC lib required for the dense path.
