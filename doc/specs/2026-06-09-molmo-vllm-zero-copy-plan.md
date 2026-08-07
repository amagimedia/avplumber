# Molmo vLLM Zero-Copy AVP Plan

Date: 2026-06-09

## Goal

Add a local Molmo2/vLLM video inference path for AVPlumber that keeps video frames and model input tensors in GPU memory:

```text
AVP CUDA NV12 frame
-> CUDA preprocess
-> torch CUDA tensor shaped for Molmo2 video inputs
-> in-process local vLLM Molmo2
-> AVP frame metadata
-> existing CUDA draw nodes
```

The first implementation is an inline, pass-through Python node. It must not block the main graph while vLLM runs.

## Non-Goals

- No OpenAI HTTP API.
- No local vLLM HTTP API for the strict path.
- No PIL, NumPy image conversion, `hwdownload`, `hwupload`, or CPU media copies in strict zero-copy mode.
- No ByteTrack integration in phase 1.
- No FFmpeg CUDA conversion patch required for the Molmo path.
- No C++ Molmo node in phase 1.

## Model Target

Default model:

```text
allenai/Molmo2-VideoPoint-4B
```

Reasons:

- It is the video/pointing variant.
- It exposes Molmo2 video processor configuration.
- vLLM's Molmo2 model schema includes `pixel_values_videos`.
- It matches the grounding/pointing visualization use case better than plain image QA.

Relevant model defaults:

```text
default frame size: 378x378
patch size: 14
patch grid: 27x27
patches per frame: 729
patch vector length: 14 * 14 * 3 = 588
```

The node should read model/processor config where possible, validate it, and fail closed in strict mode if config is unsupported.

References:

- https://huggingface.co/allenai/Molmo2-VideoPoint-4B/blob/main/video_preprocessor_config.json
- https://huggingface.co/allenai/Molmo2-VideoPoint-4B/blob/main/config.json
- https://docs.vllm.ai/en/latest/api/vllm/model_executor/models/molmo2/

## Node Shape

Add:

```text
/home/jp/git/avplumber/src/nodes/neural_net/vlm/molmo/node.py
```

Primary class:

```python
class MolmoVllmAsync(PythonNode):
    ...
```

The node is SISO and inline:

```text
video -> molmo_vllm_async -> draw_bbox -> draw_keypoints -> draw_bbox_labels
```

It forwards every frame immediately. It samples frames by timestamp, preprocesses sampled frames immediately into node-owned CUDA tensors, and runs vLLM in a background worker. The graph thread never waits for vLLM completion.

## Why Inline Instead Of Split/Join

`join_metadata` waits while both branches are live and either input queue is empty. A sparse side metadata branch can therefore backpressure unless it also passes every frame.

For phase 1, inline pass-through is simpler:

- No `split`.
- No `join_metadata`.
- No side branch PTS alignment problem.
- Metadata is attached in-place to frames when cached results are fresh.

Existing `split(drop=true)` plus `join_metadata` remains useful later if Molmo must be a completely optional side branch.

## Zero-Copy Definition

Allowed:

- CUDA frame plane reads.
- CUDA kernel writing a transformed tensor.
- Torch/CuPy views over GPU buffers.
- DLPack or CUDA array interface exchange between Torch and CuPy.

Not allowed in strict mode:

- CPU image download.
- PIL image conversion.
- NumPy image conversion.
- HTTP upload of image/video bytes.
- FFmpeg `hwdownload`/`hwupload` workaround.

The preprocess copy from NV12 CUDA frame into Molmo CUDA tensor is acceptable. It is a GPU-to-GPU transform, not a CPU copy.

## CUDA Preprocess

Add:

```text
/home/jp/git/avplumber/src/nodes/neural_net/preprocess/molmo2_preprocess.cu
```

Phase 1 launcher:

```text
CuPy RawModule loaded from the .cu source file at node startup
```

This avoids AVP Makefile/PTX integration for the prototype. Later, move to AVP-built PTX if needed.

Kernel behavior:

```text
input:
  NV12 CUDA frame planes
  Y pointer, Y pitch
  UV pointer, UV pitch
  source width/height

output:
  torch CUDA tensor backing pixel_values_videos

steps:
  NV12 limited-range BT.709 -> RGB
  square resize to molmo_frame_size
  normalize with mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
  patchify directly into [frame_index, patch_index, patch_element]
```

The initial input policy is square resize, not letterbox. This matches Molmo2's public processor defaults. Aspect-preserving letterbox can be added later with AVP's existing `model_content_width`, `model_content_height`, and offset mapping in draw nodes.

## Python GPU Interop

Use existing AVP Python zero-copy patterns:

- `/home/jp/git/avplumber/src/nodes/neural_net/examples/pytorch_cuda.py`
- `/home/jp/git/avplumber/src/nodes/neural_net/scene_cut/cuda_scene_detect.py`

Input plane wrapping:

```python
y_cp = cupy.ndarray(
    shape=(height * pitch_y,),
    dtype=cupy.uint8,
    memptr=cupy.cuda.MemoryPointer(
        cupy.cuda.UnownedMemory(y_ptr, height * pitch_y, owner=frame),
        0,
    ),
).reshape(height, pitch_y)
```

Output tensor pool:

```python
pixel_values = torch.empty(
    (window_frames, patches_per_frame, patch_vec),
    device="cuda",
    dtype=torch.float16,
)
```

CuPy writes into this tensor through DLPack or another zero-copy CUDA view.

## Frame Lifetime Rule

Preprocess sampled frames immediately in `process()`.

Do not queue raw `AVFrame` pointers or `frame.data_ptr` values for the async worker. Once the frame is forwarded downstream, AVP/FFmpeg may eventually recycle the backing CUDA frame.

Flow:

```text
process(frame):
  if sampled:
    launch CUDA preprocess into node-owned tensor slot
  attach any fresh cached metadata
  forward frame immediately
  if a window is complete:
    enqueue tensor/window descriptor to the vLLM worker
```

The async worker owns only node-owned tensors and metadata, not AVFrame memory.

## Tensor Pool

Use a persistent tensor pool:

```text
num_buffers = max_inflight + window_queue_size + 1
```

Default tensor shape:

```text
window_frames = 16
molmo_frame_size = 378
patch_size = 14
patches_per_frame = 729
patch_vec = 588
dtype = float16
shape = [16, 729, 588]
```

One default window tensor is about 13 MiB. Three buffers are about 40 MiB. Model weights and vLLM KV/cache dominate GPU memory, not this tensor pool.

## Windowing And Sampling

Initial defaults:

```text
sample_fps = 8
window_frames = 16
window_stride = 16
```

That gives tumbling two-second windows at 8 fps.

Sampling should use timestamps:

```text
next_sample_time += 1 / sample_fps
select frame when pts_time >= next_sample_time
```

If PTS is invalid, fall back to frame-count sampling using inferred fps.

## Async Queueing

The graph thread cannot block on vLLM.

Defaults:

```text
max_inflight = 1
window_queue_size = 1
queue_policy = "replace_pending_with_newest"
```

Meaning:

- One window may be running in vLLM.
- One newer window may wait.
- If another window completes while one is pending, replace the pending window with the newest one.
- Do not build an old-window backlog.

## vLLM Placement

Run vLLM in-process inside `MolmoVllmAsync`.

Reason:

- Shortest path to strict zero-copy.
- No HTTP image/video payloads.
- No CUDA IPC process boundary.
- vLLM and Torch can consume the same CUDA tensor objects.

The implementation still needs a vLLM Molmo2 patch/helper that accepts preprocessed CUDA tensors and bypasses the normal media loader/processor path.

## vLLM GPU Memory Defaults

For a target with about 22 GB GPU memory:

```text
gpu_memory_utilization = 0.75
max_model_len = 8192
max_num_seqs = 1
max_num_batched_tokens = 8192
```

This reserves about 25 percent of VRAM outside vLLM for AVP CUDA frames, CUDA contexts, tensor pool, driver overhead, and fragmentation.

Do not start with high-throughput server settings. This is one live graph, not a multi-client vLLM serving workload.

## Frame Size Policy

Default:

```text
molmo_frame_size = 378
patch_size = 14
```

Allow any size divisible by `patch_size`, but require an explicit flag for non-default sizes:

```text
allow_non_default_frame_size = true
```

Validation:

```text
molmo_frame_size % patch_size == 0
patch_size == model_config.vit_config.image_patch_size
warn if molmo_frame_size != model_config.vit_config.image_default_input_size[0]
recompute video_grids/token_pooling for that size
strict_zero_copy fails closed on unsupported shapes
```

## Prompt API

Phase 1 uses a static prompt parameter:

```python
MolmoVllmAsync({
    "model_id": "allenai/Molmo2-VideoPoint-4B",
    "prompt_id": "default_objects",
    "prompt": "Find people, vehicles, balls, and important objects. Return JSON only...",
    "max_new_tokens": 512,
    "temperature": 0.0,
})
```

Prompt from frame metadata or a control edge can be added later.

## Prompt Coordinate Contract

Ask Molmo for integer coordinates from 0 to 1000 relative to the square image:

```json
{
  "objects": [
    {
      "label": "person",
      "confidence": 0.82,
      "bbox": [100, 180, 310, 920],
      "point": [205, 530]
    }
  ]
}
```

The node converts 1000-scale coordinates into Molmo model coordinates:

```text
x_model = x_1000 / 1000 * molmo_frame_size
y_model = y_1000 / 1000 * molmo_frame_size
```

## Metadata Outputs

Attach metadata only when a cached result is fresh.

Freshness policy:

```text
frame.pts >= result.window_end_pts
frame.pts <= result.window_end_pts + visualize_ttl
```

Default:

```text
visualize_ttl_frames = 16
```

### `molmo_detections`

Used by `draw_bbox` and `draw_bbox_labels`.

```json
{
  "schema": "yolo_detections_v1",
  "coord_space": "model",
  "model_width": 378,
  "model_height": 378,
  "detections": [
    {
      "label": "person",
      "conf": 0.82,
      "xyxy": [37.8, 68.04, 117.18, 347.76],
      "source": "molmo"
    }
  ]
}
```

### `molmo_points`

Used by `draw_keypoints` for CUDA circles.

```json
{
  "schema": "pose_keypoints_v1",
  "coord_space": "model",
  "model_width": 378,
  "model_height": 378,
  "num_keypoints": 1,
  "poses": [
    {
      "keypoints": [77.49, 200.34, 0.82]
    }
  ]
}
```

### `molmo_raw`

Used for debugging and observability.

```json
{
  "prompt_id": "default_objects",
  "window_start_pts": "...",
  "window_end_pts": "...",
  "generated_text": "...",
  "parse_status": "ok",
  "invalid_object_count": 0,
  "latency_ms": 734
}
```

## Output Parsing

Parsing policy:

```text
valid object with bbox -> emit bbox
valid object with point -> emit point
invalid object -> skip object and count it
invalid whole JSON -> no draw metadata, raw metadata parse_status="invalid_json"
```

Do not retry synchronously in the graph thread.

## Visualization Nodes

Use existing nodes:

```python
DrawBBox({
    "metadata_key": "molmo_detections",
    "bbox_thickness": 2,
})

DrawKeypoints({
    "metadata_key": "molmo_points",
    "radius": 5,
    "color": "yellow",
    "min_conf": 0.2,
})

DrawBBoxLabels({
    "metadata_key": "molmo_detections",
})
```

Existing draw nodes already scale `coord_space="model"` metadata from `model_width`/`model_height` to the actual video frame.

## Dependency Policy

Strict mode:

```text
strict_zero_copy = true
```

Startup must fail if any required dependency is unavailable:

- Torch with CUDA.
- CuPy for the target CUDA runtime.
- vLLM.
- Molmo2 vLLM tensor-input patch/helper.
- CUDA preprocess kernel compile/load.

Non-strict mode:

```text
strict_zero_copy = false
```

The node forwards frames unchanged and attaches `molmo_raw` status metadata such as `unavailable`. It should not silently use a CPU image fallback.

## Target Setup Notes

For Fedora 44 NVIDIA GPU targets, prefer one NVIDIA driver packaging method and do not mix RPM Fusion NVIDIA packages with NVIDIA CUDA repo driver packages unless intentionally switching.

Generic NVIDIA repo compute/headless path:

```bash
sudo dnf install -y kernel-devel-matched kernel-headers gcc make dkms pciutils dnf-plugins-core

sudo dnf config-manager addrepo \
  --from-repofile=https://developer.download.nvidia.com/compute/cuda/repos/fedora44/x86_64/cuda-fedora44.repo

sudo dnf clean expire-cache
sudo dnf install -y nvidia-driver-cuda kmod-nvidia-latest-dkms nvidia-modprobe nvidia-persistenced
sudo systemctl enable nvidia-persistenced
```

After reboot:

```bash
nvidia-smi
modinfo -F version nvidia
dkms status | grep -i nvidia || true
lsmod | grep '^nvidia'
```

CUDA toolkit:

```bash
sudo dnf install -y cuda-toolkit
```

CuPy example for CUDA 12 runtimes:

```bash
pip install cupy-cuda12x
```

For CUDA 13 runtimes, verify CuPy package support on the target before treating strict mode as available.

Reference:

- https://docs.nvidia.com/datacenter/tesla/driver-installation-guide/fedora.html

## Planned Files

Phase 1:

```text
/home/jp/git/avplumber/src/nodes/neural_net/vlm/molmo/node.py
/home/jp/git/avplumber/src/nodes/neural_net/preprocess/molmo2_preprocess.cu
/home/jp/git/avplumber/src/nodes/neural_net/vlm/molmo/example.py
```

External/local patch area:

```text
vLLM Molmo2 input mapper/helper for preprocessed CUDA tensors
```

No AVP C++ node is required for phase 1.

## Stop Behavior

On `doStop()`:

```text
set stop_event
stop accepting new windows
wake worker
wait up to worker_join_timeout_ms
release tensor pool references
release vLLM/model resources if the API supports explicit shutdown
```

On EOF:

```text
forward EOF
do not flush extra inference by default
stop worker
```

## Validation Plan

1. Unit-test JSON parsing and 1000-scale coordinate conversion.
2. Unit-test metadata emission for `molmo_detections`, `molmo_points`, and `molmo_raw`.
3. Run CuPy kernel on synthetic CUDA input/output tensors.
4. Run the Python node with `strict_zero_copy=false` and verify frame pass-through.
5. Run strict mode on a Fedora NVIDIA target and verify:
   - no PIL/NumPy image path,
   - no `hwdownload`,
   - no local HTTP vLLM path,
   - CUDA preprocess writes the expected tensor shape.
6. Confirm overlays persist for `visualize_ttl_frames`.
7. Monitor GPU memory with `nvidia-smi`; start with `gpu_memory_utilization=0.75`.

## Future Work

- Add ByteTrack phase 2 by feeding `molmo_detections` into existing `player_tracker`.
- Add aspect-preserving letterbox preprocess and set draw node `model_content_*` mapping.
- Move CuPy startup-compiled kernel to AVP-built PTX.
- Add prompt from metadata or a control edge.
- Add generic `draw_points` if `draw_keypoints` schema becomes awkward for non-pose points.
- Add optional side-branch split/join graph if Molmo must be isolated from the main video path.
