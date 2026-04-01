# YOLO Pose Decoder and draw_keypoints Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add YOLO pose/keypoint decoding and a CUDA circle-drawing node to visualize court landmarks.

**Architecture:** New `PoseDecoder` (header-only, CPU decode) follows the existing `DetectionDecoder`/`SegmentationDecoder` pattern. New `draw_keypoints` node extends `CudaOverlayBase` with a simple filled-circle CUDA kernel operating on NV12 planes. Keypoint data flows as JSON metadata through the existing `av_dict` mechanism.

**Tech Stack:** C++17, CUDA (PTX kernels), TensorRT (inference), nlohmann/json (metadata), FFmpeg libavutil (frame side data)

---

### Task 1: Add PoseResult struct and ModelRunner fields

**Files:**
- Modify: `src/nodes/hwaccel/cuda_infer_yolo_base.hpp`

- [ ] **Step 1: Add PoseResult struct after SegmentationResult**

In `src/nodes/hwaccel/cuda_infer_yolo_base.hpp`, after the `SegmentationResult` struct (line ~92) and before `DecodeParams`, add:

```cpp
struct PoseResult : DetectionResult {
    std::vector<float> keypoints;  // flat [x, y, conf, x, y, conf, ...] per detection
    int num_keypoints = 0;         // keypoints per detection (e.g. 34)
};
```

- [ ] **Step 2: Add forward declaration for PoseDecoder**

After the existing forward declarations (line ~148), add:

```cpp
class PoseDecoder;
```

- [ ] **Step 3: Add pose_decoder and num_classes to ModelRunner**

In the `ModelRunner` struct, after the `seg_decoder` field (line ~198), add:

```cpp
std::unique_ptr<PoseDecoder> pose_decoder;
int num_classes = -1;  // -1 = auto-detect; required for pose to disambiguate class scores from keypoints
```

- [ ] **Step 4: Verify local build compiles**

```bash
make -j$(nproc)
```

Expected: compiles cleanly (no new code references the new fields yet).

- [ ] **Step 5: Commit**

```bash
git add src/nodes/hwaccel/cuda_infer_yolo_base.hpp
git commit -m "feat: add PoseResult struct and pose_decoder field to ModelRunner"
```

---

### Task 2: Implement PoseDecoder

**Files:**
- Create: `src/nodes/hwaccel/yolo_decode_pose.hpp`

- [ ] **Step 1: Create yolo_decode_pose.hpp**

Create `src/nodes/hwaccel/yolo_decode_pose.hpp` with the following content:

```cpp
#pragma once
#include "cuda_infer_yolo_base.hpp"

namespace yolo_base {

class PoseDecoder {
private:
    int num_classes_ = 1;

public:
    explicit PoseDecoder(int num_classes) : num_classes_(num_classes) {}

    PoseResult decode(
        const std::vector<const float*>& host_outputs,
        const std::vector<nvinfer1::Dims>& output_dims,
        const DecodeParams& params
    ) {
        PoseResult result;
        if (host_outputs.empty() || !host_outputs[0]) return result;

        const float* out = host_outputs[0];
        const nvinfer1::Dims& d = output_dims[0];

        int count = 0, attrs = 0;
        bool attrs_first = false;

        if (d.nbDims == 2) {
            int a = d.d[0], b = d.d[1];
            if (a <= b && a >= 6) { attrs = a; count = b; attrs_first = true; }
            else if (b >= 6) { attrs = b; count = a; attrs_first = false; }
            else if (a >= 6) { attrs = a; count = b; attrs_first = true; }
            else return result;
        } else if (d.nbDims == 3 && d.d[0] == 1) {
            int d1 = d.d[1], d2 = d.d[2];
            if (d1 <= d2 && d1 >= 6) { attrs = d1; count = d2; attrs_first = true; }
            else if (d2 >= 6) { attrs = d2; count = d1; attrs_first = false; }
            else if (d1 >= 6) { attrs = d1; count = d2; attrs_first = true; }
            else return result;
        } else {
            return result;
        }

        if (count <= 0 || attrs < 6) return result;

        auto at = [&](int det, int attr) -> float {
            return attrs_first ? out[attr * count + det] : out[det * attrs + attr];
        };

        // Layout: [4 box] + [num_classes] + [num_keypoints * 3]
        int kpt_start = 4 + num_classes_;
        int kpt_total_attrs = attrs - kpt_start;
        if (kpt_total_attrs < 3 || kpt_total_attrs % 3 != 0) return result;
        result.num_keypoints = kpt_total_attrs / 3;

        for (int i = 0; i < count; ++i) {
            Detection det;
            det.model_index = params.model_index;

            if (params.box_format == OutputBoxFormat::EndToEndXYXY) {
                det.x1 = at(i, 0); det.y1 = at(i, 1);
                det.x2 = at(i, 2); det.y2 = at(i, 3);
                det.conf = at(i, 4);
                det.cls = (int)std::round(at(i, 5));
            } else { // RawCXCYWH
                float cx = at(i, 0), cy = at(i, 1);
                float w = at(i, 2), h = at(i, 3);
                int best_cls = 0;
                float best = 0.0f;
                for (int c = 4; c < kpt_start; ++c) {
                    float s = at(i, c);
                    if (s > best) { best = s; best_cls = c - 4; }
                }
                det.x1 = cx - w * 0.5f; det.y1 = cy - h * 0.5f;
                det.x2 = cx + w * 0.5f; det.y2 = cy + h * 0.5f;
                det.conf = best;
                det.cls = best_cls;
            }

            if (det.conf < params.conf_thresh) continue;

            if (det.cls >= 0 && (size_t)det.cls < params.class_index_remap.size()) {
                det.cls = params.class_index_remap[(size_t)det.cls];
            }

            result.detections.push_back(det);

            // Extract keypoint triplets (x, y, conf)
            for (int k = 0; k < kpt_total_attrs; ++k) {
                result.keypoints.push_back(at(i, kpt_start + k));
            }
        }

        return result;
    }
};

} // namespace yolo_base
```

- [ ] **Step 2: Verify local build compiles**

The header is not yet included anywhere, so this is just a syntax check:

```bash
make -j$(nproc)
```

Expected: compiles cleanly.

- [ ] **Step 3: Commit**

```bash
git add src/nodes/hwaccel/yolo_decode_pose.hpp
git commit -m "feat: add PoseDecoder for YOLO pose model keypoint extraction"
```

---

### Task 3: Integrate PoseDecoder into cuda_infer_yolo node

**Files:**
- Modify: `src/nodes/hwaccel/cuda_infer_yolo.cpp`

- [ ] **Step 1: Add include for pose decoder**

At the top of `src/nodes/hwaccel/cuda_infer_yolo.cpp`, after the existing includes (line 3), add:

```cpp
#include "yolo_decode_pose.hpp"
```

- [ ] **Step 2: Add metadata_key_pose_ member and param parsing**

In the `CudaInferYolo` class, after `metadata_key_segmentation_` (line 14), add:

```cpp
std::string metadata_key_pose_ = "yolo_pose";
```

In the `create()` method, after the `metadata_key_segmentation` param parsing (line 283), add:

```cpp
if (params.count("metadata_key_pose")) r->metadata_key_pose_ = params["metadata_key_pose"].get<std::string>();
```

- [ ] **Step 3: Parse num_classes in model config**

In the `create()` method, inside the models loop, after the `class_index_remap` parsing block (around line 353), add:

```cpp
int num_classes = -1;
if (mp.count("num_classes")) {
    num_classes = mp["num_classes"].get<int>();
}
model.num_classes = num_classes;
```

- [ ] **Step 4: Create PoseDecoder in the model factory block**

In the `create()` method, after the segmentation decoder creation (line ~359), add the pose case:

```cpp
} else if (model.task_type == TaskType::Pose) {
    int nc = model.num_classes;
    if (nc < 1) nc = 1;  // default for pose models
    model.pose_decoder = std::make_unique<PoseDecoder>(nc);
}
```

- [ ] **Step 5: Add pose branch in process loop**

In the `process()` method, after the `TaskType::Segmentation` block (around line 201, after the `sr.gpu_mask_buf` unref), add the pose handling:

```cpp
} else if (model.task_type == TaskType::Pose && model.pose_decoder) {
    PoseResult pr = model.pose_decoder->decode(host_outputs, output_dims, dp);
    if (model.include_in_detection_metadata) {
        all_dets.insert(all_dets.end(), pr.detections.begin(), pr.detections.end());
    }

    // Build pose metadata
    if (!pr.detections.empty()) {
        Parameters pose_md;
        pose_md["coord_space"] = "model";
        pose_md["model_width"] = input_w_;
        pose_md["model_height"] = input_h_;
        pose_md["num_keypoints"] = pr.num_keypoints;
        pose_md["poses"] = Parameters::array();

        int kpt_stride = pr.num_keypoints * 3;
        for (size_t di = 0; di < pr.detections.size(); ++di) {
            const Detection& d = pr.detections[di];
            Parameters item;
            item["cls"] = d.cls;
            item["conf"] = d.conf;
            item["xyxy"] = {d.x1, d.y1, d.x2, d.y2};
            item["model_index"] = d.model_index;
            if (d.model_index >= 0 && (size_t)d.model_index < models_.size()) {
                item["engine_name"] = models_[(size_t)d.model_index].engine_name;
                const auto& cn = models_[(size_t)d.model_index].class_names;
                if (d.cls >= 0 && (size_t)d.cls < cn.size()) {
                    item["label"] = cn[(size_t)d.cls];
                }
            }
            // Flat keypoint array [x1, y1, c1, x2, y2, c2, ...]
            item["keypoints"] = Parameters::array();
            size_t kpt_off = di * (size_t)kpt_stride;
            for (int k = 0; k < kpt_stride && kpt_off + (size_t)k < pr.keypoints.size(); ++k) {
                item["keypoints"].push_back(pr.keypoints[kpt_off + (size_t)k]);
            }
            pose_md["poses"].push_back(item);
        }

        av_dict_set(&frm.raw()->metadata, metadata_key_pose_.c_str(), pose_md.dump().c_str(), 0);
    }

    if (debug_log_metadata_ && debug_log_every_n_ > 0 &&
        (frame_counter_ % (uint64_t)debug_log_every_n_) == 0) {
        logstream << "cuda_infer_yolo: pose model=" << model.engine_name
                  << " detections=" << pr.detections.size()
                  << " keypoints_per_det=" << pr.num_keypoints;
    }
}
```

- [ ] **Step 6: Verify local build compiles**

```bash
make -j$(nproc)
```

Expected: compiles cleanly.

- [ ] **Step 7: Commit**

```bash
git add src/nodes/hwaccel/cuda_infer_yolo.cpp
git commit -m "feat: integrate PoseDecoder into cuda_infer_yolo process loop"
```

---

### Task 4: Create draw_keypoints CUDA kernel

**Files:**
- Create: `src/nodes/hwaccel/draw_keypoints.cu`

- [ ] **Step 1: Create draw_keypoints.cu**

Create `src/nodes/hwaccel/draw_keypoints.cu`:

```cpp
// Draw filled circles at keypoint positions on an NV12 CUDA frame.
#include <stdint.h>
#include <cuda_runtime.h>

// Keypoint position in frame coordinates (uploaded from host)
struct KeypointPos {
    float x;
    float y;
};

extern "C" __global__ void kDrawKeypointsNV12Luma(
    uint8_t* __restrict__ y_plane, size_t pitch_y,
    const KeypointPos* __restrict__ points, int num_points,
    int radius, int y_color,
    int frame_w, int frame_h)
{
    const int px = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    const int py = (int)(blockIdx.y * blockDim.y + threadIdx.y);
    if (px >= frame_w || py >= frame_h) return;

    const int r2 = radius * radius;
    for (int i = 0; i < num_points; ++i) {
        float dx = (float)px - points[i].x;
        float dy = (float)py - points[i].y;
        if (dx * dx + dy * dy <= (float)r2) {
            y_plane[(size_t)py * pitch_y + (size_t)px] = (uint8_t)y_color;
            return;
        }
    }
}

extern "C" __global__ void kDrawKeypointsNV12Chroma(
    uint8_t* __restrict__ uv_plane, size_t pitch_uv,
    const KeypointPos* __restrict__ points, int num_points,
    int radius, int u_color, int v_color,
    int frame_w, int frame_h)
{
    const int uv_x = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    const int uv_y = (int)(blockIdx.y * blockDim.y + threadIdx.y);
    const int uv_frame_w = (frame_w + 1) >> 1;
    const int uv_frame_h = (frame_h + 1) >> 1;
    if (uv_x >= uv_frame_w || uv_y >= uv_frame_h) return;

    const int luma_x = (uv_x << 1);
    const int luma_y = (uv_y << 1);

    const int r2 = radius * radius;
    for (int i = 0; i < num_points; ++i) {
        float dx = (float)luma_x - points[i].x;
        float dy = (float)luma_y - points[i].y;
        if (dx * dx + dy * dy <= (float)r2) {
            uint8_t* row = uv_plane + (size_t)uv_y * pitch_uv;
            row[(size_t)(uv_x << 1) + 0] = (uint8_t)u_color;
            row[(size_t)(uv_x << 1) + 1] = (uint8_t)v_color;
            return;
        }
    }
}
```

- [ ] **Step 2: Commit**

```bash
git add src/nodes/hwaccel/draw_keypoints.cu
git commit -m "feat: add CUDA kernel for drawing keypoint circles on NV12"
```

---

### Task 5: Add draw_keypoints to Makefile

**Files:**
- Modify: `Makefile`

- [ ] **Step 1: Add variable definitions in the HAVE_CUDA+HAVE_NVCC block**

In the `Makefile`, inside the `ifeq ($(HAVE_CUDA)$(HAVE_NVCC),11)` block (after line 92, the `draw_segmask.cpp` entry), add:

```makefile
NODES_SRC += $(SRCDIR)/nodes/hwaccel/draw_keypoints.cpp
```

After the `DRAW_SEGMASK_PTX_H` definition (line 104), add:

```makefile
BUILD_DRAW_KEYPOINTS_PTX = 1
DRAW_KEYPOINTS_KERNEL = $(SRCDIR)/nodes/hwaccel/draw_keypoints.cu
DRAW_KEYPOINTS_PTX = objs/$(SRCDIR)/nodes/hwaccel/draw_keypoints.ptx
DRAW_KEYPOINTS_PTX_H = objs/$(SRCDIR)/nodes/hwaccel/draw_keypoints.ptx.h
```

In the `else` branch (line ~108), after `BUILD_DRAW_SEGMASK_PTX = 0`, add:

```makefile
BUILD_DRAW_KEYPOINTS_PTX = 0
```

- [ ] **Step 2: Add build rules after the draw_segmask rules**

After the `endif` of the `BUILD_DRAW_SEGMASK_PTX` block (line 321), add:

```makefile
ifeq ($(BUILD_DRAW_KEYPOINTS_PTX),1)
$(DRAW_KEYPOINTS_PTX): $(DRAW_KEYPOINTS_KERNEL)
	@mkdir -p $(dir $@)
	$(NVCC) -ptx -o $@ $<

$(DRAW_KEYPOINTS_PTX_H): $(DRAW_KEYPOINTS_PTX)
	@mkdir -p $(dir $@)
	@if [ ! -s $< ]; then echo "Error: PTX file $< is empty or missing" >&2; exit 1; fi
	xxd -i $< | sed -E 's/unsigned int objs_src_nodes_hwaccel_draw_keypoints_ptx_len/const unsigned int avpl_draw_keypoints_ptx_len/; s/unsigned char objs_src_nodes_hwaccel_draw_keypoints_ptx/const char avpl_draw_keypoints_ptx/' > $@
	@if [ ! -s $@ ]; then echo "Error: Generated header $@ is empty. Check PTX file: $<" >&2; exit 1; fi

objs/src/nodes/hwaccel/draw_keypoints.o: $(DRAW_KEYPOINTS_PTX_H)
endif
```

- [ ] **Step 3: Verify the Makefile parses correctly**

```bash
make -n -j1 2>&1 | head -5
```

Expected: no Makefile syntax errors.

- [ ] **Step 4: Commit**

```bash
git add Makefile
git commit -m "build: add draw_keypoints CUDA kernel compilation rules"
```

---

### Task 6: Implement draw_keypoints node

**Files:**
- Create: `src/nodes/hwaccel/draw_keypoints.cpp`

- [ ] **Step 1: Create draw_keypoints.cpp**

Create `src/nodes/hwaccel/draw_keypoints.cpp`:

```cpp
#include "cuda_overlay_base.hpp"

#include <cstdint>
#include <vector>

#include "../../../objs/src/nodes/hwaccel/draw_keypoints.ptx.h"

using cuda_overlay::DrawColor;

class DrawKeypoints : public CudaOverlayBase {
private:
    std::string metadata_key_;
    DrawColor color_{};
    int radius_ = 3;
    double min_conf_ = 0.0;
    double model_content_width_ = 0.0;
    double model_content_height_ = 0.0;
    double model_content_offset_x_ = 0.0;
    double model_content_offset_y_ = 0.0;
    int debug_log_every_n_ = 0;

    CUdeviceptr gpu_points_buf_ = 0;
    size_t gpu_points_capacity_ = 0;

    const char* nodeName() const override { return "draw_keypoints"; }

    bool remapModelCoord(double x, double y,
                         double model_w, double model_h,
                         double& out_x, double& out_y) const {
        if (model_content_width_ > 0.0 && model_content_height_ > 0.0) {
            const double content_x = std::max(0.0, std::min(x - model_content_offset_x_, model_content_width_));
            const double content_y = std::max(0.0, std::min(y - model_content_offset_y_, model_content_height_));
            out_x = content_x * ((double)input_params_.width / model_content_width_);
            out_y = content_y * ((double)input_params_.height / model_content_height_);
            return true;
        }
        const double sx = model_w > 0.0 ? (double)input_params_.width / model_w : 1.0;
        const double sy = model_h > 0.0 ? (double)input_params_.height / model_h : 1.0;
        out_x = x * sx;
        out_y = y * sy;
        return true;
    }

    struct KeypointPos {
        float x;
        float y;
    };

    void drawOnFrame(const av::VideoFrame& input, av::VideoFrame& output) override {
        if (!loadKernels(avpl_draw_keypoints_ptx, avpl_draw_keypoints_ptx_len,
                         "kDrawKeypointsNV12Luma", "kDrawKeypointsNV12Chroma")) {
            throw Error("draw_keypoints: failed to initialize CUDA kernels");
        }

        const AVFrame* raw = input.raw();
        if (!raw || !raw->metadata) return;

        AVDictionaryEntry* entry = av_dict_get(raw->metadata, metadata_key_.c_str(), nullptr, 0);
        if (!entry || !entry->value) return;

        // Parse pose metadata
        std::vector<KeypointPos> points;
        double model_w = 0, model_h = 0;
        try {
            Parameters md = Parameters::parse(entry->value);
            model_w = md.value("model_width", 0.0);
            model_h = md.value("model_height", 0.0);
            int num_kpts = md.value("num_keypoints", 0);
            if (num_kpts <= 0) return;

            if (!md.contains("poses") || !md["poses"].is_array()) return;

            for (const auto& pose : md["poses"]) {
                if (!pose.contains("keypoints") || !pose["keypoints"].is_array()) continue;
                const auto& kpts = pose["keypoints"];
                int n = (int)kpts.size() / 3;
                for (int k = 0; k < n; ++k) {
                    float kx = kpts[(size_t)(k * 3 + 0)].get<float>();
                    float ky = kpts[(size_t)(k * 3 + 1)].get<float>();
                    float kc = kpts[(size_t)(k * 3 + 2)].get<float>();
                    if ((double)kc < min_conf_) continue;

                    double fx, fy;
                    remapModelCoord((double)kx, (double)ky, model_w, model_h, fx, fy);
                    // Clamp to frame bounds
                    if (fx < 0 || fy < 0 || fx >= input_params_.width || fy >= input_params_.height) continue;
                    points.push_back({(float)fx, (float)fy});
                }
            }
        } catch (const std::exception&) {
            return;
        }

        if (points.empty()) return;

        if (debug_log_every_n_ > 0 && (frame_counter_ % (uint64_t)debug_log_every_n_) == 0) {
            logstream << "draw_keypoints: frame=" << frame_counter_
                      << " points=" << points.size();
        }

        // Upload points to GPU
        size_t pts_bytes = points.size() * sizeof(KeypointPos);
        if (pts_bytes > gpu_points_capacity_) {
            if (gpu_points_buf_) cuMemFree(gpu_points_buf_);
            gpu_points_capacity_ = pts_bytes * 2;  // over-allocate
            if (CUDA_OVERLAY_CHECK_CU(cuMemAlloc(&gpu_points_buf_, gpu_points_capacity_))) {
                gpu_points_buf_ = 0;
                gpu_points_capacity_ = 0;
                return;
            }
        }
        if (CUDA_OVERLAY_CHECK_CU(cuMemcpyHtoDAsync(gpu_points_buf_, points.data(), pts_bytes, cuda_dev_ctx_->stream)))
            return;

        int num_points = (int)points.size();
        int radius = radius_;
        int frame_w = output.width();
        int frame_h = output.height();
        int y_color = color_.y;
        int u_color = color_.u;
        int v_color = color_.v;

        CUdeviceptr y_plane = (CUdeviceptr)(uintptr_t)output.raw()->data[0];
        size_t pitch_y = (size_t)output.raw()->linesize[0];
        CUdeviceptr uv_plane = (CUdeviceptr)(uintptr_t)output.raw()->data[1];
        size_t pitch_uv = (size_t)output.raw()->linesize[1];

        const unsigned int block_x = 32;
        const unsigned int block_y = 8;
        unsigned int grid_x = ((unsigned int)frame_w + block_x - 1) / block_x;
        unsigned int grid_y = ((unsigned int)frame_h + block_y - 1) / block_y;

        // Luma kernel
        void* y_args[] = {
            (void*)&y_plane, (void*)&pitch_y,
            (void*)&gpu_points_buf_, (void*)&num_points,
            (void*)&radius, (void*)&y_color,
            (void*)&frame_w, (void*)&frame_h
        };
        if (CUDA_OVERLAY_CHECK_CU(cuLaunchKernel(draw_luma_kernel_,
                                    grid_x, grid_y, 1,
                                    block_x, block_y, 1,
                                    0, cuda_dev_ctx_->stream, y_args, nullptr))) {
            logstream << "draw_keypoints: failed launching luma kernel";
            return;
        }

        // Chroma kernel
        unsigned int uv_grid_x = (((unsigned int)frame_w + 1) / 2 + block_x - 1) / block_x;
        unsigned int uv_grid_y = (((unsigned int)frame_h + 1) / 2 + block_y - 1) / block_y;
        void* uv_args[] = {
            (void*)&uv_plane, (void*)&pitch_uv,
            (void*)&gpu_points_buf_, (void*)&num_points,
            (void*)&radius, (void*)&u_color, (void*)&v_color,
            (void*)&frame_w, (void*)&frame_h
        };
        if (CUDA_OVERLAY_CHECK_CU(cuLaunchKernel(draw_chroma_kernel_,
                                    uv_grid_x, uv_grid_y, 1,
                                    block_x, block_y, 1,
                                    0, cuda_dev_ctx_->stream, uv_args, nullptr))) {
            logstream << "draw_keypoints: failed launching chroma kernel";
            return;
        }

        CUDA_OVERLAY_CHECK_CU(cuStreamSynchronize(cuda_dev_ctx_->stream));
    }

public:
    DrawKeypoints(std::unique_ptr<Source<av::VideoFrame>> &&source,
                  std::unique_ptr<Sink<av::VideoFrame>> &&sink,
                  std::string metadata_key,
                  DrawColor color,
                  int radius,
                  double min_conf,
                  double model_content_width,
                  double model_content_height,
                  double model_content_offset_x,
                  double model_content_offset_y,
                  VideoParameters input_params,
                  av::Rational frame_rate,
                  av::Rational timebase,
                  int debug_log_every_n)
        : CudaOverlayBase(std::move(source), std::move(sink)),
          metadata_key_(std::move(metadata_key)),
          color_(color),
          radius_(radius),
          min_conf_(min_conf),
          model_content_width_(model_content_width),
          model_content_height_(model_content_height),
          model_content_offset_x_(model_content_offset_x),
          model_content_offset_y_(model_content_offset_y),
          debug_log_every_n_(debug_log_every_n) {
        input_params_ = input_params;
        frame_rate_ = frame_rate;
        timebase_ = timebase;
    }

    ~DrawKeypoints() {
        if (gpu_points_buf_) {
            cuMemFree(gpu_points_buf_);
            gpu_points_buf_ = 0;
        }
    }

    static std::shared_ptr<DrawKeypoints> create(NodeCreationInfo &nci) {
        EdgeManager &edges = nci.edges;
        const Parameters &params = nci.params;

        auto src_edge = edges.find<av::VideoFrame>(params["src"]);
        const auto upstream = resolveUpstreamInfo(src_edge, params);

        const std::string metadata_key = params.value("metadata_key", std::string("yolo_pose"));
        const int radius = params.value("radius", 3);
        const double min_conf = params.value("min_conf", 0.0);
        const int debug_log_every_n = params.value("debug_log_every_n", 0);
        const double model_content_width = params.value("model_content_width", 0.0);
        const double model_content_height = params.value("model_content_height", 0.0);
        const double model_content_offset_x = params.value("model_content_offset_x", 0.0);
        const double model_content_offset_y = params.value("model_content_offset_y", 0.0);

        DrawColor color{};
        std::string color_name = params.value("color", std::string("white"));
        if (!cuda_overlay::tryParseNamedColor(color_name, color)) {
            throw Error("draw_keypoints: unknown color: " + color_name);
        }

        return NodeSISO<av::VideoFrame, av::VideoFrame>::template createCommon<DrawKeypoints>(
            edges, params, metadata_key, color, radius, min_conf,
            model_content_width, model_content_height, model_content_offset_x, model_content_offset_y,
            upstream.input_params, upstream.frame_rate, upstream.timebase, debug_log_every_n);
    }
};

DECLNODE(draw_keypoints, DrawKeypoints)
```

- [ ] **Step 2: Verify remote build compiles**

Rsync and build on remote:

```bash
rsync -avz --relative -e "ssh -i /home/jp/work-misc-stuff/awsdev.pem" \
  /home/jp/git/avplumber/./{src/nodes/hwaccel/draw_keypoints.cpp,src/nodes/hwaccel/draw_keypoints.cu,src/nodes/hwaccel/yolo_decode_pose.hpp,src/nodes/hwaccel/cuda_infer_yolo_base.hpp,src/nodes/hwaccel/cuda_infer_yolo.cpp,Makefile} \
  fedora@172.17.36.132:/home/fedora/avplumber/
```

Then on remote:

```bash
make clean && make -j8 \
  HAVE_CUDA=1 HAVE_TENSORRT=1 HAVE_NVCC=1 \
  NVCC=/usr/local/cuda-13.0/bin/nvcc \
  TENSORRT_ROOT=/opt/tensorrt \
  PKG_CONFIG_PATH=/apps/ffmpeg/lib/pkgconfig \
  CXXFLAGS+=' -I/usr/local/cuda-13.0/include -I/usr/local/cuda-13.0/targets/x86_64-linux/include' \
  LFLAGS+=' -L/apps/ffmpeg/lib -Wl,-rpath,/apps/ffmpeg/lib -L/usr/local/cuda-13.0/targets/x86_64-linux/lib -Wl,-rpath,/usr/local/cuda-13.0/targets/x86_64-linux/lib'
```

Expected: compiles and links cleanly.

- [ ] **Step 3: Commit**

```bash
git add src/nodes/hwaccel/draw_keypoints.cpp
git commit -m "feat: add draw_keypoints node for rendering pose keypoints on NV12"
```

---

### Task 7: Update example pipeline

**Files:**
- Modify: `examples/yolo/yolo_reframe_crop.avplumber`

- [ ] **Step 1: Add court-pose model to cuda_infer_yolo models array**

In `examples/yolo/yolo_reframe_crop.avplumber`, in the `cuda_infer_yolo` node (line 24), add the court-pose model as the 5th entry in the `models` array, after the court-segmentation entry:

```json
, { "engine": "/home/fedora/tensorrt/court-pose-2/court-pose.plan", "task_type": "pose", "class_names": ["court"], "num_classes": 1, "output_box_format": "raw_cxcywh", "include_in_detection_metadata": false }
```

Also add `"metadata_key_pose": "yolo_pose"` to the top-level params of the node (alongside the existing `metadata_key_detection` and `metadata_key_segmentation`).

- [ ] **Step 2: Add draw_keypoints node in the draw chain**

After the `draw_segmask` node (line 31) and before the `smooth_crop_viewport` node, insert:

```
node.add { "type": "draw_keypoints", "src": "v_segmask", "dst": "v_keypoints", "group": "in", "name": "Draw_Keypoints", "metadata_key": "yolo_pose", "radius": 5, "color": "white", "min_conf": 0.3, "model_content_width": 960, "model_content_height": 540, "model_content_offset_x": 0, "model_content_offset_y": 210, "width": 1920, "height": 1080, "pixel_format": "cuda", "real_pixel_format": "nv12", "debug_log_every_n": 30 }
```

Then update the `smooth_crop_viewport` node's `src` from `"v_segmask"` to `"v_keypoints"`.

Also update `draw_bbox`'s `src` from `"v_smoothed_crop_md"` — check that it still chains correctly through `v_keypoints` → `v_smoothed_crop_md` → `v_annotated_cuda`.

- [ ] **Step 3: Commit**

```bash
git add examples/yolo/yolo_reframe_crop.avplumber
git commit -m "feat: add court-pose model and draw_keypoints to reframe pipeline"
```

---

### Task 8: Remote build and smoke test

- [ ] **Step 1: Rsync all changed files to remote**

```bash
rsync -avz --relative -e "ssh -i /home/jp/work-misc-stuff/awsdev.pem" \
  /home/jp/git/avplumber/./{src/nodes/hwaccel/draw_keypoints.cpp,src/nodes/hwaccel/draw_keypoints.cu,src/nodes/hwaccel/yolo_decode_pose.hpp,src/nodes/hwaccel/cuda_infer_yolo_base.hpp,src/nodes/hwaccel/cuda_infer_yolo.cpp,Makefile,examples/yolo/yolo_reframe_crop.avplumber} \
  fedora@172.17.36.132:/home/fedora/avplumber/
```

- [ ] **Step 2: Clean build on remote**

```bash
ssh -i /home/jp/work-misc-stuff/awsdev.pem fedora@172.17.36.132 \
  "cd /home/fedora/avplumber && make clean && make -j8 \
  HAVE_CUDA=1 HAVE_TENSORRT=1 HAVE_NVCC=1 \
  NVCC=/usr/local/cuda-13.0/bin/nvcc \
  TENSORRT_ROOT=/opt/tensorrt \
  PKG_CONFIG_PATH=/apps/ffmpeg/lib/pkgconfig \
  CXXFLAGS+=' -I/usr/local/cuda-13.0/include -I/usr/local/cuda-13.0/targets/x86_64-linux/include' \
  LFLAGS+=' -L/apps/ffmpeg/lib -Wl,-rpath,/apps/ffmpeg/lib -L/usr/local/cuda-13.0/targets/x86_64-linux/lib -Wl,-rpath,/usr/local/cuda-13.0/targets/x86_64-linux/lib'"
```

Expected: compiles and links cleanly.

- [ ] **Step 3: Smoke test with the example pipeline**

```bash
ssh -i /home/jp/work-misc-stuff/awsdev.pem fedora@172.17.36.132 \
  "cd /home/fedora/avplumber && timeout 10 ./avplumber -s examples/yolo/yolo_reframe_crop.avplumber 2>&1 | tail -30"
```

Look for:
- No crash on startup
- `cuda_infer_yolo: pose model=court-pose.plan` log lines appear
- `draw_keypoints: frame=... points=...` log lines appear
- Keypoints visible in output file (inspect manually via ffplay or screenshot)
