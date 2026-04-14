# Court Polygon Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a `court_polygon` node that converts pose keypoints into a segmentation mask compatible with the existing ball_tracker and shot_classifier consumers.

**Architecture:** Single `NodeSISO<VideoFrame, VideoFrame>` in `src/nodes/neural_net/sport_specific/court_polygon.cpp`. Reads `yolo_pose` metadata, rasterizes a filled polygon on CPU, attaches CPU + GPU side data in the existing YOLO seg mask format. No downstream changes required.

**Tech Stack:** C++, FFmpeg libavutil side data, CUDA driver API (cuMemAlloc/cuMemcpyHtoD via cuda_loader)

**Spec:** `doc/specs/2026-04-14-court-polygon-design.md`

---

### Task 1: Move GpuMaskSideDataHeader to shared header

`GpuMaskSideDataHeader` is currently defined in `infer_trt_base.hpp` which pulls in TensorRT. The court_polygon node needs this struct but has no TensorRT dependency. Move it to `yolo_side_data.hpp` where the other side data constants live, and update `infer_trt_base.hpp` to not duplicate it.

**Files:**
- Modify: `src/nodes/neural_net/common/yolo_side_data.hpp`
- Modify: `src/nodes/neural_net/common/infer_trt_base.hpp`

- [ ] **Step 1: Add GpuMaskSideDataHeader to yolo_side_data.hpp**

Add the struct and required include after the existing side data type constants in `src/nodes/neural_net/common/yolo_side_data.hpp`:

```cpp
#pragma once

extern "C" {
#include <libavutil/frame.h>
}

#include <cstdint>

// YOLO segmentation side-data ids used across nodes. Kept separate from
// TensorRT-specific headers so non-TRT nodes can parse side-data safely.
static const AVFrameSideDataType AV_FRAME_DATA_YOLO_SEG_MASKS = (AVFrameSideDataType)0x59534D00;
static const AVFrameSideDataType AV_FRAME_DATA_YOLO_SEG_MASKS_GPU = (AVFrameSideDataType)0x59534D01;

// Header for GPU mask side data (lives in CPU memory, gpu_ptr is a CUdeviceptr)
struct GpuMaskSideDataHeader {
    uint64_t gpu_ptr;
    uint32_t num_masks;
    uint32_t proto_w;
    uint32_t proto_h;
    uint32_t model_w;
    uint32_t model_h;
};
```

- [ ] **Step 2: Remove duplicate from infer_trt_base.hpp**

In `src/nodes/neural_net/common/infer_trt_base.hpp`, remove the duplicated constants and struct (lines 28-40). They are now provided by `yolo_side_data.hpp`. The file already includes headers that transitively include `yolo_side_data.hpp` via other nodes, but to be safe, add an explicit include if not present.

Remove these lines from `infer_trt_base.hpp`:

```cpp
// Project-local side data type for segmentation masks
static const AVFrameSideDataType AV_FRAME_DATA_YOLO_SEG_MASKS = (AVFrameSideDataType)0x59534D00;
static const AVFrameSideDataType AV_FRAME_DATA_YOLO_SEG_MASKS_GPU = (AVFrameSideDataType)0x59534D01;

// Header for GPU mask side data (lives in CPU memory, gpu_ptr is a CUdeviceptr)
struct GpuMaskSideDataHeader {
    uint64_t gpu_ptr;
    uint32_t num_masks;
    uint32_t proto_w;
    uint32_t proto_h;
    uint32_t model_w;
    uint32_t model_h;
};
```

And add at the top of `infer_trt_base.hpp` (after the existing includes):

```cpp
#include "yolo_side_data.hpp"
```

- [ ] **Step 3: Verify build**

Run: `make clean && make -j$(nproc) NEURAL_NET_COMMON=1 NEURAL_NET_SPECIFIC=1 HAVE_CUDA=1`

This must compile without errors or duplicate symbol warnings. All existing consumers of either header should work unchanged.

- [ ] **Step 4: Commit**

```bash
git add src/nodes/neural_net/common/yolo_side_data.hpp src/nodes/neural_net/common/infer_trt_base.hpp
git commit -m "refactor: move GpuMaskSideDataHeader to yolo_side_data.hpp

Shared side data struct used by both TRT inference and the new
court_polygon node. Moving it out of infer_trt_base.hpp avoids
pulling in TensorRT headers where they are not needed."
```

---

### Task 2: Implement court_polygon node — skeleton, param parsing, metadata read

Create the node file with the class structure, parameter parsing, pose metadata reading, and passthrough for frames with no valid pose. No rasterization yet — just the node shell that compiles and passes frames through.

**Files:**
- Create: `src/nodes/neural_net/sport_specific/court_polygon.cpp`

- [ ] **Step 1: Create court_polygon.cpp with class skeleton**

Create `src/nodes/neural_net/sport_specific/court_polygon.cpp`:

```cpp
#include "../../node_common.hpp"
#include "../common/yolo_side_data.hpp"

extern "C" {
#include <libavutil/dict.h>
#include <libavutil/frame.h>
#include <libavutil/hwcontext.h>
#include <libavutil/hwcontext_cuda.h>
}

#include <cuda_loader/cuda_drvapi_dynlink_cuda.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

struct CourtKeypoint {
    float x = 0.0f;
    float y = 0.0f;
    float conf = 0.0f;
    bool visible = false;
};

class CourtPolygon : public NodeSISO<av::VideoFrame, av::VideoFrame> {
private:
    // Params
    std::string pose_metadata_key_ = "yolo_pose";
    std::string metadata_key_out_ = "yolo_seg";
    int mask_w_ = 240;
    int mask_h_ = 136;
    float min_keypoint_conf_ = 0.3f;
    int min_visible_keypoints_ = 6;
    float min_court_area_ = 0.05f;
    std::vector<int> winding_order_ = {0, 3, 5, 7, 9, 10, 11, 8, 6, 4, 2, 1};
    int debug_log_every_n_ = 0;

    // CUDA state
    CUcontext cu_ctx_ = nullptr;
    CUdeviceptr gpu_mask_buf_ = 0;
    size_t gpu_mask_buf_size_ = 0;

    // Frame counter
    uint64_t frame_counter_ = 0;

    // Reusable CPU mask buffer
    std::vector<float> cpu_mask_;

    bool shouldDebugLog() const {
        return debug_log_every_n_ > 0 && (frame_counter_ % (uint64_t)debug_log_every_n_) == 0;
    }

    bool initCudaContext(const av::VideoFrame& frm) {
        if (!frm.raw() || !frm.raw()->hw_frames_ctx || !frm.raw()->hw_frames_ctx->data) {
            return false;
        }
        AVHWFramesContext* fctx = (AVHWFramesContext*)frm.raw()->hw_frames_ctx->data;
        if (!fctx || !fctx->device_ctx || !fctx->device_ctx->hwctx) {
            return false;
        }
        AVCUDADeviceContext* cuda_dev = (AVCUDADeviceContext*)fctx->device_ctx->hwctx;
        if (!cuda_dev || !cuda_dev->cuda_ctx) {
            return false;
        }
        cu_ctx_ = cuda_dev->cuda_ctx;
        return cuCtxSetCurrent(cu_ctx_) == CUDA_SUCCESS;
    }

    bool parsePoseKeypoints(const av::VideoFrame& frm,
                            std::vector<CourtKeypoint>& keypoints_out,
                            float& model_w_out, float& model_h_out,
                            float& det_conf_out) const {
        const AVFrame* raw = frm.raw();
        if (!raw || !raw->metadata) return false;

        AVDictionaryEntry* entry = av_dict_get(raw->metadata, pose_metadata_key_.c_str(), nullptr, 0);
        if (!entry || !entry->value) return false;

        Parameters pose_md;
        try {
            pose_md = Parameters::parse(entry->value);
        } catch (...) {
            return false;
        }

        model_w_out = pose_md.value("model_width", 0.0f);
        model_h_out = pose_md.value("model_height", 0.0f);
        if (model_w_out <= 0.0f || model_h_out <= 0.0f) return false;

        if (!pose_md.contains("poses") || !pose_md["poses"].is_array() || pose_md["poses"].empty()) {
            return false;
        }

        // Find highest confidence pose detection
        int best_idx = -1;
        float best_conf = -1.0f;
        for (size_t i = 0; i < pose_md["poses"].size(); ++i) {
            float conf = pose_md["poses"][i].value("conf", 0.0f);
            if (conf > best_conf) {
                best_conf = conf;
                best_idx = (int)i;
            }
        }
        if (best_idx < 0) return false;

        const auto& pose = pose_md["poses"][(size_t)best_idx];
        det_conf_out = best_conf;

        if (!pose.contains("keypoints") || !pose["keypoints"].is_array()) return false;
        const auto& kpts = pose["keypoints"];

        int num_kpts = pose_md.value("num_keypoints", 0);
        if (num_kpts <= 0 || (int)kpts.size() < num_kpts * 3) return false;

        keypoints_out.resize((size_t)num_kpts);
        for (int i = 0; i < num_kpts; ++i) {
            keypoints_out[(size_t)i].x = kpts[(size_t)(i * 3)].get<float>();
            keypoints_out[(size_t)i].y = kpts[(size_t)(i * 3 + 1)].get<float>();
            keypoints_out[(size_t)i].conf = kpts[(size_t)(i * 3 + 2)].get<float>();
            keypoints_out[(size_t)i].visible = keypoints_out[(size_t)i].conf >= min_keypoint_conf_;
        }
        return true;
    }

public:
    using NodeSISO<av::VideoFrame, av::VideoFrame>::NodeSISO;

    ~CourtPolygon() override {
        if (gpu_mask_buf_ && cu_ctx_) {
            cuCtxSetCurrent(cu_ctx_);
            cuMemFree(gpu_mask_buf_);
            gpu_mask_buf_ = 0;
        }
    }

    void process() override {
        av::VideoFrame frm = this->source_->get();
        if (!frm) return;

        if (isEofMarker(frm)) {
            frame_counter_ = 0;
            this->sink_->put(frm);
            return;
        }

        ++frame_counter_;

        // Parse pose keypoints
        std::vector<CourtKeypoint> keypoints;
        float model_w = 0.0f, model_h = 0.0f, det_conf = 0.0f;
        if (!parsePoseKeypoints(frm, keypoints, model_w, model_h, det_conf)) {
            if (shouldDebugLog()) {
                logstream << "court_polygon: frame=" << frame_counter_ << " no_pose";
            }
            this->sink_->put(frm);
            return;
        }

        // TODO: Task 3 — build polygon, validate, rasterize
        // TODO: Task 4 — attach side data

        if (shouldDebugLog()) {
            int visible = 0;
            for (const auto& kp : keypoints) {
                if (kp.visible) ++visible;
            }
            logstream << "court_polygon: frame=" << frame_counter_
                      << " keypoints=" << visible << "/" << keypoints.size()
                      << " model=" << model_w << "x" << model_h
                      << " conf=" << det_conf;
        }

        this->sink_->put(frm);
    }

    static std::shared_ptr<CourtPolygon> create(NodeCreationInfo& nci) {
        EdgeManager& edges = nci.edges;
        const Parameters& params = nci.params;
        auto r = NodeSISO<av::VideoFrame, av::VideoFrame>::template createCommon<CourtPolygon>(edges, params);

        if (params.count("pose_metadata_key")) r->pose_metadata_key_ = params["pose_metadata_key"].get<std::string>();
        if (params.count("metadata_key_out")) r->metadata_key_out_ = params["metadata_key_out"].get<std::string>();
        if (params.count("mask_w")) r->mask_w_ = params["mask_w"].get<int>();
        if (params.count("mask_h")) r->mask_h_ = params["mask_h"].get<int>();
        if (params.count("min_keypoint_conf")) r->min_keypoint_conf_ = params["min_keypoint_conf"].get<float>();
        if (params.count("min_visible_keypoints")) r->min_visible_keypoints_ = params["min_visible_keypoints"].get<int>();
        if (params.count("min_court_area")) r->min_court_area_ = params["min_court_area"].get<float>();
        if (params.count("winding_order")) {
            r->winding_order_.clear();
            for (const auto& item : params["winding_order"]) {
                r->winding_order_.push_back(item.get<int>());
            }
        }
        if (params.count("debug_log_every_n")) r->debug_log_every_n_ = params["debug_log_every_n"].get<int>();

        return r;
    }
};

DECLNODE(court_polygon, CourtPolygon)
```

- [ ] **Step 2: Verify build**

Run: `make clean && make -j$(nproc) NEURAL_NET_COMMON=1 NEURAL_NET_SPECIFIC=1 HAVE_CUDA=1`

The node should compile and appear in the generated node factory. Verify:

```bash
grep court_polygon objs/graph_factory.generated.cpp
```

Expected: a line registering the `court_polygon` factory.

- [ ] **Step 3: Commit**

```bash
git add src/nodes/neural_net/sport_specific/court_polygon.cpp
git commit -m "feat: add court_polygon node skeleton with param parsing

Reads yolo_pose metadata and parses keypoints. Passthrough only —
polygon rasterization and side data attachment in next commits."
```

---

### Task 3: Implement polygon building, validation, and rasterization

Add the core geometry: build the polygon from keypoints in winding order, validate convexity, rasterize via CPU scanline fill, and check area.

**Files:**
- Modify: `src/nodes/neural_net/sport_specific/court_polygon.cpp`

- [ ] **Step 1: Add polygon building and validation methods**

Add the `PolygonVertex` struct inside the `CourtPolygon` class (private section, alongside `CourtKeypoint`-using code), then add the methods after `parsePoseKeypoints`:

```cpp
    struct PolygonVertex {
        float x = 0.0f;
        float y = 0.0f;
    };

    bool buildPolygon(const std::vector<CourtKeypoint>& keypoints,
                      std::vector<PolygonVertex>& polygon_out) const {
        polygon_out.clear();
        for (int idx : winding_order_) {
            if (idx < 0 || idx >= (int)keypoints.size()) continue;
            if (!keypoints[(size_t)idx].visible) continue;
            polygon_out.push_back({keypoints[(size_t)idx].x, keypoints[(size_t)idx].y});
        }
        return (int)polygon_out.size() >= min_visible_keypoints_;
    }

    static bool isConvex(const std::vector<PolygonVertex>& poly) {
        const int n = (int)poly.size();
        if (n < 3) return false;

        bool got_sign = false;
        bool positive = false;

        for (int i = 0; i < n; ++i) {
            const auto& a = poly[(size_t)i];
            const auto& b = poly[(size_t)((i + 1) % n)];
            const auto& c = poly[(size_t)((i + 2) % n)];

            float cross = (b.x - a.x) * (c.y - b.y) - (b.y - a.y) * (c.x - b.x);

            if (std::abs(cross) < 1e-6f) continue; // collinear, skip

            if (!got_sign) {
                positive = (cross > 0.0f);
                got_sign = true;
            } else if ((cross > 0.0f) != positive) {
                return false;
            }
        }
        return got_sign; // at least one non-collinear triple
    }
```

- [ ] **Step 2: Add scanline rasterizer**

Add this private method after `isConvex`:

```cpp
    float rasterizePolygon(const std::vector<PolygonVertex>& poly,
                           float model_w, float model_h) {
        const size_t total_pixels = (size_t)mask_w_ * (size_t)mask_h_;
        cpu_mask_.assign(total_pixels, 0.0f);

        // Map polygon vertices to mask space
        std::vector<PolygonVertex> mask_poly(poly.size());
        for (size_t i = 0; i < poly.size(); ++i) {
            mask_poly[i].x = poly[i].x / model_w * (float)mask_w_;
            mask_poly[i].y = poly[i].y / model_h * (float)mask_h_;
        }

        const int n = (int)mask_poly.size();
        size_t filled = 0;

        for (int row = 0; row < mask_h_; ++row) {
            float y = (float)row + 0.5f;

            // Find edge intersections with this scanline
            std::vector<float> intersections;
            for (int i = 0; i < n; ++i) {
                const auto& a = mask_poly[(size_t)i];
                const auto& b = mask_poly[(size_t)((i + 1) % n)];

                if ((a.y <= y && b.y > y) || (b.y <= y && a.y > y)) {
                    float t = (y - a.y) / (b.y - a.y);
                    intersections.push_back(a.x + t * (b.x - a.x));
                }
            }

            std::sort(intersections.begin(), intersections.end());

            // Fill between pairs
            for (size_t i = 0; i + 1 < intersections.size(); i += 2) {
                int x_start = std::max(0, (int)std::ceil(intersections[i]));
                int x_end = std::min(mask_w_, (int)std::floor(intersections[i + 1]) + 1);
                for (int x = x_start; x < x_end; ++x) {
                    cpu_mask_[(size_t)row * (size_t)mask_w_ + (size_t)x] = 1.0f;
                    ++filled;
                }
            }
        }

        return total_pixels > 0 ? (float)filled / (float)total_pixels : 0.0f;
    }
```

- [ ] **Step 3: Wire into process()**

Replace the `// TODO: Task 3` block in `process()` with:

```cpp
        // Build polygon from visible keypoints in winding order
        std::vector<PolygonVertex> polygon;
        if (!buildPolygon(keypoints, polygon)) {
            if (shouldDebugLog()) {
                int visible = 0;
                for (const auto& kp : keypoints) if (kp.visible) ++visible;
                logstream << "court_polygon: frame=" << frame_counter_
                          << " too_few_keypoints visible=" << visible
                          << " required=" << min_visible_keypoints_;
            }
            this->sink_->put(frm);
            return;
        }

        // Validate convexity
        if (!isConvex(polygon)) {
            if (shouldDebugLog()) {
                logstream << "court_polygon: frame=" << frame_counter_
                          << " non_convex vertices=" << polygon.size();
            }
            this->sink_->put(frm);
            return;
        }

        // Rasterize polygon
        float court_area = rasterizePolygon(polygon, model_w, model_h);

        // Area check
        if (court_area < min_court_area_) {
            if (shouldDebugLog()) {
                logstream << "court_polygon: frame=" << frame_counter_
                          << " area_too_small=" << (int)(court_area * 100) << "%"
                          << " min=" << (int)(min_court_area_ * 100) << "%";
            }
            this->sink_->put(frm);
            return;
        }

        // TODO: Task 4 — attach side data
```

Also update the debug log at the end of process() to include polygon info:

```cpp
        if (shouldDebugLog()) {
            logstream << "court_polygon: frame=" << frame_counter_
                      << " vertices=" << polygon.size()
                      << " area=" << (int)(court_area * 100) << "%"
                      << " model=" << model_w << "x" << model_h
                      << " conf=" << det_conf;
        }
```

- [ ] **Step 4: Verify build**

Run: `make -j$(nproc) NEURAL_NET_COMMON=1 NEURAL_NET_SPECIFIC=1 HAVE_CUDA=1`

- [ ] **Step 5: Commit**

```bash
git add src/nodes/neural_net/sport_specific/court_polygon.cpp
git commit -m "feat: court_polygon polygon building, validation, rasterization

Build polygon from keypoints in winding order, validate convexity,
CPU scanline fill rasterization, area check for closeup rejection."
```

---

### Task 4: Attach CPU and GPU side data + write metadata

Complete the node by attaching the rasterized mask as both CPU and GPU side data, and writing the yolo_seg-compatible metadata.

**Files:**
- Modify: `src/nodes/neural_net/sport_specific/court_polygon.cpp`

- [ ] **Step 1: Add CPU side data attachment method**

Add this private method after `rasterizePolygon`:

```cpp
    void attachCpuSideData(av::VideoFrame& frm) const {
        const size_t header_size = 16;
        const size_t mask_data_size = cpu_mask_.size() * sizeof(float);
        const size_t total_size = header_size + mask_data_size;

        AVBufferRef* buf = av_buffer_alloc((int)total_size);
        if (!buf) return;

        uint32_t* header = (uint32_t*)buf->data;
        header[0] = 1;                  // num_masks
        header[1] = (uint32_t)mask_w_;  // mask_w
        header[2] = (uint32_t)mask_h_;  // mask_h
        header[3] = 0;                  // reserved

        std::memcpy(buf->data + header_size, cpu_mask_.data(), mask_data_size);
        av_frame_new_side_data_from_buf(frm.raw(), AV_FRAME_DATA_YOLO_SEG_MASKS, buf);
    }
```

- [ ] **Step 2: Add GPU side data attachment method**

Add this private method after `attachCpuSideData`:

```cpp
    void attachGpuSideData(av::VideoFrame& frm, float model_w, float model_h) {
        if (!cu_ctx_) {
            if (!initCudaContext(frm)) return;
        }
        if (cuCtxSetCurrent(cu_ctx_) != CUDA_SUCCESS) return;

        const size_t mask_bytes = cpu_mask_.size() * sizeof(float);

        // Allocate or reuse GPU buffer
        if (!gpu_mask_buf_ || gpu_mask_buf_size_ != mask_bytes) {
            if (gpu_mask_buf_) {
                cuMemFree(gpu_mask_buf_);
                gpu_mask_buf_ = 0;
            }
            if (cuMemAlloc(&gpu_mask_buf_, mask_bytes) != CUDA_SUCCESS) {
                gpu_mask_buf_ = 0;
                return;
            }
            gpu_mask_buf_size_ = mask_bytes;
        }

        // Upload mask to GPU
        if (cuMemcpyHtoD(gpu_mask_buf_, cpu_mask_.data(), mask_bytes) != CUDA_SUCCESS) {
            return;
        }

        // Create header
        auto* header = (GpuMaskSideDataHeader*)av_malloc(sizeof(GpuMaskSideDataHeader));
        if (!header) return;

        header->gpu_ptr = (uint64_t)gpu_mask_buf_;
        header->num_masks = 1;
        header->proto_w = (uint32_t)mask_w_;
        header->proto_h = (uint32_t)mask_h_;
        header->model_w = (uint32_t)model_w;
        header->model_h = (uint32_t)model_h;

        // Wrap in AVBufferRef — free callback only frees the header,
        // not the GPU buffer (which is owned by this node and reused)
        AVBufferRef* sd_buf = av_buffer_create(
            (uint8_t*)header, sizeof(GpuMaskSideDataHeader),
            [](void*, uint8_t* data) { av_free(data); },
            nullptr, 0);

        if (sd_buf) {
            av_frame_new_side_data_from_buf(frm.raw(), AV_FRAME_DATA_YOLO_SEG_MASKS_GPU, sd_buf);
        } else {
            av_free(header);
        }
    }
```

- [ ] **Step 3: Add metadata writing method**

Add this private method after `attachGpuSideData`:

```cpp
    void writeSegMetadata(av::VideoFrame& frm, float model_w, float model_h, float det_conf) const {
        Parameters seg_md;
        seg_md["model_width"] = model_w;
        seg_md["model_height"] = model_h;

        Parameters det;
        det["cls"] = 0;
        det["conf"] = det_conf;
        det["xyxy"] = {0.0f, 0.0f, model_w, model_h};
        det["label"] = "court";

        seg_md["detections"] = Parameters::array();
        seg_md["detections"].push_back(det);

        av_dict_set(&frm.raw()->metadata, metadata_key_out_.c_str(), seg_md.dump().c_str(), 0);
    }
```

- [ ] **Step 4: Wire into process()**

Replace the `// TODO: Task 4` comment in `process()` with:

```cpp
        // Attach CPU side data
        attachCpuSideData(frm);

        // Attach GPU side data (skipped if not a CUDA frame)
        attachGpuSideData(frm, model_w, model_h);

        // Write yolo_seg-compatible metadata
        writeSegMetadata(frm, model_w, model_h, det_conf);
```

- [ ] **Step 5: Verify build**

Run: `make -j$(nproc) NEURAL_NET_COMMON=1 NEURAL_NET_SPECIFIC=1 HAVE_CUDA=1`

- [ ] **Step 6: Commit**

```bash
git add src/nodes/neural_net/sport_specific/court_polygon.cpp
git commit -m "feat: court_polygon CPU+GPU side data and metadata output

Attaches AV_FRAME_DATA_YOLO_SEG_MASKS (CPU) and
AV_FRAME_DATA_YOLO_SEG_MASKS_GPU side data. Writes yolo_seg
metadata. Drop-in replacement for segmentation model output."
```

---

### Task 5: Sync to remote, build, and test end-to-end

Build on the remote GPU instance and run the tracker pipeline with the pose model + court_polygon replacing the segmentation model.

**Files:**
- No code changes — integration testing only

- [ ] **Step 1: Rsync changed files to remote**

```bash
rsync -avz --relative -e "ssh -i /home/jp/work-misc-stuff/awsdev.pem" \
  /home/jp/git/avplumber/./src/nodes/neural_net/common/yolo_side_data.hpp \
  /home/jp/git/avplumber/./src/nodes/neural_net/common/infer_trt_base.hpp \
  /home/jp/git/avplumber/./src/nodes/neural_net/sport_specific/court_polygon.cpp \
  fedora@172.17.36.132:/home/fedora/avplumber/
```

- [ ] **Step 2: Build on remote**

```bash
ssh -i /home/jp/work-misc-stuff/awsdev.pem fedora@172.17.36.132 \
  'cd /home/fedora/avplumber && make clean && make -j8 \
  NEURAL_NET_COMMON=1 NEURAL_NET_SPECIFIC=1 HAVE_CUDA=1 \
  HAVE_NVOF_FRUC=1 HAVE_NVCC=1 \
  NVCC=/usr/local/cuda-13.0/bin/nvcc \
  TENSORRT_ROOT=/opt/tensorrt \
  PKG_CONFIG_PATH=/usr/local/lib/pkgconfig \
  CXXFLAGS+=" -I/usr/local/include -I/usr/local/cuda-13.0/include -I/usr/local/cuda-13.0/targets/x86_64-linux/include" \
  LFLAGS+=" -L/usr/local/lib -Wl,-rpath,/usr/local/lib -L/usr/local/cuda-13.0/targets/x86_64-linux/lib -Wl,-rpath,/usr/local/cuda-13.0/targets/x86_64-linux/lib"'
```

Expected: clean build, no errors.

- [ ] **Step 3: Verify node registration**

```bash
ssh -i /home/jp/work-misc-stuff/awsdev.pem fedora@172.17.36.132 \
  'grep court_polygon /home/fedora/avplumber/objs/graph_factory.generated.cpp'
```

Expected: factory registration line for `court_polygon`.

- [ ] **Step 4: Run tracker-vod pipeline with court_polygon**

Create or modify a test `.avplumber` script on the remote that uses the pose model + `court_polygon` instead of the segmentation model. Run it with the demo NBA clip and verify:

1. `shot_classifier` logs show reasonable `court_coverage` values
2. `ball_tracker` court bounds veto fires on out-of-court detections
3. `draw_segmask` renders a visible court overlay
4. On closeup shots, no mask is produced (debug logs show `area_too_small` or `too_few_keypoints`)

- [ ] **Step 5: Commit test results notes (if any param tuning needed)**

If any default parameter values needed adjustment during testing, update them in the source and commit.
