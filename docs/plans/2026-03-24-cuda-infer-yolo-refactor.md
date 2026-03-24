# cuda_infer_yolo Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor the monolithic `cuda_infer_yolo` node into a base class + task-specific decoders, fix all code quality issues, and add segmentation support.

**Architecture:** Base class handles TRT/CUDA infrastructure (engine loading, preprocess, inference, cleanup). DetectionDecoder and SegmentationDecoder handle output interpretation. The node orchestrates process() and metadata assembly. GPU masks use AVBufferRef with custom release callbacks; CPU masks use AVFrame side data.

**Tech Stack:** C++17, TensorRT, CUDA Driver API (no runtime), FFmpeg libavutil, nlohmann/json

**Spec:** `docs/specs/2026-03-24-cuda-infer-yolo-refactor-design.md`

---

## File Map

| File | Action | Responsibility |
|------|--------|---------------|
| `src/nodes/hwaccel/cuda_infer_yolo_base.hpp` | Create | Structs, enums, utility functions, CudaInferYoloBase class declaration |
| `src/nodes/hwaccel/cuda_infer_yolo_base.cpp` | Create | Base class implementation: engine loading, bindings, preprocess, inference, sync, cleanup |
| `src/nodes/hwaccel/yolo_decode_detection.hpp` | Create | DetectionDecoder: unified decode with accessor lambda |
| `src/nodes/hwaccel/yolo_decode_segmentation.hpp` | Create | SegmentationDecoder: detection+coefficient decode, mask assembly dispatch, GPU/CPU paths |
| `src/nodes/hwaccel/yolo_mask_assemble.cu` | Create | CUDA kernel: coefficients x prototypes matmul + sigmoid |
| `src/nodes/hwaccel/cuda_infer_yolo.cpp` | Rewrite | Node: DECLNODE, create(), process(), metadata assembly (replaces current 907-line monolith) |
| `src/nodes/join_metadata.cpp` | Modify | Add side data copying from auxiliary to primary frame |
| `Makefile` | Modify | Add PTX build rule for yolo_mask_assemble.cu |
| `examples/yolo/yolo_bbox.avplumber` | Modify | Add explicit task_type per model, update metadata key names |
| `examples/yolo/rtmp_input_hw_dec_cuda_yolo.avplumber` | Modify | Same param updates |
| `examples/rtmp_input_hw_dec_cuda_yolo.avplumber` | Modify | Same param updates |
| `library_examples/obs-avplumber-source/examples/rtmp_input_hw_dec_cuda_yolo.txt` | Modify | Same param updates |
| `node_documentation/cuda_infer_yolo.md` | Modify | Update documentation for new params |

---

## Task 1: Create base header with structs, enums, and utilities

**Files:**
- Create: `src/nodes/hwaccel/cuda_infer_yolo_base.hpp`

This is the foundation. All other files depend on the types defined here.

- [ ] **Step 1: Create the base header file**

Write `cuda_infer_yolo_base.hpp` containing:

```cpp
#pragma once

#include "../node_common.hpp"
#include "../../hwaccel.hpp"
#include <cuda_loader/cuda_drvapi_dynlink_cuda.h>

extern "C" {
#include <libavutil/dict.h>
#include <libavutil/hwcontext.h>
#include <libavutil/hwcontext_cuda.h>
#include <libavutil/buffer.h>
#include <libavutil/frame.h>
}

#include <NvInfer.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

// Project-local side data type for segmentation masks
static const AVFrameSideDataType AV_FRAME_DATA_YOLO_SEG_MASKS = (AVFrameSideDataType)0x59534D00;

namespace yolo_base {

// --- CUDA error checking (shared with cuda_overlay_base.hpp pattern) ---
inline int check_cu(CUresult err, const char *func) {
    if (err == CUDA_SUCCESS) return 0;
    const char *err_name = nullptr;
    const char *err_string = nullptr;
    if (cuGetErrorName && cuGetErrorString) {
        cuGetErrorName(err, &err_name);
        cuGetErrorString(err, &err_string);
    }
    logstream << "cuda function: " << func << " failed: "
              << (err_name ? err_name : "?") << ": " << (err_string ? err_string : "?");
    return -1;
}
#define YOLO_CHECK_CU(x) yolo_base::check_cu((x), #x)

// --- TensorRT logger ---
class TRTLogger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity == Severity::kERROR || severity == Severity::kINTERNAL_ERROR || severity == Severity::kWARNING) {
            logstream << "tensorrt: " << (msg ? msg : "");
        }
    }
};

// --- Enums ---
enum class TaskType { Detection, Segmentation, Pose };
enum class OutputBoxFormat { EndToEndXYXY, RawCXCYWH };

// --- Detection struct ---
struct Detection {
    float x1 = 0.0f, y1 = 0.0f, x2 = 0.0f, y2 = 0.0f;
    float conf = 0.0f;
    int cls = -1;
    int model_index = -1;
};

// --- Decode results ---
struct DetectionResult {
    std::vector<Detection> detections;
};

struct SegmentationResult : DetectionResult {
    AVBufferRef* gpu_mask_buf = nullptr;
    int mask_proto_w = 0, mask_proto_h = 0;
    int num_masks = 0;
    std::vector<float> cpu_masks;
    int cpu_mask_w = 0, cpu_mask_h = 0;
};

struct DecodeParams {
    int model_index;
    float conf_thresh;
    OutputBoxFormat box_format;
    const std::vector<int>& class_index_remap;
};

// --- Utility functions ---
inline float halfToFloat(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1Fu;
    uint32_t frac = h & 0x03FFu;
    uint32_t out = 0;
    if (exp == 0) {
        if (frac == 0) {
            out = sign;
        } else {
            exp = 1;
            while ((frac & 0x0400u) == 0) { frac <<= 1; --exp; }
            frac &= 0x03FFu;
            out = sign | ((exp + (127 - 15)) << 23) | (frac << 13);
        }
    } else if (exp == 0x1Fu) {
        out = sign | 0x7F800000u | (frac << 13);
    } else {
        out = sign | ((exp + (127 - 15)) << 23) | (frac << 13);
    }
    float f;
    memcpy(&f, &out, sizeof(float));
    return f;
}

inline size_t elementSize(nvinfer1::DataType dt) {
    switch (dt) {
        case nvinfer1::DataType::kFLOAT: return 4;
        case nvinfer1::DataType::kHALF: return 2;
        case nvinfer1::DataType::kINT8: return 1;
        case nvinfer1::DataType::kINT32: return 4;
        case nvinfer1::DataType::kBOOL: return 1;
        default: return 0;
    }
}

inline size_t volume(const nvinfer1::Dims& d) {
    size_t v = 1;
    for (int i = 0; i < d.nbDims; ++i) {
        if (d.d[i] <= 0) return 0;
        v *= (size_t)d.d[i];
    }
    return v;
}

// Forward declarations for decoders
class DetectionDecoder;
class SegmentationDecoder;

// --- Output tensor info (supports multiple output tensors per model) ---
struct OutputTensor {
    std::string name;
    nvinfer1::Dims dims{};
    nvinfer1::DataType dtype = nvinfer1::DataType::kFLOAT;
    size_t tensor_index = 0;  // index into tensor_ptrs/tensor_bytes
    std::vector<float> host_output;
    std::vector<uint16_t> host_output_half;
};

// --- ModelRunner ---
struct ModelRunner {
    std::string engine_path;
    std::string engine_name;

    // TRT objects
    nvinfer1::IRuntime* trt_runtime = nullptr;
    nvinfer1::ICudaEngine* trt_engine = nullptr;
    nvinfer1::IExecutionContext* trt_ctx = nullptr;

    // IO tensor management
    std::vector<std::string> io_tensor_names;
    std::vector<size_t> tensor_bytes;
    std::vector<CUdeviceptr> tensor_ptrs;
    std::unordered_map<std::string, size_t> tensor_index;

    // Input
    std::string input_tensor_name;
    nvinfer1::Dims input_dims{};
    nvinfer1::DataType input_dtype = nvinfer1::DataType::kFLOAT;
    int input_w = 0, input_h = 0;

    // Outputs (vector: detection has 1, segmentation has 2)
    std::vector<OutputTensor> outputs;

    // Preprocess
    CUfunction preprocess_kernel = nullptr;
    CUstream stream = nullptr;

    // Config
    OutputBoxFormat output_box_format = OutputBoxFormat::EndToEndXYXY;
    TaskType task_type = TaskType::Detection;
    std::vector<std::string> class_names;
    std::vector<int> class_index_remap;

    // Decoder (only one is non-null)
    std::unique_ptr<DetectionDecoder> det_decoder;
    std::unique_ptr<SegmentationDecoder> seg_decoder;
};

// --- CudaInferYoloBase ---
class CudaInferYoloBase {
protected:
    AVCUDADeviceContext* cuda_dev_ctx_ = nullptr;
    CUcontext cu_ctx_ = nullptr;
    TRTLogger trt_logger_;
    std::vector<ModelRunner> models_;
    int input_w_ = 0, input_h_ = 0;
    nvinfer1::DataType input_dtype_ = nvinfer1::DataType::kFLOAT;
    bool input_bgr_order_ = false;
    CUmodule preprocess_module_ = nullptr;
    bool initialized_ = false;

    // Cached metadata JSON fragment for static model info
    std::string cached_models_json_;

    bool initCudaContextFromFrame(const av::VideoFrame& frm);
    bool loadPreprocessModule();
    bool parseEngine(ModelRunner& model);
    bool allocateBindings(ModelRunner& model);
    bool ensureCompatibleInput(const ModelRunner& model, size_t model_index);
    bool configureRunnerPreprocess(ModelRunner& model);
    bool ensureInitialized(const av::VideoFrame& frm);

    bool runPreprocessNV12(const av::VideoFrame& frm, ModelRunner& model);
    bool runInference(ModelRunner& model);
    bool syncModel(ModelRunner& model);

    void cleanupModel(ModelRunner& model);
    void cleanupAll();

    AVPixelFormat hwSwFormat(const av::VideoFrame& frm) const;

public:
    virtual ~CudaInferYoloBase();
};

} // namespace yolo_base
```

- [ ] **Step 2: Verify it compiles**

The header is include-only at this stage. Add a temporary `#include` in the existing `cuda_infer_yolo.cpp` and run:
```bash
make -j8 HAVE_CUDA=1 HAVE_TENSORRT=1 HAVE_NVCC=1 ...
```
Remove the temporary include after verification. (Full build command from CLAUDE.md remote build section.)

- [ ] **Step 3: Commit**

```bash
git add src/nodes/hwaccel/cuda_infer_yolo_base.hpp
git commit -m "feat(yolo): add base header with structs, enums, and utilities"
```

---

## Task 2: Implement base class methods

**Files:**
- Create: `src/nodes/hwaccel/cuda_infer_yolo_base.cpp`

Extract all TRT/CUDA infrastructure from current `cuda_infer_yolo.cpp` into the base implementation. Key changes from current code:
- `allocateBindings` enumerates ALL output tensors (not just first)
- `syncModel` handles half-to-float for all output tensors
- `cleanupModel` syncs stream before destroying
- `ensureInitialized` rolls back on partial failure
- Remove `shortEngineName` (use `std::filesystem::path::filename()`)

- [ ] **Step 1: Create `cuda_infer_yolo_base.cpp`**

Implement all methods declared in the base header. Key method signatures:

```cpp
#include "cuda_infer_yolo_base.hpp"
// PTX blob
#include "../../../objs/src/nodes/hwaccel/yolo_preprocess.ptx.h"

namespace yolo_base {

CudaInferYoloBase::~CudaInferYoloBase() {
    cleanupAll();
}

void CudaInferYoloBase::cleanupAll() {
    if (cu_ctx_) {
        YOLO_CHECK_CU(cuCtxSetCurrent(cu_ctx_));
    }
    for (ModelRunner& model : models_) {
        cleanupModel(model);
    }
    if (preprocess_module_) {
        YOLO_CHECK_CU(cuModuleUnload(preprocess_module_));
        preprocess_module_ = nullptr;
    }
}

void CudaInferYoloBase::cleanupModel(ModelRunner& model) {
    // Sync stream before destroying anything
    if (model.stream) {
        YOLO_CHECK_CU(cuStreamSynchronize(model.stream));
        YOLO_CHECK_CU(cuStreamDestroy(model.stream));
        model.stream = nullptr;
    }
    // Clean up decoder GPU resources
    if (model.seg_decoder) {
        model.seg_decoder->cleanup();
    }
    // Free GPU tensors
    for (CUdeviceptr ptr : model.tensor_ptrs) {
        if (ptr) YOLO_CHECK_CU(cuMemFree(ptr));
    }
    model.tensor_ptrs.clear();
    model.tensor_bytes.clear();
    model.io_tensor_names.clear();
    model.tensor_index.clear();
    model.outputs.clear();
    // Destroy TRT objects
    if (model.trt_ctx) { delete model.trt_ctx; model.trt_ctx = nullptr; }
    if (model.trt_engine) { delete model.trt_engine; model.trt_engine = nullptr; }
    if (model.trt_runtime) { delete model.trt_runtime; model.trt_runtime = nullptr; }
}

bool CudaInferYoloBase::ensureInitialized(const av::VideoFrame& frm) {
    if (initialized_) return true;
    if (!initCudaContextFromFrame(frm)) return false;
    if (!loadPreprocessModule()) return false;
    for (size_t i = 0; i < models_.size(); ++i) {
        ModelRunner& model = models_[i];
        if (!parseEngine(model) || !allocateBindings(model) ||
            !ensureCompatibleInput(model, i) || !configureRunnerPreprocess(model)) {
            // Rollback: clean up models 0..i
            for (size_t j = 0; j <= i; ++j) {
                cleanupModel(models_[j]);
            }
            return false;
        }
    }
    // Cache static model info JSON
    // ... (build cached_models_json_ once)
    initialized_ = true;
    return true;
}

// allocateBindings: enumerate ALL output tensors into model.outputs vector
bool CudaInferYoloBase::allocateBindings(ModelRunner& model) {
    // ... (same structure as current, but collect all outputs not just first)
    // For each non-input tensor: create OutputTensor entry in model.outputs
}

// syncModel: sync stream, convert half->float for ALL output tensors
bool CudaInferYoloBase::syncModel(ModelRunner& model) {
    if (YOLO_CHECK_CU(cuStreamSynchronize(model.stream))) return false;
    for (OutputTensor& ot : model.outputs) {
        if (ot.dtype == nvinfer1::DataType::kHALF) {
            for (size_t i = 0; i < ot.host_output_half.size(); ++i) {
                ot.host_output[i] = halfToFloat(ot.host_output_half[i]);
            }
        }
    }
    return true;
}

// runInference: enqueueV3 + async D2H for all output tensors
bool CudaInferYoloBase::runInference(ModelRunner& model) {
    if (!model.trt_ctx->enqueueV3(reinterpret_cast<cudaStream_t>(model.stream))) {
        logstream << "cuda_infer_yolo: enqueueV3 failed for " << model.engine_name;
        return false;
    }
    for (OutputTensor& ot : model.outputs) {
        size_t idx = ot.tensor_index;
        size_t bytes = model.tensor_bytes[idx];
        void* dst = (ot.dtype == nvinfer1::DataType::kHALF)
            ? (void*)ot.host_output_half.data()
            : (void*)ot.host_output.data();
        if (YOLO_CHECK_CU(cuMemcpyDtoHAsync(dst, model.tensor_ptrs[idx], bytes, model.stream))) {
            return false;
        }
    }
    return true;
}

// ... remaining methods: initCudaContextFromFrame, loadPreprocessModule,
//     parseEngine, configureRunnerPreprocess, runPreprocessNV12, hwSwFormat
//     (these are mostly unchanged from current code, just moved into namespace)

} // namespace yolo_base
```

- [ ] **Step 2: Verify compilation**

This file won't be linked yet (no node references it). Verify it compiles as an object:
```bash
# On remote: add cuda_infer_yolo_base.cpp to NODES_SRC or compile manually
g++ -c src/nodes/hwaccel/cuda_infer_yolo_base.cpp -Isrc -Ideps/... [flags]
```

- [ ] **Step 3: Commit**

```bash
git add src/nodes/hwaccel/cuda_infer_yolo_base.cpp
git commit -m "feat(yolo): implement base class - engine loading, inference, cleanup"
```

---

## Task 3: Create DetectionDecoder

**Files:**
- Create: `src/nodes/hwaccel/yolo_decode_detection.hpp`

This replaces the ~130 lines of duplicated `decodeYoloOutput` with a single unified path.

- [ ] **Step 1: Write the detection decoder**

```cpp
#pragma once
#include "cuda_infer_yolo_base.hpp"

namespace yolo_base {

class DetectionDecoder {
public:
    DetectionResult decode(
        const std::vector<const float*>& host_outputs,
        const std::vector<nvinfer1::Dims>& output_dims,
        const DecodeParams& params
    ) {
        DetectionResult result;
        if (host_outputs.empty() || !host_outputs[0]) return result;

        const float* out = host_outputs[0];
        const nvinfer1::Dims& d = output_dims[0];

        int count = 0, attrs = 0;
        bool attrs_first = false;

        // Determine layout from dims
        if (d.nbDims == 2) {
            int a = d.d[0], b = d.d[1];
            if (a >= 6 && a > b) { attrs = a; count = b; attrs_first = true; }
            else if (b >= 6) { attrs = b; count = a; attrs_first = false; }
            else return result;
        } else if (d.nbDims == 3 && d.d[0] == 1) {
            int d1 = d.d[1], d2 = d.d[2];
            if (d1 >= 6 && d1 >= d2) { attrs = d1; count = d2; attrs_first = true; }
            else if (d2 >= 6) { attrs = d2; count = d1; attrs_first = false; }
            else return result;
        } else {
            return result;
        }

        if (count <= 0 || attrs < 6) return result;

        // Unified accessor: handles both memory layouts
        auto at = [&](int det, int attr) -> float {
            return attrs_first ? out[attr * count + det] : out[det * attrs + attr];
        };

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
                for (int c = 4; c < attrs; ++c) {
                    float s = at(i, c);
                    if (s > best) { best = s; best_cls = c - 4; }
                }
                det.x1 = cx - w * 0.5f; det.y1 = cy - h * 0.5f;
                det.x2 = cx + w * 0.5f; det.y2 = cy + h * 0.5f;
                det.conf = best;
                det.cls = best_cls;
            }

            if (det.conf < params.conf_thresh) continue;

            // Apply class index remapping
            if (det.cls >= 0 && (size_t)det.cls < params.class_index_remap.size()) {
                det.cls = params.class_index_remap[(size_t)det.cls];
            }

            result.detections.push_back(det);
        }

        return result;
    }
};

} // namespace yolo_base
```

- [ ] **Step 2: Commit**

```bash
git add src/nodes/hwaccel/yolo_decode_detection.hpp
git commit -m "feat(yolo): add DetectionDecoder with unified accessor decode"
```

---

## Task 4: Rewrite the node (detection-only, no segmentation yet)

**Files:**
- Rewrite: `src/nodes/hwaccel/cuda_infer_yolo.cpp`

This is the critical step. Rewrite the node to use the base class and detection decoder. Must produce identical detection output to the current node for the `yolo_bbox.avplumber` example.

- [ ] **Step 1: Rewrite `cuda_infer_yolo.cpp`**

The node now:
- Inherits from `CudaInferYoloBase` and `NodeSingleInput<av::VideoFrame>`
- Manually manages two `EdgeSink<av::VideoFrame>` members: `sink_` (required dst) and `sink_seg_` (optional dst_seg, nullptr if no seg models)
- `create()` parses params including `task_type` per model, creates decoder instances
- `process()` calls base methods then dispatches to decoders

Key structure:
```cpp
#include "cuda_infer_yolo_base.hpp"
#include "yolo_decode_detection.hpp"

using namespace yolo_base;

class CudaInferYolo : public NodeSingleInput<av::VideoFrame>, public CudaInferYoloBase {
protected:
    std::unique_ptr<EdgeSink<av::VideoFrame>> sink_;
    std::unique_ptr<EdgeSink<av::VideoFrame>> sink_seg_;  // optional

    std::string metadata_key_detection_ = "yolo_detections";
    std::string metadata_key_segmentation_ = "yolo_segmentation";
    bool debug_log_metadata_ = false;
    int debug_log_every_n_ = 30;
    int infer_every_n_ = 1;
    float conf_thresh_ = 0.25f;
    int max_det_ = 300;
    int mask_gpu_every_n_ = 1;
    int mask_cpu_every_n_ = 2;
    int mask_cpu_resolution_ = 120;
    uint64_t frame_counter_ = 0;
    uint64_t infer_counter_ = 0;

public:
    CudaInferYolo(std::unique_ptr<Source<av::VideoFrame>> source,
                  std::unique_ptr<EdgeSink<av::VideoFrame>> sink,
                  std::unique_ptr<EdgeSink<av::VideoFrame>> sink_seg)
        : NodeSingleInput(std::move(source))
        , sink_(std::move(sink))
        , sink_seg_(std::move(sink_seg)) {}

    void process() override {
        av::VideoFrame frm = this->source_->get();
        if (!frm) return;

        ++frame_counter_;
        if (infer_every_n_ > 1 && (frame_counter_ % (uint64_t)infer_every_n_) != 0) {
            sink_->put(frm);
            return;
        }
        // Validate frame format
        if (frm.raw()->format != AV_PIX_FMT_CUDA) { sink_->put(frm); return; }
        if (!ensureInitialized(frm)) return;
        if (frm.width() != input_w_ || frm.height() != input_h_) return;
        if (hwSwFormat(frm) != AV_PIX_FMT_NV12) return;
        if (YOLO_CHECK_CU(cuCtxSetCurrent(cu_ctx_))) return;

        ++infer_counter_;

        // Run inference on all models
        for (ModelRunner& model : models_) {
            if (!runPreprocessNV12(frm, model)) return;
            if (!runInference(model)) return;
        }

        // Collect detections
        std::vector<Detection> all_dets;
        for (size_t mi = 0; mi < models_.size(); ++mi) {
            ModelRunner& model = models_[mi];
            if (!syncModel(model)) return;

            // Build host_outputs and output_dims from model.outputs
            std::vector<const float*> host_outputs;
            std::vector<nvinfer1::Dims> output_dims;
            for (const OutputTensor& ot : model.outputs) {
                host_outputs.push_back(ot.host_output.data());
                output_dims.push_back(ot.dims);
            }

            DecodeParams dp{(int)mi, conf_thresh_, model.output_box_format, model.class_index_remap};

            if (model.task_type == TaskType::Detection && model.det_decoder) {
                DetectionResult dr = model.det_decoder->decode(host_outputs, output_dims, dp);
                all_dets.insert(all_dets.end(), dr.detections.begin(), dr.detections.end());
            }
            // Segmentation handled in Task 7
        }

        // Finalize: sort by confidence, truncate to max_det
        std::sort(all_dets.begin(), all_dets.end(),
            [](const Detection& a, const Detection& b) { return a.conf > b.conf; });
        if (max_det_ > 0 && (int)all_dets.size() > max_det_) {
            all_dets.resize((size_t)max_det_);
        }

        // Build and attach detection metadata
        std::string md = buildDetectionMetadata(all_dets);
        av_dict_set(&frm.raw()->metadata, metadata_key_detection_.c_str(), md.c_str(), 0);

        if (debug_log_metadata_ && debug_log_every_n_ > 0 &&
            (frame_counter_ % (uint64_t)debug_log_every_n_) == 0) {
            logstream << "cuda_infer_yolo: " << md;
        }

        sink_->put(frm);
    }

    std::string buildDetectionMetadata(const std::vector<Detection>& dets) const {
        // Use cached_models_json_ for static part
        Parameters j;
        j["coord_space"] = "model";
        j["model_width"] = input_w_;
        j["model_height"] = input_h_;
        // ... models array from cache, detections array from dets
        // Add label lookup from model.class_names
    }

    static std::shared_ptr<CudaInferYolo> create(NodeCreationInfo& nci) {
        EdgeManager& edges = nci.edges;
        const Parameters& params = nci.params;

        auto src = edges.find<av::VideoFrame>(params["src"]);
        auto dst = edges.find<av::VideoFrame>(params["dst"]);
        std::unique_ptr<EdgeSink<av::VideoFrame>> sink_seg;
        if (params.count("dst_seg")) {
            auto dst_seg = edges.find<av::VideoFrame>(params["dst_seg"]);
            sink_seg = dst_seg->makeSink();
        }

        auto r = std::make_shared<CudaInferYolo>(
            src->makeSource(), dst->makeSink(), std::move(sink_seg));

        // Parse global params
        if (params.count("conf_thresh")) r->conf_thresh_ = params["conf_thresh"];
        if (params.count("max_det")) r->max_det_ = params["max_det"];
        if (params.count("infer_every_n")) r->infer_every_n_ = params["infer_every_n"];
        if (params.count("metadata_key_detection")) r->metadata_key_detection_ = params["metadata_key_detection"].get<std::string>();
        if (params.count("metadata_key_segmentation")) r->metadata_key_segmentation_ = params["metadata_key_segmentation"].get<std::string>();
        if (params.count("debug_log_metadata")) r->debug_log_metadata_ = params["debug_log_metadata"];
        if (params.count("debug_log_every_n")) r->debug_log_every_n_ = params["debug_log_every_n"];
        if (params.count("mask_gpu_every_n")) r->mask_gpu_every_n_ = params["mask_gpu_every_n"];
        if (params.count("mask_cpu_every_n")) r->mask_cpu_every_n_ = params["mask_cpu_every_n"];
        if (params.count("mask_cpu_resolution")) r->mask_cpu_resolution_ = params["mask_cpu_resolution"];
        if (params.count("input_format")) {
            std::string ifmt = params["input_format"].get<std::string>();
            r->input_bgr_order_ = (ifmt == "BGR" || ifmt == "bgr");
        }

        // Parse models
        auto parseFmt = [](const std::string& fmt) -> OutputBoxFormat {
            if (fmt == "end2end_xyxy") return OutputBoxFormat::EndToEndXYXY;
            if (fmt == "raw_cxcywh") return OutputBoxFormat::RawCXCYWH;
            throw Error("cuda_infer_yolo: output_box_format must be 'end2end_xyxy' or 'raw_cxcywh', got: " + fmt);
        };
        auto parseTaskType = [](const std::string& tt) -> TaskType {
            if (tt == "detection") return TaskType::Detection;
            if (tt == "segmentation") return TaskType::Segmentation;
            if (tt == "pose") return TaskType::Pose;
            throw Error("cuda_infer_yolo: task_type must be 'detection', 'segmentation', or 'pose', got: " + tt);
        };

        for (const auto& mp : params["models"]) {
            ModelRunner model;
            model.engine_path = mp["engine"].get<std::string>();
            model.engine_name = std::filesystem::path(model.engine_path).filename().string();
            model.output_box_format = parseFmt(mp["output_box_format"].get<std::string>());
            model.task_type = parseTaskType(mp["task_type"].get<std::string>());

            if (mp.count("class_names")) {
                for (const auto& n : mp["class_names"]) model.class_names.push_back(n.get<std::string>());
            }
            if (mp.count("class_index_remap")) {
                for (const auto& c : mp["class_index_remap"]) model.class_index_remap.push_back(c.get<int>());
            }

            // Create decoder
            if (model.task_type == TaskType::Detection) {
                model.det_decoder = std::make_unique<DetectionDecoder>();
            }
            // Segmentation decoder created in Task 7

            r->models_.push_back(std::move(model));
        }

        return r;
    }
};

DECLNODE(cuda_infer_yolo, CudaInferYolo)
```

- [ ] **Step 2: Update example files**

Update `examples/yolo/yolo_bbox.avplumber`:
- Add `"task_type": "detection"` to each model in the models array
- Replace `"metadata_key_out": "yolo_detections_v1"` with `"metadata_key_detection": "yolo_detections"`
- Remove `"iou_thresh": 0.45`
- Update downstream nodes that reference `"yolo_detections_v1"` to `"yolo_detections"`

Apply same changes to:
- `examples/rtmp_input_hw_dec_cuda_yolo.avplumber`
- `examples/yolo/rtmp_input_hw_dec_cuda_yolo.avplumber`
- `library_examples/obs-avplumber-source/examples/rtmp_input_hw_dec_cuda_yolo.txt`

Also update `basketball_analysis` metadata_key_in param from `"yolo_detections_v1"` to `"yolo_detections"` in the example files.

- [ ] **Step 3: Build and verify on remote**

```bash
# Rsync all changed files to remote
rsync -avz --relative -e "ssh -i /home/jp/work-misc-stuff/awsdev.pem" \
  /home/jp/git/avplumber/./{src/nodes/hwaccel/cuda_infer_yolo_base.hpp,src/nodes/hwaccel/cuda_infer_yolo_base.cpp,src/nodes/hwaccel/cuda_infer_yolo.cpp,src/nodes/hwaccel/yolo_decode_detection.hpp,examples/yolo/yolo_bbox.avplumber} \
  fedora@172.17.36.132:/home/fedora/avplumber/

# SSH to remote, clean build
ssh -i /home/jp/work-misc-stuff/awsdev.pem fedora@172.17.36.132
cd /home/fedora/avplumber
make clean && make -j8 HAVE_CUDA=1 HAVE_TENSORRT=1 HAVE_NVCC=1 \
  NVCC=/usr/local/cuda-13.0/bin/nvcc TENSORRT_ROOT=/opt/tensorrt \
  PKG_CONFIG_PATH=/apps/ffmpeg/lib/pkgconfig \
  CXXFLAGS+=' -I/usr/local/cuda-13.0/include -I/usr/local/cuda-13.0/targets/x86_64-linux/include' \
  LFLAGS+=' -L/apps/ffmpeg/lib -Wl,-rpath,/apps/ffmpeg/lib -L/usr/local/cuda-13.0/targets/x86_64-linux/lib -Wl,-rpath,/usr/local/cuda-13.0/targets/x86_64-linux/lib'
```

- [ ] **Step 4: Run the example pipeline on remote**

```bash
./avplumber -p 20200 -s examples/yolo/yolo_bbox.avplumber
```

Verify:
- No crashes or errors
- Detection metadata appears in debug log output
- Output .ts file contains video with bounding boxes
- All 3 models (ball, player-ball, hoop-ball) produce detections

- [ ] **Step 5: Commit**

```bash
git add src/nodes/hwaccel/cuda_infer_yolo.cpp \
        examples/yolo/yolo_bbox.avplumber \
        examples/rtmp_input_hw_dec_cuda_yolo.avplumber \
        examples/yolo/rtmp_input_hw_dec_cuda_yolo.avplumber \
        library_examples/obs-avplumber-source/examples/rtmp_input_hw_dec_cuda_yolo.txt
git commit -m "feat(yolo): rewrite node using base class + detection decoder

Replaces monolithic 907-line file with modular base+decoder architecture.
Detection behavior identical to previous version.
Updates example files with explicit task_type and new metadata key names."
```

---

## Task 5: Update join_metadata to copy side data

**Files:**
- Modify: `src/nodes/join_metadata.cpp` (line 39, inside the `ts_primary == ts_auxiliary` branch)

- [ ] **Step 1: Add side data copying**

After the existing `av_dict_copy` line (line 39), add side data copy logic:

```cpp
// Copy side data from auxiliary to primary (for segmentation masks, etc.)
if (auxiliary->raw()->nb_side_data > 0) {
    for (int i = 0; i < auxiliary->raw()->nb_side_data; ++i) {
        AVFrameSideData* sd_src = auxiliary->raw()->side_data[i];
        if (!sd_src || !sd_src->buf) continue;
        // Only copy if not already present on primary
        if (av_frame_get_side_data(primary->raw(), sd_src->type)) continue;
        AVBufferRef* ref = av_buffer_ref(sd_src->buf);
        if (!ref) continue;
        AVFrameSideData* sd_dst = av_frame_new_side_data_from_buf(primary->raw(), sd_src->type, ref);
        if (!sd_dst) {
            av_buffer_unref(&ref);
            logstream << "join_metadata: failed to copy side data type " << (int)sd_src->type;
        }
    }
}
```

- [ ] **Step 2: Verify build**

Build and run the same yolo_bbox.avplumber example. Side data copying should be a no-op for detection-only pipelines (no side data present).

- [ ] **Step 3: Commit**

```bash
git add src/nodes/join_metadata.cpp
git commit -m "feat(join_metadata): copy AVFrame side data from auxiliary to primary"
```

---

## Task 6: Add yolo_mask_assemble.cu and Makefile rule

**Files:**
- Create: `src/nodes/hwaccel/yolo_mask_assemble.cu`
- Modify: `Makefile` (add PTX build rule following existing pattern)

- [ ] **Step 1: Write the mask assembly CUDA kernel**

```cuda
// yolo_mask_assemble.cu
// For each detection: dot-product of 32 coefficients against 32 prototype
// channels at each spatial position, followed by sigmoid activation.
//
// Grid: (proto_w + 31) / 32, (proto_h + 7) / 8, num_detections
// Block: 32, 8, 1
//
// Inputs:
//   prototypes: [32, proto_h, proto_w] float
//   coefficients: [num_detections, 32] float
//   proto_h, proto_w: prototype spatial dims
//   num_coefficients: 32 (or configurable)
//
// Output:
//   masks: [num_detections, proto_h, proto_w] float (after sigmoid)

extern "C" __global__ void kMaskAssemble(
    const float* __restrict__ prototypes,  // [32, proto_h, proto_w]
    const float* __restrict__ coefficients, // [num_dets, 32]
    float* __restrict__ masks,              // [num_dets, proto_h, proto_w]
    int proto_h,
    int proto_w,
    int num_coefficients
) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int det = blockIdx.z;

    if (x >= proto_w || y >= proto_h) return;

    const int spatial_idx = y * proto_w + x;
    const int spatial_size = proto_h * proto_w;

    float sum = 0.0f;
    for (int c = 0; c < num_coefficients; ++c) {
        sum += coefficients[det * num_coefficients + c] * prototypes[c * spatial_size + spatial_idx];
    }

    // Sigmoid
    sum = 1.0f / (1.0f + expf(-sum));

    masks[det * spatial_size + spatial_idx] = sum;
}

// Bilinear downsample kernel for CPU mask path
extern "C" __global__ void kMaskDownsample(
    const float* __restrict__ src,   // [proto_h, proto_w]
    float* __restrict__ dst,          // [dst_h, dst_w]
    int src_h, int src_w,
    int dst_h, int dst_w
) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= dst_w || y >= dst_h) return;

    float src_x = (float)x * (float)src_w / (float)dst_w;
    float src_y = (float)y * (float)src_h / (float)dst_h;

    int x0 = (int)src_x;
    int y0 = (int)src_y;
    int x1 = min(x0 + 1, src_w - 1);
    int y1 = min(y0 + 1, src_h - 1);

    float fx = src_x - (float)x0;
    float fy = src_y - (float)y0;

    float v = src[y0 * src_w + x0] * (1.0f - fx) * (1.0f - fy)
            + src[y0 * src_w + x1] * fx * (1.0f - fy)
            + src[y1 * src_w + x0] * (1.0f - fx) * fy
            + src[y1 * src_w + x1] * fx * fy;

    dst[y * dst_w + x] = v;
}
```

- [ ] **Step 2: Add Makefile PTX rule**

Add after the existing YOLO_PREPROCESS_PTX block (around line 107):

```makefile
YOLO_MASK_ASSEMBLE_KERNEL = $(SRCDIR)/nodes/hwaccel/yolo_mask_assemble.cu
YOLO_MASK_ASSEMBLE_PTX = objs/$(SRCDIR)/nodes/hwaccel/yolo_mask_assemble.ptx
YOLO_MASK_ASSEMBLE_PTX_H = objs/$(SRCDIR)/nodes/hwaccel/yolo_mask_assemble.ptx.h
```

And add the build rules (following the pattern at line 284-295):

```makefile
$(YOLO_MASK_ASSEMBLE_PTX): $(YOLO_MASK_ASSEMBLE_KERNEL)
	@mkdir -p $(dir $@)
	$(NVCC) -ptx -o $@ $<

$(YOLO_MASK_ASSEMBLE_PTX_H): $(YOLO_MASK_ASSEMBLE_PTX)
	@mkdir -p $(dir $@)
	@if [ ! -s $< ]; then echo "Error: PTX file $< is empty or missing" >&2; exit 1; fi
	xxd -i $< | sed -E 's/unsigned int objs_src_nodes_hwaccel_yolo_mask_assemble_ptx_len/const unsigned int avpl_yolo_mask_assemble_ptx_len/; s/unsigned char objs_src_nodes_hwaccel_yolo_mask_assemble_ptx/const char avpl_yolo_mask_assemble_ptx/' > $@
	@if [ ! -s $@ ]; then echo "Error: Generated header $@ is empty. Check PTX file: $<" >&2; exit 1; fi

objs/src/nodes/hwaccel/cuda_infer_yolo.o: $(YOLO_MASK_ASSEMBLE_PTX_H)
```

- [ ] **Step 3: Verify PTX compilation on remote**

```bash
# Rsync .cu and Makefile
# On remote:
make clean && make -j8 [full flags]
# Verify objs/src/nodes/hwaccel/yolo_mask_assemble.ptx.h is generated
ls -la objs/src/nodes/hwaccel/yolo_mask_assemble.ptx.h
```

- [ ] **Step 4: Commit**

```bash
git add src/nodes/hwaccel/yolo_mask_assemble.cu Makefile
git commit -m "feat(yolo): add mask assembly CUDA kernel with downsample"
```

---

## Task 7: Create SegmentationDecoder

**Files:**
- Create: `src/nodes/hwaccel/yolo_decode_segmentation.hpp`

- [ ] **Step 1: Write the segmentation decoder**

```cpp
#pragma once
#include "cuda_infer_yolo_base.hpp"
#include "../../../objs/src/nodes/hwaccel/yolo_mask_assemble.ptx.h"

namespace yolo_base {

class SegmentationDecoder {
private:
    CUmodule mask_module_ = nullptr;
    CUfunction mask_assemble_kernel_ = nullptr;
    CUfunction mask_downsample_kernel_ = nullptr;
    CUdeviceptr gpu_mask_scratch_ = 0;  // [max_det, proto_h, proto_w]
    CUdeviceptr gpu_coeff_buf_ = 0;     // [max_det, 32]
    CUdeviceptr gpu_downsample_buf_ = 0; // [cpu_res, cpu_res] per detection
    size_t proto_h_ = 0, proto_w_ = 0;
    int max_det_ = 0;
    CUcontext cu_ctx_ = nullptr;

public:
    bool init(const ModelRunner& model, CUcontext cu_ctx, int max_det) {
        cu_ctx_ = cu_ctx;
        max_det_ = max_det;

        // Load PTX module
        const std::string ptx_str(avpl_yolo_mask_assemble_ptx,
            avpl_yolo_mask_assemble_ptx + avpl_yolo_mask_assemble_ptx_len);
        if (YOLO_CHECK_CU(cuModuleLoadDataEx(&mask_module_, ptx_str.c_str(), 0, nullptr, nullptr)))
            return false;
        if (YOLO_CHECK_CU(cuModuleGetFunction(&mask_assemble_kernel_, mask_module_, "kMaskAssemble")))
            return false;
        if (YOLO_CHECK_CU(cuModuleGetFunction(&mask_downsample_kernel_, mask_module_, "kMaskDownsample")))
            return false;

        // Get prototype dims from second output tensor
        if (model.outputs.size() < 2) return false;
        const nvinfer1::Dims& proto_dims = model.outputs[1].dims;
        // Expected: [1, 32, proto_h, proto_w] or [32, proto_h, proto_w]
        if (proto_dims.nbDims == 4) {
            proto_h_ = proto_dims.d[2]; proto_w_ = proto_dims.d[3];
        } else if (proto_dims.nbDims == 3) {
            proto_h_ = proto_dims.d[1]; proto_w_ = proto_dims.d[2];
        } else return false;

        // Allocate scratch buffers
        size_t mask_bytes = max_det_ * proto_h_ * proto_w_ * sizeof(float);
        size_t coeff_bytes = max_det_ * 32 * sizeof(float);
        if (YOLO_CHECK_CU(cuMemAlloc(&gpu_mask_scratch_, mask_bytes))) return false;
        if (YOLO_CHECK_CU(cuMemAlloc(&gpu_coeff_buf_, coeff_bytes))) return false;

        return true;
    }

    SegmentationResult decode(
        const std::vector<const float*>& host_outputs,
        const std::vector<nvinfer1::Dims>& output_dims,
        const DecodeParams& params,
        bool emit_gpu_mask,
        bool emit_cpu_mask,
        int cpu_mask_resolution,
        CUstream stream
    ) {
        SegmentationResult result;
        if (host_outputs.size() < 2) return result;

        // 1. Decode detections + extract 32 coefficients from output0
        //    output0 layout: [1, 4+num_classes+32, N] or similar
        //    Reuse detection accessor logic, plus extract last 32 attrs as coefficients
        // ... (decode detections similar to DetectionDecoder)
        // ... (extract coefficients into a host buffer)

        // 2. If neither GPU nor CPU mask needed, return detections only
        if (!emit_gpu_mask && !emit_cpu_mask) return result;
        if (result.detections.empty()) return result;

        int num_dets = std::min((int)result.detections.size(), max_det_);
        result.num_masks = num_dets;
        result.mask_proto_w = (int)proto_w_;
        result.mask_proto_h = (int)proto_h_;

        // 3. Upload coefficients to GPU
        // ... cuMemcpyHtoDAsync(gpu_coeff_buf_, ...)

        // 4. Launch mask assembly kernel
        int proto_h = (int)proto_h_, proto_w = (int)proto_w_, num_coeff = 32;
        // prototypes are already on GPU (output1 tensor)
        // ... cuLaunchKernel(mask_assemble_kernel_, ...)

        // 5. GPU path: wrap in AVBufferRef
        if (emit_gpu_mask) {
            size_t mask_bytes = num_dets * proto_h_ * proto_w_ * sizeof(float);
            CUdeviceptr gpu_out = 0;
            YOLO_CHECK_CU(cuMemAlloc(&gpu_out, mask_bytes));
            YOLO_CHECK_CU(cuMemcpyDtoDAsync(gpu_out, gpu_mask_scratch_, mask_bytes, stream));
            // Wrap in AVBufferRef with custom free callback
            result.gpu_mask_buf = av_buffer_create(
                (uint8_t*)(uintptr_t)gpu_out, mask_bytes,
                [](void*, uint8_t* data) {
                    cuMemFree((CUdeviceptr)(uintptr_t)data);
                }, nullptr, 0);
        }

        // 6. CPU path: downsample + D2H
        if (emit_cpu_mask) {
            result.cpu_mask_w = cpu_mask_resolution;
            result.cpu_mask_h = cpu_mask_resolution;
            size_t ds_bytes = cpu_mask_resolution * cpu_mask_resolution * sizeof(float);
            if (!gpu_downsample_buf_) {
                YOLO_CHECK_CU(cuMemAlloc(&gpu_downsample_buf_, ds_bytes));
            }
            result.cpu_masks.resize(num_dets * cpu_mask_resolution * cpu_mask_resolution);
            for (int d = 0; d < num_dets; ++d) {
                // Launch downsample kernel per detection
                // ... cuLaunchKernel(mask_downsample_kernel_, ...)
                // D2H copy
                // ... cuMemcpyDtoHAsync into result.cpu_masks + offset
            }
        }

        return result;
    }

    void cleanup() {
        if (cu_ctx_) YOLO_CHECK_CU(cuCtxSetCurrent(cu_ctx_));
        if (gpu_mask_scratch_) { YOLO_CHECK_CU(cuMemFree(gpu_mask_scratch_)); gpu_mask_scratch_ = 0; }
        if (gpu_coeff_buf_) { YOLO_CHECK_CU(cuMemFree(gpu_coeff_buf_)); gpu_coeff_buf_ = 0; }
        if (gpu_downsample_buf_) { YOLO_CHECK_CU(cuMemFree(gpu_downsample_buf_)); gpu_downsample_buf_ = 0; }
        if (mask_module_) { YOLO_CHECK_CU(cuModuleUnload(mask_module_)); mask_module_ = nullptr; }
    }
};

} // namespace yolo_base
```

- [ ] **Step 2: Commit**

```bash
git add src/nodes/hwaccel/yolo_decode_segmentation.hpp
git commit -m "feat(yolo): add SegmentationDecoder with GPU/CPU mask paths"
```

---

## Task 8: Integrate segmentation into the node

**Files:**
- Modify: `src/nodes/hwaccel/cuda_infer_yolo.cpp`

Wire up segmentation decoder creation in `create()` and segmentation output in `process()`.

- [ ] **Step 1: Add segmentation decoder include and creation**

In `cuda_infer_yolo.cpp`, add:
```cpp
#include "yolo_decode_segmentation.hpp"
```

In `create()`, after the model parsing loop where detection decoder is created, add:
```cpp
if (model.task_type == TaskType::Segmentation) {
    model.seg_decoder = std::make_unique<SegmentationDecoder>();
}
```

- [ ] **Step 2: Add segmentation processing to process()**

In the model dispatch loop:
```cpp
if (model.task_type == TaskType::Segmentation && model.seg_decoder) {
    bool emit_gpu = (infer_counter_ % (uint64_t)mask_gpu_every_n_ == 0);
    bool emit_cpu = (infer_counter_ % (uint64_t)mask_cpu_every_n_ == 0);
    SegmentationResult sr = model.seg_decoder->decode(
        host_outputs, output_dims, dp,
        emit_gpu, emit_cpu, mask_cpu_resolution_, model.stream);

    // Add seg detections to all_dets
    all_dets.insert(all_dets.end(), sr.detections.begin(), sr.detections.end());

    // CPU path: attach side data
    if (emit_cpu && !sr.cpu_masks.empty()) {
        size_t header_size = 16;
        size_t mask_data_size = sr.cpu_masks.size() * sizeof(float);
        size_t total_size = header_size + mask_data_size;
        AVBufferRef* buf = av_buffer_alloc(total_size);
        if (buf) {
            uint32_t* header = (uint32_t*)buf->data;
            header[0] = (uint32_t)sr.num_masks;
            header[1] = (uint32_t)sr.cpu_mask_w;
            header[2] = (uint32_t)sr.cpu_mask_h;
            header[3] = 0; // reserved
            memcpy(buf->data + header_size, sr.cpu_masks.data(), mask_data_size);
            av_frame_new_side_data_from_buf(frm.raw(), AV_FRAME_DATA_YOLO_SEG_MASKS, buf);
        }
    }

    // GPU path: output mask AVFrame
    if (emit_gpu && sr.gpu_mask_buf && sink_seg_) {
        av::VideoFrame mask_frame;
        // Create minimal frame with PTS and the mask buffer
        mask_frame.raw()->pts = frm.raw()->pts;
        // Attach gpu_mask_buf as the frame data
        // ... (set frame dimensions to proto_w x proto_h, attach buffer)
        sink_seg_->put(mask_frame);
    } else if (sr.gpu_mask_buf) {
        // Not emitting: release the buffer
        av_buffer_unref(&sr.gpu_mask_buf);
    }
}
```

- [ ] **Step 3: Initialize segmentation decoder in ensureInitialized**

After `ensureInitialized` succeeds in `process()` (or in a post-init hook), initialize segmentation decoders:
```cpp
for (ModelRunner& model : models_) {
    if (model.task_type == TaskType::Segmentation && model.seg_decoder) {
        if (!model.seg_decoder->init(model, cu_ctx_, max_det_)) {
            logstream << "cuda_infer_yolo: failed to init seg decoder for " << model.engine_name;
            return;
        }
    }
}
```

- [ ] **Step 4: Build and verify on remote**

```bash
# Full rsync + clean build + run yolo_bbox.avplumber
# Verify detection-only pipeline still works (no seg models in this example)
```

- [ ] **Step 5: Commit**

```bash
git add src/nodes/hwaccel/cuda_infer_yolo.cpp
git commit -m "feat(yolo): integrate segmentation decoder into node process loop"
```

---

## Task 9: Update documentation

**Files:**
- Modify: `node_documentation/cuda_infer_yolo.md`

- [ ] **Step 1: Update node documentation**

Update the documentation to reflect:
- New `task_type` parameter per model (required: `"detection"` or `"segmentation"`)
- New params: `metadata_key_detection`, `metadata_key_segmentation`, `dst_seg`, `mask_gpu_every_n`, `mask_cpu_every_n`, `mask_cpu_resolution`
- Removed: `iou_thresh`, `metadata_key_out`
- Document the side data binary format for CPU masks
- Add example config with a segmentation model

- [ ] **Step 2: Commit**

```bash
git add node_documentation/cuda_infer_yolo.md
git commit -m "docs(yolo): update cuda_infer_yolo documentation for multi-task support"
```

---

## Execution Order & Dependencies

```
Task 1 (base header)
  └─> Task 2 (base implementation)
       └─> Task 3 (detection decoder)
            └─> Task 4 (rewrite node, detection-only) ← CRITICAL CHECKPOINT: verify pipeline works
                 ├─> Task 5 (join_metadata side data)
                 └─> Task 6 (mask assembly kernel + Makefile)
                      └─> Task 7 (segmentation decoder)
                           └─> Task 8 (integrate segmentation into node) ← CHECKPOINT: verify seg works
                                └─> Task 9 (documentation)
```

**Checkpoints:**
- After Task 4: Full pipeline must work with detection-only models. Run `yolo_bbox.avplumber` on remote.
- After Task 8: Segmentation integration complete. Test with a seg model if available, otherwise verify detection-only still works and seg code paths compile cleanly.
