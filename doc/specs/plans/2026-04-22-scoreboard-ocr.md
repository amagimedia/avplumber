# Scoreboard OCR Node Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `scoreboard_ocr` node that reads NBA scoreboard text (team names, scores, period, time, shot clock) from YOLO-detected bounding boxes using TensorRT-accelerated PP-OCR recognition.

**Architecture:** The node is a `NodeSISO<VideoFrame, VideoFrame>` that receives 1080p GPU NV12 frames with YOLO detection metadata already merged via `join_metadata`. It filters detections for scoreboard classes (Period, Shot Clock, Team Name, Team Points, Time Remaining), crops those regions from the GPU frame via a CUDA kernel that also resizes + pads to the fixed 48×320 NCHW tensor expected by the PP-OCRv4 English recognition TensorRT engine, runs inference, CTC-decodes the output, and writes structured `scoreboard` JSON metadata onto the frame. An `ocr_every_n` parameter (default 25) skips frames and carries forward the last result.

**Tech Stack:** C++17, CUDA (driver API via cuda_loader), TensorRT (via existing `CudaInferTrtBase`/`ModelRunner`), PP-OCRv4 English recognition model (`en_PP-OCRv4_rec`), `ppocr_keys_v1.txt` character dictionary from `deps/RapidOcrOnnx/models/`.

---

## File Structure

| File | Action | Responsibility |
|------|--------|----------------|
| `src/nodes/neural_net/preprocess/nv12_crop_resize_pad.cu` | Create | CUDA kernel: crop sub-rect from NV12 GPU frame, bilinear resize to height=48, pad to width=320, convert to RGB NCHW float32, normalize to [-1,1] |
| `src/nodes/neural_net/sport_specific/scoreboard_ocr.cpp` | Create | The node: detection filtering, coord scaling, kernel dispatch, TRT inference, CTC decode, metadata output |
| `Makefile` | Modify | Add PTX kernel build rule, add node source under `NEURAL_NET_SPECIFIC` |
| `doc/models.md` | Modify | Add PP-OCRv4 English rec model entry with download + trtexec commands |
| `examples/yolo/yolo_infer_all_players_tracker_live_teams_siglip_nba.avplumber` | Already modified | `scoreboard_ocr` node is already wired in |

---

### Task 1: CUDA Kernel — NV12 Crop + Resize + Pad to NCHW

**Files:**
- Create: `src/nodes/neural_net/preprocess/nv12_crop_resize_pad.cu`

This kernel takes a sub-rectangle from a full-resolution NV12 GPU frame, bilinear-resizes it to height=48 with aspect-preserving width, zero-pads to width=320, converts NV12→RGB, and normalizes to [-1, 1] (PP-OCR normalization: `pixel/127.5 - 1.0`). It writes a single NCHW float32 tensor of shape `[1, 3, 48, 320]`.

The kernel is launched once per crop. The caller sets the crop rect via kernel args — no batch dimension inside the kernel. Multiple crops are serialized on the same CUDA stream (scoreboard has at most 5-7 small boxes, each ~80×40 source pixels → trivial GPU time).

- [ ] **Step 1: Write the CUDA kernel**

Create `src/nodes/neural_net/preprocess/nv12_crop_resize_pad.cu`:

```cuda
#include <stdint.h>
#include <cuda_runtime.h>

__device__ __forceinline__ float clamp_f(float v, float lo, float hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

__device__ __forceinline__ void nv12_to_rgb_norm(
    uint8_t y8, uint8_t u8, uint8_t v8,
    float& r, float& g, float& b)
{
    float y = (float)y8;
    float u = (float)u8 - 128.0f;
    float v = (float)v8 - 128.0f;
    float yf = 1.164384f * (y - 16.0f);
    r = clamp_f((yf + 1.792741f * v) / 127.5f - 1.0f, -1.0f, 1.0f);
    g = clamp_f((yf - 0.213249f * u - 0.532909f * v) / 127.5f - 1.0f, -1.0f, 1.0f);
    b = clamp_f((yf + 2.112402f * u) / 127.5f - 1.0f, -1.0f, 1.0f);
}

// Crop a sub-rect from an NV12 frame, bilinear resize to (dst_h x dst_w_content),
// pad to (dst_h x dst_w_padded) with zeros, output RGB NCHW float32 normalized to [-1,1].
//
// Grid: (dst_w_padded, dst_h) with 1 thread per output pixel.
// Padded region (x >= dst_w_content) writes 0.0f (gray in [-1,1] space).
extern "C" __global__ void kNV12_crop_resize_pad_RGB(
    const uint8_t* __restrict__ y_plane, int y_pitch,
    const uint8_t* __restrict__ uv_plane, int uv_pitch,
    float* __restrict__ out_nchw,
    int src_x1, int src_y1, int src_w, int src_h,
    int dst_h, int dst_w_content, int dst_w_padded)
{
    const int dx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    const int dy = (int)(blockIdx.y * blockDim.y + threadIdx.y);
    if (dx >= dst_w_padded || dy >= dst_h) return;

    const int plane_size = dst_w_padded * dst_h;
    const int idx = dy * dst_w_padded + dx;

    if (dx >= dst_w_content) {
        out_nchw[idx + 0 * plane_size] = 0.0f;
        out_nchw[idx + 1 * plane_size] = 0.0f;
        out_nchw[idx + 2 * plane_size] = 0.0f;
        return;
    }

    // Map output pixel back to source coordinates (bilinear)
    float sx = (float)src_x1 + ((float)dx + 0.5f) * ((float)src_w / (float)dst_w_content) - 0.5f;
    float sy = (float)src_y1 + ((float)dy + 0.5f) * ((float)src_h / (float)dst_h) - 0.5f;
    sx = clamp_f(sx, (float)src_x1, (float)(src_x1 + src_w - 1));
    sy = clamp_f(sy, (float)src_y1, (float)(src_y1 + src_h - 1));

    int sx0 = (int)sx;
    int sy0 = (int)sy;
    int sx1 = sx0 + 1 < src_x1 + src_w ? sx0 + 1 : sx0;
    int sy1 = sy0 + 1 < src_y1 + src_h ? sy0 + 1 : sy0;
    float fx = sx - (float)sx0;
    float fy = sy - (float)sy0;

    // Sample 4 NV12 pixels
    auto sample = [&](int px, int py, float& r, float& g, float& b) {
        uint8_t Y = y_plane[py * y_pitch + px];
        int uvx = px >> 1;
        int uvy = py >> 1;
        uint8_t U = uv_plane[uvy * uv_pitch + (uvx << 1) + 0];
        uint8_t V = uv_plane[uvy * uv_pitch + (uvx << 1) + 1];
        nv12_to_rgb_norm(Y, U, V, r, g, b);
    };

    float r00, g00, b00, r10, g10, b10, r01, g01, b01, r11, g11, b11;
    sample(sx0, sy0, r00, g00, b00);
    sample(sx1, sy0, r10, g10, b10);
    sample(sx0, sy1, r01, g01, b01);
    sample(sx1, sy1, r11, g11, b11);

    float w00 = (1.0f - fx) * (1.0f - fy);
    float w10 = fx * (1.0f - fy);
    float w01 = (1.0f - fx) * fy;
    float w11 = fx * fy;

    out_nchw[idx + 0 * plane_size] = r00*w00 + r10*w10 + r01*w01 + r11*w11;
    out_nchw[idx + 1 * plane_size] = g00*w00 + g10*w10 + g01*w01 + g11*w11;
    out_nchw[idx + 2 * plane_size] = b00*w00 + b10*w10 + b01*w01 + b11*w11;
}
```

- [ ] **Step 2: Commit**

```bash
git add src/nodes/neural_net/preprocess/nv12_crop_resize_pad.cu
git commit -m "feat: add NV12 crop+resize+pad CUDA kernel for OCR preprocessing"
```

---

### Task 2: Scoreboard OCR Node

**Files:**
- Create: `src/nodes/neural_net/sport_specific/scoreboard_ocr.cpp`

The node extends `CudaInferTrtBase` to reuse engine loading, binding allocation, inference, and sync. It overrides preprocessing to use the crop+resize+pad kernel instead of `runPreprocessNV12`. The CTC greedy decode logic (from RapidOcrOnnx reference, Apache-2.0) is embedded directly — ~25 lines.

Key design:
- Reads `detection_metadata_key` (default `"yolo_players"`) from frame metadata
- Filters detections by label matching `scoreboard_labels` param (default `["Period", "Shot Clock", "Team Name", "Team Points", "Time Remaining"]`)
- Scales XYXY from model coords → source resolution using `model_width`/`model_height` from detection metadata
- For each scoreboard detection: dispatches the crop+resize+pad kernel, runs TRT inference, CTC-decodes
- Pairs two "Team Name" and two "Team Points" detections by Y-proximity: left X → team A, right X → team B
- Writes `output_metadata_key` (default `"scoreboard"`) JSON onto the frame
- `ocr_every_n` (default 25): on skipped frames, re-writes the last cached scoreboard JSON
- Loads character dictionary from `keys_file` param (path to `ppocr_keys_v1.txt` or `en_dict.txt`)

- [ ] **Step 1: Write the node**

Create `src/nodes/neural_net/sport_specific/scoreboard_ocr.cpp`:

```cpp
#include "../../node_common.hpp"
#include "../common/infer_trt_base.hpp"

extern "C" {
#include <libavutil/dict.h>
}

#include "../../../../objs/src/nodes/neural_net/preprocess/nv12_crop_resize_pad.ptx.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <string>
#include <vector>

namespace {

constexpr int kOcrDstHeight = 48;
constexpr int kOcrDstWidthPadded = 320;
constexpr float kPadValue = 0.0f;

struct ScoreboardDetection {
    std::string label;
    float x1, y1, x2, y2;
    float conf;
};

struct OcrResult {
    std::string label;
    std::string text;
    float mean_conf;
};

struct TeamPair {
    std::string name;
    std::string points;
    float name_conf;
    float points_conf;
    float center_x;
};

// CTC greedy decode: argmax per timestep, dedup consecutive, skip blank (index 0).
// Reference: RapidOcrOnnx CrnnNet::scoreToTextLine (Apache-2.0).
std::string ctcGreedyDecode(const float* output, int seq_len, int num_classes,
                            const std::vector<std::string>& keys, float& mean_conf) {
    std::string result;
    std::vector<float> scores;
    int last_index = 0;
    for (int t = 0; t < seq_len; ++t) {
        const float* row = output + t * num_classes;
        int best_idx = 0;
        float best_val = row[0];
        for (int c = 1; c < num_classes; ++c) {
            if (row[c] > best_val) {
                best_val = row[c];
                best_idx = c;
            }
        }
        if (best_idx > 0 && best_idx < (int)keys.size() && best_idx != last_index) {
            result.append(keys[best_idx]);
            scores.push_back(best_val);
        }
        last_index = best_idx;
    }
    mean_conf = 0.0f;
    if (!scores.empty()) {
        float sum = 0.0f;
        for (float s : scores) sum += s;
        mean_conf = sum / (float)scores.size();
    }
    return result;
}

} // namespace

class ScoreboardOcr : public NodeSISO<av::VideoFrame, av::VideoFrame>, public yolo_base::CudaInferTrtBase {
    std::string detection_metadata_key_ = "yolo_players";
    std::string output_metadata_key_ = "scoreboard";
    std::vector<std::string> scoreboard_labels_ = {"Period", "Shot Clock", "Team Name", "Team Points", "Time Remaining"};
    float min_conf_ = 0.4f;
    int ocr_every_n_ = 25;
    int debug_log_every_n_ = 0;
    std::string keys_file_;

    std::vector<std::string> keys_;
    CUmodule crop_module_ = nullptr;
    CUfunction crop_kernel_ = nullptr;
    CUdeviceptr d_crop_tensor_ = 0;
    bool ocr_initialized_ = false;
    uint64_t frame_counter_ = 0;
    std::string cached_scoreboard_json_;

    bool initOcr(const av::VideoFrame& frm) {
        if (ocr_initialized_) return true;
        if (!ensureInitialized(frm)) return false;

        // Load crop kernel PTX
        const std::string ptx_str(avpl_ocr_crop_ptx, avpl_ocr_crop_ptx + avpl_ocr_crop_ptx_len);
        if (CUDA_CHECK_CU(cuModuleLoadDataEx(&crop_module_, ptx_str.c_str(), 0, nullptr, nullptr))) {
            logstream << "scoreboard_ocr: failed to load crop PTX";
            return false;
        }
        if (CUDA_CHECK_CU(cuModuleGetFunction(&crop_kernel_, crop_module_, "kNV12_crop_resize_pad_RGB"))) {
            logstream << "scoreboard_ocr: failed to get crop kernel";
            return false;
        }

        // Allocate GPU buffer for one crop tensor: 1 x 3 x 48 x 320 float32
        const size_t tensor_bytes = 3 * kOcrDstHeight * kOcrDstWidthPadded * sizeof(float);
        if (CUDA_CHECK_CU(cuMemAlloc(&d_crop_tensor_, tensor_bytes))) {
            logstream << "scoreboard_ocr: failed to alloc crop tensor";
            return false;
        }

        // Load character keys
        std::ifstream kin(keys_file_);
        if (!kin) {
            logstream << "scoreboard_ocr: cannot open keys file: " << keys_file_;
            return false;
        }
        std::string line;
        keys_.clear();
        keys_.push_back("#"); // index 0 = CTC blank
        while (std::getline(kin, line)) {
            keys_.push_back(line);
        }
        keys_.push_back(" ");
        logstream << "scoreboard_ocr: loaded " << keys_.size() << " keys";

        ocr_initialized_ = true;
        return true;
    }

    std::vector<ScoreboardDetection> extractScoreboardDetections(const Parameters& md) {
        std::vector<ScoreboardDetection> dets;
        if (!md.contains("detections") || !md["detections"].is_array()) return dets;

        const double model_w = md.value("model_width", 960.0);
        const double model_h = md.value("model_height", 544.0);
        // Scale factors are computed per-frame in process() where we know source dims

        for (const auto& det : md["detections"]) {
            if (!det.is_object()) continue;
            if (!det.contains("label") || !det["label"].is_string()) continue;
            const std::string label = det["label"].get<std::string>();
            bool match = false;
            for (const auto& want : scoreboard_labels_) {
                if (label == want) { match = true; break; }
            }
            if (!match) continue;
            float conf = det.value("conf", 0.0f);
            if (conf < min_conf_) continue;
            if (!det.contains("xyxy") || !det["xyxy"].is_array() || det["xyxy"].size() < 4) continue;
            ScoreboardDetection sd;
            sd.label = label;
            sd.x1 = det["xyxy"][0].get<float>();
            sd.y1 = det["xyxy"][1].get<float>();
            sd.x2 = det["xyxy"][2].get<float>();
            sd.y2 = det["xyxy"][3].get<float>();
            sd.conf = conf;
            dets.push_back(sd);
        }
        return dets;
    }

    OcrResult runOcrOnCrop(const av::VideoFrame& frm, const ScoreboardDetection& det,
                           double scale_x, double scale_y) {
        OcrResult result;
        result.label = det.label;

        // Scale model coords to source resolution
        int sx1 = std::max(0, (int)(det.x1 * scale_x));
        int sy1 = std::max(0, (int)(det.y1 * scale_y));
        int sx2 = std::min(frm.width(), (int)(det.x2 * scale_x));
        int sy2 = std::min(frm.height(), (int)(det.y2 * scale_y));
        int src_w = sx2 - sx1;
        int src_h = sy2 - sy1;
        if (src_w <= 0 || src_h <= 0) return result;

        // Compute content width preserving aspect ratio
        int dst_w_content = (int)((float)src_w * ((float)kOcrDstHeight / (float)src_h));
        dst_w_content = std::max(1, std::min(dst_w_content, kOcrDstWidthPadded));

        const AVFrame* raw = frm.raw();
        const CUdeviceptr y_plane = (CUdeviceptr)(uintptr_t)raw->data[0];
        const int y_pitch = raw->linesize[0];
        const CUdeviceptr uv_plane = (CUdeviceptr)(uintptr_t)raw->data[1];
        const int uv_pitch = raw->linesize[1];

        // Get model's input tensor GPU pointer
        auto& model = models_[0];
        auto it = model.tensor_index.find(model.input_tensor_name);
        if (it == model.tensor_index.end()) return result;
        CUdeviceptr input_ptr = model.tensor_ptrs[it->second];

        // Launch crop+resize+pad kernel writing directly to TRT input tensor
        const int dst_h = kOcrDstHeight;
        const int dst_w_pad = kOcrDstWidthPadded;
        void* args[] = {
            (void*)&y_plane, (void*)&y_pitch,
            (void*)&uv_plane, (void*)&uv_pitch,
            (void*)&input_ptr,
            (void*)&sx1, (void*)&sy1, (void*)&src_w, (void*)&src_h,
            (void*)&dst_h, (void*)&dst_w_content, (void*)&dst_w_pad
        };
        const unsigned bx = 32, by = 8;
        const unsigned gx = ((unsigned)dst_w_pad + bx - 1) / bx;
        const unsigned gy = ((unsigned)dst_h + by - 1) / by;
        if (CUDA_CHECK_CU(cuLaunchKernel(crop_kernel_, gx, gy, 1, bx, by, 1, 0, model.stream, args, nullptr))) {
            return result;
        }

        // Run TRT inference + sync
        if (!runInference(model) || !syncModel(model)) return result;

        // CTC decode
        const auto& out = model.outputs[0];
        int seq_len = 1, num_classes = 1;
        if (out.dims.nbDims == 3) {
            seq_len = out.dims.d[1];
            num_classes = out.dims.d[2];
        } else if (out.dims.nbDims == 2) {
            seq_len = out.dims.d[0];
            num_classes = out.dims.d[1];
        }
        result.text = ctcGreedyDecode(out.host_output.data(), seq_len, num_classes, keys_, result.mean_conf);
        return result;
    }

    Parameters buildScoreboardJson(const std::vector<OcrResult>& results) {
        Parameters sb;

        // Collect team names and team points, pair by closest Y then assign A/B by X
        std::vector<const OcrResult*> names, points;
        for (const auto& r : results) {
            if (r.text.empty()) continue;
            if (r.label == "Team Name") names.push_back(&r);
            else if (r.label == "Team Points") points.push_back(&r);
            else if (r.label == "Period") {
                sb["period"] = {{"text", r.text}, {"conf", r.mean_conf}};
            } else if (r.label == "Shot Clock") {
                sb["shot_clock"] = {{"text", r.text}, {"conf", r.mean_conf}};
            } else if (r.label == "Time Remaining") {
                sb["time_remaining"] = {{"text", r.text}, {"conf", r.mean_conf}};
            }
        }

        // Simple pairing: if we have 2 names and 2 points, sort both by detection order
        // (which comes from YOLO left-to-right), assign first=team_a, second=team_b
        if (names.size() >= 1) {
            sb["team_a"] = Parameters::object();
            sb["team_a"]["name"] = {{"text", names[0]->text}, {"conf", names[0]->mean_conf}};
            if (points.size() >= 1) {
                sb["team_a"]["points"] = {{"text", points[0]->text}, {"conf", points[0]->mean_conf}};
            }
        }
        if (names.size() >= 2) {
            sb["team_b"] = Parameters::object();
            sb["team_b"]["name"] = {{"text", names[1]->text}, {"conf", names[1]->mean_conf}};
            if (points.size() >= 2) {
                sb["team_b"]["points"] = {{"text", points[1]->text}, {"conf", points[1]->mean_conf}};
            }
        }

        return sb;
    }

public:
    using NodeSISO<av::VideoFrame, av::VideoFrame>::NodeSISO;

    ~ScoreboardOcr() {
        if (cu_ctx_) {
            cuCtxSetCurrent(cu_ctx_);
            if (d_crop_tensor_) cuMemFree(d_crop_tensor_);
            if (crop_module_) cuModuleUnload(crop_module_);
        }
    }

    void process() override {
        av::VideoFrame frm = this->source_->get();
        if (!frm) return;
        if (isEofMarker(frm)) {
            cached_scoreboard_json_.clear();
            frame_counter_ = 0;
            this->sink_->put(frm);
            return;
        }

        ++frame_counter_;
        const AVFrame* raw = frm.raw();

        // On skipped frames, re-apply cached result
        if ((frame_counter_ % (uint64_t)ocr_every_n_) != 1 && !cached_scoreboard_json_.empty()) {
            if (raw && raw->metadata) {
                av_dict_set(&frm.raw()->metadata, output_metadata_key_.c_str(), cached_scoreboard_json_.c_str(), 0);
            }
            this->sink_->put(frm);
            return;
        }

        if (!raw || !raw->metadata) {
            this->sink_->put(frm);
            return;
        }

        if (!initOcr(frm)) {
            this->sink_->put(frm);
            return;
        }

        // Read detection metadata
        AVDictionaryEntry* entry = av_dict_get(raw->metadata, detection_metadata_key_.c_str(), nullptr, 0);
        if (!entry || !entry->value) {
            this->sink_->put(frm);
            return;
        }

        Parameters md;
        try {
            md = Parameters::parse(entry->value);
        } catch (...) {
            this->sink_->put(frm);
            return;
        }

        auto dets = extractScoreboardDetections(md);
        if (dets.empty()) {
            this->sink_->put(frm);
            return;
        }

        const double model_w = md.value("model_width", 960.0);
        const double model_h = md.value("model_height", 544.0);
        const double scale_x = (double)frm.width() / model_w;
        const double scale_y = (double)frm.height() / model_h;

        if (cu_ctx_) cuCtxSetCurrent(cu_ctx_);

        std::vector<OcrResult> results;
        for (const auto& det : dets) {
            OcrResult r = runOcrOnCrop(frm, det, scale_x, scale_y);
            if (!r.text.empty()) results.push_back(std::move(r));
        }

        Parameters sb = buildScoreboardJson(results);
        cached_scoreboard_json_ = sb.dump();
        av_dict_set(&frm.raw()->metadata, output_metadata_key_.c_str(), cached_scoreboard_json_.c_str(), 0);

        if (debug_log_every_n_ > 0 && (frame_counter_ % (uint64_t)debug_log_every_n_) == 0) {
            logstream << "scoreboard_ocr: frame=" << frame_counter_
                      << " dets=" << dets.size()
                      << " results=" << results.size()
                      << " json=" << cached_scoreboard_json_;
        }

        this->sink_->put(frm);
    }

    static std::shared_ptr<ScoreboardOcr> create(NodeCreationInfo& nci) {
        EdgeManager& edges = nci.edges;
        const Parameters& params = nci.params;
        auto r = NodeSISO<av::VideoFrame, av::VideoFrame>::template createCommon<ScoreboardOcr>(edges, params);

        if (params.count("detection_metadata_key")) r->detection_metadata_key_ = params["detection_metadata_key"].get<std::string>();
        if (params.count("output_metadata_key")) r->output_metadata_key_ = params["output_metadata_key"].get<std::string>();
        if (params.count("min_conf")) r->min_conf_ = params["min_conf"].get<float>();
        if (params.count("ocr_every_n")) r->ocr_every_n_ = std::max(1, params["ocr_every_n"].get<int>());
        if (params.count("debug_log_every_n")) r->debug_log_every_n_ = params["debug_log_every_n"].get<int>();
        if (params.count("scoreboard_labels")) {
            r->scoreboard_labels_.clear();
            for (const auto& item : params["scoreboard_labels"]) r->scoreboard_labels_.push_back(item.get<std::string>());
        }

        // TRT engine setup (reuses CudaInferTrtBase infrastructure)
        if (!params.count("ocr_model")) {
            throw Error("scoreboard_ocr: ocr_model param required (path to TRT .plan)");
        }
        if (!params.count("ocr_keys")) {
            throw Error("scoreboard_ocr: ocr_keys param required (path to keys .txt)");
        }
        r->keys_file_ = params["ocr_keys"].get<std::string>();

        yolo_base::ModelRunner model;
        model.engine_path = params["ocr_model"].get<std::string>();
        model.engine_name = std::filesystem::path(model.engine_path).filename().string();
        r->models_.push_back(std::move(model));

        return r;
    }
};

DECLNODE(scoreboard_ocr, ScoreboardOcr)
```

- [ ] **Step 2: Commit**

```bash
git add src/nodes/neural_net/sport_specific/scoreboard_ocr.cpp
git commit -m "feat: add scoreboard_ocr node with TRT inference and CTC decode"
```

---

### Task 3: Makefile Changes

**Files:**
- Modify: `Makefile` (lines ~142-145, the `HAVE_CUDA + NEURAL_NET_SPECIFIC + HAVE_NVCC` section)

Add the PTX kernel build rule for the crop kernel, with the scoreboard_ocr.o as its dependent.

- [ ] **Step 1: Add PTX kernel build rule and verify node is auto-discovered**

In the Makefile, find the block at line 142:

```makefile
ifeq ($(HAVE_CUDA)$(NEURAL_NET_SPECIFIC)$(HAVE_NVCC),111)
$(eval $(call ptx_kernel,$(SRCDIR)/nodes/neural_net/sport_specific/jersey_color_extract.cu,avpl_jersey_uv_mean_ptx,objs/src/nodes/neural_net/sport_specific/jersey_color_extract.o))
$(eval $(call ptx_kernel,$(SRCDIR)/nodes/neural_net/sport_specific/player_mask_feature_encoder.cu,avpl_player_mask_feature_encoder_ptx,objs/src/nodes/neural_net/sport_specific/player_mask_feature_encoder.o))
endif
```

Add after the `player_mask_feature_encoder` line:

```makefile
$(eval $(call ptx_kernel,$(SRCDIR)/nodes/neural_net/preprocess/nv12_crop_resize_pad.cu,avpl_ocr_crop_ptx,objs/src/nodes/neural_net/sport_specific/scoreboard_ocr.o))
```

The node `.cpp` itself is auto-discovered by the existing `find` at line 48:
```makefile
NODES_SRC += $(shell find $(SRCDIR)/nodes/neural_net/sport_specific -maxdepth 1 -name '*.cpp')
```

- [ ] **Step 2: Commit**

```bash
git add Makefile
git commit -m "build: add PTX kernel rule for scoreboard OCR crop kernel"
```

---

### Task 4: Model Documentation

**Files:**
- Modify: `doc/models.md` (append after the `siglip-base-patch16-224` section)

Add download instructions, conversion commands, and the model inventory entry for the PP-OCRv4 English recognition model.

- [ ] **Step 1: Add model entry to doc/models.md**

Append before the `## Base Weights` section:

```markdown
### en-ppocr-v4-rec (English OCR recognition)

English-only PP-OCRv4 recognition model for scoreboard text (team names, scores, time).
Input: `[1, 3, 48, 320]` (fixed, padded). Output: `[1, 80, num_classes]` (CTC sequence).
Uses `en_dict.txt` or `ppocr_keys_v1.txt` from `deps/RapidOcrOnnx/models/`.

| Stage | Path | Size | Notes |
|-------|------|------|-------|
| Paddle | `en_PP-OCRv4_rec_infer/` | ~10 MB | Downloaded from PaddleOCR |
| .onnx | `en-ppocr-v4-rec/en_PP-OCRv4_rec.onnx` | ~12 MB | Converted via paddle2onnx |
| .plan | `en-ppocr-v4-rec/en_PP-OCRv4_rec_48x320.plan` | ~8 MB | FP16, fixed 48x320 |

Download and convert:
\`\`\`bash
# 1. Download Paddle inference model
cd /home/fedora/tensorrt
mkdir -p en-ppocr-v4-rec && cd en-ppocr-v4-rec
wget https://paddleocr.bj.bcebos.com/PP-OCRv4/english/en_PP-OCRv4_rec_infer.tar
tar xf en_PP-OCRv4_rec_infer.tar

# 2. Convert Paddle → ONNX (in venv with paddle2onnx installed)
source /home/fedora/tensorrt/.venv/bin/activate
pip install paddle2onnx
paddle2onnx \
  --model_dir en_PP-OCRv4_rec_infer \
  --model_filename inference.pdmodel \
  --params_filename inference.pdiparams \
  --save_file en_PP-OCRv4_rec.onnx \
  --opset_version 14

# 3. Convert ONNX → TensorRT .plan (fixed input 48x320)
/opt/tensorrt/bin/trtexec \
  --onnx=en_PP-OCRv4_rec.onnx \
  --saveEngine=en_PP-OCRv4_rec_48x320.plan \
  --fp16 \
  --minShapes=x:1x3x48x320 \
  --optShapes=x:1x3x48x320 \
  --maxShapes=x:1x3x48x320
\`\`\`
```

- [ ] **Step 2: Commit**

```bash
git add doc/models.md
git commit -m "docs: add PP-OCRv4 English rec model entry with conversion commands"
```

---

### Task 5: Update Example Pipeline

**Files:**
- Verify: `examples/yolo/yolo_infer_all_players_tracker_live_teams_siglip_nba.avplumber`

The example was already updated earlier in this session. Verify the `scoreboard_ocr` node config points to the correct model path and keys file.

- [ ] **Step 1: Update model path to match Task 4**

The current example references `/home/fedora/tensorrt/ppocr/ch_PP-OCRv4_rec.onnx`. Update to the TRT plan path from Task 4:

Change:
```
"ocr_model": "/home/fedora/tensorrt/ppocr/ch_PP-OCRv4_rec.onnx"
"ocr_keys": "/home/fedora/tensorrt/ppocr/ppocr_keys_v1.txt"
```
To:
```
"ocr_model": "/home/fedora/tensorrt/en-ppocr-v4-rec/en_PP-OCRv4_rec_48x320.plan"
"ocr_keys": "/home/fedora/tensorrt/en-ppocr-v4-rec/en_dict.txt"
```

- [ ] **Step 2: Commit**

```bash
git add examples/yolo/yolo_infer_all_players_tracker_live_teams_siglip_nba.avplumber
git commit -m "fix(example): update scoreboard OCR model path to TRT plan"
```

---

### Task 6: Remote Build and Smoke Test

**Files:** None (remote execution only)

- [ ] **Step 1: Rsync all changed files to remote**

```bash
rsync -avz --relative -e "ssh -i /home/jp/work-misc-stuff/awsdev.pem" \
  /home/jp/git/avplumber/./{src/nodes/neural_net/preprocess/nv12_crop_resize_pad.cu,src/nodes/neural_net/sport_specific/scoreboard_ocr.cpp,Makefile,doc/models.md,examples/yolo/yolo_infer_all_players_tracker_live_teams_siglip_nba.avplumber} \
  fedora@172.17.36.132:/home/fedora/avplumber/

```

- [ ] **Step 2: Download and convert PP-OCRv4 English rec model on remote**

SSH to remote and run the commands from Task 4 step 1.

- [ ] **Step 3: Build on remote**

```bash
ssh -i /home/jp/work-misc-stuff/awsdev.pem fedora@172.17.36.132
cd /home/fedora/avplumber
make clean
make -j8 \
  NEURAL_NET_COMMON=1 \
  NEURAL_NET_SPECIFIC=1 \
  HAVE_CUDA=1 \
  HAVE_NVOF_FRUC=1 \
  HAVE_NVCC=1 \
  NVCC=/usr/local/cuda-13.0/bin/nvcc \
  TENSORRT_ROOT=/opt/tensorrt \
  PKG_CONFIG_PATH=/usr/local/lib/pkgconfig \
  CXXFLAGS+=' -I/usr/local/include -I/usr/local/cuda-13.0/include -I/usr/local/cuda-13.0/targets/x86_64-linux/include' \
  LFLAGS+=' -L/usr/local/lib -Wl,-rpath,/usr/local/lib -L/usr/local/cuda-13.0/targets/x86_64-linux/lib -Wl,-rpath,/usr/local/cuda-13.0/targets/x86_64-linux/lib'
```

Expected: Build succeeds, `scoreboard_ocr` appears in `graph_factory.generated.cpp`.

- [ ] **Step 4: Run the example pipeline**

```bash
LD_LIBRARY_PATH=/usr/local/lib:/opt/tensorrt/lib:/usr/local/cuda-13.0/targets/x86_64-linux/lib \
  ./avplumber -p 20200 -s examples/yolo/yolo_infer_all_players_tracker_live_teams_siglip_nba.avplumber
```

Expected: Pipeline starts, scoreboard_ocr logs show recognized text every 25 frames. Look for log lines like:
```
scoreboard_ocr: frame=25 dets=5 results=4 json={"period":{"text":"3","conf":0.95},...}
```
