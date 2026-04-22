#include "../../node_common.hpp"
#include "../common/infer_trt_base.hpp"
#include "../common/player_feature_side_data.hpp"
#include "../common/yolo_side_data.hpp"
#include <cuda_loader/cuda_drvapi_dynlink_cuda.h>

extern "C" {
#include <libavutil/buffer.h>
#include <libavutil/dict.h>
#include <libavutil/frame.h>
#include <libavutil/hwcontext.h>
#include <libavutil/hwcontext_cuda.h>
}

#include <NvInfer.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "../../../../objs/src/nodes/neural_net/sport_specific/player_mask_feature_encoder.ptx.h"

namespace {

using yolo_base::elementSize;
using yolo_base::halfToFloat;
using yolo_base::TRTLogger;

bool iequals(const std::string& a, const std::string& b) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i) {
        if (std::tolower((unsigned char)a[i]) != std::tolower((unsigned char)b[i])) return false;
    }
    return true;
}

float bboxIoU(float ax1, float ay1, float ax2, float ay2,
              float bx1, float by1, float bx2, float by2) {
    const float ix1 = std::max(ax1, bx1);
    const float iy1 = std::max(ay1, by1);
    const float ix2 = std::min(ax2, bx2);
    const float iy2 = std::min(ay2, by2);
    const float iw = std::max(0.0f, ix2 - ix1);
    const float ih = std::max(0.0f, iy2 - iy1);
    const float inter = iw * ih;
    const float ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter;
    return ua > 0.0f ? inter / ua : 0.0f;
}

float centerDistance(float ax1, float ay1, float ax2, float ay2,
                     float bx1, float by1, float bx2, float by2) {
    const float acx = 0.5f * (ax1 + ax2);
    const float acy = 0.5f * (ay1 + ay2);
    const float bcx = 0.5f * (bx1 + bx2);
    const float bcy = 0.5f * (by1 + by2);
    const float dx = acx - bcx;
    const float dy = acy - bcy;
    return std::sqrt(dx * dx + dy * dy);
}

int checkCu(CUresult err, const char* func) {
    if (err == CUDA_SUCCESS) return 0;
    const char* err_name = nullptr;
    const char* err_string = nullptr;
    if (cuGetErrorName && cuGetErrorString) {
        cuGetErrorName(err, &err_name);
        cuGetErrorString(err, &err_string);
    }
    logstream << "cuda function: " << func << " failed: "
              << (err_name ? err_name : "?") << ": " << (err_string ? err_string : "?");
    return -1;
}

#define PLAYER_FEATURE_CHECK_CU(x) checkCu((x), #x)

struct ParsedTracked {
    int det_index = -1;
    int track_id = -1;
    float x1 = 0.0f;
    float y1 = 0.0f;
    float x2 = 0.0f;
    float y2 = 0.0f;
};

struct ParsedSeg {
    int det_index = -1;
    int plane_index = -1;
    float x1 = 0.0f;
    float y1 = 0.0f;
    float x2 = 0.0f;
    float y2 = 0.0f;
};

struct MatchedCrop {
    int player_det_index = -1;
    int track_id = -1;
    int seg_det_index = -1;
    int seg_plane_index = -1;
    int bbox_xyxy[4] = {};
};

struct EmbedRunner {
    std::string engine_path;
    std::string input_name;
    std::string output_name;
    nvinfer1::IRuntime* runtime = nullptr;
    nvinfer1::ICudaEngine* engine = nullptr;
    nvinfer1::IExecutionContext* ctx = nullptr;
    nvinfer1::DataType input_dtype = nvinfer1::DataType::kFLOAT;
    nvinfer1::DataType output_dtype = nvinfer1::DataType::kFLOAT;
    int input_w = 224;
    int input_h = 224;
    int max_batch = 16;
    int embed_dim = 0;
    CUstream stream = nullptr;
    CUdeviceptr d_input = 0;
    CUdeviceptr d_output = 0;
    size_t input_bytes = 0;
    size_t output_bytes = 0;
    std::vector<float> host_output;
    std::vector<uint16_t> host_output_half;

    void cleanup() {
        if (stream) {
            cuStreamDestroy(stream);
            stream = nullptr;
        }
        if (d_input) {
            cuMemFree(d_input);
            d_input = 0;
        }
        if (d_output) {
            cuMemFree(d_output);
            d_output = 0;
        }
        if (ctx) {
            delete ctx;
            ctx = nullptr;
        }
        if (engine) {
            delete engine;
            engine = nullptr;
        }
        if (runtime) {
            delete runtime;
            runtime = nullptr;
        }
        host_output.clear();
        host_output_half.clear();
    }
};

bool matchesLabel(const Parameters& det, const std::vector<std::string>& labels) {
    if (labels.empty()) return true;
    if (!det.contains("label") || !det["label"].is_string()) return false;
    const std::string lbl = det["label"].get<std::string>();
    for (const auto& want : labels) {
        if (iequals(lbl, want)) return true;
    }
    return false;
}

bool parseBBox(const Parameters& det, float& x1, float& y1, float& x2, float& y2) {
    if (!det.contains("xyxy") || !det["xyxy"].is_array() || det["xyxy"].size() < 4) return false;
    x1 = det["xyxy"][0].get<float>();
    y1 = det["xyxy"][1].get<float>();
    x2 = det["xyxy"][2].get<float>();
    y2 = det["xyxy"][3].get<float>();
    return x2 > x1 && y2 > y1;
}

} // namespace

class PlayerMaskFeatureEncoder : public NodeSISO<av::VideoFrame, av::VideoFrame> {
    std::string player_metadata_key_ = "yolo_players";
    std::string seg_metadata_key_ = "yolo_players_seg";
    std::vector<std::string> player_labels_ = {"Player"};
    std::vector<std::string> seg_labels_ = {"player"};
    int seg_side_data_slot_ = 1;
    std::string engine_path_;
    float iou_match_threshold_ = 0.15f;
    float fallback_center_distance_px_ = 60.0f;
    float mask_threshold_ = 0.5f;
    int body_region_mode_ = 0; // 0 = full, 1 = torso
    float torso_x_margin_rel_ = 0.14f;
    float torso_y_start_rel_ = 0.18f;
    float torso_y_end_rel_ = 0.58f;
    int input_size_ = 224;
    int max_batch_ = 16;
    int min_crop_width_ = 8;
    int min_crop_height_ = 8;
    int debug_log_every_n_ = 0;

    AVCUDADeviceContext* cuda_dev_ctx_ = nullptr;
    CUcontext cu_ctx_ = nullptr;
    CUmodule preprocess_module_ = nullptr;
    CUfunction preprocess_kernel_ = nullptr;
    CUdeviceptr d_bboxes_ = 0;
    CUdeviceptr d_plane_indices_ = 0;
    int capacity_ = 0;
    TRTLogger trt_logger_;
    EmbedRunner runner_;
    bool initialized_ = false;
    uint64_t frame_counter_ = 0;

    bool initCudaContextFromFrame(const av::VideoFrame& frm) {
        if (!frm.raw() || !frm.raw()->hw_frames_ctx || !frm.raw()->hw_frames_ctx->data) return false;
        AVHWFramesContext* fctx = (AVHWFramesContext*)frm.raw()->hw_frames_ctx->data;
        if (!fctx || !fctx->device_ctx || !fctx->device_ctx->hwctx) return false;
        AVCUDADeviceContext* next = (AVCUDADeviceContext*)fctx->device_ctx->hwctx;
        if (!next || !next->cuda_ctx) return false;
        cuda_dev_ctx_ = next;
        cu_ctx_ = next->cuda_ctx;
        return PLAYER_FEATURE_CHECK_CU(cuCtxSetCurrent(cu_ctx_)) == 0;
    }

    bool loadPreprocessKernel() {
        if (preprocess_module_ && preprocess_kernel_) return true;
        const std::string ptx_str(avpl_player_mask_feature_encoder_ptx,
                                  avpl_player_mask_feature_encoder_ptx + avpl_player_mask_feature_encoder_ptx_len);
        if (PLAYER_FEATURE_CHECK_CU(cuModuleLoadDataEx(&preprocess_module_, ptx_str.c_str(), 0, nullptr, nullptr))) return false;
        const char* kname = (runner_.input_dtype == nvinfer1::DataType::kHALF)
            ? "kPlayerMaskToNCHW_fp16"
            : "kPlayerMaskToNCHW_fp32";
        if (PLAYER_FEATURE_CHECK_CU(cuModuleGetFunction(&preprocess_kernel_, preprocess_module_, kname))) return false;
        return true;
    }

    bool initRunner() {
        std::ifstream f(engine_path_, std::ios::binary);
        if (!f) {
            logstream << "player_mask_feature_encoder: cannot open engine file " << engine_path_;
            return false;
        }
        f.seekg(0, std::ios::end);
        std::streamsize size = f.tellg();
        if (size <= 0) return false;
        f.seekg(0, std::ios::beg);
        std::vector<char> blob((size_t)size);
        if (!f.read(blob.data(), size)) return false;

        runner_.engine_path = engine_path_;
        runner_.runtime = nvinfer1::createInferRuntime(trt_logger_);
        if (!runner_.runtime) return false;
        runner_.engine = runner_.runtime->deserializeCudaEngine(blob.data(), blob.size());
        if (!runner_.engine) return false;
        runner_.ctx = runner_.engine->createExecutionContext();
        if (!runner_.ctx) return false;

        const int nb = runner_.engine->getNbIOTensors();
        for (int i = 0; i < nb; ++i) {
            const char* name = runner_.engine->getIOTensorName(i);
            if (!name) return false;
            const auto mode = runner_.engine->getTensorIOMode(name);
            if (mode == nvinfer1::TensorIOMode::kINPUT && runner_.input_name.empty()) {
                runner_.input_name = name;
                runner_.input_dtype = runner_.engine->getTensorDataType(name);
                nvinfer1::Dims dims = runner_.engine->getTensorShape(name);
                if (dims.nbDims != 4) {
                    logstream << "player_mask_feature_encoder: expected 4D input tensor";
                    return false;
                }
                runner_.input_h = dims.d[2] > 0 ? dims.d[2] : input_size_;
                runner_.input_w = dims.d[3] > 0 ? dims.d[3] : input_size_;
            } else if (mode == nvinfer1::TensorIOMode::kOUTPUT && runner_.output_name.empty()) {
                runner_.output_name = name;
                runner_.output_dtype = runner_.engine->getTensorDataType(name);
            }
        }
        if (runner_.input_name.empty() || runner_.output_name.empty()) return false;

        runner_.max_batch = max_batch_;
        nvinfer1::Dims input_dims;
        input_dims.nbDims = 4;
        input_dims.d[0] = runner_.max_batch;
        input_dims.d[1] = 3;
        input_dims.d[2] = runner_.input_h;
        input_dims.d[3] = runner_.input_w;
        if (!runner_.ctx->setInputShape(runner_.input_name.c_str(), input_dims)) {
            logstream << "player_mask_feature_encoder: setInputShape failed for max batch";
            return false;
        }
        const nvinfer1::Dims out_dims = runner_.ctx->getTensorShape(runner_.output_name.c_str());
        int embed_dim = 1;
        for (int i = 1; i < out_dims.nbDims; ++i) {
            if (out_dims.d[i] <= 0) {
                logstream << "player_mask_feature_encoder: unresolved output dims";
                return false;
            }
            embed_dim *= out_dims.d[i];
        }
        if (embed_dim <= 0) return false;
        runner_.embed_dim = embed_dim;

        runner_.input_bytes = (size_t)runner_.max_batch * 3u * (size_t)runner_.input_h * (size_t)runner_.input_w * elementSize(runner_.input_dtype);
        runner_.output_bytes = (size_t)runner_.max_batch * (size_t)runner_.embed_dim * elementSize(runner_.output_dtype);
        if (PLAYER_FEATURE_CHECK_CU(cuMemAlloc(&runner_.d_input, runner_.input_bytes))) return false;
        if (PLAYER_FEATURE_CHECK_CU(cuMemAlloc(&runner_.d_output, runner_.output_bytes))) return false;
        if (PLAYER_FEATURE_CHECK_CU(cuStreamCreate(&runner_.stream, 0))) return false;
        if (!runner_.ctx->setTensorAddress(runner_.input_name.c_str(), reinterpret_cast<void*>(runner_.d_input))) return false;
        if (!runner_.ctx->setTensorAddress(runner_.output_name.c_str(), reinterpret_cast<void*>(runner_.d_output))) return false;

        if (runner_.output_dtype == nvinfer1::DataType::kHALF) {
            runner_.host_output_half.resize((size_t)runner_.max_batch * (size_t)runner_.embed_dim);
        } else {
            runner_.host_output.resize((size_t)runner_.max_batch * (size_t)runner_.embed_dim);
        }
        return true;
    }

    bool ensureCapacity(int needed) {
        if (needed <= capacity_) return true;
        int next = std::max(8, 1);
        while (next < needed) next <<= 1;
        if (d_bboxes_) {
            cuMemFree(d_bboxes_);
            d_bboxes_ = 0;
        }
        if (d_plane_indices_) {
            cuMemFree(d_plane_indices_);
            d_plane_indices_ = 0;
        }
        if (PLAYER_FEATURE_CHECK_CU(cuMemAlloc(&d_bboxes_, (size_t)next * 4u * sizeof(int)))) return false;
        if (PLAYER_FEATURE_CHECK_CU(cuMemAlloc(&d_plane_indices_, (size_t)next * sizeof(int)))) return false;
        capacity_ = next;
        return true;
    }

    bool ensureInitialized(const av::VideoFrame& frm) {
        if (initialized_) return true;
        if (!initCudaContextFromFrame(frm)) return false;
        if (!initRunner()) return false;
        if (!loadPreprocessKernel()) return false;
        initialized_ = true;
        return true;
    }

    void cleanup() {
        if (cu_ctx_) {
            cuCtxSetCurrent(cu_ctx_);
        }
        if (d_bboxes_) {
            cuMemFree(d_bboxes_);
            d_bboxes_ = 0;
        }
        if (d_plane_indices_) {
            cuMemFree(d_plane_indices_);
            d_plane_indices_ = 0;
        }
        capacity_ = 0;
        if (preprocess_module_) {
            cuModuleUnload(preprocess_module_);
            preprocess_module_ = nullptr;
            preprocess_kernel_ = nullptr;
        }
        runner_.cleanup();
        initialized_ = false;
    }

    void parseTracked(const Parameters& player_md, std::vector<ParsedTracked>& tracked) const {
        tracked.clear();
        if (!player_md.contains("detections") || !player_md["detections"].is_array()) return;
        for (int i = 0; i < (int)player_md["detections"].size(); ++i) {
            const auto& det = player_md["detections"][i];
            if (!det.is_object() || !matchesLabel(det, player_labels_)) continue;
            ParsedTracked p;
            p.det_index = i;
            p.track_id = det.value("track_id", -1);
            if (p.track_id < 0) continue;
            if (!parseBBox(det, p.x1, p.y1, p.x2, p.y2)) continue;
            tracked.push_back(p);
        }
    }

    void parseSeg(const Parameters& seg_md, std::vector<ParsedSeg>& segs) const {
        segs.clear();
        if (!seg_md.contains("detections") || !seg_md["detections"].is_array()) return;
        int plane_index = 0;
        for (int i = 0; i < (int)seg_md["detections"].size(); ++i) {
            const auto& det = seg_md["detections"][i];
            if (!det.is_object()) {
                ++plane_index;
                continue;
            }
            ParsedSeg s;
            s.det_index = i;
            s.plane_index = plane_index++;
            if (!matchesLabel(det, seg_labels_)) continue;
            if (!parseBBox(det, s.x1, s.y1, s.x2, s.y2)) continue;
            segs.push_back(s);
        }
    }

    void matchPlayersToSegs(const std::vector<ParsedTracked>& tracked,
                            const std::vector<ParsedSeg>& segs,
                            std::vector<MatchedCrop>& matched) const {
        matched.clear();
        std::vector<char> seg_used(segs.size(), 0);
        for (const auto& tr : tracked) {
            int best_idx = -1;
            float best_iou = 0.0f;
            for (int si = 0; si < (int)segs.size(); ++si) {
                if (seg_used[(size_t)si]) continue;
                const auto& sg = segs[(size_t)si];
                const float iou = bboxIoU(tr.x1, tr.y1, tr.x2, tr.y2, sg.x1, sg.y1, sg.x2, sg.y2);
                if (iou >= iou_match_threshold_ && iou > best_iou) {
                    best_iou = iou;
                    best_idx = si;
                }
            }
            if (best_idx < 0) {
                float best_center = fallback_center_distance_px_;
                for (int si = 0; si < (int)segs.size(); ++si) {
                    if (seg_used[(size_t)si]) continue;
                    const auto& sg = segs[(size_t)si];
                    const float center = centerDistance(tr.x1, tr.y1, tr.x2, tr.y2, sg.x1, sg.y1, sg.x2, sg.y2);
                    if (center <= best_center) {
                        best_center = center;
                        best_idx = si;
                    }
                }
            }
            if (best_idx < 0) continue;
            seg_used[(size_t)best_idx] = 1;
            const auto& sg = segs[(size_t)best_idx];
            MatchedCrop m;
            m.player_det_index = tr.det_index;
            m.track_id = tr.track_id;
            m.seg_det_index = sg.det_index;
            m.seg_plane_index = sg.plane_index;
            m.bbox_xyxy[0] = (int)std::floor(sg.x1);
            m.bbox_xyxy[1] = (int)std::floor(sg.y1);
            m.bbox_xyxy[2] = (int)std::ceil(sg.x2);
            m.bbox_xyxy[3] = (int)std::ceil(sg.y2);
            if ((m.bbox_xyxy[2] - m.bbox_xyxy[0]) < min_crop_width_ ||
                (m.bbox_xyxy[3] - m.bbox_xyxy[1]) < min_crop_height_) {
                continue;
            }
            matched.push_back(m);
        }
    }

    bool runPreprocess(const av::VideoFrame& frm,
                       const GpuMaskSideDataHeader& header,
                       const std::vector<MatchedCrop>& matched) {
        if (matched.empty()) return true;
        std::vector<int> host_bboxes((size_t)matched.size() * 4u);
        std::vector<int> host_plane_indices(matched.size());
        for (size_t i = 0; i < matched.size(); ++i) {
            std::memcpy(host_bboxes.data() + i * 4u, matched[i].bbox_xyxy, sizeof(matched[i].bbox_xyxy));
            host_plane_indices[i] = matched[i].seg_plane_index;
        }
        if (!ensureCapacity((int)matched.size())) return false;
        if (PLAYER_FEATURE_CHECK_CU(cuMemcpyHtoDAsync(d_bboxes_, host_bboxes.data(), host_bboxes.size() * sizeof(int), runner_.stream))) return false;
        if (PLAYER_FEATURE_CHECK_CU(cuMemcpyHtoDAsync(d_plane_indices_, host_plane_indices.data(), host_plane_indices.size() * sizeof(int), runner_.stream))) return false;

        const CUdeviceptr dY = (CUdeviceptr)(uintptr_t)frm.raw()->data[0];
        const CUdeviceptr dUV = (CUdeviceptr)(uintptr_t)frm.raw()->data[1];
        const size_t pitchY = (size_t)frm.raw()->linesize[0];
        const size_t pitchUV = (size_t)frm.raw()->linesize[1];
        const CUdeviceptr masks = (CUdeviceptr)(uintptr_t)header.gpu_ptr;
        const int frame_w = frm.width();
        const int frame_h = frm.height();
        const int proto_w = (int)header.proto_w;
        const int proto_h = (int)header.proto_h;
        const int model_w = (int)header.model_w;
        const int model_h = (int)header.model_h;
        const int body_region_mode = body_region_mode_;
        const float torso_x_margin_rel = torso_x_margin_rel_;
        const float torso_y_start_rel = torso_y_start_rel_;
        const float torso_y_end_rel = torso_y_end_rel_;
        const int out_w = runner_.input_w;
        const int out_h = runner_.input_h;
        const float mask_threshold = mask_threshold_;
        void* out = reinterpret_cast<void*>(runner_.d_input);

        void* args[] = {
            (void*)&dY,
            (void*)&pitchY,
            (void*)&dUV,
            (void*)&pitchUV,
            (void*)&frame_w,
            (void*)&frame_h,
            (void*)&masks,
            (void*)&proto_w,
            (void*)&proto_h,
            (void*)&model_w,
            (void*)&model_h,
            (void*)&body_region_mode,
            (void*)&torso_x_margin_rel,
            (void*)&torso_y_start_rel,
            (void*)&torso_y_end_rel,
            (void*)&d_bboxes_,
            (void*)&d_plane_indices_,
            (void*)&out_w,
            (void*)&out_h,
            (void*)&mask_threshold,
            (void*)&out
        };
        const unsigned int blockX = 16;
        const unsigned int blockY = 16;
        const unsigned int gridX = ((unsigned int)out_w + blockX - 1u) / blockX;
        const unsigned int gridY = ((unsigned int)out_h + blockY - 1u) / blockY;
        const unsigned int gridZ = (unsigned int)matched.size();
        if (PLAYER_FEATURE_CHECK_CU(cuLaunchKernel(preprocess_kernel_, gridX, gridY, gridZ, blockX, blockY, 1, 0, runner_.stream, args, nullptr))) {
            return false;
        }
        return true;
    }

    bool runEmbedding(int batch_size, std::vector<float>& out_embeddings) {
        if (batch_size <= 0) return true;
        nvinfer1::Dims input_dims;
        input_dims.nbDims = 4;
        input_dims.d[0] = batch_size;
        input_dims.d[1] = 3;
        input_dims.d[2] = runner_.input_h;
        input_dims.d[3] = runner_.input_w;
        if (!runner_.ctx->setInputShape(runner_.input_name.c_str(), input_dims)) return false;
        if (!runner_.ctx->setTensorAddress(runner_.input_name.c_str(), reinterpret_cast<void*>(runner_.d_input))) return false;
        if (!runner_.ctx->setTensorAddress(runner_.output_name.c_str(), reinterpret_cast<void*>(runner_.d_output))) return false;
        if (!runner_.ctx->enqueueV3(reinterpret_cast<cudaStream_t>(runner_.stream))) return false;

        const size_t elems = (size_t)batch_size * (size_t)runner_.embed_dim;
        out_embeddings.assign(elems, 0.0f);
        if (runner_.output_dtype == nvinfer1::DataType::kHALF) {
            if (PLAYER_FEATURE_CHECK_CU(cuMemcpyDtoHAsync(runner_.host_output_half.data(), runner_.d_output, elems * sizeof(uint16_t), runner_.stream))) return false;
            if (PLAYER_FEATURE_CHECK_CU(cuStreamSynchronize(runner_.stream))) return false;
            for (size_t i = 0; i < elems; ++i) out_embeddings[i] = halfToFloat(runner_.host_output_half[i]);
        } else {
            if (PLAYER_FEATURE_CHECK_CU(cuMemcpyDtoHAsync(out_embeddings.data(), runner_.d_output, elems * sizeof(float), runner_.stream))) return false;
            if (PLAYER_FEATURE_CHECK_CU(cuStreamSynchronize(runner_.stream))) return false;
        }
        return true;
    }

    void attachFeatureSideData(AVFrame* raw,
                               const std::vector<MatchedCrop>& matched,
                               const std::vector<float>& embeddings) {
        if (!raw || matched.empty()) return;
        const size_t header_size = sizeof(PlayerFeatureSideDataHeader);
        const size_t item_size = matched.size() * sizeof(PlayerFeatureSideDataItem);
        const size_t vec_size = embeddings.size() * sizeof(float);
        AVBufferRef* buf = av_buffer_alloc(header_size + item_size + vec_size);
        if (!buf) return;
        auto* header = (PlayerFeatureSideDataHeader*)buf->data;
        header->num_items = (uint32_t)matched.size();
        header->feature_dim = (uint32_t)runner_.embed_dim;
        header->reserved0 = 0;
        header->reserved1 = 0;
        auto* items = (PlayerFeatureSideDataItem*)(buf->data + header_size);
        uint8_t* vec_ptr = buf->data + header_size + item_size;
        for (size_t i = 0; i < matched.size(); ++i) {
            items[i].det_index = matched[i].player_det_index;
            items[i].track_id = matched[i].track_id;
            items[i].seg_index = matched[i].seg_det_index;
            items[i].reserved = 0;
        }
        std::memcpy(vec_ptr, embeddings.data(), vec_size);
        av_frame_new_side_data_from_buf(raw, AV_FRAME_DATA_PLAYER_FEATURE, buf);
    }

public:
    using NodeSISO<av::VideoFrame, av::VideoFrame>::NodeSISO;

    ~PlayerMaskFeatureEncoder() override {
        cleanup();
    }

    void process() override {
        av::VideoFrame frm = this->source_->get();
        if (!frm) return;
        if (isEofMarker(frm)) {
            this->sink_->put(frm);
            return;
        }

        ++frame_counter_;
        if (!frm.raw() || frm.raw()->format != AV_PIX_FMT_CUDA || !frm.raw()->metadata) {
            this->sink_->put(frm);
            return;
        }
        if (!ensureInitialized(frm)) {
            this->sink_->put(frm);
            return;
        }
        if (PLAYER_FEATURE_CHECK_CU(cuCtxSetCurrent(cu_ctx_))) {
            this->sink_->put(frm);
            return;
        }

        AVDictionaryEntry* player_entry = av_dict_get(frm.raw()->metadata, player_metadata_key_.c_str(), nullptr, 0);
        AVDictionaryEntry* seg_entry = av_dict_get(frm.raw()->metadata, seg_metadata_key_.c_str(), nullptr, 0);
        const AVFrameSideData* sd = av_frame_get_side_data(frm.raw(), yoloSegGpuSideDataType(seg_side_data_slot_));
        if (!player_entry || !player_entry->value || !seg_entry || !seg_entry->value || !sd || sd->size < (int)sizeof(GpuMaskSideDataHeader)) {
            this->sink_->put(frm);
            return;
        }

        Parameters player_md;
        Parameters seg_md;
        try {
            player_md = Parameters::parse(player_entry->value);
            seg_md = Parameters::parse(seg_entry->value);
        } catch (...) {
            this->sink_->put(frm);
            return;
        }
        if (!player_md.contains("detections") || !player_md["detections"].is_array()) {
            this->sink_->put(frm);
            return;
        }

        for (auto& det : player_md["detections"]) {
            if (!det.is_object()) continue;
            det["embed_valid"] = 0;
        }

        std::vector<ParsedTracked> tracked;
        std::vector<ParsedSeg> segs;
        std::vector<MatchedCrop> matched;
        parseTracked(player_md, tracked);
        parseSeg(seg_md, segs);
        matchPlayersToSegs(tracked, segs, matched);

        const auto* header = (const GpuMaskSideDataHeader*)sd->data;
        std::vector<float> embeddings;
        if (!matched.empty() &&
            runPreprocess(frm, *header, matched) &&
            runEmbedding((int)matched.size(), embeddings)) {
            for (size_t i = 0; i < matched.size(); ++i) {
                if ((size_t)matched[i].player_det_index < player_md["detections"].size()) {
                    auto& det = player_md["detections"][matched[i].player_det_index];
                    det["embed_valid"] = 1;
                    det["embed_index"] = (int)i;
                }
            }
            attachFeatureSideData(frm.raw(), matched, embeddings);
            av_dict_set(&frm.raw()->metadata, player_metadata_key_.c_str(), player_md.dump().c_str(), 0);
        }

        if (debug_log_every_n_ > 0 && (frame_counter_ % (uint64_t)debug_log_every_n_) == 0) {
            logstream << "player_mask_feature_encoder: frame=" << frame_counter_
                      << " tracked=" << tracked.size()
                      << " seg=" << segs.size()
                      << " matched=" << matched.size()
                      << " body=" << (body_region_mode_ == 1 ? "torso" : "full")
                      << " batch=" << matched.size()
                      << " feature_dim=" << runner_.embed_dim;
        }

        this->sink_->put(frm);
    }

    static std::shared_ptr<PlayerMaskFeatureEncoder> create(NodeCreationInfo& nci) {
        EdgeManager& edges = nci.edges;
        const Parameters& params = nci.params;
        std::shared_ptr<Edge<av::VideoFrame>> src = edges.find<av::VideoFrame>(params["src"]);
        std::shared_ptr<Edge<av::VideoFrame>> dst = edges.find<av::VideoFrame>(params["dst"]);
        auto r = std::make_shared<PlayerMaskFeatureEncoder>(src->makeSource(), dst->makeSink());
        if (params.count("player_metadata_key")) r->player_metadata_key_ = params["player_metadata_key"].get<std::string>();
        if (params.count("seg_metadata_key")) r->seg_metadata_key_ = params["seg_metadata_key"].get<std::string>();
        if (params.count("player_labels")) {
            auto labels = jsonToStringList(params["player_labels"]);
            r->player_labels_.assign(labels.begin(), labels.end());
        }
        if (params.count("seg_labels")) {
            auto labels = jsonToStringList(params["seg_labels"]);
            r->seg_labels_.assign(labels.begin(), labels.end());
        }
        if (params.count("seg_side_data_slot")) r->seg_side_data_slot_ = params["seg_side_data_slot"].get<int>();
        if (params.count("engine")) r->engine_path_ = params["engine"].get<std::string>();
        if (params.count("iou_match_threshold")) r->iou_match_threshold_ = params["iou_match_threshold"].get<float>();
        if (params.count("fallback_center_distance_px")) r->fallback_center_distance_px_ = params["fallback_center_distance_px"].get<float>();
        if (params.count("mask_threshold")) r->mask_threshold_ = params["mask_threshold"].get<float>();
        if (params.count("body_region")) {
            const std::string body = params["body_region"].get<std::string>();
            r->body_region_mode_ = iequals(body, "torso") ? 1 : 0;
        }
        if (params.count("torso_x_margin_rel")) r->torso_x_margin_rel_ = params["torso_x_margin_rel"].get<float>();
        if (params.count("torso_y_start_rel")) r->torso_y_start_rel_ = params["torso_y_start_rel"].get<float>();
        if (params.count("torso_y_end_rel")) r->torso_y_end_rel_ = params["torso_y_end_rel"].get<float>();
        if (params.count("input_size")) r->input_size_ = params["input_size"].get<int>();
        if (params.count("max_batch")) r->max_batch_ = params["max_batch"].get<int>();
        if (params.count("min_crop_width")) r->min_crop_width_ = params["min_crop_width"].get<int>();
        if (params.count("min_crop_height")) r->min_crop_height_ = params["min_crop_height"].get<int>();
        if (params.count("debug_log_every_n")) r->debug_log_every_n_ = params["debug_log_every_n"].get<int>();
        if (r->engine_path_.empty()) throw Error("player_mask_feature_encoder: missing required parameter: engine");
        if (!yoloSegIsValidSlot(r->seg_side_data_slot_)) {
            throw Error("player_mask_feature_encoder: seg_side_data_slot out of range");
        }
        if (r->max_batch_ <= 0) throw Error("player_mask_feature_encoder: max_batch must be positive");
        return r;
    }
};

DECLNODE(player_mask_feature_encoder, PlayerMaskFeatureEncoder)
