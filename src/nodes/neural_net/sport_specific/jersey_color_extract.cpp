#include "../../node_common.hpp"
#include "../common/yolo_side_data.hpp"
#include <cuda_loader/cuda_drvapi_dynlink_cuda.h>

extern "C" {
#include <libavutil/dict.h>
#include <libavutil/frame.h>
#include <libavutil/hwcontext.h>
#include <libavutil/hwcontext_cuda.h>
}

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <sstream>
#include <string>
#include <vector>

#include "../../../../objs/src/nodes/neural_net/sport_specific/jersey_color_extract.ptx.h"

namespace {

bool iequals(const std::string& a, const std::string& b) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i) {
        if (std::tolower((unsigned char)a[i]) != std::tolower((unsigned char)b[i])) return false;
    }
    return true;
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

#define JERSEY_CHECK_CU(x) checkCu((x), #x)

struct PackedDetection {
    int array_index = -1;
    int plane_index = -1;
    int x1 = 0;
    int y1 = 0;
    int x2 = 0;
    int y2 = 0;
};

struct GpuMaskRelease {
    CUcontext ctx = nullptr;
    CUdeviceptr ptr = 0;
};

void freeGpuMaskSideData(void* opaque, uint8_t* data) {
    GpuMaskRelease* release = (GpuMaskRelease*)opaque;
    if (release) {
        if (release->ctx && cuCtxSetCurrent) {
            cuCtxSetCurrent(release->ctx);
        }
        if (release->ptr && cuMemFree) {
            cuMemFree(release->ptr);
        }
        av_free(release);
    }
    av_free(data);
}

void appendSegPointerDebug(std::ostringstream& oss, CUcontext expected_ctx, CUdeviceptr ptr) {
    if (!ptr) {
        oss << " gpu_ptr=0";
        return;
    }

    CUcontext current_ctx = nullptr;
    if (cuCtxGetCurrent) {
        CUresult current_res = cuCtxGetCurrent(&current_ctx);
        if (current_res == CUDA_SUCCESS) {
            oss << " current_ctx=" << current_ctx;
        } else {
            oss << " current_ctx=<error:" << (int)current_res << ">";
        }
    }
    oss << " expected_ctx=" << expected_ctx
        << " gpu_ptr=" << (const void*)(uintptr_t)ptr;

    CUdeviceptr base_ptr = 0;
    size_t base_size = 0;
    if (cuMemGetAddressRange) {
        CUresult range_res = cuMemGetAddressRange(&base_ptr, &base_size, ptr);
        if (range_res == CUDA_SUCCESS) {
            oss << " range_base=" << (const void*)(uintptr_t)base_ptr
                << " range_bytes=" << base_size;
        } else {
            oss << " range_base=<error:" << (int)range_res << ">";
        }
    } else {
        oss << " range_base=<unavailable>";
    }
}

} // namespace

class JerseyColorExtract : public NodeSISO<av::VideoFrame, av::VideoFrame>, public ReportsFinishByFlag {
    std::string metadata_key_ = "yolo_players_seg";
    std::vector<std::string> target_labels_ = {"player"};
    int target_class_ = -1;
    bool label_case_sensitive_ = false;
    float mask_threshold_ = 0.5f;
    int min_pixels_ = 32;
    int body_region_mode_ = 0; // 0 = full, 1 = torso
    float torso_x_margin_rel_ = 0.14f;
    float torso_y_start_rel_ = 0.18f;
    float torso_y_end_rel_ = 0.58f;
    float sample_inner_x_margin_rel_ = 0.0f;
    float sample_top_y_exclusion_rel_ = 0.0f;
    int skin_y_min_ = 45;
    int skin_y_max_ = 235;
    int skin_u_min_ = 80;
    int skin_u_max_ = 132;
    int skin_v_min_ = 126;
    int skin_v_max_ = 182;
    int skin_neutral_y_min_ = 185;
    int skin_neutral_u_tol_ = 10;
    int skin_neutral_v_tol_ = 10;
    int initial_capacity_ = 32;
    int debug_log_every_n_ = 0;
    int side_data_slot_ = 0;
    std::string debug_output_metadata_key_;
    int debug_output_side_data_slot_ = -1;

    AVCUDADeviceContext* cuda_dev_ctx_ = nullptr;
    CUcontext cu_ctx_ = nullptr;
    CUmodule cu_module_ = nullptr;
    CUfunction kernel_ = nullptr;

    CUdeviceptr d_bboxes_ = 0;
    CUdeviceptr d_plane_indices_ = 0;
    CUdeviceptr d_out_best_yuv_ = 0;
    CUdeviceptr d_out_best_count_ = 0;
    CUdeviceptr d_out_cloth_count_ = 0;
    CUdeviceptr d_out_skin_count_ = 0;
    CUdeviceptr d_out_confidence_ = 0;
    CUdeviceptr d_out_uv_hist_ = 0;
    CUdeviceptr d_out_l_hist_ = 0;
    static constexpr int kUVHistSize = 256;
    static constexpr int kLHistSize = 16;
    int capacity_ = 0;
    uint64_t frame_counter_ = 0;

    bool matchesTarget(const Parameters& det) const {
        if (target_class_ >= 0 && det.contains("cls")) {
            if (det["cls"].get<int>() == target_class_) return true;
        }
        if (!target_labels_.empty() && det.contains("label")) {
            const std::string lbl = det["label"].get<std::string>();
            for (const auto& t : target_labels_) {
                if (label_case_sensitive_) {
                    if (lbl == t) return true;
                } else {
                    if (iequals(lbl, t)) return true;
                }
            }
        }
        return target_class_ < 0 && target_labels_.empty();
    }

    void releaseBuffers() {
        if (!cu_ctx_) return;
        JERSEY_CHECK_CU(cuCtxSetCurrent(cu_ctx_));
        if (d_bboxes_) { JERSEY_CHECK_CU(cuMemFree(d_bboxes_)); d_bboxes_ = 0; }
        if (d_plane_indices_) { JERSEY_CHECK_CU(cuMemFree(d_plane_indices_)); d_plane_indices_ = 0; }
        if (d_out_best_yuv_) { JERSEY_CHECK_CU(cuMemFree(d_out_best_yuv_)); d_out_best_yuv_ = 0; }
        if (d_out_best_count_) { JERSEY_CHECK_CU(cuMemFree(d_out_best_count_)); d_out_best_count_ = 0; }
        if (d_out_cloth_count_) { JERSEY_CHECK_CU(cuMemFree(d_out_cloth_count_)); d_out_cloth_count_ = 0; }
        if (d_out_skin_count_) { JERSEY_CHECK_CU(cuMemFree(d_out_skin_count_)); d_out_skin_count_ = 0; }
        if (d_out_confidence_) { JERSEY_CHECK_CU(cuMemFree(d_out_confidence_)); d_out_confidence_ = 0; }
        if (d_out_uv_hist_) { JERSEY_CHECK_CU(cuMemFree(d_out_uv_hist_)); d_out_uv_hist_ = 0; }
        if (d_out_l_hist_) { JERSEY_CHECK_CU(cuMemFree(d_out_l_hist_)); d_out_l_hist_ = 0; }
        capacity_ = 0;
    }

    bool initCudaContextFromFrame(const av::VideoFrame& frm) {
        if (!frm.raw() || !frm.raw()->hw_frames_ctx || !frm.raw()->hw_frames_ctx->data) return false;
        AVHWFramesContext* fctx = (AVHWFramesContext*)frm.raw()->hw_frames_ctx->data;
        if (!fctx || !fctx->device_ctx || !fctx->device_ctx->hwctx) return false;
        AVCUDADeviceContext* next = (AVCUDADeviceContext*)fctx->device_ctx->hwctx;
        if (!next || !next->cuda_ctx) return false;
        cuda_dev_ctx_ = next;
        cu_ctx_ = next->cuda_ctx;
        return JERSEY_CHECK_CU(cuCtxSetCurrent(cu_ctx_)) == 0;
    }

    bool loadKernel() {
        if (cu_module_ && kernel_) return true;
        if (!cu_ctx_) return false;
        if (JERSEY_CHECK_CU(cuCtxSetCurrent(cu_ctx_))) return false;
        const std::string ptx_str(avpl_jersey_uv_mean_ptx,
                                  avpl_jersey_uv_mean_ptx + avpl_jersey_uv_mean_ptx_len);
        if (JERSEY_CHECK_CU(cuModuleLoadDataEx(&cu_module_, ptx_str.c_str(), 0, nullptr, nullptr))) return false;
        if (JERSEY_CHECK_CU(cuModuleGetFunction(&kernel_, cu_module_, "kJerseyUVMean"))) return false;
        return true;
    }

    bool ensureCapacity(int needed) {
        if (needed <= capacity_) return true;
        int next = std::max(initial_capacity_, 1);
        while (next < needed) next <<= 1;
        releaseBuffers();
        if (JERSEY_CHECK_CU(cuCtxSetCurrent(cu_ctx_))) return false;
        if (JERSEY_CHECK_CU(cuMemAlloc(&d_bboxes_, (size_t)next * 4u * sizeof(int)))) return false;
        if (JERSEY_CHECK_CU(cuMemAlloc(&d_plane_indices_, (size_t)next * sizeof(int)))) return false;
        if (JERSEY_CHECK_CU(cuMemAlloc(&d_out_best_yuv_, (size_t)next * 3u * sizeof(float)))) return false;
        if (JERSEY_CHECK_CU(cuMemAlloc(&d_out_best_count_, (size_t)next * sizeof(int)))) return false;
        if (JERSEY_CHECK_CU(cuMemAlloc(&d_out_cloth_count_, (size_t)next * sizeof(int)))) return false;
        if (JERSEY_CHECK_CU(cuMemAlloc(&d_out_skin_count_, (size_t)next * sizeof(int)))) return false;
        if (JERSEY_CHECK_CU(cuMemAlloc(&d_out_confidence_, (size_t)next * sizeof(float)))) return false;
        if (JERSEY_CHECK_CU(cuMemAlloc(&d_out_uv_hist_, (size_t)next * kUVHistSize * sizeof(float)))) return false;
        if (JERSEY_CHECK_CU(cuMemAlloc(&d_out_l_hist_, (size_t)next * kLHistSize * sizeof(float)))) return false;
        capacity_ = next;
        return true;
    }

    bool packDetections(Parameters& md, std::vector<PackedDetection>& packed) const {
        packed.clear();
        if (!md.contains("detections") || !md["detections"].is_array()) return false;

        int plane_index = 0;
        for (int i = 0; i < (int)md["detections"].size(); ++i) {
            const auto& det = md["detections"][i];
            if (!det.is_object()) continue;
            if (!det.contains("xyxy") || !det["xyxy"].is_array() || det["xyxy"].size() < 4) {
                ++plane_index;
                continue;
            }
            if (!matchesTarget(det)) {
                ++plane_index;
                continue;
            }

            PackedDetection item;
            item.array_index = i;
            item.plane_index = plane_index;
            item.x1 = (int)std::lround(det["xyxy"][0].get<double>());
            item.y1 = (int)std::lround(det["xyxy"][1].get<double>());
            item.x2 = (int)std::lround(det["xyxy"][2].get<double>());
            item.y2 = (int)std::lround(det["xyxy"][3].get<double>());
            if (item.x2 > item.x1 && item.y2 > item.y1) {
                packed.push_back(item);
            }
            ++plane_index;
        }
        return !packed.empty();
    }

public:
    using NodeSISO<av::VideoFrame, av::VideoFrame>::NodeSISO;
    bool consumeEofIfPresent() override {
        return false;
    }

    ~JerseyColorExtract() {
        if (cu_ctx_) {
            JERSEY_CHECK_CU(cuCtxSetCurrent(cu_ctx_));
        }
        releaseBuffers();
        if (cu_module_) {
            JERSEY_CHECK_CU(cuModuleUnload(cu_module_));
            cu_module_ = nullptr;
        }
    }

    void process() override {
        av::VideoFrame frm = this->source_->get();
        if (isEofMarker(frm)) {
            this->sink_->put(frm);
            this->finished_ = true;
            return;
        }
        if (!frm) return;

        ++frame_counter_;
        const AVFrame* raw = frm.raw();
        if (!raw || !raw->metadata) {
            this->sink_->put(frm);
            return;
        }

        AVDictionaryEntry* entry = av_dict_get(raw->metadata, metadata_key_.c_str(), nullptr, 0);
        if (!entry || !entry->value) {
            this->sink_->put(frm);
            return;
        }

        const AVFrameSideData* sd = av_frame_get_side_data(raw, yoloSegGpuSideDataType(side_data_slot_));
        if (!sd || !sd->buf || sd->buf->size < (int)sizeof(GpuMaskSideDataHeader)) {
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

        const GpuMaskSideDataHeader* header = (const GpuMaskSideDataHeader*)sd->buf->data;
        const CUdeviceptr gpu_masks = (CUdeviceptr)header->gpu_ptr;
        if (!gpu_masks || header->num_masks == 0 || header->proto_w == 0 || header->proto_h == 0) {
            this->sink_->put(frm);
            return;
        }

        std::vector<PackedDetection> packed;
        if (!packDetections(md, packed)) {
            this->sink_->put(frm);
            return;
        }

        size_t invalid_plane_count = 0;
        for (const auto& p : packed) {
            if (p.plane_index < 0 || p.plane_index >= (int)header->num_masks) {
                ++invalid_plane_count;
            }
        }
        if (invalid_plane_count > 0) {
            std::ostringstream err;
            err << "jersey_color_extract: mask index out of range"
                << " frame=" << frame_counter_
                << " slot=" << side_data_slot_
                << " invalid=" << invalid_plane_count
                << " packed=" << packed.size()
                << " num_masks=" << header->num_masks;
            throw Error(err.str());
        }

        if (!initCudaContextFromFrame(frm) || !loadKernel() || !ensureCapacity((int)packed.size())) {
            this->sink_->put(frm);
            return;
        }

        if (debug_log_every_n_ > 0 && (frame_counter_ % (uint64_t)debug_log_every_n_) == 0) {
            std::ostringstream dbg;
            dbg << "jersey_color_extract: gpu side data"
                << " frame=" << frame_counter_
                << " slot=" << side_data_slot_
                << " sd_type=" << (int)yoloSegGpuSideDataType(side_data_slot_)
                << " packed=" << packed.size()
                << " num_masks=" << header->num_masks
                << " proto=" << header->proto_w << "x" << header->proto_h
                << " model=" << header->model_w << "x" << header->model_h
                << " uv_pitch=" << raw->linesize[1]
                << " first_plane=" << (packed.empty() ? -1 : packed.front().plane_index);
            appendSegPointerDebug(dbg, cu_ctx_, gpu_masks);
            logstream << dbg.str();
        }

        std::vector<int> host_bboxes;
        std::vector<int> host_plane_indices;
        host_bboxes.reserve(packed.size() * 4u);
        host_plane_indices.reserve(packed.size());
        for (const auto& p : packed) {
            host_bboxes.push_back(p.x1);
            host_bboxes.push_back(p.y1);
            host_bboxes.push_back(p.x2);
            host_bboxes.push_back(p.y2);
            host_plane_indices.push_back(p.plane_index);
        }

        if (JERSEY_CHECK_CU(cuMemcpyHtoDAsync(d_bboxes_, host_bboxes.data(), host_bboxes.size() * sizeof(int), cuda_dev_ctx_->stream))) {
            this->sink_->put(frm);
            return;
        }
        if (JERSEY_CHECK_CU(cuMemcpyHtoDAsync(d_plane_indices_, host_plane_indices.data(), host_plane_indices.size() * sizeof(int), cuda_dev_ctx_->stream))) {
            this->sink_->put(frm);
            return;
        }

        const CUdeviceptr y_plane = (CUdeviceptr)(uintptr_t)raw->data[0];
        const int y_pitch = raw->linesize[0];
        const CUdeviceptr uv_plane = (CUdeviceptr)(uintptr_t)raw->data[1];
        const int uv_pitch = raw->linesize[1];
        const int chroma_w = frm.width() / 2;
        const int chroma_h = frm.height() / 2;
        const int proto_w = (int)header->proto_w;
        const int proto_h = (int)header->proto_h;
        const int model_w = (int)header->model_w;
        const int model_h = (int)header->model_h;
        const int body_region_mode = body_region_mode_;
        const float mask_threshold = mask_threshold_;
        const float torso_x_margin_rel = torso_x_margin_rel_;
        const float torso_y_start_rel = torso_y_start_rel_;
        const float torso_y_end_rel = torso_y_end_rel_;
        const float sample_inner_x_margin_rel = sample_inner_x_margin_rel_;
        const float sample_top_y_exclusion_rel = sample_top_y_exclusion_rel_;
        const int skin_y_min = skin_y_min_;
        const int skin_y_max = skin_y_max_;
        const int skin_u_min = skin_u_min_;
        const int skin_u_max = skin_u_max_;
        const int skin_v_min = skin_v_min_;
        const int skin_v_max = skin_v_max_;
        const int skin_neutral_y_min = skin_neutral_y_min_;
        const int skin_neutral_u_tol = skin_neutral_u_tol_;
        const int skin_neutral_v_tol = skin_neutral_v_tol_;
        CUdeviceptr d_debug_masks = 0;
        const bool emit_debug_masks = !debug_output_metadata_key_.empty() && yoloSegIsValidSlot(debug_output_side_data_slot_);
        const size_t debug_mask_bytes = emit_debug_masks
                                            ? ((size_t)packed.size() * (size_t)proto_w * (size_t)proto_h * sizeof(float))
                                            : 0;
        if (emit_debug_masks) {
            if (JERSEY_CHECK_CU(cuMemAlloc(&d_debug_masks, debug_mask_bytes))) {
                this->sink_->put(frm);
                return;
            }
        }

        void* args[] = {
            (void*)&y_plane,
            (void*)&y_pitch,
            (void*)&uv_plane,
            (void*)&uv_pitch,
            (void*)&chroma_w,
            (void*)&chroma_h,
            (void*)&gpu_masks,
            (void*)&proto_w,
            (void*)&proto_h,
            (void*)&model_w,
            (void*)&model_h,
            (void*)&d_bboxes_,
            (void*)&d_plane_indices_,
            (void*)&mask_threshold,
            (void*)&body_region_mode,
            (void*)&torso_x_margin_rel,
            (void*)&torso_y_start_rel,
            (void*)&torso_y_end_rel,
            (void*)&sample_inner_x_margin_rel,
            (void*)&sample_top_y_exclusion_rel,
            (void*)&skin_y_min,
            (void*)&skin_y_max,
            (void*)&skin_u_min,
            (void*)&skin_u_max,
            (void*)&skin_v_min,
            (void*)&skin_v_max,
            (void*)&skin_neutral_y_min,
            (void*)&skin_neutral_u_tol,
            (void*)&skin_neutral_v_tol,
            (void*)&d_debug_masks,
            (void*)&d_out_best_yuv_,
            (void*)&d_out_best_count_,
            (void*)&d_out_cloth_count_,
            (void*)&d_out_skin_count_,
            (void*)&d_out_confidence_,
            (void*)&d_out_uv_hist_,
            (void*)&d_out_l_hist_
        };

        if (JERSEY_CHECK_CU(cuLaunchKernel(kernel_,
                                           (unsigned int)packed.size(), 1, 1,
                                           32, 8, 1,
                                           0, cuda_dev_ctx_->stream, args, nullptr))) {
            if (d_debug_masks) JERSEY_CHECK_CU(cuMemFree(d_debug_masks));
            this->sink_->put(frm);
            return;
        }
        if (JERSEY_CHECK_CU(cuStreamSynchronize(cuda_dev_ctx_->stream))) {
            if (d_debug_masks) JERSEY_CHECK_CU(cuMemFree(d_debug_masks));
            this->sink_->put(frm);
            return;
        }

        std::vector<float> host_best_yuv(packed.size() * 3u, 0.0f);
        std::vector<int> host_best_count(packed.size(), 0);
        std::vector<int> host_cloth_count(packed.size(), 0);
        std::vector<int> host_skin_count(packed.size(), 0);
        std::vector<float> host_confidence(packed.size(), 0.0f);
        if (JERSEY_CHECK_CU(cuMemcpyDtoH(host_best_yuv.data(), d_out_best_yuv_, host_best_yuv.size() * sizeof(float)))) {
            if (d_debug_masks) JERSEY_CHECK_CU(cuMemFree(d_debug_masks));
            this->sink_->put(frm);
            return;
        }
        if (JERSEY_CHECK_CU(cuMemcpyDtoH(host_best_count.data(), d_out_best_count_, host_best_count.size() * sizeof(int)))) {
            if (d_debug_masks) JERSEY_CHECK_CU(cuMemFree(d_debug_masks));
            this->sink_->put(frm);
            return;
        }
        if (JERSEY_CHECK_CU(cuMemcpyDtoH(host_cloth_count.data(), d_out_cloth_count_, host_cloth_count.size() * sizeof(int)))) {
            if (d_debug_masks) JERSEY_CHECK_CU(cuMemFree(d_debug_masks));
            this->sink_->put(frm);
            return;
        }
        if (JERSEY_CHECK_CU(cuMemcpyDtoH(host_skin_count.data(), d_out_skin_count_, host_skin_count.size() * sizeof(int)))) {
            if (d_debug_masks) JERSEY_CHECK_CU(cuMemFree(d_debug_masks));
            this->sink_->put(frm);
            return;
        }
        if (JERSEY_CHECK_CU(cuMemcpyDtoH(host_confidence.data(), d_out_confidence_, host_confidence.size() * sizeof(float)))) {
            if (d_debug_masks) JERSEY_CHECK_CU(cuMemFree(d_debug_masks));
            this->sink_->put(frm);
            return;
        }
        std::vector<float> host_uv_hist(packed.size() * kUVHistSize, 0.0f);
        std::vector<float> host_l_hist(packed.size() * kLHistSize, 0.0f);
        if (JERSEY_CHECK_CU(cuMemcpyDtoH(host_uv_hist.data(), d_out_uv_hist_, host_uv_hist.size() * sizeof(float)))) {
            if (d_debug_masks) JERSEY_CHECK_CU(cuMemFree(d_debug_masks));
            this->sink_->put(frm);
            return;
        }
        if (JERSEY_CHECK_CU(cuMemcpyDtoH(host_l_hist.data(), d_out_l_hist_, host_l_hist.size() * sizeof(float)))) {
            if (d_debug_masks) JERSEY_CHECK_CU(cuMemFree(d_debug_masks));
            this->sink_->put(frm);
            return;
        }

        int with_color = 0;
        Parameters debug_md;
        if (emit_debug_masks) {
            debug_md["detections"] = Parameters::array();
        }
        for (size_t i = 0; i < packed.size(); ++i) {
            auto& det = md["detections"][packed[i].array_index];
            det["jersey_pixels"] = host_cloth_count[i];
            det["jersey_mode_pixels"] = host_best_count[i];
            det["jersey_cloth_pixels"] = host_cloth_count[i];
            det["jersey_skin_pixels"] = host_skin_count[i];
            det["jersey_confidence"] = host_confidence[i];
            det["jersey_mode_ratio"] = host_cloth_count[i] > 0
                                           ? (double)host_best_count[i] / (double)host_cloth_count[i]
                                           : 0.0;
            if (host_cloth_count[i] >= min_pixels_ && host_best_count[i] > 0) {
                Parameters uv_hist_arr = Parameters::array();
                for (int b = 0; b < kUVHistSize; ++b) {
                    uv_hist_arr.push_back(host_uv_hist[i * kUVHistSize + b]);
                }
                det["jersey_uv_hist"] = uv_hist_arr;

                Parameters l_hist_arr = Parameters::array();
                for (int b = 0; b < kLHistSize; ++b) {
                    l_hist_arr.push_back(host_l_hist[i * kLHistSize + b]);
                }
                det["jersey_l_hist"] = l_hist_arr;
                ++with_color;
            }
            if (emit_debug_masks) {
                Parameters debug_det = Parameters::object();
                debug_det["xyxy"] = det["xyxy"];
                debug_det["cls"] = 0;
                debug_det["label"] = "jersey_sample";
                debug_det["source_det_index"] = packed[i].array_index;
                debug_md["detections"].push_back(debug_det);
            }
        }

        const std::string serialized = md.dump();
        av_dict_set(&frm.raw()->metadata, metadata_key_.c_str(), serialized.c_str(), 0);
        if (emit_debug_masks) {
            auto* header_out = (GpuMaskSideDataHeader*)av_malloc(sizeof(GpuMaskSideDataHeader));
            auto* release = (GpuMaskRelease*)av_malloc(sizeof(GpuMaskRelease));
            bool attached = false;
            if (header_out && release) {
                header_out->gpu_ptr = (uint64_t)(uintptr_t)d_debug_masks;
                header_out->num_masks = (uint32_t)packed.size();
                header_out->proto_w = (uint32_t)proto_w;
                header_out->proto_h = (uint32_t)proto_h;
                header_out->model_w = (uint32_t)model_w;
                header_out->model_h = (uint32_t)model_h;
                release->ctx = cu_ctx_;
                release->ptr = d_debug_masks;
                AVBufferRef* sd_buf = av_buffer_create(
                    (uint8_t*)header_out, sizeof(GpuMaskSideDataHeader),
                    freeGpuMaskSideData, release, 0);
                if (sd_buf) {
                    av_frame_new_side_data_from_buf(frm.raw(), yoloSegGpuSideDataType(debug_output_side_data_slot_), sd_buf);
                    const std::string debug_serialized = debug_md.dump();
                    av_dict_set(&frm.raw()->metadata, debug_output_metadata_key_.c_str(), debug_serialized.c_str(), 0);
                    d_debug_masks = 0;
                    attached = true;
                }
            }
            if (!attached) {
                if (header_out) av_free(header_out);
                if (release) av_free(release);
            }
            if (d_debug_masks) {
                JERSEY_CHECK_CU(cuMemFree(d_debug_masks));
                d_debug_masks = 0;
            }
        }

        if (debug_log_every_n_ > 0 && (frame_counter_ % (uint64_t)debug_log_every_n_) == 0) {
            double sum_conf_pct = 0.0;
            double sum_mode_pct = 0.0;
            int conf_count = 0;
            std::ostringstream sample_stream;
            const size_t sample_count = std::min<size_t>(packed.size(), 4);
            for (size_t i = 0; i < packed.size(); ++i) {
                const double conf_pct = (double)host_confidence[i] * 100.0;
                const double mode_pct = host_cloth_count[i] > 0
                                            ? (100.0 * (double)host_best_count[i] / (double)host_cloth_count[i])
                                            : 0.0;
                if (host_cloth_count[i] > 0) {
                    sum_conf_pct += conf_pct;
                    sum_mode_pct += mode_pct;
                    ++conf_count;
                }
                if (i < sample_count) {
                    if (i > 0) sample_stream << " ";
                    sample_stream << "[det=" << packed[i].array_index
                                  << " conf_pct=" << (int)std::lround(conf_pct)
                                  << " mode_pct=" << (int)std::lround(mode_pct)
                                  << " cloth=" << host_cloth_count[i]
                                  << " skin=" << host_skin_count[i]
                                  << " valid=" << ((host_cloth_count[i] >= min_pixels_ && host_best_count[i] > 0) ? 1 : 0)
                                  << "]";
                }
            }
            logstream << "jersey_color_extract: frame=" << frame_counter_
                      << " packed=" << packed.size()
                      << " valid=" << with_color
                      << " avg_conf_pct=" << (conf_count > 0 ? (sum_conf_pct / (double)conf_count) : 0.0)
                      << " avg_mode_pct=" << (conf_count > 0 ? (sum_mode_pct / (double)conf_count) : 0.0)
                      << " proto=" << proto_w << "x" << proto_h
                      << " torso_rel=[" << torso_y_start_rel_ << "," << torso_y_end_rel_ << "]"
                      << " sample_rel=[x:" << sample_inner_x_margin_rel_
                      << ",y:" << sample_top_y_exclusion_rel_ << "]"
                      << " skin_u=[" << skin_u_min_ << "," << skin_u_max_ << "]"
                      << " skin_v=[" << skin_v_min_ << "," << skin_v_max_ << "]"
                      << " samples=" << sample_stream.str();
        }

        this->sink_->put(frm);
    }

    static std::shared_ptr<JerseyColorExtract> create(NodeCreationInfo& nci) {
        EdgeManager& edges = nci.edges;
        const Parameters& params = nci.params;
        auto r = NodeSISO<av::VideoFrame, av::VideoFrame>::template createCommon<JerseyColorExtract>(edges, params);

        if (params.count("metadata_key")) r->metadata_key_ = params["metadata_key"].get<std::string>();
        if (params.count("target_labels")) {
            r->target_labels_.clear();
            for (const auto& item : params["target_labels"]) {
                r->target_labels_.push_back(item.get<std::string>());
            }
        }
        if (params.count("target_class")) r->target_class_ = params["target_class"].get<int>();
        if (params.count("label_case_sensitive")) r->label_case_sensitive_ = params["label_case_sensitive"].get<bool>();
        if (params.count("mask_threshold")) r->mask_threshold_ = params["mask_threshold"].get<float>();
        if (params.count("min_pixels")) r->min_pixels_ = params["min_pixels"].get<int>();
        if (params.count("debug_log_every_n")) r->debug_log_every_n_ = params["debug_log_every_n"].get<int>();
        if (params.count("initial_capacity")) r->initial_capacity_ = params["initial_capacity"].get<int>();
        if (params.count("side_data_slot")) {
            r->side_data_slot_ = params["side_data_slot"].get<int>();
            if (!yoloSegIsValidSlot(r->side_data_slot_)) {
                throw Error("jersey_color_extract: side_data_slot out of range [0," + std::to_string(kMaxYoloSegSlots - 1) + "]");
            }
        }
        if (params.count("body_region")) {
            const std::string body = params["body_region"].get<std::string>();
            r->body_region_mode_ = iequals(body, "torso") ? 1 : 0;
        }
        if (params.count("torso_x_margin_rel")) r->torso_x_margin_rel_ = params["torso_x_margin_rel"].get<float>();
        if (params.count("torso_y_start_rel")) r->torso_y_start_rel_ = params["torso_y_start_rel"].get<float>();
        if (params.count("torso_y_end_rel")) r->torso_y_end_rel_ = params["torso_y_end_rel"].get<float>();
        if (params.count("sample_inner_x_margin_rel")) r->sample_inner_x_margin_rel_ = params["sample_inner_x_margin_rel"].get<float>();
        if (params.count("sample_top_y_exclusion_rel")) r->sample_top_y_exclusion_rel_ = params["sample_top_y_exclusion_rel"].get<float>();
        if (params.count("skin_y_min")) r->skin_y_min_ = params["skin_y_min"].get<int>();
        if (params.count("skin_y_max")) r->skin_y_max_ = params["skin_y_max"].get<int>();
        if (params.count("skin_u_min")) r->skin_u_min_ = params["skin_u_min"].get<int>();
        if (params.count("skin_u_max")) r->skin_u_max_ = params["skin_u_max"].get<int>();
        if (params.count("skin_v_min")) r->skin_v_min_ = params["skin_v_min"].get<int>();
        if (params.count("skin_v_max")) r->skin_v_max_ = params["skin_v_max"].get<int>();
        if (params.count("skin_neutral_y_min")) r->skin_neutral_y_min_ = params["skin_neutral_y_min"].get<int>();
        if (params.count("skin_neutral_u_tol")) r->skin_neutral_u_tol_ = params["skin_neutral_u_tol"].get<int>();
        if (params.count("skin_neutral_v_tol")) r->skin_neutral_v_tol_ = params["skin_neutral_v_tol"].get<int>();
        if (params.count("debug_output_metadata_key")) r->debug_output_metadata_key_ = params["debug_output_metadata_key"].get<std::string>();
        if (params.count("debug_output_side_data_slot")) {
            r->debug_output_side_data_slot_ = params["debug_output_side_data_slot"].get<int>();
            if (r->debug_output_side_data_slot_ >= 0 && !yoloSegIsValidSlot(r->debug_output_side_data_slot_)) {
                throw Error("jersey_color_extract: debug_output_side_data_slot out of range [0," + std::to_string(kMaxYoloSegSlots - 1) + "]");
            }
        }
        return r;
    }
};

DECLNODE(jersey_color_extract, JerseyColorExtract)
