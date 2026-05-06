#include "../../node_common.hpp"
#include "../common/yolo_side_data.hpp"
#include <cuda_loader/cuda_drvapi_dynlink_cuda.h>

extern "C" {
#include <libavutil/buffer.h>
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

#include "../../../../objs/src/nodes/neural_net/sport_specific/player_torso_seg.ptx.h"

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

#define TORSO_SEG_CHECK_CU(x) checkCu((x), #x)

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

} // namespace

class PlayerTorsoSeg : public NodeSISO<av::VideoFrame, av::VideoFrame>, public ReportsFinishByFlag {
    std::string metadata_key_ = "yolo_players_seg";
    std::string output_metadata_key_ = "yolo_players_torso_seg";
    std::vector<std::string> target_labels_ = {"player"};
    int target_class_ = -1;
    bool label_case_sensitive_ = false;
    int input_side_data_slot_ = 1;
    int output_side_data_slot_ = 2;
    float mask_threshold_ = 0.5f;
    float torso_x_margin_rel_ = 0.10f;
    float torso_y_start_rel_ = 0.16f;
    float torso_y_end_rel_ = 0.60f;
    float sample_inner_x_margin_rel_ = 0.18f;
    float sample_top_y_exclusion_rel_ = 0.12f;
    bool skin_filter_ = true;
    int skin_y_min_ = 45;
    int skin_y_max_ = 235;
    int skin_u_min_ = 80;
    int skin_u_max_ = 132;
    int skin_v_min_ = 126;
    int skin_v_max_ = 182;
    int skin_neutral_y_min_ = 0;
    int skin_neutral_u_tol_ = 18;
    int skin_neutral_v_tol_ = 18;
    int initial_capacity_ = 32;
    int debug_log_every_n_ = 0;

    AVCUDADeviceContext* cuda_dev_ctx_ = nullptr;
    CUcontext cu_ctx_ = nullptr;
    CUmodule cu_module_ = nullptr;
    CUfunction kernel_ = nullptr;
    CUdeviceptr d_bboxes_ = 0;
    CUdeviceptr d_plane_indices_ = 0;
    int capacity_ = 0;
    uint64_t frame_counter_ = 0;

    bool matchesTarget(const Parameters& det) const {
        if (target_class_ >= 0 && det.contains("cls")) {
            if (det["cls"].get<int>() == target_class_) return true;
        }
        if (!target_labels_.empty() && det.contains("label")) {
            const std::string lbl = det["label"].get<std::string>();
            for (const auto& t : target_labels_) {
                if (label_case_sensitive_ ? (lbl == t) : iequals(lbl, t)) return true;
            }
        }
        return target_class_ < 0 && target_labels_.empty();
    }

    void releaseBuffers() {
        if (!cu_ctx_) return;
        TORSO_SEG_CHECK_CU(cuCtxSetCurrent(cu_ctx_));
        if (d_bboxes_) { TORSO_SEG_CHECK_CU(cuMemFree(d_bboxes_)); d_bboxes_ = 0; }
        if (d_plane_indices_) { TORSO_SEG_CHECK_CU(cuMemFree(d_plane_indices_)); d_plane_indices_ = 0; }
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
        return TORSO_SEG_CHECK_CU(cuCtxSetCurrent(cu_ctx_)) == 0;
    }

    bool loadKernel() {
        if (cu_module_ && kernel_) return true;
        if (!cu_ctx_) return false;
        if (TORSO_SEG_CHECK_CU(cuCtxSetCurrent(cu_ctx_))) return false;
        const std::string ptx_str(avpl_player_torso_seg_ptx,
                                  avpl_player_torso_seg_ptx + avpl_player_torso_seg_ptx_len);
        if (TORSO_SEG_CHECK_CU(cuModuleLoadDataEx(&cu_module_, ptx_str.c_str(), 0, nullptr, nullptr))) return false;
        if (TORSO_SEG_CHECK_CU(cuModuleGetFunction(&kernel_, cu_module_, "kPlayerTorsoSegMask"))) return false;
        return true;
    }

    bool ensureCapacity(int needed) {
        if (needed <= capacity_) return true;
        int next = std::max(initial_capacity_, 1);
        while (next < needed) next <<= 1;
        releaseBuffers();
        if (TORSO_SEG_CHECK_CU(cuCtxSetCurrent(cu_ctx_))) return false;
        if (TORSO_SEG_CHECK_CU(cuMemAlloc(&d_bboxes_, (size_t)next * 4u * sizeof(int)))) return false;
        if (TORSO_SEG_CHECK_CU(cuMemAlloc(&d_plane_indices_, (size_t)next * sizeof(int)))) return false;
        capacity_ = next;
        return true;
    }

    bool packDetections(const Parameters& md, std::vector<PackedDetection>& packed, Parameters& out_md) const {
        packed.clear();
        out_md = Parameters::object();
        for (auto it = md.begin(); it != md.end(); ++it) {
            if (it.key() != "detections") {
                out_md[it.key()] = it.value();
            }
        }
        out_md["detections"] = Parameters::array();
        if (!md.contains("detections") || !md["detections"].is_array()) return false;

        int plane_index = 0;
        for (int i = 0; i < (int)md["detections"].size(); ++i) {
            const auto& det = md["detections"][i];
            if (!det.is_object()) continue;

            const bool has_bbox = det.contains("xyxy") && det["xyxy"].is_array() && det["xyxy"].size() >= 4;
            if (!has_bbox) {
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
                Parameters out_det = det;
                out_det["cls"] = 0;
                out_det["label"] = "torso";
                out_det["source_det_index"] = item.array_index;
                out_det["source_plane_index"] = item.plane_index;
                out_md["detections"].push_back(out_det);
            }
            ++plane_index;
        }
        return !packed.empty();
    }

    void attachEmptyMetadata(av::VideoFrame& frm) {
        Parameters out_md = Parameters::object();
        out_md["detections"] = Parameters::array();
        av_dict_set(&frm.raw()->metadata, output_metadata_key_.c_str(), out_md.dump().c_str(), 0);
    }

public:
    using NodeSISO<av::VideoFrame, av::VideoFrame>::NodeSISO;
    bool consumeEofIfPresent() override {
        return false;
    }

    ~PlayerTorsoSeg() {
        if (cu_ctx_) {
            TORSO_SEG_CHECK_CU(cuCtxSetCurrent(cu_ctx_));
        }
        releaseBuffers();
        if (cu_module_) {
            TORSO_SEG_CHECK_CU(cuModuleUnload(cu_module_));
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
        AVFrame* raw = frm.raw();
        if (!raw || !raw->metadata) {
            this->sink_->put(frm);
            return;
        }

        AVDictionaryEntry* entry = av_dict_get(raw->metadata, metadata_key_.c_str(), nullptr, 0);
        if (!entry || !entry->value) {
            attachEmptyMetadata(frm);
            this->sink_->put(frm);
            return;
        }

        const AVFrameSideData* sd = av_frame_get_side_data(raw, yoloSegGpuSideDataType(input_side_data_slot_));
        if (!sd || !sd->buf || sd->buf->size < (int)sizeof(GpuMaskSideDataHeader)) {
            attachEmptyMetadata(frm);
            this->sink_->put(frm);
            return;
        }

        Parameters md;
        try {
            md = Parameters::parse(entry->value);
        } catch (...) {
            attachEmptyMetadata(frm);
            this->sink_->put(frm);
            return;
        }

        const auto* header = (const GpuMaskSideDataHeader*)sd->buf->data;
        const CUdeviceptr gpu_masks = (CUdeviceptr)header->gpu_ptr;
        if (!gpu_masks || header->num_masks == 0 || header->proto_w == 0 || header->proto_h == 0) {
            attachEmptyMetadata(frm);
            this->sink_->put(frm);
            return;
        }

        std::vector<PackedDetection> packed;
        Parameters out_md;
        if (!packDetections(md, packed, out_md)) {
            attachEmptyMetadata(frm);
            this->sink_->put(frm);
            return;
        }

        size_t invalid_plane_count = 0;
        for (const auto& p : packed) {
            if (p.plane_index < 0 || p.plane_index >= (int)header->num_masks) ++invalid_plane_count;
        }
        if (invalid_plane_count > 0) {
            std::ostringstream err;
            err << "player_torso_seg: mask index out of range"
                << " frame=" << frame_counter_
                << " input_slot=" << input_side_data_slot_
                << " invalid=" << invalid_plane_count
                << " packed=" << packed.size()
                << " num_masks=" << header->num_masks;
            throw Error(err.str());
        }

        if (!initCudaContextFromFrame(frm) || !loadKernel() || !ensureCapacity((int)packed.size())) {
            this->sink_->put(frm);
            return;
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

        if (TORSO_SEG_CHECK_CU(cuMemcpyHtoDAsync(d_bboxes_, host_bboxes.data(), host_bboxes.size() * sizeof(int), cuda_dev_ctx_->stream)) ||
            TORSO_SEG_CHECK_CU(cuMemcpyHtoDAsync(d_plane_indices_, host_plane_indices.data(), host_plane_indices.size() * sizeof(int), cuda_dev_ctx_->stream))) {
            this->sink_->put(frm);
            return;
        }

        const int proto_w = (int)header->proto_w;
        const int proto_h = (int)header->proto_h;
        const int model_w = (int)header->model_w;
        const int model_h = (int)header->model_h;
        const size_t mask_bytes = packed.size() * (size_t)proto_w * (size_t)proto_h * sizeof(float);
        CUdeviceptr out_masks = 0;
        if (TORSO_SEG_CHECK_CU(cuMemAlloc(&out_masks, mask_bytes))) {
            this->sink_->put(frm);
            return;
        }

        const CUdeviceptr y_plane = (CUdeviceptr)(uintptr_t)raw->data[0];
        const CUdeviceptr uv_plane = (CUdeviceptr)(uintptr_t)raw->data[1];
        const int y_pitch = raw->linesize[0];
        const int uv_pitch = raw->linesize[1];
        const int frame_w = frm.width();
        const int frame_h = frm.height();
        const int num_dets = (int)packed.size();
        const int skin_filter_enabled = (skin_filter_ && y_plane && uv_plane) ? 1 : 0;

        const unsigned int block_x = 16;
        const unsigned int block_y = 16;
        const unsigned int grid_x = ((unsigned int)proto_w + block_x - 1) / block_x;
        const unsigned int grid_y = ((unsigned int)proto_h + block_y - 1) / block_y;
        void* args[] = {
            (void*)&y_plane, (void*)&y_pitch,
            (void*)&uv_plane, (void*)&uv_pitch,
            (void*)&frame_w, (void*)&frame_h,
            (void*)&gpu_masks,
            (void*)&out_masks,
            (void*)&proto_w, (void*)&proto_h,
            (void*)&model_w, (void*)&model_h,
            (void*)&d_bboxes_,
            (void*)&d_plane_indices_,
            (void*)&num_dets,
            (void*)&mask_threshold_,
            (void*)&torso_x_margin_rel_,
            (void*)&torso_y_start_rel_,
            (void*)&torso_y_end_rel_,
            (void*)&sample_inner_x_margin_rel_,
            (void*)&sample_top_y_exclusion_rel_,
            (void*)&skin_filter_enabled,
            (void*)&skin_y_min_,
            (void*)&skin_y_max_,
            (void*)&skin_u_min_,
            (void*)&skin_u_max_,
            (void*)&skin_v_min_,
            (void*)&skin_v_max_,
            (void*)&skin_neutral_y_min_,
            (void*)&skin_neutral_u_tol_,
            (void*)&skin_neutral_v_tol_
        };

        if (TORSO_SEG_CHECK_CU(cuLaunchKernel(kernel_,
                                              grid_x, grid_y, (unsigned int)num_dets,
                                              block_x, block_y, 1,
                                              0, cuda_dev_ctx_->stream, args, nullptr)) ||
            TORSO_SEG_CHECK_CU(cuStreamSynchronize(cuda_dev_ctx_->stream))) {
            TORSO_SEG_CHECK_CU(cuMemFree(out_masks));
            this->sink_->put(frm);
            return;
        }

        auto* header_out = (GpuMaskSideDataHeader*)av_malloc(sizeof(GpuMaskSideDataHeader));
        auto* release = (GpuMaskRelease*)av_malloc(sizeof(GpuMaskRelease));
        if (!header_out || !release) {
            if (header_out) av_free(header_out);
            if (release) av_free(release);
            TORSO_SEG_CHECK_CU(cuMemFree(out_masks));
            this->sink_->put(frm);
            return;
        }

        header_out->gpu_ptr = (uint64_t)(uintptr_t)out_masks;
        header_out->num_masks = (uint32_t)num_dets;
        header_out->proto_w = (uint32_t)proto_w;
        header_out->proto_h = (uint32_t)proto_h;
        header_out->model_w = (uint32_t)model_w;
        header_out->model_h = (uint32_t)model_h;
        release->ctx = cu_ctx_;
        release->ptr = out_masks;

        AVBufferRef* sd_buf = av_buffer_create((uint8_t*)header_out, sizeof(GpuMaskSideDataHeader),
                                               freeGpuMaskSideData, release, 0);
        if (!sd_buf) {
            av_free(header_out);
            av_free(release);
            TORSO_SEG_CHECK_CU(cuMemFree(out_masks));
            this->sink_->put(frm);
            return;
        }

        av_frame_remove_side_data(raw, yoloSegGpuSideDataType(output_side_data_slot_));
        AVFrameSideData* attached = av_frame_new_side_data_from_buf(raw, yoloSegGpuSideDataType(output_side_data_slot_), sd_buf);
        if (!attached) {
            av_buffer_unref(&sd_buf);
            this->sink_->put(frm);
            return;
        }

        av_dict_set(&raw->metadata, output_metadata_key_.c_str(), out_md.dump().c_str(), 0);

        if (debug_log_every_n_ > 0 && (frame_counter_ % (uint64_t)debug_log_every_n_) == 0) {
            logstream << "player_torso_seg: frame=" << frame_counter_
                      << " masks=" << num_dets
                      << " input_slot=" << input_side_data_slot_
                      << " output_slot=" << output_side_data_slot_
                      << " proto=" << proto_w << "x" << proto_h
                      << " skin_filter=" << skin_filter_enabled;
        }

        this->sink_->put(frm);
    }

    static std::shared_ptr<PlayerTorsoSeg> create(NodeCreationInfo& nci) {
        EdgeManager& edges = nci.edges;
        const Parameters& params = nci.params;
        auto r = NodeSISO<av::VideoFrame, av::VideoFrame>::template createCommon<PlayerTorsoSeg>(edges, params);

        if (params.count("metadata_key")) r->metadata_key_ = params["metadata_key"].get<std::string>();
        if (params.count("output_metadata_key")) r->output_metadata_key_ = params["output_metadata_key"].get<std::string>();
        if (params.count("target_labels")) {
            r->target_labels_.clear();
            for (const auto& item : params["target_labels"]) r->target_labels_.push_back(item.get<std::string>());
        }
        if (params.count("target_class")) r->target_class_ = params["target_class"].get<int>();
        if (params.count("label_case_sensitive")) r->label_case_sensitive_ = params["label_case_sensitive"].get<bool>();
        if (params.count("input_side_data_slot")) r->input_side_data_slot_ = params["input_side_data_slot"].get<int>();
        if (params.count("output_side_data_slot")) r->output_side_data_slot_ = params["output_side_data_slot"].get<int>();
        if (!yoloSegIsValidSlot(r->input_side_data_slot_) || !yoloSegIsValidSlot(r->output_side_data_slot_)) {
            throw Error("player_torso_seg: side-data slot out of range [0," + std::to_string(kMaxYoloSegSlots - 1) + "]");
        }
        if (r->input_side_data_slot_ == r->output_side_data_slot_) {
            throw Error("player_torso_seg: input_side_data_slot and output_side_data_slot must be different");
        }
        if (params.count("mask_threshold")) r->mask_threshold_ = params["mask_threshold"].get<float>();
        if (params.count("torso_x_margin_rel")) r->torso_x_margin_rel_ = params["torso_x_margin_rel"].get<float>();
        if (params.count("torso_y_start_rel")) r->torso_y_start_rel_ = params["torso_y_start_rel"].get<float>();
        if (params.count("torso_y_end_rel")) r->torso_y_end_rel_ = params["torso_y_end_rel"].get<float>();
        if (params.count("sample_inner_x_margin_rel")) r->sample_inner_x_margin_rel_ = params["sample_inner_x_margin_rel"].get<float>();
        if (params.count("sample_top_y_exclusion_rel")) r->sample_top_y_exclusion_rel_ = params["sample_top_y_exclusion_rel"].get<float>();
        if (params.count("skin_filter")) r->skin_filter_ = params["skin_filter"].get<bool>();
        if (params.count("skin_y_min")) r->skin_y_min_ = params["skin_y_min"].get<int>();
        if (params.count("skin_y_max")) r->skin_y_max_ = params["skin_y_max"].get<int>();
        if (params.count("skin_u_min")) r->skin_u_min_ = params["skin_u_min"].get<int>();
        if (params.count("skin_u_max")) r->skin_u_max_ = params["skin_u_max"].get<int>();
        if (params.count("skin_v_min")) r->skin_v_min_ = params["skin_v_min"].get<int>();
        if (params.count("skin_v_max")) r->skin_v_max_ = params["skin_v_max"].get<int>();
        if (params.count("skin_neutral_y_min")) r->skin_neutral_y_min_ = params["skin_neutral_y_min"].get<int>();
        if (params.count("skin_neutral_u_tol")) r->skin_neutral_u_tol_ = params["skin_neutral_u_tol"].get<int>();
        if (params.count("skin_neutral_v_tol")) r->skin_neutral_v_tol_ = params["skin_neutral_v_tol"].get<int>();
        if (params.count("initial_capacity")) r->initial_capacity_ = params["initial_capacity"].get<int>();
        if (params.count("debug_log_every_n")) r->debug_log_every_n_ = params["debug_log_every_n"].get<int>();
        return r;
    }
};

DECLNODE(player_torso_seg, PlayerTorsoSeg)
