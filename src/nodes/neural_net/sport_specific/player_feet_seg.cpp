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
#include <climits>
#include <sstream>
#include <string>
#include <vector>

#include "../../../../objs/src/nodes/neural_net/sport_specific/player_feet_seg.ptx.h"

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

#define FEET_SEG_CHECK_CU(x) checkCu((x), #x)

struct PackedDetection {
    int array_index = -1;
    int plane_index = -1;
    int x1 = 0;
    int y1 = 0;
    int x2 = 0;
    int y2 = 0;
    int track_id = -1;
    std::string team_ab;
};

struct PlayerDet {
    float x1 = 0.0f;
    float y1 = 0.0f;
    float x2 = 0.0f;
    float y2 = 0.0f;
    int track_id = -1;
    std::string team_ab;
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

class PlayerFeetSeg : public NodeSISO<av::VideoFrame, av::VideoFrame>, public ReportsFinishByFlag {
    std::string metadata_key_ = "yolo_players_seg";
    std::string player_metadata_key_ = "yolo_players";
    std::string output_metadata_key_ = "player_feet";
    std::vector<std::string> target_labels_ = {"player"};
    int target_class_ = -1;
    bool label_case_sensitive_ = false;
    int input_side_data_slot_ = 1;
    int output_side_data_slot_ = 3;
    float mask_threshold_ = 0.5f;
    float foot_x_margin_rel_ = 0.02f;
    float foot_y_start_rel_ = 0.70f;
    float foot_y_end_margin_rel_ = 0.00f;
    int min_pixels_ = 10;
    float player_iou_threshold_ = 0.10f;
    float fallback_center_distance_px_ = 60.0f;
    int initial_capacity_ = 32;
    int debug_log_every_n_ = 0;

    AVCUDADeviceContext* cuda_dev_ctx_ = nullptr;
    CUcontext cu_ctx_ = nullptr;
    CUmodule cu_module_ = nullptr;
    CUfunction kernel_ = nullptr;
    CUdeviceptr d_bboxes_ = 0;
    CUdeviceptr d_plane_indices_ = 0;
    CUdeviceptr d_count_ = 0;
    CUdeviceptr d_sum_x100_ = 0;
    CUdeviceptr d_sum_y100_ = 0;
    CUdeviceptr d_min_x100_ = 0;
    CUdeviceptr d_max_x100_ = 0;
    CUdeviceptr d_min_y100_ = 0;
    CUdeviceptr d_max_y100_ = 0;
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

    static double round3(double v) {
        return std::round(v * 1000.0) / 1000.0;
    }

    static double clamp01(double v) {
        return std::max(0.0, std::min(1.0, v));
    }

    static float boxIou(float ax1, float ay1, float ax2, float ay2,
                        float bx1, float by1, float bx2, float by2) {
        const float ix1 = std::max(ax1, bx1);
        const float iy1 = std::max(ay1, by1);
        const float ix2 = std::min(ax2, bx2);
        const float iy2 = std::min(ay2, by2);
        const float iw = std::max(0.0f, ix2 - ix1);
        const float ih = std::max(0.0f, iy2 - iy1);
        const float inter = iw * ih;
        const float area_a = std::max(0.0f, ax2 - ax1) * std::max(0.0f, ay2 - ay1);
        const float area_b = std::max(0.0f, bx2 - bx1) * std::max(0.0f, by2 - by1);
        const float denom = area_a + area_b - inter;
        return denom > 0.0f ? inter / denom : 0.0f;
    }

    static float centerDistance(float ax1, float ay1, float ax2, float ay2,
                                float bx1, float by1, float bx2, float by2) {
        const float acx = (ax1 + ax2) * 0.5f;
        const float acy = (ay1 + ay2) * 0.5f;
        const float bcx = (bx1 + bx2) * 0.5f;
        const float bcy = (by1 + by2) * 0.5f;
        const float dx = acx - bcx;
        const float dy = acy - bcy;
        return std::sqrt(dx * dx + dy * dy);
    }

    void releaseBuffers() {
        if (!cu_ctx_) return;
        FEET_SEG_CHECK_CU(cuCtxSetCurrent(cu_ctx_));
        if (d_bboxes_) { FEET_SEG_CHECK_CU(cuMemFree(d_bboxes_)); d_bboxes_ = 0; }
        if (d_plane_indices_) { FEET_SEG_CHECK_CU(cuMemFree(d_plane_indices_)); d_plane_indices_ = 0; }
        if (d_count_) { FEET_SEG_CHECK_CU(cuMemFree(d_count_)); d_count_ = 0; }
        if (d_sum_x100_) { FEET_SEG_CHECK_CU(cuMemFree(d_sum_x100_)); d_sum_x100_ = 0; }
        if (d_sum_y100_) { FEET_SEG_CHECK_CU(cuMemFree(d_sum_y100_)); d_sum_y100_ = 0; }
        if (d_min_x100_) { FEET_SEG_CHECK_CU(cuMemFree(d_min_x100_)); d_min_x100_ = 0; }
        if (d_max_x100_) { FEET_SEG_CHECK_CU(cuMemFree(d_max_x100_)); d_max_x100_ = 0; }
        if (d_min_y100_) { FEET_SEG_CHECK_CU(cuMemFree(d_min_y100_)); d_min_y100_ = 0; }
        if (d_max_y100_) { FEET_SEG_CHECK_CU(cuMemFree(d_max_y100_)); d_max_y100_ = 0; }
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
        return FEET_SEG_CHECK_CU(cuCtxSetCurrent(cu_ctx_)) == 0;
    }

    bool loadKernel() {
        if (cu_module_ && kernel_) return true;
        if (!cu_ctx_) return false;
        if (FEET_SEG_CHECK_CU(cuCtxSetCurrent(cu_ctx_))) return false;
        const std::string ptx_str(avpl_player_feet_seg_ptx,
                                  avpl_player_feet_seg_ptx + avpl_player_feet_seg_ptx_len);
        if (FEET_SEG_CHECK_CU(cuModuleLoadDataEx(&cu_module_, ptx_str.c_str(), 0, nullptr, nullptr))) return false;
        if (FEET_SEG_CHECK_CU(cuModuleGetFunction(&kernel_, cu_module_, "kPlayerFeetSegMaskAndStats"))) return false;
        return true;
    }

    bool ensureCapacity(int needed) {
        if (needed <= capacity_) return true;
        int next = std::max(initial_capacity_, 1);
        while (next < needed) next <<= 1;
        releaseBuffers();
        if (FEET_SEG_CHECK_CU(cuCtxSetCurrent(cu_ctx_))) return false;
        if (FEET_SEG_CHECK_CU(cuMemAlloc(&d_bboxes_, (size_t)next * 4u * sizeof(int)))) return false;
        if (FEET_SEG_CHECK_CU(cuMemAlloc(&d_plane_indices_, (size_t)next * sizeof(int)))) return false;
        if (FEET_SEG_CHECK_CU(cuMemAlloc(&d_count_, (size_t)next * sizeof(int)))) return false;
        if (FEET_SEG_CHECK_CU(cuMemAlloc(&d_sum_x100_, (size_t)next * sizeof(int)))) return false;
        if (FEET_SEG_CHECK_CU(cuMemAlloc(&d_sum_y100_, (size_t)next * sizeof(int)))) return false;
        if (FEET_SEG_CHECK_CU(cuMemAlloc(&d_min_x100_, (size_t)next * sizeof(int)))) return false;
        if (FEET_SEG_CHECK_CU(cuMemAlloc(&d_max_x100_, (size_t)next * sizeof(int)))) return false;
        if (FEET_SEG_CHECK_CU(cuMemAlloc(&d_min_y100_, (size_t)next * sizeof(int)))) return false;
        if (FEET_SEG_CHECK_CU(cuMemAlloc(&d_max_y100_, (size_t)next * sizeof(int)))) return false;
        capacity_ = next;
        return true;
    }

    void attachEmptyMetadata(av::VideoFrame& frm) {
        Parameters out_md = Parameters::object();
        out_md["schema"] = "player_feet_seg_v1";
        out_md["coord_space"] = "model";
        out_md["detections"] = Parameters::array();
        av_dict_set(&frm.raw()->metadata, output_metadata_key_.c_str(), out_md.dump().c_str(), 0);
    }

    static bool readMetadata(const AVFrame* raw, const std::string& key, Parameters& out) {
        if (!raw || !raw->metadata) return false;
        AVDictionaryEntry* entry = av_dict_get(raw->metadata, key.c_str(), nullptr, 0);
        if (!entry || !entry->value) return false;
        try {
            out = Parameters::parse(entry->value);
            return out.is_object() && !out.is_null();
        } catch (...) {
            return false;
        }
    }

    void parsePlayers(const Parameters& md, std::vector<PlayerDet>& players) const {
        players.clear();
        if (!md.contains("detections") || !md["detections"].is_array()) return;
        for (const auto& det : md["detections"]) {
            if (!det.is_object()) continue;
            if (det.value("label", std::string()) != "Player") continue;
            if (!det.contains("xyxy") || !det["xyxy"].is_array() || det["xyxy"].size() < 4) continue;
            PlayerDet p;
            p.x1 = det["xyxy"][0].get<float>();
            p.y1 = det["xyxy"][1].get<float>();
            p.x2 = det["xyxy"][2].get<float>();
            p.y2 = det["xyxy"][3].get<float>();
            p.track_id = det.value("track_id", -1);
            p.team_ab = det.value("team_ab", std::string());
            players.push_back(p);
        }
    }

    void attachTrackIdentity(PackedDetection& item, const std::vector<PlayerDet>& players) const {
        if (players.empty()) return;

        int best_iou_idx = -1;
        float best_iou = player_iou_threshold_;
        for (int i = 0; i < (int)players.size(); ++i) {
            const auto& p = players[(size_t)i];
            float iou = boxIou((float)item.x1, (float)item.y1, (float)item.x2, (float)item.y2,
                               p.x1, p.y1, p.x2, p.y2);
            if (iou > best_iou) {
                best_iou = iou;
                best_iou_idx = i;
            }
        }

        int best_idx = best_iou_idx;
        if (best_idx < 0) {
            float best_dist = fallback_center_distance_px_;
            for (int i = 0; i < (int)players.size(); ++i) {
                const auto& p = players[(size_t)i];
                float dist = centerDistance((float)item.x1, (float)item.y1, (float)item.x2, (float)item.y2,
                                            p.x1, p.y1, p.x2, p.y2);
                if (dist < best_dist) {
                    best_dist = dist;
                    best_idx = i;
                }
            }
        }

        if (best_idx >= 0) {
            const auto& p = players[(size_t)best_idx];
            item.track_id = p.track_id;
            item.team_ab = p.team_ab;
        }
    }

    bool packDetections(const Parameters& seg_md, const std::vector<PlayerDet>& players,
                        std::vector<PackedDetection>& packed) const {
        packed.clear();
        if (!seg_md.contains("detections") || !seg_md["detections"].is_array()) return false;

        int plane_index = 0;
        for (int i = 0; i < (int)seg_md["detections"].size(); ++i) {
            const auto& det = seg_md["detections"][i];
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
                attachTrackIdentity(item, players);
                packed.push_back(item);
            }
            ++plane_index;
        }
        return !packed.empty();
    }

    double confidenceFor(const PackedDetection& p, int count,
                         double min_x, double max_x, double min_y, double max_y,
                         int model_w, int model_h) const {
        if (count <= 0 || min_x > max_x || min_y > max_y) return 0.0;
        const double bw = std::max(1.0, (double)(p.x2 - p.x1));
        const double bh = std::max(1.0, (double)(p.y2 - p.y1));
        const double width_ratio = std::max(0.0, max_x - min_x) / bw;
        const double height_ratio = std::max(0.0, max_y - min_y) / bh;
        const double bottom_gap_rel = clamp01(((double)p.y2 - max_y) / bh);
        const double pixel_factor = clamp01((double)count / (double)std::max(1, min_pixels_ * 2));
        const double bottom_factor = 1.0 - clamp01(bottom_gap_rel / 0.18);
        const double width_factor = width_ratio < 0.05 ? clamp01(width_ratio / 0.05) :
                                    (width_ratio > 0.90 ? clamp01((1.0 - width_ratio) / 0.10) : 1.0);
        const double height_factor = height_ratio < 0.02 ? clamp01(height_ratio / 0.02) : 1.0;
        double conf = 0.35 * pixel_factor + 0.35 * bottom_factor + 0.20 * width_factor + 0.10 * height_factor;

        const bool edge_cropped = p.y2 >= model_h - 2 || p.x1 <= 1 || p.x2 >= model_w - 2;
        if (edge_cropped) conf *= 0.75;
        if (count < min_pixels_) conf *= 0.5;
        return clamp01(conf);
    }

    Parameters buildOutputMetadata(const std::vector<PackedDetection>& packed,
                                   const std::vector<int>& count,
                                   const std::vector<int>& sum_x100,
                                   const std::vector<int>& sum_y100,
                                   const std::vector<int>& min_x100,
                                   const std::vector<int>& max_x100,
                                   const std::vector<int>& min_y100,
                                   const std::vector<int>& max_y100,
                                   int model_w, int model_h) const {
        Parameters out_md = Parameters::object();
        out_md["schema"] = "player_feet_seg_v1";
        out_md["coord_space"] = "model";
        out_md["model_width"] = model_w;
        out_md["model_height"] = model_h;
        out_md["detections"] = Parameters::array();

        for (int i = 0; i < (int)packed.size(); ++i) {
            const PackedDetection& p = packed[(size_t)i];
            const int c = count[(size_t)i];
            const bool valid = c > 0 &&
                               min_x100[(size_t)i] != INT_MAX && min_y100[(size_t)i] != INT_MAX &&
                               max_x100[(size_t)i] != INT_MIN && max_y100[(size_t)i] != INT_MIN;

            const double fallback_x = ((double)p.x1 + (double)p.x2) * 0.5;
            const double fallback_y = (double)p.y2;

            double min_x = fallback_x, max_x = fallback_x, min_y = fallback_y, max_y = fallback_y;
            double foot_x = fallback_x, foot_y = fallback_y, centroid_x = fallback_x, centroid_y = fallback_y;
            if (valid) {
                min_x = (double)min_x100[(size_t)i] / 100.0;
                max_x = (double)max_x100[(size_t)i] / 100.0;
                min_y = (double)min_y100[(size_t)i] / 100.0;
                max_y = (double)max_y100[(size_t)i] / 100.0;
                foot_x = (min_x + max_x) * 0.5;
                foot_y = max_y;
                centroid_x = (double)sum_x100[(size_t)i] / (double)c / 100.0;
                centroid_y = (double)sum_y100[(size_t)i] / (double)c / 100.0;
            }

            const double conf = valid ?
                confidenceFor(p, c, min_x, max_x, min_y, max_y, model_w, model_h) : 0.0;

            Parameters det = Parameters::object();
            det["label"] = "feet";
            det["cls"] = 0;
            det["conf"] = round3(conf);
            det["source"] = valid ? "player_seg_bottom_mask" : "bbox_bottom_fallback";
            det["valid"] = valid && c >= min_pixels_;
            det["xyxy"] = Parameters::array({round3(min_x), round3(min_y), round3(max_x), round3(max_y)});
            det["foot_point"] = Parameters::array({round3(foot_x), round3(foot_y)});
            det["left_point"] = Parameters::array({round3(min_x), round3(foot_y)});
            det["right_point"] = Parameters::array({round3(max_x), round3(foot_y)});
            det["centroid"] = Parameters::array({round3(centroid_x), round3(centroid_y)});
            det["pixels"] = c;
            det["source_det_index"] = p.array_index;
            det["source_plane_index"] = p.plane_index;
            if (p.track_id >= 0) det["track_id"] = p.track_id;
            if (!p.team_ab.empty() && p.team_ab != "?") det["team_ab"] = p.team_ab;
            const double bw = std::max(1.0, (double)(p.x2 - p.x1));
            const double bh = std::max(1.0, (double)(p.y2 - p.y1));
            if (valid) {
                det["width_ratio"] = round3(std::max(0.0, max_x - min_x) / bw);
                det["height_ratio"] = round3(std::max(0.0, max_y - min_y) / bh);
                det["bottom_gap_ratio"] = round3(clamp01(((double)p.y2 - max_y) / bh));
            }
            out_md["detections"].push_back(det);
        }

        return out_md;
    }

public:
    using NodeSISO<av::VideoFrame, av::VideoFrame>::NodeSISO;
    bool consumeEofIfPresent() override {
        return false;
    }

    ~PlayerFeetSeg() {
        if (cu_ctx_) {
            FEET_SEG_CHECK_CU(cuCtxSetCurrent(cu_ctx_));
        }
        releaseBuffers();
        if (cu_module_) {
            FEET_SEG_CHECK_CU(cuModuleUnload(cu_module_));
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

        const AVFrameSideData* sd = av_frame_get_side_data(raw, yoloSegGpuSideDataType(input_side_data_slot_));
        if (!sd || !sd->buf || sd->buf->size < (int)sizeof(GpuMaskSideDataHeader)) {
            attachEmptyMetadata(frm);
            this->sink_->put(frm);
            return;
        }

        Parameters seg_md;
        if (!readMetadata(raw, metadata_key_, seg_md)) {
            attachEmptyMetadata(frm);
            this->sink_->put(frm);
            return;
        }

        Parameters player_md;
        std::vector<PlayerDet> players;
        if (readMetadata(raw, player_metadata_key_, player_md)) parsePlayers(player_md, players);

        const auto* header = (const GpuMaskSideDataHeader*)sd->buf->data;
        const CUdeviceptr gpu_masks = (CUdeviceptr)header->gpu_ptr;
        if (!gpu_masks || header->num_masks == 0 || header->proto_w == 0 || header->proto_h == 0) {
            attachEmptyMetadata(frm);
            this->sink_->put(frm);
            return;
        }

        std::vector<PackedDetection> packed;
        if (!packDetections(seg_md, players, packed)) {
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
            err << "player_feet_seg: mask index out of range"
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

        const int num_dets = (int)packed.size();
        std::vector<int> zero((size_t)num_dets, 0);
        std::vector<int> min_init((size_t)num_dets, INT_MAX);
        std::vector<int> max_init((size_t)num_dets, INT_MIN);

        if (FEET_SEG_CHECK_CU(cuMemcpyHtoDAsync(d_bboxes_, host_bboxes.data(), host_bboxes.size() * sizeof(int), cuda_dev_ctx_->stream)) ||
            FEET_SEG_CHECK_CU(cuMemcpyHtoDAsync(d_plane_indices_, host_plane_indices.data(), host_plane_indices.size() * sizeof(int), cuda_dev_ctx_->stream)) ||
            FEET_SEG_CHECK_CU(cuMemcpyHtoDAsync(d_count_, zero.data(), zero.size() * sizeof(int), cuda_dev_ctx_->stream)) ||
            FEET_SEG_CHECK_CU(cuMemcpyHtoDAsync(d_sum_x100_, zero.data(), zero.size() * sizeof(int), cuda_dev_ctx_->stream)) ||
            FEET_SEG_CHECK_CU(cuMemcpyHtoDAsync(d_sum_y100_, zero.data(), zero.size() * sizeof(int), cuda_dev_ctx_->stream)) ||
            FEET_SEG_CHECK_CU(cuMemcpyHtoDAsync(d_min_x100_, min_init.data(), min_init.size() * sizeof(int), cuda_dev_ctx_->stream)) ||
            FEET_SEG_CHECK_CU(cuMemcpyHtoDAsync(d_max_x100_, max_init.data(), max_init.size() * sizeof(int), cuda_dev_ctx_->stream)) ||
            FEET_SEG_CHECK_CU(cuMemcpyHtoDAsync(d_min_y100_, min_init.data(), min_init.size() * sizeof(int), cuda_dev_ctx_->stream)) ||
            FEET_SEG_CHECK_CU(cuMemcpyHtoDAsync(d_max_y100_, max_init.data(), max_init.size() * sizeof(int), cuda_dev_ctx_->stream))) {
            this->sink_->put(frm);
            return;
        }

        const int proto_w = (int)header->proto_w;
        const int proto_h = (int)header->proto_h;
        const int model_w = (int)header->model_w;
        const int model_h = (int)header->model_h;
        const size_t mask_bytes = packed.size() * (size_t)proto_w * (size_t)proto_h * sizeof(float);
        CUdeviceptr out_masks = 0;
        if (FEET_SEG_CHECK_CU(cuMemAlloc(&out_masks, mask_bytes))) {
            this->sink_->put(frm);
            return;
        }

        const unsigned int block_x = 16;
        const unsigned int block_y = 16;
        const unsigned int grid_x = ((unsigned int)proto_w + block_x - 1) / block_x;
        const unsigned int grid_y = ((unsigned int)proto_h + block_y - 1) / block_y;
        void* args[] = {
            (void*)&gpu_masks,
            (void*)&out_masks,
            (void*)&proto_w, (void*)&proto_h,
            (void*)&model_w, (void*)&model_h,
            (void*)&d_bboxes_,
            (void*)&d_plane_indices_,
            (void*)&num_dets,
            (void*)&mask_threshold_,
            (void*)&foot_x_margin_rel_,
            (void*)&foot_y_start_rel_,
            (void*)&foot_y_end_margin_rel_,
            (void*)&d_count_,
            (void*)&d_sum_x100_,
            (void*)&d_sum_y100_,
            (void*)&d_min_x100_,
            (void*)&d_max_x100_,
            (void*)&d_min_y100_,
            (void*)&d_max_y100_
        };

        if (FEET_SEG_CHECK_CU(cuLaunchKernel(kernel_,
                                            grid_x, grid_y, (unsigned int)num_dets,
                                            block_x, block_y, 1,
                                            0, cuda_dev_ctx_->stream, args, nullptr))) {
            FEET_SEG_CHECK_CU(cuMemFree(out_masks));
            this->sink_->put(frm);
            return;
        }

        std::vector<int> h_count((size_t)num_dets);
        std::vector<int> h_sum_x100((size_t)num_dets);
        std::vector<int> h_sum_y100((size_t)num_dets);
        std::vector<int> h_min_x100((size_t)num_dets);
        std::vector<int> h_max_x100((size_t)num_dets);
        std::vector<int> h_min_y100((size_t)num_dets);
        std::vector<int> h_max_y100((size_t)num_dets);

        if (FEET_SEG_CHECK_CU(cuMemcpyDtoHAsync(h_count.data(), d_count_, h_count.size() * sizeof(int), cuda_dev_ctx_->stream)) ||
            FEET_SEG_CHECK_CU(cuMemcpyDtoHAsync(h_sum_x100.data(), d_sum_x100_, h_sum_x100.size() * sizeof(int), cuda_dev_ctx_->stream)) ||
            FEET_SEG_CHECK_CU(cuMemcpyDtoHAsync(h_sum_y100.data(), d_sum_y100_, h_sum_y100.size() * sizeof(int), cuda_dev_ctx_->stream)) ||
            FEET_SEG_CHECK_CU(cuMemcpyDtoHAsync(h_min_x100.data(), d_min_x100_, h_min_x100.size() * sizeof(int), cuda_dev_ctx_->stream)) ||
            FEET_SEG_CHECK_CU(cuMemcpyDtoHAsync(h_max_x100.data(), d_max_x100_, h_max_x100.size() * sizeof(int), cuda_dev_ctx_->stream)) ||
            FEET_SEG_CHECK_CU(cuMemcpyDtoHAsync(h_min_y100.data(), d_min_y100_, h_min_y100.size() * sizeof(int), cuda_dev_ctx_->stream)) ||
            FEET_SEG_CHECK_CU(cuMemcpyDtoHAsync(h_max_y100.data(), d_max_y100_, h_max_y100.size() * sizeof(int), cuda_dev_ctx_->stream)) ||
            FEET_SEG_CHECK_CU(cuStreamSynchronize(cuda_dev_ctx_->stream))) {
            FEET_SEG_CHECK_CU(cuMemFree(out_masks));
            this->sink_->put(frm);
            return;
        }

        Parameters out_md = buildOutputMetadata(packed, h_count, h_sum_x100, h_sum_y100,
                                                h_min_x100, h_max_x100, h_min_y100, h_max_y100,
                                                model_w, model_h);

        auto* header_out = (GpuMaskSideDataHeader*)av_malloc(sizeof(GpuMaskSideDataHeader));
        auto* release = (GpuMaskRelease*)av_malloc(sizeof(GpuMaskRelease));
        if (!header_out || !release) {
            if (header_out) av_free(header_out);
            if (release) av_free(release);
            FEET_SEG_CHECK_CU(cuMemFree(out_masks));
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
            FEET_SEG_CHECK_CU(cuMemFree(out_masks));
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
            int valid_count = 0;
            for (int c : h_count) {
                if (c >= min_pixels_) ++valid_count;
            }
            logstream << "player_feet_seg: frame=" << frame_counter_
                      << " feet=" << num_dets
                      << " valid=" << valid_count
                      << " input_slot=" << input_side_data_slot_
                      << " output_slot=" << output_side_data_slot_
                      << " proto=" << proto_w << "x" << proto_h;
        }

        this->sink_->put(frm);
    }

    static std::shared_ptr<PlayerFeetSeg> create(NodeCreationInfo& nci) {
        EdgeManager& edges = nci.edges;
        const Parameters& params = nci.params;
        auto r = NodeSISO<av::VideoFrame, av::VideoFrame>::template createCommon<PlayerFeetSeg>(edges, params);

        if (params.count("metadata_key")) r->metadata_key_ = params["metadata_key"].get<std::string>();
        if (params.count("player_metadata_key")) r->player_metadata_key_ = params["player_metadata_key"].get<std::string>();
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
            throw Error("player_feet_seg: side-data slot out of range [0," + std::to_string(kMaxYoloSegSlots - 1) + "]");
        }
        if (r->input_side_data_slot_ == r->output_side_data_slot_) {
            throw Error("player_feet_seg: input_side_data_slot and output_side_data_slot must be different");
        }
        if (params.count("mask_threshold")) r->mask_threshold_ = params["mask_threshold"].get<float>();
        if (params.count("foot_x_margin_rel")) r->foot_x_margin_rel_ = params["foot_x_margin_rel"].get<float>();
        if (params.count("foot_y_start_rel")) r->foot_y_start_rel_ = params["foot_y_start_rel"].get<float>();
        if (params.count("foot_y_end_margin_rel")) r->foot_y_end_margin_rel_ = params["foot_y_end_margin_rel"].get<float>();
        if (params.count("min_pixels")) r->min_pixels_ = std::max(1, params["min_pixels"].get<int>());
        if (params.count("player_iou_threshold")) r->player_iou_threshold_ = params["player_iou_threshold"].get<float>();
        if (params.count("fallback_center_distance_px")) r->fallback_center_distance_px_ = params["fallback_center_distance_px"].get<float>();
        if (params.count("initial_capacity")) r->initial_capacity_ = params["initial_capacity"].get<int>();
        if (params.count("debug_log_every_n")) r->debug_log_every_n_ = params["debug_log_every_n"].get<int>();
        return r;
    }
};

DECLNODE(player_feet_seg, PlayerFeetSeg)
