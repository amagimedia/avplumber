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

    std::string pose_metadata_key_ = "yolo_pose";
    std::string metadata_key_out_ = "yolo_seg";
    int mask_w_ = 240;
    int mask_h_ = 136;
    float min_keypoint_conf_ = 0.3f;
    int min_visible_keypoints_ = 6;
    float min_court_area_ = 0.05f;
    int expected_num_keypoints_ = 12;
    float expand_ratio_ = 1.0f;
    float expand_px_ = 0.0f;
    float snap_left_px_ = 0.0f;
    float snap_right_px_ = 0.0f;
    float snap_top_px_ = 0.0f;
    float snap_bottom_px_ = 0.0f;
    std::vector<int> winding_order_ = {0, 3, 5, 7, 9, 10, 11, 8, 6, 4, 2, 1};
    int debug_log_every_n_ = 0;

    CUcontext cu_ctx_ = nullptr;
    uint64_t frame_counter_ = 0;

    std::vector<float> cpu_mask_;

    struct PolygonVertex {
        float x = 0.0f;
        float y = 0.0f;
    };

    static float cross(const PolygonVertex& o, const PolygonVertex& a, const PolygonVertex& b) {
        return (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x);
    }

    bool shouldDebugLog() const {
        return debug_log_every_n_ > 0 && (frame_counter_ % (uint64_t)debug_log_every_n_) == 0;
    }

    bool initCudaContext(const av::VideoFrame& frm) {
        const AVFrame* raw = frm.raw();
        if (!raw || !raw->hw_frames_ctx) return false;
        AVHWFramesContext* fctx = (AVHWFramesContext*)raw->hw_frames_ctx->data;
        if (!fctx || !fctx->device_ctx) return false;
        AVCUDADeviceContext* cuda_dev = (AVCUDADeviceContext*)fctx->device_ctx->hwctx;
        if (!cuda_dev) return false;
        cu_ctx_ = cuda_dev->cuda_ctx;
        cuCtxSetCurrent(cu_ctx_);
        return true;
    }

    bool parsePoseKeypoints(const av::VideoFrame& frm,
                            std::vector<CourtKeypoint>& keypoints_out,
                            float& model_w_out, float& model_h_out,
                            float& det_conf_out) {
        const AVFrame* raw = frm.raw();
        if (!raw || !raw->metadata) return false;

        AVDictionaryEntry* entry = av_dict_get(raw->metadata, pose_metadata_key_.c_str(), nullptr, 0);
        if (!entry || !entry->value) return false;

        try {
            Parameters md = Parameters::parse(entry->value);
            model_w_out = md.value("model_width", 960.0f);
            model_h_out = md.value("model_height", 544.0f);

            if (!md.contains("poses") || !md["poses"].is_array() || md["poses"].empty()) return false;

            // Find highest-confidence pose
            float best_conf = -1.0f;
            const Parameters* best_pose = nullptr;
            for (const auto& pose : md["poses"]) {
                float conf = pose.value("conf", 0.0f);
                if (conf > best_conf) {
                    best_conf = conf;
                    best_pose = &pose;
                }
            }
            if (!best_pose) return false;
            det_conf_out = best_conf;

            if (!best_pose->contains("keypoints") || !(*best_pose)["keypoints"].is_array()) return false;
            const auto& kpts_array = (*best_pose)["keypoints"];

            // Flat array, stride 3: x, y, conf
            if ((kpts_array.size() % 3) != 0) return false;
            size_t num_kpts = kpts_array.size() / 3;
            if (md.value("num_keypoints", -1) != expected_num_keypoints_) return false;
            if ((int)num_kpts != expected_num_keypoints_) return false;
            keypoints_out.resize(num_kpts);
            for (size_t i = 0; i < num_kpts; i++) {
                CourtKeypoint kp;
                kp.x = kpts_array[i * 3 + 0].get<float>();
                kp.y = kpts_array[i * 3 + 1].get<float>();
                kp.conf = kpts_array[i * 3 + 2].get<float>();
                kp.visible = (kp.conf >= min_keypoint_conf_);
                keypoints_out[i] = kp;
            }
            return true;
        } catch (...) {
            return false;
        }
    }

    bool buildPolygon(const std::vector<CourtKeypoint>& keypoints,
                      std::vector<PolygonVertex>& polygon_out) {
        std::vector<PolygonVertex> points;
        for (int idx : winding_order_) {
            if (idx < 0 || idx >= (int)keypoints.size()) continue;
            if (!keypoints[idx].visible) continue;
            PolygonVertex v;
            v.x = keypoints[idx].x;
            v.y = keypoints[idx].y;
            points.push_back(v);
        }
        if ((int)points.size() < min_visible_keypoints_) return false;

        std::sort(points.begin(), points.end(),
                  [](const PolygonVertex& a, const PolygonVertex& b) {
                      if (a.x != b.x) return a.x < b.x;
                      return a.y < b.y;
                  });
        points.erase(std::unique(points.begin(), points.end(),
                                 [](const PolygonVertex& a, const PolygonVertex& b) {
                                     return std::abs(a.x - b.x) < 1e-3f && std::abs(a.y - b.y) < 1e-3f;
                                 }),
                     points.end());
        if ((int)points.size() < min_visible_keypoints_) return false;

        std::vector<PolygonVertex> hull;
        hull.reserve(points.size() * 2);

        for (const auto& p : points) {
            while (hull.size() >= 2 &&
                   cross(hull[hull.size() - 2], hull[hull.size() - 1], p) <= 0.0f) {
                hull.pop_back();
            }
            hull.push_back(p);
        }

        size_t lower_size = hull.size();
        for (int i = (int)points.size() - 2; i >= 0; --i) {
            const auto& p = points[(size_t)i];
            while (hull.size() > lower_size &&
                   cross(hull[hull.size() - 2], hull[hull.size() - 1], p) <= 0.0f) {
                hull.pop_back();
            }
            hull.push_back(p);
        }

        if (!hull.empty()) hull.pop_back();
        polygon_out = std::move(hull);
        return (int)polygon_out.size() >= 3;
    }

    static bool isConvex(const std::vector<PolygonVertex>& poly) {
        int n = (int)poly.size();
        if (n < 3) return false;
        int sign = 0;
        for (int i = 0; i < n; i++) {
            const PolygonVertex& a = poly[i];
            const PolygonVertex& b = poly[(i + 1) % n];
            const PolygonVertex& c = poly[(i + 2) % n];
            float cross = (b.x - a.x) * (c.y - b.y) - (b.y - a.y) * (c.x - b.x);
            if (std::abs(cross) < 1e-6f) continue; // collinear
            int s = (cross > 0.0f) ? 1 : -1;
            if (sign == 0) {
                sign = s;
            } else if (s != sign) {
                return false;
            }
        }
        return sign != 0;
    }

    void expandPolygon(std::vector<PolygonVertex>& poly, float model_w, float model_h) const {
        if (poly.size() < 3) return;
        if (expand_ratio_ <= 1.0f && expand_px_ <= 0.0f) return;

        float cx = 0.0f;
        float cy = 0.0f;
        for (const auto& v : poly) {
            cx += v.x;
            cy += v.y;
        }
        cx /= (float)poly.size();
        cy /= (float)poly.size();

        for (auto& v : poly) {
            float dx = v.x - cx;
            float dy = v.y - cy;
            float len = std::sqrt(dx * dx + dy * dy);
            float scale = expand_ratio_;
            if (len > 1e-4f && expand_px_ > 0.0f) {
                scale += expand_px_ / len;
            }
            v.x = cx + dx * scale;
            v.y = cy + dy * scale;
            v.x = std::max(0.0f, std::min(v.x, model_w - 1.0f));
            v.y = std::max(0.0f, std::min(v.y, model_h - 1.0f));
        }
    }

    void snapPolygonToBorders(std::vector<PolygonVertex>& poly, float model_w, float model_h) const {
        if (poly.size() < 3) return;

        float min_x = poly[0].x, max_x = poly[0].x;
        float min_y = poly[0].y, max_y = poly[0].y;
        for (const auto& v : poly) {
            min_x = std::min(min_x, v.x);
            max_x = std::max(max_x, v.x);
            min_y = std::min(min_y, v.y);
            max_y = std::max(max_y, v.y);
        }

        const bool snap_left = snap_left_px_ > 0.0f && min_x <= snap_left_px_;
        const bool snap_right = snap_right_px_ > 0.0f && max_x >= (model_w - 1.0f - snap_right_px_);
        const bool snap_top = snap_top_px_ > 0.0f && min_y <= snap_top_px_;
        const bool snap_bottom = snap_bottom_px_ > 0.0f && max_y >= (model_h - 1.0f - snap_bottom_px_);

        const float edge_eps_x = 12.0f;
        const float edge_eps_y = 12.0f;
        for (auto& v : poly) {
            if (snap_left && v.x <= min_x + edge_eps_x) v.x = 0.0f;
            if (snap_right && v.x >= max_x - edge_eps_x) v.x = model_w - 1.0f;
            if (snap_top && v.y <= min_y + edge_eps_y) v.y = 0.0f;
            if (snap_bottom && v.y >= max_y - edge_eps_y) v.y = model_h - 1.0f;
        }
    }

    float rasterizePolygon(const std::vector<PolygonVertex>& poly,
                           float model_w, float model_h) {
        int total = mask_w_ * mask_h_;
        cpu_mask_.assign(total, 0.0f);

        int n = (int)poly.size();
        if (n < 3) return 0.0f;

        // Map vertices to mask space
        std::vector<PolygonVertex> mverts(n);
        for (int i = 0; i < n; i++) {
            mverts[i].x = poly[i].x / model_w * (float)mask_w_;
            mverts[i].y = poly[i].y / model_h * (float)mask_h_;
        }

        int filled = 0;
        for (int row = 0; row < mask_h_; row++) {
            float y = (float)row + 0.5f;
            std::vector<float> xs;
            for (int i = 0; i < n; i++) {
                const PolygonVertex& va = mverts[i];
                const PolygonVertex& vb = mverts[(i + 1) % n];
                float y0 = va.y, y1 = vb.y;
                if ((y0 <= y && y < y1) || (y1 <= y && y < y0)) {
                    float t = (y - y0) / (y1 - y0);
                    xs.push_back(va.x + t * (vb.x - va.x));
                }
            }
            std::sort(xs.begin(), xs.end());
            for (size_t k = 0; k + 1 < xs.size(); k += 2) {
                int x0 = (int)std::ceil(xs[k]);
                int x1 = (int)std::floor(xs[k + 1]);
                x0 = std::max(x0, 0);
                x1 = std::min(x1, mask_w_ - 1);
                for (int col = x0; col <= x1; col++) {
                    cpu_mask_[row * mask_w_ + col] = 1.0f;
                    filled++;
                }
            }
        }

        return (total > 0) ? (float)filled / (float)total : 0.0f;
    }

    void attachCpuSideData(av::VideoFrame& frm) {
        size_t header_size = 16;
        size_t mask_data_size = cpu_mask_.size() * sizeof(float);
        size_t total_size = header_size + mask_data_size;

        AVBufferRef* buf = av_buffer_alloc(total_size);
        if (!buf) return;

        uint32_t* header = (uint32_t*)buf->data;
        header[0] = 1;              // num_masks
        header[1] = (uint32_t)mask_w_;
        header[2] = (uint32_t)mask_h_;
        header[3] = 0;              // reserved
        memcpy(buf->data + header_size, cpu_mask_.data(), mask_data_size);

        av_frame_new_side_data_from_buf(frm.raw(), AV_FRAME_DATA_YOLO_SEG_MASKS, buf);
    }

    void attachGpuSideData(av::VideoFrame& frm, float model_w, float model_h) {
        if (!cu_ctx_) {
            if (!initCudaContext(frm)) return;
        }

        size_t mask_bytes = cpu_mask_.size() * sizeof(float);
        CUdeviceptr gpu_mask_buf = 0;
        if (cuMemAlloc(&gpu_mask_buf, mask_bytes) != CUDA_SUCCESS) return;
        if (cuMemcpyHtoD(gpu_mask_buf, cpu_mask_.data(), mask_bytes) != CUDA_SUCCESS) {
            cuMemFree(gpu_mask_buf);
            return;
        }

        AVBufferRef* gpu_ref = av_buffer_create(
            (uint8_t*)(uintptr_t)gpu_mask_buf, (int)mask_bytes,
            [](void* opaque, uint8_t* data) {
                CUcontext ctx = (CUcontext)opaque;
                if (ctx) cuCtxSetCurrent(ctx);
                cuMemFree((CUdeviceptr)(uintptr_t)data);
            },
            cu_ctx_, 0);
        if (!gpu_ref) {
            cuMemFree(gpu_mask_buf);
            return;
        }

        auto* hdr = (GpuMaskSideDataHeader*)av_malloc(sizeof(GpuMaskSideDataHeader));
        if (!hdr) {
            av_buffer_unref(&gpu_ref);
            return;
        }

        hdr->gpu_ptr = (uint64_t)gpu_mask_buf;
        hdr->num_masks = 1;
        hdr->proto_w = (uint32_t)mask_w_;
        hdr->proto_h = (uint32_t)mask_h_;
        hdr->model_w = (uint32_t)model_w;
        hdr->model_h = (uint32_t)model_h;

        AVBufferRef* sd_buf = av_buffer_create(
            (uint8_t*)hdr, sizeof(GpuMaskSideDataHeader),
            [](void* opaque, uint8_t* data) {
                av_buffer_unref((AVBufferRef**)&opaque);
                av_free(data);
            },
            gpu_ref, 0);
        if (!sd_buf) {
            av_buffer_unref(&gpu_ref);
            av_free(hdr);
            return;
        }

        av_frame_new_side_data_from_buf(frm.raw(), AV_FRAME_DATA_YOLO_SEG_MASKS_GPU, sd_buf);
    }

    void writeSegMetadata(av::VideoFrame& frm, float model_w, float model_h, float det_conf) {
        Parameters out;
        out["model_width"] = (int)model_w;
        out["model_height"] = (int)model_h;
        Parameters det;
        det["cls"] = 0;
        det["conf"] = det_conf;
        det["xyxy"] = {0, 0, (int)model_w, (int)model_h};
        det["label"] = "court";
        out["detections"] = Parameters::array({det});
        std::string serialized = out.dump();
        av_dict_set(&frm.raw()->metadata, metadata_key_out_.c_str(), serialized.c_str(), 0);
    }

public:
    using NodeSISO<av::VideoFrame, av::VideoFrame>::NodeSISO;

    void process() override {
        av::VideoFrame frm = this->source_->get();
        if (!frm) return;

        if (isEofMarker(frm)) {
            frame_counter_ = 0;
            this->sink_->put(frm);
            return;
        }

        ++frame_counter_;

        // 1. Parse pose keypoints
        std::vector<CourtKeypoint> keypoints;
        float model_w = 960.0f, model_h = 544.0f, det_conf = 0.0f;
        if (!parsePoseKeypoints(frm, keypoints, model_w, model_h, det_conf)) {
            if (shouldDebugLog()) {
                logstream << "court_polygon: frame=" << frame_counter_
                          << " skipped - invalid pose metadata or keypoint count (expected "
                          << expected_num_keypoints_ << ")";
            }
            this->sink_->put(frm);
            return;
        }

        // 2. Build polygon from visible keypoints
        std::vector<PolygonVertex> polygon;
        if (!buildPolygon(keypoints, polygon)) {
            if (shouldDebugLog()) {
                logstream << "court_polygon: frame=" << frame_counter_
                          << " skipped - too few visible keypoints ("
                          << polygon.size() << " < " << min_visible_keypoints_ << ")";
            }
            this->sink_->put(frm);
            return;
        }

        // 3. Validate convexity
        if (!isConvex(polygon)) {
            if (shouldDebugLog()) {
                logstream << "court_polygon: frame=" << frame_counter_
                          << " skipped - polygon not convex (vertices=" << polygon.size() << ")";
            }
            this->sink_->put(frm);
            return;
        }

        // 4. Expand polygon to better cover the court before rasterization.
        expandPolygon(polygon, model_w, model_h);
        snapPolygonToBorders(polygon, model_w, model_h);

        // 5. Rasterize polygon
        float area_ratio = rasterizePolygon(polygon, model_w, model_h);

        // 6. Check area
        if (area_ratio < min_court_area_) {
            if (shouldDebugLog()) {
                logstream << "court_polygon: frame=" << frame_counter_
                          << " skipped - area too small (" << (int)(area_ratio * 100) << "% < "
                          << (int)(min_court_area_ * 100) << "%)";
            }
            this->sink_->put(frm);
            return;
        }

        // 7. Attach CPU side data
        attachCpuSideData(frm);

        // 8. Attach GPU side data (only for CUDA frames)
        if (frm.raw() && frm.raw()->hw_frames_ctx) {
            attachGpuSideData(frm, model_w, model_h);
        }

        // 9. Write seg metadata
        writeSegMetadata(frm, model_w, model_h, det_conf);

        if (shouldDebugLog()) {
            logstream << "court_polygon: frame=" << frame_counter_
                      << " vertices=" << polygon.size()
                      << " area=" << (int)(area_ratio * 100) << "%"
                      << " model=" << (int)model_w << "x" << (int)model_h
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
        if (params.count("expected_num_keypoints")) r->expected_num_keypoints_ = params["expected_num_keypoints"].get<int>();
        if (params.count("expand_ratio")) r->expand_ratio_ = params["expand_ratio"].get<float>();
        if (params.count("expand_px")) r->expand_px_ = params["expand_px"].get<float>();
        if (params.count("snap_left_px")) r->snap_left_px_ = params["snap_left_px"].get<float>();
        if (params.count("snap_right_px")) r->snap_right_px_ = params["snap_right_px"].get<float>();
        if (params.count("snap_top_px")) r->snap_top_px_ = params["snap_top_px"].get<float>();
        if (params.count("snap_bottom_px")) r->snap_bottom_px_ = params["snap_bottom_px"].get<float>();
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
