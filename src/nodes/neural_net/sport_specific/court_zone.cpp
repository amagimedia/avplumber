#include "../../node_common.hpp"
#include "../common/yolo_side_data.hpp"

extern "C" {
#include <libavutil/dict.h>
#include <libavutil/frame.h>
}

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

class CourtZone : public NodeSISO<av::VideoFrame, av::VideoFrame>, public ReportsFinishByFlag {
    std::string handler_metadata_key_ = "ball_handler";
    std::string ball_metadata_key_ = "yolo_ball";
    std::string player_metadata_key_ = "yolo_players";
    std::string seg_metadata_key_ = "yolo_seg";
    std::string output_metadata_key_ = "court_zone";
    int seg_side_data_slot_ = 0;
    float mask_threshold_ = 0.5f;
    int line_search_radius_rows_ = 3;
    float line_inside_margin_px_ = 12.0f;
    float inside_three_fallback_radius_rel_ = 0.24f;
    float paint_depth_rel_ = 0.16f;
    float paint_half_height_rel_ = 0.18f;
    float corner_baseline_rel_ = 0.22f;
    float corner_sideline_rel_ = 0.16f;
    float top_band_rel_ = 0.12f;
    int debug_log_every_n_ = 0;
    uint64_t frame_counter_ = 0;

    struct MaskInfo {
        int num_masks = 0;
        int w = 0;
        int h = 0;
        const float* data = nullptr;
    };

    struct Point {
        float x = 0.0f;
        float y = 0.0f;
        bool valid = false;
    };

    struct Bounds {
        int min_x = 0;
        int min_y = 0;
        int max_x = 0;
        int max_y = 0;
        bool valid = false;
    };

    struct CourtContext {
        float model_w = 960.0f;
        float model_h = 544.0f;
        bool has_hoop = false;
        float hoop_x = 0.0f;
        float hoop_y = 0.0f;
        bool hoop_right = true;
        bool has_court_mask = false;
        bool has_line_mask = false;
        int court_mask_idx = -1;
        int line_mask_idx = -1;
        MaskInfo masks;
        Bounds court_bounds_mask;
    };

    static Parameters tryParse(const AVFrame* raw, const std::string& key) {
        if (!raw || !raw->metadata) return {};
        AVDictionaryEntry* entry = av_dict_get(raw->metadata, key.c_str(), nullptr, 0);
        if (!entry || !entry->value) return {};
        try { return Parameters::parse(entry->value); } catch (...) { return {}; }
    }

    static bool readCpuMasks(const AVFrame* raw, int slot, MaskInfo& out) {
        if (!raw) return false;
        AVFrameSideData* sd = av_frame_get_side_data(raw, yoloSegCpuSideDataType(slot));
        if (!sd || !sd->buf || sd->buf->size < 16) return false;
        const uint32_t* header = (const uint32_t*)sd->buf->data;
        out.num_masks = (int)header[0];
        out.w = (int)header[1];
        out.h = (int)header[2];
        size_t expected = 16 + (size_t)out.num_masks * (size_t)out.w * (size_t)out.h * sizeof(float);
        if ((size_t)sd->buf->size < expected) return false;
        out.data = (const float*)(sd->buf->data + 16);
        return out.num_masks > 0 && out.w > 0 && out.h > 0;
    }

    static const float* maskPtr(const MaskInfo& masks, int idx) {
        if (!masks.data || idx < 0 || idx >= masks.num_masks) return nullptr;
        return masks.data + (size_t)idx * (size_t)masks.w * (size_t)masks.h;
    }

    static bool findMaskIndex(const Parameters& seg_md, const std::string& label, int& out_idx) {
        if (!seg_md.contains("detections") || !seg_md["detections"].is_array()) return false;
        for (size_t i = 0; i < seg_md["detections"].size(); ++i) {
            const auto& det = seg_md["detections"][i];
            if (det.value("label", std::string()) == label) {
                out_idx = (int)i;
                return true;
            }
        }
        return false;
    }

    Bounds computeBounds(const float* mask, int w, int h) const {
        Bounds b;
        if (!mask || w <= 0 || h <= 0) return b;
        b.min_x = w;
        b.min_y = h;
        b.max_x = -1;
        b.max_y = -1;
        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                if (mask[y * w + x] < mask_threshold_) continue;
                b.min_x = std::min(b.min_x, x);
                b.min_y = std::min(b.min_y, y);
                b.max_x = std::max(b.max_x, x);
                b.max_y = std::max(b.max_y, y);
            }
        }
        b.valid = (b.max_x >= b.min_x && b.max_y >= b.min_y);
        return b;
    }

    static int clampi(int v, int lo, int hi) {
        return std::max(lo, std::min(v, hi));
    }

    static int roundToMask(float v, float model_dim, int mask_dim) {
        if (mask_dim <= 0 || model_dim <= 0.0f) return 0;
        float scaled = v * (float)mask_dim / model_dim;
        return clampi((int)std::lround(scaled), 0, mask_dim - 1);
    }

    bool sampleMask(const float* mask, int w, int h, float model_x, float model_y,
                    float model_w, float model_h) const {
        if (!mask || w <= 0 || h <= 0 || model_w <= 0.0f || model_h <= 0.0f) return false;
        int mx = roundToMask(model_x, model_w, w);
        int my = roundToMask(model_y, model_h, h);
        return mask[my * w + mx] >= mask_threshold_;
    }

    static std::string sideLabel(float y, float hoop_y, const std::string& left_name, const std::string& right_name) {
        return (y < hoop_y) ? left_name : right_name;
    }

    Point parseHandlerPoint(const Parameters& handler_md) const {
        Point p;
        if (!handler_md.contains("detections") || !handler_md["detections"].is_array()) return p;
        for (const auto& det : handler_md["detections"]) {
            if (!det.contains("xyxy") || !det["xyxy"].is_array() || det["xyxy"].size() < 4) continue;
            p.x = (det["xyxy"][0].get<float>() + det["xyxy"][2].get<float>()) * 0.5f;
            p.y = det["xyxy"][3].get<float>();
            p.valid = true;
            return p;
        }
        return p;
    }

    Point parseBallPoint(const Parameters& ball_md) const {
        Point p;
        if (!ball_md.contains("detections") || !ball_md["detections"].is_array()) return p;
        for (const auto& det : ball_md["detections"]) {
            if (!det.contains("xyxy") || !det["xyxy"].is_array() || det["xyxy"].size() < 4) continue;
            p.x = (det["xyxy"][0].get<float>() + det["xyxy"][2].get<float>()) * 0.5f;
            p.y = (det["xyxy"][1].get<float>() + det["xyxy"][3].get<float>()) * 0.5f;
            p.valid = true;
            return p;
        }
        return p;
    }

    void parseHoop(const Parameters& players_md, CourtContext& ctx) const {
        if (!players_md.contains("detections") || !players_md["detections"].is_array()) return;
        float best_conf = -1.0f;
        for (const auto& det : players_md["detections"]) {
            if (det.value("label", std::string()) != "Hoop") continue;
            if (!det.contains("xyxy") || !det["xyxy"].is_array() || det["xyxy"].size() < 4) continue;
            float conf = det.value("conf", 0.0f);
            if (conf < best_conf) continue;
            best_conf = conf;
            ctx.hoop_x = (det["xyxy"][0].get<float>() + det["xyxy"][2].get<float>()) * 0.5f;
            ctx.hoop_y = (det["xyxy"][1].get<float>() + det["xyxy"][3].get<float>()) * 0.5f;
            ctx.has_hoop = true;
        }
        if (ctx.has_hoop) ctx.hoop_right = ctx.hoop_x >= (ctx.model_w * 0.5f);
    }

    void parseCourt(const AVFrame* raw, const Parameters& seg_md, CourtContext& ctx) const {
        if (seg_md.contains("model_width")) ctx.model_w = seg_md["model_width"].get<float>();
        if (seg_md.contains("model_height")) ctx.model_h = seg_md["model_height"].get<float>();
        if (!readCpuMasks(raw, seg_side_data_slot_, ctx.masks)) return;

        int idx = -1;
        if (findMaskIndex(seg_md, "basketball-court", idx) && idx < ctx.masks.num_masks) {
            ctx.court_mask_idx = idx;
            ctx.has_court_mask = true;
        } else if (ctx.masks.num_masks > 0) {
            ctx.court_mask_idx = 0;
            ctx.has_court_mask = true;
        }

        idx = -1;
        if (findMaskIndex(seg_md, "three point line", idx) && idx < ctx.masks.num_masks) {
            ctx.line_mask_idx = idx;
            ctx.has_line_mask = true;
        }

        if (ctx.has_court_mask) {
            ctx.court_bounds_mask = computeBounds(maskPtr(ctx.masks, ctx.court_mask_idx), ctx.masks.w, ctx.masks.h);
        }
    }

    bool lineBoundaryAtY(const CourtContext& ctx, float model_y, int& line_x_mask) const {
        if (!ctx.has_line_mask) return false;
        const float* line_mask = maskPtr(ctx.masks, ctx.line_mask_idx);
        if (!line_mask) return false;

        int mask_y = roundToMask(model_y, ctx.model_h, ctx.masks.h);
        int best_row = -1;
        int best_x = 0;
        for (int d = 0; d <= line_search_radius_rows_; ++d) {
            for (int sign = -1; sign <= 1; sign += 2) {
                if (d == 0 && sign > 0) continue;
                int row = clampi(mask_y + d * sign, 0, ctx.masks.h - 1);
                int found_x = ctx.hoop_right ? -1 : ctx.masks.w;
                bool found = false;
                for (int x = 0; x < ctx.masks.w; ++x) {
                    if (line_mask[row * ctx.masks.w + x] < mask_threshold_) continue;
                    found = true;
                    if (ctx.hoop_right) found_x = std::max(found_x, x);
                    else found_x = std::min(found_x, x);
                }
                if (found) {
                    best_row = row;
                    best_x = found_x;
                    break;
                }
            }
            if (best_row >= 0) break;
        }
        if (best_row < 0) return false;
        line_x_mask = best_x;
        return true;
    }

    bool insideThreePointArea(const CourtContext& ctx, const Point& p) const {
        if (!p.valid || !ctx.has_court_mask || !ctx.has_hoop) return false;
        const float* court_mask = maskPtr(ctx.masks, ctx.court_mask_idx);
        if (!sampleMask(court_mask, ctx.masks.w, ctx.masks.h, p.x, p.y, ctx.model_w, ctx.model_h)) return false;

        int line_x_mask = 0;
        int point_x_mask = roundToMask(p.x, ctx.model_w, ctx.masks.w);
        float margin_mask = line_inside_margin_px_ * (float)ctx.masks.w / std::max(1.0f, ctx.model_w);
        if (lineBoundaryAtY(ctx, p.y, line_x_mask)) {
            if (ctx.hoop_right) return (float)point_x_mask >= (float)line_x_mask - margin_mask;
            return (float)point_x_mask <= (float)line_x_mask + margin_mask;
        }

        float dx = p.x - ctx.hoop_x;
        float dy = p.y - ctx.hoop_y;
        float radius = inside_three_fallback_radius_rel_ * (ctx.court_bounds_mask.valid ? (float)(ctx.court_bounds_mask.max_x - ctx.court_bounds_mask.min_x + 1) * ctx.model_w / (float)ctx.masks.w : ctx.model_w);
        return std::sqrt(dx * dx + dy * dy) <= radius;
    }

    std::string classifyZone(const CourtContext& ctx, const Point& p,
                             bool* inside_court_out = nullptr,
                             bool* inside_three_out = nullptr,
                             float* dx_out = nullptr,
                             float* dy_out = nullptr,
                             float* dist_out = nullptr) const {
        if (inside_court_out) *inside_court_out = false;
        if (inside_three_out) *inside_three_out = false;
        if (dx_out) *dx_out = 0.0f;
        if (dy_out) *dy_out = 0.0f;
        if (dist_out) *dist_out = 0.0f;

        if (!p.valid || !ctx.has_hoop || !ctx.has_court_mask || !ctx.court_bounds_mask.valid) return "unknown";

        const float* court_mask = maskPtr(ctx.masks, ctx.court_mask_idx);
        bool inside_court = sampleMask(court_mask, ctx.masks.w, ctx.masks.h, p.x, p.y, ctx.model_w, ctx.model_h);
        if (inside_court_out) *inside_court_out = inside_court;
        if (!inside_court) return "unknown";

        float dx = p.x - ctx.hoop_x;
        float dy = p.y - ctx.hoop_y;
        float dist = std::sqrt(dx * dx + dy * dy);
        if (dx_out) *dx_out = dx;
        if (dy_out) *dy_out = dy;
        if (dist_out) *dist_out = dist;

        bool inside_three = insideThreePointArea(ctx, p);
        if (inside_three_out) *inside_three_out = inside_three;

        float court_min_x = (float)ctx.court_bounds_mask.min_x * ctx.model_w / (float)ctx.masks.w;
        float court_max_x = (float)(ctx.court_bounds_mask.max_x + 1) * ctx.model_w / (float)ctx.masks.w;
        float court_min_y = (float)ctx.court_bounds_mask.min_y * ctx.model_h / (float)ctx.masks.h;
        float court_max_y = (float)(ctx.court_bounds_mask.max_y + 1) * ctx.model_h / (float)ctx.masks.h;
        float court_w = std::max(1.0f, court_max_x - court_min_x);
        float court_h = std::max(1.0f, court_max_y - court_min_y);
        float half_x = (court_min_x + court_max_x) * 0.5f;

        bool backcourt = ctx.hoop_right ? (p.x < half_x) : (p.x > half_x);
        if (backcourt) return "backcourt";

        bool toward_center = ctx.hoop_right ? (dx <= 0.0f) : (dx >= 0.0f);
        float paint_depth = paint_depth_rel_ * court_w;
        float paint_half_h = paint_half_height_rel_ * court_h;
        if (toward_center && std::abs(dx) <= paint_depth && std::abs(dy) <= paint_half_h) {
            return "paint";
        }

        float baseline_dist = ctx.hoop_right ? (court_max_x - p.x) : (p.x - court_min_x);
        float sideline_top = p.y - court_min_y;
        float sideline_bottom = court_max_y - p.y;
        bool near_baseline = baseline_dist <= corner_baseline_rel_ * court_w;
        bool near_sideline = std::min(sideline_top, sideline_bottom) <= corner_sideline_rel_ * court_h;
        if (near_baseline && near_sideline && inside_three) {
            return sideLabel(p.y, ctx.hoop_y, "left_corner", "right_corner");
        }

        if (std::abs(dy) <= top_band_rel_ * court_h) return "top";

        return sideLabel(p.y, ctx.hoop_y, "left_wing", "right_wing");
    }

public:
    using NodeSISO<av::VideoFrame, av::VideoFrame>::NodeSISO;
    bool consumeEofIfPresent() override {
        return false;
    }

    void process() override {
        av::VideoFrame frm = this->source_->get();
        if (isEofMarker(frm)) {
            frame_counter_ = 0;
            this->sink_->put(frm);
            this->finished_ = true;
            return;
        }
        if (!frm) return;

        ++frame_counter_;

        const AVFrame* raw = frm.raw();
        auto handler_md = tryParse(raw, handler_metadata_key_);
        auto ball_md = tryParse(raw, ball_metadata_key_);
        auto players_md = tryParse(raw, player_metadata_key_);
        auto seg_md = tryParse(raw, seg_metadata_key_);

        CourtContext ctx;
        if (players_md.contains("model_width")) ctx.model_w = players_md["model_width"].get<float>();
        if (players_md.contains("model_height")) ctx.model_h = players_md["model_height"].get<float>();
        parseHoop(players_md, ctx);
        parseCourt(raw, seg_md, ctx);

        Point handler_pt = parseHandlerPoint(handler_md);
        Point ball_pt = parseBallPoint(ball_md);

        Parameters out_md;
        if (handler_pt.valid) out_md["zone_source"] = "handler";
        else if (ball_pt.valid) out_md["zone_source"] = "ball";

        if (ctx.has_hoop) out_md["hoop_side"] = ctx.hoop_right ? "right" : "left";

        Point focus_pt = handler_pt.valid ? handler_pt : ball_pt;
        bool focus_inside_court = false;
        bool focus_inside_three = false;
        float focus_dx = 0.0f, focus_dy = 0.0f, focus_dist = 0.0f;

        if (handler_pt.valid) {
            std::string zone = classifyZone(ctx, handler_pt);
            if (zone != "unknown") out_md["handler_zone"] = zone;
            else out_md["handler_zone"] = "unknown";
        }
        if (ball_pt.valid) {
            std::string zone = classifyZone(ctx, ball_pt);
            if (zone != "unknown") out_md["ball_zone"] = zone;
            else out_md["ball_zone"] = "unknown";
        }
        if (focus_pt.valid) {
            classifyZone(ctx, focus_pt, &focus_inside_court, &focus_inside_three, &focus_dx, &focus_dy, &focus_dist);
            out_md["inside_court"] = focus_inside_court;
            out_md["inside_three_point_area"] = focus_inside_three;
            if (ctx.has_hoop) {
                out_md["relative_to_hoop_x"] = (int)std::lround(focus_dx);
                out_md["relative_to_hoop_y"] = (int)std::lround(focus_dy);
                out_md["distance_to_hoop"] = (int)std::lround(focus_dist);
            }
        }

        if (!out_md.empty()) {
            av_dict_set(&frm.raw()->metadata, output_metadata_key_.c_str(), out_md.dump().c_str(), 0);
            if (debug_log_every_n_ > 0 && (frame_counter_ % (uint64_t)debug_log_every_n_) == 0) {
                logstream << "court_zone: frame=" << frame_counter_ << " json=" << out_md.dump();
            }
        }

        this->sink_->put(frm);
    }

    static std::shared_ptr<CourtZone> create(NodeCreationInfo& nci) {
        EdgeManager& edges = nci.edges;
        const Parameters& params = nci.params;
        auto r = NodeSISO<av::VideoFrame, av::VideoFrame>::template createCommon<CourtZone>(edges, params);
        if (params.count("handler_metadata_key")) r->handler_metadata_key_ = params["handler_metadata_key"].get<std::string>();
        if (params.count("ball_metadata_key")) r->ball_metadata_key_ = params["ball_metadata_key"].get<std::string>();
        if (params.count("player_metadata_key")) r->player_metadata_key_ = params["player_metadata_key"].get<std::string>();
        if (params.count("seg_metadata_key")) r->seg_metadata_key_ = params["seg_metadata_key"].get<std::string>();
        if (params.count("output_metadata_key")) r->output_metadata_key_ = params["output_metadata_key"].get<std::string>();
        if (params.count("seg_side_data_slot")) r->seg_side_data_slot_ = params["seg_side_data_slot"].get<int>();
        if (params.count("mask_threshold")) r->mask_threshold_ = params["mask_threshold"].get<float>();
        if (params.count("line_search_radius_rows")) r->line_search_radius_rows_ = std::max(0, params["line_search_radius_rows"].get<int>());
        if (params.count("line_inside_margin_px")) r->line_inside_margin_px_ = params["line_inside_margin_px"].get<float>();
        if (params.count("inside_three_fallback_radius_rel")) r->inside_three_fallback_radius_rel_ = params["inside_three_fallback_radius_rel"].get<float>();
        if (params.count("paint_depth_rel")) r->paint_depth_rel_ = params["paint_depth_rel"].get<float>();
        if (params.count("paint_half_height_rel")) r->paint_half_height_rel_ = params["paint_half_height_rel"].get<float>();
        if (params.count("corner_baseline_rel")) r->corner_baseline_rel_ = params["corner_baseline_rel"].get<float>();
        if (params.count("corner_sideline_rel")) r->corner_sideline_rel_ = params["corner_sideline_rel"].get<float>();
        if (params.count("top_band_rel")) r->top_band_rel_ = params["top_band_rel"].get<float>();
        if (params.count("debug_log_every_n")) r->debug_log_every_n_ = params["debug_log_every_n"].get<int>();
        return r;
    }
};

DECLNODE(court_zone, CourtZone)
