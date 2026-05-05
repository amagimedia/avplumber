#include "../../node_common.hpp"
#include "../common/yolo_side_data.hpp"

extern "C" {
#include <libavutil/dict.h>
#include <libavutil/frame.h>
}

#include <algorithm>
#include <cmath>
#include <deque>
#include <limits>
#include <string>
#include <vector>

class ShotAttemptDetector : public NodeSISO<av::VideoFrame, av::VideoFrame>, public ReportsFinishByFlag {
    std::string ball_metadata_key_ = "yolo_ball";
    std::string player_metadata_key_ = "yolo_players";
    std::string handler_metadata_key_ = "ball_handler";
    std::string feet_metadata_key_ = "player_feet";
    std::string camera_shot_metadata_key_ = "camera_shot_info";
    std::string seg_metadata_key_ = "yolo_seg";
    std::string output_metadata_key_ = "shot_events";
    double min_flight_speed_ = 6.0;
    double max_vy_for_release_ = -2.0;
    double hoop_arrival_dist_ = 40.0;
    int seg_side_data_slot_ = 0;
    double mask_threshold_ = 0.5;
    double line_inside_margin_px_ = 12.0;
    int line_search_radius_rows_ = 16;
    int line_barrier_radius_px_ = 2;
    double foot_court_snap_radius_px_ = 36.0;
    double hoop_seed_inward_px_ = 42.0;
    double hoop_seed_down_px_ = 48.0;
    double hoop_seed_search_radius_px_ = 240.0;
    uint64_t shooter_hold_frames_ = 20;
    int shooter_confirm_frames_ = 2;
    int outcome_vector_window_frames_ = 6;
    double outcome_min_speed_ = 2.0;
    double outcome_min_downward_vy_ = 1.5;
    double outcome_min_downward_ratio_ = 0.45;
    int release_confirm_frames_ = 3;
    int cooldown_frames_ = 30;
    double hoop_min_conf_ = 0.3;
    int debug_log_every_n_ = 0;

    uint64_t frame_counter_ = 0;
    bool in_flight_ = false;
    uint64_t flight_start_frame_ = 0;
    int flight_frames_ = 0;
    bool release_emitted_ = false;
    bool hoop_emitted_ = false;
    uint64_t cooldown_until_ = 0;
    int total_releases_ = 0;
    int total_hoop_arrivals_ = 0;
    uint64_t last_handler_frame_ = 0;
    double prev_hoop_dist_ = 1e9;
    int pending_attempt_points_ = 0;
    std::string pending_attempt_type_;
    std::string pending_attempt_type_source_;
    double pending_attempt_confidence_ = -1.0;
    double pending_three_point_line_signed_distance_px_ = 0.0;
    double pending_three_point_line_y_delta_px_ = 0.0;
    std::string pending_three_point_line_relation_;

    struct FlightSample {
        uint64_t frame = 0;
        double cx = 0.0;
        double cy = 0.0;
        double vx = 0.0;
        double vy = 0.0;
        bool has_velocity = false;
    };

    struct HandlerState {
        bool found = false;
        int track_id = -1;
        double foot_x = 0.0;
        double foot_y = 0.0;
    };

    struct MaskInfo {
        int num_masks = 0;
        int w = 0;
        int h = 0;
        const float* data = nullptr;
    };

    struct CourtContext {
        bool usable = false;
        bool has_court = false;
        bool has_line = false;
        bool has_hoop = false;
        bool hoop_right = true;
        double hoop_x = 0.0;
        double hoop_y = 0.0;
        double model_w = 960.0;
        double model_h = 544.0;
        MaskInfo masks;
        int court_mask_idx = -1;
        int line_mask_idx = -1;
    };

    struct AttemptValue {
        bool known = false;
        std::string type;
        int points = 0;
        std::string source;
        double confidence = 0.0;
        double signed_line_distance_px = 0.0; // positive = inside/on 2PT side
        double line_y_delta_px = 0.0;
        std::string line_relation;
    };

    HandlerState last_handler_;
    HandlerState stable_handler_;
    uint64_t stable_handler_frame_ = 0;
    HandlerState handler_candidate_;
    int handler_candidate_frames_ = 0;
    std::deque<FlightSample> flight_samples_;

    struct ShotOutcome {
        bool classified = false;
        std::string result;
        double confidence = 0.0;
        double vx = 0.0;
        double vy = 0.0;
    };

    static Parameters tryParse(const AVFrame* raw, const std::string& key) {
        if (!raw || !raw->metadata) return {};
        AVDictionaryEntry* entry = av_dict_get(raw->metadata, key.c_str(), nullptr, 0);
        if (!entry || !entry->value) return {};
        try { return Parameters::parse(entry->value); } catch (...) { return {}; }
    }

    struct BallState {
        bool detected = false;
        double cx = 0, cy = 0;
        double vx = 0, vy = 0;
        double speed = 0;
        std::string source;
    };

    struct HoopState {
        bool found = false;
        double cx = 0, cy = 0;
        double w = 0, h = 0;
    };

    BallState parseBall(const Parameters& md) const {
        BallState b;
        if (!md.contains("detections") || !md["detections"].is_array()) return b;
        for (const auto& det : md["detections"]) {
            if (!det.contains("xyxy") || !det["xyxy"].is_array() || det["xyxy"].size() < 4) continue;
            double x1 = det["xyxy"][0].get<double>(), y1 = det["xyxy"][1].get<double>();
            double x2 = det["xyxy"][2].get<double>(), y2 = det["xyxy"][3].get<double>();
            b.detected = true;
            b.cx = (x1 + x2) * 0.5;
            b.cy = (y1 + y2) * 0.5;
            b.source = det.value("source", std::string("detected"));
            if (det.contains("velocity_x") && det.contains("velocity_y")) {
                b.vx = det["velocity_x"].get<double>();
                b.vy = det["velocity_y"].get<double>();
                b.speed = std::sqrt(b.vx * b.vx + b.vy * b.vy);
            }
            break;
        }
        return b;
    }

    HoopState parseHoop(const Parameters& md) const {
        HoopState h;
        if (!md.contains("detections") || !md["detections"].is_array()) return h;
        for (const auto& det : md["detections"]) {
            if (det.value("label", std::string()) != "Hoop") continue;
            if (det.value("conf", 0.0) < hoop_min_conf_) continue;
            if (!det.contains("xyxy") || !det["xyxy"].is_array() || det["xyxy"].size() < 4) continue;
            double x1 = det["xyxy"][0].get<double>(), y1 = det["xyxy"][1].get<double>();
            double x2 = det["xyxy"][2].get<double>(), y2 = det["xyxy"][3].get<double>();
            h.found = true;
            h.cx = (x1 + x2) * 0.5;
            h.cy = (y1 + y2) * 0.5;
            h.w = x2 - x1;
            h.h = y2 - y1;
            break;
        }
        return h;
    }

    HandlerState parseFeetPoint(const Parameters& feet_md, int track_id) const {
        HandlerState h;
        if (track_id < 0 || !feet_md.contains("detections") || !feet_md["detections"].is_array()) return h;
        for (const auto& det : feet_md["detections"]) {
            if (!det.is_object()) continue;
            if (det.value("track_id", -1) != track_id) continue;
            if (!det.contains("foot_point") || !det["foot_point"].is_array() || det["foot_point"].size() < 2) continue;
            h.found = true;
            h.track_id = track_id;
            h.foot_x = det["foot_point"][0].get<double>();
            h.foot_y = det["foot_point"][1].get<double>();
            return h;
        }
        return h;
    }

    HandlerState parseHandler(const Parameters& handler_md, const Parameters& feet_md) const {
        HandlerState h;
        if (!handler_md.contains("detections") || !handler_md["detections"].is_array()) return h;
        for (const auto& det : handler_md["detections"]) {
            if (det.value("label", std::string()) != "BallHandler") continue;
            if (!det.contains("xyxy") || !det["xyxy"].is_array() || det["xyxy"].size() < 4) continue;
            const int track_id = det.value("track_id", -1);
            HandlerState fh = parseFeetPoint(feet_md, track_id);
            if (fh.found) return fh;
            h.found = true;
            h.track_id = track_id;
            h.foot_x = (det["xyxy"][0].get<double>() + det["xyxy"][2].get<double>()) * 0.5;
            h.foot_y = det["xyxy"][3].get<double>();
            return h;
        }
        return h;
    }

    static bool sameHandlerIdentity(const HandlerState& a, const HandlerState& b) {
        if (!a.found || !b.found) return false;
        if (a.track_id >= 0 && b.track_id >= 0) return a.track_id == b.track_id;
        const double dx = a.foot_x - b.foot_x;
        const double dy = a.foot_y - b.foot_y;
        return dx * dx + dy * dy <= 48.0 * 48.0;
    }

    void observeHandlerForShooter(const HandlerState& handler) {
        if (!handler.found) return;

        if (!handler_candidate_.found || !sameHandlerIdentity(handler_candidate_, handler)) {
            handler_candidate_ = handler;
            handler_candidate_frames_ = 1;
        } else {
            handler_candidate_ = handler;
            ++handler_candidate_frames_;
        }

        const bool same_as_stable = stable_handler_.found && sameHandlerIdentity(stable_handler_, handler);
        if (!stable_handler_.found || same_as_stable || handler_candidate_frames_ >= shooter_confirm_frames_) {
            stable_handler_ = handler;
            stable_handler_frame_ = frame_counter_;
        }
    }

    HandlerState shooterForRelease(const HandlerState& current_handler) const {
        if (stable_handler_.found && frame_counter_ >= stable_handler_frame_ &&
            frame_counter_ - stable_handler_frame_ <= shooter_hold_frames_) {
            return stable_handler_;
        }
        if (current_handler.found) return current_handler;
        if (last_handler_.found && frame_counter_ >= last_handler_frame_ &&
            frame_counter_ - last_handler_frame_ <= shooter_hold_frames_) {
            return last_handler_;
        }
        return HandlerState();
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

    static int roundToMask(double v, double model_dim, int mask_dim) {
        if (mask_dim <= 0 || model_dim <= 0.0) return 0;
        double scaled = v * (double)mask_dim / model_dim;
        return std::max(0, std::min((int)std::lround(scaled), mask_dim - 1));
    }

    CourtContext parseCourtContext(const AVFrame* raw, const Parameters& seg_md, const HoopState& hoop) const {
        CourtContext ctx;
        ctx.model_w = seg_md.value("model_width", 960.0);
        ctx.model_h = seg_md.value("model_height", 544.0);
        ctx.has_hoop = hoop.found;
        ctx.hoop_x = hoop.cx;
        ctx.hoop_y = hoop.cy;
        ctx.hoop_right = hoop.found ? (hoop.cx >= ctx.model_w * 0.5) : true;
        if (!hoop.found || !readCpuMasks(raw, seg_side_data_slot_, ctx.masks)) return ctx;

        int idx = -1;
        if (findMaskIndex(seg_md, "basketball-court", idx) && idx < ctx.masks.num_masks) {
            ctx.court_mask_idx = idx;
            ctx.has_court = true;
        } else if (ctx.masks.num_masks > 0) {
            ctx.court_mask_idx = 0;
            ctx.has_court = true;
        }

        idx = -1;
        if (findMaskIndex(seg_md, "three point line", idx) && idx < ctx.masks.num_masks) {
            ctx.line_mask_idx = idx;
            ctx.has_line = true;
        }
        ctx.usable = ctx.has_line;
        return ctx;
    }

    bool isLineBarrier(const CourtContext& ctx, int mask_x, int mask_y) const {
        if (!ctx.has_line) return false;
        const float* line_mask = maskPtr(ctx.masks, ctx.line_mask_idx);
        if (!line_mask) return false;
        const int radius = std::max(0, line_barrier_radius_px_);
        for (int y = std::max(0, mask_y - radius); y <= std::min(ctx.masks.h - 1, mask_y + radius); ++y) {
            for (int x = std::max(0, mask_x - radius); x <= std::min(ctx.masks.w - 1, mask_x + radius); ++x) {
                if (line_mask[y * ctx.masks.w + x] >= (float)mask_threshold_) return true;
            }
        }
        return false;
    }

    bool nearestCourtPixel(const CourtContext& ctx, double model_x, double model_y,
                           double search_radius_px, bool avoid_line,
                           int& out_x, int& out_y, double* dist_model_out = nullptr) const {
        if (!ctx.has_court) return false;
        const float* court_mask = maskPtr(ctx.masks, ctx.court_mask_idx);
        if (!court_mask) return false;
        const int center_x = roundToMask(model_x, ctx.model_w, ctx.masks.w);
        const int center_y = roundToMask(model_y, ctx.model_h, ctx.masks.h);
        const int rx = std::max(1, (int)std::ceil(search_radius_px * (double)ctx.masks.w / std::max(1.0, ctx.model_w)));
        const int ry = std::max(1, (int)std::ceil(search_radius_px * (double)ctx.masks.h / std::max(1.0, ctx.model_h)));
        double best_score = std::numeric_limits<double>::infinity();
        bool found = false;
        for (int y = std::max(0, center_y - ry); y <= std::min(ctx.masks.h - 1, center_y + ry); ++y) {
            for (int x = std::max(0, center_x - rx); x <= std::min(ctx.masks.w - 1, center_x + rx); ++x) {
                if (court_mask[y * ctx.masks.w + x] < (float)mask_threshold_) continue;
                if (avoid_line && isLineBarrier(ctx, x, y)) continue;
                const double px = ((double)x + 0.5) * ctx.model_w / (double)ctx.masks.w;
                const double py = ((double)y + 0.5) * ctx.model_h / (double)ctx.masks.h;
                const double dx = px - model_x;
                const double dy = py - model_y;
                const double score = dx * dx + dy * dy;
                if (score < best_score) {
                    best_score = score;
                    out_x = x;
                    out_y = y;
                    found = true;
                }
            }
        }
        if (found && dist_model_out) *dist_model_out = std::sqrt(best_score);
        return found;
    }

    bool nearestLinePoint(const CourtContext& ctx, double model_x, double model_y,
                          int& line_x_mask, int& line_y_mask,
                          double* dist_model_out = nullptr) const {
        if (!ctx.usable) return false;
        const float* line_mask = maskPtr(ctx.masks, ctx.line_mask_idx);
        if (!line_mask) return false;
        double best_score = std::numeric_limits<double>::infinity();
        bool found = false;
        for (int y = 0; y < ctx.masks.h; ++y) {
            for (int x = 0; x < ctx.masks.w; ++x) {
                if (line_mask[y * ctx.masks.w + x] < (float)mask_threshold_) continue;
                const double px = ((double)x + 0.5) * ctx.model_w / (double)ctx.masks.w;
                const double py = ((double)y + 0.5) * ctx.model_h / (double)ctx.masks.h;
                const double dx = px - model_x;
                const double dy = py - model_y;
                const double score = dx * dx + dy * dy;
                if (score < best_score) {
                    best_score = score;
                    line_x_mask = x;
                    line_y_mask = y;
                    found = true;
                }
            }
        }
        if (found && dist_model_out) *dist_model_out = std::sqrt(best_score);
        return found;
    }

    bool buildInsideThreeComponent(const CourtContext& ctx, std::vector<uint8_t>& component) const {
        if (!ctx.has_court || !ctx.has_line || !ctx.has_hoop) return false;
        const float* court_mask = maskPtr(ctx.masks, ctx.court_mask_idx);
        if (!court_mask) return false;

        const double seed_x = ctx.hoop_x + (ctx.hoop_right ? -hoop_seed_inward_px_ : hoop_seed_inward_px_);
        const double seed_y = ctx.hoop_y + hoop_seed_down_px_;
        int seed_mx = 0;
        int seed_my = 0;
        if (!nearestCourtPixel(ctx, seed_x, seed_y, hoop_seed_search_radius_px_, true, seed_mx, seed_my)) {
            if (!nearestCourtPixel(ctx, ctx.hoop_x, ctx.hoop_y, hoop_seed_search_radius_px_, true, seed_mx, seed_my)) {
                return false;
            }
        }

        component.assign((size_t)ctx.masks.w * (size_t)ctx.masks.h, 0);
        std::deque<int> q;
        const int seed_idx = seed_my * ctx.masks.w + seed_mx;
        component[seed_idx] = 1;
        q.push_back(seed_idx);

        const int dirs[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
        while (!q.empty()) {
            const int idx = q.front();
            q.pop_front();
            const int x = idx % ctx.masks.w;
            const int y = idx / ctx.masks.w;
            for (const auto& d : dirs) {
                const int nx = x + d[0];
                const int ny = y + d[1];
                if (nx < 0 || nx >= ctx.masks.w || ny < 0 || ny >= ctx.masks.h) continue;
                const int ni = ny * ctx.masks.w + nx;
                if (component[ni]) continue;
                if (court_mask[ni] < (float)mask_threshold_) continue;
                if (isLineBarrier(ctx, nx, ny)) continue;
                component[ni] = 1;
                q.push_back(ni);
            }
        }
        return true;
    }

    bool nearestLinePointAround(const CourtContext& ctx, double model_x, double model_y,
                                int& line_x_mask, int& line_y_mask) const {
        if (!ctx.usable) return false;
        const float* line_mask = maskPtr(ctx.masks, ctx.line_mask_idx);
        if (!line_mask) return false;
        const int mask_x = roundToMask(model_x, ctx.model_w, ctx.masks.w);
        const int mask_y = roundToMask(model_y, ctx.model_h, ctx.masks.h);
        const int radius = std::max(0, line_search_radius_rows_);
        double best_score = std::numeric_limits<double>::infinity();
        bool found = false;
        for (int y = std::max(0, mask_y - radius); y <= std::min(ctx.masks.h - 1, mask_y + radius); ++y) {
            for (int x = 0; x < ctx.masks.w; ++x) {
                if (line_mask[y * ctx.masks.w + x] < (float)mask_threshold_) continue;
                const double dx = (double)x - (double)mask_x;
                const double dy = (double)y - (double)mask_y;
                const double score = dx * dx + dy * dy * 0.35;
                if (score < best_score) {
                    best_score = score;
                    line_x_mask = x;
                    line_y_mask = y;
                    found = true;
                }
            }
        }
        return found;
    }

    AttemptValue classifyAttemptValue(const CourtContext& ctx, const HandlerState& shooter) const {
        AttemptValue out;
        if (!ctx.usable || !shooter.found) return out;

        std::vector<uint8_t> inside_component;
        if (buildInsideThreeComponent(ctx, inside_component)) {
            int foot_x_mask = 0;
            int foot_y_mask = 0;
            double snap_dist = 0.0;
            const bool have_foot_court = nearestCourtPixel(ctx, shooter.foot_x, shooter.foot_y,
                                                           foot_court_snap_radius_px_, true,
                                                           foot_x_mask, foot_y_mask, &snap_dist);
            const int raw_foot_x = roundToMask(shooter.foot_x, ctx.model_w, ctx.masks.w);
            const int raw_foot_y = roundToMask(shooter.foot_y, ctx.model_h, ctx.masks.h);
            const bool on_line = isLineBarrier(ctx, raw_foot_x, raw_foot_y) ||
                                 (have_foot_court && isLineBarrier(ctx, foot_x_mask, foot_y_mask));
            if (have_foot_court || on_line) {
                if (!have_foot_court) {
                    foot_x_mask = raw_foot_x;
                    foot_y_mask = raw_foot_y;
                }
                const bool inside_three = on_line ||
                    inside_component[(size_t)foot_y_mask * (size_t)ctx.masks.w + (size_t)foot_x_mask] != 0;

                int line_x_mask = 0;
                int line_y_mask = 0;
                double line_dist = 0.0;
                const bool have_line_point =
                    nearestLinePoint(ctx, shooter.foot_x, shooter.foot_y, line_x_mask, line_y_mask, &line_dist);
                double y_delta = 0.0;
                if (have_line_point) {
                    const double line_y_model = ((double)line_y_mask + 0.5) * ctx.model_h / (double)ctx.masks.h;
                    y_delta = std::abs(shooter.foot_y - line_y_model);
                }

                out.known = true;
                out.type = inside_three ? "2pt" : "3pt";
                out.points = inside_three ? 2 : 3;
                out.source = "feet_three_point_line_floodfill";
                out.signed_line_distance_px = inside_three ? line_dist : -line_dist;
                out.line_y_delta_px = y_delta;
                if (line_dist <= line_inside_margin_px_) out.line_relation = "on_or_near_line";
                else out.line_relation = inside_three ? "inside_arc" : "behind_arc";

                const double margin_conf =
                    std::min(1.0, line_dist / std::max(1.0, line_inside_margin_px_ * 3.0));
                const double snap_conf = have_foot_court
                    ? 1.0 - std::min(1.0, snap_dist / std::max(1.0, foot_court_snap_radius_px_))
                    : 0.4;
                out.confidence = std::max(0.0, std::min(1.0, 0.55 * margin_conf + 0.45 * snap_conf));
                return out;
            }
        }

        int line_x_mask = 0;
        int line_y_mask = 0;
        if (!nearestLinePointAround(ctx, shooter.foot_x, shooter.foot_y, line_x_mask, line_y_mask)) return out;
        const double line_x_model = ((double)line_x_mask + 0.5) * ctx.model_w / (double)ctx.masks.w;
        const double line_y_model = ((double)line_y_mask + 0.5) * ctx.model_h / (double)ctx.masks.h;
        const double signed_inside_px = ctx.hoop_right
            ? shooter.foot_x - line_x_model
            : line_x_model - shooter.foot_x;
        const bool inside_three = signed_inside_px >= -line_inside_margin_px_;
        const double abs_margin = std::abs(signed_inside_px);
        const double y_delta = std::abs(shooter.foot_y - line_y_model);
        out.known = true;
        out.type = inside_three ? "2pt" : "3pt";
        out.points = inside_three ? 2 : 3;
        out.source = "feet_three_point_line_nearest";
        out.signed_line_distance_px = signed_inside_px;
        out.line_y_delta_px = y_delta;
        if (std::abs(signed_inside_px) <= line_inside_margin_px_) out.line_relation = "on_or_near_line";
        else out.line_relation = inside_three ? "inside_arc" : "behind_arc";
        const double margin_conf = std::min(1.0, abs_margin / std::max(1.0, line_inside_margin_px_ * 3.0));
        const double row_conf = 1.0 - std::min(1.0, y_delta / std::max(1.0, (double)line_search_radius_rows_ * ctx.model_h / std::max(1, ctx.masks.h)));
        out.confidence = std::max(0.0, std::min(1.0, 0.65 * margin_conf + 0.35 * row_conf));
        return out;
    }

    void resetFlight() {
        in_flight_ = false;
        flight_frames_ = 0;
        release_emitted_ = false;
        hoop_emitted_ = false;
        prev_hoop_dist_ = 1e9;
        pending_attempt_points_ = 0;
        pending_attempt_type_.clear();
        pending_attempt_type_source_.clear();
        pending_attempt_confidence_ = -1.0;
        pending_three_point_line_signed_distance_px_ = 0.0;
        pending_three_point_line_y_delta_px_ = 0.0;
        pending_three_point_line_relation_.clear();
        flight_samples_.clear();
    }

    void recordFlightSample(const BallState& ball) {
        if (!ball.detected) return;
        FlightSample s;
        s.frame = frame_counter_;
        s.cx = ball.cx;
        s.cy = ball.cy;
        s.vx = ball.vx;
        s.vy = ball.vy;
        s.has_velocity = ball.speed > 0.0;
        flight_samples_.push_back(s);

        const uint64_t keep_frames = (uint64_t)std::max(1, outcome_vector_window_frames_ * 3);
        while (!flight_samples_.empty() && frame_counter_ - flight_samples_.front().frame > keep_frames) {
            flight_samples_.pop_front();
        }
    }

    ShotOutcome classifyNearHoopOutcome(const BallState& ball) const {
        ShotOutcome out;
        if (!ball.detected) return out;

        double vx = ball.vx;
        double vy = ball.vy;
        double speed = ball.speed;

        if (speed < outcome_min_speed_) {
            const FlightSample* first = nullptr;
            for (const auto& s : flight_samples_) {
                if (frame_counter_ - s.frame <= (uint64_t)std::max(1, outcome_vector_window_frames_)) {
                    first = &s;
                    break;
                }
            }
            if (first && first->frame != frame_counter_) {
                vx = ball.cx - first->cx;
                vy = ball.cy - first->cy;
                speed = std::sqrt(vx * vx + vy * vy);
            }
        }

        if (speed < outcome_min_speed_) return out;

        const double downward_ratio = vy / std::max(speed, 1e-6);
        const bool downward = vy >= outcome_min_downward_vy_ &&
                              downward_ratio >= outcome_min_downward_ratio_;
        out.classified = true;
        out.result = downward ? "scored" : "missed";
        out.vx = vx;
        out.vy = vy;
        out.confidence = std::min(1.0, std::max(0.0, std::abs(downward_ratio)));
        return out;
    }

public:
    using NodeSISO<av::VideoFrame, av::VideoFrame>::NodeSISO;
    bool consumeEofIfPresent() override {
        return false;
    }

    void process() override {
        av::VideoFrame frm = this->source_->get();

        if (isEofMarker(frm)) {
            logstream << "shot_attempt_detector: total releases=" << total_releases_
                      << " hoop_arrivals=" << total_hoop_arrivals_
                      << " in " << frame_counter_ << " frames";
            frame_counter_ = 0;
            resetFlight();
            cooldown_until_ = 0;
            total_releases_ = 0;
            total_hoop_arrivals_ = 0;
            last_handler_frame_ = 0;
            stable_handler_frame_ = 0;
            last_handler_ = HandlerState();
            stable_handler_ = HandlerState();
            handler_candidate_ = HandlerState();
            handler_candidate_frames_ = 0;
            this->sink_->put(frm);
            this->finished_ = true;
            return;
        }
        if (!frm) return;

        ++frame_counter_;

        const AVFrame* raw = frm.raw();
        auto ball_md = tryParse(raw, ball_metadata_key_);
        auto player_md = tryParse(raw, player_metadata_key_);
        auto handler_md = tryParse(raw, handler_metadata_key_);
        auto feet_md = tryParse(raw, feet_metadata_key_);
        auto shot_md = tryParse(raw, camera_shot_metadata_key_);

        bool wide_shot = shot_md.value("camera_shot_type", std::string()) == "wide";

        BallState ball = parseBall(ball_md);
        HoopState hoop = parseHoop(player_md);
        HandlerState handler_state = parseHandler(handler_md, feet_md);
        bool handler = handler_state.found;
        if (handler) {
            last_handler_ = handler_state;
            last_handler_frame_ = frame_counter_;
            observeHandlerForShooter(handler_state);
        }

        auto seg_md = tryParse(raw, seg_metadata_key_);

        Parameters events;
        events["release"] = false;
        events["hoop_arrival"] = false;
        events["in_flight"] = false;
        events["ball_detected"] = ball.detected;
        events["hoop_detected"] = hoop.found;
        events["total_releases"] = total_releases_;
        events["total_hoop_arrivals"] = total_hoop_arrivals_;

        bool in_cooldown = frame_counter_ < cooldown_until_;

        if (handler || !ball.detected) {
            if (in_flight_ && !hoop_emitted_) {
                resetFlight();
            }
        }

        bool ball_in_air = ball.detected && !handler;
        events["ball_in_air"] = ball_in_air;
        if (ball_in_air || in_flight_) recordFlightSample(ball);

        double hoop_dist = 1e9;
        if (ball.detected && hoop.found) {
            double dx = ball.cx - hoop.cx, dy = ball.cy - hoop.cy;
            hoop_dist = std::sqrt(dx * dx + dy * dy);
            events["ball_hoop_dist"] = (int)std::round(hoop_dist);
        }

        if (!in_cooldown && wide_shot && ball_in_air) {
            bool has_arc = ball.vy <= max_vy_for_release_;
            bool approaching_hoop = hoop.found && hoop_dist < prev_hoop_dist_ && hoop_dist < 120.0;
            if (!in_flight_ && ball.speed >= min_flight_speed_ && (has_arc || approaching_hoop)) {
                in_flight_ = true;
                flight_start_frame_ = frame_counter_;
                flight_frames_ = 0;
            }
            if (in_flight_) flight_frames_++;

            if (flight_frames_ >= release_confirm_frames_ && !release_emitted_) {
                HandlerState shooter = shooterForRelease(handler_state);
                const CourtContext court = parseCourtContext(raw, seg_md, hoop);
                const AttemptValue attempt = classifyAttemptValue(court, shooter);
                pending_attempt_type_ = attempt.known ? attempt.type : std::string("unknown");
                pending_attempt_points_ = attempt.known ? attempt.points : 0;
                pending_attempt_type_source_ = attempt.known ? attempt.source : std::string();
                pending_attempt_confidence_ = attempt.known ? attempt.confidence : -1.0;
                pending_three_point_line_signed_distance_px_ =
                    attempt.known ? attempt.signed_line_distance_px : 0.0;
                pending_three_point_line_y_delta_px_ = attempt.known ? attempt.line_y_delta_px : 0.0;
                pending_three_point_line_relation_ = attempt.known ? attempt.line_relation : std::string();
                release_emitted_ = true;
                total_releases_++;
                events["release"] = true;
                events["release_frame"] = flight_start_frame_;
                events["total_releases"] = total_releases_;
                if (shooter.track_id >= 0) events["shooter_id"] = shooter.track_id;
                events["attempt_type"] = pending_attempt_type_;
                if (pending_attempt_points_ > 0) events["attempt_points"] = pending_attempt_points_;
                if (attempt.known) {
                    events["attempt_type_source"] = attempt.source;
                    events["attempt_confidence"] = std::round(attempt.confidence * 1000.0) / 1000.0;
                    events["three_point_line_signed_distance_px"] =
                        (int)std::lround(attempt.signed_line_distance_px);
                    events["three_point_line_y_delta_px"] =
                        (int)std::lround(attempt.line_y_delta_px);
                    events["three_point_line_relation"] = attempt.line_relation;
                }
                logstream << "shot_attempt_detector: RELEASE frame=" << flight_start_frame_
                          << " ball=[" << (int)ball.cx << "," << (int)ball.cy << "]"
                          << " speed=" << (int)ball.speed
                          << " vel=[" << (int)ball.vx << "," << (int)ball.vy << "]"
                          << " attempt=" << pending_attempt_type_
                          << (attempt.known ? " source=" + attempt.source : "");
            }
        }

        if (in_flight_ && release_emitted_ && !hoop_emitted_ && ball.detected && hoop.found) {
            double dx = ball.cx - hoop.cx;
            double dy = ball.cy - hoop.cy;
            double dist = std::sqrt(dx * dx + dy * dy);

            if (dist <= hoop_arrival_dist_) {
                ShotOutcome outcome = classifyNearHoopOutcome(ball);
                hoop_emitted_ = true;
                total_hoop_arrivals_++;
                events["hoop_arrival"] = true;
                events["ball_hoop_dist"] = (int)std::round(dist);
                events["total_hoop_arrivals"] = total_hoop_arrivals_;
                if (outcome.classified) {
                    events["shot_result"] = true;
                    events["result"] = outcome.result;
                    events["result_source"] = "ball_vector";
                    events["result_conf"] = outcome.confidence;
                    events["result_vx"] = outcome.vx;
                    events["result_vy"] = outcome.vy;
                    events["attempt_type"] = pending_attempt_type_.empty() ? std::string("unknown") : pending_attempt_type_;
                    if (pending_attempt_points_ > 0) {
                        events["attempt_points"] = pending_attempt_points_;
                        if (outcome.result == "scored") events["points"] = pending_attempt_points_;
                    }
                    if (!pending_attempt_type_source_.empty()) {
                        events["attempt_type_source"] = pending_attempt_type_source_;
                        events["attempt_confidence"] =
                            std::round(pending_attempt_confidence_ * 1000.0) / 1000.0;
                        events["three_point_line_signed_distance_px"] =
                            (int)std::lround(pending_three_point_line_signed_distance_px_);
                        events["three_point_line_y_delta_px"] =
                            (int)std::lround(pending_three_point_line_y_delta_px_);
                        events["three_point_line_relation"] = pending_three_point_line_relation_;
                    }
                }
                cooldown_until_ = frame_counter_ + (uint64_t)cooldown_frames_;
                logstream << "shot_attempt_detector: HOOP_ARRIVAL frame=" << frame_counter_
                          << " ball=[" << (int)ball.cx << "," << (int)ball.cy << "]"
                          << " hoop=[" << (int)hoop.cx << "," << (int)hoop.cy << "]"
                          << " dist=" << (int)dist
                          << " flight_frames=" << flight_frames_
                          << (outcome.classified ? " result=" + outcome.result : " result=unknown");
                resetFlight();
            }
        }

        events["in_flight"] = in_flight_ && release_emitted_;
        prev_hoop_dist_ = hoop_dist;

        std::string serialized = events.dump();
        av_dict_set(&frm.raw()->metadata, output_metadata_key_.c_str(), serialized.c_str(), 0);

        if (debug_log_every_n_ > 0 && (frame_counter_ % (uint64_t)debug_log_every_n_) == 0) {
            logstream << "shot_attempt_detector: frame=" << frame_counter_
                      << " flight=" << in_flight_
                      << " releases=" << total_releases_
                      << " hoop_arrivals=" << total_hoop_arrivals_;
        }

        this->sink_->put(frm);
    }

    static std::shared_ptr<ShotAttemptDetector> create(NodeCreationInfo& nci) {
        EdgeManager& edges = nci.edges;
        const Parameters& params = nci.params;
        auto r = NodeSISO<av::VideoFrame, av::VideoFrame>::template createCommon<ShotAttemptDetector>(edges, params);

        if (params.count("ball_metadata_key")) r->ball_metadata_key_ = params["ball_metadata_key"].get<std::string>();
        if (params.count("player_metadata_key")) r->player_metadata_key_ = params["player_metadata_key"].get<std::string>();
        if (params.count("handler_metadata_key")) r->handler_metadata_key_ = params["handler_metadata_key"].get<std::string>();
        if (params.count("feet_metadata_key")) r->feet_metadata_key_ = params["feet_metadata_key"].get<std::string>();
        if (params.count("camera_shot_metadata_key")) r->camera_shot_metadata_key_ = params["camera_shot_metadata_key"].get<std::string>();
        if (params.count("seg_metadata_key")) r->seg_metadata_key_ = params["seg_metadata_key"].get<std::string>();
        if (params.count("output_metadata_key")) r->output_metadata_key_ = params["output_metadata_key"].get<std::string>();
        if (params.count("min_flight_speed")) r->min_flight_speed_ = params["min_flight_speed"].get<double>();
        if (params.count("max_vy_for_release")) r->max_vy_for_release_ = params["max_vy_for_release"].get<double>();
        if (params.count("hoop_arrival_dist")) r->hoop_arrival_dist_ = params["hoop_arrival_dist"].get<double>();
        if (params.count("seg_side_data_slot")) r->seg_side_data_slot_ = params["seg_side_data_slot"].get<int>();
        if (params.count("mask_threshold")) r->mask_threshold_ = params["mask_threshold"].get<double>();
        if (params.count("line_inside_margin_px")) r->line_inside_margin_px_ = params["line_inside_margin_px"].get<double>();
        if (params.count("line_search_radius_rows")) r->line_search_radius_rows_ = std::max(0, params["line_search_radius_rows"].get<int>());
        if (params.count("line_barrier_radius_px")) r->line_barrier_radius_px_ = std::max(0, params["line_barrier_radius_px"].get<int>());
        if (params.count("foot_court_snap_radius_px")) r->foot_court_snap_radius_px_ = std::max(0.0, params["foot_court_snap_radius_px"].get<double>());
        if (params.count("hoop_seed_inward_px")) r->hoop_seed_inward_px_ = std::max(0.0, params["hoop_seed_inward_px"].get<double>());
        if (params.count("hoop_seed_down_px")) r->hoop_seed_down_px_ = params["hoop_seed_down_px"].get<double>();
        if (params.count("hoop_seed_search_radius_px")) r->hoop_seed_search_radius_px_ = std::max(1.0, params["hoop_seed_search_radius_px"].get<double>());
        if (params.count("shooter_hold_frames")) r->shooter_hold_frames_ = params["shooter_hold_frames"].get<uint64_t>();
        if (params.count("shooter_confirm_frames")) r->shooter_confirm_frames_ = std::max(1, params["shooter_confirm_frames"].get<int>());
        if (params.count("outcome_vector_window_frames")) r->outcome_vector_window_frames_ = std::max(1, params["outcome_vector_window_frames"].get<int>());
        if (params.count("outcome_min_speed")) r->outcome_min_speed_ = params["outcome_min_speed"].get<double>();
        if (params.count("outcome_min_downward_vy")) r->outcome_min_downward_vy_ = params["outcome_min_downward_vy"].get<double>();
        if (params.count("outcome_min_downward_ratio")) r->outcome_min_downward_ratio_ = params["outcome_min_downward_ratio"].get<double>();
        if (params.count("release_confirm_frames")) r->release_confirm_frames_ = params["release_confirm_frames"].get<int>();
        if (params.count("cooldown_frames")) r->cooldown_frames_ = params["cooldown_frames"].get<int>();
        if (params.count("hoop_min_conf")) r->hoop_min_conf_ = params["hoop_min_conf"].get<double>();
        if (params.count("debug_log_every_n")) r->debug_log_every_n_ = params["debug_log_every_n"].get<int>();

        return r;
    }
};

DECLNODE(shot_attempt_detector, ShotAttemptDetector)
