#include "cuda_overlay_base.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <deque>
#include <limits>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "../../../../objs/src/nodes/neural_net/draw/draw_tactical_court.ptx.h"

using cuda_overlay::DrawColor;

// Renders the tactical (top-down) court panel. The ground-plane homography is
// computed upstream by the court_calibration node (pyplumber) and consumed
// here via `court_calib` frame metadata; this node only projects and draws.
// Player foot points come from player_feet_seg metadata (model coordinates).

namespace {

struct Point2D {
    double x = 0.0;
    double y = 0.0;
};

struct TacticalPoint {
    float x;
    float y;
    float radius;
    int y_color;
    int u_color;
    int v_color;
};

struct PlayerDot {
    int id = -1;
    std::string team;
    bool has_ball = false;
    Point2D foot;
};

struct ProjectedPlayerDot {
    PlayerDot player;
    Point2D projected;
};

struct PlayerRenderState {
    Point2D projected;
    std::string team;
    bool has_ball = false;
    int seen_frames = 0;
    int missed_frames = 0;
    bool visible = false;
};

struct TrailPoint {
    float x = 0.0f;
    float y = 0.0f;
    std::string team;
};

struct PanelGeometry {
    int panel_x = 0;
    int panel_y = 0;
    int panel_w = 0;
    int panel_h = 0;
    int court_x = 0;
    int court_y = 0;
    int court_w = 0;
    int court_h = 0;
};

struct CourtMapper {
    double source_to_court[9] = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
    double source_w = 1.0;
    double source_h = 1.0;
    bool hoop_on_left = true;
    int court_x = 0;
    int court_y = 0;
    int court_w = 0;
    int court_h = 0;
};

Point2D operator*(const Point2D& a, double s) {
    return Point2D{a.x * s, a.y * s};
}

Point2D operator+(const Point2D& a, const Point2D& b) {
    return Point2D{a.x + b.x, a.y + b.y};
}

double clampDouble(double v, double lo, double hi) {
    return std::max(lo, std::min(hi, v));
}

bool parsePointArray(const Parameters& arr, Point2D& out) {
    if (!arr.is_array() || arr.size() < 2) return false;
    out.x = arr[0].get<double>();
    out.y = arr[1].get<double>();
    return std::isfinite(out.x) && std::isfinite(out.y);
}

} // namespace

class DrawTacticalCourt : public CudaOverlayBase {
    std::string calib_metadata_key_ = "court_calib";
    std::string feet_metadata_key_ = "player_feet";
    int panel_width_ = 360;
    int panel_height_ = 220;
    int padding_left_ = 28;
    int padding_bottom_ = 116;
    int inner_padding_ = 14;
    int line_thickness_ = 2;
    int trail_length_ = 18;
    int max_players_per_team_ = 5;
    int max_player_dots_ = 10;
    bool show_trails_ = false;
    bool show_unknown_players_ = false;
    double min_foot_confidence_ = 0.15;
    int overlay_hold_frames_ = 30;
    int player_min_seen_frames_ = 2;
    int player_hold_frames_ = 8;
    double player_smoothing_alpha_ = 0.35;
    double player_max_jump_px_ = 46.0;
    float background_opacity_ = 0.58f;
    int debug_log_every_n_ = 0;

    DrawColor background_color_{16, 128, 128};
    DrawColor court_line_color_{235, 128, 128};
    DrawColor three_point_color_{210, 16, 146};
    DrawColor hoop_color_{156, 44, 200};
    DrawColor team_a_color_{169, 166, 16};
    DrawColor team_b_color_{173, 42, 26};
    DrawColor unknown_color_{200, 128, 128};

    CUdeviceptr d_points_ = 0;
    size_t d_points_capacity_ = 0;
    std::unordered_map<int, std::deque<TrailPoint>> trails_;
    bool last_hoop_on_left_ = true;
    std::unordered_map<int, PlayerRenderState> player_states_;
    std::vector<TacticalPoint> cached_points_;
    bool have_cached_overlay_ = false;
    int cached_overlay_age_ = 0;
    PanelGeometry cached_geom_;
    int cached_output_w_ = 0;
    int cached_output_h_ = 0;
    int cached_active_hoop_on_left_ = 1;

    const char* nodeName() const override { return "draw_tactical_court"; }

    void onKernelsUnloaded() override {
        if (d_points_) {
            cuMemFree(d_points_);
            d_points_ = 0;
            d_points_capacity_ = 0;
        }
    }

    DrawColor teamColor(const std::string& team) const {
        if (team == "A") return team_a_color_;
        if (team == "B") return team_b_color_;
        return unknown_color_;
    }

    PanelGeometry makePanelGeometry(const av::VideoFrame& output) const {
        PanelGeometry g;
        g.panel_w = std::min(panel_width_, std::max(1, output.width() - padding_left_ * 2));
        g.panel_h = std::min(panel_height_, std::max(1, output.height() - padding_bottom_ - 8));
        g.panel_x = std::max(0, padding_left_);
        g.panel_y = std::max(0, output.height() - padding_bottom_ - g.panel_h);
        g.court_x = g.panel_x + inner_padding_;
        g.court_y = g.panel_y + inner_padding_;
        g.court_w = std::max(16, g.panel_w - inner_padding_ * 2);
        g.court_h = std::max(16, g.panel_h - inner_padding_ * 2);
        return g;
    }

    bool readMetadata(const AVFrame* raw, const std::string& key, Parameters& out) const {
        if (!raw || !raw->metadata) return false;
        AVDictionaryEntry* entry = av_dict_get(raw->metadata, key.c_str(), nullptr, 0);
        if (!entry || !entry->value) return false;
        try {
            out = Parameters::parse(entry->value);
            return out.is_object();
        } catch (const std::exception&) {
            return false;
        }
    }

    bool parseCalibration(const av::VideoFrame& frm,
                          const PanelGeometry& geom,
                          CourtMapper& mapper) const {
        Parameters md;
        if (!readMetadata(frm.raw(), calib_metadata_key_, md)) return false;
        if (!md.value("valid", false)) return false;
        if (!md.contains("h") || !md["h"].is_array() || md["h"].size() < 9) return false;
        for (int i = 0; i < 9; ++i) {
            mapper.source_to_court[i] = md["h"][(size_t)i].get<double>();
            if (!std::isfinite(mapper.source_to_court[i])) return false;
        }
        mapper.source_w = md.value("source_w", (double)frm.width());
        mapper.source_h = md.value("source_h", (double)frm.height());
        if (mapper.source_w <= 1.0 || mapper.source_h <= 1.0) return false;
        mapper.hoop_on_left = md.value("hoop_on_left", true);
        mapper.court_x = geom.court_x;
        mapper.court_y = geom.court_y;
        mapper.court_w = geom.court_w;
        mapper.court_h = geom.court_h;
        return true;
    }

    void parsePlayers(const av::VideoFrame& frm, std::vector<PlayerDot>& players) const {
        players.clear();
        Parameters md;
        if (!readMetadata(frm.raw(), feet_metadata_key_, md)) return;
        if (!md.contains("detections") || !md["detections"].is_array()) return;
        const double mw = md.value("model_width", 960.0);
        const double mh = md.value("model_height", 544.0);
        if (mw <= 1.0 || mh <= 1.0) return;
        const double sx = (double)frm.width() / mw;
        const double sy = (double)frm.height() / mh;
        for (const auto& det : md["detections"]) {
            if (!det.is_object()) continue;
            if (!det.value("valid", false)) continue;
            if (det.value("conf", 0.0) < min_foot_confidence_) continue;
            if (!det.contains("foot_point")) continue;
            PlayerDot dot;
            if (!parsePointArray(det["foot_point"], dot.foot)) continue;
            dot.foot.x *= sx;
            dot.foot.y *= sy;
            dot.id = det.value("track_id", -1);
            dot.team = det.value("team_ab", std::string());
            players.push_back(dot);
        }
    }

    Point2D courtFtToPanel(double x_ft, double y_ft, const CourtMapper& mapper) const {
        return Point2D{
            (double)mapper.court_x + (x_ft / 94.0) * (double)mapper.court_w,
            (double)mapper.court_y + (y_ft / 50.0) * (double)mapper.court_h
        };
    }

    bool sourcePointToPanel(const Point2D& src, const CourtMapper& mapper, Point2D& out) const {
        const double x = src.x / mapper.source_w;
        const double y = src.y / mapper.source_h;
        const double* h = mapper.source_to_court;
        const double den = h[6] * x + h[7] * y + h[8];
        if (std::fabs(den) < 1e-9 || !std::isfinite(den)) return false;
        const double cx = (h[0] * x + h[1] * y + h[2]) / den;
        const double cy = (h[3] * x + h[4] * y + h[5]) / den;
        if (!std::isfinite(cx) || !std::isfinite(cy)) return false;
        if (cx < -0.05 || cx > 1.05 || cy < -0.05 || cy > 1.05) return false;
        const double x_ft = clampDouble(cx * 94.0, 0.0, 94.0);
        const double y_ft = clampDouble(cy * 50.0, 0.0, 50.0);
        out = courtFtToPanel(x_ft, y_ft, mapper);
        return std::isfinite(out.x) && std::isfinite(out.y);
    }

    void appendCourtSegment(double x1,
                            double y1,
                            double x2,
                            double y2,
                            const CourtMapper& mapper,
                            const DrawColor& color,
                            float radius,
                            std::vector<TacticalPoint>& points) const {
        const double dx = x2 - x1;
        const double dy = y2 - y1;
        const int steps = std::max(2, (int)std::ceil(std::hypot(dx, dy) / 1.25));
        for (int i = 0; i <= steps; ++i) {
            const double t = (double)i / (double)steps;
            const Point2D p = courtFtToPanel(x1 + dx * t, y1 + dy * t, mapper);
            points.push_back(TacticalPoint{(float)p.x, (float)p.y, radius, color.y, color.u, color.v});
        }
    }

    void appendCourtArc(double cx,
                        double cy,
                        double r,
                        double theta_from,
                        double theta_to,
                        const CourtMapper& mapper,
                        const DrawColor& color,
                        float radius,
                        std::vector<TacticalPoint>& points) const {
        const int steps = std::max(12, (int)std::ceil(std::fabs(theta_to - theta_from) * r / 1.0));
        for (int i = 0; i <= steps; ++i) {
            const double t = theta_from + (theta_to - theta_from) * (double)i / (double)steps;
            const Point2D p = courtFtToPanel(cx + r * std::cos(t), cy + r * std::sin(t), mapper);
            points.push_back(TacticalPoint{(float)p.x, (float)p.y, radius, color.y, color.u, color.v});
        }
    }

    void appendThreePointHalf(bool left,
                              const CourtMapper& mapper,
                              const DrawColor& color,
                              float radius,
                              std::vector<TacticalPoint>& points) const {
        const double hoop_x = left ? 5.25 : 88.75;
        const double hoop_y = 25.0;
        const double r = 23.75;
        const double corner_y_top = 3.0;
        const double corner_y_bottom = 47.0;
        const double corner_lat = 22.0;
        const double theta_max = std::asin(corner_lat / r);
        const double corner_dx = std::sqrt(std::max(0.0, r * r - corner_lat * corner_lat));
        const double corner_x = left ? hoop_x + corner_dx : hoop_x - corner_dx;

        if (left) {
            appendCourtSegment(0.0, corner_y_top, corner_x, corner_y_top, mapper, color, radius, points);
            appendCourtSegment(0.0, corner_y_bottom, corner_x, corner_y_bottom, mapper, color, radius, points);
        } else {
            appendCourtSegment(corner_x, corner_y_top, 94.0, corner_y_top, mapper, color, radius, points);
            appendCourtSegment(corner_x, corner_y_bottom, 94.0, corner_y_bottom, mapper, color, radius, points);
        }

        const int arc_steps = 72;
        for (int i = 0; i <= arc_steps; ++i) {
            const double t = -theta_max + (2.0 * theta_max * (double)i / (double)arc_steps);
            const double x = left ? hoop_x + r * std::cos(t) : hoop_x - r * std::cos(t);
            const double y = hoop_y + r * std::sin(t);
            const Point2D p = courtFtToPanel(x, y, mapper);
            points.push_back(TacticalPoint{(float)p.x, (float)p.y, radius, color.y, color.u, color.v});
        }

        const Point2D hoop_p = courtFtToPanel(hoop_x, hoop_y, mapper);
        points.push_back(TacticalPoint{(float)hoop_p.x, (float)hoop_p.y, left == mapper.hoop_on_left ? 3.6f : 2.2f,
                                       (left == mapper.hoop_on_left ? hoop_color_ : color).y,
                                       (left == mapper.hoop_on_left ? hoop_color_ : color).u,
                                       (left == mapper.hoop_on_left ? hoop_color_ : color).v});
    }

    void appendPaintHalf(bool left,
                         const CourtMapper& mapper,
                         const DrawColor& color,
                         float radius,
                         std::vector<TacticalPoint>& points) const {
        // NBA key: 16 ft wide, free-throw line 19 ft from baseline, circle r=6.
        const double ft_line = left ? 19.0 : 75.0;
        const double base = left ? 0.0 : 94.0;
        appendCourtSegment(base, 17.0, ft_line, 17.0, mapper, color, radius, points);
        appendCourtSegment(base, 33.0, ft_line, 33.0, mapper, color, radius, points);
        appendCourtSegment(ft_line, 17.0, ft_line, 33.0, mapper, color, radius, points);
        appendCourtArc(ft_line, 25.0, 6.0, -M_PI / 2.0, M_PI / 2.0, mapper, color, radius, points);
    }

    void appendCourtModel(const CourtMapper& mapper, std::vector<TacticalPoint>& points) const {
        const float thin = 0.85f;
        // Court boundary.
        appendCourtSegment(0.0, 0.0, 94.0, 0.0, mapper, court_line_color_, thin, points);
        appendCourtSegment(0.0, 50.0, 94.0, 50.0, mapper, court_line_color_, thin, points);
        appendCourtSegment(0.0, 0.0, 0.0, 50.0, mapper, court_line_color_, thin, points);
        appendCourtSegment(94.0, 0.0, 94.0, 50.0, mapper, court_line_color_, thin, points);
        // Center line + circle.
        appendCourtSegment(47.0, 0.0, 47.0, 50.0, mapper, court_line_color_, thin, points);
        appendCourtArc(47.0, 25.0, 6.0, 0.0, 2.0 * M_PI, mapper, court_line_color_, thin, points);
        // Paint both ends.
        appendPaintHalf(true, mapper, court_line_color_, thin, points);
        appendPaintHalf(false, mapper, court_line_color_, thin, points);
        // Three-point lines: active half highlighted.
        const bool active_left = mapper.hoop_on_left;
        appendThreePointHalf(!active_left, mapper, court_line_color_, thin, points);
        appendThreePointHalf(active_left, mapper, three_point_color_, 1.25f, points);
    }

    bool uploadPoints(const std::vector<TacticalPoint>& points) {
        const size_t bytes = points.size() * sizeof(TacticalPoint);
        if (bytes == 0) return true;
        if (bytes > d_points_capacity_) {
            if (d_points_) cuMemFree(d_points_);
            d_points_capacity_ = bytes * 2;
            if (CUDA_OVERLAY_CHECK_CU(cuMemAlloc(&d_points_, d_points_capacity_))) {
                d_points_ = 0;
                d_points_capacity_ = 0;
                return false;
            }
        }
        return CUDA_OVERLAY_CHECK_CU(cuMemcpyHtoDAsync(d_points_, points.data(), bytes, cuda_dev_ctx_->stream)) == 0;
    }

    bool renderPoints(av::VideoFrame& output,
                      const PanelGeometry& geom,
                      const std::vector<TacticalPoint>& points,
                      int active_hoop_on_left) {
        if (!uploadPoints(points)) {
            logstream << "draw_tactical_court: failed uploading points";
            return false;
        }

        const unsigned int block_x = 32;
        const unsigned int block_y = 8;
        const unsigned int grid_x = ((unsigned int)output.width() + block_x - 1) / block_x;
        const unsigned int grid_y = ((unsigned int)output.height() + block_y - 1) / block_y;
        const int uv_width = (output.width() + 1) / 2;
        const int uv_height = (output.height() + 1) / 2;
        const unsigned int uv_grid_x = ((unsigned int)uv_width + block_x - 1) / block_x;
        const unsigned int uv_grid_y = ((unsigned int)uv_height + block_y - 1) / block_y;

        CUdeviceptr y_plane = (CUdeviceptr)(uintptr_t)output.raw()->data[0];
        size_t pitch_y = (size_t)output.raw()->linesize[0];
        CUdeviceptr uv_plane = (CUdeviceptr)(uintptr_t)output.raw()->data[1];
        size_t pitch_uv = (size_t)output.raw()->linesize[1];
        int width = output.width();
        int height = output.height();
        CUdeviceptr points_ptr = d_points_;
        int num_points = (int)points.size();
        int bg_y = background_color_.y;
        int bg_u = background_color_.u;
        int bg_v = background_color_.v;
        float bg_alpha = background_opacity_;
        int line_y = court_line_color_.y;
        int line_u = court_line_color_.u;
        int line_v = court_line_color_.v;
        int three_y = three_point_color_.y;
        int three_u = three_point_color_.u;
        int three_v = three_point_color_.v;
        int hoop_y = hoop_color_.y;
        int hoop_u = hoop_color_.u;
        int hoop_v = hoop_color_.v;
        int line_thickness = line_thickness_;

        void* y_args[] = {
            (void*)&y_plane, (void*)&pitch_y,
            (void*)&width, (void*)&height,
            (void*)&geom.panel_x, (void*)&geom.panel_y, (void*)&geom.panel_w, (void*)&geom.panel_h,
            (void*)&geom.court_x, (void*)&geom.court_y, (void*)&geom.court_w, (void*)&geom.court_h,
            (void*)&points_ptr, (void*)&num_points,
            (void*)&bg_y, (void*)&bg_u, (void*)&bg_v, (void*)&bg_alpha,
            (void*)&line_y, (void*)&line_u, (void*)&line_v,
            (void*)&three_y, (void*)&three_u, (void*)&three_v,
            (void*)&hoop_y, (void*)&hoop_u, (void*)&hoop_v,
            (void*)&line_thickness, (void*)&active_hoop_on_left
        };
        if (CUDA_OVERLAY_CHECK_CU(cuLaunchKernel(draw_luma_kernel_,
                                    grid_x, grid_y, 1,
                                    block_x, block_y, 1,
                                    0, cuda_dev_ctx_->stream, y_args, nullptr))) {
            logstream << "draw_tactical_court: failed launching luma kernel";
            return false;
        }

        void* uv_args[] = {
            (void*)&uv_plane, (void*)&pitch_uv,
            (void*)&width, (void*)&height,
            (void*)&geom.panel_x, (void*)&geom.panel_y, (void*)&geom.panel_w, (void*)&geom.panel_h,
            (void*)&geom.court_x, (void*)&geom.court_y, (void*)&geom.court_w, (void*)&geom.court_h,
            (void*)&points_ptr, (void*)&num_points,
            (void*)&bg_y, (void*)&bg_u, (void*)&bg_v, (void*)&bg_alpha,
            (void*)&line_y, (void*)&line_u, (void*)&line_v,
            (void*)&three_y, (void*)&three_u, (void*)&three_v,
            (void*)&hoop_y, (void*)&hoop_u, (void*)&hoop_v,
            (void*)&line_thickness, (void*)&active_hoop_on_left
        };
        if (CUDA_OVERLAY_CHECK_CU(cuLaunchKernel(draw_chroma_kernel_,
                                    uv_grid_x, uv_grid_y, 1,
                                    block_x, block_y, 1,
                                    0, cuda_dev_ctx_->stream, uv_args, nullptr))) {
            logstream << "draw_tactical_court: failed launching chroma kernel";
            return false;
        }
        return CUDA_OVERLAY_CHECK_CU(cuStreamSynchronize(cuda_dev_ctx_->stream)) == 0;
    }

    void cacheOverlay(const av::VideoFrame& output,
                      const PanelGeometry& geom,
                      const std::vector<TacticalPoint>& points,
                      int active_hoop_on_left) {
        cached_points_ = points;
        cached_geom_ = geom;
        cached_output_w_ = output.width();
        cached_output_h_ = output.height();
        cached_active_hoop_on_left_ = active_hoop_on_left;
        cached_overlay_age_ = 0;
        have_cached_overlay_ = true;
    }

    bool tryRenderCachedOverlay(av::VideoFrame& output) {
        if (!have_cached_overlay_) return false;
        if (cached_overlay_age_ >= overlay_hold_frames_) return false;
        if (cached_output_w_ != output.width() || cached_output_h_ != output.height()) return false;
        ++cached_overlay_age_;
        return renderPoints(output, cached_geom_, cached_points_, cached_active_hoop_on_left_);
    }

    void appendPlayerAndTrailPoints(const std::vector<PlayerDot>& players,
                                    const CourtMapper& mapper,
                                    std::vector<TacticalPoint>& points) {
        std::vector<ProjectedPlayerDot> team_a;
        std::vector<ProjectedPlayerDot> team_b;
        std::vector<ProjectedPlayerDot> unknown;
        std::unordered_set<int> observed_ids;
        team_a.reserve(8);
        team_b.reserve(8);
        unknown.reserve(8);

        auto addProjected = [&](const ProjectedPlayerDot& item) {
            if (item.player.team == "A") {
                team_a.push_back(item);
            } else if (item.player.team == "B") {
                team_b.push_back(item);
            } else if (show_unknown_players_) {
                unknown.push_back(item);
            }
        };

        for (const PlayerDot& player : players) {
            Point2D projected;
            if (!sourcePointToPanel(player.foot, mapper, projected)) continue;

            ProjectedPlayerDot item{player, projected};
            if (player.id >= 0) {
                observed_ids.insert(player.id);
                PlayerRenderState& state = player_states_[player.id];
                Point2D stable = projected;

                if (state.seen_frames > 0) {
                    const double jump = std::hypot(projected.x - state.projected.x,
                                                   projected.y - state.projected.y);
                    const bool hold_jump = state.visible &&
                                           player_hold_frames_ > 0 &&
                                           player_max_jump_px_ > 0.0 &&
                                           jump > player_max_jump_px_ &&
                                           state.missed_frames < player_hold_frames_;
                    if (hold_jump) {
                        stable = state.projected;
                        ++state.missed_frames;
                    } else {
                        const double alpha = clampDouble(player_smoothing_alpha_, 0.0, 1.0);
                        stable = state.projected * (1.0 - alpha) + projected * alpha;
                        state.missed_frames = 0;
                    }
                } else {
                    state.missed_frames = 0;
                }

                state.projected = stable;
                state.team = player.team;
                state.has_ball = player.has_ball;
                state.seen_frames = std::min(state.seen_frames + 1, 1000000);
                state.visible = state.seen_frames >= std::max(1, player_min_seen_frames_);
                if (!state.visible) continue;

                item.projected = stable;
            }

            addProjected(item);
        }

        std::vector<int> erase_ids;
        for (auto& kv : player_states_) {
            const int id = kv.first;
            PlayerRenderState& state = kv.second;
            if (observed_ids.find(id) != observed_ids.end()) continue;

            if (state.visible && state.missed_frames < player_hold_frames_) {
                ++state.missed_frames;
                PlayerDot held;
                held.id = id;
                held.team = state.team;
                held.has_ball = false;
                ProjectedPlayerDot item{held, state.projected};
                addProjected(item);
            } else {
                erase_ids.push_back(id);
            }
        }
        for (int id : erase_ids) player_states_.erase(id);

        auto prioritize = [](std::vector<ProjectedPlayerDot>& dots) {
            std::stable_sort(dots.begin(), dots.end(),
                             [](const ProjectedPlayerDot& a, const ProjectedPlayerDot& b) {
                                 if (a.player.has_ball != b.player.has_ball) return a.player.has_ball;
                                 if (a.player.id >= 0 && b.player.id >= 0) return a.player.id < b.player.id;
                                 return a.player.id > b.player.id;
                             });
        };
        prioritize(team_a);
        prioritize(team_b);
        prioritize(unknown);

        std::vector<ProjectedPlayerDot> selected;
        selected.reserve((size_t)std::max(0, max_player_dots_));
        auto take = [&](const std::vector<ProjectedPlayerDot>& src, int cap) {
            const int remaining = std::max(0, max_player_dots_ - (int)selected.size());
            const int n = std::min({(int)src.size(), std::max(0, cap), remaining});
            for (int i = 0; i < n; ++i) selected.push_back(src[(size_t)i]);
        };
        take(team_a, max_players_per_team_);
        take(team_b, max_players_per_team_);
        if (show_unknown_players_) take(unknown, max_player_dots_);

        std::unordered_set<int> current_ids;
        if (show_trails_) {
            for (const ProjectedPlayerDot& item : selected) {
                const int id = item.player.id;
                if (id >= 0) {
                    current_ids.insert(id);
                    auto& hist = trails_[id];
                    if (hist.empty() ||
                        std::hypot(hist.back().x - (float)item.projected.x, hist.back().y - (float)item.projected.y) > 0.5f) {
                        hist.push_back(TrailPoint{(float)item.projected.x, (float)item.projected.y, item.player.team});
                    }
                    while ((int)hist.size() > trail_length_) hist.pop_front();
                }
            }

            for (const auto& kv : trails_) {
                if (current_ids.find(kv.first) == current_ids.end()) continue;
                const auto& hist = kv.second;
                if (hist.size() < 2) continue;
                const size_t keep_from = hist.size() > 1 ? hist.size() - 1 : 0;
                for (size_t i = 0; i < keep_from; ++i) {
                    const DrawColor color = teamColor(hist[i].team);
                    points.push_back(TacticalPoint{hist[i].x, hist[i].y, 1.8f, color.y, color.u, color.v});
                }
            }
        } else if (!trails_.empty()) {
            trails_.clear();
        }

        for (const ProjectedPlayerDot& item : selected) {
            const DrawColor color = teamColor(item.player.team);
            points.push_back(TacticalPoint{(float)item.projected.x, (float)item.projected.y, item.player.has_ball ? 6.0f : 5.4f,
                                           color.y, color.u, color.v});
        }
    }

    void drawOnFrame(const av::VideoFrame& input, av::VideoFrame& output) override {
        if (!loadKernels(avpl_draw_tactical_court_ptx, avpl_draw_tactical_court_ptx_len,
                         "kDrawTacticalCourtNV12Luma", "kDrawTacticalCourtNV12Chroma")) {
            throw Error("draw_tactical_court: failed to initialize CUDA kernels");
        }
        const PanelGeometry geom = makePanelGeometry(output);

        CourtMapper mapper;
        if (!parseCalibration(input, geom, mapper)) {
            tryRenderCachedOverlay(output);
            return;
        }
        if (mapper.hoop_on_left != last_hoop_on_left_) {
            trails_.clear();
            player_states_.clear();
            last_hoop_on_left_ = mapper.hoop_on_left;
        }

        std::vector<PlayerDot> players;
        parsePlayers(input, players);

        std::vector<TacticalPoint> points;
        points.reserve(1200);
        appendCourtModel(mapper, points);
        appendPlayerAndTrailPoints(players, mapper, points);

        const int active_hoop_on_left = mapper.hoop_on_left ? 1 : 0;
        if (!renderPoints(output, geom, points, active_hoop_on_left)) return;
        cacheOverlay(output, geom, points, active_hoop_on_left);

        if (debug_log_every_n_ > 0 && (frame_counter_ % (uint64_t)debug_log_every_n_) == 0) {
            logstream << "draw_tactical_court: frame=" << frame_counter_
                      << " players=" << players.size()
                      << " points=" << points.size()
                      << " hoop_on_left=" << (mapper.hoop_on_left ? 1 : 0)
                      << " cached_age=" << cached_overlay_age_;
        }
    }

public:
    using CudaOverlayBase::CudaOverlayBase;

    static std::shared_ptr<DrawTacticalCourt> create(NodeCreationInfo& nci) {
        EdgeManager& edges = nci.edges;
        const Parameters& params = nci.params;

        auto src_edge = edges.find<av::VideoFrame>(params["src"]);
        auto dst_edge = edges.find<av::VideoFrame>(params["dst"]);
        auto r = std::make_shared<DrawTacticalCourt>(src_edge->makeSource(), dst_edge->makeSink());

        UpstreamInfo info = resolveUpstreamInfo(src_edge, params);
        r->input_params_ = info.input_params;
        r->frame_rate_ = info.frame_rate;
        r->timebase_ = info.timebase;

        r->calib_metadata_key_ = params.value("calib_metadata_key", std::string("court_calib"));
        r->feet_metadata_key_ = params.value("feet_metadata_key", std::string("player_feet"));
        r->panel_width_ = params.value("panel_width", 360);
        r->panel_height_ = params.value("panel_height", 220);
        r->padding_left_ = params.value("padding_left", 28);
        r->padding_bottom_ = params.value("padding_bottom", 116);
        r->inner_padding_ = params.value("inner_padding", 14);
        r->line_thickness_ = params.value("line_thickness", 2);
        r->trail_length_ = params.value("trail_length", 18);
        r->max_players_per_team_ = params.value("max_players_per_team", 5);
        r->max_player_dots_ = params.value("max_player_dots", 10);
        r->show_trails_ = params.value("show_trails", false);
        r->show_unknown_players_ = params.value("show_unknown_players", false);
        r->min_foot_confidence_ = params.value("min_foot_confidence", 0.15);
        r->overlay_hold_frames_ = params.value("overlay_hold_frames", 30);
        r->player_min_seen_frames_ = params.value("player_min_seen_frames", 2);
        r->player_hold_frames_ = params.value("player_hold_frames", 8);
        r->player_smoothing_alpha_ = params.value("player_smoothing_alpha", 0.35);
        r->player_max_jump_px_ = params.value("player_max_jump_px", 46.0);
        r->background_opacity_ = params.value("background_opacity", 0.58f);
        r->debug_log_every_n_ = params.value("debug_log_every_n", 0);

        auto parse_color = [&](const char* key, const char* fallback, DrawColor& out) {
            const std::string name = params.value(key, std::string(fallback));
            if (!cuda_overlay::tryParseNamedColor(name, out)) {
                throw Error(std::string("draw_tactical_court: unknown color for ") + key + ": " + name);
            }
        };
        parse_color("background_color", "black", r->background_color_);
        parse_color("court_line_color", "white", r->court_line_color_);
        parse_color("three_point_color", "yellow", r->three_point_color_);
        parse_color("hoop_color", "orange", r->hoop_color_);
        parse_color("team_a_color", "light_blue", r->team_a_color_);
        parse_color("team_b_color", "green", r->team_b_color_);
        parse_color("unknown_color", "light_gray", r->unknown_color_);

        return r;
    }
};

DECLNODE(draw_tactical_court, DrawTacticalCourt)
