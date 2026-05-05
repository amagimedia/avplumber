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

struct MaskInfo {
    int num_masks = 0;
    int w = 0;
    int h = 0;
    const float* data = nullptr;
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

struct CourtPosePoint {
    int keypoint_index = -1;
    Point2D image;
    Point2D court_ft;
    double conf = 0.0;
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
    Point2D hoop;
    Point2D u; // source-space direction from hoop toward court interior
    Point2D v; // source-space lateral direction, positive toward screen-down
    double depth_px_per_ft = 1.0;
    double lateral_px_per_ft = 1.0;
    bool hoop_on_left = true;
    int court_x = 0;
    int court_y = 0;
    int court_w = 0;
    int court_h = 0;
    bool has_homography = false;
    double source_to_court[9] = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
    double homography_source_w = 1.0;
    double homography_source_h = 1.0;
    int homography_inliers = 0;
    double homography_error_ft = 0.0;
    bool used_line = false;
    bool used_pose = false;
    bool used_pose_keypoints = false;
};

Point2D operator+(const Point2D& a, const Point2D& b) {
    return Point2D{a.x + b.x, a.y + b.y};
}

Point2D operator-(const Point2D& a, const Point2D& b) {
    return Point2D{a.x - b.x, a.y - b.y};
}

Point2D operator*(const Point2D& a, double s) {
    return Point2D{a.x * s, a.y * s};
}

double dotPoint(const Point2D& a, const Point2D& b) {
    return a.x * b.x + a.y * b.y;
}

double lengthPoint(const Point2D& p) {
    return std::sqrt(dotPoint(p, p));
}

bool normalizePoint(Point2D& p) {
    const double len = lengthPoint(p);
    if (len < 1e-6 || !std::isfinite(len)) return false;
    p.x /= len;
    p.y /= len;
    return true;
}

double clampDouble(double v, double lo, double hi) {
    return std::max(lo, std::min(hi, v));
}

bool readCpuMasks(const AVFrame* raw, int slot, MaskInfo& out) {
    if (!raw) return false;
    AVFrameSideData* sd = av_frame_get_side_data(raw, yoloSegCpuSideDataType(slot));
    if (!sd || !sd->buf || sd->buf->size < 16) return false;
    const uint32_t* header = (const uint32_t*)sd->buf->data;
    out.num_masks = (int)header[0];
    out.w = (int)header[1];
    out.h = (int)header[2];
    const size_t expected = 16 + (size_t)out.num_masks * (size_t)out.w * (size_t)out.h * sizeof(float);
    if ((size_t)sd->buf->size < expected) return false;
    out.data = (const float*)(sd->buf->data + 16);
    return out.num_masks > 0 && out.w > 0 && out.h > 0;
}

double percentile(std::vector<double> values, double q) {
    if (values.empty()) return 0.0;
    q = std::max(0.0, std::min(1.0, q));
    const size_t idx = std::min(values.size() - 1, (size_t)std::llround(q * (double)(values.size() - 1)));
    std::nth_element(values.begin(), values.begin() + (ptrdiff_t)idx, values.end());
    return values[idx];
}

bool parsePointArray(const Parameters& arr, Point2D& out) {
    if (!arr.is_array() || arr.size() < 2) return false;
    out.x = arr[0].get<double>();
    out.y = arr[1].get<double>();
    return std::isfinite(out.x) && std::isfinite(out.y);
}

bool canonicalCourtPosePoint(int idx, Point2D& out) {
    switch (idx) {
    case 0: out = Point2D{0.0, 0.0}; return true;
    case 1: out = Point2D{0.0, 25.0}; return true;
    case 2: out = Point2D{0.0, 50.0}; return true;
    case 3: out = Point2D{23.5, 0.0}; return true;
    case 4: out = Point2D{23.5, 50.0}; return true;
    case 5: out = Point2D{47.0, 0.0}; return true;
    case 6: out = Point2D{47.0, 50.0}; return true;
    case 7: out = Point2D{70.5, 0.0}; return true;
    case 8: out = Point2D{70.5, 50.0}; return true;
    case 9: out = Point2D{94.0, 0.0}; return true;
    case 10: out = Point2D{94.0, 25.0}; return true;
    case 11: out = Point2D{94.0, 50.0}; return true;
    default: return false;
    }
}

} // namespace

class DrawTacticalCourt : public CudaOverlayBase {
    std::string metadata_key_ = "frame_dump";
    std::string court_seg_metadata_key_ = "yolo_seg";
    std::string pose_metadata_key_ = "yolo_pose";
    int court_seg_slot_ = 0;
    float mask_threshold_ = 0.5f;
    bool require_wide_shot_ = true;
    int panel_width_ = 360;
    int panel_height_ = 220;
    int padding_left_ = 28;
    int padding_bottom_ = 116;
    int inner_padding_ = 14;
    int line_thickness_ = 2;
    int max_line_points_ = 512;
    int trail_length_ = 18;
    int max_players_per_team_ = 5;
    int max_player_dots_ = 10;
    bool show_trails_ = false;
    bool show_ball_ring_ = false;
    bool show_ball_ = true;
    bool show_unknown_players_ = false;
    double min_foot_confidence_ = 0.15;
    double pose_min_keypoint_conf_ = 0.05;
    int pose_homography_min_keypoints_ = 4;
    double pose_homography_inlier_error_ft_ = 8.0;
    double pose_homography_hoop_error_ft_ = 0.0;
    int mapper_hold_frames_ = 12;
    int overlay_hold_frames_ = 30;
    int side_switch_confirm_frames_ = 4;
    double mapper_smoothing_alpha_ = 0.25;
    double mapper_player_jump_gate_px_ = 55.0;
    int mapper_player_jump_gate_min_tracks_ = 3;
    int mapper_bad_hold_frames_ = 10;
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
    DrawColor ball_color_{81, 90, 240};
    DrawColor team_a_color_{169, 166, 16};
    DrawColor team_b_color_{173, 42, 26};
    DrawColor unknown_color_{200, 128, 128};
    DrawColor ball_ring_color_{235, 128, 128};

    CUdeviceptr d_points_ = 0;
    size_t d_points_capacity_ = 0;
    std::unordered_map<int, std::deque<TrailPoint>> trails_;
    CourtMapper last_mapper_;
    bool have_last_mapper_ = false;
    int last_mapper_age_ = 0;
    int mapper_bad_frames_ = 0;
    bool pending_hoop_on_left_ = true;
    int pending_side_frames_ = 0;
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

    bool parseFrameDump(const av::VideoFrame& frm,
                        std::vector<PlayerDot>& players,
                        Point2D& hoop,
                        bool& have_hoop,
                        Point2D& ball,
                        bool& have_ball) const {
        players.clear();
        have_hoop = false;
        have_ball = false;
        const AVFrame* raw = frm.raw();
        if (!raw || !raw->metadata) return false;
        AVDictionaryEntry* entry = av_dict_get(raw->metadata, metadata_key_.c_str(), nullptr, 0);
        if (!entry || !entry->value) return false;

        try {
            Parameters md = Parameters::parse(entry->value);
            if (require_wide_shot_ && md.value("camera_shot", std::string()) != "wide") return false;

            if (md.contains("hoop") && md["hoop"].is_object()) {
                const auto& h = md["hoop"];
                if (h.contains("box") && h["box"].is_array() && h["box"].size() >= 4) {
                    hoop.x = (h["box"][0].get<double>() + h["box"][2].get<double>()) * 0.5;
                    hoop.y = (h["box"][1].get<double>() + h["box"][3].get<double>()) * 0.5;
                    have_hoop = true;
                }
            }

            if (md.contains("ball") && md["ball"].is_object()) {
                const auto& b = md["ball"];
                if (b.contains("box") && b["box"].is_array() && b["box"].size() >= 4) {
                    ball.x = (b["box"][0].get<double>() + b["box"][2].get<double>()) * 0.5;
                    ball.y = (b["box"][1].get<double>() + b["box"][3].get<double>()) * 0.5;
                    have_ball = std::isfinite(ball.x) && std::isfinite(ball.y);
                }
            }

            if (!md.contains("players") || !md["players"].is_array()) return false;
            for (const auto& p : md["players"]) {
                if (!p.is_object()) continue;
                if (!p.contains("foot") || !p["foot"].is_object()) continue;
                const auto& foot = p["foot"];
                if (foot.contains("valid") && foot["valid"].is_boolean() && !foot["valid"].get<bool>()) continue;
                if (foot.value("confidence", 1.0) < min_foot_confidence_) continue;
                if (!foot.contains("point")) continue;
                PlayerDot dot;
                dot.id = p.value("id", -1);
                dot.team = p.value("team", std::string());
                dot.has_ball = p.value("has_ball", false);
                if (!parsePointArray(foot["point"], dot.foot)) continue;
                players.push_back(dot);
            }
        } catch (const std::exception&) {
            return false;
        }
        return true;
    }

    std::vector<std::string> courtSegLabels(const AVFrame* raw) const {
        std::vector<std::string> labels;
        if (!raw || !raw->metadata) return labels;
        AVDictionaryEntry* entry = av_dict_get(raw->metadata, court_seg_metadata_key_.c_str(), nullptr, 0);
        if (!entry || !entry->value) return labels;
        try {
            Parameters md = Parameters::parse(entry->value);
            if (!md.contains("detections") || !md["detections"].is_array()) return labels;
            for (const auto& det : md["detections"]) {
                if (!det.is_object()) continue;
                labels.push_back(det.value("label", std::string()));
            }
        } catch (const std::exception&) {
            labels.clear();
        }
        return labels;
    }

    int maskArea(const MaskInfo& masks, int idx) const {
        if (idx < 0 || idx >= masks.num_masks) return 0;
        const float* data = masks.data + (size_t)idx * (size_t)masks.w * (size_t)masks.h;
        int area = 0;
        for (int i = 0; i < masks.w * masks.h; ++i) {
            if (data[i] >= mask_threshold_) ++area;
        }
        return area;
    }

    bool chooseCourtAndLineMasks(const MaskInfo& masks,
                                 const std::vector<std::string>& labels,
                                 int& court_idx,
                                 std::vector<int>& line_indices) const {
        court_idx = -1;
        int best_area = 0;
        line_indices.clear();
        const bool have_labels = labels.size() >= (size_t)masks.num_masks;

        for (int i = 0; i < masks.num_masks; ++i) {
            const std::string label = have_labels ? labels[(size_t)i] : std::string();
            const int area = maskArea(masks, i);
            if (have_labels && label == "three point line") {
                line_indices.push_back(i);
                continue;
            }
            if ((!have_labels || label == "basketball-court") && area > best_area) {
                best_area = area;
                court_idx = i;
            }
        }

        if (court_idx < 0) {
            for (int i = 0; i < masks.num_masks; ++i) {
                const int area = maskArea(masks, i);
                if (area > best_area) {
                    best_area = area;
                    court_idx = i;
                }
            }
        }
        return !line_indices.empty() || (court_idx >= 0 && best_area > 64);
    }

    std::vector<Point2D> collectLineSamples(const MaskInfo& masks,
                                            const std::vector<int>& line_indices,
                                            int frame_w,
                                            int frame_h) const {
        std::vector<Point2D> samples;
        if (line_indices.empty() || masks.w <= 0 || masks.h <= 0) return samples;

        int count = 0;
        for (int idx : line_indices) {
            if (idx < 0 || idx >= masks.num_masks) continue;
            const float* data = masks.data + (size_t)idx * (size_t)masks.w * (size_t)masks.h;
            for (int i = 0; i < masks.w * masks.h; ++i) {
                if (data[i] >= mask_threshold_) ++count;
            }
        }
        if (count <= 0) return samples;

        const int max_samples = std::max(64, max_line_points_ * 4);
        const int stride = std::max(1, count / max_samples);
        const double sx = (double)frame_w / (double)masks.w;
        const double sy = (double)frame_h / (double)masks.h;
        int seen = 0;

        samples.reserve((size_t)std::min(count, max_samples + 1));
        for (int idx : line_indices) {
            if (idx < 0 || idx >= masks.num_masks) continue;
            const float* data = masks.data + (size_t)idx * (size_t)masks.w * (size_t)masks.h;
            for (int y = 0; y < masks.h; ++y) {
                for (int x = 0; x < masks.w; ++x) {
                    if (data[(size_t)y * (size_t)masks.w + (size_t)x] < mask_threshold_) continue;
                    if ((seen++ % stride) != 0) continue;
                    samples.push_back(Point2D{((double)x + 0.5) * sx, ((double)y + 0.5) * sy});
                    if ((int)samples.size() >= max_samples) return samples;
                }
            }
        }
        return samples;
    }

    bool parseCourtPosePoints(const av::VideoFrame& frm,
                              std::vector<CourtPosePoint>& points_out,
                              int& visible_keypoints_out,
                              double& pose_conf_out) const {
        points_out.clear();
        visible_keypoints_out = 0;
        pose_conf_out = 0.0;
        const AVFrame* raw = frm.raw();
        if (!raw || !raw->metadata) return false;
        AVDictionaryEntry* entry = av_dict_get(raw->metadata, pose_metadata_key_.c_str(), nullptr, 0);
        if (!entry || !entry->value) return false;

        try {
            Parameters md = Parameters::parse(entry->value);
            if (!md.contains("poses") || !md["poses"].is_array() || md["poses"].empty()) return false;

            const Parameters* best_pose = nullptr;
            double best_conf = -1.0;
            for (const auto& pose : md["poses"]) {
                if (!pose.is_object()) continue;
                const double conf = pose.value("conf", 0.0);
                if (conf > best_conf) {
                    best_conf = conf;
                    best_pose = &pose;
                }
            }
            if (!best_pose || !best_pose->contains("keypoints") || !(*best_pose)["keypoints"].is_array()) return false;
            pose_conf_out = best_conf;

            const double model_w = md.value("model_width", (double)frm.width());
            const double model_h = md.value("model_height", (double)frm.height());
            if (model_w <= 1.0 || model_h <= 1.0) return false;

            const auto& kpts = (*best_pose)["keypoints"];
            if ((kpts.size() % 3) != 0) return false;
            const int available_keypoints = (int)(kpts.size() / 3);
            const int declared_keypoints = md.value("num_keypoints", available_keypoints);
            const double sx = (double)frm.width() / model_w;
            const double sy = (double)frm.height() / model_h;

            static constexpr std::array<int, 12> kOld33ToPose12 = {
                0, 3, 5, 12, 14, 15, 17, 18, 20, 27, 30, 32
            };

            points_out.reserve(12);
            for (int new_idx = 0; new_idx < 12; ++new_idx) {
                const int src_idx = declared_keypoints >= 33 ? kOld33ToPose12[(size_t)new_idx] : new_idx;
                if (src_idx < 0 || src_idx >= available_keypoints) continue;
                const size_t off = (size_t)src_idx * 3u;
                const double conf = kpts[off + 2].get<double>();
                if (conf < pose_min_keypoint_conf_) continue;
                const double x = kpts[off + 0].get<double>() * sx;
                const double y = kpts[off + 1].get<double>() * sy;
                if (!std::isfinite(x) || !std::isfinite(y)) continue;
                Point2D court;
                if (!canonicalCourtPosePoint(new_idx, court)) continue;
                points_out.push_back(CourtPosePoint{new_idx, Point2D{x, y}, court, conf});
            }
            visible_keypoints_out = (int)points_out.size();
            return !points_out.empty();
        } catch (const std::exception&) {
            return false;
        }
    }

    bool poseCentroid(const std::vector<CourtPosePoint>& points,
                      Point2D& centroid_out,
                      int min_points) const {
        const double min_centroid_conf = std::max(0.15, pose_min_keypoint_conf_);
        Point2D sum;
        int count = 0;
        for (const CourtPosePoint& p : points) {
            if (p.conf < min_centroid_conf) continue;
            sum = sum + p.image;
            ++count;
        }
        if (count < min_points) return false;
        centroid_out = sum * (1.0 / (double)count);
        return std::isfinite(centroid_out.x) && std::isfinite(centroid_out.y);
    }

    bool solve8x8(double a[8][8], double b[8], double x[8]) const {
        for (int col = 0; col < 8; ++col) {
            int pivot = col;
            double best = std::fabs(a[col][col]);
            for (int row = col + 1; row < 8; ++row) {
                const double v = std::fabs(a[row][col]);
                if (v > best) {
                    best = v;
                    pivot = row;
                }
            }
            if (best < 1e-10 || !std::isfinite(best)) return false;
            if (pivot != col) {
                for (int c = col; c < 8; ++c) std::swap(a[col][c], a[pivot][c]);
                std::swap(b[col], b[pivot]);
            }

            const double div = a[col][col];
            for (int c = col; c < 8; ++c) a[col][c] /= div;
            b[col] /= div;

            for (int row = 0; row < 8; ++row) {
                if (row == col) continue;
                const double f = a[row][col];
                if (std::fabs(f) < 1e-14) continue;
                for (int c = col; c < 8; ++c) a[row][c] -= f * a[col][c];
                b[row] -= f * b[col];
            }
        }

        for (int i = 0; i < 8; ++i) {
            x[i] = b[i];
            if (!std::isfinite(x[i])) return false;
        }
        return true;
    }

    bool solveHomography(const std::vector<CourtPosePoint>& points,
                         const std::vector<int>& indices,
                         double source_w,
                         double source_h,
                         double h[9]) const {
        if (indices.size() < 4 || source_w <= 1.0 || source_h <= 1.0) return false;

        double ata[8][8] = {};
        double atb[8] = {};
        auto accumulate = [&](const double row[8], double rhs) {
            for (int r = 0; r < 8; ++r) {
                atb[r] += row[r] * rhs;
                for (int c = 0; c < 8; ++c) ata[r][c] += row[r] * row[c];
            }
        };

        for (int idx : indices) {
            if (idx < 0 || idx >= (int)points.size()) return false;
            const CourtPosePoint& p = points[(size_t)idx];
            const double x = p.image.x / source_w;
            const double y = p.image.y / source_h;
            const double X = p.court_ft.x / 94.0;
            const double Y = p.court_ft.y / 50.0;
            if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(X) || !std::isfinite(Y)) return false;
            const double row_x[8] = {x, y, 1.0, 0.0, 0.0, 0.0, -X * x, -X * y};
            const double row_y[8] = {0.0, 0.0, 0.0, x, y, 1.0, -Y * x, -Y * y};
            accumulate(row_x, X);
            accumulate(row_y, Y);
        }

        double solution[8] = {};
        if (!solve8x8(ata, atb, solution)) return false;
        for (int i = 0; i < 8; ++i) h[i] = solution[i];
        h[8] = 1.0;
        return true;
    }

    bool applyHomographyNorm(const double h[9],
                             const Point2D& src,
                             double source_w,
                             double source_h,
                             Point2D& court_norm) const {
        if (source_w <= 1.0 || source_h <= 1.0) return false;
        const double x = src.x / source_w;
        const double y = src.y / source_h;
        const double den = h[6] * x + h[7] * y + h[8];
        if (std::fabs(den) < 1e-9 || !std::isfinite(den)) return false;
        court_norm.x = (h[0] * x + h[1] * y + h[2]) / den;
        court_norm.y = (h[3] * x + h[4] * y + h[5]) / den;
        return std::isfinite(court_norm.x) && std::isfinite(court_norm.y);
    }

    double homographyPointErrorFt(const double h[9],
                                  const CourtPosePoint& p,
                                  double source_w,
                                  double source_h) const {
        Point2D projected;
        if (!applyHomographyNorm(h, p.image, source_w, source_h, projected)) {
            return std::numeric_limits<double>::infinity();
        }
        const double x_ft = projected.x * 94.0;
        const double y_ft = projected.y * 50.0;
        return std::hypot(x_ft - p.court_ft.x, y_ft - p.court_ft.y);
    }

    bool buildPoseHomographyMapper(const av::VideoFrame& frm,
                                   const Point2D& hoop,
                                   bool have_hoop,
                                   const std::vector<CourtPosePoint>& pose_points,
                                   int court_x,
                                   int court_y,
                                   int court_w,
                                   int court_h,
                                   CourtMapper& out) const {
        if ((int)pose_points.size() < std::max(4, pose_homography_min_keypoints_)) return false;

        const double source_w = (double)frm.width();
        const double source_h = (double)frm.height();
        const int n = (int)pose_points.size();
        double best_h[9] = {};
        int best_inliers = 0;
        double best_error = std::numeric_limits<double>::infinity();

        for (int a = 0; a < n - 3; ++a) {
            for (int b = a + 1; b < n - 2; ++b) {
                for (int c = b + 1; c < n - 1; ++c) {
                    for (int d = c + 1; d < n; ++d) {
                        std::vector<int> subset = {a, b, c, d};
                        double h[9] = {};
                        if (!solveHomography(pose_points, subset, source_w, source_h, h)) continue;

                        int inliers = 0;
                        double error_sum = 0.0;
                        for (int i = 0; i < n; ++i) {
                            const double err = homographyPointErrorFt(h, pose_points[(size_t)i], source_w, source_h);
                            if (err <= pose_homography_inlier_error_ft_) {
                                ++inliers;
                                error_sum += err;
                            }
                        }
                        if (inliers < std::max(4, pose_homography_min_keypoints_)) continue;
                        const double avg_error = error_sum / (double)std::max(1, inliers);
                        if (inliers > best_inliers ||
                            (inliers == best_inliers && avg_error < best_error)) {
                            std::copy(h, h + 9, best_h);
                            best_inliers = inliers;
                            best_error = avg_error;
                        }
                    }
                }
            }
        }

        if (best_inliers < std::max(4, pose_homography_min_keypoints_)) return false;

        std::vector<int> inlier_indices;
        inlier_indices.reserve((size_t)best_inliers);
        for (int i = 0; i < n; ++i) {
            const double err = homographyPointErrorFt(best_h, pose_points[(size_t)i], source_w, source_h);
            if (err <= pose_homography_inlier_error_ft_) inlier_indices.push_back(i);
        }
        if (inlier_indices.size() >= 4) {
            double refit_h[9] = {};
            if (solveHomography(pose_points, inlier_indices, source_w, source_h, refit_h)) {
                std::copy(refit_h, refit_h + 9, best_h);
            }
        }

        double error_sum = 0.0;
        for (int idx : inlier_indices) {
            error_sum += homographyPointErrorFt(best_h, pose_points[(size_t)idx], source_w, source_h);
        }
        best_error = error_sum / (double)std::max<size_t>(1, inlier_indices.size());

        bool hoop_on_left = have_hoop ? hoop.x < source_w * 0.5 : true;
        if (have_hoop) {
            if (pose_homography_hoop_error_ft_ > 0.0) {
                Point2D hoop_norm;
                if (!applyHomographyNorm(best_h, hoop, source_w, source_h, hoop_norm)) return false;
                const double hoop_x_ft = hoop_norm.x * 94.0;
                const double hoop_y_ft = hoop_norm.y * 50.0;
                const double left_err = std::hypot(hoop_x_ft - 5.25, hoop_y_ft - 25.0);
                const double right_err = std::hypot(hoop_x_ft - 88.75, hoop_y_ft - 25.0);
                if (std::min(left_err, right_err) > pose_homography_hoop_error_ft_) return false;
                hoop_on_left = left_err <= right_err;
            }
        } else {
            double avg_x = 0.0;
            for (const CourtPosePoint& p : pose_points) avg_x += p.court_ft.x;
            avg_x /= (double)pose_points.size();
            hoop_on_left = avg_x < 47.0;
        }

        out.hoop = hoop;
        out.u = Point2D{hoop_on_left ? 1.0 : -1.0, 0.0};
        out.v = Point2D{0.0, 1.0};
        out.depth_px_per_ft = 1.0;
        out.lateral_px_per_ft = 1.0;
        out.hoop_on_left = hoop_on_left;
        out.court_x = court_x;
        out.court_y = court_y;
        out.court_w = court_w;
        out.court_h = court_h;
        out.has_homography = true;
        std::copy(best_h, best_h + 9, out.source_to_court);
        out.homography_source_w = source_w;
        out.homography_source_h = source_h;
        out.homography_inliers = (int)inlier_indices.size();
        out.homography_error_ft = best_error;
        out.used_pose = true;
        out.used_pose_keypoints = true;
        return true;
    }

    bool buildCourtMapper(const av::VideoFrame& frm,
                          const Point2D& hoop,
                          bool have_hoop,
                          const std::vector<Point2D>& line_samples,
                          const std::vector<CourtPosePoint>& pose_points,
                          int court_x,
                          int court_y,
                          int court_w,
                          int court_h,
                          CourtMapper& out) const {
        if (buildPoseHomographyMapper(frm, hoop, have_hoop, pose_points,
                                      court_x, court_y, court_w, court_h,
                                      out)) {
            return true;
        }
        if (!have_hoop) return false;

        const bool hoop_on_left = hoop.x < (double)frm.width() * 0.5;
        const Point2D expected = hoop_on_left ? Point2D{1.0, 0.0} : Point2D{-1.0, 0.0};
        const double max_len = (double)std::max(frm.width(), frm.height()) * 0.90;

        Point2D line_dir;
        int line_count = 0;
        for (const Point2D& p : line_samples) {
            Point2D vec = p - hoop;
            const double len = lengthPoint(vec);
            if (len < 20.0 || len > max_len) continue;
            line_dir = line_dir + vec;
            ++line_count;
        }

        bool line_valid = line_count >= 8 && normalizePoint(line_dir);
        if (line_valid && dotPoint(line_dir, expected) < 0.0) line_dir = line_dir * -1.0;

        Point2D pose_dir;
        bool pose_valid = false;
        Point2D pose_centroid;
        const bool have_pose_centroid = poseCentroid(pose_points, pose_centroid, 3);
        if (have_pose_centroid) {
            pose_dir = pose_centroid - hoop;
            pose_valid = lengthPoint(pose_dir) > 20.0 && normalizePoint(pose_dir);
            if (pose_valid && dotPoint(pose_dir, expected) < 0.0) pose_dir = pose_dir * -1.0;
        }

        Point2D u;
        bool used_line = false;
        bool used_pose = false;
        if (line_valid) {
            u = line_dir;
            used_line = true;
            if (pose_valid && dotPoint(line_dir, pose_dir) > 0.20) {
                u = line_dir * 0.82 + pose_dir * 0.18;
                used_pose = true;
            }
        } else if (pose_valid) {
            u = pose_dir;
            used_pose = true;
        } else {
            return false;
        }
        if (!normalizePoint(u)) return false;

        Point2D v{-u.y, u.x};
        if (v.y < 0.0) v = v * -1.0;
        if (!normalizePoint(v)) return false;

        std::vector<double> depths;
        std::vector<double> laterals;
        depths.reserve(line_samples.size());
        laterals.reserve(line_samples.size());
        for (const Point2D& p : line_samples) {
            const Point2D vec = p - hoop;
            const double len = lengthPoint(vec);
            if (len < 12.0 || len > max_len) continue;
            const double depth = dotPoint(vec, u);
            const double lateral = std::fabs(dotPoint(vec, v));
            if (depth > 8.0) depths.push_back(depth);
            if (lateral > 4.0) laterals.push_back(lateral);
        }

        const double fallback_px_per_ft = std::max(4.0, ((double)frm.width() * 0.18) / 23.75);
        double depth_px_per_ft = fallback_px_per_ft;
        if (depths.size() >= 8) {
            depth_px_per_ft = percentile(depths, 0.75) / 23.75;
        }
        if (!std::isfinite(depth_px_per_ft) || depth_px_per_ft < 2.0) {
            depth_px_per_ft = fallback_px_per_ft;
        }

        double lateral_px_per_ft = depth_px_per_ft;
        if (laterals.size() >= 8) {
            lateral_px_per_ft = percentile(laterals, 0.90) / 22.0;
        }
        if (!std::isfinite(lateral_px_per_ft) || lateral_px_per_ft < 2.0) {
            lateral_px_per_ft = depth_px_per_ft;
        }

        out.hoop = hoop;
        out.u = u;
        out.v = v;
        out.depth_px_per_ft = clampDouble(depth_px_per_ft, 2.0, 90.0);
        out.lateral_px_per_ft = clampDouble(lateral_px_per_ft, 2.0, 90.0);
        out.hoop_on_left = hoop_on_left;
        out.court_x = court_x;
        out.court_y = court_y;
        out.court_w = court_w;
        out.court_h = court_h;
        out.used_line = used_line;
        out.used_pose = used_pose;
        return true;
    }

    CourtMapper stabilizeMapper(const CourtMapper& candidate) {
        CourtMapper mapper = candidate;
        if (!have_last_mapper_) {
            pending_side_frames_ = 0;
            return mapper;
        }

        if (mapper.hoop_on_left != last_mapper_.hoop_on_left) {
            if (pending_side_frames_ == 0 || pending_hoop_on_left_ != mapper.hoop_on_left) {
                pending_hoop_on_left_ = mapper.hoop_on_left;
                pending_side_frames_ = 1;
            } else {
                ++pending_side_frames_;
            }
            if (pending_side_frames_ < side_switch_confirm_frames_) {
                mapper = last_mapper_;
                mapper.court_x = candidate.court_x;
                mapper.court_y = candidate.court_y;
                mapper.court_w = candidate.court_w;
                mapper.court_h = candidate.court_h;
                mapper.used_line = candidate.used_line;
                mapper.used_pose = candidate.used_pose;
                mapper.used_pose_keypoints = candidate.used_pose_keypoints;
                return mapper;
            }

            pending_side_frames_ = 0;
            trails_.clear();
            player_states_.clear();
            return mapper;
        }
        pending_side_frames_ = 0;

        const double alpha = clampDouble(mapper_smoothing_alpha_, 0.0, 1.0);
        if (alpha <= 0.0) return mapper;

        if (mapper.has_homography || last_mapper_.has_homography) {
            if (mapper.has_homography && last_mapper_.has_homography &&
                std::fabs(mapper.homography_source_w - last_mapper_.homography_source_w) < 1e-3 &&
                std::fabs(mapper.homography_source_h - last_mapper_.homography_source_h) < 1e-3) {
                for (int i = 0; i < 9; ++i) {
                    mapper.source_to_court[i] = last_mapper_.source_to_court[i] * (1.0 - alpha) +
                                                candidate.source_to_court[i] * alpha;
                }
                const double scale = std::fabs(mapper.source_to_court[8]) > 1e-9 ? mapper.source_to_court[8] : 1.0;
                for (double& v : mapper.source_to_court) v /= scale;
                mapper.homography_error_ft = last_mapper_.homography_error_ft * (1.0 - alpha) +
                                             candidate.homography_error_ft * alpha;
                mapper.homography_inliers = candidate.homography_inliers;
            }
            return mapper;
        }

        mapper.hoop = last_mapper_.hoop * (1.0 - alpha) + candidate.hoop * alpha;
        Point2D u = last_mapper_.u * (1.0 - alpha) + candidate.u * alpha;
        if (normalizePoint(u)) {
            mapper.u = u;
            mapper.v = Point2D{-u.y, u.x};
            if (mapper.v.y < 0.0) mapper.v = mapper.v * -1.0;
            normalizePoint(mapper.v);
        }
        mapper.depth_px_per_ft = last_mapper_.depth_px_per_ft * (1.0 - alpha) +
                                 candidate.depth_px_per_ft * alpha;
        mapper.lateral_px_per_ft = last_mapper_.lateral_px_per_ft * (1.0 - alpha) +
                                   candidate.lateral_px_per_ft * alpha;
        mapper.depth_px_per_ft = clampDouble(mapper.depth_px_per_ft, 2.0, 90.0);
        mapper.lateral_px_per_ft = clampDouble(mapper.lateral_px_per_ft, 2.0, 90.0);
        return mapper;
    }

    bool mapperRejectedByPlayerContinuity(const CourtMapper& candidate,
                                          const std::vector<PlayerDot>& players) const {
        if (!have_last_mapper_ || mapper_player_jump_gate_px_ <= 0.0) return false;
        if (candidate.hoop_on_left != last_mapper_.hoop_on_left) return false;

        std::vector<double> jumps;
        jumps.reserve(players.size());
        for (const PlayerDot& player : players) {
            if (player.id < 0) continue;

            Point2D candidate_pt;
            Point2D last_pt;
            if (!sourcePointToPanel(player.foot, candidate, candidate_pt)) continue;
            if (!sourcePointToPanel(player.foot, last_mapper_, last_pt)) continue;

            const double jump = std::hypot(candidate_pt.x - last_pt.x, candidate_pt.y - last_pt.y);
            if (std::isfinite(jump)) jumps.push_back(jump);
        }

        if ((int)jumps.size() < std::max(1, mapper_player_jump_gate_min_tracks_)) return false;
        const double median_jump = percentile(jumps, 0.50);
        const double high_jump = percentile(jumps, 0.80);
        return median_jump > mapper_player_jump_gate_px_ ||
               high_jump > mapper_player_jump_gate_px_ * 1.35;
    }

    Point2D courtFtToPanel(double x_ft, double y_ft, const CourtMapper& mapper) const {
        return Point2D{
            (double)mapper.court_x + (x_ft / 94.0) * (double)mapper.court_w,
            (double)mapper.court_y + (y_ft / 50.0) * (double)mapper.court_h
        };
    }

    bool sourcePointToPanel(const Point2D& src, const CourtMapper& mapper, Point2D& out) const {
        if (mapper.has_homography) {
            Point2D court_norm;
            if (!applyHomographyNorm(mapper.source_to_court, src,
                                     mapper.homography_source_w,
                                     mapper.homography_source_h,
                                     court_norm)) {
                return false;
            }
            if (court_norm.x < -0.05 || court_norm.x > 1.05 ||
                court_norm.y < -0.05 || court_norm.y > 1.05) {
                return false;
            }
            const double x_ft = clampDouble(court_norm.x * 94.0, 0.0, 94.0);
            const double y_ft = clampDouble(court_norm.y * 50.0, 0.0, 50.0);
            out = courtFtToPanel(x_ft, y_ft, mapper);
            return std::isfinite(out.x) && std::isfinite(out.y);
        }

        const Point2D vec = src - mapper.hoop;
        const double depth_ft = dotPoint(vec, mapper.u) / mapper.depth_px_per_ft;
        const double lateral_ft = dotPoint(vec, mapper.v) / mapper.lateral_px_per_ft;
        if (!std::isfinite(depth_ft) || !std::isfinite(lateral_ft)) return false;

        const double hoop_x_ft = mapper.hoop_on_left ? 5.25 : 88.75;
        double x_ft = mapper.hoop_on_left ? hoop_x_ft + depth_ft : hoop_x_ft - depth_ft;
        double y_ft = 25.0 + lateral_ft;
        if (x_ft < -2.0 || x_ft > 96.0 || y_ft < -2.0 || y_ft > 52.0) return false;

        x_ft = clampDouble(x_ft, 0.0, 94.0);
        y_ft = clampDouble(y_ft, 0.0, 50.0);
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

    void appendCanonicalCourtPoints(const CourtMapper& mapper,
                                    std::vector<TacticalPoint>& points) const {
        const bool active_left = mapper.hoop_on_left;
        appendThreePointHalf(!active_left, mapper, court_line_color_, 0.85f, points);
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
            if (show_ball_ring_ && item.player.has_ball) {
                points.push_back(TacticalPoint{(float)item.projected.x, (float)item.projected.y, 7.0f,
                                               ball_ring_color_.y, ball_ring_color_.u, ball_ring_color_.v});
            }
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

        std::vector<PlayerDot> players;
        Point2D hoop;
        Point2D ball;
        bool have_hoop = false;
        bool have_ball = false;
        if (!parseFrameDump(input, players, hoop, have_hoop, ball, have_ball)) {
            tryRenderCachedOverlay(output);
            return;
        }

        MaskInfo masks;
        if (!readCpuMasks(input.raw(), court_seg_slot_, masks)) {
            tryRenderCachedOverlay(output);
            return;
        }
        const std::vector<std::string> labels = courtSegLabels(input.raw());
        int court_idx = -1;
        std::vector<int> line_indices;
        if (!chooseCourtAndLineMasks(masks, labels, court_idx, line_indices)) {
            tryRenderCachedOverlay(output);
            return;
        }

        std::vector<Point2D> line_samples = collectLineSamples(masks, line_indices, output.width(), output.height());
        std::vector<CourtPosePoint> pose_points;
        int pose_keypoints = 0;
        double pose_conf = 0.0;
        parseCourtPosePoints(input, pose_points, pose_keypoints, pose_conf);

        CourtMapper mapper;
        const bool built_mapper = buildCourtMapper(input, hoop, have_hoop, line_samples,
                                                   pose_points,
                                                   geom.court_x, geom.court_y, geom.court_w, geom.court_h,
                                                   mapper);
        if (built_mapper) {
            const bool reject_mapper_jump = mapperRejectedByPlayerContinuity(mapper, players) &&
                                            mapper_bad_frames_ < mapper_bad_hold_frames_;
            if (reject_mapper_jump && have_last_mapper_) {
                ++mapper_bad_frames_;
                mapper = last_mapper_;
                mapper.court_x = geom.court_x;
                mapper.court_y = geom.court_y;
                mapper.court_w = geom.court_w;
                mapper.court_h = geom.court_h;
                ++last_mapper_age_;
            } else {
                mapper_bad_frames_ = 0;
                mapper = stabilizeMapper(mapper);
            }
            if (have_last_mapper_ && mapper.hoop_on_left != last_mapper_.hoop_on_left) {
                trails_.clear();
                player_states_.clear();
            }
            last_mapper_ = mapper;
            have_last_mapper_ = true;
            last_mapper_age_ = 0;
        } else if (have_last_mapper_ && last_mapper_age_ < mapper_hold_frames_) {
            mapper = last_mapper_;
            mapper.court_x = geom.court_x;
            mapper.court_y = geom.court_y;
            mapper.court_w = geom.court_w;
            mapper.court_h = geom.court_h;
            ++last_mapper_age_;
        } else {
            tryRenderCachedOverlay(output);
            return;
        }

        std::vector<TacticalPoint> points;
        points.reserve((size_t)std::max(0, max_player_dots_) * (show_ball_ring_ ? 2u : 1u) + 1u);
        appendPlayerAndTrailPoints(players, mapper, points);
        if (show_ball_ && have_ball) {
            Point2D projected_ball;
            if (sourcePointToPanel(ball, mapper, projected_ball)) {
                points.push_back(TacticalPoint{(float)projected_ball.x, (float)projected_ball.y, 3.8f,
                                               ball_color_.y, ball_color_.u, ball_color_.v});
            }
        }
        const int active_hoop_on_left = mapper.hoop_on_left ? 1 : 0;
        if (!renderPoints(output, geom, points, active_hoop_on_left)) return;
        cacheOverlay(output, geom, points, active_hoop_on_left);

        if (debug_log_every_n_ > 0 && (frame_counter_ % (uint64_t)debug_log_every_n_) == 0) {
            logstream << "draw_tactical_court: frame=" << frame_counter_
                      << " players=" << players.size()
                      << " ball=" << (have_ball ? 1 : 0)
                      << " points=" << points.size()
                      << " line_samples=" << line_samples.size()
                      << " pose_kpts=" << pose_keypoints
                      << " pose_conf=" << pose_conf
                      << " homography=" << (mapper.has_homography ? 1 : 0)
                      << " homography_inliers=" << mapper.homography_inliers
                      << " homography_err_ft=" << mapper.homography_error_ft
                      << " mapper_line=" << (mapper.used_line ? 1 : 0)
                      << " mapper_pose=" << (mapper.used_pose ? 1 : 0)
                      << " mapper_pose_kpts=" << (mapper.used_pose_keypoints ? 1 : 0)
                      << " depth_px_ft=" << mapper.depth_px_per_ft
                      << " lateral_px_ft=" << mapper.lateral_px_per_ft
                      << " cached_age=" << cached_overlay_age_
                      << " panel=[" << geom.panel_x << "," << geom.panel_y << "," << geom.panel_w << "," << geom.panel_h << "]";
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

        r->metadata_key_ = params.value("metadata_key", std::string("frame_dump"));
        r->court_seg_metadata_key_ = params.value("court_seg_metadata_key", std::string("yolo_seg"));
        r->pose_metadata_key_ = params.value("pose_metadata_key", std::string("yolo_pose"));
        r->court_seg_slot_ = params.value("court_seg_slot", 0);
        r->mask_threshold_ = params.value("mask_threshold", 0.5f);
        r->require_wide_shot_ = params.value("require_wide_shot", true);
        r->panel_width_ = params.value("panel_width", 360);
        r->panel_height_ = params.value("panel_height", 220);
        r->padding_left_ = params.value("padding_left", 28);
        r->padding_bottom_ = params.value("padding_bottom", 116);
        r->inner_padding_ = params.value("inner_padding", 14);
        r->line_thickness_ = params.value("line_thickness", 2);
        r->max_line_points_ = params.value("max_line_points", 512);
        r->trail_length_ = params.value("trail_length", 18);
        r->max_players_per_team_ = params.value("max_players_per_team", 5);
        r->max_player_dots_ = params.value("max_player_dots", 10);
        r->show_trails_ = params.value("show_trails", false);
        r->show_ball_ring_ = params.value("show_ball_ring", false);
        r->show_ball_ = params.value("show_ball", true);
        r->show_unknown_players_ = params.value("show_unknown_players", false);
        r->min_foot_confidence_ = params.value("min_foot_confidence", 0.15);
        r->pose_min_keypoint_conf_ = params.value("pose_min_keypoint_conf", 0.05);
        r->pose_homography_min_keypoints_ = params.value("pose_homography_min_keypoints", 4);
        r->pose_homography_inlier_error_ft_ = params.value("pose_homography_inlier_error_ft", 8.0);
        r->pose_homography_hoop_error_ft_ = params.value("pose_homography_hoop_error_ft", 0.0);
        r->mapper_hold_frames_ = params.value("mapper_hold_frames", 12);
        r->overlay_hold_frames_ = params.value("overlay_hold_frames", 30);
        r->side_switch_confirm_frames_ = params.value("side_switch_confirm_frames", 4);
        r->mapper_smoothing_alpha_ = params.value("mapper_smoothing_alpha", 0.25);
        r->mapper_player_jump_gate_px_ = params.value("mapper_player_jump_gate_px", 55.0);
        r->mapper_player_jump_gate_min_tracks_ = params.value("mapper_player_jump_gate_min_tracks", 3);
        r->mapper_bad_hold_frames_ = params.value("mapper_bad_hold_frames", 10);
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
        parse_color("ball_color", "red", r->ball_color_);
        parse_color("team_a_color", "light_blue", r->team_a_color_);
        parse_color("team_b_color", "green", r->team_b_color_);
        parse_color("unknown_color", "light_gray", r->unknown_color_);
        parse_color("ball_ring_color", "white", r->ball_ring_color_);

        return r;
    }
};

DECLNODE(draw_tactical_court, DrawTacticalCourt)
