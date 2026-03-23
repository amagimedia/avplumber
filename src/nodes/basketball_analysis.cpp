#include "node_common.hpp"

extern "C" {
#include <libavutil/dict.h>
}

#include <algorithm>
#include <cmath>
#include <deque>
#include <limits>
#include <string>
#include <tuple>
#include <vector>

namespace {

struct DetectionBox {
    int cls = -1;
    std::string label;
    bool has_label = false;
    double conf = 0.0;
    double x1 = 0.0, y1 = 0.0, x2 = 0.0, y2 = 0.0;
    int model_index = -1;
    std::string engine_name;
    bool has_engine_name = false;
    bool extrapolated = false;
};

struct MetadataEnvelope {
    int version = 1;
    std::string coord_space = "model";
    double model_width = 0.0, model_height = 0.0;
    Parameters thresholds;
};

static double centerX(const DetectionBox& b) { return (b.x1 + b.x2) * 0.5; }
static double centerY(const DetectionBox& b) { return (b.y1 + b.y2) * 0.5; }

static double centerDist(const DetectionBox& a, const DetectionBox& b) {
    double dx = centerX(a) - centerX(b), dy = centerY(a) - centerY(b);
    return std::sqrt(dx * dx + dy * dy);
}

static bool boxesIntersect(const DetectionBox& a, const DetectionBox& b) {
    return a.x1 <= b.x2 && a.x2 >= b.x1 &&
           a.y1 <= b.y2 && a.y2 >= b.y1;
}

// Distance from point (px,py) to nearest edge of box. Zero when inside.
static double edgeDistance(double px, double py, const DetectionBox& box) {
    double dx = std::max({0.0, box.x1 - px, px - box.x2});
    double dy = std::max({0.0, box.y1 - py, py - box.y2});
    return std::sqrt(dx * dx + dy * dy);
}

static double nearestPlayerEdgeDist(double px, double py,
                                     const std::vector<DetectionBox>& players) {
    double best = std::numeric_limits<double>::max();
    for (const auto& p : players) best = std::min(best, edgeDistance(px, py, p));
    return best;
}

static double lerp(double a, double b, double t) { return a + (b - a) * t; }

static DetectionBox lerpBox(const DetectionBox& a, const DetectionBox& b, double t) {
    DetectionBox r = b;
    r.x1 = lerp(a.x1, b.x1, t); r.y1 = lerp(a.y1, b.y1, t);
    r.x2 = lerp(a.x2, b.x2, t); r.y2 = lerp(a.y2, b.y2, t);
    r.conf = lerp(a.conf, b.conf, t);
    return r;
}

static std::vector<std::pair<int, int>> matchBoxes(
    const std::vector<DetectionBox>& last,
    const std::vector<DetectionBox>& cur,
    double max_dist) {
    std::vector<std::tuple<double, int, int>> cands;
    for (int i = 0; i < (int)last.size(); i++)
        for (int j = 0; j < (int)cur.size(); j++) {
            double d = centerDist(last[i], cur[j]);
            if (d <= max_dist) cands.emplace_back(d, i, j);
        }
    std::sort(cands.begin(), cands.end());
    std::vector<bool> lu(last.size(), false), cu(cur.size(), false);
    std::vector<std::pair<int, int>> out;
    for (auto& [d, i, j] : cands) {
        if (!lu[i] && !cu[j]) { out.emplace_back(i, j); lu[i] = cu[j] = true; }
    }
    return out;
}

// ── Least-squares fits ──────────────────────────────────────────

static bool fitQuadratic(const std::vector<double>& t,
                          const std::vector<double>& y,
                          double& a, double& b, double& c, double& rmse) {
    int n = (int)t.size();
    if (n < 3) return false;
    double s0 = n, s1 = 0, s2 = 0, s3 = 0, s4 = 0;
    double sy = 0, sty = 0, st2y = 0;
    for (int i = 0; i < n; i++) {
        double ti = t[i], ti2 = ti * ti;
        s1 += ti; s2 += ti2; s3 += ti2 * ti; s4 += ti2 * ti2;
        sy += y[i]; sty += ti * y[i]; st2y += ti2 * y[i];
    }
    double det = s4*(s2*s0 - s1*s1) - s3*(s3*s0 - s1*s2) + s2*(s3*s1 - s2*s2);
    if (std::abs(det) < 1e-12) return false;
    a = (st2y*(s2*s0 - s1*s1) - s3*(sty*s0 - s1*sy) + s2*(sty*s1 - s2*sy)) / det;
    b = (s4*(sty*s0 - s1*sy) - st2y*(s3*s0 - s1*s2) + s2*(s3*sy - sty*s2)) / det;
    c = (s4*(s2*sy - sty*s1) - s3*(s3*sy - sty*s2) + st2y*(s3*s1 - s2*s2)) / det;
    double sse = 0;
    for (int i = 0; i < n; i++) { double e = y[i] - (a*t[i]*t[i] + b*t[i] + c); sse += e*e; }
    rmse = std::sqrt(sse / n);
    return true;
}

// ── Ball candidate ──────────────────────────────────────────────

struct PositionEntry { uint64_t frame; double cx, cy; };

struct TrajectoryFit {
    double ax = 0, bx = 0, cx_coeff = 0;
    double ay = 0, by = 0, cy_coeff = 0;
    double residual = std::numeric_limits<double>::max();
    bool valid = false;
};

struct BallCandidate {
    int id = -1;
    std::deque<PositionEntry> history;
    DetectionBox last_det;
    uint64_t last_match_frame = 0;
    bool valid = false;
    double speed = 0.0;
    TrajectoryFit traj;
    double trajectory_score = 0.0;
    uint64_t last_near_player_frame = 0;
    double last_near_player_dist = std::numeric_limits<double>::max();
    int consecutive_near_frames = 0;
    bool armed = false;
};

} // namespace

enum class MergeMode { PFB, PFB_GAP, BALL_ONLY };
enum class AnalysisEventType { NONE, SHOT };

static const char* analysisEventTypeName(AnalysisEventType type) {
    switch (type) {
    case AnalysisEventType::SHOT: return "shot";
    case AnalysisEventType::NONE: default: return "none";
    }
}

class BasketballAnalysis : public NodeSISO<av::VideoFrame, av::VideoFrame>, public IInputReset {
private:
    struct AnalysisEventSnapshot {
        AnalysisEventType type = AnalysisEventType::NONE;
        int ball_id = -1;
        uint64_t frame = 0;
        double travel_px = 0.0;
        int tracked_frames = 0;
        int extrapolated_frames = 0;
    };

    // ── Config ──────────────────────────────────────────────
    std::string metadata_key_in_       = "yolo_detections_v1";
    std::string metadata_key_out_      = "basketball_analysis_v1";
    int    ball_model_index_           = 0;
    int    context_model_index_        = 1;
    std::vector<int> passthrough_model_indices_;
    std::vector<std::string> passthrough_labels_; // if non-empty, only pass through these labels
    std::string ball_label_            = "basketball";
    std::string player_label_          = "player";
    std::string pfb_ball_label_        = "ball";
    double min_speed_px_per_frame_     = 5.0;
    double arm_dist_px_                = 50.0;
    int    arm_frames_                 = 2;
    int    confirm_frames_             = 3;
    int    max_hold_frames_            = 12;
    double max_match_distance_px_      = 200.0;
    int    history_frames_             = 6;
    int    candidate_drop_frames_      = 3;
    int    pfb_possession_grace_frames_ = 2;
    int    release_confirm_frames_      = 2;
    int    release_possession_memory_frames_ = 6;
    int    shot_make_min_frames_        = 4;
    double shot_make_min_travel_px_     = 70.0;
    int    shot_hoop_memory_frames_     = 24;
    double hoop_min_conf_               = 0.80;

    // ── State ───────────────────────────────────────────────
    MergeMode mode_ = MergeMode::PFB;
    uint64_t  frame_counter_ = 0;

    // Ball candidates
    std::vector<BallCandidate> candidates_;
    int next_candidate_id_ = 0;
    int active_ball_id_    = -1;

    // Player state
    std::vector<DetectionBox> last_player_dets_;
    bool     have_last_player_dets_ = false;
    uint64_t last_player_frame_     = 0;
    int      hold_frames_emitted_   = 0;
    DetectionBox last_pfb_possession_ball_det_;
    bool     have_last_pfb_possession_ball_ = false;
    uint64_t last_pfb_possession_frame_     = 0;
    uint64_t last_pfb_visible_frame_        = 0;
    bool     have_last_pfb_visible_         = false;
    bool     current_pfb_visible_           = false;
    bool     current_pfb_possession_        = false;
    bool     recent_pfb_possession_         = false;
    bool     awaiting_fresh_pfb_release_    = false;
    int      pending_release_ball_id_       = -1;
    int      pending_release_frames_        = 0;

    // Buffer & confirmation
    std::vector<av::VideoFrame> gap_buffer_;
    int consecutive_player_frames_ = 0;

    // Stats for logging
    uint64_t ball_only_entered_frame_       = 0;
    int      ball_only_tracked_frames_      = 0;
    int      ball_only_extrapolated_frames_ = 0;

    // Cooldown: don't re-enter BALL_ONLY too quickly after returning to PFB
    uint64_t last_ball_only_exit_frame_     = 0;
    static constexpr int kBallOnlyCooldownFrames = 30; // ~1 second at 30fps

    // Active shot: track a confirmed release until it reaches the hoop zone or ends.
    int      active_shot_id_               = -1;
    int      shot_extrapolated_frames_     = 0;

    // Hoop tracking: use hoop bbox to validate shots
    DetectionBox last_hoop_det_;
    bool     have_hoop_                    = false;
    uint64_t last_hoop_frame_              = 0;
    std::string hoop_label_                = "hoop";
    static constexpr int kHoopHoldFrames   = 2;

    // Shot outcome counting (persists across resetState — game total)
    int      total_shots_detected_         = 0;
    int      last_event_ball_id_           = -1;
    double   shot_start_x_                 = 0;
    double   shot_start_y_                 = 0;
    int      shot_tracking_frames_         = 0;
    bool     shot_counted_                 = false;
    bool     shot_reached_hoop_zone_       = false;
    bool     shot_scored_                  = false;
    DetectionBox last_shot_eval_det_;
    bool     have_last_shot_eval_det_      = false;
    AnalysisEventSnapshot last_event_;

    // ── Reset ───────────────────────────────────────────────

    void resetState() {
        mode_ = MergeMode::PFB;
        candidates_.clear();
        next_candidate_id_ = 0;
        active_ball_id_ = -1;
        last_player_dets_.clear();
        have_last_player_dets_ = false;
        last_player_frame_ = 0;
        hold_frames_emitted_ = 0;
        have_last_pfb_possession_ball_ = false;
        last_pfb_possession_frame_ = 0;
        last_pfb_visible_frame_ = 0;
        have_last_pfb_visible_ = false;
        current_pfb_visible_ = false;
        current_pfb_possession_ = false;
        recent_pfb_possession_ = false;
        awaiting_fresh_pfb_release_ = false;
        pending_release_ball_id_ = -1;
        pending_release_frames_ = 0;
        gap_buffer_.clear();
        consecutive_player_frames_ = 0;
        ball_only_entered_frame_ = 0;
        ball_only_tracked_frames_ = 0;
        ball_only_extrapolated_frames_ = 0;
        last_ball_only_exit_frame_ = 0;
        active_shot_id_ = -1;
        shot_extrapolated_frames_ = 0;
        have_hoop_ = false;
        last_hoop_frame_ = 0;
        last_event_ball_id_ = -1;
        shot_counted_ = false;
        shot_reached_hoop_zone_ = false;
        shot_scored_ = false;
        have_last_shot_eval_det_ = false;
        last_event_ = AnalysisEventSnapshot{};
    }

    bool hasFreshHoop() const {
        return have_hoop_
            && (frame_counter_ - last_hoop_frame_) <= (uint64_t)kHoopHoldFrames;
    }

    bool hasShotHoopReference() const {
        return have_hoop_
            && (frame_counter_ - last_hoop_frame_) <= (uint64_t)shot_hoop_memory_frames_;
    }

    bool shouldDrawHeldHoop() const {
        if (!have_hoop_) return false;
        if (hasFreshHoop()) return true;
        return active_shot_id_ >= 0 || mode_ == MergeMode::BALL_ONLY;
    }

    int currentExtrapolatedFrames() const {
        return mode_ == MergeMode::BALL_ONLY ? ball_only_extrapolated_frames_
                                             : shot_extrapolated_frames_;
    }

    const char* activeEventName() const {
        if (active_shot_id_ >= 0 || active_ball_id_ >= 0)
            return "shot";
        return "none";
    }

    void recordCompletedEvent(AnalysisEventType type, int ball_id, double travel_px) {
        last_event_.type = type;
        last_event_.ball_id = ball_id;
        last_event_.frame = frame_counter_;
        last_event_.travel_px = travel_px;
        last_event_.tracked_frames = shot_tracking_frames_;
        last_event_.extrapolated_frames = currentExtrapolatedFrames();
    }

    // ── Parsing ─────────────────────────────────────────────

    bool parseDetections(const av::VideoFrame& frm, MetadataEnvelope& env,
                         std::vector<DetectionBox>& dets) const {
        dets.clear();
        env = MetadataEnvelope{};
        env.model_width = frm.width();
        env.model_height = frm.height();
        const AVFrame* raw = frm.raw();
        if (!raw || !raw->metadata) return false;
        AVDictionaryEntry* entry = av_dict_get(raw->metadata,
                                               metadata_key_in_.c_str(), nullptr, 0);
        if (!entry || !entry->value) return false;
        try {
            Parameters md = Parameters::parse(entry->value);
            env.version      = md.value("version", 1);
            env.coord_space  = md.value("coord_space", std::string("model"));
            env.model_width  = md.value("model_width",  (double)frm.width());
            env.model_height = md.value("model_height", (double)frm.height());
            if (md.contains("thresholds")) env.thresholds = md["thresholds"];
            if (!md.contains("detections") || !md["detections"].is_array()) return true;
            for (const auto& item : md["detections"]) {
                if (!item.is_object()) continue;
                if (!item.contains("xyxy") || !item["xyxy"].is_array()
                    || item["xyxy"].size() < 4) continue;
                DetectionBox det;
                det.cls         = item.value("cls", -1);
                det.conf        = item.value("conf", 0.0);
                det.model_index = item.value("model_index", -1);
                det.x1 = item["xyxy"][0].get<double>();
                det.y1 = item["xyxy"][1].get<double>();
                det.x2 = item["xyxy"][2].get<double>();
                det.y2 = item["xyxy"][3].get<double>();
                if (item.contains("label") && item["label"].is_string()) {
                    det.label = item["label"].get<std::string>();
                    det.has_label = true;
                }
                if (item.contains("engine_name") && item["engine_name"].is_string()) {
                    det.engine_name = item["engine_name"].get<std::string>();
                    det.has_engine_name = true;
                }
                dets.push_back(det);
            }
            return true;
        } catch (const std::exception&) { return false; }
    }

    // ── Metadata output ─────────────────────────────────────

    Parameters buildMetadata(const MetadataEnvelope& env,
                              const std::vector<DetectionBox>& dets,
                              const std::string& mode_tag) const {
        Parameters md;
        md["version"]      = 1;
        md["coord_space"]  = env.coord_space;
        md["model_width"]  = env.model_width;
        md["model_height"] = env.model_height;
        if (!env.thresholds.is_null()) md["thresholds"] = env.thresholds;
        md["detections"] = Parameters::array();
        for (const auto& det : dets) {
            Parameters item;
            item["cls"]         = det.cls;
            item["conf"]        = det.conf;
            item["xyxy"]        = {det.x1, det.y1, det.x2, det.y2};
            item["model_index"] = det.model_index;
            if (det.has_label)       item["label"]       = det.label;
            if (det.has_engine_name) item["engine_name"]  = det.engine_name;
            if (det.extrapolated)    item["extrapolated"] = true;
            md["detections"].push_back(item);
        }
        md["basketball_analysis"] = {
            {"mode", mode_tag},
            {"active_ball_candidate_id", active_ball_id_},
            {"analysis", {
                {"event", {
                    {"type", analysisEventTypeName(last_event_.type)},
                    {"ball_id", last_event_.ball_id},
                    {"frame", last_event_.frame},
                    {"travel_px", last_event_.travel_px},
                    {"tracked_frames", last_event_.tracked_frames},
                    {"extrapolated_frames", last_event_.extrapolated_frames}
                }},
                {"state", {
                    {"active_event", activeEventName()},
                    {"shot_attempt", shot_counted_},
                    {"has_fresh_hoop", hasFreshHoop()},
                    {"ball_only_mode", mode_ == MergeMode::BALL_ONLY},
                    {"pending_release_frames", pending_release_frames_},
                    {"reached_hoop_zone", shot_reached_hoop_zone_},
                    {"scored", shot_scored_}
                }},
                {"totals", {
                    {"shots", total_shots_detected_}
                }}
            }}
        };
        return md;
    }

    void appendPassthroughDetections(av::VideoFrame& frm,
                                     std::vector<DetectionBox>& dets) const {
        if (passthrough_model_indices_.empty()) return;
        MetadataEnvelope tmp_env;
        std::vector<DetectionBox> orig_dets;
        parseDetections(frm, tmp_env, orig_dets);

        bool saw_passthrough_hoop = false;
        for (const auto& d : orig_dets) {
            for (int mi : passthrough_model_indices_) {
                if (d.model_index == mi) {
                    if (!passthrough_labels_.empty()) {
                        if (!d.has_label) break; // no label -> skip
                        bool found = false;
                        for (const auto& lbl : passthrough_labels_) {
                            if (d.label == lbl) { found = true; break; }
                        }
                        if (!found) break;
                    }
                    if (d.has_label && d.label == hoop_label_ && d.conf < hoop_min_conf_) break;
                    if (d.has_label && d.label == hoop_label_) saw_passthrough_hoop = true;
                    dets.push_back(d);
                    break;
                }
            }
        }

        // Keep the last valid hoop visible through an active shot event.
        if (!saw_passthrough_hoop && shouldDrawHeldHoop()) dets.push_back(last_hoop_det_);
    }

    void writeMetadata(av::VideoFrame& frm, const MetadataEnvelope& env,
                       const std::vector<DetectionBox>& dets,
                       const std::string& mode_tag) {
        auto all = dets;
        appendPassthroughDetections(frm, all);
        const std::string s = buildMetadata(env, all, mode_tag).dump();
        av_dict_set(&frm.raw()->metadata, metadata_key_out_.c_str(), s.c_str(), 0);
    }

    // ── Filtering helpers ───────────────────────────────────

    std::vector<DetectionBox> filterByModel(const std::vector<DetectionBox>& dets,
                                             int mi) const {
        std::vector<DetectionBox> r;
        for (const auto& d : dets) if (d.model_index == mi) r.push_back(d);
        return r;
    }

    std::vector<DetectionBox> filterByLabel(const std::vector<DetectionBox>& dets,
                                             const std::string& lbl) const {
        std::vector<DetectionBox> r;
        for (const auto& d : dets)
            if (d.has_label && d.label == lbl) r.push_back(d);
        return r;
    }

    std::vector<DetectionBox> withoutLabel(const std::vector<DetectionBox>& dets,
                                           const std::string& lbl) const {
        std::vector<DetectionBox> r;
        r.reserve(dets.size());
        for (const auto& d : dets)
            if (!d.has_label || d.label != lbl) r.push_back(d);
        return r;
    }

    DetectionBox chooseBestPfbBall(const std::vector<DetectionBox>& pfb_ball_dets,
                                   const std::vector<DetectionBox>& player_dets) const {
        if (pfb_ball_dets.empty()) return {};

        const auto& prox_players = player_dets.empty() ? last_player_dets_ : player_dets;
        if (!prox_players.empty()) {
            double best_dist = std::numeric_limits<double>::max();
            int best_idx = -1;
            for (int i = 0; i < (int)pfb_ball_dets.size(); i++) {
                double dist = nearestPlayerEdgeDist(centerX(pfb_ball_dets[i]),
                                                    centerY(pfb_ball_dets[i]),
                                                    prox_players);
                if (dist < best_dist) {
                    best_dist = dist;
                    best_idx = i;
                }
            }
            if (best_idx >= 0) return pfb_ball_dets[best_idx];
        }

        DetectionBox best = pfb_ball_dets.front();
        for (const auto& d : pfb_ball_dets) {
            if (d.conf > best.conf) best = d;
        }
        return best;
    }

    std::vector<DetectionBox> buildPfbOutputDetections(
        const std::vector<DetectionBox>& pfb_dets,
        const std::vector<DetectionBox>& player_dets,
        const std::vector<DetectionBox>& pfb_ball_dets) const {
        std::vector<DetectionBox> out = withoutLabel(pfb_dets, pfb_ball_label_);

        // Only expose a single analysis-approved ball, never the raw set of PFB balls.
        if (active_shot_id_ >= 0) {
            DetectionBox shot = getShotDetection();
            if (shot.has_label) out.push_back(shot);
            return out;
        }

        if (!pfb_ball_dets.empty()) {
            DetectionBox selected = chooseBestPfbBall(pfb_ball_dets, player_dets);
            if (selected.has_label || selected.cls >= 0) out.push_back(selected);
        }

        return out;
    }

    void updatePfbPossession(const std::vector<DetectionBox>& pfb_dets,
                             const std::vector<DetectionBox>& pfb_ball_dets,
                             const std::vector<DetectionBox>& player_dets) {
        const auto& prox_players = player_dets.empty() ? last_player_dets_ : player_dets;
        current_pfb_visible_ = !pfb_dets.empty();
        current_pfb_possession_ = false;

        if (current_pfb_visible_) {
            have_last_pfb_visible_ = true;
            last_pfb_visible_frame_ = frame_counter_;
        }

        if (!prox_players.empty()) {
            double best_dist = std::numeric_limits<double>::max();
            int best_idx = -1;
            for (int i = 0; i < (int)pfb_ball_dets.size(); i++) {
                double dist = nearestPlayerEdgeDist(centerX(pfb_ball_dets[i]),
                                                    centerY(pfb_ball_dets[i]),
                                                    prox_players);
                if (dist < best_dist) {
                    best_dist = dist;
                    best_idx = i;
                }
            }
            if (best_idx >= 0 && best_dist <= arm_dist_px_) {
                current_pfb_possession_ = true;
                recent_pfb_possession_ = true;
                awaiting_fresh_pfb_release_ = false;
                last_pfb_possession_ball_det_ = pfb_ball_dets[best_idx];
                have_last_pfb_possession_ball_ = true;
                last_pfb_possession_frame_ = frame_counter_;
                pending_release_ball_id_ = -1;
                pending_release_frames_ = 0;
            }
        }

        if (!current_pfb_possession_) {
            recent_pfb_possession_ = have_last_pfb_possession_ball_
                && (frame_counter_ - last_pfb_possession_frame_)
                       <= (uint64_t)pfb_possession_grace_frames_;
        }
    }

    bool candidateMatchesRecentPfbPossession(const BallCandidate& c) const {
        if (!have_last_pfb_possession_ball_) return false;
        return centerDist(c.last_det, last_pfb_possession_ball_det_) <= max_match_distance_px_;
    }

    bool hasRecentReleaseSource() const {
        return have_last_pfb_possession_ball_
            && (frame_counter_ - last_pfb_possession_frame_)
                   <= (uint64_t)release_possession_memory_frames_;
    }

    DetectionBox expandedHoopZone() const {
        DetectionBox zone = last_hoop_det_;
        const double hoop_w = std::max(1.0, last_hoop_det_.x2 - last_hoop_det_.x1);
        const double hoop_h = std::max(1.0, last_hoop_det_.y2 - last_hoop_det_.y1);
        const double pad_x = hoop_w * 1.10;
        const double pad_top = hoop_h * 1.10;
        const double pad_bottom = hoop_h * 1.85;
        zone.x1 -= pad_x;
        zone.x2 += pad_x;
        zone.y1 -= pad_top;
        zone.y2 += pad_bottom;
        return zone;
    }

    DetectionBox goodShotDropZone() const {
        DetectionBox zone = last_hoop_det_;
        const double hoop_w = std::max(1.0, last_hoop_det_.x2 - last_hoop_det_.x1);
        const double hoop_h = std::max(1.0, last_hoop_det_.y2 - last_hoop_det_.y1);
        zone.x1 -= hoop_w * 0.35;
        zone.x2 += hoop_w * 0.35;
        zone.y1 = centerY(last_hoop_det_) - hoop_h * 0.25;
        zone.y2 += hoop_h * 1.5;
        return zone;
    }

    bool ballCenterInside(const DetectionBox& box, const DetectionBox& ball_det) const {
        const double bx = centerX(ball_det);
        const double by = centerY(ball_det);
        return bx >= box.x1 && bx <= box.x2 && by >= box.y1 && by <= box.y2;
    }

    bool isGoodShotGeometry(const DetectionBox& ball_det) const {
        if (boxesIntersect(ball_det, last_hoop_det_)) return true;
        if (!shot_reached_hoop_zone_) return false;

        const double hoop_w = std::max(1.0, last_hoop_det_.x2 - last_hoop_det_.x1);
        const double hoop_center_y = centerY(last_hoop_det_);
        const double bx = centerX(ball_det);
        const double by = centerY(ball_det);
        if (!ballCenterInside(goodShotDropZone(), ball_det)) return false;
        if (bx < last_hoop_det_.x1 - hoop_w || bx > last_hoop_det_.x2 + hoop_w) return false;
        if (by < hoop_center_y) return false;
        if (!have_last_shot_eval_det_) return true;

        const double prev_y = centerY(last_shot_eval_det_);
        return prev_y <= by && prev_y <= last_hoop_det_.y2 + hoop_w;
    }

    void finalizeShotOutcome(int ball_id, const DetectionBox* det) {
        (void)ball_id;
        (void)det;
    }

    void countShotAttemptIfNeeded(int ball_id, double travel_px) {
        if (ball_id == last_event_ball_id_) return;
        total_shots_detected_++;
        last_event_ball_id_ = ball_id;
        shot_counted_ = true;
        recordCompletedEvent(AnalysisEventType::SHOT, ball_id, travel_px);
    }

    double shotTravelPx(const DetectionBox& ball_det) const {
        const double dx = centerX(ball_det) - shot_start_x_;
        const double dy = centerY(ball_det) - shot_start_y_;
        return std::sqrt(dx * dx + dy * dy);
    }

    // ── Ball candidate tracker ──────────────────────────────

    void fitCandidateTrajectory(BallCandidate& c) {
        c.traj = TrajectoryFit{};
        c.trajectory_score = 0;
        if ((int)c.history.size() < 3) return;
        std::vector<double> tv, xv, yv;
        uint64_t t0 = c.history.front().frame;
        for (const auto& e : c.history) {
            tv.push_back((double)(e.frame - t0));
            xv.push_back(e.cx);
            yv.push_back(e.cy);
        }
        double ax, bx, cxc, rmx, ay, by, cyc, rmy;
        bool fx = fitQuadratic(tv, xv, ax, bx, cxc, rmx);
        bool fy = fitQuadratic(tv, yv, ay, by, cyc, rmy);
        if (fx && fy) {
            c.traj.ax = ax; c.traj.bx = bx; c.traj.cx_coeff = cxc;
            c.traj.ay = ay; c.traj.by = by; c.traj.cy_coeff = cyc;
            c.traj.residual = std::sqrt(rmx * rmx + rmy * rmy);
            c.traj.valid = true;
            c.trajectory_score = 1.0 / (c.traj.residual + 1.0);
        }
    }

    void updateCandidates(const std::vector<DetectionBox>& ball_dets,
                          const std::vector<DetectionBox>& player_dets) {
        // Match detections to existing candidates by nearest center
        std::vector<std::tuple<double, int, int>> pairs;
        for (int ci = 0; ci < (int)candidates_.size(); ci++)
            for (int di = 0; di < (int)ball_dets.size(); di++) {
                DetectionBox match_ref = candidates_[ci].last_det;
                const bool is_tracked_release = candidates_[ci].id == active_shot_id_
                    || candidates_[ci].id == active_ball_id_;
                if (is_tracked_release &&
                    candidates_[ci].last_match_frame < frame_counter_ &&
                    candidates_[ci].traj.valid) {
                    match_ref = extrapolateBall(
                        candidates_[ci],
                        (int)(frame_counter_ - candidates_[ci].last_match_frame));
                }
                double d = centerDist(match_ref, ball_dets[di]);
                if (d <= max_match_distance_px_) pairs.emplace_back(d, ci, di);
            }
        std::sort(pairs.begin(), pairs.end());

        std::vector<bool> cm(candidates_.size(), false), dm(ball_dets.size(), false);
        for (auto& [d, ci, di] : pairs) {
            if (cm[ci] || dm[di]) continue;
            cm[ci] = dm[di] = true;
            BallCandidate& c = candidates_[ci];
            c.last_det = ball_dets[di];
            c.last_match_frame = frame_counter_;
            double cx = centerX(ball_dets[di]), cy = centerY(ball_dets[di]);
            c.history.push_back({frame_counter_, cx, cy});
            while ((int)c.history.size() > history_frames_) c.history.pop_front();
            if (c.history.size() >= 2) {
                auto& prev = c.history[c.history.size() - 2];
                double dx = cx - prev.cx, dy = cy - prev.cy;
                double df = (double)(frame_counter_ - prev.frame);
                if (df > 0) c.speed = std::sqrt(dx*dx + dy*dy) / df;
            }
            c.valid = c.speed >= min_speed_px_per_frame_;
            fitCandidateTrajectory(c);
        }

        // New candidates for unmatched detections
        for (int di = 0; di < (int)ball_dets.size(); di++) {
            if (dm[di]) continue;
            BallCandidate nc;
            nc.id = next_candidate_id_++;
            nc.last_det = ball_dets[di];
            nc.last_match_frame = frame_counter_;
            nc.history.push_back({frame_counter_,
                                  centerX(ball_dets[di]), centerY(ball_dets[di])});
            candidates_.push_back(std::move(nc));
        }

        // Drop stale (active ball exempt)
        candidates_.erase(
            std::remove_if(candidates_.begin(), candidates_.end(),
                [this](const BallCandidate& c) {
                    if (c.id == active_ball_id_ || c.id == active_shot_id_) return false;
                    return (frame_counter_ - c.last_match_frame) >
                           (uint64_t)candidate_drop_frames_;
                }),
            candidates_.end());

        // Update player proximity & arming
        // Use last known player positions when current detections are empty
        const auto& proximity_players = player_dets.empty()
            ? last_player_dets_ : player_dets;
        for (auto& c : candidates_) {
            if (current_pfb_possession_) {
                c.consecutive_near_frames = 0;
                c.armed = false;
                continue;
            }
            if (proximity_players.empty()) {
                c.consecutive_near_frames = 0;
                continue;
            }
            double bcx = centerX(c.last_det), bcy = centerY(c.last_det);
            double dist = nearestPlayerEdgeDist(bcx, bcy, proximity_players);
            if (dist <= arm_dist_px_) {
                c.consecutive_near_frames++;
                c.last_near_player_frame = frame_counter_;
                c.last_near_player_dist = dist;
                if (c.consecutive_near_frames >= arm_frames_ && !c.armed) {
                    logstream << "basketball_analysis: arming: ball candidate #" << c.id
                              << " within " << (int)dist
                              << "px of player box edge for "
                              << c.consecutive_near_frames << " consecutive frames"
                              << ", speed=" << (int)c.speed << "px/f";
                    c.armed = true;
                }
            } else {
                c.consecutive_near_frames = 0;
            }
        }
    }

    // Update active shot tracking in PFB mode:
    // PFB remains authoritative during possession; basketball activates only
    // after a confirmed release that continues from a recent PFB-held ball.
    void updateActiveShot(const std::vector<DetectionBox>& player_dets) {
        const auto& prox_players = player_dets.empty() ? last_player_dets_ : player_dets;

        // Check if current shot is still valid
        if (active_shot_id_ >= 0) {
            BallCandidate* shot = findCandidate(active_shot_id_);
            if (!shot) {
                logstream << "basketball_analysis: shot tracking lost: active shot candidate #"
                          << active_shot_id_
                          << " was removed before event completion";
                finalizeShotOutcome(active_shot_id_, nullptr);
                active_shot_id_ = -1;
                shot_extrapolated_frames_ = 0;
                return;
            }
            bool matched = (shot->last_match_frame == frame_counter_);
            if (matched) {
                shot_extrapolated_frames_ = 0;
                // Check if ball reached hoop
                trackBallEvent(shot->last_det, shot->id);
                // Ball returned near player → shot over
                if (!prox_players.empty()) {
                    double bcx = centerX(shot->last_det), bcy = centerY(shot->last_det);
                    double dist = nearestPlayerEdgeDist(bcx, bcy, prox_players);
                    if (dist <= arm_dist_px_) {
                        finalizeShotOutcome(shot->id, &shot->last_det);
                        logstream << "basketball_analysis: shot attempt ended: ball #" << shot->id
                                  << " returned near player (dist=" << (int)dist << "px)";
                        active_shot_id_ = -1;
                        return;
                    }
                }
            } else {
                shot_extrapolated_frames_++;
                if (shot->traj.valid) {
                    DetectionBox extrap = extrapolateBall(*shot, shot_extrapolated_frames_);
                    trackBallEvent(extrap, shot->id);
                } else {
                    DetectionBox held = shot->last_det;
                    held.extrapolated = true;
                    trackBallEvent(held, shot->id);
                }
                if (shot_extrapolated_frames_ > max_hold_frames_) {
                    DetectionBox final_det = shot->traj.valid
                        ? extrapolateBall(*shot, shot_extrapolated_frames_)
                        : shot->last_det;
                    finalizeShotOutcome(shot->id, &final_det);
                    logstream << "basketball_analysis: shot ended: ball #" << shot->id
                              << " lost after " << shot_extrapolated_frames_ << " frames";
                    active_shot_id_ = -1;
                    shot_extrapolated_frames_ = 0;
                    return;
                }
            }
            return; // already tracking a shot
        }

        if (current_pfb_possession_ || !hasRecentReleaseSource() || prox_players.empty()) {
            pending_release_ball_id_ = -1;
            pending_release_frames_ = 0;
            return;
        }

        if (awaiting_fresh_pfb_release_) {
            pending_release_ball_id_ = -1;
            pending_release_frames_ = 0;
            return;
        }

        BallCandidate* best_release = nullptr;
        double best_release_dist = std::numeric_limits<double>::max();
        double best_player_dist = 0.0;
        for (auto& c : candidates_) {
            if (!c.valid) continue;
            if (c.last_match_frame != frame_counter_) continue;
            if (!candidateMatchesRecentPfbPossession(c)) continue;
            double bcx = centerX(c.last_det), bcy = centerY(c.last_det);
            double dist = nearestPlayerEdgeDist(bcx, bcy, prox_players);
            if (dist <= arm_dist_px_) continue;
            double release_dist = centerDist(c.last_det, last_pfb_possession_ball_det_);
            if (!best_release || release_dist < best_release_dist) {
                best_release = &c;
                best_release_dist = release_dist;
                best_player_dist = dist;
            }
        }

        if (!best_release) {
            pending_release_ball_id_ = -1;
            pending_release_frames_ = 0;
            return;
        }

        if (pending_release_ball_id_ == best_release->id) {
            pending_release_frames_++;
        } else {
            pending_release_ball_id_ = best_release->id;
            pending_release_frames_ = 1;
        }

        if (pending_release_frames_ < release_confirm_frames_) return;

        const int confirmed_frames = pending_release_frames_;
        active_shot_id_ = best_release->id;
        shot_extrapolated_frames_ = 0;
        awaiting_fresh_pfb_release_ = true;
        pending_release_ball_id_ = -1;
        pending_release_frames_ = 0;
        startEventTracking(best_release->last_det);
        logstream << "basketball_analysis: confirmed release: ball #"
                  << best_release->id
                  << " (dist_from_player=" << (int)best_player_dist
                  << "px, release_from_pfb=" << (int)best_release_dist
                  << "px, speed=" << (int)best_release->speed << "px/f"
                  << ", confirm_frames=" << confirmed_frames
                  << (hasFreshHoop() ? ", hoop visible" : ", hoop not visible")
                  << ")";
    }

    // Get the active shot ball detection (for overlaying on PFB output)
    // Returns null if no active shot or ball lost
    const DetectionBox* getActiveShotDet() const {
        if (active_shot_id_ < 0) return nullptr;
        for (const auto& c : candidates_) {
            if (c.id != active_shot_id_) continue;
            if (c.last_match_frame == frame_counter_)
                return &c.last_det;
            // Extrapolate if recently lost
            if (shot_extrapolated_frames_ <= max_hold_frames_ && c.traj.valid)
                return &c.last_det; // caller will extrapolate
            return nullptr;
        }
        return nullptr;
    }

    DetectionBox getShotDetection() const {
        if (active_shot_id_ < 0) return {};
        for (const auto& c : candidates_) {
            if (c.id != active_shot_id_) continue;
            if (c.last_match_frame == frame_counter_)
                return c.last_det;
            // Extrapolate
            if (c.traj.valid && shot_extrapolated_frames_ <= max_hold_frames_) {
                int ahead = (int)(frame_counter_ - c.last_match_frame);
                return extrapolateBall(c, ahead);
            }
            return c.last_det; // hold last position
        }
        return {};
    }

    BallCandidate* findArmedValidCandidate() {
        if (active_shot_id_ < 0) return nullptr;

        // Cooldown: don't re-enter BALL_ONLY too soon after leaving it
        if (frame_counter_ - last_ball_only_exit_frame_ < (uint64_t)kBallOnlyCooldownFrames)
            return nullptr;

        // Return the active shot candidate
        BallCandidate* shot = findCandidate(active_shot_id_);
        if (shot && shot->valid) return shot;
        return nullptr;
    }

    BallCandidate* findCandidate(int id) {
        for (auto& c : candidates_) if (c.id == id) return &c;
        return nullptr;
    }

    DetectionBox extrapolateBall(const BallCandidate& c, int frames_ahead) const {
        DetectionBox det = c.last_det;
        det.extrapolated = true;
        if (!c.traj.valid || c.history.empty()) return det;
        uint64_t t0 = c.history.front().frame;
        double t = (double)(c.last_match_frame - t0) + frames_ahead;
        double px = c.traj.ax*t*t + c.traj.bx*t + c.traj.cx_coeff;
        double py = c.traj.ay*t*t + c.traj.by*t + c.traj.cy_coeff;
        double w = det.x2 - det.x1, h = det.y2 - det.y1;
        det.x1 = px - w*0.5; det.y1 = py - h*0.5;
        det.x2 = px + w*0.5; det.y2 = py + h*0.5;
        return det;
    }

    // Track shot progress during BALL_ONLY mode.
    void trackBallEvent(const DetectionBox& ball_det, int ball_id) {
        shot_tracking_frames_++;

        if (!hasShotHoopReference()) return;
        if (ball_det.extrapolated) return;

        const double travel = shotTravelPx(ball_det);
        if (shot_tracking_frames_ < shot_make_min_frames_ ||
            travel < shot_make_min_travel_px_) {
            return;
        }

        if (boxesIntersect(ball_det, expandedHoopZone()) ||
            ballCenterInside(expandedHoopZone(), ball_det)) {
            shot_reached_hoop_zone_ = true;
        }

        if (shot_reached_hoop_zone_ && !shot_counted_) {
            countShotAttemptIfNeeded(ball_id, travel);
            logstream << "basketball_analysis: shot confirmed at hoop: ball #"
                      << ball_id
                      << ", traveled " << (int)travel
                      << "px in " << shot_tracking_frames_ << " frames)";
        }

        if (isGoodShotGeometry(ball_det)) {
            shot_scored_ = true;
            logstream << "basketball_analysis: shot reached relaxed hoop zone: ball #"
                      << ball_id
                      << ", traveled " << (int)travel
                      << "px in " << shot_tracking_frames_ << " frames)";
        }

        last_shot_eval_det_ = ball_det;
        have_last_shot_eval_det_ = true;
    }

    void startEventTracking(const DetectionBox& ball_det) {
        shot_start_x_ = centerX(ball_det);
        shot_start_y_ = centerY(ball_det);
        shot_tracking_frames_ = 0;
        shot_counted_ = false;
        shot_reached_hoop_zone_ = false;
        shot_scored_ = false;
        have_last_shot_eval_det_ = false;
    }

    // Count a miss only if the shot reached the padded hoop zone
    // without ever entering the actual hoop box.
    void countEventOnBallOnlyExit(int ball_id, const BallCandidate* active) {
        finalizeShotOutcome(ball_id, active ? &active->last_det : nullptr);
    }

    // ── Buffer operations ───────────────────────────────────

    void flushGapBufferRaw() {
        for (auto& f : gap_buffer_) {
            MetadataEnvelope env;
            std::vector<DetectionBox> existing;
            parseDetections(f, env, existing);
            auto pfb = withoutLabel(filterByModel(existing, context_model_index_), pfb_ball_label_);
            writeMetadata(f, env, pfb, "pfb");
            this->sink_->put(f);
        }
        gap_buffer_.clear();
    }

    void flushGapBufferInterp(const std::vector<DetectionBox>& resume_players) {
        if (!have_last_player_dets_ || gap_buffer_.empty()) {
            flushGapBufferRaw();
            return;
        }
        auto matches = matchBoxes(last_player_dets_, resume_players,
                                  max_match_distance_px_);
        if (matches.empty()) { flushGapBufferRaw(); return; }

        double max_disp = 0;
        for (auto& [li, ci] : matches)
            max_disp = std::max(max_disp,
                                centerDist(last_player_dets_[li], resume_players[ci]));

        logstream << "basketball_analysis: PFB gap interpolated: " << gap_buffer_.size()
                  << " frames, " << matches.size() << "/" << resume_players.size()
                  << " players matched (max displacement="
                  << (int)max_disp << "px)";

        const int n = (int)gap_buffer_.size();
        for (int i = 0; i < n; i++) {
            double t = (double)(i + 1) / (double)(n + 1);
            MetadataEnvelope env;
            std::vector<DetectionBox> existing;
            parseDetections(gap_buffer_[i], env, existing);
            // Keep only context-model non-player detections, add interpolated players
            std::vector<DetectionBox> result;
            for (auto& d : existing)
                if (d.model_index == context_model_index_ &&
                    (!d.has_label || (d.label != player_label_ && d.label != pfb_ball_label_)))
                    result.push_back(d);
            for (auto& [li, ci] : matches)
                result.push_back(lerpBox(last_player_dets_[li],
                                         resume_players[ci], t));
            writeMetadata(gap_buffer_[i], env, result, "pfb_interpolated");
            this->sink_->put(gap_buffer_[i]);
        }
        gap_buffer_.clear();
    }

    void flushGapBufferBallOnly(const BallCandidate& active) {
        for (auto& f : gap_buffer_) {
            MetadataEnvelope env;
            std::vector<DetectionBox> existing;
            parseDetections(f, env, existing);
            writeMetadata(f, env, {active.last_det}, "ball_only");
            this->sink_->put(f);
        }
        gap_buffer_.clear();
    }

    // Emit one frame from front with held player positions, keep rest.
    void emitOneHeld() {
        if (gap_buffer_.empty()) return;
        MetadataEnvelope env;
        std::vector<DetectionBox> existing;
        parseDetections(gap_buffer_.front(), env, existing);
        // Keep only context-model non-player detections, add held players
        std::vector<DetectionBox> result;
        for (auto& d : existing)
            if (d.model_index == context_model_index_ &&
                (!d.has_label || (d.label != player_label_ && d.label != pfb_ball_label_)))
                result.push_back(d);
        if (have_last_player_dets_)
            for (auto& d : last_player_dets_) result.push_back(d);
        writeMetadata(gap_buffer_.front(), env, result, "pfb_held");
        this->sink_->put(gap_buffer_.front());
        gap_buffer_.erase(gap_buffer_.begin());
        hold_frames_emitted_++;
    }

    void flushGapBufferPFB() {
        for (auto& f : gap_buffer_) {
            MetadataEnvelope env;
            std::vector<DetectionBox> existing;
            parseDetections(f, env, existing);
            auto pfb = withoutLabel(filterByModel(existing, context_model_index_), pfb_ball_label_);
            writeMetadata(f, env, pfb, "pfb");
            this->sink_->put(f);
        }
        gap_buffer_.clear();
    }

    void flushGapBufferEmpty() {
        for (auto& f : gap_buffer_) {
            MetadataEnvelope env;
            std::vector<DetectionBox> existing;
            parseDetections(f, env, existing);
            writeMetadata(f, env, {}, "empty");
            this->sink_->put(f);
        }
        gap_buffer_.clear();
    }

    // ── Process: PFB mode ───────────────────────────────────

    void processPFB(av::VideoFrame* pfrm,
                    const MetadataEnvelope& env,
                    const std::vector<DetectionBox>& pfb_dets,
                    const std::vector<DetectionBox>& player_dets,
                    const std::vector<DetectionBox>& pfb_ball_dets,
                    bool has_pfb) {
        if (has_pfb) {
            if (!player_dets.empty()) {
                last_player_dets_     = player_dets;
                have_last_player_dets_ = true;
                last_player_frame_    = frame_counter_;
            }
            hold_frames_emitted_  = 0;
            av::VideoFrame frm = *pfrm;
            this->source_->pop();
            writeMetadata(frm, env,
                          buildPfbOutputDetections(pfb_dets, player_dets, pfb_ball_dets),
                          active_shot_id_ >= 0 ? "pfb+shot" : "pfb");
            this->sink_->put(frm);
        } else {
            mode_ = MergeMode::PFB_GAP;
            av::VideoFrame frm = *pfrm;
            this->source_->pop();
            gap_buffer_.push_back(std::move(frm));
        }
    }

    // ── Process: PFB_GAP mode ───────────────────────────────

    void processPFBGap(av::VideoFrame* pfrm,
                       const MetadataEnvelope& env,
                       const std::vector<DetectionBox>& pfb_dets,
                       const std::vector<DetectionBox>& player_dets,
                       const std::vector<DetectionBox>& pfb_ball_dets,
                       bool has_pfb) {
        if (has_pfb) {
            // Players returned — interpolate gap
            if (!player_dets.empty()) {
                flushGapBufferInterp(player_dets);
                last_player_dets_      = player_dets;
                have_last_player_dets_ = true;
                last_player_frame_     = frame_counter_;
            } else {
                flushGapBufferPFB();
            }
            hold_frames_emitted_   = 0;
            av::VideoFrame frm = *pfrm;
            this->source_->pop();
            writeMetadata(frm, env,
                          buildPfbOutputDetections(pfb_dets, player_dets, pfb_ball_dets),
                          active_shot_id_ >= 0 ? "pfb+shot" : "pfb");
            this->sink_->put(frm);
            mode_ = MergeMode::PFB;
            return;
        }

        av::VideoFrame frm = *pfrm;
        this->source_->pop();
        gap_buffer_.push_back(std::move(frm));

        if ((int)gap_buffer_.size() >= confirm_frames_) {
            BallCandidate* armed = findArmedValidCandidate();
            if (armed) {
                // ── Switch to BALL_ONLY ──
                active_ball_id_                = armed->id;
                ball_only_entered_frame_       = frame_counter_;
                ball_only_tracked_frames_      = 0;
                ball_only_extrapolated_frames_ = 0;
                hold_frames_emitted_           = 0;
                active_shot_id_                = -1; // BALL_ONLY takes over
                // The release is tracked, but it still counts only after hoop reach.

                logstream << "basketball_analysis: PFB -> BALL_ONLY (tracked release): "
                          << "ball candidate #" << armed->id
                          << ", was near player for " << armed->consecutive_near_frames
                          << " frames (edge dist=" << (int)armed->last_near_player_dist << "px)"
                          << ", players lost for " << (frame_counter_ - last_player_frame_)
                          << " frames"
                          << ", ball speed=" << (int)armed->speed << "px/f"
                          << ", trajectory_score=" << armed->trajectory_score;

                flushGapBufferBallOnly(*armed);
                mode_ = MergeMode::BALL_ONLY;
            } else if (hold_frames_emitted_ < max_hold_frames_) {
                emitOneHeld();
            } else {
                if (hold_frames_emitted_ == max_hold_frames_) {
                    logstream << "basketball_analysis: PFB hold timeout: "
                              << "no player detection for "
                              << (frame_counter_ - last_player_frame_) << " frames"
                              << ", last real detection at frame " << last_player_frame_
                              << ", held " << max_hold_frames_
                              << " frames with stale positions";
                }
                flushGapBufferEmpty();
                hold_frames_emitted_++;
            }
        }
    }

    // ── Process: BALL_ONLY mode ─────────────────────────────

    void processBALLONLY(av::VideoFrame* pfrm,
                         const MetadataEnvelope& env,
                         const std::vector<DetectionBox>& pfb_dets,
                         const std::vector<DetectionBox>& player_dets,
                         bool has_pfb) {
        BallCandidate* active = findCandidate(active_ball_id_);

        if (has_pfb) {
            // Start / continue return-to-PFB confirmation
            av::VideoFrame frm = *pfrm;
            this->source_->pop();
            gap_buffer_.push_back(std::move(frm));
            consecutive_player_frames_++;

            if (consecutive_player_frames_ >= confirm_frames_) {
                int total = ball_only_tracked_frames_ +
                            ball_only_extrapolated_frames_;
                logstream << "basketball_analysis: BALL_ONLY -> PFB: "
                          << "player detected for "
                          << consecutive_player_frames_ << " consecutive frames"
                          << ", was in BALL_ONLY for " << total << " frames ("
                          << ball_only_tracked_frames_ << " tracked, "
                          << ball_only_extrapolated_frames_ << " extrapolated)";

                countEventOnBallOnlyExit(active_ball_id_, active);
                flushGapBufferPFB();
                if (!player_dets.empty()) {
                    last_player_dets_      = player_dets;
                    have_last_player_dets_ = true;
                    last_player_frame_     = frame_counter_;
                }
                mode_                  = MergeMode::PFB;
                hold_frames_emitted_   = 0;
                consecutive_player_frames_ = 0;
                active_ball_id_        = -1;
                last_ball_only_exit_frame_ = frame_counter_;
                for (auto& c : candidates_) c.armed = false;
            }
            return;
        }

        // No players — cancel any return-to-PFB confirmation
        if (!gap_buffer_.empty()) {
            if (active) flushGapBufferBallOnly(*active);
            else        flushGapBufferRaw();
            consecutive_player_frames_ = 0;
        }

        av::VideoFrame frm = *pfrm;
        this->source_->pop();

        bool matched = active && (active->last_match_frame == frame_counter_);

        if (matched) {
            trackBallEvent(active->last_det, active->id);
            writeMetadata(frm, env, {active->last_det}, "ball_only");
            this->sink_->put(frm);
            ball_only_tracked_frames_++;
            hold_frames_emitted_ = 0;
        } else if (active && active->traj.valid &&
                   hold_frames_emitted_ < max_hold_frames_) {
            int ahead = (int)(frame_counter_ - active->last_match_frame);
            DetectionBox extrap = extrapolateBall(*active, ahead);
            trackBallEvent(extrap, active->id);
            writeMetadata(frm, env, {extrap}, "ball_only_extrapolated");
            this->sink_->put(frm);
            ball_only_extrapolated_frames_++;
            hold_frames_emitted_++;
        } else if (active && hold_frames_emitted_ < max_hold_frames_) {
            // No trajectory fit — hold last position
            DetectionBox held = active->last_det;
            held.extrapolated = true;
            trackBallEvent(held, active->id);
            writeMetadata(frm, env, {held}, "ball_only_held");
            this->sink_->put(frm);
            hold_frames_emitted_++;
        } else {
            writeMetadata(frm, env, {}, "empty");
            this->sink_->put(frm);
            hold_frames_emitted_++;
            if (hold_frames_emitted_ >= max_hold_frames_ || !active) {
                uint64_t last_real = active ? active->last_match_frame : ball_only_entered_frame_;
                double residual = active ? active->traj.residual : 0.0;
                logstream << "basketball_analysis: BALL_ONLY extrapolation timeout: "
                          << "no ball detection for "
                          << (frame_counter_ - last_real) << " frames"
                          << ", last real detection at frame " << last_real
                          << ", trajectory residual was " << residual << "px";
                countEventOnBallOnlyExit(active_ball_id_, active);
                mode_                      = MergeMode::PFB;
                active_ball_id_            = -1;
                hold_frames_emitted_       = 0;
                last_ball_only_exit_frame_ = frame_counter_;
                for (auto& c : candidates_) c.armed = false;
            }
        }
    }

public:
    using NodeSISO::NodeSISO;

    void resetInput() override { resetState(); }

    void process() override {
        av::VideoFrame* pfrm = this->source_->peek();
        if (!pfrm) return;
        if (!*pfrm) { this->source_->pop(); return; }
        if (isEofMarker(*pfrm)) {
            av::VideoFrame eof = *pfrm;
            this->source_->pop();
            flushGapBufferRaw();
            logstream << "basketball_analysis: GAME STATS: "
                      << total_shots_detected_ << " shots in "
                      << frame_counter_ << " frames";
            resetState();
            this->sink_->put(eof);
            return;
        }

        frame_counter_++;

        MetadataEnvelope env;
        std::vector<DetectionBox> all_dets;
        parseDetections(*pfrm, env, all_dets);

        for (const auto& d : all_dets) {
            if (d.has_label && d.label == hoop_label_ && d.conf >= hoop_min_conf_) {
                last_hoop_det_ = d;
                have_hoop_ = true;
                last_hoop_frame_ = frame_counter_;
                break;
            }
        }

        auto ball_dets    = filterByLabel(filterByModel(all_dets, ball_model_index_),
                                           ball_label_);
        auto pfb_dets     = filterByModel(all_dets, context_model_index_);
        auto player_dets  = filterByLabel(pfb_dets, player_label_);
        auto pfb_ball_dets = filterByLabel(pfb_dets, pfb_ball_label_);
        bool has_pfb = !pfb_dets.empty();

        updatePfbPossession(pfb_dets, pfb_ball_dets, player_dets);
        updateCandidates(ball_dets, player_dets);

        // Track shots in PFB mode (armed ball leaving player)
        if (mode_ != MergeMode::BALL_ONLY)
            updateActiveShot(player_dets);

        switch (mode_) {
        case MergeMode::PFB:
            processPFB(pfrm, env, pfb_dets, player_dets, pfb_ball_dets, has_pfb);
            break;
        case MergeMode::PFB_GAP:
            processPFBGap(pfrm, env, pfb_dets, player_dets, pfb_ball_dets, has_pfb);
            break;
        case MergeMode::BALL_ONLY:
            processBALLONLY(pfrm, env, pfb_dets, player_dets, has_pfb);
            break;
        }
    }

    static std::shared_ptr<BasketballAnalysis> create(NodeCreationInfo& nci) {
        EdgeManager& edges = nci.edges;
        const Parameters& params = nci.params;
        auto r = NodeSISO<av::VideoFrame, av::VideoFrame>::
                     template createCommon<BasketballAnalysis>(edges, params);
        if (params.count("metadata_key_in"))
            r->metadata_key_in_ = params["metadata_key_in"].get<std::string>();
        if (params.count("metadata_key_out"))
            r->metadata_key_out_ = params["metadata_key_out"].get<std::string>();
        if (params.count("ball_model_index"))
            r->ball_model_index_ = params["ball_model_index"];
        if (params.count("context_model_index"))
            r->context_model_index_ = params["context_model_index"];
        if (params.count("ball_label"))
            r->ball_label_ = params["ball_label"].get<std::string>();
        if (params.count("player_label"))
            r->player_label_ = params["player_label"].get<std::string>();
        if (params.count("pfb_ball_label"))
            r->pfb_ball_label_ = params["pfb_ball_label"].get<std::string>();
        if (params.count("min_speed_px_per_frame"))
            r->min_speed_px_per_frame_ = params["min_speed_px_per_frame"];
        if (params.count("arm_dist_px"))
            r->arm_dist_px_ = params["arm_dist_px"];
        if (params.count("arm_frames"))
            r->arm_frames_ = params["arm_frames"];
        if (params.count("confirm_frames"))
            r->confirm_frames_ = params["confirm_frames"];
        if (params.count("max_hold_frames"))
            r->max_hold_frames_ = params["max_hold_frames"];
        if (params.count("max_match_distance_px"))
            r->max_match_distance_px_ = params["max_match_distance_px"];
        if (params.count("history_frames"))
            r->history_frames_ = params["history_frames"];
        if (params.count("candidate_drop_frames"))
            r->candidate_drop_frames_ = params["candidate_drop_frames"];
        if (params.count("pfb_possession_grace_frames"))
            r->pfb_possession_grace_frames_ = params["pfb_possession_grace_frames"];
        if (params.count("release_confirm_frames"))
            r->release_confirm_frames_ = params["release_confirm_frames"];
        if (params.count("release_possession_memory_frames"))
            r->release_possession_memory_frames_ = params["release_possession_memory_frames"];
        if (params.count("shot_make_min_frames"))
            r->shot_make_min_frames_ = params["shot_make_min_frames"];
        if (params.count("shot_make_min_travel_px"))
            r->shot_make_min_travel_px_ = params["shot_make_min_travel_px"];
        if (params.count("shot_hoop_memory_frames"))
            r->shot_hoop_memory_frames_ = params["shot_hoop_memory_frames"];
        if (params.count("hoop_label"))
            r->hoop_label_ = params["hoop_label"].get<std::string>();
        if (params.count("hoop_min_conf"))
            r->hoop_min_conf_ = params["hoop_min_conf"];
        if (params.count("passthrough_model_indices")) {
            for (const auto& idx : params["passthrough_model_indices"])
                r->passthrough_model_indices_.push_back(idx.get<int>());
        }
        if (params.count("passthrough_labels")) {
            for (const auto& lbl : params["passthrough_labels"])
                r->passthrough_labels_.push_back(lbl.get<std::string>());
        }
        return r;
    }
};

DECLNODE(basketball_analysis, BasketballAnalysis)
