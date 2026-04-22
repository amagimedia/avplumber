#include "../../node_common.hpp"

extern "C" {
#include <libavutil/dict.h>
}

#include <algorithm>
#include <cctype>
#include <cmath>
#include <limits>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

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

struct TrackColor {
    float y_ema = 0.0f;
    float u_ema = 0.0f;
    float v_ema = 0.0f;
    float last_confidence = 0.0f;
    int assigned_team = -1;
    int initial_candidate_team = -1;
    int initial_candidate_frames = 0;
    uint32_t hits = 0;
    uint64_t last_frame = 0;
};

struct ParsedTracked {
    int det_index = -1;
    int track_id = -1;
    float x1 = 0, y1 = 0, x2 = 0, y2 = 0;
};

struct ParsedSeg {
    int det_index = -1;
    float x1 = 0, y1 = 0, x2 = 0, y2 = 0;
    float y = 0;
    float u = 0, v = 0;
    int pixels = 0;
    float confidence = 0.0f;
};

struct CurrentEvidence {
    bool has_sample = false;
    float y = 0.0f;
    float u = 0.0f;
    float v = 0.0f;
    float confidence = 0.0f;
    int pixels = 0;
};

struct RecentAppearance {
    float x1 = 0.0f;
    float y1 = 0.0f;
    float x2 = 0.0f;
    float y2 = 0.0f;
    float y = 0.0f;
    float u = 0.0f;
    float v = 0.0f;
    float confidence = 0.0f;
    int team = -1;
    uint64_t frame = 0;
};

struct HandoffMatch {
    bool found = false;
    int team = -1;
    float center_rel = 0.0f;
    float color_distance = 0.0f;
    uint64_t age = 0;
};

} // namespace

class TeamClassifier : public NodeSISO<av::VideoFrame, av::VideoFrame> {
    std::string player_metadata_key_ = "yolo_players";
    std::string seg_metadata_key_ = "yolo_players_seg";
    std::string shot_metadata_key_;
    std::vector<std::string> player_labels_ = {"Player"};
    std::vector<std::string> seg_labels_ = {"player"};
    std::string debug_seg_metadata_key_;
    float iou_match_threshold_ = 0.3f;
    float fallback_center_distance_px_ = 30.0f;
    float ema_alpha_track_ = 0.2f;
    float ema_alpha_centroid_ = 0.05f;
    float luma_weight_ = 0.35f;
    float min_jersey_confidence_ = 0.15f;
    float soft_assignment_margin_ = 2.0f;
    float initial_assignment_margin_ = 4.0f;
    float assignment_margin_ = 8.0f;
    float bootstrap_axis_min_separation_ = 6.0f;
    int initial_assignment_min_hits_ = 3;
    int initial_assignment_confirm_frames_ = 3;
    uint64_t bootstrap_frames_ = 60;
    int bootstrap_min_tracks_ = 6;
    uint64_t track_idle_frames_ = 300;
    uint64_t handoff_max_age_frames_ = 3;
    float handoff_max_center_distance_rel_ = 0.85f;
    float handoff_min_size_ratio_ = 0.45f;
    float handoff_max_color_distance_ = 18.0f;
    int min_jersey_pixels_ = 32;
    std::string output_field_ = "team";
    std::string output_team_color_field_;
    bool write_back_to_seg_ = true;
    bool rewrite_seg_cls_ = true;
    bool rewrite_label_ = false;
    std::vector<std::string> team_label_names_ = {"PlayerA", "PlayerB"};
    std::string unknown_label_ = "PlayerUnknown";
    int debug_log_every_n_ = 0;

    std::unordered_map<int, TrackColor> tracks_;
    float centroids_[2][3] = {{0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f}};
    float identity_axis_[3] = {0.0f, 0.0f, 0.0f};
    float identity_midpoint_[3] = {0.0f, 0.0f, 0.0f};
    float identity_axis_separation_ = 0.0f;
    bool bootstrapped_ = false;
    uint64_t frame_counter_ = 0;
    std::vector<RecentAppearance> recent_appearances_;

    void resetState() {
        tracks_.clear();
        centroids_[0][0] = centroids_[0][1] = centroids_[0][2] = 0.0f;
        centroids_[1][0] = centroids_[1][1] = centroids_[1][2] = 0.0f;
        identity_axis_[0] = identity_axis_[1] = identity_axis_[2] = 0.0f;
        identity_midpoint_[0] = identity_midpoint_[1] = identity_midpoint_[2] = 0.0f;
        identity_axis_separation_ = 0.0f;
        bootstrapped_ = false;
        frame_counter_ = 0;
        recent_appearances_.clear();
    }

    void toWeightedColor(float y, float u, float v, float (&out)[3]) const {
        out[0] = y * luma_weight_;
        out[1] = u;
        out[2] = v;
    }

    bool matchesLabel(const Parameters& det, const std::vector<std::string>& labels) const {
        if (labels.empty()) return true;
        if (!det.contains("label") || !det["label"].is_string()) return false;
        const std::string lbl = det["label"].get<std::string>();
        for (const auto& want : labels) {
            if (iequals(lbl, want)) return true;
        }
        return false;
    }

    static bool parseBBox(const Parameters& det, float& x1, float& y1, float& x2, float& y2) {
        if (!det.contains("xyxy") || !det["xyxy"].is_array() || det["xyxy"].size() < 4) return false;
        x1 = det["xyxy"][0].get<float>();
        y1 = det["xyxy"][1].get<float>();
        x2 = det["xyxy"][2].get<float>();
        y2 = det["xyxy"][3].get<float>();
        return x2 > x1 && y2 > y1;
    }

    static bool parseJerseyColor(const Parameters& det, float& y, float& u, float& v, float& confidence) {
        if (!det.contains("jersey_y")) return false;
        if (!det.contains("jersey_uv") || !det["jersey_uv"].is_array() || det["jersey_uv"].size() < 2) return false;
        y = det["jersey_y"].get<float>();
        u = det["jersey_uv"][0].get<float>();
        v = det["jersey_uv"][1].get<float>();
        confidence = det.value("jersey_confidence", 0.0f);
        return true;
    }

    float teamDistance(float y, float u, float v, int team) const {
        const float dy = (y - centroids_[team][0]) * luma_weight_;
        const float du = u - centroids_[team][1];
        const float dv = v - centroids_[team][2];
        return std::sqrt(dy * dy + du * du + dv * dv);
    }

    float axisProjection(float y, float u, float v) const {
        float weighted[3];
        toWeightedColor(y, u, v, weighted);
        return (weighted[0] - identity_midpoint_[0]) * identity_axis_[0] +
               (weighted[1] - identity_midpoint_[1]) * identity_axis_[1] +
               (weighted[2] - identity_midpoint_[2]) * identity_axis_[2];
    }

    float weightedColorDistance(float y0, float u0, float v0, float y1, float u1, float v1) const {
        const float dy = (y0 - y1) * luma_weight_;
        const float du = u0 - u1;
        const float dv = v0 - v1;
        return std::sqrt(dy * dy + du * du + dv * dv);
    }

    static float bboxWidth(float x1, float x2) {
        return std::max(0.0f, x2 - x1);
    }

    static float bboxHeight(float y1, float y2) {
        return std::max(0.0f, y2 - y1);
    }

    static float bboxDiagonal(float x1, float y1, float x2, float y2) {
        const float w = bboxWidth(x1, x2);
        const float h = bboxHeight(y1, y2);
        return std::sqrt(w * w + h * h);
    }

    void pruneRecentAppearances() {
        for (auto it = recent_appearances_.begin(); it != recent_appearances_.end();) {
            if (frame_counter_ - it->frame > handoff_max_age_frames_) {
                it = recent_appearances_.erase(it);
            } else {
                ++it;
            }
        }
    }

    HandoffMatch findRecentHandoff(const ParsedTracked& tr,
                                   const CurrentEvidence& sample,
                                   int expected_team) const {
        HandoffMatch best;
        if (!sample.has_sample || expected_team < 0) return best;

        const float tr_diag = std::max(1.0f, bboxDiagonal(tr.x1, tr.y1, tr.x2, tr.y2));
        const float tr_area = bboxWidth(tr.x1, tr.x2) * bboxHeight(tr.y1, tr.y2);
        float best_score = std::numeric_limits<float>::max();
        for (auto it = recent_appearances_.rbegin(); it != recent_appearances_.rend(); ++it) {
            if (it->team != expected_team) continue;
            const uint64_t age = frame_counter_ - it->frame;
            if (age > handoff_max_age_frames_) continue;

            const float recent_diag = std::max(1.0f, bboxDiagonal(it->x1, it->y1, it->x2, it->y2));
            const float recent_area = bboxWidth(it->x1, it->x2) * bboxHeight(it->y1, it->y2);
            const float area_min = std::max(1.0f, std::min(tr_area, recent_area));
            const float area_max = std::max(1.0f, std::max(tr_area, recent_area));
            const float size_ratio = area_min / area_max;
            if (size_ratio < handoff_min_size_ratio_) continue;

            const float center_dist = centerDistance(tr.x1, tr.y1, tr.x2, tr.y2, it->x1, it->y1, it->x2, it->y2);
            const float center_rel = center_dist / std::max(tr_diag, recent_diag);
            if (center_rel > handoff_max_center_distance_rel_) continue;

            const float color_distance = weightedColorDistance(sample.y, sample.u, sample.v, it->y, it->u, it->v);
            if (color_distance > handoff_max_color_distance_) continue;

            const float score = center_rel * 16.0f + color_distance + (float)age * 1.5f;
            if (score < best_score) {
                best_score = score;
                best.found = true;
                best.team = it->team;
                best.center_rel = center_rel;
                best.color_distance = color_distance;
                best.age = age;
            }
        }
        return best;
    }

    void bootstrapCentroids() {
        struct Sample {
            float y = 0.0f;
            float u = 0.0f;
            float v = 0.0f;
        };
        std::vector<Sample> samples;
        samples.reserve(tracks_.size());
        for (const auto& kv : tracks_) {
            if (kv.second.hits == 0 || kv.second.last_confidence < min_jersey_confidence_) continue;
            samples.push_back({kv.second.y_ema, kv.second.u_ema, kv.second.v_ema});
        }
        if ((int)samples.size() < bootstrap_min_tracks_) return;

        centroids_[0][0] = samples.front().y;
        centroids_[0][1] = samples.front().u;
        centroids_[0][2] = samples.front().v;

        float best_d = -1.0f;
        size_t best_i = 0;
        for (size_t i = 0; i < samples.size(); ++i) {
            const float d = teamDistance(samples[i].y, samples[i].u, samples[i].v, 0);
            if (d > best_d) {
                best_d = d;
                best_i = i;
            }
        }
        centroids_[1][0] = samples[best_i].y;
        centroids_[1][1] = samples[best_i].u;
        centroids_[1][2] = samples[best_i].v;

        for (int iter = 0; iter < 10; ++iter) {
            float sum_y[2] = {0.0f, 0.0f};
            float sum_u[2] = {0.0f, 0.0f};
            float sum_v[2] = {0.0f, 0.0f};
            int count[2] = {0, 0};
            for (const auto& s : samples) {
                const float d0 = teamDistance(s.y, s.u, s.v, 0);
                const float d1 = teamDistance(s.y, s.u, s.v, 1);
                const int k = (d0 <= d1) ? 0 : 1;
                sum_y[k] += s.y;
                sum_u[k] += s.u;
                sum_v[k] += s.v;
                count[k] += 1;
            }
            for (int k = 0; k < 2; ++k) {
                if (count[k] > 0) {
                    centroids_[k][0] = sum_y[k] / (float)count[k];
                    centroids_[k][1] = sum_u[k] / (float)count[k];
                    centroids_[k][2] = sum_v[k] / (float)count[k];
                }
            }
        }

        float weighted0[3];
        float weighted1[3];
        toWeightedColor(centroids_[0][0], centroids_[0][1], centroids_[0][2], weighted0);
        toWeightedColor(centroids_[1][0], centroids_[1][1], centroids_[1][2], weighted1);

        const float dx = weighted1[0] - weighted0[0];
        const float dy = weighted1[1] - weighted0[1];
        const float dz = weighted1[2] - weighted0[2];
        const float sep = std::sqrt(dx * dx + dy * dy + dz * dz);
        if (sep < bootstrap_axis_min_separation_) return;

        identity_axis_[0] = dx / sep;
        identity_axis_[1] = dy / sep;
        identity_axis_[2] = dz / sep;
        identity_midpoint_[0] = 0.5f * (weighted0[0] + weighted1[0]);
        identity_midpoint_[1] = 0.5f * (weighted0[1] + weighted1[1]);
        identity_midpoint_[2] = 0.5f * (weighted0[2] + weighted1[2]);
        identity_axis_separation_ = sep;
        bootstrapped_ = true;

        logstream << "team_classifier: bootstrap"
                  << " frame=" << frame_counter_
                  << " tracks=" << samples.size()
                  << " axis_sep=" << sep
                  << " centroid0_yuv=[" << centroids_[0][0] << "," << centroids_[0][1] << "," << centroids_[0][2] << "]"
                  << " centroid1_yuv=[" << centroids_[1][0] << "," << centroids_[1][1] << "," << centroids_[1][2] << "]";
    }

    void pruneTracks() {
        for (auto it = tracks_.begin(); it != tracks_.end();) {
            if (frame_counter_ - it->second.last_frame > track_idle_frames_) {
                it = tracks_.erase(it);
            } else {
                ++it;
            }
        }
    }

public:
    using NodeSISO<av::VideoFrame, av::VideoFrame>::NodeSISO;

    void process() override {
        av::VideoFrame frm = this->source_->get();
        if (!frm) return;
        if (isEofMarker(frm)) {
            resetState();
            this->sink_->put(frm);
            return;
        }

        ++frame_counter_;
        const AVFrame* raw = frm.raw();
        if (!shot_metadata_key_.empty() && raw && raw->metadata) {
            AVDictionaryEntry* shot_entry = av_dict_get(raw->metadata, shot_metadata_key_.c_str(), nullptr, 0);
            if (shot_entry && shot_entry->value) {
                try {
                    Parameters shot_md = Parameters::parse(shot_entry->value);
                    if (shot_md.value("shot_transition", false)) {
                        const std::string shot_type = shot_md.value("shot_type", std::string("?"));
                        resetState();
                        ++frame_counter_;
                        if (debug_log_every_n_ > 0) {
                            logstream << "team_classifier: reset"
                                      << " frame=" << frame_counter_
                                      << " shot_type=" << shot_type;
                        }
                    }
                } catch (...) {}
            }
        }
        if (!raw || !raw->metadata) {
            this->sink_->put(frm);
            return;
        }

        AVDictionaryEntry* player_entry = av_dict_get(raw->metadata, player_metadata_key_.c_str(), nullptr, 0);
        AVDictionaryEntry* seg_entry = av_dict_get(raw->metadata, seg_metadata_key_.c_str(), nullptr, 0);
        AVDictionaryEntry* debug_seg_entry = debug_seg_metadata_key_.empty()
                                                 ? nullptr
                                                 : av_dict_get(raw->metadata, debug_seg_metadata_key_.c_str(), nullptr, 0);
        if (!player_entry || !player_entry->value || !seg_entry || !seg_entry->value) {
            this->sink_->put(frm);
            return;
        }

        Parameters player_md;
        Parameters seg_md;
        Parameters debug_seg_md;
        bool has_debug_seg_md = false;
        try {
            player_md = Parameters::parse(player_entry->value);
            seg_md = Parameters::parse(seg_entry->value);
            if (debug_seg_entry && debug_seg_entry->value) {
                debug_seg_md = Parameters::parse(debug_seg_entry->value);
                has_debug_seg_md = debug_seg_md.contains("detections") && debug_seg_md["detections"].is_array();
            }
        } catch (...) {
            this->sink_->put(frm);
            return;
        }
        if (!player_md.contains("detections") || !player_md["detections"].is_array() ||
            !seg_md.contains("detections") || !seg_md["detections"].is_array()) {
            this->sink_->put(frm);
            return;
        }

        std::vector<ParsedTracked> tracked;
        std::vector<ParsedSeg> segs;

        for (int i = 0; i < (int)player_md["detections"].size(); ++i) {
            auto& det = player_md["detections"][i];
            if (!det.is_object() || !matchesLabel(det, player_labels_)) continue;
            ParsedTracked p;
            if (!parseBBox(det, p.x1, p.y1, p.x2, p.y2)) continue;
            p.det_index = i;
            p.track_id = det.value("track_id", -1);
            if (p.track_id >= 0) tracked.push_back(p);
        }

        for (int i = 0; i < (int)seg_md["detections"].size(); ++i) {
            auto& det = seg_md["detections"][i];
            if (!det.is_object() || !matchesLabel(det, seg_labels_)) continue;
            ParsedSeg s;
            if (!parseBBox(det, s.x1, s.y1, s.x2, s.y2)) continue;
            if (!parseJerseyColor(det, s.y, s.u, s.v, s.confidence)) continue;
            s.pixels = det.value("jersey_cloth_pixels", det.value("jersey_pixels", 0));
            s.det_index = i;
            segs.push_back(s);
        }

        std::unordered_map<int, int> debug_det_by_source_index;
        if (has_debug_seg_md) {
            for (int i = 0; i < (int)debug_seg_md["detections"].size(); ++i) {
                auto& det = debug_seg_md["detections"][i];
                if (!det.is_object()) continue;
                const int source_det_index = det.value("source_det_index", -1);
                if (source_det_index >= 0) {
                    debug_det_by_source_index[source_det_index] = i;
                }
                det["cls"] = 2;
                det["team"] = -1;
            }
        }

        std::vector<int> track_to_seg(tracked.size(), -1);
        std::vector<bool> seg_used(segs.size(), false);
        std::vector<CurrentEvidence> current_evidence(tracked.size());

        for (size_t ti = 0; ti < tracked.size(); ++ti) {
            float best_iou = 0.0f;
            int best_si = -1;
            for (size_t si = 0; si < segs.size(); ++si) {
                if (seg_used[si]) continue;
                const float iou = bboxIoU(tracked[ti].x1, tracked[ti].y1, tracked[ti].x2, tracked[ti].y2,
                                          segs[si].x1, segs[si].y1, segs[si].x2, segs[si].y2);
                if (iou > best_iou) {
                    best_iou = iou;
                    best_si = (int)si;
                }
            }
            if (best_si >= 0 && best_iou >= iou_match_threshold_) {
                track_to_seg[ti] = best_si;
                seg_used[(size_t)best_si] = true;
            }
        }

        if (fallback_center_distance_px_ > 0.0f) {
            for (size_t ti = 0; ti < tracked.size(); ++ti) {
                if (track_to_seg[ti] >= 0) continue;
                float best_d = std::numeric_limits<float>::max();
                int best_si = -1;
                for (size_t si = 0; si < segs.size(); ++si) {
                    if (seg_used[si]) continue;
                    const float d = centerDistance(tracked[ti].x1, tracked[ti].y1, tracked[ti].x2, tracked[ti].y2,
                                                   segs[si].x1, segs[si].y1, segs[si].x2, segs[si].y2);
                    if (d < best_d) {
                        best_d = d;
                        best_si = (int)si;
                    }
                }
                if (best_si >= 0 && best_d <= fallback_center_distance_px_) {
                    track_to_seg[ti] = best_si;
                    seg_used[(size_t)best_si] = true;
                }
            }
        }

        int matched = 0;
        for (size_t ti = 0; ti < tracked.size(); ++ti) {
            const int si = track_to_seg[ti];
            if (si < 0) continue;
            const ParsedSeg& seg = segs[(size_t)si];
            CurrentEvidence& evidence = current_evidence[ti];
            evidence.has_sample = true;
            evidence.y = seg.y;
            evidence.u = seg.u;
            evidence.v = seg.v;
            evidence.confidence = seg.confidence;
            evidence.pixels = seg.pixels;
            if (seg.pixels < min_jersey_pixels_ || seg.confidence < min_jersey_confidence_) continue;
            TrackColor& tc = tracks_[tracked[ti].track_id];
            if (tc.hits == 0) {
                tc.y_ema = seg.y;
                tc.u_ema = seg.u;
                tc.v_ema = seg.v;
            } else {
                tc.y_ema = (1.0f - ema_alpha_track_) * tc.y_ema + ema_alpha_track_ * seg.y;
                tc.u_ema = (1.0f - ema_alpha_track_) * tc.u_ema + ema_alpha_track_ * seg.u;
                tc.v_ema = (1.0f - ema_alpha_track_) * tc.v_ema + ema_alpha_track_ * seg.v;
            }
            tc.last_confidence = seg.confidence;
            tc.hits += 1;
            tc.last_frame = frame_counter_;
            matched += 1;
        }

        if (!bootstrapped_ && frame_counter_ >= bootstrap_frames_) {
            bootstrapCentroids();
        }

        for (auto& det : seg_md["detections"]) {
            if (!det.is_object() || !matchesLabel(det, seg_labels_)) continue;
            if (rewrite_seg_cls_) det["cls"] = -1;
            if (write_back_to_seg_) det[output_field_] = -1;
        }

        int assigned_known = 0;
        int sticky_mismatch = 0;
        int strong_lock_count = 0;
        int weak_lock_count = 0;
        int assigned_team_count[2] = {0, 0};
        int axis_team_count[2] = {0, 0};
        int axis_eligible_count = 0;
        int current_sample_count = 0;
        int handoff_lock_count = 0;
        int handoff_match_count = 0;
        for (const ParsedTracked& tr : tracked) {
            auto& det = player_md["detections"][tr.det_index];
            int team = -1;
            auto it = tracks_.find(tr.track_id);
            const CurrentEvidence& evidence = current_evidence[(size_t)(&tr - tracked.data())];
            if (bootstrapped_ && it != tracks_.end() && it->second.hits > 0) {
                TrackColor& tc = it->second;
                const bool use_current_sample = evidence.has_sample &&
                                                evidence.pixels >= min_jersey_pixels_ &&
                                                evidence.confidence >= min_jersey_confidence_;
                const float sample_y = use_current_sample ? evidence.y : tc.y_ema;
                const float sample_u = use_current_sample ? evidence.u : tc.u_ema;
                const float sample_v = use_current_sample ? evidence.v : tc.v_ema;
                const float sample_confidence = use_current_sample ? evidence.confidence : tc.last_confidence;
                if (use_current_sample) {
                    current_sample_count += 1;
                }
                const float projection = axisProjection(sample_y, sample_u, sample_v);
                const int candidate_team = (projection <= 0.0f) ? 0 : 1;
                const float margin = std::fabs(projection);
                const bool confident_now = sample_confidence >= min_jersey_confidence_;
                const bool has_enough_hits = tc.hits >= (uint32_t)initial_assignment_min_hits_;
                const bool can_lock_strong = confident_now &&
                                             has_enough_hits &&
                                             margin >= initial_assignment_margin_;
                const bool can_lock_weak = confident_now &&
                                           has_enough_hits &&
                                           margin >= soft_assignment_margin_;
                const bool can_update_centroid = confident_now && margin >= assignment_margin_;
                if (confident_now) {
                    axis_team_count[candidate_team] += 1;
                    axis_eligible_count += 1;
                }

                const HandoffMatch handoff = (tc.assigned_team < 0)
                                                 ? findRecentHandoff(tr, evidence, candidate_team)
                                                 : HandoffMatch();
                if (handoff.found) {
                    handoff_match_count += 1;
                }

                if (tc.assigned_team < 0) {
                    if (handoff.found && confident_now && margin >= soft_assignment_margin_) {
                        tc.assigned_team = handoff.team;
                        tc.initial_candidate_team = -1;
                        tc.initial_candidate_frames = 0;
                        handoff_lock_count += 1;
                        logstream << "team_classifier: lock"
                                  << " mode=handoff"
                                  << " frame=" << frame_counter_
                                  << " track_id=" << tr.track_id
                                  << " team=" << handoff.team
                                  << " conf_pct=" << (100.0f * sample_confidence)
                                  << " axis_margin=" << margin
                                  << " handoff_age=" << handoff.age
                                  << " handoff_center_rel=" << handoff.center_rel
                                  << " handoff_color_dist=" << handoff.color_distance;
                    } else if (can_lock_strong) {
                        tc.assigned_team = candidate_team;
                        tc.initial_candidate_team = -1;
                        tc.initial_candidate_frames = 0;
                        strong_lock_count += 1;
                        logstream << "team_classifier: lock"
                                  << " mode=strong"
                                  << " frame=" << frame_counter_
                                  << " track_id=" << tr.track_id
                                  << " team=" << candidate_team
                                  << " hits=" << tc.hits
                                  << " conf_pct=" << (100.0f * sample_confidence)
                                  << " axis_margin=" << margin;
                    } else if (can_lock_weak) {
                        if (tc.initial_candidate_team == candidate_team) {
                            tc.initial_candidate_frames += 1;
                        } else {
                            tc.initial_candidate_team = candidate_team;
                            tc.initial_candidate_frames = 1;
                        }
                        if (tc.initial_candidate_frames >= initial_assignment_confirm_frames_) {
                            tc.assigned_team = candidate_team;
                            tc.initial_candidate_team = -1;
                            tc.initial_candidate_frames = 0;
                            weak_lock_count += 1;
                            logstream << "team_classifier: lock"
                                      << " mode=weak"
                                      << " frame=" << frame_counter_
                                      << " track_id=" << tr.track_id
                                      << " team=" << candidate_team
                                      << " hits=" << tc.hits
                                      << " conf_pct=" << (100.0f * sample_confidence)
                                      << " axis_margin=" << margin;
                        }
                    } else {
                        tc.initial_candidate_team = -1;
                        tc.initial_candidate_frames = 0;
                    }
                }

                team = tc.assigned_team;
                if (team >= 0 && candidate_team != team) {
                    sticky_mismatch += 1;
                }
                if (team >= 0 && can_update_centroid && candidate_team == team) {
                    centroids_[team][0] = (1.0f - ema_alpha_centroid_) * centroids_[team][0] + ema_alpha_centroid_ * sample_y;
                    centroids_[team][1] = (1.0f - ema_alpha_centroid_) * centroids_[team][1] + ema_alpha_centroid_ * sample_u;
                    centroids_[team][2] = (1.0f - ema_alpha_centroid_) * centroids_[team][2] + ema_alpha_centroid_ * sample_v;
                }
                if (team >= 0 && !output_team_color_field_.empty()) {
                    Parameters team_color = Parameters::array();
                    team_color.push_back(centroids_[team][1]);
                    team_color.push_back(centroids_[team][2]);
                    det[output_team_color_field_] = team_color;
                }
            }
            det[output_field_] = team;
            if (team >= 0) {
                ++assigned_known;
                assigned_team_count[team] += 1;
            }
        }

        for (size_t ti = 0; ti < tracked.size(); ++ti) {
            auto& player_det = player_md["detections"][tracked[ti].det_index];
            const int team = player_det.value(output_field_, -1);
            const int si = track_to_seg[ti];
            if (si < 0) continue;
            auto& seg_det = seg_md["detections"][segs[(size_t)si].det_index];
            if (seg_det.contains("jersey_pixels")) player_det["jersey_pixels"] = seg_det["jersey_pixels"];
            if (seg_det.contains("jersey_mode_pixels")) player_det["jersey_mode_pixels"] = seg_det["jersey_mode_pixels"];
            if (seg_det.contains("jersey_cloth_pixels")) player_det["jersey_cloth_pixels"] = seg_det["jersey_cloth_pixels"];
            if (seg_det.contains("jersey_skin_pixels")) player_det["jersey_skin_pixels"] = seg_det["jersey_skin_pixels"];
            if (seg_det.contains("jersey_confidence")) player_det["jersey_confidence"] = seg_det["jersey_confidence"];
            if (seg_det.contains("jersey_mode_ratio")) player_det["jersey_mode_ratio"] = seg_det["jersey_mode_ratio"];
            if (seg_det.contains("jersey_y")) player_det["jersey_y"] = seg_det["jersey_y"];
            if (seg_det.contains("jersey_uv")) player_det["jersey_uv"] = seg_det["jersey_uv"];
            if (write_back_to_seg_) seg_det[output_field_] = team;
            if (rewrite_seg_cls_) seg_det["cls"] = team;
            if (has_debug_seg_md) {
                auto dbg_it = debug_det_by_source_index.find(segs[(size_t)si].det_index);
                if (dbg_it != debug_det_by_source_index.end()) {
                    auto& debug_det = debug_seg_md["detections"][dbg_it->second];
                    debug_det["cls"] = team >= 0 ? team : 2;
                    debug_det["team"] = team;
                }
            }
            if (!output_team_color_field_.empty() && player_det.contains(output_team_color_field_)) {
                seg_det[output_team_color_field_] = player_det[output_team_color_field_];
            }
            if (rewrite_label_) {
                if (team >= 0 && team < (int)team_label_names_.size()) {
                    player_det["label"] = team_label_names_[(size_t)team];
                } else {
                    player_det["label"] = unknown_label_;
                }
            }
        }

        pruneTracks();
        pruneRecentAppearances();
        for (size_t ti = 0; ti < tracked.size(); ++ti) {
            const int team = player_md["detections"][tracked[ti].det_index].value(output_field_, -1);
            const CurrentEvidence& evidence = current_evidence[ti];
            if (team < 0 || !evidence.has_sample) continue;
            if (evidence.pixels < min_jersey_pixels_ || evidence.confidence < min_jersey_confidence_) continue;
            RecentAppearance recent;
            recent.x1 = tracked[ti].x1;
            recent.y1 = tracked[ti].y1;
            recent.x2 = tracked[ti].x2;
            recent.y2 = tracked[ti].y2;
            recent.y = evidence.y;
            recent.u = evidence.u;
            recent.v = evidence.v;
            recent.confidence = evidence.confidence;
            recent.team = team;
            recent.frame = frame_counter_;
            recent_appearances_.push_back(recent);
        }

        av_dict_set(&frm.raw()->metadata, player_metadata_key_.c_str(), player_md.dump().c_str(), 0);
        av_dict_set(&frm.raw()->metadata, seg_metadata_key_.c_str(), seg_md.dump().c_str(), 0);
        if (has_debug_seg_md) {
            av_dict_set(&frm.raw()->metadata, debug_seg_metadata_key_.c_str(), debug_seg_md.dump().c_str(), 0);
        }

        if (debug_log_every_n_ > 0 && (frame_counter_ % (uint64_t)debug_log_every_n_) == 0) {
            logstream << "team_classifier: frame=" << frame_counter_
                      << " tracked=" << tracked.size()
                      << " seg=" << segs.size()
                      << " matched=" << matched
                      << " assigned_known=" << assigned_known
                      << " assigned_t0=" << assigned_team_count[0]
                      << " assigned_t1=" << assigned_team_count[1]
                      << " sticky_mismatch=" << sticky_mismatch
                      << " axis_eligible=" << axis_eligible_count
                      << " axis_t0=" << axis_team_count[0]
                      << " axis_t1=" << axis_team_count[1]
                      << " current_samples=" << current_sample_count
                      << " handoff_matches=" << handoff_match_count
                      << " handoff_locks=" << handoff_lock_count
                      << " strong_locks=" << strong_lock_count
                      << " weak_locks=" << weak_lock_count
                      << " bootstrapped=" << (bootstrapped_ ? 1 : 0)
                      << " axis_sep=" << identity_axis_separation_
                      << " min_conf=" << min_jersey_confidence_
                      << " soft_margin=" << soft_assignment_margin_
                      << " init_margin=" << initial_assignment_margin_
                      << " init_hits=" << initial_assignment_min_hits_
                      << " init_confirm=" << initial_assignment_confirm_frames_
                      << " centroid_margin=" << assignment_margin_;
        }

        this->sink_->put(frm);
    }

    static std::shared_ptr<TeamClassifier> create(NodeCreationInfo& nci) {
        EdgeManager& edges = nci.edges;
        const Parameters& params = nci.params;
        auto r = NodeSISO<av::VideoFrame, av::VideoFrame>::template createCommon<TeamClassifier>(edges, params);
        if (params.count("player_metadata_key")) r->player_metadata_key_ = params["player_metadata_key"].get<std::string>();
        if (params.count("seg_metadata_key")) r->seg_metadata_key_ = params["seg_metadata_key"].get<std::string>();
        if (params.count("debug_seg_metadata_key")) r->debug_seg_metadata_key_ = params["debug_seg_metadata_key"].get<std::string>();
        if (params.count("shot_metadata_key")) r->shot_metadata_key_ = params["shot_metadata_key"].get<std::string>();
        if (params.count("player_labels")) {
            r->player_labels_.clear();
            for (const auto& item : params["player_labels"]) r->player_labels_.push_back(item.get<std::string>());
        }
        if (params.count("seg_labels")) {
            r->seg_labels_.clear();
            for (const auto& item : params["seg_labels"]) r->seg_labels_.push_back(item.get<std::string>());
        }
        if (params.count("iou_match_threshold")) r->iou_match_threshold_ = params["iou_match_threshold"].get<float>();
        if (params.count("fallback_center_distance_px")) r->fallback_center_distance_px_ = params["fallback_center_distance_px"].get<float>();
        if (params.count("ema_alpha_track")) r->ema_alpha_track_ = params["ema_alpha_track"].get<float>();
        if (params.count("ema_alpha_centroid")) r->ema_alpha_centroid_ = params["ema_alpha_centroid"].get<float>();
        if (params.count("luma_weight")) r->luma_weight_ = params["luma_weight"].get<float>();
        if (params.count("min_jersey_confidence")) r->min_jersey_confidence_ = params["min_jersey_confidence"].get<float>();
        if (params.count("soft_assignment_margin")) r->soft_assignment_margin_ = params["soft_assignment_margin"].get<float>();
        if (params.count("initial_assignment_margin")) r->initial_assignment_margin_ = params["initial_assignment_margin"].get<float>();
        if (params.count("assignment_margin")) r->assignment_margin_ = params["assignment_margin"].get<float>();
        if (params.count("bootstrap_axis_min_separation")) r->bootstrap_axis_min_separation_ = params["bootstrap_axis_min_separation"].get<float>();
        if (params.count("initial_assignment_min_hits")) r->initial_assignment_min_hits_ = params["initial_assignment_min_hits"].get<int>();
        if (params.count("initial_assignment_confirm_frames")) r->initial_assignment_confirm_frames_ = params["initial_assignment_confirm_frames"].get<int>();
        if (params.count("bootstrap_frames")) r->bootstrap_frames_ = params["bootstrap_frames"].get<uint64_t>();
        if (params.count("bootstrap_min_tracks")) r->bootstrap_min_tracks_ = params["bootstrap_min_tracks"].get<int>();
        if (params.count("track_idle_frames")) r->track_idle_frames_ = params["track_idle_frames"].get<uint64_t>();
        if (params.count("handoff_max_age_frames")) r->handoff_max_age_frames_ = params["handoff_max_age_frames"].get<uint64_t>();
        if (params.count("handoff_max_center_distance_rel")) r->handoff_max_center_distance_rel_ = params["handoff_max_center_distance_rel"].get<float>();
        if (params.count("handoff_min_size_ratio")) r->handoff_min_size_ratio_ = params["handoff_min_size_ratio"].get<float>();
        if (params.count("handoff_max_color_distance")) r->handoff_max_color_distance_ = params["handoff_max_color_distance"].get<float>();
        if (params.count("min_jersey_pixels")) r->min_jersey_pixels_ = params["min_jersey_pixels"].get<int>();
        if (params.count("output_field")) r->output_field_ = params["output_field"].get<std::string>();
        if (params.count("output_team_color_field")) r->output_team_color_field_ = params["output_team_color_field"].get<std::string>();
        if (params.count("write_back_to_seg")) r->write_back_to_seg_ = params["write_back_to_seg"].get<bool>();
        if (params.count("rewrite_seg_cls")) r->rewrite_seg_cls_ = params["rewrite_seg_cls"].get<bool>();
        if (params.count("rewrite_label")) r->rewrite_label_ = params["rewrite_label"].get<bool>();
        if (params.count("team_label_names")) {
            r->team_label_names_.clear();
            for (const auto& item : params["team_label_names"]) r->team_label_names_.push_back(item.get<std::string>());
        }
        if (params.count("unknown_label")) r->unknown_label_ = params["unknown_label"].get<std::string>();
        if (params.count("debug_log_every_n")) r->debug_log_every_n_ = params["debug_log_every_n"].get<int>();
        return r;
    }
};

DECLNODE(team_classifier, TeamClassifier)
