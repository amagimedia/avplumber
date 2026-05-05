#include "../../node_common.hpp"

extern "C" {
#include <libavutil/dict.h>
}

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

constexpr int kUVHistBins = 256;
constexpr int kLHistBins = 16;

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

float chiSquareDistance(const float* a, const float* b, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; ++i) {
        const float d = a[i] - b[i];
        const float s = a[i] + b[i];
        if (s > 1e-7f) sum += (d * d) / s;
    }
    return sum;
}

void normalizeHistogram(float* hist, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; ++i) sum += hist[i];
    if (sum <= 1e-7f) return;
    const float inv = 1.0f / sum;
    for (int i = 0; i < n; ++i) hist[i] *= inv;
}

bool isZeroHistogram(const float* hist, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; ++i) sum += hist[i];
    return sum < 1e-7f;
}

void emaUpdateHistogram(float* ema, const float* sample, int n, float alpha) {
    for (int i = 0; i < n; ++i) {
        ema[i] = (1.0f - alpha) * ema[i] + alpha * sample[i];
    }
    normalizeHistogram(ema, n);
}

bool parseBBox(const Parameters& det, float& x1, float& y1, float& x2, float& y2) {
    if (!det.contains("xyxy") || !det["xyxy"].is_array() || det["xyxy"].size() < 4) return false;
    x1 = det["xyxy"][0].get<float>();
    y1 = det["xyxy"][1].get<float>();
    x2 = det["xyxy"][2].get<float>();
    y2 = det["xyxy"][3].get<float>();
    return x2 > x1 && y2 > y1;
}

bool matchesLabel(const Parameters& det, const std::vector<std::string>& labels) {
    if (labels.empty()) return true;
    if (!det.contains("label") || !det["label"].is_string()) return false;
    const std::string lbl = det["label"].get<std::string>();
    for (const auto& want : labels) {
        if (iequals(lbl, want)) return true;
    }
    return false;
}

int topBin(const float* hist, int n) {
    int top = 0;
    float best = -1.0f;
    for (int i = 0; i < n; ++i) {
        if (hist[i] > best) {
            best = hist[i];
            top = i;
        }
    }
    return top;
}

float histRangeSum(const float* hist, int begin, int end) {
    float sum = 0.0f;
    begin = std::max(0, begin);
    end = std::max(begin, end);
    for (int i = begin; i < end; ++i) sum += hist[i];
    return sum;
}

float histMeanBin(const float* hist, int n) {
    float weighted = 0.0f;
    float sum = 0.0f;
    for (int i = 0; i < n; ++i) {
        weighted += hist[i] * (float)i;
        sum += hist[i];
    }
    return sum > 1e-7f ? weighted / sum : 0.0f;
}

struct PlayerDet {
    int det_index = -1;
    int track_id = -1;
    float x1 = 0.0f;
    float y1 = 0.0f;
    float x2 = 0.0f;
    float y2 = 0.0f;
};

struct TorsoDet {
    int det_index = -1;
    int source_det_index = -1;
    int source_plane_index = -1;
    float x1 = 0.0f;
    float y1 = 0.0f;
    float x2 = 0.0f;
    float y2 = 0.0f;
    bool has_hist = false;
    std::array<float, kUVHistBins> uv_hist{};
    std::array<float, kLHistBins> l_hist{};
    int pixels = 0;
    int mode_pixels = 0;
    int cloth_pixels = 0;
    int skin_pixels = 0;
    float confidence = 0.0f;
    float mode_ratio = 0.0f;
    int matched_player = -1;
    float match_iou = 0.0f;
    float match_center_distance = 0.0f;
};

struct Sample {
    std::array<float, kUVHistBins> uv_hist{};
    std::array<float, kLHistBins> l_hist{};
    int pixels = 0;
    float confidence = 0.0f;
    uint64_t frame = 0;
};

struct Assignment {
    int team = -1;
    float distance_a = 0.0f;
    float distance_b = 0.0f;
    float margin = 0.0f;
    float confidence = 0.0f;
    std::string reason = "unassigned";
};

struct TrackFallbackState {
    int team = -1;
    float confidence = 0.0f;
    float margin = 0.0f;
    float distance_a = 0.0f;
    float distance_b = 0.0f;
    uint64_t frame = 0;
};

} // namespace

class TorsoTeamClassifier : public NodeSISO<av::VideoFrame, av::VideoFrame>, public ReportsFinishByFlag {
    std::string player_metadata_key_ = "yolo_players";
    std::string torso_metadata_key_ = "yolo_players_torso_seg";
    std::string player_seg_metadata_key_ = "yolo_players_seg";
    std::string output_player_metadata_key_ = "yolo_players_team";
    std::string debug_metadata_key_ = "team_classifier_debug";
    std::string camera_shot_metadata_key_ = "camera_shot_info";
    std::vector<std::string> player_labels_ = {"Player"};
    std::vector<std::string> torso_labels_ = {"torso"};
    std::vector<std::string> team_ab_ = {"A", "B"};

    float iou_match_threshold_ = 0.10f;
    float fallback_center_distance_px_ = 60.0f;
    float uv_weight_ = 1.0f;
    float l_weight_ = 1.0f;
    float min_jersey_confidence_ = 0.05f;
    float assignment_margin_ = 0.03f;
    float prototype_update_margin_ = 0.06f;
    float ema_alpha_centroid_ = 0.02f;
    float bootstrap_min_prototype_distance_ = 0.06f;
    int min_jersey_pixels_ = 32;
    int bootstrap_min_samples_ = 12;
    int bootstrap_min_cluster_size_ = 3;
    uint64_t bootstrap_frames_ = 10;
    int max_bootstrap_samples_ = 600;
    bool require_wide_shot_ = true;
    bool require_player_match_for_training_ = true;
    bool require_player_match_for_assignment_ = true;
    bool unknown_on_low_margin_ = true;
    bool write_back_to_torso_ = true;
    bool rewrite_torso_cls_ = true;
    bool write_back_to_player_seg_ = true;
    bool rewrite_player_seg_cls_ = false;
    bool strip_input_metadata_ = false;
    bool tracker_fallback_enabled_ = true;
    uint64_t tracker_fallback_max_age_frames_ = 3;
    float tracker_fallback_min_margin_ = 0.06f;
    float tracker_fallback_min_confidence_ = 0.02f;
    int debug_log_every_n_ = 0;

    uint64_t frame_counter_ = 0;
    uint64_t wide_frame_counter_ = 0;
    bool bootstrapped_ = false;
    std::array<float, kUVHistBins> proto_uv_[2]{};
    std::array<float, kLHistBins> proto_l_[2]{};
    std::vector<Sample> bootstrap_samples_;
    std::unordered_map<int, TrackFallbackState> track_fallback_;
    int bootstrap_cluster_count_[2] = {0, 0};

    void resetState() {
        frame_counter_ = 0;
        wide_frame_counter_ = 0;
        bootstrapped_ = false;
        proto_uv_[0].fill(0.0f);
        proto_uv_[1].fill(0.0f);
        proto_l_[0].fill(0.0f);
        proto_l_[1].fill(0.0f);
        bootstrap_samples_.clear();
        track_fallback_.clear();
        bootstrap_cluster_count_[0] = 0;
        bootstrap_cluster_count_[1] = 0;
    }

    float combinedHistDistance(const float* uv_a, const float* l_a,
                               const float* uv_b, const float* l_b) const {
        return uv_weight_ * chiSquareDistance(uv_a, uv_b, kUVHistBins) +
               l_weight_ * chiSquareDistance(l_a, l_b, kLHistBins);
    }

    float prototypeDistance() const {
        if (!bootstrapped_) return 0.0f;
        return combinedHistDistance(proto_uv_[0].data(), proto_l_[0].data(),
                                    proto_uv_[1].data(), proto_l_[1].data());
    }

    std::string readShotType(const AVFrame* raw) const {
        if (!raw || !raw->metadata || camera_shot_metadata_key_.empty()) return {};
        AVDictionaryEntry* entry = av_dict_get(raw->metadata, camera_shot_metadata_key_.c_str(), nullptr, 0);
        if (!entry || !entry->value) return {};
        try {
            const Parameters shot_md = Parameters::parse(entry->value);
            return shot_md.value("camera_shot_type", std::string());
        } catch (...) {
            return {};
        }
    }

    bool readMetadata(const AVFrame* raw, const std::string& key, Parameters& out) const {
        if (!raw || !raw->metadata) return false;
        AVDictionaryEntry* entry = av_dict_get(raw->metadata, key.c_str(), nullptr, 0);
        if (!entry || !entry->value) return false;
        try {
            out = Parameters::parse(entry->value);
            return out.contains("detections") && out["detections"].is_array();
        } catch (...) {
            return false;
        }
    }

    bool parseJerseyHistogram(const Parameters& det,
                              std::array<float, kUVHistBins>& uv_hist,
                              std::array<float, kLHistBins>& l_hist) const {
        if (!det.contains("jersey_uv_hist") || !det["jersey_uv_hist"].is_array() ||
            (int)det["jersey_uv_hist"].size() < kUVHistBins) return false;
        if (!det.contains("jersey_l_hist") || !det["jersey_l_hist"].is_array() ||
            (int)det["jersey_l_hist"].size() < kLHistBins) return false;

        for (int i = 0; i < kUVHistBins; ++i) uv_hist[i] = det["jersey_uv_hist"][i].get<float>();
        for (int i = 0; i < kLHistBins; ++i) l_hist[i] = det["jersey_l_hist"][i].get<float>();
        normalizeHistogram(uv_hist.data(), kUVHistBins);
        normalizeHistogram(l_hist.data(), kLHistBins);
        return !isZeroHistogram(uv_hist.data(), kUVHistBins) || !isZeroHistogram(l_hist.data(), kLHistBins);
    }

    void parsePlayers(const Parameters& player_md, std::vector<PlayerDet>& players) const {
        players.clear();
        if (!player_md.contains("detections") || !player_md["detections"].is_array()) return;
        for (int i = 0; i < (int)player_md["detections"].size(); ++i) {
            const auto& det = player_md["detections"][i];
            if (!det.is_object() || !matchesLabel(det, player_labels_)) continue;
            PlayerDet p;
            p.det_index = i;
            p.track_id = det.value("track_id", -1);
            if (!parseBBox(det, p.x1, p.y1, p.x2, p.y2)) continue;
            players.push_back(p);
        }
    }

    void parseTorsos(const Parameters& torso_md, std::vector<TorsoDet>& torsos) const {
        torsos.clear();
        if (!torso_md.contains("detections") || !torso_md["detections"].is_array()) return;
        for (int i = 0; i < (int)torso_md["detections"].size(); ++i) {
            const auto& det = torso_md["detections"][i];
            if (!det.is_object() || !matchesLabel(det, torso_labels_)) continue;
            TorsoDet t;
            t.det_index = i;
            if (!parseBBox(det, t.x1, t.y1, t.x2, t.y2)) continue;
            t.source_det_index = det.value("source_det_index", -1);
            t.source_plane_index = det.value("source_plane_index", -1);
            t.pixels = det.value("jersey_pixels", 0);
            t.mode_pixels = det.value("jersey_mode_pixels", 0);
            t.cloth_pixels = det.value("jersey_cloth_pixels", t.pixels);
            t.skin_pixels = det.value("jersey_skin_pixels", 0);
            t.confidence = det.value("jersey_confidence", 0.0f);
            t.mode_ratio = det.value("jersey_mode_ratio", 0.0f);
            t.has_hist = parseJerseyHistogram(det, t.uv_hist, t.l_hist);
            torsos.push_back(t);
        }
    }

    void matchPlayersToTorsos(const std::vector<PlayerDet>& players, std::vector<TorsoDet>& torsos) const {
        std::vector<bool> torso_used(torsos.size(), false);
        for (size_t pi = 0; pi < players.size(); ++pi) {
            float best_iou = 0.0f;
            int best_ti = -1;
            for (size_t ti = 0; ti < torsos.size(); ++ti) {
                if (torso_used[ti]) continue;
                const float iou = bboxIoU(players[pi].x1, players[pi].y1, players[pi].x2, players[pi].y2,
                                          torsos[ti].x1, torsos[ti].y1, torsos[ti].x2, torsos[ti].y2);
                if (iou > best_iou) {
                    best_iou = iou;
                    best_ti = (int)ti;
                }
            }
            if (best_ti >= 0 && best_iou >= iou_match_threshold_) {
                torsos[(size_t)best_ti].matched_player = (int)pi;
                torsos[(size_t)best_ti].match_iou = best_iou;
                torsos[(size_t)best_ti].match_center_distance =
                    centerDistance(players[pi].x1, players[pi].y1, players[pi].x2, players[pi].y2,
                                   torsos[(size_t)best_ti].x1, torsos[(size_t)best_ti].y1,
                                   torsos[(size_t)best_ti].x2, torsos[(size_t)best_ti].y2);
                torso_used[(size_t)best_ti] = true;
            }
        }

        if (fallback_center_distance_px_ <= 0.0f) return;
        for (size_t pi = 0; pi < players.size(); ++pi) {
            bool already_matched = false;
            for (const auto& torso : torsos) {
                if (torso.matched_player == (int)pi) {
                    already_matched = true;
                    break;
                }
            }
            if (already_matched) continue;

            float best_d = std::numeric_limits<float>::max();
            int best_ti = -1;
            for (size_t ti = 0; ti < torsos.size(); ++ti) {
                if (torso_used[ti]) continue;
                const float d = centerDistance(players[pi].x1, players[pi].y1, players[pi].x2, players[pi].y2,
                                               torsos[ti].x1, torsos[ti].y1, torsos[ti].x2, torsos[ti].y2);
                if (d < best_d) {
                    best_d = d;
                    best_ti = (int)ti;
                }
            }
            if (best_ti >= 0 && best_d <= fallback_center_distance_px_) {
                torsos[(size_t)best_ti].matched_player = (int)pi;
                torsos[(size_t)best_ti].match_center_distance = best_d;
                torsos[(size_t)best_ti].match_iou =
                    bboxIoU(players[pi].x1, players[pi].y1, players[pi].x2, players[pi].y2,
                            torsos[(size_t)best_ti].x1, torsos[(size_t)best_ti].y1,
                            torsos[(size_t)best_ti].x2, torsos[(size_t)best_ti].y2);
                torso_used[(size_t)best_ti] = true;
            }
        }
    }

    bool validSample(const TorsoDet& torso, std::string* reason = nullptr) const {
        if (require_player_match_for_training_ && torso.matched_player < 0) {
            if (reason) *reason = "unmatched_player";
            return false;
        }
        if (!torso.has_hist) {
            if (reason) *reason = "no_hist";
            return false;
        }
        if (torso.cloth_pixels < min_jersey_pixels_) {
            if (reason) *reason = "low_pixels";
            return false;
        }
        if (torso.confidence < min_jersey_confidence_) {
            if (reason) *reason = "low_confidence";
            return false;
        }
        if (isZeroHistogram(torso.uv_hist.data(), kUVHistBins) &&
            isZeroHistogram(torso.l_hist.data(), kLHistBins)) {
            if (reason) *reason = "zero_hist";
            return false;
        }
        if (reason) *reason = "valid";
        return true;
    }

    bool assignableTorso(const TorsoDet& torso, std::string* reason = nullptr) const {
        if (require_player_match_for_assignment_ && torso.matched_player < 0) {
            if (reason) *reason = "unmatched_player";
            return false;
        }
        return validSample(torso, reason);
    }

    void appendBootstrapSample(const TorsoDet& torso) {
        if ((int)bootstrap_samples_.size() >= max_bootstrap_samples_) return;
        Sample s;
        s.uv_hist = torso.uv_hist;
        s.l_hist = torso.l_hist;
        s.pixels = torso.cloth_pixels;
        s.confidence = torso.confidence;
        s.frame = frame_counter_;
        bootstrap_samples_.push_back(s);
    }

    bool bootstrapPrototypes() {
        if ((int)bootstrap_samples_.size() < bootstrap_min_samples_) return false;
        if (wide_frame_counter_ < bootstrap_frames_) return false;

        std::memcpy(proto_uv_[0].data(), bootstrap_samples_[0].uv_hist.data(), sizeof(float) * kUVHistBins);
        std::memcpy(proto_l_[0].data(), bootstrap_samples_[0].l_hist.data(), sizeof(float) * kLHistBins);

        float best_d = -1.0f;
        size_t best_i = 0;
        for (size_t i = 0; i < bootstrap_samples_.size(); ++i) {
            const float d = combinedHistDistance(bootstrap_samples_[i].uv_hist.data(), bootstrap_samples_[i].l_hist.data(),
                                                 proto_uv_[0].data(), proto_l_[0].data());
            if (d > best_d) {
                best_d = d;
                best_i = i;
            }
        }
        std::memcpy(proto_uv_[1].data(), bootstrap_samples_[best_i].uv_hist.data(), sizeof(float) * kUVHistBins);
        std::memcpy(proto_l_[1].data(), bootstrap_samples_[best_i].l_hist.data(), sizeof(float) * kLHistBins);

        int count[2] = {0, 0};
        for (int iter = 0; iter < 12; ++iter) {
            std::array<float, kUVHistBins> sum_uv[2]{};
            std::array<float, kLHistBins> sum_l[2]{};
            count[0] = 0;
            count[1] = 0;
            for (const auto& s : bootstrap_samples_) {
                const float d0 = combinedHistDistance(s.uv_hist.data(), s.l_hist.data(),
                                                      proto_uv_[0].data(), proto_l_[0].data());
                const float d1 = combinedHistDistance(s.uv_hist.data(), s.l_hist.data(),
                                                      proto_uv_[1].data(), proto_l_[1].data());
                const int k = d0 <= d1 ? 0 : 1;
                for (int b = 0; b < kUVHistBins; ++b) sum_uv[k][b] += s.uv_hist[b];
                for (int b = 0; b < kLHistBins; ++b) sum_l[k][b] += s.l_hist[b];
                count[k] += 1;
            }
            for (int k = 0; k < 2; ++k) {
                if (count[k] <= 0) continue;
                const float inv = 1.0f / (float)count[k];
                for (int b = 0; b < kUVHistBins; ++b) proto_uv_[k][b] = sum_uv[k][b] * inv;
                for (int b = 0; b < kLHistBins; ++b) proto_l_[k][b] = sum_l[k][b] * inv;
                normalizeHistogram(proto_uv_[k].data(), kUVHistBins);
                normalizeHistogram(proto_l_[k].data(), kLHistBins);
            }
        }

        if (count[0] < bootstrap_min_cluster_size_ || count[1] < bootstrap_min_cluster_size_) {
            return false;
        }

        const float sep = combinedHistDistance(proto_uv_[0].data(), proto_l_[0].data(),
                                               proto_uv_[1].data(), proto_l_[1].data());
        if (sep < bootstrap_min_prototype_distance_) return false;

        if (histMeanBin(proto_l_[0].data(), kLHistBins) > histMeanBin(proto_l_[1].data(), kLHistBins)) {
            std::swap(proto_uv_[0], proto_uv_[1]);
            std::swap(proto_l_[0], proto_l_[1]);
            std::swap(count[0], count[1]);
        }

        bootstrap_cluster_count_[0] = count[0];
        bootstrap_cluster_count_[1] = count[1];
        bootstrapped_ = true;
        logstream << "torso_team_classifier: bootstrap"
                  << " frame=" << frame_counter_
                  << " wide_frames=" << wide_frame_counter_
                  << " samples=" << bootstrap_samples_.size()
                  << " cluster_a=" << bootstrap_cluster_count_[0]
                  << " cluster_b=" << bootstrap_cluster_count_[1]
                  << " prototype_sep=" << sep
                  << " l_mean_a=" << histMeanBin(proto_l_[0].data(), kLHistBins)
                  << " l_mean_b=" << histMeanBin(proto_l_[1].data(), kLHistBins);
        return true;
    }

    Assignment classifyTorso(const TorsoDet& torso) const {
        Assignment a;
        std::string reason;
        if (!assignableTorso(torso, &reason)) {
            a.reason = reason;
            return a;
        }
        if (!bootstrapped_) {
            a.reason = "bootstrap_pending";
            return a;
        }

        a.distance_a = combinedHistDistance(torso.uv_hist.data(), torso.l_hist.data(),
                                            proto_uv_[0].data(), proto_l_[0].data());
        a.distance_b = combinedHistDistance(torso.uv_hist.data(), torso.l_hist.data(),
                                            proto_uv_[1].data(), proto_l_[1].data());
        a.team = a.distance_a <= a.distance_b ? 0 : 1;
        a.margin = std::fabs(a.distance_a - a.distance_b);
        a.confidence = a.margin / std::max(1e-6f, a.distance_a + a.distance_b);
        if (unknown_on_low_margin_ && a.margin < assignment_margin_) {
            a.team = -1;
            a.reason = "low_margin";
        } else {
            a.reason = "nearest_prototype";
        }
        return a;
    }

    bool isReliablePrimaryAssignment(const Assignment& a) const {
        return a.team >= 0 &&
               a.reason == "nearest_prototype" &&
               a.margin >= tracker_fallback_min_margin_ &&
               a.confidence >= tracker_fallback_min_confidence_;
    }

    bool applyTrackerFallback(const PlayerDet& player, Assignment& assignment) const {
        if (!tracker_fallback_enabled_) return false;
        if (assignment.team >= 0) return false;
        if (player.track_id < 0) return false;

        auto it = track_fallback_.find(player.track_id);
        if (it == track_fallback_.end()) return false;

        const TrackFallbackState& st = it->second;
        if (st.team < 0) return false;
        const uint64_t age = frame_counter_ >= st.frame ? frame_counter_ - st.frame : 0;
        if (age == 0 || age > tracker_fallback_max_age_frames_) return false;
        if (st.margin < tracker_fallback_min_margin_ ||
            st.confidence < tracker_fallback_min_confidence_) return false;

        assignment.team = st.team;
        assignment.confidence = st.confidence;
        assignment.margin = st.margin;
        assignment.distance_a = st.distance_a;
        assignment.distance_b = st.distance_b;
        assignment.reason = "tracker_fallback";
        return true;
    }

    void updateTrackFallbackState(const PlayerDet& player, const Assignment& assignment) {
        if (!tracker_fallback_enabled_) return;
        if (player.track_id < 0 || !isReliablePrimaryAssignment(assignment)) return;

        TrackFallbackState st;
        st.team = assignment.team;
        st.confidence = assignment.confidence;
        st.margin = assignment.margin;
        st.distance_a = assignment.distance_a;
        st.distance_b = assignment.distance_b;
        st.frame = frame_counter_;
        track_fallback_[player.track_id] = st;
    }

    void pruneTrackFallbackState() {
        if (!tracker_fallback_enabled_) return;
        for (auto it = track_fallback_.begin(); it != track_fallback_.end();) {
            const uint64_t age = frame_counter_ >= it->second.frame ? frame_counter_ - it->second.frame : 0;
            if (age > tracker_fallback_max_age_frames_) {
                it = track_fallback_.erase(it);
            } else {
                ++it;
            }
        }
    }

    std::string teamAb(int team) const {
        if (team >= 0 && team < (int)team_ab_.size()) return team_ab_[(size_t)team];
        return "?";
    }

    void appendTorsoStats(Parameters& det, const TorsoDet& torso) const {
        const int l_top = topBin(torso.l_hist.data(), kLHistBins);
        const int uv_top = topBin(torso.uv_hist.data(), kUVHistBins);
        det["torso_pixels_used"] = torso.cloth_pixels;
        det["torso_mode_pixels"] = torso.mode_pixels;
        det["torso_skin_pixels"] = torso.skin_pixels;
        det["torso_jersey_confidence"] = torso.confidence;
        det["torso_mode_ratio"] = torso.mode_ratio;
        det["torso_l_top_bin"] = l_top;
        det["torso_l_top_ratio"] = torso.l_hist[(size_t)l_top];
        det["torso_l_mean_bin"] = histMeanBin(torso.l_hist.data(), kLHistBins);
        det["torso_l_dark_ratio"] = histRangeSum(torso.l_hist.data(), 0, 5);
        det["torso_l_light_ratio"] = histRangeSum(torso.l_hist.data(), 11, kLHistBins);
        det["torso_uv_top_bin"] = uv_top;
        det["torso_uv_top_ratio"] = torso.uv_hist[(size_t)uv_top];
    }

    void writeAssignmentFields(Parameters& det, const TorsoDet* torso, const Assignment& a) const {
        det["team"] = a.team;
        det["team_ab"] = teamAb(a.team);
        det["team_confidence"] = a.confidence;
        det["team_margin"] = a.margin;
        det["team_distance_a"] = a.distance_a;
        det["team_distance_b"] = a.distance_b;
        det["team_decision_reason"] = a.reason;
        det["team_tracker_fallback"] = a.reason == "tracker_fallback";
        det["team_classifier"] = "torso_team_classifier";
        if (torso) {
            det["team_torso_det_index"] = torso->det_index;
            det["team_torso_source_det_index"] = torso->source_det_index;
            det["team_torso_source_plane_index"] = torso->source_plane_index;
            det["team_match_iou"] = torso->match_iou;
            det["team_match_center_distance"] = torso->match_center_distance;
            appendTorsoStats(det, *torso);
        }
    }

    Parameters buildDebugMetadata(bool wide_shot,
                                  int players,
                                  int torsos,
                                  int valid_samples,
                                  int matched_players,
                                  int assigned_a,
                                  int assigned_b,
                                  int unknown,
                                  int tracker_fallback_used) const {
        Parameters dbg = Parameters::object();
        dbg["classifier"] = "torso_team_classifier";
        dbg["algorithm"] = "torso_uv_l_hist_kmeans";
        dbg["frame"] = frame_counter_;
        dbg["wide_frame"] = wide_frame_counter_;
        dbg["wide_shot"] = wide_shot;
        dbg["bootstrapped"] = bootstrapped_;
        dbg["bootstrap_samples"] = bootstrap_samples_.size();
        dbg["bootstrap_cluster_a"] = bootstrap_cluster_count_[0];
        dbg["bootstrap_cluster_b"] = bootstrap_cluster_count_[1];
        dbg["current_players"] = players;
        dbg["current_torsos"] = torsos;
        dbg["current_valid_samples"] = valid_samples;
        dbg["matched_players"] = matched_players;
        dbg["assigned_a"] = assigned_a;
        dbg["assigned_b"] = assigned_b;
        dbg["unknown"] = unknown;
        dbg["tracker_fallback_used"] = tracker_fallback_used;
        dbg["tracker_fallback_tracks"] = track_fallback_.size();
        dbg["prototype_distance"] = prototypeDistance();
        dbg["uv_weight"] = uv_weight_;
        dbg["l_weight"] = l_weight_;
        if (bootstrapped_) {
            Parameters proto = Parameters::array();
            for (int team = 0; team < 2; ++team) {
                Parameters p = Parameters::object();
                const int l_top = topBin(proto_l_[team].data(), kLHistBins);
                const int uv_top = topBin(proto_uv_[team].data(), kUVHistBins);
                p["team"] = team;
                p["team_ab"] = teamAb(team);
                p["l_top_bin"] = l_top;
                p["l_top_ratio"] = proto_l_[team][(size_t)l_top];
                p["l_mean_bin"] = histMeanBin(proto_l_[team].data(), kLHistBins);
                p["l_dark_ratio"] = histRangeSum(proto_l_[team].data(), 0, 5);
                p["l_light_ratio"] = histRangeSum(proto_l_[team].data(), 11, kLHistBins);
                p["uv_top_bin"] = uv_top;
                p["uv_top_ratio"] = proto_uv_[team][(size_t)uv_top];
                proto.push_back(p);
            }
            dbg["prototypes"] = proto;
        }
        return dbg;
    }

    void setMetadata(av::VideoFrame& frm,
                     const Parameters& player_team_md,
                     const Parameters& torso_md,
                     const Parameters* player_seg_md,
                     const Parameters& debug_md) const {
        AVFrame* raw = frm.raw();
        if (!raw) return;
        if (strip_input_metadata_) {
            av_dict_free(&raw->metadata);
        }
        const std::string player_s = player_team_md.dump();
        const std::string torso_s = torso_md.dump();
        const std::string debug_s = debug_md.dump();
        av_dict_set(&raw->metadata, output_player_metadata_key_.c_str(), player_s.c_str(), 0);
        av_dict_set(&raw->metadata, torso_metadata_key_.c_str(), torso_s.c_str(), 0);
        if (player_seg_md) {
            const std::string player_seg_s = player_seg_md->dump();
            av_dict_set(&raw->metadata, player_seg_metadata_key_.c_str(), player_seg_s.c_str(), 0);
        }
        if (!debug_metadata_key_.empty()) {
            av_dict_set(&raw->metadata, debug_metadata_key_.c_str(), debug_s.c_str(), 0);
        }
    }

public:
    using NodeSISO<av::VideoFrame, av::VideoFrame>::NodeSISO;
    bool consumeEofIfPresent() override {
        return false;
    }

    void process() override {
        av::VideoFrame frm = this->source_->get();
        if (isEofMarker(frm)) {
            resetState();
            this->sink_->put(frm);
            this->finished_ = true;
            return;
        }
        if (!frm) return;

        ++frame_counter_;
        AVFrame* raw = frm.raw();
        const std::string shot_type = readShotType(raw);
        const bool wide_shot = !require_wide_shot_ || shot_type == "wide";
        if (wide_shot) ++wide_frame_counter_;
        if (!wide_shot) {
            track_fallback_.clear();
        } else {
            pruneTrackFallbackState();
        }

        Parameters player_md;
        Parameters torso_md;
        Parameters player_seg_md;
        const bool have_players = readMetadata(raw, player_metadata_key_, player_md);
        const bool have_torso = readMetadata(raw, torso_metadata_key_, torso_md);
        const bool have_player_seg = !player_seg_metadata_key_.empty() &&
                                     readMetadata(raw, player_seg_metadata_key_, player_seg_md);
        if (!have_players || !have_torso) {
            Parameters empty_players = Parameters::object();
            empty_players["detections"] = Parameters::array();
            Parameters empty_torso = Parameters::object();
            empty_torso["detections"] = Parameters::array();
            Parameters dbg = buildDebugMetadata(wide_shot, 0, 0, 0, 0, 0, 0, 0, 0);
            dbg["missing_players"] = !have_players;
            dbg["missing_torso"] = !have_torso;
            setMetadata(frm, have_players ? player_md : empty_players,
                        have_torso ? torso_md : empty_torso,
                        have_player_seg ? &player_seg_md : nullptr,
                        dbg);
            this->sink_->put(frm);
            return;
        }

        Parameters player_team_md = player_md;
        std::vector<PlayerDet> players;
        std::vector<TorsoDet> torsos;
        parsePlayers(player_md, players);
        parseTorsos(torso_md, torsos);
        matchPlayersToTorsos(players, torsos);

        for (const auto& p : players) {
            auto& det = player_team_md["detections"][p.det_index];
            Assignment unknown;
            unknown.reason = wide_shot ? "unmatched_torso" : "not_wide";
            writeAssignmentFields(det, nullptr, unknown);
        }
        for (auto& det : torso_md["detections"]) {
            if (!det.is_object() || !matchesLabel(det, torso_labels_)) continue;
            Assignment unknown;
            unknown.reason = wide_shot ? "unassigned" : "not_wide";
            if (write_back_to_torso_) writeAssignmentFields(det, nullptr, unknown);
            if (rewrite_torso_cls_) det["cls"] = -1;
        }
        if (have_player_seg) {
            for (auto& det : player_seg_md["detections"]) {
                if (!det.is_object()) continue;
                det["team"] = -1;
                det["team_ab"] = "?";
                det["team_decision_reason"] = wide_shot ? "unassigned" : "not_wide";
            }
        }

        int valid_samples = 0;
        int matched_players = 0;
        int assigned_a = 0;
        int assigned_b = 0;
        int unknown_count = 0;
        int tracker_fallback_used = 0;
        std::vector<Assignment> assignments(torsos.size());

        for (const auto& torso : torsos) {
            if (torso.matched_player >= 0) ++matched_players;
            std::string reason;
            if (wide_shot && validSample(torso, &reason)) {
                ++valid_samples;
                if (!bootstrapped_) appendBootstrapSample(torso);
            }
        }

        if (wide_shot && !bootstrapped_) {
            bootstrapPrototypes();
        }

        if (wide_shot) {
            for (size_t ti = 0; ti < torsos.size(); ++ti) {
                assignments[ti] = classifyTorso(torsos[ti]);
                if (assignments[ti].team < 0 &&
                    torsos[ti].matched_player >= 0 &&
                    torsos[ti].matched_player < (int)players.size() &&
                    applyTrackerFallback(players[(size_t)torsos[ti].matched_player], assignments[ti])) {
                    ++tracker_fallback_used;
                }
                if (assignments[ti].team == 0) ++assigned_a;
                else if (assignments[ti].team == 1) ++assigned_b;
                else ++unknown_count;

                if (bootstrapped_ && assignments[ti].team >= 0 &&
                    assignments[ti].reason == "nearest_prototype" &&
                    assignments[ti].margin >= prototype_update_margin_) {
                    emaUpdateHistogram(proto_uv_[assignments[ti].team].data(),
                                       torsos[ti].uv_hist.data(), kUVHistBins, ema_alpha_centroid_);
                    emaUpdateHistogram(proto_l_[assignments[ti].team].data(),
                                       torsos[ti].l_hist.data(), kLHistBins, ema_alpha_centroid_);
                }
            }
        } else {
            unknown_count = (int)torsos.size();
            for (auto& a : assignments) a.reason = "not_wide";
        }

        for (size_t ti = 0; ti < torsos.size(); ++ti) {
            const TorsoDet& torso = torsos[ti];
            const Assignment& assignment = assignments[ti];

            if (torso.det_index >= 0 && torso.det_index < (int)torso_md["detections"].size()) {
                auto& torso_det = torso_md["detections"][torso.det_index];
                if (write_back_to_torso_) writeAssignmentFields(torso_det, &torso, assignment);
                if (rewrite_torso_cls_) torso_det["cls"] = assignment.team;
            }

            if (torso.matched_player >= 0 && torso.matched_player < (int)players.size()) {
                const PlayerDet& player = players[(size_t)torso.matched_player];
                if (player.det_index >= 0 && player.det_index < (int)player_team_md["detections"].size()) {
                    auto& player_det = player_team_md["detections"][player.det_index];
                    writeAssignmentFields(player_det, &torso, assignment);
                }
                updateTrackFallbackState(player, assignment);
            }

            if (have_player_seg && write_back_to_player_seg_ &&
                torso.source_det_index >= 0 &&
                torso.source_det_index < (int)player_seg_md["detections"].size()) {
                auto& seg_det = player_seg_md["detections"][torso.source_det_index];
                writeAssignmentFields(seg_det, &torso, assignment);
                if (rewrite_player_seg_cls_) seg_det["cls"] = assignment.team;
            }
        }

        const Parameters debug_md = buildDebugMetadata(wide_shot,
                                                       (int)players.size(),
                                                       (int)torsos.size(),
                                                       valid_samples,
                                                       matched_players,
                                                       assigned_a,
                                                       assigned_b,
                                                       unknown_count,
                                                       tracker_fallback_used);

        setMetadata(frm, player_team_md, torso_md, have_player_seg ? &player_seg_md : nullptr, debug_md);

        if (debug_log_every_n_ > 0 && (frame_counter_ % (uint64_t)debug_log_every_n_) == 0) {
            logstream << "torso_team_classifier: frame=" << frame_counter_
                      << " shot=" << (shot_type.empty() ? std::string("<missing>") : shot_type)
                      << " wide=" << (wide_shot ? 1 : 0)
                      << " players=" << players.size()
                      << " torsos=" << torsos.size()
                      << " matched=" << matched_players
                      << " valid=" << valid_samples
                      << " assigned_a=" << assigned_a
                      << " assigned_b=" << assigned_b
                      << " unknown=" << unknown_count
                      << " tracker_fallback=" << tracker_fallback_used
                      << " bootstrapped=" << (bootstrapped_ ? 1 : 0)
                      << " samples=" << bootstrap_samples_.size()
                      << " proto_sep=" << prototypeDistance();
        }

        this->sink_->put(frm);
    }

    static std::shared_ptr<TorsoTeamClassifier> create(NodeCreationInfo& nci) {
        EdgeManager& edges = nci.edges;
        const Parameters& params = nci.params;
        auto r = NodeSISO<av::VideoFrame, av::VideoFrame>::template createCommon<TorsoTeamClassifier>(edges, params);

        if (params.count("player_metadata_key")) r->player_metadata_key_ = params["player_metadata_key"].get<std::string>();
        if (params.count("torso_metadata_key")) r->torso_metadata_key_ = params["torso_metadata_key"].get<std::string>();
        if (params.count("player_seg_metadata_key")) r->player_seg_metadata_key_ = params["player_seg_metadata_key"].get<std::string>();
        if (params.count("output_player_metadata_key")) r->output_player_metadata_key_ = params["output_player_metadata_key"].get<std::string>();
        if (params.count("debug_metadata_key")) r->debug_metadata_key_ = params["debug_metadata_key"].get<std::string>();
        if (params.count("camera_shot_metadata_key")) r->camera_shot_metadata_key_ = params["camera_shot_metadata_key"].get<std::string>();
        if (params.count("player_labels")) {
            r->player_labels_.clear();
            for (const auto& item : params["player_labels"]) r->player_labels_.push_back(item.get<std::string>());
        }
        if (params.count("torso_labels")) {
            r->torso_labels_.clear();
            for (const auto& item : params["torso_labels"]) r->torso_labels_.push_back(item.get<std::string>());
        }
        if (params.count("team_ab")) {
            r->team_ab_.clear();
            for (const auto& item : params["team_ab"]) r->team_ab_.push_back(item.get<std::string>());
        }
        if (params.count("iou_match_threshold")) r->iou_match_threshold_ = params["iou_match_threshold"].get<float>();
        if (params.count("fallback_center_distance_px")) r->fallback_center_distance_px_ = params["fallback_center_distance_px"].get<float>();
        if (params.count("uv_weight")) r->uv_weight_ = params["uv_weight"].get<float>();
        if (params.count("l_weight")) r->l_weight_ = params["l_weight"].get<float>();
        if (params.count("min_jersey_confidence")) r->min_jersey_confidence_ = params["min_jersey_confidence"].get<float>();
        if (params.count("assignment_margin")) r->assignment_margin_ = params["assignment_margin"].get<float>();
        if (params.count("prototype_update_margin")) r->prototype_update_margin_ = params["prototype_update_margin"].get<float>();
        if (params.count("ema_alpha_centroid")) r->ema_alpha_centroid_ = params["ema_alpha_centroid"].get<float>();
        if (params.count("bootstrap_min_prototype_distance")) r->bootstrap_min_prototype_distance_ = params["bootstrap_min_prototype_distance"].get<float>();
        if (params.count("min_jersey_pixels")) r->min_jersey_pixels_ = params["min_jersey_pixels"].get<int>();
        if (params.count("bootstrap_min_samples")) r->bootstrap_min_samples_ = params["bootstrap_min_samples"].get<int>();
        if (params.count("bootstrap_min_cluster_size")) r->bootstrap_min_cluster_size_ = params["bootstrap_min_cluster_size"].get<int>();
        if (params.count("bootstrap_frames")) r->bootstrap_frames_ = params["bootstrap_frames"].get<uint64_t>();
        if (params.count("max_bootstrap_samples")) r->max_bootstrap_samples_ = params["max_bootstrap_samples"].get<int>();
        if (params.count("require_wide_shot")) r->require_wide_shot_ = params["require_wide_shot"].get<bool>();
        if (params.count("require_player_match_for_training")) r->require_player_match_for_training_ = params["require_player_match_for_training"].get<bool>();
        if (params.count("require_player_match_for_assignment")) r->require_player_match_for_assignment_ = params["require_player_match_for_assignment"].get<bool>();
        if (params.count("unknown_on_low_margin")) r->unknown_on_low_margin_ = params["unknown_on_low_margin"].get<bool>();
        if (params.count("write_back_to_torso")) r->write_back_to_torso_ = params["write_back_to_torso"].get<bool>();
        if (params.count("rewrite_torso_cls")) r->rewrite_torso_cls_ = params["rewrite_torso_cls"].get<bool>();
        if (params.count("write_back_to_player_seg")) r->write_back_to_player_seg_ = params["write_back_to_player_seg"].get<bool>();
        if (params.count("rewrite_player_seg_cls")) r->rewrite_player_seg_cls_ = params["rewrite_player_seg_cls"].get<bool>();
        if (params.count("strip_input_metadata")) r->strip_input_metadata_ = params["strip_input_metadata"].get<bool>();
        if (params.count("tracker_fallback_enabled")) r->tracker_fallback_enabled_ = params["tracker_fallback_enabled"].get<bool>();
        if (params.count("tracker_fallback_max_age_frames")) r->tracker_fallback_max_age_frames_ = params["tracker_fallback_max_age_frames"].get<uint64_t>();
        if (params.count("tracker_fallback_min_margin")) r->tracker_fallback_min_margin_ = params["tracker_fallback_min_margin"].get<float>();
        if (params.count("tracker_fallback_min_confidence")) r->tracker_fallback_min_confidence_ = params["tracker_fallback_min_confidence"].get<float>();
        if (params.count("debug_log_every_n")) r->debug_log_every_n_ = params["debug_log_every_n"].get<int>();

        if (r->bootstrap_min_samples_ < 2) r->bootstrap_min_samples_ = 2;
        if (r->bootstrap_min_cluster_size_ < 1) r->bootstrap_min_cluster_size_ = 1;
        if (r->max_bootstrap_samples_ < r->bootstrap_min_samples_) {
            r->max_bootstrap_samples_ = r->bootstrap_min_samples_;
        }
        return r;
    }
};

DECLNODE(torso_team_classifier, TorsoTeamClassifier)
