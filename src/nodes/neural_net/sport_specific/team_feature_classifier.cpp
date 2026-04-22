#include "../../node_common.hpp"
#include "../common/player_feature_side_data.hpp"

extern "C" {
#include <libavutil/dict.h>
}

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstring>
#include <limits>
#include <string>
#include <unordered_map>
#include <unordered_set>
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

bool matchesLabel(const Parameters& det, const std::vector<std::string>& labels) {
    if (labels.empty()) return true;
    if (!det.contains("label") || !det["label"].is_string()) return false;
    const std::string lbl = det["label"].get<std::string>();
    for (const auto& want : labels) {
        if (iequals(lbl, want)) return true;
    }
    return false;
}

bool parseBBox(const Parameters& det, float& x1, float& y1, float& x2, float& y2) {
    if (!det.contains("xyxy") || !det["xyxy"].is_array() || det["xyxy"].size() < 4) return false;
    x1 = det["xyxy"][0].get<float>();
    y1 = det["xyxy"][1].get<float>();
    x2 = det["xyxy"][2].get<float>();
    y2 = det["xyxy"][3].get<float>();
    return x2 > x1 && y2 > y1;
}

float l2Normalize(std::vector<float>& v) {
    float sum = 0.0f;
    for (float x : v) sum += x * x;
    if (sum <= 1e-12f) return 0.0f;
    const float inv = 1.0f / std::sqrt(sum);
    for (float& x : v) x *= inv;
    return 1.0f / inv;
}

float l2Normalize(float* v, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; ++i) sum += v[i] * v[i];
    if (sum <= 1e-12f) return 0.0f;
    const float inv = 1.0f / std::sqrt(sum);
    for (int i = 0; i < n; ++i) v[i] *= inv;
    return 1.0f / inv;
}

float dotProduct(const float* a, const float* b, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; ++i) sum += a[i] * b[i];
    return sum;
}

float cosineDistance(const float* a, const float* b, int n) {
    return 1.0f - dotProduct(a, b, n);
}

struct ParsedTracked {
    int det_index = -1;
    int track_id = -1;
    float x1 = 0.0f;
    float y1 = 0.0f;
    float x2 = 0.0f;
    float y2 = 0.0f;
};

struct ParsedSeg {
    int det_index = -1;
    float x1 = 0.0f;
    float y1 = 0.0f;
    float x2 = 0.0f;
    float y2 = 0.0f;
};

struct EmbedItem {
    int det_index = -1;
    int track_id = -1;
    int seg_index = -1;
    const float* vec = nullptr;
};

struct TrackState {
    std::vector<float> embed_ema;
    bool has_embed = false;
    int hits = 0;
    int assigned_team = -1;
    float last_margin = 0.0f;
    uint64_t last_frame = 0;
};

} // namespace

class TeamFeatureClassifier : public NodeSISO<av::VideoFrame, av::VideoFrame> {
    std::string player_metadata_key_ = "yolo_players";
    std::string seg_metadata_key_ = "yolo_players_seg";
    std::string shot_metadata_key_ = "shot_info";
    std::vector<std::string> player_labels_ = {"Player"};
    std::vector<std::string> seg_labels_ = {"player"};
    float iou_match_threshold_ = 0.15f;
    float fallback_center_distance_px_ = 60.0f;
    float ema_alpha_track_ = 0.2f;
    float ema_alpha_centroid_ = 0.05f;
    float assignment_margin_ = 0.05f;
    float soft_assignment_margin_ = 0.02f;
    float bootstrap_min_prototype_distance_ = 0.08f;
    int bootstrap_min_tracks_ = 6;
    int bootstrap_min_hits_ = 3;
    int bootstrap_min_cluster_size_ = 2;
    float bootstrap_max_ratio_ = 2.0f;
    uint64_t bootstrap_frames_ = 60;
    uint64_t track_idle_frames_ = 300;
    int centroid_update_min_tracks_per_team_ = 2;
    float centroid_update_max_ratio_ = 2.5f;
    bool centroid_update_wide_only_ = true;
    bool write_back_to_seg_ = true;
    bool rewrite_seg_cls_ = true;
    bool rewrite_label_ = false;
    std::string output_field_ = "team";
    std::string output_ab_field_ = "team_ab";
    std::vector<std::string> team_ab_ = {"A", "B"};
    std::vector<std::string> team_label_names_ = {"PlayerA", "PlayerB"};
    std::string unknown_label_ = "PlayerUnknown";
    int debug_log_every_n_ = 0;

    uint64_t frame_counter_ = 0;
    bool bootstrapped_ = false;
    int embed_dim_ = 0;
    std::unordered_map<int, TrackState> tracks_;
    std::vector<float> proto_[2];

    void resetState() {
        bootstrapped_ = false;
        embed_dim_ = 0;
        tracks_.clear();
        proto_[0].clear();
        proto_[1].clear();
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

    void parseTracked(const Parameters& player_md, std::vector<ParsedTracked>& tracked) const {
        tracked.clear();
        if (!player_md.contains("detections") || !player_md["detections"].is_array()) return;
        for (int i = 0; i < (int)player_md["detections"].size(); ++i) {
            const auto& det = player_md["detections"][i];
            if (!det.is_object() || !matchesLabel(det, player_labels_)) continue;
            ParsedTracked p;
            p.det_index = i;
            p.track_id = det.value("track_id", -1);
            if (p.track_id < 0) continue;
            if (!parseBBox(det, p.x1, p.y1, p.x2, p.y2)) continue;
            tracked.push_back(p);
        }
    }

    void parseSeg(const Parameters& seg_md, std::vector<ParsedSeg>& segs) const {
        segs.clear();
        if (!seg_md.contains("detections") || !seg_md["detections"].is_array()) return;
        for (int i = 0; i < (int)seg_md["detections"].size(); ++i) {
            const auto& det = seg_md["detections"][i];
            if (!det.is_object() || !matchesLabel(det, seg_labels_)) continue;
            ParsedSeg s;
            s.det_index = i;
            if (!parseBBox(det, s.x1, s.y1, s.x2, s.y2)) continue;
            segs.push_back(s);
        }
    }

    bool isWideShot(const AVFrame* raw) {
        if (!raw || !raw->metadata || shot_metadata_key_.empty()) return true;
        AVDictionaryEntry* shot_entry = av_dict_get(raw->metadata, shot_metadata_key_.c_str(), nullptr, 0);
        if (!shot_entry || !shot_entry->value) return true;
        try {
            Parameters shot_md = Parameters::parse(shot_entry->value);
            if (shot_md.value("shot_transition", false)) {
                resetState();
            }
            return iequals(shot_md.value("shot_type", std::string("wide")), "wide");
        } catch (...) {
            return true;
        }
    }

    bool parseFeatureSideData(const AVFrame* raw, std::vector<EmbedItem>& items, int& embed_dim) const {
        items.clear();
        embed_dim = 0;
        if (!raw) return false;
        const AVFrameSideData* sd = av_frame_get_side_data(raw, AV_FRAME_DATA_PLAYER_FEATURE);
        if (!sd || sd->size < (int)sizeof(PlayerFeatureSideDataHeader)) return false;
        const auto* header = (const PlayerFeatureSideDataHeader*)sd->data;
        const size_t item_offset = sizeof(PlayerFeatureSideDataHeader);
        const size_t items_bytes = (size_t)header->num_items * sizeof(PlayerFeatureSideDataItem);
        const size_t vec_offset = item_offset + items_bytes;
        const size_t vec_bytes = (size_t)header->num_items * (size_t)header->feature_dim * sizeof(float);
        if ((size_t)sd->size < vec_offset + vec_bytes) return false;
        const auto* item_ptr = (const PlayerFeatureSideDataItem*)(sd->data + item_offset);
        const float* vec_ptr = (const float*)(sd->data + vec_offset);
        embed_dim = (int)header->feature_dim;
        for (uint32_t i = 0; i < header->num_items; ++i) {
            EmbedItem item;
            item.det_index = item_ptr[i].det_index;
            item.track_id = item_ptr[i].track_id;
            item.seg_index = item_ptr[i].seg_index;
            item.vec = vec_ptr + (size_t)i * (size_t)header->feature_dim;
            items.push_back(item);
        }
        return !items.empty() && embed_dim > 0;
    }

    void updateTracks(const std::vector<EmbedItem>& items) {
        for (const auto& item : items) {
            if (item.track_id < 0 || !item.vec) continue;
            TrackState& st = tracks_[item.track_id];
            if ((int)st.embed_ema.size() != embed_dim_) st.embed_ema.assign((size_t)embed_dim_, 0.0f);
            if (!st.has_embed) {
                std::memcpy(st.embed_ema.data(), item.vec, (size_t)embed_dim_ * sizeof(float));
                l2Normalize(st.embed_ema);
                st.has_embed = true;
            } else {
                for (int i = 0; i < embed_dim_; ++i) {
                    st.embed_ema[(size_t)i] = (1.0f - ema_alpha_track_) * st.embed_ema[(size_t)i] + ema_alpha_track_ * item.vec[i];
                }
                l2Normalize(st.embed_ema);
            }
            st.hits += 1;
            st.last_frame = frame_counter_;
        }
    }

    bool bootstrapPrototypes(bool wide_only, const std::vector<ParsedTracked>& tracked) {
        if (bootstrapped_ || frame_counter_ < bootstrap_frames_ || !wide_only) return bootstrapped_;
        std::vector<const TrackState*> samples;
        samples.reserve(tracked.size());
        for (const auto& tr : tracked) {
            auto it = tracks_.find(tr.track_id);
            if (it == tracks_.end()) continue;
            const TrackState& st = it->second;
            if (!st.has_embed || st.hits < bootstrap_min_hits_) continue;
            samples.push_back(&st);
        }
        if ((int)samples.size() < bootstrap_min_tracks_) return false;

        proto_[0] = samples.front()->embed_ema;
        float best_d = -1.0f;
        size_t best_i = 0;
        for (size_t i = 0; i < samples.size(); ++i) {
            const float d = cosineDistance(samples[i]->embed_ema.data(), proto_[0].data(), embed_dim_);
            if (d > best_d) {
                best_d = d;
                best_i = i;
            }
        }
        proto_[1] = samples[best_i]->embed_ema;

        int final_count0 = 0;
        int final_count1 = 0;
        for (int iter = 0; iter < 12; ++iter) {
            std::vector<float> sum0((size_t)embed_dim_, 0.0f);
            std::vector<float> sum1((size_t)embed_dim_, 0.0f);
            int count0 = 0;
            int count1 = 0;
            for (const TrackState* st : samples) {
                const float d0 = cosineDistance(st->embed_ema.data(), proto_[0].data(), embed_dim_);
                const float d1 = cosineDistance(st->embed_ema.data(), proto_[1].data(), embed_dim_);
                std::vector<float>& sum = (d0 <= d1) ? sum0 : sum1;
                int& count = (d0 <= d1) ? count0 : count1;
                for (int i = 0; i < embed_dim_; ++i) sum[(size_t)i] += st->embed_ema[(size_t)i];
                count += 1;
            }
            if (count0 > 0) {
                for (int i = 0; i < embed_dim_; ++i) proto_[0][(size_t)i] = sum0[(size_t)i] / (float)count0;
                l2Normalize(proto_[0]);
            }
            if (count1 > 0) {
                for (int i = 0; i < embed_dim_; ++i) proto_[1][(size_t)i] = sum1[(size_t)i] / (float)count1;
                l2Normalize(proto_[1]);
            }
            final_count0 = count0;
            final_count1 = count1;
        }

        const float sep = cosineDistance(proto_[0].data(), proto_[1].data(), embed_dim_);
        if (final_count0 < bootstrap_min_cluster_size_ || final_count1 < bootstrap_min_cluster_size_) return false;
        const float min_count = (float)std::min(final_count0, final_count1);
        const float max_count = (float)std::max(final_count0, final_count1);
        if (min_count <= 0.0f || max_count / min_count > bootstrap_max_ratio_) return false;
        if (sep < bootstrap_min_prototype_distance_) return false;
        bootstrapped_ = true;
        logstream << "team_feature_classifier: bootstrap"
                  << " frame=" << frame_counter_
                  << " tracks=" << samples.size()
                  << " c0=" << final_count0
                  << " c1=" << final_count1
                  << " proto_sep=" << sep;
        return true;
    }

    void assignTeams(Parameters& player_md, const std::vector<ParsedTracked>& tracked) {
        for (auto& det : player_md["detections"]) {
            if (!det.is_object()) continue;
            det[output_field_] = -1;
            det[output_ab_field_] = "?";
        }
        if (!bootstrapped_) return;

        for (const auto& tr : tracked) {
            auto it = tracks_.find(tr.track_id);
            if (it == tracks_.end() || !it->second.has_embed) continue;
            TrackState& st = it->second;
            const float d0 = cosineDistance(st.embed_ema.data(), proto_[0].data(), embed_dim_);
            const float d1 = cosineDistance(st.embed_ema.data(), proto_[1].data(), embed_dim_);
            const int best = (d0 <= d1) ? 0 : 1;
            const float margin = std::fabs(d0 - d1);
            if (margin >= assignment_margin_ || (st.assigned_team == best && margin >= soft_assignment_margin_)) {
                st.assigned_team = best;
                st.last_margin = margin;
            }
            const int team = st.assigned_team;
            auto& det = player_md["detections"][tr.det_index];
            det[output_field_] = team;
            det[output_ab_field_] = (team >= 0 && team < (int)team_ab_.size()) ? team_ab_[(size_t)team] : "?";
            if (team >= 0) {
                det["team_conf"] = margin;
                if (rewrite_label_ && team < (int)team_label_names_.size()) {
                    det["label"] = team_label_names_[(size_t)team];
                }
            }
        }
    }

    void updateCentroidsFromAssigned(const std::vector<EmbedItem>& items, bool wide) {
        if (!bootstrapped_ || ema_alpha_centroid_ <= 0.0f) return;
        if (centroid_update_wide_only_ && !wide) return;
        std::unordered_set<int> visible_tracks;
        visible_tracks.reserve(items.size());
        for (const auto& item : items) {
            if (item.track_id >= 0) visible_tracks.insert(item.track_id);
        }
        if (visible_tracks.empty()) return;

        std::vector<float> sum0((size_t)embed_dim_, 0.0f);
        std::vector<float> sum1((size_t)embed_dim_, 0.0f);
        int count0 = 0;
        int count1 = 0;
        for (auto& kv : tracks_) {
            TrackState& st = kv.second;
            if (!visible_tracks.count(kv.first)) continue;
            if (!st.has_embed || st.assigned_team < 0 || st.last_margin < assignment_margin_) continue;
            std::vector<float>& sum = (st.assigned_team == 0) ? sum0 : sum1;
            int& count = (st.assigned_team == 0) ? count0 : count1;
            for (int i = 0; i < embed_dim_; ++i) sum[(size_t)i] += st.embed_ema[(size_t)i];
            count += 1;
        }
        if (count0 < centroid_update_min_tracks_per_team_ || count1 < centroid_update_min_tracks_per_team_) return;

        const float min_count = (float)std::min(count0, count1);
        const float max_count = (float)std::max(count0, count1);
        if (min_count <= 0.0f || max_count / min_count > centroid_update_max_ratio_) return;

        std::vector<float> mean0((size_t)embed_dim_, 0.0f);
        std::vector<float> mean1((size_t)embed_dim_, 0.0f);
        for (int i = 0; i < embed_dim_; ++i) {
            mean0[(size_t)i] = sum0[(size_t)i] / (float)count0;
            mean1[(size_t)i] = sum1[(size_t)i] / (float)count1;
        }
        l2Normalize(mean0);
        l2Normalize(mean1);

        std::vector<float> next0 = proto_[0];
        std::vector<float> next1 = proto_[1];
        for (int i = 0; i < embed_dim_; ++i) {
            next0[(size_t)i] = (1.0f - ema_alpha_centroid_) * next0[(size_t)i] + ema_alpha_centroid_ * mean0[(size_t)i];
            next1[(size_t)i] = (1.0f - ema_alpha_centroid_) * next1[(size_t)i] + ema_alpha_centroid_ * mean1[(size_t)i];
        }
        l2Normalize(next0);
        l2Normalize(next1);
        if (cosineDistance(next0.data(), next1.data(), embed_dim_) < bootstrap_min_prototype_distance_) return;

        proto_[0].swap(next0);
        proto_[1].swap(next1);
    }

    void writeSegTeams(const std::vector<ParsedTracked>& tracked,
                       Parameters& player_md,
                       Parameters& seg_md) const {
        if (!write_back_to_seg_) return;
        std::vector<ParsedSeg> segs;
        parseSeg(seg_md, segs);
        std::vector<char> used(segs.size(), 0);
        for (const auto& tr : tracked) {
            const auto& player_det = player_md["detections"][tr.det_index];
            const int team = player_det.value(output_field_, -1);
            int best_idx = -1;
            float best_iou = 0.0f;
            for (int si = 0; si < (int)segs.size(); ++si) {
                if (used[(size_t)si]) continue;
                const auto& sg = segs[(size_t)si];
                const float iou = bboxIoU(tr.x1, tr.y1, tr.x2, tr.y2, sg.x1, sg.y1, sg.x2, sg.y2);
                if (iou >= iou_match_threshold_ && iou > best_iou) {
                    best_iou = iou;
                    best_idx = si;
                }
            }
            if (best_idx < 0) {
                float best_center = fallback_center_distance_px_;
                for (int si = 0; si < (int)segs.size(); ++si) {
                    if (used[(size_t)si]) continue;
                    const auto& sg = segs[(size_t)si];
                    const float center = centerDistance(tr.x1, tr.y1, tr.x2, tr.y2, sg.x1, sg.y1, sg.x2, sg.y2);
                    if (center <= best_center) {
                        best_center = center;
                        best_idx = si;
                    }
                }
            }
            if (best_idx < 0) continue;
            used[(size_t)best_idx] = 1;
            auto& seg_det = seg_md["detections"][segs[(size_t)best_idx].det_index];
            seg_det[output_field_] = team;
            seg_det[output_ab_field_] = player_det.value(output_ab_field_, std::string("?"));
            if (rewrite_seg_cls_) seg_det["cls"] = team;
            if (rewrite_label_) {
                if (team >= 0 && team < (int)team_label_names_.size()) seg_det["label"] = team_label_names_[(size_t)team];
                else seg_det["label"] = unknown_label_;
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
        if (!raw || !raw->metadata) {
            this->sink_->put(frm);
            return;
        }

        AVDictionaryEntry* player_entry = av_dict_get(raw->metadata, player_metadata_key_.c_str(), nullptr, 0);
        AVDictionaryEntry* seg_entry = av_dict_get(raw->metadata, seg_metadata_key_.c_str(), nullptr, 0);
        if (!player_entry || !player_entry->value || !seg_entry || !seg_entry->value) {
            this->sink_->put(frm);
            return;
        }

        Parameters player_md;
        Parameters seg_md;
        try {
            player_md = Parameters::parse(player_entry->value);
            seg_md = Parameters::parse(seg_entry->value);
        } catch (...) {
            this->sink_->put(frm);
            return;
        }

        std::vector<ParsedTracked> tracked;
        parseTracked(player_md, tracked);
        const bool wide = isWideShot(raw);

        std::vector<EmbedItem> embed_items;
        int embed_dim = 0;
        if (parseFeatureSideData(raw, embed_items, embed_dim)) {
            if (embed_dim_ == 0) embed_dim_ = embed_dim;
            if (embed_dim == embed_dim_) updateTracks(embed_items);
        }

        pruneTracks();
        bootstrapPrototypes(wide, tracked);
        assignTeams(player_md, tracked);
        updateCentroidsFromAssigned(embed_items, wide);
        writeSegTeams(tracked, player_md, seg_md);

        av_dict_set(&frm.raw()->metadata, player_metadata_key_.c_str(), player_md.dump().c_str(), 0);
        av_dict_set(&frm.raw()->metadata, seg_metadata_key_.c_str(), seg_md.dump().c_str(), 0);

        if (debug_log_every_n_ > 0 && (frame_counter_ % (uint64_t)debug_log_every_n_) == 0) {
            int known = 0;
            int team0 = 0;
            int team1 = 0;
            for (const auto& tr : tracked) {
                const auto& det = player_md["detections"][tr.det_index];
                const int team = det.value(output_field_, -1);
                if (team >= 0) known += 1;
                if (team == 0) team0 += 1;
                if (team == 1) team1 += 1;
            }
            logstream << "team_feature_classifier: frame=" << frame_counter_
                      << " wide=" << (wide ? 1 : 0)
                      << " tracked=" << tracked.size()
                      << " embeds=" << embed_items.size()
                      << " bootstrapped=" << (bootstrapped_ ? 1 : 0)
                      << " assigned_known=" << known
                      << " assigned_t0=" << team0
                      << " assigned_t1=" << team1;
        }

        this->sink_->put(frm);
    }

    static std::shared_ptr<TeamFeatureClassifier> create(NodeCreationInfo& nci) {
        EdgeManager& edges = nci.edges;
        const Parameters& params = nci.params;
        std::shared_ptr<Edge<av::VideoFrame>> src = edges.find<av::VideoFrame>(params["src"]);
        std::shared_ptr<Edge<av::VideoFrame>> dst = edges.find<av::VideoFrame>(params["dst"]);
        auto r = std::make_shared<TeamFeatureClassifier>(src->makeSource(), dst->makeSink());
        if (params.count("player_metadata_key")) r->player_metadata_key_ = params["player_metadata_key"].get<std::string>();
        if (params.count("seg_metadata_key")) r->seg_metadata_key_ = params["seg_metadata_key"].get<std::string>();
        if (params.count("shot_metadata_key")) r->shot_metadata_key_ = params["shot_metadata_key"].get<std::string>();
        if (params.count("player_labels")) {
            auto labels = jsonToStringList(params["player_labels"]);
            r->player_labels_.assign(labels.begin(), labels.end());
        }
        if (params.count("seg_labels")) {
            auto labels = jsonToStringList(params["seg_labels"]);
            r->seg_labels_.assign(labels.begin(), labels.end());
        }
        if (params.count("iou_match_threshold")) r->iou_match_threshold_ = params["iou_match_threshold"].get<float>();
        if (params.count("fallback_center_distance_px")) r->fallback_center_distance_px_ = params["fallback_center_distance_px"].get<float>();
        if (params.count("ema_alpha_track")) r->ema_alpha_track_ = params["ema_alpha_track"].get<float>();
        if (params.count("ema_alpha_centroid")) r->ema_alpha_centroid_ = params["ema_alpha_centroid"].get<float>();
        if (params.count("assignment_margin")) r->assignment_margin_ = params["assignment_margin"].get<float>();
        if (params.count("soft_assignment_margin")) r->soft_assignment_margin_ = params["soft_assignment_margin"].get<float>();
        if (params.count("bootstrap_min_prototype_distance")) r->bootstrap_min_prototype_distance_ = params["bootstrap_min_prototype_distance"].get<float>();
        if (params.count("bootstrap_min_tracks")) r->bootstrap_min_tracks_ = params["bootstrap_min_tracks"].get<int>();
        if (params.count("bootstrap_min_hits")) r->bootstrap_min_hits_ = params["bootstrap_min_hits"].get<int>();
        if (params.count("bootstrap_min_cluster_size")) r->bootstrap_min_cluster_size_ = params["bootstrap_min_cluster_size"].get<int>();
        if (params.count("bootstrap_max_ratio")) r->bootstrap_max_ratio_ = params["bootstrap_max_ratio"].get<float>();
        if (params.count("bootstrap_frames")) r->bootstrap_frames_ = params["bootstrap_frames"].get<uint64_t>();
        if (params.count("track_idle_frames")) r->track_idle_frames_ = params["track_idle_frames"].get<uint64_t>();
        if (params.count("centroid_update_min_tracks_per_team")) r->centroid_update_min_tracks_per_team_ = params["centroid_update_min_tracks_per_team"].get<int>();
        if (params.count("centroid_update_max_ratio")) r->centroid_update_max_ratio_ = params["centroid_update_max_ratio"].get<float>();
        if (params.count("centroid_update_wide_only")) r->centroid_update_wide_only_ = params["centroid_update_wide_only"].get<bool>();
        if (params.count("write_back_to_seg")) r->write_back_to_seg_ = params["write_back_to_seg"].get<bool>();
        if (params.count("rewrite_seg_cls")) r->rewrite_seg_cls_ = params["rewrite_seg_cls"].get<bool>();
        if (params.count("rewrite_label")) r->rewrite_label_ = params["rewrite_label"].get<bool>();
        if (params.count("output_field")) r->output_field_ = params["output_field"].get<std::string>();
        if (params.count("output_ab_field")) r->output_ab_field_ = params["output_ab_field"].get<std::string>();
        if (params.count("debug_log_every_n")) r->debug_log_every_n_ = params["debug_log_every_n"].get<int>();
        return r;
    }
};

DECLNODE(team_feature_classifier, TeamFeatureClassifier)
