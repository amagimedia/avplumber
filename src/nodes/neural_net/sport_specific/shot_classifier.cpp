#include "../../node_common.hpp"

extern "C" {
#include <libavutil/dict.h>
#include <libavutil/frame.h>
}

#include "../common/yolo_side_data.hpp"

#include <cmath>
#include <cstdint>
#include <string>
#include <unordered_set>
#include <vector>

class ShotClassifier : public NodeSISO<av::VideoFrame, av::VideoFrame>, public ReportsFinishByFlag {

    // Seg mask config
    std::string seg_metadata_key_ = "yolo_seg";
    int seg_side_data_slot_ = 0;
    float seg_mask_threshold_ = 0.5f;
    std::unordered_set<int> court_class_indices_ = {0, 1}; // "three point line" + "basketball-court"

    // Player detection config
    std::string player_metadata_key_ = "yolo_players";
    std::vector<std::string> player_labels_ = {"Player"};
    float player_min_conf_ = 0.25f;

    // Classification thresholds
    float wide_court_threshold_ = 0.25f;       // court coverage above this = definitely wide
    float closeup_court_threshold_ = 0.05f;    // court coverage below this = definitely closeup
    int ambiguous_min_players_ = 3;            // need this many valid players for wide
    int high_player_override_ = 7;             // this many players = wide regardless of court coverage
    float player_height_fraction_ = 0.25f;     // expected player height as fraction of frame height
    float player_height_tolerance_ = 0.45f;    // +/- tolerance on player height
    float player_min_aspect_ratio_ = 0.75f;    // min h/w to accept standing or slightly tilted players

    // Hysteresis
    int min_stable_frames_ = 6;
    int reuse_last_court_coverage_frames_ = 0;

    // Output
    std::string metadata_key_out_ = "camera_shot_info";
    int debug_log_every_n_ = 1;

    // State
    std::string prev_shot_type_;
    std::string candidate_type_;
    int candidate_count_ = 0;
    uint64_t frame_counter_ = 0;
    bool have_last_court_coverage_ = false;
    float last_court_coverage_ = 0.0f;
    uint64_t last_court_coverage_frame_ = 0;

    bool isValidWidePlayer(float bbox_h, float bbox_w, float frame_h) const {
        // Must be roughly upright, but allow slight camera tilt/perspective.
        if (bbox_w <= 0.0f || (bbox_h / bbox_w) < player_min_aspect_ratio_) return false;
        // Height should be ~player_height_fraction_ of frame, within tolerance
        float expected_h = frame_h * player_height_fraction_;
        float min_h = expected_h * (1.0f - player_height_tolerance_);
        float max_h = expected_h * (1.0f + player_height_tolerance_);
        return bbox_h >= min_h && bbox_h <= max_h;
    }

    int countValidWidePlayers(const Parameters& md, float frame_h) const {
        int count = 0;
        if (!md.contains("detections") || !md["detections"].is_array()) return 0;

        float model_w = md.value("model_width", 960.0);
        float model_h = md.value("model_height", 544.0);
        float scale_y = frame_h / model_h;

        for (const auto& det : md["detections"]) {
            if (!det.contains("xyxy") || !det["xyxy"].is_array() || det["xyxy"].size() < 4) continue;
            float conf = det.value("conf", 0.0);
            if (conf < player_min_conf_) continue;

            // Check label
            bool label_match = false;
            if (det.contains("label")) {
                std::string lbl = det["label"].get<std::string>();
                for (const auto& pl : player_labels_) {
                    if (lbl == pl) { label_match = true; break; }
                }
            }
            if (!label_match) continue;

            float y1 = det["xyxy"][1].get<float>();
            float x2 = det["xyxy"][2].get<float>();
            float y2 = det["xyxy"][3].get<float>();
            float x1 = det["xyxy"][0].get<float>();
            float bbox_w = (x2 - x1) * (frame_h / model_h); // scale to frame space
            float bbox_h = (y2 - y1) * scale_y;

            if (isValidWidePlayer(bbox_h, bbox_w, frame_h)) {
                count++;
            }
        }
        return count;
    }

public:
    using NodeSISO<av::VideoFrame, av::VideoFrame>::NodeSISO;
    bool consumeEofIfPresent() override {
        return false;
    }

    void process() override {
        av::VideoFrame frm = this->source_->get();

        if (isEofMarker(frm)) {
            prev_shot_type_.clear();
            candidate_type_.clear();
            candidate_count_ = 0;
            frame_counter_ = 0;
            have_last_court_coverage_ = false;
            last_court_coverage_ = 0.0f;
            last_court_coverage_frame_ = 0;
            this->sink_->put(frm);
            this->finished_ = true;
            return;
        }
        if (!frm) return;

        ++frame_counter_;

        const AVFrame* raw = frm.raw();
        if (!raw) {
            this->sink_->put(frm);
            return;
        }

        float frame_h = (float)frm.height();

        // === Step 1: Compute court coverage from seg mask ===
        std::vector<int> court_mask_indices;
        bool have_seg_metadata = false;
        if (raw->metadata) {
            AVDictionaryEntry* entry = av_dict_get(raw->metadata, seg_metadata_key_.c_str(), nullptr, 0);
            if (entry && entry->value) {
                have_seg_metadata = true;
                try {
                    Parameters seg_md = Parameters::parse(entry->value);
                    if (seg_md.contains("detections") && seg_md["detections"].is_array()) {
                        int idx = 0;
                        for (const auto& det : seg_md["detections"]) {
                            if (det.contains("cls")) {
                                int cls = det["cls"].get<int>();
                                if (court_class_indices_.count(cls)) {
                                    court_mask_indices.push_back(idx);
                                }
                            }
                            idx++;
                        }
                    }
                } catch (...) {}
            }
        }

        float court_coverage = 0.0f;
        bool have_current_court_coverage = false;
        bool reused_court_coverage = false;
        const AVFrameSideData* sd = av_frame_get_side_data(raw, yoloSegCpuSideDataType(seg_side_data_slot_));
        if (sd && sd->size >= 16 && !court_mask_indices.empty()) {
            const uint32_t* header = (const uint32_t*)sd->data;
            uint32_t num_masks = header[0];
            uint32_t mask_w = header[1];
            uint32_t mask_h = header[2];
            size_t pixels_per_mask = (size_t)mask_w * (size_t)mask_h;

            const float* mask_data = (const float*)(sd->data + 16);

            // Count pixels where ANY court class exceeds threshold (union, not sum).
            size_t court_pixels = 0;
            for (size_t p = 0; p < pixels_per_mask; p++) {
                for (int mi : court_mask_indices) {
                    if ((uint32_t)mi >= num_masks) continue;
                    if (mask_data[(size_t)mi * pixels_per_mask + p] >= seg_mask_threshold_) {
                        court_pixels++;
                        break;
                    }
                }
            }
            if (pixels_per_mask > 0) {
                court_coverage = (float)court_pixels / (float)pixels_per_mask;
                have_current_court_coverage = true;
            }
        }
        if (!have_current_court_coverage && have_seg_metadata) {
            // A present segmentation metadata key means the model ran but found no
            // usable court mask. Treat that as an authoritative zero.
            have_current_court_coverage = true;
        }
        if (have_current_court_coverage) {
            have_last_court_coverage_ = true;
            last_court_coverage_ = court_coverage;
            last_court_coverage_frame_ = frame_counter_;
        } else if (reuse_last_court_coverage_frames_ > 0 && have_last_court_coverage_
                   && frame_counter_ > last_court_coverage_frame_
                   && frame_counter_ - last_court_coverage_frame_ <= (uint64_t)reuse_last_court_coverage_frames_) {
            court_coverage = last_court_coverage_;
            reused_court_coverage = true;
        }

        // === Step 2: Always count valid wide-shot-sized players ===
        int valid_players = 0;
        {
            Parameters player_md;
            if (raw->metadata) {
                AVDictionaryEntry* entry = av_dict_get(raw->metadata, player_metadata_key_.c_str(), nullptr, 0);
                if (entry && entry->value) {
                    try {
                        player_md = Parameters::parse(entry->value);
                        valid_players = countValidWidePlayers(player_md, frame_h);
                    } catch (...) {}
                }
            }
        }

        // === Step 3: Classify ===
        // High player count overrides missing court (seg model failure on some angles)
        // Otherwise require both court coverage AND enough players
        std::string raw_type;
        bool enough_players = (valid_players >= ambiguous_min_players_);
        bool high_player_override = (valid_players >= high_player_override_);

        if (high_player_override) {
            raw_type = "wide";
        } else if (court_coverage <= closeup_court_threshold_ || !enough_players) {
            raw_type = "closeup";
        } else {
            raw_type = "wide";
        }

        // === Step 4: Hysteresis — require min_stable_frames_ before changing ===
        if (raw_type == candidate_type_) {
            candidate_count_++;
        } else {
            candidate_type_ = raw_type;
            candidate_count_ = 1;
        }

        std::string shot_type = prev_shot_type_.empty() ? raw_type : prev_shot_type_;
        bool transition = false;

        if (candidate_count_ >= min_stable_frames_ && candidate_type_ != prev_shot_type_) {
            shot_type = candidate_type_;
            transition = !prev_shot_type_.empty();
        }
        prev_shot_type_ = shot_type;

        // === Step 5: Write metadata ===
        Parameters out_md;
        out_md["camera_shot_type"] = shot_type;
        out_md["camera_shot_transition"] = transition;
        out_md["court_coverage"] = court_coverage;
        if (reused_court_coverage) {
            out_md["court_coverage_cached"] = true;
            out_md["court_coverage_age_frames"] = (int64_t)(frame_counter_ - last_court_coverage_frame_);
        }

        std::string serialized = out_md.dump();
        av_dict_set(&frm.raw()->metadata, metadata_key_out_.c_str(), serialized.c_str(), 0);

        if (debug_log_every_n_ > 0 && (frame_counter_ % (uint64_t)debug_log_every_n_) == 0) {
            logstream << "shot_classifier: frame=" << frame_counter_
                      << " court=" << (int)(court_coverage * 100) << "%"
                      << (reused_court_coverage ? " cached" : "")
                      << " players=" << valid_players
                      << " raw=" << raw_type
                      << " camera_shot=" << shot_type
                      << (transition ? " TRANSITION" : "");
        }

        this->sink_->put(frm);
    }

    static std::shared_ptr<ShotClassifier> create(NodeCreationInfo& nci) {
        EdgeManager& edges = nci.edges;
        const Parameters& params = nci.params;
        auto r = NodeSISO<av::VideoFrame, av::VideoFrame>::template createCommon<ShotClassifier>(edges, params);
        r->auto_eof_ = false;

        if (params.count("seg_metadata_key")) r->seg_metadata_key_ = params["seg_metadata_key"].get<std::string>();
        if (params.count("seg_side_data_slot")) {
            r->seg_side_data_slot_ = params["seg_side_data_slot"].get<int>();
            if (!yoloSegIsValidSlot(r->seg_side_data_slot_)) {
                throw Error("shot_classifier: seg_side_data_slot out of range [0," + std::to_string(kMaxYoloSegSlots - 1) + "]");
            }
        }
        if (params.count("player_metadata_key")) r->player_metadata_key_ = params["player_metadata_key"].get<std::string>();
        if (params.count("player_labels")) {
            r->player_labels_.clear();
            for (const auto& l : params["player_labels"]) r->player_labels_.push_back(l.get<std::string>());
        }
        if (params.count("player_min_conf")) r->player_min_conf_ = params["player_min_conf"];
        if (params.count("seg_mask_threshold")) r->seg_mask_threshold_ = params["seg_mask_threshold"];
        if (params.count("court_class_indices")) {
            r->court_class_indices_.clear();
            for (const auto& item : params["court_class_indices"]) r->court_class_indices_.insert(item.get<int>());
        }
        if (params.count("wide_court_threshold")) r->wide_court_threshold_ = params["wide_court_threshold"];
        if (params.count("closeup_court_threshold")) r->closeup_court_threshold_ = params["closeup_court_threshold"];
        if (params.count("ambiguous_min_players")) r->ambiguous_min_players_ = params["ambiguous_min_players"];
        if (params.count("high_player_override")) r->high_player_override_ = params["high_player_override"];
        if (params.count("player_height_fraction")) r->player_height_fraction_ = params["player_height_fraction"];
        if (params.count("player_height_tolerance")) r->player_height_tolerance_ = params["player_height_tolerance"];
        if (params.count("player_min_aspect_ratio")) r->player_min_aspect_ratio_ = params["player_min_aspect_ratio"];
        if (params.count("min_stable_frames")) r->min_stable_frames_ = params["min_stable_frames"];
        if (params.count("reuse_last_court_coverage_frames")) {
            r->reuse_last_court_coverage_frames_ = params["reuse_last_court_coverage_frames"];
        }
        if (params.count("metadata_key_out")) r->metadata_key_out_ = params["metadata_key_out"].get<std::string>();
        if (params.count("debug_log_every_n")) r->debug_log_every_n_ = params["debug_log_every_n"];

        return r;
    }
};

DECLNODE(shot_classifier, ShotClassifier)
