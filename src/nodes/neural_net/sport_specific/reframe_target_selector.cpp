#include "../../node_common.hpp"

extern "C" {
#include <libavutil/dict.h>
}

#include "reframe_metadata_utils.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

class ReframeTargetSelector : public NodeSISO<av::VideoFrame, av::VideoFrame>, public ReportsFinishByFlag {
    using DetectionBox = avp_sport_reframe::DetectionBox;

    std::string ball_metadata_key_ = "yolo_ball";
    std::string handler_metadata_key_ = "ball_handler";
    std::string probable_handler_metadata_key_ = "probable_ball_handler";
    std::string scene_diff_metadata_key_ = "scene_diff";
    std::string camera_shot_metadata_key_;
    std::string output_metadata_key_ = "reframe_targets";

    std::string ball_label_ = "basketball";
    std::string handler_label_ = "BallHandler";
    std::string probable_handler_label_ = "ProbableBallHandler";

    double min_ball_conf_ = 0.04;
    double min_handler_conf_ = 0.0;
    double min_probable_conf_ = 0.02;
    double scene_diff_mean_abs_reset_ = 60.0;
    double hold_conf_decay_ = 0.92;
    double ball_target_weight_ = 0.75;
    double handler_target_weight_ = 1.20;
    double probable_target_weight_ = 0.80;
    double min_context_score_ = 0.02;
    int hold_last_target_frames_ = 12;
    int max_context_targets_ = 1;
    bool include_context_with_primary_ = true;
    bool require_wide_shot_ = true;
    int debug_log_every_n_ = 0;

    uint64_t frame_counter_ = 0;
    uint64_t stat_ball_ = 0;
    uint64_t stat_handler_ = 0;
    uint64_t stat_probable_ = 0;
    uint64_t stat_context_ = 0;
    uint64_t stat_held_ = 0;
    uint64_t stat_none_ = 0;
    uint64_t stat_resets_ = 0;
    std::vector<DetectionBox> held_targets_;
    std::vector<std::string> held_sources_;
    int held_age_ = 0;

    struct TargetCandidate {
        DetectionBox box;
        std::string source;
        double source_weight = 1.0;
        double weighted_score = 0.0;
    };

    static bool bestByConf(const DetectionBox& a, const DetectionBox& b) {
        if (a.conf != b.conf) return a.conf < b.conf;
        return a.track_id > b.track_id;
    }

    bool pickBestLabel(const AVFrame* raw,
                       const std::string& key,
                       const std::string& label,
                       double min_conf,
                       double& model_w,
                       double& model_h,
                       DetectionBox& out) const {
        std::vector<DetectionBox> dets;
        if (!avp_sport_reframe::parseDetections(raw, key, dets, model_w, model_h)) return false;
        bool found = false;
        DetectionBox best;
        best.conf = -1.0;
        for (const auto& d : dets) {
            if (!avp_sport_reframe::labelMatches(d, label)) continue;
            if (d.conf < min_conf) continue;
            if (!found || bestByConf(best, d)) {
                best = d;
                found = true;
            }
        }
        if (!found) return false;
        out = best;
        return true;
    }

    TargetCandidate makeCandidate(const DetectionBox& box,
                                  const std::string& source,
                                  double source_weight) const {
        TargetCandidate c;
        c.box = box;
        c.source = source;
        c.source_weight = std::max(0.0, source_weight);
        c.weighted_score = box.conf * c.source_weight;
        return c;
    }

    static bool betterCandidate(const TargetCandidate& a, const TargetCandidate& b) {
        if (a.weighted_score != b.weighted_score) return a.weighted_score > b.weighted_score;
        if (a.box.conf != b.box.conf) return a.box.conf > b.box.conf;
        return a.source < b.source;
    }

    static bool sameTrackOrOverlap(const DetectionBox& a, const DetectionBox& b) {
        if (a.has_track_id && b.has_track_id && a.track_id == b.track_id && a.label == b.label) return true;
        const double ix1 = std::max(a.x1, b.x1);
        const double iy1 = std::max(a.y1, b.y1);
        const double ix2 = std::min(a.x2, b.x2);
        const double iy2 = std::min(a.y2, b.y2);
        const double iw = std::max(0.0, ix2 - ix1);
        const double ih = std::max(0.0, iy2 - iy1);
        const double inter = iw * ih;
        if (!(inter > 0.0)) return false;
        const double aa = std::max(0.0, a.x2 - a.x1) * std::max(0.0, a.y2 - a.y1);
        const double ba = std::max(0.0, b.x2 - b.x1) * std::max(0.0, b.y2 - b.y1);
        const double uni = aa + ba - inter;
        return uni > 0.0 && inter / uni >= 0.80;
    }

    std::vector<TargetCandidate> selectWeightedTargets(const std::vector<TargetCandidate>& candidates,
                                                       std::string& selected_source) const {
        std::vector<TargetCandidate> sorted;
        for (const auto& c : candidates) {
            if (c.weighted_score > 0.0 && std::isfinite(c.weighted_score)) sorted.push_back(c);
        }
        std::sort(sorted.begin(), sorted.end(), betterCandidate);
        if (sorted.empty()) return {};

        std::vector<TargetCandidate> out;
        out.push_back(sorted[0]);
        selected_source = sorted[0].source;

        if (!include_context_with_primary_ || max_context_targets_ <= 0) return out;
        for (size_t i = 1; i < sorted.size() && (int)out.size() - 1 < max_context_targets_; ++i) {
            if (sorted[i].weighted_score < min_context_score_) continue;
            bool duplicate = false;
            for (const auto& selected : out) {
                if (sameTrackOrOverlap(selected.box, sorted[i].box)) {
                    duplicate = true;
                    break;
                }
            }
            if (!duplicate) out.push_back(sorted[i]);
        }
        return out;
    }

    double sourceWeight(const std::string& source) const {
        std::string src = source;
        const std::string held_prefix = "held_";
        if (src.rfind(held_prefix, 0) == 0) src = src.substr(held_prefix.size());
        if (src == "ball") return ball_target_weight_;
        if (src == "confirmed_handler") return handler_target_weight_;
        if (src == "probable_handler") return probable_target_weight_;
        return 1.0;
    }

    void writeOutput(av::VideoFrame& frm,
                     double model_w,
                     double model_h,
                     const std::vector<DetectionBox>& selected,
                     const std::string& selected_source,
                     const std::vector<std::string>& detection_sources,
                     bool reset) {
        Parameters out = avp_sport_reframe::emptyYoloMetadata(model_w, model_h);
        out["strategy"] = "weighted_source_with_context_and_hold";
        out["selected_source"] = selected_source;
        out["target_count"] = selected.size();
        out["held_age_frames"] = held_age_;
        out["ball_target_weight"] = ball_target_weight_;
        out["handler_target_weight"] = handler_target_weight_;
        out["probable_target_weight"] = probable_target_weight_;
        out["reset"] = reset;
        for (size_t i = 0; i < selected.size(); ++i) {
            const DetectionBox& d = selected[i];
            Parameters det;
            det["label"] = d.label;
            det["conf"] = d.conf;
            det["xyxy"] = {d.x1, d.y1, d.x2, d.y2};
            if (d.has_track_id) det["track_id"] = d.track_id;
            if (d.cls >= 0) det["cls"] = d.cls;
            if (d.model_index >= 0) det["model_index"] = d.model_index;
            det["source"] = i < detection_sources.size() ? detection_sources[i] : selected_source;
            det["target_role"] = i == 0 ? "primary" : "context";
            if (i < detection_sources.size()) {
                const std::string& src = detection_sources[i];
                const double weight = sourceWeight(src);
                det["source_weight"] = weight;
                det["weighted_score"] = d.conf * weight;
            }
            out["detections"].push_back(det);
        }
        av_dict_set(&frm.raw()->metadata, output_metadata_key_.c_str(), out.dump().c_str(), 0);
    }

    void rememberTargets(const std::vector<DetectionBox>& targets,
                         const std::vector<std::string>& sources) {
        held_targets_ = targets;
        held_sources_ = sources;
        held_age_ = 0;
    }

    bool heldTargets(std::vector<DetectionBox>& targets,
                     std::vector<std::string>& sources) {
        if (hold_last_target_frames_ <= 0 || held_targets_.empty() || held_age_ >= hold_last_target_frames_) {
            return false;
        }
        ++held_age_;
        const double decay = std::pow(hold_conf_decay_, std::max(0, held_age_));
        targets = held_targets_;
        sources.clear();
        for (size_t i = 0; i < targets.size(); ++i) {
            targets[i].conf = avp_sport_reframe::clampDouble(targets[i].conf * decay, 0.0, 1.0);
            const std::string src = i < held_sources_.size() ? held_sources_[i] : std::string("unknown");
            sources.push_back("held_" + src);
        }
        return true;
    }

    void clearHeldTargets() {
        held_targets_.clear();
        held_sources_.clear();
        held_age_ = 0;
    }

public:
    using NodeSISO<av::VideoFrame, av::VideoFrame>::NodeSISO;

    bool consumeEofIfPresent() override {
        return false;
    }

    ~ReframeTargetSelector() {
        if (frame_counter_ == 0) return;
        logstream << "reframe_target_selector: === summary ===";
        logstream << "  total frames:       " << frame_counter_;
        logstream << "  selected ball:      " << stat_ball_;
        logstream << "  selected handler:   " << stat_handler_;
        logstream << "  selected probable:  " << stat_probable_;
        logstream << "  context targets:    " << stat_context_;
        logstream << "  held targets:       " << stat_held_;
        logstream << "  selected none:      " << stat_none_;
        logstream << "  resets:             " << stat_resets_;
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
        double model_w = 960.0;
        double model_h = 544.0;

        bool reset = false;
        if (raw && avp_sport_reframe::sceneResetRequested(raw, scene_diff_metadata_key_, scene_diff_mean_abs_reset_)) {
            reset = true;
        }
        if (raw && require_wide_shot_ &&
            !avp_sport_reframe::cameraShotIsWide(raw, camera_shot_metadata_key_, true)) {
            reset = true;
        }
        if (reset) ++stat_resets_;

        DetectionBox ball;
        DetectionBox handler;
        DetectionBox probable;
        const bool have_ball = pickBestLabel(raw, ball_metadata_key_, ball_label_, min_ball_conf_, model_w, model_h, ball);
        const bool have_handler = pickBestLabel(raw, handler_metadata_key_, handler_label_, min_handler_conf_, model_w, model_h, handler);
        const bool have_probable = pickBestLabel(raw, probable_handler_metadata_key_, probable_handler_label_, min_probable_conf_, model_w, model_h, probable);

        std::vector<DetectionBox> selected;
        std::vector<std::string> sources;
        std::string source = "none";
        if (reset) {
            clearHeldTargets();
            ++stat_none_;
        } else {
            std::vector<TargetCandidate> candidates;
            if (have_ball) candidates.push_back(makeCandidate(ball, "ball", ball_target_weight_));
            if (have_handler) candidates.push_back(makeCandidate(handler, "confirmed_handler", handler_target_weight_));
            if (have_probable) candidates.push_back(makeCandidate(probable, "probable_handler", probable_target_weight_));

            std::vector<TargetCandidate> weighted = selectWeightedTargets(candidates, source);
            for (const auto& c : weighted) {
                selected.push_back(c.box);
                sources.push_back(c.source);
            }
            if (!selected.empty()) {
                if (source == "ball") ++stat_ball_;
                else if (source == "confirmed_handler") ++stat_handler_;
                else if (source == "probable_handler") ++stat_probable_;
                if (selected.size() > 1) stat_context_ += selected.size() - 1;
                rememberTargets(selected, sources);
            } else {
                if (heldTargets(selected, sources)) {
                    source = "held_last_target";
                    ++stat_held_;
                } else {
                    ++stat_none_;
                }
            }
        }

        writeOutput(frm, model_w, model_h, selected, source, sources, reset);

        if (debug_log_every_n_ > 0 && (frame_counter_ % (uint64_t)debug_log_every_n_) == 0) {
            logstream << "reframe_target_selector: frame=" << frame_counter_
                      << " source=" << source
                      << " reset=" << (reset ? 1 : 0)
                      << " targets=" << selected.size()
                      << " held_age=" << held_age_
                      << " ball=" << (have_ball ? 1 : 0)
                      << " handler=" << (have_handler ? 1 : 0)
                      << " probable=" << (have_probable ? 1 : 0)
                      << " ball_score=" << (have_ball ? ball.conf * ball_target_weight_ : 0.0)
                      << " handler_score=" << (have_handler ? handler.conf * handler_target_weight_ : 0.0)
                      << " probable_score=" << (have_probable ? probable.conf * probable_target_weight_ : 0.0);
        }

        this->sink_->put(frm);
    }

    static std::shared_ptr<ReframeTargetSelector> create(NodeCreationInfo& nci) {
        EdgeManager& edges = nci.edges;
        const Parameters& params = nci.params;
        auto r = NodeSISO<av::VideoFrame, av::VideoFrame>::template createCommon<ReframeTargetSelector>(edges, params);
        r->auto_eof_ = false;

        if (params.count("ball_metadata_key")) r->ball_metadata_key_ = params["ball_metadata_key"].get<std::string>();
        if (params.count("handler_metadata_key")) r->handler_metadata_key_ = params["handler_metadata_key"].get<std::string>();
        if (params.count("probable_handler_metadata_key")) r->probable_handler_metadata_key_ = params["probable_handler_metadata_key"].get<std::string>();
        if (params.count("scene_diff_metadata_key")) r->scene_diff_metadata_key_ = params["scene_diff_metadata_key"].get<std::string>();
        if (params.count("camera_shot_metadata_key")) r->camera_shot_metadata_key_ = params["camera_shot_metadata_key"].get<std::string>();
        if (params.count("output_metadata_key")) r->output_metadata_key_ = params["output_metadata_key"].get<std::string>();
        if (params.count("ball_label")) r->ball_label_ = params["ball_label"].get<std::string>();
        if (params.count("handler_label")) r->handler_label_ = params["handler_label"].get<std::string>();
        if (params.count("probable_handler_label")) r->probable_handler_label_ = params["probable_handler_label"].get<std::string>();
        if (params.count("min_ball_conf")) r->min_ball_conf_ = params["min_ball_conf"].get<double>();
        if (params.count("min_handler_conf")) r->min_handler_conf_ = params["min_handler_conf"].get<double>();
        if (params.count("min_probable_conf")) r->min_probable_conf_ = params["min_probable_conf"].get<double>();
        if (params.count("scene_diff_mean_abs_reset")) r->scene_diff_mean_abs_reset_ = params["scene_diff_mean_abs_reset"].get<double>();
        if (params.count("hold_conf_decay")) r->hold_conf_decay_ = params["hold_conf_decay"].get<double>();
        if (params.count("ball_target_weight")) r->ball_target_weight_ = params["ball_target_weight"].get<double>();
        if (params.count("handler_target_weight")) r->handler_target_weight_ = params["handler_target_weight"].get<double>();
        if (params.count("probable_target_weight")) r->probable_target_weight_ = params["probable_target_weight"].get<double>();
        if (params.count("min_context_score")) r->min_context_score_ = params["min_context_score"].get<double>();
        if (params.count("hold_last_target_frames")) r->hold_last_target_frames_ = params["hold_last_target_frames"].get<int>();
        if (params.count("max_context_targets")) r->max_context_targets_ = params["max_context_targets"].get<int>();
        if (params.count("include_context_with_primary")) r->include_context_with_primary_ = params["include_context_with_primary"].get<bool>();
        if (params.count("require_wide_shot")) r->require_wide_shot_ = params["require_wide_shot"].get<bool>();
        if (params.count("debug_log_every_n")) r->debug_log_every_n_ = params["debug_log_every_n"].get<int>();
        r->hold_conf_decay_ = avp_sport_reframe::clampDouble(r->hold_conf_decay_, 0.0, 1.0);
        r->ball_target_weight_ = std::max(0.0, r->ball_target_weight_);
        r->handler_target_weight_ = std::max(0.0, r->handler_target_weight_);
        r->probable_target_weight_ = std::max(0.0, r->probable_target_weight_);
        r->min_context_score_ = std::max(0.0, r->min_context_score_);
        r->hold_last_target_frames_ = std::max(0, r->hold_last_target_frames_);
        r->max_context_targets_ = std::max(0, r->max_context_targets_);
        return r;
    }
};

DECLNODE(reframe_target_selector, ReframeTargetSelector)
