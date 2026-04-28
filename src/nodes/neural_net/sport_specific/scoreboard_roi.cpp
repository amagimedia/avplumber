#include "../../node_common.hpp"

extern "C" {
#include <libavutil/dict.h>
}

#include <algorithm>
#include <string>
#include <vector>

namespace {

struct ScoreboardRoiField {
    std::string name;
    std::string label;
    std::string kind;
    float x = 0.0f;
    float y = 0.0f;
    float w = 0.0f;
    float h = 0.0f;
    bool team_probe = false;
};

static float valueFloat(const Parameters& obj, const char* key, float def) {
    return obj.contains(key) ? obj[key].get<float>() : def;
}

} // namespace

class ScoreboardRoi : public NodeSISO<av::VideoFrame, av::VideoFrame>, public ReportsFinishByFlag {
    std::string metadata_key_out_ = "scoreboard_rois";
    std::string preset_ = "balanced";
    float lower_third_x_rel_ = 0.0f;
    float lower_third_y_rel_ = 0.908f;
    float lower_third_w_rel_ = 1.0f;
    float lower_third_h_rel_ = 0.075f;
    int team_probe_frames_ = 150;
    int debug_log_every_n_ = 0;
    uint64_t frame_counter_ = 0;
    std::vector<ScoreboardRoiField> fields_;

    void setBalancedPreset() {
        fields_ = {
            {"team_a_name", "team_1", "Team Name", 0.030f, 0.05f, 0.095f, 0.86f, true},
            {"team_b_name", "team_2", "Team Name", 0.196f, 0.05f, 0.105f, 0.86f, true},
            {"team_a_score", "score_1", "Team Points", 0.140f, 0.05f, 0.052f, 0.86f, false},
            {"team_b_score", "score_2", "Team Points", 0.313f, 0.05f, 0.050f, 0.86f, false},
            {"period", "quarter", "Period", 0.366f, 0.08f, 0.040f, 0.78f, false},
            {"game_clock", "time", "Time Remaining", 0.403f, 0.08f, 0.066f, 0.78f, false},
            {"shot_clock", "countdown", "Shot Clock", 0.470f, 0.08f, 0.036f, 0.78f, false},
        };
        lower_third_x_rel_ = 0.0f;
        lower_third_y_rel_ = 0.908f;
        lower_third_w_rel_ = 1.0f;
        lower_third_h_rel_ = 0.075f;
        team_probe_frames_ = 150;
    }

    void applyPreset(const std::string& preset) {
        preset_ = preset;
        if (preset == "balanced") {
            setBalancedPreset();
        } else if (preset == "lite") {
            setBalancedPreset();
            team_probe_frames_ = 0;
            fields_.erase(std::remove_if(fields_.begin(), fields_.end(),
                                         [](const ScoreboardRoiField& f) { return f.team_probe; }),
                          fields_.end());
        } else if (preset == "debug") {
            setBalancedPreset();
            team_probe_frames_ = 300;
        } else if (preset == "heavy") {
            setBalancedPreset();
            team_probe_frames_ = 450;
        }
    }

    bool includeField(const ScoreboardRoiField& field) const {
        if (!field.team_probe) return true;
        if (team_probe_frames_ == 0) return false;
        return frame_counter_ <= (uint64_t)team_probe_frames_;
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
        AVFrame* raw = frm.raw();
        if (!raw) {
            this->sink_->put(frm);
            return;
        }

        const int frame_w = frm.width();
        const int frame_h = frm.height();
        const float lx = std::clamp(lower_third_x_rel_, 0.0f, 1.0f);
        const float ly = std::clamp(lower_third_y_rel_, 0.0f, 1.0f);
        const float lw = std::clamp(lower_third_w_rel_, 0.0f, 1.0f - lx);
        const float lh = std::clamp(lower_third_h_rel_, 0.0f, 1.0f - ly);

        Parameters out;
        out["schema"] = "scoreboard_roi_v1";
        out["source"] = "fixed_lower_third";
        out["preset"] = preset_;
        out["frame"] = frame_counter_;
        out["model_width"] = frame_w;
        out["model_height"] = frame_h;
        out["coordinate_space"] = "frame";
        out["detections"] = Parameters::array();

        if (frame_w > 0 && frame_h > 0 && lw > 0.0f && lh > 0.0f) {
            out["lower_third_xyxy"] = {
                lx * (float)frame_w,
                ly * (float)frame_h,
                (lx + lw) * (float)frame_w,
                (ly + lh) * (float)frame_h,
            };

            int cls = 0;
            for (const auto& field : fields_) {
                if (!includeField(field) || field.w <= 0.0f || field.h <= 0.0f) {
                    ++cls;
                    continue;
                }
                const float fx1 = std::clamp(lx + field.x * lw, 0.0f, 1.0f);
                const float fy1 = std::clamp(ly + field.y * lh, 0.0f, 1.0f);
                const float fx2 = std::clamp(lx + (field.x + field.w) * lw, 0.0f, 1.0f);
                const float fy2 = std::clamp(ly + (field.y + field.h) * lh, 0.0f, 1.0f);
                if (fx2 <= fx1 || fy2 <= fy1) {
                    ++cls;
                    continue;
                }

                Parameters det;
                det["cls"] = cls;
                det["conf"] = 1.0f;
                det["label"] = field.label;
                det["kind"] = field.kind;
                det["field"] = field.name;
                det["team_probe"] = field.team_probe;
                det["xyxy"] = {
                    fx1 * (float)frame_w,
                    fy1 * (float)frame_h,
                    fx2 * (float)frame_w,
                    fy2 * (float)frame_h,
                };
                out["detections"].push_back(std::move(det));
                ++cls;
            }
        }

        std::string serialized = out.dump();
        av_dict_set(&raw->metadata, metadata_key_out_.c_str(), serialized.c_str(), 0);

        if (debug_log_every_n_ > 0 && (frame_counter_ % (uint64_t)debug_log_every_n_) == 0) {
            logstream << "scoreboard_roi: frame=" << frame_counter_
                      << " preset=" << preset_
                      << " detections=" << out["detections"].size()
                      << " model=" << frame_w << "x" << frame_h
                      << " key=" << metadata_key_out_;
        }

        this->sink_->put(frm);
    }

    static std::shared_ptr<ScoreboardRoi> create(NodeCreationInfo& nci) {
        EdgeManager& edges = nci.edges;
        const Parameters& params = nci.params;
        auto r = NodeSISO<av::VideoFrame, av::VideoFrame>::template createCommon<ScoreboardRoi>(edges, params);

        r->applyPreset(params.value("preset", std::string("balanced")));
        if (params.count("metadata_key_out")) r->metadata_key_out_ = params["metadata_key_out"].get<std::string>();
        if (params.count("output_metadata_key")) r->metadata_key_out_ = params["output_metadata_key"].get<std::string>();
        if (params.count("team_probe_frames")) r->team_probe_frames_ = std::max(0, params["team_probe_frames"].get<int>());
        if (params.count("debug_log_every_n")) r->debug_log_every_n_ = params["debug_log_every_n"].get<int>();
        if (params.count("lower_third") && params["lower_third"].is_object()) {
            const auto& lt = params["lower_third"];
            r->lower_third_x_rel_ = valueFloat(lt, "x", r->lower_third_x_rel_);
            r->lower_third_y_rel_ = valueFloat(lt, "y", r->lower_third_y_rel_);
            r->lower_third_w_rel_ = valueFloat(lt, "w", r->lower_third_w_rel_);
            r->lower_third_h_rel_ = valueFloat(lt, "h", r->lower_third_h_rel_);
        }
        if (params.count("lower_third_x")) r->lower_third_x_rel_ = params["lower_third_x"].get<float>();
        if (params.count("lower_third_y")) r->lower_third_y_rel_ = params["lower_third_y"].get<float>();
        if (params.count("lower_third_w")) r->lower_third_w_rel_ = params["lower_third_w"].get<float>();
        if (params.count("lower_third_h")) r->lower_third_h_rel_ = params["lower_third_h"].get<float>();
        if (params.count("roi_fields") && params["roi_fields"].is_array()) {
            r->fields_.clear();
            for (const auto& item : params["roi_fields"]) {
                if (!item.is_object()) continue;
                ScoreboardRoiField f;
                f.name = item.value("name", std::string());
                f.label = item.value("label", f.name);
                f.kind = item.value("kind", std::string());
                f.x = item.value("x", 0.0f);
                f.y = item.value("y", 0.0f);
                f.w = item.value("w", 0.0f);
                f.h = item.value("h", 0.0f);
                f.team_probe = item.value("team_probe", false);
                if (f.name.empty()) f.name = f.label;
                if (!f.label.empty() && f.w > 0.0f && f.h > 0.0f) r->fields_.push_back(std::move(f));
            }
        }
        if (r->fields_.empty()) r->applyPreset("balanced");

        return r;
    }
};

DECLNODE(scoreboard_roi, ScoreboardRoi)
