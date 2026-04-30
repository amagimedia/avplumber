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
    bool scan_lower_third_ = false;
    std::string scan_label_ = "Text";
    float scan_window_h_rel_ = 0.24f;
    float scan_step_x_rel_ = 0.045f;
    float scan_step_y_rel_ = 0.18f;
    float scan_y_margin_rel_ = 0.02f;
    int scan_max_detections_ = 240;
    std::vector<float> scan_window_widths_rel_ = {0.045f, 0.075f};

    void setBalancedPreset() {
        scan_lower_third_ = false;
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
        } else if (preset == "scan_lower_third" || preset == "scan") {
            scan_lower_third_ = true;
            fields_.clear();
            lower_third_x_rel_ = 0.0f;
            lower_third_y_rel_ = 0.66f;
            lower_third_w_rel_ = 1.0f;
            lower_third_h_rel_ = 0.32f;
            team_probe_frames_ = 0;
        }
    }

    bool includeField(const ScoreboardRoiField& field) const {
        if (!field.team_probe) return true;
        if (team_probe_frames_ == 0) return false;
        return frame_counter_ <= (uint64_t)team_probe_frames_;
    }

    void appendDetection(Parameters& out, int cls, const std::string& label, const std::string& kind,
                         const std::string& field_name, float x1, float y1, float x2, float y2,
                         bool team_probe = false) const {
        Parameters det;
        det["cls"] = cls;
        det["conf"] = 1.0f;
        det["label"] = label;
        det["kind"] = kind;
        det["field"] = field_name;
        det["team_probe"] = team_probe;
        det["xyxy"] = {x1, y1, x2, y2};
        out["detections"].push_back(std::move(det));
    }

    void appendScanDetections(Parameters& out, int frame_w, int frame_h,
                              float lx, float ly, float lw, float lh) const {
        const float step_x = std::max(0.005f, scan_step_x_rel_);
        const float step_y = std::max(0.005f, scan_step_y_rel_);
        const float win_h = std::clamp(scan_window_h_rel_, 0.02f, 1.0f);
        const float y_margin = std::clamp(scan_y_margin_rel_, 0.0f, 0.45f);
        int cls = 0;
        int emitted = 0;
        for (size_t wi = 0; wi < scan_window_widths_rel_.size(); ++wi) {
            const float win_w = std::clamp(scan_window_widths_rel_[wi], 0.01f, 1.0f);
            if (win_w <= 0.0f || win_h <= 0.0f) continue;
            int row = 0;
            for (float y = y_margin; y + win_h <= 1.0f - y_margin + 1e-4f; y += step_y, ++row) {
                int col = 0;
                for (float x = 0.0f; x + win_w <= 1.0f + 1e-4f; x += step_x, ++col) {
                    if (scan_max_detections_ > 0 && emitted >= scan_max_detections_) return;
                    const float fx1 = std::clamp(lx + x * lw, 0.0f, 1.0f);
                    const float fy1 = std::clamp(ly + y * lh, 0.0f, 1.0f);
                    const float fx2 = std::clamp(lx + std::min(1.0f, x + win_w) * lw, 0.0f, 1.0f);
                    const float fy2 = std::clamp(ly + std::min(1.0f, y + win_h) * lh, 0.0f, 1.0f);
                    if (fx2 <= fx1 || fy2 <= fy1) continue;
                    appendDetection(out, cls++, scan_label_, "lower_third_scan",
                                    "scan_w" + std::to_string(wi) + "_r" + std::to_string(row) + "_c" + std::to_string(col),
                                    fx1 * (float)frame_w, fy1 * (float)frame_h,
                                    fx2 * (float)frame_w, fy2 * (float)frame_h);
                    ++emitted;
                }
            }
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
        out["source"] = scan_lower_third_ ? "lower_third_scan" : "fixed_lower_third";
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

            if (scan_lower_third_) {
                appendScanDetections(out, frame_w, frame_h, lx, ly, lw, lh);
            } else {
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
                    appendDetection(out, cls, field.label, field.kind, field.name,
                                    fx1 * (float)frame_w, fy1 * (float)frame_h,
                                    fx2 * (float)frame_w, fy2 * (float)frame_h,
                                    field.team_probe);
                    ++cls;
                }
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
        if (params.count("scan_label")) r->scan_label_ = params["scan_label"].get<std::string>();
        if (params.count("scan_window_h_rel")) r->scan_window_h_rel_ = params["scan_window_h_rel"].get<float>();
        if (params.count("scan_step_x_rel")) r->scan_step_x_rel_ = params["scan_step_x_rel"].get<float>();
        if (params.count("scan_step_y_rel")) r->scan_step_y_rel_ = params["scan_step_y_rel"].get<float>();
        if (params.count("scan_y_margin_rel")) r->scan_y_margin_rel_ = params["scan_y_margin_rel"].get<float>();
        if (params.count("scan_max_detections")) r->scan_max_detections_ = std::max(0, params["scan_max_detections"].get<int>());
        if (params.count("scan_window_widths_rel") && params["scan_window_widths_rel"].is_array()) {
            r->scan_window_widths_rel_.clear();
            for (const auto& item : params["scan_window_widths_rel"]) {
                float v = item.get<float>();
                if (v > 0.0f) r->scan_window_widths_rel_.push_back(v);
            }
            if (r->scan_window_widths_rel_.empty()) r->scan_window_widths_rel_ = {0.045f, 0.075f};
        }
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
        if (!r->scan_lower_third_ && r->fields_.empty()) r->applyPreset("balanced");

        return r;
    }
};

DECLNODE(scoreboard_roi, ScoreboardRoi)
