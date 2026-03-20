#include "node_common.hpp"

extern "C" {
#include <libavutil/dict.h>
}

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <unordered_set>
#include <vector>

namespace {
struct DetectionBox {
    int cls = -1;
    std::string label;
    bool has_label = false;
    double conf = 0.0;
    double x1 = 0.0;
    double y1 = 0.0;
    double x2 = 0.0;
    double y2 = 0.0;
    int model_index = -1;
    std::string engine_name;
    bool has_engine_name = false;
};

struct MetadataEnvelope {
    int version = 1;
    std::string coord_space = "model";
    double model_width = 0.0;
    double model_height = 0.0;
    Parameters thresholds;
};

static bool finiteBox(const DetectionBox& box) {
    return std::isfinite(box.x1) && std::isfinite(box.y1)
        && std::isfinite(box.x2) && std::isfinite(box.y2)
        && box.x2 > box.x1 && box.y2 > box.y1;
}
}

class MergeBall : public NodeSISO<av::VideoFrame, av::VideoFrame>, public IInputReset {
private:
    std::string metadata_key_in_ = "yolo_detections_v1";
    std::string metadata_key_out_ = "merge_ball_v1";
    int ball_model_index_ = 0;
    int context_model_index_ = 1;
    std::string ball_label_ = "basketball";
    std::unordered_set<std::string> context_presence_labels_ = {"foot", "player", "ball"};
    std::unordered_set<std::string> context_output_labels_ = {"foot", "player"};
    double min_conf_ = 0.0;
    int fallback_delay_frames_ = 6;
    int debug_log_every_n_ = 0;
    uint64_t frame_counter_ = 0;
    int consecutive_context_misses_ = 0;

    void resetState() {
        consecutive_context_misses_ = 0;
    }

    bool parseDetections(const av::VideoFrame& frm,
                         MetadataEnvelope& env_out,
                         std::vector<DetectionBox>& dets_out) const {
        dets_out.clear();
        env_out = MetadataEnvelope{};
        env_out.model_width = frm.width();
        env_out.model_height = frm.height();

        const AVFrame* raw = frm.raw();
        if (!raw || !raw->metadata) return false;

        AVDictionaryEntry* entry = av_dict_get(raw->metadata, metadata_key_in_.c_str(), nullptr, 0);
        if (!entry || !entry->value) return false;

        try {
            Parameters md = Parameters::parse(entry->value);
            env_out.version = md.value("version", 1);
            env_out.coord_space = md.value("coord_space", std::string("model"));
            env_out.model_width = md.value("model_width", (double)frm.width());
            env_out.model_height = md.value("model_height", (double)frm.height());
            if (md.contains("thresholds")) {
                env_out.thresholds = md["thresholds"];
            }
            if (!md.contains("detections") || !md["detections"].is_array()) {
                return true;
            }

            for (const auto& item : md["detections"]) {
                if (!item.is_object()) continue;
                if (!item.contains("xyxy") || !item["xyxy"].is_array() || item["xyxy"].size() < 4) continue;

                DetectionBox det;
                det.cls = item.value("cls", -1);
                det.conf = item.value("conf", 0.0);
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

                if (det.conf < min_conf_) continue;
                if (!finiteBox(det)) continue;
                dets_out.push_back(det);
            }

            std::sort(dets_out.begin(), dets_out.end(), [](const DetectionBox& a, const DetectionBox& b) {
                return a.conf > b.conf;
            });
            return true;
        } catch (const std::exception&) {
            return false;
        }
    }

    std::vector<DetectionBox> selectDetections(const std::vector<DetectionBox>& dets, std::string& mode_out) {
        std::vector<DetectionBox> context_dets;
        bool have_context_presence = false;
        DetectionBox best_ball_label;
        DetectionBox best_ball_any;
        bool have_best_ball_label = false;
        bool have_best_ball_any = false;

        for (const DetectionBox& det : dets) {
            if (det.model_index == context_model_index_ && det.has_label) {
                if (context_presence_labels_.count(det.label) > 0) {
                    have_context_presence = true;
                }
                if (context_output_labels_.count(det.label) > 0) {
                    context_dets.push_back(det);
                }
            }
            if (det.model_index == ball_model_index_) {
                if (!have_best_ball_any || det.conf > best_ball_any.conf) {
                    best_ball_any = det;
                    have_best_ball_any = true;
                }
                if ((!ball_label_.empty() && det.has_label && det.label == ball_label_) || ball_label_.empty()) {
                    if (!have_best_ball_label || det.conf > best_ball_label.conf) {
                        best_ball_label = det;
                        have_best_ball_label = true;
                    }
                }
            }
        }

        if (have_context_presence) {
            consecutive_context_misses_ = 0;
            mode_out = "context";
            return context_dets;
        }
        ++consecutive_context_misses_;
        if (consecutive_context_misses_ < std::max(1, fallback_delay_frames_)) {
            mode_out = "grace";
            return {};
        }
        if (have_best_ball_label) {
            mode_out = "ball_fallback";
            return {best_ball_label};
        }
        if (have_best_ball_any) {
            mode_out = "ball_fallback";
            return {best_ball_any};
        }
        mode_out = "empty";
        return {};
    }

    Parameters buildOutputMetadata(const MetadataEnvelope& env,
                                   const std::vector<DetectionBox>& dets,
                                   const std::string& mode) const {
        Parameters md;
        md["version"] = 1;
        md["coord_space"] = env.coord_space;
        md["model_width"] = env.model_width;
        md["model_height"] = env.model_height;
        if (!env.thresholds.is_null()) {
            md["thresholds"] = env.thresholds;
        }
        md["detections"] = Parameters::array();
        for (const DetectionBox& det : dets) {
            Parameters item;
            item["cls"] = det.cls;
            item["conf"] = det.conf;
            item["xyxy"] = {det.x1, det.y1, det.x2, det.y2};
            item["model_index"] = det.model_index;
            if (det.has_label) {
                item["label"] = det.label;
            }
            if (det.has_engine_name) {
                item["engine_name"] = det.engine_name;
            }
            md["detections"].push_back(item);
        }
        md["merge_ball"] = {
            {"mode", mode},
            {"ball_model_index", ball_model_index_},
            {"context_model_index", context_model_index_},
            {"fallback_delay_frames", fallback_delay_frames_},
            {"consecutive_context_misses", consecutive_context_misses_}
        };
        return md;
    }

    void maybeLog(const std::vector<DetectionBox>& dets, const std::string& mode) const {
        if (debug_log_every_n_ <= 0) return;
        if ((frame_counter_ % (uint64_t)debug_log_every_n_) != 0) return;
        logstream << "merge_ball: frame=" << frame_counter_
                  << " mode=" << mode
                  << " detections=" << dets.size();
    }

public:
    using NodeSISO::NodeSISO;

    void resetInput() override {
        resetState();
    }

    void process() override {
        av::VideoFrame frm = this->source_->get();
        if (!frm) return;

        if (isEofMarker(frm)) {
            resetState();
            this->sink_->put(frm);
            return;
        }

        ++frame_counter_;

        MetadataEnvelope env;
        std::vector<DetectionBox> dets;
        (void)parseDetections(frm, env, dets);

        std::string mode;
        const std::vector<DetectionBox> merged = selectDetections(dets, mode);
        const Parameters md = buildOutputMetadata(env, merged, mode);
        const std::string serialized = md.dump();
        av_dict_set(&frm.raw()->metadata, metadata_key_out_.c_str(), serialized.c_str(), 0);
        maybeLog(merged, mode);
        this->sink_->put(frm);
    }

    static std::shared_ptr<MergeBall> create(NodeCreationInfo& nci) {
        EdgeManager& edges = nci.edges;
        const Parameters& params = nci.params;
        auto r = NodeSISO<av::VideoFrame, av::VideoFrame>::template createCommon<MergeBall>(edges, params);
        if (params.count("metadata_key_in")) r->metadata_key_in_ = params["metadata_key_in"].get<std::string>();
        if (params.count("metadata_key_out")) r->metadata_key_out_ = params["metadata_key_out"].get<std::string>();
        if (params.count("ball_model_index")) r->ball_model_index_ = params["ball_model_index"];
        if (params.count("context_model_index")) r->context_model_index_ = params["context_model_index"];
        if (params.count("ball_label")) r->ball_label_ = params["ball_label"].get<std::string>();
        if (params.count("context_presence_labels")) {
            if (!params["context_presence_labels"].is_array()) {
                throw Error("merge_ball: context_presence_labels must be a string array");
            }
            r->context_presence_labels_.clear();
            for (const auto& item : params["context_presence_labels"]) {
                if (!item.is_string()) {
                    throw Error("merge_ball: context_presence_labels must be a string array");
                }
                r->context_presence_labels_.insert(item.get<std::string>());
            }
        }
        if (params.count("context_output_labels")) {
            if (!params["context_output_labels"].is_array()) {
                throw Error("merge_ball: context_output_labels must be a string array");
            }
            r->context_output_labels_.clear();
            for (const auto& item : params["context_output_labels"]) {
                if (!item.is_string()) {
                    throw Error("merge_ball: context_output_labels must be a string array");
                }
                r->context_output_labels_.insert(item.get<std::string>());
            }
        }
        if (params.count("min_conf")) r->min_conf_ = params["min_conf"];
        if (params.count("fallback_delay_frames")) r->fallback_delay_frames_ = params["fallback_delay_frames"];
        if (params.count("debug_log_every_n")) r->debug_log_every_n_ = params["debug_log_every_n"];
        return r;
    }
};

DECLNODE(merge_ball, MergeBall)