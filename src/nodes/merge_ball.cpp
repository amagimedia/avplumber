#include "node_common.hpp"

extern "C" {
#include <libavutil/dict.h>
}

#include <string>
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
}

class MergeBall : public NodeSISO<av::VideoFrame, av::VideoFrame> {
private:
    std::string metadata_key_in_ = "yolo_detections_v1";
    std::string metadata_key_out_ = "merge_ball_v1";
    double min_conf_ = 0.0;

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
                dets_out.push_back(det);
            }
            return true;
        } catch (const std::exception&) {
            return false;
        }
    }

    Parameters buildOutputMetadata(const MetadataEnvelope& env,
                                   const std::vector<DetectionBox>& dets) const {
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
        return md;
    }

public:
    using NodeSISO::NodeSISO;

    void process() override {
        av::VideoFrame frm = this->source_->get();
        if (!frm) return;

        if (isEofMarker(frm)) {
            this->sink_->put(frm);
            return;
        }

        MetadataEnvelope env;
        std::vector<DetectionBox> dets;
        (void)parseDetections(frm, env, dets);

        const Parameters md = buildOutputMetadata(env, dets);
        const std::string serialized = md.dump();
        av_dict_set(&frm.raw()->metadata, metadata_key_out_.c_str(), serialized.c_str(), 0);
        this->sink_->put(frm);
    }

    static std::shared_ptr<MergeBall> create(NodeCreationInfo& nci) {
        EdgeManager& edges = nci.edges;
        const Parameters& params = nci.params;
        auto r = NodeSISO<av::VideoFrame, av::VideoFrame>::template createCommon<MergeBall>(edges, params);
        if (params.count("metadata_key_in")) r->metadata_key_in_ = params["metadata_key_in"].get<std::string>();
        if (params.count("metadata_key_out")) r->metadata_key_out_ = params["metadata_key_out"].get<std::string>();
        if (params.count("min_conf")) r->min_conf_ = params["min_conf"];
        return r;
    }
};

DECLNODE(merge_ball, MergeBall)
