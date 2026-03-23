#include "node_common.hpp"

extern "C" {
#include <libavutil/dict.h>
}

#include <algorithm>
#include <cmath>
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
    return std::sqrt(dx*dx + dy*dy);
}
static double lerp(double a, double b, double t) { return a + (b - a) * t; }
static DetectionBox lerpBox(const DetectionBox& a, const DetectionBox& b, double t) {
    DetectionBox r = b; // inherit label, cls, model_index etc. from b
    r.x1 = lerp(a.x1, b.x1, t); r.y1 = lerp(a.y1, b.y1, t);
    r.x2 = lerp(a.x2, b.x2, t); r.y2 = lerp(a.y2, b.y2, t);
    r.conf = lerp(a.conf, b.conf, t);
    return r;
}
} // namespace

class InterpDetections : public NodeSISO<av::VideoFrame, av::VideoFrame>, public IInputReset {
private:
    std::string metadata_key_ = "merge_ball_v1";
    std::string interp_label_ = "player";
    int max_gap_frames_ = 3;
    double max_match_distance_px_ = 200.0;

    std::vector<av::VideoFrame> gap_buffer_;
    std::vector<DetectionBox> last_player_dets_;
    bool have_last_player_dets_ = false;

    void resetState() {
        gap_buffer_.clear();
        last_player_dets_.clear();
        have_last_player_dets_ = false;
    }

    bool parseDetections(const av::VideoFrame& frm, MetadataEnvelope& env,
                         std::vector<DetectionBox>& dets) const {
        dets.clear();
        env = MetadataEnvelope{};
        env.model_width = frm.width();
        env.model_height = frm.height();

        const AVFrame* raw = frm.raw();
        if (!raw || !raw->metadata) return false;
        AVDictionaryEntry* entry = av_dict_get(raw->metadata, metadata_key_.c_str(), nullptr, 0);
        if (!entry || !entry->value) return false;

        try {
            Parameters md = Parameters::parse(entry->value);
            env.version = md.value("version", 1);
            env.coord_space = md.value("coord_space", std::string("model"));
            env.model_width = md.value("model_width", (double)frm.width());
            env.model_height = md.value("model_height", (double)frm.height());
            if (md.contains("thresholds")) env.thresholds = md["thresholds"];
            if (!md.contains("detections") || !md["detections"].is_array()) return true;

            for (const auto& item : md["detections"]) {
                if (!item.is_object()) continue;
                if (!item.contains("xyxy") || !item["xyxy"].is_array()
                    || item["xyxy"].size() < 4) continue;
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
                dets.push_back(det);
            }
            return true;
        } catch (const std::exception&) { return false; }
    }

    Parameters buildMetadata(const MetadataEnvelope& env,
                              const std::vector<DetectionBox>& dets) const {
        Parameters md;
        md["version"] = 1;
        md["coord_space"] = env.coord_space;
        md["model_width"] = env.model_width;
        md["model_height"] = env.model_height;
        if (!env.thresholds.is_null()) md["thresholds"] = env.thresholds;
        md["detections"] = Parameters::array();
        for (const auto& det : dets) {
            Parameters item;
            item["cls"] = det.cls;
            item["conf"] = det.conf;
            item["xyxy"] = {det.x1, det.y1, det.x2, det.y2};
            item["model_index"] = det.model_index;
            if (det.has_label) item["label"] = det.label;
            if (det.has_engine_name) item["engine_name"] = det.engine_name;
            md["detections"].push_back(item);
        }
        return md;
    }

    void writeMetadata(av::VideoFrame& frm, const MetadataEnvelope& env,
                       const std::vector<DetectionBox>& dets) {
        const std::string s = buildMetadata(env, dets).dump();
        av_dict_set(&frm.raw()->metadata, metadata_key_.c_str(), s.c_str(), 0);
    }

    // Greedy nearest-center matching within max_match_distance_px_.
    // Returns (last_idx, current_idx) pairs.
    std::vector<std::pair<int,int>> matchBoxes(const std::vector<DetectionBox>& last,
                                               const std::vector<DetectionBox>& cur) const {
        std::vector<std::tuple<double,int,int>> candidates;
        for (int i = 0; i < (int)last.size(); i++) {
            for (int j = 0; j < (int)cur.size(); j++) {
                double d = centerDist(last[i], cur[j]);
                if (d <= max_match_distance_px_)
                    candidates.emplace_back(d, i, j);
            }
        }
        std::sort(candidates.begin(), candidates.end());

        std::vector<bool> last_used(last.size(), false), cur_used(cur.size(), false);
        std::vector<std::pair<int,int>> matches;
        for (auto& [d, i, j] : candidates) {
            if (!last_used[i] && !cur_used[j]) {
                matches.emplace_back(i, j);
                last_used[i] = cur_used[j] = true;
            }
        }
        return matches;
    }

    void flushGapBufferRaw() {
        for (auto& f : gap_buffer_) this->sink_->put(f);
        gap_buffer_.clear();
    }

    // Emit buffered gap frames with linearly interpolated player boxes.
    // The resume frame (with resume_dets) is peeked but not yet consumed.
    void flushGapBufferInterp(const std::vector<DetectionBox>& resume_dets) {
        if (!have_last_player_dets_ || gap_buffer_.empty()) {
            flushGapBufferRaw();
            return;
        }
        const auto matches = matchBoxes(last_player_dets_, resume_dets);
        if (matches.empty()) {
            flushGapBufferRaw();
            return;
        }

        const int n = (int)gap_buffer_.size();
        logstream << "interp_detections: interpolating " << n << " gap frame(s) with "
                  << matches.size() << "/" << resume_dets.size() << " matched player boxes";
        for (int i = 0; i < n; i++) {
            const double t = (double)(i + 1) / (double)(n + 1);

            MetadataEnvelope env;
            std::vector<DetectionBox> existing;
            parseDetections(gap_buffer_[i], env, existing);

            // Keep non-player detections, splice in interpolated players.
            std::vector<DetectionBox> result;
            for (auto& d : existing) {
                if (!d.has_label || d.label != interp_label_) result.push_back(d);
            }
            for (auto& [li, ci] : matches) {
                result.push_back(lerpBox(last_player_dets_[li], resume_dets[ci], t));
            }

            writeMetadata(gap_buffer_[i], env, result);
            this->sink_->put(gap_buffer_[i]);
        }
        gap_buffer_.clear();
    }

public:
    using NodeSISO::NodeSISO;

    void resetInput() override { resetState(); }

    void process() override {
        av::VideoFrame* pfrm = this->source_->peek();
        if (!pfrm) return;

        if (!*pfrm) {
            this->source_->pop();
            return;
        }

        if (isEofMarker(*pfrm)) {
            av::VideoFrame eof = *pfrm;
            this->source_->pop();
            flushGapBufferRaw();
            resetState();
            this->sink_->put(eof);
            return;
        }

        MetadataEnvelope env;
        std::vector<DetectionBox> all_dets;
        parseDetections(*pfrm, env, all_dets);

        std::vector<DetectionBox> player_dets;
        for (auto& d : all_dets) {
            if (d.has_label && d.label == interp_label_) player_dets.push_back(d);
        }

        if (!player_dets.empty()) {
            // Resume frame: interpolate gap buffer while it's still peeked in the queue.
            flushGapBufferInterp(player_dets);
            av::VideoFrame frm = *pfrm;
            this->source_->pop();
            last_player_dets_ = std::move(player_dets);
            have_last_player_dets_ = true;
            this->sink_->put(frm);
        } else {
            // Gap frame: consume into local buffer.
            av::VideoFrame frm = *pfrm;
            this->source_->pop();
            gap_buffer_.push_back(std::move(frm));
            // Buffer full: emit oldest as-is (no resume found within max_gap_frames_).
            if ((int)gap_buffer_.size() > max_gap_frames_) {
                this->sink_->put(gap_buffer_.front());
                gap_buffer_.erase(gap_buffer_.begin());
            }
        }
    }

    static std::shared_ptr<InterpDetections> create(NodeCreationInfo& nci) {
        EdgeManager& edges = nci.edges;
        const Parameters& params = nci.params;
        auto r = NodeSISO<av::VideoFrame, av::VideoFrame>::template createCommon<InterpDetections>(
            edges, params);
        if (params.count("metadata_key"))
            r->metadata_key_ = params["metadata_key"].get<std::string>();
        if (params.count("interp_label"))
            r->interp_label_ = params["interp_label"].get<std::string>();
        if (params.count("max_gap_frames"))
            r->max_gap_frames_ = params["max_gap_frames"];
        if (params.count("max_match_distance_px"))
            r->max_match_distance_px_ = params["max_match_distance_px"];
        return r;
    }
};

DECLNODE(interp_detections, InterpDetections)
