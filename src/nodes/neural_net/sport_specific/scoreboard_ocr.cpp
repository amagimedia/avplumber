#include "../../node_common.hpp"
#include "../common/infer_trt_base.hpp"

extern "C" {
#include <libavutil/dict.h>
}

#include "../../../../objs/src/nodes/neural_net/preprocess/nv12_crop_resize_pad.ptx.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <string>
#include <vector>

namespace {

constexpr int kOcrDstHeight = 48;
constexpr int kOcrDstWidthPadded = 320;

struct ScoreboardDetection {
    std::string label;
    float x1, y1, x2, y2;
    float conf;
    int cls = -1;
};

float bboxIoU(float ax1, float ay1, float ax2, float ay2,
              float bx1, float by1, float bx2, float by2) {
    const float ix1 = std::max(ax1, bx1), iy1 = std::max(ay1, by1);
    const float ix2 = std::min(ax2, bx2), iy2 = std::min(ay2, by2);
    const float iw = std::max(0.0f, ix2 - ix1), ih = std::max(0.0f, iy2 - iy1);
    const float inter = iw * ih;
    const float ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter;
    return ua > 0.0f ? inter / ua : 0.0f;
}

// NMS across all scoreboard classes. For overlapping boxes, prefer:
// 1. Team Name / Team Points over other classes (they're smaller, more precise)
// 2. Higher confidence otherwise
void nmsScoreboard(std::vector<ScoreboardDetection>& dets, float iou_thresh) {
    auto priority = [](const std::string& label) -> int {
        if (label == "Team Name" || label == "Team Points") return 2;
        if (label == "Period" || label == "Shot Clock") return 1;
        return 0; // Time Remaining is lowest priority (tends to be oversized)
    };

    std::sort(dets.begin(), dets.end(), [&](const ScoreboardDetection& a, const ScoreboardDetection& b) {
        int pa = priority(a.label), pb = priority(b.label);
        if (pa != pb) return pa > pb;
        return a.conf > b.conf;
    });

    std::vector<bool> suppressed(dets.size(), false);
    for (size_t i = 0; i < dets.size(); ++i) {
        if (suppressed[i]) continue;
        for (size_t j = i + 1; j < dets.size(); ++j) {
            if (suppressed[j]) continue;
            if (bboxIoU(dets[i].x1, dets[i].y1, dets[i].x2, dets[i].y2,
                        dets[j].x1, dets[j].y1, dets[j].x2, dets[j].y2) >= iou_thresh) {
                suppressed[j] = true;
            }
        }
    }

    std::vector<ScoreboardDetection> kept;
    for (size_t i = 0; i < dets.size(); ++i) {
        if (!suppressed[i]) kept.push_back(std::move(dets[i]));
    }
    dets = std::move(kept);
}

// Limit detections per class: Team Name/Points keep 2, others keep 1
void limitPerClass(std::vector<ScoreboardDetection>& dets) {
    std::unordered_map<std::string, int> counts;
    std::vector<ScoreboardDetection> kept;
    for (auto& d : dets) {
        int max_per_class = (d.label == "Team Name" || d.label == "Team Points") ? 2 : 1;
        if (counts[d.label] < max_per_class) {
            counts[d.label]++;
            kept.push_back(std::move(d));
        }
    }
    dets = std::move(kept);
}

struct OcrResult {
    std::string label;
    std::string text;
    float mean_conf = 0.0f;
    float center_x = 0.0f;
};

// CTC greedy decode: argmax per timestep, dedup consecutive, skip blank (index 0).
// Algorithm reference: RapidOcrOnnx CrnnNet::scoreToTextLine (Apache-2.0).
std::string ctcGreedyDecode(const float* output, int seq_len, int num_classes,
                            const std::vector<std::string>& keys, float& mean_conf) {
    std::string result;
    std::vector<float> scores;
    int last_index = 0;
    for (int t = 0; t < seq_len; ++t) {
        const float* row = output + t * num_classes;
        int best_idx = 0;
        float best_val = row[0];
        for (int c = 1; c < num_classes; ++c) {
            if (row[c] > best_val) {
                best_val = row[c];
                best_idx = c;
            }
        }
        if (best_idx > 0 && best_idx < (int)keys.size() && best_idx != last_index) {
            result.append(keys[best_idx]);
            scores.push_back(best_val);
        }
        last_index = best_idx;
    }
    mean_conf = 0.0f;
    if (!scores.empty()) {
        float sum = 0.0f;
        for (float s : scores) sum += s;
        mean_conf = sum / (float)scores.size();
    }
    return result;
}

} // namespace

class ScoreboardOcr : public NodeSISO<av::VideoFrame, av::VideoFrame>, public yolo_base::CudaInferTrtBase {
    std::string detection_metadata_key_ = "yolo_players";
    std::string output_metadata_key_ = "scoreboard";
    std::vector<std::string> scoreboard_labels_ = {"Period", "Shot Clock", "Team Name", "Team Points", "Time Remaining"};
    float min_conf_ = 0.4f;
    float nms_iou_thresh_ = 0.3f;
    int ocr_every_n_ = 25;
    int debug_log_every_n_ = 0;
    std::string keys_file_;

    std::vector<std::string> keys_;
    CUmodule crop_module_ = nullptr;
    CUfunction crop_kernel_ = nullptr;
    bool ocr_initialized_ = false;
    uint64_t frame_counter_ = 0;
    std::string cached_scoreboard_json_;

    bool initOcr(const av::VideoFrame& frm) {
        if (ocr_initialized_) return true;
        if (!ensureInitialized(frm)) return false;

        const std::string ptx_str(avpl_ocr_crop_ptx, avpl_ocr_crop_ptx + avpl_ocr_crop_ptx_len);
        if (CUDA_CHECK_CU(cuModuleLoadDataEx(&crop_module_, ptx_str.c_str(), 0, nullptr, nullptr))) {
            logstream << "scoreboard_ocr: failed to load crop PTX";
            return false;
        }
        if (CUDA_CHECK_CU(cuModuleGetFunction(&crop_kernel_, crop_module_, "kNV12_crop_resize_pad_RGB"))) {
            logstream << "scoreboard_ocr: failed to get crop kernel";
            return false;
        }

        std::ifstream kin(keys_file_);
        if (!kin) {
            logstream << "scoreboard_ocr: cannot open keys file: " << keys_file_;
            return false;
        }
        std::string line;
        keys_.clear();
        keys_.push_back("#"); // index 0 = CTC blank
        while (std::getline(kin, line)) {
            keys_.push_back(line);
        }
        keys_.push_back(" ");
        logstream << "scoreboard_ocr: loaded " << keys_.size() << " keys from " << keys_file_;

        ocr_initialized_ = true;
        return true;
    }

    std::vector<ScoreboardDetection> extractScoreboardDetections(const Parameters& md) {
        std::vector<ScoreboardDetection> dets;
        if (!md.contains("detections") || !md["detections"].is_array()) return dets;
        for (const auto& det : md["detections"]) {
            if (!det.is_object()) continue;
            if (!det.contains("label") || !det["label"].is_string()) continue;
            const std::string label = det["label"].get<std::string>();
            bool match = false;
            for (const auto& want : scoreboard_labels_) {
                if (label == want) { match = true; break; }
            }
            if (!match) continue;
            float conf = det.value("conf", 0.0f);
            if (conf < min_conf_) continue;
            if (!det.contains("xyxy") || !det["xyxy"].is_array() || det["xyxy"].size() < 4) continue;
            ScoreboardDetection sd;
            sd.label = label;
            sd.x1 = det["xyxy"][0].get<float>();
            sd.y1 = det["xyxy"][1].get<float>();
            sd.x2 = det["xyxy"][2].get<float>();
            sd.y2 = det["xyxy"][3].get<float>();
            sd.conf = conf;
            sd.cls = det.value("cls", -1);
            dets.push_back(sd);
        }
        return dets;
    }

    OcrResult runOcrOnCrop(const av::VideoFrame& frm, const ScoreboardDetection& det,
                           double scale_x, double scale_y) {
        OcrResult result;
        result.label = det.label;
        result.center_x = (det.x1 + det.x2) * 0.5f;

        int sx1 = std::max(0, (int)(det.x1 * scale_x));
        int sy1 = std::max(0, (int)(det.y1 * scale_y));
        int sx2 = std::min(frm.width(), (int)(det.x2 * scale_x));
        int sy2 = std::min(frm.height(), (int)(det.y2 * scale_y));
        int src_w = sx2 - sx1;
        int src_h = sy2 - sy1;
        if (src_w <= 0 || src_h <= 0) return result;

        int dst_w_content = (int)((float)src_w * ((float)kOcrDstHeight / (float)src_h));
        dst_w_content = std::max(1, std::min(dst_w_content, kOcrDstWidthPadded));

        const AVFrame* raw = frm.raw();
        const CUdeviceptr y_plane = (CUdeviceptr)(uintptr_t)raw->data[0];
        const int y_pitch = raw->linesize[0];
        const CUdeviceptr uv_plane = (CUdeviceptr)(uintptr_t)raw->data[1];
        const int uv_pitch = raw->linesize[1];

        auto& model = models_[0];
        auto it = model.tensor_index.find(model.input_tensor_name);
        if (it == model.tensor_index.end()) return result;
        CUdeviceptr input_ptr = model.tensor_ptrs[it->second];

        const int dst_h = kOcrDstHeight;
        const int dst_w_pad = kOcrDstWidthPadded;
        void* args[] = {
            (void*)&y_plane, (void*)&y_pitch,
            (void*)&uv_plane, (void*)&uv_pitch,
            (void*)&input_ptr,
            (void*)&sx1, (void*)&sy1, (void*)&src_w, (void*)&src_h,
            (void*)&dst_h, (void*)&dst_w_content, (void*)&dst_w_pad
        };
        const unsigned bx = 32, by = 8;
        const unsigned gx = ((unsigned)dst_w_pad + bx - 1) / bx;
        const unsigned gy = ((unsigned)dst_h + by - 1) / by;
        if (CUDA_CHECK_CU(cuLaunchKernel(crop_kernel_, gx, gy, 1, bx, by, 1, 0, model.stream, args, nullptr))) {
            return result;
        }

        if (!runInference(model) || !syncModel(model)) return result;

        const auto& out = model.outputs[0];
        int seq_len = 1, num_classes = 1;
        if (out.dims.nbDims == 3) {
            seq_len = out.dims.d[1];
            num_classes = out.dims.d[2];
        } else if (out.dims.nbDims == 2) {
            seq_len = out.dims.d[0];
            num_classes = out.dims.d[1];
        }
        result.text = ctcGreedyDecode(out.host_output.data(), seq_len, num_classes, keys_, result.mean_conf);
        return result;
    }

    static std::string extractLeadingAbbrev(const std::string& text) {
        std::string abbrev;
        for (char c : text) {
            if (std::isupper((unsigned char)c)) abbrev += c;
            else break;
        }
        return abbrev;
    }

    static bool isTeamAbbrev(const std::string& abbrev) {
        if (abbrev.length() < 2 || abbrev.length() > 3) return false;
        static const std::vector<std::string> ignore = {
            "NBA", "NHL", "NFL", "MLB", "MLS", "ESPN", "TNT", "ABC", "CBS", "FOX",
            "1ST", "2ND", "3RD", "4TH", "OT",
            "PM", "AM", "ET", "PT", "CT", "MT",
            "FG", "FT", "PTS", "REB", "AST", "STL", "BLK", "TO", "PF",
            "SJ", "VS"
        };
        for (const auto& ign : ignore) {
            if (abbrev == ign) return false;
        }
        return true;
    }

    struct ExtractedTeam {
        std::string abbrev;
        float center_x;
        float conf;
    };

    Parameters buildScoreboardJson(const std::vector<OcrResult>& results,
                                    const std::vector<OcrResult>& all_results) {
        Parameters sb;

        std::vector<const OcrResult*> names, points;

        for (const auto& r : results) {
            if (r.text.empty()) continue;
            if (r.label == "Team Name") names.push_back(&r);
            else if (r.label == "Team Points") points.push_back(&r);
            else if (r.label == "Period") {
                sb["period"] = {{"text", r.text}, {"conf", r.mean_conf}};
            } else if (r.label == "Shot Clock") {
                sb["shot_clock"] = {{"text", r.text}, {"conf", r.mean_conf}};
            } else if (r.label == "Time Remaining") {
                sb["time_remaining"] = {{"text", r.text}, {"conf", r.mean_conf}};
            }
        }

        // Scan ALL pre-NMS results for leading uppercase abbreviations that look like team names
        std::vector<ExtractedTeam> found_teams;
        for (const auto& r : all_results) {
            if (r.text.empty()) continue;
            std::string abbrev = extractLeadingAbbrev(r.text);
            if (isTeamAbbrev(abbrev)) {
                bool already_found = false;
                for (const auto& ft : found_teams) {
                    if (ft.abbrev == abbrev) { already_found = true; break; }
                }
                if (!already_found) {
                    found_teams.push_back({abbrev, r.center_x, r.mean_conf});
                }
            }
        }

        // If explicit Team Name detections found real abbreviations, prefer those
        std::vector<ExtractedTeam> team_abbrevs;
        for (const auto* n : names) {
            std::string abbrev = extractLeadingAbbrev(n->text);
            if (isTeamAbbrev(abbrev)) {
                team_abbrevs.push_back({abbrev, n->center_x, n->mean_conf});
            }
        }
        if (team_abbrevs.size() < 2 && found_teams.size() > team_abbrevs.size()) {
            team_abbrevs = std::move(found_teams);
        }

        // Sort by X: leftmost = team_a
        std::sort(team_abbrevs.begin(), team_abbrevs.end(),
                  [](const ExtractedTeam& a, const ExtractedTeam& b) { return a.center_x < b.center_x; });

        auto by_x = [](const OcrResult* a, const OcrResult* b) { return a->center_x < b->center_x; };
        std::sort(points.begin(), points.end(), by_x);

        if (team_abbrevs.size() >= 1) {
            sb["team_a"] = Parameters::object();
            sb["team_a"]["name"] = {{"text", team_abbrevs[0].abbrev}, {"conf", team_abbrevs[0].conf}};
            if (points.size() >= 1) {
                sb["team_a"]["points"] = {{"text", points[0]->text}, {"conf", points[0]->mean_conf}};
            }
        }
        if (team_abbrevs.size() >= 2) {
            sb["team_b"] = Parameters::object();
            sb["team_b"]["name"] = {{"text", team_abbrevs[1].abbrev}, {"conf", team_abbrevs[1].conf}};
            if (points.size() >= 2) {
                sb["team_b"]["points"] = {{"text", points[1]->text}, {"conf", points[1]->mean_conf}};
            }
        }

        return sb;
    }

public:
    using NodeSISO<av::VideoFrame, av::VideoFrame>::NodeSISO;

    ~ScoreboardOcr() {
        if (cu_ctx_) {
            cuCtxSetCurrent(cu_ctx_);
            if (crop_module_) cuModuleUnload(crop_module_);
        }
    }

    void process() override {
        av::VideoFrame frm = this->source_->get();
        if (!frm) return;
        if (isEofMarker(frm)) {
            cached_scoreboard_json_.clear();
            frame_counter_ = 0;
            this->sink_->put(frm);
            return;
        }

        ++frame_counter_;
        const AVFrame* raw = frm.raw();

        if ((frame_counter_ % (uint64_t)ocr_every_n_) != 1 && !cached_scoreboard_json_.empty()) {
            if (raw && raw->metadata) {
                av_dict_set(&frm.raw()->metadata, output_metadata_key_.c_str(), cached_scoreboard_json_.c_str(), 0);
            }
            this->sink_->put(frm);
            return;
        }

        if (!raw || !raw->metadata) {
            this->sink_->put(frm);
            return;
        }

        if (!initOcr(frm)) {
            this->sink_->put(frm);
            return;
        }

        AVDictionaryEntry* entry = av_dict_get(raw->metadata, detection_metadata_key_.c_str(), nullptr, 0);
        if (!entry || !entry->value) {
            this->sink_->put(frm);
            return;
        }

        Parameters md;
        try {
            md = Parameters::parse(entry->value);
        } catch (...) {
            this->sink_->put(frm);
            return;
        }

        auto dets = extractScoreboardDetections(md);
        if (dets.empty()) {
            this->sink_->put(frm);
            return;
        }

        const double model_w = md.value("model_width", 960.0);
        const double model_h = md.value("model_height", 544.0);
        const double scale_x = (double)frm.width() / model_w;
        const double scale_y = (double)frm.height() / model_h;

        if (cu_ctx_) cuCtxSetCurrent(cu_ctx_);

        // OCR all detections (pre-NMS) so we can scan everything for team abbreviations
        std::vector<OcrResult> all_results;
        for (const auto& det : dets) {
            OcrResult r = runOcrOnCrop(frm, det, scale_x, scale_y);
            if (!r.text.empty()) all_results.push_back(std::move(r));
        }

        // NMS + per-class limits for the main scoreboard fields
        nmsScoreboard(dets, nms_iou_thresh_);
        limitPerClass(dets);

        // Keep only the OCR results whose detections survived NMS (match by label+center_x)
        std::vector<OcrResult> results;
        for (const auto& det : dets) {
            float cx = (det.x1 + det.x2) * 0.5f;
            for (const auto& r : all_results) {
                if (r.label == det.label && std::fabs(r.center_x - cx) < 1.0f) {
                    results.push_back(r);
                    break;
                }
            }
        }

        Parameters sb = buildScoreboardJson(results, all_results);
        cached_scoreboard_json_ = sb.dump();
        av_dict_set(&frm.raw()->metadata, output_metadata_key_.c_str(), cached_scoreboard_json_.c_str(), 0);

        if (debug_log_every_n_ > 0) {
            std::string det_detail;
            for (const auto& r : results) {
                if (!det_detail.empty()) det_detail += ", ";
                det_detail += r.label + "=\"" + r.text + "\"(" + std::to_string(r.mean_conf).substr(0,4) + ")";
            }
            logstream << "scoreboard_ocr: frame=" << frame_counter_
                      << " dets=" << dets.size()
                      << " results=" << results.size()
                      << " [" << det_detail << "]"
                      << " json=" << cached_scoreboard_json_;
        }

        this->sink_->put(frm);
    }

    static std::shared_ptr<ScoreboardOcr> create(NodeCreationInfo& nci) {
        EdgeManager& edges = nci.edges;
        const Parameters& params = nci.params;
        auto r = NodeSISO<av::VideoFrame, av::VideoFrame>::template createCommon<ScoreboardOcr>(edges, params);

        if (params.count("detection_metadata_key")) r->detection_metadata_key_ = params["detection_metadata_key"].get<std::string>();
        if (params.count("output_metadata_key")) r->output_metadata_key_ = params["output_metadata_key"].get<std::string>();
        if (params.count("min_conf")) r->min_conf_ = params["min_conf"].get<float>();
        if (params.count("nms_iou_thresh")) r->nms_iou_thresh_ = params["nms_iou_thresh"].get<float>();
        if (params.count("ocr_every_n")) r->ocr_every_n_ = std::max(1, params["ocr_every_n"].get<int>());
        if (params.count("debug_log_every_n")) r->debug_log_every_n_ = params["debug_log_every_n"].get<int>();
        if (params.count("scoreboard_labels")) {
            r->scoreboard_labels_.clear();
            for (const auto& item : params["scoreboard_labels"]) r->scoreboard_labels_.push_back(item.get<std::string>());
        }

        if (!params.count("ocr_model")) {
            throw Error("scoreboard_ocr: ocr_model param required (path to TRT .plan)");
        }
        if (!params.count("ocr_keys")) {
            throw Error("scoreboard_ocr: ocr_keys param required (path to keys .txt)");
        }
        r->keys_file_ = params["ocr_keys"].get<std::string>();

        yolo_base::ModelRunner model;
        model.engine_path = params["ocr_model"].get<std::string>();
        model.engine_name = std::filesystem::path(model.engine_path).filename().string();
        r->models_.push_back(std::move(model));

        return r;
    }
};

DECLNODE(scoreboard_ocr, ScoreboardOcr)
