#include "../../node_common.hpp"
#include "../common/infer_trt_base.hpp"

extern "C" {
#include <libavutil/dict.h>
}

#include "../../../../objs/src/nodes/neural_net/preprocess/nv12_crop_resize_pad.ptx.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
#include <limits>
#include <string>
#include <unordered_map>
#include <unordered_set>
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
    float crop_pad_x_team_name_rel_ = 0.22f;
    float crop_pad_y_team_name_rel_ = 0.18f;
    float crop_pad_x_points_rel_ = 0.16f;
    float crop_pad_y_points_rel_ = 0.18f;
    float crop_pad_x_clock_rel_ = 0.12f;
    float crop_pad_y_clock_rel_ = 0.16f;
    float crop_pad_x_default_rel_ = 0.10f;
    float crop_pad_y_default_rel_ = 0.12f;
    float stable_box_alpha_ = 0.70f;
    float stable_box_max_jump_rel_ = 0.12f;
    float team_name_label_bonus_ = 1.8f;
    float team_points_label_bonus_ = 1.2f;
    float generic_label_bonus_ = 0.8f;
    float evidence_decay_ = 0.96f;
    float lock_min_score_ = 3.5f;
    float lock_min_gap_ = 1.0f;
    float pair_lock_min_score_ = 2.5f;
    float pair_lock_min_gap_ = 0.6f;
    int pair_warmup_samples_ = 3;
    int pair_hard_lock_hits_ = 2;
    int unlock_bad_frames_ = 12;

    std::vector<std::string> keys_;
    CUmodule crop_module_ = nullptr;
    CUfunction crop_kernel_ = nullptr;
    bool ocr_initialized_ = false;
    uint64_t frame_counter_ = 0;
    std::string cached_scoreboard_json_;
    std::string locked_team_a_;
    std::string locked_team_b_;
    int lock_inconsistent_frames_ = 0;
    uint64_t ocr_sample_counter_ = 0;
    bool hard_lock_active_ = false;

    struct TeamEvidence {
        float left_score = 0.0f;
        float right_score = 0.0f;
        int left_hits = 0;
        int right_hits = 0;
        uint64_t last_frame_seen = 0;
    };
    std::unordered_map<std::string, TeamEvidence> team_evidence_;

    struct PairEvidence {
        float score = 0.0f;
        int hits = 0;
    };
    std::unordered_map<std::string, PairEvidence> pair_evidence_;
    struct StableBox {
        bool valid = false;
        float x1 = 0.0f, y1 = 0.0f, x2 = 0.0f, y2 = 0.0f;
        uint64_t last_frame = 0;
    };
    std::unordered_map<std::string, StableBox> stable_boxes_;

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

    static std::string canonLabel(const std::string& label) {
        // Map scoreboard-model class names to canonical internal labels so the rest
        // of the pipeline doesn't need to know which detector produced them.
        if (label == "team_1" || label == "team_2") return "Team Name";
        if (label == "score_1" || label == "score_2") return "Team Points";
        if (label == "time") return "Time Remaining";
        if (label == "quarter") return "Period";
        if (label == "countdown") return "Shot Clock";
        return label;
    }

    std::vector<ScoreboardDetection> extractScoreboardDetections(const Parameters& md) {
        std::vector<ScoreboardDetection> dets;
        if (!md.contains("detections") || !md["detections"].is_array()) return dets;
        for (const auto& det : md["detections"]) {
            if (!det.is_object()) continue;
            if (!det.contains("label") || !det["label"].is_string()) continue;
            const std::string raw_label = det["label"].get<std::string>();
            const std::string label = canonLabel(raw_label);
            bool match = false;
            for (const auto& want : scoreboard_labels_) {
                if (label == want || raw_label == want) { match = true; break; }
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

    static float bboxCenterX(const ScoreboardDetection& det) { return (det.x1 + det.x2) * 0.5f; }
    static float bboxCenterY(const ScoreboardDetection& det) { return (det.y1 + det.y2) * 0.5f; }
    static float bboxWidth(const ScoreboardDetection& det) { return det.x2 - det.x1; }
    static float bboxHeight(const ScoreboardDetection& det) { return det.y2 - det.y1; }

    static bool parseStrictUnsigned(const std::string& text, int min_val, int max_val) {
        if (text.empty()) return false;
        int value = 0;
        bool have_digit = false;
        for (char c : text) {
            if (std::isdigit((unsigned char)c)) {
                have_digit = true;
                value = value * 10 + (c - '0');
                if (value > max_val) return false;
            } else if (!std::isspace((unsigned char)c)) {
                return false;
            }
        }
        return have_digit && value >= min_val && value <= max_val;
    }

    static bool looksLikeGameClock(const std::string& text) {
        for (size_t i = 0; i + 3 < text.size(); ++i) {
            if (!std::isdigit((unsigned char)text[i])) continue;
            size_t colon = text.find(':', i);
            if (colon == std::string::npos || colon + 2 >= text.size()) continue;
            if (!std::isdigit((unsigned char)text[colon + 1]) || !std::isdigit((unsigned char)text[colon + 2])) continue;
            int mm = 0;
            bool ok = true;
            for (size_t j = i; j < colon; ++j) {
                if (!std::isdigit((unsigned char)text[j])) { ok = false; break; }
                mm = mm * 10 + (text[j] - '0');
            }
            if (!ok || mm > 12) continue;
            int ss = (text[colon + 1] - '0') * 10 + (text[colon + 2] - '0');
            if (ss < 60) return true;
        }
        return false;
    }

    static bool looksLikePeriod(const std::string& text) {
        std::string up;
        up.reserve(text.size());
        for (char c : text) up.push_back((char)std::toupper((unsigned char)c));
        return up.find("1ST") != std::string::npos || up.find("IST") != std::string::npos ||
               up.find("2ND") != std::string::npos || up.find("3RD") != std::string::npos ||
               up.find("4TH") != std::string::npos || up.find("OT") != std::string::npos;
    }

    int ocrQualityScore(const std::string& label, const OcrResult& r) const {
        int score = 0;
        if (r.text.empty()) return score;
        if (label == "Team Name") {
            if (!extractAbbrevCandidates(r.text).empty()) score += 1000;
        } else if (label == "Team Points") {
            if (parseStrictUnsigned(r.text, 0, 199)) score += 1000;
        } else if (label == "Shot Clock") {
            if (parseStrictUnsigned(r.text, 0, 24)) score += 1000;
        } else if (label == "Time Remaining") {
            if (looksLikeGameClock(r.text)) score += 1000;
        } else if (label == "Period") {
            if (looksLikePeriod(r.text)) score += 1000;
        }
        score += (int)std::lround(r.mean_conf * 100.0f);
        return score;
    }

    void cropPaddingForLabel(const std::string& label, float& pad_x_rel, float& pad_y_rel) const {
        if (label == "Team Name") {
            pad_x_rel = crop_pad_x_team_name_rel_;
            pad_y_rel = crop_pad_y_team_name_rel_;
        } else if (label == "Team Points") {
            pad_x_rel = crop_pad_x_points_rel_;
            pad_y_rel = crop_pad_y_points_rel_;
        } else if (label == "Shot Clock" || label == "Time Remaining" || label == "Period") {
            pad_x_rel = crop_pad_x_clock_rel_;
            pad_y_rel = crop_pad_y_clock_rel_;
        } else {
            pad_x_rel = crop_pad_x_default_rel_;
            pad_y_rel = crop_pad_y_default_rel_;
        }
    }

    ScoreboardDetection makeAdjustedCrop(const ScoreboardDetection& det, float model_w, float model_h,
                                         float shift_x_rel, float shift_y_rel, float scale_x_rel, float scale_y_rel) const {
        ScoreboardDetection out = det;
        float w = std::max(1.0f, bboxWidth(det));
        float h = std::max(1.0f, bboxHeight(det));
        float cx = bboxCenterX(det) + shift_x_rel * w;
        float cy = bboxCenterY(det) + shift_y_rel * h;
        float new_w = w * scale_x_rel;
        float new_h = h * scale_y_rel;
        out.x1 = std::max(0.0f, cx - new_w * 0.5f);
        out.y1 = std::max(0.0f, cy - new_h * 0.5f);
        out.x2 = std::min(model_w, cx + new_w * 0.5f);
        out.y2 = std::min(model_h, cy + new_h * 0.5f);
        return out;
    }

    OcrResult runOcrOnce(const av::VideoFrame& frm, const ScoreboardDetection& det,
                         double scale_x, double scale_y, float center_x_override) {
        OcrResult result;
        result.label = det.label;
        result.center_x = center_x_override;

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

    OcrResult runBestOcrOnCrop(const av::VideoFrame& frm, const ScoreboardDetection& det,
                               double scale_x, double scale_y, float model_w, float model_h) {
        float pad_x_rel = 0.0f, pad_y_rel = 0.0f;
        cropPaddingForLabel(det.label, pad_x_rel, pad_y_rel);
        const float center_x = bboxCenterX(det);

        std::vector<ScoreboardDetection> variants;
        variants.push_back(makeAdjustedCrop(det, model_w, model_h, 0.0f, 0.0f, 1.0f + pad_x_rel, 1.0f + pad_y_rel));

        if (det.label == "Team Name" || det.label == "Team Points" || det.label == "Shot Clock" || det.label == "Time Remaining") {
            variants.push_back(makeAdjustedCrop(det, model_w, model_h, -0.08f, 0.0f, 1.0f + pad_x_rel, 1.0f + pad_y_rel));
            variants.push_back(makeAdjustedCrop(det, model_w, model_h, 0.08f, 0.0f, 1.0f + pad_x_rel, 1.0f + pad_y_rel));
            variants.push_back(makeAdjustedCrop(det, model_w, model_h, 0.0f, -0.05f, 1.0f + pad_x_rel, 1.0f + pad_y_rel));
            variants.push_back(makeAdjustedCrop(det, model_w, model_h, 0.0f, 0.05f, 1.0f + pad_x_rel, 1.0f + pad_y_rel));
        }

        OcrResult best;
        int best_score = std::numeric_limits<int>::min();
        for (const auto& var : variants) {
            OcrResult cur = runOcrOnce(frm, var, scale_x, scale_y, center_x);
            int score = ocrQualityScore(det.label, cur);
            if (score > best_score) {
                best_score = score;
                best = std::move(cur);
            }
        }
        return best;
    }

    static std::string stableBoxKey(const std::string& label, int slot) {
        return label + "#" + std::to_string(slot);
    }

    void stabilizeDetections(std::vector<ScoreboardDetection>& dets, float model_w, float model_h) {
        std::unordered_map<std::string, std::vector<size_t>> by_label;
        for (size_t i = 0; i < dets.size(); ++i) by_label[dets[i].label].push_back(i);

        const float max_jump_px = stable_box_max_jump_rel_ * std::min(model_w, model_h);
        for (auto& [label, idxs] : by_label) {
            std::sort(idxs.begin(), idxs.end(), [&](size_t a, size_t b) { return bboxCenterX(dets[a]) < bboxCenterX(dets[b]); });
            for (size_t slot = 0; slot < idxs.size(); ++slot) {
                auto& det = dets[idxs[slot]];
                auto& st = stable_boxes_[stableBoxKey(label, (int)slot)];
                if (!st.valid || frame_counter_ <= st.last_frame + (uint64_t)ocr_every_n_ * 2ULL) {
                    const float cx_det = bboxCenterX(det), cy_det = bboxCenterY(det);
                    const float cx_st = (st.x1 + st.x2) * 0.5f, cy_st = (st.y1 + st.y2) * 0.5f;
                    const float jump = st.valid ? std::sqrt((cx_det - cx_st) * (cx_det - cx_st) + (cy_det - cy_st) * (cy_det - cy_st))
                                                : 0.0f;
                    if (!st.valid || jump > max_jump_px) {
                        st.x1 = det.x1; st.y1 = det.y1; st.x2 = det.x2; st.y2 = det.y2;
                    } else {
                        st.x1 = stable_box_alpha_ * st.x1 + (1.0f - stable_box_alpha_) * det.x1;
                        st.y1 = stable_box_alpha_ * st.y1 + (1.0f - stable_box_alpha_) * det.y1;
                        st.x2 = stable_box_alpha_ * st.x2 + (1.0f - stable_box_alpha_) * det.x2;
                        st.y2 = stable_box_alpha_ * st.y2 + (1.0f - stable_box_alpha_) * det.y2;
                    }
                } else {
                    st.x1 = det.x1; st.y1 = det.y1; st.x2 = det.x2; st.y2 = det.y2;
                }
                st.last_frame = frame_counter_;
                st.valid = true;
                det.x1 = std::max(0.0f, st.x1);
                det.y1 = std::max(0.0f, st.y1);
                det.x2 = std::min(model_w, st.x2);
                det.y2 = std::min(model_h, st.y2);
            }
        }
    }

    static bool isTeamAbbrev(const std::string& abbrev) {
        if (abbrev.length() < 2 || abbrev.length() > 3) return false;
        static const std::unordered_set<std::string> nba = {
            "ATL", "BOS", "BKN", "BRK", "CHA", "CHI", "CLE", "DAL", "DEN", "DET",
            "GS", "GSW", "HOU", "IND", "LAC", "LAL", "MEM", "MIA", "MIL", "MIN",
            "NOP", "NO", "NY", "NYK", "OKC", "ORL", "PHI", "PHX", "POR", "SAC",
            "SA", "SAS", "TOR", "UTA", "WAS"
        };
        static const std::vector<std::string> ignore = {
            "NBA", "NHL", "NFL", "MLB", "MLS", "ESPN", "TNT", "ABC", "CBS", "FOX",
            "1ST", "2ND", "3RD", "4TH", "OT",
            "PM", "AM", "ET", "PT", "CT", "MT",
            "FG", "FT", "PTS", "REB", "AST", "STL", "BLK", "TO", "PF",
            "SJ", "VS", "Q", "FGM", "FGA"
        };
        for (const auto& ign : ignore) {
            if (abbrev == ign) return false;
        }
        return nba.count(abbrev) > 0;
    }

    static std::vector<std::string> extractAbbrevCandidates(const std::string& text) {
        std::vector<std::string> out;
        std::unordered_set<std::string> seen;
        std::string up;
        up.reserve(text.size());
        for (char c : text) up.push_back((char)std::toupper((unsigned char)c));

        for (size_t i = 0; i < up.size();) {
            if (!std::isupper((unsigned char)up[i])) {
                ++i;
                continue;
            }
            size_t j = i;
            std::string run;
            while (j < up.size() && std::isupper((unsigned char)up[j]) && run.size() < 4) {
                run.push_back(up[j]);
                ++j;
            }
            if (isTeamAbbrev(run) && !seen.count(run)) {
                seen.insert(run);
                out.push_back(run);
            }
            i = std::max(j, i + 1);
        }
        return out;
    }

    struct ExtractedTeam {
        std::string abbrev;
        float center_x;
        float conf;
    };

    float labelBonus(const std::string& label) const {
        if (label == "Team Name") return team_name_label_bonus_;
        if (label == "Team Points") return team_points_label_bonus_;
        return generic_label_bonus_;
    }

    void decayEvidence() {
        for (auto it = team_evidence_.begin(); it != team_evidence_.end();) {
            it->second.left_score *= evidence_decay_;
            it->second.right_score *= evidence_decay_;
            if (it->second.left_score < 0.05f && it->second.right_score < 0.05f) {
                it = team_evidence_.erase(it);
            } else {
                ++it;
            }
        }
        for (auto it = pair_evidence_.begin(); it != pair_evidence_.end();) {
            it->second.score *= evidence_decay_;
            if (it->second.score < 0.05f) it = pair_evidence_.erase(it);
            else ++it;
        }
    }

    void updateTeamEvidence(const std::vector<OcrResult>& all_results, double model_w) {
        decayEvidence();
        const float mid_x = (float)(model_w * 0.5);
        for (const auto& r : all_results) {
            if (r.text.empty()) continue;
            const auto cands = extractAbbrevCandidates(r.text);
            if (cands.empty()) continue;
            const float add = std::max(0.05f, r.mean_conf) * labelBonus(r.label);
            for (const auto& abbrev : cands) {
                auto& ev = team_evidence_[abbrev];
                if (r.center_x < mid_x) {
                    ev.left_score += add;
                    ev.left_hits++;
                } else {
                    ev.right_score += add;
                    ev.right_hits++;
                }
                ev.last_frame_seen = frame_counter_;
            }
        }
    }

    struct SideWinner {
        std::string abbrev;
        float score = 0.0f;
        float second = 0.0f;
        int hits = 0;
    };

    SideWinner bestForSide(bool left) const {
        SideWinner w;
        for (const auto& [abbr, ev] : team_evidence_) {
            float score = left ? ev.left_score : ev.right_score;
            if (score > w.score) {
                w.second = w.score;
                w.score = score;
                w.abbrev = abbr;
                w.hits = left ? ev.left_hits : ev.right_hits;
            } else if (score > w.second) {
                w.second = score;
            }
        }
        return w;
    }

    static std::string makePairKey(const std::string& a, const std::string& b) {
        return a + "|" + b;
    }

    void updatePairEvidence(const std::vector<ExtractedTeam>& team_abbrevs) {
        if (team_abbrevs.size() < 2) return;
        const auto& a = team_abbrevs[0];
        const auto& b = team_abbrevs[1];
        if (a.abbrev.empty() || b.abbrev.empty() || a.abbrev == b.abbrev) return;
        auto& ev = pair_evidence_[makePairKey(a.abbrev, b.abbrev)];
        ev.score += std::max(0.05f, (a.conf + b.conf) * 0.5f);
        ev.hits++;
    }

    bool maybeLockFromBestPair() {
        std::string best_key;
        float best = 0.0f;
        float second = 0.0f;
        int best_hits = 0;
        for (const auto& [key, ev] : pair_evidence_) {
            if (ev.score > best) {
                second = best;
                best = ev.score;
                best_key = key;
                best_hits = ev.hits;
            } else if (ev.score > second) {
                second = ev.score;
            }
        }
        if (best_key.empty() || best_hits < 2 || best < pair_lock_min_score_ || (best - second) < pair_lock_min_gap_) {
            return false;
        }
        size_t sep = best_key.find('|');
        if (sep == std::string::npos) return false;
        locked_team_a_ = best_key.substr(0, sep);
        locked_team_b_ = best_key.substr(sep + 1);
        lock_inconsistent_frames_ = 0;
        return true;
    }

    void maybeUpdateLock() {
        if (hard_lock_active_) return;
        if (!locked_team_a_.empty() && !locked_team_b_.empty()) return;
        if (maybeLockFromBestPair()) return;
        SideWinner left = bestForSide(true);
        SideWinner right = bestForSide(false);

        if (left.abbrev.empty() || right.abbrev.empty() || left.abbrev == right.abbrev) return;

        const bool strong =
            left.score >= lock_min_score_ &&
            right.score >= lock_min_score_ &&
            (left.score - left.second) >= lock_min_gap_ &&
            (right.score - right.second) >= lock_min_gap_ &&
            left.hits >= 2 && right.hits >= 2;

        if (strong) {
            locked_team_a_ = left.abbrev;
            locked_team_b_ = right.abbrev;
            lock_inconsistent_frames_ = 0;
        }
    }

    void checkLockConsistency(const Parameters& sb) {
        if (hard_lock_active_) return;
        if (locked_team_a_.empty() || locked_team_b_.empty()) return;
        bool inconsistent = false;
        if (sb.contains("team_a") && sb["team_a"].contains("name")) {
            std::string a = sb["team_a"]["name"].value("text", std::string());
            if (!a.empty() && a != locked_team_a_) inconsistent = true;
        }
        if (sb.contains("team_b") && sb["team_b"].contains("name")) {
            std::string b = sb["team_b"]["name"].value("text", std::string());
            if (!b.empty() && b != locked_team_b_) inconsistent = true;
        }
        if (inconsistent) lock_inconsistent_frames_++;
        else lock_inconsistent_frames_ = 0;
        if (lock_inconsistent_frames_ >= unlock_bad_frames_) {
            locked_team_a_.clear();
            locked_team_b_.clear();
            lock_inconsistent_frames_ = 0;
        }
    }

    void applyLockedTeams(Parameters& sb, const std::vector<const OcrResult*>& points) {
        if (!locked_team_a_.empty()) {
            if (!sb.contains("team_a") || !sb["team_a"].is_object()) sb["team_a"] = Parameters::object();
            sb["team_a"]["name"] = {{"text", locked_team_a_}, {"conf", 1.0f}};
        }
        if (!locked_team_b_.empty()) {
            if (!sb.contains("team_b") || !sb["team_b"].is_object()) sb["team_b"] = Parameters::object();
            sb["team_b"]["name"] = {{"text", locked_team_b_}, {"conf", 1.0f}};
        }
        if (sb.contains("team_a") && points.size() >= 1) {
            sb["team_a"]["points"] = {{"text", points[0]->text}, {"conf", points[0]->mean_conf}};
        }
        if (sb.contains("team_b") && points.size() >= 2) {
            sb["team_b"]["points"] = {{"text", points[1]->text}, {"conf", points[1]->mean_conf}};
        }
    }

    void collectSideCandidates(const std::vector<OcrResult>& all_results, double model_w,
                               std::unordered_map<std::string, float>& left,
                               std::unordered_map<std::string, float>& right) const {
        const float mid_x = (float)(model_w * 0.5);
        for (const auto& r : all_results) {
            if (r.text.empty()) continue;
            const float score = std::max(0.05f, r.mean_conf) * labelBonus(r.label);
            for (const auto& abbrev : extractAbbrevCandidates(r.text)) {
                if (r.center_x < mid_x) left[abbrev] = std::max(left[abbrev], score);
                else right[abbrev] = std::max(right[abbrev], score);
            }
        }
    }

    void updatePairEvidenceFromCandidates(const std::unordered_map<std::string, float>& left,
                                          const std::unordered_map<std::string, float>& right) {
        if (left.empty() || right.empty()) return;
        for (const auto& [la, ls] : left) {
            for (const auto& [ra, rs] : right) {
                if (la == ra) continue;
                auto& ev = pair_evidence_[makePairKey(la, ra)];
                ev.score += 0.5f * (ls + rs);
                ev.hits++;
            }
        }
    }

    void maybeHardLockFromPairs() {
        if (hard_lock_active_ || ocr_sample_counter_ < (uint64_t)pair_warmup_samples_) return;
        std::string best_key;
        float best = 0.0f;
        float second = 0.0f;
        int best_hits = 0;
        for (const auto& [key, ev] : pair_evidence_) {
            if (ev.score > best) {
                second = best;
                best = ev.score;
                best_key = key;
                best_hits = ev.hits;
            } else if (ev.score > second) {
                second = ev.score;
            }
        }
        if (best_key.empty()) return;
        if (best < pair_lock_min_score_ || best_hits < pair_hard_lock_hits_ || (best - second) < pair_lock_min_gap_) return;
        size_t sep = best_key.find('|');
        if (sep == std::string::npos) return;
        locked_team_a_ = best_key.substr(0, sep);
        locked_team_b_ = best_key.substr(sep + 1);
        hard_lock_active_ = true;
        lock_inconsistent_frames_ = 0;
    }

    Parameters buildScoreboardJson(const std::vector<OcrResult>& results,
                                    const std::vector<OcrResult>& all_results,
                                    double model_w) {
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

        updateTeamEvidence(all_results, model_w);
        std::unordered_map<std::string, float> left_candidates;
        std::unordered_map<std::string, float> right_candidates;
        collectSideCandidates(all_results, model_w, left_candidates, right_candidates);
        if (!left_candidates.empty() || !right_candidates.empty()) {
            ++ocr_sample_counter_;
            updatePairEvidenceFromCandidates(left_candidates, right_candidates);
            maybeHardLockFromPairs();
        }

        // Scan ALL pre-NMS OCR results for NBA-like abbreviations anywhere in the text
        std::vector<ExtractedTeam> found_teams;
        for (const auto& r : all_results) {
            if (r.text.empty()) continue;
            for (const auto& abbrev : extractAbbrevCandidates(r.text)) {
                bool already_found = false;
                for (const auto& ft : found_teams) {
                    if (ft.abbrev == abbrev && std::fabs(ft.center_x - r.center_x) < 1.0f) {
                        already_found = true;
                        break;
                    }
                }
                if (!already_found) {
                    found_teams.push_back({abbrev, r.center_x, r.mean_conf});
                }
            }
        }

        // If explicit Team Name detections found real abbreviations, prefer those
        std::vector<ExtractedTeam> team_abbrevs;
        for (const auto* n : names) {
            const auto cands = extractAbbrevCandidates(n->text);
            if (!cands.empty()) {
                std::string abbrev = cands.front();
                team_abbrevs.push_back({abbrev, n->center_x, n->mean_conf});
            }
        }
        if (team_abbrevs.size() < 2 && found_teams.size() > team_abbrevs.size()) {
            team_abbrevs = std::move(found_teams);
        }

        // Sort by X: leftmost = team_a
        std::sort(team_abbrevs.begin(), team_abbrevs.end(),
                  [](const ExtractedTeam& a, const ExtractedTeam& b) { return a.center_x < b.center_x; });
        updatePairEvidence(team_abbrevs);

        auto by_x_result = [](const OcrResult* a, const OcrResult* b) { return a->center_x < b->center_x; };
        std::sort(names.begin(), names.end(), by_x_result);
        std::sort(points.begin(), points.end(), by_x_result);

        // Emit team_a/team_b purely by spatial position. Keep raw OCR text when abbrev
        // extraction fails — downstream consumers (game_state, external LLMs) can still use
        // the text, and emitting raw text unlocks points on frames where one name OCR missed.
        auto nameField = [&](const OcrResult* r) -> Parameters {
            auto cands = extractAbbrevCandidates(r->text);
            const std::string& text = cands.empty() ? r->text : cands.front();
            return Parameters{{"text", text}, {"conf", r->mean_conf}};
        };

        if (!names.empty() || !points.empty()) {
            sb["team_a"] = Parameters::object();
            if (!names.empty()) sb["team_a"]["name"] = nameField(names.front());
            if (!points.empty()) sb["team_a"]["points"] = {{"text", points.front()->text}, {"conf", points.front()->mean_conf}};
        }
        if (names.size() >= 2 || points.size() >= 2) {
            sb["team_b"] = Parameters::object();
            if (names.size() >= 2) sb["team_b"]["name"] = nameField(names.back());
            if (points.size() >= 2) sb["team_b"]["points"] = {{"text", points.back()->text}, {"conf", points.back()->mean_conf}};
        }

        checkLockConsistency(sb);
        maybeUpdateLock();
        if (!locked_team_a_.empty() || !locked_team_b_.empty()) {
            applyLockedTeams(sb, points);
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
            locked_team_a_.clear();
            locked_team_b_.clear();
            lock_inconsistent_frames_ = 0;
            ocr_sample_counter_ = 0;
            hard_lock_active_ = false;
            team_evidence_.clear();
            pair_evidence_.clear();
            stable_boxes_.clear();
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

        const float model_w_f = (float)model_w;
        const float model_h_f = (float)model_h;

        // OCR all detections (pre-NMS) so we can scan everything for team abbreviations
        std::vector<OcrResult> all_results;
        for (const auto& det : dets) {
            OcrResult r = runBestOcrOnCrop(frm, det, scale_x, scale_y, model_w_f, model_h_f);
            if (!r.text.empty()) all_results.push_back(std::move(r));
        }

        // NMS + per-class limits for the main scoreboard fields
        nmsScoreboard(dets, nms_iou_thresh_);
        limitPerClass(dets);
        stabilizeDetections(dets, model_w_f, model_h_f);

        // OCR stabilized post-NMS boxes for the main scoreboard fields
        std::vector<OcrResult> results;
        for (const auto& det : dets) {
            OcrResult r = runBestOcrOnCrop(frm, det, scale_x, scale_y, model_w_f, model_h_f);
            if (!r.text.empty()) results.push_back(std::move(r));
        }

        Parameters sb = buildScoreboardJson(results, all_results, model_w);
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
        if (params.count("crop_pad_x_team_name_rel")) r->crop_pad_x_team_name_rel_ = params["crop_pad_x_team_name_rel"].get<float>();
        if (params.count("crop_pad_y_team_name_rel")) r->crop_pad_y_team_name_rel_ = params["crop_pad_y_team_name_rel"].get<float>();
        if (params.count("crop_pad_x_points_rel")) r->crop_pad_x_points_rel_ = params["crop_pad_x_points_rel"].get<float>();
        if (params.count("crop_pad_y_points_rel")) r->crop_pad_y_points_rel_ = params["crop_pad_y_points_rel"].get<float>();
        if (params.count("crop_pad_x_clock_rel")) r->crop_pad_x_clock_rel_ = params["crop_pad_x_clock_rel"].get<float>();
        if (params.count("crop_pad_y_clock_rel")) r->crop_pad_y_clock_rel_ = params["crop_pad_y_clock_rel"].get<float>();
        if (params.count("crop_pad_x_default_rel")) r->crop_pad_x_default_rel_ = params["crop_pad_x_default_rel"].get<float>();
        if (params.count("crop_pad_y_default_rel")) r->crop_pad_y_default_rel_ = params["crop_pad_y_default_rel"].get<float>();
        if (params.count("stable_box_alpha")) r->stable_box_alpha_ = params["stable_box_alpha"].get<float>();
        if (params.count("stable_box_max_jump_rel")) r->stable_box_max_jump_rel_ = params["stable_box_max_jump_rel"].get<float>();
        if (params.count("team_name_label_bonus")) r->team_name_label_bonus_ = params["team_name_label_bonus"].get<float>();
        if (params.count("team_points_label_bonus")) r->team_points_label_bonus_ = params["team_points_label_bonus"].get<float>();
        if (params.count("generic_label_bonus")) r->generic_label_bonus_ = params["generic_label_bonus"].get<float>();
        if (params.count("evidence_decay")) r->evidence_decay_ = params["evidence_decay"].get<float>();
        if (params.count("lock_min_score")) r->lock_min_score_ = params["lock_min_score"].get<float>();
        if (params.count("lock_min_gap")) r->lock_min_gap_ = params["lock_min_gap"].get<float>();
        if (params.count("pair_lock_min_score")) r->pair_lock_min_score_ = params["pair_lock_min_score"].get<float>();
        if (params.count("pair_lock_min_gap")) r->pair_lock_min_gap_ = params["pair_lock_min_gap"].get<float>();
        if (params.count("pair_warmup_samples")) r->pair_warmup_samples_ = std::max(1, params["pair_warmup_samples"].get<int>());
        if (params.count("pair_hard_lock_hits")) r->pair_hard_lock_hits_ = std::max(1, params["pair_hard_lock_hits"].get<int>());
        if (params.count("unlock_bad_frames")) r->unlock_bad_frames_ = params["unlock_bad_frames"].get<int>();
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
