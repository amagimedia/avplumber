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
    std::string raw_label;
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

// Limit detections per class: Team Name/Points keep 2, scalar fields keep 1.
// Generic lower-third Text scan boxes intentionally keep more than one candidate.
void limitPerClass(std::vector<ScoreboardDetection>& dets, int max_text_detections) {
    std::unordered_map<std::string, int> counts;
    std::vector<ScoreboardDetection> kept;
    for (auto& d : dets) {
        int max_per_class = 1;
        if (d.label == "Team Name" || d.label == "Team Points") {
            max_per_class = 2;
        } else if (d.label == "Text") {
            max_per_class = std::max(1, max_text_detections);
        }
        if (counts[d.label] < max_per_class) {
            counts[d.label]++;
            kept.push_back(std::move(d));
        }
    }
    dets = std::move(kept);
}

struct OcrResult {
    std::string label;
    std::string raw_label;
    std::string text;
    float mean_conf = 0.0f;
    float det_conf = 0.0f;
    float center_x = 0.0f;
    float center_y = 0.0f;
    float x1 = 0.0f;
    float y1 = 0.0f;
    float x2 = 0.0f;
    float y2 = 0.0f;
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

class ScoreboardOcr : public NodeSISO<av::VideoFrame, av::VideoFrame>, public yolo_base::CudaInferTrtBase, public ReportsFinishByFlag {
    std::string detection_metadata_key_ = "yolo_players";
    std::string output_metadata_key_ = "scoreboard";
    std::vector<std::string> scoreboard_labels_ = {"Period", "Shot Clock", "Team Name", "Team Points", "Time Remaining", "Text"};
    float min_conf_ = 0.4f;
    float nms_iou_thresh_ = 0.3f;
    int ocr_every_n_ = 25;
    int debug_log_every_n_ = 0;
    std::string keys_file_;
    float crop_pad_x_team_name_rel_ = 0.22f;
    float crop_pad_y_team_name_rel_ = 0.18f;
    float crop_pad_x_points_rel_ = 0.60f;
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
    int inferred_score_confirm_samples_ = 2;
    float inferred_score_min_candidate_conf_ = 0.22f;
    int max_text_detections_ = 32;
    bool clock_anchor_score_scan_ = true;
    int clock_anchor_score_max_crops_ = 160;
    float clock_anchor_score_min_clock_conf_ = 0.45f;
    std::vector<float> clock_anchor_score_x_offsets_rel_ = {
        -0.285f, -0.245f, -0.205f, -0.165f, -0.125f,
        -0.105f, -0.075f, -0.050f, -0.025f,
         0.025f,  0.050f,  0.075f,  0.105f, 0.135f
    };
    std::vector<float> clock_anchor_score_y_offsets_rel_ = {-0.095f, -0.075f};
    std::vector<float> clock_anchor_score_widths_rel_ = {0.026f, 0.034f};
    std::vector<float> clock_anchor_score_heights_rel_ = {0.055f, 0.075f};
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
    int pending_inferred_score_a_ = -1;
    int pending_inferred_score_b_ = -1;
    int pending_inferred_score_hits_ = 0;

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
        if (label == "Basketball-Scoreboard") return "Scoreboard";
        if (label == "scoreboard_text" || label == "lower_third_text") return "Text";
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
            sd.raw_label = raw_label;
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

    static std::string upperAscii(std::string s) {
        for (char& c : s) c = (char)std::toupper((unsigned char)c);
        return s;
    }

    static bool parseLooseUnsigned(const std::string& text, int min_val, int max_val, int& out) {
        if (text.empty()) return false;
        std::string digits;
        digits.reserve(text.size());
        for (char c : text) {
            if (std::isdigit((unsigned char)c)) digits.push_back(c);
            else if (!std::isspace((unsigned char)c) && c != ':') return false;
        }
        if (digits.empty() || digits.size() > 3) return false;
        int value = 0;
        for (char c : digits) {
            value = value * 10 + (c - '0');
            if (value > max_val) return false;
        }
        if (value < min_val || value > max_val) return false;
        out = value;
        return true;
    }

    static bool parseStrictUnsigned(const std::string& text, int min_val, int max_val) {
        int out = -1;
        return parseLooseUnsigned(text, min_val, max_val, out);
    }

    // Find the first all-digit token in mixed text (e.g. "ATL 2 LAL 0" → 2).
    // Used when the score crop is wide enough to include adjacent team-name
    // letters; PPOCRv3 needs that horizontal context to read single digits well.
    static bool parseFirstDigitToken(const std::string& text, int min_val, int max_val, int& out) {
        size_t i = 0;
        while (i < text.size()) {
            if (!std::isdigit((unsigned char)text[i])) { ++i; continue; }
            size_t j = i;
            while (j < text.size() && std::isdigit((unsigned char)text[j])) ++j;
            const size_t len = j - i;
            if (len > 0 && len <= 3) {
                int v = 0;
                for (size_t k = i; k < j; ++k) v = v * 10 + (text[k] - '0');
                if (v >= min_val && v <= max_val) { out = v; return true; }
            }
            i = j;
        }
        return false;
    }

    static bool parseStandaloneUnsigned(const std::string& text, int min_val, int max_val, int& out) {
        if (text.empty()) return false;
        std::string digits;
        digits.reserve(text.size());
        for (char c : text) {
            if (std::isdigit((unsigned char)c)) {
                digits.push_back(c);
            } else if (!std::isspace((unsigned char)c)) {
                return false;
            }
        }
        if (digits.empty() || digits.size() > 3) return false;
        int value = 0;
        for (char c : digits) {
            value = value * 10 + (c - '0');
            if (value > max_val) return false;
        }
        if (value < min_val || value > max_val) return false;
        out = value;
        return true;
    }

    static bool parseGameClockSec(const std::string& text, int& out) {
        const std::string up = upperAscii(text);
        if (up.find("AM") != std::string::npos || up.find("PM") != std::string::npos) return false;
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
            if (ss < 60) {
                out = mm * 60 + ss;
                return true;
            }
        }
        return false;
    }

    static bool looksLikeGameClock(const std::string& text) {
        int sec = -1;
        return parseGameClockSec(text, sec);
    }

    static std::string compactAlnumUpper(const std::string& text) {
        const std::string up = upperAscii(text);
        std::string out;
        out.reserve(up.size());
        for (char c : up) {
            if (std::isalnum((unsigned char)c)) out.push_back(c);
        }
        return out;
    }

    static int parseExplicitPeriodNum(const std::string& text) {
        const std::string tok = compactAlnumUpper(text);
        if (tok == "1ST" || tok == "IST" || tok == "Q1" || tok == "1Q" ||
            tok == "QTR1" || tok == "1QTR") {
            return 1;
        }
        if (tok == "2ND" || tok == "Q2" || tok == "2Q" || tok == "QTR2" || tok == "2QTR") return 2;
        if (tok == "3RD" || tok == "Q3" || tok == "3Q" || tok == "QTR3" || tok == "3QTR") return 3;
        if (tok == "4TH" || tok == "Q4" || tok == "4Q" || tok == "QTR4" || tok == "4QTR") return 4;
        if (tok == "OT" || tok == "1OT") return 5;
        if (tok == "2OT") return 6;
        if (tok == "3OT") return 7;
        if (tok == "4OT") return 8;
        return -1;
    }

    static int parsePeriodNum(const std::string& text) {
        int explicit_num = parseExplicitPeriodNum(text);
        if (explicit_num > 0) return explicit_num;

        const std::string tok = compactAlnumUpper(text);
        if (tok.size() == 1 && std::isdigit((unsigned char)tok[0])) {
            int n = tok[0] - '0';
            if (n >= 1 && n <= 9) return n;
        }
        return -1;
    }

    static bool hasExplicitPeriodToken(const std::string& text) {
        return parseExplicitPeriodNum(text) > 0;
    }

    static bool hasGenericTextPeriodToken(const std::string& text) {
        int n = parseExplicitPeriodNum(text);
        return n >= 1 && n <= 4;
    }

    static bool looksLikePeriod(const std::string& text) {
        return parsePeriodNum(text) > 0;
    }

    static bool rawLabelIs(const OcrResult& r, const std::string& label) {
        return r.raw_label == label || r.label == label;
    }

    static bool rawLabelIs(const ScoreboardDetection& d, const std::string& label) {
        return d.raw_label == label || d.label == label;
    }

    int ocrQualityScore(const std::string& label, const OcrResult& r) const {
        int score = 0;
        if (r.text.empty()) return score;
        if (label == "Team Name") {
            if (!extractAbbrevCandidates(r.text).empty()) score += 1000;
        } else if (label == "Team Points") {
            int v = -1;
            if (parseStrictUnsigned(r.text, 0, 199) || parseFirstDigitToken(r.text, 0, 199, v)) score += 1000;
        } else if (label == "Shot Clock") {
            if (parseStrictUnsigned(r.text, 0, 24)) score += 1000;
        } else if (label == "Time Remaining") {
            if (looksLikeGameClock(r.text)) score += 1000;
        } else if (label == "Period") {
            if (looksLikePeriod(r.text)) score += 1000;
        } else if (label == "Text") {
            int value = -1;
            if (looksLikeGameClock(r.text) ||
                hasGenericTextPeriodToken(r.text) ||
                parseStandaloneUnsigned(r.text, 0, 199, value)) {
                score += 1000;
            }
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
                         float model_w, float model_h, float center_x_override) {
        OcrResult result;
        result.label = det.label;
        result.raw_label = det.raw_label;
        result.det_conf = det.conf;
        result.center_x = center_x_override;
        result.center_y = bboxCenterY(det);
        result.x1 = det.x1;
        result.y1 = det.y1;
        result.x2 = det.x2;
        result.y2 = det.y2;

        const double scale_x = model_w > 0.0f ? (double)frm.width() / (double)model_w : 1.0;
        const double scale_y = model_h > 0.0f ? (double)frm.height() / (double)model_h : 1.0;

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
                               float model_w, float model_h) {
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
            OcrResult cur = runOcrOnce(frm, var, model_w, model_h, center_x);
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
            float add = std::max(0.05f, r.mean_conf) * labelBonus(r.label);
            if (r.raw_label == "team_1" || r.raw_label == "team_2") add *= 2.0f;
            for (const auto& abbrev : cands) {
                auto& ev = team_evidence_[abbrev];
                if (r.raw_label == "team_1" || (r.raw_label != "team_2" && r.center_x < mid_x)) {
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

    struct BestTextField {
        const OcrResult* result = nullptr;
        int value = -1;
        float score = -1.0f;
    };

    static void considerBest(BestTextField& best, const OcrResult& r, int value, float score) {
        if (score > best.score) {
            best.result = &r;
            best.value = value;
            best.score = score;
        }
    }

    void inferTextFields(const std::vector<OcrResult>& all_results,
                         BestTextField& game_clock,
                         BestTextField& shot_clock,
                         BestTextField& period) const {
        for (const auto& r : all_results) {
            if (r.text.empty()) continue;
            int sec = -1;
            if (parseGameClockSec(r.text, sec)) {
                float score = r.mean_conf;
                if (rawLabelIs(r, "time") || rawLabelIs(r, "Time Remaining")) score += 3.0f;
                if (rawLabelIs(r, "countdown") || rawLabelIs(r, "Shot Clock")) score -= 1.0f;
                considerBest(game_clock, r, sec, score);
            }

            int shot_sec = -1;
            if (!rawLabelIs(r, "score_anchor") && parseLooseUnsigned(r.text, 0, 24, shot_sec)) {
                float score = r.mean_conf;
                if (rawLabelIs(r, "countdown") || rawLabelIs(r, "Shot Clock")) score += 3.0f;
                if (rawLabelIs(r, "score_1") || rawLabelIs(r, "score_2") || rawLabelIs(r, "Team Points")) score -= 2.0f;
                if (rawLabelIs(r, "time") || rawLabelIs(r, "Time Remaining")) score -= 1.0f;
                considerBest(shot_clock, r, shot_sec, score);
            }

            int period_num = parsePeriodNum(r.text);
            if (period_num > 0) {
                if (rawLabelIs(r, "score_anchor")) continue;
                if (r.label == "Text" && !hasGenericTextPeriodToken(r.text)) continue;
                float score = r.mean_conf;
                if (rawLabelIs(r, "quarter") || rawLabelIs(r, "Period")) score += 3.0f;
                considerBest(period, r, period_num, score);
            }
        }
    }

    struct PointCandidate {
        const OcrResult* result = nullptr;
        int value = -1;
        float score = -1.0f;
    };

    struct BestPointPair {
        const OcrResult* left = nullptr;
        const OcrResult* right = nullptr;
        int left_value = -1;
        int right_value = -1;
        float score = -1.0f;
    };

    static bool sameSpatialResult(const OcrResult& a, const OcrResult* b) {
        if (!b) return false;
        const float acx = (a.x1 + a.x2) * 0.5f;
        const float acy = (a.y1 + a.y2) * 0.5f;
        const float bcx = (b->x1 + b->x2) * 0.5f;
        const float bcy = (b->y1 + b->y2) * 0.5f;
        const float dx = acx - bcx;
        const float dy = acy - bcy;
        if (std::sqrt(dx * dx + dy * dy) <= 2.0f) return true;
        return bboxIoU(a.x1, a.y1, a.x2, a.y2, b->x1, b->y1, b->x2, b->y2) > 0.55f;
    }

    static float closeness01(float dist, float max_dist) {
        if (max_dist <= 0.0f) return 0.0f;
        return std::clamp(1.0f - dist / max_dist, 0.0f, 1.0f);
    }

    static bool isPointLabel(const OcrResult& r) {
        return rawLabelIs(r, "score_1") || rawLabelIs(r, "score_2") ||
               rawLabelIs(r, "Team Points");
    }

    BestPointPair inferPointFields(const std::vector<OcrResult>& all_results,
                                   double model_w,
                                   double model_h,
                                   const BestTextField& game_clock,
                                   const BestTextField& shot_clock,
                                   const BestTextField& period) const {
        BestPointPair empty;
        if (!game_clock.result || game_clock.result->mean_conf < 0.45f || game_clock.value < 0) {
            return empty;
        }

        std::vector<PointCandidate> candidates;
        const float mw = (float)model_w;
        const float mh = (float)model_h;
        const float clock_x = game_clock.result->center_x;
        const float clock_y = game_clock.result->center_y;
        const float min_score_row_above_clock = 0.025f * mh;
        const float max_score_row_above_clock = 0.20f * mh;

        for (const auto& r : all_results) {
            if (r.text.empty()) continue;
            int value = -1;
            if (!parseStandaloneUnsigned(r.text, 0, 199, value)) {
                // Score crops widened to capture team-name context produce mixed
                // text like "ATL 2"; pull the first digit token only when the
                // detection is actually labeled as a score box.
                const bool is_score_label = rawLabelIs(r, "Team Points") ||
                                            rawLabelIs(r, "score_1") ||
                                            rawLabelIs(r, "score_2") ||
                                            rawLabelIs(r, "score_anchor");
                if (!is_score_label) continue;
                if (!parseFirstDigitToken(r.text, 0, 199, value)) continue;
            }
            if (sameSpatialResult(r, game_clock.result)) continue;
            const bool score_anchor = rawLabelIs(r, "score_anchor");
            if (score_anchor && value > 9) continue;
            if (!score_anchor && r.label == "Text" &&
                period.value == 1 && game_clock.value >= 600 && value > 30) {
                // Early first-quarter scorebugs are often next to small logos or ad glyphs.
                // Generic scan crops can merge that extra digit with the true score ("92"
                // instead of "2"). A 30+ point score in the first two minutes is not plausible.
                continue;
            }
            const float min_candidate_conf = score_anchor
                ? std::min(inferred_score_min_candidate_conf_, 0.10f)
                : inferred_score_min_candidate_conf_;
            if (r.mean_conf < min_candidate_conf) continue;
            if (std::fabs(r.center_x - clock_x) > 0.35f * mw) continue;

            const bool explicit_points =
                rawLabelIs(r, "score_1") || rawLabelIs(r, "score_2") ||
                rawLabelIs(r, "Team Points") || score_anchor;
            const float score_row_above_clock = clock_y - r.center_y;
            const bool score_row_near_clock =
                score_row_above_clock >= min_score_row_above_clock &&
                score_row_above_clock <= max_score_row_above_clock;
            const bool same_row_before_clock =
                r.center_x < clock_x - 0.030f * mw &&
                std::fabs(score_row_above_clock) <= 0.120f * mh;
            if (explicit_points) {
                if (!score_row_near_clock && !same_row_before_clock &&
                    (r.center_y < clock_y - 0.20f * mh || r.center_y > clock_y + 0.08f * mh)) {
                    continue;
                }
            } else {
                // In a layout-agnostic lower-third scan, scores are the numeric pair above
                // the game-clock/quarter row. Some scorebugs put both scores before the
                // clock on the same row, so keep those left-of-clock numbers for pair scoring.
                if (!score_row_near_clock && !same_row_before_clock) {
                    continue;
                }
            }

            float score = r.mean_conf;
            if (explicit_points) {
                score += 3.0f;
            } else if (r.label == "Text") {
                score += 0.2f;
                if (score_row_near_clock) {
                    score += 0.8f * closeness01(std::fabs(score_row_above_clock - 0.10f * mh), 0.10f * mh);
                } else if (same_row_before_clock) {
                    score += 0.5f * closeness01(std::fabs(score_row_above_clock), 0.08f * mh);
                }
            }
            if (rawLabelIs(r, "countdown") || rawLabelIs(r, "Shot Clock")) score -= 2.0f;
            if (sameSpatialResult(r, shot_clock.result)) score -= 2.5f;

            if (game_clock.result) {
                const float dx = std::fabs(r.center_x - game_clock.result->center_x);
                const float dy = std::fabs(r.center_y - game_clock.result->center_y);
                score += 0.8f * closeness01(dx, 0.32f * mw);
                score += 0.5f * closeness01(dy, 0.14f * mh);
            }
            if (shot_clock.result) {
                const float dy = std::fabs(r.center_y - shot_clock.result->center_y);
                score += 0.2f * closeness01(dy, 0.14f * mh);
            }
            candidates.push_back({&r, value, score});
        }

        BestPointPair best;
        const float min_dx = 0.035f * mw;
        const float max_dx = 0.25f * mw;
        const float max_row_dy = 0.035f * mh;
        for (size_t i = 0; i < candidates.size(); ++i) {
            for (size_t j = i + 1; j < candidates.size(); ++j) {
                const PointCandidate* a = &candidates[i];
                const PointCandidate* b = &candidates[j];
                if (a->result->center_x > b->result->center_x) std::swap(a, b);

                const float dx = std::fabs(a->result->center_x - b->result->center_x);
                const float dy = std::fabs(a->result->center_y - b->result->center_y);
                if (dx < min_dx || dx > max_dx || dy > max_row_dy) continue;
                if (bboxIoU(a->result->x1, a->result->y1, a->result->x2, a->result->y2,
                            b->result->x1, b->result->y1, b->result->x2, b->result->y2) > 0.15f) {
                    continue;
                }

                if (game_clock.result) {
                    const float pair_cx = (a->result->center_x + b->result->center_x) * 0.5f;
                    const float pair_cy = (a->result->center_y + b->result->center_y) * 0.5f;
                    if (std::fabs(pair_cx - game_clock.result->center_x) > 0.22f * mw) continue;
                    const bool anchored_pair =
                        rawLabelIs(*a->result, "score_anchor") || rawLabelIs(*b->result, "score_anchor");
                    const float left_gap = game_clock.result->center_x - a->result->center_x;
                    const float right_gap = b->result->center_x - game_clock.result->center_x;
                    const float pair_above_clock = game_clock.result->center_y - pair_cy;
                    const bool above_flank_layout =
                        a->result->center_x < game_clock.result->center_x - 0.018f * mw &&
                        b->result->center_x > game_clock.result->center_x + 0.006f * mw;
                    const bool before_clock_layout =
                        b->result->center_x < game_clock.result->center_x - 0.030f * mw &&
                        dx >= 0.085f * mw &&
                        dx <= 0.340f * mw &&
                        pair_above_clock >= -0.120f * mh &&
                        pair_above_clock <= max_score_row_above_clock;

                    if (!above_flank_layout && !before_clock_layout) continue;
                    if (before_clock_layout && anchored_pair) continue;
                    const float pair_min_conf = std::min(a->result->mean_conf, b->result->mean_conf);
                    const float pair_max_conf = std::max(a->result->mean_conf, b->result->mean_conf);
                    const float pair_avg_conf = (a->result->mean_conf + b->result->mean_conf) * 0.5f;
                    if (anchored_pair && above_flank_layout) {
                        // Clock-anchored crops intentionally look near the time display, so they
                        // produce many plausible but weak digits from logos and clock separators.
                        // Keep this layout for compact scorebugs, but require a strong pair.
                        if (pair_avg_conf < 0.60f && !(pair_max_conf >= 0.90f && pair_min_conf >= 0.22f)) {
                            continue;
                        }
                    }
                    if (anchored_pair) {
                        if (above_flank_layout) {
                            // The clock-anchored scan probes boxes around the scoreboard clock.
                            // Digits too close to the clock on the right are often team logos
                            // decoded as numbers, while the actual right score sits in the
                            // next flanking score column.
                            if (right_gap < 0.055f * mw || dx < 0.085f * mw) continue;
                            if (left_gap > 0.14f * mw || right_gap > 0.18f * mw) continue;
                        } else if (before_clock_layout) {
                            if (left_gap < 0.10f * mw || left_gap > 0.34f * mw) continue;
                        }
                    }
                    const bool explicit_pair =
                        isPointLabel(*a->result) || isPointLabel(*b->result) || anchored_pair;
                    if (before_clock_layout && !explicit_pair &&
                        (a->result->mean_conf + b->result->mean_conf) * 0.5f < 0.45f) {
                        continue;
                    }
                    if (explicit_pair) {
                        if (pair_above_clock < -0.080f * mh ||
                            pair_above_clock > max_score_row_above_clock) {
                            continue;
                        }
                    } else if (!before_clock_layout &&
                               (pair_above_clock < min_score_row_above_clock ||
                                pair_above_clock > max_score_row_above_clock)) {
                        continue;
                    }
                    float pair_score = a->score + b->score;
                    pair_score += 1.5f * closeness01(dy, max_row_dy);
                    if (before_clock_layout) {
                        const float right_before_gap = game_clock.result->center_x - b->result->center_x;
                        pair_score += 0.9f * closeness01(std::fabs(dx - 0.180f * mw), 0.140f * mw);
                        pair_score += 0.9f * closeness01(std::fabs(left_gap - 0.260f * mw), 0.140f * mw);
                        pair_score += 0.8f * closeness01(std::fabs(right_before_gap - 0.105f * mw), 0.090f * mw);
                        pair_score += 0.8f * closeness01(std::fabs(pair_above_clock), 0.120f * mh);
                        pair_score += 2.4f * pair_avg_conf;
                    } else {
                        pair_score += 0.9f * closeness01(std::fabs(dx - 0.105f * mw), 0.085f * mw);
                    }
                    if (anchored_pair && above_flank_layout) {
                        pair_score += 0.6f * closeness01(std::fabs(left_gap - 0.035f * mw), 0.060f * mw);
                        pair_score += 1.1f * closeness01(std::fabs(right_gap - 0.085f * mw), 0.060f * mw);
                    } else if (!before_clock_layout) {
                        pair_score += 0.5f * closeness01(std::fabs(pair_cx - game_clock.result->center_x), 0.28f * mw);
                    }
                    pair_score += 0.6f * closeness01(std::fabs(pair_cy - game_clock.result->center_y), 0.14f * mh);
                    if (!explicit_pair && !before_clock_layout) {
                        pair_score += 0.8f * closeness01(std::fabs(pair_above_clock - 0.10f * mh), 0.10f * mh);
                    }
                    if (shot_clock.result) {
                        pair_score += 0.3f * closeness01(std::fabs(pair_cx - shot_clock.result->center_x), 0.35f * mw);
                    }

                    if (pair_score > best.score) {
                        best.left = a->result;
                        best.right = b->result;
                        best.left_value = a->value;
                        best.right_value = b->value;
                        best.score = pair_score;
                    }
                }
            }
        }
        if (best.score < 3.8f) return empty;
        return best;
    }

    bool inferredScorePairIsConfirmed(const BestPointPair& pair) {
        if (!pair.left || !pair.right || pair.left_value < 0 || pair.right_value < 0) {
            pending_inferred_score_a_ = -1;
            pending_inferred_score_b_ = -1;
            pending_inferred_score_hits_ = 0;
            return false;
        }
        if (pair.left_value == pending_inferred_score_a_ && pair.right_value == pending_inferred_score_b_) {
            pending_inferred_score_hits_++;
        } else {
            pending_inferred_score_a_ = pair.left_value;
            pending_inferred_score_b_ = pair.right_value;
            pending_inferred_score_hits_ = 1;
        }
        return pending_inferred_score_hits_ >= inferred_score_confirm_samples_;
    }

    void appendClockAnchoredScoreOcr(const av::VideoFrame& frm,
                                     std::vector<OcrResult>& all_results,
                                     double model_w,
                                     double model_h) {
        if (!clock_anchor_score_scan_ || clock_anchor_score_max_crops_ <= 0) return;

        BestTextField game_clock, shot_clock, period;
        inferTextFields(all_results, game_clock, shot_clock, period);
        if (!game_clock.result || game_clock.result->mean_conf < clock_anchor_score_min_clock_conf_) return;

        const float mw = (float)model_w;
        const float mh = (float)model_h;
        const float clock_x = game_clock.result->center_x;
        const float clock_y = game_clock.result->center_y;
        int crops = 0;

        for (float y_off : clock_anchor_score_y_offsets_rel_) {
            for (float x_off : clock_anchor_score_x_offsets_rel_) {
                for (float h_rel : clock_anchor_score_heights_rel_) {
                    for (float w_rel : clock_anchor_score_widths_rel_) {
                        if (crops >= clock_anchor_score_max_crops_) return;
                        const float w = std::max(6.0f, w_rel * mw);
                        const float h = std::max(10.0f, h_rel * mh);
                        const float cx = clock_x + x_off * mw;
                        const float cy = clock_y + y_off * mh;

                        ScoreboardDetection det;
                        det.label = "Team Points";
                        det.raw_label = "score_anchor";
                        det.conf = 0.85f;
                        det.cls = -1;
                        det.x1 = std::clamp(cx - w * 0.5f, 0.0f, mw);
                        det.y1 = std::clamp(cy - h * 0.5f, 0.0f, mh);
                        det.x2 = std::clamp(cx + w * 0.5f, 0.0f, mw);
                        det.y2 = std::clamp(cy + h * 0.5f, 0.0f, mh);
                        if (det.x2 <= det.x1 || det.y2 <= det.y1) continue;
                        OcrResult probe_box;
                        probe_box.x1 = det.x1;
                        probe_box.y1 = det.y1;
                        probe_box.x2 = det.x2;
                        probe_box.y2 = det.y2;
                        if (sameSpatialResult(probe_box, game_clock.result)) {
                            continue;
                        }

                        ++crops;
                        OcrResult r = runBestOcrOnCrop(frm, det, (float)model_w, (float)model_h);
                        int value = -1;
                        const float min_anchor_conf = std::min(inferred_score_min_candidate_conf_, 0.10f);
                        if (!r.text.empty() &&
                            r.mean_conf >= min_anchor_conf &&
                            parseStandaloneUnsigned(r.text, 0, 9, value)) {
                            all_results.push_back(std::move(r));
                        }
                    }
                }
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
                                    double model_w,
                                    double model_h) {
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

        auto by_x_result = [](const OcrResult* a, const OcrResult* b) { return a->center_x < b->center_x; };
        std::sort(names.begin(), names.end(), by_x_result);
        std::sort(points.begin(), points.end(), by_x_result);

        BestTextField best_game_clock, best_shot_clock, best_period;
        inferTextFields(all_results, best_game_clock, best_shot_clock, best_period);
        BestPointPair inferred_points = inferPointFields(all_results, model_w, model_h, best_game_clock, best_shot_clock, best_period);
        const bool inferred_points_ready = inferredScorePairIsConfirmed(inferred_points);

        // Emit team_a/team_b purely by spatial position. Keep raw OCR text when abbrev
        // extraction fails — downstream consumers (game_state, external LLMs) can still use
        // the text, and emitting raw text unlocks points on frames where one name OCR missed.
        auto nameField = [&](const OcrResult* r) -> Parameters {
            auto cands = extractAbbrevCandidates(r->text);
            const std::string& text = cands.empty() ? r->text : cands.front();
            return Parameters{{"text", text}, {"conf", r->mean_conf}};
        };

        const OcrResult* points_a = points.size() >= 1 ? points.front() : (inferred_points_ready ? inferred_points.left : nullptr);
        const OcrResult* points_b = points.size() >= 2 ? points.back() : (inferred_points_ready ? inferred_points.right : nullptr);

        if (!names.empty() || !points.empty()) {
            sb["team_a"] = Parameters::object();
            if (!names.empty()) sb["team_a"]["name"] = nameField(names.front());
            if (points_a) sb["team_a"]["points"] = {{"text", points_a->text}, {"conf", points_a->mean_conf}};
        }
        if (names.size() >= 2 || points.size() >= 2 || (inferred_points_ready && inferred_points.left && inferred_points.right)) {
            sb["team_b"] = Parameters::object();
            if (names.size() >= 2) sb["team_b"]["name"] = nameField(names.back());
            if (points_b) sb["team_b"]["points"] = {{"text", points_b->text}, {"conf", points_b->mean_conf}};
        }
        if (points.size() < 2 && inferred_points_ready && inferred_points.left && inferred_points.right) {
            if (!sb.contains("team_a") || !sb["team_a"].is_object()) sb["team_a"] = Parameters::object();
            sb["team_a"]["points"] = {{"text", inferred_points.left->text}, {"conf", inferred_points.left->mean_conf}};
            if (!sb.contains("team_b") || !sb["team_b"].is_object()) sb["team_b"] = Parameters::object();
            sb["team_b"]["points"] = {{"text", inferred_points.right->text}, {"conf", inferred_points.right->mean_conf}};
            sb["score_source"] = "lower_third_scan";
            sb["score_pair_conf"] = inferred_points.score;
        }

        if (best_period.result) {
            sb["period"] = {{"text", best_period.result->text}, {"conf", best_period.result->mean_conf}};
        }
        if (best_game_clock.result) {
            sb["time_remaining"] = {{"text", best_game_clock.result->text}, {"conf", best_game_clock.result->mean_conf}};
        }
        if (best_shot_clock.result) {
            sb["shot_clock"] = {{"text", best_shot_clock.result->text}, {"conf", best_shot_clock.result->mean_conf}};
        }

        return sb;
    }

public:
    using NodeSISO<av::VideoFrame, av::VideoFrame>::NodeSISO;
    bool consumeEofIfPresent() override {
        return false;
    }

    ~ScoreboardOcr() {
        if (cu_ctx_) {
            cuCtxSetCurrent(cu_ctx_);
            if (crop_module_) cuModuleUnload(crop_module_);
        }
    }

    void process() override {
        av::VideoFrame frm = this->source_->get();
        if (isEofMarker(frm)) {
            cached_scoreboard_json_.clear();
            frame_counter_ = 0;
            locked_team_a_.clear();
            locked_team_b_.clear();
            lock_inconsistent_frames_ = 0;
            ocr_sample_counter_ = 0;
            hard_lock_active_ = false;
            pending_inferred_score_a_ = -1;
            pending_inferred_score_b_ = -1;
            pending_inferred_score_hits_ = 0;
            team_evidence_.clear();
            pair_evidence_.clear();
            stable_boxes_.clear();
            this->sink_->put(frm);
            this->finished_ = true;
            return;
        }
        if (!frm) return;

        ++frame_counter_;
        const AVFrame* raw = frm.raw();

        if ((frame_counter_ % (uint64_t)ocr_every_n_) != 1 && !cached_scoreboard_json_.empty()) {
            if (raw) av_dict_set(&frm.raw()->metadata, output_metadata_key_.c_str(), cached_scoreboard_json_.c_str(), 0);
            this->sink_->put(frm);
            return;
        }

        if (!raw) {
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

        if (cu_ctx_) cuCtxSetCurrent(cu_ctx_);

        const float model_w_f = (float)model_w;
        const float model_h_f = (float)model_h;

        std::vector<OcrResult> all_results;
        std::vector<OcrResult> results;
        // OCR all detections (pre-NMS) so we can scan everything for team abbreviations.
        for (const auto& det : dets) {
            OcrResult r = runBestOcrOnCrop(frm, det, model_w_f, model_h_f);
            if (!r.text.empty()) all_results.push_back(std::move(r));
        }

        appendClockAnchoredScoreOcr(frm, all_results, model_w, model_h);

        // NMS + per-class limits for the main scoreboard fields.
        nmsScoreboard(dets, nms_iou_thresh_);
        limitPerClass(dets, max_text_detections_);
        stabilizeDetections(dets, model_w_f, model_h_f);

        // OCR stabilized post-NMS boxes for the main scoreboard fields.
        for (const auto& det : dets) {
            OcrResult r = runBestOcrOnCrop(frm, det, model_w_f, model_h_f);
            if (!r.text.empty()) results.push_back(std::move(r));
        }

        Parameters sb = buildScoreboardJson(results, all_results, model_w, model_h);
        cached_scoreboard_json_ = sb.dump();
        av_dict_set(&frm.raw()->metadata, output_metadata_key_.c_str(), cached_scoreboard_json_.c_str(), 0);

        if (debug_log_every_n_ > 0) {
            std::string det_detail;
            for (const auto& r : results) {
                if (!det_detail.empty()) det_detail += ", ";
                det_detail += r.raw_label + "/" + r.label + "=\"" + r.text + "\"(" + std::to_string(r.mean_conf).substr(0,4) + ")";
            }
            std::string anchor_detail;
            std::string numeric_detail;
            int anchor_numeric = 0;
            int numeric_count = 0;
            for (const auto& r : all_results) {
                int value = -1;
                if (!parseStandaloneUnsigned(r.text, 0, 199, value)) continue;
                ++numeric_count;
                if (numeric_count <= 20) {
                    if (!numeric_detail.empty()) numeric_detail += ", ";
                    numeric_detail += r.raw_label + ":" + r.text + "@" +
                                      std::to_string((int)std::lround(r.center_x)) +
                                      "," + std::to_string((int)std::lround(r.center_y)) +
                                      "(" + std::to_string(r.mean_conf).substr(0,4) + ")";
                }
                if (rawLabelIs(r, "score_anchor")) {
                    ++anchor_numeric;
                    if (anchor_numeric <= 12) {
                        if (!anchor_detail.empty()) anchor_detail += ", ";
                        anchor_detail += r.text + "@" + std::to_string((int)std::lround(r.center_x)) +
                                         "," + std::to_string((int)std::lround(r.center_y)) +
                                         "(" + std::to_string(r.mean_conf).substr(0,4) + ")";
                    }
                }
            }
            logstream << "scoreboard_ocr: frame=" << frame_counter_
                      << " dets=" << dets.size()
                      << " results=" << results.size()
                      << " all_results=" << all_results.size()
                      << " score_anchor_numeric=" << anchor_numeric
                      << " score_anchor=[" << anchor_detail << "]"
                      << " numeric=[" << numeric_detail << "]"
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
        if (params.count("inferred_score_confirm_samples")) r->inferred_score_confirm_samples_ = std::max(1, params["inferred_score_confirm_samples"].get<int>());
        if (params.count("inferred_score_min_candidate_conf")) r->inferred_score_min_candidate_conf_ = params["inferred_score_min_candidate_conf"].get<float>();
        if (params.count("max_text_detections")) r->max_text_detections_ = std::max(1, params["max_text_detections"].get<int>());
        if (params.count("clock_anchor_score_scan")) r->clock_anchor_score_scan_ = params["clock_anchor_score_scan"].get<bool>();
        if (params.count("clock_anchor_score_max_crops")) r->clock_anchor_score_max_crops_ = std::max(0, params["clock_anchor_score_max_crops"].get<int>());
        if (params.count("clock_anchor_score_min_clock_conf")) r->clock_anchor_score_min_clock_conf_ = params["clock_anchor_score_min_clock_conf"].get<float>();
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
