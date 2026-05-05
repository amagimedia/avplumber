#pragma once

#include "../../../node_common.hpp"
#include "../../common/yolo_side_data.hpp"
#include "types.hpp"

extern "C" {
#include <libavutil/dict.h>
#include <libavutil/frame.h>
}

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <map>
#include <string>
#include <unordered_map>
#include <unordered_set>

namespace metadata_dump {

class JsonFileWriter {
public:
    bool open(const std::string& path) {
        if (path.empty()) return false;
        out_.open(path, std::ios::out | std::ios::trunc);
        if (!out_) return false;
        opened_ = true;
        return true;
    }
    void writeLine(const std::string& json) {
        if (!opened_) return;
        out_ << json << '\n';
        out_.flush();
    }
    void close() {
        if (!opened_) return;
        out_.close();
        opened_ = false;
    }
    bool opened() const { return opened_; }

private:
    std::ofstream out_;
    bool opened_ = false;
};

inline Parameters tryParse(const AVFrame* raw, const std::string& key) {
    if (!raw || !raw->metadata) return {};
    AVDictionaryEntry* entry = av_dict_get(raw->metadata, key.c_str(), nullptr, 0);
    if (!entry || !entry->value) return {};
    try { return Parameters::parse(entry->value); } catch (...) { return {}; }
}

inline int ri(float v) { return (int)std::round(v); }

template <typename T>
inline T getOr(const Parameters& obj, const char* key, const T& fallback) {
    if (!obj.is_object() || !obj.contains(key) || obj[key].is_null()) return fallback;
    try {
        return obj[key].get<T>();
    } catch (...) {
        return fallback;
    }
}

// Per-frame source<->model scale. Set once on the first frame, then reused.
struct Scaler {
    int source_w = 0;
    int source_h = 0;
    float model_w = 960.0f;
    float model_h = 544.0f;
    float scale_x = 1.0f;
    float scale_y = 1.0f;

    void initFromFrame(int sw, int sh, float mw, float mh) {
        source_w = sw;
        source_h = sh;
        model_w = mw;
        model_h = mh;
        scale_x = (sw > 0 && mw > 0) ? (float)sw / mw : 1.0f;
        scale_y = (sh > 0 && mh > 0) ? (float)sh / mh : 1.0f;
    }
    int scaleX(float v) const { return ri(v * scale_x); }
    int scaleY(float v) const { return ri(v * scale_y); }
    int scaleDist(float v) const { return ri(v * 0.5f * (scale_x + scale_y)); }
    Parameters scaleBox(const Parameters& det) const {
        if (!det.contains("xyxy") || !det["xyxy"].is_array() || det["xyxy"].size() < 4) return nullptr;
        return Parameters::array({
            scaleX(det["xyxy"][0].get<float>()),
            scaleY(det["xyxy"][1].get<float>()),
            scaleX(det["xyxy"][2].get<float>()),
            scaleY(det["xyxy"][3].get<float>())
        });
    }
};

inline bool haveScore(const ScoreState& s) { return s.a >= 0 && s.b >= 0; }
inline bool sameScore(const ScoreState& a, const ScoreState& b) { return a.a == b.a && a.b == b.b; }

inline std::string attemptTypeForPoints(int points) {
    if (points == 2) return "2pt";
    if (points == 3) return "3pt";
    return "unknown";
}

inline int visualTeamIndex(const std::string& team) {
    if (team == "A") return 0;
    if (team == "B") return 1;
    return -1;
}

inline int scoreboardSideIndex(const std::string& side) {
    if (side == "team_a") return 0;
    if (side == "team_b") return 1;
    return -1;
}

inline std::string visualTeamName(int idx) {
    if (idx == 0) return "A";
    if (idx == 1) return "B";
    return "?";
}

inline std::string scoreboardSideName(int idx) {
    if (idx == 0) return "team_a";
    if (idx == 1) return "team_b";
    return "";
}

inline bool isKnownNbaAbbrev(const std::string& abbrev) {
    static const std::unordered_set<std::string> nba = {
        "ATL", "BOS", "BKN", "BRK", "CHA", "CHI", "CLE", "DAL", "DEN", "DET",
        "GS", "GSW", "HOU", "IND", "LAC", "LAL", "MEM", "MIA", "MIL", "MIN",
        "NOP", "NO", "NY", "NYK", "OKC", "ORL", "PHI", "PHX", "POR", "SAC",
        "SA", "SAS", "TOR", "UTA", "WAS"
    };
    return nba.count(abbrev) > 0;
}

inline std::string nbaFullName(const std::string& abbrev) {
    static const std::unordered_map<std::string, std::string> names = {
        {"ATL", "Atlanta Hawks"}, {"BOS", "Boston Celtics"}, {"BKN", "Brooklyn Nets"},
        {"CHA", "Charlotte Hornets"}, {"CHI", "Chicago Bulls"}, {"CLE", "Cleveland Cavaliers"},
        {"DAL", "Dallas Mavericks"}, {"DEN", "Denver Nuggets"}, {"DET", "Detroit Pistons"},
        {"GSW", "Golden State Warriors"}, {"HOU", "Houston Rockets"}, {"IND", "Indiana Pacers"},
        {"LAC", "LA Clippers"}, {"LAL", "Los Angeles Lakers"}, {"MEM", "Memphis Grizzlies"},
        {"MIA", "Miami Heat"}, {"MIL", "Milwaukee Bucks"}, {"MIN", "Minnesota Timberwolves"},
        {"NOP", "New Orleans Pelicans"}, {"NYK", "New York Knicks"}, {"OKC", "Oklahoma City Thunder"},
        {"ORL", "Orlando Magic"}, {"PHI", "Philadelphia 76ers"}, {"PHX", "Phoenix Suns"},
        {"POR", "Portland Trail Blazers"}, {"SAC", "Sacramento Kings"}, {"SAS", "San Antonio Spurs"},
        {"TOR", "Toronto Raptors"}, {"UTA", "Utah Jazz"}, {"WAS", "Washington Wizards"}
    };
    auto it = names.find(abbrev);
    return it == names.end() ? std::string() : it->second;
}

inline std::string cleanNbaAbbrev(const Parameters& obj, const char* key) {
    if (!obj.is_object() || !obj.contains(key) || !obj[key].is_string()) return {};
    std::string s = obj[key].get<std::string>();
    std::string up;
    up.reserve(s.size());
    for (char c : s) {
        if (std::isalpha((unsigned char)c)) up.push_back((char)std::toupper((unsigned char)c));
    }
    return isKnownNbaAbbrev(up) ? up : std::string();
}

inline Parameters scoreJson(const ScoreState& s) {
    Parameters out;
    if (s.a >= 0) out["team_a"] = s.a;
    if (s.b >= 0) out["team_b"] = s.b;
    if (s.period > 0) out["period"] = s.period;
    if (s.clock_sec >= 0) out["period_clock_remaining_sec"] = s.clock_sec;
    if (s.shot_clock_sec >= 0) out["shot_clock_sec"] = s.shot_clock_sec;
    return out;
}

inline bool readScoreState(const Parameters& frame_json, ScoreState& out) {
    if (!frame_json.contains("game") || !frame_json["game"].is_object()) return false;
    const auto& gs = frame_json["game"];
    if (!gs.contains("score_a") || !gs.contains("score_b")) return false;
    try {
        out.a = gs["score_a"].get<int>();
        out.b = gs["score_b"].get<int>();
    } catch (...) {
        return false;
    }
    out.period = gs.value("period", -1);
    out.clock_sec = gs.value("period_clock_remaining_sec", gs.value("clock_sec", -1));
    out.shot_clock_sec = gs.value("shot_clock_sec", -1);
    return haveScore(out);
}

inline bool isKnownZone(const std::string& zone) {
    return !zone.empty() && zone != "unknown";
}

inline std::string bestCourtZone(const Parameters& court_zone_md, std::string* source = nullptr) {
    const std::string handler_zone = getOr<std::string>(court_zone_md, "handler_zone", std::string());
    const std::string ball_zone = getOr<std::string>(court_zone_md, "ball_zone", std::string());
    if (isKnownZone(handler_zone)) {
        if (source) *source = "handler";
        return handler_zone;
    }
    if (isKnownZone(ball_zone)) {
        if (source) *source = "ball";
        return ball_zone;
    }
    if (source) source->clear();
    return {};
}

inline std::string dominantZone(const std::map<std::string, int>& zone_frames) {
    std::string best;
    int best_count = 0;
    for (const auto& kv : zone_frames) {
        if (kv.second > best_count) {
            best = kv.first;
            best_count = kv.second;
        }
    }
    return best_count >= 5 ? best : std::string();
}

inline bool readCpuMasks(const AVFrame* raw, int slot, MaskInfo& out) {
    if (!raw) return false;
    AVFrameSideData* sd = av_frame_get_side_data(raw, yoloSegCpuSideDataType(slot));
    if (!sd || !sd->buf || sd->buf->size < 16) return false;
    const uint32_t* header = (const uint32_t*)sd->buf->data;
    out.num_masks = (int)header[0];
    out.w = (int)header[1];
    out.h = (int)header[2];
    size_t expected = 16 + (size_t)out.num_masks * (size_t)out.w * (size_t)out.h * sizeof(float);
    if ((size_t)sd->buf->size < expected) return false;
    out.data = (const float*)(sd->buf->data + 16);
    return out.num_masks > 0 && out.w > 0 && out.h > 0;
}

inline Parameters dedupeContour(Parameters& pts) {
    if (pts.size() <= 1) return pts;
    Parameters deduped = Parameters::array();
    deduped.push_back(pts[0]);
    for (size_t i = 1; i < pts.size(); ++i) {
        if (pts[i] != pts[i - 1]) deduped.push_back(pts[i]);
    }
    if (deduped.size() > 1 && deduped.back() == deduped.front()) deduped.erase(deduped.size() - 1);
    return deduped;
}

inline Parameters traceContourRegion(const float* mask, int w, int h,
                                     int rx0, int ry0, int rx1, int ry1,
                                     float scale_x, float scale_y,
                                     float mask_threshold, int contour_simplify_step) {
    Parameters pts = Parameters::array();
    int step = std::max(1, contour_simplify_step);
    rx0 = std::max(0, rx0); ry0 = std::max(0, ry0);
    rx1 = std::min(w, rx1); ry1 = std::min(h, ry1);
    if (rx0 >= rx1 || ry0 >= ry1) return pts;

    for (int x = rx0; x < rx1; x += step) {
        for (int y = ry0; y < ry1; ++y) {
            if (mask[y * w + x] >= mask_threshold) {
                pts.push_back(Parameters::array({ri(x * scale_x), ri(y * scale_y)}));
                break;
            }
        }
    }
    for (int y = ry0; y < ry1; y += step) {
        for (int x = rx1 - 1; x >= rx0; --x) {
            if (mask[y * w + x] >= mask_threshold) {
                pts.push_back(Parameters::array({ri(x * scale_x), ri(y * scale_y)}));
                break;
            }
        }
    }
    for (int x = rx1 - 1; x >= rx0; x -= step) {
        for (int y = ry1 - 1; y >= ry0; --y) {
            if (mask[y * w + x] >= mask_threshold) {
                pts.push_back(Parameters::array({ri(x * scale_x), ri(y * scale_y)}));
                break;
            }
        }
    }
    for (int y = ry1 - 1; y >= ry0; y -= step) {
        for (int x = rx0; x < rx1; ++x) {
            if (mask[y * w + x] >= mask_threshold) {
                pts.push_back(Parameters::array({ri(x * scale_x), ri(y * scale_y)}));
                break;
            }
        }
    }

    return dedupeContour(pts);
}

inline Parameters traceContour(const float* mask, int w, int h,
                               float scale_x, float scale_y,
                               float mask_threshold, int contour_simplify_step) {
    return traceContourRegion(mask, w, h, 0, 0, w, h, scale_x, scale_y,
                              mask_threshold, contour_simplify_step);
}

}  // namespace metadata_dump
