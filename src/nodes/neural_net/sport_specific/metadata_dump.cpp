#include "../../node_common.hpp"
#include "../common/yolo_side_data.hpp"

extern "C" {
#include <libavutil/dict.h>
#include <libavutil/frame.h>
}

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <vector>

class MetadataDump : public NodeSISO<av::VideoFrame, av::VideoFrame> {
    std::string player_metadata_key_ = "yolo_players";
    std::string ball_metadata_key_ = "yolo_ball";
    std::string ball_handler_metadata_key_ = "ball_handler";
    std::string possession_metadata_key_ = "possession_state";
    std::string court_zone_metadata_key_ = "court_zone";
    std::string camera_shot_metadata_key_ = "camera_shot_info";
    std::string shot_events_metadata_key_ = "shot_events";
    std::string scoreboard_metadata_key_ = "scoreboard";
    std::string game_state_metadata_key_ = "game_state";
    std::string viewport_metadata_key_ = "smoothed_crop_viewport_v1";
    std::string player_seg_metadata_key_ = "yolo_players_seg";
    std::string court_seg_metadata_key_ = "yolo_seg";
    std::string output_metadata_key_ = "frame_dump";
    std::string output_file_;
    std::string output_file_court_;
    std::string output_file_outlines_;
    std::string output_file_trail_;
    std::string output_file_events_;
    std::string output_file_summary_;
    int summary_update_every_n_ = 50;
    std::string video_label_;
    int fps_ = 0;
    int schema_ = 2;
    int dump_every_n_ = 1;
    int debug_log_every_n_ = 0;
    int shot_result_wait_frames_ = 25;

    int court_seg_slot_ = 0;
    int player_seg_slot_ = 1;
    float mask_threshold_ = 0.5f;
    int contour_simplify_step_ = 2;

    uint64_t frame_counter_ = 0;
    std::string cached_json_;

    // Scale state — set on first frame
    bool header_written_ = false;
    int source_w_ = 0;
    int source_h_ = 0;
    float model_w_ = 960.0f;
    float model_h_ = 544.0f;
    float scale_x_ = 1.0f;
    float scale_y_ = 1.0f;

    // Event-detection state
    int prev_total_releases_ = 0;
    int prev_total_arrivals_ = 0;
    int prev_handler_id_ = -1;
    int last_known_handler_id_ = -1;
    std::string last_known_handler_team_;

    int pending_new_handler_id_ = -1;
    std::string pending_new_handler_team_;
    int pending_new_handler_frames_ = 0;
    int possession_change_confirm_frames_ = 6;

    // Clock-movement tracking
    int last_game_clock_sec_ = -1;
    float last_game_clock_change_t_ = -1.0f;
    bool clock_currently_stopped_ = false;
    float clock_stopped_threshold_s_ = 2.5f;

    // Clip-summary counters
    std::map<std::string, int> camera_shot_frames_;
    std::map<std::string, int> possessions_by_team_;
    int shots_made_ = 0;
    int shots_missed_ = 0;
    int shots_unknown_ = 0;
    int shot_attempts_total_ = 0;
    int first_game_clock_sec_ = -1;
    int latest_game_clock_sec_ = -1;
    std::string prev_possessing_team_;
    int prev_period_num_ = -1;
    int prev_shot_clock_sec_ = -1;
    std::string prev_camera_shot_;

    struct PendingRelease {
        uint64_t frame = 0;
        std::string shooting_team;
    };
    bool have_last_release_ = false;
    PendingRelease last_release_;

    struct PendingShot {
        uint64_t arrival_frame = 0;
        int64_t arrival_pts = 0;
        std::string shooting_team;
        int score_a_start = -1;
        int score_b_start = -1;
        int frames_waited = 0;
    };
    bool have_pending_shot_ = false;
    PendingShot pending_shot_;

    struct ShotRecord {
        int release_frame = -1;
        int arrival_frame = -1;
        int hoop_dist = -1;
        int release_hoop_dist = -1;
        std::string attempt_type;   // "2pt" | "3pt" | "unknown"
        int game_clock_sec = -1;    // at release
        int period_num = -1;        // at release
        std::string result;
        int points = 0;
    };

    struct PossessionAcc {
        int possession_id = -1;
        std::string team;
        uint64_t start_frame = 0;
        int64_t start_pts = 0;
        uint64_t end_frame = 0;
        int64_t end_pts = 0;
        int total_frames = 0;
        int frames_controlled = 0;
        int frames_loose = 0;
        int frames_in_flight = 0;
        std::vector<int> handler_ids;
        std::vector<std::string> zones_visited;
        std::vector<ShotRecord> shots;
    };
    bool poss_active_ = false;
    PossessionAcc poss_acc_;
    bool have_deferred_poss_ = false;
    PossessionAcc deferred_poss_;
    std::string output_file_possessions_;

    struct JsonFileWriter {
        std::ofstream out;
        bool opened = false;

        bool open(const std::string& path) {
            if (path.empty()) return false;
            out.open(path, std::ios::out | std::ios::trunc);
            if (!out) return false;
            opened = true;
            return true;
        }
        void writeLine(const std::string& json) {
            if (!opened) return;
            out << json << '\n';
            out.flush();
        }
        void close() {
            if (!opened) return;
            out.close();
            opened = false;
        }
    };

    JsonFileWriter file_main_;
    JsonFileWriter file_court_;
    JsonFileWriter file_outlines_;
    JsonFileWriter file_trail_;
    JsonFileWriter file_events_;
    JsonFileWriter file_possessions_;

    static Parameters tryParse(const AVFrame* raw, const std::string& key) {
        if (!raw || !raw->metadata) return {};
        AVDictionaryEntry* entry = av_dict_get(raw->metadata, key.c_str(), nullptr, 0);
        if (!entry || !entry->value) return {};
        try { return Parameters::parse(entry->value); } catch (...) { return {}; }
    }

    static int ri(float v) { return (int)std::round(v); }

    int scaleX(float v) const { return ri(v * scale_x_); }
    int scaleY(float v) const { return ri(v * scale_y_); }
    int scaleDist(float v) const { return ri(v * 0.5f * (scale_x_ + scale_y_)); }

    template <typename T>
    static T getOr(const Parameters& obj, const char* key, const T& fallback) {
        if (!obj.is_object() || !obj.contains(key) || obj[key].is_null()) return fallback;
        try {
            return obj[key].get<T>();
        } catch (...) {
            return fallback;
        }
    }

    Parameters scaleBox(const Parameters& det) const {
        if (!det.contains("xyxy") || !det["xyxy"].is_array() || det["xyxy"].size() < 4) return nullptr;
        return Parameters::array({
            scaleX(det["xyxy"][0].get<float>()),
            scaleY(det["xyxy"][1].get<float>()),
            scaleX(det["xyxy"][2].get<float>()),
            scaleY(det["xyxy"][3].get<float>())
        });
    }

    struct MaskInfo {
        int num_masks;
        int w;
        int h;
        const float* data;
    };

    static bool readCpuMasks(const AVFrame* raw, int slot, MaskInfo& out) {
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

    static Parameters dedupeContour(Parameters& pts) {
        if (pts.size() <= 1) return pts;
        Parameters deduped = Parameters::array();
        deduped.push_back(pts[0]);
        for (size_t i = 1; i < pts.size(); ++i) {
            if (pts[i] != pts[i - 1]) deduped.push_back(pts[i]);
        }
        if (deduped.size() > 1 && deduped.back() == deduped.front()) deduped.erase(deduped.size() - 1);
        return deduped;
    }

    Parameters traceContour(const float* mask, int w, int h, float scale_x, float scale_y) const {
        return traceContourRegion(mask, w, h, 0, 0, w, h, scale_x, scale_y);
    }

    Parameters traceContourRegion(const float* mask, int w, int h,
                                  int rx0, int ry0, int rx1, int ry1,
                                  float scale_x, float scale_y) const {
        Parameters pts = Parameters::array();
        int step = std::max(1, contour_simplify_step_);
        rx0 = std::max(0, rx0); ry0 = std::max(0, ry0);
        rx1 = std::min(w, rx1); ry1 = std::min(h, ry1);
        if (rx0 >= rx1 || ry0 >= ry1) return pts;

        for (int x = rx0; x < rx1; x += step) {
            for (int y = ry0; y < ry1; ++y) {
                if (mask[y * w + x] >= mask_threshold_) {
                    pts.push_back(Parameters::array({ri(x * scale_x), ri(y * scale_y)}));
                    break;
                }
            }
        }
        for (int y = ry0; y < ry1; y += step) {
            for (int x = rx1 - 1; x >= rx0; --x) {
                if (mask[y * w + x] >= mask_threshold_) {
                    pts.push_back(Parameters::array({ri(x * scale_x), ri(y * scale_y)}));
                    break;
                }
            }
        }
        for (int x = rx1 - 1; x >= rx0; x -= step) {
            for (int y = ry1 - 1; y >= ry0; --y) {
                if (mask[y * w + x] >= mask_threshold_) {
                    pts.push_back(Parameters::array({ri(x * scale_x), ri(y * scale_y)}));
                    break;
                }
            }
        }
        for (int y = ry1 - 1; y >= ry0; y -= step) {
            for (int x = rx0; x < rx1; ++x) {
                if (mask[y * w + x] >= mask_threshold_) {
                    pts.push_back(Parameters::array({ri(x * scale_x), ri(y * scale_y)}));
                    break;
                }
            }
        }

        return dedupeContour(pts);
    }

    struct DumpResult {
        Parameters main;
        Parameters court;
        Parameters outlines;
        Parameters trail;
    };

    Parameters buildHeader() const {
        Parameters h;
        h["type"] = "header";
        h["schema"] = schema_;
        h["source_w"] = source_w_;
        h["source_h"] = source_h_;
        h["model_w"] = model_w_;
        h["model_h"] = model_h_;
        h["coord_space"] = "source";
        if (fps_ > 0) h["fps"] = fps_;
        if (!video_label_.empty()) h["video"] = video_label_;
        return h;
    }

    DumpResult buildDump(const av::VideoFrame& frm) {
        const AVFrame* raw = frm.raw();
        DumpResult result;
        Parameters& out = result.main;

        int64_t pts = frm.pts().timestamp();
        out["type"] = "frame";
        out["frame"] = frame_counter_;
        out["pts"] = pts;
        if (fps_ > 0) out["t"] = std::round((float)frame_counter_ / (float)fps_ * 1000.0f) / 1000.0f;

        auto players_md = tryParse(raw, player_metadata_key_);
        auto ball_md = tryParse(raw, ball_metadata_key_);
        auto handler_md = tryParse(raw, ball_handler_metadata_key_);
        auto possession_md = tryParse(raw, possession_metadata_key_);
        auto court_zone_md = tryParse(raw, court_zone_metadata_key_);
        auto shot_md = tryParse(raw, camera_shot_metadata_key_);
        auto shot_events_md = tryParse(raw, shot_events_metadata_key_);
        auto game_state_md = tryParse(raw, game_state_metadata_key_);
        auto viewport_md = tryParse(raw, viewport_metadata_key_);
        auto player_seg_md = tryParse(raw, player_seg_metadata_key_);
        auto court_seg_md = tryParse(raw, court_seg_metadata_key_);

        // Camera shot type + transition
        if (shot_md.contains("camera_shot_type")) out["camera_shot"] = shot_md["camera_shot_type"];
        if (shot_md.contains("camera_shot_transition") && shot_md["camera_shot_transition"].get<bool>())
            out["camera_shot_transition"] = true;

        // Shot attempt state. Keep only fields useful to downstream reasoning;
        // counters are retained because event synthesis uses their increments.
        if (shot_events_md.is_object() && !shot_events_md.is_null()) {
            Parameters shot;
            const bool in_flight = getOr<bool>(shot_events_md, "in_flight", false);
            const bool shot_result = getOr<bool>(shot_events_md, "shot_result", false);
            const int total_releases = getOr<int>(shot_events_md, "total_releases", 0);
            const int total_arrivals = getOr<int>(shot_events_md, "total_hoop_arrivals", 0);
            if (in_flight) shot["in_flight"] = true;
            if (total_releases > 0) shot["total_releases"] = total_releases;
            if (total_arrivals > 0) shot["total_hoop_arrivals"] = total_arrivals;
            if (shot_result) shot["result_event"] = true;
            if (shot_events_md.contains("result")) shot["result"] = shot_events_md["result"];
            if (shot_events_md.contains("result_conf")) {
                shot["confidence"] = std::round(shot_events_md["result_conf"].get<float>() * 1000.0f) / 1000.0f;
            }
            if (shot_events_md.contains("attempt_type")) shot["attempt_type"] = shot_events_md["attempt_type"];
            if (shot_events_md.contains("points")) shot["points"] = shot_events_md["points"];
            if (shot_events_md.contains("ball_hoop_dist")) {
                shot["ball_hoop_dist"] = scaleDist(shot_events_md["ball_hoop_dist"].get<float>());
            }
            if (!shot.empty()) out["shot"] = shot;
        }

        // Possession state
        if (possession_md.is_object() && !possession_md.is_null()) {
            Parameters possession_out;
            if (possession_md.contains("possessing_team")) possession_out["team"] = possession_md["possessing_team"];
            if (possession_md.contains("handler_id")) possession_out["handler_id"] = possession_md["handler_id"];
            if (possession_md.contains("handler_team")) possession_out["handler_team"] = possession_md["handler_team"];
            if (possession_md.contains("ball_state")) possession_out["state"] = possession_md["ball_state"];
            if (possession_md.contains("frames_in_possession")) possession_out["frames_in_possession"] = possession_md["frames_in_possession"];
            if (possession_md.contains("possession_id")) possession_out["possession_id"] = possession_md["possession_id"];
            if (!possession_out.empty()) out["possession"] = possession_out;
        }

        // Parsed game state. Raw OCR strings are intentionally omitted; they are noisy.
        if (game_state_md.is_object() && !game_state_md.is_null()) {
            Parameters game_out;
            if (game_state_md.contains("period_num")) game_out["period"] = game_state_md["period_num"];
            if (game_state_md.contains("game_clock_sec")) game_out["clock_sec"] = game_state_md["game_clock_sec"];
            if (game_state_md.contains("shot_clock_sec")) game_out["shot_clock_sec"] = game_state_md["shot_clock_sec"];
            if (!game_out.empty()) out["game"] = game_out;
        }

        // Court zone — omit placeholder values when zone unknown
        if (court_zone_md.is_object() && !court_zone_md.is_null()) {
            Parameters zone_out;
            const std::string handler_zone = getOr<std::string>(court_zone_md, "handler_zone", std::string());
            const std::string ball_zone = getOr<std::string>(court_zone_md, "ball_zone", std::string());
            const bool any_known = (!handler_zone.empty() && handler_zone != "unknown") ||
                                   (!ball_zone.empty() && ball_zone != "unknown") ||
                                   getOr<bool>(court_zone_md, "inside_court", false);
            if (any_known) {
                if (court_zone_md.contains("zone_source")) zone_out["zone_source"] = court_zone_md["zone_source"];
                if (!handler_zone.empty() && handler_zone != "unknown") zone_out["handler_zone"] = handler_zone;
                if (!ball_zone.empty() && ball_zone != "unknown") zone_out["ball_zone"] = ball_zone;
                if (court_zone_md.contains("hoop_side")) zone_out["hoop_side"] = court_zone_md["hoop_side"];
                if (court_zone_md.contains("inside_court")) zone_out["inside_court"] = court_zone_md["inside_court"];
                if (court_zone_md.contains("inside_three_point_area")) zone_out["inside_three_point_area"] = court_zone_md["inside_three_point_area"];
                float raw_rx = getOr<float>(court_zone_md, "relative_to_hoop_x", 0.0f);
                float raw_ry = getOr<float>(court_zone_md, "relative_to_hoop_y", 0.0f);
                float raw_d = getOr<float>(court_zone_md, "distance_to_hoop", 0.0f);
                if (raw_rx != 0.0f || raw_ry != 0.0f) {
                    zone_out["relative_to_hoop_x"] = scaleX(raw_rx);
                    zone_out["relative_to_hoop_y"] = scaleY(raw_ry);
                }
                if (raw_d > 0.0f) zone_out["distance_to_hoop"] = scaleDist(raw_d);
            }
            if (!zone_out.empty()) out["court_zone"] = zone_out;
        }

        // Ball handler track_id
        int handler_track_id = -1;
        if (handler_md.contains("detections") && handler_md["detections"].is_array()) {
            for (const auto& det : handler_md["detections"]) {
                if (!det.is_object()) continue;
                if (getOr<std::string>(det, "label", std::string()) == "BallHandler") {
                    handler_track_id = getOr<int>(det, "track_id", -1);
                    break;
                }
            }
        }

        // Player seg masks — source-space contour scaling
        MaskInfo player_masks = {};
        bool have_player_masks = readCpuMasks(raw, player_seg_slot_, player_masks);

        std::vector<int> seg_det_to_mask;
        if (have_player_masks && player_seg_md.contains("detections") && player_seg_md["detections"].is_array()) {
            int mask_idx = 0;
            for (size_t i = 0; i < player_seg_md["detections"].size(); ++i) {
                if (mask_idx < player_masks.num_masks) {
                    seg_det_to_mask.push_back(mask_idx);
                    ++mask_idx;
                } else {
                    seg_det_to_mask.push_back(-1);
                }
            }
        }

        struct SegMatch { int seg_det_idx; std::string team_ab; };
        std::vector<std::pair<int, SegMatch>> track_to_seg;
        if (player_seg_md.contains("detections") && player_seg_md["detections"].is_array()) {
            if (players_md.contains("detections") && players_md["detections"].is_array()) {
                for (size_t si = 0; si < player_seg_md["detections"].size(); ++si) {
                    const auto& seg_det = player_seg_md["detections"][si];
                    if (!seg_det.contains("xyxy") || !seg_det["xyxy"].is_array() || seg_det["xyxy"].size() < 4) continue;
                    float sx1 = seg_det["xyxy"][0].get<float>(), sy1 = seg_det["xyxy"][1].get<float>();
                    float sx2 = seg_det["xyxy"][2].get<float>(), sy2 = seg_det["xyxy"][3].get<float>();

                    float best_iou = 0.15f;
                    int best_track = -1;
                    for (size_t pi = 0; pi < players_md["detections"].size(); ++pi) {
                        const auto& p = players_md["detections"][pi];
                        if (getOr<std::string>(p, "label", std::string()) != "Player") continue;
                        if (!p.contains("xyxy") || !p["xyxy"].is_array() || p["xyxy"].size() < 4) continue;
                        float px1 = p["xyxy"][0].get<float>(), py1 = p["xyxy"][1].get<float>();
                        float px2 = p["xyxy"][2].get<float>(), py2 = p["xyxy"][3].get<float>();
                        float ix1 = std::max(sx1, px1), iy1 = std::max(sy1, py1);
                        float ix2 = std::min(sx2, px2), iy2 = std::min(sy2, py2);
                        float iw = std::max(0.0f, ix2 - ix1), ih = std::max(0.0f, iy2 - iy1);
                        float inter = iw * ih;
                        float ua = (sx2 - sx1) * (sy2 - sy1) + (px2 - px1) * (py2 - py1) - inter;
                        float iou = (ua > 0.0f) ? inter / ua : 0.0f;
                        if (iou > best_iou) { best_iou = iou; best_track = getOr<int>(p, "track_id", -1); }
                    }
                    if (best_track >= 0) track_to_seg.push_back({best_track, {(int)si, getOr<std::string>(seg_det, "team_ab", std::string("?"))}});
                }
            }
        }

        // Mask→source scale for player outlines; model→mask to clip region.
        float p_mask_to_src_x = (player_masks.w > 0) ? (float)source_w_ / (float)player_masks.w : 1.0f;
        float p_mask_to_src_y = (player_masks.h > 0) ? (float)source_h_ / (float)player_masks.h : 1.0f;
        float p_model_to_mask_x = (player_masks.w > 0) ? model_w_ / (float)player_masks.w : 1.0f;
        float p_model_to_mask_y = (player_masks.h > 0) ? model_h_ / (float)player_masks.h : 1.0f;

        Parameters outlines_arr = Parameters::array();

        // Players, refs, hoop — all boxes scaled to source space
        if (players_md.contains("detections") && players_md["detections"].is_array()) {
            Parameters players_arr = Parameters::array();
            Parameters refs_arr = Parameters::array();
            Parameters hoop;

            for (const auto& det : players_md["detections"]) {
                if (!det.is_object()) continue;
                std::string label = getOr<std::string>(det, "label", std::string());
                auto box = scaleBox(det);
                if (box.is_null()) continue;
                float conf = getOr<float>(det, "conf", 0.0f);

                if (label == "Player") {
                    Parameters p;
                    int tid = getOr<int>(det, "track_id", -1);
                    if (tid >= 0) p["id"] = tid;
                    std::string tab = getOr<std::string>(det, "team_ab", std::string("?"));
                    if (tab != "?") p["team"] = tab;
                    if (det.contains("team_lock_age")) p["team_lock_age"] = det["team_lock_age"].get<uint32_t>();
                    p["box"] = box;
                    if (tid >= 0 && tid == handler_track_id) p["has_ball"] = true;
                    if (det.contains("track_state")) {
                        std::string ts = det["track_state"].get<std::string>();
                        if (ts != "tracked") p["track_state"] = ts;
                    }

                    if (tid >= 0 && have_player_masks && det.contains("xyxy") && det["xyxy"].size() >= 4) {
                        float bx1 = det["xyxy"][0].get<float>(), by1 = det["xyxy"][1].get<float>();
                        float bx2 = det["xyxy"][2].get<float>(), by2 = det["xyxy"][3].get<float>();
                        int mx0 = (int)(bx1 / p_model_to_mask_x) - 1, my0 = (int)(by1 / p_model_to_mask_y) - 1;
                        int mx1 = (int)(bx2 / p_model_to_mask_x) + 2, my1 = (int)(by2 / p_model_to_mask_y) + 2;

                        for (const auto& ts : track_to_seg) {
                            if (ts.first == tid) {
                                int si = ts.second.seg_det_idx;
                                if (si >= 0 && si < (int)seg_det_to_mask.size()) {
                                    int mi = seg_det_to_mask[(size_t)si];
                                    if (mi >= 0 && mi < player_masks.num_masks) {
                                        const float* mdata = player_masks.data +
                                            (size_t)mi * (size_t)player_masks.w * (size_t)player_masks.h;
                                        auto contour = traceContourRegion(mdata, player_masks.w, player_masks.h,
                                                                          mx0, my0, mx1, my1,
                                                                          p_mask_to_src_x, p_mask_to_src_y);
                                        if (!contour.empty()) {
                                            Parameters oe;
                                            oe["id"] = tid;
                                            oe["outline"] = contour;
                                            outlines_arr.push_back(std::move(oe));
                                        }
                                    }
                                }
                                break;
                            }
                        }
                    }

                    players_arr.push_back(std::move(p));
                } else if (label == "Ref") {
                    Parameters r;
                    int tid = getOr<int>(det, "track_id", -1);
                    if (tid >= 0) r["id"] = tid;
                    r["box"] = box;
                    r["conf"] = (int)(conf * 100.0f + 0.5f);
                    refs_arr.push_back(std::move(r));
                } else if (label == "Hoop" && hoop.is_null()) {
                    hoop = {{"box", box}, {"conf", (int)(conf * 100.0f + 0.5f)}};
                }
            }

            if (!players_arr.empty()) out["players"] = players_arr;
            if (!refs_arr.empty()) out["refs"] = refs_arr;
            if (!hoop.is_null()) out["hoop"] = hoop;
        }

        // Ball
        if (ball_md.contains("detections") && ball_md["detections"].is_array()) {
            for (const auto& det : ball_md["detections"]) {
                if (!det.is_object()) continue;
                auto box = scaleBox(det);
                if (box.is_null()) continue;
                Parameters b;
                b["box"] = box;
                b["conf"] = (int)(getOr<float>(det, "conf", 0.0f) * 100.0f + 0.5f);
                if (det.contains("source")) b["state"] = det["source"];
                if (det.contains("coast_streak")) {
                    int cs = det["coast_streak"].get<int>();
                    if (cs > 0) b["coast_streak"] = cs;
                }
                out["ball"] = b;
                break;
            }
        }

        // Viewport
        if (viewport_md.contains("detections") && viewport_md["detections"].is_array()) {
            for (const auto& det : viewport_md["detections"]) {
                auto box = scaleBox(det);
                if (!box.is_null()) { out["viewport"] = box; break; }
            }
        }

        // Ball trail sidecar — coords in source space
        if (ball_md.contains("trail") && ball_md["trail"].is_array() && !ball_md["trail"].empty()) {
            Parameters trail_entry;
            trail_entry["type"] = "trail";
            trail_entry["frame"] = frame_counter_;
            trail_entry["pts"] = pts;
            Parameters trail = Parameters::array();
            for (const auto& pt : ball_md["trail"]) {
                if (!pt.is_array() || pt.size() < 3) continue;
                trail.push_back(Parameters::array({
                    scaleX((float)pt[0].get<int>()),
                    scaleY((float)pt[1].get<int>()),
                    pt[2].get<int64_t>()
                }));
            }
            if (!trail.empty()) {
                trail_entry["trail"] = trail;
                result.trail = trail_entry;
            }
        }

        if (!outlines_arr.empty()) {
            Parameters outlines_entry;
            outlines_entry["type"] = "outlines";
            outlines_entry["frame"] = frame_counter_;
            outlines_entry["pts"] = pts;
            outlines_entry["players"] = outlines_arr;
            result.outlines = outlines_entry;
        }

        // Court contours — source space
        MaskInfo court_masks = {};
        if (readCpuMasks(raw, court_seg_slot_, court_masks)) {
            float court_scale_x = (float)source_w_ / (float)court_masks.w;
            float court_scale_y = (float)source_h_ / (float)court_masks.h;

            std::vector<std::string> court_labels;
            if (court_seg_md.contains("detections") && court_seg_md["detections"].is_array()) {
                for (const auto& det : court_seg_md["detections"]) {
                    if (!det.is_object()) continue;
                    court_labels.push_back(getOr<std::string>(det, "label", std::string("unknown")));
                }
            }

            Parameters court_entry;
            court_entry["type"] = "court";
            court_entry["frame"] = frame_counter_;
            court_entry["pts"] = pts;
            Parameters court_arr = Parameters::array();
            for (int i = 0; i < court_masks.num_masks; ++i) {
                const float* mdata = court_masks.data + (size_t)i * (size_t)court_masks.w * (size_t)court_masks.h;
                auto contour = traceContour(mdata, court_masks.w, court_masks.h, court_scale_x, court_scale_y);
                if (contour.empty()) continue;
                Parameters entry;
                entry["label"] = (i < (int)court_labels.size()) ? court_labels[(size_t)i] : "unknown";
                entry["outline"] = contour;
                court_arr.push_back(entry);
            }
            if (!court_arr.empty()) {
                court_entry["regions"] = court_arr;
                result.court = court_entry;
            }
        }

        return result;
    }

    void emitEvent(Parameters ev, int64_t pts) {
        if (!file_events_.opened) return;
        ev["frame"] = frame_counter_;
        ev["pts"] = pts;
        if (fps_ > 0) ev["t"] = std::round((float)frame_counter_ / (float)fps_ * 1000.0f) / 1000.0f;
        file_events_.writeLine(ev.dump());
    }

    void detectAndEmitEvents(const Parameters& frame_json, int64_t pts) {
        // Update last-known handler before any events (current frame's handler, if any)
        if (frame_json.contains("possession")) {
            const auto& pos = frame_json["possession"];
            int hid = pos.value("handler_id", -1);
            std::string tm = pos.value("handler_team", std::string());
            if (hid >= 0) last_known_handler_id_ = hid;
            if (!tm.empty()) last_known_handler_team_ = tm;
        }

        // Camera shot change
        const std::string cam = frame_json.value("camera_shot", std::string());
        if (!cam.empty() && cam != prev_camera_shot_) {
            if (!prev_camera_shot_.empty()) {
                Parameters ev;
                ev["type"] = "camera_shot_change";
                ev["from"] = prev_camera_shot_;
                ev["to"] = cam;
                emitEvent(ev, pts);
            }
            prev_camera_shot_ = cam;
        }

        // Shot release / hoop arrival
        // Shot state uses cumulative counters from the concise `shot` object.
        const Parameters empty_shot = Parameters::object();
        const Parameters& se_dump = (frame_json.contains("shot") && frame_json["shot"].is_object())
            ? frame_json["shot"] : empty_shot;
        int total_rel = getOr<int>(se_dump, "total_releases", prev_total_releases_);
        int total_arr = getOr<int>(se_dump, "total_hoop_arrivals", prev_total_arrivals_);
        if (total_rel > prev_total_releases_) {
            Parameters ev;
            ev["type"] = "release";
            // At the release frame, possession often already flipped to shot_in_air.
            // Attribute to the last-known handler from prior frames.
            if (last_known_handler_id_ >= 0) ev["player_id"] = last_known_handler_id_;
            if (!last_known_handler_team_.empty()) ev["team"] = last_known_handler_team_;
            emitEvent(ev, pts);
            have_last_release_ = true;
            last_release_.frame = frame_counter_;
            last_release_.shooting_team = last_known_handler_team_;
            onShotRelease(pts, frame_json);
            prev_total_releases_ = total_rel;
        }
        if (total_arr > prev_total_arrivals_) {
            Parameters ev;
            ev["type"] = "hoop_arrival";
            int hoop_dist_val = -1;
            if (se_dump.contains("ball_hoop_dist")) hoop_dist_val = se_dump["ball_hoop_dist"].get<int>();
            if (hoop_dist_val >= 0) ev["ball_hoop_dist"] = hoop_dist_val;
            if (!last_known_handler_team_.empty()) ev["team"] = last_known_handler_team_;
            emitEvent(ev, pts);
            onShotArrival(pts, hoop_dist_val);
            // Start pending shot if a recent release is associated
            if (have_last_release_ && (frame_counter_ - last_release_.frame) < 200) {
                have_pending_shot_ = true;
                pending_shot_.arrival_frame = frame_counter_;
                pending_shot_.arrival_pts = pts;
                pending_shot_.shooting_team = last_release_.shooting_team;
                pending_shot_.frames_waited = 0;
                pending_shot_.score_a_start = -1;
                pending_shot_.score_b_start = -1;
                have_last_release_ = false;
            }
            prev_total_arrivals_ = total_arr;
        }
        if (getOr<bool>(se_dump, "result_event", false)) {
            Parameters ev;
            ev["type"] = "shot_result";
            if (!last_known_handler_team_.empty()) ev["team"] = last_known_handler_team_;
            ev["result"] = getOr<std::string>(se_dump, "result", std::string("outcome_unknown"));
            if (se_dump.contains("result_source")) ev["source"] = se_dump["result_source"];
            if (se_dump.contains("result_conf")) ev["confidence"] = se_dump["result_conf"];
            if (se_dump.contains("result_vx")) ev["vx"] = se_dump["result_vx"];
            if (se_dump.contains("result_vy")) ev["vy"] = se_dump["result_vy"];
            if (se_dump.contains("attempt_type")) ev["attempt_type"] = se_dump["attempt_type"];
            if (se_dump.contains("attempt_points")) ev["attempt_points"] = se_dump["attempt_points"];
            if (se_dump.contains("points")) ev["points"] = se_dump["points"];
            emitEvent(ev, pts);
            onShotResult(getOr<std::string>(ev, "result", std::string("outcome_unknown")),
                         getOr<int>(ev, "points", 0),
                         getOr<std::string>(ev, "attempt_type", std::string()));
            have_pending_shot_ = false;
        }

        // Possession change — debounced: new handler must persist N frames before emitting.
        if (frame_json.contains("possession")) {
            const auto& pos = frame_json["possession"];
            int hid = pos.value("handler_id", -1);
            std::string pteam = pos.value("team", std::string());

            if (hid >= 0 && hid != prev_handler_id_) {
                if (hid == pending_new_handler_id_) {
                    ++pending_new_handler_frames_;
                    if (!pteam.empty()) pending_new_handler_team_ = pteam;
                } else {
                    pending_new_handler_id_ = hid;
                    pending_new_handler_team_ = pteam;
                    pending_new_handler_frames_ = 1;
                }
                if (pending_new_handler_frames_ >= possession_change_confirm_frames_) {
                    Parameters ev;
                    ev["type"] = "possession_change";
                    if (prev_handler_id_ >= 0) ev["from_id"] = prev_handler_id_;
                    if (!prev_possessing_team_.empty()) ev["from_team"] = prev_possessing_team_;
                    ev["to_id"] = pending_new_handler_id_;
                    if (!pending_new_handler_team_.empty()) ev["to_team"] = pending_new_handler_team_;
                    emitEvent(ev, pts);
                    prev_handler_id_ = pending_new_handler_id_;
                    prev_possessing_team_ = pending_new_handler_team_;
                    pending_new_handler_id_ = -1;
                    pending_new_handler_team_.clear();
                    pending_new_handler_frames_ = 0;
                }
            } else {
                // Reverted to the last confirmed handler (flap) — drop pending.
                pending_new_handler_id_ = -1;
                pending_new_handler_team_.clear();
                pending_new_handler_frames_ = 0;
                // Do NOT emit possession_change when only the team label changes for the
                // same handler_id — that's a classifier reclassification, not a real pass.
            }
        }

        // Period change & shot clock reset
        if (frame_json.contains("game")) {
            const auto& gs = frame_json["game"];
            int pn = gs.value("period", -1);
            if (pn > 0 && pn != prev_period_num_) {
                if (prev_period_num_ > 0) {
                    Parameters ev;
                    ev["type"] = "period_change";
                    ev["from"] = prev_period_num_;
                    ev["to"] = pn;
                    emitEvent(ev, pts);
                }
                prev_period_num_ = pn;
            }
            int sc = gs.value("shot_clock_sec", -1);
            if (sc >= 0 && prev_shot_clock_sec_ >= 0 && sc > prev_shot_clock_sec_ + 1) {
                Parameters ev;
                ev["type"] = "shot_clock_reset";
                ev["to"] = sc;
                emitEvent(ev, pts);
            }
            if (sc >= 0) prev_shot_clock_sec_ = sc;
        }
    }

    void tickPendingShot(const Parameters& frame_json, int64_t pts) {
        if (!have_pending_shot_) return;
        if (!frame_json.contains("game")) {
            ++pending_shot_.frames_waited;
        } else {
            int score_a_now = -1;
            int score_b_now = -1;
            int pts_scored = 0;
            const std::string& team = pending_shot_.shooting_team;
            if (team == "A" && pending_shot_.score_a_start >= 0 && score_a_now > pending_shot_.score_a_start) {
                pts_scored = score_a_now - pending_shot_.score_a_start;
            } else if (team == "B" && pending_shot_.score_b_start >= 0 && score_b_now > pending_shot_.score_b_start) {
                pts_scored = score_b_now - pending_shot_.score_b_start;
            }
            if (pts_scored >= 1 && pts_scored <= 3) {
                Parameters ev;
                ev["type"] = "shot_result";
                ev["team"] = team;
                ev["result"] = "scored";
                ev["points"] = pts_scored;
                emitEvent(ev, pts);
                onShotResult("scored", pts_scored);
                have_pending_shot_ = false;
                return;
            }
            ++pending_shot_.frames_waited;
        }
        if (pending_shot_.frames_waited >= shot_result_wait_frames_) {
            Parameters ev;
            ev["type"] = "shot_result";
            if (!pending_shot_.shooting_team.empty()) ev["team"] = pending_shot_.shooting_team;
            // Missed is asserted ONLY when we have the shooter team's score both at shot start
            // AND at timeout, and the values match. Any absence → outcome_unknown (no guessing).
            int score_start = -1, score_now = -1;
            if (pending_shot_.shooting_team == "A") {
                score_start = pending_shot_.score_a_start;
                score_now = -1;
            } else if (pending_shot_.shooting_team == "B") {
                score_start = pending_shot_.score_b_start;
                score_now = -1;
            }
            const bool confident_missed = (score_start >= 0 && score_now >= 0 && score_now == score_start);
            ev["result"] = confident_missed ? "missed" : "outcome_unknown";
            emitEvent(ev, pts);
            onShotResult(ev.value("result", std::string("outcome_unknown")),
                         ev.value("points", 0));
            have_pending_shot_ = false;
        }
    }

    void writePossessionRecord(const PossessionAcc& a) {
        if (!file_possessions_.opened) return;
        if (a.team.empty() && a.shots.empty()) return;
        Parameters p;
        p["type"] = "possession";
        p["possession_id"] = a.possession_id;
        if (!a.team.empty()) p["team"] = a.team;
        p["start_frame"] = a.start_frame;
        p["end_frame"] = a.end_frame;
        p["start_pts"] = a.start_pts;
        p["end_pts"] = a.end_pts;
        p["frames"] = a.total_frames;
        if (fps_ > 0) p["duration_sec"] = std::round((float)a.total_frames / (float)fps_ * 100.0f) / 100.0f;
        if (!a.handler_ids.empty()) p["handler_ids"] = a.handler_ids;
        if (!a.zones_visited.empty()) p["zones_visited"] = a.zones_visited;
        Parameters bs;
        bs["controlled"] = a.frames_controlled;
        bs["loose"] = a.frames_loose;
        bs["in_flight"] = a.frames_in_flight;
        p["ball_state_frames"] = bs;
        const bool has_shots = !a.shots.empty();
        p["ended_with"] = has_shots ? "shot_attempt" : "team_change";
        if (has_shots) {
            Parameters arr = Parameters::array();
            for (const auto& sh : a.shots) {
                Parameters sj;
                if (sh.release_frame >= 0) sj["release_frame"] = sh.release_frame;
                if (sh.arrival_frame >= 0) sj["arrival_frame"] = sh.arrival_frame;
                if (sh.release_hoop_dist >= 0) sj["release_hoop_dist"] = sh.release_hoop_dist;
                if (sh.hoop_dist >= 0) sj["hoop_dist"] = sh.hoop_dist;
                if (!sh.attempt_type.empty()) sj["attempt_type"] = sh.attempt_type;
                if (sh.game_clock_sec >= 0) sj["game_clock_sec"] = sh.game_clock_sec;
                if (sh.period_num >= 0) sj["period_num"] = sh.period_num;
                if (!sh.result.empty()) sj["result"] = sh.result;
                if (sh.points > 0) sj["points"] = sh.points;
                arr.push_back(sj);
            }
            p["shots"] = arr;
        }
        // Possession counters are written here; shot counters are updated when a
        // shot result event is observed so unknown-team shots still reach summary.
        ++possessions_by_team_[a.team.empty() ? std::string("?") : a.team];
        file_possessions_.writeLine(p.dump());
    }

    Parameters buildClipSummary() const {
        Parameters s;
        s["type"] = "clip_summary";
        s["total_frames"] = frame_counter_;
        if (fps_ > 0) s["duration_sec"] = std::round((float)frame_counter_ / (float)fps_ * 100.0f) / 100.0f;
        if (!camera_shot_frames_.empty()) {
            Parameters cs;
            for (const auto& kv : camera_shot_frames_) cs[kv.first] = kv.second;
            s["camera_shot_frames"] = cs;
        }
        int poss_total = 0;
        if (!possessions_by_team_.empty()) {
            Parameters pb;
            for (const auto& kv : possessions_by_team_) { pb[kv.first] = kv.second; poss_total += kv.second; }
            s["possessions_by_team"] = pb;
            s["possessions_total"] = poss_total;
        }
        s["shot_attempts"] = shot_attempts_total_;
        if (shot_attempts_total_ > 0) {
            Parameters r;
            r["scored"] = shots_made_;
            r["missed"] = shots_missed_;
            r["outcome_unknown"] = shots_unknown_;
            s["shot_results"] = r;
        }
        if (first_game_clock_sec_ >= 0) s["first_game_clock_sec"] = first_game_clock_sec_;
        if (latest_game_clock_sec_ >= 0) s["last_game_clock_sec"] = latest_game_clock_sec_;
        return s;
    }

    void writeClipSummaryFile() {
        if (output_file_summary_.empty()) return;
        std::ofstream out(output_file_summary_, std::ios::out | std::ios::trunc);
        if (!out) return;
        out << buildClipSummary().dump(2) << '\n';
    }

    void writeClipSummary() {
        Parameters s = buildClipSummary();
        std::string json = s.dump();
        if (file_main_.opened) file_main_.writeLine(json);
        if (file_events_.opened) file_events_.writeLine(json);
        if (file_possessions_.opened) file_possessions_.writeLine(json);
        writeClipSummaryFile();
    }

    void flushDeferredPossession() {
        if (!have_deferred_poss_) return;
        have_deferred_poss_ = false;
        writePossessionRecord(deferred_poss_);
    }

    void flushPossession() {
        if (!poss_active_) return;
        poss_active_ = false;
        // If the last shot is unresolved, defer flush so the result lands here.
        const bool last_unresolved = !poss_acc_.shots.empty() && poss_acc_.shots.back().result.empty();
        if (last_unresolved && have_pending_shot_) {
            flushDeferredPossession();  // evict any older deferred
            deferred_poss_ = poss_acc_;
            have_deferred_poss_ = true;
            return;
        }
        writePossessionRecord(poss_acc_);
    }

    void updatePossessionAcc(const Parameters& frame_json, int64_t pts) {
        if (!frame_json.contains("possession")) return;
        const auto& pos = frame_json["possession"];
        int pid = pos.value("possession_id", -1);
        if (pid < 0) return;
        std::string team = pos.value("team", std::string());

        if (!poss_active_ || pid != poss_acc_.possession_id) {
            flushPossession();
            poss_acc_ = PossessionAcc();
            poss_active_ = true;
            poss_acc_.possession_id = pid;
            poss_acc_.team = team;
            poss_acc_.start_frame = frame_counter_;
            poss_acc_.start_pts = pts;
        }
        if (poss_acc_.team.empty() && !team.empty()) poss_acc_.team = team;
        poss_acc_.end_frame = frame_counter_;
        poss_acc_.end_pts = pts;
        ++poss_acc_.total_frames;
        std::string bs = pos.value("state", std::string());
        if (bs == "controlled") ++poss_acc_.frames_controlled;
        else if (bs == "loose") ++poss_acc_.frames_loose;
        else if (bs == "shot_in_air") ++poss_acc_.frames_in_flight;
        int hid = pos.value("handler_id", -1);
        if (hid >= 0 && std::find(poss_acc_.handler_ids.begin(), poss_acc_.handler_ids.end(), hid) == poss_acc_.handler_ids.end()) {
            poss_acc_.handler_ids.push_back(hid);
        }
        if (frame_json.contains("court_zone")) {
            const auto& cz = frame_json["court_zone"];
            for (const char* k : {"handler_zone", "ball_zone"}) {
                std::string z = cz.value(k, std::string());
                if (z.empty() || z == "unknown") continue;
                if (std::find(poss_acc_.zones_visited.begin(), poss_acc_.zones_visited.end(), z) == poss_acc_.zones_visited.end())
                    poss_acc_.zones_visited.push_back(z);
            }
        }
    }

    void onShotRelease(int64_t /*pts*/, const Parameters& frame_json) {
        if (!poss_active_) return;
        ShotRecord s;
        s.release_frame = (int)frame_counter_;
        if (frame_json.contains("shot") && frame_json["shot"].contains("ball_hoop_dist")) {
            s.release_hoop_dist = frame_json["shot"]["ball_hoop_dist"].get<int>();
        }
        // Shot value is authoritative only when court_zone supplied the segmentation-derived
        // inside-three flag for the release point.
        if (frame_json.contains("court_zone") && frame_json["court_zone"].contains("inside_three_point_area")) {
            s.attempt_type = frame_json["court_zone"]["inside_three_point_area"].get<bool>() ? "2pt" : "3pt";
        } else {
            s.attempt_type = "unknown";
        }
        if (frame_json.contains("game")) {
            const auto& gs = frame_json["game"];
            if (gs.contains("clock_sec")) s.game_clock_sec = gs["clock_sec"].get<int>();
            if (gs.contains("period")) s.period_num = gs["period"].get<int>();
        }
        poss_acc_.shots.push_back(s);
    }

    void onShotArrival(int64_t /*pts*/, int hoop_dist) {
        if (!poss_active_) return;
        // Update the last shot awaiting arrival; else start a new shot with arrival only.
        for (auto it = poss_acc_.shots.rbegin(); it != poss_acc_.shots.rend(); ++it) {
            if (it->arrival_frame < 0) {
                it->arrival_frame = (int)frame_counter_;
                if (hoop_dist >= 0) it->hoop_dist = hoop_dist;
                return;
            }
        }
        ShotRecord s;
        s.arrival_frame = (int)frame_counter_;
        if (hoop_dist >= 0) s.hoop_dist = hoop_dist;
        poss_acc_.shots.push_back(s);
    }

    static void applyResult(std::vector<ShotRecord>& shots, const std::string& result, int points,
                            const std::string& attempt_type = std::string()) {
        for (auto it = shots.rbegin(); it != shots.rend(); ++it) {
            if (it->result.empty()) {
                it->result = result;
                if (!attempt_type.empty() && attempt_type != "unknown") it->attempt_type = attempt_type;
                if (points > 0) it->points = points;
                return;
            }
        }
        // No pending shot — record a standalone result (rare: result without release/arrival).
        ShotRecord s;
        s.result = result;
        if (!attempt_type.empty()) s.attempt_type = attempt_type;
        if (points > 0) s.points = points;
        shots.push_back(s);
    }

    void countShotResultForSummary(const std::string& result) {
        ++shot_attempts_total_;
        if (result == "made" || result == "scored") ++shots_made_;
        else if (result == "missed") ++shots_missed_;
        else ++shots_unknown_;
    }

    void onShotResult(const std::string& result, int points, const std::string& attempt_type = std::string()) {
        countShotResultForSummary(result);

        // A deferred possession owns this result (shot was released there).
        if (have_deferred_poss_) {
            applyResult(deferred_poss_.shots, result, points, attempt_type);
            flushDeferredPossession();
            return;
        }
        if (!poss_active_) return;
        applyResult(poss_acc_.shots, result, points, attempt_type);
    }

    void resetRunState() {
        frame_counter_ = 0;
        cached_json_.clear();
        header_written_ = false;
        prev_total_releases_ = 0;
        prev_total_arrivals_ = 0;
        prev_handler_id_ = -1;
        prev_possessing_team_.clear();
        prev_period_num_ = -1;
        prev_shot_clock_sec_ = -1;
        prev_camera_shot_.clear();
        have_last_release_ = false;
        have_pending_shot_ = false;
        last_known_handler_id_ = -1;
        last_known_handler_team_.clear();
        pending_new_handler_id_ = -1;
        pending_new_handler_team_.clear();
        pending_new_handler_frames_ = 0;
        poss_active_ = false;
        poss_acc_ = PossessionAcc();
        have_deferred_poss_ = false;
        deferred_poss_ = PossessionAcc();
        last_game_clock_sec_ = -1;
        last_game_clock_change_t_ = -1.0f;
        clock_currently_stopped_ = false;
        camera_shot_frames_.clear();
        possessions_by_team_.clear();
        shots_made_ = shots_missed_ = shots_unknown_ = shot_attempts_total_ = 0;
        first_game_clock_sec_ = latest_game_clock_sec_ = -1;
    }

public:
    using NodeSISO<av::VideoFrame, av::VideoFrame>::NodeSISO;

    void process() override {
        av::VideoFrame frm = this->source_->get();
        if (!frm) return;
        if (isEofMarker(frm)) {
            flushPossession();
            flushDeferredPossession();
            writeClipSummary();
            file_main_.close();
            file_court_.close();
            file_outlines_.close();
            file_trail_.close();
            file_events_.close();
            file_possessions_.close();
            resetRunState();
            this->sink_->put(frm);
            return;
        }

        ++frame_counter_;

        if (!header_written_) {
            source_w_ = frm.width();
            source_h_ = frm.height();
            auto players_md = tryParse(frm.raw(), player_metadata_key_);
            model_w_ = getOr<float>(players_md, "model_width", 960.0f);
            model_h_ = getOr<float>(players_md, "model_height", 544.0f);
            scale_x_ = (source_w_ > 0 && model_w_ > 0) ? (float)source_w_ / model_w_ : 1.0f;
            scale_y_ = (source_h_ > 0 && model_h_ > 0) ? (float)source_h_ / model_h_ : 1.0f;

            if (!output_file_.empty() && !file_main_.opened && !file_main_.open(output_file_))
                logstream << "metadata_dump: cannot open output file: " << output_file_;
            if (!output_file_court_.empty() && !file_court_.opened && !file_court_.open(output_file_court_))
                logstream << "metadata_dump: cannot open court file: " << output_file_court_;
            if (!output_file_outlines_.empty() && !file_outlines_.opened && !file_outlines_.open(output_file_outlines_))
                logstream << "metadata_dump: cannot open outlines file: " << output_file_outlines_;
            if (!output_file_trail_.empty() && !file_trail_.opened && !file_trail_.open(output_file_trail_))
                logstream << "metadata_dump: cannot open trail file: " << output_file_trail_;
            if (!output_file_events_.empty() && !file_events_.opened && !file_events_.open(output_file_events_))
                logstream << "metadata_dump: cannot open events file: " << output_file_events_;
            if (!output_file_possessions_.empty() && !file_possessions_.opened && !file_possessions_.open(output_file_possessions_))
                logstream << "metadata_dump: cannot open possessions file: " << output_file_possessions_;

            const std::string header = buildHeader().dump();
            if (file_main_.opened) file_main_.writeLine(header);
            if (file_events_.opened) file_events_.writeLine(header);
            if (file_possessions_.opened) file_possessions_.writeLine(header);
            if (file_court_.opened) file_court_.writeLine(header);
            if (file_outlines_.opened) file_outlines_.writeLine(header);
            if (file_trail_.opened) file_trail_.writeLine(header);

            header_written_ = true;
        }

        if (dump_every_n_ > 1 && (frame_counter_ % (uint64_t)dump_every_n_) != 1) {
            if (!cached_json_.empty()) {
                av_dict_set(&frm.raw()->metadata, output_metadata_key_.c_str(), cached_json_.c_str(), 0);
            }
            this->sink_->put(frm);
            return;
        }

        DumpResult dump = buildDump(frm);
        cached_json_ = dump.main.dump();
        av_dict_set(&frm.raw()->metadata, output_metadata_key_.c_str(), cached_json_.c_str(), 0);

        int64_t pts = frm.pts().timestamp();

        // Camera-shot counter (clip summary).
        if (dump.main.contains("camera_shot")) {
            ++camera_shot_frames_[dump.main["camera_shot"].get<std::string>()];
        }

        // Clock-movement events.
        if (dump.main.contains("game") && dump.main["game"].contains("clock_sec")) {
            int sec = dump.main["game"]["clock_sec"].get<int>();
            float now_t = (fps_ > 0) ? (float)frame_counter_ / (float)fps_ : 0.0f;
            if (first_game_clock_sec_ < 0) first_game_clock_sec_ = sec;
            latest_game_clock_sec_ = sec;
            if (sec != last_game_clock_sec_) {
                if (clock_currently_stopped_) {
                    Parameters ev; ev["type"] = "clock_started"; ev["to"] = sec;
                    emitEvent(ev, pts);
                    clock_currently_stopped_ = false;
                }
                last_game_clock_sec_ = sec;
                last_game_clock_change_t_ = now_t;
            } else if (!clock_currently_stopped_ && last_game_clock_change_t_ >= 0.0f &&
                       (now_t - last_game_clock_change_t_) >= clock_stopped_threshold_s_) {
                Parameters ev; ev["type"] = "clock_stopped"; ev["at"] = sec;
                emitEvent(ev, pts);
                clock_currently_stopped_ = true;
            }
        }

        file_main_.writeLine(cached_json_);
        if (!dump.court.is_null()) file_court_.writeLine(dump.court.dump());
        if (!dump.outlines.is_null()) file_outlines_.writeLine(dump.outlines.dump());
        if (!dump.trail.is_null()) file_trail_.writeLine(dump.trail.dump());

        detectAndEmitEvents(dump.main, pts);
        tickPendingShot(dump.main, pts);
        updatePossessionAcc(dump.main, pts);

        if (!output_file_summary_.empty() && summary_update_every_n_ > 0 &&
            (frame_counter_ % (uint64_t)summary_update_every_n_) == 0) {
            writeClipSummaryFile();
        }

        if (debug_log_every_n_ > 0 && (frame_counter_ % (uint64_t)debug_log_every_n_) == 0) {
            logstream << "metadata_dump: frame=" << frame_counter_ << " " << cached_json_;
        }

        this->sink_->put(frm);
    }

    static std::shared_ptr<MetadataDump> create(NodeCreationInfo& nci) {
        EdgeManager& edges = nci.edges;
        const Parameters& params = nci.params;
        auto r = NodeSISO<av::VideoFrame, av::VideoFrame>::template createCommon<MetadataDump>(edges, params);

        if (params.count("player_metadata_key")) r->player_metadata_key_ = params["player_metadata_key"].get<std::string>();
        if (params.count("ball_metadata_key")) r->ball_metadata_key_ = params["ball_metadata_key"].get<std::string>();
        if (params.count("ball_handler_metadata_key")) r->ball_handler_metadata_key_ = params["ball_handler_metadata_key"].get<std::string>();
        if (params.count("possession_metadata_key")) r->possession_metadata_key_ = params["possession_metadata_key"].get<std::string>();
        if (params.count("court_zone_metadata_key")) r->court_zone_metadata_key_ = params["court_zone_metadata_key"].get<std::string>();
        if (params.count("camera_shot_metadata_key")) r->camera_shot_metadata_key_ = params["camera_shot_metadata_key"].get<std::string>();
        if (params.count("shot_events_metadata_key")) r->shot_events_metadata_key_ = params["shot_events_metadata_key"].get<std::string>();
        if (params.count("scoreboard_metadata_key")) r->scoreboard_metadata_key_ = params["scoreboard_metadata_key"].get<std::string>();
        if (params.count("game_state_metadata_key")) r->game_state_metadata_key_ = params["game_state_metadata_key"].get<std::string>();
        if (params.count("viewport_metadata_key")) r->viewport_metadata_key_ = params["viewport_metadata_key"].get<std::string>();
        if (params.count("player_seg_metadata_key")) r->player_seg_metadata_key_ = params["player_seg_metadata_key"].get<std::string>();
        if (params.count("court_seg_metadata_key")) r->court_seg_metadata_key_ = params["court_seg_metadata_key"].get<std::string>();
        if (params.count("output_metadata_key")) r->output_metadata_key_ = params["output_metadata_key"].get<std::string>();
        if (params.count("output_file")) r->output_file_ = params["output_file"].get<std::string>();
        if (params.count("output_file_court")) r->output_file_court_ = params["output_file_court"].get<std::string>();
        if (params.count("output_file_outlines")) r->output_file_outlines_ = params["output_file_outlines"].get<std::string>();
        if (params.count("output_file_trail")) r->output_file_trail_ = params["output_file_trail"].get<std::string>();
        if (params.count("output_file_events")) r->output_file_events_ = params["output_file_events"].get<std::string>();
        if (params.count("output_file_possessions")) r->output_file_possessions_ = params["output_file_possessions"].get<std::string>();
        if (params.count("video_label")) r->video_label_ = params["video_label"].get<std::string>();
        if (params.count("fps")) r->fps_ = params["fps"].get<int>();
        if (params.count("dump_every_n")) r->dump_every_n_ = std::max(1, params["dump_every_n"].get<int>());
        if (params.count("debug_log_every_n")) r->debug_log_every_n_ = params["debug_log_every_n"].get<int>();
        if (params.count("shot_result_wait_frames")) r->shot_result_wait_frames_ = std::max(1, params["shot_result_wait_frames"].get<int>());
        if (params.count("possession_change_confirm_frames")) r->possession_change_confirm_frames_ = std::max(1, params["possession_change_confirm_frames"].get<int>());
        if (params.count("clock_stopped_threshold_s")) r->clock_stopped_threshold_s_ = params["clock_stopped_threshold_s"].get<float>();
        if (params.count("output_file_summary")) r->output_file_summary_ = params["output_file_summary"].get<std::string>();
        if (params.count("summary_update_every_n")) r->summary_update_every_n_ = std::max(1, params["summary_update_every_n"].get<int>());
        if (params.count("court_seg_slot")) r->court_seg_slot_ = params["court_seg_slot"].get<int>();
        if (params.count("player_seg_slot")) r->player_seg_slot_ = params["player_seg_slot"].get<int>();
        if (params.count("mask_threshold")) r->mask_threshold_ = params["mask_threshold"].get<float>();
        if (params.count("contour_simplify_step")) r->contour_simplify_step_ = std::max(1, params["contour_simplify_step"].get<int>());
        return r;
    }
};

DECLNODE(metadata_dump, MetadataDump)
