#include "../../../node_common.hpp"
#include "types.hpp"
#include "io.hpp"
#include "clip_summary.hpp"
#include "score_tracker.hpp"
#include "shot_tracker.hpp"
#include "possession_tracker.hpp"

extern "C" {
#include <libavutil/dict.h>
#include <libavutil/frame.h>
}

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <map>
#include <sstream>
#include <string>
#include <vector>

using namespace metadata_dump;

class MetadataDump : public NodeSISO<av::VideoFrame, av::VideoFrame>, public ReportsFinishByFlag {
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
    std::string player_feet_metadata_key_ = "player_feet";
    std::string court_seg_metadata_key_ = "yolo_seg";
    std::string output_metadata_key_ = "frame_dump";
    std::string output_file_;
    std::string output_file_court_;
    std::string output_file_outlines_;
    std::string output_file_trail_;
    std::string output_file_events_;
    std::string output_file_summary_;
    int summary_update_every_n_ = 1;
    std::string video_label_;
    int fps_ = 0;
    int schema_ = 4;
    int dump_every_n_ = 1;
    int debug_log_every_n_ = 0;
    int shot_result_wait_frames_ = 25;
    bool include_ocr_game_state_ = false;
    bool emit_ocr_clock_events_ = false;

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
    int prev_handler_id_ = -1;

    int possession_change_confirm_frames_ = 6;

    // Clock-movement tracking
    int last_game_clock_sec_ = -1;
    float last_game_clock_change_t_ = -1.0f;
    bool clock_currently_stopped_ = false;
    float clock_stopped_threshold_s_ = 2.5f;

    ClipSummary clip_summary_;
    int prev_period_num_ = -1;
    int prev_shot_clock_sec_ = -1;
    std::string prev_camera_shot_;

    ScoreTracker score_tracker_;
    ShotTrackerState shot_tracker_;
    PossessionTrackerState possession_tracker_;

    static bool hasAttemptLocation(const std::string& source,
                                   float confidence,
                                   const std::string& relation) {
        return !source.empty() || confidence >= 0.0f || !relation.empty();
    }

    static void appendAttemptLocationFields(Parameters& out,
                                            const std::string& source,
                                            float confidence,
                                            int signed_distance_px,
                                            int y_delta_px,
                                            const std::string& relation) {
        if (!hasAttemptLocation(source, confidence, relation)) return;
        Parameters loc;
        if (!source.empty()) loc["attempt_type_source"] = source;
        if (confidence >= 0.0f) loc["attempt_confidence"] = std::round(confidence * 1000.0f) / 1000.0f;
        loc["three_point_line_signed_distance_px"] = signed_distance_px;
        loc["three_point_line_y_delta_px"] = y_delta_px;
        if (!relation.empty()) loc["three_point_line_relation"] = relation;
        out["shot_location"] = loc;
    }

    static void appendAttemptLocationFields(Parameters& out, const PendingShot& s) {
        appendAttemptLocationFields(out, s.attempt_type_source, s.attempt_confidence,
                                    s.three_point_line_signed_distance_px,
                                    s.three_point_line_y_delta_px,
                                    s.three_point_line_relation);
    }

    static void appendAttemptLocationFields(Parameters& out, const ShotRecord& s) {
        appendAttemptLocationFields(out, s.attempt_type_source, s.attempt_confidence,
                                    s.three_point_line_signed_distance_px,
                                    s.three_point_line_y_delta_px,
                                    s.three_point_line_relation);
    }

    std::string output_file_possessions_;
    std::string output_file_pbp_;
    bool emit_handler_change_events_ = false;
    // Frames at which a brief same-team handler flicker is folded back into
    // the prior touch instead of opening a new one. Default tuned for 25 fps:
    // 6 frames ≈ 240 ms — shorter than any real pass.
    int touch_merge_gap_frames_ = 6;

    JsonFileWriter file_main_;
    JsonFileWriter file_court_;
    JsonFileWriter file_outlines_;
    JsonFileWriter file_trail_;
    JsonFileWriter file_events_;
    JsonFileWriter file_possessions_;
    JsonFileWriter file_pbp_;

    int scaleX(float v) const { return ri(v * scale_x_); }
    int scaleY(float v) const { return ri(v * scale_y_); }
    int scaleDist(float v) const { return ri(v * 0.5f * (scale_x_ + scale_y_)); }

    Parameters teamIdentityJson(bool include_evidence = true) const {
        return score_tracker_.teamIdentityJson(include_evidence);
    }

    int teamIdentityEvidenceTotal() const {
        return score_tracker_.teamIdentityEvidenceTotal();
    }

    std::string scoreboardTeamAbbrevForSide(const std::string& side) const {
        return score_tracker_.scoreboardTeamAbbrevForSide(side);
    }

    std::string scoreboardTeamNameForSide(const std::string& side) const {
        return score_tracker_.scoreboardTeamNameForSide(side);
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

    Parameters traceContour(const float* mask, int w, int h, float scale_x, float scale_y) const {
        return metadata_dump::traceContour(mask, w, h, scale_x, scale_y,
                                            mask_threshold_, contour_simplify_step_);
    }

    Parameters traceContourRegion(const float* mask, int w, int h,
                                  int rx0, int ry0, int rx1, int ry1,
                                  float scale_x, float scale_y) const {
        return metadata_dump::traceContourRegion(mask, w, h, rx0, ry0, rx1, ry1,
                                                 scale_x, scale_y,
                                                 mask_threshold_, contour_simplify_step_);
    }

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
        auto player_feet_md = tryParse(raw, player_feet_metadata_key_);
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
            if (shot_events_md.contains("result_vx")) shot["result_vx"] = shot_events_md["result_vx"];
            if (shot_events_md.contains("result_vy")) shot["result_vy"] = shot_events_md["result_vy"];
            if (shot_events_md.contains("attempt_type")) shot["attempt_type"] = shot_events_md["attempt_type"];
            if (shot_events_md.contains("attempt_points")) shot["attempt_points"] = shot_events_md["attempt_points"];
            if (shot_events_md.contains("shooter_id")) shot["shooter_id"] = shot_events_md["shooter_id"];
            if (shot_events_md.contains("points")) shot["points"] = shot_events_md["points"];
            if (shot_events_md.contains("result_source")) shot["result_source"] = shot_events_md["result_source"];
            if (shot_events_md.contains("ball_hoop_dist")) {
                shot["ball_hoop_dist"] = scaleDist(shot_events_md["ball_hoop_dist"].get<float>());
            }
            if (shot_events_md.contains("attempt_type_source")) shot["attempt_type_source"] = shot_events_md["attempt_type_source"];
            if (shot_events_md.contains("attempt_confidence")) shot["attempt_confidence"] = shot_events_md["attempt_confidence"];
            if (shot_events_md.contains("three_point_line_signed_distance_px")) {
                shot["three_point_line_signed_distance_px"] =
                    scaleDist(shot_events_md["three_point_line_signed_distance_px"].get<float>());
            }
            if (shot_events_md.contains("three_point_line_y_delta_px")) {
                shot["three_point_line_y_delta_px"] =
                    scaleDist(shot_events_md["three_point_line_y_delta_px"].get<float>());
            }
            if (shot_events_md.contains("three_point_line_relation")) {
                shot["three_point_line_relation"] = shot_events_md["three_point_line_relation"];
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

        // Parsed OCR game state is intentionally opt-in. Current OCR is noisy enough
        // that exposing it by default makes the semantic dump harder to interpret.
        if (include_ocr_game_state_ && game_state_md.is_object() && !game_state_md.is_null()) {
            score_tracker_.maybeLockScoreboardTeamNames(game_state_md);
            Parameters game_out;
            if (!score_tracker_.locked_scoreboard_team[0].empty()) {
                game_out["team_a"] = score_tracker_.locked_scoreboard_team[0];
                const std::string full = nbaFullName(score_tracker_.locked_scoreboard_team[0]);
                if (!full.empty()) game_out["team_a_name"] = full;
            } else if (game_state_md.contains("team_a_abbrev")) {
                game_out["team_a"] = game_state_md["team_a_abbrev"];
                if (game_state_md.contains("team_a_name")) game_out["team_a_name"] = game_state_md["team_a_name"];
            }
            if (!score_tracker_.locked_scoreboard_team[1].empty()) {
                game_out["team_b"] = score_tracker_.locked_scoreboard_team[1];
                const std::string full = nbaFullName(score_tracker_.locked_scoreboard_team[1]);
                if (!full.empty()) game_out["team_b_name"] = full;
            } else if (game_state_md.contains("team_b_abbrev")) {
                game_out["team_b"] = game_state_md["team_b_abbrev"];
                if (game_state_md.contains("team_b_name")) game_out["team_b_name"] = game_state_md["team_b_name"];
            }
            if (game_state_md.contains("score_a")) game_out["score_a"] = game_state_md["score_a"];
            if (game_state_md.contains("score_b")) game_out["score_b"] = game_state_md["score_b"];
            if (game_state_md.contains("score_margin")) game_out["score_margin"] = game_state_md["score_margin"];
            if (game_state_md.contains("leading_team")) game_out["leading_team"] = game_state_md["leading_team"];
            if (game_state_md.contains("leading_team_name")) game_out["leading_team_name"] = game_state_md["leading_team_name"];
            if (game_out.contains("score_a") && game_out.contains("score_b")) {
                const int a = game_out["score_a"].get<int>();
                const int b = game_out["score_b"].get<int>();
                if (a > b && !score_tracker_.locked_scoreboard_team[0].empty()) {
                    game_out["leading_team"] = score_tracker_.locked_scoreboard_team[0];
                    const std::string full = nbaFullName(score_tracker_.locked_scoreboard_team[0]);
                    if (!full.empty()) game_out["leading_team_name"] = full;
                } else if (b > a && !score_tracker_.locked_scoreboard_team[1].empty()) {
                    game_out["leading_team"] = score_tracker_.locked_scoreboard_team[1];
                    const std::string full = nbaFullName(score_tracker_.locked_scoreboard_team[1]);
                    if (!full.empty()) game_out["leading_team_name"] = full;
                } else if (a == b) {
                    game_out["leading_team"] = "TIE";
                    game_out.erase("leading_team_name");
                }
            }
            if (game_state_md.contains("period_num")) game_out["period"] = game_state_md["period_num"];
            if (game_state_md.contains("game_clock_sec")) {
                game_out["clock_sec"] = game_state_md["game_clock_sec"];
                game_out["period_clock_remaining_sec"] = game_state_md["game_clock_sec"];
            }
            if (game_state_md.contains("period_clock_remaining_sec")) {
                game_out["period_clock_remaining_sec"] = game_state_md["period_clock_remaining_sec"];
            }
            if (game_state_md.contains("shot_clock_sec")) game_out["shot_clock_sec"] = game_state_md["shot_clock_sec"];
            if (!game_out.empty()) out["game"] = game_out;
        }

        if (score_tracker_.team_identity_locked || teamIdentityEvidenceTotal() > 0 ||
            !score_tracker_.locked_scoreboard_team[0].empty() || !score_tracker_.locked_scoreboard_team[1].empty()) {
            out["team_identity"] = teamIdentityJson(false);
        }

        // Court zone — concise semantic form. Detailed court geometry stays in the
        // court sidecar; the main frame stream only keeps the coarse basketball zone.
        if (court_zone_md.is_object() && !court_zone_md.is_null()) {
            Parameters zone_out;
            std::string source;
            const std::string zone = bestCourtZone(court_zone_md, &source);
            if (isKnownZone(zone)) {
                zone_out["zone"] = zone;
                if (!source.empty()) zone_out["source"] = source;
                if (court_zone_md.contains("hoop_side")) zone_out["hoop_side"] = court_zone_md["hoop_side"];
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

        auto findFootDet = [&](int track_id) -> const Parameters* {
            if (track_id < 0 || !player_feet_md.contains("detections") || !player_feet_md["detections"].is_array()) {
                return nullptr;
            }
            for (const auto& fd : player_feet_md["detections"]) {
                if (!fd.is_object()) continue;
                if (fd.value("track_id", -1) == track_id) return &fd;
            }
            return nullptr;
        };

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
                    if (const Parameters* foot_det = findFootDet(tid)) {
                        Parameters foot;
                        if (foot_det->contains("foot_point") && (*foot_det)["foot_point"].is_array() &&
                            (*foot_det)["foot_point"].size() >= 2) {
                            const float fx = (*foot_det)["foot_point"][0].get<float>();
                            const float fy = (*foot_det)["foot_point"][1].get<float>();
                            foot["point"] = Parameters::array({scaleX(fx), scaleY(fy)});
                        }
                        if (foot_det->contains("left_point") && (*foot_det)["left_point"].is_array() &&
                            (*foot_det)["left_point"].size() >= 2) {
                            const float fx = (*foot_det)["left_point"][0].get<float>();
                            const float fy = (*foot_det)["left_point"][1].get<float>();
                            foot["left"] = Parameters::array({scaleX(fx), scaleY(fy)});
                        }
                        if (foot_det->contains("right_point") && (*foot_det)["right_point"].is_array() &&
                            (*foot_det)["right_point"].size() >= 2) {
                            const float fx = (*foot_det)["right_point"][0].get<float>();
                            const float fy = (*foot_det)["right_point"][1].get<float>();
                            foot["right"] = Parameters::array({scaleX(fx), scaleY(fy)});
                        }
                        if (foot_det->contains("conf")) foot["confidence"] = (*foot_det)["conf"];
                        if (foot_det->contains("source")) foot["source"] = (*foot_det)["source"];
                        if (foot_det->contains("pixels")) foot["pixels"] = (*foot_det)["pixels"];
                        if (foot_det->contains("valid")) foot["valid"] = (*foot_det)["valid"];
                        if (!foot.empty()) p["foot"] = foot;
                    }
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
        if (!file_events_.opened()) return;
        ev["frame"] = frame_counter_;
        ev["pts"] = pts;
        if (fps_ > 0) ev["t"] = std::round((float)frame_counter_ / (float)fps_ * 1000.0f) / 1000.0f;
        file_events_.writeLine(ev.dump());
    }

    int pendingShotScoreboardMatchWaitFrames() const {
        return score_tracker_.pendingShotScoreboardMatchWaitFrames(shot_result_wait_frames_);
    }

    bool scoreCandidateCouldConfirmPendingShot() const {
        return shot_tracker_.have_pending_shot && score_tracker_.scoreCandidateCouldConfirmPendingShot();
    }

    void processScoreObservation(const Parameters& frame_json, int64_t pts) {
        const PendingShot* ps = shot_tracker_.have_pending_shot ? &shot_tracker_.pending_shot : nullptr;
        auto out = score_tracker_.processObservation(frame_json, pts, frame_counter_,
                                                      clip_summary_,
                                                      shot_result_wait_frames_, ps);
        // Score tracker emits the shot_result event but doesn't know about
        // shot_location formatting — append it on the way out.
        if (out.have_shot_result) {
            for (auto& ev : out.events) {
                if (ev.is_object() && ev.contains("type") && ev["type"] == "shot_result") {
                    appendAttemptLocationFields(ev, shot_tracker_.pending_shot);
                }
            }
        }
        for (const auto& ev : out.events) {
            emitEvent(ev, pts);
        }
        if (out.have_shot_result) {
            onShotResult(out.shot_result);
            shot_tracker_.have_pending_shot = false;
        }
    }

    void detectAndEmitEvents(const Parameters& frame_json, int64_t pts) {
        // Update last-known handler before any events (current frame's handler, if any)
        if (frame_json.contains("possession")) {
            const auto& pos = frame_json["possession"];
            int hid = pos.value("handler_id", -1);
            std::string tm = pos.value("handler_team", std::string());
            if (hid >= 0) shot_tracker_.last_known_handler_id = hid;
            if (!tm.empty()) shot_tracker_.last_known_handler_team = tm;
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
        int total_rel = getOr<int>(se_dump, "total_releases", shot_tracker_.prev_total_releases);
        int total_arr = getOr<int>(se_dump, "total_hoop_arrivals", shot_tracker_.prev_total_arrivals);
        if (total_rel > shot_tracker_.prev_total_releases) {
            const int shooter_id = getOr<int>(se_dump, "shooter_id", shot_tracker_.last_known_handler_id);
            const int assist_id = assistIdBeforeShot(possession_tracker_.poss_acc, shooter_id);
            Parameters ev;
            ev["type"] = "shot_release";
            // At the release frame, possession often already flipped to shot_in_air.
            // Prefer the shot detector's debounced pre-release shooter identity.
            if (shooter_id >= 0) {
                ev["player_id"] = shooter_id;
                ev["shooter_id"] = shooter_id;
            }
            if (assist_id >= 0) ev["assist_id"] = assist_id;
            if (!shot_tracker_.last_known_handler_team.empty()) ev["team"] = shot_tracker_.last_known_handler_team;
            if (se_dump.contains("attempt_type")) ev["attempt_type_detected"] = se_dump["attempt_type"];
            if (se_dump.contains("attempt_points")) ev["points_detected"] = se_dump["attempt_points"];
            if (se_dump.contains("attempt_type_source")) ev["attempt_type_source"] = se_dump["attempt_type_source"];
            if (se_dump.contains("attempt_confidence")) ev["attempt_confidence"] = se_dump["attempt_confidence"];
            if (se_dump.contains("three_point_line_signed_distance_px")) {
                ev["three_point_line_signed_distance_px"] = se_dump["three_point_line_signed_distance_px"];
            }
            if (se_dump.contains("three_point_line_y_delta_px")) {
                ev["three_point_line_y_delta_px"] = se_dump["three_point_line_y_delta_px"];
            }
            if (se_dump.contains("three_point_line_relation")) ev["three_point_line_relation"] = se_dump["three_point_line_relation"];
            emitEvent(ev, pts);
            shot_tracker_.have_last_release = true;
            shot_tracker_.last_release.frame = frame_counter_;
            shot_tracker_.last_release.pts = pts;
            shot_tracker_.last_release.player_id = shooter_id;
            shot_tracker_.last_release.assist_id = assist_id;
            shot_tracker_.last_release.shooting_team = shot_tracker_.last_known_handler_team;
            shot_tracker_.last_release.attempt_type = getOr<std::string>(se_dump, "attempt_type", std::string("unknown"));
            shot_tracker_.last_release.attempt_points = getOr<int>(se_dump, "attempt_points", 0);
            shot_tracker_.last_release.attempt_type_source = getOr<std::string>(se_dump, "attempt_type_source", std::string());
            shot_tracker_.last_release.attempt_confidence = getOr<float>(se_dump, "attempt_confidence", -1.0f);
            shot_tracker_.last_release.three_point_line_signed_distance_px =
                getOr<int>(se_dump, "three_point_line_signed_distance_px", 0);
            shot_tracker_.last_release.three_point_line_y_delta_px =
                getOr<int>(se_dump, "three_point_line_y_delta_px", 0);
            shot_tracker_.last_release.three_point_line_relation =
                getOr<std::string>(se_dump, "three_point_line_relation", std::string());
            shot_tracker_.last_release.score_at_release = ScoreState();
            readScoreState(frame_json, shot_tracker_.last_release.score_at_release);
            onShotRelease(pts, frame_json);
            shot_tracker_.prev_total_releases = total_rel;
        }
        if (total_arr > shot_tracker_.prev_total_arrivals) {
            Parameters ev;
            ev["type"] = "shot_hoop_arrival";
            int hoop_dist_val = -1;
            if (se_dump.contains("ball_hoop_dist")) hoop_dist_val = se_dump["ball_hoop_dist"].get<int>();
            if (hoop_dist_val >= 0) ev["ball_hoop_dist"] = hoop_dist_val;
            if (!shot_tracker_.last_known_handler_team.empty()) ev["team"] = shot_tracker_.last_known_handler_team;
            emitEvent(ev, pts);
            onShotArrival(pts, hoop_dist_val);
            // Start pending shot if a recent release is associated
            if (shot_tracker_.have_last_release && (frame_counter_ - shot_tracker_.last_release.frame) < 200) {
                shot_tracker_.have_pending_shot = true;
                shot_tracker_.pending_shot.release_frame = shot_tracker_.last_release.frame;
                shot_tracker_.pending_shot.arrival_frame = frame_counter_;
                shot_tracker_.pending_shot.arrival_pts = pts;
                shot_tracker_.pending_shot.player_id = shot_tracker_.last_release.player_id;
                shot_tracker_.pending_shot.assist_id = shot_tracker_.last_release.assist_id;
                shot_tracker_.pending_shot.shooting_team = shot_tracker_.last_release.shooting_team;
                shot_tracker_.pending_shot.attempt_type_detected = shot_tracker_.last_release.attempt_type;
                shot_tracker_.pending_shot.points_detected = shot_tracker_.last_release.attempt_points;
                shot_tracker_.pending_shot.attempt_type_source = shot_tracker_.last_release.attempt_type_source;
                shot_tracker_.pending_shot.attempt_confidence = shot_tracker_.last_release.attempt_confidence;
                shot_tracker_.pending_shot.three_point_line_signed_distance_px =
                    shot_tracker_.last_release.three_point_line_signed_distance_px;
                shot_tracker_.pending_shot.three_point_line_y_delta_px = shot_tracker_.last_release.three_point_line_y_delta_px;
                shot_tracker_.pending_shot.three_point_line_relation = shot_tracker_.last_release.three_point_line_relation;
                shot_tracker_.pending_shot.score_at_release = shot_tracker_.last_release.score_at_release;
                shot_tracker_.pending_shot.detector_result_seen = false;
                shot_tracker_.pending_shot.detector_result.clear();
                shot_tracker_.pending_shot.detector_source.clear();
                shot_tracker_.pending_shot.detector_confidence = -1.0f;
                shot_tracker_.pending_shot.frames_waited = 0;
                shot_tracker_.have_last_release = false;
            }
            shot_tracker_.prev_total_arrivals = total_arr;
        }
        if (getOr<bool>(se_dump, "result_event", false)) {
            Parameters ev;
            ev["type"] = "shot_detector_result";
            if (!shot_tracker_.last_known_handler_team.empty()) ev["team"] = shot_tracker_.last_known_handler_team;
            ev["result"] = getOr<std::string>(se_dump, "result", std::string("outcome_unknown"));
            if (se_dump.contains("result_source")) ev["source"] = se_dump["result_source"];
            if (se_dump.contains("confidence")) ev["confidence"] = se_dump["confidence"];
            if (se_dump.contains("result_vx")) ev["vx"] = se_dump["result_vx"];
            if (se_dump.contains("result_vy")) ev["vy"] = se_dump["result_vy"];
            if (se_dump.contains("attempt_type")) ev["attempt_type_detected"] = se_dump["attempt_type"];
            if (se_dump.contains("attempt_points")) ev["points_detected"] = se_dump["attempt_points"];
            if (se_dump.contains("points")) ev["points"] = se_dump["points"];
            if (se_dump.contains("attempt_type_source")) ev["attempt_type_source"] = se_dump["attempt_type_source"];
            if (se_dump.contains("attempt_confidence")) ev["attempt_confidence"] = se_dump["attempt_confidence"];
            if (se_dump.contains("three_point_line_signed_distance_px")) {
                ev["three_point_line_signed_distance_px"] = se_dump["three_point_line_signed_distance_px"];
            }
            if (se_dump.contains("three_point_line_y_delta_px")) {
                ev["three_point_line_y_delta_px"] = se_dump["three_point_line_y_delta_px"];
            }
            if (se_dump.contains("three_point_line_relation")) ev["three_point_line_relation"] = se_dump["three_point_line_relation"];
            emitEvent(ev, pts);
            if (shot_tracker_.have_pending_shot) {
                shot_tracker_.pending_shot.detector_result_seen = true;
                shot_tracker_.pending_shot.detector_result = getOr<std::string>(se_dump, "result", std::string("outcome_unknown"));
                shot_tracker_.pending_shot.detector_source = getOr<std::string>(se_dump, "result_source", std::string());
                shot_tracker_.pending_shot.detector_confidence = getOr<float>(se_dump, "confidence", -1.0f);
            }
        }

        // Possession change — debounced: new handler must persist N frames before emitting.
        if (frame_json.contains("possession")) {
            const auto& pos = frame_json["possession"];
            int hid = pos.value("handler_id", -1);
            std::string pteam = pos.value("handler_team", std::string());
            if (pteam.empty()) pteam = pos.value("team", std::string());

            if (hid >= 0 && hid != prev_handler_id_) {
                if (hid == possession_tracker_.pending_new_handler_id) {
                    ++possession_tracker_.pending_new_handler_frames;
                    if (!pteam.empty()) possession_tracker_.pending_new_handler_team = pteam;
                } else {
                    possession_tracker_.pending_new_handler_id = hid;
                    possession_tracker_.pending_new_handler_team = pteam;
                    possession_tracker_.pending_new_handler_frames = 1;
                }
                if (possession_tracker_.pending_new_handler_frames >= possession_change_confirm_frames_) {
                    Parameters ev;
                    const bool team_changed = !possession_tracker_.prev_possessing_team.empty() &&
                                              !possession_tracker_.pending_new_handler_team.empty() &&
                                              possession_tracker_.pending_new_handler_team != possession_tracker_.prev_possessing_team;
                    ev["type"] = team_changed ? "team_control_change" : "handler_change";
                    if (prev_handler_id_ >= 0) ev["from_id"] = prev_handler_id_;
                    ev["to_id"] = possession_tracker_.pending_new_handler_id;
                    if (team_changed) {
                        ev["from_team"] = possession_tracker_.prev_possessing_team;
                        ev["to_team"] = possession_tracker_.pending_new_handler_team;
                    } else if (!possession_tracker_.pending_new_handler_team.empty()) {
                        ev["team"] = possession_tracker_.pending_new_handler_team;
                    }
                    // handler_change is high-volume noise for downstream LLM analysis;
                    // touches are aggregated into the possession record instead.
                    if (team_changed || emit_handler_change_events_) emitEvent(ev, pts);
                    prev_handler_id_ = possession_tracker_.pending_new_handler_id;
                    possession_tracker_.prev_possessing_team = possession_tracker_.pending_new_handler_team;
                    possession_tracker_.pending_new_handler_id = -1;
                    possession_tracker_.pending_new_handler_team.clear();
                    possession_tracker_.pending_new_handler_frames = 0;
                }
            } else {
                // Reverted to the last confirmed handler (flap) — drop pending.
                possession_tracker_.pending_new_handler_id = -1;
                possession_tracker_.pending_new_handler_team.clear();
                possession_tracker_.pending_new_handler_frames = 0;
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
                    emitPbpPeriodBreak(prev_period_num_, pn, pts);
                    if (possession_tracker_.poss_active) possession_tracker_.poss_acc.period_changed_during = true;
                }
                prev_period_num_ = pn;
            }
            if (emit_ocr_clock_events_) {
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

        processScoreObservation(frame_json, pts);
    }

    void tickPendingShot(const Parameters& frame_json, int64_t pts) {
        if (!shot_tracker_.have_pending_shot) return;
        (void)frame_json;
        ++shot_tracker_.pending_shot.frames_waited;
        if (shot_tracker_.pending_shot.frames_waited >= shot_result_wait_frames_) {
            if (scoreCandidateCouldConfirmPendingShot() &&
                shot_tracker_.pending_shot.frames_waited <= pendingShotScoreboardMatchWaitFrames()) {
                return;
            }
            const bool detector_claimed_score = shot_tracker_.pending_shot.detector_result_seen &&
                                                (shot_tracker_.pending_shot.detector_result == "scored" ||
                                                 shot_tracker_.pending_shot.detector_result == "made");
            const bool detector_claimed_miss = shot_tracker_.pending_shot.detector_result_seen &&
                                               shot_tracker_.pending_shot.detector_result == "missed";
            const std::string result = detector_claimed_miss ? "missed" :
                                       (detector_claimed_score ? "scoreboard_unconfirmed" : "outcome_unknown");
            Parameters ev;
            ev["type"] = "shot_result";
            if (!shot_tracker_.pending_shot.shooting_team.empty()) ev["team"] = shot_tracker_.pending_shot.shooting_team;
            if (!shot_tracker_.pending_shot.shooting_team.empty()) ev["visual_team"] = shot_tracker_.pending_shot.shooting_team;
            ev["result"] = result;
            ev["source"] = "scoreboard_no_delta";
            ev["scoreboard_verified"] = false;
            ev["release_frame"] = shot_tracker_.pending_shot.release_frame;
            ev["arrival_frame"] = shot_tracker_.pending_shot.arrival_frame;
            if (shot_tracker_.pending_shot.player_id >= 0) {
                ev["player_id"] = shot_tracker_.pending_shot.player_id;
                ev["shooter_id"] = shot_tracker_.pending_shot.player_id;
            }
            if (shot_tracker_.pending_shot.assist_id >= 0) ev["assist_id"] = shot_tracker_.pending_shot.assist_id;
            if (!shot_tracker_.pending_shot.attempt_type_detected.empty()) ev["attempt_type_detected"] = shot_tracker_.pending_shot.attempt_type_detected;
            if (shot_tracker_.pending_shot.points_detected > 0) ev["points_detected"] = shot_tracker_.pending_shot.points_detected;
            if (!shot_tracker_.pending_shot.attempt_type_source.empty()) ev["attempt_type_source"] = shot_tracker_.pending_shot.attempt_type_source;
            if (shot_tracker_.pending_shot.attempt_confidence >= 0.0f) {
                ev["attempt_confidence"] = std::round(shot_tracker_.pending_shot.attempt_confidence * 1000.0f) / 1000.0f;
            }
            if (!shot_tracker_.pending_shot.attempt_type_source.empty() || !shot_tracker_.pending_shot.three_point_line_relation.empty()) {
                ev["three_point_line_signed_distance_px"] = shot_tracker_.pending_shot.three_point_line_signed_distance_px;
                ev["three_point_line_y_delta_px"] = shot_tracker_.pending_shot.three_point_line_y_delta_px;
            }
            if (!shot_tracker_.pending_shot.three_point_line_relation.empty()) {
                ev["three_point_line_relation"] = shot_tracker_.pending_shot.three_point_line_relation;
            }
            appendAttemptLocationFields(ev, shot_tracker_.pending_shot);
            if (haveScore(shot_tracker_.pending_shot.score_at_release)) ev["score_at_release"] = scoreJson(shot_tracker_.pending_shot.score_at_release);
            if (shot_tracker_.pending_shot.detector_result_seen) {
                ev["detector_result"] = shot_tracker_.pending_shot.detector_result;
                if (!shot_tracker_.pending_shot.detector_source.empty()) ev["detector_source"] = shot_tracker_.pending_shot.detector_source;
                if (shot_tracker_.pending_shot.detector_confidence >= 0.0f) {
                    ev["detector_confidence"] =
                        std::round(shot_tracker_.pending_shot.detector_confidence * 1000.0f) / 1000.0f;
                }
            }
            if (detector_claimed_score) ev["detector_disagreement"] = true;
            emitEvent(ev, pts);

            ShotResultData data;
            data.result = result;
            data.source = "scoreboard_no_delta";
            data.shooter_id = shot_tracker_.pending_shot.player_id;
            data.assist_id = shot_tracker_.pending_shot.assist_id;
            data.scoreboard_verified = false;
            data.points_detected = shot_tracker_.pending_shot.points_detected;
            data.attempt_type_detected = shot_tracker_.pending_shot.attempt_type_detected;
            data.attempt_type_source = shot_tracker_.pending_shot.attempt_type_source;
            data.attempt_confidence = shot_tracker_.pending_shot.attempt_confidence;
            data.three_point_line_signed_distance_px = shot_tracker_.pending_shot.three_point_line_signed_distance_px;
            data.three_point_line_y_delta_px = shot_tracker_.pending_shot.three_point_line_y_delta_px;
            data.three_point_line_relation = shot_tracker_.pending_shot.three_point_line_relation;
            data.visual_team = shot_tracker_.pending_shot.shooting_team;
            data.detector_result = shot_tracker_.pending_shot.detector_result;
            data.detector_disagreement = detector_claimed_score;
            onShotResult(data);
            shot_tracker_.have_pending_shot = false;
        }
    }

    // Picks the shot whose result is most informative for labelling the
    // possession outcome. Made shots win over misses, misses over unknowns,
    // and within a tier we prefer the latest shot.
    static const ShotRecord* representativeShot(const PossessionAcc& a) {
        const ShotRecord* best = nullptr;
        int best_tier = -1;
        for (const auto& sh : a.shots) {
            int tier;
            if (sh.scoreboard_verified && sh.points > 0) tier = 3;
            else if (sh.result == "missed") tier = 2;
            else if (!sh.result.empty()) tier = 1;
            else tier = 0;
            if (!best || tier >= best_tier) { best = &sh; best_tier = tier; }
        }
        return best;
    }

    // Classifies how the possession ended. Used by both the legacy
    // possessions.ndjson record and the LLM-facing pbp.ndjson stream.
    static std::string possessionOutcomeType(const PossessionAcc& a) {
        const ShotRecord* sh = representativeShot(a);
        if (sh) {
            if (sh->scoreboard_verified && sh->points > 0) {
                if (sh->points == 3) return "made_3pt";
                if (sh->points == 2) return "made_2pt";
                if (sh->points == 1) return "made_1pt";
                return "made_shot";
            }
            if (sh->result == "missed") return "missed_shot";
            return "shot_attempt_unverified";
        }
        if (a.period_changed_during) return "end_of_period";
        if (a.clock_at_end == 0) return "end_of_period";
        if (a.shot_clock_at_end == 0) return "shot_clock_violation";
        return "turnover";
    }

    static bool isMadePossessionOutcome(const std::string& outcome) {
        return outcome == "made_1pt" || outcome == "made_2pt" ||
               outcome == "made_3pt" || outcome == "made_shot";
    }

    static double roundTo(double value, double scale) {
        return std::round(value * scale) / scale;
    }

    static double secondsForFrames(int frames, int fps, double scale = 100.0) {
        if (fps <= 0) return 0.0;
        return roundTo((double)frames / (double)fps, scale);
    }

    static int assistIdBeforeShot(const PossessionAcc& a, int shooter_id) {
        if (a.touches.size() < 2) return -1;

        if (shooter_id >= 0) {
            bool found_shooter_touch = false;
            for (auto it = a.touches.rbegin(); it != a.touches.rend(); ++it) {
                if (!found_shooter_touch) {
                    if (it->track_id == shooter_id) found_shooter_touch = true;
                    continue;
                }
                if (it->track_id >= 0 && it->track_id != shooter_id) return it->track_id;
            }
        }

        const int last_id = a.touches.back().track_id;
        for (auto it = a.touches.rbegin() + 1; it != a.touches.rend(); ++it) {
            if (it->track_id >= 0 && it->track_id != last_id) return it->track_id;
        }
        return -1;
    }

    static std::string possessionStartReasonFromPrevious(const PossessionAcc* previous,
                                                         int current_period,
                                                         bool clock_stopped) {
        if (!previous) return "period_start";
        const std::string outcome = possessionOutcomeType(*previous);
        if (previous->period_changed_during || outcome == "end_of_period" ||
            (previous->period_at_end > 0 && current_period > 0 && previous->period_at_end != current_period)) {
            return "period_start";
        }
        if (isMadePossessionOutcome(outcome)) return "inbound_after_score";
        if (outcome == "shot_clock_violation") return "shot_clock_violation";
        if (outcome == "missed_shot" || outcome == "shot_attempt_unverified") return "defensive_rebound";
        if (outcome == "turnover") return clock_stopped ? "jump_ball" : "steal";
        return "jump_ball";
    }

    Parameters zoneSecondsJson(const PossessionAcc& a) const {
        Parameters zone_sec = Parameters::object();
        if (fps_ <= 0) return zone_sec;
        for (const auto& kv : a.zone_frames) {
            if (kv.second <= 0) continue;
            zone_sec[kv.first] = secondsForFrames(kv.second, fps_);
        }
        return zone_sec;
    }

    void stampNextPossessionStart(PossessionAcc& a, uint64_t next_start_frame) const {
        if (next_start_frame == 0) return;
        a.next_start_frame = next_start_frame;
        a.inter_possession_gap_frames =
            next_start_frame > a.end_frame ? (int)(next_start_frame - a.end_frame) : 0;
    }

    void refreshActiveStartReasonFromPrevious(const PossessionAcc& previous) {
        if (!possession_tracker_.poss_active) return;
        possession_tracker_.poss_acc.start_reason =
            possessionStartReasonFromPrevious(&previous, possession_tracker_.poss_acc.period_at_start, clock_currently_stopped_);
    }

    void emitPbpPeriodBreak(int from_period, int to_period, int64_t pts) {
        if (!file_pbp_.opened()) return;
        Parameters ev;
        ev["type"] = "period_break";
        ev["frame"] = frame_counter_;
        ev["pts"] = pts;
        if (fps_ > 0) ev["t"] = roundTo((double)frame_counter_ / (double)fps_, 1000.0);
        ev["from_period"] = from_period;
        ev["to_period"] = to_period;
        file_pbp_.writeLine(ev.dump());
    }

    static bool shouldWritePossession(const PossessionAcc& a) {
        return !a.team.empty() || !a.shots.empty();
    }

    void writePbpRecord(const PossessionAcc& a) {
        if (!file_pbp_.opened()) return;
        if (!shouldWritePossession(a)) return;
        Parameters p;
        p["type"] = "possession";
        p["id"] = a.possession_id;
        if (!a.team.empty()) p["team"] = a.team;
        if (!a.start_reason.empty()) p["start_reason"] = a.start_reason;
        if (fps_ > 0) {
            p["start_t"] = std::round((float)a.start_frame / (float)fps_ * 1000.0f) / 1000.0f;
            p["end_t"] = std::round((float)a.end_frame / (float)fps_ * 1000.0f) / 1000.0f;
            p["duration_sec"] = std::round((float)a.total_frames / (float)fps_ * 100.0f) / 100.0f;
            if (a.inter_possession_gap_frames >= 0) {
                p["inter_possession_gap_sec"] = secondsForFrames(a.inter_possession_gap_frames, fps_, 1000.0);
            }
        }
        if (a.total_frames > 0) {
            p["closeup_fraction"] = roundTo((double)a.closeup_frames / (double)a.total_frames, 1000.0);
        }
        Parameters outcome;
        outcome["type"] = possessionOutcomeType(a);
        const ShotRecord* sh = representativeShot(a);
        if (sh) {
            if (sh->points > 0) outcome["points"] = sh->points;
            if (!sh->attempt_type.empty()) outcome["attempt_type"] = sh->attempt_type;
            if (sh->scoreboard_verified) outcome["scoreboard_verified"] = true;
        }
        p["outcome"] = outcome;
        if (a.score_a_at_start >= 0 && a.score_b_at_start >= 0) {
            Parameters sb; sb["a"] = a.score_a_at_start; sb["b"] = a.score_b_at_start;
            p["score_before"] = sb;
        }
        if (a.score_a_at_end >= 0 && a.score_b_at_end >= 0) {
            Parameters sa; sa["a"] = a.score_a_at_end; sa["b"] = a.score_b_at_end;
            p["score_after"] = sa;
        }
        if (a.period_at_start > 0) p["period"] = a.period_at_start;
        if (a.clock_at_start >= 0) p["clock_at_start_sec"] = a.clock_at_start;
        if (a.clock_at_end >= 0) p["clock_at_end_sec"] = a.clock_at_end;
        Parameters c = Parameters::object();
        const std::string dz = dominantZone(a.zone_frames);
        if (!dz.empty()) c["dominant"] = dz;
        Parameters zone_sec = zoneSecondsJson(a);
        if (!zone_sec.empty()) c["time_per_zone_sec"] = zone_sec;
        if (!c.empty()) p["court"] = c;
        if (!a.touches.empty()) {
            Parameters arr = Parameters::array();
            for (const auto& t : a.touches) {
                Parameters tj;
                tj["id"] = t.track_id;
                if (fps_ > 0) tj["sec"] = std::round((float)t.frames / (float)fps_ * 100.0f) / 100.0f;
                else tj["frames"] = t.frames;
                arr.push_back(tj);
            }
            p["touches"] = arr;
            // pass_count = number of distinct-handler hand-offs (touches.size()-1).
            // It overcounts the rare case where the ball returns to a teammate
            // mid-possession, but undercounts nothing — close enough for an LLM.
            p["pass_count"] = (int)a.touches.size() - 1;
        }
        if (!a.shots.empty()) {
            Parameters arr = Parameters::array();
            for (const auto& s : a.shots) {
                Parameters sj;
                if (s.release_frame >= 0 && fps_ > 0) sj["release_t"] = std::round((float)s.release_frame / (float)fps_ * 1000.0f) / 1000.0f;
                if (s.arrival_frame >= 0 && fps_ > 0) sj["arrival_t"] = std::round((float)s.arrival_frame / (float)fps_ * 1000.0f) / 1000.0f;
                if (s.shooter_id >= 0) sj["shooter_id"] = s.shooter_id;
                if (s.assist_id >= 0) sj["assist_id"] = s.assist_id;
                if (!s.attempt_type.empty()) sj["attempt_type"] = s.attempt_type;
                if (!s.result.empty()) sj["result"] = s.result;
                if (s.points > 0) sj["points"] = s.points;
                if (s.scoreboard_verified) sj["scoreboard_verified"] = true;
                if (s.points_detected > 0) sj["points_detected"] = s.points_detected;
                if (!s.attempt_type_detected.empty()) sj["attempt_type_detected"] = s.attempt_type_detected;
                if (s.attempt_type_corrected_by_scoreboard) sj["attempt_type_corrected_by_scoreboard"] = true;
                appendAttemptLocationFields(sj, s);
                if (s.game_clock_sec >= 0) sj["clock_sec"] = s.game_clock_sec;
                arr.push_back(sj);
            }
            p["shots"] = arr;
        }
        file_pbp_.writeLine(p.dump());
    }

    void writePossessionRecord(const PossessionAcc& a) {
        writePbpRecord(a);
        if (!shouldWritePossession(a)) return;
        // Update clip-summary counters first so siblings that disable
        // possessions output still produce a complete summary line.
        clip_summary_.observePossession(a.team, a.total_frames);
        if (!file_possessions_.opened()) return;
        Parameters p;
        p["type"] = "possession";
        p["possession_id"] = a.possession_id;
        if (!a.team.empty()) p["team"] = a.team;
        if (!a.start_reason.empty()) p["start_reason"] = a.start_reason;
        if (score_tracker_.team_identity_locked) {
            const int v = visualTeamIndex(a.team);
            if (v >= 0 && !score_tracker_.visual_to_scoreboard[v].empty()) {
                p["scoreboard_side"] = score_tracker_.visual_to_scoreboard[v];
                const std::string abbr = scoreboardTeamAbbrevForSide(score_tracker_.visual_to_scoreboard[v]);
                if (!abbr.empty()) {
                    p["team_abbrev"] = abbr;
                    const std::string full = nbaFullName(abbr);
                    if (!full.empty()) p["team_name"] = full;
                }
            }
        }
        Parameters span;
        span["start_frame"] = a.start_frame;
        span["end_frame"] = a.end_frame;
        span["frames"] = a.total_frames;
        if (fps_ > 0) {
            span["start_t"] = std::round((float)a.start_frame / (float)fps_ * 1000.0f) / 1000.0f;
            span["end_t"] = std::round((float)a.end_frame / (float)fps_ * 1000.0f) / 1000.0f;
            span["duration_sec"] = std::round((float)a.total_frames / (float)fps_ * 100.0f) / 100.0f;
            if (a.inter_possession_gap_frames >= 0) {
                span["inter_possession_gap_sec"] = secondsForFrames(a.inter_possession_gap_frames, fps_, 1000.0);
            }
        }
        p["span"] = span;
        if (a.total_frames > 0) {
            p["closeup_fraction"] = roundTo((double)a.closeup_frames / (double)a.total_frames, 1000.0);
        }
        if (!a.handler_ids.empty()) p["handler_track_ids"] = a.handler_ids;

        Parameters court;
        const std::string dz = dominantZone(a.zone_frames);
        if (!dz.empty()) court["dominant_zone"] = dz;
        Parameters zone_counts;
        Parameters visited = Parameters::array();
        for (const auto& kv : a.zone_frames) {
            if (kv.second < 5) continue;
            zone_counts[kv.first] = kv.second;
            visited.push_back(kv.first);
        }
        if (!zone_counts.empty()) court["zone_frames"] = zone_counts;
        Parameters zone_sec = zoneSecondsJson(a);
        if (!zone_sec.empty()) court["time_per_zone_sec"] = zone_sec;
        if (!visited.empty()) court["zones_visited"] = visited;
        if (!court.empty()) p["court"] = court;

        Parameters bs;
        bs["controlled"] = a.frames_controlled;
        bs["loose"] = a.frames_loose;
        bs["in_flight"] = a.frames_in_flight;
        p["ball_state_frames"] = bs;
        const bool has_shots = !a.shots.empty();
        Parameters outcome;
        outcome["type"] = possessionOutcomeType(a);
        if (has_shots) {
            Parameters arr = Parameters::array();
            for (const auto& sh : a.shots) {
                Parameters sj;
                if (sh.release_frame >= 0) sj["release_frame"] = sh.release_frame;
                if (sh.arrival_frame >= 0) sj["arrival_frame"] = sh.arrival_frame;
                if (sh.shooter_id >= 0) sj["shooter_id"] = sh.shooter_id;
                if (sh.assist_id >= 0) sj["assist_id"] = sh.assist_id;
                if (sh.release_hoop_dist >= 0) sj["release_hoop_dist"] = sh.release_hoop_dist;
                if (sh.hoop_dist >= 0) sj["hoop_dist"] = sh.hoop_dist;
                if (!sh.attempt_type.empty()) sj["attempt_type"] = sh.attempt_type;
                if (sh.game_clock_sec >= 0) sj["game_clock_sec"] = sh.game_clock_sec;
                if (sh.period_num >= 0) sj["period_num"] = sh.period_num;
                if (!sh.result.empty()) sj["result"] = sh.result;
                if (!sh.result_source.empty()) sj["result_source"] = sh.result_source;
                if (sh.points > 0) sj["points"] = sh.points;
                if (sh.scoreboard_verified) sj["scoreboard_verified"] = true;
                else if (!sh.result_source.empty()) sj["scoreboard_verified"] = false;
                if (sh.points_detected > 0) sj["points_detected"] = sh.points_detected;
                if (!sh.attempt_type_detected.empty()) sj["attempt_type_detected"] = sh.attempt_type_detected;
                appendAttemptLocationFields(sj, sh);
                if (sh.points_scoreboard > 0) sj["points_scoreboard"] = sh.points_scoreboard;
                if (!sh.attempt_type_scoreboard.empty()) sj["attempt_type_scoreboard"] = sh.attempt_type_scoreboard;
                if (!sh.score_delta_side.empty()) sj["score_delta_side"] = sh.score_delta_side;
                if (!sh.visual_team.empty()) sj["visual_team"] = sh.visual_team;
                if (!sh.scoreboard_team_abbrev.empty()) sj["scoreboard_team_abbrev"] = sh.scoreboard_team_abbrev;
                if (!sh.scoreboard_team_name.empty()) sj["scoreboard_team_name"] = sh.scoreboard_team_name;
                if (sh.score_before_a >= 0 && sh.score_before_b >= 0) {
                    Parameters score_before;
                    score_before["team_a"] = sh.score_before_a;
                    score_before["team_b"] = sh.score_before_b;
                    sj["score_before"] = score_before;
                }
                if (sh.score_after_a >= 0 && sh.score_after_b >= 0) {
                    Parameters score_after;
                    score_after["team_a"] = sh.score_after_a;
                    score_after["team_b"] = sh.score_after_b;
                    sj["score_after"] = score_after;
                }
                if (!sh.detector_result.empty()) sj["detector_result"] = sh.detector_result;
                if (sh.detector_disagreement) sj["detector_disagreement"] = true;
                if (sh.attempt_type_corrected_by_scoreboard) sj["attempt_type_corrected_by_scoreboard"] = true;
                arr.push_back(sj);
                if (!sh.result.empty()) outcome["result"] = sh.result;
                if (sh.points > 0) outcome["points"] = sh.points;
                if (!sh.attempt_type.empty()) outcome["attempt_type"] = sh.attempt_type;
                if (sh.scoreboard_verified) outcome["scoreboard_verified"] = true;
                if (!sh.score_delta_side.empty()) outcome["score_delta_side"] = sh.score_delta_side;
                if (!sh.visual_team.empty()) outcome["visual_team"] = sh.visual_team;
                if (!sh.scoreboard_team_abbrev.empty()) outcome["scoreboard_team_abbrev"] = sh.scoreboard_team_abbrev;
                if (!sh.scoreboard_team_name.empty()) outcome["scoreboard_team_name"] = sh.scoreboard_team_name;
            }
            p["shots"] = arr;
        }
        p["outcome"] = outcome;
        file_possessions_.writeLine(p.dump());
    }

    Parameters teamIdentityForSummary() const {
        if (score_tracker_.team_identity_locked || teamIdentityEvidenceTotal() > 0 ||
            !score_tracker_.locked_scoreboard_team[0].empty() || !score_tracker_.locked_scoreboard_team[1].empty()) {
            return teamIdentityJson(true);
        }
        return Parameters();
    }

    Parameters buildClipSummary() const {
        return clip_summary_.buildJson(frame_counter_, fps_, score_tracker_.confirmed_score,
                                        include_ocr_game_state_,
                                        teamIdentityForSummary());
    }

    void writeClipSummaryFile() {
        clip_summary_.writeFile(output_file_summary_, frame_counter_, fps_,
                                 score_tracker_.confirmed_score, include_ocr_game_state_,
                                 teamIdentityForSummary());
    }

    void writeClipSummary() {
        Parameters s = buildClipSummary();
        std::string json = s.dump();
        if (file_main_.opened()) file_main_.writeLine(json);
        if (file_events_.opened()) file_events_.writeLine(json);
        if (file_possessions_.opened()) file_possessions_.writeLine(json);
        if (file_pbp_.opened()) file_pbp_.writeLine(json);
        writeClipSummaryFile();
    }

    void flushDeferredPossession() {
        if (!possession_tracker_.have_deferred_poss) return;
        possession_tracker_.have_deferred_poss = false;
        writePossessionRecord(possession_tracker_.deferred_poss);
    }

    void flushPossession(uint64_t next_start_frame = 0) {
        if (!possession_tracker_.poss_active) return;
        possession_tracker_.poss_active = false;
        stampNextPossessionStart(possession_tracker_.poss_acc, next_start_frame);
        // If the last shot is unresolved, defer flush so the result lands here.
        const bool last_unresolved = !possession_tracker_.poss_acc.shots.empty() && possession_tracker_.poss_acc.shots.back().result.empty();
        if (last_unresolved && shot_tracker_.have_pending_shot) {
            flushDeferredPossession();  // evict any older deferred
            possession_tracker_.deferred_poss = possession_tracker_.poss_acc;
            possession_tracker_.have_deferred_poss = true;
            return;
        }
        writePossessionRecord(possession_tracker_.poss_acc);
    }

    void updatePossessionAcc(const Parameters& frame_json, int64_t pts) {
        if (!frame_json.contains("possession")) return;
        const auto& pos = frame_json["possession"];
        int pid = pos.value("possession_id", -1);
        if (pid < 0) return;
        std::string team = pos.value("team", std::string());

        const bool poss_starting = !possession_tracker_.poss_active || pid != possession_tracker_.poss_acc.possession_id;
        PossessionAcc previous_acc;
        const bool have_previous_acc = poss_starting && possession_tracker_.poss_active && shouldWritePossession(possession_tracker_.poss_acc);
        if (have_previous_acc) previous_acc = possession_tracker_.poss_acc;
        if (poss_starting) {
            flushPossession(frame_counter_);
            possession_tracker_.poss_acc = PossessionAcc();
            possession_tracker_.poss_active = true;
            possession_tracker_.poss_acc.possession_id = pid;
            possession_tracker_.poss_acc.team = team;
            possession_tracker_.poss_acc.start_frame = frame_counter_;
            possession_tracker_.poss_acc.start_pts = pts;
            if (frame_json.contains("game")) {
                const auto& gs = frame_json["game"];
                possession_tracker_.poss_acc.period_at_start = gs.value("period", -1);
                if (gs.contains("period_clock_remaining_sec")) possession_tracker_.poss_acc.clock_at_start = gs["period_clock_remaining_sec"].get<int>();
                else if (gs.contains("clock_sec")) possession_tracker_.poss_acc.clock_at_start = gs["clock_sec"].get<int>();
                possession_tracker_.poss_acc.score_a_at_start = gs.value("score_a", -1);
                possession_tracker_.poss_acc.score_b_at_start = gs.value("score_b", -1);
            }
            possession_tracker_.poss_acc.start_reason = possessionStartReasonFromPrevious(
                have_previous_acc ? &previous_acc : nullptr,
                possession_tracker_.poss_acc.period_at_start,
                clock_currently_stopped_);
        }
        if (possession_tracker_.poss_acc.team.empty() && !team.empty()) possession_tracker_.poss_acc.team = team;
        possession_tracker_.poss_acc.end_frame = frame_counter_;
        possession_tracker_.poss_acc.end_pts = pts;
        ++possession_tracker_.poss_acc.total_frames;
        if (frame_json.value("camera_shot", std::string()) == "closeup") ++possession_tracker_.poss_acc.closeup_frames;
        std::string bs = pos.value("state", std::string());
        if (bs == "controlled") ++possession_tracker_.poss_acc.frames_controlled;
        else if (bs == "loose") ++possession_tracker_.poss_acc.frames_loose;
        else if (bs == "shot_in_air") ++possession_tracker_.poss_acc.frames_in_flight;
        int hid = pos.value("handler_id", -1);
        if (hid >= 0) {
            if (std::find(possession_tracker_.poss_acc.handler_ids.begin(), possession_tracker_.poss_acc.handler_ids.end(), hid) == possession_tracker_.poss_acc.handler_ids.end()) {
                possession_tracker_.poss_acc.handler_ids.push_back(hid);
            }
            const size_t n = possession_tracker_.poss_acc.touches.size();
            if (n >= 1 && possession_tracker_.poss_acc.touches.back().track_id == hid) {
                ++possession_tracker_.poss_acc.touches.back().frames;
            } else if (n >= 2 &&
                       possession_tracker_.poss_acc.touches[n - 2].track_id == hid &&
                       possession_tracker_.poss_acc.touches.back().frames <= touch_merge_gap_frames_) {
                // Bridge a brief mis-classification of the ball-handler back
                // into the prior touch — same player resumes possession after a
                // tracking flicker, not a real two-pass sequence.
                Touch& prev = possession_tracker_.poss_acc.touches[n - 2];
                prev.frames += possession_tracker_.poss_acc.touches.back().frames + 1;
                possession_tracker_.poss_acc.touches.pop_back();
            } else {
                Touch t;
                t.track_id = hid;
                t.start_frame = (int)frame_counter_;
                t.frames = 1;
                possession_tracker_.poss_acc.touches.push_back(t);
            }
        }
        if (frame_json.contains("game")) {
            const auto& gs = frame_json["game"];
            possession_tracker_.poss_acc.period_at_end = gs.value("period", possession_tracker_.poss_acc.period_at_end);
            if (gs.contains("period_clock_remaining_sec")) possession_tracker_.poss_acc.clock_at_end = gs["period_clock_remaining_sec"].get<int>();
            else if (gs.contains("clock_sec")) possession_tracker_.poss_acc.clock_at_end = gs.value("clock_sec", possession_tracker_.poss_acc.clock_at_end);
            if (gs.contains("shot_clock_sec")) possession_tracker_.poss_acc.shot_clock_at_end = gs["shot_clock_sec"].get<int>();
            possession_tracker_.poss_acc.score_a_at_end = gs.value("score_a", possession_tracker_.poss_acc.score_a_at_end);
            possession_tracker_.poss_acc.score_b_at_end = gs.value("score_b", possession_tracker_.poss_acc.score_b_at_end);
        }
        if (frame_json.contains("court_zone")) {
            const auto& cz = frame_json["court_zone"];
            std::string z = cz.value("zone", std::string());
            if (!isKnownZone(z)) z = cz.value("handler_zone", std::string());
            if (!isKnownZone(z)) z = cz.value("ball_zone", std::string());
            if (isKnownZone(z)) {
                ++possession_tracker_.poss_acc.zone_frames[z];
                if (std::find(possession_tracker_.poss_acc.zones_visited.begin(), possession_tracker_.poss_acc.zones_visited.end(), z) == possession_tracker_.poss_acc.zones_visited.end())
                    possession_tracker_.poss_acc.zones_visited.push_back(z);
            }
        }
    }

    void onShotRelease(int64_t /*pts*/, const Parameters& frame_json) {
        if (!possession_tracker_.poss_active) return;
        ShotRecord s;
        s.release_frame = (int)frame_counter_;
        const Parameters empty_shot = Parameters::object();
        const Parameters& shot = (frame_json.contains("shot") && frame_json["shot"].is_object())
            ? frame_json["shot"] : empty_shot;
        s.shooter_id = getOr<int>(shot, "shooter_id", shot_tracker_.last_known_handler_id);
        s.assist_id = assistIdBeforeShot(possession_tracker_.poss_acc, s.shooter_id);
        s.visual_team = possession_tracker_.poss_acc.team;
        if (shot.contains("ball_hoop_dist")) {
            s.release_hoop_dist = shot["ball_hoop_dist"].get<int>();
        }
        if (shot.contains("attempt_type")) {
            s.attempt_type = shot["attempt_type"].get<std::string>();
            s.attempt_type_detected = s.attempt_type;
        } else if (frame_json.contains("court_zone") && frame_json["court_zone"].contains("inside_three_point_area")) {
            s.attempt_type = frame_json["court_zone"]["inside_three_point_area"].get<bool>() ? "2pt" : "3pt";
        } else {
            s.attempt_type = "unknown";
        }
        s.points_detected = getOr<int>(shot, "attempt_points", 0);
        s.attempt_type_source = getOr<std::string>(shot, "attempt_type_source", std::string());
        s.attempt_confidence = getOr<float>(shot, "attempt_confidence", -1.0f);
        s.three_point_line_signed_distance_px = getOr<int>(shot, "three_point_line_signed_distance_px", 0);
        s.three_point_line_y_delta_px = getOr<int>(shot, "three_point_line_y_delta_px", 0);
        s.three_point_line_relation = getOr<std::string>(shot, "three_point_line_relation", std::string());
        if (frame_json.contains("game")) {
            const auto& gs = frame_json["game"];
            if (gs.contains("period_clock_remaining_sec")) s.game_clock_sec = gs["period_clock_remaining_sec"].get<int>();
            else if (gs.contains("clock_sec")) s.game_clock_sec = gs["clock_sec"].get<int>();
            if (gs.contains("period")) s.period_num = gs["period"].get<int>();
        }
        possession_tracker_.poss_acc.shots.push_back(s);
    }

    void onShotArrival(int64_t /*pts*/, int hoop_dist) {
        if (!possession_tracker_.poss_active) return;
        // Update the last shot awaiting arrival; else start a new shot with arrival only.
        for (auto it = possession_tracker_.poss_acc.shots.rbegin(); it != possession_tracker_.poss_acc.shots.rend(); ++it) {
            if (it->arrival_frame < 0) {
                it->arrival_frame = (int)frame_counter_;
                if (hoop_dist >= 0) it->hoop_dist = hoop_dist;
                return;
            }
        }
        ShotRecord s;
        s.arrival_frame = (int)frame_counter_;
        if (hoop_dist >= 0) s.hoop_dist = hoop_dist;
        possession_tracker_.poss_acc.shots.push_back(s);
    }

    static void applyResult(std::vector<ShotRecord>& shots, const ShotResultData& data) {
        for (auto it = shots.rbegin(); it != shots.rend(); ++it) {
            if (it->result.empty()) {
                it->result = data.result;
                it->result_source = data.source;
                if (data.shooter_id >= 0 && it->shooter_id < 0) it->shooter_id = data.shooter_id;
                if (data.assist_id >= 0 && it->assist_id < 0) it->assist_id = data.assist_id;
                if (!data.attempt_type.empty() && data.attempt_type != "unknown") it->attempt_type = data.attempt_type;
                if (data.points > 0) it->points = data.points;
                it->scoreboard_verified = data.scoreboard_verified;
                if (data.points_detected > 0) it->points_detected = data.points_detected;
                if (!data.attempt_type_detected.empty()) it->attempt_type_detected = data.attempt_type_detected;
                if (!data.attempt_type_source.empty()) it->attempt_type_source = data.attempt_type_source;
                if (data.attempt_confidence >= 0.0f) it->attempt_confidence = data.attempt_confidence;
                if (!data.attempt_type_source.empty() || !data.three_point_line_relation.empty()) {
                    it->three_point_line_signed_distance_px = data.three_point_line_signed_distance_px;
                    it->three_point_line_y_delta_px = data.three_point_line_y_delta_px;
                }
                if (!data.three_point_line_relation.empty()) it->three_point_line_relation = data.three_point_line_relation;
                if (data.points_scoreboard > 0) it->points_scoreboard = data.points_scoreboard;
                if (!data.attempt_type_scoreboard.empty()) it->attempt_type_scoreboard = data.attempt_type_scoreboard;
                it->score_delta_side = data.score_delta_side;
                it->visual_team = data.visual_team;
                it->scoreboard_team_abbrev = data.scoreboard_team_abbrev;
                it->scoreboard_team_name = data.scoreboard_team_name;
                it->score_before_a = data.score_before.a;
                it->score_before_b = data.score_before.b;
                it->score_after_a = data.score_after.a;
                it->score_after_b = data.score_after.b;
                it->detector_result = data.detector_result;
                it->detector_disagreement = data.detector_disagreement;
                it->attempt_type_corrected_by_scoreboard = data.attempt_type_corrected_by_scoreboard;
                return;
            }
        }
        // No pending shot — record a standalone result (rare: result without release/arrival).
        ShotRecord s;
        s.result = data.result;
        s.result_source = data.source;
        s.shooter_id = data.shooter_id;
        s.assist_id = data.assist_id;
        if (!data.attempt_type.empty()) s.attempt_type = data.attempt_type;
        if (data.points > 0) s.points = data.points;
        s.scoreboard_verified = data.scoreboard_verified;
        s.points_detected = data.points_detected;
        s.attempt_type_detected = data.attempt_type_detected;
        s.attempt_type_source = data.attempt_type_source;
        s.attempt_confidence = data.attempt_confidence;
        s.three_point_line_signed_distance_px = data.three_point_line_signed_distance_px;
        s.three_point_line_y_delta_px = data.three_point_line_y_delta_px;
        s.three_point_line_relation = data.three_point_line_relation;
        s.points_scoreboard = data.points_scoreboard;
        s.attempt_type_scoreboard = data.attempt_type_scoreboard;
        s.score_delta_side = data.score_delta_side;
        s.visual_team = data.visual_team;
        s.scoreboard_team_abbrev = data.scoreboard_team_abbrev;
        s.scoreboard_team_name = data.scoreboard_team_name;
        s.score_before_a = data.score_before.a;
        s.score_before_b = data.score_before.b;
        s.score_after_a = data.score_after.a;
        s.score_after_b = data.score_after.b;
        s.detector_result = data.detector_result;
        s.detector_disagreement = data.detector_disagreement;
        s.attempt_type_corrected_by_scoreboard = data.attempt_type_corrected_by_scoreboard;
        shots.push_back(s);
    }

    void onShotResult(const ShotResultData& data) {
        clip_summary_.observeShotResult(data.result, data.scoreboard_verified,
                                         data.source, data.detector_disagreement);

        // A deferred possession owns this result (shot was released there).
        if (possession_tracker_.have_deferred_poss) {
            applyResult(possession_tracker_.deferred_poss.shots, data);
            refreshActiveStartReasonFromPrevious(possession_tracker_.deferred_poss);
            flushDeferredPossession();
            return;
        }
        if (!possession_tracker_.poss_active) return;
        applyResult(possession_tracker_.poss_acc.shots, data);
    }

    void onShotResult(const std::string& result, int points, const std::string& attempt_type = std::string()) {
        ShotResultData data;
        data.result = result;
        data.points = points;
        data.attempt_type = attempt_type;
        onShotResult(data);
    }

    void resetRunState() {
        frame_counter_ = 0;
        cached_json_.clear();
        header_written_ = false;
        prev_handler_id_ = -1;
        prev_period_num_ = -1;
        prev_shot_clock_sec_ = -1;
        prev_camera_shot_.clear();
        shot_tracker_.reset();
        possession_tracker_.reset();
        last_game_clock_sec_ = -1;
        last_game_clock_change_t_ = -1.0f;
        clock_currently_stopped_ = false;
        clip_summary_.reset();
        score_tracker_.reset();
    }

public:
    using NodeSISO<av::VideoFrame, av::VideoFrame>::NodeSISO;
    bool consumeEofIfPresent() override {
        return false;
    }

    void process() override {
        av::VideoFrame frm = this->source_->get();
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
            file_pbp_.close();
            resetRunState();
            this->sink_->put(frm);
            this->finished_ = true;
            return;
        }
        if (!frm) return;

        ++frame_counter_;

        if (!header_written_) {
            source_w_ = frm.width();
            source_h_ = frm.height();
            auto players_md = tryParse(frm.raw(), player_metadata_key_);
            model_w_ = getOr<float>(players_md, "model_width", 960.0f);
            model_h_ = getOr<float>(players_md, "model_height", 544.0f);
            scale_x_ = (source_w_ > 0 && model_w_ > 0) ? (float)source_w_ / model_w_ : 1.0f;
            scale_y_ = (source_h_ > 0 && model_h_ > 0) ? (float)source_h_ / model_h_ : 1.0f;

            if (!output_file_.empty() && !file_main_.opened() && !file_main_.open(output_file_))
                logstream << "metadata_dump: cannot open output file: " << output_file_;
            if (!output_file_court_.empty() && !file_court_.opened() && !file_court_.open(output_file_court_))
                logstream << "metadata_dump: cannot open court file: " << output_file_court_;
            if (!output_file_outlines_.empty() && !file_outlines_.opened() && !file_outlines_.open(output_file_outlines_))
                logstream << "metadata_dump: cannot open outlines file: " << output_file_outlines_;
            if (!output_file_trail_.empty() && !file_trail_.opened() && !file_trail_.open(output_file_trail_))
                logstream << "metadata_dump: cannot open trail file: " << output_file_trail_;
            if (!output_file_events_.empty() && !file_events_.opened() && !file_events_.open(output_file_events_))
                logstream << "metadata_dump: cannot open events file: " << output_file_events_;
            if (!output_file_possessions_.empty() && !file_possessions_.opened() && !file_possessions_.open(output_file_possessions_))
                logstream << "metadata_dump: cannot open possessions file: " << output_file_possessions_;
            if (!output_file_pbp_.empty() && !file_pbp_.opened() && !file_pbp_.open(output_file_pbp_))
                logstream << "metadata_dump: cannot open pbp file: " << output_file_pbp_;

            const std::string header = buildHeader().dump();
            if (file_main_.opened()) file_main_.writeLine(header);
            if (file_events_.opened()) file_events_.writeLine(header);
            if (file_possessions_.opened()) file_possessions_.writeLine(header);
            if (file_pbp_.opened()) file_pbp_.writeLine(header);
            if (file_court_.opened()) file_court_.writeLine(header);
            if (file_outlines_.opened()) file_outlines_.writeLine(header);
            if (file_trail_.opened()) file_trail_.writeLine(header);

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
            ++clip_summary_.camera_shot_frames[dump.main["camera_shot"].get<std::string>()];
        }

        // Clock-movement events.
        if (dump.main.contains("game") && dump.main["game"].contains("clock_sec")) {
            int sec = dump.main["game"]["clock_sec"].get<int>();
            float now_t = (fps_ > 0) ? (float)frame_counter_ / (float)fps_ : 0.0f;
            if (clip_summary_.first_game_clock_sec < 0) clip_summary_.first_game_clock_sec = sec;
            clip_summary_.latest_game_clock_sec = sec;
            if (sec != last_game_clock_sec_) {
                if (emit_ocr_clock_events_ && clock_currently_stopped_) {
                    Parameters ev; ev["type"] = "clock_started"; ev["to"] = sec;
                    emitEvent(ev, pts);
                }
                clock_currently_stopped_ = false;
                last_game_clock_sec_ = sec;
                last_game_clock_change_t_ = now_t;
            } else if (emit_ocr_clock_events_ && !clock_currently_stopped_ && last_game_clock_change_t_ >= 0.0f &&
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

    void applyParams(const Parameters& params) {
        if (params.count("player_metadata_key")) player_metadata_key_ = params["player_metadata_key"].get<std::string>();
        if (params.count("ball_metadata_key")) ball_metadata_key_ = params["ball_metadata_key"].get<std::string>();
        if (params.count("ball_handler_metadata_key")) ball_handler_metadata_key_ = params["ball_handler_metadata_key"].get<std::string>();
        if (params.count("possession_metadata_key")) possession_metadata_key_ = params["possession_metadata_key"].get<std::string>();
        if (params.count("court_zone_metadata_key")) court_zone_metadata_key_ = params["court_zone_metadata_key"].get<std::string>();
        if (params.count("camera_shot_metadata_key")) camera_shot_metadata_key_ = params["camera_shot_metadata_key"].get<std::string>();
        if (params.count("shot_events_metadata_key")) shot_events_metadata_key_ = params["shot_events_metadata_key"].get<std::string>();
        if (params.count("scoreboard_metadata_key")) scoreboard_metadata_key_ = params["scoreboard_metadata_key"].get<std::string>();
        if (params.count("game_state_metadata_key")) game_state_metadata_key_ = params["game_state_metadata_key"].get<std::string>();
        if (params.count("viewport_metadata_key")) viewport_metadata_key_ = params["viewport_metadata_key"].get<std::string>();
        if (params.count("player_seg_metadata_key")) player_seg_metadata_key_ = params["player_seg_metadata_key"].get<std::string>();
        if (params.count("player_feet_metadata_key")) player_feet_metadata_key_ = params["player_feet_metadata_key"].get<std::string>();
        if (params.count("court_seg_metadata_key")) court_seg_metadata_key_ = params["court_seg_metadata_key"].get<std::string>();
        if (params.count("output_metadata_key")) output_metadata_key_ = params["output_metadata_key"].get<std::string>();
        if (params.count("output_file")) output_file_ = params["output_file"].get<std::string>();
        if (params.count("output_file_court")) output_file_court_ = params["output_file_court"].get<std::string>();
        if (params.count("output_file_outlines")) output_file_outlines_ = params["output_file_outlines"].get<std::string>();
        if (params.count("output_file_trail")) output_file_trail_ = params["output_file_trail"].get<std::string>();
        if (params.count("output_file_events")) output_file_events_ = params["output_file_events"].get<std::string>();
        if (params.count("output_file_possessions")) output_file_possessions_ = params["output_file_possessions"].get<std::string>();
        if (params.count("output_file_pbp")) output_file_pbp_ = params["output_file_pbp"].get<std::string>();
        if (params.count("emit_handler_change_events")) emit_handler_change_events_ = params["emit_handler_change_events"].get<bool>();
        if (params.count("touch_merge_gap_frames")) touch_merge_gap_frames_ = std::max(0, params["touch_merge_gap_frames"].get<int>());
        if (params.count("video_label")) video_label_ = params["video_label"].get<std::string>();
        if (params.count("fps")) fps_ = params["fps"].get<int>();
        if (params.count("dump_every_n")) dump_every_n_ = std::max(1, params["dump_every_n"].get<int>());
        if (params.count("debug_log_every_n")) debug_log_every_n_ = params["debug_log_every_n"].get<int>();
        if (params.count("shot_result_wait_frames")) shot_result_wait_frames_ = std::max(1, params["shot_result_wait_frames"].get<int>());
        if (params.count("score_change_confirm_frames")) score_tracker_.cfg.score_change_confirm_frames = std::max(1, params["score_change_confirm_frames"].get<int>());
        if (params.count("score_relock_confirm_frames")) score_tracker_.cfg.score_relock_confirm_frames = std::max(1, params["score_relock_confirm_frames"].get<int>());
        if (params.count("score_change_max_delta")) score_tracker_.cfg.score_change_max_delta = std::max(1, params["score_change_max_delta"].get<int>());
        if (params.count("score_resync_max_delta_per_side")) score_tracker_.cfg.score_resync_max_delta_per_side = std::max(1, params["score_resync_max_delta_per_side"].get<int>());
        if (params.count("score_resync_max_total_delta")) score_tracker_.cfg.score_resync_max_total_delta = std::max(1, params["score_resync_max_total_delta"].get<int>());
        if (params.count("team_identity_lock_min_evidence")) score_tracker_.cfg.team_identity_lock_min_evidence = std::max(1, params["team_identity_lock_min_evidence"].get<int>());
        if (params.count("team_identity_lock_min_margin")) score_tracker_.cfg.team_identity_lock_min_margin = std::max(0, params["team_identity_lock_min_margin"].get<int>());
        if (params.count("scoreboard_team_name_lock_hits")) score_tracker_.cfg.scoreboard_team_name_lock_hits = std::max(1, params["scoreboard_team_name_lock_hits"].get<int>());
        if (params.count("include_ocr_game_state")) include_ocr_game_state_ = params["include_ocr_game_state"].get<bool>();
        if (params.count("emit_ocr_clock_events")) emit_ocr_clock_events_ = params["emit_ocr_clock_events"].get<bool>();
        if (params.count("possession_change_confirm_frames")) possession_change_confirm_frames_ = std::max(1, params["possession_change_confirm_frames"].get<int>());
        if (params.count("clock_stopped_threshold_s")) clock_stopped_threshold_s_ = params["clock_stopped_threshold_s"].get<float>();
        if (params.count("output_file_summary")) output_file_summary_ = params["output_file_summary"].get<std::string>();
        if (params.count("summary_update_every_n")) summary_update_every_n_ = std::max(1, params["summary_update_every_n"].get<int>());
        if (params.count("court_seg_slot")) court_seg_slot_ = params["court_seg_slot"].get<int>();
        if (params.count("player_seg_slot")) player_seg_slot_ = params["player_seg_slot"].get<int>();
        if (params.count("mask_threshold")) mask_threshold_ = params["mask_threshold"].get<float>();
        if (params.count("contour_simplify_step")) contour_simplify_step_ = std::max(1, params["contour_simplify_step"].get<int>());
    }

    // Each split-node subclass uses these to disable the file outputs that
    // belong to a different sibling, so only one node ever writes a given file.
    void disableFrameOutputs() {
        output_file_.clear();
        output_file_court_.clear();
        output_file_outlines_.clear();
        output_file_trail_.clear();
        // The frame_dump metadata key is the chain hand-off; only the frame
        // node should publish it, so non-frame nodes clear it too.
        output_metadata_key_.clear();
    }
    void disableEventsOutputs() {
        output_file_events_.clear();
    }
    void disablePossessionsOutputs() {
        output_file_possessions_.clear();
        output_file_pbp_.clear();
        output_file_summary_.clear();
    }
};

// Three sibling sink nodes. Each runs the same per-frame engine
// independently (state machines are deterministic), and each writes only
// its slice of the eight output files.

class MetadataDumpFrame : public MetadataDump {
public:
    using MetadataDump::MetadataDump;
    static std::shared_ptr<MetadataDumpFrame> create(NodeCreationInfo& nci) {
        auto r = NodeSISO<av::VideoFrame, av::VideoFrame>::template createCommon<MetadataDumpFrame>(nci.edges, nci.params);
        r->applyParams(nci.params);
        r->disableEventsOutputs();
        r->disablePossessionsOutputs();
        return r;
    }
};

class MetadataDumpEvents : public MetadataDump {
public:
    using MetadataDump::MetadataDump;
    static std::shared_ptr<MetadataDumpEvents> create(NodeCreationInfo& nci) {
        auto r = NodeSISO<av::VideoFrame, av::VideoFrame>::template createCommon<MetadataDumpEvents>(nci.edges, nci.params);
        r->applyParams(nci.params);
        r->disableFrameOutputs();
        r->disablePossessionsOutputs();
        return r;
    }
};

class MetadataDumpPossessions : public MetadataDump {
public:
    using MetadataDump::MetadataDump;
    static std::shared_ptr<MetadataDumpPossessions> create(NodeCreationInfo& nci) {
        auto r = NodeSISO<av::VideoFrame, av::VideoFrame>::template createCommon<MetadataDumpPossessions>(nci.edges, nci.params);
        r->applyParams(nci.params);
        r->disableFrameOutputs();
        r->disableEventsOutputs();
        return r;
    }
};

DECLNODE(metadata_dump_frame, MetadataDumpFrame)
DECLNODE(metadata_dump_events, MetadataDumpEvents)
DECLNODE(metadata_dump_possessions, MetadataDumpPossessions)
