#pragma once

#include "../../../node_common.hpp"
#include "types.hpp"

#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <map>
#include <sstream>
#include <string>

namespace metadata_dump {

// Aggregates clip-level counters across the run and renders the final summary
// JSON. Owns no other tracker's state — buildJson() takes the cross-cutting
// inputs (frame count, confirmed score, team identity) by argument.
class ClipSummary {
public:
    std::map<std::string, int> camera_shot_frames;
    std::map<std::string, int> possessions_by_team;
    int shots_made = 0;
    int shots_missed = 0;
    int shots_unknown = 0;
    int shot_attempts_total = 0;
    int score_events_total = 0;
    int score_events_2pt = 0;
    int score_events_3pt = 0;
    int score_resyncs = 0;
    int score_resync_points_a = 0;
    int score_resync_points_b = 0;
    int scoreboard_verified_shots = 0;
    int scoreboard_unverified_shots = 0;
    int detector_score_disagreements = 0;
    int score_ocr_regressions = 0;
    int score_ocr_rejected = 0;
    int first_game_clock_sec = -1;
    int latest_game_clock_sec = -1;
    int lead_changes = 0;
    int largest_lead = 0;
    std::string current_leader;        // "A", "B", or "" while tied
    int longest_run_points = 0;
    std::string current_run_side;      // "team_a" / "team_b"
    int current_run_points = 0;
    uint64_t total_possession_frames = 0;

    void observeCameraShot(const std::string& name) {
        ++camera_shot_frames[name];
    }

    void observePossession(const std::string& team, int total_frames) {
        ++possessions_by_team[team.empty() ? std::string("?") : team];
        total_possession_frames += (uint64_t)total_frames;
    }

    void observeShotResult(const std::string& result, bool scoreboard_verified,
                           const std::string& source, bool detector_disagreement) {
        ++shot_attempts_total;
        if (result == "made" || result == "scored") ++shots_made;
        else if (result == "missed") ++shots_missed;
        else ++shots_unknown;
        if (scoreboard_verified) ++scoreboard_verified_shots;
        else if (source == "scoreboard_no_delta") ++scoreboard_unverified_shots;
        if (detector_disagreement) ++detector_score_disagreements;
    }

    void observeConfirmedScoreState(const ScoreState& after) {
        const int lead = std::abs(after.a - after.b);
        if (lead > largest_lead) largest_lead = lead;
        std::string new_leader = after.a > after.b ? "A" : (after.b > after.a ? "B" : std::string());
        if (!new_leader.empty() && !current_leader.empty() && new_leader != current_leader) ++lead_changes;
        if (!new_leader.empty()) current_leader = new_leader;
    }

    void recordScoringRun(const std::string& side, int delta) {
        if (current_run_side == side) {
            current_run_points += delta;
        } else {
            current_run_side = side;
            current_run_points = delta;
        }
        if (current_run_points > longest_run_points) longest_run_points = current_run_points;
    }

    void recordScoreEvent(int delta) {
        ++score_events_total;
        if (delta == 2) ++score_events_2pt;
        else if (delta == 3) ++score_events_3pt;
    }

    void recordScoreResync(int delta_a, int delta_b) {
        ++score_resyncs;
        score_resync_points_a += delta_a;
        score_resync_points_b += delta_b;
    }

    void observeGameClock(int sec) {
        if (first_game_clock_sec < 0) first_game_clock_sec = sec;
        latest_game_clock_sec = sec;
    }

    void reset() { *this = ClipSummary{}; }

    Parameters buildJson(uint64_t frame_counter, int fps,
                         const ScoreState& confirmed_score,
                         bool include_ocr_game_state,
                         const Parameters& team_identity_json) const {
        Parameters s;
        s["type"] = "clip_summary";
        s["total_frames"] = frame_counter;
        if (fps > 0) s["duration_sec"] = std::round((float)frame_counter / (float)fps * 100.0f) / 100.0f;
        const bool have_score = confirmed_score.a >= 0 && confirmed_score.b >= 0;
        if (have_score && fps > 0) {
            std::ostringstream final;
            final << "A " << confirmed_score.a << " — B " << confirmed_score.b
                  << " in " << std::fixed << std::setprecision(1)
                  << ((double)frame_counter / (double)fps) << "s";
            s["final"] = final.str();
        }
        if (!camera_shot_frames.empty()) {
            Parameters cs;
            for (const auto& kv : camera_shot_frames) cs[kv.first] = kv.second;
            s["camera_shot_frames"] = cs;
        }
        int poss_total = 0;
        if (!possessions_by_team.empty()) {
            Parameters pb;
            for (const auto& kv : possessions_by_team) { pb[kv.first] = kv.second; poss_total += kv.second; }
            s["possessions_by_team"] = pb;
            s["possessions_total"] = poss_total;
        }
        s["shot_attempts"] = shot_attempts_total;
        if (shot_attempts_total > 0) {
            Parameters r;
            r["scored"] = shots_made;
            r["missed"] = shots_missed;
            r["outcome_unknown"] = shots_unknown;
            r["scoreboard_verified"] = scoreboard_verified_shots;
            r["scoreboard_unverified"] = scoreboard_unverified_shots;
            r["detector_score_disagreements"] = detector_score_disagreements;
            s["shot_results"] = r;
        }
        if (score_events_total > 0) {
            Parameters se;
            se["total"] = score_events_total;
            se["two_point_deltas"] = score_events_2pt;
            se["three_point_deltas"] = score_events_3pt;
            if (score_resyncs > 0) {
                se["resyncs"] = score_resyncs;
                se["resync_points_team_a"] = score_resync_points_a;
                se["resync_points_team_b"] = score_resync_points_b;
            }
            s["score_changes"] = se;
        } else if (score_resyncs > 0) {
            Parameters se;
            se["total"] = 0;
            se["two_point_deltas"] = 0;
            se["three_point_deltas"] = 0;
            se["resyncs"] = score_resyncs;
            se["resync_points_team_a"] = score_resync_points_a;
            se["resync_points_team_b"] = score_resync_points_b;
            s["score_changes"] = se;
        }
        Parameters shape;
        if (score_events_total > 0 || score_resyncs > 0) {
            shape["lead_changes"] = lead_changes;
            shape["largest_lead"] = largest_lead;
            shape["longest_run_points"] = longest_run_points;
        }
        if (poss_total > 0) {
            if (fps > 0 && total_possession_frames > 0) {
                shape["avg_possession_sec"] =
                    std::round((float)total_possession_frames / (float)poss_total / (float)fps * 100.0f) / 100.0f;
            }
            if (fps > 0 && frame_counter > 0) {
                const float duration_min = (float)frame_counter / (float)fps / 60.0f;
                if (duration_min > 0.0f) {
                    shape["pace_per_minute"] =
                        std::round((float)poss_total / duration_min * 100.0f) / 100.0f;
                }
            }
        }
        if (!shape.empty()) s["game_shape"] = shape;
        if (score_ocr_regressions > 0 || score_ocr_rejected > 0) {
            Parameters oq;
            if (score_ocr_regressions > 0) oq["regressions_ignored"] = score_ocr_regressions;
            if (score_ocr_rejected > 0) oq["invalid_changes_ignored"] = score_ocr_rejected;
            s["score_ocr_quality"] = oq;
        }
        if (first_game_clock_sec >= 0) s["first_game_clock_sec"] = first_game_clock_sec;
        if (latest_game_clock_sec >= 0) s["last_game_clock_sec"] = latest_game_clock_sec;
        if (!team_identity_json.is_null() && !team_identity_json.empty()) {
            s["team_identity"] = team_identity_json;
        }
        s["ocr_game_state"] = include_ocr_game_state ? "included" : "disabled";
        return s;
    }

    bool writeFile(const std::string& path, uint64_t frame_counter, int fps,
                   const ScoreState& confirmed_score,
                   bool include_ocr_game_state,
                   const Parameters& team_identity_json) const {
        if (path.empty()) return false;
        std::ofstream out(path, std::ios::out | std::ios::trunc);
        if (!out) return false;
        out << buildJson(frame_counter, fps, confirmed_score,
                         include_ocr_game_state, team_identity_json).dump(2) << '\n';
        return true;
    }
};

}  // namespace metadata_dump
