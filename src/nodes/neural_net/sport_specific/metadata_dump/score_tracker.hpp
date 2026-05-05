#pragma once

#include "../../../node_common.hpp"
#include "types.hpp"
#include "io.hpp"
#include "clip_summary.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

namespace metadata_dump {

// Tracks scoreboard state with multi-frame confirmation, resync detection,
// re-lock against bad first-frame OCR, and the visual<->scoreboard team
// identity mapping. Pure: emits events through the returned ProcessResult so
// the caller decides where to write them.
class ScoreTracker {
public:
    struct Config {
        int score_change_confirm_frames = 6;
        int score_relock_confirm_frames = 30;
        int score_change_max_delta = 4;
        int score_resync_max_delta_per_side = 8;
        int score_resync_max_total_delta = 12;
        int team_identity_lock_min_evidence = 2;
        int team_identity_lock_min_margin = 1;
        int scoreboard_team_name_lock_hits = 1;
    };
    Config cfg;

    bool have_confirmed_score = false;
    ScoreState confirmed_score;
    ScoreState score_candidate;
    int score_candidate_frames = 0;
    uint64_t score_candidate_first_frame = 0;
    int64_t score_candidate_first_pts = 0;
    ScoreState score_regression_candidate;
    int score_regression_frames = 0;
    uint64_t score_regression_first_frame = 0;
    int64_t score_regression_first_pts = 0;
    bool have_relocked_away_score = false;
    ScoreState relocked_away_score;

    int team_identity_evidence[2][2] = {{0, 0}, {0, 0}};
    bool team_identity_locked = false;
    bool team_identity_just_locked = false;
    int team_identity_lock_frame = -1;
    std::string visual_to_scoreboard[2];
    std::string scoreboard_to_visual[2];
    std::string locked_scoreboard_team[2];
    std::string scoreboard_team_candidate[2];
    int scoreboard_team_candidate_hits = 0;

    struct ProcessResult {
        std::vector<Parameters> events;
        bool have_shot_result = false;
        ShotResultData shot_result;
    };

    int teamIdentityEvidenceTotal() const {
        int total = 0;
        for (int v = 0; v < 2; ++v) {
            for (int s = 0; s < 2; ++s) total += team_identity_evidence[v][s];
        }
        return total;
    }

    std::string scoreboardTeamAbbrevForSide(const std::string& side) const {
        int idx = scoreboardSideIndex(side);
        if (idx < 0) return {};
        return locked_scoreboard_team[idx];
    }

    std::string scoreboardTeamNameForSide(const std::string& side) const {
        return nbaFullName(scoreboardTeamAbbrevForSide(side));
    }

    Parameters teamIdentityJson(bool include_evidence = true) const {
        Parameters out = Parameters::object();
        out["locked"] = team_identity_locked;
        out["source"] = "scoreboard_verified_shots";
        const bool scoreboard_names_locked =
            !locked_scoreboard_team[0].empty() && !locked_scoreboard_team[1].empty();
        out["scoreboard_team_names_locked"] = scoreboard_names_locked;
        if (team_identity_locked) {
            Parameters v2s = Parameters::object();
            Parameters s2v = Parameters::object();
            Parameters visual = Parameters::object();
            Parameters scoreboard = Parameters::object();
            for (int v = 0; v < 2; ++v) {
                const std::string vt = visualTeamName(v);
                const std::string side = visual_to_scoreboard[v];
                if (side.empty()) continue;
                v2s[vt] = side;
                Parameters vo = Parameters::object();
                vo["scoreboard_side"] = side;
                const std::string abbr = scoreboardTeamAbbrevForSide(side);
                if (!abbr.empty()) {
                    vo["abbrev"] = abbr;
                    const std::string full = nbaFullName(abbr);
                    if (!full.empty()) vo["name"] = full;
                }
                visual[vt] = vo;
            }
            for (int s = 0; s < 2; ++s) {
                const std::string side = scoreboardSideName(s);
                const std::string vt = scoreboard_to_visual[s];
                if (vt.empty()) continue;
                s2v[side] = vt;
                Parameters so = Parameters::object();
                so["visual_team"] = vt;
                if (!locked_scoreboard_team[s].empty()) {
                    so["abbrev"] = locked_scoreboard_team[s];
                    const std::string full = nbaFullName(locked_scoreboard_team[s]);
                    if (!full.empty()) so["name"] = full;
                }
                scoreboard[side] = so;
            }
            if (!v2s.empty()) out["visual_to_scoreboard"] = v2s;
            if (!s2v.empty()) out["scoreboard_to_visual"] = s2v;
            if (!visual.empty()) out["visual"] = visual;
            if (!scoreboard.empty()) out["scoreboard"] = scoreboard;
            if (team_identity_lock_frame >= 0) out["locked_frame"] = team_identity_lock_frame;
        }
        if (!team_identity_locked && scoreboard_names_locked) {
            Parameters scoreboard = Parameters::object();
            for (int s = 0; s < 2; ++s) {
                Parameters so = Parameters::object();
                so["abbrev"] = locked_scoreboard_team[s];
                const std::string full = nbaFullName(locked_scoreboard_team[s]);
                if (!full.empty()) so["name"] = full;
                scoreboard[scoreboardSideName(s)] = so;
            }
            out["scoreboard"] = scoreboard;
        }
        if (include_evidence) {
            Parameters ev = Parameters::object();
            for (int v = 0; v < 2; ++v) {
                Parameters row = Parameters::object();
                row["team_a"] = team_identity_evidence[v][0];
                row["team_b"] = team_identity_evidence[v][1];
                ev[visualTeamName(v)] = row;
            }
            out["evidence"] = ev;
            out["evidence_total"] = teamIdentityEvidenceTotal();
        }
        return out;
    }

    bool plausibleScoreResync(int delta_a, int delta_b) const {
        if (delta_a < 0 || delta_b < 0) return false;
        const int total_delta = delta_a + delta_b;
        if (total_delta <= 0) return false;
        return delta_a <= cfg.score_resync_max_delta_per_side &&
               delta_b <= cfg.score_resync_max_delta_per_side &&
               total_delta <= cfg.score_resync_max_total_delta;
    }

    bool suppressedRelockedAwayScore(const ScoreState& observed, int delta_a, int delta_b) const {
        if (!have_relocked_away_score || !sameScore(observed, relocked_away_score)) return false;
        return delta_a > cfg.score_change_max_delta ||
               delta_b > cfg.score_change_max_delta ||
               (delta_a > 0 && delta_b > 0);
    }

    int pendingShotScoreboardMatchWaitFrames(int shot_result_wait_frames) const {
        return shot_result_wait_frames + std::max(0, cfg.score_change_confirm_frames);
    }

    bool scoreCandidateCouldConfirmPendingShot() const {
        if (!have_confirmed_score || score_candidate_frames <= 0) return false;
        if (sameScore(score_candidate, confirmed_score)) return false;
        if (score_candidate.a < confirmed_score.a || score_candidate.b < confirmed_score.b) return false;
        const int delta_a = score_candidate.a - confirmed_score.a;
        const int delta_b = score_candidate.b - confirmed_score.b;
        const bool one_side_changed = (delta_a > 0 && delta_b == 0) || (delta_b > 0 && delta_a == 0);
        const int delta = std::max(delta_a, delta_b);
        return one_side_changed && (delta == 2 || delta == 3);
    }

    void observeTeamIdentityEvidence(const std::string& visual_team, const std::string& side,
                                      uint64_t frame_counter) {
        const int v = visualTeamIndex(visual_team);
        const int s = scoreboardSideIndex(side);
        if (v < 0 || s < 0) return;
        ++team_identity_evidence[v][s];
        maybeLockTeamIdentity(frame_counter);
    }

    void maybeLockScoreboardTeamNames(const Parameters& game_state_md) {
        std::string a = cleanNbaAbbrev(game_state_md, "team_a_abbrev");
        std::string b = cleanNbaAbbrev(game_state_md, "team_b_abbrev");
        if (a.empty() || b.empty() || a == b) return;
        if (!locked_scoreboard_team[0].empty() && !locked_scoreboard_team[1].empty()) return;

        if (scoreboard_team_candidate[0] == a && scoreboard_team_candidate[1] == b) {
            ++scoreboard_team_candidate_hits;
        } else {
            scoreboard_team_candidate[0] = a;
            scoreboard_team_candidate[1] = b;
            scoreboard_team_candidate_hits = 1;
        }

        if (scoreboard_team_candidate_hits >= cfg.scoreboard_team_name_lock_hits) {
            locked_scoreboard_team[0] = a;
            locked_scoreboard_team[1] = b;
        }
    }

    bool buildScoreboardShotResult(const ScoreState& before,
                                   const ScoreState& after,
                                   const std::string& side,
                                   int delta,
                                   uint64_t frame_counter,
                                   int shot_result_wait_frames,
                                   const PendingShot* pending_shot,
                                   Parameters& score_ev,
                                   Parameters& shot_ev,
                                   ShotResultData& result) {
        if (!pending_shot) {
            if (delta == 2 || delta == 3) score_ev["shot_match"] = "none";
            return false;
        }
        if (delta != 2 && delta != 3) {
            score_ev["shot_match"] = "not_field_goal_delta";
            return false;
        }
        if (pending_shot->frames_waited > pendingShotScoreboardMatchWaitFrames(shot_result_wait_frames)) {
            score_ev["shot_match"] = "pending_shot_expired";
            return false;
        }

        const std::string visual_team = pending_shot->shooting_team;
        observeTeamIdentityEvidence(visual_team, side, frame_counter);
        const std::string scoreboard_team_abbrev = scoreboardTeamAbbrevForSide(side);
        const std::string scoreboard_team_name = scoreboardTeamNameForSide(side);

        const std::string scoreboard_attempt_type = attemptTypeForPoints(delta);
        score_ev["shot_match"] = "pending_shot";
        score_ev["shot_release_frame"] = pending_shot->release_frame;
        score_ev["shot_arrival_frame"] = pending_shot->arrival_frame;
        if (!visual_team.empty()) score_ev["visual_team"] = visual_team;
        if (!scoreboard_team_abbrev.empty()) score_ev["scoreboard_team_abbrev"] = scoreboard_team_abbrev;
        if (!scoreboard_team_name.empty()) score_ev["scoreboard_team_name"] = scoreboard_team_name;
        if (team_identity_locked) score_ev["team_identity"] = teamIdentityJson(false);

        shot_ev["type"] = "shot_result";
        shot_ev["result"] = "made";
        shot_ev["source"] = "scoreboard_delta";
        shot_ev["scoreboard_verified"] = true;
        shot_ev["points"] = delta;
        shot_ev["points_scoreboard"] = delta;
        shot_ev["attempt_type_scoreboard"] = scoreboard_attempt_type;
        shot_ev["score_delta_side"] = side;
        if (!scoreboard_team_abbrev.empty()) shot_ev["scoreboard_team_abbrev"] = scoreboard_team_abbrev;
        if (!scoreboard_team_name.empty()) shot_ev["scoreboard_team_name"] = scoreboard_team_name;
        shot_ev["score_before"] = scoreJson(before);
        shot_ev["score_after"] = scoreJson(after);
        shot_ev["release_frame"] = pending_shot->release_frame;
        shot_ev["arrival_frame"] = pending_shot->arrival_frame;
        if (pending_shot->player_id >= 0) {
            shot_ev["player_id"] = pending_shot->player_id;
            shot_ev["shooter_id"] = pending_shot->player_id;
        }
        if (pending_shot->assist_id >= 0) shot_ev["assist_id"] = pending_shot->assist_id;
        if (!visual_team.empty()) {
            shot_ev["tracking_team"] = visual_team;
            shot_ev["visual_team"] = visual_team;
        }
        if (team_identity_locked) shot_ev["team_identity"] = teamIdentityJson(false);
        if (!pending_shot->attempt_type_detected.empty()) shot_ev["attempt_type_detected"] = pending_shot->attempt_type_detected;
        if (pending_shot->points_detected > 0) shot_ev["points_detected"] = pending_shot->points_detected;
        if (!pending_shot->attempt_type_source.empty()) shot_ev["attempt_type_source"] = pending_shot->attempt_type_source;
        if (pending_shot->attempt_confidence >= 0.0f) {
            shot_ev["attempt_confidence"] = std::round(pending_shot->attempt_confidence * 1000.0f) / 1000.0f;
        }
        if (!pending_shot->attempt_type_source.empty() || !pending_shot->three_point_line_relation.empty()) {
            shot_ev["three_point_line_signed_distance_px"] = pending_shot->three_point_line_signed_distance_px;
            shot_ev["three_point_line_y_delta_px"] = pending_shot->three_point_line_y_delta_px;
        }
        if (!pending_shot->three_point_line_relation.empty()) {
            shot_ev["three_point_line_relation"] = pending_shot->three_point_line_relation;
        }
        // Caller (MetadataDump) appends shot_location separately if needed.
        if (pending_shot->detector_result_seen) {
            shot_ev["detector_result"] = pending_shot->detector_result;
            if (!pending_shot->detector_source.empty()) shot_ev["detector_source"] = pending_shot->detector_source;
            if (pending_shot->detector_confidence >= 0.0f) {
                shot_ev["detector_confidence"] =
                    std::round(pending_shot->detector_confidence * 1000.0f) / 1000.0f;
            }
        }

        const bool detector_disagrees = pending_shot->detector_result_seen &&
                                        pending_shot->detector_result == "missed";
        const bool corrected_attempt = pending_shot->points_detected > 0 &&
                                       pending_shot->points_detected != delta;
        if (detector_disagrees) shot_ev["detector_disagreement"] = true;
        if (corrected_attempt) shot_ev["attempt_type_corrected_by_scoreboard"] = true;

        result.result = "made";
        result.source = "scoreboard_delta";
        result.shooter_id = pending_shot->player_id;
        result.assist_id = pending_shot->assist_id;
        result.points = delta;
        result.attempt_type = scoreboard_attempt_type;
        result.scoreboard_verified = true;
        result.points_detected = pending_shot->points_detected;
        result.attempt_type_detected = pending_shot->attempt_type_detected;
        result.attempt_type_source = pending_shot->attempt_type_source;
        result.attempt_confidence = pending_shot->attempt_confidence;
        result.three_point_line_signed_distance_px = pending_shot->three_point_line_signed_distance_px;
        result.three_point_line_y_delta_px = pending_shot->three_point_line_y_delta_px;
        result.three_point_line_relation = pending_shot->three_point_line_relation;
        result.points_scoreboard = delta;
        result.attempt_type_scoreboard = scoreboard_attempt_type;
        result.score_delta_side = side;
        result.visual_team = visual_team;
        result.scoreboard_team_abbrev = scoreboard_team_abbrev;
        result.scoreboard_team_name = scoreboard_team_name;
        result.score_before = before;
        result.score_after = after;
        result.detector_result = pending_shot->detector_result;
        result.detector_disagreement = detector_disagrees;
        result.attempt_type_corrected_by_scoreboard = corrected_attempt;
        return true;
    }

    ProcessResult processObservation(const Parameters& frame_json,
                                      int64_t pts,
                                      uint64_t frame_counter,
                                      ClipSummary& clip_summary,
                                      int shot_result_wait_frames,
                                      const PendingShot* pending_shot) {
        ProcessResult out;
        ScoreState observed;
        if (!readScoreState(frame_json, observed)) return out;

        if (!have_confirmed_score) {
            if (score_candidate_frames <= 0 || !sameScore(score_candidate, observed)) {
                startCandidate(observed, pts, frame_counter);
            } else {
                ++score_candidate_frames;
            }
            if (score_candidate_frames >= cfg.score_change_confirm_frames) {
                confirmed_score = observed;
                have_confirmed_score = true;
                score_candidate_frames = 0;
                Parameters ev;
                ev["type"] = "score_initialized";
                ev["score"] = scoreJson(confirmed_score);
                ev["confirm_frames"] = cfg.score_change_confirm_frames;
                ev["first_seen_frame"] = score_candidate_first_frame;
                ev["first_seen_pts"] = score_candidate_first_pts;
                out.events.push_back(std::move(ev));
            }
            return out;
        }

        if (sameScore(observed, confirmed_score)) {
            score_candidate_frames = 0;
            return out;
        }

        if (observed.a < confirmed_score.a || observed.b < confirmed_score.b) {
            ++clip_summary.score_ocr_regressions;
            score_candidate_frames = 0;
            if (score_regression_frames <= 0 || !sameScore(score_regression_candidate, observed)) {
                score_regression_candidate = observed;
                score_regression_frames = 1;
                score_regression_first_frame = frame_counter;
                score_regression_first_pts = pts;
            } else {
                ++score_regression_frames;
            }
            if (score_regression_frames >= cfg.score_relock_confirm_frames) {
                const ScoreState bogus = confirmed_score;
                confirmed_score = observed;
                relocked_away_score = bogus;
                have_relocked_away_score = true;
                Parameters ev;
                ev["type"] = "score_relocked";
                ev["score"] = scoreJson(confirmed_score);
                ev["score_rejected"] = scoreJson(bogus);
                ev["confirm_frames"] = cfg.score_relock_confirm_frames;
                ev["first_seen_frame"] = score_regression_first_frame;
                ev["first_seen_pts"] = score_regression_first_pts;
                out.events.push_back(std::move(ev));
                score_regression_frames = 0;
                clip_summary.score_ocr_regressions = 0;
                clip_summary.score_ocr_rejected = 0;
            }
            return out;
        }
        score_regression_frames = 0;

        const int delta_a = observed.a - confirmed_score.a;
        const int delta_b = observed.b - confirmed_score.b;
        if (suppressedRelockedAwayScore(observed, delta_a, delta_b)) {
            ++clip_summary.score_ocr_rejected;
            score_candidate_frames = 0;
            return out;
        }
        const bool one_side_changed = (delta_a > 0 && delta_b == 0) || (delta_b > 0 && delta_a == 0);
        const int delta = std::max(delta_a, delta_b);
        const bool single_scoring_change =
            one_side_changed && delta > 0 && delta <= cfg.score_change_max_delta;
        const bool score_resync = !single_scoring_change && plausibleScoreResync(delta_a, delta_b);
        if (!single_scoring_change && !score_resync) {
            ++clip_summary.score_ocr_rejected;
            score_candidate_frames = 0;
            return out;
        }

        if (score_candidate_frames <= 0 || !sameScore(score_candidate, observed)) {
            startCandidate(observed, pts, frame_counter);
            return out;
        }

        ++score_candidate_frames;
        if (score_candidate_frames < cfg.score_change_confirm_frames) return out;

        const ScoreState before = confirmed_score;
        const ScoreState after = observed;
        const std::string side = delta_a >= delta_b ? "team_a" : "team_b";

        if (score_resync) {
            Parameters ev;
            ev["type"] = "score_resync";
            ev["source"] = "stable_scoreboard_span";
            ev["score_before"] = scoreJson(before);
            ev["score_after"] = scoreJson(after);
            ev["delta_team_a"] = delta_a;
            ev["delta_team_b"] = delta_b;
            ev["confirm_frames"] = cfg.score_change_confirm_frames;
            ev["first_seen_frame"] = score_candidate_first_frame;
            ev["first_seen_pts"] = score_candidate_first_pts;
            out.events.push_back(std::move(ev));
            clip_summary.recordScoreResync(delta_a, delta_b);
            clip_summary.observeConfirmedScoreState(after);
            confirmed_score = observed;
            score_candidate_frames = 0;
            return out;
        }

        Parameters ev;
        ev["type"] = "score_change";
        ev["side"] = side;
        ev["delta"] = delta;
        ev["score_before"] = scoreJson(before);
        ev["score_after"] = scoreJson(after);
        ev["confirm_frames"] = cfg.score_change_confirm_frames;
        ev["first_seen_frame"] = score_candidate_first_frame;
        ev["first_seen_pts"] = score_candidate_first_pts;

        clip_summary.recordScoreEvent(delta);
        clip_summary.recordScoringRun(side, delta);
        clip_summary.observeConfirmedScoreState(after);

        Parameters shot_ev;
        ShotResultData shot_result;
        const bool have_scoreboard_shot_result =
            buildScoreboardShotResult(before, after, side, delta,
                                      frame_counter, shot_result_wait_frames,
                                      pending_shot,
                                      ev, shot_ev, shot_result);

        out.events.push_back(std::move(ev));
        if (have_scoreboard_shot_result) {
            out.events.push_back(std::move(shot_ev));
            out.have_shot_result = true;
            out.shot_result = std::move(shot_result);
        }
        if (team_identity_just_locked) {
            Parameters identity_ev;
            identity_ev["type"] = "team_identity_locked";
            identity_ev["team_identity"] = teamIdentityJson(true);
            out.events.push_back(std::move(identity_ev));
            team_identity_just_locked = false;
        }

        confirmed_score = observed;
        score_candidate_frames = 0;
        return out;
    }

    void reset() {
        ScoreTracker fresh;
        fresh.cfg = cfg;
        *this = fresh;
    }

private:
    void startCandidate(const ScoreState& observed, int64_t pts, uint64_t frame_counter) {
        score_candidate = observed;
        score_candidate_frames = 1;
        score_candidate_first_frame = frame_counter;
        score_candidate_first_pts = pts;
    }

    void maybeLockTeamIdentity(uint64_t frame_counter) {
        if (team_identity_locked) return;
        const int normal = team_identity_evidence[0][0] + team_identity_evidence[1][1];
        const int swapped = team_identity_evidence[0][1] + team_identity_evidence[1][0];
        const bool normal_ready =
            normal >= cfg.team_identity_lock_min_evidence &&
            (normal - swapped) >= cfg.team_identity_lock_min_margin;
        const bool swapped_ready =
            swapped >= cfg.team_identity_lock_min_evidence &&
            (swapped - normal) >= cfg.team_identity_lock_min_margin;
        if (!normal_ready && !swapped_ready) return;
        if (normal_ready) {
            visual_to_scoreboard[0] = "team_a";
            visual_to_scoreboard[1] = "team_b";
            scoreboard_to_visual[0] = "A";
            scoreboard_to_visual[1] = "B";
        } else {
            visual_to_scoreboard[0] = "team_b";
            visual_to_scoreboard[1] = "team_a";
            scoreboard_to_visual[0] = "B";
            scoreboard_to_visual[1] = "A";
        }
        team_identity_locked = true;
        team_identity_just_locked = true;
        team_identity_lock_frame = (int)frame_counter;
    }
};

}  // namespace metadata_dump
