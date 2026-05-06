#pragma once

#include "../../../node_common.hpp"

#include <map>
#include <string>
#include <vector>

namespace metadata_dump {

struct ScoreState {
    int a = -1;
    int b = -1;
    int period = -1;
    int clock_sec = -1;
    int shot_clock_sec = -1;
};

struct PendingRelease {
    uint64_t frame = 0;
    int64_t pts = 0;
    int player_id = -1;
    int assist_id = -1;
    std::string shooting_team;
    std::string attempt_type;
    int attempt_points = 0;
    std::string attempt_type_source;
    float attempt_confidence = -1.0f;
    int three_point_line_signed_distance_px = 0;
    int three_point_line_y_delta_px = 0;
    std::string three_point_line_relation;
    ScoreState score_at_release;
};

struct PendingShot {
    uint64_t release_frame = 0;
    uint64_t arrival_frame = 0;
    int64_t arrival_pts = 0;
    int player_id = -1;
    int assist_id = -1;
    std::string shooting_team;
    std::string attempt_type_detected;
    int points_detected = 0;
    std::string attempt_type_source;
    float attempt_confidence = -1.0f;
    int three_point_line_signed_distance_px = 0;
    int three_point_line_y_delta_px = 0;
    std::string three_point_line_relation;
    ScoreState score_at_release;
    bool detector_result_seen = false;
    std::string detector_result;
    std::string detector_source;
    float detector_confidence = -1.0f;
    int frames_waited = 0;
};

struct ShotRecord {
    int release_frame = -1;
    int arrival_frame = -1;
    int shooter_id = -1;
    int assist_id = -1;
    int hoop_dist = -1;
    int release_hoop_dist = -1;
    std::string attempt_type;
    int game_clock_sec = -1;
    int period_num = -1;
    std::string result;
    std::string result_source;
    int points = 0;
    bool scoreboard_verified = false;
    int points_detected = 0;
    std::string attempt_type_detected;
    std::string attempt_type_source;
    float attempt_confidence = -1.0f;
    int three_point_line_signed_distance_px = 0;
    int three_point_line_y_delta_px = 0;
    std::string three_point_line_relation;
    int points_scoreboard = 0;
    std::string attempt_type_scoreboard;
    std::string score_delta_side;
    std::string visual_team;
    std::string scoreboard_team_abbrev;
    std::string scoreboard_team_name;
    int score_before_a = -1;
    int score_before_b = -1;
    int score_after_a = -1;
    int score_after_b = -1;
    std::string detector_result;
    bool detector_disagreement = false;
    bool attempt_type_corrected_by_scoreboard = false;
};

struct ShotResultData {
    std::string result;
    std::string source;
    int shooter_id = -1;
    int assist_id = -1;
    int points = 0;
    std::string attempt_type;
    bool scoreboard_verified = false;
    int points_detected = 0;
    std::string attempt_type_detected;
    std::string attempt_type_source;
    float attempt_confidence = -1.0f;
    int three_point_line_signed_distance_px = 0;
    int three_point_line_y_delta_px = 0;
    std::string three_point_line_relation;
    int points_scoreboard = 0;
    std::string attempt_type_scoreboard;
    std::string score_delta_side;
    std::string visual_team;
    std::string scoreboard_team_abbrev;
    std::string scoreboard_team_name;
    ScoreState score_before;
    ScoreState score_after;
    std::string detector_result;
    bool detector_disagreement = false;
    bool attempt_type_corrected_by_scoreboard = false;
};

struct Touch {
    int track_id = -1;
    int start_frame = 0;
    int frames = 0;
};

struct PossessionAcc {
    int possession_id = -1;
    std::string team;
    std::string start_reason;
    uint64_t start_frame = 0;
    int64_t start_pts = 0;
    uint64_t end_frame = 0;
    int64_t end_pts = 0;
    uint64_t next_start_frame = 0;
    int inter_possession_gap_frames = -1;
    int total_frames = 0;
    int frames_controlled = 0;
    int frames_loose = 0;
    int frames_in_flight = 0;
    int closeup_frames = 0;
    std::vector<int> handler_ids;
    std::vector<Touch> touches;
    std::vector<std::string> zones_visited;
    std::map<std::string, int> zone_frames;
    std::vector<ShotRecord> shots;
    int period_at_start = -1;
    int period_at_end = -1;
    bool period_changed_during = false;
    int clock_at_start = -1;
    int clock_at_end = -1;
    int shot_clock_at_end = -1;
    int score_a_at_start = -1;
    int score_b_at_start = -1;
    int score_a_at_end = -1;
    int score_b_at_end = -1;
};

struct MaskInfo {
    int num_masks = 0;
    int w = 0;
    int h = 0;
    const float* data = nullptr;
};

struct DumpResult {
    Parameters main;
    Parameters court;
    Parameters outlines;
    Parameters trail;
};

}  // namespace metadata_dump
