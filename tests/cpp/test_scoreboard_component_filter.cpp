#include "nodes/neural_net/ocr/scoreboard_component_filter.hpp"

#include <algorithm>
#include <cstdio>
#include <string>
#include <vector>

struct Detection {
    int cls;
    std::string label;
};

static int g_failures = 0;

#define CHECK(cond)                                                            \
    do {                                                                       \
        if (!(cond)) {                                                         \
            std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);       \
            ++g_failures;                                                      \
        }                                                                      \
    } while (0)

static bool has_label(const std::vector<Detection>& detections, const char* label) {
    return std::any_of(detections.begin(), detections.end(), [&](const Detection& detection) {
        return detection.label == label;
    });
}

static void test_drops_generic_team_text_when_both_sides_are_assigned() {
    std::vector<Detection> detections = {
        {3, "team_a_name_logo_score"},
        {3, "team_b_name_logo_score"},
        {0, "team_a_name"},
        {0, "team_b_name"},
        {1, "team_a_score"},
        {1, "team_b_score"},
        {0, "team_name"},
        {1, "team_score"},
        {4, "game_clock"},
        {5, "shot_clock"},
        {6, "quarter"},
    };

    scoreboard_components::dropUnassignedTeamText(detections);

    CHECK(!has_label(detections, "team_name"));
    CHECK(!has_label(detections, "team_score"));
    CHECK(has_label(detections, "team_a_name"));
    CHECK(has_label(detections, "team_b_name"));
    CHECK(has_label(detections, "team_a_score"));
    CHECK(has_label(detections, "team_b_score"));
    const int recognizer_inputs = (int)std::count_if(
        detections.begin(), detections.end(), [](const Detection& detection) {
            return detection.cls == 0 || detection.cls == 1 || detection.cls == 4 ||
                   detection.cls == 5 || detection.cls == 6;
        });
    CHECK(recognizer_inputs == 7);
}

static void test_preserves_generic_team_text_without_both_sides() {
    std::vector<Detection> detections = {
        {3, "team_a_name_logo_score"},
        {0, "team_name"},
        {1, "team_score"},
    };

    scoreboard_components::dropUnassignedTeamText(detections);

    CHECK(has_label(detections, "team_name"));
    CHECK(has_label(detections, "team_score"));
}

int main() {
    test_drops_generic_team_text_when_both_sides_are_assigned();
    test_preserves_generic_team_text_without_both_sides();

    if (g_failures != 0) {
        std::printf("FAILED: %d scoreboard component filter checks\n", g_failures);
        return 1;
    }
    std::printf("OK: all scoreboard component filter tests passed\n");
    return 0;
}
