#pragma once

#include <algorithm>
#include <string>
#include <vector>

namespace scoreboard_components {

template <typename Detection>
void dropUnassignedTeamText(std::vector<Detection>& detections) {
    const auto has_label = [&](const char* label) {
        return std::any_of(detections.begin(), detections.end(), [&](const Detection& detection) {
            return detection.label == label;
        });
    };
    if (!has_label("team_a_name_logo_score") ||
        !has_label("team_b_name_logo_score")) {
        return;
    }

    detections.erase(
        std::remove_if(detections.begin(), detections.end(), [](const Detection& detection) {
            return (detection.cls == 0 && detection.label == "team_name") ||
                   (detection.cls == 1 && detection.label == "team_score");
        }),
        detections.end());
}

} // namespace scoreboard_components
