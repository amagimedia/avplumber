#pragma once

#include <algorithm>
#include <array>
#include <cstddef>

namespace scoreboard_post {

struct Box {
    float x1 = 0.0f;
    float y1 = 0.0f;
    float x2 = 0.0f;
    float y2 = 0.0f;
};

inline float iou(const Box& a, const Box& b) {
    const float x1 = std::max(a.x1, b.x1);
    const float y1 = std::max(a.y1, b.y1);
    const float x2 = std::min(a.x2, b.x2);
    const float y2 = std::min(a.y2, b.y2);
    const float intersection = std::max(0.0f, x2 - x1) * std::max(0.0f, y2 - y1);
    const float area_a = std::max(0.0f, a.x2 - a.x1) * std::max(0.0f, a.y2 - a.y1);
    const float area_b = std::max(0.0f, b.x2 - b.x1) * std::max(0.0f, b.y2 - b.y1);
    const float denominator = area_a + area_b - intersection;
    return denominator > 0.0f ? intersection / denominator : 0.0f;
}

inline float verticalIou(const Box& a, const Box& b) {
    const float top = std::max(a.y1, b.y1);
    const float bottom = std::min(a.y2, b.y2);
    const float intersection = std::max(0.0f, bottom - top);
    const float height_a = std::max(0.0f, a.y2 - a.y1);
    const float height_b = std::max(0.0f, b.y2 - b.y1);
    const float denominator = height_a + height_b - intersection;
    return denominator > 0.0f ? intersection / denominator : 0.0f;
}

class BoxStabilizer {
public:
    struct Update {
        Box box;
        bool matched = false;
        bool relocated = false;
    };

    explicit BoxStabilizer(float alpha = 0.25f, float min_iou = 0.5f)
        : alpha_(std::clamp(alpha, 0.0f, 1.0f)),
          min_iou_(std::clamp(min_iou, 0.0f, 1.0f)) {}

    Update update(const Box& current) {
        if (!valid_) {
            reset(current);
            return {value_, false, false};
        }
        if (iou(value_, current) < min_iou_) {
            reset(current);
            return {value_, false, true};
        }
        const Box observation = robustObservation(current);
        value_ = {
            blend(value_.x1, observation.x1),
            blend(value_.y1, observation.y1),
            blend(value_.x2, observation.x2),
            blend(value_.y2, observation.y2),
        };
        return {value_, true, false};
    }

    void clear() {
        valid_ = false;
        observation_count_ = 0;
    }
    bool valid() const { return valid_; }
    const Box& value() const { return value_; }

private:
    static constexpr std::size_t kObservationWindow = 3;

    static float median(float a, float b, float c) {
        return std::max(std::min(a, b), std::min(std::max(a, b), c));
    }

    void reset(const Box& current) {
        value_ = current;
        observations_[0] = current;
        observation_count_ = 1;
        valid_ = true;
    }

    Box robustObservation(const Box& current) {
        if (observation_count_ < kObservationWindow) {
            observations_[observation_count_++] = current;
        } else {
            observations_[0] = observations_[1];
            observations_[1] = observations_[2];
            observations_[2] = current;
        }
        if (observation_count_ < kObservationWindow) return current;
        return {
            median(observations_[0].x1, observations_[1].x1, observations_[2].x1),
            median(observations_[0].y1, observations_[1].y1, observations_[2].y1),
            median(observations_[0].x2, observations_[1].x2, observations_[2].x2),
            median(observations_[0].y2, observations_[1].y2, observations_[2].y2),
        };
    }

    float blend(float previous, float current) const {
        return previous + alpha_ * (current - previous);
    }

    float alpha_;
    float min_iou_;
    Box value_;
    std::array<Box, kObservationWindow> observations_{};
    std::size_t observation_count_ = 0;
    bool valid_ = false;
};

} // namespace scoreboard_post
