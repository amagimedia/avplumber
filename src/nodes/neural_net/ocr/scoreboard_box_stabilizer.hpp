#pragma once

#include <algorithm>

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
            value_ = current;
            valid_ = true;
            return {value_, false, false};
        }
        if (iou(value_, current) < min_iou_) {
            value_ = current;
            return {value_, false, true};
        }
        value_ = {
            blend(value_.x1, current.x1),
            blend(value_.y1, current.y1),
            blend(value_.x2, current.x2),
            blend(value_.y2, current.y2),
        };
        return {value_, true, false};
    }

    void clear() { valid_ = false; }
    bool valid() const { return valid_; }
    const Box& value() const { return value_; }

private:
    float blend(float previous, float current) const {
        return previous + alpha_ * (current - previous);
    }

    float alpha_;
    float min_iou_;
    Box value_;
    bool valid_ = false;
};

} // namespace scoreboard_post
