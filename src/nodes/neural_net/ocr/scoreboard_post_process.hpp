#pragma once

#include "scoreboard_box_stabilizer.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <string>
#include <unordered_map>

namespace scoreboard_post {

class PostProcessor {
public:
    struct Level1Update {
        Box box;
        bool relocated = false;
    };

    Level1Update updateLevel1(const Box& current) {
        // Level-2 consumes a full-width strip, so Level-1 X/width variation does
        // not identify a different scoreboard. Only a move to another vertical
        // band invalidates component and temporal state.
        const bool relocated = level1_.valid()
            && verticalIou(level1_.value(), current) < 0.5f;
        if (relocated) {
            level1_.clear();
            components_.clear();
        }
        const BoxStabilizer::Update update = level1_.update(current);
        return {update.box, relocated};
    }

    Box updateComponent(const std::string& label, const Box& current) {
        auto [entry, inserted] = components_.try_emplace(label);
        (void)inserted;
        return entry->second.update(current).box;
    }

    void clear() {
        level1_.clear();
        components_.clear();
    }

    void clearComponents() { components_.clear(); }

    static std::array<int, 4> level2Crop(const Box& scorebug, int frame_width,
                                         int frame_height, int vertical_padding) {
        int top = std::max(
            0, (int)std::floor(scorebug.y1) - std::max(0, vertical_padding));
        top &= ~1;
        int bottom = std::min(
            frame_height,
            (int)std::ceil(scorebug.y2) + std::max(0, vertical_padding));
        bottom = std::min(frame_height, (bottom + 1) & ~1);
        return {0, top, frame_width, std::max(1, bottom - top)};
    }

private:
    // Relocation is evaluated above using vertical overlap. Once the band
    // matches, smooth all four coordinates even if horizontal extent changes.
    BoxStabilizer level1_{0.25f, 0.0f};
    std::unordered_map<std::string, BoxStabilizer> components_;
};

} // namespace scoreboard_post
