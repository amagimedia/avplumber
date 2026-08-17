#include "nodes/neural_net/ocr/scoreboard_post_process.hpp"

#include <cmath>
#include <cstdio>
#include <random>

using scoreboard_post::Box;
using scoreboard_post::BoxStabilizer;
using scoreboard_post::PostProcessor;

static int g_failures = 0;

#define CHECK(cond)                                                            \
    do {                                                                       \
        if (!(cond)) {                                                         \
            std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);       \
            ++g_failures;                                                      \
        }                                                                      \
    } while (0)

#define CHECK_NEAR(a, b, eps)                                                  \
    do {                                                                       \
        const double _a = (a), _b = (b);                                       \
        if (std::fabs(_a - _b) > (eps)) {                                      \
            std::printf("FAIL %s:%d: %s (%.6f) != %s (%.6f)\n", __FILE__,     \
                        __LINE__, #a, _a, #b, _b);                             \
            ++g_failures;                                                      \
        }                                                                      \
    } while (0)

static float center_x(const Box& box) { return (box.x1 + box.x2) * 0.5f; }
static float center_y(const Box& box) { return (box.y1 + box.y2) * 0.5f; }

static void test_overlapping_boxes_are_smoothed() {
    BoxStabilizer stabilizer(0.25f, 0.5f);
    const Box first{0.10f, 0.20f, 0.40f, 0.50f};
    const Box second{0.12f, 0.18f, 0.42f, 0.48f};

    stabilizer.update(first);
    const auto update = stabilizer.update(second);

    CHECK(update.matched);
    CHECK(!update.relocated);
    CHECK_NEAR(update.box.x1, 0.105f, 1e-6);
    CHECK_NEAR(update.box.y1, 0.195f, 1e-6);
    CHECK(std::fabs(center_x(update.box) - center_x(first)) <
          std::fabs(center_x(second) - center_x(first)));
}

static void test_relocation_resets_without_blending() {
    BoxStabilizer stabilizer;
    stabilizer.update({0.05f, 0.05f, 0.20f, 0.20f});
    const Box relocated{0.70f, 0.65f, 0.95f, 0.90f};

    const auto update = stabilizer.update(relocated);

    CHECK(!update.matched);
    CHECK(update.relocated);
    CHECK_NEAR(update.box.x1, relocated.x1, 1e-6);
    CHECK_NEAR(update.box.y2, relocated.y2, 1e-6);
}

static void test_random_positions_and_scales_reduce_jitter() {
    std::mt19937 random(0x51A8u);
    std::uniform_real_distribution<float> position(0.02f, 0.75f);
    std::uniform_real_distribution<float> size(0.06f, 0.22f);
    std::uniform_real_distribution<float> jitter(-0.04f, 0.04f);

    for (int scenario = 0; scenario < 500; ++scenario) {
        const float width = size(random);
        const float height = size(random);
        const float x = std::min(position(random), 0.98f - width);
        const float y = std::min(position(random), 0.98f - height);
        const Box anchor{x, y, x + width, y + height};
        const float dx = jitter(random) * width;
        const float dy = jitter(random) * height;
        const Box current{x + dx, y + dy, x + width + dx, y + height + dy};

        BoxStabilizer stabilizer(0.25f, 0.5f);
        stabilizer.update(anchor);
        const auto update = stabilizer.update(current);

        CHECK(update.matched);
        const float raw_motion = std::hypot(center_x(current) - center_x(anchor),
                                            center_y(current) - center_y(anchor));
        const float stable_motion = std::hypot(center_x(update.box) - center_x(anchor),
                                               center_y(update.box) - center_y(anchor));
        CHECK(stable_motion <= raw_motion + 1e-7f);
    }
}

static void test_clear_forgets_previous_layout() {
    BoxStabilizer stabilizer;
    stabilizer.update({0.10f, 0.10f, 0.30f, 0.30f});
    stabilizer.clear();
    const Box next{0.11f, 0.11f, 0.31f, 0.31f};

    const auto update = stabilizer.update(next);

    CHECK(!update.matched);
    CHECK(!update.relocated);
    CHECK_NEAR(update.box.x1, next.x1, 1e-6);
}

static void test_post_processor_resets_components_on_level1_relocation() {
    PostProcessor post;
    post.updateLevel1({0.05f, 0.70f, 0.45f, 0.90f});
    const Box first_component{0.10f, 0.75f, 0.20f, 0.82f};
    post.updateComponent("clock", first_component);
    post.updateComponent("clock", {0.11f, 0.75f, 0.21f, 0.82f});

    const auto relocated = post.updateLevel1({0.55f, 0.05f, 0.95f, 0.25f});
    const Box new_component{0.60f, 0.10f, 0.70f, 0.17f};
    const Box output = post.updateComponent("clock", new_component);

    CHECK(relocated.relocated);
    CHECK_NEAR(output.x1, new_component.x1, 1e-6);
    CHECK_NEAR(output.y1, new_component.y1, 1e-6);
}

static void test_level1_horizontal_extent_change_is_not_relocation() {
    PostProcessor post;
    post.updateLevel1({0.70f, 0.70f, 0.95f, 0.90f});
    post.updateComponent("clock", {0.80f, 0.75f, 0.86f, 0.82f});
    post.updateComponent("clock", {0.81f, 0.75f, 0.87f, 0.82f});

    const auto update = post.updateLevel1({0.40f, 0.70f, 0.95f, 0.90f});
    const Box current_component{0.82f, 0.75f, 0.88f, 0.82f};
    const Box output = post.updateComponent("clock", current_component);

    CHECK(!update.relocated);
    CHECK(output.x1 < current_component.x1);
    CHECK(output.x2 < current_component.x2);
}

static void test_level2_crop_is_layout_independent_and_even_aligned() {
    std::mt19937 random(0xC20Fu);
    std::uniform_int_distribution<int> frame_width(320, 3840);
    std::uniform_int_distribution<int> frame_height(180, 2160);
    std::uniform_real_distribution<float> unit(0.0f, 1.0f);

    for (int scenario = 0; scenario < 500; ++scenario) {
        const int width = frame_width(random);
        const int height = frame_height(random) & ~1;
        const float y1 = unit(random) * (height - 2.0f);
        const float y2 = y1 + 1.0f + unit(random) * (height - y1 - 1.0f);
        const Box scorebug{0.0f, y1, (float)width, y2};
        const auto crop = PostProcessor::level2Crop(scorebug, width, height, 50);

        CHECK(crop[0] == 0);
        CHECK(crop[1] >= 0);
        CHECK(crop[1] % 2 == 0);
        CHECK(crop[2] == width);
        CHECK(crop[3] > 0);
        CHECK(crop[3] % 2 == 0);
        CHECK(crop[1] + crop[3] <= height);
    }
}

int main() {
    test_overlapping_boxes_are_smoothed();
    test_relocation_resets_without_blending();
    test_random_positions_and_scales_reduce_jitter();
    test_clear_forgets_previous_layout();
    test_post_processor_resets_components_on_level1_relocation();
    test_level1_horizontal_extent_change_is_not_relocation();
    test_level2_crop_is_layout_independent_and_even_aligned();
    if (g_failures == 0) {
        std::printf("OK: all scoreboard stabilizer tests passed\n");
        return 0;
    }
    std::printf("FAILED: %d check(s)\n", g_failures);
    return 1;
}
