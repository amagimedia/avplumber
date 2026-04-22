#pragma once

#include <cstdint>

struct DrawSegMaskItem {
    int32_t bbox_x1;
    int32_t bbox_y1;
    int32_t bbox_x2;
    int32_t bbox_y2;
    int32_t mask_index;
    int32_t y_color;
    int32_t u_color;
    int32_t v_color;
};
