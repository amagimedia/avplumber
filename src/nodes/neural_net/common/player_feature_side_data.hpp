#pragma once

#include <cstdint>

extern "C" {
#include <libavutil/frame.h>
}

static const AVFrameSideDataType AV_FRAME_DATA_PLAYER_FEATURE = (AVFrameSideDataType)0x59504500;

struct PlayerFeatureSideDataHeader {
    uint32_t num_items;
    uint32_t feature_dim;
    uint32_t reserved0;
    uint32_t reserved1;
};

struct PlayerFeatureSideDataItem {
    int32_t det_index;
    int32_t track_id;
    int32_t seg_index;
    int32_t reserved;
};
