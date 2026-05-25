#pragma once

#pragma pack(push, 1)

struct TexInfo {
    uint32_t width;
    uint32_t height;
    uint32_t stride;
    uint32_t pixel_format;
    uint64_t modifier;
    uint64_t offset;
    uint64_t timestamp;
    uint64_t frame_count;
};

#pragma pack(pop)

#define PIX_FMT_RGBA ('R' << 24 | 'G' << 16 | 'B' << 8 | 'A')
#define PIX_FMT_BGRA ('B' << 24 | 'G' << 16 | 'R' << 8 | 'A')
