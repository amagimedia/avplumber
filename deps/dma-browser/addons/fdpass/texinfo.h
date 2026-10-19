#pragma once
// The 48-byte header sent ahead of every DMA-BUF fd on a window's socket: little-endian, packed.
// deps/dma-browser encodes it (src/main/capture/TexInfoEncoder.ts) and sends it with the fd as
// SCM_RIGHTS (addon.cc); avplumber's ipc_dmabuf_source receives it (TexInfoMessage.hpp). Its size
// never changes: the socket is a stream and a receiver reads exactly this many bytes per message,
// so a longer header would be glued to the next one by an older receiver.
//
//   0  u32 width
//   4  u32 height
//   8  u32 stride         bytes per row
//   12 u32 pixel_format   FOURCC 'BGRA' (DRM ARGB8888) or 'RGBA' (DRM ABGR8888)
//   16 u64 modifier       DRM format modifier
//   24 u64 offset         plane offset in bytes
//   32 u64 timestamp      CLOCK_MONOTONIC nanoseconds of the paint
//   40 u64 frame_count    the number a release acknowledgement names (dmabuf_ack.h)
#include <cstdint>

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

constexpr uint32_t TEXINFO_PIX_FMT_RGBA = ('R' << 24 | 'G' << 16 | 'B' << 8 | 'A');
constexpr uint32_t TEXINFO_PIX_FMT_BGRA = ('B' << 24 | 'G' << 16 | 'R' << 8 | 'A');
