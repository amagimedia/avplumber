#pragma once

#include <cstdint>

extern "C" {
#include <libavutil/buffer.h>
#include <libavutil/frame.h>
}

// YOLO segmentation side-data ids used across nodes. Kept separate from
// TensorRT-specific headers so non-TRT nodes can parse side-data safely.
// Slot types are derived from a base id plus a slot index so nodes can use a
// shared mapping instead of special-casing slot 0/1.
static const AVFrameSideDataType AV_FRAME_DATA_YOLO_SEG_MASKS = (AVFrameSideDataType)0x59534D00;
static const AVFrameSideDataType AV_FRAME_DATA_YOLO_SEG_MASKS_GPU = (AVFrameSideDataType)0x59534D01;
static const AVFrameSideDataType AV_FRAME_DATA_YOLO_SEG_MASKS_SLOT1 = (AVFrameSideDataType)0x59534D02;
static const AVFrameSideDataType AV_FRAME_DATA_YOLO_SEG_MASKS_GPU_SLOT1 = (AVFrameSideDataType)0x59534D03;

static constexpr int kMaxYoloSegSlots = 16;

inline bool yoloSegIsValidSlot(int slot) {
    return slot >= 0 && slot < kMaxYoloSegSlots;
}

inline AVFrameSideDataType yoloSegCpuSideDataType(int slot) {
    return (AVFrameSideDataType)((int)AV_FRAME_DATA_YOLO_SEG_MASKS + (slot * 2));
}

inline AVFrameSideDataType yoloSegGpuSideDataType(int slot) {
    return (AVFrameSideDataType)((int)AV_FRAME_DATA_YOLO_SEG_MASKS_GPU + (slot * 2));
}

inline bool yoloSegIsManagedSideDataType(AVFrameSideDataType type) {
    const int value = (int)type;
    const int first = (int)AV_FRAME_DATA_YOLO_SEG_MASKS;
    const int last_exclusive = first + (kMaxYoloSegSlots * 2);
    return value >= first && value < last_exclusive;
}

// Header for GPU mask side data (lives in CPU memory, gpu_ptr is a CUdeviceptr)
struct GpuMaskSideDataHeader {
    uint64_t gpu_ptr;
    uint32_t num_masks;
    uint32_t proto_w;
    uint32_t proto_h;
    uint32_t model_w;
    uint32_t model_h;
};

