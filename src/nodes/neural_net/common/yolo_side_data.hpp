#pragma once

#include <cstdint>

extern "C" {
#include <libavutil/frame.h>
}

// YOLO segmentation side-data ids used across nodes. Kept separate from
// TensorRT-specific headers so non-TRT nodes can parse side-data safely.
static const AVFrameSideDataType AV_FRAME_DATA_YOLO_SEG_MASKS = (AVFrameSideDataType)0x59534D00;
static const AVFrameSideDataType AV_FRAME_DATA_YOLO_SEG_MASKS_GPU = (AVFrameSideDataType)0x59534D01;

// Header for GPU mask side data (lives in CPU memory, gpu_ptr is a CUdeviceptr)
struct GpuMaskSideDataHeader {
    uint64_t gpu_ptr;
    uint32_t num_masks;
    uint32_t proto_w;
    uint32_t proto_h;
    uint32_t model_w;
    uint32_t model_h;
};

