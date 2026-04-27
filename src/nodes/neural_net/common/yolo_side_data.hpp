#pragma once

extern "C" {
#include <libavutil/frame.h>
}

// YOLO segmentation side-data ids used across nodes. Kept separate from
// TensorRT-specific headers so non-TRT nodes can parse side-data safely.
static const AVFrameSideDataType AV_FRAME_DATA_YOLO_SEG_MASKS = (AVFrameSideDataType)0x59534D00;
static const AVFrameSideDataType AV_FRAME_DATA_YOLO_SEG_MASKS_GPU = (AVFrameSideDataType)0x59534D01;

