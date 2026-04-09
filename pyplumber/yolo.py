import struct


# Matches constants in src/nodes/hwaccel/cuda_infer_yolo_base.hpp
AV_FRAME_DATA_YOLO_SEG_MASKS = 0x59534D00
AV_FRAME_DATA_YOLO_SEG_MASKS_GPU = 0x59534D01


def _to_bytes(blob):
    if isinstance(blob, bytes):
        return blob
    if isinstance(blob, bytearray):
        return bytes(blob)
    return bytes(blob)


def ParseYoloSegMasks(side_data):
    """
    Parse AV_FRAME_DATA_YOLO_SEG_MASKS (CPU masks).

    Layout (little-endian):
      u32 num_masks
      u32 cpu_mask_w
      u32 cpu_mask_h
      u32 reserved
      float32 mask_data[num_masks * cpu_mask_w * cpu_mask_h]
    """
    data = _to_bytes(side_data.data)
    if len(data) < 16:
        raise ValueError(f"YOLO CPU seg-mask side data too small: {len(data)} bytes")

    num_masks, mask_w, mask_h, reserved = struct.unpack_from("<IIII", data, 0)
    pixels_per_mask = mask_w * mask_h
    total_floats = num_masks * pixels_per_mask
    expected_size = 16 + total_floats * 4
    if len(data) < expected_size:
        raise ValueError(
            f"YOLO CPU seg-mask payload size mismatch: expected {expected_size}, got {len(data)}"
        )

    values = struct.unpack_from(f"<{total_floats}f", data[:expected_size], 16)
    masks = []
    for i in range(num_masks):
        begin = i * pixels_per_mask
        end = begin + pixels_per_mask
        masks.append(values[begin:end])

    return {
        "type": side_data.type,
        "num_masks": num_masks,
        "mask_w": mask_w,
        "mask_h": mask_h,
        "reserved": reserved,
        "masks": masks,  # each item is flat float array, row-major, size mask_w*mask_h
    }


def ParseYoloSegMasksGpu(side_data):
    """
    Parse AV_FRAME_DATA_YOLO_SEG_MASKS_GPU (GPU masks pointer metadata).

    Layout (little-endian):
      u64 gpu_ptr
      u32 num_masks
      u32 proto_w
      u32 proto_h
      u32 model_w
      u32 model_h
    """
    data = _to_bytes(side_data.data)
    expected_size = 8 + 5 * 4
    if len(data) < expected_size:
        raise ValueError(
            f"YOLO GPU seg-mask payload size mismatch: expected {expected_size}, got {len(data)}"
        )

    gpu_ptr, num_masks, proto_w, proto_h, model_w, model_h = struct.unpack_from("<QIIIII", data[:expected_size], 0)
    return {
        "type": side_data.type,
        "gpu_ptr": gpu_ptr,
        "num_masks": num_masks,
        "proto_w": proto_w,
        "proto_h": proto_h,
        "model_w": model_w,
        "model_h": model_h,
    }


def ParseYoloDetections(side_data):
    if side_data.type == AV_FRAME_DATA_YOLO_SEG_MASKS:
        return ParseYoloSegMasks(side_data)
    if side_data.type == AV_FRAME_DATA_YOLO_SEG_MASKS_GPU:
        return ParseYoloSegMasksGpu(side_data)
    return None