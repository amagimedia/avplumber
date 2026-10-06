#pragma once
// Frame bookkeeping of the draw nodes that needs no CUDA: which frames a draw node may draw on
// without copying them first, and how a frame's properties reach the frame that is drawn on.
#include "../common/yolo_side_data.hpp"

#include <cerrno>
#include <cstddef>
#include <cstdint>

extern "C" {
#include <libavutil/error.h>
#include <libavutil/frame.h>
#include <libavutil/hwcontext.h>
}

namespace cuda_overlay {

// Its address in AVHWFramesContext::user_opaque marks the pool a draw node makes the pictures it
// draws on in: the linear picture of a CUarray input and the copy of a linear input. Only draw
// nodes allocate from such a pool: the first one of a chain from its own, a later one from the
// pool of its input when that picture is still referenced elsewhere.
inline const char kPicturePoolTag = 0;

inline void markPicturePool(AVBufferRef* frames) {
    ((AVHWFramesContext*)frames->data)->user_opaque = (void*)&kPicturePoolTag;
}

/// Whether `frames` (a frame's hw_frames_ctx) is a draw node's picture pool.
inline bool isPicturePool(const AVBufferRef* frames) {
    return frames && ((const AVHWFramesContext*)frames->data)->user_opaque == &kPicturePoolTag;
}

/// Whether a draw node may draw on `frame` itself: linear CUDA memory from a draw node's picture
/// pool that no other AVFrame references. Checked on every frame. A frame that a split, a wiretap
/// or a retaining node still references, and any frame of another pool, answers no and is copied
/// before drawing.
inline bool isPrivatePicture(AVFrame* frame) {
    return frame && frame->format == AV_PIX_FMT_CUDA && isPicturePool(frame->hw_frames_ctx) &&
           av_frame_is_writable(frame) > 0;
}

/// Rows of pitch linesize[0] from the start of plane 0 to the start of plane 1, when both planes
/// lie in the frame's single buffer at that pitch and `plane1_rows` rows of plane 1 end inside it:
/// one 2D copy from plane 0 then reaches both planes and the rows between them. 0 when the planes
/// need a copy each (separate buffers, as a mapped decoder surface has, or different pitches).
inline size_t planeSpanRows(const AVFrame* frame, size_t plane1_rows) {
    const AVBufferRef* buf = frame->buf[0];
    const uintptr_t plane0 = (uintptr_t)frame->data[0], plane1 = (uintptr_t)frame->data[1];
    if (!buf || frame->buf[1] || frame->data[0] != buf->data || frame->linesize[0] <= 0 ||
        frame->linesize[1] != frame->linesize[0] || plane1 <= plane0) return 0;
    const size_t pitch = (size_t)frame->linesize[0], offset = plane1 - plane0;
    return offset % pitch == 0 && offset + plane1_rows * pitch <= (size_t)buf->size ? offset / pitch : 0;
}

/// av_frame_copy_props for a frame that is drawn on. Returns a negative AVERROR on failure.
inline int copyFrameProps(AVFrame* dst, const AVFrame* src) {
    int ret = av_frame_copy_props(dst, src);
    if (ret < 0) return ret;

    // av_frame_copy_props clones side-data payload bytes, which is unsafe for our
    // custom YOLO seg GPU side data because the payload contains a raw device
    // pointer whose lifetime is carried by the side-data buffer ref itself.
    for (int i = 0; i < src->nb_side_data; ++i) {
        const AVFrameSideData* sd_src = src->side_data[i];
        if (!sd_src || !sd_src->buf) continue;
        if (!yoloSegIsManagedSideDataType(sd_src->type)) continue;

        av_frame_remove_side_data(dst, sd_src->type);
        AVBufferRef* ref = av_buffer_ref(sd_src->buf);
        if (!ref) return AVERROR(ENOMEM);
        if (!av_frame_new_side_data_from_buf(dst, sd_src->type, ref)) {
            av_buffer_unref(&ref);
            return AVERROR(ENOMEM);
        }
    }
    return 0;
}

} // namespace cuda_overlay
