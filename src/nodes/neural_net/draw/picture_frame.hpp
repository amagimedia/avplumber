#pragma once
// Frame bookkeeping of the draw nodes that needs no CUDA: which frames a draw node may draw on
// without copying them first, and how a frame's properties reach the frame that is drawn on.
#include "../common/yolo_side_data.hpp"

#include <cerrno>

extern "C" {
#include <libavutil/error.h>
#include <libavutil/frame.h>
#include <libavutil/hwcontext.h>
}

namespace cuda_overlay {

// Its address in AVHWFramesContext::user_opaque marks the pool a draw node makes its linear
// pictures of CUarray inputs in. Only draw nodes allocate from such a pool: the first one its
// pictures, a later one the copy of a picture that is still referenced elsewhere.
inline const char kPicturePoolTag = 0;

inline void markPicturePool(AVBufferRef* frames) {
    ((AVHWFramesContext*)frames->data)->user_opaque = (void*)&kPicturePoolTag;
}

/// Whether a draw node may draw on `frame` itself: linear CUDA memory from a draw node's picture
/// pool that no other AVFrame references. Checked on every frame. A frame that a split, a wiretap
/// or a retaining node still references, and any frame of another pool, answers no and is copied
/// before drawing.
inline bool isPrivatePicture(AVFrame* frame) {
    if (!frame || frame->format != AV_PIX_FMT_CUDA || !frame->hw_frames_ctx) return false;
    const AVHWFramesContext* frames = (const AVHWFramesContext*)frame->hw_frames_ctx->data;
    return frames->user_opaque == &kPicturePoolTag && av_frame_is_writable(frame) > 0;
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
