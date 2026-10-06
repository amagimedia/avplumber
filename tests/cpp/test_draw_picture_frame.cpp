#include "nodes/neural_net/draw/picture_frame.hpp"
#include <cassert>
#include <cstring>
#include <initializer_list>

extern "C" {
#include <libavutil/buffer.h>
#include <libavutil/dict.h>
}

using namespace cuda_overlay;

// A frames context as the gate reads one: only user_opaque. No device is needed for that.
static AVBufferRef *pool(bool picture_pool) {
    AVBufferRef *frames = av_buffer_allocz(sizeof(AVHWFramesContext));
    assert(frames);
    if (picture_pool) markPicturePool(frames);
    return frames;
}

static AVFrame *frame(AVBufferRef *frames, AVPixelFormat format = AV_PIX_FMT_CUDA, int buffer_flags = 0) {
    static uint8_t storage[1];
    AVFrame *f = av_frame_alloc();
    assert(f);
    f->format = format;
    f->hw_frames_ctx = frames ? av_buffer_ref(frames) : nullptr;
    f->buf[0] = av_buffer_create(storage, sizeof(storage), [](void *, uint8_t *) {}, nullptr, buffer_flags);
    assert(f->buf[0]);
    return f;
}

static void private_picture() {
    AVBufferRef *pictures = pool(true), *other = pool(false);

    AVFrame *picture = frame(pictures);
    assert(isPrivatePicture(picture));

    // A second AVFrame on the same buffer (a split, a wiretap, a node that keeps frames).
    AVFrame *shared = av_frame_clone(picture);
    assert(shared);
    assert(!isPrivatePicture(picture) && !isPrivatePicture(shared));
    // It is copied, and within the pool it came from.
    assert(isPicturePool(shared->hw_frames_ctx));
    av_frame_free(&shared);
    assert(isPrivatePicture(picture));

    // The same single reference, but storage of another pool: an upstream node's frame. It is
    // copied into the copying node's own pool.
    AVFrame *foreign = frame(other);
    assert(!isPrivatePicture(foreign));
    assert(!isPicturePool(foreign->hw_frames_ctx) && !isPicturePool(nullptr));

    // Storage nobody may write, as decoder surfaces and zero-copy imports are flagged.
    AVFrame *readonly = frame(pictures, AV_PIX_FMT_CUDA, AV_BUFFER_FLAG_READONLY);
    assert(!isPrivatePicture(readonly));

    // Not linear CUDA memory, whatever the pool says.
    AVFrame *software = frame(pictures, AV_PIX_FMT_NV12);
    assert(!isPrivatePicture(software));
#if LIBAVUTIL_VERSION_INT >= AV_VERSION_INT(61, 5, 100)
    AVFrame *array = frame(pictures, AV_PIX_FMT_CUARRAY);
    assert(!isPrivatePicture(array));
    av_frame_free(&array);
#endif

    AVFrame *no_context = frame(nullptr);
    assert(!isPrivatePicture(no_context));
    assert(!isPrivatePicture(nullptr));

    for (AVFrame *f : {picture, foreign, readonly, software, no_context}) av_frame_free(&f);
    av_buffer_unref(&pictures);
    av_buffer_unref(&other);
}

static void props() {
    AVFrame *src = av_frame_alloc(), *dst = av_frame_alloc();
    assert(src && dst);
    src->pts = 9000;
    src->duration = 1500;
    src->time_base = AVRational{1, 90000};
    assert(av_dict_set(&src->metadata, "yolo_detections", "{\"detections\":[]}", 0) >= 0);

    AVFrameSideData *plain = av_frame_new_side_data(src, AV_FRAME_DATA_A53_CC, 3);
    assert(plain);
    memcpy(plain->data, "\x01\x02\x03", 3);

    // The mask header names device memory that the buffer's free callback releases.
    int released = 0;
    GpuMaskSideDataHeader *header = new GpuMaskSideDataHeader{0x1000, 2, 160, 160, 640, 640};
    AVBufferRef *masks = av_buffer_create(reinterpret_cast<uint8_t *>(header), sizeof(*header),
                                          [](void *opaque, uint8_t *data) {
                                              ++*static_cast<int *>(opaque);
                                              delete reinterpret_cast<GpuMaskSideDataHeader *>(data);
                                          }, &released, 0);
    assert(masks && av_frame_new_side_data_from_buf(src, yoloSegGpuSideDataType(1), masks));

    assert(copyFrameProps(dst, src) == 0);
    assert(dst->pts == 9000 && dst->duration == 1500 && dst->time_base.den == 90000);
    const AVDictionaryEntry *entry = av_dict_get(dst->metadata, "yolo_detections", nullptr, 0);
    assert(entry && !strcmp(entry->value, "{\"detections\":[]}"));
    const AVFrameSideData *copied = av_frame_get_side_data(dst, AV_FRAME_DATA_A53_CC);
    assert(copied && copied->size == 3 && !memcmp(copied->data, "\x01\x02\x03", 3));

    // The mask side data is the source's buffer, referenced once more, and there is one of it.
    int mask_entries = 0;
    for (int i = 0; i < dst->nb_side_data; ++i)
        mask_entries += dst->side_data[i]->type == yoloSegGpuSideDataType(1);
    const AVFrameSideData *mask = av_frame_get_side_data(dst, yoloSegGpuSideDataType(1));
    assert(mask_entries == 1 && mask && mask->buf && mask->buf->data == masks->data);
    assert(av_buffer_get_ref_count(masks) == 2);

    // The device memory lives as long as either frame.
    av_frame_free(&src);
    assert(released == 0);
    assert(reinterpret_cast<const GpuMaskSideDataHeader *>(mask->data)->gpu_ptr == 0x1000);
    av_frame_free(&dst);
    assert(released == 1);
}

int main() {
    private_picture();
    props();
    return 0;
}
