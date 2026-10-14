#pragma once
// Whole pictures from host memory to CUDA frames with no copy between host buffers: the caller
// writes each picture into the page-locked block this class owns and the device reads it from
// there. For any source that can put its samples where it is told (a file read, a capture SDK
// with a caller's allocator, a packet reassembler). The block holds one picture, or as many as
// the caller wants to keep for showing again.
//
// The sequence is the one FFmpeg's `hwupload_cuda=pinned=1` uses (deps/ffmpeg): copies run on a
// private stream that first waits for work queued on the device's stream, and finish before the
// frame is handed out.
#include "../../cuda.hpp"
#include "../../hwaccel.hpp"
#include "../../util.hpp"
#include "raw_frame_index.hpp"

#include <avcpp/frame.h>

#include <algorithm>
#include <atomic>

extern "C" {
#include <libavutil/hwcontext.h>
#include <libavutil/hwcontext_cuda.h>
#include <libavutil/imgutils.h>
}

class CudaPinnedUploader {
    // The frames pool, bounded when asked: FFmpeg's own grows to the most frames ever out at
    // once and keeps them for its life, so one stall of the consumers sets the device memory
    // of the source for good. The pool's state lives as long as FFmpeg's pool, which frames
    // still out keep alive after this uploader has gone.
    struct Pool {
        CUcontext context;
        int limit;   // 0: no limit
        std::atomic<int> allocated{0};
    };
    Pool *pool_ = nullptr;   // owned by frames_' pool: deleted by forget()
    static void forget(void *opaque) { delete static_cast<Pool *>(opaque); }
    static AVBufferRef *allocate(void *opaque, size_t size) {
        auto *pool = static_cast<Pool *>(opaque);
        if (pool->limit > 0 && pool->allocated.load() >= pool->limit) return nullptr;
        CUdeviceptr data = 0;
        if (cuCtxPushCurrent(pool->context) != CUDA_SUCCESS) return nullptr;
        const bool ok = cuMemAlloc(&data, size) == CUDA_SUCCESS;
        CUcontext previous;
        cuCtxPopCurrent(&previous);
        if (!ok) return nullptr;
        AVBufferRef *buffer = av_buffer_create(reinterpret_cast<uint8_t *>(data), size, &CudaPinnedUploader::free, pool, 0);
        if (!buffer) {
            free(pool, reinterpret_cast<uint8_t *>(data));
            return nullptr;
        }
        ++pool->allocated;
        return buffer;
    }
    static void free(void *opaque, uint8_t *data) {
        auto *pool = static_cast<Pool *>(opaque);
        if (cuCtxPushCurrent(pool->context) == CUDA_SUCCESS) {
            cuMemFree(reinterpret_cast<CUdeviceptr>(data));
            CUcontext previous;
            cuCtxPopCurrent(&previous);
        }
    }
    std::shared_ptr<HWAccelDevice> device_;
    AVCUDADeviceContext *cuda_ = nullptr;
    RawPictureLayout layout_;
    AVBufferRef *frames_ = nullptr;
    CUstream stream_ = nullptr;
    CUevent readers_done_ = nullptr;
    void *block_ = nullptr;

    static bool failed(CUresult result, const char *call, std::string &error) {
        if (result == CUDA_SUCCESS) return false;
        if (error.empty()) error = std::string(call) + ": " + getCudaDrvErrorString(result);
        return true;
    }
    void release() {
        if (cuda_ && cuCtxPushCurrent(cuda_->cuda_ctx) == CUDA_SUCCESS) {
            if (stream_) cuStreamSynchronize(stream_);
            if (block_) cuMemFreeHost(block_);
            if (readers_done_) cuEventDestroy(readers_done_);
            if (stream_) cuStreamDestroy(stream_);
            CUcontext previous;
            cuCtxPopCurrent(&previous);
        }
        av_buffer_unref(&frames_);
    }

public:
    /// `write_combined`: the block is write-combined memory. Nothing may ever read it on the
    /// CPU then; whether filling it is cheaper depends on who writes it and is to be measured.
    /// `pictures`: how many pictures the block holds. `pool_frames`: at most that many device
    /// frames, 0 for as many as are ever out at once.
    CudaPinnedUploader(std::shared_ptr<HWAccelDevice> device, AVPixelFormat sw_format, int width, int height,
                       bool write_combined = false, size_t pictures = 1, int pool_frames = 0):
        device_(std::move(device)), layout_(sw_format, width, height) {
        if (!device_ || device_->hardwarePixelFormat() != AV_PIX_FMT_CUDA)
            throw Error("pinned upload: a CUDA hwaccel is required");
        if (!pictures) throw Error("pinned upload: the block must hold at least one picture");
        const size_t block_bytes = size_t(layout_.frame_size) * pictures;
        cuda_ = static_cast<AVCUDADeviceContext *>(
            reinterpret_cast<AVHWDeviceContext *>(device_->deviceContext()->data)->hwctx);
        frames_ = av_hwframe_ctx_alloc(device_->deviceContext());
        if (!frames_) throw Error("pinned upload: av_hwframe_ctx_alloc failed");
        auto *frames = reinterpret_cast<AVHWFramesContext *>(frames_->data);
        frames->format = AV_PIX_FMT_CUDA;
        frames->sw_format = sw_format;
        frames->width = width;
        frames->height = height;
        // The size hwcontext_cuda gives each frame of its own pool, which lays the planes out in it.
        constexpr int kCudaFrameAlignment = 256;
        const int frame_bytes = av_image_get_buffer_size(sw_format, width, height, kCudaFrameAlignment);
        pool_ = new Pool{cuda_->cuda_ctx, pool_frames};
        frames->pool = av_buffer_pool_init2(size_t(std::max(frame_bytes, 0)), pool_, &CudaPinnedUploader::allocate,
                                            &CudaPinnedUploader::forget);
        if (!frames->pool) {
            delete pool_;
            pool_ = nullptr;
        }
        std::string error;
        if (frame_bytes <= 0 || !frames->pool)
            error = "cannot lay out a frames pool";
        else if (const int r = av_hwframe_ctx_init(frames_); r < 0)
            error = "av_hwframe_ctx_init: " + av::error2string(r);
        else if (!failed(cuCtxPushCurrent(cuda_->cuda_ctx), "cuCtxPushCurrent", error)) {
            // The bundled dynamic-loader header omits CU_STREAM_NON_BLOCKING (0x1).
            if (!failed(cuStreamCreate(&stream_, 0x1), "cuStreamCreate", error) &&
                !failed(cuEventCreate(&readers_done_, CU_EVENT_DISABLE_TIMING), "cuEventCreate", error) &&
                failed(cuMemHostAlloc(&block_, block_bytes, write_combined ? CU_MEMHOSTALLOC_WRITECOMBINED : 0),
                       "cuMemHostAlloc", error))
                error += " (" + std::to_string(block_bytes) + " bytes of page-locked memory)";
            CUcontext previous;
            cuCtxPopCurrent(&previous);
        }
        if (!error.empty()) {
            release();
            throw Error("pinned upload: " + error);
        }
    }
    CudaPinnedUploader(const CudaPinnedUploader &) = delete;
    CudaPinnedUploader &operator=(const CudaPinnedUploader &) = delete;
    ~CudaPinnedUploader() { release(); }

    /// Device frames the pool has made so far.
    int framesAllocated() const { return pool_ ? pool_->allocated.load() : 0; }

    /// Where a picture goes: `layout().frame_size` bytes, planes at `layout().plane_offset`.
    /// Free to write whenever upload() is not running.
    uint8_t *block(size_t picture = 0) { return static_cast<uint8_t *>(block_) + picture * size_t(layout_.frame_size); }
    const RawPictureLayout &layout() const { return layout_; }

    /// A picture in the block as a CUDA frame from this uploader's pool, its pixels in place.
    /// Timestamps and colour are the caller's to set. False, and `out` untouched: the device
    /// had no memory for a frame just now.
    bool upload(av::VideoFrame &out, size_t picture = 0) {
        av::VideoFrame frame;
        if (const int r = av_hwframe_get_buffer(frames_, frame.raw(), 0); r == AVERROR(ENOMEM))
            return false;
        else if (r < 0)
            throw Error("pinned upload: av_hwframe_get_buffer: " + av::error2string(r));
        fill(frame, picture);
        out = std::move(frame);
        return true;
    }
    av::VideoFrame upload(size_t picture = 0) {
        av::VideoFrame out;
        if (!upload(out, picture)) throw Error("pinned upload: av_hwframe_get_buffer: " + av::error2string(AVERROR(ENOMEM)));
        return out;
    }

private:
    void fill(av::VideoFrame &out, size_t picture) {
        std::string error;
        if (failed(cuCtxPushCurrent(cuda_->cuda_ctx), "cuCtxPushCurrent", error))
            throw Error("pinned upload: " + error);
        // A reader on the device's stream may have let go of this pool frame with its read
        // still queued: the copies wait for everything queued there so far.
        bool stop = failed(cuEventRecord(readers_done_, cuda_->stream), "cuEventRecord", error) ||
                    failed(cuStreamWaitEvent(stream_, readers_done_, 0), "cuStreamWaitEvent", error);
        for (int i = 0; !stop && i < 4 && layout_.row_bytes[i]; ++i) {
            CUDA_MEMCPY2D copy{};
            copy.srcMemoryType = CU_MEMORYTYPE_HOST;
            copy.srcHost = block(picture) + layout_.plane_offset[i];
            copy.srcPitch = size_t(layout_.row_bytes[i]);
            copy.dstMemoryType = CU_MEMORYTYPE_DEVICE;
            copy.dstDevice = reinterpret_cast<CUdeviceptr>(out.raw()->data[i]);
            copy.dstPitch = size_t(out.raw()->linesize[i]);
            copy.WidthInBytes = size_t(layout_.row_bytes[i]);
            copy.Height = size_t(layout_.rows[i]);
            stop = failed(cuMemcpy2DAsync(&copy, stream_), "cuMemcpy2DAsync", error);
        }
        // Also after a failure: neither the frame nor the block may be reused while a queued
        // copy still reads or writes it.
        failed(cuStreamSynchronize(stream_), "cuStreamSynchronize", error);
        CUcontext previous;
        cuCtxPopCurrent(&previous);
        if (!error.empty()) throw Error("pinned upload: " + error);
    }
};
