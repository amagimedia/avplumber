#include "../../node_common.hpp"
#include "../../../hwaccel.hpp"
#include <cuda_loader/cuda_drvapi_dynlink_cuda.h>

extern "C" {
#include <libavutil/dict.h>
#include <libavutil/frame.h>
#include <libavutil/hwcontext.h>
#include <libavutil/hwcontext_cuda.h>
#include <libavutil/pixfmt.h>
}

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <deque>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#include "../../../../objs/src/nodes/neural_net/scene_cut/luma_diff.ptx.h"

namespace {

int check_cu(CUresult err, const char* func) {
    if (err == CUDA_SUCCESS) return 0;
    const char* err_name = nullptr;
    const char* err_string = nullptr;
    if (cuGetErrorName && cuGetErrorString) {
        cuGetErrorName(err, &err_name);
        cuGetErrorString(err, &err_string);
    }
    logstream << "luma_diff: " << func << " failed: "
              << (err_name ? err_name : "?") << ": "
              << (err_string ? err_string : "?");
    return -1;
}

#define LUMA_DIFF_CHECK_CU(x) check_cu((x), #x)

bool isSupportedLumaCudaFormat(AVPixelFormat sw_fmt) {
    switch (sw_fmt) {
        case AV_PIX_FMT_NV12:
        case AV_PIX_FMT_YUV420P:
        case AV_PIX_FMT_YUVJ420P:
        case AV_PIX_FMT_GRAY8:
            return true;
        default:
            return false;
    }
}

std::string jsonStringEscape(const std::string& value) {
    std::ostringstream out;
    for (char c : value) {
        switch (c) {
            case '"': out << "\\\""; break;
            case '\\': out << "\\\\"; break;
            case '\b': out << "\\b"; break;
            case '\f': out << "\\f"; break;
            case '\n': out << "\\n"; break;
            case '\r': out << "\\r"; break;
            case '\t': out << "\\t"; break;
            default:
                if ((unsigned char)c < 0x20) {
                    out << "\\u00";
                    const char* hex = "0123456789abcdef";
                    out << hex[((unsigned char)c >> 4) & 0x0f]
                        << hex[(unsigned char)c & 0x0f];
                } else {
                    out << c;
                }
        }
    }
    return out.str();
}

} // namespace

class LumaDiff : public NodeSISO<av::VideoFrame, av::VideoFrame>, public ReportsFinishByFlag {
    std::string metadata_key_ = "scene_diff";
    bool strict_cuda_ = true;
    int debug_log_every_n_ = 0;
    // Number of forward diffs (scene_diff+1..+N) to emit in addition to the
    // always-emitted backward scene_diff. 0 = only scene_diff (Y_{idx-1} vs Y_idx).
    int frames_lookahead_ = 0;

    AVCUDADeviceContext* cuda_dev_ctx_ = nullptr;
    CUcontext cu_ctx_ = nullptr;
    CUstream stream_ = nullptr;
    bool owns_stream_ = false;
    CUmodule cu_module_ = nullptr;
    CUfunction kernel_ = nullptr;

    // Ring of previous Y planes. Index (ring_head_ - k + N) % N holds Y_{t-k} at frame t.
    std::vector<CUdeviceptr> d_ring_y_;
    CUdeviceptr d_scratch_y_ = 0;
    CUdeviceptr d_block_abs_ = 0;
    CUdeviceptr d_block_signed_ = 0;
    int ring_head_ = 0;
    int ring_filled_ = 0;
    int prev_w_ = 0;
    int prev_h_ = 0;
    int block_capacity_ = 0;
    uint64_t frame_counter_ = 0;

    struct DiffResult {
        float mean_abs = 0.0f;
        float mean_norm = 0.0f;
        float mean_signed = 0.0f;
    };

    struct PendingFrame {
        av::VideoFrame frame;
        uint64_t index = 0;
        int width = 0;
        int height = 0;
        // diffs[k-1] is the diff against Y_{index+k} once available.
        std::vector<std::optional<DiffResult>> diffs;
    };

    std::deque<PendingFrame> pending_;

    void releaseBuffers() {
        for (CUdeviceptr p : d_ring_y_) {
            if (p) LUMA_DIFF_CHECK_CU(cuMemFree(p));
        }
        d_ring_y_.clear();
        if (d_scratch_y_) {
            LUMA_DIFF_CHECK_CU(cuMemFree(d_scratch_y_));
            d_scratch_y_ = 0;
        }
        if (d_block_abs_) {
            LUMA_DIFF_CHECK_CU(cuMemFree(d_block_abs_));
            d_block_abs_ = 0;
        }
        if (d_block_signed_) {
            LUMA_DIFF_CHECK_CU(cuMemFree(d_block_signed_));
            d_block_signed_ = 0;
        }
        prev_w_ = 0;
        prev_h_ = 0;
        block_capacity_ = 0;
        ring_head_ = 0;
        ring_filled_ = 0;
    }

    AVPixelFormat hwSwFormat(const av::VideoFrame& frm) const {
        if (!frm.raw() || !frm.raw()->hw_frames_ctx || !frm.raw()->hw_frames_ctx->data) {
            return AV_PIX_FMT_NONE;
        }
        const AVHWFramesContext* ctx = (const AVHWFramesContext*)frm.raw()->hw_frames_ctx->data;
        return ctx ? ctx->sw_format : AV_PIX_FMT_NONE;
    }

    bool initCudaContextFromFrame(const av::VideoFrame& frm) {
        if (cu_ctx_) return true;
        if (!frm.raw() || !frm.raw()->hw_frames_ctx || !frm.raw()->hw_frames_ctx->data) {
            logstream << "luma_diff: missing hw_frames_ctx";
            return false;
        }
        AVHWFramesContext* fctx = (AVHWFramesContext*)frm.raw()->hw_frames_ctx->data;
        if (!fctx || !fctx->device_ctx || !fctx->device_ctx->hwctx) {
            logstream << "luma_diff: missing device_ctx/hwctx in frame";
            return false;
        }
        cuda_dev_ctx_ = (AVCUDADeviceContext*)fctx->device_ctx->hwctx;
        if (!cuda_dev_ctx_ || !cuda_dev_ctx_->cuda_ctx) {
            logstream << "luma_diff: missing CUDA context in frame";
            return false;
        }
        cu_ctx_ = cuda_dev_ctx_->cuda_ctx;
        if (LUMA_DIFF_CHECK_CU(cuCtxSetCurrent(cu_ctx_))) {
            return false;
        }
        stream_ = cuda_dev_ctx_->stream;
        if (!stream_) {
            if (LUMA_DIFF_CHECK_CU(cuStreamCreate(&stream_, 0))) {
                stream_ = nullptr;
                return false;
            }
            owns_stream_ = true;
        }
        return true;
    }

    bool loadKernel() {
        if (kernel_) return true;
        if (!cu_ctx_) return false;
        if (LUMA_DIFF_CHECK_CU(cuCtxSetCurrent(cu_ctx_))) return false;
        const std::string ptx(avpl_luma_diff_ptx, avpl_luma_diff_ptx + avpl_luma_diff_ptx_len);
        if (LUMA_DIFF_CHECK_CU(cuModuleLoadDataEx(&cu_module_, ptx.c_str(), 0, nullptr, nullptr))) {
            return false;
        }
        if (LUMA_DIFF_CHECK_CU(cuModuleGetFunction(&kernel_, cu_module_, "kLumaDiffReduce"))) {
            return false;
        }
        return true;
    }

    // `frames_lookahead_` is the number of forward diffs emitted per frame:
    //   scene_diff       = Y_{idx-1} vs Y_idx           (always emitted; frame 0 is a stub)
    //   scene_diff+k     = Y_{idx-1} vs Y_{idx+k}       for k = 1..frames_lookahead_
    // The ring must hold `frames_lookahead_ + 1` past Y planes so that when Y_t
    // arrives the anchor Y_{idx-1} of the oldest pending frame is still present.
    int ringCapacity() const { return frames_lookahead_ + 1; }

    bool ensureBuffers(int width, int height, int blocks) {
        if (width <= 0 || height <= 0 || blocks <= 0) return false;
        const int N = ringCapacity();
        const bool size_match = (prev_w_ == width && prev_h_ == height);
        const bool ring_ok = ((int)d_ring_y_.size() == N) && (d_scratch_y_ != 0);
        const bool caps_ok = block_capacity_ >= blocks;
        if (size_match && ring_ok && caps_ok) {
            return true;
        }
        releaseBuffers();
        const size_t pixels = (size_t)width * (size_t)height;
        d_ring_y_.assign((size_t)N, 0);
        for (int i = 0; i < N; ++i) {
            if (LUMA_DIFF_CHECK_CU(cuMemAlloc(&d_ring_y_[i], pixels))) return false;
        }
        if (LUMA_DIFF_CHECK_CU(cuMemAlloc(&d_scratch_y_, pixels))) return false;
        if (LUMA_DIFF_CHECK_CU(cuMemAlloc(&d_block_abs_, (size_t)blocks * sizeof(float)))) return false;
        if (LUMA_DIFF_CHECK_CU(cuMemAlloc(&d_block_signed_, (size_t)blocks * sizeof(float)))) return false;
        prev_w_ = width;
        prev_h_ = height;
        block_capacity_ = blocks;
        ring_head_ = 0;
        ring_filled_ = 0;
        return true;
    }

    // Launch the CUDA reduce kernel with the given prev_y buffer and read back the
    // per-block sums, returning the reduced mean_abs / mean_norm / mean_signed.
    // The kernel also writes the current Y plane into prev_y (writeback path); the
    // caller must supply a buffer whose post-launch contents are acceptable to
    // clobber with the current Y.
    bool launchDiffKernel(CUdeviceptr y_plane, int y_pitch, CUdeviceptr prev_y,
                          int width, int height, int has_prev, int blocks, int threads,
                          DiffResult* out) {
        void* args[] = {
            (void*)&y_plane,
            (void*)&y_pitch,
            (void*)&prev_y,
            (void*)&width,
            (void*)&height,
            (void*)&has_prev,
            (void*)&d_block_abs_,
            (void*)&d_block_signed_,
        };
        const unsigned int shared_bytes = (unsigned int)(threads * 2 * sizeof(float));
        if (LUMA_DIFF_CHECK_CU(cuLaunchKernel(kernel_,
                                               (unsigned int)blocks, 1, 1,
                                               (unsigned int)threads, 1, 1,
                                               shared_bytes, stream_, args, nullptr))) {
            return false;
        }
        std::vector<float> host_abs((size_t)blocks, 0.0f);
        std::vector<float> host_signed((size_t)blocks, 0.0f);
        if (LUMA_DIFF_CHECK_CU(cuMemcpyDtoHAsync(host_abs.data(), d_block_abs_,
                                                  host_abs.size() * sizeof(float), stream_)) ||
            LUMA_DIFF_CHECK_CU(cuMemcpyDtoHAsync(host_signed.data(), d_block_signed_,
                                                  host_signed.size() * sizeof(float), stream_)) ||
            LUMA_DIFF_CHECK_CU(cuStreamSynchronize(stream_))) {
            return false;
        }
        double sum_abs = 0.0;
        double sum_signed = 0.0;
        if (has_prev) {
            for (int i = 0; i < blocks; ++i) {
                sum_abs += (double)host_abs[(size_t)i];
                sum_signed += (double)host_signed[(size_t)i];
            }
        }
        const double denom = std::max(1, width * height);
        out->mean_abs = has_prev ? (float)(sum_abs / denom) : 0.0f;
        out->mean_signed = has_prev ? (float)(sum_signed / denom) : 0.0f;
        out->mean_norm = has_prev ? (float)std::fabs(sum_signed / (255.0 * denom)) : 0.0f;
        return true;
    }

    void appendDiffJson(std::ostringstream& md, const PendingFrame& pf, int k,
                        const std::optional<DiffResult>& slot) {
        md.str("");
        md.clear();
        md.precision(9);
        // k=0 is the primary diff Y_{idx-1} vs Y_idx. For k>=1 the slot may be
        // absent because we ran off the end of the stream before the future frame
        // Y_{idx+k} arrived; downstream treats missing slots as "no_forward_frame".
        const bool has_prev = (pf.index > 0) && slot.has_value();
        const char* status = "ok";
        if (pf.index == 0) {
            status = "no_prev_frame";
        } else if (!slot.has_value()) {
            status = "no_forward_frame";
        }
        md << "{"
           << "\"frame_index\":" << pf.index << ","
           << "\"lookahead\":" << k << ","
           << "\"has_prev\":" << (has_prev ? "true" : "false") << ","
           << "\"height\":" << pf.height << ","
           << "\"mean_abs\":" << (has_prev ? slot->mean_abs : 0.0f) << ","
           << "\"mean_norm\":" << (has_prev ? slot->mean_norm : 0.0f) << ","
           << "\"mean_signed\":" << (has_prev ? slot->mean_signed : 0.0f) << ","
           << "\"status\":\"" << status << "\","
           << "\"width\":" << pf.width
           << "}";
    }

    void emitPending(PendingFrame& pf) {
        std::ostringstream md;
        // Emit `frames_lookahead_ + 1` keys total: scene_diff (Y_{idx-1} vs Y_idx)
        // plus scene_diff+k for k=1..L (Y_{idx-1} vs Y_{idx+k}), where L = frames_lookahead_.
        const int L = frames_lookahead_;
        for (int k = 0; k <= L; ++k) {
            const std::string key = (k == 0) ? metadata_key_
                                             : (metadata_key_ + "+" + std::to_string(k));
            appendDiffJson(md, pf, k, pf.diffs[(size_t)k]);
            av_dict_set(&pf.frame.raw()->metadata, key.c_str(), md.str().c_str(), 0);
        }
        this->sink_->put(pf.frame);
        pf.frame = av::VideoFrame();
    }

    void writeStatusStubToCurrent(av::VideoFrame& frm, int width, int height,
                                  const std::string& status) {
        // Fallback when we cannot buffer this frame (fatal init / bad format in
        // non-strict mode). Emits stub metadata on every configured key so
        // downstream consumers behave as if no diff was available.
        std::ostringstream md;
        const int L = frames_lookahead_;
        for (int k = 0; k <= L; ++k) {
            const std::string key = (k == 0) ? metadata_key_
                                             : (metadata_key_ + "+" + std::to_string(k));
            md.str("");
            md.clear();
            md << "{"
               << "\"frame_index\":" << (frame_counter_ > 0 ? frame_counter_ - 1 : 0) << ","
               << "\"lookahead\":" << k << ","
               << "\"has_prev\":false,"
               << "\"height\":" << height << ","
               << "\"mean_abs\":0,"
               << "\"mean_norm\":0,"
               << "\"mean_signed\":0,"
               << "\"status\":\"" << jsonStringEscape(status) << "\","
               << "\"width\":" << width
               << "}";
            av_dict_set(&frm.raw()->metadata, key.c_str(), md.str().c_str(), 0);
        }
    }

    void drainPendingOnEof() {
        while (!pending_.empty()) {
            emitPending(pending_.front());
            pending_.pop_front();
        }
    }

public:
    using NodeSISO<av::VideoFrame, av::VideoFrame>::NodeSISO;

    ~LumaDiff() {
        if (cu_ctx_) {
            LUMA_DIFF_CHECK_CU(cuCtxSetCurrent(cu_ctx_));
        }
        releaseBuffers();
        if (owns_stream_ && stream_) {
            LUMA_DIFF_CHECK_CU(cuStreamDestroy(stream_));
            stream_ = nullptr;
            owns_stream_ = false;
        }
        if (cu_module_) {
            LUMA_DIFF_CHECK_CU(cuModuleUnload(cu_module_));
            cu_module_ = nullptr;
        }
    }

    bool consumeEofIfPresent() override {
        return false;
    }

    void process() override {
        av::VideoFrame frm = this->source_->get();
        if (isEofMarker(frm)) {
            drainPendingOnEof();
            this->sink_->put(frm);
            this->markFinished();
            return;
        }
        if (!frm) return;

        ++frame_counter_;
        const uint64_t this_index = frame_counter_ - 1;
        AVFrame* raw = frm.raw();
        if (!raw || raw->format != AV_PIX_FMT_CUDA) {
            if (strict_cuda_) {
                throw Error("luma_diff: input frame is not AV_PIX_FMT_CUDA");
            }
            writeStatusStubToCurrent(frm, frm.width(), frm.height(), "skipped_non_cuda");
            this->sink_->put(frm);
            return;
        }

        const AVPixelFormat sw_fmt = hwSwFormat(frm);
        if (!isSupportedLumaCudaFormat(sw_fmt)) {
            std::ostringstream msg;
            msg << "unsupported_sw_format_" << (int)sw_fmt;
            if (strict_cuda_) {
                throw Error("luma_diff: unsupported CUDA sw_format " + std::to_string((int)sw_fmt));
            }
            writeStatusStubToCurrent(frm, frm.width(), frm.height(), msg.str());
            this->sink_->put(frm);
            return;
        }

        if (!raw->data[0] || raw->linesize[0] <= 0) {
            if (strict_cuda_) {
                throw Error("luma_diff: invalid luma plane");
            }
            writeStatusStubToCurrent(frm, frm.width(), frm.height(), "invalid_luma_plane");
            this->sink_->put(frm);
            return;
        }

        if (!initCudaContextFromFrame(frm) || !loadKernel()) {
            if (strict_cuda_) {
                throw Error("luma_diff: failed to initialize CUDA");
            }
            writeStatusStubToCurrent(frm, frm.width(), frm.height(), "cuda_init_failed");
            this->sink_->put(frm);
            return;
        }

        const int width = frm.width();
        const int height = frm.height();
        const int pixels = width * height;
        const int threads = 256;
        const int blocks = std::max(1, std::min(1024, (pixels + threads - 1) / threads));
        if (!ensureBuffers(width, height, blocks)) {
            if (strict_cuda_) {
                throw Error("luma_diff: failed to allocate buffers");
            }
            writeStatusStubToCurrent(frm, width, height, "buffer_alloc_failed");
            this->sink_->put(frm);
            return;
        }

        const int L = frames_lookahead_;       // number of forward diffs per frame
        const int Ncap = ringCapacity();       // == frames_lookahead_ + 1
        const CUdeviceptr y_plane = (CUdeviceptr)(uintptr_t)raw->data[0];
        const int y_pitch = raw->linesize[0];

        // For each pending frame pf (index = idx), the incoming Y_t plays the role
        // of Y_{idx + k} where k = this_index - pf.index. The anchor Y_{idx-1} sits
        // in the ring at (k + 1) steps back from the newest ring slot. All diffs
        // share Y_t as the "current" plane; we vary the anchor.
        //
        // Emit-time note: we launch diffs into slots k=0..L (L+1 total) but only
        // when the corresponding ring slot exists (ring_filled_ >= k+1). The
        // primary k=0 case (Y_{idx-1} vs Y_idx) can be launched immediately when
        // a pending frame is enqueued below; here we only handle the forward
        // diffs (k>=1) for already-pending frames.
        for (size_t pi = 0; pi < pending_.size(); ++pi) {
            PendingFrame& pf = pending_[pi];
            const int k = (int)(this_index - pf.index);
            if (k <= 0 || k > L) continue;
            if (pf.diffs[(size_t)k].has_value()) continue;
            // Anchor Y_{pf.index - 1} is (k+1) steps back from Y_t. The ring holds
            // the most recently seen frames with newest at (ring_head_ - 1). Y_t
            // has not been inserted yet, so "1 step back" points to Y_{t-1}, which
            // for pf sitting k steps ago is Y_{pf.index - 1} iff k+1 <= ring_filled_.
            const int back = k + 1;
            if (ring_filled_ < back) continue;  // pre-history frame — leave stub
            const int slot = (ring_head_ - back + Ncap) % Ncap;
            // Always copy the anchor into scratch — the kernel's writeback path
            // clobbers its prev_arg with Y_t, and we need the ring slot preserved
            // for other pending frames still to be diffed this frame.
            if (LUMA_DIFF_CHECK_CU(cuMemcpyDtoDAsync(d_scratch_y_, d_ring_y_[slot],
                                                       (size_t)width * (size_t)height,
                                                       stream_))) {
                if (strict_cuda_) {
                    throw Error("luma_diff: dtod copy failed");
                }
                writeStatusStubToCurrent(frm, width, height, "dtod_copy_failed");
                this->sink_->put(frm);
                return;
            }
            DiffResult r;
            if (!launchDiffKernel(y_plane, y_pitch, d_scratch_y_, width, height,
                                   /*has_prev=*/1, blocks, threads, &r)) {
                if (strict_cuda_) {
                    throw Error("luma_diff: kernel launch/read failed");
                }
                writeStatusStubToCurrent(frm, width, height, "kernel_failed");
                this->sink_->put(frm);
                return;
            }
            pf.diffs[(size_t)k] = r;
        }

        // Now enqueue the current frame as pending. Its primary diff (k=0,
        // anchor = Y_{this_index - 1}) can be filled in immediately if ring_filled_
        // >= 1. Frame 0 has no predecessor → primary slot stays empty and emit
        // will report status=no_prev_frame with has_prev=false.
        PendingFrame pf;
        pf.frame = frm;
        pf.index = this_index;
        pf.width = width;
        pf.height = height;
        // diffs[0] is the primary backward diff; diffs[1..L] are the L forward diffs.
        pf.diffs.assign((size_t)(frames_lookahead_ + 1), std::nullopt);
        if (ring_filled_ >= 1) {
            // Anchor Y_{this_index - 1} is at (ring_head_ - 1) mod Ncap. Copy into
            // scratch so the ring slot survives the kernel's writeback of Y_t.
            const int slot = (ring_head_ - 1 + Ncap) % Ncap;
            if (LUMA_DIFF_CHECK_CU(cuMemcpyDtoDAsync(d_scratch_y_, d_ring_y_[slot],
                                                       (size_t)width * (size_t)height,
                                                       stream_))) {
                if (strict_cuda_) {
                    throw Error("luma_diff: dtod copy (primary) failed");
                }
                writeStatusStubToCurrent(frm, width, height, "dtod_copy_failed");
                this->sink_->put(frm);
                return;
            }
            DiffResult r;
            if (!launchDiffKernel(y_plane, y_pitch, d_scratch_y_, width, height,
                                   /*has_prev=*/1, blocks, threads, &r)) {
                if (strict_cuda_) {
                    throw Error("luma_diff: kernel launch/read (primary) failed");
                }
                writeStatusStubToCurrent(frm, width, height, "kernel_failed");
                this->sink_->put(frm);
                return;
            }
            pf.diffs[0] = r;
        }
        const bool current_primary_ready = pf.diffs[0].has_value();
        pending_.push_back(std::move(pf));

        // Store Y_t into the ring for future anchor lookups. Copy the source (with
        // pitch >= width) into d_ring_y_[ring_head_] as a compact plane.
        {
            CUDA_MEMCPY2D copy = {};
            copy.srcMemoryType = CU_MEMORYTYPE_DEVICE;
            copy.srcDevice = y_plane;
            copy.srcPitch = (size_t)y_pitch;
            copy.dstMemoryType = CU_MEMORYTYPE_DEVICE;
            copy.dstDevice = d_ring_y_[ring_head_];
            copy.dstPitch = (size_t)width;
            copy.WidthInBytes = (size_t)width;
            copy.Height = (size_t)height;
            if (LUMA_DIFF_CHECK_CU(cuMemcpy2DAsync(&copy, stream_)) ||
                LUMA_DIFF_CHECK_CU(cuStreamSynchronize(stream_))) {
                if (strict_cuda_) {
                    throw Error("luma_diff: ring insert 2D memcpy failed");
                }
                writeStatusStubToCurrent(frm, width, height, "cold_start_copy_failed");
                this->sink_->put(frm);
                return;
            }
            ring_head_ = (ring_head_ + 1) % Ncap;
            ring_filled_ = std::min(ring_filled_ + 1, Ncap);
        }

        // Drain pending frames whose forward diffs are fully resolved. A frame at
        // index idx is complete when we have seen Y_{idx + L}, i.e. when
        // (this_index - idx) >= L. Equivalently drain the oldest whenever
        // pending_.size() > L.
        while ((int)pending_.size() > L) {
            emitPending(pending_.front());
            pending_.pop_front();
        }

        if (debug_log_every_n_ > 0 && (frame_counter_ % (uint64_t)debug_log_every_n_) == 0) {
            logstream << "luma_diff: frame=" << this_index
                      << " frames_lookahead=" << frames_lookahead_
                      << " pending=" << pending_.size()
                      << " primary_ready=" << (current_primary_ready ? 1 : 0)
                      << " size=" << width << "x" << height;
        }
    }

    static std::shared_ptr<LumaDiff> create(NodeCreationInfo& nci) {
        EdgeManager& edges = nci.edges;
        const Parameters& params = nci.params;
        auto r = NodeSISO<av::VideoFrame, av::VideoFrame>::template createCommon<LumaDiff>(edges, params);
        if (params.count("metadata_key")) r->metadata_key_ = params["metadata_key"].get<std::string>();
        if (params.count("strict_cuda")) r->strict_cuda_ = params["strict_cuda"].get<bool>();
        if (params.count("debug_log_every_n")) r->debug_log_every_n_ = params["debug_log_every_n"].get<int>();
        if (params.count("frames_lookahead")) {
            const int v = params["frames_lookahead"].get<int>();
            r->frames_lookahead_ = std::max(0, std::min(16, v));
        }
        return r;
    }
};

DECLNODE(luma_diff, LumaDiff)
DECLNODE_ALIAS(cuda_scene_diff, LumaDiff)
