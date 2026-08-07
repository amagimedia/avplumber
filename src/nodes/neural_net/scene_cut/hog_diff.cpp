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

#include "../../../../objs/src/nodes/neural_net/scene_cut/hog_diff.ptx.h"

namespace {

int check_cu(CUresult err, const char* func) {
    if (err == CUDA_SUCCESS) return 0;
    const char* err_name = nullptr;
    const char* err_string = nullptr;
    if (cuGetErrorName && cuGetErrorString) {
        cuGetErrorName(err, &err_name);
        cuGetErrorString(err, &err_string);
    }
    logstream << "hog_diff: " << func << " failed: "
              << (err_name ? err_name : "?") << ": "
              << (err_string ? err_string : "?");
    return -1;
}

#define HOG_DIFF_CHECK_CU(x) check_cu((x), #x)

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

int blockNormMode(const std::string& s) {
    if (s == "l1") return 1;
    if (s == "l2hys") return 2;
    return 0;  // "l2" / default
}

} // namespace

class HogDiff : public NodeSISO<av::VideoFrame, av::VideoFrame>, public ReportsFinishByFlag {
    std::string metadata_key_ = "scene_diff";
    bool strict_cuda_ = true;
    int debug_log_every_n_ = 0;
    // Number of forward diffs (scene_diff+1..+N) emitted per frame, mirroring
    // LumaDiff. 0 = only scene_diff (D_{idx-1} vs D_idx).
    int frames_lookahead_ = 0;

    // HOG descriptor parameters (all config-plumbed; tunable without code change).
    int cell_w_ = 8;
    int cell_h_ = 8;
    int block_size_cells_ = 2;
    int block_stride_cells_ = 2;
    int num_bins_ = 9;
    bool signed_ori_ = false;
    int norm_mode_ = 0;  // 0=L2, 1=L1, 2=L2-Hys
    float gamma_ = 0.0f;
    // HOG is computed on a downscaled luma plane to bound per-frame cost; the
    // original full-resolution frame is forwarded unchanged (the node is a
    // passthrough like LumaDiff, so it stays inline on the main chain).
    int analysis_w_ = 640;
    int analysis_h_ = 360;

    AVCUDADeviceContext* cuda_dev_ctx_ = nullptr;
    CUcontext cu_ctx_ = nullptr;
    CUstream stream_ = nullptr;
    bool owns_stream_ = false;
    CUmodule cu_module_ = nullptr;
    CUfunction k_downscale_ = nullptr;
    CUfunction k_cells_ = nullptr;
    CUfunction k_blocks_ = nullptr;
    CUfunction k_l1reduce_ = nullptr;
    CUfunction k_l1sum_ = nullptr;

    // Geometry derived from (analysis size, HOG params). Descriptor layout is
    // constant for a given geometry, so the ring just holds desc_len floats each.
    int cells_x_ = 0, cells_y_ = 0;
    int num_cells_ = 0;
    int blocks_x_ = 0, blocks_y_ = 0;
    int num_blocks_ = 0;
    int desc_len_ = 0;

    // Ring of previous descriptors. Index (ring_head_ - k + N) % N holds D_{t-k}.
    std::vector<CUdeviceptr> d_ring_desc_;
    std::vector<float> ring_norm_;  // host-side L1 norm of each ring descriptor
    CUdeviceptr d_scratch_desc_ = 0;  // current descriptor (kept out of the ring during diffs)
    CUdeviceptr d_small_y_ = 0;     // downscaled luma plane (analysis_w * analysis_h)
    CUdeviceptr d_cell_hists_ = 0;
    CUdeviceptr d_block_out_ = 0;   // scratch for L1 reduce / norm sums
    int ring_head_ = 0;
    int ring_filled_ = 0;
    // Signature of the currently-allocated buffers (geometry + ring capacity).
    int sig_aw_ = 0, sig_ah_ = 0;
    int sig_cell_w_ = 0, sig_cell_h_ = 0;
    int sig_block_size_ = 0, sig_block_stride_ = 0, sig_bins_ = 0;
    int sig_ncap_ = 0;
    int block_out_capacity_ = 0;
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
        // diffs[k-1] is the diff against D_{index+k} once available.
        std::vector<std::optional<DiffResult>> diffs;
    };

    std::deque<PendingFrame> pending_;

    int ringCapacity() const { return frames_lookahead_ + 1; }

    bool computeGeometry() {
        const int width = analysis_w_;
        const int height = analysis_h_;
        if (width <= 0 || height <= 0 || cell_w_ <= 0 || cell_h_ <= 0 ||
            block_size_cells_ <= 0 || block_stride_cells_ <= 0 || num_bins_ <= 0) {
            return false;
        }
        cells_x_ = width / cell_w_;
        cells_y_ = height / cell_h_;
        num_cells_ = cells_x_ * cells_y_;
        if (cells_x_ < block_size_cells_ || cells_y_ < block_size_cells_) {
            return false;  // analysis frame too small for even one block
        }
        blocks_x_ = (cells_x_ - block_size_cells_) / block_stride_cells_ + 1;
        blocks_y_ = (cells_y_ - block_size_cells_) / block_stride_cells_ + 1;
        num_blocks_ = blocks_x_ * blocks_y_;
        desc_len_ = num_blocks_ * block_size_cells_ * block_size_cells_ * num_bins_;
        return num_blocks_ > 0 && desc_len_ > 0;
    }

    void releaseBuffers() {
        for (CUdeviceptr p : d_ring_desc_) {
            if (p) HOG_DIFF_CHECK_CU(cuMemFree(p));
        }
        d_ring_desc_.clear();
        ring_norm_.clear();
        if (d_scratch_desc_) {
            HOG_DIFF_CHECK_CU(cuMemFree(d_scratch_desc_));
            d_scratch_desc_ = 0;
        }
        if (d_small_y_) {
            HOG_DIFF_CHECK_CU(cuMemFree(d_small_y_));
            d_small_y_ = 0;
        }
        if (d_cell_hists_) {
            HOG_DIFF_CHECK_CU(cuMemFree(d_cell_hists_));
            d_cell_hists_ = 0;
        }
        if (d_block_out_) {
            HOG_DIFF_CHECK_CU(cuMemFree(d_block_out_));
            d_block_out_ = 0;
        }
        sig_aw_ = sig_ah_ = 0;
        block_out_capacity_ = 0;
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
            logstream << "hog_diff: missing hw_frames_ctx";
            return false;
        }
        AVHWFramesContext* fctx = (AVHWFramesContext*)frm.raw()->hw_frames_ctx->data;
        if (!fctx || !fctx->device_ctx || !fctx->device_ctx->hwctx) {
            logstream << "hog_diff: missing device_ctx/hwctx in frame";
            return false;
        }
        cuda_dev_ctx_ = (AVCUDADeviceContext*)fctx->device_ctx->hwctx;
        if (!cuda_dev_ctx_ || !cuda_dev_ctx_->cuda_ctx) {
            logstream << "hog_diff: missing CUDA context in frame";
            return false;
        }
        cu_ctx_ = cuda_dev_ctx_->cuda_ctx;
        if (HOG_DIFF_CHECK_CU(cuCtxSetCurrent(cu_ctx_))) return false;
        stream_ = cuda_dev_ctx_->stream;
        if (!stream_) {
            if (HOG_DIFF_CHECK_CU(cuStreamCreate(&stream_, 0))) {
                stream_ = nullptr;
                return false;
            }
            owns_stream_ = true;
        }
        return true;
    }

    bool loadKernels() {
        if (k_cells_) return true;
        if (!cu_ctx_) return false;
        if (HOG_DIFF_CHECK_CU(cuCtxSetCurrent(cu_ctx_))) return false;
        const std::string ptx(avpl_hog_diff_ptx, avpl_hog_diff_ptx + avpl_hog_diff_ptx_len);
        if (HOG_DIFF_CHECK_CU(cuModuleLoadDataEx(&cu_module_, ptx.c_str(), 0, nullptr, nullptr))) {
            return false;
        }
        const char* names[5] = { "kHogDownscale", "kHogCells", "kHogBlocks", "kHogL1Reduce", "kHogL1Sum" };
        CUfunction* fns[5] = { &k_downscale_, &k_cells_, &k_blocks_, &k_l1reduce_, &k_l1sum_ };
        for (int i = 0; i < 5; ++i) {
            if (HOG_DIFF_CHECK_CU(cuModuleGetFunction(fns[i], cu_module_, names[i]))) {
                return false;
            }
        }
        return true;
    }

    bool ensureBuffers() {
        const int Ncap = ringCapacity();
        const bool geom_match = (sig_aw_ == analysis_w_ && sig_ah_ == analysis_h_ &&
                                 sig_cell_w_ == cell_w_ && sig_cell_h_ == cell_h_ &&
                                 sig_block_size_ == block_size_cells_ &&
                                 sig_block_stride_ == block_stride_cells_ &&
                                 sig_bins_ == num_bins_ && sig_ncap_ == Ncap);
        const int l1_threads = 256;
        const int l1_blocks = std::max(1, std::min(1024, (desc_len_ + l1_threads - 1) / l1_threads));
        const bool caps_ok = block_out_capacity_ >= l1_blocks &&
                             (int)d_ring_desc_.size() == Ncap &&
                             (int)ring_norm_.size() == Ncap &&
                             d_small_y_ != 0 && d_scratch_desc_ != 0;
        if (geom_match && caps_ok) return true;

        releaseBuffers();
        d_ring_desc_.assign((size_t)Ncap, 0);
        for (int i = 0; i < Ncap; ++i) {
            if (HOG_DIFF_CHECK_CU(cuMemAlloc(&d_ring_desc_[i], (size_t)desc_len_ * sizeof(float)))) {
                return false;
            }
        }
        ring_norm_.assign((size_t)Ncap, 0.0f);
        if (HOG_DIFF_CHECK_CU(cuMemAlloc(&d_scratch_desc_, (size_t)desc_len_ * sizeof(float)))) {
            return false;
        }
        if (HOG_DIFF_CHECK_CU(cuMemAlloc(&d_small_y_, (size_t)analysis_w_ * (size_t)analysis_h_))) {
            return false;
        }
        if (HOG_DIFF_CHECK_CU(cuMemAlloc(&d_cell_hists_, (size_t)num_cells_ * (size_t)num_bins_ * sizeof(float)))) {
            return false;
        }
        if (HOG_DIFF_CHECK_CU(cuMemAlloc(&d_block_out_, (size_t)l1_blocks * sizeof(float)))) {
            return false;
        }
        sig_aw_ = analysis_w_;
        sig_ah_ = analysis_h_;
        sig_cell_w_ = cell_w_;
        sig_cell_h_ = cell_h_;
        sig_block_size_ = block_size_cells_;
        sig_block_stride_ = block_stride_cells_;
        sig_bins_ = num_bins_;
        sig_ncap_ = Ncap;
        block_out_capacity_ = l1_blocks;
        ring_head_ = 0;
        ring_filled_ = 0;
        return true;
    }

    // Downscale the full-res luma plane into the compact d_small_y_ buffer.
    bool downscaleLuma(CUdeviceptr y_plane, int y_pitch, int width, int height) {
        const int block = 16;
        const int gx = (analysis_w_ + block - 1) / block;
        const int gy = (analysis_h_ + block - 1) / block;
        void* args[] = {
            (void*)&y_plane, (void*)&y_pitch, (void*)&width, (void*)&height,
            (void*)&analysis_w_, (void*)&analysis_h_, (void*)&d_small_y_,
        };
        if (HOG_DIFF_CHECK_CU(cuLaunchKernel(k_downscale_,
                                             (unsigned int)gx, (unsigned int)gy, 1,
                                             (unsigned int)block, (unsigned int)block, 1,
                                             0, stream_, args, nullptr))) {
            return false;
        }
        return true;
    }

    // Compute the HOG descriptor for the downscaled luma into d_scratch_desc_.
    // The current descriptor is kept out of the ring until all forward/primary
    // diffs are computed, so the oldest ring slot (which may be an anchor for
    // the max forward diff) is not clobbered prematurely.
    bool computeDescriptor() {
        const int threads = 256;
        const int signed_ori_int = signed_ori_ ? 1 : 0;
        // kHogCells: one block per cell, reads the compact downscaled plane
        // (pitch == analysis_w_, width == analysis_w_, height == analysis_h_).
        {
            void* args[] = {
                (void*)&d_small_y_, (void*)&analysis_w_, (void*)&analysis_w_, (void*)&analysis_h_,
                (void*)&cell_w_, (void*)&cell_h_, (void*)&cells_x_, (void*)&num_bins_,
                (void*)&signed_ori_int, (void*)&gamma_, (void*)&d_cell_hists_,
            };
            const unsigned int shared = (unsigned int)(num_bins_ * sizeof(float));
            if (HOG_DIFF_CHECK_CU(cuLaunchKernel(k_cells_,
                                                 (unsigned int)num_cells_, 1, 1,
                                                 (unsigned int)threads, 1, 1,
                                                 shared, stream_, args, nullptr))) {
                return false;
            }
        }
        // kHogBlocks: one block per HOG block, writes the descriptor vector.
        {
            const int B = block_size_cells_ * block_size_cells_ * num_bins_;
            CUdeviceptr dst = d_scratch_desc_;
            void* args[] = {
                (void*)&d_cell_hists_, (void*)&cells_x_,
                (void*)&block_size_cells_, (void*)&block_stride_cells_,
                (void*)&num_bins_, (void*)&norm_mode_, (void*)&dst,
            };
            const unsigned int shared = (unsigned int)(B * sizeof(float));
            if (HOG_DIFF_CHECK_CU(cuLaunchKernel(k_blocks_,
                                                 (unsigned int)num_blocks_, 1, 1,
                                                 (unsigned int)threads, 1, 1,
                                                 shared, stream_, args, nullptr))) {
                return false;
            }
        }
        return true;
    }

    // Reduce d_block_out_ (per-block sums) to a host scalar.
    bool reduceBlockOut(int blocks, float* out) {
        std::vector<float> host((size_t)blocks, 0.0f);
        if (HOG_DIFF_CHECK_CU(cuMemcpyDtoHAsync(host.data(), d_block_out_,
                                                 host.size() * sizeof(float), stream_)) ||
            HOG_DIFF_CHECK_CU(cuStreamSynchronize(stream_))) {
            return false;
        }
        double s = 0.0;
        for (int i = 0; i < blocks; ++i) s += (double)host[(size_t)i];
        *out = (float)s;
        return true;
    }

    // L1 norm of the descriptor in `slot` (fills ring_norm_ entry).
    bool descriptorNorm(CUdeviceptr slot, float* out) {
        const int threads = 256;
        const int blocks = std::max(1, std::min(1024, (desc_len_ + threads - 1) / threads));
        void* args[] = { (void*)&slot, (void*)&desc_len_, (void*)&d_block_out_ };
        const unsigned int shared = (unsigned int)(threads * sizeof(float));
        if (HOG_DIFF_CHECK_CU(cuLaunchKernel(k_l1sum_,
                                             (unsigned int)blocks, 1, 1,
                                             (unsigned int)threads, 1, 1,
                                             shared, stream_, args, nullptr))) {
            return false;
        }
        return reduceBlockOut(blocks, out);
    }

    // L1 distance between descriptors in `a` and `b` slots.
    bool descriptorL1(CUdeviceptr a, CUdeviceptr b, float* out) {
        const int threads = 256;
        const int blocks = std::max(1, std::min(1024, (desc_len_ + threads - 1) / threads));
        void* args[] = { (void*)&a, (void*)&b, (void*)&desc_len_, (void*)&d_block_out_ };
        const unsigned int shared = (unsigned int)(threads * sizeof(float));
        if (HOG_DIFF_CHECK_CU(cuLaunchKernel(k_l1reduce_,
                                             (unsigned int)blocks, 1, 1,
                                             (unsigned int)threads, 1, 1,
                                             shared, stream_, args, nullptr))) {
            return false;
        }
        return reduceBlockOut(blocks, out);
    }

    DiffResult makeDiff(float sum_abs, float anchor_norm) {
        DiffResult r;
        const double denom = std::max(1, desc_len_);
        r.mean_abs = (float)(sum_abs / denom);
        r.mean_norm = (float)(sum_abs / ((double)anchor_norm + 1e-6));
        r.mean_signed = 0.0f;  // not meaningful for HOG; emitted for shape parity
        return r;
    }

    void appendDiffJson(std::ostringstream& md, const PendingFrame& pf, int k,
                        const std::optional<DiffResult>& slot) {
        md.str("");
        md.clear();
        md.precision(9);
        const bool has_prev = (pf.index > 0) && slot.has_value();
        const char* status = "ok";
        if (pf.index == 0) status = "no_prev_frame";
        else if (!slot.has_value()) status = "no_forward_frame";
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

    ~HogDiff() {
        if (cu_ctx_) {
            HOG_DIFF_CHECK_CU(cuCtxSetCurrent(cu_ctx_));
        }
        releaseBuffers();
        if (owns_stream_ && stream_) {
            HOG_DIFF_CHECK_CU(cuStreamDestroy(stream_));
            stream_ = nullptr;
            owns_stream_ = false;
        }
        if (cu_module_) {
            HOG_DIFF_CHECK_CU(cuModuleUnload(cu_module_));
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
                throw Error("hog_diff: input frame is not AV_PIX_FMT_CUDA");
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
                throw Error("hog_diff: unsupported CUDA sw_format " + std::to_string((int)sw_fmt));
            }
            writeStatusStubToCurrent(frm, frm.width(), frm.height(), msg.str());
            this->sink_->put(frm);
            return;
        }

        if (!raw->data[0] || raw->linesize[0] <= 0) {
            if (strict_cuda_) {
                throw Error("hog_diff: invalid luma plane");
            }
            writeStatusStubToCurrent(frm, frm.width(), frm.height(), "invalid_luma_plane");
            this->sink_->put(frm);
            return;
        }

        if (!initCudaContextFromFrame(frm) || !loadKernels()) {
            if (strict_cuda_) {
                throw Error("hog_diff: failed to initialize CUDA");
            }
            writeStatusStubToCurrent(frm, frm.width(), frm.height(), "cuda_init_failed");
            this->sink_->put(frm);
            return;
        }

        const int width = frm.width();
        const int height = frm.height();
        if (!computeGeometry()) {
            if (strict_cuda_) {
                throw Error("hog_diff: analysis size too small for configured HOG geometry");
            }
            writeStatusStubToCurrent(frm, width, height, "invalid_geometry");
            this->sink_->put(frm);
            return;
        }
        if (!ensureBuffers()) {
            if (strict_cuda_) {
                throw Error("hog_diff: failed to allocate buffers");
            }
            writeStatusStubToCurrent(frm, width, height, "buffer_alloc_failed");
            this->sink_->put(frm);
            return;
        }

        const int L = frames_lookahead_;
        const int Ncap = ringCapacity();
        const CUdeviceptr y_plane = (CUdeviceptr)(uintptr_t)raw->data[0];
        const int y_pitch = raw->linesize[0];

        // Downscale the full-res luma to the analysis plane; the original frame
        // is forwarded unchanged so the node stays inline on the main chain.
        if (!downscaleLuma(y_plane, y_pitch, width, height)) {
            if (strict_cuda_) {
                throw Error("hog_diff: downscale failed");
            }
            writeStatusStubToCurrent(frm, width, height, "kernel_failed");
            this->sink_->put(frm);
            return;
        }

        // D_t lands in scratch (not the ring) so ring anchors stay intact.
        if (!computeDescriptor()) {
            if (strict_cuda_) {
                throw Error("hog_diff: descriptor computation failed");
            }
            writeStatusStubToCurrent(frm, width, height, "kernel_failed");
            this->sink_->put(frm);
            return;
        }

        // L1 norm of the current descriptor (for mean_norm when it acts as anchor).
        float cur_norm = 0.0f;
        if (!descriptorNorm(d_scratch_desc_, &cur_norm)) {
            if (strict_cuda_) {
                throw Error("hog_diff: descriptor norm failed");
            }
            writeStatusStubToCurrent(frm, width, height, "kernel_failed");
            this->sink_->put(frm);
            return;
        }

        // Forward diffs for already-pending frames: anchor = D_{idx-1}, current = D_t.
        // Anchor sits (k+1) steps back from ring_head_ (the next insert slot).
        for (size_t pi = 0; pi < pending_.size(); ++pi) {
            PendingFrame& pf = pending_[pi];
            const int k = (int)(this_index - pf.index);
            if (k <= 0 || k > L) continue;
            if (pf.diffs[(size_t)k].has_value()) continue;
            const int back = k + 1;
            if (ring_filled_ < back) continue;  // pre-history frame -> stub
            const int anchor = (ring_head_ - back + Ncap) % Ncap;
            float sum_abs = 0.0f;
            if (!descriptorL1(d_scratch_desc_, d_ring_desc_[anchor], &sum_abs)) {
                if (strict_cuda_) {
                    throw Error("hog_diff: forward L1 failed");
                }
                writeStatusStubToCurrent(frm, width, height, "kernel_failed");
                this->sink_->put(frm);
                return;
            }
            pf.diffs[(size_t)k] = makeDiff(sum_abs, ring_norm_[anchor]);
        }

        // Enqueue the current frame; fill its primary diff (anchor = D_{t-1}).
        PendingFrame pf;
        pf.frame = frm;
        pf.index = this_index;
        pf.width = width;
        pf.height = height;
        pf.diffs.assign((size_t)(frames_lookahead_ + 1), std::nullopt);
        if (ring_filled_ >= 1) {
            const int anchor = (ring_head_ - 1 + Ncap) % Ncap;
            float sum_abs = 0.0f;
            if (!descriptorL1(d_scratch_desc_, d_ring_desc_[anchor], &sum_abs)) {
                if (strict_cuda_) {
                    throw Error("hog_diff: primary L1 failed");
                }
                writeStatusStubToCurrent(frm, width, height, "kernel_failed");
                this->sink_->put(frm);
                return;
            }
            pf.diffs[0] = makeDiff(sum_abs, ring_norm_[anchor]);
        }
        const bool current_primary_ready = pf.diffs[0].has_value();
        pending_.push_back(std::move(pf));

        // Commit current descriptor into the ring (D_t -> ring_head_ slot).
        if (HOG_DIFF_CHECK_CU(cuMemcpyDtoDAsync(d_ring_desc_[ring_head_], d_scratch_desc_,
                                                 (size_t)desc_len_ * sizeof(float), stream_)) ||
            HOG_DIFF_CHECK_CU(cuStreamSynchronize(stream_))) {
            if (strict_cuda_) {
                throw Error("hog_diff: ring commit copy failed");
            }
            writeStatusStubToCurrent(frm, width, height, "kernel_failed");
            this->sink_->put(frm);
            return;
        }
        ring_norm_[ring_head_] = cur_norm;
        ring_head_ = (ring_head_ + 1) % Ncap;
        ring_filled_ = std::min(ring_filled_ + 1, Ncap);

        // Drain pending frames whose forward diffs are fully resolved.
        while ((int)pending_.size() > L) {
            emitPending(pending_.front());
            pending_.pop_front();
        }

        if (debug_log_every_n_ > 0 && (frame_counter_ % (uint64_t)debug_log_every_n_) == 0) {
            logstream << "hog_diff: frame=" << this_index
                      << " frames_lookahead=" << frames_lookahead_
                      << " pending=" << pending_.size()
                      << " primary_ready=" << (current_primary_ready ? 1 : 0)
                      << " desc_len=" << desc_len_
                      << " cells=" << cells_x_ << "x" << cells_y_
                      << " blocks=" << blocks_x_ << "x" << blocks_y_
                      << " size=" << width << "x" << height;
        }
    }

    static std::shared_ptr<HogDiff> create(NodeCreationInfo& nci) {
        EdgeManager& edges = nci.edges;
        const Parameters& params = nci.params;
        auto r = NodeSISO<av::VideoFrame, av::VideoFrame>::template createCommon<HogDiff>(edges, params);
        if (params.count("metadata_key")) r->metadata_key_ = params["metadata_key"].get<std::string>();
        if (params.count("strict_cuda")) r->strict_cuda_ = params["strict_cuda"].get<bool>();
        if (params.count("debug_log_every_n")) r->debug_log_every_n_ = params["debug_log_every_n"].get<int>();
        if (params.count("frames_lookahead")) {
            const int v = params["frames_lookahead"].get<int>();
            r->frames_lookahead_ = std::max(0, std::min(16, v));
        }
        if (params.count("cell_width")) r->cell_w_ = std::max(1, params["cell_width"].get<int>());
        if (params.count("cell_height")) r->cell_h_ = std::max(1, params["cell_height"].get<int>());
        if (params.count("block_size_cells")) r->block_size_cells_ = std::max(1, params["block_size_cells"].get<int>());
        if (params.count("block_stride_cells")) r->block_stride_cells_ = std::max(1, params["block_stride_cells"].get<int>());
        if (params.count("num_bins")) r->num_bins_ = std::max(1, params["num_bins"].get<int>());
        if (params.count("signed_orientations")) r->signed_ori_ = params["signed_orientations"].get<bool>();
        if (params.count("block_norm")) r->norm_mode_ = blockNormMode(params["block_norm"].get<std::string>());
        if (params.count("gamma")) r->gamma_ = std::max(0.0f, params["gamma"].get<float>());
        if (params.count("analysis_width")) r->analysis_w_ = std::max(1, params["analysis_width"].get<int>());
        if (params.count("analysis_height")) r->analysis_h_ = std::max(1, params["analysis_height"].get<int>());
        return r;
    }
};

DECLNODE(hog_diff, HogDiff)
DECLNODE_ALIAS(cuda_hog_diff, HogDiff)
