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
#include <sstream>
#include <string>
#include <vector>

#include "../../../../objs/src/nodes/neural_net/sport_specific/scene_diff_cuda.ptx.h"

namespace {

int check_cu(CUresult err, const char* func) {
    if (err == CUDA_SUCCESS) return 0;
    const char* err_name = nullptr;
    const char* err_string = nullptr;
    if (cuGetErrorName && cuGetErrorString) {
        cuGetErrorName(err, &err_name);
        cuGetErrorString(err, &err_string);
    }
    logstream << "cuda_scene_diff: " << func << " failed: "
              << (err_name ? err_name : "?") << ": "
              << (err_string ? err_string : "?");
    return -1;
}

#define SCENE_DIFF_CHECK_CU(x) check_cu((x), #x)

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

class CudaSceneDiff : public NodeSISO<av::VideoFrame, av::VideoFrame>, public ReportsFinishByFlag {
    std::string metadata_key_ = "scene_diff";
    bool strict_cuda_ = true;
    int debug_log_every_n_ = 0;

    AVCUDADeviceContext* cuda_dev_ctx_ = nullptr;
    CUcontext cu_ctx_ = nullptr;
    CUstream stream_ = nullptr;
    bool owns_stream_ = false;
    CUmodule cu_module_ = nullptr;
    CUfunction kernel_ = nullptr;

    CUdeviceptr d_prev_y_ = 0;
    CUdeviceptr d_block_abs_ = 0;
    CUdeviceptr d_block_signed_ = 0;
    int prev_w_ = 0;
    int prev_h_ = 0;
    int block_capacity_ = 0;
    bool have_prev_ = false;
    uint64_t frame_counter_ = 0;

    void releaseBuffers() {
        if (d_prev_y_) {
            SCENE_DIFF_CHECK_CU(cuMemFree(d_prev_y_));
            d_prev_y_ = 0;
        }
        if (d_block_abs_) {
            SCENE_DIFF_CHECK_CU(cuMemFree(d_block_abs_));
            d_block_abs_ = 0;
        }
        if (d_block_signed_) {
            SCENE_DIFF_CHECK_CU(cuMemFree(d_block_signed_));
            d_block_signed_ = 0;
        }
        prev_w_ = 0;
        prev_h_ = 0;
        block_capacity_ = 0;
        have_prev_ = false;
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
            logstream << "cuda_scene_diff: missing hw_frames_ctx";
            return false;
        }
        AVHWFramesContext* fctx = (AVHWFramesContext*)frm.raw()->hw_frames_ctx->data;
        if (!fctx || !fctx->device_ctx || !fctx->device_ctx->hwctx) {
            logstream << "cuda_scene_diff: missing device_ctx/hwctx in frame";
            return false;
        }
        cuda_dev_ctx_ = (AVCUDADeviceContext*)fctx->device_ctx->hwctx;
        if (!cuda_dev_ctx_ || !cuda_dev_ctx_->cuda_ctx) {
            logstream << "cuda_scene_diff: missing CUDA context in frame";
            return false;
        }
        cu_ctx_ = cuda_dev_ctx_->cuda_ctx;
        if (SCENE_DIFF_CHECK_CU(cuCtxSetCurrent(cu_ctx_))) {
            return false;
        }
        stream_ = cuda_dev_ctx_->stream;
        if (!stream_) {
            if (SCENE_DIFF_CHECK_CU(cuStreamCreate(&stream_, 0))) {
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
        if (SCENE_DIFF_CHECK_CU(cuCtxSetCurrent(cu_ctx_))) return false;
        const std::string ptx(avpl_scene_diff_ptx, avpl_scene_diff_ptx + avpl_scene_diff_ptx_len);
        if (SCENE_DIFF_CHECK_CU(cuModuleLoadDataEx(&cu_module_, ptx.c_str(), 0, nullptr, nullptr))) {
            return false;
        }
        if (SCENE_DIFF_CHECK_CU(cuModuleGetFunction(&kernel_, cu_module_, "kSceneLumaDiffReduce"))) {
            return false;
        }
        return true;
    }

    bool ensureBuffers(int width, int height, int blocks) {
        if (width <= 0 || height <= 0 || blocks <= 0) return false;
        if (d_prev_y_ && prev_w_ == width && prev_h_ == height && block_capacity_ >= blocks) {
            return true;
        }
        releaseBuffers();
        const size_t pixels = (size_t)width * (size_t)height;
        if (SCENE_DIFF_CHECK_CU(cuMemAlloc(&d_prev_y_, pixels))) return false;
        if (SCENE_DIFF_CHECK_CU(cuMemAlloc(&d_block_abs_, (size_t)blocks * sizeof(float)))) return false;
        if (SCENE_DIFF_CHECK_CU(cuMemAlloc(&d_block_signed_, (size_t)blocks * sizeof(float)))) return false;
        prev_w_ = width;
        prev_h_ = height;
        block_capacity_ = blocks;
        have_prev_ = false;
        return true;
    }

    bool writeMetadata(av::VideoFrame& frm, bool has_prev, int width, int height, float mean_abs, float mean_norm,
                       float mean_signed, const std::string& status) {
        std::ostringstream md;
        md.precision(9);
        md << "{"
           << "\"frame_index\":" << (frame_counter_ > 0 ? frame_counter_ - 1 : 0) << ","
           << "\"has_prev\":" << (has_prev ? "true" : "false") << ","
           << "\"height\":" << height << ","
           << "\"mean_abs\":" << mean_abs << ","
           << "\"mean_norm\":" << mean_norm << ","
           << "\"mean_signed\":" << mean_signed << ","
           << "\"status\":\"" << jsonStringEscape(status) << "\","
           << "\"width\":" << width
           << "}";
        av_dict_set(&frm.raw()->metadata, metadata_key_.c_str(), md.str().c_str(), 0);
        return true;
    }

public:
    using NodeSISO<av::VideoFrame, av::VideoFrame>::NodeSISO;

    ~CudaSceneDiff() {
        if (cu_ctx_) {
            SCENE_DIFF_CHECK_CU(cuCtxSetCurrent(cu_ctx_));
        }
        releaseBuffers();
        if (owns_stream_ && stream_) {
            SCENE_DIFF_CHECK_CU(cuStreamDestroy(stream_));
            stream_ = nullptr;
            owns_stream_ = false;
        }
        if (cu_module_) {
            SCENE_DIFF_CHECK_CU(cuModuleUnload(cu_module_));
            cu_module_ = nullptr;
        }
    }

    bool consumeEofIfPresent() override {
        return false;
    }

    void process() override {
        av::VideoFrame frm = this->source_->get();
        if (isEofMarker(frm)) {
            this->sink_->put(frm);
            this->markFinished();
            return;
        }
        if (!frm) return;

        ++frame_counter_;
        AVFrame* raw = frm.raw();
        if (!raw || raw->format != AV_PIX_FMT_CUDA) {
            if (strict_cuda_) {
                throw Error("cuda_scene_diff: input frame is not AV_PIX_FMT_CUDA");
            }
            writeMetadata(frm, false, frm.width(), frm.height(), 0.0f, 0.0f, 0.0f, "skipped_non_cuda");
            this->sink_->put(frm);
            return;
        }

        const AVPixelFormat sw_fmt = hwSwFormat(frm);
        if (!isSupportedLumaCudaFormat(sw_fmt)) {
            std::ostringstream msg;
            msg << "unsupported_sw_format_" << (int)sw_fmt;
            if (strict_cuda_) {
                throw Error("cuda_scene_diff: unsupported CUDA sw_format " + std::to_string((int)sw_fmt));
            }
            writeMetadata(frm, false, frm.width(), frm.height(), 0.0f, 0.0f, 0.0f, msg.str());
            this->sink_->put(frm);
            return;
        }

        if (!raw->data[0] || raw->linesize[0] <= 0) {
            if (strict_cuda_) {
                throw Error("cuda_scene_diff: invalid luma plane");
            }
            writeMetadata(frm, false, frm.width(), frm.height(), 0.0f, 0.0f, 0.0f, "invalid_luma_plane");
            this->sink_->put(frm);
            return;
        }

        if (!initCudaContextFromFrame(frm) || !loadKernel()) {
            if (strict_cuda_) {
                throw Error("cuda_scene_diff: failed to initialize CUDA");
            }
            writeMetadata(frm, false, frm.width(), frm.height(), 0.0f, 0.0f, 0.0f, "cuda_init_failed");
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
                throw Error("cuda_scene_diff: failed to allocate buffers");
            }
            writeMetadata(frm, false, width, height, 0.0f, 0.0f, 0.0f, "buffer_alloc_failed");
            this->sink_->put(frm);
            return;
        }

        const CUdeviceptr y_plane = (CUdeviceptr)(uintptr_t)raw->data[0];
        const int y_pitch = raw->linesize[0];
        const int has_prev = have_prev_ ? 1 : 0;
        void* args[] = {
            (void*)&y_plane,
            (void*)&y_pitch,
            (void*)&d_prev_y_,
            (void*)&width,
            (void*)&height,
            (void*)&has_prev,
            (void*)&d_block_abs_,
            (void*)&d_block_signed_,
        };
        const unsigned int shared_bytes = (unsigned int)(threads * 2 * sizeof(float));
        if (SCENE_DIFF_CHECK_CU(cuLaunchKernel(kernel_,
                                               (unsigned int)blocks, 1, 1,
                                               (unsigned int)threads, 1, 1,
                                               shared_bytes, stream_, args, nullptr))) {
            throw Error("cuda_scene_diff: kernel launch failed");
        }

        std::vector<float> host_abs((size_t)blocks, 0.0f);
        std::vector<float> host_signed((size_t)blocks, 0.0f);
        if (SCENE_DIFF_CHECK_CU(cuMemcpyDtoHAsync(host_abs.data(), d_block_abs_, host_abs.size() * sizeof(float),
                                                  stream_)) ||
            SCENE_DIFF_CHECK_CU(cuMemcpyDtoHAsync(host_signed.data(), d_block_signed_,
                                                  host_signed.size() * sizeof(float),
                                                  stream_)) ||
            SCENE_DIFF_CHECK_CU(cuStreamSynchronize(stream_))) {
            throw Error("cuda_scene_diff: failed to read metrics");
        }

        double sum_abs = 0.0;
        double sum_signed = 0.0;
        if (has_prev) {
            for (int i = 0; i < blocks; ++i) {
                sum_abs += (double)host_abs[(size_t)i];
                sum_signed += (double)host_signed[(size_t)i];
            }
        }
        have_prev_ = true;

        const double denom = std::max(1, pixels);
        const float mean_abs = has_prev ? (float)(sum_abs / denom) : 0.0f;
        const float mean_signed = has_prev ? (float)(sum_signed / denom) : 0.0f;
        const float mean_norm = has_prev ? (float)std::fabs(sum_signed / (255.0 * denom)) : 0.0f;
        writeMetadata(frm, has_prev != 0, width, height, mean_abs, mean_norm, mean_signed, "ok");

        if (debug_log_every_n_ > 0 && (frame_counter_ % (uint64_t)debug_log_every_n_) == 0) {
            logstream << "cuda_scene_diff: frame=" << (frame_counter_ - 1)
                      << " has_prev=" << has_prev
                      << " mean_abs=" << mean_abs
                      << " mean_norm=" << mean_norm
                      << " size=" << width << "x" << height;
        }

        this->sink_->put(frm);
    }

    static std::shared_ptr<CudaSceneDiff> create(NodeCreationInfo& nci) {
        EdgeManager& edges = nci.edges;
        const Parameters& params = nci.params;
        auto r = NodeSISO<av::VideoFrame, av::VideoFrame>::template createCommon<CudaSceneDiff>(edges, params);
        if (params.count("metadata_key")) r->metadata_key_ = params["metadata_key"].get<std::string>();
        if (params.count("strict_cuda")) r->strict_cuda_ = params["strict_cuda"].get<bool>();
        if (params.count("debug_log_every_n")) r->debug_log_every_n_ = params["debug_log_every_n"].get<int>();
        return r;
    }
};

DECLNODE(cuda_scene_diff, CudaSceneDiff)
