#include "../../node_common.hpp"

// CudaCameraMotion — per-frame camera motion via the NVIDIA Optical Flow
// (NVOF) dense engine.
//
// Modeled on luma_diff.cpp: grabs the CUDA context/stream from the incoming
// AV_PIX_FMT_CUDA frame, keeps the previous frame's luma in an NVOF input
// buffer, runs dense optical flow between (prev -> cur), reduces the flow grid
// to background motion statistics, optionally fits a full 2x3 affine with
// OpenCV, and writes {tx, ty, affine_2x3, nvof_cost, has_prev, ...} into frame
// metadata.
//
// When HAVE_OPENCV=0, affine_2x3 falls back to a translation-only matrix built
// from the masked median tx so existing deployments keep the old behavior.
//
// The NVOF dense engine ships with the driver (libnvidia-opticalflow.so);
// headers are vendored in deps/Optical_Flow_SDK_5.0.7. Built only when
// HAVE_NVOF=1 (see Makefile gate).

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
#include <array>
#include <cstdint>
#include <cmath>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include "nvOpticalFlowCommon.h"
#include "nvOpticalFlowCuda.h"

#if HAVE_CCM_GPU_IRLS
#include "../../../../objs/src/nodes/neural_net/scene_cut/cuda_camera_motion.ptx.h"
#endif

#if HAVE_OPENCV
#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#endif

namespace {

int ccm_check_cu(CUresult err, const char* func) {
    if (err == CUDA_SUCCESS) return 0;
    const char* err_name = nullptr;
    const char* err_string = nullptr;
    if (cuGetErrorName && cuGetErrorString) {
        cuGetErrorName(err, &err_name);
        cuGetErrorString(err, &err_string);
    }
    logstream << "cuda_camera_motion: " << func << " failed: "
              << (err_name ? err_name : "?") << ": "
              << (err_string ? err_string : "?");
    return -1;
}

#define CCM_CHECK_CU(x) ccm_check_cu((x), #x)

int ccm_check_of(NV_OF_STATUS s, const char* func) {
    if (s == NV_OF_SUCCESS) return 0;
    logstream << "cuda_camera_motion: " << func << " failed: NV_OF_STATUS=" << (int)s;
    return -1;
}

#define CCM_CHECK_OF(x) ccm_check_of((x), #x)

bool isSupportedCudaFormat(AVPixelFormat sw_fmt) {
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

} // namespace

class CudaCameraMotion : public NodeSISO<av::VideoFrame, av::VideoFrame>, public ReportsFinishByFlag {
    struct FlowCell {
        float x = 0.0f;
        float y = 0.0f;
        float dx = 0.0f;
        float dy = 0.0f;
        float mag = 0.0f;
    };

    struct MotionSummary {
        std::vector<FlowCell> cells;
        float tx_median = 0.0f;
        float tx_mad = 0.0f;
        float flow_mag_min = 0.0f;
        float flow_mag_mean = 0.0f;
        float bg_cell_frac = 0.0f;
        int bg_cell_count = 0;
        int total_cell_count = 0;
        bool used_mask_fallback = false;
    };

    struct AffineSummary {
        double m00 = 1.0;
        double m01 = 0.0;
        double m02 = 0.0;
        double m10 = 0.0;
        double m11 = 1.0;
        double m12 = 0.0;
        bool valid = false;
        int point_count = 0;
        int inlier_count = 0;
        double inlier_frac = 0.0;
        double residual_px = 0.0;
    };

#if HAVE_CCM_GPU_IRLS
    struct DeviceBox {
        float x0 = 0.0f;
        float y0 = 0.0f;
        float x1 = 0.0f;
        float y1 = 0.0f;
    };

    struct GpuReduceSummary {
        double normal[6]{};
        double bx[3]{};
        double by[3]{};
        int point_count = 0;
        int inlier_count = 0;
        double residual_sum = 0.0;
        double cost_sum = 0.0;
        double mag_sum = 0.0;
        double mag_min = 0.0;
        int total_count = 0;
        int bg_count = 0;
    };

    enum class GpuIrlsEstimateStatus {
        Ok,
        Invalid,
        Error,
    };
#endif

    std::string metadata_key_ = "camera_motion";
    bool strict_cuda_ = true;
    int debug_log_every_n_ = 0;
    int grid_size_ = NV_OF_OUTPUT_VECTOR_GRID_SIZE_4;
    // Optional foreground masking excludes flow cells covered by detections so
    // the global estimate reflects background camera motion.
    std::vector<std::string> foreground_metadata_keys_;
    float mask_margin_frac_ = 0.10f;
#if HAVE_OPENCV
    std::string affine_backend_ = "opencv_ransac";
#else
    std::string affine_backend_ = "tx_median";
#endif
    int irls_iters_ = 3;
    double irls_huber_px_ = 2.0;

    AVCUDADeviceContext* cuda_dev_ctx_ = nullptr;
    CUcontext cu_ctx_ = nullptr;
    CUstream stream_ = nullptr;

    // NVOF API + session state.
    NV_OF_CUDA_API_FUNCTION_LIST of_{};
    NvOFHandle hOF_ = nullptr;
    NvOFGPUBufferHandle inBuf_ = nullptr;    // current frame luma
    NvOFGPUBufferHandle refBuf_ = nullptr;   // previous frame luma
    NvOFGPUBufferHandle outBuf_ = nullptr;   // flow vectors (SHORT2)
    NvOFGPUBufferHandle costBuf_ = nullptr;  // output cost (UINT8)
    int of_w_ = 0;
    int of_h_ = 0;
    int grid_w_ = 0;
    int grid_h_ = 0;
    bool have_prev_ = false;
    uint64_t frame_counter_ = 0;

    std::vector<NV_OF_FLOW_VECTOR> host_grid_;
    std::vector<uint8_t> host_cost_;

#if HAVE_CCM_GPU_IRLS
    CUmodule gpu_irls_module_ = nullptr;
    CUfunction gpu_irls_reduce_kernel_ = nullptr;
    CUdeviceptr d_gpu_irls_reduce_ = 0;
    CUdeviceptr d_gpu_irls_boxes_ = 0;
    int gpu_irls_reduce_blocks_ = 0;
    int gpu_irls_box_capacity_ = 0;
#endif

    void destroyOFBuffers() {
        if (of_.nvOFDestroyGPUBufferCuda) {
            if (inBuf_)   { of_.nvOFDestroyGPUBufferCuda(inBuf_);   inBuf_ = nullptr; }
            if (refBuf_)  { of_.nvOFDestroyGPUBufferCuda(refBuf_);  refBuf_ = nullptr; }
            if (outBuf_)  { of_.nvOFDestroyGPUBufferCuda(outBuf_);  outBuf_ = nullptr; }
            if (costBuf_) { of_.nvOFDestroyGPUBufferCuda(costBuf_); costBuf_ = nullptr; }
        }
        of_w_ = of_h_ = grid_w_ = grid_h_ = 0;
        have_prev_ = false;
    }

#if HAVE_CCM_GPU_IRLS
    void releaseGpuIrlsBuffers() {
        if (d_gpu_irls_reduce_) {
            CCM_CHECK_CU(cuMemFree(d_gpu_irls_reduce_));
            d_gpu_irls_reduce_ = 0;
        }
        if (d_gpu_irls_boxes_) {
            CCM_CHECK_CU(cuMemFree(d_gpu_irls_boxes_));
            d_gpu_irls_boxes_ = 0;
        }
        gpu_irls_reduce_blocks_ = 0;
        gpu_irls_box_capacity_ = 0;
    }
#endif

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
            logstream << "cuda_camera_motion: missing hw_frames_ctx";
            return false;
        }
        AVHWFramesContext* fctx = (AVHWFramesContext*)frm.raw()->hw_frames_ctx->data;
        if (!fctx || !fctx->device_ctx || !fctx->device_ctx->hwctx) {
            logstream << "cuda_camera_motion: missing device_ctx/hwctx";
            return false;
        }
        cuda_dev_ctx_ = (AVCUDADeviceContext*)fctx->device_ctx->hwctx;
        if (!cuda_dev_ctx_ || !cuda_dev_ctx_->cuda_ctx) {
            logstream << "cuda_camera_motion: missing CUDA context";
            return false;
        }
        cu_ctx_ = cuda_dev_ctx_->cuda_ctx;
        if (CCM_CHECK_CU(cuCtxSetCurrent(cu_ctx_))) return false;
        stream_ = cuda_dev_ctx_->stream;  // share ffmpeg's stream
        return true;
    }

    bool initOF() {
        if (hOF_) return true;
        if (CCM_CHECK_OF(NvOFAPICreateInstanceCuda(NV_OF_API_VERSION, &of_))) return false;
        if (CCM_CHECK_OF(of_.nvCreateOpticalFlowCuda(cu_ctx_, &hOF_))) return false;
        if (CCM_CHECK_OF(of_.nvOFSetIOCudaStreams(hOF_, (CUstream)stream_, (CUstream)stream_))) return false;
        return true;
    }

#if HAVE_CCM_GPU_IRLS
    bool loadGpuIrlsKernel() {
        if (gpu_irls_reduce_kernel_) return true;
        if (!cu_ctx_) return false;
        if (CCM_CHECK_CU(cuCtxSetCurrent(cu_ctx_))) return false;
        const std::string ptx(avpl_camera_motion_ptx, avpl_camera_motion_ptx + avpl_camera_motion_ptx_len);
        if (CCM_CHECK_CU(cuModuleLoadDataEx(&gpu_irls_module_, ptx.c_str(), 0, nullptr, nullptr))) return false;
        if (CCM_CHECK_CU(cuModuleGetFunction(&gpu_irls_reduce_kernel_, gpu_irls_module_, "kCameraMotionAffineReduce"))) return false;
        return true;
    }

    bool ensureGpuIrlsBuffers(int reduce_blocks, int box_count) {
        if (reduce_blocks <= 0) return false;
        if (gpu_irls_reduce_blocks_ < reduce_blocks) {
            if (d_gpu_irls_reduce_) {
                CCM_CHECK_CU(cuMemFree(d_gpu_irls_reduce_));
                d_gpu_irls_reduce_ = 0;
            }
            if (CCM_CHECK_CU(cuMemAlloc(&d_gpu_irls_reduce_, (size_t)reduce_blocks * 20u * sizeof(double)))) return false;
            gpu_irls_reduce_blocks_ = reduce_blocks;
        }
        if (box_count > gpu_irls_box_capacity_) {
            if (d_gpu_irls_boxes_) {
                CCM_CHECK_CU(cuMemFree(d_gpu_irls_boxes_));
                d_gpu_irls_boxes_ = 0;
            }
            const int cap = std::max(8, box_count);
            if (CCM_CHECK_CU(cuMemAlloc(&d_gpu_irls_boxes_, (size_t)cap * sizeof(DeviceBox)))) return false;
            gpu_irls_box_capacity_ = cap;
        }
        return true;
    }
#endif

    bool ensureSession(int width, int height) {
        if (hOF_ && of_w_ == width && of_h_ == height && inBuf_ && refBuf_ && outBuf_) {
            return true;
        }
        destroyOFBuffers();

        NV_OF_INIT_PARAMS init{};
        init.width = (uint32_t)width;
        init.height = (uint32_t)height;
        init.outGridSize = (NV_OF_OUTPUT_VECTOR_GRID_SIZE)grid_size_;
        init.mode = NV_OF_MODE_OPTICALFLOW;
        init.perfLevel = NV_OF_PERF_LEVEL_FAST;   // realtime-leaning
        init.enableOutputCost = NV_OF_TRUE;
        if (CCM_CHECK_OF(of_.nvOFInit(hOF_, &init))) return false;

        grid_w_ = (width + grid_size_ - 1) / grid_size_;
        grid_h_ = (height + grid_size_ - 1) / grid_size_;

        auto mk = [&](NV_OF_BUFFER_USAGE usage, NV_OF_BUFFER_FORMAT fmt, int w, int h,
                      NvOFGPUBufferHandle* out) -> bool {
            NV_OF_BUFFER_DESCRIPTOR d{};
            d.width = (uint32_t)w; d.height = (uint32_t)h;
            d.bufferUsage = usage; d.bufferFormat = fmt;
            return CCM_CHECK_OF(of_.nvOFCreateGPUBufferCuda(hOF_, &d,
                       NV_OF_CUDA_BUFFER_TYPE_CUDEVICEPTR, out)) == 0;
        };
        if (!mk(NV_OF_BUFFER_USAGE_INPUT,  NV_OF_BUFFER_FORMAT_GRAYSCALE8, width, height, &inBuf_))  return false;
        if (!mk(NV_OF_BUFFER_USAGE_INPUT,  NV_OF_BUFFER_FORMAT_GRAYSCALE8, width, height, &refBuf_)) return false;
        if (!mk(NV_OF_BUFFER_USAGE_OUTPUT, NV_OF_BUFFER_FORMAT_SHORT2,     grid_w_, grid_h_, &outBuf_)) return false;
        if (!mk(NV_OF_BUFFER_USAGE_COST,   NV_OF_BUFFER_FORMAT_UINT8,      grid_w_, grid_h_, &costBuf_)) return false;

        of_w_ = width; of_h_ = height;
        host_grid_.assign((size_t)grid_w_ * grid_h_, NV_OF_FLOW_VECTOR{0, 0});
        host_cost_.assign((size_t)grid_w_ * grid_h_, 0);
        return true;
    }

    // Copy a device luma plane (pitched) into an NVOF GRAYSCALE8 input buffer.
    bool uploadLuma(NvOFGPUBufferHandle buf, CUdeviceptr srcY, int srcPitch, int width, int height) {
        NV_OF_CUDA_BUFFER_STRIDE_INFO si{};
        of_.nvOFGPUBufferGetStrideInfo(buf, &si);
        CUdeviceptr dst = of_.nvOFGPUBufferGetCUdeviceptr(buf);
        CUDA_MEMCPY2D m{};
        m.srcMemoryType = CU_MEMORYTYPE_DEVICE; m.srcDevice = srcY;  m.srcPitch = (size_t)srcPitch;
        m.dstMemoryType = CU_MEMORYTYPE_DEVICE; m.dstDevice = dst;   m.dstPitch = si.strideInfo[0].strideXInBytes;
        m.WidthInBytes = (size_t)width; m.Height = (size_t)height;
        return CCM_CHECK_CU(cuMemcpy2DAsync(&m, stream_)) == 0;
    }

    bool downloadGrid() {
        NV_OF_CUDA_BUFFER_STRIDE_INFO osi{};
        of_.nvOFGPUBufferGetStrideInfo(outBuf_, &osi);
        CUdeviceptr odptr = of_.nvOFGPUBufferGetCUdeviceptr(outBuf_);
        size_t rowBytes = (size_t)grid_w_ * sizeof(NV_OF_FLOW_VECTOR);
        CUDA_MEMCPY2D d{};
        d.srcMemoryType = CU_MEMORYTYPE_DEVICE; d.srcDevice = odptr; d.srcPitch = osi.strideInfo[0].strideXInBytes;
        d.dstMemoryType = CU_MEMORYTYPE_HOST;   d.dstHost = host_grid_.data(); d.dstPitch = rowBytes;
        d.WidthInBytes = rowBytes; d.Height = (size_t)grid_h_;
        if (CCM_CHECK_CU(cuMemcpy2DAsync(&d, stream_))) return false;

        NV_OF_CUDA_BUFFER_STRIDE_INFO csi{};
        of_.nvOFGPUBufferGetStrideInfo(costBuf_, &csi);
        CUdeviceptr cdptr = of_.nvOFGPUBufferGetCUdeviceptr(costBuf_);
        size_t cRow = (size_t)grid_w_ * sizeof(uint8_t);
        CUDA_MEMCPY2D c{};
        c.srcMemoryType = CU_MEMORYTYPE_DEVICE; c.srcDevice = cdptr; c.srcPitch = csi.strideInfo[0].strideXInBytes;
        c.dstMemoryType = CU_MEMORYTYPE_HOST;   c.dstHost = host_cost_.data(); c.dstPitch = cRow;
        c.WidthInBytes = cRow; c.Height = (size_t)grid_h_;
        if (CCM_CHECK_CU(cuMemcpy2DAsync(&c, stream_))) return false;
        return CCM_CHECK_CU(cuStreamSynchronize(stream_)) == 0;
    }

    // Collect foreground boxes (normalized xyxy in [0,1]) from configured
    // detection metadata on the frame, so their flow cells can be excluded.
    // Best-effort + defensive: tolerates missing keys / shape variants. If
    // nothing parses, the caller falls back to a global (unmasked) median.
    std::vector<std::array<float,4>> collectMaskBoxes(const AVFrame* raw) const {
        std::vector<std::array<float,4>> boxes;
        auto tryParse = [&](const std::string& key) -> void {
            if (key.empty() || !raw || !raw->metadata) return;
            AVDictionaryEntry* e = av_dict_get(raw->metadata, key.c_str(), nullptr, 0);
            if (!e || !e->value) return;
            Parameters md;
            try { md = Parameters::parse(e->value); } catch (...) { return; }
            const Parameters* arr = nullptr;
            if (md.is_object() && md.contains("detections") && md["detections"].is_array()) arr = &md["detections"];
            else if (md.is_object() && md.contains("bboxes") && md["bboxes"].is_array()) arr = &md["bboxes"];
            else if (md.is_array()) arr = &md;
            if (!arr) return;
            for (const auto& det : *arr) {
                std::array<float,4> b{};
                bool ok = false;
                if (det.is_object() && det.contains("bbox") && det["bbox"].is_array() && det["bbox"].size() == 4) {
                    for (int i=0;i<4;i++) b[i] = det["bbox"][i].get<float>(); ok = true;
                } else if (det.is_object() && det.contains("xyxy") && det["xyxy"].is_array() && det["xyxy"].size() == 4) {
                    for (int i=0;i<4;i++) b[i] = det["xyxy"][i].get<float>(); ok = true;
                }
                if (!ok) continue;
                // Heuristic: if any coord > 1.5 treat as pixels and normalize.
                if (b[2] > 1.5f || b[3] > 1.5f) {
                    b[0] /= (float)of_w_; b[1] /= (float)of_h_;
                    b[2] /= (float)of_w_; b[3] /= (float)of_h_;
                }
                boxes.push_back(b);
            }
        };
        for (const std::string& key : foreground_metadata_keys_) {
            tryParse(key);
        }
        return boxes;
    }

    static float medianInPlace(std::vector<float>& vals) {
        if (vals.empty()) return 0.0f;
        const size_t mid = vals.size() / 2;
        std::nth_element(vals.begin(), vals.begin() + mid, vals.end());
        return vals[mid];
    }

    static bool solve6x6(double A[6][6], double b[6], double x[6]) {
        double aug[6][7]{};
        for (int r = 0; r < 6; ++r) {
            for (int c = 0; c < 6; ++c) aug[r][c] = A[r][c];
            aug[r][6] = b[r];
        }

        for (int col = 0; col < 6; ++col) {
            int pivot = col;
            double best = std::fabs(aug[col][col]);
            for (int r = col + 1; r < 6; ++r) {
                const double v = std::fabs(aug[r][col]);
                if (v > best) { best = v; pivot = r; }
            }
            if (best < 1e-9 || !std::isfinite(best)) return false;
            if (pivot != col) {
                for (int c = col; c < 7; ++c) std::swap(aug[col][c], aug[pivot][c]);
            }

            const double div = aug[col][col];
            for (int c = col; c < 7; ++c) aug[col][c] /= div;
            for (int r = 0; r < 6; ++r) {
                if (r == col) continue;
                const double f = aug[r][col];
                if (f == 0.0) continue;
                for (int c = col; c < 7; ++c) aug[r][c] -= f * aug[col][c];
            }
        }

        for (int i = 0; i < 6; ++i) {
            x[i] = aug[i][6];
            if (!std::isfinite(x[i])) return false;
        }
        return true;
    }

    static bool weightedAffineFit(const std::vector<FlowCell>& cells,
                                  const std::vector<double>& weights,
                                  double out[6]) {
        double AtA[6][6]{};
        double Atb[6]{};

        auto accum = [&](const double row[6], double target, double w) {
            if (w <= 0.0 || !std::isfinite(w)) return;
            for (int i = 0; i < 6; ++i) {
                Atb[i] += w * row[i] * target;
                for (int j = 0; j < 6; ++j) AtA[i][j] += w * row[i] * row[j];
            }
        };

        for (size_t i = 0; i < cells.size(); ++i) {
            const FlowCell& cell = cells[i];
            const double w = weights.empty() ? 1.0 : weights[i];
            const double x = cell.x;
            const double y = cell.y;
            const double row_x[6] = {x, y, 1.0, 0.0, 0.0, 0.0};
            const double row_y[6] = {0.0, 0.0, 0.0, x, y, 1.0};
            accum(row_x, x + cell.dx, w);
            accum(row_y, y + cell.dy, w);
        }

        return solve6x6(AtA, Atb, out);
    }

    static bool solveAffineFromReduced(const double normal[6], const double bx[3],
                                       const double by[3], double out[6]) {
        double A[6][6]{};
        double b[6]{};
        A[0][0] = normal[0]; A[0][1] = normal[1]; A[0][2] = normal[2];
        A[1][0] = normal[1]; A[1][1] = normal[3]; A[1][2] = normal[4];
        A[2][0] = normal[2]; A[2][1] = normal[4]; A[2][2] = normal[5];
        A[3][3] = normal[0]; A[3][4] = normal[1]; A[3][5] = normal[2];
        A[4][3] = normal[1]; A[4][4] = normal[3]; A[4][5] = normal[4];
        A[5][3] = normal[2]; A[5][4] = normal[4]; A[5][5] = normal[5];
        b[0] = bx[0]; b[1] = bx[1]; b[2] = bx[2];
        b[3] = by[0]; b[4] = by[1]; b[5] = by[2];
        return solve6x6(A, b, out);
    }

    MotionSummary summarizeMotion(const std::vector<std::array<float,4>>& boxes) {
        MotionSummary s;
        const size_t n = (size_t)grid_w_ * grid_h_;
        std::vector<FlowCell> bg; bg.reserve(n);
        std::vector<FlowCell> all; all.reserve(n);
        const float m = mask_margin_frac_;
        for (int gy = 0; gy < grid_h_; ++gy) {
            for (int gx = 0; gx < grid_w_; ++gx) {
                const NV_OF_FLOW_VECTOR& v = host_grid_[(size_t)gy * grid_w_ + gx];
                FlowCell cell;
                cell.x = (gx + 0.5f) * (float)grid_size_;
                cell.y = (gy + 0.5f) * (float)grid_size_;
                cell.dx = v.flowx / 32.0f;
                cell.dy = v.flowy / 32.0f;
                cell.mag = std::sqrt(cell.dx * cell.dx + cell.dy * cell.dy);
                all.push_back(cell);

                // Cell center in normalized coords.
                const float cxn = ((gx + 0.5f) * grid_size_) / (float)of_w_;
                const float cyn = ((gy + 0.5f) * grid_size_) / (float)of_h_;
                bool masked = false;
                for (const auto& b : boxes) {
                    const float mw = m * std::max(0.0f, b[2] - b[0]);
                    const float mh = m * std::max(0.0f, b[3] - b[1]);
                    if (cxn >= b[0]-mw && cxn <= b[2]+mw && cyn >= b[1]-mh && cyn <= b[3]+mh) {
                        masked = true; break;
                    }
                }
                if (!masked) bg.push_back(cell);
            }
        }

        s.total_cell_count = (int)all.size();
        s.bg_cell_count = (int)bg.size();
        s.bg_cell_frac = all.empty() ? 0.0f : (float)bg.size() / (float)all.size();
        s.used_mask_fallback = bg.size() < 8 && !all.empty();
        s.cells = s.used_mask_fallback ? all : bg;
        if (s.cells.empty()) return s;

        std::vector<float> txs;
        txs.reserve(s.cells.size());
        double mag_sum = 0.0;
        float mag_min = std::numeric_limits<float>::max();
        for (const FlowCell& cell : s.cells) {
            txs.push_back(cell.dx);
            mag_sum += cell.mag;
            mag_min = std::min(mag_min, cell.mag);
        }
        s.tx_median = medianInPlace(txs);
        s.flow_mag_min = (mag_min == std::numeric_limits<float>::max()) ? 0.0f : mag_min;
        s.flow_mag_mean = (float)(mag_sum / (double)s.cells.size());

        std::vector<float> abs_dev;
        abs_dev.reserve(s.cells.size());
        for (const FlowCell& cell : s.cells) {
            abs_dev.push_back(std::fabs(cell.dx - s.tx_median));
        }
        s.tx_mad = medianInPlace(abs_dev);
        return s;
    }

    AffineSummary estimateAffineIrls(const MotionSummary& motion) const {
        AffineSummary a;
        a.m02 = motion.tx_median;
        a.point_count = (int)motion.cells.size();
        if (motion.cells.size() < 8) return a;

        std::vector<double> weights(motion.cells.size(), 1.0);
        double p[6] = {1.0, 0.0, motion.tx_median, 0.0, 1.0, 0.0};
        if (!weightedAffineFit(motion.cells, weights, p)) return a;

        const double huber = std::max(0.25, irls_huber_px_);
        const int iters = std::max(0, irls_iters_);
        for (int iter = 0; iter < iters; ++iter) {
            for (size_t i = 0; i < motion.cells.size(); ++i) {
                const FlowCell& cell = motion.cells[i];
                const double px = p[0] * cell.x + p[1] * cell.y + p[2];
                const double py = p[3] * cell.x + p[4] * cell.y + p[5];
                const double ex = px - (cell.x + cell.dx);
                const double ey = py - (cell.y + cell.dy);
                const double r = std::sqrt(ex * ex + ey * ey);
                weights[i] = (r <= huber || r <= 1e-9) ? 1.0 : (huber / r);
            }
            if (!weightedAffineFit(motion.cells, weights, p)) return a;
        }

        a.m00 = p[0]; a.m01 = p[1]; a.m02 = p[2];
        a.m10 = p[3]; a.m11 = p[4]; a.m12 = p[5];

        double residual_sum = 0.0;
        for (const FlowCell& cell : motion.cells) {
            const double px = a.m00 * cell.x + a.m01 * cell.y + a.m02;
            const double py = a.m10 * cell.x + a.m11 * cell.y + a.m12;
            const double ex = px - (cell.x + cell.dx);
            const double ey = py - (cell.y + cell.dy);
            const double r = std::sqrt(ex * ex + ey * ey);
            if (r <= huber) {
                ++a.inlier_count;
                residual_sum += r;
            }
        }
        a.inlier_frac = a.point_count > 0 ? (double)a.inlier_count / (double)a.point_count : 0.0;
        a.residual_px = a.inlier_count > 0 ? residual_sum / (double)a.inlier_count : 0.0;
        a.valid = a.inlier_count >= 8 && std::isfinite(a.residual_px);
        if (!a.valid) {
            a.m00 = 1.0; a.m01 = 0.0; a.m02 = motion.tx_median;
            a.m10 = 0.0; a.m11 = 1.0; a.m12 = 0.0;
        }
        return a;
    }

#if HAVE_CCM_GPU_IRLS
    static void populateGpuMotionSummary(const GpuReduceSummary& red, bool used_mask,
                                         bool has_boxes, MotionSummary& motion, float& cost) {
        motion.flow_mag_min = (float)red.mag_min;
        motion.flow_mag_mean = red.point_count > 0 ? (float)(red.mag_sum / (double)red.point_count) : 0.0f;
        motion.bg_cell_count = red.bg_count;
        motion.total_cell_count = red.total_count;
        motion.bg_cell_frac = red.total_count > 0 ? (float)red.bg_count / (float)red.total_count : 0.0f;
        motion.used_mask_fallback = !used_mask && has_boxes;
        cost = red.total_count > 0 ? (float)(red.cost_sum / (double)red.total_count) : 0.0f;
    }

    bool reduceGpuAffine(const std::vector<std::array<float,4>>& boxes, const double p[6],
                         bool use_weights, bool allow_mask, GpuReduceSummary& out) {
        if (!loadGpuIrlsKernel()) return false;

        const int threads = 256;
        const int n = grid_w_ * grid_h_;
        const int blocks = std::max(1, std::min(512, (n + threads - 1) / threads));
        const int box_count = allow_mask ? (int)boxes.size() : 0;
        if (!ensureGpuIrlsBuffers(blocks, box_count)) return false;

        if (box_count > 0) {
            std::vector<DeviceBox> hboxes;
            hboxes.reserve((size_t)box_count);
            for (const auto& b : boxes) {
                hboxes.push_back(DeviceBox{b[0], b[1], b[2], b[3]});
            }
            if (CCM_CHECK_CU(cuMemcpyHtoDAsync(d_gpu_irls_boxes_, hboxes.data(),
                                               hboxes.size() * sizeof(DeviceBox), stream_))) return false;
        }

        NV_OF_CUDA_BUFFER_STRIDE_INFO osi{};
        of_.nvOFGPUBufferGetStrideInfo(outBuf_, &osi);
        CUdeviceptr odptr = of_.nvOFGPUBufferGetCUdeviceptr(outBuf_);
        NV_OF_CUDA_BUFFER_STRIDE_INFO csi{};
        of_.nvOFGPUBufferGetStrideInfo(costBuf_, &csi);
        CUdeviceptr cdptr = of_.nvOFGPUBufferGetCUdeviceptr(costBuf_);
        int flow_stride = (int)osi.strideInfo[0].strideXInBytes;
        int cost_stride = (int)csi.strideInfo[0].strideXInBytes;
        int iw = of_w_;
        int ih = of_h_;
        float margin = mask_margin_frac_;
        int use_w = use_weights ? 1 : 0;
        double huber = std::max(0.25, irls_huber_px_);

        void* args[] = {
            (void*)&odptr,
            (void*)&cdptr,
            (void*)&flow_stride,
            (void*)&cost_stride,
            (void*)&grid_w_,
            (void*)&grid_h_,
            (void*)&grid_size_,
            (void*)&iw,
            (void*)&ih,
            (void*)&d_gpu_irls_boxes_,
            (void*)&box_count,
            (void*)&margin,
            (void*)&use_w,
            (void*)&p[0],
            (void*)&p[1],
            (void*)&p[2],
            (void*)&p[3],
            (void*)&p[4],
            (void*)&p[5],
            (void*)&huber,
            (void*)&d_gpu_irls_reduce_,
        };
        const unsigned int shared_bytes = (unsigned int)(threads * sizeof(double));
        if (CCM_CHECK_CU(cuLaunchKernel(gpu_irls_reduce_kernel_,
                                        (unsigned int)blocks, 1, 1,
                                        (unsigned int)threads, 1, 1,
                                        shared_bytes, stream_, args, nullptr))) return false;

        std::vector<double> h((size_t)blocks * 20u, 0.0);
        if (CCM_CHECK_CU(cuMemcpyDtoHAsync(h.data(), d_gpu_irls_reduce_,
                                           h.size() * sizeof(double), stream_)) ||
            CCM_CHECK_CU(cuStreamSynchronize(stream_))) return false;

        GpuReduceSummary s;
        s.mag_min = std::numeric_limits<double>::max();
        for (int bi = 0; bi < blocks; ++bi) {
            const double* v = h.data() + (size_t)bi * 20u;
            for (int i = 0; i < 6; ++i) s.normal[i] += v[i];
            for (int i = 0; i < 3; ++i) s.bx[i] += v[6 + i];
            for (int i = 0; i < 3; ++i) s.by[i] += v[9 + i];
            s.point_count += (int)std::llround(v[12]);
            s.inlier_count += (int)std::llround(v[13]);
            s.residual_sum += v[14];
            s.cost_sum += v[15];
            s.mag_sum += v[16];
            s.mag_min = std::min(s.mag_min, v[17]);
            s.total_count += (int)std::llround(v[18]);
            s.bg_count += (int)std::llround(v[19]);
        }
        if (s.mag_min == std::numeric_limits<double>::max()) s.mag_min = 0.0;
        out = s;
        return true;
    }

    GpuIrlsEstimateStatus estimateAffineGpuIrls(const std::vector<std::array<float,4>>& boxes,
                                                MotionSummary& motion, AffineSummary& affine,
                                                float& cost, const char*& status) {
        double p[6] = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0};
        GpuReduceSummary red;
        bool used_mask = !boxes.empty();
        status = "ok";
        if (!reduceGpuAffine(boxes, p, false, true, red)) return GpuIrlsEstimateStatus::Error;
        if (red.point_count < 8 && used_mask) {
            used_mask = false;
            if (!reduceGpuAffine(boxes, p, false, false, red)) return GpuIrlsEstimateStatus::Error;
        }
        populateGpuMotionSummary(red, used_mask, !boxes.empty(), motion, cost);
        affine.point_count = red.point_count;
        if (red.point_count < 8) {
            status = "affine_insufficient_points";
            return GpuIrlsEstimateStatus::Invalid;
        }
        if (!solveAffineFromReduced(red.normal, red.bx, red.by, p)) {
            status = "affine_singular";
            return GpuIrlsEstimateStatus::Invalid;
        }

        const int iters = std::max(0, irls_iters_);
        for (int iter = 0; iter < iters; ++iter) {
            if (!reduceGpuAffine(boxes, p, true, used_mask, red)) return GpuIrlsEstimateStatus::Error;
            populateGpuMotionSummary(red, used_mask, !boxes.empty(), motion, cost);
            affine.point_count = red.point_count;
            if (red.point_count < 8) {
                status = "affine_insufficient_points";
                return GpuIrlsEstimateStatus::Invalid;
            }
            if (!solveAffineFromReduced(red.normal, red.bx, red.by, p)) {
                status = "affine_singular";
                return GpuIrlsEstimateStatus::Invalid;
            }
        }

        if (!reduceGpuAffine(boxes, p, false, used_mask, red)) return GpuIrlsEstimateStatus::Error;

        affine.m00 = p[0]; affine.m01 = p[1]; affine.m02 = p[2];
        affine.m10 = p[3]; affine.m11 = p[4]; affine.m12 = p[5];
        affine.point_count = red.point_count;
        affine.inlier_count = red.inlier_count;
        affine.inlier_frac = red.point_count > 0 ? (double)red.inlier_count / (double)red.point_count : 0.0;
        affine.residual_px = red.inlier_count > 0 ? red.residual_sum / (double)red.inlier_count : 0.0;
        affine.valid = red.inlier_count >= 8 && std::isfinite(affine.residual_px);

        motion.tx_median = (float)affine.m02;
        motion.tx_mad = 0.0f;
        populateGpuMotionSummary(red, used_mask, !boxes.empty(), motion, cost);
        if (!affine.valid) {
            status = "affine_low_inliers";
        }
        return affine.valid ? GpuIrlsEstimateStatus::Ok : GpuIrlsEstimateStatus::Invalid;
    }
#endif

    AffineSummary estimateAffine(const MotionSummary& motion) {
        AffineSummary a;
        a.m02 = motion.tx_median;
        a.point_count = (int)motion.cells.size();
        if (affine_backend_ == "tx_median" || affine_backend_ == "translation") return a;
        if (affine_backend_ == "cpu_irls" || affine_backend_ == "irls") {
            return estimateAffineIrls(motion);
        }
#if HAVE_OPENCV
        if (motion.cells.size() < 8) return a;

        std::vector<cv::Point2f> src;
        std::vector<cv::Point2f> dst;
        src.reserve(motion.cells.size());
        dst.reserve(motion.cells.size());
        for (const FlowCell& cell : motion.cells) {
            src.emplace_back(cell.x, cell.y);
            dst.emplace_back(cell.x + cell.dx, cell.y + cell.dy);
        }

        cv::Mat inliers;
        cv::Mat M = cv::estimateAffine2D(src, dst, inliers, cv::RANSAC, 2.0);
        if (M.empty()) return a;
        if (M.type() != CV_64F) M.convertTo(M, CV_64F);

        a.m00 = M.at<double>(0, 0);
        a.m01 = M.at<double>(0, 1);
        a.m02 = M.at<double>(0, 2);
        a.m10 = M.at<double>(1, 0);
        a.m11 = M.at<double>(1, 1);
        a.m12 = M.at<double>(1, 2);
        a.valid = true;

        double residual_sum = 0.0;
        for (int i = 0; i < inliers.rows; ++i) {
            if (!inliers.at<uint8_t>(i, 0)) continue;
            ++a.inlier_count;
            const FlowCell& cell = motion.cells[(size_t)i];
            const double px = a.m00 * cell.x + a.m01 * cell.y + a.m02;
            const double py = a.m10 * cell.x + a.m11 * cell.y + a.m12;
            const double ex = px - (cell.x + cell.dx);
            const double ey = py - (cell.y + cell.dy);
            residual_sum += std::sqrt(ex * ex + ey * ey);
        }
        a.inlier_frac = a.point_count > 0 ? (double)a.inlier_count / (double)a.point_count : 0.0;
        a.residual_px = a.inlier_count > 0 ? residual_sum / (double)a.inlier_count : 0.0;
#endif
        return a;
    }

    void writeMetadata(av::VideoFrame& frm, bool has_prev, const MotionSummary& motion,
                       const AffineSummary& affine, float cost, const std::string& status) {
        std::ostringstream md;
        md.precision(6);
        md << "{"
           << "\"frame_index\":" << (frame_counter_ > 0 ? frame_counter_ - 1 : 0) << ","
           << "\"has_prev\":" << (has_prev ? "true" : "false") << ","
           << "\"backend\":\"" << affine_backend_ << "\","
           << "\"tx\":" << affine.m02 << ","
           << "\"ty\":" << affine.m12 << ","
           << "\"affine_2x3\":[["
           << affine.m00 << "," << affine.m01 << "," << affine.m02 << "],["
           << affine.m10 << "," << affine.m11 << "," << affine.m12 << "]],"
           << "\"affine_valid\":" << (affine.valid ? "true" : "false") << ","
           << "\"affine_point_count\":" << affine.point_count << ","
           << "\"affine_inlier_count\":" << affine.inlier_count << ","
           << "\"affine_inlier_frac\":" << affine.inlier_frac << ","
           << "\"affine_residual_px\":" << affine.residual_px << ","
           << "\"nvof_cost\":" << cost << ","
           << "\"flow_mag_min\":" << motion.flow_mag_min << ","
           << "\"flow_mag_mean\":" << motion.flow_mag_mean << ","
           << "\"bg_cell_frac\":" << motion.bg_cell_frac << ","
           << "\"tx_mad\":" << motion.tx_mad << ","
           << "\"tx_median\":" << motion.tx_median << ","
           << "\"mask_fallback\":" << (motion.used_mask_fallback ? "true" : "false") << ","
           << "\"status\":\"" << status << "\""
           << "}";
        av_dict_set(&frm.raw()->metadata, metadata_key_.c_str(), md.str().c_str(), 0);
    }

public:
    using NodeSISO<av::VideoFrame, av::VideoFrame>::NodeSISO;

    ~CudaCameraMotion() {
        if (cu_ctx_) CCM_CHECK_CU(cuCtxSetCurrent(cu_ctx_));
        destroyOFBuffers();
#if HAVE_CCM_GPU_IRLS
        releaseGpuIrlsBuffers();
        if (gpu_irls_module_) {
            CCM_CHECK_CU(cuModuleUnload(gpu_irls_module_));
            gpu_irls_module_ = nullptr;
        }
#endif
        if (hOF_ && of_.nvOFDestroy) { of_.nvOFDestroy(hOF_); hOF_ = nullptr; }
    }

    bool consumeEofIfPresent() override { return false; }

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
            if (strict_cuda_) throw Error("cuda_camera_motion: input frame is not AV_PIX_FMT_CUDA");
            writeMetadata(frm, false, MotionSummary{}, AffineSummary{}, 0.0f, "skipped_non_cuda");
            this->sink_->put(frm); return;
        }
        if (!isSupportedCudaFormat(hwSwFormat(frm))) {
            if (strict_cuda_) throw Error("cuda_camera_motion: unsupported CUDA sw_format");
            writeMetadata(frm, false, MotionSummary{}, AffineSummary{}, 0.0f, "unsupported_sw_format");
            this->sink_->put(frm); return;
        }
        if (!raw->data[0] || raw->linesize[0] <= 0) {
            if (strict_cuda_) throw Error("cuda_camera_motion: invalid luma plane");
            writeMetadata(frm, false, MotionSummary{}, AffineSummary{}, 0.0f, "invalid_luma_plane");
            this->sink_->put(frm); return;
        }
        if (!initCudaContextFromFrame(frm) || !initOF() || !ensureSession(frm.width(), frm.height())) {
            if (strict_cuda_) throw Error("cuda_camera_motion: failed to initialize NVOF");
            writeMetadata(frm, false, MotionSummary{}, AffineSummary{}, 0.0f, "nvof_init_failed");
            this->sink_->put(frm); return;
        }

        const CUdeviceptr y_plane = (CUdeviceptr)(uintptr_t)raw->data[0];
        const int y_pitch = raw->linesize[0];

        // Upload current luma into inBuf_.
        if (!uploadLuma(inBuf_, y_plane, y_pitch, of_w_, of_h_)) {
            throw Error("cuda_camera_motion: luma upload failed");
        }

        if (!have_prev_) {
            // First frame: no reference yet. Seed refBuf_ with current and emit tx=0.
            if (!uploadLuma(refBuf_, y_plane, y_pitch, of_w_, of_h_)) {
                throw Error("cuda_camera_motion: ref seed upload failed");
            }
            CCM_CHECK_CU(cuStreamSynchronize(stream_));
            have_prev_ = true;
            writeMetadata(frm, false, MotionSummary{}, AffineSummary{}, 0.0f, "ok");
            this->sink_->put(frm);
            return;
        }

        NV_OF_EXECUTE_INPUT_PARAMS  ein{};
        NV_OF_EXECUTE_OUTPUT_PARAMS eout{};
        ein.inputFrame = inBuf_;
        ein.referenceFrame = refBuf_;
        ein.disableTemporalHints = NV_OF_FALSE;
        eout.outputBuffer = outBuf_;
        eout.outputCostBuffer = costBuf_;
        if (CCM_CHECK_OF(of_.nvOFExecute(hOF_, &ein, &eout))) {
            throw Error("cuda_camera_motion: nvOFExecute failed");
        }
        // Background flow cells exclude ball/player boxes. If too few remain,
        // use all cells to keep metadata populated.
        const size_t n = (size_t)grid_w_ * grid_h_;
        const std::vector<std::array<float,4>> mask_boxes = collectMaskBoxes(raw);
        MotionSummary motion;
        AffineSummary affine;
        float cost = 0.0f;
#if HAVE_CCM_GPU_IRLS
        if (affine_backend_ == "gpu_irls" || affine_backend_ == "cuda_irls") {
            const char* estimate_metadata_status = "ok";
            GpuIrlsEstimateStatus estimate_status = estimateAffineGpuIrls(
                mask_boxes, motion, affine, cost, estimate_metadata_status);
            if (estimate_status == GpuIrlsEstimateStatus::Error) {
                throw Error("cuda_camera_motion: gpu_irls estimation failed");
            }
            if (estimate_status == GpuIrlsEstimateStatus::Invalid) {
                writeMetadata(frm, true, motion, affine, cost, estimate_metadata_status);
                if (!uploadLuma(refBuf_, y_plane, y_pitch, of_w_, of_h_)) {
                    throw Error("cuda_camera_motion: ref roll upload failed");
                }
                CCM_CHECK_CU(cuStreamSynchronize(stream_));
                this->sink_->put(frm);
                return;
            }
        } else
#endif
        {
            if (!downloadGrid()) throw Error("cuda_camera_motion: flow download failed");
            motion = summarizeMotion(mask_boxes);
            affine = estimateAffine(motion);
            double cost_sum = 0.0;
            for (size_t i = 0; i < n; ++i) cost_sum += (double)host_cost_[i];
            cost = n ? (float)(cost_sum / (double)n) : 0.0f;
        }

        writeMetadata(frm, true, motion, affine, cost, "ok");

        // Current becomes the reference for the next frame.
        if (!uploadLuma(refBuf_, y_plane, y_pitch, of_w_, of_h_)) {
            throw Error("cuda_camera_motion: ref roll upload failed");
        }
        CCM_CHECK_CU(cuStreamSynchronize(stream_));

        if (debug_log_every_n_ > 0 && (frame_counter_ % (uint64_t)debug_log_every_n_) == 0) {
            logstream << "cuda_camera_motion: frame=" << (frame_counter_ - 1)
                      << " tx=" << affine.m02 << " ty=" << affine.m12
                      << " affine_valid=" << affine.valid << " cost=" << cost
                      << " grid=" << grid_w_ << "x" << grid_h_;
        }

        this->sink_->put(frm);
    }

    static std::shared_ptr<CudaCameraMotion> create(NodeCreationInfo& nci) {
        EdgeManager& edges = nci.edges;
        const Parameters& params = nci.params;
        auto r = NodeSISO<av::VideoFrame, av::VideoFrame>::template createCommon<CudaCameraMotion>(edges, params);
        if (params.count("metadata_key")) r->metadata_key_ = params["metadata_key"].get<std::string>();
        if (params.count("strict_cuda")) r->strict_cuda_ = params["strict_cuda"].get<bool>();
        if (params.count("debug_log_every_n")) r->debug_log_every_n_ = params["debug_log_every_n"].get<int>();
        if (params.count("grid_size")) r->grid_size_ = params["grid_size"].get<int>();
        if (params.count("foreground_metadata_keys")) {
            if (!params["foreground_metadata_keys"].is_array()) {
                throw Error("cuda_camera_motion: foreground_metadata_keys must be a string array");
            }
            for (const auto& key : params["foreground_metadata_keys"]) {
                if (!key.is_string()) {
                    throw Error("cuda_camera_motion: foreground_metadata_keys must be a string array");
                }
                r->foreground_metadata_keys_.push_back(key.get<std::string>());
            }
        }
        if (params.count("mask_margin_frac")) r->mask_margin_frac_ = params["mask_margin_frac"].get<float>();
        if (params.count("affine_backend")) r->affine_backend_ = params["affine_backend"].get<std::string>();
        if (params.count("camera_motion_backend")) r->affine_backend_ = params["camera_motion_backend"].get<std::string>();
        if (params.count("irls_iters")) r->irls_iters_ = params["irls_iters"].get<int>();
        if (params.count("irls_huber_px")) r->irls_huber_px_ = params["irls_huber_px"].get<double>();
        return r;
    }
};

DECLNODE(cuda_camera_motion, CudaCameraMotion)
