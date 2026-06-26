// CudaCameraMotion — per-frame horizontal camera translation (tx) via the
// NVIDIA Optical Flow (NVOF) dense engine.
//
// Modeled on luma_diff.cpp: grabs the CUDA context/stream from the incoming
// AV_PIX_FMT_CUDA frame, keeps the previous frame's luma in an NVOF input
// buffer, runs dense optical flow between (prev -> cur), reduces the flow grid
// to a masked-median horizontal component (tx, in pixels), and writes
// {tx, nvof_cost, has_prev, ...} into frame metadata.
//
// Reduction: median of background flowx, excluding cells covered by ball/player
// boxes (read from frame metadata, margin-grown) so tx reflects camera motion;
// global-median fallback when too few background cells survive. The contract
// consumed downstream (Sports-Reframing-Service camera_motion.py) is just
// per-frame tx.
//
// The same single pass over the grid also emits diagnostic scalars for studying
// scene-cut vs. camera-pan separation offline: flow_mag_min (p5 background cell
// |flow|), flow_mag_mean, bg_cell_frac (surviving background cells / total), and
// tx_mad (spread of background flowx about tx). Additive metadata; tx and the
// downstream contract are unchanged.
//
// The NVOF dense engine ships with the driver (libnvidia-opticalflow.so);
// headers are vendored in deps/Optical_Flow_SDK_5.0.7. Built only when
// HAVE_NVOF=1 (see Makefile gate).

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
#include <array>
#include <cmath>
#include <cstdint>
#include <sstream>
#include <string>
#include <vector>

#include "nvOpticalFlowCommon.h"
#include "nvOpticalFlowCuda.h"

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
    std::string metadata_key_ = "srs_camera_motion";
    bool strict_cuda_ = true;
    int debug_log_every_n_ = 0;
    int grid_size_ = NV_OF_OUTPUT_VECTOR_GRID_SIZE_4;
    // Foreground masking: exclude flow cells covered by ball/player boxes so the
    // global median reflects BACKGROUND (camera) motion, not players/ball.
    std::string player_metadata_key_ = "srs_yolo_players";
    std::string ball_metadata_key_ = "tracknet_ball";
    float mask_margin_frac_ = 0.10f;

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

    // Collect foreground boxes (normalized xyxy in [0,1]) from ball/player
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
        tryParse(ball_metadata_key_);
        tryParse(player_metadata_key_);
        return boxes;
    }

    // Per-frame reduction results over the NVOF grid. tx is the camera signal
    // consumed downstream; the rest are diagnostic scalars (Phase 2 "cheap wins")
    // for offline study of scene-cut vs. pan separation. All derived in one pass
    // over the same coarse grid already in host memory.
    struct CameraMotionStats {
        float tx = 0.0f;            // masked median of background flowx (px)
        float tx_mad = 0.0f;        // median abs deviation of background flowx about tx (px)
        float flow_mag_min = 0.0f;  // low-percentile (p5) background cell |flow| (px)
        float flow_mag_mean = 0.0f; // mean background cell |flow| (px)
        float bg_cell_frac = 0.0f;  // surviving background cells / total grid cells
        bool  masked_used = false;  // true if the masked set was used (else global fallback)
    };

    static float percentileSorted(std::vector<float>& v, float pct) {
        if (v.empty()) return 0.0f;
        size_t k = (size_t)(pct * (float)(v.size() - 1));
        if (k >= v.size()) k = v.size() - 1;
        std::nth_element(v.begin(), v.begin() + k, v.end());
        return v[k];
    }

    static float medianSorted(std::vector<float>& v) {
        if (v.empty()) return 0.0f;
        std::nth_element(v.begin(), v.begin() + v.size() / 2, v.end());
        return v[v.size() / 2];
    }

    // Reduce the flow grid to tx (masked median flowx) plus diagnostic scalars.
    // Cells whose center falls inside any mask box (grown by margin) are excluded
    // so tx/diagnostics reflect BACKGROUND (camera) motion. Falls back to all
    // cells if masking would leave too few samples.
    CameraMotionStats reduceGrid(const std::vector<std::array<float,4>>& boxes) {
        const size_t n = (size_t)grid_w_ * grid_h_;
        std::vector<float> fg_fx; fg_fx.reserve(n);   // background flowx
        std::vector<float> fg_mag; fg_mag.reserve(n); // background |flow|
        std::vector<float> all_fx; all_fx.reserve(n);
        std::vector<float> all_mag; all_mag.reserve(n);
        const float m = mask_margin_frac_;
        for (int gy = 0; gy < grid_h_; ++gy) {
            for (int gx = 0; gx < grid_w_; ++gx) {
                const NV_OF_FLOW_VECTOR& v = host_grid_[(size_t)gy * grid_w_ + gx];
                const float fx = v.flowx / 32.0f;
                const float fy = v.flowy / 32.0f;
                const float mag = std::sqrt(fx * fx + fy * fy);
                all_fx.push_back(fx);
                all_mag.push_back(mag);
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
                if (!masked) { fg_fx.push_back(fx); fg_mag.push_back(mag); }
            }
        }

        CameraMotionStats s;
        if (n == 0) return s;
        s.masked_used = (fg_fx.size() >= 8);
        std::vector<float>& use_fx = s.masked_used ? fg_fx : all_fx;
        std::vector<float>& use_mag = s.masked_used ? fg_mag : all_mag;
        s.bg_cell_frac = (float)fg_fx.size() / (float)n;
        if (use_fx.empty()) return s;

        s.tx = medianSorted(use_fx);
        // MAD about tx: spread of background flowx (agreement of cells).
        std::vector<float> dev; dev.reserve(use_fx.size());
        for (float fx : use_fx) dev.push_back(std::fabs(fx - s.tx));
        s.tx_mad = medianSorted(dev);
        // Magnitude stats: min (p5, robust) and mean over the used cell set.
        double mag_sum = 0.0;
        for (float mg : use_mag) mag_sum += (double)mg;
        s.flow_mag_mean = (float)(mag_sum / (double)use_mag.size());
        s.flow_mag_min = percentileSorted(use_mag, 0.05f);
        return s;
    }

    void writeMetadata(av::VideoFrame& frm, bool has_prev, const CameraMotionStats& s,
                       float cost, const std::string& status) {
        std::ostringstream md;
        md.precision(6);
        md << "{"
           << "\"frame_index\":" << (frame_counter_ > 0 ? frame_counter_ - 1 : 0) << ","
           << "\"has_prev\":" << (has_prev ? "true" : "false") << ","
           << "\"tx\":" << s.tx << ","
           << "\"nvof_cost\":" << cost << ","
           << "\"tx_mad\":" << s.tx_mad << ","
           << "\"flow_mag_min\":" << s.flow_mag_min << ","
           << "\"flow_mag_mean\":" << s.flow_mag_mean << ","
           << "\"bg_cell_frac\":" << s.bg_cell_frac << ","
           << "\"status\":\"" << status << "\""
           << "}";
        av_dict_set(&frm.raw()->metadata, metadata_key_.c_str(), md.str().c_str(), 0);
    }

public:
    using NodeSISO<av::VideoFrame, av::VideoFrame>::NodeSISO;

    ~CudaCameraMotion() {
        if (cu_ctx_) CCM_CHECK_CU(cuCtxSetCurrent(cu_ctx_));
        destroyOFBuffers();
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
            writeMetadata(frm, false, CameraMotionStats{}, 0.0f, "skipped_non_cuda");
            this->sink_->put(frm); return;
        }
        if (!isSupportedCudaFormat(hwSwFormat(frm))) {
            if (strict_cuda_) throw Error("cuda_camera_motion: unsupported CUDA sw_format");
            writeMetadata(frm, false, CameraMotionStats{}, 0.0f, "unsupported_sw_format");
            this->sink_->put(frm); return;
        }
        if (!raw->data[0] || raw->linesize[0] <= 0) {
            if (strict_cuda_) throw Error("cuda_camera_motion: invalid luma plane");
            writeMetadata(frm, false, CameraMotionStats{}, 0.0f, "invalid_luma_plane");
            this->sink_->put(frm); return;
        }
        if (!initCudaContextFromFrame(frm) || !initOF() || !ensureSession(frm.width(), frm.height())) {
            if (strict_cuda_) throw Error("cuda_camera_motion: failed to initialize NVOF");
            writeMetadata(frm, false, CameraMotionStats{}, 0.0f, "nvof_init_failed");
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
            writeMetadata(frm, false, CameraMotionStats{}, 0.0f, "ok");
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
        if (!downloadGrid()) throw Error("cuda_camera_motion: flow download failed");

        // Reduce the flow grid: tx (masked median flowx, px) reflects background
        // (camera) motion — cells covered by ball/player boxes are excluded, with
        // global fallback when too few background cells. The same pass yields the
        // diagnostic scalars (flow_mag_min/mean, bg_cell_frac, tx_mad).
        // cost = mean NVOF cost over the grid.
        const size_t n = (size_t)grid_w_ * grid_h_;
        const std::vector<std::array<float,4>> mask_boxes = collectMaskBoxes(raw);
        const CameraMotionStats stats = reduceGrid(mask_boxes);
        double cost_sum = 0.0;
        for (size_t i = 0; i < n; ++i) cost_sum += (double)host_cost_[i];
        const float cost = n ? (float)(cost_sum / (double)n) : 0.0f;

        writeMetadata(frm, true, stats, cost, "ok");

        // Current becomes the reference for the next frame.
        if (!uploadLuma(refBuf_, y_plane, y_pitch, of_w_, of_h_)) {
            throw Error("cuda_camera_motion: ref roll upload failed");
        }
        CCM_CHECK_CU(cuStreamSynchronize(stream_));

        if (debug_log_every_n_ > 0 && (frame_counter_ % (uint64_t)debug_log_every_n_) == 0) {
            logstream << "cuda_camera_motion: frame=" << (frame_counter_ - 1)
                      << " tx=" << stats.tx << " cost=" << cost
                      << " mag_min=" << stats.flow_mag_min
                      << " mag_mean=" << stats.flow_mag_mean
                      << " bg_frac=" << stats.bg_cell_frac
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
        if (params.count("player_metadata_key")) r->player_metadata_key_ = params["player_metadata_key"].get<std::string>();
        if (params.count("ball_metadata_key")) r->ball_metadata_key_ = params["ball_metadata_key"].get<std::string>();
        if (params.count("mask_margin_frac")) r->mask_margin_frac_ = params["mask_margin_frac"].get<float>();
        return r;
    }
};

DECLNODE(cuda_camera_motion, CudaCameraMotion)
