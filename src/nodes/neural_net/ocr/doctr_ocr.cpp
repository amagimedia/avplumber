#include "../../node_common.hpp"
#include "../common/infer_trt_base.hpp"
#include "doctr_recognizer.hpp"
#include "ocr_trt_runner.hpp"

extern "C" {
#include <libavutil/dict.h>
#include <libavutil/hwcontext.h>
#include <libavutil/hwcontext_cuda.h>
}

#include "../../../../objs/src/nodes/neural_net/preprocess/nv12_doctr_preprocess.ptx.h"

#include <NvInfer.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <deque>
#include <numeric>
#include <string>
#include <vector>

namespace {

constexpr int kDetBatch = 4;
constexpr int kDetH = 200;
constexpr int kDetW = 768;

struct Region {
    int x = 0, y = 0, w = 0, h = 0;
    bool drop_left = false;
    bool drop_right = false;
};

struct TextBox {
    int region = 0;
    float x1n = 0, y1n = 0, x2n = 0, y2n = 0;
    float x1 = 0, y1 = 0, x2 = 0, y2 = 0;
    float det_conf = 0;
    std::string text;
    float reco_conf = 0;
};

float sigmoid(float v) {
    if (v >= 0.0f) {
        float z = std::exp(-v);
        return 1.0f / (1.0f + z);
    }
    float z = std::exp(v);
    return z / (1.0f + z);
}

double parseFps(const std::string& fps) {
    if (fps.empty()) return 25.0;
    size_t slash = fps.find('/');
    try {
        if (slash == std::string::npos) return std::stod(fps);
        double num = std::stod(fps.substr(0, slash));
        double den = std::stod(fps.substr(slash + 1));
        return den > 0.0 ? num / den : num;
    } catch (...) {
        return 25.0;
    }
}

std::vector<Region> buildRegions(int W, int H, int region_height, const std::vector<float>& region_x,
                                  int max_regions, float region_y = -1.0f) {
    std::vector<Region> out;
    int w = std::max(1, (int)std::floor((double)W / 2.5));
    int h = std::max(1, std::min(region_height, H));
    // Vertical anchor: region_y in [0,1] is the normalized top edge of the scan
    // band. A negative value (default/unset) preserves the original bottom-anchored
    // behavior (y = H - h), so existing configs are unaffected.
    int y;
    if (region_y >= 0.0f) {
        y = std::max(0, std::min(H - h, (int)std::lround((double)region_y * H)));
    } else {
        y = std::max(0, H - h);
    }
    // Horizontal range from region_x [x_start_rel, x_end_rel], default full width.
    int x_start = 0, x_end = W;
    if (region_x.size() == 2) {
        x_start = std::max(0, std::min(W, (int)std::floor(region_x[0] * W)));
        x_end   = std::max(x_start + 1, std::min(W, (int)std::ceil(region_x[1] * W)));
    }
    int span = x_end - x_start;
    int step = std::max(1, w / 2);
    std::vector<int> xs;
    for (int x = x_start; x < std::max(x_start + 1, x_end - w + 1); x += step) xs.push_back(x);
    if (xs.empty() || xs.back() != x_end - w) xs.push_back(std::max(x_start, x_end - w));
    xs.erase(std::unique(xs.begin(), xs.end()), xs.end());
    if ((int)xs.size() > max_regions) xs.resize((size_t)max_regions);
    for (size_t i = 0; i < xs.size(); ++i) {
        Region r;
        r.x = xs[i];
        r.y = y;
        r.w = std::min(w, x_end - r.x);
        r.h = h;
        r.drop_left = r.x > x_start;
        r.drop_right = r.x + r.w < x_end;
        out.push_back(r);
    }
    while ((int)out.size() < kDetBatch) {
        Region r;
        r.x = x_start; r.y = y; r.w = 1; r.h = 1;
        out.push_back(r);
    }
    return out;
}

int autoRegionHeight(int W, int H) {
    // The detector input is 768x200. Region width is frame_width/2.5, so choose
    // a source crop height that preserves that aspect ratio before resize.
    int region_w = std::max(1, (int)std::floor((double)W / 2.5));
    int h = (int)std::lround((double)region_w * (double)kDetH / (double)kDetW);
    return std::max(1, std::min(h, H));
}

std::vector<uint8_t> morphOpen3x3(const std::vector<uint8_t>& src, int W, int H) {
    std::vector<uint8_t> er(src.size(), 0), out(src.size(), 0);
    for (int y = 1; y < H - 1; ++y) {
        for (int x = 1; x < W - 1; ++x) {
            bool ok = true;
            for (int yy = -1; yy <= 1 && ok; ++yy)
                for (int xx = -1; xx <= 1; ++xx)
                    if (!src[(y + yy) * W + (x + xx)]) { ok = false; break; }
            er[y * W + x] = ok ? 1 : 0;
        }
    }
    for (int y = 1; y < H - 1; ++y) {
        for (int x = 1; x < W - 1; ++x) {
            bool ok = false;
            for (int yy = -1; yy <= 1 && !ok; ++yy)
                for (int xx = -1; xx <= 1; ++xx)
                    if (er[(y + yy) * W + (x + xx)]) { ok = true; break; }
            out[y * W + x] = ok ? 1 : 0;
        }
    }
    return out;
}

std::vector<TextBox> boxesFromDetector(const std::vector<float>& logits, const std::vector<Region>& regions,
                                       float bin_thresh, float box_thresh, float edge_margin, int edge_margin_px,
                                       int max_boxes) {
    std::vector<TextBox> boxes;
    const int area = kDetH * kDetW;
    for (int b = 0; b < kDetBatch; ++b) {
        std::vector<float> prob(area);
        for (int y = 0; y < kDetH; ++y) {
            for (int x = 0; x < kDetW; ++x) {
                float m = -1e9f;
                for (int yy = std::max(0, y - 1); yy <= std::min(kDetH - 1, y + 1); ++yy)
                    for (int xx = std::max(0, x - 1); xx <= std::min(kDetW - 1, x + 1); ++xx)
                        m = std::max(m, logits[(size_t)b * area + yy * kDetW + xx]);
                prob[y * kDetW + x] = sigmoid(m);
            }
        }
        std::vector<uint8_t> bin(area, 0);
        for (int i = 0; i < area; ++i) bin[i] = prob[i] >= bin_thresh ? 1 : 0;
        bin = morphOpen3x3(bin, kDetW, kDetH);

        std::vector<uint8_t> seen(area, 0);
        std::deque<int> q;
        for (int i = 0; i < area; ++i) {
            if (!bin[i] || seen[i]) continue;
            int minx = kDetW, miny = kDetH, maxx = 0, maxy = 0, count = 0;
            q.clear(); q.push_back(i); seen[i] = 1;
            while (!q.empty()) {
                int cur = q.front(); q.pop_front();
                int x = cur % kDetW, y = cur / kDetW;
                minx = std::min(minx, x); maxx = std::max(maxx, x);
                miny = std::min(miny, y); maxy = std::max(maxy, y);
                ++count;
                const int nx[4] = {x - 1, x + 1, x, x};
                const int ny[4] = {y, y, y - 1, y + 1};
                for (int k = 0; k < 4; ++k) {
                    if (nx[k] < 0 || nx[k] >= kDetW || ny[k] < 0 || ny[k] >= kDetH) continue;
                    int ni = ny[k] * kDetW + nx[k];
                    if (bin[ni] && !seen[ni]) { seen[ni] = 1; q.push_back(ni); }
                }
            }
            if ((maxx - minx) < 2 || (maxy - miny) < 2 || count < 3) continue;
            float score = 0.0f;
            int n = 0;
            for (int y = miny; y <= maxy; ++y) {
                for (int x = minx; x <= maxx; ++x) { score += prob[y * kDetW + x]; ++n; }
            }
            score = n > 0 ? score / (float)n : 0.0f;
            if (score < box_thresh) continue;

            float bw = (float)(maxx - minx + 1), bh = (float)(maxy - miny + 1);
            float expand = std::max(0.0f, (bw * bh) / std::max(1.0f, 2.0f * bw + 2.0f * bh));
            float x1n = std::max(0.0f, ((float)minx - expand) / (float)kDetW);
            float y1n = std::max(0.0f, ((float)miny - expand) / (float)kDetH);
            float x2n = std::min(1.0f, ((float)maxx + 1.0f + expand) / (float)kDetW);
            float y2n = std::min(1.0f, ((float)maxy + 1.0f + expand) / (float)kDetH);
            const Region& r = regions[(size_t)b];
            float norm_bw = x2n - x1n;
            float norm_bh = y2n - y1n;
            float pad_x = edge_margin * norm_bw + (float)edge_margin_px / std::max(1.0f, (float)r.w);
            float pad_y = edge_margin * norm_bh + (float)edge_margin_px / std::max(1.0f, (float)r.h);
            x1n = std::max(0.0f, x1n - pad_x);
            y1n = std::max(0.0f, y1n - pad_y);
            x2n = std::min(1.0f, x2n + pad_x);
            y2n = std::min(1.0f, y2n + pad_y);
            TextBox tb;
            tb.region = b;
            tb.x1n = x1n; tb.y1n = y1n; tb.x2n = x2n; tb.y2n = y2n; tb.det_conf = score;
            tb.x1 = r.x + x1n * r.w; tb.x2 = r.x + x2n * r.w;
            tb.y1 = r.y + y1n * r.h; tb.y2 = r.y + y2n * r.h;
            boxes.push_back(tb);
        }
    }
    std::sort(boxes.begin(), boxes.end(), [](const TextBox& a, const TextBox& b) { return a.det_conf > b.det_conf; });
    if ((int)boxes.size() > max_boxes) boxes.resize((size_t)max_boxes);
    return boxes;
}

} // namespace

class DoctrOcr : public NodeSISO<av::VideoFrame, av::VideoFrame>, public ReportsFinishByFlag {
    std::string detector_engine_;
    std::string recognizer_engine_;
    std::string metadata_key_ = "text_detections";
    float target_fps_ = 5.0f;
    double input_fps_ = 25.0;
    int sample_every_n_ = 5;
    int region_height_ = 0;
    std::vector<float> region_x_;
    float region_y_ = -1.0f;
    int max_regions_ = 4;
    int max_boxes_ = 64;
    float det_bin_thresh_ = 0.1f;
    float det_box_thresh_ = 0.1f;
    float reco_conf_thresh_ = 0.0f;
    float edge_margin_ = 0.0f;
    int edge_margin_px_ = 2;
    int debug_log_every_n_ = 0;
    uint64_t frame_counter_ = 0;
    uint64_t sample_counter_ = 0;
    AVCUDADeviceContext* cuda_dev_ctx_ = nullptr;
    CUcontext cu_ctx_ = nullptr;
    CUmodule preprocess_module_ = nullptr;
    CUfunction preprocess_kernel_ = nullptr;
    CUdeviceptr d_boxes_ = 0;
    bool initialized_ = false;
    std::string node_label_ = "<unnamed>";
    std::shared_ptr<HWAccelDevice> debug_hwaccel_;
    bool preinitialized_from_hwaccel_ = false;
    bool frame_context_checked_ = false;
    bool last_context_reinit_ = false;
    bool last_context_match_ = false;
    int64_t last_context_init_ms_ = 0;
    ocr::TrtLogger logger_;
    ocr::TrtRunner det_;
    ocr::DoctrRecognizer recognizer_;

    bool frameCudaContext(const av::VideoFrame& frm, CUcontext& ctx, AVCUDADeviceContext** dev_ctx = nullptr) const {
        ctx = nullptr;
        if (dev_ctx) *dev_ctx = nullptr;
        if (!frm.raw() || !frm.raw()->hw_frames_ctx || !frm.raw()->hw_frames_ctx->data) {
            return false;
        }
        AVHWFramesContext* fctx = (AVHWFramesContext*)frm.raw()->hw_frames_ctx->data;
        if (!fctx || !fctx->device_ctx || !fctx->device_ctx->hwctx) return false;
        AVCUDADeviceContext* cuda_dev_ctx = (AVCUDADeviceContext*)fctx->device_ctx->hwctx;
        if (!cuda_dev_ctx || !cuda_dev_ctx->cuda_ctx) return false;
        ctx = cuda_dev_ctx->cuda_ctx;
        if (dev_ctx) *dev_ctx = cuda_dev_ctx;
        return true;
    }

    bool initContextFromHWAccel() {
        if (!debug_hwaccel_ || !debug_hwaccel_->deviceContext() || !debug_hwaccel_->deviceContext()->data) return false;
        AVHWDeviceContext* devctx = (AVHWDeviceContext*)debug_hwaccel_->deviceContext()->data;
        if (!devctx || devctx->type != AV_HWDEVICE_TYPE_CUDA || !devctx->hwctx) return false;
        cuda_dev_ctx_ = (AVCUDADeviceContext*)devctx->hwctx;
        if (!cuda_dev_ctx_ || !cuda_dev_ctx_->cuda_ctx) return false;
        cu_ctx_ = cuda_dev_ctx_->cuda_ctx;
        if (CUDA_CHECK_CU(cuCtxSetCurrent(cu_ctx_))) return false;
        return true;
    }

    void cleanupContextBoundState() {
        if (cu_ctx_) CUDA_CHECK_CU(cuCtxSetCurrent(cu_ctx_));
        det_.cleanup();
        recognizer_.cleanup();
        if (d_boxes_) { CUDA_CHECK_CU(cuMemFree(d_boxes_)); d_boxes_ = 0; }
        if (preprocess_module_) { CUDA_CHECK_CU(cuModuleUnload(preprocess_module_)); preprocess_module_ = nullptr; }
        cuda_dev_ctx_ = nullptr;
        cu_ctx_ = nullptr;
        initialized_ = false;
        preinitialized_from_hwaccel_ = false;
        frame_context_checked_ = false;
    }

    bool initInCurrentContext() {
        if (!cu_ctx_) return false;
        if (CUDA_CHECK_CU(cuCtxSetCurrent(cu_ctx_))) return false;
        const std::string ptx(avpl_doctr_preprocess_ptx, avpl_doctr_preprocess_ptx + avpl_doctr_preprocess_ptx_len);
        if (CUDA_CHECK_CU(cuModuleLoadDataEx(&preprocess_module_, ptx.c_str(), 0, nullptr, nullptr))) return false;
        if (CUDA_CHECK_CU(cuModuleGetFunction(&preprocess_kernel_, preprocess_module_, "kNV12_doctr_crop_resize_pad_f32"))) return false;
        det_.init(logger_, detector_engine_,
                  ocr::TensorContract{"", nvinfer1::DataType::kFLOAT, {kDetBatch, 3, kDetH, kDetW}},
                  ocr::TensorContract{"", nvinfer1::DataType::kFLOAT, {}});
        recognizer_.init(logger_, preprocess_module_, recognizer_engine_, max_boxes_);
        if (CUDA_CHECK_CU(cuMemAlloc(&d_boxes_, (size_t)kDetBatch * 4 * sizeof(int)))) return false;
        initialized_ = true;
        return true;
    }

    bool preinitFromHWAccel() {
        auto init_start = std::chrono::steady_clock::now();
        bool ok = initContextFromHWAccel() && initInCurrentContext();
        auto init_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - init_start).count();
        if (!ok) {
            cleanupContextBoundState();
        } else {
            preinitialized_from_hwaccel_ = true;
            frame_context_checked_ = false;
        }
        logstream << "neural_preinit"
                  << " status=" << (ok ? "ok" : "error")
                  << " type=doctr_ocr"
                  << " node=" << node_label_
                  << " preinit=1"
                  << " init_ms=" << init_ms
                  << " cuda_ctx=" << (const void*)cu_ctx_;
        return ok;
    }

    bool initFromFrame(const av::VideoFrame& frm) {
        last_context_reinit_ = false;
        last_context_match_ = false;
        last_context_init_ms_ = 0;
        if (initialized_) {
            if (preinitialized_from_hwaccel_ && !frame_context_checked_) {
                CUcontext frame_ctx = nullptr;
                AVCUDADeviceContext* frame_dev_ctx = nullptr;
                bool have_frame_ctx = frameCudaContext(frm, frame_ctx, &frame_dev_ctx);
                last_context_match_ = have_frame_ctx && frame_ctx == cu_ctx_;
                if (last_context_match_) {
                    frame_context_checked_ = true;
                    logstream << "neural_context_check"
                              << " status=ok type=doctr_ocr"
                              << " node=" << node_label_
                              << " preinit=1 match=1 reinit=0"
                              << " frame_cuda_ctx=" << (const void*)frame_ctx
                              << " init_cuda_ctx=" << (const void*)cu_ctx_;
                    return true;
                }

                auto init_start = std::chrono::steady_clock::now();
                cleanupContextBoundState();
                bool ok = false;
                if (have_frame_ctx) {
                    cuda_dev_ctx_ = frame_dev_ctx;
                    cu_ctx_ = frame_ctx;
                    ok = initInCurrentContext();
                }
                last_context_init_ms_ = std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::steady_clock::now() - init_start).count();
                last_context_reinit_ = true;
                frame_context_checked_ = ok;
                logstream << "neural_context_check"
                          << " status=" << (ok ? "ok" : "error")
                          << " type=doctr_ocr"
                          << " node=" << node_label_
                          << " preinit=1 match=0 reinit=1"
                          << " reinit_ms=" << last_context_init_ms_
                          << " frame_cuda_ctx=" << (const void*)frame_ctx
                          << " init_cuda_ctx=" << (const void*)cu_ctx_;
                return ok;
            }
            return true;
        }
        CUcontext frame_ctx = nullptr;
        AVCUDADeviceContext* frame_dev_ctx = nullptr;
        if (!frameCudaContext(frm, frame_ctx, &frame_dev_ctx)) {
            logstream << "doctr_ocr: missing CUDA hw frame context";
            return false;
        }
        cuda_dev_ctx_ = frame_dev_ctx;
        cu_ctx_ = frame_ctx;
        if (CUDA_CHECK_CU(cuCtxSetCurrent(cu_ctx_))) return false;
        yolo_base::logCudaContextPointers("doctr_ocr", node_label_, frm, cu_ctx_, debug_hwaccel_);
        return initInCurrentContext();
    }

    void launchPreprocess(const av::VideoFrame& frm, ocr::TrtRunner& runner, const std::vector<int>& boxes,
                          float mr, float mg, float mb, float sr, float sg, float sb) {
        CUDA_CHECK_CU(cuMemcpyHtoDAsync(d_boxes_, boxes.data(), boxes.size() * sizeof(int), runner.stream()));
        CUdeviceptr dY = (CUdeviceptr)(uintptr_t)frm.raw()->data[0];
        CUdeviceptr dUV = (CUdeviceptr)(uintptr_t)frm.raw()->data[1];
        int pitchY = frm.raw()->linesize[0];
        int pitchUV = frm.raw()->linesize[1];
        CUdeviceptr out = runner.inputPtr();
        int batch = runner.inputDim(0), h = runner.inputDim(2), w = runner.inputDim(3);
        void* args[] = {
            &dY, &pitchY, &dUV, &pitchUV, &out, &d_boxes_, &batch, &h, &w,
            &mr, &mg, &mb, &sr, &sg, &sb
        };
        unsigned int bx = 16, by = 16;
        unsigned int gx = (unsigned int)((w + (int)bx - 1) / (int)bx);
        unsigned int gy = (unsigned int)((h + (int)by - 1) / (int)by);
        CUDA_CHECK_CU(cuLaunchKernel(preprocess_kernel_, gx, gy, (unsigned int)batch, bx, by, 1, 0, runner.stream(), args, nullptr));
    }

    Parameters emptyPayload(bool sampled, const std::string& reason) {
        Parameters out;
        out["schema"] = "doctr_ocr_v1";
        out["ocr_sampled"] = sampled;
        out["detections"] = Parameters::array();
        out["reason"] = reason;
        return out;
    }

public:
    using NodeSISO<av::VideoFrame, av::VideoFrame>::NodeSISO;

    ~DoctrOcr() override {
        cleanupContextBoundState();
    }

    void process() override {
        auto frm = this->source_->get();
        if (!frm) return;
        ++frame_counter_;
        bool sampled = ((frame_counter_ - 1) % (uint64_t)std::max(1, sample_every_n_)) == 0;
        if (!sampled) {
            Parameters out = emptyPayload(false, "not_sampled");
            av_dict_set(&frm.raw()->metadata, metadata_key_.c_str(), out.dump().c_str(), 0);
            this->sink_->put(frm);
            return;
        }

        auto t0 = std::chrono::steady_clock::now();
        int64_t process_init_ms = 0;
        bool process_reinit = false;
        bool process_context_match = false;
        bool process_preinit = preinitialized_from_hwaccel_;
        if (!initialized_) {
            auto init_start = std::chrono::steady_clock::now();
            bool ok = initFromFrame(frm);
            auto init_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - init_start).count();
            logstream << "neural_node_init"
                      << " status=" << (ok ? "ok" : "error")
                      << " type=doctr_ocr"
                      << " node=" << node_label_
                      << " init_ms=" << init_ms
                      << " frame_wait_ms=" << init_ms
                      << " frame=" << frame_counter_
                      << " detector=\"" << detector_engine_ << "\""
                      << " recognizer=\"" << recognizer_engine_ << "\"";
            if (!ok) {
                Parameters out = emptyPayload(true, "init_failed");
                av_dict_set(&frm.raw()->metadata, metadata_key_.c_str(), out.dump().c_str(), 0);
                this->sink_->put(frm);
                return;
            }
            process_init_ms = init_ms;
        } else if (!initFromFrame(frm)) {
            Parameters out = emptyPayload(true, "init_failed");
            av_dict_set(&frm.raw()->metadata, metadata_key_.c_str(), out.dump().c_str(), 0);
            this->sink_->put(frm);
            return;
        } else {
            process_init_ms = last_context_init_ms_;
            process_reinit = last_context_reinit_;
            process_context_match = last_context_match_;
        }
        if (cu_ctx_) cuCtxSetCurrent(cu_ctx_);
        const int W = frm.raw()->width;
        const int H = frm.raw()->height;
        int effective_region_height = region_height_ > 0 ? region_height_ : autoRegionHeight(W, H);
        std::vector<Region> regions = buildRegions(W, H, effective_region_height, region_x_,
                                                     std::min(max_regions_, kDetBatch), region_y_);
        std::vector<int> det_boxes;
        det_boxes.reserve((size_t)kDetBatch * 4);
        for (int i = 0; i < kDetBatch; ++i) {
            det_boxes.insert(det_boxes.end(), {regions[(size_t)i].x, regions[(size_t)i].y, regions[(size_t)i].w, regions[(size_t)i].h});
        }
        launchPreprocess(frm, det_, det_boxes, 0.798f, 0.785f, 0.772f, 0.264f, 0.2749f, 0.287f);
        det_.infer();
        std::vector<TextBox> boxes = boxesFromDetector(
            det_.output(), regions, det_bin_thresh_, det_box_thresh_, edge_margin_, edge_margin_px_,
            recognizer_.batchSize());

        int raw_kept = (int)boxes.size();
        if (!boxes.empty()) {
            std::vector<ocr::PixelBox> rec_boxes;
            rec_boxes.reserve(boxes.size());
            for (size_t i = 0; i < boxes.size(); ++i) {
                int x1 = std::max(0, (int)std::floor(boxes[i].x1));
                int y1 = std::max(0, (int)std::floor(boxes[i].y1));
                int x2 = std::min(W, (int)std::ceil(boxes[i].x2));
                int y2 = std::min(H, (int)std::ceil(boxes[i].y2));
                rec_boxes.push_back({x1, y1, std::max(x1 + 1, x2), std::max(y1 + 1, y2)});
            }
            const std::vector<ocr::Recognition> recognized = recognizer_.recognize(frm, rec_boxes);
            for (size_t i = 0; i < boxes.size(); ++i) {
                boxes[i].text = recognized[i].text;
                boxes[i].reco_conf = recognized[i].confidence;
            }
        }

        Parameters out;
        out["schema"] = "doctr_ocr_v1";
        out["ocr_sampled"] = true;
        out["frame_width"] = frm.raw()->width;
        out["frame_height"] = frm.raw()->height;
        out["regions"] = Parameters::array();
        for (int i = 0; i < std::min((int)regions.size(), kDetBatch); ++i) {
            Parameters r;
            r["id"] = i;
            r["bbox"] = {regions[(size_t)i].x, regions[(size_t)i].y, regions[(size_t)i].w, regions[(size_t)i].h};
            r["drop_left"] = regions[(size_t)i].drop_left;
            r["drop_right"] = regions[(size_t)i].drop_right;
            out["regions"].push_back(r);
        }
        out["detections"] = Parameters::array();
        int recognized = 0;
        for (const TextBox& b : boxes) {
            if (b.text.empty() || b.reco_conf < reco_conf_thresh_) continue;
            ++recognized;
            Parameters d;
            d["text"] = b.text;
            d["bbox"] = {b.x1, b.y1, b.x2, b.y2};
            d["det_conf"] = b.det_conf;
            d["reco_conf"] = b.reco_conf;
            d["region_id"] = b.region;
            out["detections"].push_back(d);
        }
        auto t1 = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        out["stats"] = {
            {"raw_boxes", raw_kept},
            {"recognized", recognized},
            {"max_boxes", recognizer_.batchSize()},
            {"sample_every_n", sample_every_n_},
            {"target_fps", target_fps_},
            {"elapsed_ms", ms}
        };
        std::string json = out.dump();
        av_dict_set(&frm.raw()->metadata, metadata_key_.c_str(), json.c_str(), 0);
        ++sample_counter_;
        if (debug_log_every_n_ > 0 && (sample_counter_ % (uint64_t)debug_log_every_n_) == 0) {
            std::string texts;
            for (const TextBox& b : boxes) {
                if (b.text.empty()) continue;
                if (!texts.empty()) texts += "|";
                texts += b.text;
            }
            logstream << "doctr_ocr: frame=" << frame_counter_
                      << " sampled=true regions=" << kDetBatch
                      << " det_kept=" << raw_kept
                      << " recognized=" << recognized
                      << " texts=[" << texts << "]"
                      << " elapsed_ms=" << ms;
        }
        if (sample_counter_ == 1 || process_init_ms > 0 || process_reinit) {
            logstream << "neural_process"
                      << " status=ok"
                      << " type=doctr_ocr"
                      << " node=" << node_label_
                      << " frame=" << frame_counter_
                      << " preinit=" << (process_preinit ? 1 : 0)
                      << " context_match=" << (process_context_match ? 1 : 0)
                      << " reinit=" << (process_reinit ? 1 : 0)
                      << " init_ms=" << process_init_ms
                      << " process_ms=" << (int64_t)ms;
        }
        this->sink_->put(frm);
    }

    static std::shared_ptr<DoctrOcr> create(NodeCreationInfo& nci) {
        const Parameters& p = nci.params;
        auto r = NodeSISO<av::VideoFrame, av::VideoFrame>::template createCommon<DoctrOcr>(nci.edges, p);
        if (!p.count("detector_engine")) throw Error("doctr_ocr: detector_engine param required");
        if (!p.count("recognizer_engine")) throw Error("doctr_ocr: recognizer_engine param required");
        r->node_label_ = p.value("name", std::string("<unnamed>"));
        if (p.count("hwaccel")) {
            r->debug_hwaccel_ = InstanceSharedObjects<HWAccelDevice>::get(nci.instance, p["hwaccel"]);
        }
        r->detector_engine_ = p["detector_engine"].get<std::string>();
        r->recognizer_engine_ = p["recognizer_engine"].get<std::string>();
        if (p.count("metadata_key")) r->metadata_key_ = p["metadata_key"].get<std::string>();
        if (p.count("target_fps")) r->target_fps_ = std::max(0.1f, p["target_fps"].get<float>());
        if (p.count("input_fps")) r->input_fps_ = parseFps(p["input_fps"].get<std::string>());
        r->sample_every_n_ = std::max(1, (int)std::lround(r->input_fps_ / std::max(0.1f, r->target_fps_)));
        if (p.count("region_height")) r->region_height_ = std::max(0, p["region_height"].get<int>());
        if (p.count("region_x") && p["region_x"].is_array() && p["region_x"].size() == 2) {
            r->region_x_.clear();
            for (auto& v : p["region_x"]) r->region_x_.push_back(v.get<float>());
        }
        if (p.count("region_y")) r->region_y_ = p["region_y"].get<float>();
        if (p.count("max_regions")) r->max_regions_ = std::max(1, std::min(kDetBatch, p["max_regions"].get<int>()));
        if (p.count("max_boxes")) r->max_boxes_ = std::max(1, p["max_boxes"].get<int>());
        if (p.count("det_bin_thresh")) r->det_bin_thresh_ = p["det_bin_thresh"].get<float>();
        if (p.count("det_box_thresh")) r->det_box_thresh_ = p["det_box_thresh"].get<float>();
        if (p.count("reco_conf_thresh")) r->reco_conf_thresh_ = p["reco_conf_thresh"].get<float>();
        if (p.count("edge_margin")) r->edge_margin_ = std::clamp(p["edge_margin"].get<float>(), 0.0f, 1.0f);
        if (p.count("edge_margin_px")) r->edge_margin_px_ = std::max(0, p["edge_margin_px"].get<int>());
        if (p.count("debug_log_every_n")) r->debug_log_every_n_ = std::max(0, p["debug_log_every_n"].get<int>());
        if (r->debug_hwaccel_) {
            r->preinitFromHWAccel();
        }
        return r;
    }
};

DECLNODE(doctr_ocr, DoctrOcr)
