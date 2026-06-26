#include "../../node_common.hpp"
#include "../common/infer_trt_base.hpp"

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
#include <fstream>
#include <numeric>
#include <string>
#include <vector>

namespace {

constexpr int kDetBatch = 4;
constexpr int kDetH = 200;
constexpr int kDetW = 768;
constexpr int kRecBatch = 24;
constexpr int kRecH = 32;
constexpr int kRecW = 128;

// doctr's VOCABS["french"], verbatim and EXACTLY 126 codepoints. This is the vocab
// the recognizer ONNX (PARSeq) was trained/exported with: its inference head emits
// 127 classes = these 126 chars (indices 0..125) plus <eos> at index 126. The vocab
// must match byte-for-byte or the tail indices (currency/accents) decode to the wrong
// glyph and the <eos> stop index drifts. NOTE: doctr's french has NO space between the
// "~" punctuation block and the "°" currency block — an extra space here shifts every
// later index by one and makes <eos> unreachable (the whole string decodes as garbage
// after the real word). Keep this in sync with doctr.datasets.VOCABS["french"].
const std::string kFrenchVocab =
    "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    R"(!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~)"
    "°£€¥¢฿"
    "àâéèêëîïôùûüçÀÂÉÈÊËÎÏÔÙÛÜÇ";

std::vector<std::string> splitUtf8Codepoints(const std::string& text) {
    std::vector<std::string> out;
    for (size_t i = 0; i < text.size();) {
        unsigned char c = (unsigned char)text[i];
        size_t len = 1;
        if ((c & 0x80) == 0x00) len = 1;
        else if ((c & 0xE0) == 0xC0) len = 2;
        else if ((c & 0xF0) == 0xE0) len = 3;
        else if ((c & 0xF8) == 0xF0) len = 4;
        if (i + len > text.size()) break;
        out.push_back(text.substr(i, len));
        i += len;
    }
    return out;
}

const std::vector<std::string> kFrenchTokens = splitUtf8Codepoints(kFrenchVocab);

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

std::vector<Region> buildRegions(int W, int H, int region_height, int max_regions) {
    std::vector<Region> out;
    int w = std::max(1, (int)std::floor((double)W / 2.5));
    int h = std::max(1, std::min(region_height, H));
    int y = std::max(0, H - h);
    int step = std::max(1, w / 2);
    std::vector<int> xs;
    for (int x = 0; x < std::max(1, W - w + 1); x += step) xs.push_back(x);
    if (xs.empty() || xs.back() != W - w) xs.push_back(std::max(0, W - w));
    xs.erase(std::unique(xs.begin(), xs.end()), xs.end());
    if ((int)xs.size() > max_regions) xs.resize((size_t)max_regions);
    for (size_t i = 0; i < xs.size(); ++i) {
        Region r;
        r.x = xs[i];
        r.y = y;
        r.w = std::min(w, W - r.x);
        r.h = h;
        r.drop_left = r.x > 0;
        r.drop_right = r.x + r.w < W;
        out.push_back(r);
    }
    while ((int)out.size() < kDetBatch) {
        Region r;
        r.x = 0; r.y = y; r.w = 1; r.h = 1;
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

class DoctrLogger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) logstream << "doctr_ocr tensorrt: " << (msg ? msg : "");
    }
};

struct TrtRunner {
    std::string path;
    std::string input_name;
    std::vector<std::string> tensor_names;
    std::vector<CUdeviceptr> ptrs;
    std::vector<size_t> bytes;
    std::vector<float> output;
    int input_index = -1;
    int output_index = -1;
    int n = 0, c = 0, h = 0, w = 0;
    nvinfer1::IRuntime* runtime = nullptr;
    nvinfer1::ICudaEngine* engine = nullptr;
    nvinfer1::IExecutionContext* ctx = nullptr;
    CUstream stream = nullptr;

    void cleanup() {
        if (stream) { CUDA_CHECK_CU(cuStreamSynchronize(stream)); CUDA_CHECK_CU(cuStreamDestroy(stream)); stream = nullptr; }
        for (CUdeviceptr p : ptrs) if (p) CUDA_CHECK_CU(cuMemFree(p));
        ptrs.clear(); bytes.clear(); tensor_names.clear(); output.clear();
        if (ctx) { delete ctx; ctx = nullptr; }
        if (engine) { delete engine; engine = nullptr; }
        if (runtime) { delete runtime; runtime = nullptr; }
    }

    bool init(DoctrLogger& logger, int want_n, int want_c, int want_h, int want_w) {
        std::ifstream f(path, std::ios::binary);
        if (!f) { logstream << "doctr_ocr: cannot open engine " << path; return false; }
        f.seekg(0, std::ios::end);
        std::streamsize sz = f.tellg();
        f.seekg(0, std::ios::beg);
        std::vector<char> blob((size_t)sz);
        if (sz <= 0 || !f.read(blob.data(), sz)) { logstream << "doctr_ocr: failed reading engine " << path; return false; }
        runtime = nvinfer1::createInferRuntime(logger);
        if (!runtime) return false;
        engine = runtime->deserializeCudaEngine(blob.data(), blob.size());
        if (!engine) return false;
        ctx = engine->createExecutionContext();
        if (!ctx) return false;
        if (CUDA_CHECK_CU(cuStreamCreate(&stream, 0))) return false;

        int nb = engine->getNbIOTensors();
        ptrs.assign((size_t)nb, 0);
        bytes.assign((size_t)nb, 0);
        for (int i = 0; i < nb; ++i) {
            std::string name = engine->getIOTensorName(i);
            tensor_names.push_back(name);
            if (engine->getTensorIOMode(name.c_str()) == nvinfer1::TensorIOMode::kINPUT) {
                input_index = i;
                input_name = name;
                nvinfer1::Dims4 dims(want_n, want_c, want_h, want_w);
                ctx->setInputShape(name.c_str(), dims);
            }
        }
        for (int i = 0; i < nb; ++i) {
            const std::string& name = tensor_names[(size_t)i];
            nvinfer1::Dims dims = ctx->getTensorShape(name.c_str());
            if (dims.nbDims <= 0) dims = engine->getTensorShape(name.c_str());
            size_t vol = yolo_base::volume(dims);
            size_t esz = yolo_base::elementSize(engine->getTensorDataType(name.c_str()));
            if (vol == 0 || esz == 0) {
                logstream << "doctr_ocr: unsupported tensor shape for " << path << " tensor=" << name;
                return false;
            }
            bytes[(size_t)i] = vol * esz;
            if (CUDA_CHECK_CU(cuMemAlloc(&ptrs[(size_t)i], bytes[(size_t)i]))) return false;
            if (!ctx->setTensorAddress(name.c_str(), reinterpret_cast<void*>(ptrs[(size_t)i]))) return false;
            if (engine->getTensorIOMode(name.c_str()) == nvinfer1::TensorIOMode::kOUTPUT) {
                output_index = i;
                output.assign(vol, 0.0f);
            }
        }
        if (input_index < 0 || output_index < 0) return false;
        n = want_n; c = want_c; h = want_h; w = want_w;
        logstream << "doctr_ocr: loaded engine=" << path << " input=" << n << "x" << c << "x" << h << "x" << w;
        return true;
    }

    CUdeviceptr inputPtr() const { return ptrs[(size_t)input_index]; }

    bool infer() {
        if (!ctx->enqueueV3(reinterpret_cast<cudaStream_t>(stream))) {
            logstream << "doctr_ocr: enqueue failed for " << path;
            return false;
        }
        if (CUDA_CHECK_CU(cuMemcpyDtoHAsync(output.data(), ptrs[(size_t)output_index], bytes[(size_t)output_index], stream))) return false;
        if (CUDA_CHECK_CU(cuStreamSynchronize(stream))) return false;
        return true;
    }
};

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
                                       float bin_thresh, float box_thresh, int edge_margin_px, int max_boxes) {
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
            float margin = (float)edge_margin_px / std::max(1.0f, (float)r.w);
            if (r.drop_left && x1n <= margin) continue;
            if (r.drop_right && x2n >= 1.0f - margin) continue;
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

std::pair<std::string, float> decodeParseq(const float* logits, int steps, int classes) {
    std::string text;
    float conf_sum = 0.0f;
    int conf_n = 0;
    // PARSeq inference head emits |vocab| + 1 classes: vocab tokens at indices
    // [0, eos) and the <eos> stop token at index eos == |vocab|. Decoding stops at
    // the first <eos>; anything after it is padding the model is free to fill with
    // noise (this is what produced the repeated-glyph garbage tail when eos was
    // mis-set to an unreachable index). One-time guard: if the runtime class count
    // disagrees with the vocab, the vocab is out of sync with the engine.
    const int eos = (int)kFrenchTokens.size();
    if (classes != eos + 1) {
        static bool warned = false;
        if (!warned) {
            warned = true;
            logstream << "DoctrOcr: recognizer emits " << classes << " classes but vocab "
                      << "implies " << (eos + 1) << " (|vocab|=" << eos << " + <eos>); "
                      << "kFrenchVocab is out of sync with the model — decode will be wrong";
        }
    }
    for (int t = 0; t < steps; ++t) {
        const float* row = logits + t * classes;
        int best = 0;
        float maxv = row[0];
        for (int c = 1; c < classes; ++c) if (row[c] > maxv) { maxv = row[c]; best = c; }
        float denom = 0.0f;
        for (int c = 0; c < classes; ++c) denom += std::exp(row[c] - maxv);
        float prob = denom > 0.0f ? 1.0f / denom : 0.0f;
        if (best == eos) break;
        if (best >= 0 && best < eos) {
            text += kFrenchTokens[(size_t)best];
            conf_sum += prob;
            ++conf_n;
        }
    }
    return {text, conf_n ? conf_sum / (float)conf_n : 0.0f};
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
    int max_regions_ = 4;
    int max_boxes_ = 24;
    float det_bin_thresh_ = 0.1f;
    float det_box_thresh_ = 0.1f;
    float reco_conf_thresh_ = 0.0f;
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
    DoctrLogger logger_;
    TrtRunner det_;
    TrtRunner rec_;

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
        rec_.cleanup();
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
        if (CUDA_CHECK_CU(cuMemAlloc(&d_boxes_, (size_t)kRecBatch * 4 * sizeof(int)))) return false;
        det_.path = detector_engine_;
        rec_.path = recognizer_engine_;
        if (!det_.init(logger_, kDetBatch, 3, kDetH, kDetW)) return false;
        if (!rec_.init(logger_, kRecBatch, 3, kRecH, kRecW)) return false;
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

    void launchPreprocess(const av::VideoFrame& frm, TrtRunner& runner, const std::vector<int>& boxes,
                          float mr, float mg, float mb, float sr, float sg, float sb) {
        CUDA_CHECK_CU(cuMemcpyHtoDAsync(d_boxes_, boxes.data(), boxes.size() * sizeof(int), runner.stream));
        CUdeviceptr dY = (CUdeviceptr)(uintptr_t)frm.raw()->data[0];
        CUdeviceptr dUV = (CUdeviceptr)(uintptr_t)frm.raw()->data[1];
        int pitchY = frm.raw()->linesize[0];
        int pitchUV = frm.raw()->linesize[1];
        CUdeviceptr out = runner.inputPtr();
        int batch = runner.n, h = runner.h, w = runner.w;
        void* args[] = {
            &dY, &pitchY, &dUV, &pitchUV, &out, &d_boxes_, &batch, &h, &w,
            &mr, &mg, &mb, &sr, &sg, &sb
        };
        unsigned int bx = 16, by = 16;
        unsigned int gx = (unsigned int)((w + (int)bx - 1) / (int)bx);
        unsigned int gy = (unsigned int)((h + (int)by - 1) / (int)by);
        CUDA_CHECK_CU(cuLaunchKernel(preprocess_kernel_, gx, gy, (unsigned int)batch, bx, by, 1, 0, runner.stream, args, nullptr));
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
        std::vector<Region> regions = buildRegions(W, H, effective_region_height, std::min(max_regions_, kDetBatch));
        std::vector<int> det_boxes;
        det_boxes.reserve((size_t)kDetBatch * 4);
        for (int i = 0; i < kDetBatch; ++i) {
            det_boxes.insert(det_boxes.end(), {regions[(size_t)i].x, regions[(size_t)i].y, regions[(size_t)i].w, regions[(size_t)i].h});
        }
        launchPreprocess(frm, det_, det_boxes, 0.798f, 0.785f, 0.772f, 0.264f, 0.2749f, 0.287f);
        bool ok = det_.infer();
        std::vector<TextBox> boxes = ok
            ? boxesFromDetector(det_.output, regions, det_bin_thresh_, det_box_thresh_, edge_margin_px_, std::min(max_boxes_, kRecBatch))
            : std::vector<TextBox>();

        int raw_kept = (int)boxes.size();
        if (!boxes.empty()) {
            std::vector<int> rec_boxes((size_t)kRecBatch * 4, 0);
            for (size_t i = 0; i < boxes.size() && i < (size_t)kRecBatch; ++i) {
                int x1 = std::max(0, (int)std::floor(boxes[i].x1));
                int y1 = std::max(0, (int)std::floor(boxes[i].y1));
                int x2 = std::min(W, (int)std::ceil(boxes[i].x2));
                int y2 = std::min(H, (int)std::ceil(boxes[i].y2));
                rec_boxes[i * 4 + 0] = x1;
                rec_boxes[i * 4 + 1] = y1;
                rec_boxes[i * 4 + 2] = std::max(1, x2 - x1);
                rec_boxes[i * 4 + 3] = std::max(1, y2 - y1);
            }
            launchPreprocess(frm, rec_, rec_boxes, 0.694f, 0.695f, 0.693f, 0.299f, 0.296f, 0.301f);
            if (rec_.infer()) {
                const int steps = 33;
                const int classes = (rec_.n > 0 && steps > 0)
                    ? (int)(rec_.output.size() / ((size_t)rec_.n * (size_t)steps))
                    : ((int)kFrenchTokens.size() + 1);
                for (size_t i = 0; i < boxes.size(); ++i) {
                    auto decoded = decodeParseq(rec_.output.data() + i * steps * classes, steps, classes);
                    boxes[i].text = decoded.first;
                    boxes[i].reco_conf = decoded.second;
                }
            }
        }

        Parameters out;
        out["schema"] = "doctr_ocr_v1";
        out["ocr_sampled"] = true;
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
            {"max_boxes", std::min(max_boxes_, kRecBatch)},
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
        if (p.count("max_regions")) r->max_regions_ = std::max(1, std::min(kDetBatch, p["max_regions"].get<int>()));
        if (p.count("max_boxes")) r->max_boxes_ = std::max(1, std::min(kRecBatch, p["max_boxes"].get<int>()));
        if (p.count("det_bin_thresh")) r->det_bin_thresh_ = p["det_bin_thresh"].get<float>();
        if (p.count("det_box_thresh")) r->det_box_thresh_ = p["det_box_thresh"].get<float>();
        if (p.count("reco_conf_thresh")) r->reco_conf_thresh_ = p["reco_conf_thresh"].get<float>();
        if (p.count("edge_margin_px")) r->edge_margin_px_ = std::max(0, p["edge_margin_px"].get<int>());
        if (p.count("debug_log_every_n")) r->debug_log_every_n_ = std::max(0, p["debug_log_every_n"].get<int>());
        if (r->debug_hwaccel_) {
            r->preinitFromHWAccel();
        }
        return r;
    }
};

DECLNODE(doctr_ocr, DoctrOcr)
