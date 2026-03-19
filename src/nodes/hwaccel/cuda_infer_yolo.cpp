#include "../node_common.hpp"
#include "../../hwaccel.hpp"
#include <cuda_loader/cuda_drvapi_dynlink_cuda.h>

extern "C" {
#include <libavutil/hwcontext.h>
#include <libavutil/hwcontext_cuda.h>
#include <libavutil/dict.h>
}

#include <NvInfer.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <fstream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

// PTX blob for NV12->NCHW preprocess kernel.
#include "../../../objs/src/nodes/hwaccel/yolo_preprocess.ptx.h"

static int check_cu(CUresult err, const char *func) {
    if (err == CUDA_SUCCESS) return 0;
    const char *err_name = nullptr;
    const char *err_string = nullptr;
    if (cuGetErrorName && cuGetErrorString) {
        cuGetErrorName(err, &err_name);
        cuGetErrorString(err, &err_string);
    }
    logstream << "cuda function: " << func << " failed: " << (err_name ? err_name : "?") << ": " << (err_string ? err_string : "?");
    return -1;
}
#define CHECK_CU(x) check_cu((x), #x)

namespace {
class TRTLogger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        // keep warnings/errors to avoid noisy per-frame info logs
        if (severity == Severity::kERROR || severity == Severity::kINTERNAL_ERROR || severity == Severity::kWARNING) {
            logstream << "tensorrt: " << (msg ? msg : "");
        }
    }
};

struct Detection {
    float x1 = 0.0f;
    float y1 = 0.0f;
    float x2 = 0.0f;
    float y2 = 0.0f;
    float conf = 0.0f;
    int cls = -1;
};

static bool loadClassNamesFromFile(const std::string& path, std::vector<std::string>& out, std::string& err) {
    std::ifstream f(path);
    if (!f) {
        err = "cannot open class names file " + path;
        return false;
    }

    std::vector<std::string> names;
    std::string token;
    while (f >> token) {
        names.push_back(token);
    }
    if (f.bad()) {
        err = "failed reading class names file " + path;
        return false;
    }
    if (names.empty()) {
        err = "class names file is empty " + path;
        return false;
    }

    out = std::move(names);
    return true;
}

static float halfToFloat(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1Fu;
    uint32_t frac = h & 0x03FFu;
    uint32_t out = 0;
    if (exp == 0) {
        if (frac == 0) {
            out = sign;
        } else {
            // subnormal half -> normalized float
            exp = 1;
            while ((frac & 0x0400u) == 0) {
                frac <<= 1;
                --exp;
            }
            frac &= 0x03FFu;
            uint32_t exp32 = exp + (127 - 15);
            out = sign | (exp32 << 23) | (frac << 13);
        }
    } else if (exp == 0x1Fu) {
        out = sign | 0x7F800000u | (frac << 13); // inf/nan
    } else {
        uint32_t exp32 = exp + (127 - 15);
        out = sign | (exp32 << 23) | (frac << 13);
    }
    float f;
    memcpy(&f, &out, sizeof(float));
    return f;
}

static size_t elementSize(nvinfer1::DataType dt) {
    switch (dt) {
        case nvinfer1::DataType::kFLOAT: return 4;
        case nvinfer1::DataType::kHALF: return 2;
        case nvinfer1::DataType::kINT8: return 1;
        case nvinfer1::DataType::kINT32: return 4;
        case nvinfer1::DataType::kBOOL: return 1;
        default: return 0;
    }
}

static size_t volume(const nvinfer1::Dims& d) {
    size_t v = 1;
    for (int i = 0; i < d.nbDims; ++i) {
        if (d.d[i] <= 0) return 0;
        v *= (size_t)d.d[i];
    }
    return v;
}

static float iou(const Detection& a, const Detection& b) {
    const float x1 = std::max(a.x1, b.x1);
    const float y1 = std::max(a.y1, b.y1);
    const float x2 = std::min(a.x2, b.x2);
    const float y2 = std::min(a.y2, b.y2);
    const float w = std::max(0.0f, x2 - x1);
    const float h = std::max(0.0f, y2 - y1);
    const float inter = w * h;
    const float area_a = std::max(0.0f, a.x2 - a.x1) * std::max(0.0f, a.y2 - a.y1);
    const float area_b = std::max(0.0f, b.x2 - b.x1) * std::max(0.0f, b.y2 - b.y1);
    const float uni = area_a + area_b - inter;
    if (uni <= 0.0f) return 0.0f;
    return inter / uni;
}
}

class CudaInferYolo : public NodeSISO<av::VideoFrame, av::VideoFrame> {
protected:
    std::shared_ptr<HWAccelDevice> hwaccel_;
    AVCUDADeviceContext* cuda_dev_ctx_ = nullptr;
    CUcontext cu_ctx_ = nullptr;

    TRTLogger trt_logger_;
    nvinfer1::IRuntime* trt_runtime_ = nullptr;
    nvinfer1::ICudaEngine* trt_engine_ = nullptr;
    nvinfer1::IExecutionContext* trt_ctx_ = nullptr;

    std::vector<std::string> io_tensor_names_;
    std::vector<size_t> tensor_bytes_;
    std::vector<CUdeviceptr> tensor_ptrs_;
    std::unordered_map<std::string, size_t> tensor_index_;
    std::string input_tensor_name_;
    std::string output_tensor_name_; // first output only in v1
    nvinfer1::Dims input_dims_{};
    nvinfer1::Dims output_dims_{};
    int input_w_ = 0;
    int input_h_ = 0;
    bool input_bgr_order_ = false;
    nvinfer1::DataType input_dtype_ = nvinfer1::DataType::kFLOAT;
    nvinfer1::DataType output_dtype_ = nvinfer1::DataType::kFLOAT;

    CUmodule preprocess_module_ = nullptr;
    CUfunction preprocess_kernel_ = nullptr;

    std::vector<float> host_output_;
    std::vector<uint16_t> host_output_half_;

    std::string engine_path_;
    std::string metadata_key_out_ = "yolo_detections_v1";
    bool debug_log_metadata_ = false;
    int debug_log_every_n_ = 30;
    int infer_every_n_ = 1;
    float conf_thresh_ = 0.25f;
    float iou_thresh_ = 0.45f;
    int max_det_ = 300;
    uint64_t frame_counter_ = 0;
    std::vector<std::string> class_names_;

    bool initialized_ = false;

    bool initCudaContextFromFrame(const av::VideoFrame& frm) {
        if (cu_ctx_) return true;
        if (!frm.raw() || !frm.raw()->hw_frames_ctx || !frm.raw()->hw_frames_ctx->data) {
            logstream << "cuda_infer_yolo: missing hw_frames_ctx";
            return false;
        }
        AVHWFramesContext* fctx = (AVHWFramesContext*)frm.raw()->hw_frames_ctx->data;
        if (!fctx || !fctx->device_ctx || !fctx->device_ctx->hwctx) {
            logstream << "cuda_infer_yolo: missing device_ctx/hwctx in frame";
            return false;
        }
        cuda_dev_ctx_ = (AVCUDADeviceContext*)fctx->device_ctx->hwctx;
        if (!cuda_dev_ctx_ || !cuda_dev_ctx_->cuda_ctx) {
            logstream << "cuda_infer_yolo: missing cuda context in frame";
            return false;
        }
        cu_ctx_ = cuda_dev_ctx_->cuda_ctx;
        if (CHECK_CU(cuCtxSetCurrent(cu_ctx_))) {
            logstream << "cuda_infer_yolo: cuCtxSetCurrent failed";
            return false;
        }
        return true;
    }

    bool loadPreprocessKernel() {
        if (preprocess_module_ && preprocess_kernel_) return true;
        if (!cu_ctx_) return false;
        if (CHECK_CU(cuCtxSetCurrent(cu_ctx_))) return false;
        const std::string ptx_str(avpl_yolo_preprocess_ptx, avpl_yolo_preprocess_ptx + avpl_yolo_preprocess_ptx_len);
        if (CHECK_CU(cuModuleLoadDataEx(&preprocess_module_, (const void*)ptx_str.c_str(), 0, nullptr, nullptr))) {
            logstream << "cuda_infer_yolo: failed to load preprocess PTX module";
            return false;
        }
        const char* kname = (input_dtype_ == nvinfer1::DataType::kHALF) ? "kNV12_to_NCHW_fp16" : "kNV12_to_NCHW_fp32";
        if (CHECK_CU(cuModuleGetFunction(&preprocess_kernel_, preprocess_module_, kname))) {
            logstream << "cuda_infer_yolo: failed to get preprocess kernel";
            return false;
        }
        return true;
    }

    bool parseEngine() {
        std::ifstream f(engine_path_, std::ios::binary);
        if (!f) {
            logstream << "cuda_infer_yolo: cannot open engine file " << engine_path_;
            return false;
        }
        f.seekg(0, std::ios::end);
        std::streamsize size = f.tellg();
        if (size <= 0) {
            logstream << "cuda_infer_yolo: invalid engine size";
            return false;
        }
        f.seekg(0, std::ios::beg);
        std::vector<char> blob((size_t)size);
        if (!f.read(blob.data(), size)) {
            logstream << "cuda_infer_yolo: failed reading engine file";
            return false;
        }

        trt_runtime_ = nvinfer1::createInferRuntime(trt_logger_);
        if (!trt_runtime_) {
            logstream << "cuda_infer_yolo: createInferRuntime failed";
            return false;
        }
        trt_engine_ = trt_runtime_->deserializeCudaEngine(blob.data(), blob.size());
        if (!trt_engine_) {
            logstream << "cuda_infer_yolo: deserializeCudaEngine failed";
            return false;
        }
        trt_ctx_ = trt_engine_->createExecutionContext();
        if (!trt_ctx_) {
            logstream << "cuda_infer_yolo: createExecutionContext failed";
            return false;
        }
        return true;
    }

    bool allocateBindings() {
        const int nb = trt_engine_->getNbIOTensors();
        if (nb <= 1) {
            logstream << "cuda_infer_yolo: engine has insufficient bindings";
            return false;
        }

        io_tensor_names_.clear();
        tensor_bytes_.assign((size_t)nb, 0);
        tensor_ptrs_.assign((size_t)nb, 0);
        tensor_index_.clear();
        input_tensor_name_.clear();
        output_tensor_name_.clear();
        int input_count = 0;
        bool selected_image_input = false;

        for (int i = 0; i < nb; ++i) {
            const char* tensor_name_c = trt_engine_->getIOTensorName(i);
            if (!tensor_name_c) {
                logstream << "cuda_infer_yolo: null I/O tensor name";
                return false;
            }
            const std::string tensor_name = tensor_name_c;
            io_tensor_names_.push_back(tensor_name);
            tensor_index_[tensor_name] = (size_t)i;
            const auto mode = trt_engine_->getTensorIOMode(tensor_name_c);
            const bool is_input = (mode == nvinfer1::TensorIOMode::kINPUT);
            nvinfer1::Dims dims = trt_engine_->getTensorShape(tensor_name_c);
            for (int d = 0; d < dims.nbDims; ++d) {
                if (dims.d[d] <= 0) {
                    logstream << "cuda_infer_yolo: dynamic/invalid binding dims not supported in v1";
                    return false;
                }
            }
            const size_t vol = volume(dims);
            const size_t esz = elementSize(trt_engine_->getTensorDataType(tensor_name_c));
            if (vol == 0 || esz == 0) {
                logstream << "cuda_infer_yolo: unsupported binding type/shape";
                return false;
            }
            const size_t bytes = vol * esz;

            CUdeviceptr ptr = 0;
            if (CHECK_CU(cuMemAlloc(&ptr, bytes))) {
                logstream << "cuda_infer_yolo: cuMemAlloc failed for binding " << i;
                return false;
            }
            if (CHECK_CU(cuMemsetD8(ptr, 0, bytes))) {
                logstream << "cuda_infer_yolo: cuMemsetD8 failed for binding " << i;
                return false;
            }
            tensor_bytes_[(size_t)i] = bytes;
            tensor_ptrs_[(size_t)i] = ptr;

            if (is_input) {
                ++input_count;
                const bool is_image_input =
                    (dims.nbDims == 3 && dims.d[0] == 3) ||
                    (dims.nbDims == 4 && dims.d[0] == 1 && dims.d[1] == 3);
                if (input_tensor_name_.empty()) {
                    input_tensor_name_ = tensor_name;
                    input_dims_ = dims;
                }
                // Prefer image input in engines with auxiliary input tensors.
                if (is_image_input && !selected_image_input) {
                    input_tensor_name_ = tensor_name;
                    input_dims_ = dims;
                    selected_image_input = true;
                } else if (!is_image_input) {
                    logstream << "cuda_infer_yolo: auxiliary input tensor detected (ignored by preprocess): " << tensor_name;
                }
            } else if (output_tensor_name_.empty()) {
                output_tensor_name_ = tensor_name;
                output_dims_ = dims;
            }
        }

        if (input_tensor_name_.empty() || output_tensor_name_.empty()) {
            logstream << "cuda_infer_yolo: failed to identify input/output bindings";
            return false;
        }

        // Expect CHW or NCHW with batch=1.
        if (input_dims_.nbDims == 3 && input_dims_.d[0] == 3) {
            input_h_ = input_dims_.d[1];
            input_w_ = input_dims_.d[2];
        } else if (input_dims_.nbDims == 4 && input_dims_.d[0] == 1 && input_dims_.d[1] == 3) {
            input_h_ = input_dims_.d[2];
            input_w_ = input_dims_.d[3];
        } else {
            logstream << "cuda_infer_yolo: expected image input dims CHW(C=3) or NCHW(N=1,C=3)"
                      << " (engine inputs: " << input_count << ")";
            return false;
        }
        if (input_h_ <= 0 || input_w_ <= 0) {
            logstream << "cuda_infer_yolo: invalid input dims";
            return false;
        }

        output_dtype_ = trt_engine_->getTensorDataType(output_tensor_name_.c_str());
        if (!(output_dtype_ == nvinfer1::DataType::kFLOAT || output_dtype_ == nvinfer1::DataType::kHALF)) {
            logstream << "cuda_infer_yolo: output datatype must be float/half in v1";
            return false;
        }
        host_output_.resize(volume(output_dims_));
        if (output_dtype_ == nvinfer1::DataType::kHALF) {
            host_output_half_.resize(volume(output_dims_));
        }

        input_dtype_ = trt_engine_->getTensorDataType(input_tensor_name_.c_str());
        if (!(input_dtype_ == nvinfer1::DataType::kFLOAT || input_dtype_ == nvinfer1::DataType::kHALF)) {
            logstream << "cuda_infer_yolo: input datatype must be float/half in v1";
            return false;
        }

        // TRT10 API: bind tensor addresses by name.
        for (size_t i = 0; i < io_tensor_names_.size(); ++i) {
            if (!trt_ctx_->setTensorAddress(io_tensor_names_[i].c_str(), reinterpret_cast<void*>(tensor_ptrs_[i]))) {
                logstream << "cuda_infer_yolo: setTensorAddress failed for " << io_tensor_names_[i];
                return false;
            }
        }
        return true;
    }

    bool ensureInitialized(const av::VideoFrame& frm) {
        if (initialized_) return true;
        if (!initCudaContextFromFrame(frm)) return false;
        if (!parseEngine()) return false;
        if (!allocateBindings()) return false;
        if (!loadPreprocessKernel()) return false;
        initialized_ = true;
        return true;
    }

    AVPixelFormat hwSwFormat(const av::VideoFrame& frm) const {
        if (!frm.raw() || !frm.raw()->hw_frames_ctx || !frm.raw()->hw_frames_ctx->data) return AV_PIX_FMT_NONE;
        AVHWFramesContext* ctx = (AVHWFramesContext*)frm.raw()->hw_frames_ctx->data;
        if (!ctx) return AV_PIX_FMT_NONE;
        return ctx->sw_format;
    }

    bool runPreprocessNV12(const av::VideoFrame& frm) {
        const CUdeviceptr dY = (CUdeviceptr)(uintptr_t)frm.raw()->data[0];
        const CUdeviceptr dUV = (CUdeviceptr)(uintptr_t)frm.raw()->data[1];
        const size_t pitchY = (size_t)frm.raw()->linesize[0];
        const size_t pitchUV = (size_t)frm.raw()->linesize[1];
        auto it = tensor_index_.find(input_tensor_name_);
        if (it == tensor_index_.end()) {
            logstream << "cuda_infer_yolo: input tensor index missing";
            return false;
        }
        void* out = reinterpret_cast<void*>(tensor_ptrs_[it->second]);
        const int W = input_w_;
        const int H = input_h_;
        const int bgr = input_bgr_order_ ? 1 : 0;

        void* args[] = {
            (void*)&dY, (void*)&pitchY,
            (void*)&dUV, (void*)&pitchUV,
            (void*)&out,
            (void*)&W, (void*)&H,
            (void*)&bgr
        };
        const unsigned int blockX = 32;
        const unsigned int blockY = 8;
        const unsigned int gridX = (unsigned int)(W + (int)blockX - 1) / blockX;
        const unsigned int gridY = (unsigned int)(H + (int)blockY - 1) / blockY;
        if (CHECK_CU(cuLaunchKernel(preprocess_kernel_, gridX, gridY, 1, blockX, blockY, 1, 0, cuda_dev_ctx_->stream, args, nullptr))) {
            logstream << "cuda_infer_yolo: preprocess kernel launch failed";
            return false;
        }
        return true;
    }

    std::vector<Detection> decodeYoloOutput(const float* out, const nvinfer1::Dims& d) const {
        std::vector<Detection> dets;
        if (!out) return dets;

        // common YOLO export shape: [84, N] or [N, 84] (no batch dim in binding dims)
        if (d.nbDims == 2) {
            int a = d.d[0], b = d.d[1];
            int attrs = 0, count = 0;
            bool attrs_first = false;
            if (a >= 6 && b >= 1) {
                attrs = a; count = b; attrs_first = true;
            }
            if (b >= 6 && a >= 1 && b < a) {
                attrs = b; count = a; attrs_first = false;
            }
            if (attrs >= 6 && count > 0) {
                for (int i = 0; i < count; ++i) {
                    auto at = [&](int k)->float {
                        return attrs_first ? out[k * count + i] : out[i * attrs + k];
                    };
                    const float cx = at(0), cy = at(1), w = at(2), h = at(3);
                    int best_cls = 0;
                    float best = 0.0f;
                    for (int c = 4; c < attrs; ++c) {
                        float s = at(c);
                        if (s > best) {
                            best = s;
                            best_cls = c - 4;
                        }
                    }
                    if (best < conf_thresh_) continue;
                    Detection det;
                    det.x1 = cx - w * 0.5f;
                    det.y1 = cy - h * 0.5f;
                    det.x2 = cx + w * 0.5f;
                    det.y2 = cy + h * 0.5f;
                    det.conf = best;
                    det.cls = best_cls;
                    dets.push_back(det);
                }
            }
        } else if (d.nbDims == 3) {
            // common export layouts:
            // - raw head: [1,84,N] or [1,N,84]
            // - end2end:  [1,N,6] or [1,6,N] with [x1,y1,x2,y2,conf,cls]
            const int d0 = d.d[0], d1 = d.d[1], d2 = d.d[2];
            if (d0 == 1 && d2 == 6) {
                const int count = d1;
                for (int i = 0; i < count; ++i) {
                    const float x1 = out[i * 6 + 0];
                    const float y1 = out[i * 6 + 1];
                    const float x2 = out[i * 6 + 2];
                    const float y2 = out[i * 6 + 3];
                    const float conf = out[i * 6 + 4];
                    const int cls = (int)std::round(out[i * 6 + 5]);
                    if (conf < conf_thresh_) continue;
                    Detection det;
                    det.x1 = x1;
                    det.y1 = y1;
                    det.x2 = x2;
                    det.y2 = y2;
                    det.conf = conf;
                    det.cls = cls;
                    dets.push_back(det);
                }
            } else if (d0 == 1 && d1 == 6) {
                const int count = d2;
                for (int i = 0; i < count; ++i) {
                    const float x1 = out[0 * count + i];
                    const float y1 = out[1 * count + i];
                    const float x2 = out[2 * count + i];
                    const float y2 = out[3 * count + i];
                    const float conf = out[4 * count + i];
                    const int cls = (int)std::round(out[5 * count + i]);
                    if (conf < conf_thresh_) continue;
                    Detection det;
                    det.x1 = x1;
                    det.y1 = y1;
                    det.x2 = x2;
                    det.y2 = y2;
                    det.conf = conf;
                    det.cls = cls;
                    dets.push_back(det);
                }
            } else if (d0 == 1 && d1 >= 6 && d2 >= 1) {
                bool attrs_first = true;
                int attrs = d1;
                int count = d2;
                if (d2 >= 6 && d2 < d1) {
                    attrs_first = false;
                    attrs = d2;
                    count = d1;
                }
                for (int i = 0; i < count; ++i) {
                    auto at = [&](int k)->float {
                        return attrs_first ? out[k * count + i] : out[i * attrs + k];
                    };
                    const float cx = at(0), cy = at(1), w = at(2), h = at(3);
                    int best_cls = 0;
                    float best = 0.0f;
                    for (int c = 4; c < attrs; ++c) {
                        float s = at(c);
                        if (s > best) {
                            best = s;
                            best_cls = c - 4;
                        }
                    }
                    if (best < conf_thresh_) continue;
                    Detection det;
                    det.x1 = cx - w * 0.5f;
                    det.y1 = cy - h * 0.5f;
                    det.x2 = cx + w * 0.5f;
                    det.y2 = cy + h * 0.5f;
                    det.conf = best;
                    det.cls = best_cls;
                    dets.push_back(det);
                }
            }
        }

        // class-aware NMS
        std::sort(dets.begin(), dets.end(), [](const Detection& a, const Detection& b) { return a.conf > b.conf; });
        std::vector<Detection> kept;
        kept.reserve((size_t)std::min((int)dets.size(), max_det_));
        for (const Detection& dcur : dets) {
            bool drop = false;
            for (const Detection& dk : kept) {
                if (dcur.cls != dk.cls) continue;
                if (iou(dcur, dk) > iou_thresh_) {
                    drop = true;
                    break;
                }
            }
            if (!drop) {
                kept.push_back(dcur);
                if ((int)kept.size() >= max_det_) break;
            }
        }
        return kept;
    }

    std::string buildDetectionMetadata(const std::vector<Detection>& dets) const {
        Parameters j;
        j["version"] = 1;
        j["coord_space"] = "model";
        j["model_width"] = input_w_;
        j["model_height"] = input_h_;
        j["thresholds"] = {
            {"conf", conf_thresh_},
            {"iou", iou_thresh_},
            {"max_det", max_det_}
        };
        j["detections"] = Parameters::array();
        for (const Detection& d : dets) {
            Parameters item;
            item["cls"] = d.cls;
            if (d.cls >= 0 && (size_t)d.cls < class_names_.size()) {
                item["label"] = class_names_[(size_t)d.cls];
            }
            item["conf"] = d.conf;
            item["xyxy"] = {d.x1, d.y1, d.x2, d.y2};
            j["detections"].push_back(item);
        }
        return j.dump();
    }

public:
    using NodeSISO::NodeSISO;

    ~CudaInferYolo() override {
        for (CUdeviceptr p : tensor_ptrs_) {
            if (p) {
                CHECK_CU(cuMemFree(p));
            }
        }
        tensor_ptrs_.clear();
        tensor_bytes_.clear();
        io_tensor_names_.clear();
        tensor_index_.clear();

        if (preprocess_module_) {
            CHECK_CU(cuModuleUnload(preprocess_module_));
            preprocess_module_ = nullptr;
        }
        if (trt_ctx_) {
            delete trt_ctx_;
            trt_ctx_ = nullptr;
        }
        if (trt_engine_) {
            delete trt_engine_;
            trt_engine_ = nullptr;
        }
        if (trt_runtime_) {
            delete trt_runtime_;
            trt_runtime_ = nullptr;
        }
    }

    void process() override {
        av::VideoFrame frm = this->source_->get();
        if (!frm) return;

        frame_counter_++;
        if (infer_every_n_ > 1 && (frame_counter_ % (uint64_t)infer_every_n_) != 0) {
            this->sink_->put(frm);
            return;
        }

        if (frm.raw()->format != AV_PIX_FMT_CUDA) {
            logstream << "cuda_infer_yolo: non-CUDA frame, passing through";
            this->sink_->put(frm);
            return;
        }
        if (!ensureInitialized(frm)) {
            return;
        }
        if (frm.width() != input_w_ || frm.height() != input_h_) {
            logstream << "cuda_infer_yolo: input frame size mismatch, expected " << input_w_ << "x" << input_h_
                      << " got " << frm.width() << "x" << frm.height();
            return;
        }

        AVPixelFormat swfmt = hwSwFormat(frm);
        if (swfmt != AV_PIX_FMT_NV12) {
            logstream << "cuda_infer_yolo: unsupported hw sw_format (expected NV12)";
            return;
        }

        if (CHECK_CU(cuCtxSetCurrent(cu_ctx_))) {
            logstream << "cuda_infer_yolo: cuCtxSetCurrent failed in process";
            return;
        }

        if (!runPreprocessNV12(frm)) return;

        if (!trt_ctx_->enqueueV3(reinterpret_cast<cudaStream_t>(cuda_dev_ctx_->stream))) {
            logstream << "cuda_infer_yolo: enqueueV3 failed";
            return;
        }

        auto out_it = tensor_index_.find(output_tensor_name_);
        if (out_it == tensor_index_.end()) {
            logstream << "cuda_infer_yolo: output tensor index missing";
            return;
        }
        const size_t out_idx = out_it->second;
        const size_t out_bytes = tensor_bytes_[out_idx];
        if (output_dtype_ == nvinfer1::DataType::kFLOAT) {
            if (out_bytes != host_output_.size() * sizeof(float)) {
                logstream << "cuda_infer_yolo: output size mismatch";
                return;
            }
            if (CHECK_CU(cuMemcpyDtoHAsync(host_output_.data(), tensor_ptrs_[out_idx], out_bytes, cuda_dev_ctx_->stream))) {
                logstream << "cuda_infer_yolo: output D2H copy failed";
                return;
            }
        } else {
            if (out_bytes != host_output_half_.size() * sizeof(uint16_t)) {
                logstream << "cuda_infer_yolo: output half size mismatch";
                return;
            }
            if (CHECK_CU(cuMemcpyDtoHAsync(host_output_half_.data(), tensor_ptrs_[out_idx], out_bytes, cuda_dev_ctx_->stream))) {
                logstream << "cuda_infer_yolo: output D2H copy failed";
                return;
            }
        }
        if (CHECK_CU(cuStreamSynchronize(cuda_dev_ctx_->stream))) {
            logstream << "cuda_infer_yolo: stream sync failed";
            return;
        }
        if (output_dtype_ == nvinfer1::DataType::kHALF) {
            if (host_output_.size() != host_output_half_.size()) {
                logstream << "cuda_infer_yolo: output conversion buffer mismatch";
                return;
            }
            for (size_t i = 0; i < host_output_half_.size(); ++i) {
                host_output_[i] = halfToFloat(host_output_half_[i]);
            }
        }
        std::vector<Detection> dets = decodeYoloOutput(host_output_.data(), output_dims_);
        std::string md = buildDetectionMetadata(dets);
        av_dict_set(&frm.raw()->metadata, metadata_key_out_.c_str(), md.c_str(), 0);
        if (debug_log_metadata_ && debug_log_every_n_ > 0 && (frame_counter_ % (uint64_t)debug_log_every_n_) == 0) {
            logstream << "cuda_infer_yolo: " << md;
        }

        this->sink_->put(frm);
    }

    static std::shared_ptr<CudaInferYolo> create(NodeCreationInfo &nci) {
        EdgeManager &edges = nci.edges;
        const Parameters &params = nci.params;
        std::shared_ptr<Edge<av::VideoFrame>> src = edges.find<av::VideoFrame>(params["src"]);
        std::shared_ptr<Edge<av::VideoFrame>> dst = edges.find<av::VideoFrame>(params["dst"]);
        auto r = std::make_shared<CudaInferYolo>(
            make_unique<EdgeSource<av::VideoFrame>>(src),
            make_unique<EdgeSink<av::VideoFrame>>(dst)
        );
        if (!params.count("engine")) {
            throw Error("cuda_infer_yolo: missing required parameter: engine");
        }
        if (!params.count("hwaccel")) {
            throw Error("cuda_infer_yolo: missing required parameter: hwaccel");
        }
        r->engine_path_ = params["engine"].get<std::string>();
        r->hwaccel_ = InstanceSharedObjects<HWAccelDevice>::get(nci.instance, params["hwaccel"]);
        if (!r->hwaccel_) {
            throw Error("cuda_infer_yolo: failed to get hwaccel");
        }

        if (params.count("conf_thresh")) r->conf_thresh_ = params["conf_thresh"];
        if (params.count("iou_thresh")) r->iou_thresh_ = params["iou_thresh"];
        if (params.count("max_det")) r->max_det_ = params["max_det"];
        if (params.count("infer_every_n")) r->infer_every_n_ = params["infer_every_n"];
        if (params.count("metadata_key_out")) r->metadata_key_out_ = params["metadata_key_out"].get<std::string>();
        if (params.count("debug_log_metadata")) r->debug_log_metadata_ = params["debug_log_metadata"];
        if (params.count("debug_log_every_n")) r->debug_log_every_n_ = params["debug_log_every_n"];
        if (params.count("input_format")) {
            const std::string ifmt = params["input_format"].get<std::string>();
            r->input_bgr_order_ = (ifmt == "BGR" || ifmt == "bgr");
        }
        if (params.count("yolo_classes") && params.count("class_names")) {
            throw Error("cuda_infer_yolo: use either yolo_classes or class_names, not both");
        }
        if (params.count("yolo_classes")) {
            if (!params["yolo_classes"].is_string()) {
                throw Error("cuda_infer_yolo: yolo_classes must be a string path");
            }
            const std::string path = params["yolo_classes"].get<std::string>();
            std::string err;
            if (!loadClassNamesFromFile(path, r->class_names_, err)) {
                throw Error("cuda_infer_yolo: " + err);
            }
        } else if (params.count("class_names")) {
            if (params["class_names"].is_array()) {
                const std::list<std::string> names = jsonToStringList(params["class_names"]);
                r->class_names_.assign(names.begin(), names.end());
                if (r->class_names_.empty()) {
                    throw Error("cuda_infer_yolo: class_names array must not be empty");
                }
            } else {
                throw Error("cuda_infer_yolo: class_names must be a string array");
            }
        }
        return r;
    }
};

DECLNODE(cuda_infer_yolo, CudaInferYolo);

