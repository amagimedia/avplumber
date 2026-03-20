#include "../node_common.hpp"
#include "../../hwaccel.hpp"
#include <cuda_loader/cuda_drvapi_dynlink_cuda.h>

extern "C" {
#include <libavutil/dict.h>
#include <libavutil/hwcontext.h>
#include <libavutil/hwcontext_cuda.h>
}

#include <NvInfer.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <string>
#include <unordered_map>
#include <utility>
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
    int model_index = -1;
};

enum class OutputBoxFormat {
    EndToEndXYXY,
    RawCXCYWH
};

static float halfToFloat(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1Fu;
    uint32_t frac = h & 0x03FFu;
    uint32_t out = 0;
    if (exp == 0) {
        if (frac == 0) {
            out = sign;
        } else {
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
        out = sign | 0x7F800000u | (frac << 13);
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

static std::string shortEngineName(const std::string& path) {
    const size_t slash = path.find_last_of("/\\");
    if (slash == std::string::npos) return path;
    return path.substr(slash + 1);
}
}

class CudaInferYolo : public NodeSISO<av::VideoFrame, av::VideoFrame> {
protected:
    struct ModelRunner {
        std::string engine_path;
        std::string engine_name;

        nvinfer1::IRuntime* trt_runtime = nullptr;
        nvinfer1::ICudaEngine* trt_engine = nullptr;
        nvinfer1::IExecutionContext* trt_ctx = nullptr;

        std::vector<std::string> io_tensor_names;
        std::vector<size_t> tensor_bytes;
        std::vector<CUdeviceptr> tensor_ptrs;
        std::unordered_map<std::string, size_t> tensor_index;
        std::string input_tensor_name;
        std::string output_tensor_name;
        nvinfer1::Dims input_dims{};
        nvinfer1::Dims output_dims{};
        int input_w = 0;
        int input_h = 0;
        nvinfer1::DataType input_dtype = nvinfer1::DataType::kFLOAT;
        nvinfer1::DataType output_dtype = nvinfer1::DataType::kFLOAT;
        CUfunction preprocess_kernel = nullptr;
        CUstream stream = nullptr;
        std::vector<float> host_output;
        std::vector<uint16_t> host_output_half;
    };

    std::shared_ptr<HWAccelDevice> hwaccel_;
    AVCUDADeviceContext* cuda_dev_ctx_ = nullptr;
    CUcontext cu_ctx_ = nullptr;

    TRTLogger trt_logger_;
    std::vector<ModelRunner> models_;
    std::vector<std::vector<std::string>> class_names_per_model_;

    int input_w_ = 0;
    int input_h_ = 0;
    nvinfer1::DataType input_dtype_ = nvinfer1::DataType::kFLOAT;
    bool input_bgr_order_ = false;
    CUmodule preprocess_module_ = nullptr;

    std::string metadata_key_out_ = "yolo_detections_v1";
    bool debug_log_metadata_ = false;
    int debug_log_every_n_ = 30;
    int infer_every_n_ = 1;
    float conf_thresh_ = 0.25f;
    float iou_thresh_ = 0.45f;
    int max_det_ = 300;
    uint64_t frame_counter_ = 0;
    OutputBoxFormat output_box_format_ = OutputBoxFormat::EndToEndXYXY;

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

    bool loadPreprocessModule() {
        if (preprocess_module_) return true;
        if (!cu_ctx_) return false;
        if (CHECK_CU(cuCtxSetCurrent(cu_ctx_))) return false;
        const std::string ptx_str(avpl_yolo_preprocess_ptx, avpl_yolo_preprocess_ptx + avpl_yolo_preprocess_ptx_len);
        if (CHECK_CU(cuModuleLoadDataEx(&preprocess_module_, (const void*)ptx_str.c_str(), 0, nullptr, nullptr))) {
            logstream << "cuda_infer_yolo: failed to load preprocess PTX module";
            return false;
        }
        return true;
    }

    bool parseEngine(ModelRunner& model) {
        std::ifstream f(model.engine_path, std::ios::binary);
        if (!f) {
            logstream << "cuda_infer_yolo: cannot open engine file " << model.engine_path;
            return false;
        }
        f.seekg(0, std::ios::end);
        std::streamsize size = f.tellg();
        if (size <= 0) {
            logstream << "cuda_infer_yolo: invalid engine size for " << model.engine_path;
            return false;
        }
        f.seekg(0, std::ios::beg);
        std::vector<char> blob((size_t)size);
        if (!f.read(blob.data(), size)) {
            logstream << "cuda_infer_yolo: failed reading engine file " << model.engine_path;
            return false;
        }

        model.trt_runtime = nvinfer1::createInferRuntime(trt_logger_);
        if (!model.trt_runtime) {
            logstream << "cuda_infer_yolo: createInferRuntime failed for " << model.engine_path;
            return false;
        }
        model.trt_engine = model.trt_runtime->deserializeCudaEngine(blob.data(), blob.size());
        if (!model.trt_engine) {
            logstream << "cuda_infer_yolo: deserializeCudaEngine failed for " << model.engine_path;
            return false;
        }
        model.trt_ctx = model.trt_engine->createExecutionContext();
        if (!model.trt_ctx) {
            logstream << "cuda_infer_yolo: createExecutionContext failed for " << model.engine_path;
            return false;
        }
        return true;
    }

    bool allocateBindings(ModelRunner& model) {
        const int nb = model.trt_engine->getNbIOTensors();
        if (nb <= 1) {
            logstream << "cuda_infer_yolo: engine has insufficient bindings for " << model.engine_path;
            return false;
        }

        model.io_tensor_names.clear();
        model.tensor_bytes.assign((size_t)nb, 0);
        model.tensor_ptrs.assign((size_t)nb, 0);
        model.tensor_index.clear();
        model.input_tensor_name.clear();
        model.output_tensor_name.clear();
        int input_count = 0;
        bool selected_image_input = false;

        for (int i = 0; i < nb; ++i) {
            const char* tensor_name_c = model.trt_engine->getIOTensorName(i);
            if (!tensor_name_c) {
                logstream << "cuda_infer_yolo: null I/O tensor name";
                return false;
            }
            const std::string tensor_name = tensor_name_c;
            model.io_tensor_names.push_back(tensor_name);
            model.tensor_index[tensor_name] = (size_t)i;

            const auto mode = model.trt_engine->getTensorIOMode(tensor_name_c);
            const bool is_input = (mode == nvinfer1::TensorIOMode::kINPUT);
            nvinfer1::Dims dims = model.trt_engine->getTensorShape(tensor_name_c);
            for (int d = 0; d < dims.nbDims; ++d) {
                if (dims.d[d] <= 0) {
                    logstream << "cuda_infer_yolo: dynamic/invalid binding dims not supported for " << model.engine_path;
                    return false;
                }
            }
            const size_t vol = volume(dims);
            const size_t esz = elementSize(model.trt_engine->getTensorDataType(tensor_name_c));
            if (vol == 0 || esz == 0) {
                logstream << "cuda_infer_yolo: unsupported binding type/shape for " << model.engine_path;
                return false;
            }
            const size_t bytes = vol * esz;

            CUdeviceptr ptr = 0;
            if (CHECK_CU(cuMemAlloc(&ptr, bytes))) {
                logstream << "cuda_infer_yolo: cuMemAlloc failed for " << model.engine_path << " binding " << i;
                return false;
            }
            if (CHECK_CU(cuMemsetD8(ptr, 0, bytes))) {
                logstream << "cuda_infer_yolo: cuMemsetD8 failed for " << model.engine_path << " binding " << i;
                return false;
            }
            model.tensor_bytes[(size_t)i] = bytes;
            model.tensor_ptrs[(size_t)i] = ptr;

            if (is_input) {
                ++input_count;
                const bool is_image_input =
                    (dims.nbDims == 3 && dims.d[0] == 3) ||
                    (dims.nbDims == 4 && dims.d[0] == 1 && dims.d[1] == 3);
                if (model.input_tensor_name.empty()) {
                    model.input_tensor_name = tensor_name;
                    model.input_dims = dims;
                }
                if (is_image_input && !selected_image_input) {
                    model.input_tensor_name = tensor_name;
                    model.input_dims = dims;
                    selected_image_input = true;
                } else if (!is_image_input) {
                    logstream << "cuda_infer_yolo: auxiliary input tensor ignored for " << model.engine_path << ": " << tensor_name;
                }
            } else if (model.output_tensor_name.empty()) {
                model.output_tensor_name = tensor_name;
                model.output_dims = dims;
            }
        }

        if (model.input_tensor_name.empty() || model.output_tensor_name.empty()) {
            logstream << "cuda_infer_yolo: failed to identify input/output bindings for " << model.engine_path;
            return false;
        }

        if (model.input_dims.nbDims == 3 && model.input_dims.d[0] == 3) {
            model.input_h = model.input_dims.d[1];
            model.input_w = model.input_dims.d[2];
        } else if (model.input_dims.nbDims == 4 && model.input_dims.d[0] == 1 && model.input_dims.d[1] == 3) {
            model.input_h = model.input_dims.d[2];
            model.input_w = model.input_dims.d[3];
        } else {
            logstream << "cuda_infer_yolo: expected CHW or NCHW input tensor for " << model.engine_path
                      << " (engine inputs: " << input_count << ")";
            return false;
        }
        if (model.input_h <= 0 || model.input_w <= 0) {
            logstream << "cuda_infer_yolo: invalid input dims for " << model.engine_path;
            return false;
        }

        model.output_dtype = model.trt_engine->getTensorDataType(model.output_tensor_name.c_str());
        if (!(model.output_dtype == nvinfer1::DataType::kFLOAT || model.output_dtype == nvinfer1::DataType::kHALF)) {
            logstream << "cuda_infer_yolo: output datatype must be float/half for " << model.engine_path;
            return false;
        }
        model.host_output.resize(volume(model.output_dims));
        if (model.output_dtype == nvinfer1::DataType::kHALF) {
            model.host_output_half.resize(volume(model.output_dims));
        }

        model.input_dtype = model.trt_engine->getTensorDataType(model.input_tensor_name.c_str());
        if (!(model.input_dtype == nvinfer1::DataType::kFLOAT || model.input_dtype == nvinfer1::DataType::kHALF)) {
            logstream << "cuda_infer_yolo: input datatype must be float/half for " << model.engine_path;
            return false;
        }

        for (size_t i = 0; i < model.io_tensor_names.size(); ++i) {
            if (!model.trt_ctx->setTensorAddress(model.io_tensor_names[i].c_str(), reinterpret_cast<void*>(model.tensor_ptrs[i]))) {
                logstream << "cuda_infer_yolo: setTensorAddress failed for " << model.io_tensor_names[i]
                          << " in " << model.engine_path;
                return false;
            }
        }
        return true;
    }

    bool ensureCompatibleInput(const ModelRunner& model, size_t model_index) {
        if (model_index == 0) {
            input_w_ = model.input_w;
            input_h_ = model.input_h;
            input_dtype_ = model.input_dtype;
            return true;
        }
        if (model.input_w != input_w_ || model.input_h != input_h_) {
            logstream << "cuda_infer_yolo: all engines must share the same input size, model[0]="
                      << input_w_ << "x" << input_h_ << " model[" << model_index << "]="
                      << model.input_w << "x" << model.input_h;
            return false;
        }
        if (model.input_dtype != input_dtype_) {
            logstream << "cuda_infer_yolo: all engines must share the same input dtype";
            return false;
        }
        return true;
    }

    bool configureRunnerPreprocess(ModelRunner& model) {
        const char* kname = (model.input_dtype == nvinfer1::DataType::kHALF)
            ? "kNV12_to_NCHW_fp16"
            : "kNV12_to_NCHW_fp32";
        if (CHECK_CU(cuModuleGetFunction(&model.preprocess_kernel, preprocess_module_, kname))) {
            logstream << "cuda_infer_yolo: failed to get preprocess kernel for " << model.engine_path;
            return false;
        }
        if (CHECK_CU(cuStreamCreate(&model.stream, 0))) {
            logstream << "cuda_infer_yolo: failed to create CUDA stream for " << model.engine_path;
            return false;
        }
        return true;
    }

    bool ensureInitialized(const av::VideoFrame& frm) {
        if (initialized_) return true;
        if (!initCudaContextFromFrame(frm)) return false;
        if (!loadPreprocessModule()) return false;

        for (size_t i = 0; i < models_.size(); ++i) {
            ModelRunner& model = models_[i];
            if (!parseEngine(model)) return false;
            if (!allocateBindings(model)) return false;
            if (!ensureCompatibleInput(model, i)) return false;
            if (!configureRunnerPreprocess(model)) return false;
        }

        initialized_ = true;
        return true;
    }

    AVPixelFormat hwSwFormat(const av::VideoFrame& frm) const {
        if (!frm.raw() || !frm.raw()->hw_frames_ctx || !frm.raw()->hw_frames_ctx->data) return AV_PIX_FMT_NONE;
        AVHWFramesContext* ctx = (AVHWFramesContext*)frm.raw()->hw_frames_ctx->data;
        if (!ctx) return AV_PIX_FMT_NONE;
        return ctx->sw_format;
    }

    bool runPreprocessNV12(const av::VideoFrame& frm, ModelRunner& model) {
        const CUdeviceptr dY = (CUdeviceptr)(uintptr_t)frm.raw()->data[0];
        const CUdeviceptr dUV = (CUdeviceptr)(uintptr_t)frm.raw()->data[1];
        const size_t pitchY = (size_t)frm.raw()->linesize[0];
        const size_t pitchUV = (size_t)frm.raw()->linesize[1];
        auto it = model.tensor_index.find(model.input_tensor_name);
        if (it == model.tensor_index.end()) {
            logstream << "cuda_infer_yolo: input tensor index missing for " << model.engine_path;
            return false;
        }
        void* out = reinterpret_cast<void*>(model.tensor_ptrs[it->second]);
        const int W = model.input_w;
        const int H = model.input_h;
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
        if (CHECK_CU(cuLaunchKernel(model.preprocess_kernel, gridX, gridY, 1, blockX, blockY, 1, 0, model.stream, args, nullptr))) {
            logstream << "cuda_infer_yolo: preprocess kernel launch failed for " << model.engine_path;
            return false;
        }
        return true;
    }

    std::vector<Detection> decodeYoloOutput(const float* out, const nvinfer1::Dims& d, int model_index) const {
        std::vector<Detection> dets;
        if (!out) return dets;

        auto pushCenterBox = [&](float cx, float cy, float w, float h, float conf, int cls) {
            if (conf < conf_thresh_) return;
            Detection det;
            det.x1 = cx - w * 0.5f;
            det.y1 = cy - h * 0.5f;
            det.x2 = cx + w * 0.5f;
            det.y2 = cy + h * 0.5f;
            det.conf = conf;
            det.cls = cls;
            det.model_index = model_index;
            dets.push_back(det);
        };

        auto pushCornerBox = [&](float x1, float y1, float x2, float y2, float conf, int cls) {
            if (conf < conf_thresh_) return;
            Detection det;
            det.x1 = x1;
            det.y1 = y1;
            det.x2 = x2;
            det.y2 = y2;
            det.conf = conf;
            det.cls = cls;
            det.model_index = model_index;
            dets.push_back(det);
        };

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
                    pushCenterBox(cx, cy, w, h, best, best_cls);
                }
            }
        } else if (d.nbDims == 3) {
            const int d0 = d.d[0], d1 = d.d[1], d2 = d.d[2];
            if (d0 == 1 && d2 == 6) {
                const int count = d1;
                for (int i = 0; i < count; ++i) {
                    if (output_box_format_ == OutputBoxFormat::RawCXCYWH) {
                        const float cx = out[i * 6 + 0];
                        const float cy = out[i * 6 + 1];
                        const float w = out[i * 6 + 2];
                        const float h = out[i * 6 + 3];
                        const float score0 = out[i * 6 + 4];
                        const float score1 = out[i * 6 + 5];
                        const int cls = (score1 > score0) ? 1 : 0;
                        const float conf = std::max(score0, score1);
                        pushCenterBox(cx, cy, w, h, conf, cls);
                    } else {
                        const float x1 = out[i * 6 + 0];
                        const float y1 = out[i * 6 + 1];
                        const float x2 = out[i * 6 + 2];
                        const float y2 = out[i * 6 + 3];
                        const float conf = out[i * 6 + 4];
                        const int cls = (int)std::round(out[i * 6 + 5]);
                        pushCornerBox(x1, y1, x2, y2, conf, cls);
                    }
                }
            } else if (d0 == 1 && d1 == 6) {
                const int count = d2;
                for (int i = 0; i < count; ++i) {
                    if (output_box_format_ == OutputBoxFormat::RawCXCYWH) {
                        const float cx = out[0 * count + i];
                        const float cy = out[1 * count + i];
                        const float w = out[2 * count + i];
                        const float h = out[3 * count + i];
                        const float score0 = out[4 * count + i];
                        const float score1 = out[5 * count + i];
                        const int cls = (score1 > score0) ? 1 : 0;
                        const float conf = std::max(score0, score1);
                        pushCenterBox(cx, cy, w, h, conf, cls);
                    } else {
                        const float x1 = out[0 * count + i];
                        const float y1 = out[1 * count + i];
                        const float x2 = out[2 * count + i];
                        const float y2 = out[3 * count + i];
                        const float conf = out[4 * count + i];
                        const int cls = (int)std::round(out[5 * count + i]);
                        pushCornerBox(x1, y1, x2, y2, conf, cls);
                    }
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
                    pushCenterBox(cx, cy, w, h, best, best_cls);
                }
            }
        }

        return dets;
    }

    void finalizeDetections(std::vector<Detection>& dets) const {
        std::sort(dets.begin(), dets.end(), [](const Detection& a, const Detection& b) {
            return a.conf > b.conf;
        });
        if (max_det_ > 0 && (int)dets.size() > max_det_) {
            dets.resize((size_t)max_det_);
        }
    }

    std::string buildDetectionMetadata(const std::vector<Detection>& dets) const {
        Parameters j;
        j["version"] = 2;
        j["coord_space"] = "model";
        j["model_width"] = input_w_;
        j["model_height"] = input_h_;
        j["thresholds"] = {
            {"conf", conf_thresh_},
            {"iou", iou_thresh_},
            {"max_det", max_det_}
        };
        j["models"] = Parameters::array();
        for (size_t i = 0; i < models_.size(); ++i) {
            Parameters model_item;
            model_item["model_index"] = (int)i;
            model_item["engine"] = models_[i].engine_path;
            model_item["engine_name"] = models_[i].engine_name;
            j["models"].push_back(model_item);
        }
        j["detections"] = Parameters::array();
        for (const Detection& d : dets) {
            Parameters item;
            item["cls"] = d.cls;
            item["conf"] = d.conf;
            item["xyxy"] = {d.x1, d.y1, d.x2, d.y2};
            item["model_index"] = d.model_index;
            if (d.model_index >= 0 && (size_t)d.model_index < models_.size()) {
                item["engine_name"] = models_[(size_t)d.model_index].engine_name;
            }
            if (d.model_index >= 0 && (size_t)d.model_index < class_names_per_model_.size()) {
                const std::vector<std::string>& class_names = class_names_per_model_[(size_t)d.model_index];
                if (d.cls >= 0 && (size_t)d.cls < class_names.size()) {
                    item["label"] = class_names[(size_t)d.cls];
                }
            }
            j["detections"].push_back(item);
        }
        return j.dump();
    }

    void cleanupModel(ModelRunner& model) {
        if (model.stream) {
            CHECK_CU(cuStreamDestroy(model.stream));
            model.stream = nullptr;
        }
        for (CUdeviceptr ptr : model.tensor_ptrs) {
            if (ptr) {
                CHECK_CU(cuMemFree(ptr));
            }
        }
        model.tensor_ptrs.clear();
        model.tensor_bytes.clear();
        model.io_tensor_names.clear();
        model.tensor_index.clear();
        model.host_output.clear();
        model.host_output_half.clear();
        if (model.trt_ctx) {
            delete model.trt_ctx;
            model.trt_ctx = nullptr;
        }
        if (model.trt_engine) {
            delete model.trt_engine;
            model.trt_engine = nullptr;
        }
        if (model.trt_runtime) {
            delete model.trt_runtime;
            model.trt_runtime = nullptr;
        }
    }

public:
    using NodeSISO::NodeSISO;

    ~CudaInferYolo() override {
        if (cu_ctx_) {
            CHECK_CU(cuCtxSetCurrent(cu_ctx_));
        }
        for (ModelRunner& model : models_) {
            cleanupModel(model);
        }
        if (preprocess_module_) {
            CHECK_CU(cuModuleUnload(preprocess_module_));
            preprocess_module_ = nullptr;
        }
    }

    void process() override {
        av::VideoFrame frm = this->source_->get();
        if (!frm) return;

        ++frame_counter_;
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

        for (ModelRunner& model : models_) {
            if (!runPreprocessNV12(frm, model)) return;

            if (!model.trt_ctx->enqueueV3(reinterpret_cast<cudaStream_t>(model.stream))) {
                logstream << "cuda_infer_yolo: enqueueV3 failed for " << model.engine_path;
                return;
            }

            auto out_it = model.tensor_index.find(model.output_tensor_name);
            if (out_it == model.tensor_index.end()) {
                logstream << "cuda_infer_yolo: output tensor index missing for " << model.engine_path;
                return;
            }
            const size_t out_idx = out_it->second;
            const size_t out_bytes = model.tensor_bytes[out_idx];
            if (model.output_dtype == nvinfer1::DataType::kFLOAT) {
                if (out_bytes != model.host_output.size() * sizeof(float)) {
                    logstream << "cuda_infer_yolo: output size mismatch for " << model.engine_path;
                    return;
                }
                if (CHECK_CU(cuMemcpyDtoHAsync(model.host_output.data(), model.tensor_ptrs[out_idx], out_bytes, model.stream))) {
                    logstream << "cuda_infer_yolo: output D2H copy failed for " << model.engine_path;
                    return;
                }
            } else {
                if (out_bytes != model.host_output_half.size() * sizeof(uint16_t)) {
                    logstream << "cuda_infer_yolo: output half size mismatch for " << model.engine_path;
                    return;
                }
                if (CHECK_CU(cuMemcpyDtoHAsync(model.host_output_half.data(), model.tensor_ptrs[out_idx], out_bytes, model.stream))) {
                    logstream << "cuda_infer_yolo: output D2H copy failed for " << model.engine_path;
                    return;
                }
            }
        }

        std::vector<Detection> dets;
        for (size_t model_index = 0; model_index < models_.size(); ++model_index) {
            ModelRunner& model = models_[model_index];
            if (CHECK_CU(cuStreamSynchronize(model.stream))) {
                logstream << "cuda_infer_yolo: stream sync failed for " << model.engine_path;
                return;
            }
            if (model.output_dtype == nvinfer1::DataType::kHALF) {
                if (model.host_output.size() != model.host_output_half.size()) {
                    logstream << "cuda_infer_yolo: output conversion buffer mismatch for " << model.engine_path;
                    return;
                }
                for (size_t i = 0; i < model.host_output_half.size(); ++i) {
                    model.host_output[i] = halfToFloat(model.host_output_half[i]);
                }
            }

            std::vector<Detection> model_dets = decodeYoloOutput(model.host_output.data(), model.output_dims, (int)model_index);
            dets.insert(dets.end(), model_dets.begin(), model_dets.end());
        }

        finalizeDetections(dets);
        const std::string md = buildDetectionMetadata(dets);
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

        if (!params.count("engines")) {
            throw Error("cuda_infer_yolo: missing required parameter: engines");
        }
        if (!params["engines"].is_array()) {
            throw Error("cuda_infer_yolo: engines must be a string array");
        }
        for (const auto& item : params["engines"]) {
            if (!item.is_string()) {
                throw Error("cuda_infer_yolo: engines must be a string array");
            }
            ModelRunner model;
            model.engine_path = item.get<std::string>();
            model.engine_name = shortEngineName(model.engine_path);
            r->models_.push_back(std::move(model));
        }
        if (r->models_.empty()) {
            throw Error("cuda_infer_yolo: engines array must not be empty");
        }

        if (!params.count("hwaccel")) {
            throw Error("cuda_infer_yolo: missing required parameter: hwaccel");
        }
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
        if (params.count("output_box_format")) {
            const std::string fmt = params["output_box_format"].get<std::string>();
            if (fmt == "end2end_xyxy") {
                r->output_box_format_ = OutputBoxFormat::EndToEndXYXY;
            } else if (fmt == "raw_cxcywh") {
                r->output_box_format_ = OutputBoxFormat::RawCXCYWH;
            } else {
                throw Error("cuda_infer_yolo: output_box_format must be 'end2end_xyxy' or 'raw_cxcywh'");
            }
        }

        r->class_names_per_model_.resize(r->models_.size());
        if (params.count("class_names_per_model")) {
            if (!params["class_names_per_model"].is_array()) {
                throw Error("cuda_infer_yolo: class_names_per_model must be an array of string arrays");
            }
            if (params["class_names_per_model"].size() != r->models_.size()) {
                throw Error("cuda_infer_yolo: class_names_per_model must match engines length");
            }
            size_t model_index = 0;
            for (const auto& names_item : params["class_names_per_model"]) {
                if (!names_item.is_array()) {
                    throw Error("cuda_infer_yolo: class_names_per_model must be an array of string arrays");
                }
                for (const auto& name : names_item) {
                    if (!name.is_string()) {
                        throw Error("cuda_infer_yolo: class_names_per_model must be an array of string arrays");
                    }
                    r->class_names_per_model_[model_index].push_back(name.get<std::string>());
                }
                ++model_index;
            }
        }

        return r;
    }
};

DECLNODE(cuda_infer_yolo, CudaInferYolo);

