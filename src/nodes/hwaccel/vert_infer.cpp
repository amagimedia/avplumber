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
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <deque>
#include <fstream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "../../../objs/src/nodes/hwaccel/vert_preprocess.ptx.h"

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

static uint16_t floatToHalf(float f) {
    uint32_t x = 0;
    memcpy(&x, &f, sizeof(float));
    uint32_t sign = (x >> 16) & 0x8000u;
    int32_t exp = (int32_t)((x >> 23) & 0xFFu) - 127 + 15;
    uint32_t mant = x & 0x7FFFFFu;

    if (exp <= 0) {
        if (exp < -10) return (uint16_t)sign;
        mant = (mant | 0x800000u) >> (1 - exp);
        if (mant & 0x00001000u) mant += 0x00002000u;
        return (uint16_t)(sign | (mant >> 13));
    }
    if (exp >= 31) {
        return (uint16_t)(sign | 0x7C00u);
    }
    if (mant & 0x00001000u) {
        mant += 0x00002000u;
        if (mant & 0x00800000u) {
            mant = 0;
            ++exp;
            if (exp >= 31) {
                return (uint16_t)(sign | 0x7C00u);
            }
        }
    }
    return (uint16_t)(sign | ((uint32_t)exp << 10) | (mant >> 13));
}

struct ViewportState {
    float center_x = 0.0f;
    float velocity = 0.0f;
    float acceleration = 0.0f;
};

struct CropBox {
    int x1 = 0;
    int x2 = 0;
    int y = 0;
    int w = 0;
    int h = 0;
};
}

class VertInfer : public NodeSISO<av::VideoFrame, av::VideoFrame> {
protected:
    enum class WarmupMode {
        Wait,
        CenterCrop
    };

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

    std::string visual_tensor_name_;
    std::string kinematics_tensor_name_;
    std::string output_tensor_name_;
    nvinfer1::Dims visual_dims_{};
    nvinfer1::Dims kinematics_dims_{};
    nvinfer1::Dims output_dims_{};

    int input_w_ = 0;
    int input_h_ = 0;
    int visual_channels_ = 0;
    int model_history_length_ = 0;
    int history_length_ = 3;
    int viewport_width_ = 0;
    bool input_bgr_order_ = false;
    nvinfer1::DataType visual_dtype_ = nvinfer1::DataType::kFLOAT;
    nvinfer1::DataType kinematics_dtype_ = nvinfer1::DataType::kFLOAT;
    nvinfer1::DataType output_dtype_ = nvinfer1::DataType::kFLOAT;

    CUmodule preprocess_module_ = nullptr;
    CUfunction preprocess_rgb_kernel_ = nullptr;
    CUfunction preprocess_gray_kernel_ = nullptr;

    std::vector<float> host_output_;
    std::vector<uint16_t> host_output_half_;
    std::vector<float> host_kinematics_;
    std::vector<uint16_t> host_kinematics_half_;

    std::deque<av::VideoFrame> frame_history_;
    std::deque<ViewportState> kinematics_history_;
    ViewportState current_state_;
    av::Timestamp last_input_pts_ = NOTS;

    std::string engine_path_;
    std::string metadata_key_out_ = "vert_crop_v1";
    std::string visual_tensor_name_param_;
    std::string kinematics_tensor_name_param_;
    std::string output_tensor_name_param_;
    std::string visual_mode_ = "grayscale";
    float friction_ = 0.1f;
    float force_to_pixels_ratio_ = 10.0f / 1280.0f;
    float default_fps_ = 30.0f;
    int viewport_width_override_ = 0;
    int background_width_override_ = 0;
    int background_height_override_ = 0;
    WarmupMode warmup_mode_ = WarmupMode::Wait;
    int debug_log_every_n_ = 0;
    bool debug_log_action_scores_ = false;
    uint64_t frame_counter_ = 0;

    bool initialized_ = false;

    bool initCudaContextFromFrame(const av::VideoFrame& frm) {
        if (cu_ctx_) return true;
        if (!frm.raw() || !frm.raw()->hw_frames_ctx || !frm.raw()->hw_frames_ctx->data) {
            logstream << "vert_infer: missing hw_frames_ctx";
            return false;
        }
        AVHWFramesContext* fctx = (AVHWFramesContext*)frm.raw()->hw_frames_ctx->data;
        if (!fctx || !fctx->device_ctx || !fctx->device_ctx->hwctx) {
            logstream << "vert_infer: missing device_ctx/hwctx in frame";
            return false;
        }
        cuda_dev_ctx_ = (AVCUDADeviceContext*)fctx->device_ctx->hwctx;
        if (!cuda_dev_ctx_ || !cuda_dev_ctx_->cuda_ctx) {
            logstream << "vert_infer: missing cuda context in frame";
            return false;
        }
        cu_ctx_ = cuda_dev_ctx_->cuda_ctx;
        if (CHECK_CU(cuCtxSetCurrent(cu_ctx_))) {
            logstream << "vert_infer: cuCtxSetCurrent failed";
            return false;
        }
        return true;
    }

    AVPixelFormat hwSwFormat(const av::VideoFrame& frm) const {
        if (!frm.raw() || !frm.raw()->hw_frames_ctx || !frm.raw()->hw_frames_ctx->data) return AV_PIX_FMT_NONE;
        AVHWFramesContext* ctx = (AVHWFramesContext*)frm.raw()->hw_frames_ctx->data;
        if (!ctx) return AV_PIX_FMT_NONE;
        return ctx->sw_format;
    }

    bool parseEngine() {
        if (trt_engine_ && trt_ctx_) return true;

        std::ifstream ifs(engine_path_, std::ios::binary | std::ios::ate);
        if (!ifs) {
            logstream << "vert_infer: cannot open engine file " << engine_path_;
            return false;
        }
        std::streamsize size = ifs.tellg();
        if (size <= 0) {
            logstream << "vert_infer: invalid engine size";
            return false;
        }
        ifs.seekg(0, std::ios::beg);
        std::vector<char> engine_data((size_t)size);
        if (!ifs.read(engine_data.data(), size)) {
            logstream << "vert_infer: failed reading engine file";
            return false;
        }
        trt_runtime_ = nvinfer1::createInferRuntime(trt_logger_);
        if (!trt_runtime_) {
            logstream << "vert_infer: createInferRuntime failed";
            return false;
        }
        trt_engine_ = trt_runtime_->deserializeCudaEngine(engine_data.data(), engine_data.size());
        if (!trt_engine_) {
            logstream << "vert_infer: deserializeCudaEngine failed";
            return false;
        }
        trt_ctx_ = trt_engine_->createExecutionContext();
        if (!trt_ctx_) {
            logstream << "vert_infer: createExecutionContext failed";
            return false;
        }
        return true;
    }

    bool isVisualTensor(const nvinfer1::Dims& dims, int* history = nullptr, int* channels = nullptr, int* h = nullptr, int* w = nullptr) const {
        if (dims.nbDims == 4 && (dims.d[1] == 1 || dims.d[1] == 3)) {
            if (history) *history = dims.d[0];
            if (channels) *channels = dims.d[1];
            if (h) *h = dims.d[2];
            if (w) *w = dims.d[3];
            return true;
        }
        if (dims.nbDims == 5 && dims.d[0] == 1 && (dims.d[2] == 1 || dims.d[2] == 3)) {
            if (history) *history = dims.d[1];
            if (channels) *channels = dims.d[2];
            if (h) *h = dims.d[3];
            if (w) *w = dims.d[4];
            return true;
        }
        return false;
    }

    bool isKinematicsTensor(const nvinfer1::Dims& dims, int* history = nullptr) const {
        if (dims.nbDims == 2 && dims.d[1] == 3) {
            if (history) *history = dims.d[0];
            return true;
        }
        if (dims.nbDims == 3 && dims.d[0] == 1 && dims.d[2] == 3) {
            if (history) *history = dims.d[1];
            return true;
        }
        return false;
    }

    bool isActionOutputTensor(const nvinfer1::Dims& dims) const {
        return volume(dims) == 5;
    }

    int alignEvenDown(int value) const {
        return value & ~1;
    }

    int clampInt(int value, int lo, int hi) const {
        return std::max(lo, std::min(hi, value));
    }

    int defaultViewportWidthFromHeight() const {
        if (input_h_ <= 0) return 0;
        int w = (input_h_ * 9) / 16;
        if (w <= 0) return 0;
        return alignEvenDown(w);
    }

    bool allocateBindings() {
        const int nb = trt_engine_->getNbIOTensors();
        if (nb <= 1) {
            logstream << "vert_infer: engine has insufficient bindings";
            return false;
        }

        io_tensor_names_.clear();
        tensor_bytes_.assign((size_t)nb, 0);
        tensor_ptrs_.assign((size_t)nb, 0);
        tensor_index_.clear();
        visual_tensor_name_.clear();
        kinematics_tensor_name_.clear();
        output_tensor_name_.clear();

        for (int i = 0; i < nb; ++i) {
            const char* tensor_name_c = trt_engine_->getIOTensorName(i);
            if (!tensor_name_c) {
                logstream << "vert_infer: null I/O tensor name";
                return false;
            }
            const std::string tensor_name = tensor_name_c;
            io_tensor_names_.push_back(tensor_name);
            tensor_index_[tensor_name] = (size_t)i;

            nvinfer1::Dims dims = trt_engine_->getTensorShape(tensor_name_c);
            for (int d = 0; d < dims.nbDims; ++d) {
                if (dims.d[d] <= 0) {
                    logstream << "vert_infer: dynamic/invalid binding dims not supported in v1";
                    return false;
                }
            }
            const size_t vol = volume(dims);
            const size_t esz = elementSize(trt_engine_->getTensorDataType(tensor_name_c));
            if (vol == 0 || esz == 0) {
                logstream << "vert_infer: unsupported binding type/shape";
                return false;
            }
            CUdeviceptr ptr = 0;
            const size_t bytes = vol * esz;
            if (CHECK_CU(cuMemAlloc(&ptr, bytes))) {
                logstream << "vert_infer: cuMemAlloc failed for binding " << i;
                return false;
            }
            if (CHECK_CU(cuMemsetD8(ptr, 0, bytes))) {
                logstream << "vert_infer: cuMemsetD8 failed for binding " << i;
                return false;
            }
            tensor_bytes_[(size_t)i] = bytes;
            tensor_ptrs_[(size_t)i] = ptr;

            const auto mode = trt_engine_->getTensorIOMode(tensor_name_c);
            if (mode == nvinfer1::TensorIOMode::kINPUT) {
                if (!visual_tensor_name_param_.empty() && tensor_name == visual_tensor_name_param_) {
                    visual_tensor_name_ = tensor_name;
                    visual_dims_ = dims;
                } else if (!kinematics_tensor_name_param_.empty() && tensor_name == kinematics_tensor_name_param_) {
                    kinematics_tensor_name_ = tensor_name;
                    kinematics_dims_ = dims;
                }
            } else {
                if (!output_tensor_name_param_.empty() && tensor_name == output_tensor_name_param_) {
                    output_tensor_name_ = tensor_name;
                    output_dims_ = dims;
                }
            }
        }

        for (int i = 0; i < nb; ++i) {
            const std::string& tensor_name = io_tensor_names_[(size_t)i];
            const auto mode = trt_engine_->getTensorIOMode(tensor_name.c_str());
            const nvinfer1::Dims dims = trt_engine_->getTensorShape(tensor_name.c_str());
            if (mode == nvinfer1::TensorIOMode::kINPUT) {
                if (visual_tensor_name_.empty() && isVisualTensor(dims)) {
                    visual_tensor_name_ = tensor_name;
                    visual_dims_ = dims;
                    continue;
                }
                if (kinematics_tensor_name_.empty() && isKinematicsTensor(dims)) {
                    kinematics_tensor_name_ = tensor_name;
                    kinematics_dims_ = dims;
                    continue;
                }
            } else if (output_tensor_name_.empty() && isActionOutputTensor(dims)) {
                output_tensor_name_ = tensor_name;
                output_dims_ = dims;
            }
        }

        if (visual_tensor_name_.empty() || kinematics_tensor_name_.empty() || output_tensor_name_.empty()) {
            logstream << "vert_infer: failed to identify visual/kinematics/output tensors";
            return false;
        }

        if (!isVisualTensor(visual_dims_, &model_history_length_, &visual_channels_, &input_h_, &input_w_)) {
            logstream << "vert_infer: unsupported visual tensor layout";
            return false;
        }

        int kinematics_history = 0;
        if (!isKinematicsTensor(kinematics_dims_, &kinematics_history)) {
            logstream << "vert_infer: unsupported viewport_kinematics tensor layout";
            return false;
        }
        if (kinematics_history != model_history_length_) {
            logstream << "vert_infer: history length mismatch between visual and viewport_kinematics inputs";
            return false;
        }

        if (history_length_ <= 0) history_length_ = model_history_length_;
        if (history_length_ != model_history_length_) {
            std::ostringstream ss;
            ss << "vert_infer: history_length parameter (" << history_length_
               << ") does not match engine visual history length (" << model_history_length_ << ")";
            throw Error(ss.str());
        }

        visual_dtype_ = trt_engine_->getTensorDataType(visual_tensor_name_.c_str());
        kinematics_dtype_ = trt_engine_->getTensorDataType(kinematics_tensor_name_.c_str());
        output_dtype_ = trt_engine_->getTensorDataType(output_tensor_name_.c_str());

        if (!(visual_dtype_ == nvinfer1::DataType::kFLOAT || visual_dtype_ == nvinfer1::DataType::kHALF)) {
            logstream << "vert_infer: visual tensor datatype must be float/half";
            return false;
        }
        if (!(kinematics_dtype_ == nvinfer1::DataType::kFLOAT || kinematics_dtype_ == nvinfer1::DataType::kHALF)) {
            logstream << "vert_infer: viewport_kinematics datatype must be float/half";
            return false;
        }
        if (!(output_dtype_ == nvinfer1::DataType::kFLOAT || output_dtype_ == nvinfer1::DataType::kHALF)) {
            logstream << "vert_infer: output datatype must be float/half";
            return false;
        }
        if (!isActionOutputTensor(output_dims_)) {
            logstream << "vert_infer: output tensor must contain exactly 5 action scores";
            return false;
        }

        if (visual_mode_ == "grayscale" && visual_channels_ != 1) {
            throw Error("vert_infer: visual_mode=grayscale but engine expects non-grayscale visual tensor");
        }
        if (visual_mode_ == "rgb" && visual_channels_ != 3) {
            throw Error("vert_infer: visual_mode=rgb but engine expects non-RGB visual tensor");
        }

        if (background_width_override_ > 0 && background_width_override_ != input_w_) {
            throw Error("vert_infer: background_width parameter does not match engine visual width");
        }
        if (background_height_override_ > 0 && background_height_override_ != input_h_) {
            throw Error("vert_infer: background_height parameter does not match engine visual height");
        }

        viewport_width_ = viewport_width_override_ > 0 ? alignEvenDown(viewport_width_override_) : defaultViewportWidthFromHeight();
        if (viewport_width_ <= 0 || viewport_width_ > input_w_) {
            throw Error("vert_infer: invalid viewport_width after initialization");
        }

        host_output_.resize(volume(output_dims_));
        if (output_dtype_ == nvinfer1::DataType::kHALF) {
            host_output_half_.resize(volume(output_dims_));
        }
        host_kinematics_.resize(volume(kinematics_dims_));
        if (kinematics_dtype_ == nvinfer1::DataType::kHALF) {
            host_kinematics_half_.resize(volume(kinematics_dims_));
        }

        for (size_t i = 0; i < io_tensor_names_.size(); ++i) {
            if (!trt_ctx_->setTensorAddress(io_tensor_names_[i].c_str(), reinterpret_cast<void*>(tensor_ptrs_[i]))) {
                logstream << "vert_infer: setTensorAddress failed for " << io_tensor_names_[i];
                return false;
            }
        }
        return true;
    }

    bool loadPreprocessKernel() {
        if (preprocess_module_ && preprocess_rgb_kernel_ && preprocess_gray_kernel_) return true;
        if (!cu_ctx_) return false;
        if (CHECK_CU(cuCtxSetCurrent(cu_ctx_))) return false;
        const std::string ptx_str(avpl_vert_preprocess_ptx, avpl_vert_preprocess_ptx + avpl_vert_preprocess_ptx_len);
        if (CHECK_CU(cuModuleLoadDataEx(&preprocess_module_, (const void*)ptx_str.c_str(), 0, nullptr, nullptr))) {
            logstream << "vert_infer: failed to load preprocess PTX module";
            return false;
        }
        const char* rgb_name = (visual_dtype_ == nvinfer1::DataType::kHALF) ? "kNV12_to_TCHW_rgb_fp16" : "kNV12_to_TCHW_rgb_fp32";
        const char* gray_name = (visual_dtype_ == nvinfer1::DataType::kHALF) ? "kNV12_to_TCHW_gray_fp16" : "kNV12_to_TCHW_gray_fp32";
        if (CHECK_CU(cuModuleGetFunction(&preprocess_rgb_kernel_, preprocess_module_, rgb_name))) {
            logstream << "vert_infer: failed to get RGB preprocess kernel";
            return false;
        }
        if (CHECK_CU(cuModuleGetFunction(&preprocess_gray_kernel_, preprocess_module_, gray_name))) {
            logstream << "vert_infer: failed to get grayscale preprocess kernel";
            return false;
        }
        return true;
    }

    bool ensureInitialized(const av::VideoFrame& frm) {
        if (initialized_) return true;
        if (!initCudaContextFromFrame(frm)) return false;
        if (!parseEngine()) return false;
        if (!allocateBindings()) return false;
        if (!loadPreprocessKernel()) return false;
        resetState();
        initialized_ = true;
        return true;
    }

    void resetState() {
        frame_history_.clear();
        kinematics_history_.clear();
        current_state_.center_x = input_w_ * 0.5f;
        current_state_.velocity = 0.0f;
        current_state_.acceleration = 0.0f;
        last_input_pts_ = NOTS;
    }

    void pushObservation(const av::VideoFrame& frm) {
        frame_history_.push_back(frm);
        if ((int)frame_history_.size() > history_length_) frame_history_.pop_front();

        kinematics_history_.push_back(current_state_);
        if ((int)kinematics_history_.size() > history_length_) kinematics_history_.pop_front();
    }

    bool historyReady() const {
        return (int)frame_history_.size() >= history_length_;
    }

    float frameDeltaSeconds(const av::VideoFrame& frm) {
        float fallback_dt = (default_fps_ > 0.0f) ? (1.0f / default_fps_) : (1.0f / 30.0f);
        if (!frm.pts().isValid()) return fallback_dt;
        if (!last_input_pts_.isValid()) {
            last_input_pts_ = frm.pts();
            return fallback_dt;
        }
        const double diff = (frm.pts() - last_input_pts_).seconds();
        last_input_pts_ = frm.pts();
        if (!(diff > 0.0) || diff > 1.0) return fallback_dt;
        return (float)diff;
    }

    bool runVisualPreprocess() {
        auto it = tensor_index_.find(visual_tensor_name_);
        if (it == tensor_index_.end()) {
            logstream << "vert_infer: visual tensor index missing";
            return false;
        }
        if ((int)frame_history_.size() < history_length_) {
            logstream << "vert_infer: history queue shorter than history_length";
            return false;
        }

        const size_t tensor_idx = it->second;
        const size_t frame_bytes = (size_t)visual_channels_ * (size_t)input_w_ * (size_t)input_h_ * elementSize(visual_dtype_);
        const unsigned int blockX = 32;
        const unsigned int blockY = 8;
        const unsigned int gridX = (unsigned int)(input_w_ + (int)blockX - 1) / blockX;
        const unsigned int gridY = (unsigned int)(input_h_ + (int)blockY - 1) / blockY;

        for (int slot = 0; slot < history_length_; ++slot) {
            const av::VideoFrame& hist = frame_history_[(size_t)(frame_history_.size() - 1 - slot)];
            if (hist.width() != input_w_ || hist.height() != input_h_) {
                logstream << "vert_infer: history frame size mismatch";
                return false;
            }
            if (hwSwFormat(hist) != AV_PIX_FMT_NV12) {
                logstream << "vert_infer: history frame sw_format mismatch (expected NV12)";
                return false;
            }

            const CUdeviceptr dY = (CUdeviceptr)(uintptr_t)hist.raw()->data[0];
            const CUdeviceptr dUV = (CUdeviceptr)(uintptr_t)hist.raw()->data[1];
            const size_t pitchY = (size_t)hist.raw()->linesize[0];
            const size_t pitchUV = (size_t)hist.raw()->linesize[1];
            CUdeviceptr slot_ptr = tensor_ptrs_[tensor_idx] + (CUdeviceptr)((size_t)slot * frame_bytes);
            void* out = reinterpret_cast<void*>(slot_ptr);

            if (visual_channels_ == 1) {
                void* args[] = {
                    (void*)&dY, (void*)&pitchY,
                    (void*)&dUV, (void*)&pitchUV,
                    (void*)&out,
                    (void*)&input_w_, (void*)&input_h_
                };
                if (CHECK_CU(cuLaunchKernel(preprocess_gray_kernel_, gridX, gridY, 1, blockX, blockY, 1, 0,
                                            reinterpret_cast<cudaStream_t>(cuda_dev_ctx_->stream), args, nullptr))) {
                    logstream << "vert_infer: grayscale preprocess kernel launch failed";
                    return false;
                }
            } else {
                const int bgr = input_bgr_order_ ? 1 : 0;
                void* args[] = {
                    (void*)&dY, (void*)&pitchY,
                    (void*)&dUV, (void*)&pitchUV,
                    (void*)&out,
                    (void*)&input_w_, (void*)&input_h_,
                    (void*)&bgr
                };
                if (CHECK_CU(cuLaunchKernel(preprocess_rgb_kernel_, gridX, gridY, 1, blockX, blockY, 1, 0,
                                            reinterpret_cast<cudaStream_t>(cuda_dev_ctx_->stream), args, nullptr))) {
                    logstream << "vert_infer: RGB preprocess kernel launch failed";
                    return false;
                }
            }
        }

        return true;
    }

    bool copyKinematicsInput() {
        auto it = tensor_index_.find(kinematics_tensor_name_);
        if (it == tensor_index_.end()) {
            logstream << "vert_infer: viewport_kinematics tensor index missing";
            return false;
        }
        std::fill(host_kinematics_.begin(), host_kinematics_.end(), 0.0f);

        for (int t = 0; t < history_length_; ++t) {
            const ViewportState& st = kinematics_history_[(size_t)t];
            const size_t base = (size_t)t * 3u;
            host_kinematics_[base + 0] = st.center_x / (float)input_w_;
            host_kinematics_[base + 1] = st.velocity;
            host_kinematics_[base + 2] = st.acceleration / (1.0f + std::fabs(st.acceleration));
        }

        const size_t tensor_idx = it->second;
        if (kinematics_dtype_ == nvinfer1::DataType::kFLOAT) {
            const size_t bytes = host_kinematics_.size() * sizeof(float);
            if (bytes != tensor_bytes_[tensor_idx]) {
                logstream << "vert_infer: viewport_kinematics size mismatch";
                return false;
            }
            if (CHECK_CU(cuMemcpyHtoDAsync(tensor_ptrs_[tensor_idx], host_kinematics_.data(), bytes, cuda_dev_ctx_->stream))) {
                logstream << "vert_infer: viewport_kinematics H2D copy failed";
                return false;
            }
        } else {
            if (host_kinematics_half_.size() != host_kinematics_.size()) {
                logstream << "vert_infer: viewport_kinematics half buffer mismatch";
                return false;
            }
            for (size_t i = 0; i < host_kinematics_.size(); ++i) {
                host_kinematics_half_[i] = floatToHalf(host_kinematics_[i]);
            }
            const size_t bytes = host_kinematics_half_.size() * sizeof(uint16_t);
            if (bytes != tensor_bytes_[tensor_idx]) {
                logstream << "vert_infer: viewport_kinematics half size mismatch";
                return false;
            }
            if (CHECK_CU(cuMemcpyHtoDAsync(tensor_ptrs_[tensor_idx], host_kinematics_half_.data(), bytes, cuda_dev_ctx_->stream))) {
                logstream << "vert_infer: viewport_kinematics H2D copy failed";
                return false;
            }
        }

        return true;
    }

    int forceFromActionIndex(int action_idx, float& force_out) const {
        static const std::array<float, 5> kForceMap = {-1.0f, -0.3f, 0.0f, 0.3f, 1.0f};
        int idx = action_idx;
        if (idx < 0) idx = 0;
        if (idx >= (int)kForceMap.size()) idx = (int)kForceMap.size() - 1;
        force_out = kForceMap[(size_t)idx];
        return idx;
    }

    void updateViewportState(int action_idx, float dt_seconds) {
        float force = 0.0f;
        action_idx = forceFromActionIndex(action_idx, force);
        float pixel_movement = force * (force_to_pixels_ratio_ * (float)input_w_);
        pixel_movement *= (1.0f - friction_);

        current_state_.center_x += pixel_movement;
        const float min_viewport_x = viewport_width_ * 0.5f;
        const float max_viewport_x = (float)input_w_ - viewport_width_ * 0.5f;
        current_state_.center_x = std::max(min_viewport_x, std::min(max_viewport_x, current_state_.center_x));

        float prev_velocity = current_state_.velocity;
        if (dt_seconds <= 0.0f) dt_seconds = 1.0f / 30.0f;
        current_state_.velocity = (pixel_movement / (float)input_w_) / dt_seconds;
        current_state_.acceleration = (current_state_.velocity - prev_velocity) / dt_seconds;
    }

    CropBox currentCropBox() const {
        CropBox crop;
        crop.w = viewport_width_;
        crop.h = input_h_;
        crop.y = 0;

        int max_x1 = std::max(0, input_w_ - viewport_width_);
        int x1 = (int)std::lround(current_state_.center_x - viewport_width_ * 0.5f);
        x1 = clampInt(x1, 0, max_x1);
        x1 = alignEvenDown(x1);
        if (x1 > max_x1) x1 = alignEvenDown(max_x1);
        if (x1 < 0) x1 = 0;
        crop.x1 = x1;
        crop.x2 = std::min(input_w_, x1 + viewport_width_);
        crop.w = crop.x2 - crop.x1;
        return crop;
    }

    Parameters buildCropMetadata(const CropBox& crop, int action_idx, float latency_ms, const std::array<float, 5>& action_scores) const {
        Parameters j;
        j["version"] = 1;
        j["x1"] = crop.x1;
        j["x2"] = crop.x2;
        j["y"] = crop.y;
        j["w"] = crop.w;
        j["h"] = crop.h;
        j["viewport_center_x"] = current_state_.center_x;
        j["viewport_width"] = viewport_width_;
        j["action"] = action_idx;
        j["latency_ms"] = latency_ms;
        if (debug_log_action_scores_) {
            j["action_scores"] = {action_scores[0], action_scores[1], action_scores[2], action_scores[3], action_scores[4]};
        }
        return j;
    }

    void maybeLogFrame(const CropBox& crop, int action_idx, float latency_ms, const std::array<float, 5>& action_scores, bool warmup) const {
        if (debug_log_every_n_ <= 0) return;
        if ((frame_counter_ % (uint64_t)debug_log_every_n_) != 0) return;
        std::ostringstream ss;
        ss << "vert_infer: frame=" << frame_counter_;
        if (warmup) {
            ss << " warmup";
        }
        ss << " action=" << action_idx
           << " x1=" << crop.x1
           << " x2=" << crop.x2
           << " w=" << crop.w
           << " latency_ms=" << latency_ms;
        if (debug_log_action_scores_) {
            ss << " scores=[" << action_scores[0] << "," << action_scores[1] << "," << action_scores[2] << "," << action_scores[3] << "," << action_scores[4] << "]";
        }
        logstream << ss.str();
    }

    void attachMetadata(av::VideoFrame& frm, const Parameters& j) const {
        const std::string md = j.dump();
        av_dict_set(&frm.raw()->metadata, metadata_key_out_.c_str(), md.c_str(), 0);
    }

    void emitCenterCropMetadata(av::VideoFrame& frm) const {
        CropBox crop = currentCropBox();
        std::array<float, 5> scores = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        attachMetadata(frm, buildCropMetadata(crop, 2, 0.0f, scores));
        maybeLogFrame(crop, 2, 0.0f, scores, true);
    }

public:
    using NodeSISO::NodeSISO;

    ~VertInfer() override {
        for (CUdeviceptr p : tensor_ptrs_) {
            if (p) CHECK_CU(cuMemFree(p));
        }
        tensor_ptrs_.clear();
        tensor_bytes_.clear();
        if (preprocess_module_) CHECK_CU(cuModuleUnload(preprocess_module_));
        preprocess_module_ = nullptr;
        preprocess_rgb_kernel_ = nullptr;
        preprocess_gray_kernel_ = nullptr;
        delete trt_ctx_;
        delete trt_engine_;
        delete trt_runtime_;
        trt_ctx_ = nullptr;
        trt_engine_ = nullptr;
        trt_runtime_ = nullptr;
    }

    void process() override {
        av::VideoFrame frm = this->source_->get();
        if (isEofMarker(frm)) {
            this->sink_->put(frm);
            return;
        }

        if (!frm) return;
        ++frame_counter_;

        if (frm.raw()->format != AV_PIX_FMT_CUDA) {
            logstream << "vert_infer: non-CUDA frame, passing through";
            resetState();
            this->sink_->put(frm);
            return;
        }
        if (!ensureInitialized(frm)) {
            this->sink_->put(frm);
            return;
        }
        if (frm.width() != input_w_ || frm.height() != input_h_) {
            logstream << "vert_infer: input frame size mismatch, expected " << input_w_ << "x" << input_h_
                      << " got " << frm.width() << "x" << frm.height();
            resetState();
            this->sink_->put(frm);
            return;
        }
        if (hwSwFormat(frm) != AV_PIX_FMT_NV12) {
            logstream << "vert_infer: unsupported hw sw_format (expected NV12)";
            resetState();
            this->sink_->put(frm);
            return;
        }
        if (CHECK_CU(cuCtxSetCurrent(cu_ctx_))) {
            logstream << "vert_infer: cuCtxSetCurrent failed in process";
            this->sink_->put(frm);
            return;
        }

        const float dt_seconds = frameDeltaSeconds(frm);
        pushObservation(frm);
        if (!historyReady()) {
            if (warmup_mode_ == WarmupMode::CenterCrop) {
                emitCenterCropMetadata(frm);
            } else if (debug_log_every_n_ > 0 && (frame_counter_ % (uint64_t)debug_log_every_n_) == 0) {
                logstream << "vert_infer: frame=" << frame_counter_ << " warmup history=" << frame_history_.size() << "/" << history_length_;
            }
            this->sink_->put(frm);
            return;
        }

        auto start = std::chrono::steady_clock::now();
        if (!runVisualPreprocess()) {
            this->sink_->put(frm);
            return;
        }
        if (!copyKinematicsInput()) {
            this->sink_->put(frm);
            return;
        }
        if (!trt_ctx_->enqueueV3(reinterpret_cast<cudaStream_t>(cuda_dev_ctx_->stream))) {
            logstream << "vert_infer: enqueueV3 failed";
            this->sink_->put(frm);
            return;
        }

        auto out_it = tensor_index_.find(output_tensor_name_);
        if (out_it == tensor_index_.end()) {
            logstream << "vert_infer: output tensor index missing";
            this->sink_->put(frm);
            return;
        }
        const size_t out_idx = out_it->second;
        const size_t out_bytes = tensor_bytes_[out_idx];
        if (output_dtype_ == nvinfer1::DataType::kFLOAT) {
            if (out_bytes != host_output_.size() * sizeof(float)) {
                logstream << "vert_infer: output size mismatch";
                this->sink_->put(frm);
                return;
            }
            if (CHECK_CU(cuMemcpyDtoHAsync(host_output_.data(), tensor_ptrs_[out_idx], out_bytes, cuda_dev_ctx_->stream))) {
                logstream << "vert_infer: output D2H copy failed";
                this->sink_->put(frm);
                return;
            }
        } else {
            if (out_bytes != host_output_half_.size() * sizeof(uint16_t)) {
                logstream << "vert_infer: output half size mismatch";
                this->sink_->put(frm);
                return;
            }
            if (CHECK_CU(cuMemcpyDtoHAsync(host_output_half_.data(), tensor_ptrs_[out_idx], out_bytes, cuda_dev_ctx_->stream))) {
                logstream << "vert_infer: output D2H copy failed";
                this->sink_->put(frm);
                return;
            }
        }
        if (CHECK_CU(cuStreamSynchronize(cuda_dev_ctx_->stream))) {
            logstream << "vert_infer: stream sync failed";
            this->sink_->put(frm);
            return;
        }
        const float latency_ms = (float)std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - start).count() / 1000.0f;

        if (output_dtype_ == nvinfer1::DataType::kHALF) {
            if (host_output_.size() != host_output_half_.size()) {
                logstream << "vert_infer: output conversion buffer mismatch";
                this->sink_->put(frm);
                return;
            }
            for (size_t i = 0; i < host_output_half_.size(); ++i) {
                host_output_[i] = halfToFloat(host_output_half_[i]);
            }
        }

        std::array<float, 5> action_scores = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        for (size_t i = 0; i < action_scores.size() && i < host_output_.size(); ++i) {
            action_scores[i] = host_output_[i];
        }
        int action_idx = 0;
        for (int i = 1; i < 5; ++i) {
            if (action_scores[(size_t)i] > action_scores[(size_t)action_idx]) {
                action_idx = i;
            }
        }
        updateViewportState(action_idx, dt_seconds);
        CropBox crop = currentCropBox();
        attachMetadata(frm, buildCropMetadata(crop, action_idx, latency_ms, action_scores));
        maybeLogFrame(crop, action_idx, latency_ms, action_scores, false);
        this->sink_->put(frm);
    }

    static std::shared_ptr<VertInfer> create(NodeCreationInfo &nci) {
        EdgeManager &edges = nci.edges;
        const Parameters &params = nci.params;
        std::shared_ptr<Edge<av::VideoFrame>> src = edges.find<av::VideoFrame>(params["src"]);
        std::shared_ptr<Edge<av::VideoFrame>> dst = edges.find<av::VideoFrame>(params["dst"]);
        auto r = std::make_shared<VertInfer>(
            make_unique<EdgeSource<av::VideoFrame>>(src),
            make_unique<EdgeSink<av::VideoFrame>>(dst)
        );
        src->setConsumer(r);
        dst->setProducer(r);

        if (!params.count("engine")) {
            throw Error("vert_infer: missing required parameter: engine");
        }
        if (!params.count("hwaccel")) {
            throw Error("vert_infer: missing required parameter: hwaccel");
        }
        r->engine_path_ = params["engine"].get<std::string>();
        r->hwaccel_ = InstanceSharedObjects<HWAccelDevice>::get(nci.instance, params["hwaccel"]);
        if (!r->hwaccel_) {
            throw Error("vert_infer: failed to get hwaccel");
        }

        if (params.count("metadata_key_out")) r->metadata_key_out_ = params["metadata_key_out"].get<std::string>();
        if (params.count("history_length")) r->history_length_ = params["history_length"];
        if (params.count("viewport_width")) r->viewport_width_override_ = params["viewport_width"];
        if (params.count("background_width")) r->background_width_override_ = params["background_width"];
        if (params.count("background_height")) r->background_height_override_ = params["background_height"];
        if (params.count("friction")) r->friction_ = params["friction"];
        if (params.count("force_to_pixels")) r->force_to_pixels_ratio_ = params["force_to_pixels"];
        if (params.count("default_fps")) r->default_fps_ = params["default_fps"];
        if (params.count("debug_log_every_n")) r->debug_log_every_n_ = params["debug_log_every_n"];
        if (params.count("debug_log_action_scores")) r->debug_log_action_scores_ = params["debug_log_action_scores"];
        if (params.count("visual_tensor_name")) r->visual_tensor_name_param_ = params["visual_tensor_name"].get<std::string>();
        if (params.count("kinematics_tensor_name")) r->kinematics_tensor_name_param_ = params["kinematics_tensor_name"].get<std::string>();
        if (params.count("output_tensor_name")) r->output_tensor_name_param_ = params["output_tensor_name"].get<std::string>();
        if (params.count("input_format")) {
            const std::string ifmt = params["input_format"].get<std::string>();
            r->input_bgr_order_ = (ifmt == "BGR" || ifmt == "bgr");
        }
        if (params.count("visual_mode")) {
            r->visual_mode_ = params["visual_mode"].get<std::string>();
            if (r->visual_mode_ != "rgb" && r->visual_mode_ != "grayscale") {
                throw Error("vert_infer: visual_mode must be 'rgb' or 'grayscale'");
            }
        }
        if (params.count("warmup_mode")) {
            const std::string mode = params["warmup_mode"].get<std::string>();
            if (mode == "wait") {
                r->warmup_mode_ = WarmupMode::Wait;
            } else if (mode == "center_crop") {
                r->warmup_mode_ = WarmupMode::CenterCrop;
            } else {
                throw Error("vert_infer: warmup_mode must be 'wait' or 'center_crop'");
            }
        }
        return r;
    }
};

DECLNODE(vert_infer, VertInfer)
