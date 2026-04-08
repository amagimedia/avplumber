#include "../../node_common.hpp"
#include "../../../hwaccel.hpp"
#include <cuda_loader/cuda_drvapi_dynlink_cuda.h>

extern "C" {
#include <libavutil/dict.h>
#include <libavutil/hwcontext.h>
#include <libavutil/hwcontext_cuda.h>
}

#include <NvInfer.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cctype>
#include <cstring>
#include <deque>
#include <fstream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "../../../../objs/src/nodes/neural_net/utils/amagi_reframer.ptx.h"

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
        case nvinfer1::DataType::kUINT8: return 1;
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

static std::string lowercase(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
        return (char)std::tolower(c);
    });
    return s;
}

static bool containsToken(const std::string& value, const std::string& token) {
    return lowercase(value).find(lowercase(token)) != std::string::npos;
}

struct ViewportBBox {
    int x1 = 0;
    int y1 = 0;
    int x2 = 0;
    int y2 = 0;
};
}

class Reframer : public NodeSISO<av::VideoFrame, av::VideoFrame>, public IInputReset {
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

    std::string visual_tensor_name_;
    std::string kinematics_buffer_in_tensor_name_;
    std::string viewport_center_x_in_tensor_name_;
    std::string prev_velocity_in_tensor_name_;
    std::string bbox_norm_tensor_name_;
    std::string kinematics_buffer_out_tensor_name_;
    std::string viewport_center_x_out_tensor_name_;
    std::string prev_velocity_out_tensor_name_;

    nvinfer1::Dims visual_dims_{};
    nvinfer1::Dims kinematics_buffer_in_dims_{};
    nvinfer1::Dims viewport_center_x_in_dims_{};
    nvinfer1::Dims prev_velocity_in_dims_{};
    nvinfer1::Dims bbox_norm_dims_{};
    nvinfer1::Dims kinematics_buffer_out_dims_{};
    nvinfer1::Dims viewport_center_x_out_dims_{};
    nvinfer1::Dims prev_velocity_out_dims_{};

    int input_w_ = 0;
    int input_h_ = 0;
    int visual_channels_ = 0;
    int model_history_length_ = 0;
    int history_length_ = 3;
    int kinematics_dim_ = 0;
    bool input_bgr_order_ = false;

    nvinfer1::DataType visual_dtype_ = nvinfer1::DataType::kFLOAT;
    nvinfer1::DataType kinematics_buffer_in_dtype_ = nvinfer1::DataType::kFLOAT;
    nvinfer1::DataType viewport_center_x_in_dtype_ = nvinfer1::DataType::kFLOAT;
    nvinfer1::DataType prev_velocity_in_dtype_ = nvinfer1::DataType::kFLOAT;
    nvinfer1::DataType bbox_norm_dtype_ = nvinfer1::DataType::kFLOAT;
    nvinfer1::DataType kinematics_buffer_out_dtype_ = nvinfer1::DataType::kFLOAT;
    nvinfer1::DataType viewport_center_x_out_dtype_ = nvinfer1::DataType::kFLOAT;
    nvinfer1::DataType prev_velocity_out_dtype_ = nvinfer1::DataType::kFLOAT;

    CUmodule preprocess_module_ = nullptr;
    CUfunction preprocess_rgb_kernel_ = nullptr;

    std::deque<av::VideoFrame> frame_history_;

    std::vector<float> host_kinematics_buffer_in_;
    std::vector<uint16_t> host_kinematics_buffer_in_half_;
    std::vector<float> host_viewport_center_x_in_;
    std::vector<uint16_t> host_viewport_center_x_in_half_;
    std::vector<float> host_prev_velocity_in_;
    std::vector<uint16_t> host_prev_velocity_in_half_;

    std::vector<float> host_kinematics_buffer_out_;
    std::vector<uint16_t> host_kinematics_buffer_out_half_;
    std::vector<float> host_viewport_center_x_out_;
    std::vector<uint16_t> host_viewport_center_x_out_half_;
    std::vector<float> host_prev_velocity_out_;
    std::vector<uint16_t> host_prev_velocity_out_half_;
    std::vector<float> host_bbox_norm_;
    std::vector<uint16_t> host_bbox_norm_half_;

    std::string engine_path_;
    std::string metadata_key_out_ = "reframer_bbox";
    std::string visual_tensor_name_param_ = "visual";
    std::string kinematics_buffer_in_tensor_name_param_ = "kinematics_buffer_in";
    std::string viewport_center_x_in_tensor_name_param_ = "viewport_center_x_in";
    std::string prev_velocity_in_tensor_name_param_ = "prev_velocity_in";
    std::string bbox_norm_tensor_name_param_ = "bbox_norm";
    std::string kinematics_buffer_out_tensor_name_param_ = "kinematics_buffer_out";
    std::string viewport_center_x_out_tensor_name_param_ = "viewport_center_x_out";
    std::string prev_velocity_out_tensor_name_param_ = "prev_velocity_out";
    int full_frame_width_ = 0;
    int full_frame_height_ = 0;
    int debug_log_every_n_ = 0;
    uint64_t frame_counter_ = 0;

    bool initialized_ = false;

    bool initCudaContextFromFrame(const av::VideoFrame& frm) {
        if (cu_ctx_) return true;
        if (!frm.raw() || !frm.raw()->hw_frames_ctx || !frm.raw()->hw_frames_ctx->data) {
            logstream << "reframer: missing hw_frames_ctx";
            return false;
        }
        AVHWFramesContext* fctx = (AVHWFramesContext*)frm.raw()->hw_frames_ctx->data;
        if (!fctx || !fctx->device_ctx || !fctx->device_ctx->hwctx) {
            logstream << "reframer: missing device_ctx/hwctx in frame";
            return false;
        }
        cuda_dev_ctx_ = (AVCUDADeviceContext*)fctx->device_ctx->hwctx;
        if (!cuda_dev_ctx_ || !cuda_dev_ctx_->cuda_ctx) {
            logstream << "reframer: missing cuda context in frame";
            return false;
        }
        cu_ctx_ = cuda_dev_ctx_->cuda_ctx;
        if (CHECK_CU(cuCtxSetCurrent(cu_ctx_))) {
            logstream << "reframer: cuCtxSetCurrent failed";
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
            logstream << "reframer: cannot open engine file " << engine_path_;
            return false;
        }
        std::streamsize size = ifs.tellg();
        if (size <= 0) {
            logstream << "reframer: invalid engine size";
            return false;
        }
        ifs.seekg(0, std::ios::beg);
        std::vector<char> engine_data((size_t)size);
        if (!ifs.read(engine_data.data(), size)) {
            logstream << "reframer: failed reading engine file";
            return false;
        }
        trt_runtime_ = nvinfer1::createInferRuntime(trt_logger_);
        if (!trt_runtime_) {
            logstream << "reframer: createInferRuntime failed";
            return false;
        }
        trt_engine_ = trt_runtime_->deserializeCudaEngine(engine_data.data(), engine_data.size());
        if (!trt_engine_) {
            logstream << "reframer: deserializeCudaEngine failed";
            return false;
        }
        trt_ctx_ = trt_engine_->createExecutionContext();
        if (!trt_ctx_) {
            logstream << "reframer: createExecutionContext failed";
            return false;
        }
        return true;
    }

    bool isVisualTensor(const nvinfer1::Dims& dims, int* history = nullptr, int* channels = nullptr, int* h = nullptr, int* w = nullptr) const {
        if (dims.nbDims == 4 && dims.d[0] > 0 && dims.d[1] == 3 && dims.d[2] > 0 && dims.d[3] > 0) {
            if (history) *history = dims.d[0];
            if (channels) *channels = dims.d[1];
            if (h) *h = dims.d[2];
            if (w) *w = dims.d[3];
            return true;
        }
        if (dims.nbDims == 5 && dims.d[0] == 1 && dims.d[1] > 0 && dims.d[2] == 3 && dims.d[3] > 0 && dims.d[4] > 0) {
            if (history) *history = dims.d[1];
            if (channels) *channels = dims.d[2];
            if (h) *h = dims.d[3];
            if (w) *w = dims.d[4];
            return true;
        }
        return false;
    }

    bool isKinematicsBufferTensor(const nvinfer1::Dims& dims, int* history = nullptr, int* dim = nullptr) const {
        if (dims.nbDims == 2 && dims.d[0] > 0 && dims.d[1] > 0) {
            if (history) *history = dims.d[0];
            if (dim) *dim = dims.d[1];
            return true;
        }
        if (dims.nbDims == 3 && dims.d[0] == 1 && dims.d[1] > 0 && dims.d[2] > 0) {
            if (history) *history = dims.d[1];
            if (dim) *dim = dims.d[2];
            return true;
        }
        return false;
    }

    bool isScalarTensor(const nvinfer1::Dims& dims) const {
        return volume(dims) == 1;
    }

    bool isBBoxTensor(const nvinfer1::Dims& dims) const {
        return volume(dims) == 4;
    }

    int clampInt(int value, int lo, int hi) const {
        return std::max(lo, std::min(hi, value));
    }

    float clampFloat(float value, float lo, float hi) const {
        return std::max(lo, std::min(hi, value));
    }

    bool allocateBindings() {
        const int nb = trt_engine_->getNbIOTensors();
        if (nb <= 1) {
            logstream << "reframer: engine has insufficient bindings";
            return false;
        }

        io_tensor_names_.clear();
        tensor_bytes_.assign((size_t)nb, 0);
        tensor_ptrs_.assign((size_t)nb, 0);
        tensor_index_.clear();

        for (int i = 0; i < nb; ++i) {
            const char* tensor_name_c = trt_engine_->getIOTensorName(i);
            if (!tensor_name_c) {
                logstream << "reframer: null I/O tensor name";
                return false;
            }
            const std::string tensor_name = tensor_name_c;
            io_tensor_names_.push_back(tensor_name);
            tensor_index_[tensor_name] = (size_t)i;

            const nvinfer1::Dims dims = trt_engine_->getTensorShape(tensor_name_c);
            for (int d = 0; d < dims.nbDims; ++d) {
                if (dims.d[d] <= 0) {
                    logstream << "reframer: dynamic/invalid binding dims not supported in v1";
                    return false;
                }
            }
            const size_t vol = volume(dims);
            const size_t esz = elementSize(trt_engine_->getTensorDataType(tensor_name_c));
            if (vol == 0 || esz == 0) {
                logstream << "reframer: unsupported binding type/shape";
                return false;
            }

            CUdeviceptr ptr = 0;
            const size_t bytes = vol * esz;
            if (CHECK_CU(cuMemAlloc(&ptr, bytes))) {
                logstream << "reframer: cuMemAlloc failed for binding " << i;
                return false;
            }
            if (CHECK_CU(cuMemsetD8(ptr, 0, bytes))) {
                logstream << "reframer: cuMemsetD8 failed for binding " << i;
                return false;
            }
            tensor_bytes_[(size_t)i] = bytes;
            tensor_ptrs_[(size_t)i] = ptr;
        }

        auto resolveTensor = [&](const std::string& preferred_name,
                                 nvinfer1::TensorIOMode mode,
                                 auto predicate,
                                 std::string& tensor_name_out,
                                 nvinfer1::Dims& dims_out,
                                 nvinfer1::DataType& dtype_out) -> bool {
            tensor_name_out.clear();
            dims_out = nvinfer1::Dims{};

            auto pick = [&](const std::string& name) {
                const nvinfer1::Dims dims = trt_engine_->getTensorShape(name.c_str());
                if (trt_engine_->getTensorIOMode(name.c_str()) != mode) return false;
                if (!predicate(dims, name)) return false;
                tensor_name_out = name;
                dims_out = dims;
                dtype_out = trt_engine_->getTensorDataType(name.c_str());
                return true;
            };

            if (!preferred_name.empty()) {
                auto it = tensor_index_.find(preferred_name);
                if (it != tensor_index_.end() && pick(it->first)) {
                    return true;
                }
            }

            for (const std::string& name : io_tensor_names_) {
                if (pick(name)) return true;
            }
            return false;
        };

        if (!resolveTensor(visual_tensor_name_param_, nvinfer1::TensorIOMode::kINPUT,
                           [&](const nvinfer1::Dims& dims, const std::string& name) {
                               (void)name;
                               return isVisualTensor(dims);
                           },
                           visual_tensor_name_, visual_dims_, visual_dtype_)) {
            logstream << "reframer: failed to identify visual input tensor";
            return false;
        }

        if (!resolveTensor(kinematics_buffer_in_tensor_name_param_, nvinfer1::TensorIOMode::kINPUT,
                           [&](const nvinfer1::Dims& dims, const std::string& name) {
                               return isKinematicsBufferTensor(dims) && containsToken(name, "kinematics") && containsToken(name, "buffer");
                           },
                           kinematics_buffer_in_tensor_name_, kinematics_buffer_in_dims_, kinematics_buffer_in_dtype_)) {
            logstream << "reframer: failed to identify kinematics_buffer input tensor";
            return false;
        }

        if (!resolveTensor(viewport_center_x_in_tensor_name_param_, nvinfer1::TensorIOMode::kINPUT,
                           [&](const nvinfer1::Dims& dims, const std::string& name) {
                               return isScalarTensor(dims) && containsToken(name, "viewport") && containsToken(name, "center");
                           },
                           viewport_center_x_in_tensor_name_, viewport_center_x_in_dims_, viewport_center_x_in_dtype_)) {
            logstream << "reframer: failed to identify viewport_center_x input tensor";
            return false;
        }

        if (!resolveTensor(prev_velocity_in_tensor_name_param_, nvinfer1::TensorIOMode::kINPUT,
                           [&](const nvinfer1::Dims& dims, const std::string& name) {
                               return isScalarTensor(dims) && containsToken(name, "prev") && containsToken(name, "velocity");
                           },
                           prev_velocity_in_tensor_name_, prev_velocity_in_dims_, prev_velocity_in_dtype_)) {
            logstream << "reframer: failed to identify prev_velocity input tensor";
            return false;
        }

        if (!resolveTensor(bbox_norm_tensor_name_param_, nvinfer1::TensorIOMode::kOUTPUT,
                           [&](const nvinfer1::Dims& dims, const std::string& name) {
                               return isBBoxTensor(dims) && containsToken(name, "bbox");
                           },
                           bbox_norm_tensor_name_, bbox_norm_dims_, bbox_norm_dtype_)) {
            logstream << "reframer: failed to identify bbox_norm output tensor";
            return false;
        }

        if (!resolveTensor(kinematics_buffer_out_tensor_name_param_, nvinfer1::TensorIOMode::kOUTPUT,
                           [&](const nvinfer1::Dims& dims, const std::string& name) {
                               return isKinematicsBufferTensor(dims) && containsToken(name, "kinematics") && containsToken(name, "buffer");
                           },
                           kinematics_buffer_out_tensor_name_, kinematics_buffer_out_dims_, kinematics_buffer_out_dtype_)) {
            logstream << "reframer: failed to identify kinematics_buffer output tensor";
            return false;
        }

        if (!resolveTensor(viewport_center_x_out_tensor_name_param_, nvinfer1::TensorIOMode::kOUTPUT,
                           [&](const nvinfer1::Dims& dims, const std::string& name) {
                               return isScalarTensor(dims) && containsToken(name, "viewport") && containsToken(name, "center");
                           },
                           viewport_center_x_out_tensor_name_, viewport_center_x_out_dims_, viewport_center_x_out_dtype_)) {
            logstream << "reframer: failed to identify viewport_center_x output tensor";
            return false;
        }

        if (!resolveTensor(prev_velocity_out_tensor_name_param_, nvinfer1::TensorIOMode::kOUTPUT,
                           [&](const nvinfer1::Dims& dims, const std::string& name) {
                               return isScalarTensor(dims) && containsToken(name, "prev") && containsToken(name, "velocity");
                           },
                           prev_velocity_out_tensor_name_, prev_velocity_out_dims_, prev_velocity_out_dtype_)) {
            logstream << "reframer: failed to identify prev_velocity output tensor";
            return false;
        }

        int kinematics_history_in = 0;
        int kinematics_history_out = 0;
        int kinematics_dim_in = 0;
        int kinematics_dim_out = 0;
        if (!isVisualTensor(visual_dims_, &model_history_length_, &visual_channels_, &input_h_, &input_w_)) {
            logstream << "reframer: unsupported visual tensor layout";
            return false;
        }
        if (!isKinematicsBufferTensor(kinematics_buffer_in_dims_, &kinematics_history_in, &kinematics_dim_in)) {
            logstream << "reframer: unsupported kinematics_buffer input tensor layout";
            return false;
        }
        if (!isKinematicsBufferTensor(kinematics_buffer_out_dims_, &kinematics_history_out, &kinematics_dim_out)) {
            logstream << "reframer: unsupported kinematics_buffer output tensor layout";
            return false;
        }
        if (kinematics_history_in != model_history_length_ || kinematics_history_out != model_history_length_) {
            logstream << "reframer: history length mismatch across visual and kinematics tensors";
            return false;
        }
        if (kinematics_dim_in != kinematics_dim_out) {
            logstream << "reframer: kinematics_buffer input/output dimensions do not match";
            return false;
        }
        if (history_length_ <= 0) history_length_ = model_history_length_;
        if (history_length_ != model_history_length_) {
            std::ostringstream ss;
            ss << "reframer: history_length parameter (" << history_length_
               << ") does not match engine visual history length (" << model_history_length_ << ")";
            throw Error(ss.str());
        }
        if (visual_channels_ != 3) {
            throw Error("reframer: visual tensor must be RGB with 3 channels");
        }
        if (!isScalarTensor(viewport_center_x_in_dims_) || !isScalarTensor(prev_velocity_in_dims_)
                || !isScalarTensor(viewport_center_x_out_dims_) || !isScalarTensor(prev_velocity_out_dims_)) {
            logstream << "reframer: viewport_center_x and prev_velocity tensors must be scalar";
            return false;
        }
        if (!isBBoxTensor(bbox_norm_dims_)) {
            logstream << "reframer: bbox_norm tensor must contain exactly 4 values";
            return false;
        }

        auto validateFloatish = [&](nvinfer1::DataType dt, const char* label) {
            if (dt == nvinfer1::DataType::kFLOAT || dt == nvinfer1::DataType::kHALF) return true;
            logstream << "reframer: " << label << " datatype must be float/half";
            return false;
        };
        auto validateVisual = [&](nvinfer1::DataType dt) {
            if (dt == nvinfer1::DataType::kUINT8 || dt == nvinfer1::DataType::kFLOAT || dt == nvinfer1::DataType::kHALF) return true;
            logstream << "reframer: visual datatype must be uint8/float/half";
            return false;
        };
        if (!validateVisual(visual_dtype_)
                || !validateFloatish(kinematics_buffer_in_dtype_, "kinematics_buffer_in")
                || !validateFloatish(viewport_center_x_in_dtype_, "viewport_center_x_in")
                || !validateFloatish(prev_velocity_in_dtype_, "prev_velocity_in")
                || !validateFloatish(bbox_norm_dtype_, "bbox_norm")
                || !validateFloatish(kinematics_buffer_out_dtype_, "kinematics_buffer_out")
                || !validateFloatish(viewport_center_x_out_dtype_, "viewport_center_x_out")
                || !validateFloatish(prev_velocity_out_dtype_, "prev_velocity_out")) {
            return false;
        }

        kinematics_dim_ = kinematics_dim_in;

        host_kinematics_buffer_in_.resize(volume(kinematics_buffer_in_dims_));
        host_viewport_center_x_in_.resize(volume(viewport_center_x_in_dims_));
        host_prev_velocity_in_.resize(volume(prev_velocity_in_dims_));
        host_kinematics_buffer_out_.resize(volume(kinematics_buffer_out_dims_));
        host_viewport_center_x_out_.resize(volume(viewport_center_x_out_dims_));
        host_prev_velocity_out_.resize(volume(prev_velocity_out_dims_));
        host_bbox_norm_.resize(volume(bbox_norm_dims_));

        if (kinematics_buffer_in_dtype_ == nvinfer1::DataType::kHALF) host_kinematics_buffer_in_half_.resize(host_kinematics_buffer_in_.size());
        if (viewport_center_x_in_dtype_ == nvinfer1::DataType::kHALF) host_viewport_center_x_in_half_.resize(host_viewport_center_x_in_.size());
        if (prev_velocity_in_dtype_ == nvinfer1::DataType::kHALF) host_prev_velocity_in_half_.resize(host_prev_velocity_in_.size());
        if (kinematics_buffer_out_dtype_ == nvinfer1::DataType::kHALF) host_kinematics_buffer_out_half_.resize(host_kinematics_buffer_out_.size());
        if (viewport_center_x_out_dtype_ == nvinfer1::DataType::kHALF) host_viewport_center_x_out_half_.resize(host_viewport_center_x_out_.size());
        if (prev_velocity_out_dtype_ == nvinfer1::DataType::kHALF) host_prev_velocity_out_half_.resize(host_prev_velocity_out_.size());
        if (bbox_norm_dtype_ == nvinfer1::DataType::kHALF) host_bbox_norm_half_.resize(host_bbox_norm_.size());

        for (size_t i = 0; i < io_tensor_names_.size(); ++i) {
            if (!trt_ctx_->setTensorAddress(io_tensor_names_[i].c_str(), reinterpret_cast<void*>(tensor_ptrs_[i]))) {
                logstream << "reframer: setTensorAddress failed for " << io_tensor_names_[i];
                return false;
            }
        }

        return true;
    }

    bool loadPreprocessKernel() {
        if (preprocess_module_ && preprocess_rgb_kernel_) return true;
        if (!cu_ctx_) return false;
        if (CHECK_CU(cuCtxSetCurrent(cu_ctx_))) return false;
        const std::string ptx_str(avpl_amagi_reframer_ptx, avpl_amagi_reframer_ptx + avpl_amagi_reframer_ptx_len);
        if (CHECK_CU(cuModuleLoadDataEx(&preprocess_module_, (const void*)ptx_str.c_str(), 0, nullptr, nullptr))) {
            logstream << "reframer: failed to load preprocess PTX module";
            return false;
        }
        const char* rgb_name = nullptr;
        if (visual_dtype_ == nvinfer1::DataType::kHALF) {
            rgb_name = "kNV12_to_TCHW_rgb_fp16";
        } else if (visual_dtype_ == nvinfer1::DataType::kFLOAT) {
            rgb_name = "kNV12_to_TCHW_rgb_fp32";
        } else {
            rgb_name = "kNV12_to_TCHW_rgb_u8";
        }
        if (CHECK_CU(cuModuleGetFunction(&preprocess_rgb_kernel_, preprocess_module_, rgb_name))) {
            logstream << "reframer: failed to get RGB preprocess kernel";
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
        std::fill(host_kinematics_buffer_in_.begin(), host_kinematics_buffer_in_.end(), 0.0f);
        std::fill(host_viewport_center_x_in_.begin(), host_viewport_center_x_in_.end(), 0.0f);
        std::fill(host_prev_velocity_in_.begin(), host_prev_velocity_in_.end(), 0.0f);
        std::fill(host_kinematics_buffer_out_.begin(), host_kinematics_buffer_out_.end(), 0.0f);
        std::fill(host_viewport_center_x_out_.begin(), host_viewport_center_x_out_.end(), 0.0f);
        std::fill(host_prev_velocity_out_.begin(), host_prev_velocity_out_.end(), 0.0f);
        std::fill(host_bbox_norm_.begin(), host_bbox_norm_.end(), 0.0f);

        if (!host_kinematics_buffer_in_.empty() && history_length_ > 0 && kinematics_dim_ > 0) {
            const size_t base = (size_t)(history_length_ - 1) * (size_t)kinematics_dim_;
            if (base < host_kinematics_buffer_in_.size()) {
                host_kinematics_buffer_in_[base] = 0.5f;
            }
        }
        if (!host_viewport_center_x_in_.empty()) {
            host_viewport_center_x_in_[0] = full_frame_width_ > 0 ? (float)full_frame_width_ * 0.5f : 0.0f;
        }
        if (!host_prev_velocity_in_.empty()) {
            host_prev_velocity_in_[0] = 0.0f;
        }
    }

    bool historyReady() const {
        return (int)frame_history_.size() >= history_length_;
    }

    void pushObservation(const av::VideoFrame& frm) {
        frame_history_.push_back(frm);
        if ((int)frame_history_.size() > history_length_) frame_history_.pop_front();
    }

    bool copyHostFloatToDevice(const std::string& tensor_name,
                               const std::vector<float>& host_float,
                               std::vector<uint16_t>& host_half,
                               nvinfer1::DataType dtype,
                               const char* label) {
        auto it = tensor_index_.find(tensor_name);
        if (it == tensor_index_.end()) {
            logstream << "reframer: missing tensor index for " << label;
            return false;
        }
        const size_t tensor_idx = it->second;
        if (dtype == nvinfer1::DataType::kFLOAT) {
            const size_t bytes = host_float.size() * sizeof(float);
            if (bytes != tensor_bytes_[tensor_idx]) {
                logstream << "reframer: " << label << " size mismatch";
                return false;
            }
            if (CHECK_CU(cuMemcpyHtoDAsync(tensor_ptrs_[tensor_idx], host_float.data(), bytes, cuda_dev_ctx_->stream))) {
                logstream << "reframer: " << label << " H2D copy failed";
                return false;
            }
            return true;
        }

        if (host_half.size() != host_float.size()) {
            logstream << "reframer: " << label << " half buffer mismatch";
            return false;
        }
        for (size_t i = 0; i < host_float.size(); ++i) {
            host_half[i] = floatToHalf(host_float[i]);
        }
        const size_t bytes = host_half.size() * sizeof(uint16_t);
        if (bytes != tensor_bytes_[tensor_idx]) {
            logstream << "reframer: " << label << " half size mismatch";
            return false;
        }
        if (CHECK_CU(cuMemcpyHtoDAsync(tensor_ptrs_[tensor_idx], host_half.data(), bytes, cuda_dev_ctx_->stream))) {
            logstream << "reframer: " << label << " H2D copy failed";
            return false;
        }
        return true;
    }

    bool copyDeviceToHostFloat(const std::string& tensor_name,
                               std::vector<float>& host_float,
                               std::vector<uint16_t>& host_half,
                               nvinfer1::DataType dtype,
                               const char* label) {
        auto it = tensor_index_.find(tensor_name);
        if (it == tensor_index_.end()) {
            logstream << "reframer: missing tensor index for " << label;
            return false;
        }
        const size_t tensor_idx = it->second;
        if (dtype == nvinfer1::DataType::kFLOAT) {
            const size_t bytes = host_float.size() * sizeof(float);
            if (bytes != tensor_bytes_[tensor_idx]) {
                logstream << "reframer: " << label << " size mismatch";
                return false;
            }
            if (CHECK_CU(cuMemcpyDtoHAsync(host_float.data(), tensor_ptrs_[tensor_idx], bytes, cuda_dev_ctx_->stream))) {
                logstream << "reframer: " << label << " D2H copy failed";
                return false;
            }
            return true;
        }

        if (host_half.size() != host_float.size()) {
            logstream << "reframer: " << label << " half buffer mismatch";
            return false;
        }
        const size_t bytes = host_half.size() * sizeof(uint16_t);
        if (bytes != tensor_bytes_[tensor_idx]) {
            logstream << "reframer: " << label << " half size mismatch";
            return false;
        }
        if (CHECK_CU(cuMemcpyDtoHAsync(host_half.data(), tensor_ptrs_[tensor_idx], bytes, cuda_dev_ctx_->stream))) {
            logstream << "reframer: " << label << " D2H copy failed";
            return false;
        }
        return true;
    }

    void convertHalfOutputsToFloat(std::vector<float>& host_float,
                                   const std::vector<uint16_t>& host_half,
                                   nvinfer1::DataType dtype) {
        if (dtype != nvinfer1::DataType::kHALF) return;
        for (size_t i = 0; i < host_half.size() && i < host_float.size(); ++i) {
            host_float[i] = halfToFloat(host_half[i]);
        }
    }

    bool runVisualPreprocess() {
        auto it = tensor_index_.find(visual_tensor_name_);
        if (it == tensor_index_.end()) {
            logstream << "reframer: visual tensor index missing";
            return false;
        }
        if ((int)frame_history_.size() < history_length_) {
            logstream << "reframer: history queue shorter than history_length";
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
                logstream << "reframer: history frame size mismatch";
                return false;
            }
            if (hwSwFormat(hist) != AV_PIX_FMT_NV12) {
                logstream << "reframer: history frame sw_format mismatch (expected NV12)";
                return false;
            }

            const CUdeviceptr dY = (CUdeviceptr)(uintptr_t)hist.raw()->data[0];
            const CUdeviceptr dUV = (CUdeviceptr)(uintptr_t)hist.raw()->data[1];
            const size_t pitchY = (size_t)hist.raw()->linesize[0];
            const size_t pitchUV = (size_t)hist.raw()->linesize[1];
            CUdeviceptr slot_ptr = tensor_ptrs_[tensor_idx] + (CUdeviceptr)((size_t)slot * frame_bytes);
            void* out = reinterpret_cast<void*>(slot_ptr);
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
                logstream << "reframer: RGB preprocess kernel launch failed";
                return false;
            }
        }

        return true;
    }

    ViewportBBox decodeBBox() {
        std::array<float, 4> bbox_norm = {0.0f, 0.0f, 1.0f, 1.0f};
        for (size_t i = 0; i < bbox_norm.size() && i < host_bbox_norm_.size(); ++i) {
            bbox_norm[i] = clampFloat(host_bbox_norm_[i], 0.0f, 1.0f);
        }

        ViewportBBox bbox;
        bbox.x1 = clampInt((int)std::lround(bbox_norm[0] * (float)full_frame_width_), 0, full_frame_width_);
        bbox.y1 = clampInt((int)std::lround(bbox_norm[1] * (float)full_frame_height_), 0, full_frame_height_);
        bbox.x2 = clampInt((int)std::lround(bbox_norm[2] * (float)full_frame_width_), 0, full_frame_width_);
        bbox.y2 = clampInt((int)std::lround(bbox_norm[3] * (float)full_frame_height_), 0, full_frame_height_);
        if (bbox.x2 < bbox.x1) std::swap(bbox.x1, bbox.x2);
        if (bbox.y2 < bbox.y1) std::swap(bbox.y1, bbox.y2);
        return bbox;
    }

    Parameters buildMetadata(const ViewportBBox& bbox, float latency_ms) const {
        Parameters j;
        j["version"] = 1;
        j["bbox_norm"] = {host_bbox_norm_[0], host_bbox_norm_[1], host_bbox_norm_[2], host_bbox_norm_[3]};
        j["viewport_bbox"] = {bbox.x1, bbox.y1, bbox.x2, bbox.y2};
        j["viewport_center_x"] = (int)std::lround(((host_bbox_norm_[0] + host_bbox_norm_[2]) * 0.5f) * (float)full_frame_width_);
        j["full_frame_width"] = full_frame_width_;
        j["full_frame_height"] = full_frame_height_;
        j["latency_ms"] = latency_ms;
        return j;
    }

    void attachMetadata(av::VideoFrame& frm, const Parameters& j) const {
        const std::string md = j.dump();
        av_dict_set(&frm.raw()->metadata, metadata_key_out_.c_str(), md.c_str(), 0);
    }

    void maybeLogWarmup(size_t history_size, float latency_ms) const {
        if (debug_log_every_n_ <= 0) return;
        if ((frame_counter_ % (uint64_t)debug_log_every_n_) != 0) return;
        logstream << "reframer: frame=" << frame_counter_
                  << " warmup history=" << history_size << "/" << history_length_
                  << " latency_ms=" << latency_ms;
    }

    void maybeLogFrame(const ViewportBBox& bbox, float latency_ms) const {
        if (debug_log_every_n_ <= 0) return;
        if ((frame_counter_ % (uint64_t)debug_log_every_n_) != 0) return;
        logstream << "reframer: frame=" << frame_counter_
                  << " bbox=[" << bbox.x1 << "," << bbox.y1 << "," << bbox.x2 << "," << bbox.y2 << "]"
                  << " latency_ms=" << latency_ms;
    }

public:
    using NodeSISO::NodeSISO;

    ~Reframer() override {
        for (CUdeviceptr p : tensor_ptrs_) {
            if (p) CHECK_CU(cuMemFree(p));
        }
        tensor_ptrs_.clear();
        tensor_bytes_.clear();
        if (preprocess_module_) CHECK_CU(cuModuleUnload(preprocess_module_));
        preprocess_module_ = nullptr;
        preprocess_rgb_kernel_ = nullptr;
        delete trt_ctx_;
        delete trt_engine_;
        delete trt_runtime_;
        trt_ctx_ = nullptr;
        trt_engine_ = nullptr;
        trt_runtime_ = nullptr;
    }

    void resetInput() override {
        resetState();
    }

    void process() override {
        av::VideoFrame frm = this->source_->get();
        if (isEofMarker(frm)) {
            this->sink_->put(frm);
            return;
        }
        if (!frm) return;

        ++frame_counter_;
        auto process_start = std::chrono::steady_clock::now();

        if (frm.raw()->format != AV_PIX_FMT_CUDA) {
            logstream << "reframer: non-CUDA frame, passing through";
            resetState();
            this->sink_->put(frm);
            return;
        }
        if (!ensureInitialized(frm)) {
            this->sink_->put(frm);
            return;
        }
        if (frm.width() != input_w_ || frm.height() != input_h_) {
            logstream << "reframer: input frame size mismatch, expected " << input_w_ << "x" << input_h_
                      << " got " << frm.width() << "x" << frm.height();
            resetState();
            this->sink_->put(frm);
            return;
        }
        if (hwSwFormat(frm) != AV_PIX_FMT_NV12) {
            logstream << "reframer: unsupported hw sw_format (expected NV12)";
            resetState();
            this->sink_->put(frm);
            return;
        }
        if (CHECK_CU(cuCtxSetCurrent(cu_ctx_))) {
            logstream << "reframer: cuCtxSetCurrent failed in process";
            this->sink_->put(frm);
            return;
        }

        pushObservation(frm);
        if (!historyReady()) {
            const float latency_ms = (float)std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::steady_clock::now() - process_start).count() / 1000.0f;
            maybeLogWarmup(frame_history_.size(), latency_ms);
            this->sink_->put(frm);
            return;
        }

        if (!runVisualPreprocess()) {
            this->sink_->put(frm);
            return;
        }
        if (!copyHostFloatToDevice(kinematics_buffer_in_tensor_name_, host_kinematics_buffer_in_, host_kinematics_buffer_in_half_,
                                   kinematics_buffer_in_dtype_, "kinematics_buffer_in")) {
            this->sink_->put(frm);
            return;
        }
        if (!copyHostFloatToDevice(viewport_center_x_in_tensor_name_, host_viewport_center_x_in_, host_viewport_center_x_in_half_,
                                   viewport_center_x_in_dtype_, "viewport_center_x_in")) {
            this->sink_->put(frm);
            return;
        }
        if (!copyHostFloatToDevice(prev_velocity_in_tensor_name_, host_prev_velocity_in_, host_prev_velocity_in_half_,
                                   prev_velocity_in_dtype_, "prev_velocity_in")) {
            this->sink_->put(frm);
            return;
        }

        if (!trt_ctx_->enqueueV3(reinterpret_cast<cudaStream_t>(cuda_dev_ctx_->stream))) {
            logstream << "reframer: enqueueV3 failed";
            this->sink_->put(frm);
            return;
        }

        if (!copyDeviceToHostFloat(bbox_norm_tensor_name_, host_bbox_norm_, host_bbox_norm_half_, bbox_norm_dtype_, "bbox_norm")
                || !copyDeviceToHostFloat(kinematics_buffer_out_tensor_name_, host_kinematics_buffer_out_, host_kinematics_buffer_out_half_,
                                         kinematics_buffer_out_dtype_, "kinematics_buffer_out")
                || !copyDeviceToHostFloat(viewport_center_x_out_tensor_name_, host_viewport_center_x_out_, host_viewport_center_x_out_half_,
                                         viewport_center_x_out_dtype_, "viewport_center_x_out")
                || !copyDeviceToHostFloat(prev_velocity_out_tensor_name_, host_prev_velocity_out_, host_prev_velocity_out_half_,
                                         prev_velocity_out_dtype_, "prev_velocity_out")) {
            this->sink_->put(frm);
            return;
        }

        if (CHECK_CU(cuStreamSynchronize(cuda_dev_ctx_->stream))) {
            logstream << "reframer: stream sync failed";
            this->sink_->put(frm);
            return;
        }

        convertHalfOutputsToFloat(host_bbox_norm_, host_bbox_norm_half_, bbox_norm_dtype_);
        convertHalfOutputsToFloat(host_kinematics_buffer_out_, host_kinematics_buffer_out_half_, kinematics_buffer_out_dtype_);
        convertHalfOutputsToFloat(host_viewport_center_x_out_, host_viewport_center_x_out_half_, viewport_center_x_out_dtype_);
        convertHalfOutputsToFloat(host_prev_velocity_out_, host_prev_velocity_out_half_, prev_velocity_out_dtype_);

        host_kinematics_buffer_in_ = host_kinematics_buffer_out_;
        host_viewport_center_x_in_ = host_viewport_center_x_out_;
        host_prev_velocity_in_ = host_prev_velocity_out_;

        ViewportBBox bbox = decodeBBox();
        const float latency_ms = (float)std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - process_start).count() / 1000.0f;
        attachMetadata(frm, buildMetadata(bbox, latency_ms));
        maybeLogFrame(bbox, latency_ms);
        this->sink_->put(frm);
    }

    static std::shared_ptr<Reframer> create(NodeCreationInfo &nci) {
        EdgeManager &edges = nci.edges;
        const Parameters &params = nci.params;
        std::shared_ptr<Edge<av::VideoFrame>> src = edges.find<av::VideoFrame>(params["src"]);
        std::shared_ptr<Edge<av::VideoFrame>> dst = edges.find<av::VideoFrame>(params["dst"]);
        auto r = std::make_shared<Reframer>(
            make_unique<EdgeSource<av::VideoFrame>>(src),
            make_unique<EdgeSink<av::VideoFrame>>(dst)
        );
        src->setConsumer(r);
        dst->setProducer(r);

        if (!params.count("engine")) {
            throw Error("reframer: missing required parameter: engine");
        }
        if (!params.count("hwaccel")) {
            throw Error("reframer: missing required parameter: hwaccel");
        }
        if (!params.count("full_frame_width")) {
            throw Error("reframer: missing required parameter: full_frame_width");
        }
        if (!params.count("full_frame_height")) {
            throw Error("reframer: missing required parameter: full_frame_height");
        }
        r->engine_path_ = params["engine"].get<std::string>();
        r->hwaccel_ = InstanceSharedObjects<HWAccelDevice>::get(nci.instance, params["hwaccel"]);
        if (!r->hwaccel_) {
            throw Error("reframer: failed to get hwaccel");
        }

        if (params.count("metadata_key_out")) r->metadata_key_out_ = params["metadata_key_out"].get<std::string>();
        if (params.count("history_length")) r->history_length_ = params["history_length"];
        r->full_frame_width_ = params["full_frame_width"];
        r->full_frame_height_ = params["full_frame_height"];
        if (params.count("debug_log_every_n")) r->debug_log_every_n_ = params["debug_log_every_n"];
        if (params.count("visual_tensor_name")) r->visual_tensor_name_param_ = params["visual_tensor_name"].get<std::string>();
        if (params.count("kinematics_buffer_in_tensor_name")) r->kinematics_buffer_in_tensor_name_param_ = params["kinematics_buffer_in_tensor_name"].get<std::string>();
        if (params.count("viewport_center_x_in_tensor_name")) r->viewport_center_x_in_tensor_name_param_ = params["viewport_center_x_in_tensor_name"].get<std::string>();
        if (params.count("prev_velocity_in_tensor_name")) r->prev_velocity_in_tensor_name_param_ = params["prev_velocity_in_tensor_name"].get<std::string>();
        if (params.count("bbox_norm_tensor_name")) r->bbox_norm_tensor_name_param_ = params["bbox_norm_tensor_name"].get<std::string>();
        if (params.count("kinematics_buffer_out_tensor_name")) r->kinematics_buffer_out_tensor_name_param_ = params["kinematics_buffer_out_tensor_name"].get<std::string>();
        if (params.count("viewport_center_x_out_tensor_name")) r->viewport_center_x_out_tensor_name_param_ = params["viewport_center_x_out_tensor_name"].get<std::string>();
        if (params.count("prev_velocity_out_tensor_name")) r->prev_velocity_out_tensor_name_param_ = params["prev_velocity_out_tensor_name"].get<std::string>();
        if (params.count("input_format")) {
            const std::string ifmt = params["input_format"].get<std::string>();
            r->input_bgr_order_ = (ifmt == "BGR" || ifmt == "bgr");
        }
        if (r->history_length_ <= 0) {
            throw Error("reframer: history_length must be positive");
        }
        if (r->full_frame_width_ <= 0 || r->full_frame_height_ <= 0) {
            throw Error("reframer: full_frame_width and full_frame_height must be positive");
        }
        return r;
    }
};

DECLNODE(reframer, Reframer)
