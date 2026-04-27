#pragma once

#include "../../node_common.hpp"
#include "../../../hwaccel.hpp"
#include <cuda_loader/cuda_drvapi_dynlink_cuda.h>

extern "C" {
#include <libavutil/dict.h>
#include <libavutil/hwcontext.h>
#include <libavutil/hwcontext_cuda.h>
#include <libavutil/buffer.h>
#include <libavutil/frame.h>
}

#include <NvInfer.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "yolo_side_data.hpp"

namespace yolo_base {

// --- CUDA error checking ---
inline int check_cu(CUresult err, const char *func) {
    if (err == CUDA_SUCCESS) return 0;
    const char *err_name = nullptr;
    const char *err_string = nullptr;
    if (cuGetErrorName && cuGetErrorString) {
        cuGetErrorName(err, &err_name);
        cuGetErrorString(err, &err_string);
    }
    logstream << "cuda function: " << func << " failed: "
              << (err_name ? err_name : "?") << ": " << (err_string ? err_string : "?");
    return -1;
}
#define CUDA_CHECK_CU(x) yolo_base::check_cu((x), #x)

// --- TensorRT logger ---
class TRTLogger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity == Severity::kERROR || severity == Severity::kINTERNAL_ERROR || severity == Severity::kWARNING) {
            logstream << "tensorrt: " << (msg ? msg : "");
        }
    }
};

// --- Enums ---
enum class TaskType { Detection, Segmentation, Pose };
enum class OutputBoxFormat { EndToEndXYXY, RawCXCYWH };

// --- Detection struct ---
struct Detection {
    float x1 = 0.0f, y1 = 0.0f, x2 = 0.0f, y2 = 0.0f;
    float conf = 0.0f;
    int cls = -1;
    int model_index = -1;
};

// --- Decode results ---
struct DetectionResult {
    std::vector<Detection> detections;
};

struct SegmentationResult : DetectionResult {
    AVBufferRef* gpu_mask_buf = nullptr;
    int mask_proto_w = 0, mask_proto_h = 0;
    int num_masks = 0;
    std::vector<float> cpu_masks;
    int cpu_mask_w = 0, cpu_mask_h = 0;
};

struct PoseResult : DetectionResult {
    std::vector<float> keypoints;  // flat [x, y, conf, x, y, conf, ...] per detection
    int num_keypoints = 0;         // keypoints per detection (e.g. 34)
};

struct DecodeParams {
    int model_index;
    float conf_thresh;
    OutputBoxFormat box_format;
    const std::vector<int>& class_index_remap;
};

// --- Utility functions ---
inline float halfToFloat(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1Fu;
    uint32_t frac = h & 0x03FFu;
    uint32_t out = 0;
    if (exp == 0) {
        if (frac == 0) {
            out = sign;
        } else {
            exp = 1;
            while ((frac & 0x0400u) == 0) { frac <<= 1; --exp; }
            frac &= 0x03FFu;
            out = sign | ((exp + (127 - 15)) << 23) | (frac << 13);
        }
    } else if (exp == 0x1Fu) {
        out = sign | 0x7F800000u | (frac << 13);
    } else {
        out = sign | ((exp + (127 - 15)) << 23) | (frac << 13);
    }
    float f;
    memcpy(&f, &out, sizeof(float));
    return f;
}

inline size_t elementSize(nvinfer1::DataType dt) {
    switch (dt) {
        case nvinfer1::DataType::kFLOAT: return 4;
        case nvinfer1::DataType::kHALF: return 2;
        case nvinfer1::DataType::kINT8: return 1;
        case nvinfer1::DataType::kINT32: return 4;
        case nvinfer1::DataType::kINT64: return 8;
        case nvinfer1::DataType::kBOOL: return 1;
        default: return 0;
    }
}

inline size_t volume(const nvinfer1::Dims& d) {
    size_t v = 1;
    for (int i = 0; i < d.nbDims; ++i) {
        if (d.d[i] <= 0) return 0;
        v *= (size_t)d.d[i];
    }
    return v;
}

// Forward declarations for decoders
class DetectionDecoder;
class SegmentationDecoder;
class PoseDecoder;
struct DetectionDecoderDeleter {
    void operator()(DetectionDecoder* p) const;
};
struct SegmentationDecoderDeleter {
    void operator()(SegmentationDecoder* p) const;
};
struct PoseDecoderDeleter {
    void operator()(PoseDecoder* p) const;
};

// --- Output tensor info (supports multiple output tensors per model) ---
struct OutputTensor {
    std::string name;
    nvinfer1::Dims dims{};
    nvinfer1::DataType dtype = nvinfer1::DataType::kFLOAT;
    size_t tensor_index = 0;  // index into tensor_ptrs/tensor_bytes
    std::vector<float> host_output;
    std::vector<uint16_t> host_output_half;
    std::vector<int32_t> host_output_i32;
    std::vector<int64_t> host_output_i64;
};

// --- ModelRunner ---
struct ModelRunner {
    ModelRunner();
    ~ModelRunner();
    ModelRunner(const ModelRunner&) = delete;
    ModelRunner& operator=(const ModelRunner&) = delete;
    ModelRunner(ModelRunner&&) noexcept;
    ModelRunner& operator=(ModelRunner&&) noexcept;

    std::string engine_path;
    std::string engine_name;

    // TRT objects
    nvinfer1::IRuntime* trt_runtime = nullptr;
    nvinfer1::ICudaEngine* trt_engine = nullptr;
    nvinfer1::IExecutionContext* trt_ctx = nullptr;

    // IO tensor management
    std::vector<std::string> io_tensor_names;
    std::vector<size_t> tensor_bytes;
    std::vector<CUdeviceptr> tensor_ptrs;
    std::unordered_map<std::string, size_t> tensor_index;

    // Input
    std::string input_tensor_name;
    nvinfer1::Dims input_dims{};
    nvinfer1::DataType input_dtype = nvinfer1::DataType::kFLOAT;
    int input_w = 0, input_h = 0;

    // Outputs (vector: detection has 1, segmentation has 2)
    std::vector<OutputTensor> outputs;

    // Preprocess
    CUfunction preprocess_kernel = nullptr;
    CUstream stream = nullptr;

    // Config
    OutputBoxFormat output_box_format = OutputBoxFormat::EndToEndXYXY;
    TaskType task_type = TaskType::Detection;
    bool include_in_detection_metadata = true;
    std::vector<std::string> class_names;
    std::vector<int> class_index_remap;

    // Decoder (only one is non-null)
    std::unique_ptr<DetectionDecoder, DetectionDecoderDeleter> det_decoder;
    std::unique_ptr<SegmentationDecoder, SegmentationDecoderDeleter> seg_decoder;
    std::unique_ptr<PoseDecoder, PoseDecoderDeleter> pose_decoder;
    int num_classes = -1;  // -1 = auto-detect; required for pose to disambiguate class scores from keypoints
};

// --- CudaInferTrtBase ---
class CudaInferTrtBase {
protected:
    AVCUDADeviceContext* cuda_dev_ctx_ = nullptr;
    CUcontext cu_ctx_ = nullptr;
    TRTLogger trt_logger_;
    std::vector<ModelRunner> models_;
    int input_w_ = 0, input_h_ = 0;
    nvinfer1::DataType input_dtype_ = nvinfer1::DataType::kFLOAT;
    bool input_bgr_order_ = false;
    CUmodule preprocess_module_ = nullptr;
    bool initialized_ = false;

    // Cached metadata JSON fragment for static model info
    std::string cached_models_json_;

    bool initCudaContextFromFrame(const av::VideoFrame& frm);
    bool loadPreprocessModule();
    bool parseEngine(ModelRunner& model);
    bool allocateBindings(ModelRunner& model);
    bool ensureCompatibleInput(const ModelRunner& model, size_t model_index);
    bool configureRunnerPreprocess(ModelRunner& model);
    bool ensureInitialized(const av::VideoFrame& frm);

    bool runPreprocessNV12(const av::VideoFrame& frm, ModelRunner& model);
    bool runInference(ModelRunner& model);
    bool syncModel(ModelRunner& model);

    void cleanupModel(ModelRunner& model);
    void cleanupAll();

    AVPixelFormat hwSwFormat(const av::VideoFrame& frm) const;

public:
    virtual ~CudaInferTrtBase();
};

} // namespace yolo_base
