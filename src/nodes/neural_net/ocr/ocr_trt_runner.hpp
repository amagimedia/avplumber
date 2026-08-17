#pragma once

#include "../common/infer_trt_base.hpp"

#include <NvInfer.h>

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace ocr {

struct TensorContract {
    std::string name;
    nvinfer1::DataType dtype = nvinfer1::DataType::kFLOAT;
    std::vector<int64_t> shape;
};

class TrtLogger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override;
};

class TrtRunner {
public:
    TrtRunner() = default;
    ~TrtRunner();
    TrtRunner(const TrtRunner&) = delete;
    TrtRunner& operator=(const TrtRunner&) = delete;

    void init(TrtLogger& logger,
              const std::string& engine_path,
              const TensorContract& input,
              const TensorContract& output);
    void cleanup() noexcept;
    void infer();

    CUdeviceptr inputPtr() const;
    CUstream stream() const { return stream_; }
    const std::vector<float>& output() const { return output_; }
    int inputDim(size_t index) const;
    const std::string& path() const { return path_; }

private:
    std::string path_;
    std::vector<std::string> tensor_names_;
    std::vector<CUdeviceptr> ptrs_;
    std::vector<size_t> bytes_;
    std::vector<float> output_;
    int input_index_ = -1;
    int output_index_ = -1;
    nvinfer1::Dims input_dims_{};
    nvinfer1::IRuntime* runtime_ = nullptr;
    nvinfer1::ICudaEngine* engine_ = nullptr;
    nvinfer1::IExecutionContext* ctx_ = nullptr;
    CUstream stream_ = nullptr;
};

} // namespace ocr
