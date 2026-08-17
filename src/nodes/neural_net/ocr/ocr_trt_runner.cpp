#include "ocr_trt_runner.hpp"

#include <fstream>
#include <sstream>
#include <utility>

namespace {

std::string dimsString(const nvinfer1::Dims& dims) {
    std::ostringstream out;
    for (int i = 0; i < dims.nbDims; ++i) {
        if (i) out << 'x';
        out << dims.d[i];
    }
    return out.str();
}

bool shapeMatches(const nvinfer1::Dims& actual, const std::vector<int64_t>& expected) {
    if (actual.nbDims != (int)expected.size()) return false;
    for (int i = 0; i < actual.nbDims; ++i) {
        if (actual.d[i] != expected[(size_t)i]) return false;
    }
    return true;
}

nvinfer1::Dims expectedDims(const std::vector<int64_t>& shape) {
    nvinfer1::Dims dims{};
    dims.nbDims = (int)shape.size();
    if (dims.nbDims > nvinfer1::Dims::MAX_DIMS) {
        throw Error("ocr TensorRT contract has too many dimensions");
    }
    for (int i = 0; i < dims.nbDims; ++i) dims.d[i] = shape[(size_t)i];
    return dims;
}

void requireCuda(CUresult result, const char* operation) {
    if (result == CUDA_SUCCESS) return;
    const char* name = nullptr;
    const char* description = nullptr;
    if (cuGetErrorName) cuGetErrorName(result, &name);
    if (cuGetErrorString) cuGetErrorString(result, &description);
    throw Error(std::string("ocr CUDA failure in ") + operation + ": " +
                (name ? name : "unknown") + " (" +
                (description ? description : "no description") + ")");
}

} // namespace

namespace ocr {

void TrtLogger::log(Severity severity, const char* msg) noexcept {
    if (severity <= Severity::kWARNING) {
        logstream << "ocr tensorrt: " << (msg ? msg : "");
    }
}

TrtRunner::~TrtRunner() {
    cleanup();
}

void TrtRunner::cleanup() noexcept {
    if (stream_) {
        cuStreamSynchronize(stream_);
        cuStreamDestroy(stream_);
        stream_ = nullptr;
    }
    for (CUdeviceptr ptr : ptrs_) {
        if (ptr) cuMemFree(ptr);
    }
    ptrs_.clear();
    bytes_.clear();
    tensor_names_.clear();
    output_.clear();
    input_index_ = -1;
    output_index_ = -1;
    input_dims_ = {};
    if (ctx_) { delete ctx_; ctx_ = nullptr; }
    if (engine_) { delete engine_; engine_ = nullptr; }
    if (runtime_) { delete runtime_; runtime_ = nullptr; }
    path_.clear();
}

void TrtRunner::init(TrtLogger& logger,
                     const std::string& engine_path,
                     const TensorContract& input,
                     const TensorContract& output) {
    cleanup();
    path_ = engine_path;

    std::ifstream file(path_, std::ios::binary);
    if (!file) throw Error("ocr: cannot open TensorRT engine " + path_);
    file.seekg(0, std::ios::end);
    const std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    if (size <= 0) throw Error("ocr: empty TensorRT engine " + path_);
    std::vector<char> blob((size_t)size);
    if (!file.read(blob.data(), size)) throw Error("ocr: failed reading TensorRT engine " + path_);

    runtime_ = nvinfer1::createInferRuntime(logger);
    if (!runtime_) throw Error("ocr: createInferRuntime failed for " + path_);
    engine_ = runtime_->deserializeCudaEngine(blob.data(), blob.size());
    if (!engine_) throw Error("ocr: deserializeCudaEngine failed for " + path_);
    ctx_ = engine_->createExecutionContext();
    if (!ctx_) throw Error("ocr: createExecutionContext failed for " + path_);
    requireCuda(cuStreamCreate(&stream_, 0), "cuStreamCreate");

    const int tensor_count = engine_->getNbIOTensors();
    if (tensor_count != 2) {
        throw Error("ocr: engine must have exactly one input and one output: " + path_);
    }
    ptrs_.assign((size_t)tensor_count, 0);
    bytes_.assign((size_t)tensor_count, 0);
    tensor_names_.reserve((size_t)tensor_count);

    for (int i = 0; i < tensor_count; ++i) {
        const std::string name = engine_->getIOTensorName(i);
        tensor_names_.push_back(name);
        const auto mode = engine_->getTensorIOMode(name.c_str());
        if (mode == nvinfer1::TensorIOMode::kINPUT) {
            if (input_index_ >= 0) throw Error("ocr: engine has multiple inputs: " + path_);
            input_index_ = i;
            if (!input.name.empty() && name != input.name) {
                throw Error("ocr: input binding mismatch for " + path_ + ": expected " +
                            input.name + ", got " + name);
            }
            if (engine_->getTensorDataType(name.c_str()) != input.dtype) {
                throw Error("ocr: input dtype mismatch for " + path_ + " tensor=" + name);
            }
            const nvinfer1::Dims dims = expectedDims(input.shape);
            if (!ctx_->setInputShape(name.c_str(), dims)) {
                throw Error("ocr: setInputShape failed for " + path_ + " tensor=" + name);
            }
        } else if (mode == nvinfer1::TensorIOMode::kOUTPUT) {
            if (output_index_ >= 0) throw Error("ocr: engine has multiple outputs: " + path_);
            output_index_ = i;
            if (!output.name.empty() && name != output.name) {
                throw Error("ocr: output binding mismatch for " + path_ + ": expected " +
                            output.name + ", got " + name);
            }
            if (engine_->getTensorDataType(name.c_str()) != output.dtype) {
                throw Error("ocr: output dtype mismatch for " + path_ + " tensor=" + name);
            }
        }
    }
    if (input_index_ < 0 || output_index_ < 0) {
        throw Error("ocr: engine input/output binding missing: " + path_);
    }

    input_dims_ = ctx_->getTensorShape(tensor_names_[(size_t)input_index_].c_str());
    const nvinfer1::Dims output_dims = ctx_->getTensorShape(tensor_names_[(size_t)output_index_].c_str());
    if (!shapeMatches(input_dims_, input.shape)) {
        throw Error("ocr: input shape mismatch for " + path_ + ": expected " +
                    dimsString(expectedDims(input.shape)) + ", got " + dimsString(input_dims_));
    }
    if (!output.shape.empty() && !shapeMatches(output_dims, output.shape)) {
        throw Error("ocr: output shape mismatch for " + path_ + ": expected " +
                    dimsString(expectedDims(output.shape)) + ", got " + dimsString(output_dims));
    }

    for (int i = 0; i < tensor_count; ++i) {
        const std::string& name = tensor_names_[(size_t)i];
        const nvinfer1::Dims dims = ctx_->getTensorShape(name.c_str());
        const size_t volume = yolo_base::volume(dims);
        const size_t element_size = yolo_base::elementSize(engine_->getTensorDataType(name.c_str()));
        if (!volume || !element_size) {
            throw Error("ocr: unsupported tensor shape or dtype for " + path_ + " tensor=" + name);
        }
        bytes_[(size_t)i] = volume * element_size;
        requireCuda(cuMemAlloc(&ptrs_[(size_t)i], bytes_[(size_t)i]), "cuMemAlloc");
        if (!ctx_->setTensorAddress(name.c_str(), reinterpret_cast<void*>(ptrs_[(size_t)i]))) {
            throw Error("ocr: setTensorAddress failed for " + path_ + " tensor=" + name);
        }
    }
    output_.assign(yolo_base::volume(output_dims), 0.0f);
    logstream << "ocr: loaded engine=" << path_ << " input=" << dimsString(input_dims_)
              << " output=" << dimsString(output_dims);
}

CUdeviceptr TrtRunner::inputPtr() const {
    if (input_index_ < 0) throw Error("ocr: TensorRT input requested before initialization");
    return ptrs_[(size_t)input_index_];
}

int TrtRunner::inputDim(size_t index) const {
    if (index >= (size_t)input_dims_.nbDims) throw Error("ocr: TensorRT input dimension out of range");
    return input_dims_.d[index];
}

void TrtRunner::infer() {
    if (!ctx_ || output_index_ < 0) throw Error("ocr: TensorRT inference before initialization");
    if (!ctx_->enqueueV3(reinterpret_cast<cudaStream_t>(stream_))) {
        throw Error("ocr: TensorRT enqueue failed for " + path_);
    }
    requireCuda(cuMemcpyDtoHAsync(output_.data(), ptrs_[(size_t)output_index_],
                                  bytes_[(size_t)output_index_], stream_),
                "cuMemcpyDtoHAsync");
    requireCuda(cuStreamSynchronize(stream_), "cuStreamSynchronize");
}

} // namespace ocr
