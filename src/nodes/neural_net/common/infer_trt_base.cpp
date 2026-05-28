#include "infer_trt_base.hpp"
#include "../yolo/decode_detection.hpp"
#include "../yolo/decode_segmentation.hpp"
#include "../yolo/decode_pose.hpp"
#include <mutex>

// PTX blob for NV12->NCHW preprocess kernel.
#include "../../../../objs/src/nodes/neural_net/preprocess/nv12_to_nchw.ptx.h"

namespace yolo_base {

namespace {
std::mutex g_cuda_graph_capture_mutex;
constexpr unsigned int kCudaStreamDefault = 0x0;
constexpr unsigned int kCudaStreamNonBlocking = 0x1;
}

void DetectionDecoderDeleter::operator()(DetectionDecoder* p) const { delete p; }
void SegmentationDecoderDeleter::operator()(SegmentationDecoder* p) const { delete p; }
void PoseDecoderDeleter::operator()(PoseDecoder* p) const { delete p; }

ModelRunner::ModelRunner() = default;
ModelRunner::~ModelRunner() = default;
ModelRunner::ModelRunner(ModelRunner&&) noexcept = default;
ModelRunner& ModelRunner::operator=(ModelRunner&&) noexcept = default;

CudaInferTrtBase::~CudaInferTrtBase() {
    cleanupAll();
}

void CudaInferTrtBase::cleanupAll() {
    if (cu_ctx_) {
        CUDA_CHECK_CU(cuCtxSetCurrent(cu_ctx_));
    }
    for (ModelRunner& model : models_) {
        cleanupModel(model);
    }
    if (preprocess_module_) {
        CUDA_CHECK_CU(cuModuleUnload(preprocess_module_));
        preprocess_module_ = nullptr;
    }
}

void CudaInferTrtBase::cleanupModel(ModelRunner& model) {
    // Sync stream before destroying anything
    if (model.stream) {
        CUDA_CHECK_CU(cuStreamSynchronize(model.stream));
        if (model.cuda_graph_exec && cuGraphExecDestroy) {
            CUDA_CHECK_CU(cuGraphExecDestroy(model.cuda_graph_exec));
            model.cuda_graph_exec = nullptr;
        }
        for (cudaStream_t aux_stream : model.aux_streams) {
            CUDA_CHECK_CU(cuStreamDestroy(reinterpret_cast<CUstream>(aux_stream)));
        }
        model.aux_streams.clear();
        CUDA_CHECK_CU(cuStreamDestroy(model.stream));
        model.stream = nullptr;
    }
    // Clean up decoder GPU resources
    if (model.seg_decoder) {
        model.seg_decoder->cleanup();
    }
    // Free GPU tensors
    for (CUdeviceptr ptr : model.tensor_ptrs) {
        if (ptr) CUDA_CHECK_CU(cuMemFree(ptr));
    }
    model.tensor_ptrs.clear();
    model.tensor_bytes.clear();
    model.io_tensor_names.clear();
    model.tensor_index.clear();
    model.outputs.clear();
    // Destroy TRT objects
    if (model.trt_ctx) { delete model.trt_ctx; model.trt_ctx = nullptr; }
    if (model.trt_engine) { delete model.trt_engine; model.trt_engine = nullptr; }
    if (model.trt_runtime) { delete model.trt_runtime; model.trt_runtime = nullptr; }
}

bool CudaInferTrtBase::initCudaContextFromFrame(const av::VideoFrame& frm) {
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
    if (CUDA_CHECK_CU(cuCtxSetCurrent(cu_ctx_))) {
        logstream << "cuda_infer_yolo: cuCtxSetCurrent failed";
        return false;
    }
    return true;
}

bool CudaInferTrtBase::loadPreprocessModule() {
    if (preprocess_module_) return true;
    if (!cu_ctx_) return false;
    if (CUDA_CHECK_CU(cuCtxSetCurrent(cu_ctx_))) return false;
    const std::string ptx_str(avpl_yolo_preprocess_ptx, avpl_yolo_preprocess_ptx + avpl_yolo_preprocess_ptx_len);
    if (CUDA_CHECK_CU(cuModuleLoadDataEx(&preprocess_module_, (const void*)ptx_str.c_str(), 0, nullptr, nullptr))) {
        logstream << "cuda_infer_yolo: failed to load preprocess PTX module";
        return false;
    }
    return true;
}

bool CudaInferTrtBase::parseEngine(ModelRunner& model) {
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

bool CudaInferTrtBase::allocateBindings(ModelRunner& model) {
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
    model.outputs.clear();
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
        const nvinfer1::DataType dt = model.trt_engine->getTensorDataType(tensor_name_c);
        const size_t esz = elementSize(dt);
        if (vol == 0 || esz == 0) {
            logstream << "cuda_infer_yolo: unsupported binding type/shape for " << model.engine_path;
            return false;
        }
        const size_t bytes = vol * esz;

        CUdeviceptr ptr = 0;
        if (CUDA_CHECK_CU(cuMemAlloc(&ptr, bytes))) {
            logstream << "cuda_infer_yolo: cuMemAlloc failed for " << model.engine_path << " binding " << i;
            return false;
        }
        if (CUDA_CHECK_CU(cuMemsetD8(ptr, 0, bytes))) {
            logstream << "cuda_infer_yolo: cuMemsetD8 failed for " << model.engine_path << " binding " << i;
            return false;
        }
        model.tensor_bytes[(size_t)i] = bytes;
        model.tensor_ptrs[(size_t)i] = ptr;

        if (is_input) {
            ++input_count;
            const int expected_c = expected_input_channels_;
            const bool is_image_input =
                (dims.nbDims == 3 && dims.d[0] == expected_c) ||
                (dims.nbDims == 4 && dims.d[0] == 1 && dims.d[1] == expected_c);
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
        } else {
            // Collect ALL output tensors (detection has 1, segmentation has 2)
            OutputTensor ot;
            ot.name = tensor_name;
            ot.dims = dims;
            ot.dtype = dt;
            ot.tensor_index = (size_t)i;
            if (dt == nvinfer1::DataType::kFLOAT || dt == nvinfer1::DataType::kHALF) {
                ot.host_output.resize(vol);
                if (dt == nvinfer1::DataType::kHALF) {
                    ot.host_output_half.resize(vol);
                }
            } else if (dt == nvinfer1::DataType::kINT32) {
                ot.host_output.resize(vol);
                ot.host_output_i32.resize(vol);
            } else if (dt == nvinfer1::DataType::kINT64) {
                ot.host_output.resize(vol);
                ot.host_output_i64.resize(vol);
            }
            model.outputs.push_back(std::move(ot));
        }
    }

    if (model.input_tensor_name.empty() || model.outputs.empty()) {
        logstream << "cuda_infer_yolo: failed to identify input/output bindings for " << model.engine_path;
        return false;
    }

    if (model.input_dims.nbDims == 3 && model.input_dims.d[0] == expected_input_channels_) {
        model.input_c = model.input_dims.d[0];
        model.input_h = model.input_dims.d[1];
        model.input_w = model.input_dims.d[2];
    } else if (model.input_dims.nbDims == 4 && model.input_dims.d[0] == 1 && model.input_dims.d[1] == expected_input_channels_) {
        model.input_c = model.input_dims.d[1];
        model.input_h = model.input_dims.d[2];
        model.input_w = model.input_dims.d[3];
    } else {
        logstream << "cuda_infer_yolo: expected CHW or NCHW input tensor with "
                  << expected_input_channels_ << " channels for " << model.engine_path
                  << " (engine inputs: " << input_count << ")";
        return false;
    }
    if (model.input_h <= 0 || model.input_w <= 0) {
        logstream << "cuda_infer_yolo: invalid input dims for " << model.engine_path;
        return false;
    }

    // Validate output dtypes
    for (const OutputTensor& ot : model.outputs) {
        if (!(ot.dtype == nvinfer1::DataType::kFLOAT ||
              ot.dtype == nvinfer1::DataType::kHALF ||
              ot.dtype == nvinfer1::DataType::kINT32 ||
              ot.dtype == nvinfer1::DataType::kINT64)) {
            logstream << "cuda_infer_yolo: output datatype must be float/half/int32/int64 for " << model.engine_path
                      << " tensor " << ot.name;
            return false;
        }
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

bool CudaInferTrtBase::ensureCompatibleInput(const ModelRunner& model, size_t model_index) {
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

bool CudaInferTrtBase::configureRunnerPreprocess(ModelRunner& model) {
    const char* kname = (model.input_dtype == nvinfer1::DataType::kHALF)
        ? "kNV12_to_NCHW_fp16"
        : "kNV12_to_NCHW_fp32";
    if (CUDA_CHECK_CU(cuModuleGetFunction(&model.preprocess_kernel, preprocess_module_, kname))) {
        logstream << "cuda_infer_yolo: failed to get preprocess kernel for " << model.engine_path;
        return false;
    }
    return configureRunnerStream(model);
}

bool CudaInferTrtBase::configureRunnerStream(ModelRunner& model) {
    unsigned int stream_flags = use_cuda_graph_ ? kCudaStreamNonBlocking : kCudaStreamDefault;
    if (CUDA_CHECK_CU(cuStreamCreate(&model.stream, stream_flags))) {
        logstream << "cuda_infer_yolo: failed to create CUDA stream for " << model.engine_path;
        return false;
    }
    if (use_cuda_graph_) {
        int nb_aux_streams = model.trt_engine->getNbAuxStreams();
        model.aux_streams.reserve((size_t)std::max(nb_aux_streams, 0));
        for (int i = 0; i < nb_aux_streams; ++i) {
            CUstream aux_stream = nullptr;
            if (CUDA_CHECK_CU(cuStreamCreate(&aux_stream, kCudaStreamNonBlocking))) {
                logstream << "cuda_infer_yolo: failed to create TensorRT aux stream for " << model.engine_path;
                return false;
            }
            model.aux_streams.push_back(reinterpret_cast<cudaStream_t>(aux_stream));
        }
    }
    return true;
}

bool CudaInferTrtBase::ensureInitialized(const av::VideoFrame& frm) {
    if (initialized_) return true;
    if (!initCudaContextFromFrame(frm)) return false;
    if (!loadPreprocessModule()) return false;
    for (size_t i = 0; i < models_.size(); ++i) {
        ModelRunner& model = models_[i];
        if (!parseEngine(model) || !allocateBindings(model) ||
            !ensureCompatibleInput(model, i) || !configureRunnerPreprocess(model)) {
            // Rollback: clean up models 0..i
            for (size_t j = 0; j <= i; ++j) {
                cleanupModel(models_[j]);
            }
            return false;
        }
    }
    initialized_ = true;
    return true;
}

AVPixelFormat CudaInferTrtBase::hwSwFormat(const av::VideoFrame& frm) const {
    if (!frm.raw() || !frm.raw()->hw_frames_ctx || !frm.raw()->hw_frames_ctx->data) return AV_PIX_FMT_NONE;
    AVHWFramesContext* ctx = (AVHWFramesContext*)frm.raw()->hw_frames_ctx->data;
    if (!ctx) return AV_PIX_FMT_NONE;
    return ctx->sw_format;
}

bool CudaInferTrtBase::runPreprocessNV12(const av::VideoFrame& frm, ModelRunner& model) {
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
    if (CUDA_CHECK_CU(cuLaunchKernel(model.preprocess_kernel, gridX, gridY, 1, blockX, blockY, 1, 0, model.stream, args, nullptr))) {
        logstream << "cuda_infer_yolo: preprocess kernel launch failed for " << model.engine_path;
        return false;
    }
    return true;
}

bool CudaInferTrtBase::enqueueInference(ModelRunner& model) {
    if (!model.aux_streams.empty()) {
        model.trt_ctx->setAuxStreams(model.aux_streams.data(), (int32_t)model.aux_streams.size());
    }
    if (!model.trt_ctx->enqueueV3(reinterpret_cast<cudaStream_t>(model.stream))) {
        logstream << "cuda_infer_yolo: enqueueV3 failed for " << model.engine_name;
        return false;
    }
    return true;
}

bool CudaInferTrtBase::copyOutputsToHost(ModelRunner& model) {
    for (OutputTensor& ot : model.outputs) {
        size_t idx = ot.tensor_index;
        size_t bytes = model.tensor_bytes[idx];
        void* dst = nullptr;
        if (ot.dtype == nvinfer1::DataType::kHALF) {
            dst = (void*)ot.host_output_half.data();
        } else if (ot.dtype == nvinfer1::DataType::kINT32) {
            dst = (void*)ot.host_output_i32.data();
        } else if (ot.dtype == nvinfer1::DataType::kINT64) {
            dst = (void*)ot.host_output_i64.data();
        } else {
            dst = (void*)ot.host_output.data();
        }
        if (CUDA_CHECK_CU(cuMemcpyDtoHAsync(dst, model.tensor_ptrs[idx], bytes, model.stream))) {
            return false;
        }
    }
    return true;
}

void CudaInferTrtBase::disableCudaGraph(ModelRunner& model, const std::string& reason) {
    model.cuda_graph_disabled = true;
    if (!model.cuda_graph_disable_logged) {
        logstream << "cuda_infer_yolo: CUDA graph disabled for " << model.engine_name
                  << ": " << reason << "; falling back to normal TensorRT enqueue";
        model.cuda_graph_disable_logged = true;
    }
}

bool CudaInferTrtBase::ensureCudaGraph(ModelRunner& model) {
    if (model.cuda_graph_ready) return true;
    if (model.cuda_graph_disabled) return false;

    if (!cuStreamBeginCapture || !cuStreamEndCapture || !cuGraphInstantiate ||
        !cuGraphLaunch || !cuGraphExecDestroy || !cuGraphDestroy) {
        disableCudaGraph(model, "CUDA graph driver APIs are unavailable");
        return false;
    }

    std::lock_guard<std::mutex> lock(g_cuda_graph_capture_mutex);
    if (model.cuda_graph_ready || model.cuda_graph_disabled) return model.cuda_graph_ready;

    if (CUDA_CHECK_CU(cuStreamSynchronize(model.stream))) {
        disableCudaGraph(model, "stream synchronize before capture failed");
        return false;
    }

    CUresult res = cuStreamBeginCapture(model.stream, CU_STREAM_CAPTURE_MODE_RELAXED);
    if (res != CUDA_SUCCESS) {
        disableCudaGraph(model, "cuStreamBeginCapture failed");
        return false;
    }

    CUgraph graph = nullptr;
    if (!model.aux_streams.empty()) {
        model.trt_ctx->setAuxStreams(model.aux_streams.data(), (int32_t)model.aux_streams.size());
    }
    if (!model.trt_ctx->enqueueV3(reinterpret_cast<cudaStream_t>(model.stream))) {
        CUresult end_res = cuStreamEndCapture(model.stream, &graph);
        if (end_res == CUDA_SUCCESS && graph) {
            CUDA_CHECK_CU(cuGraphDestroy(graph));
        }
        disableCudaGraph(model, "enqueueV3 failed during capture");
        return false;
    }

    res = cuStreamEndCapture(model.stream, &graph);
    if (res != CUDA_SUCCESS || !graph) {
        disableCudaGraph(model, "cuStreamEndCapture failed");
        return false;
    }

    CUgraphExec graph_exec = nullptr;
    CUgraphNode error_node = nullptr;
    char log_buffer[1024] = {};
    res = cuGraphInstantiate(&graph_exec, graph, &error_node, log_buffer, sizeof(log_buffer));
    CUDA_CHECK_CU(cuGraphDestroy(graph));
    if (res != CUDA_SUCCESS || !graph_exec) {
        std::string reason = "cuGraphInstantiate failed";
        if (log_buffer[0] != '\0') {
            reason += ": ";
            reason += log_buffer;
        }
        disableCudaGraph(model, reason);
        return false;
    }

    model.cuda_graph_exec = graph_exec;
    model.cuda_graph_ready = true;
    if (!model.cuda_graph_capture_logged) {
        logstream << "cuda_infer_yolo: CUDA graph captured for " << model.engine_name;
        model.cuda_graph_capture_logged = true;
    }
    return true;
}

bool CudaInferTrtBase::runCudaGraph(ModelRunner& model) {
    if (!model.cuda_graph_ready || !model.cuda_graph_exec) return false;
    CUresult res = cuGraphLaunch(model.cuda_graph_exec, model.stream);
    if (res != CUDA_SUCCESS) {
        disableCudaGraph(model, "cuGraphLaunch failed");
        return false;
    }
    return true;
}

bool CudaInferTrtBase::runInference(ModelRunner& model) {
    if (use_cuda_graph_ && !model.cuda_graph_disabled) {
        if (model.cuda_graph_warmup_remaining > 0) {
            --model.cuda_graph_warmup_remaining;
            return enqueueInference(model) && copyOutputsToHost(model);
        }
        if (!model.cuda_graph_ready) {
            ensureCudaGraph(model);
        }
        if (model.cuda_graph_ready && runCudaGraph(model)) {
            return copyOutputsToHost(model);
        }
    }

    return enqueueInference(model) && copyOutputsToHost(model);
}

bool CudaInferTrtBase::syncModel(ModelRunner& model) {
    if (CUDA_CHECK_CU(cuStreamSynchronize(model.stream))) return false;
    for (OutputTensor& ot : model.outputs) {
        if (ot.dtype == nvinfer1::DataType::kHALF) {
            for (size_t i = 0; i < ot.host_output_half.size(); ++i) {
                ot.host_output[i] = halfToFloat(ot.host_output_half[i]);
            }
        } else if (ot.dtype == nvinfer1::DataType::kINT32) {
            for (size_t i = 0; i < ot.host_output_i32.size(); ++i) {
                ot.host_output[i] = (float)ot.host_output_i32[i];
            }
        } else if (ot.dtype == nvinfer1::DataType::kINT64) {
            for (size_t i = 0; i < ot.host_output_i64.size(); ++i) {
                ot.host_output[i] = (float)ot.host_output_i64[i];
            }
        }
    }
    return true;
}

} // namespace yolo_base
