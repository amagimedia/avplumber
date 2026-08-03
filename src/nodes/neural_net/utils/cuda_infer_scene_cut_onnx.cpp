// Pairwise CNN-embedding scene-cut detector (amagimedia/scene-cut-experiments
// export). Unlike luma_diff/hog_diff (raw CUDA metric requiring an external
// Python threshold+debounce decider), this model's KS-distance threshold is
// baked into the ONNX graph at export time, so this node writes the final
// camera_shot_info decision directly — no downstream decider node is used.
//
// Model contract (see amagimedia/scene-cut-experiments docs/onnx_scene_cut_usage.md):
//   inputs:  frame_a float32 (1,3,H,W) NCHW RGB [0,1] -- previous frame
//            frame_b float32 (1,3,H,W) NCHW RGB [0,1] -- current frame
//   output:  scene_cut_flag int64 (1,) -- 0 = same scene, 1 = cut
// Default H=202, W=360 (configurable via input_height/input_width params in case
// a future export changes resolution).
//
// Frame history: only a single-frame lookback is needed. Instead of retaining
// av::VideoFrame objects (as amagi_reframer.cpp does for its N-frame window),
// this node keeps two persistent device tensors (frame_a, frame_b) and shifts
// "current" into "previous" with a cheap device-to-device memcpy each call,
// then overwrites "current" with the freshly preprocessed frame. This avoids
// re-running the NV12->RGB preprocess kernel on the previous frame every time.
#include "../../node_common.hpp"
#include "../../../hwaccel.hpp"
#include <cuda_loader/cuda_drvapi_dynlink_cuda.h>

extern "C" {
#include <libavutil/dict.h>
#include <libavutil/hwcontext.h>
#include <libavutil/hwcontext_cuda.h>
}

#include <NvInfer.h>

#include <cstdint>
#include <cstring>
#include <fstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "../../../../objs/src/nodes/neural_net/utils/cuda_infer_scene_cut_onnx.ptx.h"

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
class SceneCutOnnxTRTLogger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity == Severity::kERROR || severity == Severity::kINTERNAL_ERROR || severity == Severity::kWARNING) {
            logstream << "cuda_infer_scene_cut_onnx tensorrt: " << (msg ? msg : "");
        }
    }
};

size_t elementSize(nvinfer1::DataType dt) {
    switch (dt) {
        case nvinfer1::DataType::kFLOAT: return 4;
        case nvinfer1::DataType::kHALF: return 2;
        case nvinfer1::DataType::kUINT8: return 1;
        case nvinfer1::DataType::kINT8: return 1;
        case nvinfer1::DataType::kINT32: return 4;
        case nvinfer1::DataType::kINT64: return 8;
        case nvinfer1::DataType::kBOOL: return 1;
        default: return 0;
    }
}

size_t volume(const nvinfer1::Dims& d) {
    size_t v = 1;
    for (int i = 0; i < d.nbDims; ++i) {
        if (d.d[i] <= 0) return 0;
        v *= (size_t)d.d[i];
    }
    return v;
}
}

class CudaInferSceneCutOnnx : public NodeSISO<av::VideoFrame, av::VideoFrame>, public ReportsFinishByFlag {
protected:
    std::shared_ptr<HWAccelDevice> hwaccel_;
    AVCUDADeviceContext* cuda_dev_ctx_ = nullptr;
    CUcontext cu_ctx_ = nullptr;

    SceneCutOnnxTRTLogger trt_logger_;
    nvinfer1::IRuntime* trt_runtime_ = nullptr;
    nvinfer1::ICudaEngine* trt_engine_ = nullptr;
    nvinfer1::IExecutionContext* trt_ctx_ = nullptr;

    std::vector<std::string> io_tensor_names_;
    std::vector<size_t> tensor_bytes_;
    std::vector<CUdeviceptr> tensor_ptrs_;
    std::unordered_map<std::string, size_t> tensor_index_;

    size_t frame_a_idx_ = 0;
    size_t frame_b_idx_ = 0;
    size_t output_idx_ = 0;
    nvinfer1::DataType output_dtype_ = nvinfer1::DataType::kINT64;
    std::vector<int64_t> host_output_i64_{0};
    std::vector<int32_t> host_output_i32_{0};

    CUmodule preprocess_module_ = nullptr;
    CUfunction preprocess_kernel_ = nullptr;

    bool initialized_ = false;
    bool has_prev_ = false;
    uint64_t frame_counter_ = 0;

    // Params
    std::string engine_path_;
    std::string metadata_key_ = "camera_shot_info";
    std::string frame_a_tensor_name_ = "frame_a";
    std::string frame_b_tensor_name_ = "frame_b";
    std::string output_tensor_name_ = "scene_cut_flag";
    int input_w_ = 360;
    int input_h_ = 202;
    int debug_log_every_n_ = 0;

    bool initCudaContextFromFrame(const av::VideoFrame& frm) {
        if (cu_ctx_) return true;
        if (!frm.raw() || !frm.raw()->hw_frames_ctx || !frm.raw()->hw_frames_ctx->data) {
            logstream << "cuda_infer_scene_cut_onnx: missing hw_frames_ctx";
            return false;
        }
        AVHWFramesContext* fctx = (AVHWFramesContext*)frm.raw()->hw_frames_ctx->data;
        if (!fctx || !fctx->device_ctx || !fctx->device_ctx->hwctx) {
            logstream << "cuda_infer_scene_cut_onnx: missing device_ctx/hwctx in frame";
            return false;
        }
        cuda_dev_ctx_ = (AVCUDADeviceContext*)fctx->device_ctx->hwctx;
        if (!cuda_dev_ctx_ || !cuda_dev_ctx_->cuda_ctx) {
            logstream << "cuda_infer_scene_cut_onnx: missing CUDA context in frame";
            return false;
        }
        cu_ctx_ = cuda_dev_ctx_->cuda_ctx;
        if (CHECK_CU(cuCtxSetCurrent(cu_ctx_))) {
            logstream << "cuda_infer_scene_cut_onnx: cuCtxSetCurrent failed";
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
            logstream << "cuda_infer_scene_cut_onnx: cannot open engine file " << engine_path_;
            return false;
        }
        std::streamsize size = ifs.tellg();
        if (size <= 0) {
            logstream << "cuda_infer_scene_cut_onnx: invalid engine size";
            return false;
        }
        ifs.seekg(0, std::ios::beg);
        std::vector<char> engine_data((size_t)size);
        if (!ifs.read(engine_data.data(), size)) {
            logstream << "cuda_infer_scene_cut_onnx: failed reading engine file";
            return false;
        }
        trt_runtime_ = nvinfer1::createInferRuntime(trt_logger_);
        if (!trt_runtime_) {
            logstream << "cuda_infer_scene_cut_onnx: createInferRuntime failed";
            return false;
        }
        trt_engine_ = trt_runtime_->deserializeCudaEngine(engine_data.data(), engine_data.size());
        if (!trt_engine_) {
            logstream << "cuda_infer_scene_cut_onnx: deserializeCudaEngine failed";
            return false;
        }
        trt_ctx_ = trt_engine_->createExecutionContext();
        if (!trt_ctx_) {
            logstream << "cuda_infer_scene_cut_onnx: createExecutionContext failed";
            return false;
        }
        return true;
    }

    // Resolves frame_a/frame_b/scene_cut_flag by exact name (unlike Reframer's
    // heuristic predicate matching -- this model's I/O contract is fixed and
    // documented, so exact-name lookup is simpler and safer). Handles both
    // static and dynamic-shape engines: dynamic input dims are resolved via
    // setInputShape() to the configured input_w_/input_h_ before allocation.
    bool allocateBindings() {
        const int nb = trt_engine_->getNbIOTensors();
        if (nb < 3) {
            logstream << "cuda_infer_scene_cut_onnx: engine has fewer than 3 I/O tensors (need frame_a, frame_b, "
                      << output_tensor_name_ << ")";
            return false;
        }

        io_tensor_names_.clear();
        for (int i = 0; i < nb; ++i) {
            const char* name_c = trt_engine_->getIOTensorName(i);
            if (!name_c) {
                logstream << "cuda_infer_scene_cut_onnx: null I/O tensor name";
                return false;
            }
            io_tensor_names_.push_back(name_c);
        }

        auto findTensor = [&](const std::string& name) -> bool {
            return std::find(io_tensor_names_.begin(), io_tensor_names_.end(), name) != io_tensor_names_.end();
        };
        if (!findTensor(frame_a_tensor_name_)) {
            logstream << "cuda_infer_scene_cut_onnx: engine has no input tensor named " << frame_a_tensor_name_;
            return false;
        }
        if (!findTensor(frame_b_tensor_name_)) {
            logstream << "cuda_infer_scene_cut_onnx: engine has no input tensor named " << frame_b_tensor_name_;
            return false;
        }
        if (!findTensor(output_tensor_name_)) {
            logstream << "cuda_infer_scene_cut_onnx: engine has no output tensor named " << output_tensor_name_;
            return false;
        }

        // Resolve dynamic input shapes (trtexec --minShapes/--optShapes/--maxShapes
        // builds always produce an optimization profile requiring an explicit
        // setInputShape call, even when min==opt==max).
        const nvinfer1::Dims4 fixed_input_dims{1, 3, input_h_, input_w_};
        for (const std::string& name : {frame_a_tensor_name_, frame_b_tensor_name_}) {
            if (trt_engine_->getTensorIOMode(name.c_str()) != nvinfer1::TensorIOMode::kINPUT) {
                logstream << "cuda_infer_scene_cut_onnx: " << name << " is not an input tensor";
                return false;
            }
            const nvinfer1::Dims engine_dims = trt_engine_->getTensorShape(name.c_str());
            bool dynamic = false;
            for (int d = 0; d < engine_dims.nbDims; ++d) {
                if (engine_dims.d[d] <= 0) { dynamic = true; break; }
            }
            if (dynamic || engine_dims.nbDims != 4) {
                if (!trt_ctx_->setInputShape(name.c_str(), fixed_input_dims)) {
                    logstream << "cuda_infer_scene_cut_onnx: setInputShape failed for " << name;
                    return false;
                }
            }
        }

        io_tensor_names_.clear();
        tensor_bytes_.assign((size_t)nb, 0);
        tensor_ptrs_.assign((size_t)nb, 0);
        tensor_index_.clear();
        for (int i = 0; i < nb; ++i) {
            const char* name_c = trt_engine_->getIOTensorName(i);
            const std::string name = name_c;
            io_tensor_names_.push_back(name);
            tensor_index_[name] = (size_t)i;

            // After setInputShape, the EXECUTION CONTEXT (not the engine) reports
            // the concrete resolved shape for both inputs and outputs.
            const nvinfer1::Dims dims = trt_ctx_->getTensorShape(name_c);
            const size_t vol = volume(dims);
            const size_t esz = elementSize(trt_engine_->getTensorDataType(name_c));
            if (vol == 0 || esz == 0) {
                logstream << "cuda_infer_scene_cut_onnx: unresolved/unsupported shape or dtype for tensor " << name;
                return false;
            }
            const size_t bytes = vol * esz;
            CUdeviceptr ptr = 0;
            if (CHECK_CU(cuMemAlloc(&ptr, bytes))) {
                logstream << "cuda_infer_scene_cut_onnx: cuMemAlloc failed for tensor " << name;
                return false;
            }
            if (CHECK_CU(cuMemsetD8(ptr, 0, bytes))) {
                logstream << "cuda_infer_scene_cut_onnx: cuMemsetD8 failed for tensor " << name;
                return false;
            }
            tensor_bytes_[(size_t)i] = bytes;
            tensor_ptrs_[(size_t)i] = ptr;

            if (name == frame_a_tensor_name_) frame_a_idx_ = (size_t)i;
            if (name == frame_b_tensor_name_) frame_b_idx_ = (size_t)i;
            if (name == output_tensor_name_) {
                output_idx_ = (size_t)i;
                output_dtype_ = trt_engine_->getTensorDataType(name_c);
                if (output_dtype_ != nvinfer1::DataType::kINT64 && output_dtype_ != nvinfer1::DataType::kINT32) {
                    logstream << "cuda_infer_scene_cut_onnx: " << output_tensor_name_ << " must be int32 or int64, got other dtype";
                    return false;
                }
            }
        }

        for (size_t i = 0; i < io_tensor_names_.size(); ++i) {
            if (!trt_ctx_->setTensorAddress(io_tensor_names_[i].c_str(), reinterpret_cast<void*>(tensor_ptrs_[i]))) {
                logstream << "cuda_infer_scene_cut_onnx: setTensorAddress failed for " << io_tensor_names_[i];
                return false;
            }
        }
        return true;
    }

    bool loadPreprocessKernel() {
        if (preprocess_module_ && preprocess_kernel_) return true;
        if (!cu_ctx_) return false;
        if (CHECK_CU(cuCtxSetCurrent(cu_ctx_))) return false;
        const std::string ptx_str(avpl_scene_cut_onnx_ptx, avpl_scene_cut_onnx_ptx + avpl_scene_cut_onnx_ptx_len);
        if (CHECK_CU(cuModuleLoadDataEx(&preprocess_module_, (const void*)ptx_str.c_str(), 0, nullptr, nullptr))) {
            logstream << "cuda_infer_scene_cut_onnx: failed to load preprocess module";
            return false;
        }
        if (CHECK_CU(cuModuleGetFunction(&preprocess_kernel_, preprocess_module_, "kNV12_to_CHW_rgb01_fp32"))) {
            logstream << "cuda_infer_scene_cut_onnx: failed to get preprocess kernel";
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
        initialized_ = true;
        return true;
    }

    bool runPreprocessInto(const av::VideoFrame& frm, CUdeviceptr dst_ptr) {
        const CUdeviceptr dY = (CUdeviceptr)(uintptr_t)frm.raw()->data[0];
        const CUdeviceptr dUV = (CUdeviceptr)(uintptr_t)frm.raw()->data[1];
        const size_t pitchY = (size_t)frm.raw()->linesize[0];
        const size_t pitchUV = (size_t)frm.raw()->linesize[1];
        void* out = reinterpret_cast<void*>(dst_ptr);
        const unsigned int blockX = 32, blockY = 8;
        const unsigned int gridX = (unsigned int)(input_w_ + (int)blockX - 1) / blockX;
        const unsigned int gridY = (unsigned int)(input_h_ + (int)blockY - 1) / blockY;
        void* args[] = {
            (void*)&dY, (void*)&pitchY,
            (void*)&dUV, (void*)&pitchUV,
            (void*)&out,
            (void*)&input_w_, (void*)&input_h_,
        };
        if (CHECK_CU(cuLaunchKernel(preprocess_kernel_, gridX, gridY, 1, blockX, blockY, 1, 0,
                                    reinterpret_cast<cudaStream_t>(cuda_dev_ctx_->stream), args, nullptr))) {
            logstream << "cuda_infer_scene_cut_onnx: preprocess kernel launch failed";
            return false;
        }
        return true;
    }

    void writeMetadata(av::VideoFrame& frm, bool transition, const char* reason) const {
        Parameters j;
        j["camera_shot_transition"] = transition;
        j["camera_shot_type"] = "unknown";
        (void)reason;
        const std::string md = j.dump();
        av_dict_set(&frm.raw()->metadata, metadata_key_.c_str(), md.c_str(), 0);
    }

    void maybeLog(bool transition) const {
        if (debug_log_every_n_ <= 0) return;
        if ((frame_counter_ % (uint64_t)debug_log_every_n_) != 0) return;
        logstream << "cuda_infer_scene_cut_onnx: frame=" << frame_counter_
                  << " has_prev=" << has_prev_ << " transition=" << transition;
    }

public:
    using NodeSISO::NodeSISO;
    bool consumeEofIfPresent() override { return false; }

    ~CudaInferSceneCutOnnx() override {
        for (CUdeviceptr p : tensor_ptrs_) {
            if (p) CHECK_CU(cuMemFree(p));
        }
        tensor_ptrs_.clear();
        tensor_bytes_.clear();
        if (preprocess_module_) CHECK_CU(cuModuleUnload(preprocess_module_));
        preprocess_module_ = nullptr;
        preprocess_kernel_ = nullptr;
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
            this->finished_ = true;
            return;
        }
        if (!frm) return;
        ++frame_counter_;

        if (frm.raw()->format != AV_PIX_FMT_CUDA) {
            logstream << "cuda_infer_scene_cut_onnx: non-CUDA frame, passing through";
            this->sink_->put(frm);
            return;
        }
        if (!ensureInitialized(frm)) {
            writeMetadata(frm, false, "init_failed");
            this->sink_->put(frm);
            return;
        }
        if (frm.width() != input_w_ || frm.height() != input_h_) {
            logstream << "cuda_infer_scene_cut_onnx: frame size mismatch, expected " << input_w_ << "x" << input_h_
                      << " got " << frm.width() << "x" << frm.height();
            writeMetadata(frm, false, "size_mismatch");
            this->sink_->put(frm);
            return;
        }
        if (hwSwFormat(frm) != AV_PIX_FMT_NV12) {
            logstream << "cuda_infer_scene_cut_onnx: unsupported hw sw_format (expected NV12)";
            writeMetadata(frm, false, "format_mismatch");
            this->sink_->put(frm);
            return;
        }
        if (CHECK_CU(cuCtxSetCurrent(cu_ctx_))) {
            writeMetadata(frm, false, "cuda_ctx_failed");
            this->sink_->put(frm);
            return;
        }

        if (!has_prev_) {
            // No previous frame yet: seed frame_b with the current frame so the
            // NEXT call's device-to-device shift has real data, but do not run
            // inference against an undefined frame_a (all-zero at this point).
            if (!runPreprocessInto(frm, tensor_ptrs_[frame_b_idx_])) {
                writeMetadata(frm, false, "preprocess_failed");
                this->sink_->put(frm);
                return;
            }
            has_prev_ = true;
            writeMetadata(frm, false, "no_prev");
            maybeLog(false);
            this->sink_->put(frm);
            return;
        }

        // Shift last call's "current" tensor into "previous", then overwrite
        // "current" with the freshly preprocessed frame.
        if (CHECK_CU(cuMemcpyDtoDAsync(tensor_ptrs_[frame_a_idx_], tensor_ptrs_[frame_b_idx_],
                                       tensor_bytes_[frame_b_idx_], cuda_dev_ctx_->stream))) {
            writeMetadata(frm, false, "shift_failed");
            this->sink_->put(frm);
            return;
        }
        if (!runPreprocessInto(frm, tensor_ptrs_[frame_b_idx_])) {
            writeMetadata(frm, false, "preprocess_failed");
            this->sink_->put(frm);
            return;
        }

        if (!trt_ctx_->enqueueV3(reinterpret_cast<cudaStream_t>(cuda_dev_ctx_->stream))) {
            logstream << "cuda_infer_scene_cut_onnx: enqueueV3 failed";
            writeMetadata(frm, false, "infer_failed");
            this->sink_->put(frm);
            return;
        }

        bool copy_ok = false;
        if (output_dtype_ == nvinfer1::DataType::kINT64) {
            copy_ok = !CHECK_CU(cuMemcpyDtoHAsync(host_output_i64_.data(), tensor_ptrs_[output_idx_],
                                                   tensor_bytes_[output_idx_], cuda_dev_ctx_->stream));
        } else {
            copy_ok = !CHECK_CU(cuMemcpyDtoHAsync(host_output_i32_.data(), tensor_ptrs_[output_idx_],
                                                   tensor_bytes_[output_idx_], cuda_dev_ctx_->stream));
        }
        if (!copy_ok) {
            writeMetadata(frm, false, "output_copy_failed");
            this->sink_->put(frm);
            return;
        }
        if (CHECK_CU(cuStreamSynchronize(cuda_dev_ctx_->stream))) {
            writeMetadata(frm, false, "stream_sync_failed");
            this->sink_->put(frm);
            return;
        }

        const bool transition = output_dtype_ == nvinfer1::DataType::kINT64
            ? (host_output_i64_[0] != 0)
            : (host_output_i32_[0] != 0);
        writeMetadata(frm, transition, "ok");
        maybeLog(transition);
        this->sink_->put(frm);
    }

    static std::shared_ptr<CudaInferSceneCutOnnx> create(NodeCreationInfo &nci) {
        EdgeManager &edges = nci.edges;
        const Parameters &params = nci.params;
        std::shared_ptr<Edge<av::VideoFrame>> src = edges.find<av::VideoFrame>(params["src"]);
        std::shared_ptr<Edge<av::VideoFrame>> dst = edges.find<av::VideoFrame>(params["dst"]);
        auto r = std::make_shared<CudaInferSceneCutOnnx>(
            make_unique<EdgeSource<av::VideoFrame>>(src),
            make_unique<EdgeSink<av::VideoFrame>>(dst)
        );
        src->setConsumer(r);
        dst->setProducer(r);

        if (!params.count("engine")) {
            throw Error("cuda_infer_scene_cut_onnx: missing required parameter: engine");
        }
        if (!params.count("hwaccel")) {
            throw Error("cuda_infer_scene_cut_onnx: missing required parameter: hwaccel");
        }
        r->engine_path_ = params["engine"].get<std::string>();
        r->hwaccel_ = InstanceSharedObjects<HWAccelDevice>::get(nci.instance, params["hwaccel"]);
        if (!r->hwaccel_) {
            throw Error("cuda_infer_scene_cut_onnx: failed to get hwaccel");
        }

        if (params.count("metadata_key")) r->metadata_key_ = params["metadata_key"].get<std::string>();
        if (params.count("frame_a_tensor_name")) r->frame_a_tensor_name_ = params["frame_a_tensor_name"].get<std::string>();
        if (params.count("frame_b_tensor_name")) r->frame_b_tensor_name_ = params["frame_b_tensor_name"].get<std::string>();
        if (params.count("output_tensor_name")) r->output_tensor_name_ = params["output_tensor_name"].get<std::string>();
        if (params.count("input_width")) r->input_w_ = params["input_width"];
        if (params.count("input_height")) r->input_h_ = params["input_height"];
        if (params.count("debug_log_every_n")) r->debug_log_every_n_ = params["debug_log_every_n"];

        if (r->input_w_ <= 0 || r->input_h_ <= 0) {
            throw Error("cuda_infer_scene_cut_onnx: input_width and input_height must be positive");
        }
        return r;
    }
};

DECLNODE(cuda_infer_scene_cut_onnx, CudaInferSceneCutOnnx)
