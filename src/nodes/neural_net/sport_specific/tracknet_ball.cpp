#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <limits>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "../../node_common.hpp"
#include "../common/infer_trt_base.hpp"

// PTX blob for TrackNet triplet preprocessing.
#include "../../../../objs/src/nodes/neural_net/sport_specific/tracknet_ball_preprocess.ptx.h"

extern "C" {
#include <libavutil/dict.h>
#include <libavutil/frame.h>
#include <libavutil/pixfmt.h>
}

using namespace yolo_base;

namespace {

float clamp01(float v) {
    if (!std::isfinite(v)) return 0.0f;
    return std::max(0.0f, std::min(1.0f, v));
}

int jsonIntParam(const Parameters& params, const char* key, int fallback) {
    if (!params.count(key) || params[key].is_null()) return fallback;
    return params[key].get<int>();
}

float jsonFloatParam(const Parameters& params, const char* key, float fallback) {
    if (!params.count(key) || params[key].is_null()) return fallback;
    return params[key].get<float>();
}

bool jsonBoolParam(const Parameters& params, const char* key, bool fallback) {
    if (!params.count(key) || params[key].is_null()) return fallback;
    return params[key].get<bool>();
}

void rejectLegacySamplingParams(const Parameters& params) {
    for (const char* key : {"sample_every_n", "tracknet_sample_every_n", "infer_every_n"}) {
        if (params.count(key) && !params[key].is_null()) {
            throw Error("tracknet_ball: legacy sample_every_n parameters were removed; "
                        "use auto_sample_min_fps and auto_sample_every_n");
        }
    }
}

double jsonFpsParam(const Parameters& params, const char* key, double fallback) {
    if (!params.count(key) || params[key].is_null()) return fallback;
    const Parameters& value = params[key];
    if (value.is_number()) return value.get<double>();
    if (value.is_string()) return parseRatio(value.get<std::string>()).getDouble();
    throw Error(std::string("tracknet_ball: ") + key + " must be a number or ratio string");
}

std::string jsonStringParam(const Parameters& params, const char* key, const std::string& fallback) {
    if (!params.count(key) || params[key].is_null()) return fallback;
    return params[key].get<std::string>();
}

enum class TrackNetOutputMode {
    Detection,
    Raw,
    Both,
    SrsBall,
    SrsBallAndRaw,
};

enum class TrackNetTripletAlignment {
    Center,
    Latest,
};

enum class TrackNetPreprocessMode {
    Resize,
    SrsAffine,
};

enum class TrackNetSampleFillMode {
    None,
    Hold,
};

bool emitsDetectionMetadata(TrackNetOutputMode mode) {
    return mode == TrackNetOutputMode::Detection || mode == TrackNetOutputMode::Both;
}

bool emitsRawMetadata(TrackNetOutputMode mode) {
    return mode == TrackNetOutputMode::Raw || mode == TrackNetOutputMode::Both ||
           mode == TrackNetOutputMode::SrsBallAndRaw;
}

bool emitsSrsBallMetadata(TrackNetOutputMode mode) {
    return mode == TrackNetOutputMode::SrsBall || mode == TrackNetOutputMode::SrsBallAndRaw;
}

int inputChannels(const nvinfer1::Dims& dims) {
    if (dims.nbDims == 3) return dims.d[0];
    if (dims.nbDims == 4 && dims.d[0] == 1) return dims.d[1];
    return 0;
}

bool setInputGeometryFromDims(ModelRunner& model, int expected_channels) {
    if (model.input_dims.nbDims == 3 && model.input_dims.d[0] == expected_channels) {
        model.input_h = model.input_dims.d[1];
        model.input_w = model.input_dims.d[2];
        return true;
    }
    if (model.input_dims.nbDims == 4 && model.input_dims.d[0] == 1 &&
        model.input_dims.d[1] == expected_channels) {
        model.input_h = model.input_dims.d[2];
        model.input_w = model.input_dims.d[3];
        return true;
    }
    return false;
}

TrackNetOutputMode parseOutputModeString(const std::string& mode) {
    if (mode == "detection" || mode == "detections") return TrackNetOutputMode::Detection;
    if (mode == "raw") return TrackNetOutputMode::Raw;
    if (mode == "both" || mode == "detection_and_raw") return TrackNetOutputMode::Both;
    if (mode == "srs_ball" || mode == "srs") return TrackNetOutputMode::SrsBall;
    if (mode == "srs_ball_and_raw" || mode == "srs_and_raw") return TrackNetOutputMode::SrsBallAndRaw;
    throw Error("tracknet_ball: output_mode must be 'detection', 'raw', 'both', 'srs_ball', or 'srs_ball_and_raw'");
}

TrackNetTripletAlignment parseTripletAlignmentString(const std::string& alignment) {
    if (alignment == "center") return TrackNetTripletAlignment::Center;
    if (alignment == "latest" || alignment == "vod_latest") return TrackNetTripletAlignment::Latest;
    throw Error("tracknet_ball: triplet_alignment must be 'center' or 'latest'");
}

TrackNetPreprocessMode parsePreprocessModeString(const std::string& mode) {
    if (mode == "resize" || mode == "scale") return TrackNetPreprocessMode::Resize;
    if (mode == "srs_affine" || mode == "srs") return TrackNetPreprocessMode::SrsAffine;
    throw Error("tracknet_ball: preprocess_mode must be 'resize' or 'srs_affine'");
}

TrackNetSampleFillMode parseSampleFillModeString(const std::string& mode) {
    if (mode == "none" || mode == "off" || mode == "disabled") return TrackNetSampleFillMode::None;
    if (mode == "hold" || mode == "last" || mode == "copy_last") return TrackNetSampleFillMode::Hold;
    throw Error("tracknet_ball: sample_fill_mode must be 'none' or 'hold'");
}

std::string dtypeName(nvinfer1::DataType dtype) {
    switch (dtype) {
        case nvinfer1::DataType::kFLOAT: return "float32";
        case nvinfer1::DataType::kHALF: return "float16";
        case nvinfer1::DataType::kINT8: return "int8";
        case nvinfer1::DataType::kINT32: return "int32";
        case nvinfer1::DataType::kINT64: return "int64";
        default: return "unknown";
    }
}

Parameters dimsToJson(const nvinfer1::Dims& dims) {
    Parameters out = Parameters::array();
    for (int i = 0; i < dims.nbDims; ++i) {
        out.push_back(dims.d[i]);
    }
    return out;
}

float sigmoidFloat(float v) {
    if (!std::isfinite(v)) return 0.0f;
    if (v >= 0.0f) {
        const float z = std::exp(-v);
        return 1.0f / (1.0f + z);
    }
    const float z = std::exp(v);
    return z / (1.0f + z);
}

Parameters tensorValuesToJson(const std::vector<float>& values, int max_elements, size_t* emitted_count = nullptr,
                              size_t* nonfinite_count = nullptr) {
    const size_t limit = max_elements > 0 ? std::min(values.size(), (size_t)max_elements) : values.size();
    Parameters out = Parameters::array();
    for (size_t i = 0; i < limit; ++i) {
        if (std::isfinite(values[i])) {
            out.push_back(values[i]);
        } else {
            out.push_back(nullptr);
            if (nonfinite_count) ++(*nonfinite_count);
        }
    }
    if (emitted_count) *emitted_count += limit;
    return out;
}

} // namespace

class TrackNetBall : public NodeSISO<av::VideoFrame, av::VideoFrame>,
                     public IInputReset,
                     public ReportsFinishByFlag,
                     public CudaInferTrtBase {
protected:
    std::string metadata_key_detection_ = "yolo_ball";
    std::string metadata_key_raw_ = "tracknet_raw";
    std::string metadata_key_srs_ball_ = "tracknet_ball_srs";
    std::string target_label_ = "basketball";
    TrackNetOutputMode output_mode_ = TrackNetOutputMode::Detection;
    TrackNetTripletAlignment triplet_alignment_ = TrackNetTripletAlignment::Center;
    TrackNetPreprocessMode preprocess_mode_ = TrackNetPreprocessMode::Resize;
    TrackNetSampleFillMode sample_fill_mode_ = TrackNetSampleFillMode::None;
    float conf_thresh_ = 0.5f;
    float visible_thresh_ = 0.5f;
    float srs_score_threshold_ = 0.5f;
    bool emit_invisible_ = false;
    bool srs_use_hm_weight_ = true;
    int raw_output_max_elements_ = 0;
    int srs_channel_ = 2;
    int output_model_width_ = 0;
    int output_model_height_ = 0;
    bool debug_log_metadata_ = false;
    int debug_log_every_n_ = 0;
    double auto_sample_min_fps_ = 0.0;
    int auto_sample_every_n_ = 1;
    int active_sample_every_n_ = 1;
    bool auto_sample_decided_ = false;
    bool auto_sample_have_pts_ = false;
    av::Timestamp auto_sample_prev_pts_;
    double estimated_input_fps_ = 0.0;
    std::map<std::string, std::string> last_tracknet_metadata_;

    std::deque<av::VideoFrame> frame_buffer_;
    bool first_boundary_emitted_ = false;
    bool center_output_started_ = false;
    int source_w_ = 0;
    int source_h_ = 0;
    AVPixelFormat source_sw_format_ = AV_PIX_FMT_NONE;

    int output_tensor_index_ = -1;
    int srs_heatmap_tensor_index_ = -1;
    bool output_contract_validated_ = false;
    uint64_t frame_counter_ = 0;
    uint64_t infer_counter_ = 0;
    uint64_t skipped_sample_frames_ = 0;
    uint64_t detected_frames_ = 0;
    uint64_t empty_frames_ = 0;
    std::array<uint64_t, 10> conf_histogram_{};
    std::map<int, uint64_t> detection_count_histogram_;
    std::string node_label_ = "<unnamed>";

public:
    TrackNetBall(std::unique_ptr<Source<av::VideoFrame>> source,
                 std::unique_ptr<EdgeSink<av::VideoFrame>> sink)
        : NodeSISO(std::move(source), std::move(sink)) {}

    ~TrackNetBall() {
        const uint64_t total = detected_frames_ + empty_frames_;
        if (total == 0) return;
        logstream << "tracknet_ball: detection summary:"
                  << " detected frames: " << detected_frames_
                  << " / total inferred frames: " << total
                  << " (" << (100.0 * (double)detected_frames_ / (double)std::max<uint64_t>(1, total)) << "%)"
                  << ", active_sample_every_n: " << active_sample_every_n_
                  << ", auto_sample_min_fps: " << auto_sample_min_fps_
                  << ", auto_sample_every_n: " << auto_sample_every_n_
                  << ", sample_fill_mode: " << (sample_fill_mode_ == TrackNetSampleFillMode::Hold ? "hold" : "none")
                  << ", estimated_input_fps: " << estimated_input_fps_
                  << ", skipped sample frames: " << skipped_sample_frames_;
        logstream << "tracknet_ball: confidence histogram:";
        for (int i = 0; i < 10; ++i) {
            logstream << "  " << (i * 0.1) << "-" << ((i + 1) * 0.1) << ": "
                      << conf_histogram_[(size_t)i] << " detections";
        }
    }

    bool consumeEofIfPresent() override {
        return false;
    }

    void resetInput() override {
        resetBufferedFrames();
    }

    void process() override {
        av::VideoFrame frm = this->source_->get();
        if (isEofMarker(frm)) {
            flushBufferedBoundaryFrames();
            this->sink_->put(frm);
            this->finished_ = true;
            return;
        }
        if (!frm) return;

        ++frame_counter_;
        updateAutoSampling(frm);

        if (!isSupportedCudaFrame(frm)) {
            flushBufferedBoundaryFrames();
            this->sink_->put(frm);
            return;
        }

        if (!shouldSampleCurrentFrame()) {
            ++skipped_sample_frames_;
            applyHeldMetadata(frm);
            this->sink_->put(frm);
            return;
        }

        if (!initialized_) {
            auto init_start = std::chrono::steady_clock::now();
            bool ok = ensureTracknetInitialized(frm);
            auto init_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - init_start).count();
            logstream << "neural_node_init"
                      << " status=" << (ok ? "ok" : "error")
                      << " type=tracknet_ball"
                      << " node=" << node_label_
                      << " init_ms=" << init_ms
                      << " frame_wait_ms=" << init_ms
                      << " frame=" << frame_counter_
                      << " models=" << models_.size();
            if (!ok) {
                return;
            }
        } else if (!ensureTracknetInitialized(frm)) {
            return;
        }

        if (CUDA_CHECK_CU(cuCtxSetCurrent(cu_ctx_))) {
            logstream << "tracknet_ball: cuCtxSetCurrent failed in process";
            return;
        }

        if (!bufferCompatible(frm)) {
            flushBufferedBoundaryFrames();
            setBufferGeometry(frm);
        } else if (source_w_ == 0 || source_h_ == 0) {
            setBufferGeometry(frm);
        }

        if (triplet_alignment_ == TrackNetTripletAlignment::Latest) {
            processLatestAlignedFrame(frm);
        } else {
            processCenterAlignedFrame(frm);
        }
    }

protected:
    void storeTracknetMetadata(const std::string& key, const std::string& value) {
        last_tracknet_metadata_[key] = value;
    }

    void applyHeldMetadata(av::VideoFrame& frm) {
        if (sample_fill_mode_ != TrackNetSampleFillMode::Hold || last_tracknet_metadata_.empty() || !frm.raw()) {
            return;
        }
        for (const auto& [key, value] : last_tracknet_metadata_) {
            av_dict_set(&frm.raw()->metadata, key.c_str(), value.c_str(), 0);
        }
    }

    bool shouldSampleCurrentFrame() const {
        return active_sample_every_n_ <= 1 ||
               (((frame_counter_ > 0 ? frame_counter_ - 1 : 0) % (uint64_t)active_sample_every_n_) == 0);
    }

    void updateAutoSampling(const av::VideoFrame& frm) {
        if (auto_sample_decided_ || auto_sample_min_fps_ <= 0.0 || auto_sample_every_n_ <= 1) {
            return;
        }
        const av::Timestamp pts = frm.pts();
        if (!pts.isValid()) return;
        if (!auto_sample_have_pts_) {
            auto_sample_prev_pts_ = pts;
            auto_sample_have_pts_ = true;
            return;
        }

        const double delta_sec = (pts - auto_sample_prev_pts_).seconds();
        auto_sample_prev_pts_ = pts;
        if (!std::isfinite(delta_sec) || delta_sec <= 0.0) return;

        estimated_input_fps_ = 1.0 / delta_sec;
        active_sample_every_n_ = estimated_input_fps_ >= auto_sample_min_fps_ ? auto_sample_every_n_ : 1;
        auto_sample_decided_ = true;
        logstream << "tracknet_ball: auto sampling estimated_input_fps=" << estimated_input_fps_
                  << " min_fps=" << auto_sample_min_fps_
                  << " active_sample_every_n=" << active_sample_every_n_;
    }

    void processCenterAlignedFrame(const av::VideoFrame& frm) {
        frame_buffer_.push_back(frm);
        if (frame_buffer_.size() < 3) return;

        if (!first_boundary_emitted_) {
            this->sink_->put(frame_buffer_.front());
            first_boundary_emitted_ = true;
        }

        av::VideoFrame& center = frame_buffer_[1];
        if (!runTripletInference(frame_buffer_[0], center, center, frame_buffer_[2])) {
            return;
        }

        center_output_started_ = true;
        this->sink_->put(center);
        frame_buffer_.pop_front();
    }

    void processLatestAlignedFrame(const av::VideoFrame& frm) {
        frame_buffer_.push_back(frm);

        av::VideoFrame& latest = frame_buffer_.back();
        if (frame_buffer_.size() == 1) {
            if (!runTripletInference(latest, latest, latest, latest)) return;
            this->sink_->put(latest);
            return;
        }

        if (frame_buffer_.size() == 2) {
            if (!runTripletInference(frame_buffer_[0], frame_buffer_[0], latest, latest)) return;
            this->sink_->put(latest);
            return;
        }

        if (!runTripletInference(frame_buffer_[0], frame_buffer_[1], latest, latest)) return;
        this->sink_->put(latest);
        frame_buffer_.pop_front();
    }

    bool isSupportedCudaFrame(const av::VideoFrame& frm) {
        if (!frm.raw() || frm.raw()->format != AV_PIX_FMT_CUDA) {
            logstream << "tracknet_ball: non-CUDA frame, passing through";
            return false;
        }
        AVPixelFormat swfmt = hwSwFormat(frm);
        if (swfmt != AV_PIX_FMT_NV12) {
            logstream << "tracknet_ball: unsupported hw sw_format (expected NV12)";
            return false;
        }
        return true;
    }

    void setBufferGeometry(const av::VideoFrame& frm) {
        source_w_ = frm.width();
        source_h_ = frm.height();
        source_sw_format_ = hwSwFormat(frm);
    }

    bool bufferCompatible(const av::VideoFrame& frm) const {
        if (source_w_ == 0 || source_h_ == 0) return true;
        return frm.width() == source_w_
            && frm.height() == source_h_
            && hwSwFormat(frm) == source_sw_format_;
    }

    void resetBufferedFrames() {
        frame_buffer_.clear();
        first_boundary_emitted_ = false;
        center_output_started_ = false;
        source_w_ = 0;
        source_h_ = 0;
        source_sw_format_ = AV_PIX_FMT_NONE;
    }

    void flushBufferedBoundaryFrames() {
        if (triplet_alignment_ == TrackNetTripletAlignment::Latest) {
            resetBufferedFrames();
            return;
        }

        if (frame_buffer_.empty()) {
            resetBufferedFrames();
            return;
        }

        if (center_output_started_) {
            if (frame_buffer_.size() >= 2) {
                this->sink_->put(frame_buffer_.back());
            } else if (!first_boundary_emitted_) {
                this->sink_->put(frame_buffer_.front());
            }
        } else {
            while (!frame_buffer_.empty()) {
                this->sink_->put(frame_buffer_.front());
                frame_buffer_.pop_front();
            }
        }

        resetBufferedFrames();
    }

    bool loadTracknetPreprocessModule() {
        if (preprocess_module_) return true;
        if (!cu_ctx_) return false;
        if (CUDA_CHECK_CU(cuCtxSetCurrent(cu_ctx_))) return false;
        const std::string ptx_str(avpl_tracknet_ball_preprocess_ptx,
                                  avpl_tracknet_ball_preprocess_ptx + avpl_tracknet_ball_preprocess_ptx_len);
        if (CUDA_CHECK_CU(cuModuleLoadDataEx(&preprocess_module_, ptx_str.c_str(), 0, nullptr, nullptr))) {
            logstream << "tracknet_ball: failed to load preprocess PTX module";
            return false;
        }
        return true;
    }

    bool configureTracknetPreprocess(ModelRunner& model) {
        const char* kname = nullptr;
        if (preprocess_mode_ == TrackNetPreprocessMode::SrsAffine) {
            kname = (model.input_dtype == nvinfer1::DataType::kHALF)
                ? "kTrackNetNV12TripletToNCHW9SrsAffine_fp16"
                : "kTrackNetNV12TripletToNCHW9SrsAffine_fp32";
        } else {
            kname = (model.input_dtype == nvinfer1::DataType::kHALF)
                ? "kTrackNetNV12TripletToNCHW9_fp16"
                : "kTrackNetNV12TripletToNCHW9_fp32";
        }
        if (CUDA_CHECK_CU(cuModuleGetFunction(&model.preprocess_kernel, preprocess_module_, kname))) {
            logstream << "tracknet_ball: failed to get preprocess kernel for " << model.engine_path;
            return false;
        }
        return configureTracknetStream(model);
    }

    bool configureTracknetStream(ModelRunner& model) {
        constexpr unsigned int kCudaStreamDefault = 0x0;
        constexpr unsigned int kCudaStreamNonBlocking = 0x1;
        unsigned int stream_flags = use_cuda_graph_ ? kCudaStreamNonBlocking : kCudaStreamDefault;
        if (CUDA_CHECK_CU(cuStreamCreate(&model.stream, stream_flags))) {
            logstream << "tracknet_ball: failed to create CUDA stream for " << model.engine_path;
            return false;
        }
        if (use_cuda_graph_) {
            int nb_aux_streams = model.trt_engine->getNbAuxStreams();
            model.aux_streams.reserve((size_t)std::max(nb_aux_streams, 0));
            for (int i = 0; i < nb_aux_streams; ++i) {
                CUstream aux_stream = nullptr;
                if (CUDA_CHECK_CU(cuStreamCreate(&aux_stream, kCudaStreamNonBlocking))) {
                    logstream << "tracknet_ball: failed to create TensorRT aux stream for " << model.engine_path;
                    return false;
                }
                model.aux_streams.push_back(reinterpret_cast<cudaStream_t>(aux_stream));
            }
        }
        return true;
    }

    bool allocateTracknetBindings(ModelRunner& model) {
        constexpr int kExpectedChannels = 9;
        const int nb = model.trt_engine->getNbIOTensors();
        if (nb <= 1) {
            logstream << "tracknet_ball: engine has insufficient bindings for " << model.engine_path;
            return false;
        }

        model.io_tensor_names.clear();
        model.tensor_bytes.assign((size_t)nb, 0);
        model.tensor_ptrs.assign((size_t)nb, 0);
        model.tensor_index.clear();
        model.input_tensor_name.clear();
        model.outputs.clear();
        int input_count = 0;
        bool selected_triplet_input = false;

        for (int i = 0; i < nb; ++i) {
            const char* tensor_name_c = model.trt_engine->getIOTensorName(i);
            if (!tensor_name_c) {
                logstream << "tracknet_ball: null I/O tensor name";
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
                    logstream << "tracknet_ball: dynamic/invalid binding dims not supported for " << model.engine_path;
                    return false;
                }
            }

            const size_t vol = volume(dims);
            const nvinfer1::DataType dt = model.trt_engine->getTensorDataType(tensor_name_c);
            const size_t esz = elementSize(dt);
            if (vol == 0 || esz == 0) {
                logstream << "tracknet_ball: unsupported binding type/shape for " << model.engine_path;
                return false;
            }

            CUdeviceptr ptr = 0;
            const size_t bytes = vol * esz;
            if (CUDA_CHECK_CU(cuMemAlloc(&ptr, bytes))) {
                logstream << "tracknet_ball: cuMemAlloc failed for " << model.engine_path << " binding " << i;
                return false;
            }
            if (CUDA_CHECK_CU(cuMemsetD8(ptr, 0, bytes))) {
                logstream << "tracknet_ball: cuMemsetD8 failed for " << model.engine_path << " binding " << i;
                return false;
            }
            model.tensor_bytes[(size_t)i] = bytes;
            model.tensor_ptrs[(size_t)i] = ptr;

            if (is_input) {
                ++input_count;
                const bool is_triplet_input =
                    (dims.nbDims == 3 && dims.d[0] == kExpectedChannels) ||
                    (dims.nbDims == 4 && dims.d[0] == 1 && dims.d[1] == kExpectedChannels);
                if (model.input_tensor_name.empty()) {
                    model.input_tensor_name = tensor_name;
                    model.input_dims = dims;
                }
                if (is_triplet_input && !selected_triplet_input) {
                    model.input_tensor_name = tensor_name;
                    model.input_dims = dims;
                    selected_triplet_input = true;
                } else if (!is_triplet_input) {
                    logstream << "tracknet_ball: auxiliary input tensor ignored for "
                              << model.engine_path << ": " << tensor_name;
                }
            } else {
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
            logstream << "tracknet_ball: failed to identify input/output bindings for " << model.engine_path;
            return false;
        }
        if (!setInputGeometryFromDims(model, kExpectedChannels) || model.input_h <= 0 || model.input_w <= 0) {
            logstream << "tracknet_ball: expected CHW or NCHW input tensor with 9 channels for "
                      << model.engine_path << " (engine inputs: " << input_count << ")";
            return false;
        }

        for (const OutputTensor& ot : model.outputs) {
            if (!(ot.dtype == nvinfer1::DataType::kFLOAT ||
                  ot.dtype == nvinfer1::DataType::kHALF ||
                  ot.dtype == nvinfer1::DataType::kINT32 ||
                  ot.dtype == nvinfer1::DataType::kINT64)) {
                logstream << "tracknet_ball: output datatype must be float/half/int32/int64 for "
                          << model.engine_path << " tensor " << ot.name;
                return false;
            }
        }

        model.input_dtype = model.trt_engine->getTensorDataType(model.input_tensor_name.c_str());
        if (!(model.input_dtype == nvinfer1::DataType::kFLOAT || model.input_dtype == nvinfer1::DataType::kHALF)) {
            logstream << "tracknet_ball: input datatype must be float/half for " << model.engine_path;
            return false;
        }

        for (size_t i = 0; i < model.io_tensor_names.size(); ++i) {
            if (!model.trt_ctx->setTensorAddress(
                    model.io_tensor_names[i].c_str(), reinterpret_cast<void*>(model.tensor_ptrs[i]))) {
                logstream << "tracknet_ball: setTensorAddress failed for " << model.io_tensor_names[i]
                          << " in " << model.engine_path;
                return false;
            }
        }
        return true;
    }

    bool ensureTracknetInitialized(const av::VideoFrame& frm) {
        if (initialized_) return true;
        if (models_.size() != 1) {
            logstream << "tracknet_ball: requires exactly one TensorRT engine";
            return false;
        }
        if (!initCudaContextFromFrame(frm)) return false;
        if (!loadTracknetPreprocessModule()) return false;

        ModelRunner& model = models_[0];
        if (!parseEngine(model) || !allocateTracknetBindings(model) ||
            !ensureCompatibleInput(model, 0) || !configureTracknetPreprocess(model) ||
            !validateOutputContract(model)) {
            cleanupModel(model);
            return false;
        }

        initialized_ = true;
        return true;
    }

    struct SrsCandidate {
        float x = 0.0f;
        float y = 0.0f;
        float score = 0.0f;
    };

    bool getHeatmapShape(const OutputTensor& ot, int& channels, int& height, int& width) const {
        channels = 0;
        height = 0;
        width = 0;
        if (ot.dims.nbDims == 4 && ot.dims.d[0] == 1) {
            channels = ot.dims.d[1];
            height = ot.dims.d[2];
            width = ot.dims.d[3];
            return channels > 0 && height > 0 && width > 0;
        }
        if (ot.dims.nbDims == 3) {
            channels = ot.dims.d[0];
            height = ot.dims.d[1];
            width = ot.dims.d[2];
            return channels > 0 && height > 0 && width > 0;
        }
        return false;
    }

    bool validateOutputContract(ModelRunner& model) {
        if (inputChannels(model.input_dims) != 9 || model.input_w <= 0 || model.input_h <= 0) {
            logstream << "tracknet_ball: engine must have NCHW input with 9 channels";
            return false;
        }

        output_tensor_index_ = -1;
        srs_heatmap_tensor_index_ = -1;
        if (emitsDetectionMetadata(output_mode_)) {
            for (size_t i = 0; i < model.outputs.size(); ++i) {
                const OutputTensor& ot = model.outputs[i];
                if (volume(ot.dims) >= 6) {
                    output_tensor_index_ = (int)i;
                    if (debug_log_metadata_) {
                        logstream << "tracknet_ball: detection output tensor=" << ot.name
                                  << " dims=" << ot.dims.nbDims
                                  << " volume=" << volume(ot.dims)
                                  << " input=" << model.input_w << "x" << model.input_h;
                    }
                    break;
                }
            }

            if (output_tensor_index_ < 0) {
                logstream << "tracknet_ball: failed to find output tensor with [x1,y1,x2,y2,score,visible]";
                return false;
            }
        }

        if (emitsSrsBallMetadata(output_mode_)) {
            for (size_t i = 0; i < model.outputs.size(); ++i) {
                int channels = 0;
                int heatmap_h = 0;
                int heatmap_w = 0;
                if (getHeatmapShape(model.outputs[i], channels, heatmap_h, heatmap_w) &&
                    channels > srs_channel_) {
                    srs_heatmap_tensor_index_ = (int)i;
                    if (debug_log_metadata_) {
                        logstream << "tracknet_ball: SRS heatmap output tensor=" << model.outputs[i].name
                                  << " channels=" << channels
                                  << " size=" << heatmap_w << "x" << heatmap_h
                                  << " dtype=" << dtypeName(model.outputs[i].dtype);
                    }
                    break;
                }
            }

            if (srs_heatmap_tensor_index_ < 0) {
                logstream << "tracknet_ball: failed to find SRS heatmap output tensor with channel "
                          << srs_channel_ << " (expected CHW or NCHW logits)";
                return false;
            }
        }

        if (debug_log_metadata_ && emitsRawMetadata(output_mode_)) {
            for (const OutputTensor& ot : model.outputs) {
                logstream << "tracknet_ball: raw output tensor=" << ot.name
                          << " dims=" << ot.dims.nbDims
                          << " volume=" << volume(ot.dims)
                          << " dtype=" << dtypeName(ot.dtype);
            }
        }

        output_contract_validated_ = true;
        return true;
    }

    bool runTripletPreprocess(const av::VideoFrame& f0,
                              const av::VideoFrame& f1,
                              const av::VideoFrame& f2,
                              ModelRunner& model) {
        auto it = model.tensor_index.find(model.input_tensor_name);
        if (it == model.tensor_index.end()) {
            logstream << "tracknet_ball: input tensor index missing for " << model.engine_path;
            return false;
        }

        const AVFrame* r0 = f0.raw();
        const AVFrame* r1 = f1.raw();
        const AVFrame* r2 = f2.raw();
        const CUdeviceptr y0 = (CUdeviceptr)(uintptr_t)r0->data[0];
        const CUdeviceptr uv0 = (CUdeviceptr)(uintptr_t)r0->data[1];
        const CUdeviceptr y1 = (CUdeviceptr)(uintptr_t)r1->data[0];
        const CUdeviceptr uv1 = (CUdeviceptr)(uintptr_t)r1->data[1];
        const CUdeviceptr y2 = (CUdeviceptr)(uintptr_t)r2->data[0];
        const CUdeviceptr uv2 = (CUdeviceptr)(uintptr_t)r2->data[1];
        const size_t pitch_y0 = (size_t)r0->linesize[0];
        const size_t pitch_uv0 = (size_t)r0->linesize[1];
        const size_t pitch_y1 = (size_t)r1->linesize[0];
        const size_t pitch_uv1 = (size_t)r1->linesize[1];
        const size_t pitch_y2 = (size_t)r2->linesize[0];
        const size_t pitch_uv2 = (size_t)r2->linesize[1];

        void* out = reinterpret_cast<void*>(model.tensor_ptrs[it->second]);
        const int src_w = f1.width();
        const int src_h = f1.height();
        const int dst_w = model.input_w;
        const int dst_h = model.input_h;

        void* args[] = {
            (void*)&y0, (void*)&pitch_y0,
            (void*)&uv0, (void*)&pitch_uv0,
            (void*)&y1, (void*)&pitch_y1,
            (void*)&uv1, (void*)&pitch_uv1,
            (void*)&y2, (void*)&pitch_y2,
            (void*)&uv2, (void*)&pitch_uv2,
            (void*)&out,
            (void*)&src_w, (void*)&src_h,
            (void*)&dst_w, (void*)&dst_h
        };

        const unsigned int block_x = 32;
        const unsigned int block_y = 8;
        const unsigned int grid_x = (unsigned int)(dst_w + (int)block_x - 1) / block_x;
        const unsigned int grid_y = (unsigned int)(dst_h + (int)block_y - 1) / block_y;
        if (CUDA_CHECK_CU(cuLaunchKernel(model.preprocess_kernel, grid_x, grid_y, 1,
                                         block_x, block_y, 1, 0, model.stream, args, nullptr))) {
            logstream << "tracknet_ball: preprocess kernel launch failed for " << model.engine_path;
            return false;
        }
        return true;
    }

    bool runTripletInference(const av::VideoFrame& f0,
                             const av::VideoFrame& f1,
                             av::VideoFrame& output_frame,
                             const av::VideoFrame& f2) {
        if (!output_contract_validated_) return false;

        ++infer_counter_;
        ModelRunner& model = models_[0];
        if (!runTripletPreprocess(f0, f1, f2, model)) return false;
        // Keep TensorRT enqueue capture/replay behavior from the shared base.
        if (!runInference(model)) return false;
        if (!syncModel(model)) return false;

        const int metadata_w = output_model_width_ > 0 ? output_model_width_ : output_frame.width();
        const int metadata_h = output_model_height_ > 0 ? output_model_height_ : output_frame.height();
        if (emitsDetectionMetadata(output_mode_)) {
            const std::string md = buildDetectionMetadata(model, metadata_w, metadata_h);
            av_dict_set(&output_frame.raw()->metadata, metadata_key_detection_.c_str(), md.c_str(), 0);
            storeTracknetMetadata(metadata_key_detection_, md);

            if (debug_log_metadata_ && debug_log_every_n_ > 0 &&
                (infer_counter_ % (uint64_t)debug_log_every_n_) == 0) {
                logstream << "tracknet_ball: frame=" << frame_counter_ << " " << md;
            }
        }

        if (emitsRawMetadata(output_mode_)) {
            const std::string raw_md = buildRawMetadata(model, output_frame.width(), output_frame.height());
            av_dict_set(&output_frame.raw()->metadata, metadata_key_raw_.c_str(), raw_md.c_str(), 0);
            storeTracknetMetadata(metadata_key_raw_, raw_md);

            if (debug_log_metadata_ && debug_log_every_n_ > 0 &&
                (infer_counter_ % (uint64_t)debug_log_every_n_) == 0) {
                logstream << "tracknet_ball: frame=" << frame_counter_
                          << " raw_metadata_key=" << metadata_key_raw_
                          << " raw_metadata_bytes=" << raw_md.size()
                          << " outputs=" << model.outputs.size();
            }
        }
        if (emitsSrsBallMetadata(output_mode_)) {
            const std::string srs_md = buildSrsBallMetadata(model, output_frame.width(), output_frame.height());
            av_dict_set(&output_frame.raw()->metadata, metadata_key_srs_ball_.c_str(), srs_md.c_str(), 0);
            storeTracknetMetadata(metadata_key_srs_ball_, srs_md);

            if (debug_log_metadata_ && debug_log_every_n_ > 0 &&
                (infer_counter_ % (uint64_t)debug_log_every_n_) == 0) {
                logstream << "tracknet_ball: frame=" << frame_counter_
                          << " srs_metadata_key=" << metadata_key_srs_ball_
                          << " " << srs_md;
            }
        }
        return true;
    }

    std::string buildRawMetadata(const ModelRunner& model, int source_w, int source_h) {
        Parameters j;
        j["schema"] = "tracknet_raw_outputs_v1";
        j["coord_space"] = "tensor";
        j["source_width"] = source_w;
        j["source_height"] = source_h;

        Parameters input;
        input["name"] = model.input_tensor_name;
        input["dtype"] = dtypeName(model.input_dtype);
        input["dims"] = dimsToJson(model.input_dims);
        input["width"] = model.input_w;
        input["height"] = model.input_h;
        input["channels"] = inputChannels(model.input_dims);
        j["input"] = input;

        j["models"] = Parameters::array();
        Parameters model_item;
        model_item["model_index"] = 0;
        model_item["engine"] = model.engine_path;
        model_item["engine_name"] = model.engine_name;
        j["models"].push_back(model_item);

        size_t emitted_elements = 0;
        size_t nonfinite_elements = 0;
        size_t total_elements = 0;
        j["outputs"] = Parameters::array();
        for (const OutputTensor& ot : model.outputs) {
            const size_t tensor_elements = ot.host_output.size();
            total_elements += tensor_elements;

            Parameters item;
            item["name"] = ot.name;
            item["dtype"] = dtypeName(ot.dtype);
            item["dims"] = dimsToJson(ot.dims);
            item["size"] = tensorElementsToJsonSize(tensor_elements);
            item["values"] = tensorValuesToJson(
                ot.host_output, raw_output_max_elements_, &emitted_elements, &nonfinite_elements);
            if (raw_output_max_elements_ > 0 && tensor_elements > (size_t)raw_output_max_elements_) {
                item["truncated"] = true;
                item["emitted_size"] = raw_output_max_elements_;
            }
            j["outputs"].push_back(item);
        }
        j["output_count"] = model.outputs.size();
        j["total_elements"] = tensorElementsToJsonSize(total_elements);
        j["emitted_elements"] = tensorElementsToJsonSize(emitted_elements);
        if (nonfinite_elements > 0) {
            j["nonfinite_elements"] = tensorElementsToJsonSize(nonfinite_elements);
            j["nonfinite_encoding"] = "null";
        }
        return j.dump();
    }

    std::vector<SrsCandidate> extractSrsCandidates(const ModelRunner& model, int source_w, int source_h) const {
        std::vector<SrsCandidate> candidates;
        if (source_w <= 0 || source_h <= 0) {
            return candidates;
        }
        if (srs_heatmap_tensor_index_ < 0 || (size_t)srs_heatmap_tensor_index_ >= model.outputs.size()) {
            return candidates;
        }

        const OutputTensor& ot = model.outputs[(size_t)srs_heatmap_tensor_index_];
        int channels = 0;
        int heatmap_h = 0;
        int heatmap_w = 0;
        if (!getHeatmapShape(ot, channels, heatmap_h, heatmap_w) || srs_channel_ < 0 || srs_channel_ >= channels) {
            return candidates;
        }

        const size_t plane_size = (size_t)heatmap_w * (size_t)heatmap_h;
        const size_t offset = (size_t)srs_channel_ * plane_size;
        if (offset + plane_size > ot.host_output.size() || plane_size == 0) {
            return candidates;
        }

        std::vector<float> heatmap(plane_size);
        std::vector<uint8_t> active(plane_size, 0);
        bool any_active = false;
        for (size_t i = 0; i < plane_size; ++i) {
            const float v = sigmoidFloat(ot.host_output[offset + i]);
            heatmap[i] = v;
            if (v > srs_score_threshold_) {
                active[i] = 1;
                any_active = true;
            }
        }
        if (!any_active) {
            return candidates;
        }

        std::vector<uint8_t> visited(plane_size, 0);
        std::vector<int> stack;
        stack.reserve(256);
        const float src_scale = (float)std::max(source_w, source_h);
        const float inv_factor = src_scale / (float)heatmap_w;
        const float src_cx = 0.5f * (float)source_w;
        const float src_cy = 0.5f * (float)source_h;
        const float hm_cx = 0.5f * (float)heatmap_w;
        const float hm_cy = 0.5f * (float)heatmap_h;

        for (int start_y = 0; start_y < heatmap_h; ++start_y) {
            for (int start_x = 0; start_x < heatmap_w; ++start_x) {
                const int start_idx = start_y * heatmap_w + start_x;
                if (!active[(size_t)start_idx] || visited[(size_t)start_idx]) continue;

                stack.clear();
                stack.push_back(start_idx);
                visited[(size_t)start_idx] = 1;

                float sum_w = 0.0f;
                float sum_x = 0.0f;
                float sum_y = 0.0f;
                int count = 0;

                while (!stack.empty()) {
                    const int idx = stack.back();
                    stack.pop_back();
                    const int y = idx / heatmap_w;
                    const int x = idx - y * heatmap_w;
                    const float w = heatmap[(size_t)idx];

                    if (srs_use_hm_weight_) {
                        sum_w += w;
                        sum_x += (float)x * w;
                        sum_y += (float)y * w;
                    } else {
                        sum_x += (float)x;
                        sum_y += (float)y;
                    }
                    ++count;

                    for (int dy = -1; dy <= 1; ++dy) {
                        const int ny = y + dy;
                        if (ny < 0 || ny >= heatmap_h) continue;
                        for (int dx = -1; dx <= 1; ++dx) {
                            if (dx == 0 && dy == 0) continue;
                            const int nx = x + dx;
                            if (nx < 0 || nx >= heatmap_w) continue;
                            const int nidx = ny * heatmap_w + nx;
                            if (!active[(size_t)nidx] || visited[(size_t)nidx]) continue;
                            visited[(size_t)nidx] = 1;
                            stack.push_back(nidx);
                        }
                    }
                }

                if (count <= 0) continue;
                const float denom = srs_use_hm_weight_ ? sum_w : (float)count;
                if (denom <= 0.0f || !std::isfinite(denom)) continue;
                const float hm_x = sum_x / denom;
                const float hm_y = sum_y / denom;
                const float src_x = (hm_x - hm_cx) * inv_factor + src_cx;
                const float src_y = (hm_y - hm_cy) * inv_factor + src_cy;
                const float score = srs_use_hm_weight_ ? sum_w : (float)count;
                candidates.push_back({src_x, src_y, score});
            }
        }
        return candidates;
    }

    std::string buildSrsBallMetadata(const ModelRunner& model, int source_w, int source_h) {
        Parameters j;
        j["frame"] = infer_counter_ > 0 ? (int64_t)(infer_counter_ - 1) : 0;
        j["bboxes"] = Parameters::array();

        const std::vector<SrsCandidate> candidates = extractSrsCandidates(model, source_w, source_h);
        if (candidates.empty() || source_w <= 0 || source_h <= 0) {
            return j.dump();
        }

        const SrsCandidate* best = nullptr;
        for (const SrsCandidate& candidate : candidates) {
            if (!best || candidate.score > best->score) {
                best = &candidate;
            }
        }
        if (!best) {
            return j.dump();
        }

        Parameters bbox;
        bbox["x"] = best->x / (float)source_w;
        bbox["y"] = best->y / (float)source_h;
        bbox["score"] = best->score;
        j["bboxes"].push_back(bbox);
        return j.dump();
    }

    std::string buildDetectionMetadata(const ModelRunner& model, int metadata_w, int metadata_h) {
        Parameters j;
        j["coord_space"] = "model";
        j["model_width"] = metadata_w;
        j["model_height"] = metadata_h;

        j["models"] = Parameters::array();
        Parameters model_item;
        model_item["model_index"] = 0;
        model_item["engine"] = model.engine_path;
        model_item["engine_name"] = model.engine_name;
        j["models"].push_back(model_item);

        j["detections"] = Parameters::array();
        if (output_tensor_index_ < 0 || (size_t)output_tensor_index_ >= model.outputs.size()) {
            ++empty_frames_;
            ++detection_count_histogram_[0];
            return j.dump();
        }

        const OutputTensor& ot = model.outputs[(size_t)output_tensor_index_];
        if (ot.host_output.size() < 6 || metadata_w <= 0 || metadata_h <= 0) {
            ++empty_frames_;
            ++detection_count_histogram_[0];
            return j.dump();
        }

        const float x1n = clamp01(ot.host_output[0]);
        const float y1n = clamp01(ot.host_output[1]);
        const float x2n = clamp01(ot.host_output[2]);
        const float y2n = clamp01(ot.host_output[3]);
        const float score = std::isfinite(ot.host_output[4]) ? ot.host_output[4] : 0.0f;
        const float visible = std::isfinite(ot.host_output[5]) ? ot.host_output[5] : 0.0f;

        const bool keep = score >= conf_thresh_ && (emit_invisible_ || visible >= visible_thresh_);
        if (!keep) {
            ++empty_frames_;
            ++detection_count_histogram_[0];
            return j.dump();
        }

        Parameters item;
        item["cls"] = 0;
        item["conf"] = score;
        item["xyxy"] = {
            x1n * (float)metadata_w,
            y1n * (float)metadata_h,
            x2n * (float)metadata_w,
            y2n * (float)metadata_h
        };
        item["model_index"] = 0;
        item["engine_name"] = model.engine_name;
        item["label"] = target_label_;
        item["visible"] = visible;
        item["tracknet_score"] = score;
        j["detections"].push_back(item);

        ++detected_frames_;
        ++detection_count_histogram_[1];
        int bucket = std::max(0, std::min((int)(score * 10.0f), 9));
        ++conf_histogram_[(size_t)bucket];
        return j.dump();
    }

    static std::string parseEnginePath(const Parameters& params) {
        if (params.count("engine") && params["engine"].is_string()) {
            return params["engine"].get<std::string>();
        }
        if (params.count("models") && params["models"].is_array() && params["models"].size() == 1 &&
            params["models"][0].is_object() &&
            params["models"][0].count("engine") && params["models"][0]["engine"].is_string()) {
            return params["models"][0]["engine"].get<std::string>();
        }
        throw Error("tracknet_ball: missing required parameter: engine (or models[0].engine)");
    }

    static std::string parseLabel(const Parameters& params) {
        std::string label = jsonStringParam(params, "target_label", "basketball");
        label = jsonStringParam(params, "label", label);
        if (params.count("models") && params["models"].is_array() && params["models"].size() == 1 &&
            params["models"][0].is_object() && params["models"][0].count("class_names") &&
            params["models"][0]["class_names"].is_array() && !params["models"][0]["class_names"].empty() &&
            params["models"][0]["class_names"][0].is_string()) {
            label = params["models"][0]["class_names"][0].get<std::string>();
        }
        return label;
    }

    static int64_t tensorElementsToJsonSize(size_t value) {
        return value > (size_t)std::numeric_limits<int64_t>::max()
            ? std::numeric_limits<int64_t>::max()
            : (int64_t)value;
    }

public:
    static std::shared_ptr<TrackNetBall> create(NodeCreationInfo &nci) {
        EdgeManager &edges = nci.edges;
        const Parameters &params = nci.params;

        std::shared_ptr<Edge<av::VideoFrame>> src = edges.find<av::VideoFrame>(params["src"]);
        std::shared_ptr<Edge<av::VideoFrame>> dst = edges.find<av::VideoFrame>(params["dst"]);

        auto r = std::make_shared<TrackNetBall>(src->makeSource(), dst->makeSink());
        r->node_label_ = params.value("name", std::string("<unnamed>"));
        std::shared_ptr<HWAccelDevice> debug_hwaccel;
        if (params.count("hwaccel")) {
            debug_hwaccel = InstanceSharedObjects<HWAccelDevice>::get(nci.instance, params["hwaccel"]);
        }
        r->setCudaContextDebugInfo("tracknet_ball", r->node_label_, debug_hwaccel);

        r->metadata_key_detection_ = jsonStringParam(params, "metadata_key_detection", r->metadata_key_detection_);
        r->metadata_key_detection_ = jsonStringParam(params, "metadata_key", r->metadata_key_detection_);
        r->metadata_key_raw_ = jsonStringParam(params, "metadata_key_raw", r->metadata_key_raw_);
        r->metadata_key_raw_ = jsonStringParam(params, "raw_metadata_key", r->metadata_key_raw_);
        r->metadata_key_srs_ball_ = jsonStringParam(params, "metadata_key_srs", r->metadata_key_srs_ball_);
        r->metadata_key_srs_ball_ = jsonStringParam(params, "srs_metadata_key", r->metadata_key_srs_ball_);
        r->output_mode_ = parseOutputModeString(jsonStringParam(params, "output_mode", "detection"));
        const std::string default_alignment = emitsSrsBallMetadata(r->output_mode_) ? "latest" : "center";
        r->triplet_alignment_ = parseTripletAlignmentString(
            jsonStringParam(params, "triplet_alignment", default_alignment));
        const std::string default_preprocess = emitsSrsBallMetadata(r->output_mode_) ? "srs_affine" : "resize";
        r->preprocess_mode_ = parsePreprocessModeString(
            jsonStringParam(params, "preprocess_mode", default_preprocess));
        r->sample_fill_mode_ = parseSampleFillModeString(
            jsonStringParam(params, "sample_fill_mode", "none"));
        r->sample_fill_mode_ = parseSampleFillModeString(
            jsonStringParam(params, "tracknet_sample_fill_mode",
                            r->sample_fill_mode_ == TrackNetSampleFillMode::Hold ? "hold" : "none"));
        if (r->output_mode_ == TrackNetOutputMode::Raw && params.count("metadata_key") &&
            !params.count("metadata_key_raw") && !params.count("raw_metadata_key")) {
            r->metadata_key_raw_ = params["metadata_key"].get<std::string>();
        }
        if (emitsSrsBallMetadata(r->output_mode_) && params.count("metadata_key") &&
            !params.count("metadata_key_srs") && !params.count("srs_metadata_key")) {
            r->metadata_key_srs_ball_ = params["metadata_key"].get<std::string>();
        }
        r->target_label_ = parseLabel(params);
        r->conf_thresh_ = jsonFloatParam(params, "conf_thresh", r->conf_thresh_);
        r->visible_thresh_ = jsonFloatParam(params, "visible_thresh", r->visible_thresh_);
        r->srs_score_threshold_ = jsonFloatParam(params, "srs_score_threshold", r->srs_score_threshold_);
        r->emit_invisible_ = jsonBoolParam(params, "emit_invisible", r->emit_invisible_);
        r->srs_use_hm_weight_ = jsonBoolParam(params, "srs_use_hm_weight", r->srs_use_hm_weight_);
        r->use_cuda_graph_ = jsonBoolParam(params, "use_cuda_graph", r->use_cuda_graph_);
        r->debug_log_metadata_ = jsonBoolParam(params, "debug_log_metadata", r->debug_log_metadata_);
        r->debug_log_every_n_ = jsonIntParam(params, "debug_log_every_n", r->debug_log_every_n_);
        rejectLegacySamplingParams(params);
        r->auto_sample_min_fps_ = jsonFpsParam(params, "auto_sample_min_fps", r->auto_sample_min_fps_);
        r->auto_sample_min_fps_ = jsonFpsParam(params, "tracknet_auto_sample_min_fps", r->auto_sample_min_fps_);
        r->auto_sample_min_fps_ = jsonFpsParam(params, "auto_sample_fps_threshold", r->auto_sample_min_fps_);
        r->auto_sample_every_n_ = jsonIntParam(params, "auto_sample_every_n", r->auto_sample_every_n_);
        r->auto_sample_every_n_ = jsonIntParam(params, "tracknet_auto_sample_every_n", r->auto_sample_every_n_);
        r->auto_sample_every_n_ = jsonIntParam(params, "auto_sample_divisor", r->auto_sample_every_n_);
        r->raw_output_max_elements_ = jsonIntParam(params, "raw_output_max_elements", r->raw_output_max_elements_);
        r->raw_output_max_elements_ = jsonIntParam(
            params, "raw_output_max_elements_per_tensor", r->raw_output_max_elements_);
        r->srs_channel_ = jsonIntParam(params, "srs_channel", r->srs_channel_);
        r->output_model_width_ = jsonIntParam(params, "output_model_width", r->output_model_width_);
        r->output_model_height_ = jsonIntParam(params, "output_model_height", r->output_model_height_);
        if ((r->output_model_width_ < 0) || (r->output_model_height_ < 0)) {
            throw Error("tracknet_ball: output_model_width/output_model_height must be >= 0");
        }
        if (r->raw_output_max_elements_ < 0) {
            throw Error("tracknet_ball: raw_output_max_elements must be >= 0");
        }
        if (r->auto_sample_min_fps_ < 0.0) {
            throw Error("tracknet_ball: auto_sample_min_fps must be >= 0");
        }
        if (r->auto_sample_every_n_ < 1) {
            throw Error("tracknet_ball: auto_sample_every_n must be >= 1");
        }
        if (r->auto_sample_min_fps_ > 0.0 && r->auto_sample_every_n_ > 1 &&
            r->triplet_alignment_ != TrackNetTripletAlignment::Latest) {
            throw Error("tracknet_ball: auto sampling every N frames requires triplet_alignment='latest'");
        }
        if (r->srs_channel_ < 0) {
            throw Error("tracknet_ball: srs_channel must be >= 0");
        }
        if (r->srs_score_threshold_ < 0.0f || r->srs_score_threshold_ > 1.0f) {
            throw Error("tracknet_ball: srs_score_threshold must be between 0 and 1");
        }

        ModelRunner model;
        model.engine_path = parseEnginePath(params);
        model.engine_name = std::filesystem::path(model.engine_path).filename().string();
        model.class_names.push_back(r->target_label_);
        r->models_.push_back(std::move(model));

        return r;
    }
};

DECLNODE(tracknet_ball, TrackNetBall)
