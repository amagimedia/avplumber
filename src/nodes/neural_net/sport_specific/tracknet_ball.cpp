#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <deque>
#include <filesystem>
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

std::string jsonStringParam(const Parameters& params, const char* key, const std::string& fallback) {
    if (!params.count(key) || params[key].is_null()) return fallback;
    return params[key].get<std::string>();
}

} // namespace

class TrackNetBall : public NodeSISO<av::VideoFrame, av::VideoFrame>,
                     public IInputReset,
                     public ReportsFinishByFlag,
                     public CudaInferTrtBase {
protected:
    std::string metadata_key_detection_ = "yolo_ball";
    std::string target_label_ = "basketball";
    float conf_thresh_ = 0.5f;
    float visible_thresh_ = 0.5f;
    bool emit_invisible_ = false;
    int output_model_width_ = 0;
    int output_model_height_ = 0;
    bool debug_log_metadata_ = false;
    int debug_log_every_n_ = 0;

    std::deque<av::VideoFrame> frame_buffer_;
    bool first_boundary_emitted_ = false;
    bool center_output_started_ = false;
    int source_w_ = 0;
    int source_h_ = 0;
    AVPixelFormat source_sw_format_ = AV_PIX_FMT_NONE;

    int output_tensor_index_ = -1;
    bool output_contract_validated_ = false;
    uint64_t frame_counter_ = 0;
    uint64_t infer_counter_ = 0;
    uint64_t detected_frames_ = 0;
    uint64_t empty_frames_ = 0;
    std::array<uint64_t, 10> conf_histogram_{};
    std::map<int, uint64_t> detection_count_histogram_;

public:
    TrackNetBall(std::unique_ptr<Source<av::VideoFrame>> source,
                 std::unique_ptr<EdgeSink<av::VideoFrame>> sink)
        : NodeSISO(std::move(source), std::move(sink)) {
        expected_input_channels_ = 9;
    }

    ~TrackNetBall() {
        const uint64_t total = detected_frames_ + empty_frames_;
        if (total == 0) return;
        logstream << "tracknet_ball: detection summary:"
                  << " detected frames: " << detected_frames_
                  << " / total inferred frames: " << total
                  << " (" << (100.0 * (double)detected_frames_ / (double)std::max<uint64_t>(1, total)) << "%)";
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

        if (!isSupportedCudaFrame(frm)) {
            flushBufferedBoundaryFrames();
            this->sink_->put(frm);
            return;
        }

        if (!ensureTracknetInitialized(frm)) {
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

        frame_buffer_.push_back(frm);
        if (frame_buffer_.size() < 3) {
            return;
        }

        if (!first_boundary_emitted_) {
            this->sink_->put(frame_buffer_.front());
            first_boundary_emitted_ = true;
        }

        av::VideoFrame& center = frame_buffer_[1];
        if (!runTripletInference(frame_buffer_[0], center, frame_buffer_[2])) {
            return;
        }

        center_output_started_ = true;
        this->sink_->put(center);
        frame_buffer_.pop_front();
    }

protected:
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
        const char* kname = (model.input_dtype == nvinfer1::DataType::kHALF)
            ? "kTrackNetNV12TripletToNCHW9_fp16"
            : "kTrackNetNV12TripletToNCHW9_fp32";
        if (CUDA_CHECK_CU(cuModuleGetFunction(&model.preprocess_kernel, preprocess_module_, kname))) {
            logstream << "tracknet_ball: failed to get preprocess kernel for " << model.engine_path;
            return false;
        }
        return configureRunnerStream(model);
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
        if (!parseEngine(model) || !allocateBindings(model) ||
            !ensureCompatibleInput(model, 0) || !configureTracknetPreprocess(model) ||
            !validateOutputContract(model)) {
            cleanupModel(model);
            return false;
        }

        initialized_ = true;
        return true;
    }

    bool validateOutputContract(ModelRunner& model) {
        if (model.input_c != 9 || model.input_w <= 0 || model.input_h <= 0) {
            logstream << "tracknet_ball: engine must have NCHW input with 9 channels";
            return false;
        }

        for (size_t i = 0; i < model.outputs.size(); ++i) {
            const OutputTensor& ot = model.outputs[i];
            if (volume(ot.dims) >= 6) {
                output_tensor_index_ = (int)i;
                output_contract_validated_ = true;
                if (debug_log_metadata_) {
                    logstream << "tracknet_ball: output tensor=" << ot.name
                              << " dims=" << ot.dims.nbDims
                              << " volume=" << volume(ot.dims)
                              << " input=" << model.input_w << "x" << model.input_h;
                }
                return true;
            }
        }

        logstream << "tracknet_ball: failed to find output tensor with [x1,y1,x2,y2,score,visible]";
        return false;
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
                             av::VideoFrame& center,
                             const av::VideoFrame& f2) {
        if (!output_contract_validated_ || output_tensor_index_ < 0) return false;

        ++infer_counter_;
        ModelRunner& model = models_[0];
        if (!runTripletPreprocess(f0, center, f2, model)) return false;
        // Keep TensorRT enqueue capture/replay behavior from the shared base.
        if (!runInference(model)) return false;
        if (!syncModel(model)) return false;

        const int metadata_w = output_model_width_ > 0 ? output_model_width_ : center.width();
        const int metadata_h = output_model_height_ > 0 ? output_model_height_ : center.height();
        const std::string md = buildDetectionMetadata(model, metadata_w, metadata_h);
        av_dict_set(&center.raw()->metadata, metadata_key_detection_.c_str(), md.c_str(), 0);

        if (debug_log_metadata_ && debug_log_every_n_ > 0 &&
            (infer_counter_ % (uint64_t)debug_log_every_n_) == 0) {
            logstream << "tracknet_ball: frame=" << frame_counter_ << " " << md;
        }
        return true;
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

public:
    static std::shared_ptr<TrackNetBall> create(NodeCreationInfo &nci) {
        EdgeManager &edges = nci.edges;
        const Parameters &params = nci.params;

        std::shared_ptr<Edge<av::VideoFrame>> src = edges.find<av::VideoFrame>(params["src"]);
        std::shared_ptr<Edge<av::VideoFrame>> dst = edges.find<av::VideoFrame>(params["dst"]);

        auto r = std::make_shared<TrackNetBall>(src->makeSource(), dst->makeSink());

        r->metadata_key_detection_ = jsonStringParam(params, "metadata_key_detection", r->metadata_key_detection_);
        r->metadata_key_detection_ = jsonStringParam(params, "metadata_key", r->metadata_key_detection_);
        r->target_label_ = parseLabel(params);
        r->conf_thresh_ = jsonFloatParam(params, "conf_thresh", r->conf_thresh_);
        r->visible_thresh_ = jsonFloatParam(params, "visible_thresh", r->visible_thresh_);
        r->emit_invisible_ = jsonBoolParam(params, "emit_invisible", r->emit_invisible_);
        r->use_cuda_graph_ = jsonBoolParam(params, "use_cuda_graph", r->use_cuda_graph_);
        r->debug_log_metadata_ = jsonBoolParam(params, "debug_log_metadata", r->debug_log_metadata_);
        r->debug_log_every_n_ = jsonIntParam(params, "debug_log_every_n", r->debug_log_every_n_);
        r->output_model_width_ = jsonIntParam(params, "output_model_width", r->output_model_width_);
        r->output_model_height_ = jsonIntParam(params, "output_model_height", r->output_model_height_);
        if ((r->output_model_width_ < 0) || (r->output_model_height_ < 0)) {
            throw Error("tracknet_ball: output_model_width/output_model_height must be >= 0");
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
