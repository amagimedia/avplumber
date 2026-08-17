#include "../../node_common.hpp"
#include "../common/infer_trt_base.hpp"
#include "doctr_recognizer.hpp"
#include "ocr_trt_runner.hpp"
#include "scoreboard_post_process.hpp"

extern "C" {
#include <libavutil/dict.h>
#include <libavutil/hwcontext.h>
#include <libavutil/hwcontext_cuda.h>
}

#include "../../../../objs/src/nodes/neural_net/preprocess/nv12_doctr_preprocess.ptx.h"

#include <NvInfer.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr int kLevel1H = 384;
constexpr int kLevel1W = 640;
constexpr int kLevel1Rows = 300;
constexpr int kLevel2Size = 1024;
constexpr int kLevel2Rows = 50;
constexpr int kClassCount = 7;

const char* const kClassNames[kClassCount] = {
    "team_name", "team_score", "team_logo", "team_name_logo_score",
    "game_clock", "shot_clock", "quarter",
};

using Box = scoreboard_post::Box;

struct Detection {
    Box box;
    float confidence = 0.0f;
    int cls = -1;
    std::string label;
    std::string text;
    float recognition_confidence = 0.0f;
};

float boxIou(const Box& a, const Box& b) {
    return scoreboard_post::iou(a, b);
}

float centerX(const Box& box) { return (box.x1 + box.x2) * 0.5f; }
float centerY(const Box& box) { return (box.y1 + box.y2) * 0.5f; }

Box clampBox(Box box, int width, int height) {
    box.x1 = std::clamp(box.x1, 0.0f, (float)width);
    box.x2 = std::clamp(box.x2, 0.0f, (float)width);
    box.y1 = std::clamp(box.y1, 0.0f, (float)height);
    box.y2 = std::clamp(box.y2, 0.0f, (float)height);
    return box;
}

bool validBox(const Box& box) {
    return std::isfinite(box.x1) && std::isfinite(box.y1) &&
           std::isfinite(box.x2) && std::isfinite(box.y2) &&
           box.x2 > box.x1 && box.y2 > box.y1;
}

double parseFps(const std::string& fps) {
    if (fps.empty()) return 25.0;
    const size_t slash = fps.find('/');
    try {
        if (slash == std::string::npos) return std::stod(fps);
        const double numerator = std::stod(fps.substr(0, slash));
        const double denominator = std::stod(fps.substr(slash + 1));
        return denominator > 0.0 ? numerator / denominator : numerator;
    } catch (...) {
        return 25.0;
    }
}

void requireCuda(CUresult result, const char* operation) {
    if (result == CUDA_SUCCESS) return;
    const char* name = nullptr;
    const char* description = nullptr;
    cuGetErrorName(result, &name);
    cuGetErrorString(result, &description);
    throw Error(std::string("scoreboard_two_stage CUDA failure in ") + operation + ": " +
                (name ? name : "unknown") + " (" +
                (description ? description : "no description") + ")");
}

Parameters detectionJson(const Detection& detection) {
    Parameters result;
    result["bbox"] = {detection.box.x1, detection.box.y1,
                      detection.box.x2, detection.box.y2};
    result["det_conf"] = detection.confidence;
    result["reco_conf"] = detection.recognition_confidence;
    result["text"] = detection.text;
    result["label"] = detection.label;
    result["base_label"] = kClassNames[detection.cls];
    result["region_id"] = 0;
    return result;
}

} // namespace

class ScoreboardTwoStageOcr : public NodeSISO<av::VideoFrame, av::VideoFrame>, public ReportsFinishByFlag {
    std::string level1_engine_;
    std::string level2_engine_;
    std::string recognizer_engine_;
    std::string metadata_key_ = "text_detections";
    std::string node_label_ = "<unnamed>";
    float target_fps_ = 2.0f;
    double input_fps_ = 25.0;
    int sample_every_n_ = 13;
    float level1_confidence_ = 0.5f;
    float level2_confidence_ = 0.25f;
    float recognition_confidence_ = 0.0f;
    int vertical_padding_ = 50;
    int max_boxes_ = 48;
    float edge_margin_ = 0.2f;
    int edge_margin_px_ = 0;
    std::vector<float> region_x_;
    int debug_log_every_n_ = 0;
    bool post_process_ = false;
    uint64_t frame_counter_ = 0;
    uint64_t sample_counter_ = 0;
    scoreboard_post::PostProcessor post_processor_;

    std::shared_ptr<HWAccelDevice> hwaccel_;
    AVCUDADeviceContext* cuda_device_context_ = nullptr;
    CUcontext cuda_context_ = nullptr;
    CUmodule preprocess_module_ = nullptr;
    CUfunction level1_preprocess_ = nullptr;
    CUfunction level2_preprocess_ = nullptr;
    CUdeviceptr device_box_ = 0;
    bool initialized_ = false;
    bool preinitialized_ = false;
    bool frame_context_checked_ = false;
    ocr::TrtLogger logger_;
    ocr::TrtRunner level1_;
    ocr::TrtRunner level2_;
    ocr::DoctrRecognizer recognizer_;

    static bool frameCudaContext(const av::VideoFrame& frame, CUcontext& context,
                                 AVCUDADeviceContext** device_context = nullptr) {
        context = nullptr;
        if (device_context) *device_context = nullptr;
        if (!frame.raw() || !frame.raw()->hw_frames_ctx || !frame.raw()->hw_frames_ctx->data) return false;
        AVHWFramesContext* frames_context =
            reinterpret_cast<AVHWFramesContext*>(frame.raw()->hw_frames_ctx->data);
        if (!frames_context || !frames_context->device_ctx || !frames_context->device_ctx->hwctx) return false;
        auto* cuda = reinterpret_cast<AVCUDADeviceContext*>(frames_context->device_ctx->hwctx);
        if (!cuda || !cuda->cuda_ctx) return false;
        context = cuda->cuda_ctx;
        if (device_context) *device_context = cuda;
        return true;
    }

    void cleanup() noexcept {
        if (cuda_context_) cuCtxSetCurrent(cuda_context_);
        level1_.cleanup();
        level2_.cleanup();
        recognizer_.cleanup();
        if (device_box_) { cuMemFree(device_box_); device_box_ = 0; }
        if (preprocess_module_) { cuModuleUnload(preprocess_module_); preprocess_module_ = nullptr; }
        level1_preprocess_ = nullptr;
        level2_preprocess_ = nullptr;
        cuda_device_context_ = nullptr;
        cuda_context_ = nullptr;
        initialized_ = false;
        preinitialized_ = false;
        frame_context_checked_ = false;
    }

    void initInCurrentContext() {
        if (!cuda_context_) throw Error("scoreboard_two_stage: no CUDA context");
        requireCuda(cuCtxSetCurrent(cuda_context_), "cuCtxSetCurrent(init)");
        const std::string ptx(avpl_doctr_preprocess_ptx,
                              avpl_doctr_preprocess_ptx + avpl_doctr_preprocess_ptx_len);
        requireCuda(cuModuleLoadDataEx(&preprocess_module_, ptx.c_str(), 0, nullptr, nullptr),
                    "cuModuleLoadDataEx");
        requireCuda(cuModuleGetFunction(&level1_preprocess_, preprocess_module_,
                                        "kNV12_scoreboard_letterbox_f32"),
                    "cuModuleGetFunction(level1)");
        requireCuda(cuModuleGetFunction(&level2_preprocess_, preprocess_module_,
                                        "kNV12_scoreboard_letterbox_u8"),
                    "cuModuleGetFunction(level2)");
        level1_.init(logger_, level1_engine_,
                     ocr::TensorContract{"images", nvinfer1::DataType::kFLOAT, {1, 3, kLevel1H, kLevel1W}},
                     ocr::TensorContract{"output0", nvinfer1::DataType::kFLOAT, {1, kLevel1Rows, 6}});
        level2_.init(logger_, level2_engine_,
                     ocr::TensorContract{"images_uint8", nvinfer1::DataType::kUINT8, {1, 3, kLevel2Size, kLevel2Size}},
                     ocr::TensorContract{"output0", nvinfer1::DataType::kFLOAT, {1, kLevel2Rows, 6}});
        recognizer_.init(logger_, preprocess_module_, recognizer_engine_, max_boxes_);
        requireCuda(cuMemAlloc(&device_box_, 4 * sizeof(int)), "cuMemAlloc(box)");
        initialized_ = true;
    }

    void preinitialize() {
        if (!hwaccel_ || !hwaccel_->deviceContext() || !hwaccel_->deviceContext()->data) {
            throw Error("scoreboard_two_stage: CUDA hwaccel is required");
        }
        AVHWDeviceContext* device =
            reinterpret_cast<AVHWDeviceContext*>(hwaccel_->deviceContext()->data);
        if (!device || device->type != AV_HWDEVICE_TYPE_CUDA || !device->hwctx) {
            throw Error("scoreboard_two_stage: hwaccel is not CUDA");
        }
        cuda_device_context_ = reinterpret_cast<AVCUDADeviceContext*>(device->hwctx);
        cuda_context_ = cuda_device_context_->cuda_ctx;
        const auto start = std::chrono::steady_clock::now();
        initInCurrentContext();
        preinitialized_ = true;
        const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - start).count();
        logstream << "neural_preinit status=ok type=scoreboard_two_stage_ocr node="
                  << node_label_ << " init_ms=" << elapsed;
    }

    void ensureFrameContext(const av::VideoFrame& frame) {
        CUcontext frame_context = nullptr;
        AVCUDADeviceContext* frame_device_context = nullptr;
        if (!frameCudaContext(frame, frame_context, &frame_device_context)) {
            throw Error("scoreboard_two_stage: input is not a CUDA hardware frame");
        }
        if (!initialized_) {
            cuda_context_ = frame_context;
            cuda_device_context_ = frame_device_context;
            initInCurrentContext();
            return;
        }
        if (preinitialized_ && !frame_context_checked_) {
            if (frame_context != cuda_context_) {
                cleanup();
                cuda_context_ = frame_context;
                cuda_device_context_ = frame_device_context;
                initInCurrentContext();
            }
            frame_context_checked_ = true;
        }
        requireCuda(cuCtxSetCurrent(cuda_context_), "cuCtxSetCurrent(frame)");
    }

    void launchLevel1(const av::VideoFrame& frame) {
        const int box[4] = {0, 0, frame.raw()->width, frame.raw()->height};
        requireCuda(cuMemcpyHtoDAsync(device_box_, box, sizeof(box), level1_.stream()),
                    "cuMemcpyHtoDAsync(level1 box)");
        CUdeviceptr y = (CUdeviceptr)(uintptr_t)frame.raw()->data[0];
        CUdeviceptr uv = (CUdeviceptr)(uintptr_t)frame.raw()->data[1];
        int y_pitch = frame.raw()->linesize[0], uv_pitch = frame.raw()->linesize[1];
        CUdeviceptr output = level1_.inputPtr();
        int height = kLevel1H, width = kLevel1W;
        float pad = 114.0f / 255.0f;
        void* args[] = {&y, &y_pitch, &uv, &uv_pitch, &output, &device_box_, &height, &width, &pad};
        constexpr unsigned int block = 16;
        requireCuda(cuLaunchKernel(level1_preprocess_, (width + 15) / 16, (height + 15) / 16, 1,
                                   block, block, 1, 0, level1_.stream(), args, nullptr),
                    "cuLaunchKernel(level1)");
        level1_.infer();
    }

    void launchLevel2(const av::VideoFrame& frame, const int box[4]) {
        requireCuda(cuMemcpyHtoDAsync(device_box_, box, 4 * sizeof(int), level2_.stream()),
                    "cuMemcpyHtoDAsync(level2 box)");
        CUdeviceptr y = (CUdeviceptr)(uintptr_t)frame.raw()->data[0];
        CUdeviceptr uv = (CUdeviceptr)(uintptr_t)frame.raw()->data[1];
        int y_pitch = frame.raw()->linesize[0], uv_pitch = frame.raw()->linesize[1];
        CUdeviceptr output = level2_.inputPtr();
        int height = kLevel2Size, width = kLevel2Size;
        void* args[] = {&y, &y_pitch, &uv, &uv_pitch, &output, &device_box_, &height, &width};
        constexpr unsigned int block = 16;
        requireCuda(cuLaunchKernel(level2_preprocess_, (width + 15) / 16, (height + 15) / 16, 1,
                                   block, block, 1, 0, level2_.stream(), args, nullptr),
                    "cuLaunchKernel(level2)");
        level2_.infer();
    }

    bool selectedByRegion(const Box& box, int frame_width) const {
        if (region_x_.size() != 2) return true;
        const float normalized_x = centerX(box) / std::max(1.0f, (float)frame_width);
        return normalized_x >= region_x_[0] && normalized_x < region_x_[1];
    }

    bool decodeLevel1(int frame_width, int frame_height, Detection& best) const {
        const float scale = std::min((float)kLevel1W / frame_width, (float)kLevel1H / frame_height);
        const int content_width = std::max(1, std::min(kLevel1W, (int)std::lround(frame_width * scale)));
        const int content_height = std::max(1, std::min(kLevel1H, (int)std::lround(frame_height * scale)));
        const float scale_x = (float)content_width / frame_width;
        const float scale_y = (float)content_height / frame_height;
        const int pad_x = (kLevel1W - content_width) / 2;
        const int pad_y = (kLevel1H - content_height) / 2;
        bool found = false;
        for (int row = 0; row < kLevel1Rows; ++row) {
            const float* item = level1_.output().data() + row * 6;
            if (!std::isfinite(item[4]) || item[4] < level1_confidence_) continue;
            const int cls = (int)std::lround(item[5]);
            if (cls != 0 || (found && item[4] <= best.confidence)) continue;
            Box box{
                (item[0] - pad_x) / scale_x,
                (item[1] - pad_y) / scale_y,
                (item[2] - pad_x) / scale_x,
                (item[3] - pad_y) / scale_y,
            };
            box = clampBox(box, frame_width, frame_height);
            if (!validBox(box)) continue;
            best = Detection{box, item[4], 0, "scoreboard", "", 0.0f};
            found = true;
        }
        return found;
    }

    std::vector<Detection> decodeLevel2(const int crop[4], int frame_width, int frame_height) const {
        const float scale = std::min((float)kLevel2Size / crop[2], (float)kLevel2Size / crop[3]);
        const int content_width = std::max(1, std::min(kLevel2Size, (int)std::lround(crop[2] * scale)));
        const int content_height = std::max(1, std::min(kLevel2Size, (int)std::lround(crop[3] * scale)));
        const float scale_x = (float)content_width / crop[2];
        const float scale_y = (float)content_height / crop[3];
        const int pad_x = (kLevel2Size - content_width) / 2;
        const int pad_y = (kLevel2Size - content_height) / 2;
        std::vector<Detection> decoded;
        for (int row = 0; row < kLevel2Rows; ++row) {
            const float* item = level2_.output().data() + row * 6;
            if (!std::isfinite(item[4]) || item[4] < level2_confidence_) continue;
            const int cls = (int)std::lround(item[5]);
            if (cls < 0 || cls >= kClassCount) continue;
            Box box{
                crop[0] + (item[0] * kLevel2Size - pad_x) / scale_x,
                crop[1] + (item[1] * kLevel2Size - pad_y) / scale_y,
                crop[0] + (item[2] * kLevel2Size - pad_x) / scale_x,
                crop[1] + (item[3] * kLevel2Size - pad_y) / scale_y,
            };
            box = clampBox(box, frame_width, frame_height);
            if (validBox(box) && selectedByRegion(box, frame_width)) {
                decoded.push_back({box, item[4], cls, kClassNames[cls], "", 0.0f});
            }
        }
        std::sort(decoded.begin(), decoded.end(), [](const Detection& a, const Detection& b) {
            return a.confidence > b.confidence;
        });
        std::vector<Detection> deduplicated;
        for (const Detection& detection : decoded) {
            const bool duplicate = std::any_of(deduplicated.begin(), deduplicated.end(),
                [&](const Detection& kept) {
                    return detection.cls == kept.cls && boxIou(detection.box, kept.box) >= 0.5f;
                });
            if (!duplicate) deduplicated.push_back(detection);
        }
        return deduplicated;
    }

    static void keepSingleton(std::vector<Detection>& detections, int cls) {
        bool kept = false;
        detections.erase(std::remove_if(detections.begin(), detections.end(), [&](const Detection& detection) {
            if (detection.cls != cls) return false;
            if (!kept) { kept = true; return false; }
            return true;
        }), detections.end());
    }

    static int countClass(const std::vector<Detection>& detections, int cls) {
        return (int)std::count_if(detections.begin(), detections.end(),
                                  [&](const Detection& detection) { return detection.cls == cls; });
    }

    static void assignTeamLabels(std::vector<Detection>& detections) {
        std::vector<const Detection*> groups;
        for (const Detection& detection : detections) {
            if (detection.cls == 3) groups.push_back(&detection);
        }
        if (groups.size() != 2) return;
        const float dx = std::fabs(centerX(groups[0]->box) - centerX(groups[1]->box));
        const float dy = std::fabs(centerY(groups[0]->box) - centerY(groups[1]->box));
        if ((dx >= dy && centerX(groups[0]->box) > centerX(groups[1]->box)) ||
            (dx < dy && centerY(groups[0]->box) > centerY(groups[1]->box))) {
            std::swap(groups[0], groups[1]);
        }
        for (Detection& detection : detections) {
            if (detection.cls > 3) continue;
            const float x = centerX(detection.box), y = centerY(detection.box);
            int group = -1;
            for (int index = 0; index < 2; ++index) {
                const Box& anchor = groups[(size_t)index]->box;
                if (x >= anchor.x1 && x <= anchor.x2 && y >= anchor.y1 && y <= anchor.y2) {
                    if (group >= 0) { group = -1; break; }
                    group = index;
                }
            }
            if (group >= 0) {
                detection.label = std::string(group == 0 ? "team_a_" : "team_b_") +
                                  std::string(kClassNames[detection.cls]).substr(5);
            }
        }
    }

    static void keepHighestPerLabel(std::vector<Detection>& detections) {
        std::vector<Detection> unique;
        unique.reserve(detections.size());
        for (const Detection& detection : detections) {
            const bool already_kept = std::any_of(unique.begin(), unique.end(), [&](const Detection& kept) {
                return kept.label == detection.label;
            });
            if (!already_kept) unique.push_back(detection);
        }
        detections = std::move(unique);
    }

    void recognizeText(const av::VideoFrame& frame, std::vector<Detection>& detections,
                       int frame_width, int frame_height) {
        std::vector<size_t> indices;
        std::vector<ocr::PixelBox> boxes;
        for (size_t i = 0; i < detections.size(); ++i) {
            const int cls = detections[i].cls;
            if (!(cls == 0 || cls == 1 || cls == 4 || cls == 5 || cls == 6)) continue;
            if ((int)boxes.size() >= recognizer_.batchSize()) break;
            const Box& box = detections[i].box;
            const float margin_x = edge_margin_ * (box.x2 - box.x1) + edge_margin_px_;
            const float margin_y = edge_margin_ * (box.y2 - box.y1) + edge_margin_px_;
            const int x1 = std::max(0, (int)std::floor(box.x1 - margin_x));
            const int y1 = std::max(0, (int)std::floor(box.y1 - margin_y));
            const int x2 = std::min(frame_width, (int)std::ceil(box.x2 + margin_x));
            const int y2 = std::min(frame_height, (int)std::ceil(box.y2 + margin_y));
            indices.push_back(i);
            boxes.push_back({x1, y1, std::max(x1 + 1, x2), std::max(y1 + 1, y2)});
        }
        const std::vector<ocr::Recognition> results = recognizer_.recognize(frame, boxes);
        for (size_t i = 0; i < results.size(); ++i) {
            if (results[i].confidence >= recognition_confidence_) {
                detections[indices[i]].text = results[i].text;
                detections[indices[i]].recognition_confidence = results[i].confidence;
            }
        }
    }

    Parameters emptyPayload(bool sampled, const std::string& reason) const {
        Parameters output;
        output["schema"] = "scoreboard_two_stage_v1";
        output["ocr_sampled"] = sampled;
        output["detections"] = Parameters::array();
        output["draw_detections"] = Parameters::array();
        output["reason"] = reason;
        return output;
    }

public:
    using NodeSISO<av::VideoFrame, av::VideoFrame>::NodeSISO;

    ~ScoreboardTwoStageOcr() override { cleanup(); }

    void process() override {
        av::VideoFrame frame = this->source_->get();
        if (!frame) return;
        ++frame_counter_;
        const bool sampled = ((frame_counter_ - 1) % (uint64_t)std::max(1, sample_every_n_)) == 0;
        if (!sampled) {
            const Parameters output = emptyPayload(false, "not_sampled");
            av_dict_set(&frame.raw()->metadata, metadata_key_.c_str(), output.dump().c_str(), 0);
            this->sink_->put(frame);
            return;
        }
        if (frame.raw()->format != AV_PIX_FMT_CUDA) {
            throw Error("scoreboard_two_stage: expected CUDA NV12 frame");
        }
        const auto* frames_context = reinterpret_cast<const AVHWFramesContext*>(
            frame.raw()->hw_frames_ctx ? frame.raw()->hw_frames_ctx->data : nullptr);
        if (!frames_context || frames_context->sw_format != AV_PIX_FMT_NV12) {
            throw Error("scoreboard_two_stage: expected NV12 CUDA frame storage");
        }
        const auto start = std::chrono::steady_clock::now();
        ensureFrameContext(frame);
        const int width = frame.raw()->width, height = frame.raw()->height;
        launchLevel1(frame);
        Detection scoreboard;
        if (!decodeLevel1(width, height, scoreboard)) {
            if (post_process_) post_processor_.missLevel1();
            const Parameters output = emptyPayload(true, "scoreboard_not_detected");
            av_dict_set(&frame.raw()->metadata, metadata_key_.c_str(), output.dump().c_str(), 0);
            this->sink_->put(frame);
            return;
        }

        const Box raw_scoreboard_box = scoreboard.box;
        bool level1_relocated = false;
        if (post_process_) {
            const auto level1_update = post_processor_.updateLevel1(scoreboard.box);
            scoreboard.box = clampBox(level1_update.box, width, height);
            level1_relocated = level1_update.relocated;
        }
        // Padding already absorbs small Level-1 jitter. Keep the current raw box for
        // the Level-2 crop so temporal output smoothing cannot change model sampling.
        const auto level2_crop = scoreboard_post::PostProcessor::level2Crop(
            raw_scoreboard_box, width, height, vertical_padding_);
        int crop[4] = {
            level2_crop[0], level2_crop[1], level2_crop[2], level2_crop[3],
        };
        launchLevel2(frame, crop);
        std::vector<Detection> detections = decodeLevel2(crop, width, height);

        Parameters output;
        output["schema"] = "scoreboard_two_stage_v1";
        output["ocr_sampled"] = true;
        output["frame_width"] = width;
        output["frame_height"] = height;
        output["regions"] = Parameters::array({{
            {"id", 0},
            {"bbox", {scoreboard.box.x1, scoreboard.box.y1,
                      scoreboard.box.x2 - scoreboard.box.x1,
                      scoreboard.box.y2 - scoreboard.box.y1}},
        }});
        output["level1_detection"] = {
            {"bbox", {scoreboard.box.x1, scoreboard.box.y1, scoreboard.box.x2, scoreboard.box.y2}},
            {"det_conf", scoreboard.confidence},
            {"label", "scoreboard"},
        };
        if (post_process_) {
            output["post_process"] = {
                {"enabled", true},
                {"level1_relocated", level1_relocated},
                {"raw_level1_bbox", {raw_scoreboard_box.x1, raw_scoreboard_box.y1,
                                      raw_scoreboard_box.x2, raw_scoreboard_box.y2}},
                {"raw_level2_detections", Parameters::array()},
            };
        }
        output["level2_crop"] = {crop[0], crop[1], crop[2], crop[3]};

        const bool ambiguous = countClass(detections, 3) > 2 || countClass(detections, 4) > 1;
        output["detections"] = Parameters::array();
        output["draw_detections"] = Parameters::array();
        if (ambiguous) {
            if (post_process_) post_processor_.missComponents();
            output["reason"] = "ambiguous_multiple_games";
            for (const Detection& detection : detections) {
                if (post_process_) {
                    output["post_process"]["raw_level2_detections"].push_back(
                        detectionJson(detection));
                }
            }
        } else {
            keepSingleton(detections, 4);
            keepSingleton(detections, 5);
            keepSingleton(detections, 6);
            assignTeamLabels(detections);
            for (Detection& detection : detections) {
                if (detection.cls == 4) detection.label = "clock";
            }
            keepHighestPerLabel(detections);
            recognizeText(frame, detections, width, height);
            if (post_process_) {
                for (Detection& detection : detections) {
                    output["post_process"]["raw_level2_detections"].push_back(
                        detectionJson(detection));
                    detection.box = clampBox(
                        post_processor_.updateComponent(detection.label, detection.box),
                        width, height);
                }
            }
            std::sort(detections.begin(), detections.end(), [](const Detection& a, const Detection& b) {
                if (a.label != b.label) return a.label < b.label;
                return a.confidence > b.confidence;
            });
            for (const Detection& detection : detections) {
                output["detections"].push_back(detectionJson(detection));
                output["draw_detections"].push_back(detectionJson(detection));
            }
            output["reason"] = detections.empty() ? "components_not_detected" : "ok";
        }
        const double elapsed_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - start).count();
        output["stats"] = {
            {"level2_boxes", detections.size()},
            {"elapsed_ms", elapsed_ms},
            {"sample_every_n", sample_every_n_},
        };
        av_dict_set(&frame.raw()->metadata, metadata_key_.c_str(), output.dump().c_str(), 0);
        ++sample_counter_;
        if (debug_log_every_n_ > 0 && sample_counter_ % (uint64_t)debug_log_every_n_ == 0) {
            logstream << "scoreboard_two_stage_ocr frame=" << frame_counter_
                      << " components=" << detections.size()
                      << " ambiguous=" << (ambiguous ? 1 : 0)
                      << " elapsed_ms=" << elapsed_ms;
        }
        this->sink_->put(frame);
    }

    static std::shared_ptr<ScoreboardTwoStageOcr> create(NodeCreationInfo& nci) {
        const Parameters& params = nci.params;
        auto result = NodeSISO<av::VideoFrame, av::VideoFrame>::template createCommon<ScoreboardTwoStageOcr>(
            nci.edges, params);
        for (const char* required : {"level1_engine", "level2_engine", "recognizer_engine", "hwaccel"}) {
            if (!params.count(required)) throw Error(std::string("scoreboard_two_stage: ") + required + " param required");
        }
        result->node_label_ = params.value("name", std::string("<unnamed>"));
        result->level1_engine_ = params["level1_engine"].get<std::string>();
        result->level2_engine_ = params["level2_engine"].get<std::string>();
        result->recognizer_engine_ = params["recognizer_engine"].get<std::string>();
        result->hwaccel_ = InstanceSharedObjects<HWAccelDevice>::get(nci.instance, params["hwaccel"]);
        if (params.count("metadata_key")) result->metadata_key_ = params["metadata_key"].get<std::string>();
        if (params.count("target_fps")) result->target_fps_ = std::max(0.1f, params["target_fps"].get<float>());
        if (params.count("input_fps")) result->input_fps_ = parseFps(params["input_fps"].get<std::string>());
        result->sample_every_n_ = std::max(1, (int)std::lround(result->input_fps_ / result->target_fps_));
        if (params.count("level1_confidence")) result->level1_confidence_ = params["level1_confidence"].get<float>();
        if (params.count("level2_confidence")) result->level2_confidence_ = params["level2_confidence"].get<float>();
        if (params.count("reco_conf_thresh")) result->recognition_confidence_ = params["reco_conf_thresh"].get<float>();
        if (params.count("vertical_padding")) result->vertical_padding_ = std::max(0, params["vertical_padding"].get<int>());
        if (params.count("max_boxes")) result->max_boxes_ = std::max(1, params["max_boxes"].get<int>());
        if (params.count("edge_margin")) result->edge_margin_ = std::clamp(params["edge_margin"].get<float>(), 0.0f, 1.0f);
        if (params.count("edge_margin_px")) result->edge_margin_px_ = std::max(0, params["edge_margin_px"].get<int>());
        if (params.count("region_x") && params["region_x"].is_array() && params["region_x"].size() == 2) {
            result->region_x_ = {params["region_x"][0].get<float>(), params["region_x"][1].get<float>()};
            if (result->region_x_[0] < 0.0f || result->region_x_[1] > 1.0f ||
                result->region_x_[0] >= result->region_x_[1]) {
                throw Error("scoreboard_two_stage: region_x must be [start,end] within [0,1]");
            }
        }
        if (params.count("debug_log_every_n")) result->debug_log_every_n_ = std::max(0, params["debug_log_every_n"].get<int>());
        if (params.count("post_process")) result->post_process_ = params["post_process"].get<bool>();
        result->preinitialize();
        return result;
    }
};

DECLNODE(scoreboard_two_stage_ocr, ScoreboardTwoStageOcr)
