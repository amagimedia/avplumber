#include <algorithm>
#include <array>
#include <map>
#include "../common/infer_trt_base.hpp"
#include "decode_detection.hpp"

using namespace yolo_base;

class CudaInferRTDetr : public NodeSingleInput<av::VideoFrame>, public CudaInferTrtBase {
protected:
    std::unique_ptr<EdgeSink<av::VideoFrame>> sink_;

    std::string metadata_key_detection_ = "yolo_detections";
    std::string output_contract_ = "rtdetr_e2e_v1";
    bool debug_log_metadata_ = false;
    int debug_log_every_n_ = 30;
    int infer_every_n_ = 1;
    float conf_thresh_ = 0.25f;
    int max_det_ = 300;
    bool boxes_normalized_ = false;

    uint64_t frame_counter_ = 0;
    uint64_t infer_counter_ = 0;
    bool contract_validated_ = false;
    // e2e contract indices
    int boxes_output_index_ = -1;
    int scores_output_index_ = -1;
    int labels_output_index_ = -1;
    int det_count_ = 0;
    // combined contract
    int combined_output_index_ = -1;
    int combined_num_classes_ = 0;
    std::map<int, uint64_t> detection_count_histogram_;
    std::array<uint64_t, 10> conf_histogram_{};
    RTDetrDetectionDecoder decoder_;

public:
    CudaInferRTDetr(std::unique_ptr<Source<av::VideoFrame>> source,
                    std::unique_ptr<EdgeSink<av::VideoFrame>> sink)
        : NodeSingleInput(std::move(source))
        , sink_(std::move(sink)) {}

    ~CudaInferRTDetr() {
        if (detection_count_histogram_.empty()) return;
        uint64_t total = 0;
        uint64_t with_dets = 0;
        for (auto& [count, frames] : detection_count_histogram_) {
            total += frames;
            if (count > 0) with_dets += frames;
        }
        logstream << "cuda_infer_rtdetr: detection summary:"
                  << " detected frames: " << with_dets
                  << " / total frames: " << total
                  << " (" << (100.0 * with_dets / std::max<uint64_t>(1, total)) << "%)";
    }

    void logOutputTensors(const ModelRunner& model) {
        for (size_t i = 0; i < model.outputs.size(); ++i) {
            const OutputTensor& ot = model.outputs[i];
            logstream << "cuda_infer_rtdetr: output[" << i << "] name=" << ot.name
                      << " dtype=" << (int)ot.dtype
                      << " dims=" << ot.dims.nbDims
                      << ":" << (ot.dims.nbDims > 0 ? ot.dims.d[0] : -1)
                      << "," << (ot.dims.nbDims > 1 ? ot.dims.d[1] : -1)
                      << "," << (ot.dims.nbDims > 2 ? ot.dims.d[2] : -1);
        }
    }

    bool validateE2eV1(ModelRunner& model) {
        if (boxes_normalized_) {
            logstream << "cuda_infer_rtdetr: boxes_normalized=true is unsupported in e2e_v1";
            return false;
        }

        int bi = -1, si = -1, li = -1;
        int box_count = -1;
        for (size_t i = 0; i < model.outputs.size(); ++i) {
            const OutputTensor& ot = model.outputs[i];
            if (ot.dims.nbDims == 3 && ot.dims.d[0] == 1 && ot.dims.d[2] == 4) {
                bi = (int)i;
                box_count = ot.dims.d[1];
            } else if (ot.dtype == nvinfer1::DataType::kINT64 || ot.dtype == nvinfer1::DataType::kINT32) {
                // Integer tensor is labels
                li = (int)i;
            } else if (ot.dims.nbDims == 2 && ot.dims.d[0] == 1) {
                // Float 2D tensor is scores
                si = (int)i;
            } else if (ot.dims.nbDims == 1) {
                si = (int)i;
            }
        }

        if (bi < 0 || si < 0 || li < 0) {
            logstream << "cuda_infer_rtdetr: failed to match required outputs for contract rtdetr_e2e_v1"
                      << " (need boxes[1,N,4], scores[1,N]/[N], labels[1,N]/[N])";
            return false;
        }

        const OutputTensor& scores_ot = model.outputs[(size_t)si];
        const OutputTensor& labels_ot = model.outputs[(size_t)li];
        int scores_count = (scores_ot.dims.nbDims == 2) ? scores_ot.dims.d[1] : scores_ot.dims.d[0];
        int labels_count = (labels_ot.dims.nbDims == 2) ? labels_ot.dims.d[1] : labels_ot.dims.d[0];
        if (box_count <= 0 || scores_count != box_count || labels_count != box_count) {
            logstream << "cuda_infer_rtdetr: output count mismatch boxes=" << box_count
                      << " scores=" << scores_count << " labels=" << labels_count;
            return false;
        }

        boxes_output_index_ = bi;
        scores_output_index_ = si;
        labels_output_index_ = li;
        det_count_ = box_count;
        return true;
    }

    bool validateCombinedV1(ModelRunner& model) {
        // Expect single output [1, N, 4+num_classes] or [N, 4+num_classes]
        for (size_t i = 0; i < model.outputs.size(); ++i) {
            const OutputTensor& ot = model.outputs[i];
            int n = -1, cols = -1;
            if (ot.dims.nbDims == 3 && ot.dims.d[0] == 1 && ot.dims.d[2] > 4) {
                n = ot.dims.d[1];
                cols = ot.dims.d[2];
            } else if (ot.dims.nbDims == 2 && ot.dims.d[1] > 4) {
                n = ot.dims.d[0];
                cols = ot.dims.d[1];
            }
            if (n > 0 && cols > 4) {
                combined_output_index_ = (int)i;
                det_count_ = n;
                combined_num_classes_ = cols - 4;
                logstream << "cuda_infer_rtdetr: combined_v1 matched output[" << i
                          << "] det_count=" << det_count_
                          << " num_classes=" << combined_num_classes_;
                return true;
            }
        }
        logstream << "cuda_infer_rtdetr: failed to match output for contract rtdetr_combined_v1"
                  << " (need [1,N,4+C] or [N,4+C])";
        return false;
    }

    bool validateOutputContract() {
        if (models_.size() != 1) {
            logstream << "cuda_infer_rtdetr: requires exactly one model";
            return false;
        }

        ModelRunner& model = models_[0];
        if (model.input_dims.nbDims != 4 || model.input_dims.d[0] != 1) {
            logstream << "cuda_infer_rtdetr: requires NCHW batch=1 input";
            return false;
        }

        bool ok = false;
        if (output_contract_ == "rtdetr_e2e_v1") {
            ok = validateE2eV1(model);
        } else if (output_contract_ == "rtdetr_combined_v1") {
            ok = validateCombinedV1(model);
        } else {
            logstream << "cuda_infer_rtdetr: unsupported output_contract: " << output_contract_;
            return false;
        }

        if (ok) {
            contract_validated_ = true;
            if (debug_log_metadata_) {
                logstream << "cuda_infer_rtdetr: output_contract=" << output_contract_
                          << " det_count=" << det_count_;
                logOutputTensors(model);
            }
        }
        return ok;
    }

    std::string buildDetectionMetadata(const std::vector<Detection>& dets) const {
        Parameters j;
        j["coord_space"] = "model";
        j["model_width"] = input_w_;
        j["model_height"] = input_h_;
        j["models"] = Parameters::array();

        for (size_t i = 0; i < models_.size(); ++i) {
            Parameters model_item;
            model_item["model_index"] = (int)i;
            model_item["engine"] = models_[i].engine_path;
            model_item["engine_name"] = models_[i].engine_name;
            j["models"].push_back(model_item);
        }

        j["detections"] = Parameters::array();
        for (const Detection& d : dets) {
            Parameters item;
            item["cls"] = d.cls;
            item["conf"] = d.conf;
            item["xyxy"] = {d.x1, d.y1, d.x2, d.y2};
            item["model_index"] = d.model_index;
            if (d.model_index >= 0 && (size_t)d.model_index < models_.size()) {
                item["engine_name"] = models_[(size_t)d.model_index].engine_name;
                const std::vector<std::string>& class_names = models_[(size_t)d.model_index].class_names;
                if (d.cls >= 0 && (size_t)d.cls < class_names.size()) {
                    item["label"] = class_names[(size_t)d.cls];
                }
            }
            j["detections"].push_back(item);
        }
        return j.dump();
    }

    void process() override {
        av::VideoFrame frm = this->source_->get();
        if (!frm) return;

        ++frame_counter_;
        if (infer_every_n_ > 1 && (frame_counter_ % (uint64_t)infer_every_n_) != 0) {
            sink_->put(frm);
            return;
        }

        if (frm.raw()->format != AV_PIX_FMT_CUDA) {
            logstream << "cuda_infer_rtdetr: non-CUDA frame, passing through";
            sink_->put(frm);
            return;
        }
        if (!ensureInitialized(frm)) return;
        if (!contract_validated_ && !validateOutputContract()) return;

        if (frm.width() != input_w_ || frm.height() != input_h_) {
            logstream << "cuda_infer_rtdetr: input frame size mismatch, expected " << input_w_ << "x" << input_h_
                      << " got " << frm.width() << "x" << frm.height();
            return;
        }

        if (hwSwFormat(frm) != AV_PIX_FMT_NV12) {
            logstream << "cuda_infer_rtdetr: unsupported hw sw_format (expected NV12)";
            return;
        }

        if (CUDA_CHECK_CU(cuCtxSetCurrent(cu_ctx_))) {
            logstream << "cuda_infer_rtdetr: cuCtxSetCurrent failed in process";
            return;
        }

        ++infer_counter_;
        ModelRunner& model = models_[0];
        if (!runPreprocessNV12(frm, model)) return;
        if (!runInference(model)) return;
        if (!syncModel(model)) return;

        std::vector<Detection> all_dets;
        DecodeParams dp{0, conf_thresh_, OutputBoxFormat::EndToEndXYXY, model.class_index_remap};

        if (output_contract_ == "rtdetr_combined_v1" && combined_output_index_ >= 0) {
            const OutputTensor& ot = model.outputs[(size_t)combined_output_index_];
            DetectionResult dr = decoder_.decodeCombined(
                ot.host_output.data(), det_count_, combined_num_classes_,
                (float)input_w_, (float)input_h_, dp);
            all_dets = std::move(dr.detections);
        } else if (boxes_output_index_ >= 0 && scores_output_index_ >= 0 && labels_output_index_ >= 0) {
            const OutputTensor& boxes_ot = model.outputs[(size_t)boxes_output_index_];
            const OutputTensor& scores_ot = model.outputs[(size_t)scores_output_index_];
            const OutputTensor& labels_ot = model.outputs[(size_t)labels_output_index_];
            DetectionResult dr = decoder_.decode(
                boxes_ot.host_output.data(),
                scores_ot.host_output.data(),
                labels_ot.host_output.data(),
                det_count_, dp);
            all_dets = std::move(dr.detections);
        } else {
            logstream << "cuda_infer_rtdetr: decode failed; writing empty detections";
        }

        std::sort(all_dets.begin(), all_dets.end(),
            [](const Detection& a, const Detection& b) { return a.conf > b.conf; });
        if (max_det_ > 0 && (int)all_dets.size() > max_det_) {
            all_dets.resize((size_t)max_det_);
        }

        ++detection_count_histogram_[(int)all_dets.size()];
        for (const Detection& d : all_dets) {
            int bucket = std::min((int)(d.conf * 10.0f), 9);
            ++conf_histogram_[(size_t)bucket];
        }

        std::string md = buildDetectionMetadata(all_dets);
        av_dict_set(&frm.raw()->metadata, metadata_key_detection_.c_str(), md.c_str(), 0);

        if (debug_log_metadata_ && debug_log_every_n_ > 0 &&
            (frame_counter_ % (uint64_t)debug_log_every_n_) == 0) {
            logstream << "cuda_infer_rtdetr: " << md;
        }
        sink_->put(frm);
    }

    static std::shared_ptr<CudaInferRTDetr> create(NodeCreationInfo &nci) {
        EdgeManager &edges = nci.edges;
        const Parameters &params = nci.params;

        if (!params.count("models") || !params["models"].is_array() || params["models"].empty()) {
            throw Error("cuda_infer_rtdetr: missing or empty required parameter: models");
        }
        if (params["models"].size() != 1) {
            throw Error("cuda_infer_rtdetr: v1 requires models array of size 1");
        }

        std::shared_ptr<Edge<av::VideoFrame>> src = edges.find<av::VideoFrame>(params["src"]);
        std::shared_ptr<Edge<av::VideoFrame>> dst = edges.find<av::VideoFrame>(params["dst"]);

        auto r = std::make_shared<CudaInferRTDetr>(src->makeSource(), dst->makeSink());

        if (params.count("conf_thresh")) r->conf_thresh_ = params["conf_thresh"];
        if (params.count("max_det")) r->max_det_ = params["max_det"];
        if (params.count("infer_every_n")) r->infer_every_n_ = params["infer_every_n"];
        if (params.count("metadata_key_detection")) r->metadata_key_detection_ = params["metadata_key_detection"].get<std::string>();
        if (params.count("debug_log_metadata")) r->debug_log_metadata_ = params["debug_log_metadata"];
        if (params.count("debug_log_every_n")) r->debug_log_every_n_ = params["debug_log_every_n"];
        if (params.count("input_format")) {
            std::string ifmt = params["input_format"].get<std::string>();
            r->input_bgr_order_ = (ifmt == "BGR" || ifmt == "bgr");
        }

        const auto& mp = params["models"][0];
        if (!mp.is_object()) {
            throw Error("cuda_infer_rtdetr: models[0] must be an object");
        }
        if (!mp.count("engine") || !mp["engine"].is_string()) {
            throw Error("cuda_infer_rtdetr: model requires 'engine' string");
        }
        if (!mp.count("output_contract") || !mp["output_contract"].is_string()) {
            throw Error("cuda_infer_rtdetr: model requires 'output_contract' string");
        }

        r->output_contract_ = mp["output_contract"].get<std::string>();
        if (mp.count("boxes_normalized")) {
            r->boxes_normalized_ = mp["boxes_normalized"].get<bool>();
        }

        ModelRunner model;
        model.engine_path = mp["engine"].get<std::string>();
        model.engine_name = std::filesystem::path(model.engine_path).filename().string();

        if (mp.count("class_names")) {
            if (!mp["class_names"].is_array()) {
                throw Error("cuda_infer_rtdetr: class_names must be string array");
            }
            for (const auto& name : mp["class_names"]) {
                if (!name.is_string()) {
                    throw Error("cuda_infer_rtdetr: class_names must be string array");
                }
                model.class_names.push_back(name.get<std::string>());
            }
        }
        if (mp.count("class_index_remap")) {
            if (!mp["class_index_remap"].is_array()) {
                throw Error("cuda_infer_rtdetr: class_index_remap must be int array");
            }
            for (const auto& cls_item : mp["class_index_remap"]) {
                if (!cls_item.is_number_integer()) {
                    throw Error("cuda_infer_rtdetr: class_index_remap must be int array");
                }
                model.class_index_remap.push_back(cls_item.get<int>());
            }
        }

        r->models_.push_back(std::move(model));
        return r;
    }
};

DECLNODE(cuda_infer_rtdetr, CudaInferRTDetr)
