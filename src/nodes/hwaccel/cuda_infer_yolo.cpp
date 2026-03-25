#include "cuda_infer_yolo_base.hpp"
#include "yolo_decode_detection.hpp"
#include "yolo_decode_segmentation.hpp"

using namespace yolo_base;

class CudaInferYolo : public NodeSingleInput<av::VideoFrame>, public CudaInferYoloBase {
protected:
    std::unique_ptr<EdgeSink<av::VideoFrame>> sink_;
    std::unique_ptr<EdgeSink<av::VideoFrame>> sink_seg_;  // optional, for segmentation mask output

    std::string metadata_key_detection_ = "yolo_detections";
    std::string metadata_key_segmentation_ = "yolo_segmentation";
    bool debug_log_metadata_ = false;
    int debug_log_every_n_ = 30;
    int infer_every_n_ = 1;
    float conf_thresh_ = 0.25f;
    int max_det_ = 300;
    int mask_gpu_every_n_ = 1;
    int mask_cpu_every_n_ = 2;
    int mask_cpu_resolution_ = 120;
    uint64_t frame_counter_ = 0;
    uint64_t infer_counter_ = 0;
    bool seg_decoders_initialized_ = false;

public:
    CudaInferYolo(std::unique_ptr<Source<av::VideoFrame>> source,
                  std::unique_ptr<EdgeSink<av::VideoFrame>> sink,
                  std::unique_ptr<EdgeSink<av::VideoFrame>> sink_seg)
        : NodeSingleInput(std::move(source))
        , sink_(std::move(sink))
        , sink_seg_(std::move(sink_seg)) {}

    void process() override {
        av::VideoFrame frm = this->source_->get();
        if (!frm) return;

        ++frame_counter_;
        if (infer_every_n_ > 1 && (frame_counter_ % (uint64_t)infer_every_n_) != 0) {
            sink_->put(frm);
            return;
        }

        if (frm.raw()->format != AV_PIX_FMT_CUDA) {
            logstream << "cuda_infer_yolo: non-CUDA frame, passing through";
            sink_->put(frm);
            return;
        }
        if (!ensureInitialized(frm)) {
            return;
        }
        // Initialize segmentation decoders after engine loading (needs output tensor dims)
        if (!seg_decoders_initialized_) {
            seg_decoders_initialized_ = true;
            for (ModelRunner& model : models_) {
                if (model.task_type == TaskType::Segmentation && model.seg_decoder) {
                    if (!model.seg_decoder->init(model, cu_ctx_, max_det_)) {
                        logstream << "cuda_infer_yolo: failed to init seg decoder for " << model.engine_name;
                        return;
                    }
                }
            }
        }
        if (frm.width() != input_w_ || frm.height() != input_h_) {
            logstream << "cuda_infer_yolo: input frame size mismatch, expected " << input_w_ << "x" << input_h_
                      << " got " << frm.width() << "x" << frm.height();
            return;
        }

        AVPixelFormat swfmt = hwSwFormat(frm);
        if (swfmt != AV_PIX_FMT_NV12) {
            logstream << "cuda_infer_yolo: unsupported hw sw_format (expected NV12)";
            return;
        }

        if (YOLO_CHECK_CU(cuCtxSetCurrent(cu_ctx_))) {
            logstream << "cuda_infer_yolo: cuCtxSetCurrent failed in process";
            return;
        }

        ++infer_counter_;

        // Run inference on all models
        for (ModelRunner& model : models_) {
            if (!runPreprocessNV12(frm, model)) return;
            if (!runInference(model)) return;
        }

        // Collect detections from all models
        std::vector<Detection> all_dets;
        for (size_t mi = 0; mi < models_.size(); ++mi) {
            ModelRunner& model = models_[mi];
            if (!syncModel(model)) return;

            // Build host_outputs and output_dims from model.outputs
            std::vector<const float*> host_outputs;
            std::vector<nvinfer1::Dims> output_dims;
            for (const OutputTensor& ot : model.outputs) {
                host_outputs.push_back(ot.host_output.data());
                output_dims.push_back(ot.dims);
            }

            DecodeParams dp{(int)mi, conf_thresh_, model.output_box_format, model.class_index_remap};

            if (model.task_type == TaskType::Detection && model.det_decoder) {
                DetectionResult dr = model.det_decoder->decode(host_outputs, output_dims, dp);
                all_dets.insert(all_dets.end(), dr.detections.begin(), dr.detections.end());
            } else if (model.task_type == TaskType::Segmentation && model.seg_decoder) {
                bool emit_gpu = (mask_gpu_every_n_ > 0) && (infer_counter_ % (uint64_t)mask_gpu_every_n_ == 0);
                bool emit_cpu = (mask_cpu_every_n_ > 0) && (infer_counter_ % (uint64_t)mask_cpu_every_n_ == 0);
                SegmentationResult sr = model.seg_decoder->decode(
                    host_outputs, output_dims, dp,
                    emit_gpu, emit_cpu, mask_cpu_resolution_, model.stream);

                // Add seg detections to all_dets
                all_dets.insert(all_dets.end(), sr.detections.begin(), sr.detections.end());

                // CPU path: attach side data to frame
                if (emit_cpu && !sr.cpu_masks.empty()) {
                    size_t header_size = 16;
                    size_t mask_data_size = sr.cpu_masks.size() * sizeof(float);
                    size_t total_size = header_size + mask_data_size;
                    AVBufferRef* buf = av_buffer_alloc((size_t)total_size);
                    if (buf) {
                        uint32_t* header = (uint32_t*)buf->data;
                        header[0] = (uint32_t)sr.num_masks;
                        header[1] = (uint32_t)sr.cpu_mask_w;
                        header[2] = (uint32_t)sr.cpu_mask_h;
                        header[3] = 0; // reserved
                        memcpy(buf->data + header_size, sr.cpu_masks.data(), mask_data_size);
                        av_frame_new_side_data_from_buf(frm.raw(), AV_FRAME_DATA_YOLO_SEG_MASKS, buf);
                    }
                }

                // GPU path: output mask AVFrame on dst_seg edge
                if (emit_gpu && sr.gpu_mask_buf && sink_seg_) {
                    av::VideoFrame mask_frame;
                    mask_frame.raw()->pts = frm.raw()->pts;
                    mask_frame.raw()->width = sr.mask_proto_w;
                    mask_frame.raw()->height = sr.mask_proto_h;
                    // Attach the GPU mask buffer as the frame's data[0] via buf[0]
                    mask_frame.raw()->buf[0] = sr.gpu_mask_buf;
                    mask_frame.raw()->data[0] = sr.gpu_mask_buf->data;
                    mask_frame.raw()->linesize[0] = sr.mask_proto_w * (int)sizeof(float);
                    sr.gpu_mask_buf = nullptr; // ownership transferred to frame
                    sink_seg_->put(mask_frame);
                } else if (sr.gpu_mask_buf) {
                    av_buffer_unref(&sr.gpu_mask_buf);
                }
            }
        }

        // Finalize: sort by confidence, truncate to max_det
        std::sort(all_dets.begin(), all_dets.end(),
            [](const Detection& a, const Detection& b) { return a.conf > b.conf; });
        if (max_det_ > 0 && (int)all_dets.size() > max_det_) {
            all_dets.resize((size_t)max_det_);
        }

        // Build and attach detection metadata
        std::string md = buildDetectionMetadata(all_dets);
        av_dict_set(&frm.raw()->metadata, metadata_key_detection_.c_str(), md.c_str(), 0);

        if (debug_log_metadata_ && debug_log_every_n_ > 0 &&
            (frame_counter_ % (uint64_t)debug_log_every_n_) == 0) {
            logstream << "cuda_infer_yolo: " << md;
        }

        sink_->put(frm);
    }

    std::string buildDetectionMetadata(const std::vector<Detection>& dets) const {
        Parameters j;
        j["coord_space"] = "model";
        j["model_width"] = input_w_;
        j["model_height"] = input_h_;

        // Build models array (could cache but models_ is small)
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

    static std::shared_ptr<CudaInferYolo> create(NodeCreationInfo &nci) {
        EdgeManager &edges = nci.edges;
        const Parameters &params = nci.params;

        if (!params.count("models") || !params["models"].is_array() || params["models"].empty()) {
            throw Error("cuda_infer_yolo: missing or empty required parameter: models (array of model objects)");
        }

        std::shared_ptr<Edge<av::VideoFrame>> src = edges.find<av::VideoFrame>(params["src"]);
        std::shared_ptr<Edge<av::VideoFrame>> dst = edges.find<av::VideoFrame>(params["dst"]);

        std::unique_ptr<EdgeSink<av::VideoFrame>> sink_seg;
        if (params.count("dst_seg")) {
            std::shared_ptr<Edge<av::VideoFrame>> dst_seg = edges.find<av::VideoFrame>(params["dst_seg"]);
            sink_seg = dst_seg->makeSink();
        }

        auto r = std::make_shared<CudaInferYolo>(
            src->makeSource(), dst->makeSink(), std::move(sink_seg));

        // Parse global params
        if (params.count("conf_thresh")) r->conf_thresh_ = params["conf_thresh"];
        if (params.count("max_det")) r->max_det_ = params["max_det"];
        if (params.count("infer_every_n")) r->infer_every_n_ = params["infer_every_n"];
        if (params.count("metadata_key_detection")) r->metadata_key_detection_ = params["metadata_key_detection"].get<std::string>();
        if (params.count("metadata_key_segmentation")) r->metadata_key_segmentation_ = params["metadata_key_segmentation"].get<std::string>();
        if (params.count("debug_log_metadata")) r->debug_log_metadata_ = params["debug_log_metadata"];
        if (params.count("debug_log_every_n")) r->debug_log_every_n_ = params["debug_log_every_n"];
        if (params.count("mask_gpu_every_n")) r->mask_gpu_every_n_ = params["mask_gpu_every_n"];
        if (params.count("mask_cpu_every_n")) r->mask_cpu_every_n_ = params["mask_cpu_every_n"];
        if (params.count("mask_cpu_resolution")) r->mask_cpu_resolution_ = params["mask_cpu_resolution"];
        if (params.count("input_format")) {
            std::string ifmt = params["input_format"].get<std::string>();
            r->input_bgr_order_ = (ifmt == "BGR" || ifmt == "bgr");
        }

        // Parse models
        auto parseFmt = [](const std::string& fmt) -> OutputBoxFormat {
            if (fmt == "end2end_xyxy") return OutputBoxFormat::EndToEndXYXY;
            if (fmt == "raw_cxcywh") return OutputBoxFormat::RawCXCYWH;
            throw Error("cuda_infer_yolo: output_box_format must be 'end2end_xyxy' or 'raw_cxcywh', got: " + fmt);
        };
        auto parseTaskType = [](const std::string& tt) -> TaskType {
            if (tt == "detection") return TaskType::Detection;
            if (tt == "segmentation") return TaskType::Segmentation;
            if (tt == "pose") return TaskType::Pose;
            throw Error("cuda_infer_yolo: task_type must be 'detection', 'segmentation', or 'pose', got: " + tt);
        };

        for (const auto& mp : params["models"]) {
            if (!mp.is_object()) {
                throw Error("cuda_infer_yolo: each item in models must be an object");
            }
            if (!mp.count("engine") || !mp["engine"].is_string()) {
                throw Error("cuda_infer_yolo: each model must have an 'engine' string");
            }

            ModelRunner model;
            model.engine_path = mp["engine"].get<std::string>();
            model.engine_name = std::filesystem::path(model.engine_path).filename().string();

            if (mp.count("output_box_format")) {
                model.output_box_format = parseFmt(mp["output_box_format"].get<std::string>());
            }

            // task_type defaults to Detection if not specified (backward compat)
            if (mp.count("task_type")) {
                model.task_type = parseTaskType(mp["task_type"].get<std::string>());
            }

            if (mp.count("class_names")) {
                if (!mp["class_names"].is_array()) {
                    throw Error("cuda_infer_yolo: class_names must be a string array");
                }
                for (const auto& name : mp["class_names"]) {
                    if (!name.is_string()) {
                        throw Error("cuda_infer_yolo: class_names must be a string array");
                    }
                    model.class_names.push_back(name.get<std::string>());
                }
            }
            if (mp.count("class_index_remap")) {
                if (!mp["class_index_remap"].is_array()) {
                    throw Error("cuda_infer_yolo: class_index_remap must be an int array");
                }
                for (const auto& cls_item : mp["class_index_remap"]) {
                    if (!cls_item.is_number_integer()) {
                        throw Error("cuda_infer_yolo: class_index_remap must be an int array");
                    }
                    model.class_index_remap.push_back(cls_item.get<int>());
                }
            }

            // Create decoder
            if (model.task_type == TaskType::Detection) {
                model.det_decoder = std::make_unique<DetectionDecoder>();
            } else if (model.task_type == TaskType::Segmentation) {
                model.seg_decoder = std::make_unique<SegmentationDecoder>();
            }

            r->models_.push_back(std::move(model));
        }

        return r;
    }
};

DECLNODE(cuda_infer_yolo, CudaInferYolo)
