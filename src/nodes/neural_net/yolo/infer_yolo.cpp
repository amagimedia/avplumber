#include <array>
#include <chrono>
#include <map>
#include <sstream>
#include "../common/infer_trt_base.hpp"
#include "decode_detection.hpp"
#include "decode_segmentation.hpp"
#include "decode_pose.hpp"

using namespace yolo_base;

namespace {

void appendSegPointerDebug(std::ostringstream& oss, CUcontext expected_ctx, CUdeviceptr ptr) {
    if (!ptr) {
        oss << " gpu_ptr=0";
        return;
    }

    CUcontext current_ctx = nullptr;
    if (cuCtxGetCurrent) {
        CUresult current_res = cuCtxGetCurrent(&current_ctx);
        if (current_res == CUDA_SUCCESS) {
            oss << " current_ctx=" << current_ctx;
        } else {
            oss << " current_ctx=<error:" << (int)current_res << ">";
        }
    }
    oss << " expected_ctx=" << expected_ctx
        << " gpu_ptr=" << (const void*)(uintptr_t)ptr;

    CUdeviceptr base_ptr = 0;
    size_t base_size = 0;
    if (cuMemGetAddressRange) {
        CUresult range_res = cuMemGetAddressRange(&base_ptr, &base_size, ptr);
        if (range_res == CUDA_SUCCESS) {
            oss << " range_base=" << (const void*)(uintptr_t)base_ptr
                << " range_bytes=" << base_size;
        } else {
            oss << " range_base=<error:" << (int)range_res << ">";
        }
    } else {
        oss << " range_base=<unavailable>";
    }
}

} // namespace

class CudaInferYolo : public NodeSingleInput<av::VideoFrame>, public CudaInferTrtBase, public ReportsFinishByFlag {
protected:
    std::unique_ptr<EdgeSink<av::VideoFrame>> sink_;
    std::unique_ptr<EdgeSink<av::VideoFrame>> sink_seg_;  // optional, for segmentation mask output

    std::string metadata_key_detection_ = "yolo_detections";
    std::string metadata_key_segmentation_ = "yolo_segmentation";
    std::string metadata_key_pose_ = "yolo_pose";
    bool debug_log_metadata_ = false;
    int debug_log_every_n_ = 30;
    int infer_every_n_ = 1;
    float conf_thresh_ = 0.25f;
    int max_det_ = 300;
    int mask_gpu_every_n_ = 1;
    int mask_cpu_every_n_ = 2;
    int mask_cpu_resolution_ = 120;
    int side_data_slot_ = 0;
    uint64_t frame_counter_ = 0;
    uint64_t infer_counter_ = 0;
    std::map<int, uint64_t> detection_count_histogram_;
    std::array<uint64_t, 10> conf_histogram_{}; // buckets: [0.0,0.1), [0.1,0.2), ... [0.9,1.0]
    bool seg_decoders_initialized_ = false;
    std::string node_label_ = "<unnamed>";

public:
    CudaInferYolo(std::unique_ptr<Source<av::VideoFrame>> source,
                  std::unique_ptr<EdgeSink<av::VideoFrame>> sink,
                  std::unique_ptr<EdgeSink<av::VideoFrame>> sink_seg)
        : NodeSingleInput(std::move(source))
        , sink_(std::move(sink))
        , sink_seg_(std::move(sink_seg)) {}

    ~CudaInferYolo() {
        if (detection_count_histogram_.empty()) return;
        uint64_t total = 0;
        uint64_t with_dets = 0;
        for (auto& [count, frames] : detection_count_histogram_) {
            total += frames;
            if (count > 0) with_dets += frames;
        }
        logstream << "cuda_infer_yolo: detection summary:"
                  << " detected frames: " << with_dets
                  << " / total frames: " << total
                  << " (" << (100.0 * with_dets / total) << "%)";
        logstream << "cuda_infer_yolo: detection count histogram:";
        for (auto& [count, frames] : detection_count_histogram_) {
            logstream << "  " << count << " detections: " << frames << " frames";
        }
        logstream << "cuda_infer_yolo: confidence histogram:";
        for (int i = 0; i < 10; ++i) {
            logstream << "  " << (i * 0.1) << "-" << ((i + 1) * 0.1) << ": " << conf_histogram_[(size_t)i] << " detections";
        }
    }

    bool consumeEofIfPresent() override {
        return false;
    }

    void process() override {
        av::VideoFrame frm = this->source_->get();
        if (isEofMarker(frm)) {
            sink_->put(frm);
            if (sink_seg_) {
                sink_seg_->put(createEofMarker<av::VideoFrame>());
            }
            this->finished_ = true;
            return;
        }
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
        if (!initialized_) {
            auto init_start = std::chrono::steady_clock::now();
            bool ok = ensureInitialized(frm);
            auto init_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - init_start).count();
            logstream << "neural_node_init"
                      << " status=" << (ok ? "ok" : "error")
                      << " type=cuda_infer_yolo"
                      << " node=" << node_label_
                      << " init_ms=" << init_ms
                      << " frame_wait_ms=" << init_ms
                      << " frame=" << frame_counter_
                      << " models=" << models_.size();
            if (!ok) {
                return;
            }
        } else if (!ensureInitialized(frm)) {
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

        if (CUDA_CHECK_CU(cuCtxSetCurrent(cu_ctx_))) {
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

            DecodeParams dp{
                (int)mi,
                conf_thresh_,
                model.output_box_format,
                model.class_index_remap,
                model.nms_iou_thresh,
                model.nms_class_agnostic
            };

            if (model.task_type == TaskType::Detection && model.det_decoder) {
                DetectionResult dr = model.det_decoder->decode(host_outputs, output_dims, dp);
                if (model.include_in_detection_metadata) {
                    all_dets.insert(all_dets.end(), dr.detections.begin(), dr.detections.end());
                }
            } else if (model.task_type == TaskType::Segmentation && model.seg_decoder) {
                bool emit_gpu = (mask_gpu_every_n_ > 0) && (infer_counter_ % (uint64_t)mask_gpu_every_n_ == 0);
                bool emit_cpu = (mask_cpu_every_n_ > 0) && (infer_counter_ % (uint64_t)mask_cpu_every_n_ == 0);
                SegmentationResult sr = model.seg_decoder->decode(
                    host_outputs, output_dims, dp,
                    emit_gpu, emit_cpu, mask_cpu_resolution_, model.stream);

                if (model.include_in_detection_metadata) {
                    all_dets.insert(all_dets.end(), sr.detections.begin(), sr.detections.end());
                }

                // Write segmentation detections to separate metadata key (preserving mask order)
                // Truncate to match the number of assembled masks
                {
                    std::vector<Detection> seg_dets = sr.detections;
                    if (sr.num_masks > 0 && (int)seg_dets.size() > sr.num_masks) {
                        seg_dets.resize((size_t)sr.num_masks);
                    }
                    std::string seg_md = buildDetectionMetadata(seg_dets);
                    av_dict_set(&frm.raw()->metadata, metadata_key_segmentation_.c_str(), seg_md.c_str(), 0);
                }

                if (debug_log_metadata_ && debug_log_every_n_ > 0 &&
                    (frame_counter_ % (uint64_t)debug_log_every_n_) == 0) {
                    logstream << "cuda_infer_yolo: seg model=" << model.engine_name
                              << " detections=" << sr.detections.size()
                              << " masks=" << sr.num_masks
                              << " proto=" << sr.mask_proto_w << "x" << sr.mask_proto_h
                              << " gpu=" << (emit_gpu && sr.gpu_mask_buf ? "yes" : "no")
                              << " cpu=" << (emit_cpu && !sr.cpu_masks.empty() ? "yes" : "no");
                }

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
                        av_frame_new_side_data_from_buf(frm.raw(), yoloSegCpuSideDataType(side_data_slot_), buf);
                    }
                }

                // GPU path: output mask AVFrame on dst_seg edge
                if (emit_gpu && sr.gpu_mask_buf && sink_seg_) {
                    av::VideoFrame mask_frame;
                    mask_frame.raw()->pts = frm.raw()->pts;
                    mask_frame.raw()->width = sr.mask_proto_w;
                    mask_frame.raw()->height = sr.mask_proto_h;
                    mask_frame.raw()->buf[0] = av_buffer_ref(sr.gpu_mask_buf);
                    mask_frame.raw()->data[0] = sr.gpu_mask_buf->data;
                    mask_frame.raw()->linesize[0] = sr.mask_proto_w * (int)sizeof(float);
                    sink_seg_->put(mask_frame);
                }

                // GPU path: also attach as side data on passthrough frame
                if (emit_gpu && sr.gpu_mask_buf) {
                    auto* header = (GpuMaskSideDataHeader*)av_malloc(sizeof(GpuMaskSideDataHeader));
                    if (header) {
                        header->gpu_ptr = (uint64_t)(uintptr_t)sr.gpu_mask_buf->data;
                        header->num_masks = (uint32_t)sr.num_masks;
                        header->proto_w = (uint32_t)sr.mask_proto_w;
                        header->proto_h = (uint32_t)sr.mask_proto_h;
                        header->model_w = (uint32_t)input_w_;
                        header->model_h = (uint32_t)input_h_;
                        const CUdeviceptr gpu_masks = (CUdeviceptr)(uintptr_t)header->gpu_ptr;
                        if (!gpu_masks) {
                            throw Error("cuda_infer_yolo: segmentation GPU mask buffer is null before side-data attach");
                        }
                        if (header->num_masks == 0 || header->proto_w == 0 || header->proto_h == 0) {
                            std::ostringstream err;
                            err << "cuda_infer_yolo: invalid segmentation GPU side data header"
                                << " slot=" << side_data_slot_
                                << " num_masks=" << header->num_masks
                                << " proto=" << header->proto_w << "x" << header->proto_h
                                << " model=" << header->model_w << "x" << header->model_h;
                            throw Error(err.str());
                        }
                        if (debug_log_metadata_ && debug_log_every_n_ > 0 &&
                            (frame_counter_ % (uint64_t)debug_log_every_n_) == 0) {
                            std::ostringstream dbg;
                            dbg << "cuda_infer_yolo: seg attach"
                                << " frame=" << frame_counter_
                                << " model=" << model.engine_name
                                << " slot=" << side_data_slot_
                                << " sd_type=" << (int)yoloSegGpuSideDataType(side_data_slot_)
                                << " num_masks=" << header->num_masks
                                << " proto=" << header->proto_w << "x" << header->proto_h
                                << " model_dims=" << header->model_w << "x" << header->model_h;
                            appendSegPointerDebug(dbg, cu_ctx_, gpu_masks);
                            logstream << dbg.str();
                        }
                        // Transfer ownership of GPU buffer to the side data
                        AVBufferRef* gpu_ref = sr.gpu_mask_buf;
                        sr.gpu_mask_buf = nullptr;
                        AVBufferRef* sd_buf = av_buffer_create(
                            (uint8_t*)header, sizeof(GpuMaskSideDataHeader),
                            [](void* opaque, uint8_t* data) {
                                av_buffer_unref((AVBufferRef**)&opaque);
                                av_free(data);
                            }, gpu_ref, 0);
                        if (sd_buf) {
                            av_frame_new_side_data_from_buf(frm.raw(), yoloSegGpuSideDataType(side_data_slot_), sd_buf);
                        } else {
                            av_buffer_unref(&gpu_ref);
                            av_free(header);
                        }
                    }
                }
                if (sr.gpu_mask_buf) {
                    av_buffer_unref(&sr.gpu_mask_buf);
                }
            } else if (model.task_type == TaskType::Pose && model.pose_decoder) {
                PoseResult pr = model.pose_decoder->decode(host_outputs, output_dims, dp);
                if (model.include_in_detection_metadata) {
                    all_dets.insert(all_dets.end(), pr.detections.begin(), pr.detections.end());
                }

                // Build pose metadata.
                // Keep pose metadata consistent with node-level max_det to avoid rendering low-confidence duplicates.
                size_t pose_emit_count = pr.detections.size();
                if (max_det_ > 0 && pose_emit_count > (size_t)max_det_) {
                    pose_emit_count = (size_t)max_det_;
                }
                if (pose_emit_count > 0) {
                    Parameters pose_md;
                    pose_md["coord_space"] = "model";
                    pose_md["model_width"] = input_w_;
                    pose_md["model_height"] = input_h_;
                    pose_md["num_keypoints"] = pr.num_keypoints;
                    pose_md["poses"] = Parameters::array();

                    int kpt_stride = pr.num_keypoints * 3;
                    for (size_t di = 0; di < pose_emit_count; ++di) {
                        const Detection& d = pr.detections[di];
                        Parameters item;
                        item["cls"] = d.cls;
                        item["conf"] = d.conf;
                        item["xyxy"] = {d.x1, d.y1, d.x2, d.y2};
                        item["model_index"] = d.model_index;
                        if (d.model_index >= 0 && (size_t)d.model_index < models_.size()) {
                            item["engine_name"] = models_[(size_t)d.model_index].engine_name;
                            const auto& cn = models_[(size_t)d.model_index].class_names;
                            if (d.cls >= 0 && (size_t)d.cls < cn.size()) {
                                item["label"] = cn[(size_t)d.cls];
                            }
                        }
                        // Flat keypoint array [x1, y1, c1, x2, y2, c2, ...]
                        item["keypoints"] = Parameters::array();
                        size_t kpt_off = di * (size_t)kpt_stride;
                        for (int k = 0; k < kpt_stride && kpt_off + (size_t)k < pr.keypoints.size(); ++k) {
                            item["keypoints"].push_back(pr.keypoints[kpt_off + (size_t)k]);
                        }
                        pose_md["poses"].push_back(item);
                    }

                    av_dict_set(&frm.raw()->metadata, metadata_key_pose_.c_str(), pose_md.dump().c_str(), 0);
                }

                if (debug_log_metadata_ && debug_log_every_n_ > 0 &&
                    (frame_counter_ % (uint64_t)debug_log_every_n_) == 0) {
                    logstream << "cuda_infer_yolo: pose model=" << model.engine_name
                              << " detections_raw=" << pr.detections.size()
                              << " detections_emitted=" << pose_emit_count
                              << " keypoints_per_det=" << pr.num_keypoints;
                }
            }
        }

        // Finalize: sort by confidence, truncate to max_det
        std::sort(all_dets.begin(), all_dets.end(),
            [](const Detection& a, const Detection& b) { return a.conf > b.conf; });
        if (max_det_ > 0 && (int)all_dets.size() > max_det_) {
            all_dets.resize((size_t)max_det_);
        }

        // Track detection statistics
        ++detection_count_histogram_[(int)all_dets.size()];
        for (const Detection& d : all_dets) {
            int bucket = std::min((int)(d.conf * 10.0f), 9);
            ++conf_histogram_[(size_t)bucket];
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
        r->node_label_ = params.value("name", std::string("<unnamed>"));
        std::shared_ptr<HWAccelDevice> debug_hwaccel;
        if (params.count("hwaccel")) {
            debug_hwaccel = InstanceSharedObjects<HWAccelDevice>::get(nci.instance, params["hwaccel"]);
        }
        r->setCudaContextDebugInfo("cuda_infer_yolo", r->node_label_, debug_hwaccel);

        // Parse global params
        if (params.count("conf_thresh")) r->conf_thresh_ = params["conf_thresh"];
        if (params.count("max_det")) r->max_det_ = params["max_det"];
        if (params.count("infer_every_n")) r->infer_every_n_ = params["infer_every_n"];
        if (params.count("use_cuda_graph")) r->use_cuda_graph_ = params["use_cuda_graph"];
        if (params.count("metadata_key_detection")) r->metadata_key_detection_ = params["metadata_key_detection"].get<std::string>();
        if (params.count("metadata_key_segmentation")) r->metadata_key_segmentation_ = params["metadata_key_segmentation"].get<std::string>();
        if (params.count("metadata_key_pose")) r->metadata_key_pose_ = params["metadata_key_pose"].get<std::string>();
        if (params.count("debug_log_metadata")) r->debug_log_metadata_ = params["debug_log_metadata"];
        if (params.count("debug_log_every_n")) r->debug_log_every_n_ = params["debug_log_every_n"];
        if (params.count("mask_gpu_every_n")) r->mask_gpu_every_n_ = params["mask_gpu_every_n"];
        if (params.count("mask_cpu_every_n")) r->mask_cpu_every_n_ = params["mask_cpu_every_n"];
        if (params.count("mask_cpu_resolution")) r->mask_cpu_resolution_ = params["mask_cpu_resolution"];
        if (params.count("side_data_slot")) {
            r->side_data_slot_ = params["side_data_slot"].get<int>();
            if (!yoloSegIsValidSlot(r->side_data_slot_)) {
                throw Error("cuda_infer_yolo: side_data_slot out of range [0," + std::to_string(kMaxYoloSegSlots - 1) + "]");
            }
        }
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

            if (mp.count("include_in_detection_metadata")) {
                model.include_in_detection_metadata = mp["include_in_detection_metadata"].get<bool>();
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
            if (mp.count("nms_iou_thresh")) {
                model.nms_iou_thresh = mp["nms_iou_thresh"].get<float>();
                if (model.nms_iou_thresh < 0.0f || model.nms_iou_thresh > 1.0f) {
                    throw Error("cuda_infer_yolo: nms_iou_thresh must be in [0,1]");
                }
            }
            if (mp.count("nms_class_agnostic")) {
                model.nms_class_agnostic = mp["nms_class_agnostic"].get<bool>();
            }

            int num_classes = -1;
            if (mp.count("num_classes")) {
                num_classes = mp["num_classes"].get<int>();
            }
            model.num_classes = num_classes;

            // Create decoder
            if (model.task_type == TaskType::Detection) {
                model.det_decoder.reset(new DetectionDecoder());
            } else if (model.task_type == TaskType::Segmentation) {
                model.seg_decoder.reset(new SegmentationDecoder());
            } else if (model.task_type == TaskType::Pose) {
                int nc = model.num_classes;
                if (nc < 1) nc = 1;  // default for pose models
                float nms_iou = 0.45f;
                if (mp.count("nms_iou_thresh")) nms_iou = mp["nms_iou_thresh"].get<float>();
                model.pose_decoder.reset(new PoseDecoder(nc, nms_iou));
            }

            r->models_.push_back(std::move(model));
        }

        return r;
    }
};

DECLNODE(cuda_infer_yolo, CudaInferYolo)
