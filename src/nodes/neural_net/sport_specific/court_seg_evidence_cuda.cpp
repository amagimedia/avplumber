#include "../../node_common.hpp"
#include "../common/yolo_side_data.hpp"
#include <cuda_loader/cuda_drvapi_dynlink_cuda.h>

extern "C" {
#include <libavutil/buffer.h>
#include <libavutil/dict.h>
#include <libavutil/frame.h>
#include <libavutil/hwcontext.h>
#include <libavutil/hwcontext_cuda.h>
}

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <sstream>
#include <string>
#include <vector>

#include "../../../../objs/src/nodes/neural_net/sport_specific/court_seg_evidence_cuda.ptx.h"

namespace {

int checkCu(CUresult err, const char* func) {
    if (err == CUDA_SUCCESS) return 0;
    const char* err_name = nullptr;
    const char* err_string = nullptr;
    if (cuGetErrorName && cuGetErrorString) {
        cuGetErrorName(err, &err_name);
        cuGetErrorString(err, &err_string);
    }
    logstream << "cuda function: " << func << " failed: "
              << (err_name ? err_name : "?") << ": " << (err_string ? err_string : "?");
    return -1;
}

#define COURT_EVIDENCE_CHECK_CU(x) checkCu((x), #x)

} // namespace

class CourtSegEvidenceCuda : public NodeSISO<av::VideoFrame, av::VideoFrame>, public ReportsFinishByFlag {
    int side_data_slot_ = 0;
    int output_side_data_slot_ = 0;
    int mask_w_ = 272;
    int mask_h_ = 272;
    float mask_threshold_ = 0.5f;
    int emit_cpu_every_n_ = 1;
    bool emit_luma_ = false;
    bool clear_cpu_side_data_ = true;
    std::string output_metadata_key_ = "court_seg_evidence";
    int debug_log_every_n_ = 0;

    AVCUDADeviceContext* cuda_dev_ctx_ = nullptr;
    CUcontext cu_ctx_ = nullptr;
    CUmodule cu_module_ = nullptr;
    CUfunction downsample_kernel_ = nullptr;
    CUdeviceptr d_downsample_ = 0;
    CUdeviceptr d_counts_ = 0;
    size_t downsample_capacity_bytes_ = 0;
    size_t counts_capacity_bytes_ = 0;
    uint64_t frame_counter_ = 0;

    std::vector<float> host_masks_;
    std::vector<uint32_t> host_counts_;

    bool shouldEmitCpu() const {
        return emit_cpu_every_n_ > 0 &&
               ((frame_counter_ - 1) % (uint64_t)emit_cpu_every_n_) == 0;
    }

    bool initCudaContextFromFrame(const av::VideoFrame& frm) {
        if (!frm.raw() || !frm.raw()->hw_frames_ctx || !frm.raw()->hw_frames_ctx->data) return false;
        AVHWFramesContext* fctx = (AVHWFramesContext*)frm.raw()->hw_frames_ctx->data;
        if (!fctx || !fctx->device_ctx || !fctx->device_ctx->hwctx) return false;
        AVCUDADeviceContext* next = (AVCUDADeviceContext*)fctx->device_ctx->hwctx;
        if (!next || !next->cuda_ctx) return false;
        cuda_dev_ctx_ = next;
        cu_ctx_ = next->cuda_ctx;
        return COURT_EVIDENCE_CHECK_CU(cuCtxSetCurrent(cu_ctx_)) == 0;
    }

    bool loadKernel() {
        if (cu_module_ && downsample_kernel_) return true;
        if (!cu_ctx_) return false;
        if (COURT_EVIDENCE_CHECK_CU(cuCtxSetCurrent(cu_ctx_))) return false;
        const std::string ptx_str(avpl_court_seg_evidence_cuda_ptx,
                                  avpl_court_seg_evidence_cuda_ptx + avpl_court_seg_evidence_cuda_ptx_len);
        if (COURT_EVIDENCE_CHECK_CU(cuModuleLoadDataEx(&cu_module_, ptx_str.c_str(), 0, nullptr, nullptr))) return false;
        if (COURT_EVIDENCE_CHECK_CU(cuModuleGetFunction(&downsample_kernel_, cu_module_, "kCourtSegEvidenceDownsample"))) return false;
        return true;
    }

    void releaseBuffers() {
        if (!cu_ctx_) return;
        COURT_EVIDENCE_CHECK_CU(cuCtxSetCurrent(cu_ctx_));
        if (d_downsample_) {
            COURT_EVIDENCE_CHECK_CU(cuMemFree(d_downsample_));
            d_downsample_ = 0;
        }
        if (d_counts_) {
            COURT_EVIDENCE_CHECK_CU(cuMemFree(d_counts_));
            d_counts_ = 0;
        }
        downsample_capacity_bytes_ = 0;
        counts_capacity_bytes_ = 0;
    }

    bool ensureBuffers(uint32_t num_masks, bool need_masks) {
        const size_t counts_bytes = (size_t)num_masks * sizeof(uint32_t);
        if (counts_bytes > counts_capacity_bytes_) {
            if (d_counts_) {
                COURT_EVIDENCE_CHECK_CU(cuMemFree(d_counts_));
                d_counts_ = 0;
                counts_capacity_bytes_ = 0;
            }
            if (COURT_EVIDENCE_CHECK_CU(cuMemAlloc(&d_counts_, counts_bytes))) return false;
            counts_capacity_bytes_ = counts_bytes;
        }

        if (!need_masks) return true;
        const size_t mask_pixels = (size_t)mask_w_ * (size_t)mask_h_;
        const size_t mask_bytes = (size_t)num_masks * mask_pixels * sizeof(float);
        if (mask_bytes <= downsample_capacity_bytes_) return true;
        if (d_downsample_) {
            COURT_EVIDENCE_CHECK_CU(cuMemFree(d_downsample_));
            d_downsample_ = 0;
            downsample_capacity_bytes_ = 0;
        }
        if (COURT_EVIDENCE_CHECK_CU(cuMemAlloc(&d_downsample_, mask_bytes))) return false;
        downsample_capacity_bytes_ = mask_bytes;
        return true;
    }

    void attachEmptyEvidence(av::VideoFrame& frm, const char* status) {
        if (!frm.raw()) return;
        Parameters md = Parameters::object();
        md["version"] = 1;
        md["status"] = status;
        md["source"] = "gpu_yolo_seg_masks";
        md["side_data_slot"] = side_data_slot_;
        md["cpu_masks_attached"] = false;
        md["coverages"] = Parameters::array();
        av_dict_set(&frm.raw()->metadata, output_metadata_key_.c_str(), md.dump().c_str(), 0);
    }

    void attachEvidenceMetadata(av::VideoFrame& frm,
                                const GpuMaskSideDataHeader& header,
                                bool cpu_attached) {
        const uint64_t denom = (uint64_t)mask_w_ * (uint64_t)mask_h_;
        Parameters md = Parameters::object();
        md["version"] = 1;
        md["status"] = "ok";
        md["source"] = "gpu_yolo_seg_masks";
        md["side_data_slot"] = side_data_slot_;
        md["cpu_side_data_slot"] = output_side_data_slot_;
        md["num_masks"] = (int)header.num_masks;
        md["proto_w"] = (int)header.proto_w;
        md["proto_h"] = (int)header.proto_h;
        md["model_w"] = (int)header.model_w;
        md["model_h"] = (int)header.model_h;
        md["mask_w"] = mask_w_;
        md["mask_h"] = mask_h_;
        md["mask_threshold"] = mask_threshold_;
        md["cpu_masks_attached"] = cpu_attached;
        md["coverages"] = Parameters::array();
        for (uint32_t i = 0; i < header.num_masks && i < host_counts_.size(); ++i) {
            md["coverages"].push_back(denom > 0 ? (double)host_counts_[i] / (double)denom : 0.0);
        }
        av_dict_set(&frm.raw()->metadata, output_metadata_key_.c_str(), md.dump().c_str(), 0);
    }

    bool attachLumaSideData(av::VideoFrame& frm) {
        if (!emit_luma_) return false;
        AVFrame* raw = frm.raw();
        if (!raw || !raw->data[0]) return false;
        const int w = frm.width();
        const int h = frm.height();
        if (w <= 0 || h <= 0) return false;
        const size_t header_size = 16;
        AVBufferRef* buf = av_buffer_alloc(header_size + (size_t)w * (size_t)h);
        if (!buf) return false;
        uint32_t* header = (uint32_t*)buf->data;
        header[0] = (uint32_t)w;
        header[1] = (uint32_t)h;
        header[2] = 0;
        header[3] = 0;
        CUDA_MEMCPY2D cp;
        memset(&cp, 0, sizeof(cp));
        cp.srcMemoryType = CU_MEMORYTYPE_DEVICE;
        cp.srcDevice = (CUdeviceptr)raw->data[0];
        cp.srcPitch = (size_t)raw->linesize[0];
        cp.dstMemoryType = CU_MEMORYTYPE_HOST;
        cp.dstHost = buf->data + header_size;
        cp.dstPitch = (size_t)w;
        cp.WidthInBytes = (size_t)w;
        cp.Height = (size_t)h;
        if (COURT_EVIDENCE_CHECK_CU(cuMemcpy2DAsync(&cp, cuda_dev_ctx_->stream)) ||
            COURT_EVIDENCE_CHECK_CU(cuStreamSynchronize(cuda_dev_ctx_->stream))) {
            av_buffer_unref(&buf);
            return false;
        }
        av_frame_remove_side_data(raw, AV_FRAME_DATA_COURT_LUMA);
        AVFrameSideData* attached = av_frame_new_side_data_from_buf(raw, AV_FRAME_DATA_COURT_LUMA, buf);
        if (!attached) {
            av_buffer_unref(&buf);
            return false;
        }
        return true;
    }

    bool attachCpuSideData(av::VideoFrame& frm, uint32_t num_masks) {
        const size_t mask_pixels = (size_t)mask_w_ * (size_t)mask_h_;
        const size_t header_size = 16;
        const size_t mask_data_size = (size_t)num_masks * mask_pixels * sizeof(float);
        AVBufferRef* buf = av_buffer_alloc(header_size + mask_data_size);
        if (!buf) return false;

        uint32_t* header = (uint32_t*)buf->data;
        header[0] = num_masks;
        header[1] = (uint32_t)mask_w_;
        header[2] = (uint32_t)mask_h_;
        header[3] = 0;
        if (mask_data_size > 0) {
            memcpy(buf->data + header_size, host_masks_.data(), mask_data_size);
        }

        AVFrame* raw = frm.raw();
        av_frame_remove_side_data(raw, yoloSegCpuSideDataType(output_side_data_slot_));
        AVFrameSideData* attached = av_frame_new_side_data_from_buf(raw, yoloSegCpuSideDataType(output_side_data_slot_), buf);
        if (!attached) {
            av_buffer_unref(&buf);
            return false;
        }
        return true;
    }

public:
    using NodeSISO<av::VideoFrame, av::VideoFrame>::NodeSISO;

    bool consumeEofIfPresent() override {
        return false;
    }

    ~CourtSegEvidenceCuda() {
        if (cu_ctx_) {
            COURT_EVIDENCE_CHECK_CU(cuCtxSetCurrent(cu_ctx_));
        }
        releaseBuffers();
        if (cu_module_) {
            COURT_EVIDENCE_CHECK_CU(cuModuleUnload(cu_module_));
            cu_module_ = nullptr;
        }
    }

    void process() override {
        av::VideoFrame frm = this->source_->get();
        if (isEofMarker(frm)) {
            frame_counter_ = 0;
            this->sink_->put(frm);
            this->finished_ = true;
            return;
        }
        if (!frm) return;

        ++frame_counter_;
        AVFrame* raw = frm.raw();
        if (!raw) {
            this->sink_->put(frm);
            return;
        }
        if (clear_cpu_side_data_) {
            av_frame_remove_side_data(raw, yoloSegCpuSideDataType(output_side_data_slot_));
        }

        const AVFrameSideData* sd = av_frame_get_side_data(raw, yoloSegGpuSideDataType(side_data_slot_));
        if (!sd || !sd->buf || sd->buf->size < (int)sizeof(GpuMaskSideDataHeader)) {
            attachEmptyEvidence(frm, "no_gpu_masks");
            this->sink_->put(frm);
            return;
        }

        const auto* header = (const GpuMaskSideDataHeader*)sd->buf->data;
        const CUdeviceptr gpu_masks = (CUdeviceptr)header->gpu_ptr;
        if (!gpu_masks || header->num_masks == 0 || header->proto_w == 0 || header->proto_h == 0) {
            attachEmptyEvidence(frm, "invalid_gpu_masks");
            this->sink_->put(frm);
            return;
        }

        const bool emit_cpu = shouldEmitCpu();
        if (!initCudaContextFromFrame(frm) || !loadKernel() || !ensureBuffers(header->num_masks, emit_cpu)) {
            attachEmptyEvidence(frm, "cuda_unavailable");
            this->sink_->put(frm);
            return;
        }

        host_counts_.assign((size_t)header->num_masks, 0);
        if (COURT_EVIDENCE_CHECK_CU(cuMemcpyHtoDAsync(d_counts_, host_counts_.data(),
                                                      (size_t)header->num_masks * sizeof(uint32_t),
                                                      cuda_dev_ctx_->stream))) {
            attachEmptyEvidence(frm, "count_clear_failed");
            this->sink_->put(frm);
            return;
        }

        const int num_masks = (int)header->num_masks;
        const int proto_w = (int)header->proto_w;
        const int proto_h = (int)header->proto_h;
        const int write_masks = emit_cpu ? 1 : 0;
        void* args[] = {
            (void*)&gpu_masks,
            (void*)&d_downsample_,
            (void*)&d_counts_,
            (void*)&num_masks,
            (void*)&proto_w,
            (void*)&proto_h,
            (void*)&mask_w_,
            (void*)&mask_h_,
            (void*)&mask_threshold_,
            (void*)&write_masks,
        };

        const unsigned int block_x = 16;
        const unsigned int block_y = 16;
        const unsigned int grid_x = ((unsigned int)mask_w_ + block_x - 1) / block_x;
        const unsigned int grid_y = ((unsigned int)mask_h_ + block_y - 1) / block_y;
        if (COURT_EVIDENCE_CHECK_CU(cuLaunchKernel(downsample_kernel_,
                                                  grid_x, grid_y, (unsigned int)num_masks,
                                                  block_x, block_y, 1,
                                                  0, cuda_dev_ctx_->stream, args, nullptr))) {
            attachEmptyEvidence(frm, "kernel_failed");
            this->sink_->put(frm);
            return;
        }

        if (COURT_EVIDENCE_CHECK_CU(cuMemcpyDtoHAsync(host_counts_.data(), d_counts_,
                                                      (size_t)num_masks * sizeof(uint32_t),
                                                      cuda_dev_ctx_->stream))) {
            attachEmptyEvidence(frm, "count_copy_failed");
            this->sink_->put(frm);
            return;
        }

        bool cpu_attached = false;
        if (emit_cpu) {
            const size_t mask_pixels = (size_t)mask_w_ * (size_t)mask_h_;
            const size_t mask_values = (size_t)num_masks * mask_pixels;
            host_masks_.assign(mask_values, 0.0f);
            if (COURT_EVIDENCE_CHECK_CU(cuMemcpyDtoHAsync(host_masks_.data(), d_downsample_,
                                                          mask_values * sizeof(float),
                                                          cuda_dev_ctx_->stream))) {
                attachEmptyEvidence(frm, "mask_copy_failed");
                this->sink_->put(frm);
                return;
            }
        }

        if (COURT_EVIDENCE_CHECK_CU(cuStreamSynchronize(cuda_dev_ctx_->stream))) {
            attachEmptyEvidence(frm, "stream_sync_failed");
            this->sink_->put(frm);
            return;
        }

        if (emit_cpu) {
            cpu_attached = attachCpuSideData(frm, header->num_masks);
            attachLumaSideData(frm);
        }
        attachEvidenceMetadata(frm, *header, cpu_attached);

        if (debug_log_every_n_ > 0 && (frame_counter_ % (uint64_t)debug_log_every_n_) == 0) {
            std::ostringstream oss;
            oss << "court_seg_evidence_cuda: frame=" << frame_counter_
                << " masks=" << header->num_masks
                << " proto=" << header->proto_w << "x" << header->proto_h
                << " cpu_masks=" << (cpu_attached ? "yes" : "no")
                << " coverages=[";
            for (size_t i = 0; i < host_counts_.size(); ++i) {
                if (i) oss << ",";
                const double denom = (double)mask_w_ * (double)mask_h_;
                oss << (denom > 0.0 ? (double)host_counts_[i] / denom : 0.0);
            }
            oss << "]";
            logstream << oss.str();
        }

        this->sink_->put(frm);
    }

    static std::shared_ptr<CourtSegEvidenceCuda> create(NodeCreationInfo& nci) {
        EdgeManager& edges = nci.edges;
        const Parameters& params = nci.params;
        auto r = NodeSISO<av::VideoFrame, av::VideoFrame>::template createCommon<CourtSegEvidenceCuda>(edges, params);
        if (params.count("side_data_slot")) r->side_data_slot_ = params["side_data_slot"].get<int>();
        if (params.count("output_side_data_slot")) r->output_side_data_slot_ = params["output_side_data_slot"].get<int>();
        if (!yoloSegIsValidSlot(r->side_data_slot_) || !yoloSegIsValidSlot(r->output_side_data_slot_)) {
            throw Error("court_seg_evidence_cuda: side-data slot out of range [0," + std::to_string(kMaxYoloSegSlots - 1) + "]");
        }
        if (params.count("mask_w")) r->mask_w_ = params["mask_w"].get<int>();
        if (params.count("mask_h")) r->mask_h_ = params["mask_h"].get<int>();
        if (params.count("mask_resolution")) {
            r->mask_w_ = params["mask_resolution"].get<int>();
            r->mask_h_ = params["mask_resolution"].get<int>();
        }
        if (r->mask_w_ <= 0 || r->mask_h_ <= 0) {
            throw Error("court_seg_evidence_cuda: mask dimensions must be positive");
        }
        if (params.count("mask_threshold")) r->mask_threshold_ = params["mask_threshold"].get<float>();
        if (params.count("emit_cpu_every_n")) r->emit_cpu_every_n_ = params["emit_cpu_every_n"].get<int>();
        if (params.count("emit_luma")) r->emit_luma_ = params["emit_luma"].get<bool>();
        if (params.count("clear_cpu_side_data")) r->clear_cpu_side_data_ = params["clear_cpu_side_data"].get<bool>();
        if (params.count("output_metadata_key")) r->output_metadata_key_ = params["output_metadata_key"].get<std::string>();
        if (params.count("debug_log_every_n")) r->debug_log_every_n_ = params["debug_log_every_n"].get<int>();
        return r;
    }
};

DECLNODE(court_seg_evidence_cuda, CourtSegEvidenceCuda)
