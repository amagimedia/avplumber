#pragma once
#include "../common/infer_trt_base.hpp"

// PTX blob for mask assembly kernels
#include "../../../../objs/src/nodes/neural_net/preprocess/mask_assemble.ptx.h"

namespace yolo_base {

class SegmentationDecoder {
private:
    CUmodule mask_module_ = nullptr;
    CUfunction mask_assemble_kernel_ = nullptr;
    CUfunction mask_downsample_kernel_ = nullptr;
    CUdeviceptr gpu_mask_scratch_ = 0;   // [max_det, proto_h, proto_w]
    CUdeviceptr gpu_coeff_buf_ = 0;      // [max_det, 32]
    CUdeviceptr gpu_proto_buf_ = 0;      // [32, proto_h, proto_w] — prototypes uploaded from host
    CUdeviceptr gpu_downsample_buf_ = 0; // [cpu_res, cpu_res] per detection
    size_t proto_h_ = 0, proto_w_ = 0;
    int num_proto_channels_ = 32;
    int max_det_ = 0;
    CUcontext cu_ctx_ = nullptr;

public:
    bool init(const ModelRunner& model, CUcontext cu_ctx, int max_det) {
        cu_ctx_ = cu_ctx;
        max_det_ = max_det;

        // Ensure correct CUDA context (TensorRT engine loading may have changed it)
        if (CUDA_CHECK_CU(cuCtxSetCurrent(cu_ctx_))) return false;

        // Load PTX module
        const std::string ptx_str(avpl_yolo_mask_assemble_ptx,
            avpl_yolo_mask_assemble_ptx + avpl_yolo_mask_assemble_ptx_len);
        if (CUDA_CHECK_CU(cuModuleLoadDataEx(&mask_module_, ptx_str.c_str(), 0, nullptr, nullptr)))
            return false;
        if (CUDA_CHECK_CU(cuModuleGetFunction(&mask_assemble_kernel_, mask_module_, "kMaskAssemble")))
            return false;
        if (CUDA_CHECK_CU(cuModuleGetFunction(&mask_downsample_kernel_, mask_module_, "kMaskDownsample")))
            return false;

        // Get prototype dims from second output tensor
        // Expected output1: [1, 32, proto_h, proto_w] or [32, proto_h, proto_w]
        if (model.outputs.size() < 2) {
            logstream << "cuda_infer_yolo: segmentation model needs 2 output tensors, got " << model.outputs.size();
            return false;
        }
        const nvinfer1::Dims& proto_dims = model.outputs[1].dims;
        if (proto_dims.nbDims == 4) {
            num_proto_channels_ = proto_dims.d[1];
            proto_h_ = proto_dims.d[2];
            proto_w_ = proto_dims.d[3];
        } else if (proto_dims.nbDims == 3) {
            num_proto_channels_ = proto_dims.d[0];
            proto_h_ = proto_dims.d[1];
            proto_w_ = proto_dims.d[2];
        } else {
            logstream << "cuda_infer_yolo: unexpected prototype dims (nbDims=" << proto_dims.nbDims << ")";
            return false;
        }

        // Allocate scratch buffers
        size_t mask_bytes = (size_t)max_det_ * proto_h_ * proto_w_ * sizeof(float);
        size_t coeff_bytes = (size_t)max_det_ * (size_t)num_proto_channels_ * sizeof(float);
        size_t proto_bytes = (size_t)num_proto_channels_ * proto_h_ * proto_w_ * sizeof(float);
        if (CUDA_CHECK_CU(cuMemAlloc(&gpu_mask_scratch_, mask_bytes))) return false;
        if (CUDA_CHECK_CU(cuMemAlloc(&gpu_coeff_buf_, coeff_bytes))) return false;
        if (CUDA_CHECK_CU(cuMemAlloc(&gpu_proto_buf_, proto_bytes))) return false;

        return true;
    }

    SegmentationResult decode(
        const std::vector<const float*>& host_outputs,
        const std::vector<nvinfer1::Dims>& output_dims,
        const DecodeParams& params,
        bool emit_gpu_mask,
        bool emit_cpu_mask,
        int cpu_mask_resolution,
        CUstream stream
    ) {
        SegmentationResult result;
        if (host_outputs.size() < 2 || !host_outputs[0] || !host_outputs[1]) return result;
        auto log_ctx = [&](const char* stage, CUdeviceptr ptr = 0) {
            if (!cuCtxGetCurrent) return;
            CUcontext current = nullptr;
            CUcontext stream_ctx = nullptr;
            CUresult err = cuCtxGetCurrent(&current);
            if (err != CUDA_SUCCESS) {
                CUDA_CHECK_CU(err);
                return;
            }
            if (cuStreamGetCtx) {
                CUresult stream_err = cuStreamGetCtx(stream, &stream_ctx);
                if (stream_err != CUDA_SUCCESS) {
                    CUDA_CHECK_CU(stream_err);
                }
            }
            logstream << "cuda_infer_yolo: seg_ctx"
                      << " stage=" << stage
                      << " current=" << (void*)current
                      << " decoder=" << (void*)cu_ctx_
                      << " stream_ctx=" << (void*)stream_ctx
                      << " stream=" << (void*)stream
                      << " scratch=" << (void*)(uintptr_t)gpu_mask_scratch_
                      << " coeff=" << (void*)(uintptr_t)gpu_coeff_buf_
                      << " proto=" << (void*)(uintptr_t)gpu_proto_buf_
                      << " ptr=" << (void*)(uintptr_t)ptr;
        };

        const float* out0 = host_outputs[0];
        const nvinfer1::Dims& d = output_dims[0];

        // 1. Determine layout (same logic as DetectionDecoder, but attrs includes mask coefficients)
        int count = 0, attrs = 0;
        bool attrs_first = false;

        if (d.nbDims == 2) {
            int a = d.d[0], b = d.d[1];
            if (a <= b && a >= 6) { attrs = a; count = b; attrs_first = true; }
            else if (b >= 6) { attrs = b; count = a; attrs_first = false; }
            else if (a >= 6) { attrs = a; count = b; attrs_first = true; }
            else return result;
        } else if (d.nbDims == 3 && d.d[0] == 1) {
            int d1 = d.d[1], d2 = d.d[2];
            if (d1 <= d2 && d1 >= 6) { attrs = d1; count = d2; attrs_first = true; }
            else if (d2 >= 6) { attrs = d2; count = d1; attrs_first = false; }
            else if (d1 >= 6) { attrs = d1; count = d2; attrs_first = true; }
            else return result;
        } else {
            return result;
        }

        if (count <= 0 || attrs < 6) return result;

        // Unified accessor
        auto at = [&](int det, int attr) -> float {
            return attrs_first ? out0[attr * count + det] : out0[det * attrs + attr];
        };

        // Determine number of class scores:
        // attrs = 4 (box) + num_classes + num_proto_channels (mask coefficients)
        // For segmentation models, the last num_proto_channels_ attrs are mask coefficients
        int num_class_attrs = attrs - 4 - num_proto_channels_;
        if (num_class_attrs < 1) {
            // Fallback: assume all attrs after box are class scores (no mask coefficients in output0)
            // This shouldn't happen for proper seg models, but handle gracefully
            num_class_attrs = attrs - 4;
        }
        int coeff_start = 4 + num_class_attrs;  // index where mask coefficients begin

        // Host buffer for coefficients of accepted detections
        std::vector<float> host_coefficients;

        for (int i = 0; i < count; ++i) {
            Detection det;
            det.model_index = params.model_index;

            if (params.box_format == OutputBoxFormat::EndToEndXYXY) {
                det.x1 = at(i, 0); det.y1 = at(i, 1);
                det.x2 = at(i, 2); det.y2 = at(i, 3);
                det.conf = at(i, 4);
                det.cls = (int)std::round(at(i, 5));
            } else { // RawCXCYWH
                float cx = at(i, 0), cy = at(i, 1);
                float w = at(i, 2), h = at(i, 3);
                int best_cls = 0;
                float best = 0.0f;
                for (int c = 4; c < coeff_start; ++c) {
                    float s = at(i, c);
                    if (s > best) { best = s; best_cls = c - 4; }
                }
                det.x1 = cx - w * 0.5f; det.y1 = cy - h * 0.5f;
                det.x2 = cx + w * 0.5f; det.y2 = cy + h * 0.5f;
                det.conf = best;
                det.cls = best_cls;
            }

            if (det.conf < params.conf_thresh) continue;

            // Apply class index remapping
            if (det.cls >= 0 && (size_t)det.cls < params.class_index_remap.size()) {
                det.cls = params.class_index_remap[(size_t)det.cls];
            }

            result.detections.push_back(det);

            // Extract mask coefficients for this detection
            if (coeff_start + num_proto_channels_ <= attrs) {
                for (int c = 0; c < num_proto_channels_; ++c) {
                    host_coefficients.push_back(at(i, coeff_start + c));
                }
            }
        }

        // 2. If neither GPU nor CPU mask needed, or no detections, return early
        if (!emit_gpu_mask && !emit_cpu_mask) return result;
        if (result.detections.empty()) return result;

        int num_dets = std::min((int)result.detections.size(), max_det_);
        result.num_masks = num_dets;
        result.mask_proto_w = (int)proto_w_;
        result.mask_proto_h = (int)proto_h_;

        // Truncate coefficients to max_det
        if ((int)result.detections.size() > max_det_) {
            host_coefficients.resize((size_t)max_det_ * (size_t)num_proto_channels_);
        }

        // 3. Upload coefficients and prototypes to GPU
        size_t coeff_bytes = (size_t)num_dets * (size_t)num_proto_channels_ * sizeof(float);
        if (CUDA_CHECK_CU(cuMemcpyHtoDAsync(gpu_coeff_buf_, host_coefficients.data(), coeff_bytes, stream)))
            return result;

        // Upload prototypes (output1 host data) to GPU
        const float* proto_host = host_outputs[1];
        size_t proto_bytes = (size_t)num_proto_channels_ * proto_h_ * proto_w_ * sizeof(float);
        if (CUDA_CHECK_CU(cuMemcpyHtoDAsync(gpu_proto_buf_, proto_host, proto_bytes, stream)))
            return result;

        // 4. Launch mask assembly kernel
        int proto_h = (int)proto_h_, proto_w = (int)proto_w_;
        int num_coeff = num_proto_channels_;
        void* mask_args[] = {
            (void*)&gpu_proto_buf_,
            (void*)&gpu_coeff_buf_,
            (void*)&gpu_mask_scratch_,
            (void*)&proto_h,
            (void*)&proto_w,
            (void*)&num_coeff
        };
        unsigned int gridX = ((unsigned int)proto_w + 31) / 32;
        unsigned int gridY = ((unsigned int)proto_h + 7) / 8;
        unsigned int gridZ = (unsigned int)num_dets;
        if (CUDA_CHECK_CU(cuLaunchKernel(mask_assemble_kernel_, gridX, gridY, gridZ, 32, 8, 1, 0, stream, mask_args, nullptr)))
            return result;
        if (CUDA_CHECK_CU(cuStreamSynchronize(stream))) {
            log_ctx("after_mask_assemble_sync_failed");
            return result;
        }

        // Keep using the same active context that owns the stream and scratch buffers.
        // Switching contexts here can make the subsequent output allocation/copy invalid.
        // 5. GPU path: wrap assembled masks in AVBufferRef
        if (emit_gpu_mask) {
            log_ctx("before_gpu_output");
            size_t mask_bytes = (size_t)num_dets * proto_h_ * proto_w_ * sizeof(float);
            CUdeviceptr gpu_out = 0;
            if (CUDA_CHECK_CU(cuMemAlloc(&gpu_out, mask_bytes))) {
                // continue without GPU mask
            } else {
                log_ctx("after_gpu_out_alloc", gpu_out);
                if (CUDA_CHECK_CU(cuMemcpyDtoDAsync(gpu_out, gpu_mask_scratch_, mask_bytes, stream))) {
                    log_ctx("gpu_copy_failed", gpu_out);
                    CUDA_CHECK_CU(cuMemFree(gpu_out));
                } else {
                    // Wrap in AVBufferRef with custom free callback
                    result.gpu_mask_buf = av_buffer_create(
                        (uint8_t*)(uintptr_t)gpu_out, (size_t)mask_bytes,
                        [](void*, uint8_t* data) {
                            cuMemFree((CUdeviceptr)(uintptr_t)data);
                        }, nullptr, 0);
                    if (!result.gpu_mask_buf) {
                        CUDA_CHECK_CU(cuMemFree(gpu_out));
                    }
                }
            }
        }

        // 6. CPU path: downsample per detection + D2H
        if (emit_cpu_mask) {
            result.cpu_mask_w = cpu_mask_resolution;
            result.cpu_mask_h = cpu_mask_resolution;
            size_t ds_pixels = (size_t)cpu_mask_resolution * (size_t)cpu_mask_resolution;
            size_t ds_bytes = ds_pixels * sizeof(float);

            if (!gpu_downsample_buf_) {
                if (CUDA_CHECK_CU(cuMemAlloc(&gpu_downsample_buf_, ds_bytes))) {
                    return result;
                }
            }

            result.cpu_masks.resize((size_t)num_dets * ds_pixels);

            int ds_h = cpu_mask_resolution, ds_w = cpu_mask_resolution;
            unsigned int ds_gridX = ((unsigned int)ds_w + 31) / 32;
            unsigned int ds_gridY = ((unsigned int)ds_h + 7) / 8;

            for (int d = 0; d < num_dets; ++d) {
                // Source: gpu_mask_scratch_ + offset for this detection
                CUdeviceptr src_mask = gpu_mask_scratch_ + (size_t)d * proto_h_ * proto_w_ * sizeof(float);
                void* ds_args[] = {
                    (void*)&src_mask,
                    (void*)&gpu_downsample_buf_,
                    (void*)&proto_h,
                    (void*)&proto_w,
                    (void*)&ds_h,
                    (void*)&ds_w
                };
                if (CUDA_CHECK_CU(cuLaunchKernel(mask_downsample_kernel_, ds_gridX, ds_gridY, 1, 32, 8, 1, 0, stream, ds_args, nullptr)))
                    break;

                // D2H copy
                float* host_dst = result.cpu_masks.data() + (size_t)d * ds_pixels;
                if (CUDA_CHECK_CU(cuMemcpyDtoHAsync(host_dst, gpu_downsample_buf_, ds_bytes, stream)))
                    break;
            }

            // Sync to ensure CPU masks are ready
            CUDA_CHECK_CU(cuStreamSynchronize(stream));
        }

        return result;
    }

    void cleanup() {
        if (cu_ctx_) CUDA_CHECK_CU(cuCtxSetCurrent(cu_ctx_));
        if (gpu_mask_scratch_) { CUDA_CHECK_CU(cuMemFree(gpu_mask_scratch_)); gpu_mask_scratch_ = 0; }
        if (gpu_coeff_buf_) { CUDA_CHECK_CU(cuMemFree(gpu_coeff_buf_)); gpu_coeff_buf_ = 0; }
        if (gpu_proto_buf_) { CUDA_CHECK_CU(cuMemFree(gpu_proto_buf_)); gpu_proto_buf_ = 0; }
        if (gpu_downsample_buf_) { CUDA_CHECK_CU(cuMemFree(gpu_downsample_buf_)); gpu_downsample_buf_ = 0; }
        if (mask_module_) { CUDA_CHECK_CU(cuModuleUnload(mask_module_)); mask_module_ = nullptr; }
    }
};

} // namespace yolo_base
