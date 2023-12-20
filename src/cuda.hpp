#include <cuda_loader/cuda_wrapper_include.h>

struct CUDAState {
    bool has_errors;

    CUdevice device;
    CUcontext cu_ctx;
};

extern CUDAState global_cuda;
