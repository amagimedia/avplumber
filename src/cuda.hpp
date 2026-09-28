#include <cuda_loader/cuda_wrapper_include.h>

// Driver initialization result; has no CUDA context of its own.
struct CUDAState {
    bool has_errors;

    CUdevice device;
};

extern CUDAState global_cuda;
