#include "cuda.hpp"
#include "util.hpp"

CUDAState global_cuda;

__attribute__((constructor)) void init_global_cuda() {
    CUresult status = cuInit_drvapi(0, __CUDA_API_VERSION);
    if (CUDA_SUCCESS != status) {
        logstream << "failed to initialize CUDA (cuInit_drvapi)";
        global_cuda.has_errors = true;
        return;
    }

    if (!cuDeviceGetCount) {
        logstream << "failed to load CUDA functions silently";
        global_cuda.has_errors = true;
        return;
    }

    int cuda_error = 0;
    int device_count = 0;
    cuda_error |= CHECK_CU(cuDeviceGetCount(&device_count));
    logstream << "initializing cuda. Device count: " << device_count;

    cuda_error |= CHECK_CU(cuDeviceGet(&global_cuda.device, 0));
    cuda_error |=
        CHECK_CU(cuCtxCreate(&global_cuda.cu_ctx, 0, global_cuda.device));
    
    if (cuda_error) {
        logstream << "failed to initialize CUDA context";
        global_cuda.has_errors = true;
        return;
    }

    logstream << "cuda initialized successfully";
}
