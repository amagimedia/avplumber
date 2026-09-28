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

    // No context is created here: nodes run in the context of their hwaccel device, and an
    // unused process-wide context would still hold device memory.
    if (cuda_error || device_count < 1) {
        logstream << "failed to find a CUDA device";
        global_cuda.has_errors = true;
        return;
    }

    logstream << "cuda initialized successfully";
}
