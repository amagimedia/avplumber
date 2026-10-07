// Loads the CUDA driver entry points cuda_rect_draw calls (cuCtxSetCurrent,
// cuModuleLoadDataEx, …). FFmpeg's own hwdevice init does not fill this table.

#include <cuda_loader/cuda_drvapi_dynlink_cuda.h>

#include <cstdio>

extern "C" void avp_cpp_cuda_init(void) {
    static int once = 0;
    if (once) {
        return;
    }
    once = 1;
    const CUresult status = cuInit_drvapi(0, __CUDA_API_VERSION);
    if (status != CUDA_SUCCESS) {
        std::fprintf(stderr, "cuInit_drvapi failed: %d\n", static_cast<int>(status));
    }
}
