#pragma once

#include <cuda_loader/cuda_drvapi_dynlink_cuda.h>
#include <dlfcn.h>

extern "C" {
#include <libavutil/pixfmt.h>
#include <libavutil/version.h>
}

namespace avp::cuda {

inline bool isArrayFormat(AVPixelFormat format) {
#if LIBAVUTIL_VERSION_INT >= AV_VERSION_INT(61, 5, 100)
    return format == AV_PIX_FMT_CUARRAY;
#else
    return false;
#endif
}

// The bundled CUDA loader predates planar arrays. Resolve this entry point only
// for array users; linear CUDA input must still work with older drivers.
// The returned plane is borrowed: keep the source frame alive until reads finish.
inline CUresult arrayGetPlane(CUarray *plane, CUarray array, unsigned index) {
    using GetPlane = CUresult (CUDAAPI *)(CUarray *, CUarray, unsigned);
    struct Driver {
        void *handle = dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
        GetPlane get = handle ? reinterpret_cast<GetPlane>(dlsym(handle, "cuArrayGetPlane")) : nullptr;
        ~Driver() { if (handle) dlclose(handle); }
    };
    static const Driver driver;
    return driver.get ? driver.get(plane, array, index) : CUDA_ERROR_NOT_FOUND;
}

} // namespace avp::cuda
