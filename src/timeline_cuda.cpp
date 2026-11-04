#include "timeline.hpp"
// A file of its own: nvToolsExtCuda.h includes the real cuda.h, which the dynamic loader's
// header avplumber uses elsewhere replaces.
#if __has_include(<nvtx3/nvToolsExtCuda.h>)
#include <nvtx3/nvToolsExtCuda.h>
#endif

namespace avp::timeline {

void nameCudaStream([[maybe_unused]] void *stream, [[maybe_unused]] const std::string &name) {
#ifdef AVP_HAVE_NVTX
    nvtxNameCuStreamA(static_cast<CUstream>(stream), name.c_str());
#endif
}

}  // namespace avp::timeline
