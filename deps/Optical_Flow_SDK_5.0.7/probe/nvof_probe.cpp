// Standalone NVOF dense-CUDA link/init probe.
//
// Purpose: prove the headers compile and the driver's
// libnvidia-opticalflow.so.1 links + initializes on this GPU, BEFORE wiring an
// avplumber node. It does NOT compute flow on real frames — it creates a CUDA
// context, creates an NVOF CUDA API instance, queries capabilities, and exits.
//
// Build (on a box with CUDA toolkit + NVIDIA driver):
//   g++ -std=c++17 nvof_probe.cpp -o nvof_probe \
//       -I../NvOFInterface \
//       -I/usr/local/cuda/include \
//       -L/usr/local/cuda/lib64 -lcuda -lcudart -lnvidia-opticalflow
//
// Run (needs an NVIDIA GPU with an optical-flow engine, e.g. Turing T4):
//   ./nvof_probe
// Expected: prints API version, creates instance, prints a capability, "OK".

#include <cstdio>
#include <cstring>
#include <cuda.h>
#include <cuda_runtime.h>

#include "nvOpticalFlowCommon.h"
#include "nvOpticalFlowCuda.h"

#define CK_CU(x) do { CUresult r=(x); if(r!=CUDA_SUCCESS){ const char* s=nullptr; cuGetErrorString(r,&s); fprintf(stderr,"CU error %d (%s) at %s:%d\n",r,s?s:"?",__FILE__,__LINE__); return 2; } } while(0)
#define CK_OF(x) do { NV_OF_STATUS s=(x); if(s!=NV_OF_SUCCESS){ fprintf(stderr,"NVOF error %d at %s:%d\n",(int)s,__FILE__,__LINE__); return 3; } } while(0)

int main() {
    CK_CU(cuInit(0));
    int devCount = 0;
    CK_CU(cuDeviceGetCount(&devCount));
    if (devCount < 1) { fprintf(stderr, "no CUDA device\n"); return 4; }

    // Establish a primary context via the runtime API (avoids the CUDA 13
    // driver cuCtxCreate_v4 signature change), then retain it for NVOF.
    if (cudaSetDevice(0) != cudaSuccess) { fprintf(stderr, "cudaSetDevice failed\n"); return 5; }
    if (cudaFree(0) != cudaSuccess) { fprintf(stderr, "cudaFree(0) ctx init failed\n"); return 5; }

    CUdevice dev;
    CK_CU(cuDeviceGet(&dev, 0));
    char name[128] = {0};
    CK_CU(cuDeviceGetName(name, sizeof(name), dev));
    CUcontext ctx = nullptr;
    CK_CU(cuCtxGetCurrent(&ctx));
    if (ctx == nullptr) { CK_CU(cuDevicePrimaryCtxRetain(&ctx, dev)); CK_CU(cuCtxSetCurrent(ctx)); }
    printf("CUDA device 0: %s\n", name);

    // Populate the NVOF CUDA API function-pointer table from the driver lib.
    NV_OF_CUDA_API_FUNCTION_LIST of;
    memset(&of, 0, sizeof(of));
    printf("NV_OF_API_VERSION = 0x%x\n", (unsigned)NV_OF_API_VERSION);
    CK_OF(NvOFAPICreateInstanceCuda(NV_OF_API_VERSION, &of));
    printf("NvOFAPICreateInstanceCuda: OK (function table populated)\n");

    // Create an NVOF handle bound to the CUDA context.
    NvOFHandle hOF = nullptr;
    CK_OF(of.nvCreateOpticalFlowCuda(ctx, &hOF));
    printf("nvCreateOpticalFlowCuda: OK\n");

    // Query a capability to confirm the engine is live (max supported width).
    uint32_t vals[64];
    uint32_t size = 0;
    NV_OF_STATUS capStatus = of.nvOFGetCaps(hOF, NV_OF_CAPS_WIDTH_MAX, vals, &size);
    if (capStatus == NV_OF_SUCCESS && size > 0) {
        printf("NV_OF_CAPS_WIDTH_MAX = %u\n", vals[0]);
    } else {
        printf("nvOFGetCaps(WIDTH_MAX) status=%d size=%u (engine present, cap query non-fatal)\n",
               (int)capStatus, size);
    }

    of.nvOFDestroy(hOF);
    cudaDeviceReset();
    printf("OK: NVOF dense CUDA engine reachable on this GPU.\n");
    return 0;
}
