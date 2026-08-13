// NVOF dense-flow probe (Step 2a): compute real optical flow on two synthetic
// grayscale frames and verify the recovered horizontal shift.
//
// frame A = vertical-stripe texture; frame B = A shifted right by SHIFT px.
// A rightward image shift means content moved +x, so NVOF flow (reference->input
// mapping) should report a consistent horizontal component of magnitude ~SHIFT.
// We report the median flowx over the grid; |median| ~= SHIFT proves the engine,
// the GRAYSCALE8 upload, the S10.5 fixed-point decode, and the execute path.
//
// Build:
//   g++ -std=c++17 -O2 nvof_flow_probe.cpp -o nvof_flow_probe \
//     -I../NvOFInterface -I/usr/local/cuda/include \
//     -L/usr/local/cuda/lib64 -L/usr/local/cuda/lib64/stubs \
//     -lcuda -lcudart -lnvidia-opticalflow
// Run on a GPU with an OF engine (Turing+). Exit 0 on success.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>
#include <cmath>
#include <cuda.h>
#include <cuda_runtime.h>

#include "nvOpticalFlowCommon.h"
#include "nvOpticalFlowCuda.h"

#define CK_CU(x) do { CUresult r=(x); if(r!=CUDA_SUCCESS){ const char* s=nullptr; cuGetErrorString(r,&s); fprintf(stderr,"CU error %d (%s) %s:%d\n",r,s?s:"?",__FILE__,__LINE__); return 2; } } while(0)
#define CK_RT(x) do { cudaError_t e=(x); if(e!=cudaSuccess){ fprintf(stderr,"RT error %s %s:%d\n",cudaGetErrorString(e),__FILE__,__LINE__); return 2; } } while(0)
#define CK_OF(x) do { NV_OF_STATUS s=(x); if(s!=NV_OF_SUCCESS){ fprintf(stderr,"NVOF error %d %s:%d\n",(int)s,__FILE__,__LINE__); return 3; } } while(0)

static const int W = 256, H = 256, SHIFT = 8;
static const int GRID = NV_OF_OUTPUT_VECTOR_GRID_SIZE_4;  // 4x4 output grid

int main() {
    CK_CU(cuInit(0));
    CK_RT(cudaSetDevice(0));
    CK_RT(cudaFree(0));
    CUcontext ctx = nullptr;
    CK_CU(cuCtxGetCurrent(&ctx));

    NV_OF_CUDA_API_FUNCTION_LIST of; memset(&of, 0, sizeof(of));
    CK_OF(NvOFAPICreateInstanceCuda(NV_OF_API_VERSION, &of));
    NvOFHandle hOF = nullptr;
    CK_OF(of.nvCreateOpticalFlowCuda(ctx, &hOF));
    CK_OF(of.nvOFSetIOCudaStreams(hOF, 0, 0));  // default stream

    // Init: optical-flow mode, 4x4 grid, slow/best-quality perf level.
    NV_OF_INIT_PARAMS init; memset(&init, 0, sizeof(init));
    init.width = W; init.height = H;
    init.outGridSize = (NV_OF_OUTPUT_VECTOR_GRID_SIZE)GRID;
    init.mode = NV_OF_MODE_OPTICALFLOW;
    init.perfLevel = NV_OF_PERF_LEVEL_SLOW;
    CK_OF(of.nvOFInit(hOF, &init));

    // Allocate two input (GRAYSCALE8) buffers + one output (flow vector) buffer.
    auto makeBuf = [&](NV_OF_BUFFER_USAGE usage, NV_OF_BUFFER_FORMAT fmt, int w, int h, NvOFGPUBufferHandle* out) -> NV_OF_STATUS {
        NV_OF_BUFFER_DESCRIPTOR d; memset(&d, 0, sizeof(d));
        d.width = w; d.height = h; d.bufferUsage = usage; d.bufferFormat = fmt;
        return of.nvOFCreateGPUBufferCuda(hOF, &d, NV_OF_CUDA_BUFFER_TYPE_CUDEVICEPTR, out);
    };
    const int gw = (W + GRID - 1) / GRID, gh = (H + GRID - 1) / GRID;
    NvOFGPUBufferHandle inBuf=nullptr, refBuf=nullptr, outBuf=nullptr;
    CK_OF(makeBuf(NV_OF_BUFFER_USAGE_INPUT,  NV_OF_BUFFER_FORMAT_GRAYSCALE8, W, H, &inBuf));
    CK_OF(makeBuf(NV_OF_BUFFER_USAGE_INPUT,  NV_OF_BUFFER_FORMAT_GRAYSCALE8, W, H, &refBuf));
    CK_OF(makeBuf(NV_OF_BUFFER_USAGE_OUTPUT, NV_OF_BUFFER_FORMAT_SHORT2,     gw, gh, &outBuf));

    // Build host frames: A = vertical stripes; B = A shifted right by SHIFT.
    std::vector<uint8_t> A(W*H), B(W*H);
    for (int y=0;y<H;y++) for (int x=0;x<W;x++) {
        uint8_t v = (uint8_t)(((x/8)%2) ? 220 : 30);
        A[y*W+x] = v;
        int sx = x - SHIFT; B[y*W+x] = (sx>=0) ? (uint8_t)(((sx/8)%2)?220:30) : 30;
    }

    // Upload to the OF input buffers (respect stride).
    auto upload = [&](NvOFGPUBufferHandle buf, const std::vector<uint8_t>& host) -> NV_OF_STATUS {
        NV_OF_CUDA_BUFFER_STRIDE_INFO si; memset(&si,0,sizeof(si));
        of.nvOFGPUBufferGetStrideInfo(buf, &si);
        CUdeviceptr dptr = of.nvOFGPUBufferGetCUdeviceptr(buf);
        CUDA_MEMCPY2D m; memset(&m,0,sizeof(m));
        m.srcMemoryType=CU_MEMORYTYPE_HOST; m.srcHost=host.data(); m.srcPitch=W;
        m.dstMemoryType=CU_MEMORYTYPE_DEVICE; m.dstDevice=dptr; m.dstPitch=si.strideInfo[0].strideXInBytes;
        m.WidthInBytes=W; m.Height=H;
        return (cuMemcpy2D(&m)==CUDA_SUCCESS) ? NV_OF_SUCCESS : NV_OF_ERR_GENERIC;
    };
    CK_OF(upload(inBuf, B));    // input = current frame (B)
    CK_OF(upload(refBuf, A));   // reference = previous frame (A)

    NV_OF_EXECUTE_INPUT_PARAMS  ein;  memset(&ein,0,sizeof(ein));
    NV_OF_EXECUTE_OUTPUT_PARAMS eout; memset(&eout,0,sizeof(eout));
    ein.inputFrame = inBuf; ein.referenceFrame = refBuf; ein.disableTemporalHints = NV_OF_TRUE;
    eout.outputBuffer = outBuf;
    CK_OF(of.nvOFExecute(hOF, &ein, &eout));
    CK_RT(cudaDeviceSynchronize());

    // Download flow grid (S10.5 int16 x,y per cell) and take median of flowx.
    NV_OF_CUDA_BUFFER_STRIDE_INFO osi; memset(&osi,0,sizeof(osi));
    of.nvOFGPUBufferGetStrideInfo(outBuf, &osi);
    CUdeviceptr odptr = of.nvOFGPUBufferGetCUdeviceptr(outBuf);
    size_t rowBytes = (size_t)gw * sizeof(NV_OF_FLOW_VECTOR);
    std::vector<NV_OF_FLOW_VECTOR> grid((size_t)gw*gh);
    CUDA_MEMCPY2D d; memset(&d,0,sizeof(d));
    d.srcMemoryType=CU_MEMORYTYPE_DEVICE; d.srcDevice=odptr; d.srcPitch=osi.strideInfo[0].strideXInBytes;
    d.dstMemoryType=CU_MEMORYTYPE_HOST; d.dstHost=grid.data(); d.dstPitch=rowBytes;
    d.WidthInBytes=rowBytes; d.Height=gh;
    CK_CU(cuMemcpy2D(&d));

    std::vector<float> fx; fx.reserve(grid.size());
    for (auto& v : grid) fx.push_back(v.flowx / 32.0f);  // S10.5 -> px
    std::sort(fx.begin(), fx.end());
    float medx = fx[fx.size()/2];
    printf("grid %dx%d  median flowx = %.2f px (expected ~ %+d for a +%d px image shift)\n",
           gw, gh, medx, -SHIFT, SHIFT);

    of.nvOFDestroyGPUBufferCuda(inBuf);
    of.nvOFDestroyGPUBufferCuda(refBuf);
    of.nvOFDestroyGPUBufferCuda(outBuf);
    of.nvOFDestroy(hOF);

    // Accept either sign convention; magnitude must be within 2px of SHIFT.
    if (std::fabs(std::fabs(medx) - SHIFT) <= 2.0f) {
        printf("OK: NVOF dense flow recovered the synthetic shift.\n");
        return 0;
    }
    fprintf(stderr, "FAIL: |median flowx|=%.2f not within 2px of %d\n", std::fabs(medx), SHIFT);
    return 1;
}
