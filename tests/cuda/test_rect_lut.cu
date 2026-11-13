// Source-code LUT fused into existing samplers vs pre-transforming the source.
// Strict byte parity across plain/affine, CUDA pointers/arrays, and all filters.
// nvcc -std=c++17 -O2 tests/cuda/test_rect_lut.cu -lcuda -o /tmp/test_rect_lut
#include <cuda.h>
#include <fstream>
#include <iostream>
#include "../../src/nodes/hwaccel/cuda_rect_scale.cu"
#include "../../src/nodes/hwaccel/cuda_rect_sampler.h"
#include "rect_test_harness.cuh"

int main(int argc, char **argv) {
    CUcontext context = nullptr;
    try {
        check(cuInit(0), "init");
        check(cuDevicePrimaryCtxRetain(&context, 0), "context");
        check(cuCtxSetCurrent(context), "set context");
        unsigned char lut[3][256];
        for (int p = 0; p < 3; ++p) for (int i = 0; i < 256; ++i)
            lut[p][i] = static_cast<unsigned char>((i * (p + 1) + p * 37) % 256);
        if (argc > 1) {
            std::ifstream f(argv[1], std::ios::binary);
            if (!f.read(reinterpret_cast<char *>(lut), sizeof(lut))) throw std::runtime_error("need 768 LUT bytes");
        }
        int cases = 0;
        for (const auto &size : std::vector<std::pair<int,int>>{{12,22}, {240,144}}) {
            std::mt19937 r1(42), r2(42);
            Frame source(NV12, size.first, size.second, r1), washed(NV12, size.first, size.second, r2);
            for (int p = 0; p < 2; ++p) {
                for (int y = 0; y < washed.rows(p); ++y) for (int x = 0; x < washed.width; ++x) {
                    auto &v = washed.host[p][size_t(y) * washed.pitch + x];
                    v = lut[p ? 1 + x % 2 : 0][v];
                }
                check(cuMemcpyHtoD(washed.linear[p], washed.host[p].data(), washed.host[p].size()), "upload transformed reference");
            }
            for (int filter : {AVP_RECT_FILTER_BILINEAR, AVP_RECT_FILTER_BICUBIC, AVP_RECT_FILTER_BICUBIC_A0,
                               AVP_RECT_FILTER_MULTISAMPLE4, AVP_RECT_FILTER_MULTISAMPLE8, AVP_RECT_FILTER_AREA})
            for (bool arrays : {false, true}) for (bool affine : {false, true}) {
                std::cerr << "case " << cases << " source=" << source.width << 'x' << source.height
                          << " filter=" << filter << " array=" << arrays << " affine=" << affine << '\n';
                const Draw d{&source, 0,0, source.width,source.height, 0,0,1080,1920,filter,-.5f};
                auto actual = entry(d, NV12, arrays);
                Draw ref = d; ref.source = &washed;
                auto expected = entry(ref, NV12, false);
                actual.yuv_lut = 1;
                std::memcpy(actual.lut, lut, sizeof(lut));
                if (affine) for (auto *e : {&actual, &expected}) {
                    e->affine = 1;
                    for (int p = 0; p < 2; ++p) {
                        const float sx = float(e->sw[p]) / e->dw[p], sy = float(e->sh[p]) / e->dh[p];
                        e->m[p][0] = sx; e->m[p][4] = sy;
                        e->edge[p][2] = float(e->sw[p]); e->edge[p][3] = float(e->sh[p]);
                        e->aa[p][0] = 1/sx; e->aa[p][1] = 1/sy;
                        e->footprint[p][0] = sx; e->footprint[p][1] = sy;
                    }
                }
                Canvas a(NV12,1080,1920), b(NV12,1080,1920);
                const Kernel kernel = affine ? (arrays ? composite_planes_affine : composite_planes_yuv_affine)
                                             : composite_planes_filter;
                a.draw(kernel, {actual}); b.draw(kernel, {expected});
                if (a.pixels != b.pixels) throw std::runtime_error("LUT/resample pixel mismatch, case " + std::to_string(cases));
                ++cases;
            }
        }
        check(cuDevicePrimaryCtxRelease(0), "release context");
        std::cout << "ALL PASS: " << cases << " exact LUT/resample comparisons\n";
    } catch (const std::exception &error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        if (context) cuDevicePrimaryCtxRelease(0);
        return 1;
    }
}
