// CUDA integration: real multi-planar CUarrays versus linear device buffers,
// through the production compositor kernels. No decoder or video encode needed.
// Tests exact pixels for crop/resize, overlapping layers, depth/chroma promotion,
// and mixed storage. CPU transfers only create fixtures and inspect results.
// nvcc -std=c++17 -O2 tests/cuda/nvdec/compositor_arrays.cu -lcuda -o /tmp/compositor_arrays
#include <cuda.h>
#include <algorithm>
#include <array>
#include <cstring>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>
#include "../../../src/nodes/hwaccel/cuda_rect_scale.cu"
#include "../../../src/nodes/hwaccel/cuda_rect_sampler.h"

static void check(CUresult result, const char *what) {
    if (result == CUDA_SUCCESS) return;
    const char *message = nullptr;
    cuGetErrorString(result, &message);
    throw std::runtime_error(std::string(what) + ": " + (message ? message : "CUDA error"));
}

struct Format {
    const char *name;
    int bytes, shift, sub_y;
    CUarray_format array_format;
};
static const Format formats[] = {
    {"nv12", 1, 0, 1, CU_AD_FORMAT_NV12},
    {"p010", 2, 6, 1, CU_AD_FORMAT_P016},
    {"p210", 2, 6, 0, CU_AD_FORMAT_P216},
};

struct Frame {
    const Format &format;
    int width, height, pitch;
    std::array<CUdeviceptr, 2> linear{};
    std::array<CUtexObject, 2> texture{};
    CUarray array = nullptr;

    Frame(const Format &fmt, int w, int h, std::mt19937 &rng) : format(fmt), width(w), height(h), pitch(w * fmt.bytes) {
        CUDA_ARRAY3D_DESCRIPTOR desc{};
        desc.Width = w;
        desc.Height = h;
        desc.Format = fmt.array_format;
        desc.NumChannels = 3;
        desc.Flags = CUDA_ARRAY3D_SURFACE_LDST | CUDA_ARRAY3D_VIDEO_ENCODE_DECODE;
        check(cuArray3DCreate(&array, &desc), "create multi-planar array");
        for (unsigned p = 0; p < 2; ++p) {
            const int rows = p ? (h + (1 << fmt.sub_y) - 1) >> fmt.sub_y : h;
            std::vector<unsigned char> pixels(size_t(pitch) * rows);
            if (fmt.bytes == 1) {
                for (auto &value : pixels) value = static_cast<unsigned char>(rng());
            } else {
                for (size_t i = 0; i < pixels.size(); i += 2) {
                    const uint16_t value = uint16_t((rng() % 1024) << fmt.shift);
                    std::memcpy(pixels.data() + i, &value, sizeof(value));
                }
            }
            check(cuMemAlloc(&linear[p], pixels.size()), "allocate linear reference");
            check(cuMemcpyHtoD(linear[p], pixels.data(), pixels.size()), "upload linear reference");
            CUarray plane = nullptr;
            check(cuArrayGetPlane(&plane, array, p), "get plane");
            CUDA_MEMCPY2D copy{};
            copy.srcMemoryType = CU_MEMORYTYPE_HOST;
            copy.srcHost = pixels.data();
            copy.srcPitch = pitch;
            copy.dstMemoryType = CU_MEMORYTYPE_ARRAY;
            copy.dstArray = plane;
            copy.WidthInBytes = pitch;
            copy.Height = rows;
            check(cuMemcpy2D(&copy), "upload array fixture");
            CUDA_RESOURCE_DESC resource;
            CUDA_TEXTURE_DESC sampler;
            avp::mixer::rectTextureDesc(plane, resource, sampler);
            check(cuTexObjectCreate(&texture[p], &resource, &sampler, nullptr), "create plane texture");
        }
    }
    ~Frame() {
        for (auto handle : texture) if (handle) cuTexObjectDestroy(handle);
        for (auto pointer : linear) if (pointer) cuMemFree(pointer);
        if (array) cuArrayDestroy(array);
    }
    Frame(const Frame &) = delete;
    Frame &operator=(const Frame &) = delete;
};

static AvpRectLayer layer(const Frame &source, const Format &destination, int offset) {
    AvpRectLayer out{};
    out.kind = source.format.bytes != destination.bytes || source.format.sub_y != destination.sub_y
        ? AVP_RECT_KIND_PROMOTE : AVP_RECT_KIND_YUV;
    out.src_bytes = source.format.bytes;
    out.src_shift = source.format.shift;
    out.mul = source.format.bytes == destination.bytes ? 1.f : 4.f;
    for (int p = 0; p < 2; ++p) {
        const int src_y = p ? source.format.sub_y : 0, dst_y = p ? destination.sub_y : 0;
        out.src[p] = source.linear[p];
        out.src_pitch[p] = source.pitch;
        out.sx[p] = 6 >> p;
        out.sy[p] = 4 >> src_y;
        out.sw[p] = (source.width - 12) >> p;
        out.sh[p] = (source.height - 8) >> src_y;
        out.dx[p] = (-10 + offset) >> p;
        out.dy[p] = (6 + offset) >> dst_y;
        out.dw[p] = (114 - offset) >> p;
        out.dh[p] = (72 - offset) >> dst_y;
    }
    return out;
}

static std::vector<unsigned char> render(const Format &format, const std::vector<AvpRectLayer> &layers,
                                          bool arrays, int mode) {
    constexpr int width = 128, height = 96;
    const int pitch = width * format.bytes, uv_height = height >> format.sub_y;
    const size_t y_size = size_t(pitch) * height, uv_size = size_t(pitch) * uv_height;
    CUdeviceptr output = 0, table = 0;
    check(cuMemAlloc(&output, y_size + uv_size), "allocate canvas");
    check(cuMemAlloc(&table, layers.size() * sizeof(AvpRectLayer)), "allocate table");
    check(cuMemcpyHtoD(table, layers.data(), layers.size() * sizeof(AvpRectLayer)), "upload table");
    auto *y = reinterpret_cast<unsigned char *>(output), *uv = y + y_size;
    const dim3 grid(1, (height + 7) / 8, 2), block(32, 8);
    const size_t shared = ((layers.size() + 31) / 32) * sizeof(unsigned);
    const int scale = format.bytes == 1 ? 1 : 4;
#define DRAW_ARGS reinterpret_cast<const AvpRectLayer *>(table), int(layers.size()), y, pitch, uv, pitch, \
    width, height, width / 2, uv_height, format.bytes, format.shift, scale, 1, format.sub_y, \
    16 * scale, 128 * scale, 0, 100.f, 1000.f
    if (mode == 2) {
        if (arrays) composite_planes_opacity_array<<<grid, block, shared>>>(DRAW_ARGS);
        else composite_planes_opacity<<<grid, block, shared>>>(DRAW_ARGS);
    } else if (mode == 1) {
        if (arrays) composite_planes_array<<<grid, block, shared>>>(DRAW_ARGS);
        else composite_planes<<<grid, block, shared>>>(DRAW_ARGS);
    } else {
        if (arrays) composite_planes_yuv_array<<<grid, block, shared>>>(DRAW_ARGS);
        else composite_planes_yuv<<<grid, block, shared>>>(DRAW_ARGS);
    }
#undef DRAW_ARGS
    check(cuCtxSynchronize(), "render");
    std::vector<unsigned char> pixels(y_size + uv_size);
    check(cuMemcpyDtoH(pixels.data(), output, pixels.size()), "download result");
    cuMemFree(table);
    cuMemFree(output);
    return pixels;
}

static AvpRectLayer keyLayer(CUdeviceptr pixels, const Format &format, float opacity) {
    AvpRectLayer out{};
    out.kind = AVP_RECT_KIND_RGBA;
    out.src[0] = pixels;
    out.src_pitch[0] = 32 * 4;
    out.step = 4;
    out.r_off = 0; out.g_off = 1; out.b_off = 2; out.a_off = 3;
    out.sw[0] = out.sh[0] = 32;
    out.mul = opacity;
    for (int p = 0; p < 2; ++p) {
        out.dx[p] = 18 >> p; out.dy[p] = 10 >> (p ? format.sub_y : 0);
        out.dw[p] = 50 >> p; out.dh[p] = 42 >> (p ? format.sub_y : 0);
    }
    return out;
}

static void verify(const Frame &a, const Frame &b, const Format &destination, CUdeviceptr key) {
    for (int mode = 0; mode < 3; ++mode) {
        std::vector<AvpRectLayer> linear{layer(a, destination, 0), layer(b, destination, 22)};
        if (mode) linear.push_back(keyLayer(key, destination, mode == 2 ? .4f : 1.f));
        const auto reference = render(destination, linear, false, mode);
        for (unsigned mask : {1u, 2u, 3u}) {
            auto textured = linear;
            for (unsigned i = 0; i < 2; ++i) {
                if (!(mask & (1u << i))) continue;
                textured[i].yuv_texture = 1;
                for (unsigned p = 0; p < 2; ++p) textured[i].src[p] = (i ? b : a).texture[p];
            }
            const auto actual = render(destination, textured, true, mode);
            if (actual != reference) {
                const auto mismatch = std::mismatch(actual.begin(), actual.end(), reference.begin());
                throw std::runtime_error(std::string(a.format.name) + "/" + b.format.name + " -> " +
                    destination.name + " differs at byte " + std::to_string(mismatch.first - actual.begin()));
            }
        }
    }
    std::cout << a.format.name << '/' << b.format.name << " -> " << destination.name
              << ": exact parity, array/mixed storage, YUV/RGBA/faded RGBA\n";
}

int main() {
    CUcontext context = nullptr;
    try {
        check(cuInit(0), "initialize CUDA");
        check(cuDevicePrimaryCtxRetain(&context, 0), "retain primary context");
        check(cuCtxSetCurrent(context), "set context");
        std::mt19937 rng(541);
        CUdeviceptr key = 0;
        std::vector<unsigned char> rgba(32 * 32 * 4);
        for (auto &value : rgba) value = static_cast<unsigned char>(rng());
        check(cuMemAlloc(&key, rgba.size()), "allocate RGBA key");
        check(cuMemcpyHtoD(key, rgba.data(), rgba.size()), "upload RGBA key");
        {
            Frame nv12(formats[0], 80, 64, rng), p010(formats[1], 80, 64, rng), p210(formats[2], 80, 64, rng);
            verify(nv12, nv12, formats[0], key);
            verify(p010, p010, formats[1], key);
            verify(p210, p210, formats[2], key);
            verify(nv12, p010, formats[1], key);
            verify(nv12, p210, formats[2], key);
            verify(p010, p210, formats[2], key);
        }
        check(cuMemFree(key), "release RGBA key");
        check(cuDevicePrimaryCtxRelease(0), "release context");
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        if (context) cuDevicePrimaryCtxRelease(0);
        return 1;
    }
}
