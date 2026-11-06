#pragma once
// Shared by the rect kernel's NVIDIA tests (test_rect_filters.cu, test_rect_affine.cu): semiplanar
// noise frames held as host bytes, linear device buffers and CUarray plane textures; a canvas the
// kernel entries draw into; and the CPU reference samplers (bilinear, cubic, multisample) that
// reproduce the kernel's tap positions in float. Include after the kernel source and
// cuda_rect_sampler.h.
#include <cuda.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

static void check(CUresult result, const char *what) {
    if (result == CUDA_SUCCESS) return;
    const char *message = nullptr;
    cuGetErrorString(result, &message);
    throw std::runtime_error(std::string(what) + ": " + (message ? message : "CUDA error"));
}

struct Format {
    const char *name;
    int bytes, shift, depth, sub_y;
    CUarray_format array_format;
};
static const Format NV12{"nv12", 1, 0, 8, 1, CU_AD_FORMAT_NV12}, P010{"p010", 2, 6, 10, 1, CU_AD_FORMAT_P016},
                    P210{"p210", 2, 6, 10, 0, CU_AD_FORMAT_P216};

// Logical code of the sample at byte offset `at`.
static int decode(const unsigned char *at, const Format &format) {
    if (format.bytes == 1) return *at;
    uint16_t word;
    std::memcpy(&word, at, 2);
    return word >> format.shift;
}

// A semiplanar frame of noise held three ways: host bytes for the reference, linear device
// buffers (padded pitch) and a multi-planar CUarray behind the production plane textures.
struct Frame {
    const Format &format;
    int width, height, pitch;
    std::array<std::vector<unsigned char>, 2> host;
    std::array<CUdeviceptr, 2> linear{};
    std::array<CUtexObject, 2> texture{};
    CUarray array = nullptr;

    int rows(int plane) const { return plane ? (height + (1 << format.sub_y) - 1) >> format.sub_y : height; }
    // Lane `c` of lane group (x, y): a luma sample, or one of a chroma pair.
    int code(int plane, int x, int y, int c) const {
        return decode(&host[plane][size_t(y) * pitch + (x * (plane ? 2 : 1) + c) * format.bytes], format);
    }

    Frame(const Format &fmt, int w, int h, std::mt19937 &rng)
        : format(fmt), width(w), height(h), pitch((w * fmt.bytes + 63) / 64 * 64 + 64) {
        CUDA_ARRAY3D_DESCRIPTOR desc{};
        desc.Width = w;
        desc.Height = h;
        desc.Format = fmt.array_format;
        desc.NumChannels = 3;
        desc.Flags = CUDA_ARRAY3D_SURFACE_LDST | CUDA_ARRAY3D_VIDEO_ENCODE_DECODE;
        check(cuArray3DCreate(&array, &desc), "create multi-planar array");
        for (unsigned p = 0; p < 2; ++p) {
            host[p].assign(size_t(pitch) * rows(p), 0xee);   // the padding never holds a valid 10-bit word
            for (int y = 0; y < rows(p); ++y)
                for (int x = 0; x < w; ++x) {
                    unsigned char *at = &host[p][size_t(y) * pitch + x * fmt.bytes];
                    if (fmt.bytes == 1) {
                        *at = static_cast<unsigned char>(rng());
                    } else {
                        const uint16_t word = uint16_t((rng() % 1024) << fmt.shift);
                        std::memcpy(at, &word, 2);
                    }
                }
            check(cuMemAlloc(&linear[p], host[p].size()), "allocate linear plane");
            check(cuMemcpyHtoD(linear[p], host[p].data(), host[p].size()), "upload linear plane");
            CUarray plane = nullptr;
            check(cuArrayGetPlane(&plane, array, p), "get plane");
            CUDA_MEMCPY2D copy{};
            copy.srcMemoryType = CU_MEMORYTYPE_HOST;
            copy.srcHost = host[p].data();
            copy.srcPitch = pitch;
            copy.dstMemoryType = CU_MEMORYTYPE_ARRAY;
            copy.dstArray = plane;
            copy.WidthInBytes = size_t(w) * fmt.bytes;
            copy.Height = rows(p);
            check(cuMemcpy2D(&copy), "upload array plane");
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

// A layer in luma pixels (all even, as the host aligns them) and its table entry, built the way
// CudaRectDraw::fillTableEntry builds it.
struct Draw {
    const Frame *source;
    int cx, cy, cw, ch;   // crop
    int dx, dy, dw, dh;   // destination on the canvas
    int filter = AVP_RECT_FILTER_BILINEAR;
    float param = 0.f;
};

static AvpRectLayer entry(const Draw &draw, const Format &canvas, bool arrays) {
    const Frame &source = *draw.source;
    AvpRectLayer out{};
    out.kind = &source.format == &canvas ? AVP_RECT_KIND_YUV
             : source.format.depth > canvas.depth ? AVP_RECT_KIND_DEMOTE : AVP_RECT_KIND_PROMOTE;
    out.yuv_texture = arrays;
    out.src_bytes = source.format.bytes;
    out.src_shift = source.format.shift;
    out.mul = std::ldexp(1.f, canvas.depth - source.format.depth);
    out.filter = draw.filter;
    out.filter_param = draw.param;
    for (int p = 0; p < 2; ++p) {
        const int src_y = p ? source.format.sub_y : 0, dst_y = p ? canvas.sub_y : 0;
        out.src[p] = arrays ? source.texture[p] : source.linear[p];
        out.src_pitch[p] = arrays ? 0 : source.pitch;
        out.sx[p] = draw.cx >> p; out.sy[p] = draw.cy >> src_y;
        out.sw[p] = draw.cw >> p; out.sh[p] = draw.ch >> src_y;
        out.dx[p] = draw.dx >> p; out.dy[p] = draw.dy >> dst_y;
        out.dw[p] = draw.dw >> p; out.dh[p] = draw.dh >> dst_y;
    }
    return out;
}

static std::vector<AvpRectLayer> table(const std::vector<Draw> &draws, const Format &canvas, bool arrays) {
    std::vector<AvpRectLayer> out;
    for (const Draw &draw : draws) out.push_back(entry(draw, canvas, arrays));
    return out;
}

using Kernel = decltype(&composite_planes_yuv);

// A canvas the kernels draw into; its padded pitch must come back untouched.
struct Canvas {
    const Format &format;
    int width, height, pitch, uv_rows;
    size_t y_size, uv_size;
    CUdeviceptr device = 0, device_table = 0;
    std::vector<unsigned char> pixels;
    static constexpr int kMaxLayers = 8;
    static constexpr unsigned char kUntouched = 0xa5;

    Canvas(const Format &fmt, int w, int h)
        : format(fmt), width(w), height(h), pitch((w * fmt.bytes + 63) / 64 * 64 + 64), uv_rows(h >> fmt.sub_y),
          y_size(size_t(pitch) * h), uv_size(size_t(pitch) * uv_rows), pixels(y_size + uv_size) {
        check(cuMemAlloc(&device, pixels.size()), "allocate canvas");
        check(cuMemAlloc(&device_table, kMaxLayers * sizeof(AvpRectLayer)), "allocate table");
        check(cuMemsetD8(device, kUntouched, pixels.size()), "fill canvas");
    }
    ~Canvas() { cuMemFree(device_table); cuMemFree(device); }
    Canvas(const Canvas &) = delete;
    Canvas &operator=(const Canvas &) = delete;

    void upload(const std::vector<AvpRectLayer> &layers) {
        if (layers.size() > kMaxLayers) throw std::runtime_error("test table too long");
        check(cuMemcpyHtoD(device_table, layers.data(), layers.size() * sizeof(AvpRectLayer)), "upload table");
    }
    // One launch, as CudaRectDraw::draw issues it; not synchronized.
    void launch(Kernel kernel, int layers) {
        auto *y = reinterpret_cast<unsigned char *>(device), *uv = y + y_size;
        const dim3 grid((width + 32 * AVP_RECT_PX - 1) / (32 * AVP_RECT_PX), (height + 7) / 8, 2), block(32, 8);
        const int scale = 1 << (format.depth - 8);
        kernel<<<grid, block, (size_t(layers) + 31) / 32 * sizeof(unsigned)>>>(
            reinterpret_cast<const AvpRectLayer *>(device_table), layers, y, pitch, uv, pitch,
            width, height, width / 2, uv_rows, format.bytes, format.shift, scale, 1, format.sub_y,
            16 * scale, 128 * scale, 0, 100.f, 1000.f);
    }
    void draw(Kernel kernel, const std::vector<AvpRectLayer> &layers) {
        upload(layers);
        launch(kernel, int(layers.size()));
        check(cuCtxSynchronize(), "render");
        check(cuMemcpyDtoH(pixels.data(), device, pixels.size()), "download canvas");
        for (int p = 0; p < 2; ++p)
            for (int y = 0; y < (p ? uv_rows : height); ++y)
                for (int byte = width * format.bytes; byte < pitch; ++byte)
                    if (pixels[(p ? y_size : 0) + size_t(y) * pitch + byte] != kUntouched)
                        throw std::runtime_error("the kernel wrote beyond the canvas width");
    }
    int code(int plane, int x, int y, int c) const {
        return decode(&pixels[(plane ? y_size : 0) + size_t(y) * pitch + (x * (plane ? 2 : 1) + c) * format.bytes], format);
    }
    int clearCode(int plane) const { return (plane ? 128 : 16) << (format.depth - 8); }
};

// ---------------------------------------------------------------------------
// CPU reference. Tap positions are computed in float exactly as the kernel computes them (a
// position one float step away moves a noise sample by more than rounding hides); the taps are
// then combined in double. The kernel combines them in float, where nvcc fuses some products
// and sums, so a result within kTieSlack of a rounding tie may land on either code.
// ---------------------------------------------------------------------------
static constexpr double kTieSlack = 0.01;

static float sourceCoord(int o, int sn, int dn) { return (o + 0.5f) * sn / dn - 0.5f; }

struct PlaneView {   // one plane of a layer's source rect
    const Frame &frame;
    int plane, sx, sy, sw, sh;
    double tap(int x, int y, int c) const {
        return frame.code(plane, sx + std::max(0, std::min(x, sw - 1)), sy + std::max(0, std::min(y, sh - 1)), c);
    }
};

static double bilinear(const PlaneView &v, float fx, float fy, int c) {
    const int ix = int(std::floor(fx)), iy = int(std::floor(fy));
    const double tx = double(fx) - ix, ty = double(fy) - iy;
    const double top = v.tap(ix, iy, c) + tx * (v.tap(ix + 1, iy, c) - v.tap(ix, iy, c));
    const double bottom = v.tap(ix, iy + 1, c) + tx * (v.tap(ix + 1, iy + 1, c) - v.tap(ix, iy + 1, c));
    return top + ty * (bottom - top);
}

static std::array<double, 4> cubicWeights(double x, double A) {
    std::array<double, 4> w;
    w[0] = ((A * (x + 1) - 5 * A) * (x + 1) + 8 * A) * (x + 1) - 4 * A;
    w[1] = ((A + 2) * x - (A + 3)) * x * x + 1;
    w[2] = ((A + 2) * (1 - x) - (A + 3)) * (1 - x) * (1 - x) + 1;
    w[3] = 1 - w[0] - w[1] - w[2];
    return w;
}

static double bicubic(const PlaneView &v, float fx, float fy, double A, int c) {
    const int ix = int(std::floor(fx)), iy = int(std::floor(fy));
    const auto wx = cubicWeights(double(fx) - ix, A), wy = cubicWeights(double(fy) - iy, A);
    double value = 0;
    for (int r = 0; r < 4; ++r) {
        double line = 0;
        for (int k = 0; k < 4; ++k) line += v.tap(ix - 1 + k, iy - 1 + r, c) * wx[k];
        value += line * wy[r];
    }
    return value;
}

static double multisample(const PlaneView &v, float fx, float fy, float xs, float ys, int samples, int c) {
    float dx[8], dy[8];
    if (samples == 4) {
        const float wx = std::min(std::max(0.5f * (xs - 1.f), 0.f), 1.f), wy = std::min(std::max(0.5f * (ys - 1.f), 0.f), 1.f);
        const float gx = wx / (0.5f + wx), gy = wy / (0.5f + wy);
        for (int k = 0; k < 4; ++k) { dx[k] = k & 1 ? gx : -gx; dy[k] = k & 2 ? gy : -gy; }
    } else {
        // Direct3D standard 8-sample pattern, sixteenths of the area one output sample covers.
        static const int pattern[8][2] = {{1, -3}, {-1, 3}, {5, 1}, {-3, -5}, {-5, 5}, {-7, -1}, {3, 7}, {7, -7}};
        for (int k = 0; k < 8; ++k) {
            // volatile: the product is rounded to float before it is added to the position.
            volatile float ox = float(pattern[k][0]) * (xs * 0.0625f), oy = float(pattern[k][1]) * (ys * 0.0625f);
            dx[k] = ox; dy[k] = oy;
        }
    }
    double sum = 0;
    for (int k = 0; k < samples; ++k) sum += bilinear(v, fx + dx[k], fy + dy[k], c);
    return sum / samples;
}
