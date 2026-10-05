// NVIDIA integration test of the rect kernel's layer filters (bicubic, multisample 4 and 8) and
// of sources deeper than the canvas (P010 and P210 on NV12).
// Every sample a filtered or demoted layer draws is compared with a CPU reference of the same
// formulas, for 2x enlarging, 3:1 and 1.5:1 shrinking, crops, rects partly off the canvas,
// NV12/P010/P210 and the depth promotions and demotions; linear buffers and CUarray planes must
// agree byte for byte. A 10-bit frame copied 1:1 onto NV12 must be min(255, (v + 2) >> 2) of
// every code. The *_filter entries are also drawn against the existing entries for tables
// without a filter, and each filter is timed on 1920x1080 -> 640x360 and 1280x720 -> 1920x1080.
//   nvcc -std=c++17 -O2 tests/cuda/test_rect_filters.cu -lcuda -o /tmp/test_rect_filters
#include <cuda.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
#include "../../src/nodes/hwaccel/cuda_rect_scale.cu"
#include "../../src/nodes/hwaccel/cuda_rect_sampler.h"

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

// The unrounded value `layer` gives lane `c` of canvas lane group (X, Y) of `plane`; false
// outside the layer's rect.
static bool reference(const AvpRectLayer &layer, const Frame &source, const Format &canvas, int plane, int X, int Y, int c,
                      double &value) {
    const int ox = X - layer.dx[plane], oy = Y - layer.dy[plane];
    if (ox < 0 || oy < 0 || ox >= layer.dw[plane] || oy >= layer.dh[plane]) return false;
    const PlaneView view{source, plane, layer.sx[plane], layer.sy[plane], layer.sw[plane], layer.sh[plane]};
    const float fx = sourceCoord(ox, view.sw, layer.dw[plane]), fy = sourceCoord(oy, view.sh, layer.dh[plane]);
    const float xs = float(view.sw) / layer.dw[plane], ys = float(view.sh) / layer.dh[plane];
    switch (layer.filter) {
    case AVP_RECT_FILTER_BICUBIC: value = bicubic(view, fx, fy, layer.filter_param, c); break;
    case AVP_RECT_FILTER_BICUBIC_A0: value = bicubic(view, fx, fy, 0., c); break;   // the same cubic, all 16 taps
    case AVP_RECT_FILTER_MULTISAMPLE4: value = multisample(view, fx, fy, xs, ys, 4, c); break;
    case AVP_RECT_FILTER_MULTISAMPLE8: value = multisample(view, fx, fy, xs, ys, 8, c); break;
    default: value = bilinear(view, fx, fy, c);
    }
    if (layer.kind != AVP_RECT_KIND_YUV) value *= layer.mul;   // promoted or demoted codes
    value = std::min(std::max(value, 0.), double((256 << (canvas.depth - 8)) - 1));
    return true;
}

struct Tally {
    size_t samples = 0, off_tie = 0;   // off_tie: not the code the reference rounds to (within the slack)
    double worst = 0;                  // largest distance between a drawn code and the reference value
    size_t limited = 0;                // samples the cubic took beyond the code range
};

// A canvas holding one layer against the reference, sample by sample.
static void compare(const Canvas &out, const AvpRectLayer &layer, const Frame &source, const std::string &what, Tally &tally) {
    const int max_code = (256 << (out.format.depth - 8)) - 1;
    for (int p = 0; p < 2; ++p)
        for (int y = 0; y < (p ? out.uv_rows : out.height); ++y)
            for (int x = 0; x < (p ? out.width / 2 : out.width); ++x)
                for (int c = 0; c < (p ? 2 : 1); ++c) {
                    const int got = out.code(p, x, y, c);
                    double want;
                    if (!reference(layer, source, out.format, p, x, y, c, want)) {
                        if (got != out.clearCode(p))
                            throw std::runtime_error(what + ": drew outside the layer at plane " + std::to_string(p) + " " +
                                                     std::to_string(x) + "," + std::to_string(y));
                        continue;
                    }
                    ++tally.samples;
                    const double distance = std::abs(got - want);
                    tally.worst = std::max(tally.worst, distance);
                    if (distance > 0.5 + kTieSlack)
                        throw std::runtime_error(what + ": plane " + std::to_string(p) + " " + std::to_string(x) + "," +
                                                 std::to_string(y) + " lane " + std::to_string(c) + " is " + std::to_string(got) +
                                                 ", reference " + std::to_string(want));
                    if (got != int(want + 0.5)) ++tally.off_tie;
                    if (want == 0 || want == max_code) ++tally.limited;
                }
}

struct Geometry { const char *name; int cx, cy, cw, ch, dx, dy, dw, dh; };
struct Filter { const char *name; int code; float param; };

// Sources are 240x144, the canvas 512x288.
static const Geometry geometries[] = {
    {"2x enlarge", 0, 0, 240, 144, 16, 0, 480, 288},
    {"3:1 shrink", 0, 0, 240, 144, 100, 40, 80, 48},
    {"1.5:1 shrink", 0, 0, 240, 144, 20, 20, 160, 96},
    {"crop, 2x enlarge", 36, 20, 120, 72, 8, 4, 240, 144},
    {"crop, 3:1 shrink, off the top left", 36, 20, 180, 108, -20, -12, 60, 36},
    {"2x enlarge, off the bottom right", 0, 0, 240, 144, 300, 200, 480, 288},
    {"wider and flatter", 0, 0, 240, 144, 0, 100, 400, 48},
};
static const Filter filters[] = {
    {"bilinear", AVP_RECT_FILTER_BILINEAR, 0.f},
    {"bicubic A=0", AVP_RECT_FILTER_BICUBIC, 0.f},
    {"bicubic A=0 2x2", AVP_RECT_FILTER_BICUBIC_A0, 0.f},
    {"bicubic A=-0.5", AVP_RECT_FILTER_BICUBIC, -0.5f},
    {"bicubic A=-0.75", AVP_RECT_FILTER_BICUBIC, -0.75f},
    {"multisample 4", AVP_RECT_FILTER_MULTISAMPLE4, 0.f},
    {"multisample 8", AVP_RECT_FILTER_MULTISAMPLE8, 0.f},
};

// The entry CudaRectDraw::draw launches for a table of YUV layers holding this filter: the lean
// pair has the filters the scaler's defaults use, the full entry also the 4x4 cubic and 8 samples.
static Kernel filterKernel(const Filter &filter, bool arrays) {
    if (filter.code == AVP_RECT_FILTER_BICUBIC || filter.code == AVP_RECT_FILTER_MULTISAMPLE8) return composite_planes_filter;
    return arrays ? composite_planes_yuv_array_filter : composite_planes_yuv_filter;
}

static void verifyFilters(const Frame &source, const Format &canvas_format) {
    for (const Filter &filter : filters) {
        Tally tally;
        for (const Geometry &g : geometries) {
            const Draw draw{&source, g.cx, g.cy, g.cw, g.ch, g.dx, g.dy, g.dw, g.dh, filter.code, filter.param};
            const std::string what = std::string(source.format.name) + " -> " + canvas_format.name + ", " + filter.name + ", " + g.name;
            Canvas linear(canvas_format, 512, 288), arrays(canvas_format, 512, 288);
            const AvpRectLayer layer = entry(draw, canvas_format, false);
            linear.draw(filterKernel(filter, false), {layer});
            compare(linear, layer, source, what, tally);
            arrays.draw(filterKernel(filter, true), {entry(draw, canvas_format, true)});
            if (arrays.pixels != linear.pixels) throw std::runtime_error(what + ": array and linear sources differ");
        }
        std::printf("PASS %s -> %s, %-16s %zu samples in %zu geometries: %zu on the other side of a rounding tie, "
                    "largest distance from the reference %.4f codes, %zu at a code limit; arrays identical\n",
                    source.format.name, canvas_format.name, filter.name, tally.samples, std::size(geometries),
                    tally.off_tie, tally.worst, tally.limited);
    }
}

// The *_filter entries must draw layers without a filter, RGB(A) keys and key fades exactly as
// the entries they stand in for, and a filtered or demoted layer must not disturb its
// neighbours. `left_source` may be deeper than the canvas; `right_source` is not.
static void verifyEntries(const Frame &left_source, const Frame &right_source, const Format &canvas_format,
                          CUdeviceptr key_pixels) {
    const std::string names = std::string(left_source.format.name) + " and " + right_source.format.name + " -> " +
                              canvas_format.name;
    const bool demoted = left_source.format.depth > canvas_format.depth;
    auto key = [&](float opacity) {
        AvpRectLayer out{};
        out.kind = AVP_RECT_KIND_RGBA;
        out.src[0] = key_pixels;
        out.src_pitch[0] = 32 * 4;
        out.step = 4;
        out.r_off = 0; out.g_off = 1; out.b_off = 2; out.a_off = 3;
        out.sw[0] = out.sh[0] = 32;
        out.mul = opacity;
        for (int p = 0; p < 2; ++p) {   // over the right-hand layer only
            out.dx[p] = 300 >> p; out.dy[p] = 40 >> (p ? canvas_format.sub_y : 0);
            out.dw[p] = 120 >> p; out.dh[p] = 100 >> (p ? canvas_format.sub_y : 0);
        }
        return out;
    };
    auto same = [&](Kernel filter_kernel, Kernel plain_kernel, const std::vector<AvpRectLayer> &layers, const char *what) {
        Canvas a(canvas_format, 512, 288), b(canvas_format, 512, 288);
        a.draw(filter_kernel, layers);
        b.draw(plain_kernel, layers);
        if (a.pixels != b.pixels)
            throw std::runtime_error(names + ": " + what + " differs from the entry without filters");
    };
    const Draw left{&left_source, 0, 0, 240, 144, 0, 0, 256, 288}, right{&right_source, 36, 20, 180, 108, 256, 0, 256, 200};
    for (bool arrays : {false, true}) {
        // The entries without filters cannot draw a demoted layer, so they get the right one alone;
        // only that half is compared with them.
        std::vector<AvpRectLayer> faded = table(demoted ? std::vector<Draw>{right} : std::vector<Draw>{left, right},
                                                canvas_format, arrays);
        if (!demoted) {
            const std::vector<AvpRectLayer> yuv = faded;
            same(arrays ? composite_planes_yuv_array_filter : composite_planes_yuv_filter,
                 arrays ? composite_planes_yuv_array : composite_planes_yuv, yuv, "yuv filter entry");
            auto keyed = yuv;
            keyed.push_back(key(1.f));
            same(composite_planes_filter, arrays ? composite_planes_yuv_array : composite_planes_yuv, yuv, "full filter entry, yuv only");
            same(composite_planes_filter, arrays ? composite_planes_array : composite_planes, keyed, "full filter entry with a key");
        }
        faded.push_back(key(0.4f));
        if (!demoted)
            same(composite_planes_filter, arrays ? composite_planes_opacity_array : composite_planes_opacity, faded,
                 "full filter entry with a faded key");

        // A filtered (or demoted) layer on the left: the right half, key included, must not change,
        // and the left half must be what the entry for YUV-only tables draws.
        for (const Filter &filter : filters) {
            if (filter.code == AVP_RECT_FILTER_BILINEAR && !demoted) continue;
            Draw filtered_left = left;
            filtered_left.filter = filter.code;
            filtered_left.param = filter.param;
            auto mixed = table({filtered_left, right}, canvas_format, arrays);
            Canvas lean(canvas_format, 512, 288), full(canvas_format, 512, 288), plain(canvas_format, 512, 288);
            lean.draw(filterKernel(filter, arrays), mixed);
            mixed.push_back(key(0.4f));
            full.draw(composite_planes_filter, mixed);
            plain.draw(arrays ? composite_planes_opacity_array : composite_planes_opacity, faded);
            for (int p = 0; p < 2; ++p)
                for (int y = 0; y < (p ? full.uv_rows : full.height); ++y)
                    for (int x = 0; x < (p ? 256 : 512); ++x)
                        for (int c = 0; c < (p ? 2 : 1); ++c) {
                            const bool is_left = x < (p ? 128 : 256);
                            if (full.code(p, x, y, c) != (is_left ? lean : plain).code(p, x, y, c))
                                throw std::runtime_error(names + ", " + filter.name + ": a filtered layer next to a key differs on the " +
                                                         (is_left ? "filtered" : "unfiltered") + " side");
                        }
        }
    }
    if (demoted)
        std::printf("PASS %s: a demoted layer under every filter leaves its neighbour and a faded key as the "
                    "plain entries draw them, linear and arrays\n", names.c_str());
    else
        std::printf("PASS %s: filter entries equal the plain entries without a filter (yuv, key, faded key, "
                    "arrays); a filtered layer leaves its neighbours and a key unchanged\n", names.c_str());
}

// A frame deeper than the canvas copied 1:1: every drawn code against the integer formula, with
// no float reference in between. Luma and 4:2:0 chroma are min(max, (v + r/2) / r) of the source
// code v, r = 2^(depth difference); chroma of a 4:2:2 source on a 4:2:0 canvas is that of the mean of
// the two source rows a canvas row covers, min(max, (a + b + r) / (2r)).
static void verifyDemoteCopy(const Frame &source, const Format &canvas_format) {
    const int ratio = 1 << (source.format.depth - canvas_format.depth), max_code = (256 << (canvas_format.depth - 8)) - 1;
    const bool halves_rows = source.format.sub_y < canvas_format.sub_y;
    const Draw copy{&source, 0, 0, source.width, source.height, 0, 0, source.width, source.height};
    std::vector<bool> seen(size_t(256) << (source.format.depth - 8));
    const struct { Kernel kernel; bool arrays; const char *name; } runs[] = {
        {composite_planes_yuv_filter, false, "linear"},
        {composite_planes_yuv_array_filter, true, "array"},
        {composite_planes_filter, false, "linear, full entry"},
        {composite_planes_filter, true, "array, full entry"},
    };
    for (const auto &run : runs) {
        Canvas out(canvas_format, source.width, source.height);
        out.draw(run.kernel, {entry(copy, canvas_format, run.arrays)});
        for (int p = 0; p < 2; ++p)
            for (int y = 0; y < (p ? out.uv_rows : out.height); ++y)
                for (int x = 0; x < (p ? out.width / 2 : out.width); ++x)
                    for (int c = 0; c < (p ? 2 : 1); ++c) {
                        int want;
                        if (p && halves_rows) {
                            want = (source.code(p, x, 2 * y, c) + source.code(p, x, 2 * y + 1, c) + ratio) / (2 * ratio);
                        } else {
                            const int v = source.code(p, x, y, c);
                            if (!p) seen[v] = true;
                            want = (v + ratio / 2) / ratio;
                        }
                        want = std::min(want, max_code);
                        if (out.code(p, x, y, c) != want)
                            throw std::runtime_error(std::string(source.format.name) + " -> " + canvas_format.name + " 1:1 " +
                                                     run.name + ": plane " + std::to_string(p) + " " + std::to_string(x) + "," +
                                                     std::to_string(y) + " is " + std::to_string(out.code(p, x, y, c)) +
                                                     ", want " + std::to_string(want));
                    }
    }
    if (std::find(seen.begin(), seen.end(), false) != seen.end())
        throw std::runtime_error("the 1:1 fixture does not hold every source code; the test is weaker than it says");
    std::printf("PASS %s -> %s 1:1: every sample equals min(%d, (v + %d) >> %d)%s, all %zu source codes present; "
                "linear and array sources, lean and full filter entries\n", source.format.name, canvas_format.name,
                max_code, ratio / 2, source.format.depth - canvas_format.depth,
                halves_rows ? " (chroma: of the mean of its two source rows)" : "", seen.size());
}

// Median and 90th percentile of 200 launches of one full-canvas layer (table already on the device).
static void timeLayer(const char *label, const Frame &source, const Format &canvas_format, int cw, int ch,
                      Kernel kernel, const Filter &filter, bool arrays) {
    Canvas canvas(canvas_format, cw, ch);
    canvas.upload({entry({&source, 0, 0, source.width, source.height, 0, 0, cw, ch, filter.code, filter.param},
                         canvas_format, arrays)});
    CUevent start, stop;
    check(cuEventCreate(&start, CU_EVENT_DEFAULT), "event");
    check(cuEventCreate(&stop, CU_EVENT_DEFAULT), "event");
    for (int i = 0; i < 20; ++i) canvas.launch(kernel, 1);
    check(cuCtxSynchronize(), "warm up");
    std::vector<float> ms;
    for (int i = 0; i < 200; ++i) {
        check(cuEventRecord(start, nullptr), "record");
        canvas.launch(kernel, 1);
        check(cuEventRecord(stop, nullptr), "record");
        check(cuEventSynchronize(stop), "wait");
        float elapsed = 0;
        check(cuEventElapsedTime(&elapsed, start, stop), "elapsed");
        ms.push_back(elapsed);
    }
    std::sort(ms.begin(), ms.end());
    std::printf("%-34s %-22s median %.3f ms  p90 %.3f ms\n", label, filter.name, ms[ms.size() / 2], ms[ms.size() * 9 / 10]);
    cuEventDestroy(start);
    cuEventDestroy(stop);
}

static void timings(std::mt19937 &rng) {
    static const Filter plain{"bilinear, plain entry", AVP_RECT_FILTER_BILINEAR, 0.f};
    {   // A 10-bit frame onto an 8-bit canvas, next to the same-depth copy the plain entry draws.
        Frame deep(P010, 1920, 1080, rng);
        for (bool arrays : {false, true}) {
            const std::string storage = arrays ? " array" : " linear";
            timeLayer(("1920x1080 p010 -> p010 1:1" + storage).c_str(), deep, P010, 1920, 1080,
                      arrays ? composite_planes_yuv_array : composite_planes_yuv, plain, arrays);
            timeLayer(("1920x1080 p010 -> nv12 1:1" + storage).c_str(), deep, NV12, 1920, 1080,
                      filterKernel(filters[0], arrays), filters[0], arrays);
            for (const Filter &filter : filters)
                if (filter.code == AVP_RECT_FILTER_BILINEAR || filter.code == AVP_RECT_FILTER_MULTISAMPLE4)
                    timeLayer(("1920x1080 p010 -> nv12 640x360" + storage).c_str(), deep, NV12, 640, 360,
                              filterKernel(filter, arrays), filter, arrays);
        }
    }
    struct Case { const char *name; int sw, sh, cw, ch; };
    for (const Case &c : {Case{"1920x1080 -> 640x360", 1920, 1080, 640, 360}, Case{"1280x720 -> 1920x1080", 1280, 720, 1920, 1080}})
        for (const Format *format : {&NV12, &P010}) {
            Frame source(*format, c.sw, c.sh, rng);
            for (bool arrays : {false, true}) {
                const std::string label = std::string(c.name) + " " + format->name + (arrays ? " array" : " linear");
                timeLayer(label.c_str(), source, *format, c.cw, c.ch,
                          arrays ? composite_planes_yuv_array : composite_planes_yuv, plain, arrays);
                for (const Filter &filter : filters)
                    timeLayer(label.c_str(), source, *format, c.cw, c.ch, filterKernel(filter, arrays), filter, arrays);
            }
        }
}

int main(int argc, char **argv) {
    const bool timing_only = argc > 1 && std::string(argv[1]) == "--timings";
    CUcontext context = nullptr;
    try {
        check(cuInit(0), "initialize CUDA");
        check(cuDevicePrimaryCtxRetain(&context, 0), "retain primary context");
        check(cuCtxSetCurrent(context), "set context");
        std::mt19937 rng(20261005);
        if (!timing_only) {
            CUdeviceptr key = 0;
            std::vector<unsigned char> rgba(32 * 32 * 4);
            for (auto &value : rgba) value = static_cast<unsigned char>(rng());
            check(cuMemAlloc(&key, rgba.size()), "allocate RGBA key");
            check(cuMemcpyHtoD(key, rgba.data(), rgba.size()), "upload RGBA key");
            {
                Frame nv12(NV12, 240, 144, rng), p010(P010, 240, 144, rng), p210(P210, 240, 144, rng);
                const std::pair<const Frame *, const Format *> pairs[] = {
                    {&nv12, &NV12}, {&p010, &P010}, {&p210, &P210}, {&nv12, &P010}, {&nv12, &P210}, {&p010, &P210}};
                const Frame *deeper[] = {&p010, &p210};   // on an NV12 canvas
                for (const Frame *source : deeper) verifyDemoteCopy(*source, NV12);
                for (const auto &[source, canvas_format] : pairs) verifyFilters(*source, *canvas_format);
                for (const Frame *source : deeper) verifyFilters(*source, NV12);
                for (const auto &[source, canvas_format] : pairs) verifyEntries(*source, *source, *canvas_format, key);
                for (const Frame *source : deeper) verifyEntries(*source, nv12, NV12, key);
            }
            check(cuMemFree(key), "release RGBA key");
        }
        timings(rng);
        check(cuDevicePrimaryCtxRelease(0), "release context");
        std::cout << (timing_only ? "TIMINGS DONE\n" : "ALL PASS\n");
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        if (context) cuDevicePrimaryCtxRelease(0);
        return 1;
    }
}
