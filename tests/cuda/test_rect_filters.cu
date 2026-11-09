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

#include "rect_test_harness.cuh"

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
    case AVP_RECT_FILTER_AREA: value = area(view, fx, fy, xs, ys, c); break;
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
    {"8:1 shrink", 0, 0, 240, 144, 10, 10, 30, 18},
};
static const Filter filters[] = {
    {"bilinear", AVP_RECT_FILTER_BILINEAR, 0.f},
    {"bicubic A=0", AVP_RECT_FILTER_BICUBIC, 0.f},
    {"bicubic A=0 2x2", AVP_RECT_FILTER_BICUBIC_A0, 0.f},
    {"bicubic A=-0.5", AVP_RECT_FILTER_BICUBIC, -0.5f},
    {"bicubic A=-0.75", AVP_RECT_FILTER_BICUBIC, -0.75f},
    {"multisample 4", AVP_RECT_FILTER_MULTISAMPLE4, 0.f},
    {"multisample 8", AVP_RECT_FILTER_MULTISAMPLE8, 0.f},
    {"area", AVP_RECT_FILTER_AREA, 0.f},
};

// The entry CudaRectDraw::draw launches for a table of YUV layers holding this filter: the lean
// pair has the filters the scaler's defaults use, the full entry also the 4x4 cubic and 8 samples.
static Kernel filterKernel(const Filter &filter, bool arrays) {
    if (filter.code == AVP_RECT_FILTER_BICUBIC || filter.code == AVP_RECT_FILTER_MULTISAMPLE8 || filter.code == AVP_RECT_FILTER_AREA)
        return composite_planes_filter;
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
