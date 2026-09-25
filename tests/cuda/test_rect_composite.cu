// NVIDIA integration test of the single-launch composite kernel against the
// per-layer kernels it replaced (legacy_rect_kernels.cuh): the same layer set
// drawn both ways must give identical canvases (NV12 and P210, scale/blit, NV12->P210 promotion,
// opaque RGB and blended RGBA, overlapping z-order, partial off-canvas rects).
// Then times both paths on a 1080x1920 sixteen-box grid.
//   nvcc -std=c++17 -O2 tests/cuda/test_rect_composite.cu -lcuda -o /tmp/test_rect_composite
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
#include <cuda.h>
#include "../../src/nodes/hwaccel/cuda_rect_scale.cu"
#include "../../src/nodes/hwaccel/cuda_rect_sampler.h"
#include "legacy_rect_kernels.cuh"

static void check(cudaError_t r, const char *what) {
    if (r != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(r));
}

struct Plane {
    unsigned char *dev = nullptr;
    int pitch = 0, rows = 0;
    std::vector<unsigned char> host;
    Plane() = default;
    Plane(int pitch_, int rows_, std::mt19937 *rng = nullptr) : pitch(pitch_), rows(rows_), host(size_t(pitch_) * rows_) {
        if (rng) for (auto &b : host) b = (unsigned char)(*rng)();
        check(cudaMalloc(&dev, host.size()), "cudaMalloc");
        check(cudaMemcpy(dev, host.data(), host.size(), cudaMemcpyHostToDevice), "upload");
    }
    ~Plane() { if (dev) cudaFree(dev); }
    Plane(const Plane &) = delete;
    Plane &operator=(const Plane &) = delete;
    std::vector<unsigned char> download() const {
        std::vector<unsigned char> out(host.size());
        check(cudaMemcpy(out.data(), dev, out.size(), cudaMemcpyDeviceToHost), "download");
        return out;
    }
};

// Semiplanar formats only: NV12 (8-bit 4:2:0), P010 (10-bit 4:2:0), P210 (10-bit 4:2:2).
struct Fmt {
    const char *name; int bytes; int shift; int depth; int sub_x; int sub_y; int planes; int lanes0;
};
static const Fmt NV12{"nv12", 1, 0, 8, 1, 1, 2, 1}, P010{"p010", 2, 6, 10, 1, 1, 2, 1}, P210{"p210", 2, 6, 10, 1, 0, 2, 1},
                 RGB0{"rgb0", 1, 0, 8, 0, 0, 1, 4};

// libavutil's lumaRectToPlaneRegion for semiplanar: plane 0 bytes = x*bytes; plane 1 bytes = ceil(x/2)*2*bytes.
static void planeRegion(const Fmt &f, int lx, int ly, int lw, int lh, int plane, int &bx, int &by, int &bw, int &bh) {
    if (lx < 0 || lx + lw < 0) { bx = by = bw = bh = 0; return; }   // av_image_get_linesize errors on negative widths
    auto linesize = [&](int w) { return plane ? ((w + 1) >> 1) * 2 * f.bytes : w * f.bytes * f.lanes0; };
    const int sy = plane ? f.sub_y : 0;
    bx = linesize(lx); bw = linesize(lx + lw) - bx;
    by = ly >> sy; bh = ((ly + lh + (1 << sy) - 1) >> sy) - by;
}

struct Frame {   // a semiplanar surface
    const Fmt *fmt; int w, h; Plane y, uv;
    Frame(const Fmt &f, int w_, int h_, std::mt19937 *rng) : fmt(&f), w(w_), h(h_),
        y(((w_ * f.bytes * f.lanes0 + 255) / 256) * 256, h_, rng),
        uv(((w_ * f.bytes + 255) / 256) * 256, f.planes > 1 ? (h_ + (1 << f.sub_y) - 1) >> f.sub_y : 1, rng) {
        if (rng && f.bytes == 2) {   // keep 10-bit codes in range with zero padding, like real frames
            auto fix = [&](Plane &p) {
                for (size_t i = 0; i + 1 < p.host.size(); i += 2) {
                    uint16_t v = (uint16_t)((((*rng)() % 1024) << f.shift));
                    memcpy(&p.host[i], &v, 2);
                }
                check(cudaMemcpy(p.dev, p.host.data(), p.host.size(), cudaMemcpyHostToDevice), "upload");
            };
            fix(y); fix(uv);
        }
    }
};

struct Rgba {   // packed 8-bit RGBA (bgra-like order chosen by offsets)
    int w, h; Plane px; int step = 4, r_off = 2, g_off = 1, b_off = 0, a_off = 3;
    Rgba(int w_, int h_, std::mt19937 *rng) : w(w_), h(h_), px(w_ * 4, h_, rng) {}
};

// Packed RGBA in a CUDA array behind a texture object, as a zero-copy DMA-BUF import presents it.
// The texture is created with the production descriptor (rectTextureDesc), so a wrong read mode
// there shows up here as a parity failure instead of on the live output.
struct RgbaTex {
    cudaArray_t array = nullptr; CUtexObject tex = 0;
    RgbaTex(const Rgba &src) {
        cudaChannelFormatDesc ch = cudaCreateChannelDesc<uchar4>();
        check(cudaMallocArray(&array, &ch, src.w, src.h), "cudaMallocArray");
        check(cudaMemcpy2DToArray(array, 0, 0, src.px.dev, src.px.pitch, src.w * 4, src.h, cudaMemcpyDeviceToDevice), "to array");
        CUDA_RESOURCE_DESC res; CUDA_TEXTURE_DESC td;
        avp::mixer::rectTextureDesc((CUarray)array, res, td);
        if (cuTexObjectCreate(&tex, &res, &td, nullptr) != CUDA_SUCCESS) throw std::runtime_error("cuTexObjectCreate");
    }
    ~RgbaTex() { if (tex) cuTexObjectDestroy(tex); if (array) cudaFreeArray(array); }
};

struct Layer {
    int kind;                 // AVP_RECT_KIND_*
    const Frame *yuv = nullptr;
    const Rgba *rgb = nullptr;
    const RgbaTex *tex = nullptr;   // *_TEX kinds: same pixels as rgb, fetched through the texture
    int cx, cy, cw, ch;       // crop (source pixels)
    int dx, dy, dw, dh;       // destination on the canvas (luma pixels), chroma-aligned
};

static int clearValue(const Fmt &f, int plane) { return plane ? 1 << (f.depth - 1) : 16 << (f.depth - 8); }

// Reference: memset + one launch per layer per plane, the production path before batching.
static void drawLayered(const Fmt &cf, Frame &canvas, const std::vector<Layer> &layers, int transfer) {
    const int cw_bytes = canvas.w * cf.bytes * cf.lanes0;
    if (cf.planes == 1) {
        check(cudaMemset2D(canvas.y.dev, canvas.y.pitch, 0, cw_bytes, canvas.y.rows), "memset rgb");
    } else if (cf.bytes == 2) {
        check(cudaMemset2D(canvas.y.dev, canvas.y.pitch, 0, canvas.y.pitch, canvas.y.rows), "memset");
        std::vector<uint16_t> row(canvas.y.pitch / 2, (uint16_t)(clearValue(cf, 0) << cf.shift));
        for (int r = 0; r < canvas.y.rows; ++r) check(cudaMemcpy(canvas.y.dev + r * canvas.y.pitch, row.data(), cw_bytes, cudaMemcpyHostToDevice), "clear");
        std::vector<uint16_t> crow(canvas.uv.pitch / 2, (uint16_t)(clearValue(cf, 1) << cf.shift));
        for (int r = 0; r < canvas.uv.rows; ++r) check(cudaMemcpy(canvas.uv.dev + r * canvas.uv.pitch, crow.data(), cw_bytes, cudaMemcpyHostToDevice), "clear");
    } else {
        check(cudaMemset2D(canvas.y.dev, canvas.y.pitch, clearValue(cf, 0), cw_bytes, canvas.y.rows), "memset y");
        check(cudaMemset2D(canvas.uv.dev, canvas.uv.pitch, clearValue(cf, 1), cw_bytes, canvas.uv.rows), "memset uv");
    }
    const int dst_scale = 1 << (cf.depth - 8);
    for (const Layer &L : layers) {
        if (L.kind >= AVP_RECT_KIND_RGB) {
            const int bw = 1 << cf.sub_x, bh = 1 << cf.sub_y;
            const dim3 grid(((L.dw + bw - 1) / bw + 31) / 32, ((L.dh + bh - 1) / bh + 7) / 8), block(32, 8);
            if (L.kind == AVP_RECT_KIND_RGB || L.kind == AVP_RECT_KIND_RGB_TEX)
                rgb_to_yuv<<<grid, block>>>(L.rgb->px.dev, L.rgb->px.pitch, L.cx, L.cy, L.cw, L.ch,
                    L.rgb->step, L.rgb->r_off, L.rgb->g_off, L.rgb->b_off,
                    canvas.y.dev, canvas.y.pitch, canvas.uv.dev, canvas.uv.pitch,
                    L.dx, L.dy, L.dw, L.dh, canvas.w, canvas.h, cf.bytes, cf.shift, dst_scale, cf.sub_x, cf.sub_y,
                    transfer, 203.f, 1000.f);
            else
                rgba_over_yuv<<<grid, block>>>(L.rgb->px.dev, L.rgb->px.pitch, L.cx, L.cy, L.cw, L.ch,
                    L.rgb->step, L.rgb->r_off, L.rgb->g_off, L.rgb->b_off, L.rgb->a_off,
                    canvas.y.dev, canvas.y.pitch, canvas.uv.dev, canvas.uv.pitch,
                    L.dx, L.dy, L.dw, L.dh, canvas.w, canvas.h, cf.bytes, cf.shift, dst_scale, cf.sub_x, cf.sub_y,
                    transfer, 203.f, 1000.f, 0);
            check(cudaGetLastError(), "rgb launch");
            continue;
        }
        const Fmt &sf = *L.yuv->fmt;
        for (int p = 0; p < cf.planes; ++p) {
            const int lanes = p ? 2 : cf.lanes0;
            int sx, sy, sw, sh, dx, dy, dw, dh, ox, oy, cwp, chp;
            planeRegion(sf, L.cx, L.cy, L.cw, L.ch, p, sx, sy, sw, sh);
            planeRegion(cf, L.dx, L.dy, L.dw, L.dh, p, dx, dy, dw, dh);
            planeRegion(cf, 0, 0, canvas.w, canvas.h, p, ox, oy, cwp, chp);
            sx /= lanes * sf.bytes; sw /= lanes * sf.bytes;
            dx /= lanes * cf.bytes; dw /= lanes * cf.bytes; cwp /= lanes * cf.bytes;
            if (dw <= 0 || dh <= 0) continue;   // the real code would fail the launch; skip for the test
            const Plane &src = p ? L.yuv->uv : L.yuv->y;
            Plane &dst = p ? canvas.uv : canvas.y;
            const dim3 grid((dw + 31) / 32, (dh + 7) / 8), block(32, 8);
            if (L.kind == AVP_RECT_KIND_YUV)
                scale_plane<<<grid, block>>>(src.dev, src.pitch, sx, sy, sw, sh, dst.dev, dst.pitch, dx, dy, dw, dh,
                                             cwp, chp, lanes, cf.bytes, cf.shift);
            else
                convert_scale_plane<<<grid, block>>>(src.dev, src.pitch, sx, sy, sw, sh, sf.bytes, sf.shift,
                                                     dst.dev, dst.pitch, dx, dy, dw, dh, cwp, chp, lanes, cf.bytes, cf.shift,
                                                     float(1 << (cf.depth - sf.depth)));
            check(cudaGetLastError(), "scale launch");
        }
    }
}

// Candidate: the rect table + one launch, mirroring CudaRectDraw::fillTableEntry.
static void fillTable(const Fmt &cf, const std::vector<Layer> &layers, std::vector<AvpRectLayer> &table) {
    table.clear();
    for (const Layer &L : layers) {
        AvpRectLayer e{};
        if (L.kind >= AVP_RECT_KIND_RGB) {
            e.kind = L.kind; e.step = L.rgb->step; e.r_off = L.rgb->r_off; e.g_off = L.rgb->g_off; e.b_off = L.rgb->b_off; e.a_off = L.rgb->a_off;
            if (L.tex) { e.src[0] = L.tex->tex; e.src_pitch[0] = 0; }
            else { e.src[0] = (unsigned long long)(uintptr_t)L.rgb->px.dev; e.src_pitch[0] = L.rgb->px.pitch; }
            e.sx[0] = L.cx; e.sy[0] = L.cy; e.sw[0] = L.cw; e.sh[0] = L.ch;
            e.dx[0] = L.dx; e.dy[0] = L.dy; e.dw[0] = L.dw; e.dh[0] = L.dh;
            const int bw = 1 << cf.sub_x, bh = 1 << cf.sub_y;
            e.dx[1] = L.dx >> cf.sub_x; e.dy[1] = L.dy >> cf.sub_y;
            e.dw[1] = (L.dx + L.dw + bw - 1) / bw - e.dx[1];
            e.dh[1] = (L.dy + L.dh + bh - 1) / bh - e.dy[1];
            if (L.dx + L.dw <= 0) e.dw[1] = 0;
            if (L.dy + L.dh <= 0) e.dh[1] = 0;
        } else {
            const Fmt &sf = *L.yuv->fmt;
            e.kind = L.kind; e.src_bytes = sf.bytes; e.src_shift = sf.shift;
            e.mul = L.kind == AVP_RECT_KIND_PROMOTE ? float(1 << (cf.depth - sf.depth)) : 1.f;
            for (int p = 0; p < cf.planes; ++p) {
                const int lanes = p ? 2 : cf.lanes0;
                int sx, sy, sw, sh, dx, dy, dw, dh;
                planeRegion(sf, L.cx, L.cy, L.cw, L.ch, p, sx, sy, sw, sh);
                planeRegion(cf, L.dx, L.dy, L.dw, L.dh, p, dx, dy, dw, dh);
                sx /= lanes * sf.bytes; sw /= lanes * sf.bytes;
                dx /= lanes * cf.bytes; dw /= lanes * cf.bytes;
                const Plane &src = p ? L.yuv->uv : L.yuv->y;
                e.src[p] = (unsigned long long)(uintptr_t)src.dev; e.src_pitch[p] = src.pitch;
                e.sx[p] = sx; e.sy[p] = sy; e.sw[p] = sw; e.sh[p] = sh;
                e.dx[p] = dx; e.dy[p] = dy; e.dw[p] = dw; e.dh[p] = dh;
            }
        }
        table.push_back(e);
    }
}

static void drawBatched(const Fmt &cf, Frame &canvas, const std::vector<AvpRectLayer> &table, AvpRectLayer *dev_table, int transfer) {
    if (!table.empty())
        check(cudaMemcpyAsync(dev_table, table.data(), table.size() * sizeof(AvpRectLayer), cudaMemcpyHostToDevice, 0), "table");
    int ox, oy, chroma_w = 0, chroma_h = 0;
    if (cf.planes > 1) { planeRegion(cf, 0, 0, canvas.w, canvas.h, 1, ox, oy, chroma_w, chroma_h); chroma_w /= 2 * cf.bytes; }
    const dim3 grid((canvas.w + 32 * AVP_RECT_PX - 1) / (32 * AVP_RECT_PX), (canvas.h + 7) / 8, cf.planes), block(32, 8);
    bool any_rgb = false;
    for (const auto &e : table) any_rgb = any_rgb || e.kind >= AVP_RECT_KIND_RGB;
    const int clear0 = cf.planes == 1 ? 0 : clearValue(cf, 0);
    if (cf.planes == 1)
        composite_planes_packed4<<<grid, block, (table.size() + 31) / 32 * sizeof(unsigned int)>>>(dev_table, (int)table.size(), canvas.y.dev, canvas.y.pitch, canvas.uv.dev, canvas.uv.pitch,
            canvas.w, canvas.h, chroma_w, chroma_h, cf.bytes, cf.shift, 1 << (cf.depth - 8), cf.sub_x, cf.sub_y,
            clear0, clearValue(cf, 1), transfer, 203.f, 1000.f);
    else if (any_rgb)
        composite_planes<<<grid, block, (table.size() + 31) / 32 * sizeof(unsigned int)>>>(dev_table, (int)table.size(), canvas.y.dev, canvas.y.pitch, canvas.uv.dev, canvas.uv.pitch,
            canvas.w, canvas.h, chroma_w, chroma_h, cf.bytes, cf.shift, 1 << (cf.depth - 8), cf.sub_x, cf.sub_y,
            clear0, clearValue(cf, 1), transfer, 203.f, 1000.f);
    else
        composite_planes_yuv<<<grid, block, (table.size() + 31) / 32 * sizeof(unsigned int)>>>(dev_table, (int)table.size(), canvas.y.dev, canvas.y.pitch, canvas.uv.dev, canvas.uv.pitch,
            canvas.w, canvas.h, chroma_w, chroma_h, cf.bytes, cf.shift, 1 << (cf.depth - 8), cf.sub_x, cf.sub_y,
            clear0, clearValue(cf, 1), transfer, 203.f, 1000.f);
    check(cudaGetLastError(), "composite launch");
}

// Bit-exact except where a blended sample rounds differently because nvcc contracts
// a*x + (1-a)*y into an FMA in one kernel and not the other: allow one code, rarely.
static void comparePlanes(const Plane &a, const Plane &b, int width_bytes, int bytes, int shift, const char *what) {
    const auto A = a.download(), B = b.download();
    size_t samples = 0, off_by_one = 0;
    for (int r = 0; r < a.rows; ++r)
        for (int c = 0; c < width_bytes; c += bytes) {
            int va, vb;
            if (bytes == 2) { uint16_t wa, wb; memcpy(&wa, &A[size_t(r) * a.pitch + c], 2); memcpy(&wb, &B[size_t(r) * b.pitch + c], 2); va = wa >> shift; vb = wb >> shift; }
            else { va = A[size_t(r) * a.pitch + c]; vb = B[size_t(r) * b.pitch + c]; }
            ++samples;
            if (va == vb) continue;
            if (va - vb == 1 || vb - va == 1) { ++off_by_one; continue; }
            throw std::runtime_error(std::string(what) + ": mismatch at row " + std::to_string(r) + " byte " +
                                     std::to_string(c) + " (" + std::to_string(va) + " vs " + std::to_string(vb) + ")");
        }
    if (off_by_one * 1000 > samples)
        throw std::runtime_error(std::string(what) + ": too many one-code differences: " + std::to_string(off_by_one) + " of " + std::to_string(samples));
    if (off_by_one) std::cout << "  " << what << ": " << off_by_one << " of " << samples << " samples differ by one code (blend rounding)\n";
}

static void scenario(const char *name, const Fmt &cf, int cw, int ch, const std::vector<Layer> &layers, int transfer,
                     AvpRectLayer *dev_table) {
    Frame ref(cf, cw, ch, nullptr), out(cf, cw, ch, nullptr);
    drawLayered(cf, ref, layers, transfer);
    std::vector<AvpRectLayer> table;
    fillTable(cf, layers, table);
    drawBatched(cf, out, table, dev_table, transfer);
    check(cudaDeviceSynchronize(), "sync");
    comparePlanes(ref.y, out.y, cw * cf.bytes * cf.lanes0, cf.bytes, cf.shift, (std::string(name) + " plane0").c_str());
    if (cf.planes > 1)
        comparePlanes(ref.uv, out.uv, cw * cf.bytes, cf.bytes, cf.shift, (std::string(name) + " chroma").c_str());
    std::cout << "PASS " << name << " (" << layers.size() << " layers, " << cf.name << " " << cw << "x" << ch << ")\n";
}

int main() {
    std::mt19937 rng(7);
    AvpRectLayer *dev_table = nullptr;
    check(cudaMalloc(&dev_table, sizeof(AvpRectLayer) * 512), "table alloc");

    Frame a(NV12, 640, 360, &rng), b(NV12, 1280, 720, &rng), c(NV12, 302, 170, &rng);
    Frame pa(P210, 640, 360, &rng), pb(P210, 1280, 720, &rng);
    Frame qa(P010, 640, 360, &rng);
    Rgba g(400, 300, &rng), g2(128, 64, &rng);

    // 1. NV12 canvas: blit, up/down scale, overlaps in order, crop, partial off right/bottom edge.
    scenario("nv12 mixed", NV12, 1080, 1920, {
        {AVP_RECT_KIND_YUV, &b, nullptr, nullptr, 0, 0, 1280, 720, 0, 0, 1080, 608},
        {AVP_RECT_KIND_YUV, &a, nullptr, nullptr, 0, 0, 640, 360, 0, 0, 640, 360},          // exact blit
        {AVP_RECT_KIND_YUV, &c, nullptr, nullptr, 10, 6, 280, 150, 100, 700, 980, 520},     // crop + upscale
        {AVP_RECT_KIND_YUV, &b, nullptr, nullptr, 0, 0, 1280, 720, 540, 1500, 900, 506},    // off the right/bottom edge
        {AVP_RECT_KIND_YUV, &a, nullptr, nullptr, 0, 0, 640, 360, 200, 200, 320, 180},      // on top of the first
        {AVP_RECT_KIND_RGB, nullptr, &g, nullptr, 0, 0, 400, 300, 640, 1200, 400, 300},     // opaque graphic
        {AVP_RECT_KIND_RGBA, nullptr, &g2, nullptr, 0, 0, 128, 64, 300, 1000, 512, 256},    // blended over video
        {AVP_RECT_KIND_RGBA, nullptr, &g, nullptr, 0, 0, 400, 300, 0, 0, 1080, 1920},       // blended over everything
    }, 2, dev_table);

    // 2. Empty canvas: only the clear; and an odd width for the tail-store path.
    scenario("nv12 clear", NV12, 720, 480, {}, 2, dev_table);
    scenario("nv12 odd width", NV12, 1082, 606, {
        {AVP_RECT_KIND_YUV, &b, nullptr, nullptr, 0, 0, 1280, 720, 0, 0, 1082, 606},
        {AVP_RECT_KIND_YUV, &a, nullptr, nullptr, 0, 0, 640, 360, 542, 300, 540, 304},
    }, 2, dev_table);

    // 3. Sixteen-box grid, the demo's heaviest scene.
    {
        std::vector<Layer> grid;
        for (int i = 0; i < 16; ++i) {
            const int col = i % 2, row = i / 2;
            grid.push_back({AVP_RECT_KIND_YUV, (i % 3) ? &b : &a, nullptr, nullptr, 0, 0, (i % 3) ? 1280 : 640, (i % 3) ? 720 : 360,
                            col * 540, row * 240, 540, 240});
        }
        scenario("nv12 grid16", NV12, 1080, 1920, grid, 2, dev_table);
    }

    // Exercise culling-mask word boundaries and every bit up to the layer limit.
    for (int count : {31, 32, 33, 255, 256, 257, 310, 511, 512}) {
        std::vector<Layer> grid;
        for (int i = 0; i < count; ++i)
            grid.push_back({AVP_RECT_KIND_YUV, i % 2 ? &a : &b, nullptr, nullptr,
                           0, 0, i % 2 ? 640 : 1280, i % 2 ? 360 : 720,
                           (i % 32) * 16, (i / 32) * 32, 16, 32});
        const std::string label = "nv12 layers " + std::to_string(count);
        scenario(label.c_str(), NV12, 512, 512, grid, 2, dev_table);
    }

    // 4. P210 canvas (HLG): P210 sources, NV12 promoted, graphics converted with the HLG transfer.
    scenario("p210 hlg", P210, 1080, 1920, {
        {AVP_RECT_KIND_YUV, &pb, nullptr, nullptr, 0, 0, 1280, 720, 0, 0, 1080, 608},
        {AVP_RECT_KIND_PROMOTE, &a, nullptr, nullptr, 0, 0, 640, 360, 0, 700, 1080, 608},
        {AVP_RECT_KIND_PROMOTE, &c, nullptr, nullptr, 20, 10, 260, 150, 100, 1400, 520, 300},
        {AVP_RECT_KIND_YUV, &pa, nullptr, nullptr, 0, 0, 640, 360, 540, 1400, 540, 304},
        {AVP_RECT_KIND_RGB, nullptr, &g, nullptr, 0, 0, 400, 300, 640, 40, 400, 300},
        {AVP_RECT_KIND_RGBA, nullptr, &g2, nullptr, 0, 0, 128, 64, 0, 0, 1080, 1920},
    }, 0, dev_table);

    // 5. P010 canvas: 4:2:0 10-bit, promotion from NV12, PQ graphics.
    scenario("p010 pq", P010, 960, 540, {
        {AVP_RECT_KIND_YUV, &qa, nullptr, nullptr, 0, 0, 640, 360, 0, 0, 960, 540},
        {AVP_RECT_KIND_PROMOTE, &b, nullptr, nullptr, 0, 0, 1280, 720, 480, 270, 480, 270},
        {AVP_RECT_KIND_RGBA, nullptr, &g, nullptr, 0, 0, 400, 300, 100, 100, 400, 300},
    }, 1, dev_table);

    // 6. Packed RGB canvas (DMA-BUF browser scale test): rgb0 sources scaled and blitted.
    {
        Frame ra(RGB0, 640, 360, &rng), rb(RGB0, 1280, 720, &rng);
        scenario("rgb0 packed", RGB0, 1920, 1080, {
            {AVP_RECT_KIND_YUV, &rb, nullptr, nullptr, 0, 0, 1280, 720, 0, 0, 1920, 1080},
            {AVP_RECT_KIND_YUV, &ra, nullptr, nullptr, 0, 0, 640, 360, 1280, 720, 640, 360},
            {AVP_RECT_KIND_YUV, &ra, nullptr, nullptr, 10, 10, 300, 200, 100, 100, 900, 600},
        }, 2, dev_table);
    }

    // 7. Texture-backed RGBA (zero-copy DMA-BUF): opaque and blended, NV12 and P210 HLG canvases.
    {
        RgbaTex tg(g), tg2(g2);
        scenario("nv12 texture rgba", NV12, 1080, 1920, {
            {AVP_RECT_KIND_YUV, &b, nullptr, nullptr, 0, 0, 1280, 720, 0, 0, 1080, 608},
            {AVP_RECT_KIND_RGB_TEX, nullptr, &g, &tg, 0, 0, 400, 300, 640, 1200, 400, 300},
            {AVP_RECT_KIND_RGB_TEX, nullptr, &g2, &tg2, 10, 8, 100, 50, 0, 700, 1080, 540},
            {AVP_RECT_KIND_RGBA_TEX, nullptr, &g2, &tg2, 0, 0, 128, 64, 300, 100, 512, 256},
            {AVP_RECT_KIND_RGBA_TEX, nullptr, &g, &tg, 0, 0, 400, 300, 0, 0, 1080, 1920},
        }, 2, dev_table);
        scenario("p210 hlg texture rgba", P210, 1080, 1920, {
            {AVP_RECT_KIND_YUV, &pb, nullptr, nullptr, 0, 0, 1280, 720, 0, 0, 1080, 608},
            {AVP_RECT_KIND_RGB_TEX, nullptr, &g, &tg, 0, 0, 400, 300, 640, 40, 400, 300},
            {AVP_RECT_KIND_RGBA_TEX, nullptr, &g2, &tg2, 0, 0, 128, 64, 0, 0, 1080, 1920},
        }, 0, dev_table);
    }

    // Timing: grid16 on 1080x1920 NV12, both paths, median of runs.
    {
        std::vector<Layer> grid;
        for (int i = 0; i < 16; ++i)
            grid.push_back({AVP_RECT_KIND_YUV, &b, nullptr, nullptr, 0, 0, 1280, 720, (i % 2) * 540, (i / 2) * 240, 540, 240});
        Frame canvas(NV12, 1080, 1920, nullptr);
        std::vector<AvpRectLayer> table;
        fillTable(NV12, grid, table);
        cudaEvent_t t0, t1;
        check(cudaEventCreate(&t0), "event"); check(cudaEventCreate(&t1), "event");
        auto timeIt = [&](auto &&fn, const char *label) {
            for (int i = 0; i < 20; ++i) fn();   // warm up
            check(cudaDeviceSynchronize(), "sync");
            std::vector<float> ms;
            for (int i = 0; i < 200; ++i) {
                cudaEventRecord(t0);
                fn();
                cudaEventRecord(t1);
                cudaEventSynchronize(t1);
                float m = 0; cudaEventElapsedTime(&m, t0, t1); ms.push_back(m);
            }
            std::sort(ms.begin(), ms.end());
            printf("%-28s median %.3f ms  p90 %.3f ms per frame\n", label, ms[ms.size() / 2], ms[ms.size() * 9 / 10]);
        };
        timeIt([&] { drawLayered(NV12, canvas, grid, 2); }, "per-layer (2 memset + 32)");
        timeIt([&] { drawBatched(NV12, canvas, table, dev_table, 2); }, "batched yuv (1 launch)");
        std::vector<Layer> mixed = grid;
        mixed.push_back({AVP_RECT_KIND_RGBA, nullptr, &g2, nullptr, 0, 0, 128, 64, 0, 0, 256, 128});
        std::vector<AvpRectLayer> mixed_table;
        fillTable(NV12, mixed, mixed_table);
        timeIt([&] { drawLayered(NV12, canvas, mixed, 2); }, "per-layer +1 rgba");
        timeIt([&] { drawBatched(NV12, canvas, mixed_table, dev_table, 2); }, "batched full +1 rgba");
    }
    cudaFree(dev_table);
    std::cout << "ALL PASS\n";
    return 0;
}
