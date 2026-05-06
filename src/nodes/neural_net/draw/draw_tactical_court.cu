// Draw a lower-corner tactical court overlay onto an NV12 CUDA frame.
#include <stdint.h>
#include <cuda_runtime.h>

struct TacticalPoint {
    float x;
    float y;
    float radius;
    int y_color;
    int u_color;
    int v_color;
};

namespace {
__device__ __forceinline__ bool in_rect(int x, int y, int rx, int ry, int rw, int rh) {
    return x >= rx && x < rx + rw && y >= ry && y < ry + rh;
}

__device__ __forceinline__ bool near_panel_line(int x, int y,
                                                 int court_x, int court_y,
                                                 int court_w, int court_h,
                                                 int thickness) {
    if (!in_rect(x, y, court_x, court_y, court_w, court_h)) return false;
    const bool outer = x < court_x + thickness || x >= court_x + court_w - thickness ||
                       y < court_y + thickness || y >= court_y + court_h - thickness;
    const int mid_x = court_x + court_w / 2;
    const int mid_dx = x >= mid_x ? x - mid_x : mid_x - x;
    const bool midline = mid_dx <= thickness / 2;
    return outer || midline;
}

__device__ __forceinline__ int blend_channel(int src, int dst, float alpha) {
    const float v = (float)src * (1.0f - alpha) + (float)dst * alpha;
    return (int)fminf(255.0f, fmaxf(0.0f, v + 0.5f));
}

__device__ __forceinline__ bool point_hit(int x, int y, const TacticalPoint& p) {
    const float dx = (float)x - p.x;
    const float dy = (float)y - p.y;
    return dx * dx + dy * dy <= p.radius * p.radius;
}

__device__ __forceinline__ bool near_three_point_line(int x, int y,
                                                       int court_x, int court_y,
                                                       int court_w, int court_h,
                                                       bool left_side,
                                                       int thickness) {
    if (!in_rect(x, y, court_x, court_y, court_w, court_h)) return false;
    const float xf = ((float)x - (float)court_x) * 94.0f / fmaxf(1.0f, (float)court_w);
    const float yf = ((float)y - (float)court_y) * 50.0f / fmaxf(1.0f, (float)court_h);
    const float hoop_x = left_side ? 5.25f : 88.75f;
    const float hoop_y = 25.0f;
    const float r = 23.75f;
    const float corner_lat = 22.0f;
    const float corner_x = left_side
        ? hoop_x + sqrtf(fmaxf(0.0f, r * r - corner_lat * corner_lat))
        : hoop_x - sqrtf(fmaxf(0.0f, r * r - corner_lat * corner_lat));
    const float thick_ft = fmaxf(0.35f, (float)thickness * 50.0f / fmaxf(1.0f, (float)court_h));

    const bool corner_line = left_side
        ? (xf >= 0.0f && xf <= corner_x + thick_ft &&
           (fabsf(yf - 3.0f) <= thick_ft || fabsf(yf - 47.0f) <= thick_ft))
        : (xf <= 94.0f && xf >= corner_x - thick_ft &&
           (fabsf(yf - 3.0f) <= thick_ft || fabsf(yf - 47.0f) <= thick_ft));

    const float dx = xf - hoop_x;
    const float dy = yf - hoop_y;
    const float dist = sqrtf(dx * dx + dy * dy);
    const bool arc_gate = fabsf(dy) <= corner_lat + thick_ft &&
                          (left_side ? xf >= corner_x - thick_ft : xf <= corner_x + thick_ft);
    const bool arc_line = arc_gate && fabsf(dist - r) <= thick_ft;
    return corner_line || arc_line;
}

__device__ __forceinline__ bool near_hoop(int x, int y,
                                           int court_x, int court_y,
                                           int court_w, int court_h,
                                           bool left_side,
                                           int thickness) {
    if (!in_rect(x, y, court_x, court_y, court_w, court_h)) return false;
    const float xf = ((float)x - (float)court_x) * 94.0f / fmaxf(1.0f, (float)court_w);
    const float yf = ((float)y - (float)court_y) * 50.0f / fmaxf(1.0f, (float)court_h);
    const float hoop_x = left_side ? 5.25f : 88.75f;
    const float hoop_y = 25.0f;
    const float dx = xf - hoop_x;
    const float dy = yf - hoop_y;
    const float dist = sqrtf(dx * dx + dy * dy);
    const float hoop_r = 1.05f;
    const float thick_ft = fmaxf(0.28f, (float)thickness * 50.0f / fmaxf(1.0f, (float)court_h));
    return fabsf(dist - hoop_r) <= thick_ft;
}

__device__ __forceinline__ void resolve_pixel_color(
    int x, int y,
    int court_x, int court_y, int court_w, int court_h,
    const TacticalPoint* points, int num_points,
    int line_y, int line_u, int line_v,
    int three_y, int three_u, int three_v,
    int hoop_y, int hoop_u, int hoop_v,
    int line_thickness,
    int active_hoop_on_left,
    int& y_out, int& u_out, int& v_out,
    bool& draw_color)
{
    if (near_panel_line(x, y, court_x, court_y, court_w, court_h, line_thickness)) {
        y_out = line_y;
        u_out = line_u;
        v_out = line_v;
        draw_color = true;
    }

    if (near_three_point_line(x, y, court_x, court_y, court_w, court_h, true, line_thickness)) {
        const bool active = active_hoop_on_left != 0;
        y_out = active ? three_y : line_y;
        u_out = active ? three_u : line_u;
        v_out = active ? three_v : line_v;
        draw_color = true;
    }
    if (near_three_point_line(x, y, court_x, court_y, court_w, court_h, false, line_thickness)) {
        const bool active = active_hoop_on_left == 0;
        y_out = active ? three_y : line_y;
        u_out = active ? three_u : line_u;
        v_out = active ? three_v : line_v;
        draw_color = true;
    }

    if (near_hoop(x, y, court_x, court_y, court_w, court_h, true, line_thickness) ||
        near_hoop(x, y, court_x, court_y, court_w, court_h, false, line_thickness)) {
        y_out = hoop_y;
        u_out = hoop_u;
        v_out = hoop_v;
        draw_color = true;
    }

    for (int i = 0; i < num_points; ++i) {
        const TacticalPoint p = points[i];
        if (point_hit(x, y, p)) {
            y_out = p.y_color;
            u_out = p.u_color;
            v_out = p.v_color;
            draw_color = true;
        }
    }
}
} // namespace

extern "C" __global__ void kDrawTacticalCourtNV12Luma(
    uint8_t* __restrict__ y_plane, size_t pitch_y,
    int width, int height,
    int panel_x, int panel_y, int panel_w, int panel_h,
    int court_x, int court_y, int court_w, int court_h,
    const TacticalPoint* __restrict__ points, int num_points,
    int bg_y, int bg_u, int bg_v, float bg_alpha,
    int line_y, int line_u, int line_v,
    int three_y, int three_u, int three_v,
    int hoop_y, int hoop_u, int hoop_v,
    int line_thickness, int active_hoop_on_left)
{
    const int x = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    const int y = (int)(blockIdx.y * blockDim.y + threadIdx.y);
    if (x >= width || y >= height) return;
    if (!in_rect(x, y, panel_x, panel_y, panel_w, panel_h)) return;

    uint8_t* y_px = &y_plane[(size_t)y * pitch_y + (size_t)x];
    int out_y = blend_channel((int)(*y_px), bg_y, bg_alpha);
    int out_u = bg_u;
    int out_v = bg_v;
    bool draw_color = false;
    resolve_pixel_color(x, y, court_x, court_y, court_w, court_h,
                        points, num_points,
                        line_y, line_u, line_v,
                        three_y, three_u, three_v,
                        hoop_y, hoop_u, hoop_v,
                        line_thickness, active_hoop_on_left,
                        out_y, out_u, out_v, draw_color);
    *y_px = (uint8_t)out_y;
}

extern "C" __global__ void kDrawTacticalCourtNV12Chroma(
    uint8_t* __restrict__ uv_plane, size_t pitch_uv,
    int width, int height,
    int panel_x, int panel_y, int panel_w, int panel_h,
    int court_x, int court_y, int court_w, int court_h,
    const TacticalPoint* __restrict__ points, int num_points,
    int bg_y, int bg_u, int bg_v, float bg_alpha,
    int line_y, int line_u, int line_v,
    int three_y, int three_u, int three_v,
    int hoop_y, int hoop_u, int hoop_v,
    int line_thickness, int active_hoop_on_left)
{
    const int uv_x = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    const int uv_y = (int)(blockIdx.y * blockDim.y + threadIdx.y);
    const int uv_width = (width + 1) >> 1;
    const int uv_height = (height + 1) >> 1;
    if (uv_x >= uv_width || uv_y >= uv_height) return;

    const int x = uv_x << 1;
    const int y = uv_y << 1;
    if (!in_rect(x, y, panel_x, panel_y, panel_w, panel_h)) return;

    uint8_t* row = uv_plane + (size_t)uv_y * pitch_uv;
    const size_t uv_idx = (size_t)(uv_x << 1);

    int out_y = bg_y;
    int out_u = blend_channel((int)row[uv_idx + 0], bg_u, bg_alpha);
    int out_v = blend_channel((int)row[uv_idx + 1], bg_v, bg_alpha);
    bool draw_color = false;
    resolve_pixel_color(x, y, court_x, court_y, court_w, court_h,
                        points, num_points,
                        line_y, line_u, line_v,
                        three_y, three_u, three_v,
                        hoop_y, hoop_u, hoop_v,
                        line_thickness, active_hoop_on_left,
                        out_y, out_u, out_v, draw_color);
    row[uv_idx + 0] = (uint8_t)out_u;
    row[uv_idx + 1] = (uint8_t)out_v;
}
