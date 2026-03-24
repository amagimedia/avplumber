#include "cuda_overlay_base.hpp"

#include <array>
#include <cstdio>
#include <cstring>

#include "../../../objs/src/nodes/hwaccel/draw_text.ptx.h"

using cuda_overlay::DrawColor;

namespace {

constexpr int kMaxOverlayChars = 96;

struct OverlayText {
    std::array<char, kMaxOverlayChars> line1{};
    std::array<char, kMaxOverlayChars> line2{};
    int line1_len = 0;
    int line2_len = 0;
    int origin_x = 48;
    int origin_y = 48;
    int font_scale = 5;
    int line_spacing = 16;
    int bg_x = 0;
    int bg_y = 0;
    int bg_w = 0;
    int bg_h = 0;
};

static int boundedStringLength(const std::array<char, kMaxOverlayChars>& text) {
    int len = 0;
    while (len < (int)text.size() && text[(size_t)len] != '\0') {
        ++len;
    }
    return len;
}

static OverlayText makeDefaultOverlay(int origin_x, int origin_y, int font_scale, int line_spacing) {
    OverlayText overlay;
    overlay.origin_x = origin_x;
    overlay.origin_y = origin_y;
    overlay.font_scale = font_scale;
    overlay.line_spacing = line_spacing;
    std::snprintf(overlay.line1.data(), overlay.line1.size(), "SHOTS: 0");
    overlay.line2[0] = '\0';
    overlay.line1_len = boundedStringLength(overlay.line1);
    overlay.line2_len = boundedStringLength(overlay.line2);
    return overlay;
}

} // namespace

class DrawText : public CudaOverlayBase {
private:
    std::string metadata_key_ = "basketball_analysis_v1";
    std::string analysis_object_key_ = "basketball_analysis";
    int origin_x_ = 48;
    int origin_y_ = 48;
    int font_scale_ = 5;
    int line_spacing_ = 16;
    int debug_log_every_n_ = 0;
    DrawColor text_color_{235, 128, 128};
    DrawColor background_color_{16, 128, 128};
    bool draw_background_ = true;

    CUdeviceptr line1_symbol_ = 0;
    CUdeviceptr line2_symbol_ = 0;
    size_t line_symbol_size_ = 0;

    const char* nodeName() const override { return "draw_text"; }

    void onKernelsUnloaded() override {
        line1_symbol_ = 0;
        line2_symbol_ = 0;
        line_symbol_size_ = 0;
    }

    bool loadTextKernels() {
        if (!loadKernels(avpl_draw_text_ptx, avpl_draw_text_ptx_len,
                         "kDrawTextNV12Luma", "kDrawTextNV12Chroma")) {
            return false;
        }
        if (line1_symbol_ && line2_symbol_) return true;

        size_t line1_bytes = 0;
        size_t line2_bytes = 0;
        if (CUDA_OVERLAY_CHECK_CU(cuModuleGetGlobal(&line1_symbol_, &line1_bytes, draw_module_, "gDrawTextLine1"))) {
            logstream << "draw_text: failed to get line1 symbol";
            return false;
        }
        if (CUDA_OVERLAY_CHECK_CU(cuModuleGetGlobal(&line2_symbol_, &line2_bytes, draw_module_, "gDrawTextLine2"))) {
            logstream << "draw_text: failed to get line2 symbol";
            return false;
        }
        if (line1_bytes < kMaxOverlayChars || line2_bytes < kMaxOverlayChars) {
            logstream << "draw_text: PTX line buffers are too small";
            return false;
        }
        line_symbol_size_ = std::min(line1_bytes, line2_bytes);
        return true;
    }

    static const Parameters* findObjectChild(const Parameters* parent, const char* key) {
        if (!parent || !parent->contains(key)) return nullptr;
        const auto& child = (*parent)[key];
        if (!child.is_object()) return nullptr;
        return &child;
    }

    static int readIntOrDefault(const Parameters* parent, const char* key, int fallback) {
        if (!parent || !parent->contains(key)) return fallback;
        const auto& value = (*parent)[key];
        if (value.is_number_integer()) return value.get<int>();
        if (value.is_number_float()) return (int)std::lround(value.get<double>());
        return fallback;
    }

    void finalizeOverlayLayout(OverlayText& overlay) const {
        overlay.line1_len = boundedStringLength(overlay.line1);
        overlay.line2_len = boundedStringLength(overlay.line2);

        const int char_advance = std::max(1, 6 * overlay.font_scale);
        const int line_height = std::max(1, 7 * overlay.font_scale);
        const int text_w = std::max(overlay.line1_len, overlay.line2_len) * char_advance;
        const int text_h = line_height * (overlay.line2_len > 0 ? 2 : 1)
                         + (overlay.line2_len > 0 ? overlay.line_spacing : 0);
        const int pad_x = std::max(10, overlay.font_scale * 2);
        const int pad_y = std::max(8, overlay.font_scale);

        overlay.bg_x = overlay.origin_x - pad_x;
        overlay.bg_y = overlay.origin_y - pad_y;
        overlay.bg_w = text_w + pad_x * 2;
        overlay.bg_h = text_h + pad_y * 2;
    }

    OverlayText buildOverlayText(const av::VideoFrame& frm) const {
        OverlayText overlay = makeDefaultOverlay(origin_x_, origin_y_, font_scale_, line_spacing_);

        const AVFrame* raw = frm.raw();
        if (!raw || !raw->metadata) {
            finalizeOverlayLayout(overlay);
            return overlay;
        }

        AVDictionaryEntry* entry = av_dict_get(raw->metadata, metadata_key_.c_str(), nullptr, 0);
        if (!entry || !entry->value) {
            finalizeOverlayLayout(overlay);
            return overlay;
        }

        try {
            const Parameters md = Parameters::parse(entry->value);
            const Parameters* analysis_root = findObjectChild(&md, analysis_object_key_.c_str());
            const Parameters* analysis = findObjectChild(analysis_root, "analysis");
            const Parameters* totals = findObjectChild(analysis, "totals");

            const int shots = readIntOrDefault(totals, "shots", 0);

            std::snprintf(overlay.line1.data(), overlay.line1.size(),
                          "SHOTS: %d", shots);
            overlay.line2[0] = '\0';
        } catch (const std::exception&) {
        }

        finalizeOverlayLayout(overlay);
        return overlay;
    }

    bool copyOverlayTextToGpu(const OverlayText& overlay) const {
        if (!line1_symbol_ || !line2_symbol_ || line_symbol_size_ < kMaxOverlayChars) {
            return false;
        }
        if (CUDA_OVERLAY_CHECK_CU(cuMemcpyHtoDAsync(line1_symbol_, overlay.line1.data(),
                                       std::min(line_symbol_size_, overlay.line1.size()),
                                       cuda_dev_ctx_->stream))) {
            logstream << "draw_text: failed copying line1 text";
            return false;
        }
        if (CUDA_OVERLAY_CHECK_CU(cuMemcpyHtoDAsync(line2_symbol_, overlay.line2.data(),
                                       std::min(line_symbol_size_, overlay.line2.size()),
                                       cuda_dev_ctx_->stream))) {
            logstream << "draw_text: failed copying line2 text";
            return false;
        }
        return true;
    }

    bool drawTextOnFrame(av::VideoFrame& frm, const OverlayText& overlay) {
        const unsigned int block_x = 32;
        const unsigned int block_y = 8;

        const int launch_w = std::max(1, overlay.bg_w);
        const int launch_h = std::max(1, overlay.bg_h);
        const unsigned int grid_x = (unsigned int)(launch_w + (int)block_x - 1) / block_x;
        const unsigned int grid_y = (unsigned int)(launch_h + (int)block_y - 1) / block_y;
        const unsigned int uv_grid_x = (unsigned int)(((launch_w + 1) / 2) + (int)block_x - 1) / block_x;
        const unsigned int uv_grid_y = (unsigned int)(((launch_h + 1) / 2) + (int)block_y - 1) / block_y;

        CUdeviceptr y_plane = (CUdeviceptr)(uintptr_t)frm.raw()->data[0];
        size_t pitch_y = (size_t)frm.raw()->linesize[0];
        CUdeviceptr uv_plane = (CUdeviceptr)(uintptr_t)frm.raw()->data[1];
        size_t pitch_uv = (size_t)frm.raw()->linesize[1];
        int width = frm.width();
        int height = frm.height();
        int origin_x = overlay.origin_x;
        int origin_y = overlay.origin_y;
        int font_scale = overlay.font_scale;
        int line_spacing = overlay.line_spacing;
        int line1_len = overlay.line1_len;
        int line2_len = overlay.line2_len;
        int bg_x = overlay.bg_x;
        int bg_y = overlay.bg_y;
        int bg_w = overlay.bg_w;
        int bg_h = overlay.bg_h;
        int draw_background = draw_background_ ? 1 : 0;
        int text_y = text_color_.y;
        int text_u = text_color_.u;
        int text_v = text_color_.v;
        int bg_y_color = background_color_.y;
        int bg_u = background_color_.u;
        int bg_v = background_color_.v;

        void* y_args[] = {
            (void*)&y_plane, (void*)&pitch_y,
            (void*)&width, (void*)&height,
            (void*)&origin_x, (void*)&origin_y,
            (void*)&font_scale, (void*)&line_spacing,
            (void*)&line1_len, (void*)&line2_len,
            (void*)&bg_x, (void*)&bg_y, (void*)&bg_w, (void*)&bg_h,
            (void*)&draw_background,
            (void*)&text_y, (void*)&bg_y_color
        };
        if (CUDA_OVERLAY_CHECK_CU(cuLaunchKernel(draw_luma_kernel_,
                                    grid_x, grid_y, 1,
                                    block_x, block_y, 1,
                                    0, cuda_dev_ctx_->stream, y_args, nullptr))) {
            logstream << "draw_text: failed launching luma kernel";
            return false;
        }

        void* uv_args[] = {
            (void*)&uv_plane, (void*)&pitch_uv,
            (void*)&width, (void*)&height,
            (void*)&origin_x, (void*)&origin_y,
            (void*)&font_scale, (void*)&line_spacing,
            (void*)&line1_len, (void*)&line2_len,
            (void*)&bg_x, (void*)&bg_y, (void*)&bg_w, (void*)&bg_h,
            (void*)&draw_background,
            (void*)&text_u, (void*)&text_v,
            (void*)&bg_u, (void*)&bg_v
        };
        if (CUDA_OVERLAY_CHECK_CU(cuLaunchKernel(draw_chroma_kernel_,
                                    uv_grid_x, uv_grid_y, 1,
                                    block_x, block_y, 1,
                                    0, cuda_dev_ctx_->stream, uv_args, nullptr))) {
            logstream << "draw_text: failed launching chroma kernel";
            return false;
        }

        return CUDA_OVERLAY_CHECK_CU(cuStreamSynchronize(cuda_dev_ctx_->stream)) == 0;
    }

    void maybeLogOverlay(const OverlayText& overlay) const {
        if (debug_log_every_n_ <= 0) return;
        if ((frame_counter_ % (uint64_t)debug_log_every_n_) != 0) return;
        logstream << "draw_text: frame=" << frame_counter_
                  << " metadata_key=" << metadata_key_
                  << " analysis_object_key=" << analysis_object_key_
                  << " line1=\"" << overlay.line1.data() << "\""
                  << " line2=\"" << overlay.line2.data() << "\"";
    }

    void drawOnFrame(const av::VideoFrame& input, av::VideoFrame& output) override {
        if (!loadTextKernels()) {
            throw Error("draw_text: failed to initialize CUDA kernels");
        }

        const OverlayText overlay = buildOverlayText(input);

        if (!copyOverlayTextToGpu(overlay)) {
            throw Error("draw_text: failed to upload overlay text");
        }
        if (!drawTextOnFrame(output, overlay)) {
            throw Error("draw_text: failed drawing text");
        }
        maybeLogOverlay(overlay);
    }

public:
    DrawText(std::unique_ptr<Source<av::VideoFrame>>&& source,
             std::unique_ptr<Sink<av::VideoFrame>>&& sink,
             std::string metadata_key,
             std::string analysis_object_key,
             int origin_x,
             int origin_y,
             int font_scale,
             int line_spacing,
             DrawColor text_color,
             DrawColor background_color,
             bool draw_background,
             VideoParameters input_params,
             av::Rational frame_rate,
             av::Rational timebase,
             int debug_log_every_n)
        : CudaOverlayBase(std::move(source), std::move(sink)),
          metadata_key_(std::move(metadata_key)),
          analysis_object_key_(std::move(analysis_object_key)),
          origin_x_(origin_x),
          origin_y_(origin_y),
          font_scale_(font_scale),
          line_spacing_(line_spacing),
          debug_log_every_n_(debug_log_every_n),
          text_color_(text_color),
          background_color_(background_color),
          draw_background_(draw_background) {
        input_params_ = input_params;
        frame_rate_ = frame_rate;
        timebase_ = timebase;
        if (origin_x_ < 0 || origin_y_ < 0) {
            throw Error("draw_text: origin_x and origin_y must be non-negative");
        }
        if (font_scale_ <= 0) {
            throw Error("draw_text: font_scale must be positive");
        }
        if (line_spacing_ < 0) {
            throw Error("draw_text: line_spacing must be non-negative");
        }
        if (metadata_key_.empty()) {
            throw Error("draw_text: metadata_key must not be empty");
        }
        if (analysis_object_key_.empty()) {
            throw Error("draw_text: analysis_object_key must not be empty");
        }
    }

    static std::shared_ptr<DrawText> create(NodeCreationInfo& nci) {
        EdgeManager& edges = nci.edges;
        const Parameters& params = nci.params;

        auto src_edge = edges.find<av::VideoFrame>(params["src"]);
        const auto upstream = resolveUpstreamInfo(src_edge, params);

        const std::string metadata_key = params.value("metadata_key", std::string("basketball_analysis_v1"));
        const std::string analysis_object_key = params.value("analysis_object_key", std::string("basketball_analysis"));
        const int origin_x = params.value("origin_x", 48);
        const int origin_y = params.value("origin_y", 48);
        const int font_scale = params.value("font_scale", 5);
        const int line_spacing = params.value("line_spacing", 16);
        const int debug_log_every_n = params.value("debug_log_every_n", 0);

        DrawColor text_color{235, 128, 128};
        if (params.count("text_color")) {
            if (!params["text_color"].is_string() || !cuda_overlay::tryParseNamedColor(params["text_color"].get<std::string>(), text_color)) {
                throw Error("draw_text: text_color must be one of: white, black, red, green, yellow, light_blue");
            }
        }

        DrawColor background_color{16, 128, 128};
        bool draw_background = true;
        if (params.count("background_color")) {
            if (!params["background_color"].is_string()) {
                throw Error("draw_text: background_color must be a string color name or \"none\"");
            }
            const std::string background_name = cuda_overlay::normalizeColorName(params["background_color"].get<std::string>());
            if (background_name == "none") {
                draw_background = false;
            } else if (!cuda_overlay::tryParseNamedColor(background_name, background_color)) {
                throw Error("draw_text: background_color must be one of: white, black, red, green, yellow, light_blue, none");
            }
        }

        return NodeSISO<av::VideoFrame, av::VideoFrame>::template createCommon<DrawText>(
            edges, params, metadata_key, analysis_object_key, origin_x, origin_y,
            font_scale, line_spacing, text_color, background_color, draw_background,
            upstream.input_params, upstream.frame_rate, upstream.timebase, debug_log_every_n);
    }
};

DECLNODE(draw_text, DrawText)
