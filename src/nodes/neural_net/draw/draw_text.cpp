#include "cuda_overlay_base.hpp"
#include "draw_batch_shared.hpp"

#include <array>
#include <cstdio>
#include <cstring>

#include "../../../../objs/src/nodes/neural_net/draw/draw_text.ptx.h"

using cuda_overlay::DrawColor;

namespace {

constexpr int kMaxOverlayChars = 96;
struct OverlayText {
    std::array<char, kMaxOverlayChars> line1{};
    std::array<char, kMaxOverlayChars> line2{};
    std::array<char, kMaxOverlayChars> line3{};
    int line1_len = 0;
    int line2_len = 0;
    int line3_len = 0;
    int origin_x = 48;
    int origin_y = 48;
    int font_scale = 2;
    int line_spacing = 12;
    int glyph_preset = (int)cuda_overlay::GlyphPreset::k10x14;
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

static OverlayText makeDefaultOverlay(int origin_x, int origin_y, int font_scale, int line_spacing, int glyph_preset) {
    OverlayText overlay;
    overlay.origin_x = origin_x;
    overlay.origin_y = origin_y;
    overlay.font_scale = font_scale;
    overlay.line_spacing = line_spacing;
    overlay.glyph_preset = glyph_preset;
    std::snprintf(overlay.line1.data(), overlay.line1.size(), "SHOTS: 0");
    overlay.line2[0] = '\0';
    overlay.line3[0] = '\0';
    overlay.line1_len = boundedStringLength(overlay.line1);
    overlay.line2_len = boundedStringLength(overlay.line2);
    overlay.line3_len = boundedStringLength(overlay.line3);
    return overlay;
}

} // namespace

class DrawText : public CudaOverlayBase {
private:
    std::string metadata_key_ = "basketball_analysis_v1";
    std::string analysis_object_key_ = "basketball_analysis";
    int origin_x_ = 48;
    int origin_y_ = 48;
    int font_scale_ = 2;
    int line_spacing_ = 12;
    cuda_overlay::GlyphPreset glyph_preset_ = cuda_overlay::GlyphPreset::k10x14;
    int debug_log_every_n_ = 0;
    DrawColor text_color_{235, 128, 128};
    DrawColor background_color_{16, 128, 128};
    bool draw_background_ = true;
    float background_opacity_ = 1.0f;

    cuda_overlay::DeviceBuffer<cuda_overlay::BatchedTextLabel> d_labels_;
    cuda_overlay::DeviceBuffer<char> d_text_blob_;

    const char* nodeName() const override { return "draw_text"; }

    void onKernelsUnloaded() override {
        d_labels_.release(cu_ctx_);
        d_text_blob_.release(cu_ctx_);
    }

    bool loadTextKernels() {
        return loadKernels(avpl_draw_text_ptx, avpl_draw_text_ptx_len,
                           "kDrawTextNV12Luma", "kDrawTextNV12Chroma");
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
        overlay.line3_len = boundedStringLength(overlay.line3);

        const int glyph_w = std::max(1, cuda_overlay::glyphBaseWidth((cuda_overlay::GlyphPreset)overlay.glyph_preset));
        const int glyph_h = std::max(1, cuda_overlay::glyphBaseHeight((cuda_overlay::GlyphPreset)overlay.glyph_preset));
        const int char_advance = std::max(1, cuda_overlay::glyphAdvance((cuda_overlay::GlyphPreset)overlay.glyph_preset) * overlay.font_scale);
        const int line_height = std::max(1, glyph_h * overlay.font_scale);
        const int max_line_len = std::max(overlay.line1_len, std::max(overlay.line2_len, overlay.line3_len));
        const int line_count = 1 + (overlay.line2_len > 0 ? 1 : 0) + (overlay.line3_len > 0 ? 1 : 0);
        const int text_w = max_line_len * char_advance;
        const int text_h = line_height * line_count + (line_count > 1 ? overlay.line_spacing * (line_count - 1) : 0);
        const int pad_x = std::max(10, overlay.font_scale * 2);
        const int pad_y = std::max(8, overlay.font_scale);

        overlay.bg_x = overlay.origin_x - pad_x;
        overlay.bg_y = overlay.origin_y - pad_y;
        overlay.bg_w = text_w + pad_x * 2;
        overlay.bg_h = text_h + pad_y * 2;
    }

    OverlayText buildOverlayText(const av::VideoFrame& frm) const {
        OverlayText overlay = makeDefaultOverlay(origin_x_, origin_y_, font_scale_, line_spacing_, (int)glyph_preset_);

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
            overlay.line3[0] = '\0';
        } catch (const std::exception&) {
        }

        finalizeOverlayLayout(overlay);
        return overlay;
    }

    bool drawTextOnFrame(av::VideoFrame& frm, const OverlayText& overlay) {
        const unsigned int block_x = 32;
        const unsigned int block_y = 8;
        const unsigned int grid_x = (unsigned int)(frm.width() + (int)block_x - 1) / block_x;
        const unsigned int grid_y = (unsigned int)(frm.height() + (int)block_y - 1) / block_y;
        const unsigned int uv_grid_x = (unsigned int)(((frm.width() + 1) / 2) + (int)block_x - 1) / block_x;
        const unsigned int uv_grid_y = (unsigned int)(((frm.height() + 1) / 2) + (int)block_y - 1) / block_y;

        CUdeviceptr y_plane = (CUdeviceptr)(uintptr_t)frm.raw()->data[0];
        size_t pitch_y = (size_t)frm.raw()->linesize[0];
        CUdeviceptr uv_plane = (CUdeviceptr)(uintptr_t)frm.raw()->data[1];
        size_t pitch_uv = (size_t)frm.raw()->linesize[1];
        int width = frm.width();
        int height = frm.height();
        std::vector<cuda_overlay::BatchedTextLabel> batched_labels;
        std::vector<char> text_blob;
        batched_labels.reserve(1);
        cuda_overlay::BatchedTextLabel label;
        label.line1_len = overlay.line1_len;
        label.line2_len = overlay.line2_len;
        label.line3_len = overlay.line3_len;
        label.origin_x = overlay.origin_x;
        label.origin_y = overlay.origin_y;
        label.font_scale = overlay.font_scale;
        label.line_spacing = overlay.line_spacing;
        label.glyph_preset = overlay.glyph_preset;
        label.bg_x = overlay.bg_x;
        label.bg_y = overlay.bg_y;
        label.bg_w = overlay.bg_w;
        label.bg_h = overlay.bg_h;
        label.draw_background = draw_background_ ? 1 : 0;
        label.background_opacity = background_opacity_;
        label.text_y = text_color_.y;
        label.text_u = text_color_.u;
        label.text_v = text_color_.v;
        label.bg_y_color = background_color_.y;
        label.bg_u = background_color_.u;
        label.bg_v = background_color_.v;
        label.line1_offset = cuda_overlay::appendTextBlob(text_blob, overlay.line1.data(), overlay.line1_len);
        label.line2_offset = cuda_overlay::appendTextBlob(text_blob, overlay.line2.data(), overlay.line2_len);
        label.line3_offset = cuda_overlay::appendTextBlob(text_blob, overlay.line3.data(), overlay.line3_len);
        batched_labels.push_back(label);

        if (!d_labels_.upload(batched_labels, cu_ctx_, cuda_dev_ctx_->stream)) {
            logstream << "draw_text: failed uploading label descriptors";
            return false;
        }
        if (!d_text_blob_.uploadBytes(text_blob.data(), text_blob.size(), cu_ctx_, cuda_dev_ctx_->stream)) {
            logstream << "draw_text: failed uploading text blob";
            return false;
        }
        CUdeviceptr labels_ptr = d_labels_.ptr();
        int num_labels = (int)batched_labels.size();
        CUdeviceptr text_blob_ptr = d_text_blob_.ptr();

        void* y_args[] = {
            (void*)&y_plane, (void*)&pitch_y,
            (void*)&width, (void*)&height,
            (void*)&labels_ptr, (void*)&num_labels,
            (void*)&text_blob_ptr
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
            (void*)&labels_ptr, (void*)&num_labels,
            (void*)&text_blob_ptr
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
             cuda_overlay::GlyphPreset glyph_preset,
             DrawColor text_color,
             DrawColor background_color,
             bool draw_background,
             float background_opacity,
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
          glyph_preset_(glyph_preset),
          debug_log_every_n_(debug_log_every_n),
          text_color_(text_color),
          background_color_(background_color),
          draw_background_(draw_background),
          background_opacity_(background_opacity) {
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
        const int font_scale = params.value("font_scale", 2);
        const int line_spacing = params.value("line_spacing", 12);
        const int debug_log_every_n = params.value("debug_log_every_n", 0);
        const std::string glyph_preset_name = params.value("glyph_preset", std::string("10x14"));
        cuda_overlay::GlyphPreset glyph_preset = cuda_overlay::GlyphPreset::k10x14;
        if (!cuda_overlay::tryParseGlyphPreset(glyph_preset_name, glyph_preset)) {
            throw Error("draw_text: glyph_preset must be one of: 5x7, 10x14");
        }

        DrawColor text_color{235, 128, 128};
        if (params.count("text_color")) {
            if (!params["text_color"].is_string() || !cuda_overlay::tryParseNamedColor(params["text_color"].get<std::string>(), text_color)) {
                throw Error("draw_text: text_color must be one of: white, black, red, green, yellow, light_blue");
            }
        }

        DrawColor background_color{16, 128, 128};
        bool draw_background = true;
        float background_opacity = 1.0f;
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
        if (params.count("background_opacity")) {
            background_opacity = params["background_opacity"];
            if (background_opacity < 0.0f || background_opacity > 1.0f) {
                throw Error("draw_text: background_opacity must be in [0,1]");
            }
        }

        return NodeSISO<av::VideoFrame, av::VideoFrame>::template createCommon<DrawText>(
            edges, params, metadata_key, analysis_object_key, origin_x, origin_y,
            font_scale, line_spacing, glyph_preset, text_color, background_color, draw_background, background_opacity,
            upstream.input_params, upstream.frame_rate, upstream.timebase, debug_log_every_n);
    }
};

DECLNODE(draw_text, DrawText)
