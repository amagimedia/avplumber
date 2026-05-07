#include "cuda_overlay_base.hpp"
#include "draw_batch_shared.hpp"

#include <algorithm>
#include <string>
#include <vector>

#include "../../../../objs/src/nodes/neural_net/draw/draw_text.ptx.h"

namespace {

constexpr int kMaxLines = 3;
constexpr int kMaxLineChars = 96;

static std::vector<std::string> splitLines(const std::string& text) {
    std::vector<std::string> lines;
    size_t start = 0;
    while (start <= text.size() && (int)lines.size() < kMaxLines) {
        const size_t end = text.find('\n', start);
        std::string line = end == std::string::npos ? text.substr(start) : text.substr(start, end - start);
        if ((int)line.size() > kMaxLineChars) line.resize(kMaxLineChars);
        lines.push_back(std::move(line));
        if (end == std::string::npos) break;
        start = end + 1;
    }
    return lines;
}

} // namespace

class StaticTextOverlay : public CudaOverlayBase {
private:
    std::vector<std::string> lines_;
    int x_ = 60;
    int y_ = 60;
    int font_scale_ = 3;
    int line_spacing_ = 8;
    int padding_x_ = 12;
    int padding_y_ = 10;
    cuda_overlay::GlyphPreset glyph_preset_ = cuda_overlay::GlyphPreset::k10x14;
    cuda_overlay::DrawColor text_color_{235, 128, 128};
    cuda_overlay::DrawColor background_color_{16, 128, 128};
    bool draw_background_ = true;
    float background_opacity_ = 0.75f;

    cuda_overlay::DeviceBuffer<cuda_overlay::BatchedTextLabel> d_labels_;
    cuda_overlay::DeviceBuffer<char> d_text_blob_;

    const char* nodeName() const override { return "static_text_overlay"; }

    void onKernelsUnloaded() override {
        d_labels_.release(cu_ctx_);
        d_text_blob_.release(cu_ctx_);
    }

    bool loadTextKernels() {
        return loadKernels(avpl_draw_text_ptx, avpl_draw_text_ptx_len,
                           "kDrawTextNV12Luma", "kDrawTextNV12Chroma");
    }

    cuda_overlay::BatchedTextLabel buildLabel(std::vector<char>& text_blob) const {
        cuda_overlay::BatchedTextLabel label;
        label.font_scale = font_scale_;
        label.line_spacing = line_spacing_;
        label.glyph_preset = (int)glyph_preset_;
        label.draw_background = draw_background_ ? 1 : 0;
        label.background_opacity = background_opacity_;
        label.text_y = text_color_.y;
        label.text_u = text_color_.u;
        label.text_v = text_color_.v;
        label.bg_y_color = background_color_.y;
        label.bg_u = background_color_.u;
        label.bg_v = background_color_.v;

        if (!lines_.empty()) {
            label.line1_len = (int)lines_[0].size();
            label.line1_offset = cuda_overlay::appendTextBlob(text_blob, lines_[0].data(), label.line1_len);
        }
        if (lines_.size() > 1) {
            label.line2_len = (int)lines_[1].size();
            label.line2_offset = cuda_overlay::appendTextBlob(text_blob, lines_[1].data(), label.line2_len);
        }
        if (lines_.size() > 2) {
            label.line3_len = (int)lines_[2].size();
            label.line3_offset = cuda_overlay::appendTextBlob(text_blob, lines_[2].data(), label.line3_len);
        }

        const int line_count = (label.line1_len > 0 ? 1 : 0) + (label.line2_len > 0 ? 1 : 0) + (label.line3_len > 0 ? 1 : 0);
        const int max_len = std::max(label.line1_len, std::max(label.line2_len, label.line3_len));
        const int char_advance = std::max(1, cuda_overlay::glyphAdvance(glyph_preset_) * font_scale_);
        const int line_height = std::max(1, cuda_overlay::glyphBaseHeight(glyph_preset_) * font_scale_);
        const int text_w = max_len * char_advance;
        const int text_h = line_height * line_count + (line_count > 1 ? line_spacing_ * (line_count - 1) : 0);

        label.bg_w = std::max(1, std::min(input_params_.width, text_w + padding_x_ * 2));
        label.bg_h = std::max(1, std::min(input_params_.height, text_h + padding_y_ * 2));
        label.bg_x = std::max(0, std::min(x_, input_params_.width - label.bg_w));
        label.bg_y = std::max(0, std::min(y_, input_params_.height - label.bg_h));
        label.origin_x = label.bg_x + padding_x_;
        label.origin_y = label.bg_y + padding_y_;
        return label;
    }

    void drawOnFrame(const av::VideoFrame&, av::VideoFrame& output) override {
        if (!loadTextKernels()) {
            throw Error("static_text_overlay: failed to initialize text kernels");
        }
        if (lines_.empty()) return;

        std::vector<char> text_blob;
        std::vector<cuda_overlay::BatchedTextLabel> labels;
        labels.push_back(buildLabel(text_blob));

        if (!d_labels_.upload(labels, cu_ctx_, cuda_dev_ctx_->stream)) {
            throw Error("static_text_overlay: failed uploading label descriptors");
        }
        if (!d_text_blob_.uploadBytes(text_blob.data(), text_blob.size(), cu_ctx_, cuda_dev_ctx_->stream)) {
            throw Error("static_text_overlay: failed uploading text");
        }

        const unsigned int block_x = 32;
        const unsigned int block_y = 8;
        const unsigned int grid_x = (unsigned int)(output.width() + (int)block_x - 1) / block_x;
        const unsigned int grid_y = (unsigned int)(output.height() + (int)block_y - 1) / block_y;
        const unsigned int uv_grid_x = (unsigned int)(((output.width() + 1) / 2) + (int)block_x - 1) / block_x;
        const unsigned int uv_grid_y = (unsigned int)(((output.height() + 1) / 2) + (int)block_y - 1) / block_y;

        CUdeviceptr y_plane = (CUdeviceptr)(uintptr_t)output.raw()->data[0];
        size_t pitch_y = (size_t)output.raw()->linesize[0];
        CUdeviceptr uv_plane = (CUdeviceptr)(uintptr_t)output.raw()->data[1];
        size_t pitch_uv = (size_t)output.raw()->linesize[1];
        int width = output.width();
        int height = output.height();
        CUdeviceptr labels_ptr = d_labels_.ptr();
        CUdeviceptr text_blob_ptr = d_text_blob_.ptr();
        int num_labels = (int)labels.size();

        void* y_args[] = { (void*)&y_plane, (void*)&pitch_y, (void*)&width, (void*)&height,
                           (void*)&labels_ptr, (void*)&num_labels, (void*)&text_blob_ptr };
        if (CUDA_OVERLAY_CHECK_CU(cuLaunchKernel(draw_luma_kernel_, grid_x, grid_y, 1,
                                                 block_x, block_y, 1, 0, cuda_dev_ctx_->stream, y_args, nullptr))) {
            throw Error("static_text_overlay: luma kernel failed");
        }

        void* uv_args[] = { (void*)&uv_plane, (void*)&pitch_uv, (void*)&width, (void*)&height,
                            (void*)&labels_ptr, (void*)&num_labels, (void*)&text_blob_ptr };
        if (CUDA_OVERLAY_CHECK_CU(cuLaunchKernel(draw_chroma_kernel_, uv_grid_x, uv_grid_y, 1,
                                                 block_x, block_y, 1, 0, cuda_dev_ctx_->stream, uv_args, nullptr))) {
            throw Error("static_text_overlay: chroma kernel failed");
        }
        if (CUDA_OVERLAY_CHECK_CU(cuStreamSynchronize(cuda_dev_ctx_->stream))) {
            throw Error("static_text_overlay: stream synchronize failed");
        }
    }

public:
    StaticTextOverlay(std::unique_ptr<Source<av::VideoFrame>>&& source,
                      std::unique_ptr<Sink<av::VideoFrame>>&& sink,
                      std::string text,
                      int x,
                      int y,
                      int font_scale,
                      int line_spacing,
                      int padding_x,
                      int padding_y,
                      cuda_overlay::GlyphPreset glyph_preset,
                      cuda_overlay::DrawColor text_color,
                      cuda_overlay::DrawColor background_color,
                      bool draw_background,
                      float background_opacity,
                      VideoParameters input_params,
                      av::Rational frame_rate,
                      av::Rational timebase)
        : CudaOverlayBase(std::move(source), std::move(sink)),
          lines_(splitLines(std::move(text))),
          x_(x),
          y_(y),
          font_scale_(font_scale),
          line_spacing_(line_spacing),
          padding_x_(padding_x),
          padding_y_(padding_y),
          glyph_preset_(glyph_preset),
          text_color_(text_color),
          background_color_(background_color),
          draw_background_(draw_background),
          background_opacity_(background_opacity) {
        input_params_ = input_params;
        frame_rate_ = frame_rate;
        timebase_ = timebase;
        if (lines_.empty()) throw Error("static_text_overlay: text must not be empty");
        if (font_scale_ <= 0) throw Error("static_text_overlay: font_scale must be positive");
        if (line_spacing_ < 0) throw Error("static_text_overlay: line_spacing must be non-negative");
        if (padding_x_ < 0 || padding_y_ < 0) throw Error("static_text_overlay: padding must be non-negative");
        if (background_opacity_ < 0.f || background_opacity_ > 1.f) throw Error("static_text_overlay: background_opacity must be in [0,1]");
    }

    static std::shared_ptr<StaticTextOverlay> create(NodeCreationInfo& nci) {
        EdgeManager& edges = nci.edges;
        const Parameters& params = nci.params;

        auto src_edge = edges.find<av::VideoFrame>(params["src"]);
        const auto upstream = resolveUpstreamInfo(src_edge, params);

        const std::string text = params.value("text", std::string());
        const int x = params.value("x", 60);
        const int y = params.value("y", 60);
        const int font_scale = params.value("font_scale", 3);
        const int line_spacing = params.value("line_spacing", 8);
        const int padding_x = params.value("padding_x", 12);
        const int padding_y = params.value("padding_y", 10);

        cuda_overlay::GlyphPreset glyph_preset = cuda_overlay::GlyphPreset::k10x14;
        const std::string glyph_preset_name = params.value("glyph_preset", std::string("10x14"));
        if (!cuda_overlay::tryParseGlyphPreset(glyph_preset_name, glyph_preset)) {
            throw Error("static_text_overlay: glyph_preset must be one of: 5x7, 10x14");
        }

        cuda_overlay::DrawColor text_color{235, 128, 128};
        if (params.count("text_color") &&
            (!params["text_color"].is_string() ||
             !cuda_overlay::tryParseNamedColor(params["text_color"].get<std::string>(), text_color))) {
            throw Error("static_text_overlay: text_color must be a supported color name");
        }

        cuda_overlay::DrawColor background_color{16, 128, 128};
        bool draw_background = true;
        if (params.count("background_color")) {
            if (!params["background_color"].is_string()) {
                throw Error("static_text_overlay: background_color must be a supported color name or none");
            }
            const std::string bg_name = cuda_overlay::normalizeColorName(params["background_color"].get<std::string>());
            if (bg_name == "none") {
                draw_background = false;
            } else if (!cuda_overlay::tryParseNamedColor(bg_name, background_color)) {
                throw Error("static_text_overlay: background_color must be a supported color name or none");
            }
        }
        const float background_opacity = params.value("background_opacity", 0.75f);

        return NodeSISO<av::VideoFrame, av::VideoFrame>::template createCommon<StaticTextOverlay>(
            edges, params, text, x, y, font_scale, line_spacing, padding_x, padding_y,
            glyph_preset, text_color, background_color, draw_background, background_opacity,
            upstream.input_params, upstream.frame_rate, upstream.timebase);
    }
};

DECLNODE(static_text_overlay, StaticTextOverlay)
