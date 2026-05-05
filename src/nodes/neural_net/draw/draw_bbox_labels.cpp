#include "cuda_overlay_base.hpp"
#include "draw_batch_shared.hpp"

#include <array>
#include <cstdio>
#include <string>
#include <unordered_set>
#include <vector>

#include "../../../../objs/src/nodes/neural_net/draw/draw_text.ptx.h"

using cuda_overlay::DrawColor;

namespace {

constexpr int kMaxLabelChars = 64;
constexpr int kMaxGpuChars = 96;
constexpr int kMaxLabelLines = 3;

enum class FieldToken {
    TrackId,
    TeamAB,
    JerseyPct,
    Conf,
    Velocity,
    Label,
    Cls,
};

struct TemplateToken {
    bool is_literal = true;
    std::string literal;
    FieldToken field = FieldToken::TrackId;
};

using TemplateLine = std::vector<TemplateToken>;

struct LabelLayout {
    std::array<char, kMaxGpuChars> line1{};
    std::array<char, kMaxGpuChars> line2{};
    std::array<char, kMaxGpuChars> line3{};
    int line1_len = 0;
    int line2_len = 0;
    int line3_len = 0;
    int line_count = 0;
    int origin_x = 0;
    int origin_y = 0;
    int font_scale = 2;
    int line_spacing = 8;
    int glyph_preset = (int)cuda_overlay::GlyphPreset::k10x14;
    int bg_x = 0;
    int bg_y = 0;
    int bg_w = 0;
    int bg_h = 0;
};

static int boundedStringLength(const std::array<char, kMaxGpuChars>& text) {
    int len = 0;
    while (len < (int)text.size() && text[(size_t)len] != '\0') {
        ++len;
    }
    return len;
}

static std::vector<std::string> splitLines(const std::string& text) {
    std::vector<std::string> out;
    size_t start = 0;
    while (start <= text.size()) {
        const size_t end = text.find('\n', start);
        if (end == std::string::npos) {
            out.push_back(text.substr(start));
            break;
        }
        out.push_back(text.substr(start, end - start));
        start = end + 1;
    }
    return out;
}

static bool parseFieldToken(const std::string& token, FieldToken& out_field) {
    if (token == "track_id") {
        out_field = FieldToken::TrackId;
        return true;
    }
    if (token == "team_ab") {
        out_field = FieldToken::TeamAB;
        return true;
    }
    if (token == "jersey_pct") {
        out_field = FieldToken::JerseyPct;
        return true;
    }
    if (token == "conf") {
        out_field = FieldToken::Conf;
        return true;
    }
    if (token == "velocity") {
        out_field = FieldToken::Velocity;
        return true;
    }
    if (token == "label") {
        out_field = FieldToken::Label;
        return true;
    }
    if (token == "cls") {
        out_field = FieldToken::Cls;
        return true;
    }
    return false;
}

static bool compileTemplateLine(const std::string& line, TemplateLine& out_line, std::string& err_out) {
    out_line.clear();
    std::string literal;
    size_t i = 0;
    while (i < line.size()) {
        if (line[i] != '{') {
            literal.push_back(line[i]);
            ++i;
            continue;
        }
        const size_t close = line.find('}', i + 1);
        if (close == std::string::npos) {
            err_out = "missing closing } in label_template";
            return false;
        }
        if (!literal.empty()) {
            TemplateToken token;
            token.is_literal = true;
            token.literal = literal;
            out_line.push_back(std::move(token));
            literal.clear();
        }
        const std::string key = line.substr(i + 1, close - i - 1);
        FieldToken field;
        if (!parseFieldToken(key, field)) {
            err_out = "unknown token {" + key + "} in label_template";
            return false;
        }
        TemplateToken token;
        token.is_literal = false;
        token.field = field;
        out_line.push_back(std::move(token));
        i = close + 1;
    }
    if (!literal.empty()) {
        TemplateToken token;
        token.is_literal = true;
        token.literal = literal;
        out_line.push_back(std::move(token));
    }
    return true;
}

} // namespace

class DrawBBoxLabels : public CudaOverlayBase {
private:
    std::vector<std::string> metadata_keys_;
    std::string camera_shot_metadata_key_ = "camera_shot_info";
    double min_conf_ = 0.0;
    std::unordered_set<int> allowed_classes_;
    std::unordered_set<std::string> allowed_labels_;
    double model_content_width_ = 0.0;
    double model_content_height_ = 0.0;
    double model_content_offset_x_ = 0.0;
    double model_content_offset_y_ = 0.0;
    int debug_log_every_n_ = 0;

    std::string label_template_ = "ID:{track_id}\nV:{velocity}";
    std::vector<TemplateLine> compiled_template_;
    int velocity_precision_ = 1;
    bool show_predicted_labels_ = false;
    bool show_untracked_ = false;
    bool require_wide_shot_ = false;

    int font_scale_ = 2;
    int line_spacing_ = 8;
    int offset_x_ = 0;
    int offset_y_ = 0;
    cuda_overlay::GlyphPreset glyph_preset_ = cuda_overlay::GlyphPreset::k10x14;

    DrawColor text_color_{235, 128, 128};
    DrawColor background_color_{16, 128, 128};
    bool draw_background_ = true;
    float background_opacity_ = 0.75f;

    cuda_overlay::DeviceBuffer<cuda_overlay::BatchedTextLabel> d_labels_;
    cuda_overlay::DeviceBuffer<char> d_text_blob_;

    const char* nodeName() const override { return "draw_bbox_labels"; }

    std::string readShotType(const av::VideoFrame& frm) const {
        const AVFrame* raw = frm.raw();
        if (!raw || !raw->metadata || camera_shot_metadata_key_.empty()) return {};

        AVDictionaryEntry* entry = av_dict_get(raw->metadata, camera_shot_metadata_key_.c_str(), nullptr, 0);
        if (!entry || !entry->value) return {};

        try {
            Parameters md = Parameters::parse(entry->value);
            return md.value("camera_shot_type", std::string());
        } catch (const std::exception&) {
            return {};
        }
    }

    void onKernelsUnloaded() override {
        d_labels_.release(cu_ctx_);
        d_text_blob_.release(cu_ctx_);
    }

    bool loadTextKernels() {
        return loadKernels(avpl_draw_text_ptx, avpl_draw_text_ptx_len,
                           "kDrawTextNV12Luma", "kDrawTextNV12Chroma");
    }

    bool compileTemplate(std::string& err_out) {
        compiled_template_.clear();
        for (const std::string& line : splitLines(label_template_)) {
            if (line.empty()) continue;
            TemplateLine compiled_line;
            if (!compileTemplateLine(line, compiled_line, err_out)) {
                return false;
            }
            if (!compiled_line.empty()) {
                compiled_template_.push_back(std::move(compiled_line));
            }
        }
        if (compiled_template_.empty()) {
            err_out = "label_template produced zero compiled lines";
            return false;
        }
        if ((int)compiled_template_.size() > kMaxLabelLines) {
            compiled_template_.resize(kMaxLabelLines);
        }
        return true;
    }

    static void appendBounded(std::string& out, const std::string& chunk) {
        if ((int)out.size() >= kMaxLabelChars) return;
        const int space_left = kMaxLabelChars - (int)out.size();
        out.append(chunk.substr(0, (size_t)std::max(0, space_left)));
    }

    bool formatField(const cuda_overlay::ParsedYoloDetection& det, FieldToken field, std::string& out) const {
        char buf[96];
        switch (field) {
        case FieldToken::TrackId:
            if (!det.has_track_id) return false;
            if (det.track_id < 0 && !show_untracked_) return false;
            std::snprintf(buf, sizeof(buf), "%d", det.track_id);
            out = buf;
            return true;
        case FieldToken::TeamAB:
            if (!det.has_team || det.team < 0) {
                out = "?";
                return true;
            }
            if (det.team == 0) {
                out = "A";
                return true;
            }
            if (det.team == 1) {
                out = "B";
                return true;
            }
            out = "?";
            return true;
        case FieldToken::JerseyPct:
            if (!det.has_jersey_mode_ratio) return false;
            std::snprintf(buf, sizeof(buf), "%d", (int)std::lround(det.jersey_mode_ratio * 100.0));
            out = buf;
            return true;
        case FieldToken::Conf:
            std::snprintf(buf, sizeof(buf), "%.2f", det.conf);
            out = buf;
            return true;
        case FieldToken::Velocity:
            if (!det.has_velocity) return false;
            std::snprintf(buf, sizeof(buf), "%.*f,%.*f", velocity_precision_, det.velocity_x,
                          velocity_precision_, det.velocity_y);
            out = buf;
            return true;
        case FieldToken::Label:
            if (!det.has_label) return false;
            out = det.label;
            return true;
        case FieldToken::Cls:
            if (!det.has_cls) return false;
            std::snprintf(buf, sizeof(buf), "%d", det.cls);
            out = buf;
            return true;
        default:
            return false;
        }
    }

    bool buildLabelLines(const cuda_overlay::ParsedYoloDetection& det, std::vector<std::string>& lines_out) const {
        lines_out.clear();
        if (det.has_predicted && det.predicted && !show_predicted_labels_) return false;

        for (const TemplateLine& line : compiled_template_) {
            std::string formatted;
            bool valid_line = true;
            for (const TemplateToken& token : line) {
                if (token.is_literal) {
                    appendBounded(formatted, token.literal);
                } else {
                    std::string field_str;
                    if (!formatField(det, token.field, field_str)) {
                        valid_line = false;
                        break;
                    }
                    appendBounded(formatted, field_str);
                }
            }
            if (valid_line && !formatted.empty()) {
                lines_out.push_back(formatted);
                if ((int)lines_out.size() >= kMaxLabelLines) break;
            }
        }
        return !lines_out.empty();
    }

    void finalizeLayout(LabelLayout& layout, const cuda_overlay::ParsedYoloDetection& det,
                        const std::vector<std::string>& lines) const {
        layout.font_scale = font_scale_;
        layout.line_spacing = line_spacing_;
        layout.glyph_preset = (int)glyph_preset_;
        layout.line1.fill('\0');
        layout.line2.fill('\0');
        layout.line3.fill('\0');
        if (!lines.empty()) std::snprintf(layout.line1.data(), layout.line1.size(), "%s", lines[0].c_str());
        if (lines.size() > 1) std::snprintf(layout.line2.data(), layout.line2.size(), "%s", lines[1].c_str());
        if (lines.size() > 2) std::snprintf(layout.line3.data(), layout.line3.size(), "%s", lines[2].c_str());
        layout.line1_len = boundedStringLength(layout.line1);
        layout.line2_len = boundedStringLength(layout.line2);
        layout.line3_len = boundedStringLength(layout.line3);
        layout.line_count = (layout.line1_len > 0 ? 1 : 0) + (layout.line2_len > 0 ? 1 : 0) + (layout.line3_len > 0 ? 1 : 0);

        const int char_advance = std::max(1, cuda_overlay::glyphAdvance(glyph_preset_) * font_scale_);
        const int line_height = std::max(1, cuda_overlay::glyphBaseHeight(glyph_preset_) * font_scale_);
        const int max_line_len = std::max(layout.line1_len, std::max(layout.line2_len, layout.line3_len));
        const int text_w = max_line_len * char_advance;
        const int text_h = line_height * layout.line_count + (layout.line_count > 1 ? line_spacing_ * (layout.line_count - 1) : 0);
        const int pad_x = std::max(4, font_scale_);
        const int pad_y = std::max(3, font_scale_);

        int origin_x = det.x1 + offset_x_;
        int origin_y = det.y1 - text_h - 2 + offset_y_;
        if (origin_y < 0) {
            origin_y = det.y1 + 2 + offset_y_;
        }

        int bg_x = origin_x - pad_x;
        int bg_y = origin_y - pad_y;
        int bg_w = text_w + pad_x * 2;
        int bg_h = text_h + pad_y * 2;
        bg_w = std::max(1, std::min(bg_w, input_params_.width));
        bg_h = std::max(1, std::min(bg_h, input_params_.height));
        bg_x = std::max(0, std::min(bg_x, input_params_.width - bg_w));
        bg_y = std::max(0, std::min(bg_y, input_params_.height - bg_h));

        layout.bg_x = bg_x;
        layout.bg_y = bg_y;
        layout.bg_w = bg_w;
        layout.bg_h = bg_h;
        layout.origin_x = bg_x + pad_x;
        layout.origin_y = bg_y + pad_y;
    }

    bool drawLabelsOnFrame(av::VideoFrame& frm, const std::vector<LabelLayout>& labels) {
        if (labels.empty()) return true;
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
        batched_labels.reserve(labels.size());

        for (const LabelLayout& label : labels) {
            cuda_overlay::BatchedTextLabel batched;
            batched.line1_len = label.line1_len;
            batched.line2_len = label.line2_len;
            batched.line3_len = label.line3_len;
            batched.origin_x = label.origin_x;
            batched.origin_y = label.origin_y;
            batched.font_scale = label.font_scale;
            batched.line_spacing = label.line_spacing;
            batched.glyph_preset = label.glyph_preset;
            batched.bg_x = label.bg_x;
            batched.bg_y = label.bg_y;
            batched.bg_w = label.bg_w;
            batched.bg_h = label.bg_h;
            batched.draw_background = draw_background_ ? 1 : 0;
            batched.background_opacity = background_opacity_;
            batched.text_y = text_color_.y;
            batched.text_u = text_color_.u;
            batched.text_v = text_color_.v;
            batched.bg_y_color = background_color_.y;
            batched.bg_u = background_color_.u;
            batched.bg_v = background_color_.v;
            batched.line1_offset = cuda_overlay::appendTextBlob(text_blob, label.line1.data(), label.line1_len);
            batched.line2_offset = cuda_overlay::appendTextBlob(text_blob, label.line2.data(), label.line2_len);
            batched.line3_offset = cuda_overlay::appendTextBlob(text_blob, label.line3.data(), label.line3_len);
            batched_labels.push_back(batched);
        }

        if (!d_labels_.upload(batched_labels, cu_ctx_, cuda_dev_ctx_->stream)) {
            logstream << "draw_bbox_labels: failed uploading label descriptors";
            return false;
        }
        if (!d_text_blob_.uploadBytes(text_blob.data(), text_blob.size(), cu_ctx_, cuda_dev_ctx_->stream)) {
            logstream << "draw_bbox_labels: failed uploading label text blob";
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
            return false;
        }
        return CUDA_OVERLAY_CHECK_CU(cuStreamSynchronize(cuda_dev_ctx_->stream)) == 0;
    }

    void drawOnFrame(const av::VideoFrame& input, av::VideoFrame& output) override {
        if (!loadTextKernels()) {
            throw Error("draw_bbox_labels: failed to initialize text kernels");
        }

        const AVFrame* raw = input.raw();
        if (!raw || !raw->metadata) return;

        if (require_wide_shot_) {
            const std::string shot_type = readShotType(input);
            if (shot_type != "wide") {
                if (debug_log_every_n_ > 0 && (frame_counter_ % (uint64_t)debug_log_every_n_) == 0) {
                    logstream << "draw_bbox_labels: frame=" << frame_counter_
                              << " suppressed shot_type=" << (shot_type.empty() ? "<missing>" : shot_type);
                }
                return;
            }
        }

        cuda_overlay::YoloParseConfig cfg;
        cfg.frame_width = input_params_.width;
        cfg.frame_height = input_params_.height;
        cfg.min_conf = min_conf_;
        cfg.allowed_classes = &allowed_classes_;
        cfg.allowed_labels = &allowed_labels_;
        cfg.model_content_width = model_content_width_;
        cfg.model_content_height = model_content_height_;
        cfg.model_content_offset_x = model_content_offset_x_;
        cfg.model_content_offset_y = model_content_offset_y_;

        int labels_drawn = 0;
        std::vector<LabelLayout> labels_to_draw;
        for (const std::string& key : metadata_keys_) {
            AVDictionaryEntry* entry = av_dict_get(raw->metadata, key.c_str(), nullptr, 0);
            if (!entry || !entry->value) continue;
            try {
                const Parameters md = Parameters::parse(entry->value);
                std::vector<cuda_overlay::ParsedYoloDetection> detections;
                cuda_overlay::parseYoloDetections(md, cfg, detections);
                for (const auto& det : detections) {
                    std::vector<std::string> lines;
                    if (!buildLabelLines(det, lines)) continue;
                    LabelLayout label;
                    finalizeLayout(label, det, lines);
                    labels_to_draw.push_back(label);
                    labels_drawn++;
                }
            } catch (const std::exception&) {
                continue;
            }
        }

        if (!drawLabelsOnFrame(output, labels_to_draw)) {
            throw Error("draw_bbox_labels: failed to draw labels");
        }

        if (debug_log_every_n_ > 0 && (frame_counter_ % (uint64_t)debug_log_every_n_) == 0) {
            logstream << "draw_bbox_labels: frame=" << frame_counter_
                      << " labels=" << labels_drawn
                      << " preset=" << ((glyph_preset_ == cuda_overlay::GlyphPreset::k10x14) ? "10x14" : "5x7");
        }
    }

public:
    DrawBBoxLabels(std::unique_ptr<Source<av::VideoFrame>>&& source,
                   std::unique_ptr<Sink<av::VideoFrame>>&& sink,
                   std::vector<std::string> metadata_keys,
                   std::string camera_shot_metadata_key,
                   double min_conf,
                   std::unordered_set<int> allowed_classes,
                   std::unordered_set<std::string> allowed_labels,
                   double model_content_width,
                   double model_content_height,
                   double model_content_offset_x,
                   double model_content_offset_y,
                   std::string label_template,
                   int velocity_precision,
                   bool show_predicted_labels,
                   bool show_untracked,
                   bool require_wide_shot,
                   int font_scale,
                   int line_spacing,
                   int offset_x,
                   int offset_y,
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
          metadata_keys_(std::move(metadata_keys)),
          camera_shot_metadata_key_(std::move(camera_shot_metadata_key)),
          min_conf_(min_conf),
          allowed_classes_(std::move(allowed_classes)),
          allowed_labels_(std::move(allowed_labels)),
          model_content_width_(model_content_width),
          model_content_height_(model_content_height),
          model_content_offset_x_(model_content_offset_x),
          model_content_offset_y_(model_content_offset_y),
          label_template_(std::move(label_template)),
          velocity_precision_(velocity_precision),
          show_predicted_labels_(show_predicted_labels),
          show_untracked_(show_untracked),
          require_wide_shot_(require_wide_shot),
          font_scale_(font_scale),
          line_spacing_(line_spacing),
          offset_x_(offset_x),
          offset_y_(offset_y),
          glyph_preset_(glyph_preset),
          text_color_(text_color),
          background_color_(background_color),
          draw_background_(draw_background),
          background_opacity_(background_opacity),
          debug_log_every_n_(debug_log_every_n) {
        input_params_ = input_params;
        frame_rate_ = frame_rate;
        timebase_ = timebase;
        if (metadata_keys_.empty()) {
            throw Error("draw_bbox_labels: metadata_keys must be non-empty (or pass metadata_key)");
        }
        if (font_scale_ <= 0) {
            throw Error("draw_bbox_labels: font_scale must be positive");
        }
        if (line_spacing_ < 0) {
            throw Error("draw_bbox_labels: line_spacing must be non-negative");
        }
        if (velocity_precision_ < 0 || velocity_precision_ > 6) {
            throw Error("draw_bbox_labels: velocity_precision must be in [0,6]");
        }
        if (background_opacity_ < 0.f || background_opacity_ > 1.f) {
            throw Error("draw_bbox_labels: background_opacity must be in [0,1]");
        }
        std::string err;
        if (!compileTemplate(err)) {
            throw Error("draw_bbox_labels: " + err);
        }
    }

    static std::shared_ptr<DrawBBoxLabels> create(NodeCreationInfo& nci) {
        EdgeManager& edges = nci.edges;
        const Parameters& params = nci.params;

        auto src_edge = edges.find<av::VideoFrame>(params["src"]);
        const auto upstream = resolveUpstreamInfo(src_edge, params);

        std::vector<std::string> metadata_keys;
        if (params.count("metadata_keys") && params["metadata_keys"].is_array()) {
            for (const auto &item : params["metadata_keys"]) {
                if (!item.is_string()) {
                    throw Error("draw_bbox_labels: metadata_keys entries must be strings");
                }
                metadata_keys.push_back(item.get<std::string>());
            }
        }
        if (metadata_keys.empty()) {
            metadata_keys.push_back(params.value("metadata_key", std::string("yolo_players")));
        } else {
            std::unordered_set<std::string> seen;
            std::vector<std::string> unique_keys;
            unique_keys.reserve(metadata_keys.size());
            for (std::string &k : metadata_keys) {
                if (seen.insert(k).second) unique_keys.push_back(std::move(k));
            }
            metadata_keys = std::move(unique_keys);
        }

        const double min_conf = params.value("min_conf", 0.0);
        const std::string camera_shot_metadata_key = params.value("camera_shot_metadata_key", std::string("camera_shot_info"));
        const double model_content_width = params.value("model_content_width", 0.0);
        const double model_content_height = params.value("model_content_height", 0.0);
        const double model_content_offset_x = params.value("model_content_offset_x", 0.0);
        const double model_content_offset_y = params.value("model_content_offset_y", 0.0);
        const std::string label_template = params.value("label_template", std::string("ID:{track_id}\nV:{velocity}"));
        const int velocity_precision = params.value("velocity_precision", 1);
        const bool show_predicted_labels = params.value("show_predicted_labels", false);
        const bool show_untracked = params.value("show_untracked", false);
        const bool require_wide_shot = params.value("require_wide_shot", false);
        const int font_scale = params.value("font_scale", 2);
        const int line_spacing = params.value("line_spacing", 8);
        const int offset_x = params.value("offset_x", 0);
        const int offset_y = params.value("offset_y", 0);
        const int debug_log_every_n = params.value("debug_log_every_n", 0);
        const std::string glyph_preset_name = params.value("glyph_preset", std::string("10x14"));
        cuda_overlay::GlyphPreset glyph_preset = cuda_overlay::GlyphPreset::k10x14;
        if (!cuda_overlay::tryParseGlyphPreset(glyph_preset_name, glyph_preset)) {
            throw Error("draw_bbox_labels: glyph_preset must be one of: 5x7, 10x14");
        }

        std::unordered_set<int> allowed_classes;
        if (params.count("allowed_classes") && params["allowed_classes"].is_array()) {
            for (const auto& item : params["allowed_classes"]) {
                allowed_classes.insert(item.get<int>());
            }
        }
        std::unordered_set<std::string> allowed_labels;
        if (params.count("allowed_labels") && params["allowed_labels"].is_array()) {
            for (const auto& item : params["allowed_labels"]) {
                allowed_labels.insert(item.get<std::string>());
            }
        }

        DrawColor text_color{235, 128, 128};
        if (params.count("text_color")) {
            if (!params["text_color"].is_string() ||
                !cuda_overlay::tryParseNamedColor(params["text_color"].get<std::string>(), text_color)) {
                throw Error("draw_bbox_labels: text_color must be a supported color name");
            }
        }

        DrawColor background_color{16, 128, 128};
        bool draw_background = true;
        if (params.count("background_color")) {
            if (!params["background_color"].is_string()) {
                throw Error("draw_bbox_labels: background_color must be a string color name or \"none\"");
            }
            const std::string bg_name = cuda_overlay::normalizeColorName(params["background_color"].get<std::string>());
            if (bg_name == "none") {
                draw_background = false;
            } else if (!cuda_overlay::tryParseNamedColor(bg_name, background_color)) {
                throw Error("draw_bbox_labels: background_color must be a supported color name or none");
            }
        }

        float background_opacity = params.value("background_opacity", 0.75f);
        if (background_opacity < 0.f || background_opacity > 1.f) {
            throw Error("draw_bbox_labels: background_opacity must be in [0,1]");
        }

        return NodeSISO<av::VideoFrame, av::VideoFrame>::template createCommon<DrawBBoxLabels>(
            edges, params,
            std::move(metadata_keys),
            camera_shot_metadata_key,
            min_conf,
            std::move(allowed_classes),
            std::move(allowed_labels),
            model_content_width,
            model_content_height,
            model_content_offset_x,
            model_content_offset_y,
            label_template,
            velocity_precision,
            show_predicted_labels,
            show_untracked,
            require_wide_shot,
            font_scale,
            line_spacing,
            offset_x,
            offset_y,
            glyph_preset,
            text_color,
            background_color,
            draw_background,
            background_opacity,
            upstream.input_params,
            upstream.frame_rate,
            upstream.timebase,
            debug_log_every_n);
    }
};

DECLNODE(draw_bbox_labels, DrawBBoxLabels)
