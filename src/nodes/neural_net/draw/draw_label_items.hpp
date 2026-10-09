#pragma once
// draw_bbox_labels' reading of a frame: one text label per detection, formatted from the node's
// template and laid out beside its box.
#include "draw_batch_shared.hpp"
#include "draw_payloads.hpp"

#include <array>
#include <cstdio>

namespace cuda_overlay {

class LabelItems {
    static constexpr int kMaxLabelChars = 64;
    static constexpr int kMaxGpuChars = 96;
    static constexpr int kMaxLabelLines = 3;

    enum class FieldToken { TrackId, Conf, Velocity, Label, Cls };

    struct TemplateToken {
        bool is_literal = true;
        std::string literal;
        FieldToken field = FieldToken::TrackId;
    };
    using TemplateLine = std::vector<TemplateToken>;

    std::vector<std::string> metadata_keys_;
    std::string camera_shot_metadata_key_ = "camera_shot_info";
    DetectionFilter filter_;
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
    GlyphPreset glyph_preset_ = GlyphPreset::k10x14;

    DrawColor text_color_{235, 128, 128};
    DrawColor background_color_{16, 128, 128};
    bool draw_background_ = true;
    float background_opacity_ = 0.75f;
    std::string node_;

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
        if (token == "track_id") out_field = FieldToken::TrackId;
        else if (token == "conf") out_field = FieldToken::Conf;
        else if (token == "velocity") out_field = FieldToken::Velocity;
        else if (token == "label") out_field = FieldToken::Label;
        else if (token == "cls") out_field = FieldToken::Cls;
        else return false;
        return true;
    }

    static bool compileTemplateLine(const std::string& line, TemplateLine& out_line, std::string& err_out) {
        out_line.clear();
        std::string literal;
        const auto flushLiteral = [&]() {
            if (literal.empty()) return;
            TemplateToken token;
            token.is_literal = true;
            token.literal = literal;
            out_line.push_back(std::move(token));
            literal.clear();
        };
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
            flushLiteral();
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
        flushLiteral();
        return true;
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

    bool formatField(const ParsedYoloDetection& det, FieldToken field, std::string& out) const {
        char buf[96];
        switch (field) {
        case FieldToken::TrackId:
            if (!det.has_track_id) return false;
            if (det.track_id < 0 && !show_untracked_) return false;
            std::snprintf(buf, sizeof(buf), "%d", det.track_id);
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

    bool buildLabelLines(const ParsedYoloDetection& det, std::vector<std::string>& lines_out) const {
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

    /// The label of `det` with its lines appended to `blob`: the text sits above the box, or below
    /// its top edge when it would leave the frame, and its background is clamped into the frame.
    BatchedTextLabel layout(const ParsedYoloDetection& det, const std::vector<std::string>& lines,
                            const FrameGeometry& frame, std::vector<char>& blob) const {
        std::array<std::array<char, kMaxGpuChars>, kMaxLabelLines> text{};
        std::array<int, kMaxLabelLines> len{};
        for (size_t i = 0; i < lines.size() && i < (size_t)kMaxLabelLines; ++i) {
            std::snprintf(text[i].data(), text[i].size(), "%s", lines[i].c_str());
            while (len[i] < (int)text[i].size() && text[i][(size_t)len[i]] != '\0') ++len[i];
        }
        const int line_count = (len[0] > 0 ? 1 : 0) + (len[1] > 0 ? 1 : 0) + (len[2] > 0 ? 1 : 0);

        const int char_advance = std::max(1, glyphAdvance(glyph_preset_) * font_scale_);
        const int line_height = std::max(1, glyphBaseHeight(glyph_preset_) * font_scale_);
        const int max_line_len = std::max(len[0], std::max(len[1], len[2]));
        const int text_w = max_line_len * char_advance;
        const int text_h = line_height * line_count + (line_count > 1 ? line_spacing_ * (line_count - 1) : 0);
        const int pad_x = std::max(4, font_scale_);
        const int pad_y = std::max(3, font_scale_);

        const int origin_x = det.x1 + offset_x_;
        int origin_y = det.y1 - text_h - 2 + offset_y_;
        if (origin_y < 0) {
            origin_y = det.y1 + 2 + offset_y_;
        }

        int bg_x = origin_x - pad_x;
        int bg_y = origin_y - pad_y;
        int bg_w = text_w + pad_x * 2;
        int bg_h = text_h + pad_y * 2;
        bg_w = std::max(1, std::min(bg_w, frame.width));
        bg_h = std::max(1, std::min(bg_h, frame.height));
        bg_x = std::max(0, std::min(bg_x, frame.width - bg_w));
        bg_y = std::max(0, std::min(bg_y, frame.height - bg_h));

        BatchedTextLabel label;
        label.line1_len = len[0];
        label.line2_len = len[1];
        label.line3_len = len[2];
        label.origin_x = bg_x + pad_x;
        label.origin_y = bg_y + pad_y;
        label.font_scale = font_scale_;
        label.line_spacing = line_spacing_;
        label.glyph_preset = (int)glyph_preset_;
        label.bg_x = bg_x;
        label.bg_y = bg_y;
        label.bg_w = bg_w;
        label.bg_h = bg_h;
        label.draw_background = draw_background_ ? 1 : 0;
        label.background_opacity = background_opacity_;
        label.text_y = text_color_.y;
        label.text_u = text_color_.u;
        label.text_v = text_color_.v;
        label.bg_y_color = background_color_.y;
        label.bg_u = background_color_.u;
        label.bg_v = background_color_.v;
        label.line1_offset = appendTextBlob(blob, text[0].data(), len[0]);
        label.line2_offset = appendTextBlob(blob, text[1].data(), len[1]);
        label.line3_offset = appendTextBlob(blob, text[2].data(), len[2]);
        return label;
    }

public:
    static LabelItems fromParams(const Parameters& params, const std::string& node) {
        LabelItems items;
        items.node_ = node;
        items.metadata_keys_ = metadataKeysParam(params, "yolo_players", node);
        items.filter_ = DetectionFilter::fromParams(params);
        items.camera_shot_metadata_key_ = params.value("camera_shot_metadata_key", std::string("camera_shot_info"));
        items.label_template_ = params.value("label_template", std::string("ID:{track_id}\nV:{velocity}"));
        items.velocity_precision_ = params.value("velocity_precision", 1);
        items.show_predicted_labels_ = params.value("show_predicted_labels", false);
        items.show_untracked_ = params.value("show_untracked", false);
        items.require_wide_shot_ = params.value("require_wide_shot", false);
        items.font_scale_ = params.value("font_scale", 2);
        items.line_spacing_ = params.value("line_spacing", 8);
        items.offset_x_ = params.value("offset_x", 0);
        items.offset_y_ = params.value("offset_y", 0);
        items.debug_log_every_n_ = params.value("debug_log_every_n", 0);
        if (!tryParseGlyphPreset(params.value("glyph_preset", std::string("10x14")), items.glyph_preset_)) {
            throw Error(node + ": glyph_preset must be one of: 5x7, 10x14");
        }
        if (params.count("text_color")) {
            if (!params["text_color"].is_string() ||
                !tryParseNamedColor(params["text_color"].get<std::string>(), items.text_color_)) {
                throw Error(node + ": text_color must be a supported color name");
            }
        }
        if (params.count("background_color")) {
            if (!params["background_color"].is_string()) {
                throw Error(node + ": background_color must be a string color name or \"none\"");
            }
            const std::string bg_name = normalizeColorName(params["background_color"].get<std::string>());
            if (bg_name == "none") {
                items.draw_background_ = false;
            } else if (!tryParseNamedColor(bg_name, items.background_color_)) {
                throw Error(node + ": background_color must be a supported color name or none");
            }
        }
        items.background_opacity_ = params.value("background_opacity", 0.75f);
        if (items.background_opacity_ < 0.f || items.background_opacity_ > 1.f) {
            throw Error(node + ": background_opacity must be in [0,1]");
        }
        if (items.metadata_keys_.empty()) {
            throw Error(node + ": metadata_keys must be non-empty (or pass metadata_key)");
        }
        if (items.font_scale_ <= 0) {
            throw Error(node + ": font_scale must be positive");
        }
        if (items.line_spacing_ < 0) {
            throw Error(node + ": line_spacing must be non-negative");
        }
        if (items.velocity_precision_ < 0 || items.velocity_precision_ > 6) {
            throw Error(node + ": velocity_precision must be in [0,6]");
        }
        std::string err;
        if (!items.compileTemplate(err)) {
            throw Error(node + ": " + err);
        }
        return items;
    }

    /// Appends the frame's labels to `labels` and their text to `blob`, in key and detection order.
    void collect(FramePayloads& payloads, const FrameGeometry& frame, uint64_t frame_counter,
                 std::vector<BatchedTextLabel>& labels, std::vector<char>& blob) const {
        if (!payloads.hasMetadata()) return;
        const bool log = debug_log_every_n_ > 0 && (frame_counter % (uint64_t)debug_log_every_n_) == 0;

        if (require_wide_shot_) {
            const std::string shot_type = payloads.shotType(camera_shot_metadata_key_);
            if (shot_type != "wide") {
                if (log) {
                    logstream << node_ << ": frame=" << frame_counter
                              << " suppressed shot_type=" << (shot_type.empty() ? "<missing>" : shot_type);
                }
                return;
            }
        }

        const YoloParseConfig cfg = filter_.config(frame);
        const size_t before = labels.size();
        for (const std::string& key : metadata_keys_) {
            const Parameters* md = payloads.get(key);
            if (!md) continue;
            try {
                std::vector<ParsedYoloDetection> detections;
                parseYoloDetections(*md, cfg, detections);
                for (const auto& det : detections) {
                    std::vector<std::string> lines;
                    if (!buildLabelLines(det, lines)) continue;
                    labels.push_back(layout(det, lines, frame, blob));
                }
            } catch (const std::exception&) {
                continue;
            }
        }

        if (log) {
            logstream << node_ << ": frame=" << frame_counter
                      << " labels=" << (labels.size() - before)
                      << " preset=" << ((glyph_preset_ == GlyphPreset::k10x14) ? "10x14" : "5x7");
        }
    }
};

} // namespace cuda_overlay
