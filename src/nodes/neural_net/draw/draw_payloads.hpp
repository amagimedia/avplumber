#pragma once
// What the draw nodes read from a frame and from their parameters before anything reaches the GPU:
// the frame's JSON metadata, each key parsed once however many layers read it, and the parameters
// the nodes share.
#include "cuda_overlay_base.hpp"

#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace cuda_overlay {

/// The JSON payloads of one frame's metadata. A key that is absent or is not JSON has no payload.
class FramePayloads {
    const AVFrame* raw_;
    std::unordered_map<std::string, std::optional<Parameters>> parsed_;

public:
    explicit FramePayloads(const AVFrame* raw) : raw_(raw) {}

    bool hasMetadata() const { return raw_ && raw_->metadata; }

    const Parameters* get(const std::string& key) {
        auto found = parsed_.find(key);
        if (found == parsed_.end()) {
            std::optional<Parameters> payload;
            if (hasMetadata()) {
                AVDictionaryEntry* entry = av_dict_get(raw_->metadata, key.c_str(), nullptr, 0);
                if (entry && entry->value) {
                    try {
                        payload = Parameters::parse(entry->value);
                    } catch (const std::exception&) {
                    }
                }
            }
            found = parsed_.emplace(key, std::move(payload)).first;
        }
        return found->second ? &*found->second : nullptr;
    }

    /// `camera_shot_type` of the payload under `key`; empty when there is none.
    std::string shotType(const std::string& key) {
        if (key.empty()) return {};
        const Parameters* md = get(key);
        if (!md) return {};
        try {
            return md->value("camera_shot_type", std::string());
        } catch (const std::exception&) {
            return {};
        }
    }
};

/// `metadata_keys` without repeats, or the single `metadata_key` (default `fallback`).
inline std::vector<std::string> metadataKeysParam(const Parameters& params, const std::string& fallback,
                                                  const std::string& node) {
    std::vector<std::string> keys;
    if (params.count("metadata_keys") && params["metadata_keys"].is_array()) {
        std::unordered_set<std::string> seen;
        for (const auto& item : params["metadata_keys"]) {
            if (!item.is_string()) throw Error(node + ": metadata_keys entries must be strings");
            std::string key = item.get<std::string>();
            if (seen.insert(key).second) keys.push_back(std::move(key));
        }
    }
    if (keys.empty()) keys.push_back(params.value("metadata_key", fallback));
    return keys;
}

/// The detection filters and the model-content remap the detection-reading nodes share.
struct DetectionFilter {
    double min_conf = 0.0;
    std::unordered_set<int> allowed_classes;
    std::unordered_set<std::string> allowed_labels;
    double model_content_width = 0.0;
    double model_content_height = 0.0;
    double model_content_offset_x = 0.0;
    double model_content_offset_y = 0.0;

    static DetectionFilter fromParams(const Parameters& params) {
        DetectionFilter filter;
        filter.min_conf = params.value("min_conf", 0.0);
        filter.model_content_width = params.value("model_content_width", 0.0);
        filter.model_content_height = params.value("model_content_height", 0.0);
        filter.model_content_offset_x = params.value("model_content_offset_x", 0.0);
        filter.model_content_offset_y = params.value("model_content_offset_y", 0.0);
        if (params.count("allowed_classes") && params["allowed_classes"].is_array()) {
            for (const auto& item : params["allowed_classes"]) filter.allowed_classes.insert(item.get<int>());
        }
        if (params.count("allowed_labels") && params["allowed_labels"].is_array()) {
            for (const auto& item : params["allowed_labels"]) filter.allowed_labels.insert(item.get<std::string>());
        }
        return filter;
    }

    YoloParseConfig config(const FrameGeometry& frame) const {
        YoloParseConfig cfg;
        cfg.frame_width = frame.width;
        cfg.frame_height = frame.height;
        cfg.min_conf = min_conf;
        cfg.allowed_classes = &allowed_classes;
        cfg.allowed_labels = &allowed_labels;
        cfg.model_content_width = model_content_width;
        cfg.model_content_height = model_content_height;
        cfg.model_content_offset_x = model_content_offset_x;
        cfg.model_content_offset_y = model_content_offset_y;
        return cfg;
    }
};

} // namespace cuda_overlay
