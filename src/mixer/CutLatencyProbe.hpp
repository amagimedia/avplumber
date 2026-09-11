#pragma once

#include "CutLatency.hpp"
#include "../avutils.hpp"
#include <atomic>
#include <charconv>
#include <memory>

namespace avp::mixer {

struct CutLatencyProbe {
    CutLatency timing;
    const std::string metadata_key;
    const std::string encoder_name;

    CutLatencyProbe(std::string mixer, std::string encoder)
        : metadata_key("avp.cut." + mixer), encoder_name(std::move(encoder)) {}

    Parameters status() {
        const auto samples = timing.snapshot();
        auto json = [](const CutLatency::Sample& sample) {
            Parameters value = {{"id", sample.id}, {"scene", sample.scene}, {"state", sample.state}};
            value["ms"] = sample.milliseconds ? Parameters(*sample.milliseconds) : Parameters(nullptr);
            value["encoded_pts"] = sample.encoded_pts ? Parameters(*sample.encoded_pts) : Parameters(nullptr);
            return value;
        };
        auto withRecent = [&](const CutLatency::Sample& sample, const std::vector<CutLatency::Sample>& recent) {
            auto value = json(sample);
            value["recent"] = Parameters::array();
            for (const auto& entry : recent) value["recent"].push_back(json(entry));
            return value;
        };
        return {{"endpoint", "encoder_output"}, {"encoder", encoder_name},
                {"direct", withRecent(samples.direct, samples.direct_recent)},
                {"previewed", withRecent(samples.previewed, samples.previewed_recent)}};
    }

    uint64_t frameToken(const av::VideoFrame& frame) const {
        if (!frame.raw()) return 0;
        auto entry = av_dict_get(frame.raw()->metadata, metadata_key.c_str(), nullptr, 0);
        if (!entry) return 0;
        uint64_t token = 0;
        const char* end = entry->value + std::char_traits<char>::length(entry->value);
        const auto parsed = std::from_chars(entry->value, end, token);
        return parsed.ec == std::errc{} && parsed.ptr == end ? token : 0;
    }
};

// Opt-in observer binding, not a graph routing or media-processing interface.
// Atomic shared_ptr access allows installation on an already-created node.
class CutLatencyObserver {
    std::shared_ptr<CutLatencyProbe> probe_;
public:
    virtual ~CutLatencyObserver() = default;
    std::shared_ptr<CutLatencyProbe> cutLatencyProbe() const { return std::atomic_load(&probe_); }
    void setCutLatencyProbe(std::shared_ptr<CutLatencyProbe> probe) { std::atomic_store(&probe_, std::move(probe)); }
};

} // namespace avp::mixer
