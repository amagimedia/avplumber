#pragma once
// Capture identity travels with pixels, independently of the presentation timestamp.
#include <algorithm>
#include <charconv>
#include <cstdint>
#include <map>
#include <set>
#include <string>
#include <stdexcept>
#include <vector>
extern "C" {
#include <libavutil/dict.h>
#include <libavutil/frame.h>
}

namespace avp::mixer {
inline constexpr const char* sourcePrefix = "avp.source.";
inline constexpr const char* captureKey = "avp.capture";

struct SourceFrame {
    uint64_t epoch = 0, number = 0, period_ns = 0;
    std::string encode() const {
        return std::to_string(epoch) + ":" + std::to_string(number) + ":" + std::to_string(period_ns);
    }
    static SourceFrame decode(const std::string& text) {
        SourceFrame f;
        const char* pos = text.data();
        const char* end = pos + text.size();
        for (auto* value : {&f.epoch, &f.number, &f.period_ns}) {
            const auto result = std::from_chars(pos, end, *value);
            if (result.ec != std::errc()) throw std::runtime_error("invalid source frame identity");
            pos = result.ptr;
            if (value != &f.period_ns) {
                if (pos == end || *pos++ != ':') throw std::runtime_error("invalid source frame identity");
            }
        }
        if (pos != end) throw std::runtime_error("invalid source frame identity");
        return f;
    }
};
inline void setFrameTag(AVFrame* frame, const char* key, const std::string& value) {
    if (av_dict_set(&frame->metadata, key, value.c_str(), 0) < 0)
        throw std::runtime_error("cannot allocate source frame metadata");
}
inline void clearSourceFrames(AVFrame* frame) {
    while (auto* e = av_dict_get(frame->metadata, sourcePrefix, nullptr, AV_DICT_IGNORE_SUFFIX))
        av_dict_set(&frame->metadata, e->key, nullptr, 0);
}
inline void mergeSourceFrames(AVFrame* out, const AVFrame* in) {
    const AVDictionaryEntry* e = nullptr;
    while ((e = av_dict_get(in->metadata, sourcePrefix, e, AV_DICT_IGNORE_SUFFIX))) {
        const auto* before = av_dict_get(out->metadata, e->key, nullptr, 0);
        if (before && std::string(before->value) == e->value) continue;
        setFrameTag(out, e->key, before ? std::string(before->value) + ";" + e->value : e->value);
    }
}
using SourceFrames = std::map<std::string, std::vector<SourceFrame>>;
inline SourceFrames sourceFrames(const AVFrame* frame) {
    SourceFrames result;
    const AVDictionaryEntry* e = nullptr;
    while ((e = av_dict_get(frame->metadata, sourcePrefix, e, AV_DICT_IGNORE_SUFFIX))) {
        auto& frames = result[std::string(e->key).substr(std::char_traits<char>::length(sourcePrefix))];
        const std::string value(e->value);
        size_t start = 0;
        do {
            const auto end = value.find(';', start);
            frames.push_back(SourceFrame::decode(value.substr(start, end - start)));
            if (end == std::string::npos) break;
            start = end + 1;
        } while (true);
    }
    return result;
}

// One observation per published program frame. Slot warm-up, queue housekeeping and aliases
// do not call this. Going off air ends the comparison; re-entry establishes a fresh baseline.
class SourceDelivery {
public:
    struct Counts { uint64_t frames = 0, repeats = 0, dropped = 0, regressions = 0, static_holds = 0; };
private:
    struct Source {
        Counts counts;
        bool visible = false;
        uint64_t epoch = 0, high = 0;
        int64_t fresh_ns = 0, repeat_ns = 0;
    };
    std::map<std::string, Source> sources_;
public:
    void reset() { sources_.clear(); }
    std::map<std::string, Counts> counts() const {
        std::map<std::string, Counts> result;
        for (const auto& [name, s] : sources_) result[name] = s.counts;
        return result;
    }
    void observe(int64_t pts_ns, int64_t output_period_ns, const SourceFrames& frame) {
        for (auto& [name, s] : sources_)
            if (!frame.count(name)) s.visible = false;
        for (const auto& [name, captures] : frame) {
            if (captures.empty()) continue;
            auto& s = sources_[name];
            ++s.counts.frames;
            const auto newest = *std::max_element(captures.begin(), captures.end(), [](auto a, auto b) {
                return a.epoch < b.epoch || (a.epoch == b.epoch && a.number < b.number);
            });
            std::set<uint64_t> numbers;
            for (auto f : captures) if (f.epoch == newest.epoch) numbers.insert(f.number);
            if (!s.visible || s.epoch != newest.epoch) {
                s.visible = true; s.epoch = newest.epoch; s.high = newest.number;
                s.fresh_ns = s.repeat_ns = pts_ns;
                continue;
            }
            if (newest.number > s.high) {
                const auto seen = std::distance(numbers.upper_bound(s.high), numbers.end());
                const uint64_t gap = newest.number - s.high - seen;
                // A faster source intentionally skips pictures when sampled at the output rate.
                const uint64_t conversion = newest.period_ns && output_period_ns > int64_t(newest.period_ns)
                    ? (pts_ns - s.fresh_ns) / newest.period_ns - 1 : 0;
                s.counts.dropped += gap > conversion ? gap - conversion : 0;
                s.high = newest.number;
                s.fresh_ns = s.repeat_ns = pts_ns;
            } else {
                if (newest.number < s.high) ++s.counts.regressions;
                // Slower native-rate sources may legitimately span multiple output ticks.
                if (!newest.period_ns) ++s.counts.static_holds;
                else if (pts_ns - s.repeat_ns + 1 >= int64_t(newest.period_ns)) {
                    ++s.counts.repeats;
                    s.repeat_ns = pts_ns;
                }
            }
        }
    }
};
} // namespace avp::mixer
