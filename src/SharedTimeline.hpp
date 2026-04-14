#pragma once
#include "instance_shared.hpp"
#include "util.hpp"
#include "graph_core.hpp"
#include <mutex>
#include <unordered_map>
#include <vector>
#include <string>
#include <optional>
#include <algorithm>

class SharedTimeline : public InstanceShared<SharedTimeline> {
    struct Entry {
        int64_t at_pts_ms;
        Parameters value;
    };

    // channel -> key -> sorted vector of {at_pts_ms, value}
    std::unordered_map<std::string,
        std::unordered_map<std::string, std::vector<Entry>>> data_;
    mutable std::mutex mutex_;

    static void upsertEntry(std::vector<Entry>& entries, int64_t at_pts_ms, const Parameters& value) {
        for (auto& e : entries) {
            if (e.at_pts_ms == at_pts_ms) {
                e.value = value;
                return;
            }
        }
        entries.push_back({at_pts_ms, value});
        std::sort(entries.begin(), entries.end(),
                  [](const Entry& a, const Entry& b) { return a.at_pts_ms < b.at_pts_ms; });
    }

public:
    struct BatchEntry {
        std::string channel;
        std::string key;
        int64_t at_pts_ms;
        Parameters value;
    };

    void set(const std::string& channel, const std::string& key,
             int64_t at_pts_ms, const Parameters& value) {
        std::lock_guard<std::mutex> lock(mutex_);
        upsertEntry(data_[channel][key], at_pts_ms, value);
    }

    void setBatch(const std::vector<BatchEntry>& batch) {
        std::lock_guard<std::mutex> lock(mutex_);
        for (const auto& entry : batch)
            upsertEntry(data_[entry.channel][entry.key], entry.at_pts_ms, entry.value);
    }

    // Returns the value from the latest entry with at_pts_ms <= frame_pts,
    // using av_compare_ts so frame timebase doesn't matter.
    std::optional<Parameters> get(const std::string& channel,
                                   const std::string& key,
                                   const av::Timestamp& frame_pts) const {
        std::lock_guard<std::mutex> lock(mutex_);
        auto ch_it = data_.find(channel);
        if (ch_it == data_.end()) return std::nullopt;
        auto key_it = ch_it->second.find(key);
        if (key_it == ch_it->second.end()) return std::nullopt;
        const auto& entries = key_it->second;
        std::optional<Parameters> result;
        for (const auto& e : entries) {
            av::Timestamp entry_ts(e.at_pts_ms, {1, 1000});
            if (frame_pts >= entry_ts)
                result = e.value;
            else
                break;
        }
        return result;
    }

    template<typename T>
    T getOr(const std::string& channel, const std::string& key,
            const av::Timestamp& frame_pts, const T& default_val) const {
        auto opt = get(channel, key, frame_pts);
        if (!opt) return default_val;
        return opt->get<T>();
    }

    void clear(const std::string& channel) {
        std::lock_guard<std::mutex> lock(mutex_);
        data_.erase(channel);
    }

    void clearAll() {
        std::lock_guard<std::mutex> lock(mutex_);
        data_.clear();
    }

    // GC: keep only the latest entry before threshold plus all entries at/after threshold
    void gc(int64_t before_pts_ms) {
        std::lock_guard<std::mutex> lock(mutex_);
        for (auto& [ch, keys] : data_) {
            for (auto& [key, entries] : keys) {
                auto it = entries.begin();
                while (it != entries.end() && std::next(it) != entries.end()
                       && std::next(it)->at_pts_ms <= before_pts_ms) {
                    it = entries.erase(it);
                }
            }
        }
    }

    Parameters dump() const {
        std::lock_guard<std::mutex> lock(mutex_);
        Parameters result = Parameters::object();
        for (const auto& [ch, keys] : data_) {
            Parameters ch_obj = Parameters::object();
            for (const auto& [key, entries] : keys) {
                Parameters arr = Parameters::array();
                for (const auto& e : entries) {
                    Parameters entry_obj;
                    entry_obj["at"] = e.at_pts_ms;
                    entry_obj["val"] = e.value;
                    arr.push_back(std::move(entry_obj));
                }
                ch_obj[key] = std::move(arr);
            }
            result[ch] = std::move(ch_obj);
        }
        return result;
    }
};


inline uint32_t parseBitmask(const Parameters& value) {
    if (value.is_string()) {
        uint32_t mask = 0;
        auto s = value.get<std::string>();
        for (size_t i = 0; i < s.size(); i++)
            if (s[i] == '1') mask |= (1u << i);
        return mask;
    }
    return value.get<uint32_t>();
}

class TimelineReader {
protected:
    std::shared_ptr<SharedTimeline> timeline_;
    std::string channel_;

    void initTimeline(NodeCreationInfo& nci) {
        if (nci.params.count("timeline")) {
            timeline_ = InstanceSharedObjects<SharedTimeline>::get(
                nci.instance, nci.params["timeline"].get<std::string>());
            if (nci.params.count("name")) {
                channel_ = nci.params["name"].get<std::string>();
            }
        }
    }

    template<typename T>
    T tlGet(const std::string& key, const av::Timestamp& pts, const T& fallback) const {
        if (!timeline_) return fallback;
        return timeline_->getOr<T>(channel_, key, pts, fallback);
    }

    std::optional<Parameters> tlGetRaw(const std::string& key, const av::Timestamp& pts) const {
        if (!timeline_) return std::nullopt;
        return timeline_->get(channel_, key, pts);
    }

    bool hasTimeline() const { return timeline_ != nullptr; }
};
