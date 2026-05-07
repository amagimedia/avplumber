#pragma once

#include "fabric_protocol.hpp"

#include <algorithm>
#include <deque>
#include <functional>
#include <limits>
#include <map>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace avp_fabric {

struct SourceCandidate {
    MediaHeader media = {};
    std::vector<uint8_t> payload;
    int64_t normalized_frame_id = 0;
};

class SourceTimeline {
public:
    using SendControl = std::function<void(uint32_t replica_id, const std::string &cmd, int64_t normalized_frame_id)>;

private:
    struct FrameSlot {
        std::unordered_map<uint32_t, SourceCandidate> media_by_replica;
        std::unordered_map<uint32_t, int64_t> status_raw_by_replica;
        bool repair_requested = false;
        uint64_t repair_requested_ns = 0;
        bool promote_sent = false;
    };

    struct ReplicaState {
        std::deque<std::pair<uint64_t, int64_t>> history;
        std::deque<int64_t> offset_samples;
        uint64_t generation = 0;
        bool generation_valid = false;
        int64_t offset = 0;
        bool offset_valid = false;
        bool eligible = false;
        uint64_t aligned_frames = 0;
        uint64_t last_seen_ns = 0;
        int64_t highest_seen_frame = std::numeric_limits<int64_t>::min();
    };

    std::string redundancy_mode_ = "single";
    uint32_t preferred_active_replica_id_ = 1;
    uint32_t current_active_replica_id_ = 1;
    uint64_t playout_delay_frames_ = 0;
    uint64_t promote_after_misses_ = 1;
    uint64_t active_timeout_ms_ = 50;
    uint64_t repair_grace_ns_ = 0;
    bool strict_frame_identity_ = false;

    static constexpr size_t alignment_window_frames_ = 50;
    static constexpr size_t alignment_required_frames_ = 10;
    static constexpr uint64_t alignment_max_wallclock_delta_ns_ = 120000000ull;

    std::map<int64_t, FrameSlot> frame_slots_;
    std::unordered_map<uint32_t, ReplicaState> replicas_;
    bool next_emit_valid_ = false;
    int64_t next_emit_frame_ = 0;
    int64_t highest_seen_frame_ = std::numeric_limits<int64_t>::min();
    uint64_t active_miss_count_ = 0;
    uint64_t last_active_media_ns_ = 0;
    bool standby_promoted_ = false;
    bool strict_redundancy_established_ = false;

public:
    SourceTimeline() = default;

    SourceTimeline(std::string redundancy_mode,
                   uint32_t active_replica_id,
                   uint64_t playout_delay_frames,
                   uint64_t promote_after_misses,
                   uint64_t active_timeout_ms,
                   uint64_t repair_grace_ms,
                   bool strict_frame_identity):
        redundancy_mode_(std::move(redundancy_mode)),
        preferred_active_replica_id_(active_replica_id),
        current_active_replica_id_(active_replica_id),
        playout_delay_frames_(playout_delay_frames),
        promote_after_misses_(promote_after_misses),
        active_timeout_ms_(active_timeout_ms),
        repair_grace_ns_(repair_grace_ms * 1000000ull),
        strict_frame_identity_(strict_frame_identity) {}

    bool isSingle() const {
        return redundancy_mode_ == "single";
    }

    int64_t ingestMedia(const MediaHeader &media,
                        const uint8_t *payload,
                        size_t payload_bytes,
                        uint64_t received_ns) {
        handleGeneration(media);
        const int64_t normalized_frame_id = normalizeFrameId(media, received_ns);
        highest_seen_frame_ = std::max(highest_seen_frame_, normalized_frame_id);
        if (!next_emit_valid_) {
            next_emit_valid_ = true;
            next_emit_frame_ = normalized_frame_id;
        }

        SourceCandidate candidate;
        candidate.media = media;
        candidate.payload.assign(payload, payload + payload_bytes);
        candidate.normalized_frame_id = normalized_frame_id;
        frame_slots_[normalized_frame_id].media_by_replica[media.replica_id] = std::move(candidate);

        if (media.replica_id == current_active_replica_id_) {
            last_active_media_ns_ = received_ns;
        }

        return normalized_frame_id;
    }

    int64_t ingestStatus(const MediaHeader &media, uint64_t received_ns, const SendControl &send_control) {
        handleGeneration(media);
        const int64_t normalized_frame_id = normalizeFrameId(media, received_ns);
        highest_seen_frame_ = std::max(highest_seen_frame_, normalized_frame_id);
        FrameSlot &slot = frame_slots_[normalized_frame_id];
        slot.status_raw_by_replica[media.replica_id] = media.pts;
        maybePromoteOnActiveTimeout(media, normalized_frame_id, send_control);
        return normalized_frame_id;
    }

    bool trySelect(SourceCandidate &selected, const SendControl &send_control) {
        if (!next_emit_valid_) return false;
        const int64_t watermark = strict_frame_identity_ ? strictWatermark() : highest_seen_frame_;
        if (watermark == std::numeric_limits<int64_t>::min()) return false;
        if (watermark < next_emit_frame_ + static_cast<int64_t>(playout_delay_frames_)) return false;

        auto it = frame_slots_.find(next_emit_frame_);
        if (it == frame_slots_.end()) {
            it = frame_slots_.lower_bound(next_emit_frame_);
            if (it == frame_slots_.end()) return false;
            logstream << "redundancy_selector skipped missing frame slot"
                      << " from_normalized_pts=" << next_emit_frame_
                      << " to_normalized_pts=" << it->first;
            next_emit_frame_ = it->first;
        }
        FrameSlot &slot = it->second;

        if (redundancy_mode_ == "hot_hot_first_complete") {
            if (!slot.media_by_replica.empty()) {
                selected = slot.media_by_replica.begin()->second;
                finishSelectedFrame(it);
                return true;
            }
            return false;
        }

        auto active = slot.media_by_replica.find(current_active_replica_id_);
        if (active != slot.media_by_replica.end()) {
            selected = active->second;
            active_miss_count_ = 0;
            finishSelectedFrame(it);
            return true;
        }

        auto active_status = slot.status_raw_by_replica.find(current_active_replica_id_);
        if (active_status != slot.status_raw_by_replica.end()) {
            if (!slot.repair_requested) {
                send_control(current_active_replica_id_, "REPAIR", next_emit_frame_);
                slot.repair_requested = true;
                slot.repair_requested_ns = monotonicNs();
                return false;
            }
            if (withinRepairGrace(slot)) return false;
            logstream << "redundancy_selector skipped active status-only frame slot"
                      << " replica_id=" << current_active_replica_id_
                      << " normalized_pts=" << next_emit_frame_;
            finishSelectedFrame(it);
            return false;
        }

        for (auto candidate_it = slot.media_by_replica.begin(); candidate_it != slot.media_by_replica.end(); ++candidate_it) {
            const uint32_t replica_id = candidate_it->first;
            if (replica_id == current_active_replica_id_) continue;
            if (!replicaEligible(replica_id)) continue;
            selected = candidate_it->second;
            logstream << "redundancy_selector elected active replica_id=" << replica_id
                      << " previous_active_replica_id=" << current_active_replica_id_
                      << " normalized_pts=" << next_emit_frame_;
            current_active_replica_id_ = replica_id;
            last_active_media_ns_ = monotonicNs();
            active_miss_count_ = 0;
            standby_promoted_ = false;
            finishSelectedFrame(it);
            return true;
        }

        for (const auto &status: slot.status_raw_by_replica) {
            const uint32_t replica_id = status.first;
            if (replica_id == current_active_replica_id_) continue;
            if (!replicaEligible(replica_id)) continue;
            if (!slot.repair_requested) {
                send_control(replica_id, "REPAIR", next_emit_frame_);
                slot.repair_requested = true;
                slot.repair_requested_ns = monotonicNs();
                active_miss_count_++;
                if (active_miss_count_ >= promote_after_misses_ && !slot.promote_sent) {
                    send_control(replica_id, "PROMOTE", next_emit_frame_ + 1);
                    slot.promote_sent = true;
                }
            } else if (!withinRepairGrace(slot)) {
                send_control(replica_id, "REPAIR", next_emit_frame_);
                slot.repair_requested_ns = monotonicNs();
            }
            break;
        }
        return false;
    }

    int64_t denormalizeFrameId(uint32_t replica_id, int64_t normalized_frame_id) const {
        auto it = replicas_.find(replica_id);
        if (it == replicas_.end() || !it->second.offset_valid) return normalized_frame_id;
        const int64_t raw = normalized_frame_id - it->second.offset;
        if (raw < 0) return normalized_frame_id;
        return raw;
    }

private:
    void finishSelectedFrame(std::map<int64_t, FrameSlot>::iterator selected_it) {
        frame_slots_.erase(frame_slots_.begin(), std::next(selected_it));
        if (frame_slots_.empty()) {
            next_emit_valid_ = false;
            return;
        }
        next_emit_frame_ = frame_slots_.begin()->first;
    }

    void maybePromoteOnActiveTimeout(const MediaHeader &media, int64_t normalized_frame_id, const SendControl &send_control) {
        if (redundancy_mode_ != "active_standby_repair") return;
        if (media.replica_id == current_active_replica_id_) return;
        if (!replicaEligible(media.replica_id)) return;
        if (standby_promoted_) return;
        if (!last_active_media_ns_) return;

        const uint64_t now = monotonicNs();
        const uint64_t timeout_ns = active_timeout_ms_ * 1000000ull;
        if (now - last_active_media_ns_ < timeout_ns) return;

        send_control(media.replica_id, "PROMOTE", normalized_frame_id);
        standby_promoted_ = true;
    }

    bool withinRepairGrace(const FrameSlot &slot) const {
        if (!repair_grace_ns_ || !slot.repair_requested_ns) return false;
        return monotonicNs() - slot.repair_requested_ns < repair_grace_ns_;
    }

    int64_t normalizeFrameId(const MediaHeader &media, uint64_t received_ns) {
        ReplicaState &state = replicas_[media.replica_id];
        state.history.emplace_back(received_ns, media.pts);
        while (state.history.size() > alignment_window_frames_) state.history.pop_front();
        state.last_seen_ns = received_ns;

        if (media.replica_id == preferred_active_replica_id_) {
            state.offset = 0;
            state.offset_valid = true;
            state.aligned_frames++;
            if (!state.eligible && state.aligned_frames >= alignment_required_frames_) {
                state.eligible = true;
                logstream << "redundancy_selector replica eligible replica_id=" << media.replica_id
                          << " generation=" << media.generation
                          << " offset=0";
            }
            state.highest_seen_frame = std::max(state.highest_seen_frame, media.pts);
            return media.pts;
        }

        if (strict_frame_identity_) {
            state.offset = 0;
            state.offset_valid = true;
            state.aligned_frames++;
            if (!state.eligible && state.aligned_frames >= alignment_required_frames_) {
                state.eligible = true;
                logstream << "redundancy_selector replica eligible replica_id=" << media.replica_id
                          << " generation=" << media.generation
                          << " offset=0 strict_frame_identity=1";
            }
            state.highest_seen_frame = std::max(state.highest_seen_frame, media.pts);
            return media.pts;
        }

        learnOffset(media, received_ns, state);
        if (!state.offset_valid) return media.pts;
        const int64_t normalized = media.pts + state.offset;
        if (normalized < 0) return media.pts;
        state.highest_seen_frame = std::max(state.highest_seen_frame, normalized);
        return normalized;
    }

    void learnOffset(const MediaHeader &media, uint64_t received_ns, ReplicaState &state) {
        auto active_it = replicas_.find(preferred_active_replica_id_);
        if (active_it == replicas_.end() || active_it->second.history.empty()) return;

        uint64_t best_dt = std::numeric_limits<uint64_t>::max();
        int64_t best_active_pts = AV_NOPTS_VALUE;
        for (const auto &sample: active_it->second.history) {
            const uint64_t sample_ns = sample.first;
            const uint64_t dt = (sample_ns > received_ns) ? sample_ns - received_ns : received_ns - sample_ns;
            if (dt < best_dt) {
                best_dt = dt;
                best_active_pts = sample.second;
            }
        }
        if (best_active_pts == AV_NOPTS_VALUE || best_dt > alignment_max_wallclock_delta_ns_) return;

        state.offset_samples.push_back(best_active_pts - media.pts);
        while (state.offset_samples.size() > alignment_window_frames_) state.offset_samples.pop_front();
        if (state.offset_samples.size() < alignment_required_frames_) return;

        std::vector<int64_t> samples(state.offset_samples.begin(), state.offset_samples.end());
        std::sort(samples.begin(), samples.end());
        state.offset = samples[samples.size() / 2];
        if (!state.offset_valid) {
            logstream << "redundancy_selector learned standby offset replica_id=" << media.replica_id
                      << " offset=" << state.offset;
        }
        state.offset_valid = true;
        state.aligned_frames++;
        if (!state.eligible && state.aligned_frames >= alignment_required_frames_) {
            state.eligible = true;
            logstream << "redundancy_selector replica eligible replica_id=" << media.replica_id
                      << " generation=" << media.generation
                      << " offset=" << state.offset;
        }
    }

    bool replicaEligible(uint32_t replica_id) const {
        auto it = replicas_.find(replica_id);
        return it != replicas_.end() && it->second.eligible;
    }

    int64_t strictWatermark() {
        const uint64_t now = monotonicNs();
        const uint64_t stale_ns = std::max<uint64_t>(active_timeout_ms_, 1) * 1000000ull;
        int64_t watermark = std::numeric_limits<int64_t>::max();
        size_t eligible_live_replicas = 0;
        for (const auto &entry: replicas_) {
            const ReplicaState &state = entry.second;
            if (!state.eligible || state.highest_seen_frame == std::numeric_limits<int64_t>::min()) continue;
            if (state.last_seen_ns && now - state.last_seen_ns > stale_ns) continue;
            watermark = std::min(watermark, state.highest_seen_frame);
            eligible_live_replicas++;
        }
        if (eligible_live_replicas >= 2) {
            strict_redundancy_established_ = true;
            return watermark == std::numeric_limits<int64_t>::max() ? std::numeric_limits<int64_t>::min() : watermark;
        }
        if (!strict_redundancy_established_) return std::numeric_limits<int64_t>::min();
        if (eligible_live_replicas < 2) {
            auto active_it = replicas_.find(current_active_replica_id_);
            if (active_it != replicas_.end() && active_it->second.eligible &&
                active_it->second.last_seen_ns && now - active_it->second.last_seen_ns <= stale_ns) {
                return active_it->second.highest_seen_frame;
            }
            return watermark == std::numeric_limits<int64_t>::max() ? std::numeric_limits<int64_t>::min() : watermark;
        }
        return std::numeric_limits<int64_t>::min();
    }

    void handleGeneration(const MediaHeader &media) {
        ReplicaState &state = replicas_[media.replica_id];
        if (state.generation_valid && state.generation == media.generation) return;
        const bool restart = state.generation_valid;
        state = ReplicaState{};
        state.generation = media.generation;
        state.generation_valid = true;
        removeReplicaSlots(media.replica_id);
        if (restart && media.replica_id == current_active_replica_id_ && media.replica_id != preferred_active_replica_id_) {
            logstream << "redundancy_selector active replica restarted as new generation; returning active owner to preferred replica"
                      << " restarted_replica_id=" << media.replica_id
                      << " preferred_replica_id=" << preferred_active_replica_id_;
            current_active_replica_id_ = preferred_active_replica_id_;
            standby_promoted_ = false;
            active_miss_count_ = 0;
            last_active_media_ns_ = 0;
        }
        if (restart) {
            logstream << "redundancy_selector generation changed replica_id=" << media.replica_id
                      << " generation=" << media.generation;
        } else {
            logstream << "redundancy_selector generation seen replica_id=" << media.replica_id
                      << " generation=" << media.generation;
        }
    }

    void removeReplicaSlots(uint32_t replica_id) {
        for (auto it = frame_slots_.begin(); it != frame_slots_.end();) {
            it->second.media_by_replica.erase(replica_id);
            it->second.status_raw_by_replica.erase(replica_id);
            if (it->second.media_by_replica.empty() && it->second.status_raw_by_replica.empty()) {
                it = frame_slots_.erase(it);
            } else {
                ++it;
            }
        }
    }
};

} // namespace avp_fabric
