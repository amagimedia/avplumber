#include "../node_common.hpp"

#include "fabric_protocol.hpp"
#include "fabric_source_timeline.hpp"

#include <algorithm>
#include <arpa/inet.h>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <deque>
#include <limits>
#include <map>
#include <netinet/in.h>
#include <string>
#include <sys/socket.h>
#include <unistd.h>
#include <unordered_map>
#include <vector>

class RedundancySelector: public NodeMultiInput<FabricPacket>,
                          public NodeSingleOutput<FabricPacket>,
                          public ReportsFinishByFlag,
                          public IStreamsInput {
    struct ReplicaState {
        std::deque<std::pair<uint64_t, int64_t>> history;
        std::deque<int64_t> offset_samples;
        int64_t offset = 0;
        bool offset_valid = false;
    };
    struct Slot {
        std::unordered_map<uint32_t, FabricPacket> by_replica;
    };
    enum class AlignmentMode {
        WallclockOffset,
        StrictFrameIdentity,
    };
    enum class Mode {
        Single,
        ActiveActive,
        ActiveStandby,
    };

    Mode mode_ = Mode::ActiveActive;
    uint32_t anchor_replica_id_ = 1;
    uint32_t active_replica_id_ = 1;
    uint32_t standby_replica_id_ = 2;
    AlignmentMode alignment_mode_ = AlignmentMode::StrictFrameIdentity;
    int alignment_window_frames_ = 50;
    int alignment_required_frames_ = 10;
    int64_t max_wallclock_delta_ns_ = 120000000;
    int64_t max_initial_frame_step_ticks_ = 900000;
    uint64_t playout_delay_frames_ = 1;
    uint64_t promote_after_misses_ = 1;
    uint64_t active_timeout_ms_ = 50;
    uint64_t repair_grace_ms_ = 0;
    std::string standby_control_host_ = "127.0.0.1";
    uint16_t standby_control_port_ = 0;
    std::unordered_map<uint32_t, uint16_t> control_ports_by_replica_;
    int control_fd_ = -1;
    std::unordered_map<uint32_t, ReplicaState> replicas_;
    std::map<int64_t, Slot> slots_;
    avp_fabric::SourceTimeline timeline_;
    bool next_emit_valid_ = false;
    int64_t next_emit_id_ = 0;
    int64_t highest_seen_id_ = std::numeric_limits<int64_t>::min();
    int64_t frame_step_ = 0;
    int64_t previous_anchor_pts_ = AV_NOPTS_VALUE;
    av::FormatContext format_ctx_;
    av::Stream stream_;
    std::string reference_url_;
    uint64_t received_ = 0;
    uint64_t emitted_ = 0;
    uint64_t stale_dropped_ = 0;
    uint64_t standby_selected_ = 0;
    uint64_t implausible_steps_ = 0;

public:
    RedundancySelector(std::unique_ptr<Sink<FabricPacket>> &&sink, const Parameters &params):
        NodeSingleOutput<FabricPacket>(std::move(sink)) {
        const std::string mode = params.value("mode", params.value("redundancy_mode", std::string("active_active")));
        if (mode == "single") {
            mode_ = Mode::Single;
        } else if (mode == "active_active" || mode == "hot_hot" || mode == "hot_hot_first_complete") {
            mode_ = Mode::ActiveActive;
        } else if (mode == "active_standby" || mode == "active_standby_repair") {
            mode_ = Mode::ActiveStandby;
        } else {
            throw Error("redundancy_selector unsupported mode: " + mode);
        }

        anchor_replica_id_ = params.value("anchor_replica_id", params.value("active_replica_id", anchor_replica_id_));
        active_replica_id_ = params.value("active_replica_id", anchor_replica_id_);
        standby_replica_id_ = params.value("standby_replica_id", standby_replica_id_);
        alignment_window_frames_ = params.value("alignment_window_frames", alignment_window_frames_);
        alignment_required_frames_ = params.value("alignment_required_frames", alignment_required_frames_);
        const std::string alignment_mode = params.value("alignment_mode", std::string("strict_frame_identity"));
        if (alignment_mode == "wallclock_offset" || alignment_mode == "wallclock") {
            alignment_mode_ = AlignmentMode::WallclockOffset;
        } else if (alignment_mode == "strict_frame_identity" || alignment_mode == "strict") {
            alignment_mode_ = AlignmentMode::StrictFrameIdentity;
        } else {
            throw Error("redundancy_selector unknown alignment_mode: " + alignment_mode);
        }
        const int max_wallclock_delta_ms = params.value("alignment_max_wallclock_delta_ms", 120);
        max_wallclock_delta_ns_ = static_cast<int64_t>(max_wallclock_delta_ms) * 1000000ll;
        max_initial_frame_step_ticks_ = params.value("max_initial_frame_step_ticks", max_initial_frame_step_ticks_);
        playout_delay_frames_ = params.value("playout_delay_frames", playout_delay_frames_);
        promote_after_misses_ = params.value("promote_after_misses", promote_after_misses_);
        active_timeout_ms_ = params.value("active_timeout_ms", active_timeout_ms_);
        repair_grace_ms_ = params.value("repair_grace_ms", repair_grace_ms_);
        standby_control_host_ = params.value("standby_control_host", standby_control_host_);
        standby_control_port_ = params.value("standby_control_port", 0);
        if (params.count("standby_control_ports")) {
            for (auto it = params["standby_control_ports"].begin(); it != params["standby_control_ports"].end(); ++it) {
                control_ports_by_replica_[static_cast<uint32_t>(std::stoul(it.key()))] = it.value().get<uint16_t>();
            }
        }
        if (standby_control_port_) {
            control_ports_by_replica_[standby_replica_id_] = standby_control_port_;
        }
        if (!control_ports_by_replica_.empty()) {
            control_fd_ = socket(AF_INET, SOCK_DGRAM, 0);
            if (control_fd_ < 0) throw Error("redundancy_selector control socket failed");
        }
        timeline_ = avp_fabric::SourceTimeline(
            "active_standby_repair", active_replica_id_, playout_delay_frames_, promote_after_misses_,
            active_timeout_ms_, repair_grace_ms_, alignment_mode_ == AlignmentMode::StrictFrameIdentity);
        reference_url_ = params.value("reference_url", std::string());
    }

    ~RedundancySelector() override {
        if (control_fd_ >= 0) close(control_fd_);
    }

    void init(EdgeManager &edges, const Parameters &params) override {
        NodeMultiInput<FabricPacket>::createSourcesFromParameters(edges, params);
        NodeSingleOutput<FabricPacket>::init(edges, params);

        for (auto &edge: source_edges_) {
            auto md = edge->template metadata<InputStreamMetadata>();
            if (md && md->source_stream.isValid()) {
                stream_ = md->source_stream;
                break;
            }
        }
        if (!stream_.isValid() && !reference_url_.empty()) {
            av::Dictionary opts;
            format_ctx_.openInput(reference_url_, opts);
            format_ctx_.findStreamInfo();
            for (size_t i = 0; i < format_ctx_.streamsCount(); ++i) {
                av::Stream st = format_ctx_.stream(i);
                if (st.isVideo()) {
                    stream_ = st;
                    break;
                }
            }
            if (stream_.isValid()) {
                AVCodecParameters *cp = stream_.raw()->codecpar;
                if (cp->extradata) {
                    av_freep(&cp->extradata);
                    cp->extradata_size = 0;
                }
                cp->codec_tag = 0;
            }
        }
        if (!stream_.isValid()) throw Error("redundancy_selector requires upstream InputStreamMetadata");
        auto out_edge = edges.find<FabricPacket>(params["dst"]);
        auto out_md = out_edge->metadata<InputStreamMetadata>(true);
        out_md->source_stream = stream_;
    }

    void process() override {
        if (mode_ == Mode::Single) {
            processSingle();
            return;
        }

        for (;;) {
            FabricPacket selected;
            if (tryEmit(selected)) {
                this->sink_->put(selected);
                emitted_++;
                return;
            }

            const int idx = findSourceWithData(-1);
            if (idx < 0) return;
            FabricPacket *queued = source_edges_[idx]->peek();
            if (queued == nullptr) continue;
            FabricPacket pkt = *queued;
            source_edges_[idx]->pop();
            if (!pkt.complete) continue;

            if (mode_ == Mode::ActiveStandby) {
                receiveActiveStandby(pkt);
            } else {
                receiveActiveActive(pkt);
            }
        }
    }

    size_t streamsCount() override { return 1; }
    av::Stream stream(size_t i) override {
        if (i != 0) throw Error("redundancy_selector stream index out of range");
        return stream_;
    }
    void discardAllStreams() override {}
    void enableStream(size_t) override {}
    av::FormatContext& formatContext() override { return format_ctx_; }

    static std::shared_ptr<RedundancySelector> create(NodeCreationInfo &nci) {
        auto edge = nci.edges.find<FabricPacket>(nci.params["dst"]);
        return std::make_shared<RedundancySelector>(make_unique<EdgeSink<FabricPacket>>(edge), nci.params);
    }

private:
    void processSingle() {
        for (;;) {
            const int idx = findSourceWithData(-1);
            if (idx < 0) return;
            FabricPacket *queued = source_edges_[idx]->peek();
            if (queued == nullptr) continue;
            FabricPacket pkt = *queued;
            source_edges_[idx]->pop();
            if (!pkt.complete || pkt.message_type != avp_fabric::MSG_MEDIA) continue;
            this->sink_->put(transparentPacket(pkt));
            emitted_++;
            return;
        }
    }

    bool tryEmit(FabricPacket &selected) {
        if (mode_ == Mode::ActiveStandby) {
            avp_fabric::SourceCandidate candidate;
            if (!timeline_.trySelect(candidate, controlSender())) return false;
            selected = fromCandidate(candidate);
            return true;
        }
        if (!next_emit_valid_ || slots_.empty()) return false;
        return alignment_mode_ == AlignmentMode::StrictFrameIdentity ? tryEmitStrict(selected) : tryEmitWallclock(selected);
    }

    void receiveActiveStandby(const FabricPacket &pkt) {
        if (pkt.raw_pts == AV_NOPTS_VALUE) {
            logstream << "redundancy_selector dropped packet without PTS"
                      << " replica_id=" << pkt.replica_id;
            return;
        }
        const avp_fabric::MediaHeader media = mediaHeaderFromPacket(pkt);
        if (pkt.message_type == avp_fabric::MSG_FRAME_STATUS) {
            timeline_.ingestStatus(media, pkt.receiver_wallclock_ns, controlSender());
        } else if (pkt.message_type == avp_fabric::MSG_MEDIA) {
            timeline_.ingestMedia(media, pkt.packet.data(), pkt.packet.size(), pkt.receiver_wallclock_ns);
        }
        received_++;
    }

    void receiveActiveActive(const FabricPacket &pkt) {
        if (pkt.message_type != avp_fabric::MSG_MEDIA) return;
        if (pkt.raw_pts == AV_NOPTS_VALUE) {
            logstream << "redundancy_selector dropped packet without PTS"
                      << " replica_id=" << pkt.replica_id;
            return;
        }

        ReplicaState &state = replicas_[pkt.replica_id];
        state.history.emplace_back(pkt.receiver_wallclock_ns, pkt.raw_pts);
        while (state.history.size() > static_cast<size_t>(alignment_window_frames_)) {
            state.history.pop_front();
        }

        if (pkt.replica_id == anchor_replica_id_) {
            observeAnchorStep(pkt);
            state.offset = 0;
            state.offset_valid = true;
        } else {
            if (alignment_mode_ == AlignmentMode::StrictFrameIdentity) {
                state.offset = 0;
                state.offset_valid = true;
            } else {
                learnOffset(pkt, state);
                if (!state.offset_valid) {
                    return;
                }
            }
        }

        const int64_t normalized_id = pkt.raw_pts + state.offset;
        slots_[normalized_id].by_replica[pkt.replica_id] = pkt;
        highest_seen_id_ = std::max(highest_seen_id_, normalized_id);
        if (!next_emit_valid_ && pkt.replica_id == anchor_replica_id_) {
            if (alignment_mode_ == AlignmentMode::StrictFrameIdentity) {
                if (frame_step_ > 0) {
                    const int64_t previous_id = normalized_id - frame_step_;
                    next_emit_id_ = slots_.find(previous_id) != slots_.end() ? previous_id : normalized_id;
                    next_emit_valid_ = true;
                }
            } else {
                next_emit_valid_ = true;
                next_emit_id_ = normalized_id;
            }
        }
        if (received_ < 12 || received_ % 250 == 0) {
            logstream << "redundancy_selector received"
                      << " replica_id=" << pkt.replica_id
                      << " raw_pts=" << pkt.raw_pts
                      << " offset=" << state.offset
                      << " normalized_id=" << normalized_id
                      << " offset_valid=" << state.offset_valid
                      << " emitted=" << emitted_;
        }
        received_++;
    }

    void observeAnchorStep(const FabricPacket &pkt) {
        if (previous_anchor_pts_ != AV_NOPTS_VALUE) {
            const int64_t step = pkt.raw_pts - previous_anchor_pts_;
            if (isPlausibleFrameStep(step, pkt)) {
                frame_step_ = step;
            } else if (step > 0) {
                if (implausible_steps_ < 5 || implausible_steps_ % 100 == 0) {
                    logstream << "redundancy_selector ignored implausible frame step"
                              << " previous_anchor_pts=" << previous_anchor_pts_
                              << " raw_pts=" << pkt.raw_pts
                              << " step=" << step
                              << " duration=" << pkt.duration
                              << " current_frame_step=" << frame_step_
                              << " implausible_steps=" << implausible_steps_;
                }
                implausible_steps_++;
            }
        }
        previous_anchor_pts_ = pkt.raw_pts;
    }

    bool isPlausibleFrameStep(int64_t step, const FabricPacket &pkt) const {
        if (step <= 0) return false;
        if (frame_step_ > 0) {
            return std::llabs(step - frame_step_) < std::max<int64_t>(1, frame_step_ / 2);
        }
        if (pkt.duration > 0) {
            const int64_t max_from_duration = pkt.duration > (std::numeric_limits<int64_t>::max() / 4)
                ? std::numeric_limits<int64_t>::max()
                : pkt.duration * 4;
            return step <= std::max<int64_t>(1, max_from_duration);
        }
        return step <= max_initial_frame_step_ticks_;
    }

    void learnOffset(const FabricPacket &pkt, ReplicaState &state) {
        auto anchor_it = replicas_.find(anchor_replica_id_);
        if (anchor_it == replicas_.end()) return;
        const auto &anchor_history = anchor_it->second.history;
        if (anchor_history.empty()) return;

        int64_t best_dt = std::numeric_limits<int64_t>::max();
        int64_t best_anchor_pts = AV_NOPTS_VALUE;
        for (const auto &sample: anchor_history) {
            const int64_t dt = std::llabs(static_cast<int64_t>(sample.first) - static_cast<int64_t>(pkt.receiver_wallclock_ns));
            if (dt < best_dt) {
                best_dt = dt;
                best_anchor_pts = sample.second;
            }
        }
        if (best_anchor_pts == AV_NOPTS_VALUE || best_dt > max_wallclock_delta_ns_) return;

        state.offset_samples.push_back(best_anchor_pts - pkt.raw_pts);
        while (state.offset_samples.size() > static_cast<size_t>(alignment_window_frames_)) {
            state.offset_samples.pop_front();
        }
        if (state.offset_samples.size() < static_cast<size_t>(alignment_required_frames_)) return;

        std::vector<int64_t> samples(state.offset_samples.begin(), state.offset_samples.end());
        std::sort(samples.begin(), samples.end());
        state.offset = samples[samples.size() / 2];
        state.offset_valid = true;
    }

    bool tryEmitWallclock(FabricPacket &selected) {
        auto it = slots_.lower_bound(next_emit_id_);
        if (it == slots_.end()) return false;
        next_emit_id_ = it->first;

        const int64_t delay_ticks = (frame_step_ > 0) ? static_cast<int64_t>(playout_delay_frames_) * frame_step_ : 0;
        if (delay_ticks > 0 && highest_seen_id_ < next_emit_id_ + delay_ticks) return false;

        Slot &slot = it->second;
        auto active = slot.by_replica.find(anchor_replica_id_);
        if (active != slot.by_replica.end()) {
            selected = transparentPacket(active->second);
            finish(it);
            return true;
        }
        auto standby = slot.by_replica.find(standby_replica_id_);
        if (standby != slot.by_replica.end()) {
            selected = transparentPacket(standby->second);
            logStandbySelected(*standby, next_emit_id_, false);
            finish(it);
            return true;
        }
        return false;
    }

    bool tryEmitStrict(FabricPacket &selected) {
        while (!slots_.empty() && slots_.begin()->first < next_emit_id_) {
            if (stale_dropped_ < 8 || stale_dropped_ % 250 == 0) {
                logstream << "redundancy_selector strict dropped stale pts=" << slots_.begin()->first
                          << " next_emit_id=" << next_emit_id_
                          << " stale_dropped=" << stale_dropped_;
            }
            stale_dropped_++;
            slots_.erase(slots_.begin());
        }
        if (slots_.empty()) return false;

        const int64_t delay_ticks = (frame_step_ > 0) ? static_cast<int64_t>(playout_delay_frames_) * frame_step_ : 0;
        if (delay_ticks > 0 && highest_seen_id_ < next_emit_id_ + delay_ticks) return false;

        auto it = slots_.find(next_emit_id_);
        if (it == slots_.end()) return false;

        Slot &slot = it->second;
        auto active = slot.by_replica.find(anchor_replica_id_);
        if (active != slot.by_replica.end()) {
            selected = transparentPacket(active->second);
            finish(it);
            return true;
        }
        auto standby = slot.by_replica.find(standby_replica_id_);
        if (standby != slot.by_replica.end()) {
            selected = transparentPacket(standby->second);
            logStandbySelected(*standby, next_emit_id_, true);
            finish(it);
            return true;
        }
        return false;
    }

    void logStandbySelected(const std::pair<const uint32_t, FabricPacket> &standby, int64_t normalized_id, bool strict) {
        if (standby_selected_ < 10 || standby_selected_ % 250 == 0) {
            logstream << "redundancy_selector"
                      << (strict ? " strict" : "")
                      << " selected standby replica_id=" << standby.first
                      << " raw_pts=" << standby.second.raw_pts
                      << " normalized_id=" << normalized_id
                      << " standby_selected=" << standby_selected_;
        }
        standby_selected_++;
    }

    FabricPacket transparentPacket(const FabricPacket &fp) {
        FabricPacket out = fp;
        applyPacketTiming(out);
        return out;
    }

    FabricPacket fromCandidate(const avp_fabric::SourceCandidate &candidate) {
        FabricPacket out;
        out.message_type = avp_fabric::MSG_MEDIA;
        out.stream_id_hash = candidate.media.stream_id_hash;
        out.replica_id = candidate.media.replica_id;
        out.generation = candidate.media.generation;
        out.time_base_num = candidate.media.time_base_num;
        out.time_base_den = candidate.media.time_base_den;
        out.media_type = candidate.media.media_type;
        out.codec = candidate.media.codec;
        out.packet_format = candidate.media.packet_format;
        out.packet_flags = candidate.media.packet_flags;
        out.duration = candidate.media.duration;
        out.width = candidate.media.width;
        out.height = candidate.media.height;
        out.pixel_format = candidate.media.pixel_format;
        out.real_pixel_format = candidate.media.real_pixel_format;
        out.sender_wallclock_ns = candidate.media.sender_wallclock_ns;
        out.receiver_wallclock_ns = avp_fabric::monotonicNs();
        out.raw_pts = candidate.media.pts;
        out.raw_dts = candidate.media.dts;
        out.packet = av::Packet(candidate.payload);
        out.complete = true;
        applyPacketTiming(out);
        return out;
    }

    void applyPacketTiming(FabricPacket &fp) {
        av::Rational tb(fp.time_base_num, fp.time_base_den);
        fp.packet.setTimeBase(tb);
        fp.packet.setPts(av::Timestamp(fp.raw_pts, tb));
        if (fp.raw_dts != AV_NOPTS_VALUE) {
            fp.packet.setDts(av::Timestamp(fp.raw_dts, tb));
        } else {
            fp.packet.setDts(av::Timestamp(fp.raw_pts, tb));
        }
        fp.packet.setDuration(static_cast<int>(fp.duration));
        fp.packet.setStreamIndex(0);
        fp.packet.setKeyPacket((fp.packet_flags & avp_fabric::FLAG_KEYFRAME) != 0);
        fp.packet.setComplete(true);
        fp.complete = true;
        fp.message_type = avp_fabric::MSG_MEDIA;
    }

    avp_fabric::MediaHeader mediaHeaderFromPacket(const FabricPacket &pkt) const {
        avp_fabric::MediaHeader media = {};
        media.stream_id_hash = pkt.stream_id_hash;
        media.replica_id = pkt.replica_id;
        media.generation = pkt.generation;
        media.pts = pkt.raw_pts;
        media.dts = pkt.raw_dts;
        media.time_base_num = pkt.time_base_num;
        media.time_base_den = pkt.time_base_den;
        media.media_type = pkt.media_type;
        media.codec = pkt.codec;
        media.packet_format = pkt.packet_format;
        media.packet_flags = pkt.packet_flags;
        media.duration = pkt.duration;
        media.width = pkt.width;
        media.height = pkt.height;
        media.pixel_format = pkt.pixel_format;
        media.real_pixel_format = pkt.real_pixel_format;
        media.sender_wallclock_ns = pkt.sender_wallclock_ns;
        return media;
    }

    avp_fabric::SourceTimeline::SendControl controlSender() {
        return [this](uint32_t replica_id, const std::string &cmd, int64_t normalized_pts) {
            sendControl(replica_id, cmd, normalized_pts);
        };
    }

    void sendControl(uint32_t replica_id, const std::string &cmd, int64_t normalized_pts) {
        if (control_fd_ < 0) return;
        auto port_it = control_ports_by_replica_.find(replica_id);
        if (port_it == control_ports_by_replica_.end() || !port_it->second) return;
        const int64_t raw_pts = timeline_.denormalizeFrameId(replica_id, normalized_pts);
        sockaddr_in addr = {};
        addr.sin_family = AF_INET;
        addr.sin_port = htons(port_it->second);
        if (inet_pton(AF_INET, standby_control_host_.c_str(), &addr.sin_addr) != 1) {
            logstream << "redundancy_selector invalid standby_control_host=" << standby_control_host_;
            return;
        }
        const std::string msg = cmd + " " + std::to_string(raw_pts);
        sendto(control_fd_, msg.data(), msg.size(), 0, reinterpret_cast<sockaddr *>(&addr), sizeof(addr));
        logstream << "redundancy_selector control " << cmd
                  << " replica_id=" << replica_id
                  << " normalized_pts=" << normalized_pts
                  << " raw_pts=" << raw_pts;
    }

    void finish(std::map<int64_t, Slot>::iterator selected_it) {
        const int64_t selected_id = selected_it->first;
        slots_.erase(slots_.begin(), std::next(selected_it));
        if (alignment_mode_ == AlignmentMode::StrictFrameIdentity && frame_step_ > 0) {
            next_emit_id_ = selected_id + frame_step_;
            return;
        }
        if (slots_.empty()) {
            next_emit_valid_ = false;
            return;
        }
        next_emit_id_ = slots_.begin()->first;
    }
};

DECLNODE(redundancy_selector, RedundancySelector);
