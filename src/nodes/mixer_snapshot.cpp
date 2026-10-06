#include "node_common.hpp"
#include "../mixer/primitives/CutLatencyProbe.hpp"
#include "../mixer/primitives/OutputSnapshot.hpp"
#include "../mixer/primitives/TickGrid.hpp"
#include "../mixer/primitives/MonotonicClock.hpp"

// The output instance records committed pictures; the two slot instances can
// substitute that same retained frame without a GPU copy or another compositor.
class MixerSnapshot : public NodeSISO<av::VideoFrame, av::VideoFrame>,
                      public IFrameRateSource, public ITimeBaseSource {
    std::shared_ptr<avp::mixer::OutputSnapshot> state_;
    int slot_;
    av::Rational fps_;
    avp::mixer::TickGrid rate_;
    int64_t latency_ns_;
    av::Timestamp last_pts_ = NOTS;
    // Set while the output instance sees its Snapshot holding; the line logged at the release
    // is how a host run shows how long a take kept the still on air.
    std::optional<int64_t> held_since_ns_;
    static constexpr const char* kMarker = "avp.mixer.snapshot";

    void endHold(avp::mixer::Snapshot<av::VideoFrame>& frames, const char* how) {
        frames.release();
        if (!held_since_ns_) return;
        logstream << "mixer_snapshot: output held " << (avp::mixer::monotonicNs() - *held_since_ns_) / 1000000
                  << " ms, released on a " << how << " frame";
        held_since_ns_.reset();
    }

public:
    MixerSnapshot(std::unique_ptr<SourceType>&& source, std::unique_ptr<SinkType>&& sink,
                  std::shared_ptr<avp::mixer::OutputSnapshot> state, int slot,
                  av::Rational fps, int64_t latency_ns)
        : NodeSISO(std::move(source), std::move(sink)), state_(std::move(state)),
          slot_(slot), fps_(fps), rate_(fps),
          latency_ns_(latency_ns) {
        if (slot_ == -1) {
            std::lock_guard<std::mutex> lock(state_->mutex);
            state_->output_connected = true;
        }
    }
    ~MixerSnapshot() override {
        if (slot_ == -1) {
            std::lock_guard<std::mutex> lock(state_->mutex);
            state_->output_connected = false;
        }
    }
    av::Rational frameRate() override { return fps_; }
    av::Rational timeBase() override { return av_inv_q(fps_.getValue()); }

    void process() override {
        // A short input wait also wakes a newly requested hold on an idle slot.
        auto* input = this->source_->peek(2);
        std::unique_lock<std::mutex> lock(state_->mutex);
        auto& frames = state_->frames;
        bool replace = slot_ == -1 ? frames.holding() : frames.replaces(slot_);
        bool release = false;
        if (slot_ == -1 && replace) {
            if (!held_since_ns_) held_since_ns_ = avp::mixer::monotonicNs();
            if (input && input->isValid() && input->pts().isValid()) {
                const auto* tag = av_dict_get(input->raw()->metadata, kMarker, nullptr, 0);
                const uint64_t generation = tag ? avp::mixer::parseFrameToken(tag->value) : 0;
                release = frames.canRelease(input->pts().timestamp({1, 1000000000}), generation);
                if (release) replace = false;
            }
        }

        av::VideoFrame output;
        if (replace) {
            // The output instance stamps its still one tick behind the clock: a tick's live frame
            // crosses the pipeline after a still of that tick would be stamped, so it lost that race,
            // was dropped below as a duplicate and left the hold's end to scheduling luck. Slot
            // instances stay on the current tick: their still must reach the output ahead of the
            // output's own.
            const auto index = rate_.atOrBefore(avp::mixer::monotonicNs() - latency_ns_) - (slot_ == -1 ? 1 : 0);
            auto pts = av::Timestamp(index, timeBase());
            // Draining is consumer-owned; no controller ever clears live edges.
            for (int i = 0; i < 8 && this->source_->peek(0); ++i) this->source_->pop();
            if (last_pts_.isValid() && pts <= last_pts_) return;
            output = *frames.frozen();
            output.setTimeBase(timeBase());
            output.setPts(pts);
            if (slot_ != -1)
                av_dict_set(&output.raw()->metadata, kMarker, std::to_string(frames.generation()).c_str(), 0);
        } else {
            if (!input) return;
            output = *input;
            if (output.isValid() && last_pts_.isValid() && output.pts().isValid() && output.pts() <= last_pts_) {
                this->source_->pop();
                // A still of its tick went out first; the hold ends all the same, so the
                // next live frame passes instead of losing the same race again.
                if (release) endHold(frames, "duplicate");
                return;
            }
        }
        // Serialize capture with a nonblocking publication. Backpressure must
        // never lock out the control thread or update the displayed snapshot.
        if (this->sink_->put(output, true)) {
            if (!replace) this->source_->pop();
            if (release) endHold(frames, "live");
            if (output.isValid() && output.pts().isValid()) {
                last_pts_ = output.pts();
                if (slot_ == -1) frames.presented(output);
            }
        } else {
            lock.unlock();
            this->edgeSink()->edge()->consumedEvent().wait(2);
        }
    }

    static std::shared_ptr<MixerSnapshot> create(NodeCreationInfo& nci) {
        auto state = InstanceSharedObjects<avp::mixer::OutputSnapshot>::get(nci.instance, nci.params.at("snapshot"));
        const auto fps = parseRatio(nci.params.value("fps", std::string("60/1")));
        const int slot = nci.params.value("slot", -1);
        if (slot < -1 || slot > 1) throw Error("mixer_snapshot: slot must be -1, 0 or 1");
        const double default_ms = 2000.0 * fps.getDenominator() / fps.getNumerator();
        const auto latency = static_cast<int64_t>(nci.params.value("latency_ms", default_ms) * 1000000);
        return NodeSISO::createCommon<MixerSnapshot>(nci.edges, nci.params, state, slot, fps, latency);
    }
};

DECLNODE(mixer_snapshot, MixerSnapshot)
