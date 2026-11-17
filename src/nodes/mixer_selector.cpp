#include "../mixer/SourceSwitcher.hpp"
#include "../mixer/orchestrator/MixerOrchestrator.hpp"
#include "../mixer/primitives/TraceRing.hpp"

// The only frame-boundary commit points: program A/B/blend, and direct/wipe.
// Both serialize with commands through MixerState, without a scheduling thread.
class MixerSelector : public MixerSourceSwitcher<av::VideoFrame>, public IReturnsObjects {
    std::shared_ptr<avp::mixer::MixerState> state_;
    bool wipe_ = false;
    bool trace_enabled_ = false;
    avp::mixer::TraceRing trace_;
public:
    using MixerSourceSwitcher::MixerSourceSwitcher;

    void process() override {
        // Input events wake immediately. The finite wait only detects a stalled
        // take when every input is silent; it never delays an available frame. Only a
        // take at a show tick holds one (below), for its destination's frame of the tick.
        this->findSourceWithData(20);
        if (this->stopping_) return;
        if (this->edgeSink()->edge()->occupied() >= int(this->edgeSink()->edge()->capacity())) {
            this->edgeSink()->edge()->consumedEvent().wait(20);
            return;
        }
        std::unique_lock<std::mutex> lock(state_->mutex);
        std::vector<const av::VideoFrame*> frames;
        for (auto& edge : source_edges_) frames.push_back(edge->peek());
        int selected = active_input_.load();
        if (state_->graph) {
            const int64_t last = last_output_pts_.isValid() ? last_output_pts_.timestamp({1, 1000000000}) : 0;
            avp::mixer::MixerOrchestrator control(state_->graph, state_);
            const auto mode = state_->transition_mode.load();
            const auto phase = state_->take_phase;
            selected = control.selectFrame(wipe_, frames, active_input_.load(), last);
            if (selected != avp::mixer::MixerOrchestrator::kWaitForDestination) active_input_.store(selected);
            if (trace_enabled_) {
                Parameters heads = Parameters::array();
                for (const auto* frame : frames)
                    heads.push_back(frame && frame->pts().isValid()
                        ? Parameters(frame->pts().timestamp({1, 1000000000})) : Parameters());
                trace_.push({{"last_ns", last}, {"heads_ns", heads},
                    {"selected", selected}, {"mode", int(mode)}, {"phase", int(phase)},
                    {"start_ns", state_->take_start_ns}});
            }
        }
        if (selected == avp::mixer::MixerOrchestrator::kWaitForDestination) {
            // Nothing goes out yet: a take at a show tick waits for the destination's frame of
            // program's tick until take_tick_wait_until_ns (selectProgram), never with the mutex.
            const int64_t wait_ns = state_->take_tick_wait_until_ns - avp::mixer::monotonicNs();
            return holdForInput(frames, state_->pvwSourceSwitcherIndex(), wait_ns, lock);
        }
        processHeads(frames, selected);
    }

    Parameters getObject(const std::string key) override {
        if (key == "trace") {
            std::lock_guard<std::mutex> lock(state_->mutex);
            return trace_.json();
        }
        throw Error("mixer_selector: unknown object: " + key);
    }

    void setObject(const std::string key, const Parameters& value) override {
        if (key == "trace") {
            std::lock_guard<std::mutex> lock(state_->mutex);
            trace_enabled_ = value.get<bool>();
            trace_.clear();
        } else {
            MixerSourceSwitcher::setObject(key, value);
        }
    }

    static std::shared_ptr<MixerSelector> create(NodeCreationInfo& nci) {
        auto output = nci.edges.find<av::VideoFrame>(nci.params.at("dst"));
        auto r = std::make_shared<MixerSelector>(make_unique<EdgeSink<av::VideoFrame>>(output));
        r->createSourcesFromParameters(nci.edges, nci.params);
        output->setProducer(r);
        r->configure(nci);
        r->drop_non_monotonic_ = true;
        r->state_ = InstanceSharedObjects<avp::mixer::MixerState>::get(nci.instance, nci.params.at("mixer"));
        r->wipe_ = nci.params.value("wipe", false);
        return r;
    }
};

DECLNODE(mixer_selector, MixerSelector)
