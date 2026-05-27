#include "node_common.hpp"
#include "../SharedTimeline.hpp"

template <typename T>
class MixerSourceSwitcher : public NodeMultiInput<T>, public NodeSingleOutput<T>,
                            public TimelineReader, public IInputsObjects,
                            public IVideoFormatSource, public IFrameRateSource,
                            public IAudioMetadataSource, public ITimeBaseSource {
    std::atomic<int> active_input_{0};
    int fallback_input_ = -1;
    int timeline_reference_input_ = -1;
    bool fallback_when_active_missing_ = true;
    int fallback_wait_ms_ = 0;
    bool drop_non_monotonic_ = false;
    uint64_t dropped_non_monotonic_ = 0;
    std::string label_;
    int last_selected_input_ = -1;
    av::Timestamp last_output_pts_ = NOTS;

    template <typename Interface>
    std::shared_ptr<Interface> upstreamInterface(const char* interface_name) {
        const int active = active_input_.load(std::memory_order_relaxed);
        if (active >= 0 && active < (int)this->source_edges_.size()) {
            auto src = this->source_edges_[active]->template findNodeUp<Interface>();
            if (src) return src;
        }
        for (auto& edge : this->source_edges_) {
            auto src = edge->template findNodeUp<Interface>();
            if (src) return src;
        }
        throw Error("source_switcher[" + label_ + "]: no upstream " + interface_name);
    }

public:
    using NodeSingleOutput<T>::NodeSingleOutput;

    bool putIfMonotonic(int input_index, T* data, bool fallback) {
        av::Timestamp pts = data->pts();
        if (drop_non_monotonic_ && last_output_pts_.isValid() && pts.isValid() && pts < last_output_pts_) {
            dropped_non_monotonic_++;
            if (dropped_non_monotonic_ <= 5 || dropped_non_monotonic_ % 30 == 0) {
                logstream << "source_switcher[" << label_ << "]: dropping non-monotonic pts "
                          << pts << " after " << last_output_pts_ << " on input " << input_index
                          << " count=" << dropped_non_monotonic_;
            }
            return false;
        }
        if (input_index != last_selected_input_) {
            if (fallback) {
                logstream << "source_switcher[" << label_ << "]: selected fallback input "
                          << input_index << " because active input has no usable frame";
            } else {
                logstream << "source_switcher[" << label_ << "]: selected input " << input_index
                          << " at pts " << pts;
            }
        }
        if (!drop_non_monotonic_ && last_output_pts_.isValid() && pts < last_output_pts_) {
            logstream << "source_switcher[" << label_ << "]: output pts went backwards "
                      << last_output_pts_ << " -> " << pts << " on input " << input_index;
        }
        last_selected_input_ = input_index;
        last_output_pts_ = pts;
        this->sink_->put(*data);
        return true;
    }

    virtual void process() override {
        if (this->findSourceWithData(0) < 0) {
            this->waitForInput();
            return;
        }

        T* timeline_reference = nullptr;
        if (timeline_reference_input_ >= 0 &&
            timeline_reference_input_ < (int)this->source_edges_.size()) {
            timeline_reference = this->source_edges_[timeline_reference_input_]->peek();
        }

        int reference_active = active_input_.load(std::memory_order_relaxed);
        if (timeline_reference) {
            av::Timestamp pts = timeline_reference->pts();
            reference_active = this->template tlGet<int>("active", pts, reference_active);
        }

        if (!timeline_reference &&
            fallback_wait_ms_ > 0 &&
            fallback_input_ >= 0 &&
            reference_active >= 0 &&
            reference_active < (int)this->source_edges_.size() &&
            reference_active != fallback_input_ &&
            !this->source_edges_[reference_active]->wait_peek(0)) {
            this->source_edges_[reference_active]->wait_peek(fallback_wait_ms_);
        }

        auto activeFor = [&](int input_index, T* data) {
            if (timeline_reference && input_index == timeline_reference_input_)
                return reference_active;
            av::Timestamp pts = data->pts();
            if (pts.isNoPts())
                pts = data->pts();
            return this->template tlGet<int>("active", pts, active_input_.load(std::memory_order_relaxed));
        };

        int output_count = 0;
        for (int i = 0; i < (int)this->source_edges_.size(); i++) {
            T* data = this->source_edges_[i]->peek();
            if (!data) continue;
            int active = activeFor(i, data);
            if (i == active) {
                if (putIfMonotonic(i, data, false))
                    output_count++;
            }
        }

        if (output_count == 0 &&
            fallback_input_ >= 0 &&
            fallback_input_ < (int)this->source_edges_.size() &&
            fallback_input_ != reference_active &&
            (fallback_when_active_missing_ || reference_active == fallback_input_)) {
            T* data = this->source_edges_[fallback_input_]->peek();
            if (data) {
                putIfMonotonic(fallback_input_, data, true);
            }
        }

        for (int i = 0; i < (int)this->source_edges_.size(); i++) {
            T* data = this->source_edges_[i]->peek();
            if (data) {
                this->source_edges_[i]->pop();
            }
        }
    }

    void setObject(const std::string key, const Parameters& value) override {
        if (key == "active") {
            active_input_.store(value.get<int>(), std::memory_order_relaxed);
            for (auto& edge : this->source_edges_) {
                edge->producedEvent().signal();
            }
        } else if (key == "drop_non_monotonic") {
            drop_non_monotonic_ = value.get<bool>();
        }
    }

    int width() override {
        return upstreamInterface<IVideoFormatSource>("video format source")->width();
    }

    int height() override {
        return upstreamInterface<IVideoFormatSource>("video format source")->height();
    }

    av::PixelFormat pixelFormat() override {
        return upstreamInterface<IVideoFormatSource>("video format source")->pixelFormat();
    }

    av::PixelFormat realPixelFormat() override {
        return upstreamInterface<IVideoFormatSource>("video format source")->realPixelFormat();
    }

    av::Rational frameRate() override {
        return upstreamInterface<IFrameRateSource>("frame rate source")->frameRate();
    }

    int sampleRate() override {
        return upstreamInterface<IAudioMetadataSource>("audio metadata source")->sampleRate();
    }

    av::SampleFormat sampleFormat() override {
        return upstreamInterface<IAudioMetadataSource>("audio metadata source")->sampleFormat();
    }

    uint64_t channelLayout() override {
        return upstreamInterface<IAudioMetadataSource>("audio metadata source")->channelLayout();
    }

    av::Rational timeBase() override {
        return upstreamInterface<ITimeBaseSource>("timebase source")->timeBase();
    }

    static std::shared_ptr<MixerSourceSwitcher> create(NodeCreationInfo& nci) {
        std::shared_ptr<Edge<T>> out_edge = nci.edges.find<T>(nci.params["dst"]);
        auto r = std::make_shared<MixerSourceSwitcher>(make_unique<EdgeSink<T>>(out_edge));
        r->createSourcesFromParameters(nci.edges, nci.params);
        out_edge->setProducer(r);
        r->initTimeline(nci);
        if (nci.params.count("name"))
            r->label_ = nci.params["name"].get<std::string>();
        else if (nci.params.count("dst"))
            r->label_ = nci.params["dst"].get<std::string>();
        else
            r->label_ = "<unnamed>";
        if (nci.params.count("active"))
            r->active_input_.store(nci.params["active"].get<int>(), std::memory_order_relaxed);
        if (nci.params.count("fallback_active"))
            r->fallback_input_ = nci.params["fallback_active"].get<int>();
        if (nci.params.count("timeline_reference_input"))
            r->timeline_reference_input_ = nci.params["timeline_reference_input"].get<int>();
        if (nci.params.count("fallback_when_active_missing"))
            r->fallback_when_active_missing_ = nci.params["fallback_when_active_missing"].get<bool>();
        if (nci.params.count("fallback_wait_ms"))
            r->fallback_wait_ms_ = nci.params["fallback_wait_ms"].get<int>();
        if (nci.params.count("drop_non_monotonic"))
            r->drop_non_monotonic_ = nci.params["drop_non_monotonic"].get<bool>();
        return r;
    }
};

DECLNODE_ATD(source_switcher, MixerSourceSwitcher);
