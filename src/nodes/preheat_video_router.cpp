#include "node_common.hpp"
#include "../SharedTimeline.hpp"

#include <algorithm>
#include <mutex>
#include <unordered_map>
#include <unordered_set>
#include <vector>

// preheat_video_router: video-only N->M router with a timeline-driven route
// table. Assumes all inputs share width/height/pixel format/frame rate/time
// base. Enforces per-output monotonic PTS by default; intended for the current
// preheated wallclock-PTS mixer pipelines rather than as a generic media router.
class PreheatVideoRouter : public NodeMultiInput<av::VideoFrame>,
                           public NodeMultiOutput<av::VideoFrame>,
                           public TimelineReader,
                           public ReportsFinishByFlag,
                           public IInputsObjects,
                           public IReturnsObjects,
                           public IVideoFormatSource,
                           public IFrameRateSource,
                           public ITimeBaseSource {
    mutable std::mutex routes_mutex_;
    mutable std::mutex stats_mutex_;
    std::vector<int> routes_;
    std::vector<int> effective_routes_;
    std::vector<std::string> output_labels_;
    std::unordered_map<std::string, size_t> output_index_by_label_;
    std::vector<av::Timestamp> last_output_pts_;
    std::vector<bool> input_finished_;
    std::vector<uint64_t> frames_drained_per_input_;
    std::vector<uint64_t> frames_enqueued_per_output_;
    std::vector<uint64_t> frames_dropped_per_output_;
    std::vector<uint64_t> backward_pts_drops_per_output_;
    std::vector<std::unordered_set<size_t>> outputs_seen_by_input_;
    std::shared_ptr<IVideoFormatSource> video_format_source_;
    std::shared_ptr<IFrameRateSource> frame_rate_source_;
    std::shared_ptr<ITimeBaseSource> time_base_source_;
    int metadata_width_ = 0;
    int metadata_height_ = 0;
    av::PixelFormat metadata_pixel_format_;
    av::PixelFormat metadata_real_pixel_format_;
    av::Rational metadata_frame_rate_{0, 1};
    av::Rational metadata_time_base_{0, 1};
    bool enforce_monotonic_pts_ = true;
    std::string label_ = "<unnamed>";
    static constexpr uint64_t kBackwardPtsDropLogLimit = 5;

    template <typename Interface>
    std::shared_ptr<Interface> upstreamInterfaceAt(size_t input_index, const char* interface_name) const {
        if (input_index >= this->source_edges_.size()) {
            throw Error("preheat_video_router[" + label_ + "]: input index " +
                        std::to_string(input_index) + " is out of range");
        }
        auto src = this->source_edges_[input_index]->template findNodeUp<Interface>();
        if (src)
            return src;
        throw Error("preheat_video_router[" + label_ + "]: input " +
                    std::to_string(input_index) + " has no upstream " + interface_name);
    }

    template <typename Interface>
    std::shared_ptr<Interface> findUpstreamInterfaceAt(size_t input_index) const {
        if (input_index >= this->source_edges_.size())
            return nullptr;
        return this->source_edges_[input_index]->template findNodeUp<Interface>();
    }

    void validateRouteInputIndex(int input_index) const {
        if (input_index < -1 || input_index >= (int)this->source_edges_.size()) {
            throw Error("preheat_video_router[" + label_ + "]: route input index " +
                        std::to_string(input_index) + " is outside [-1, " +
                        std::to_string(this->source_edges_.size() - 1) + "]");
        }
    }

    std::vector<int> parseRoutes(const Parameters& value,
                                 const std::vector<int>* base_routes = nullptr) const {
        if (value.is_array()) {
            if (value.size() != this->sink_edges_.size()) {
                throw Error("preheat_video_router[" + label_ + "]: routes size " +
                            std::to_string(value.size()) + " does not match outputs " +
                            std::to_string(this->sink_edges_.size()));
            }

            std::vector<int> parsed;
            parsed.reserve(value.size());
            for (const auto& item : value) {
                int input_index = item.get<int>();
                validateRouteInputIndex(input_index);
                parsed.push_back(input_index);
            }
            return parsed;
        }

        if (!value.is_object()) {
            throw Error("preheat_video_router[" + label_ + "]: routes must be an array or object");
        }
        if (!base_routes) {
            throw Error("preheat_video_router[" + label_ + "]: named routes require a base route table");
        }

        std::vector<int> parsed = *base_routes;
        if (parsed.size() != this->sink_edges_.size()) {
            throw Error("preheat_video_router[" + label_ + "]: base route table size " +
                        std::to_string(parsed.size()) + " does not match outputs " +
                        std::to_string(this->sink_edges_.size()));
        }

        for (auto it = value.begin(); it != value.end(); ++it) {
            auto label_it = output_index_by_label_.find(it.key());
            if (label_it == output_index_by_label_.end()) {
                throw Error("preheat_video_router[" + label_ + "]: unknown route label: " + it.key());
            }
            int input_index = it.value().get<int>();
            validateRouteInputIndex(input_index);
            parsed[label_it->second] = input_index;
        }
        return parsed;
    }

    Parameters routesToJson(const std::vector<int>& routes) const {
        Parameters arr = Parameters::array();
        for (int input_index : routes)
            arr.push_back(input_index);
        return arr;
    }

    Parameters routesToNamedJson(const std::vector<int>& routes) const {
        Parameters obj = Parameters::object();
        for (size_t out_i = 0; out_i < routes.size(); ++out_i)
            obj[output_labels_[out_i]] = routes[out_i];
        return obj;
    }

    std::vector<int> currentRoutes() const {
        std::lock_guard<std::mutex> lock(routes_mutex_);
        return routes_;
    }

    std::vector<int> effectiveRoutes() const {
        std::lock_guard<std::mutex> lock(routes_mutex_);
        return effective_routes_;
    }

    void resetOutputsForRouteChangesLocked(const std::vector<int>& old_routes,
                                           const std::vector<int>& new_routes) {
        for (size_t out_i = 0; out_i < new_routes.size(); ++out_i) {
            if (out_i >= old_routes.size() || old_routes[out_i] != new_routes[out_i])
                last_output_pts_[out_i] = NOTS;
        }
    }

    // Atomically resolve the route table for `pts` *and* publish it as the
    // current effective table so per-output backward-PTS guards stay coherent.
    // Returns the routes actually applied. Holding both locks for the whole
    // window prevents the routesAt -> applyEffectiveRoutes race where a
    // concurrent setObject("routes") could observe a stale effective_routes_
    // and reset last_output_pts_ relative to a now-outdated baseline.
    std::vector<int> resolveAndApplyRoutesAt(const av::Timestamp& pts) {
        std::scoped_lock lock(routes_mutex_, stats_mutex_);
        std::vector<int> resolved;
        if (pts.isValid()) {
            auto opt = tlGetRaw("routes", pts);
            if (opt) {
                resolved = parseRoutes(*opt, &effective_routes_);
            }
        }
        if (resolved.empty())
            resolved = routes_;
        if (effective_routes_ != resolved) {
            resetOutputsForRouteChangesLocked(effective_routes_, resolved);
            effective_routes_ = resolved;
        }
        return resolved;
    }

    void setRoutes(std::vector<int> routes) {
        {
            std::scoped_lock lock(routes_mutex_, stats_mutex_);
            resetOutputsForRouteChangesLocked(effective_routes_, routes);
            routes_ = routes;
            effective_routes_ = std::move(routes);
        }
        for (auto& edge : this->source_edges_)
            edge->producedEvent().signal();
    }

    void validateOutputLabels() {
        if (output_labels_.size() != this->sink_edges_.size()) {
            throw Error("preheat_video_router[" + label_ + "]: labels size " +
                        std::to_string(output_labels_.size()) + " does not match outputs " +
                        std::to_string(this->sink_edges_.size()));
        }
        for (size_t out_i = 0; out_i < output_labels_.size(); ++out_i) {
            const std::string& output_label = output_labels_[out_i];
            if (output_label.empty()) {
                throw Error("preheat_video_router[" + label_ + "]: output label " +
                            std::to_string(out_i) + " must not be empty");
            }
            auto [_, inserted] = output_index_by_label_.emplace(output_label, out_i);
            if (!inserted) {
                throw Error("preheat_video_router[" + label_ + "]: duplicate output label: " +
                            output_label);
            }
        }
    }

    void validateHomogeneousInputs() {
        if (this->source_edges_.empty())
            throw Error("preheat_video_router[" + label_ + "]: requires at least one input");

        // Three independent passes, one per aspect. Mismatch errors name both the
        // offending input and the input that supplied the reference, since "the
        // first input with metadata" can be a non-zero index when earlier inputs
        // lack the interface.
        size_t video_ref_idx = 0;
        std::shared_ptr<IVideoFormatSource> video_ref;
        for (size_t i = 0; i < this->source_edges_.size(); ++i) {
            auto src = findUpstreamInterfaceAt<IVideoFormatSource>(i);
            if (!src)
                continue;
            if (!video_ref) {
                video_ref = src;
                video_ref_idx = i;
                video_format_source_ = src;
                continue;
            }
            if (src->width() != video_ref->width() || src->height() != video_ref->height() ||
                src->pixelFormat() != video_ref->pixelFormat() ||
                src->realPixelFormat() != video_ref->realPixelFormat()) {
                throw Error("preheat_video_router[" + label_ + "]: input " +
                            std::to_string(i) +
                            " video format does not match input " +
                            std::to_string(video_ref_idx));
            }
        }

        size_t frame_rate_ref_idx = 0;
        std::shared_ptr<IFrameRateSource> frame_rate_ref;
        for (size_t i = 0; i < this->source_edges_.size(); ++i) {
            auto src = findUpstreamInterfaceAt<IFrameRateSource>(i);
            if (!src)
                continue;
            if (!frame_rate_ref) {
                frame_rate_ref = src;
                frame_rate_ref_idx = i;
                frame_rate_source_ = src;
                continue;
            }
            if (src->frameRate() != frame_rate_ref->frameRate()) {
                throw Error("preheat_video_router[" + label_ + "]: input " +
                            std::to_string(i) +
                            " frame rate does not match input " +
                            std::to_string(frame_rate_ref_idx));
            }
        }

        size_t time_base_ref_idx = 0;
        std::shared_ptr<ITimeBaseSource> time_base_ref;
        for (size_t i = 0; i < this->source_edges_.size(); ++i) {
            auto src = findUpstreamInterfaceAt<ITimeBaseSource>(i);
            if (!src)
                continue;
            if (!time_base_ref) {
                time_base_ref = src;
                time_base_ref_idx = i;
                time_base_source_ = src;
                continue;
            }
            if (src->timeBase() != time_base_ref->timeBase()) {
                throw Error("preheat_video_router[" + label_ + "]: input " +
                            std::to_string(i) +
                            " timebase does not match input " +
                            std::to_string(time_base_ref_idx));
            }
        }
    }

    bool allInputsFinished() const {
        return std::all_of(input_finished_.begin(), input_finished_.end(), [](bool done) {
            return done;
        });
    }

    void copyToRoutedOutputs(const av::VideoFrame& frame, const std::vector<int>& routes,
                             int input_index) {
        for (size_t out_i = 0; out_i < this->sink_edges_.size(); ++out_i) {
            if (routes[out_i] != input_index)
                continue;

            const av::Timestamp pts = frame.pts();
            if (enforce_monotonic_pts_ && pts.isValid()) {
                std::lock_guard<std::mutex> lock(stats_mutex_);
                if (last_output_pts_[out_i].isValid() && pts < last_output_pts_[out_i]) {
                    const uint64_t drop_count = ++backward_pts_drops_per_output_[out_i];
                    if (drop_count <= kBackwardPtsDropLogLimit) {
                        logstream << "preheat_video_router[" << label_ << "]: dropping backward-pts frame on output "
                                  << output_labels_[out_i] << " route=" << input_index
                                  << " pts=" << pts << " last_output_pts=" << last_output_pts_[out_i];
                    }
                    continue;
                }
            }

            bool enqueued = EdgeSink<av::VideoFrame>(this->sink_edges_[out_i]).put(frame, true);
            std::lock_guard<std::mutex> lock(stats_mutex_);
            if (enqueued) {
                frames_enqueued_per_output_[out_i]++;
                outputs_seen_by_input_[input_index].insert(out_i);
                if (pts.isValid())
                    last_output_pts_[out_i] = pts;
            } else {
                frames_dropped_per_output_[out_i]++;
            }
        }
    }

    void propagateEofForInput(int input_index) {
        std::unordered_set<size_t> target_outputs;
        std::vector<int> current_effective_routes = effectiveRoutes();
        for (size_t out_i = 0; out_i < current_effective_routes.size(); ++out_i) {
            if (current_effective_routes[out_i] == input_index)
                target_outputs.insert(out_i);
        }
        {
            std::lock_guard<std::mutex> lock(stats_mutex_);
            for (size_t out_i : outputs_seen_by_input_[input_index])
                target_outputs.insert(out_i);
        }

        av::VideoFrame eof = createEofMarker<av::VideoFrame>();
        for (size_t out_i : target_outputs) {
            bool enqueued = EdgeSink<av::VideoFrame>(this->sink_edges_[out_i]).put(eof, true);
            std::lock_guard<std::mutex> lock(stats_mutex_);
            if (enqueued)
                frames_enqueued_per_output_[out_i]++;
            else
                frames_dropped_per_output_[out_i]++;
        }
    }

    Parameters vectorToJson(const std::vector<uint64_t>& values) const {
        Parameters arr = Parameters::array();
        for (uint64_t value : values)
            arr.push_back(value);
        return arr;
    }

public:
    bool consumeEofIfPresent() override {
        return false;
    }

    void process() override {
        int srci = this->findSourceWithData(0);
        if (srci < 0) {
            if (this->stopping_ || allInputsFinished())
                this->finished_ = true;
            else
                this->waitForInput();
            return;
        }

        av::VideoFrame* frame = this->source_edges_[srci]->peek();
        if (!frame)
            return;

        if (isEofMarker(*frame)) {
            propagateEofForInput(srci);
            this->source_edges_[srci]->pop();
            input_finished_[srci] = true;
            if (allInputsFinished())
                this->finished_ = true;
            return;
        }

        std::vector<int> routes = resolveAndApplyRoutesAt(frame->pts());
        copyToRoutedOutputs(*frame, routes, srci);
        this->source_edges_[srci]->pop();
        {
            std::lock_guard<std::mutex> lock(stats_mutex_);
            frames_drained_per_input_[srci]++;
        }
    }

    void setObject(const std::string key, const Parameters& value) override {
        if (key == "routes") {
            std::vector<int> base_routes = currentRoutes();
            setRoutes(parseRoutes(value, &base_routes));
            return;
        }
        throw Error("preheat_video_router[" + label_ + "]: unknown object key: " + key);
    }

    Parameters getObject(const std::string key) override {
        if (key == "routes")
            return routesToJson(currentRoutes());
        if (key == "routes_named")
            return routesToNamedJson(currentRoutes());
        if (key != "status")
            throw Error("preheat_video_router[" + label_ + "]: unknown object key: " + key);

        Parameters status;
        status["routes"] = routesToJson(currentRoutes());
        status["routes_named"] = routesToNamedJson(currentRoutes());
        status["labels"] = output_labels_;
        {
            std::lock_guard<std::mutex> lock(stats_mutex_);
            status["frames_drained_per_input"] = vectorToJson(frames_drained_per_input_);
            status["frames_enqueued_per_output"] = vectorToJson(frames_enqueued_per_output_);
            status["frames_dropped_per_output"] = vectorToJson(frames_dropped_per_output_);
            status["backward_pts_drops_per_output"] = vectorToJson(backward_pts_drops_per_output_);
        }
        return status;
    }

    int width() override {
        if (video_format_source_)
            return video_format_source_->width();
        return metadata_width_;
    }

    int height() override {
        if (video_format_source_)
            return video_format_source_->height();
        return metadata_height_;
    }

    av::PixelFormat pixelFormat() override {
        if (video_format_source_)
            return video_format_source_->pixelFormat();
        return metadata_pixel_format_;
    }

    av::PixelFormat realPixelFormat() override {
        if (video_format_source_)
            return video_format_source_->realPixelFormat();
        return metadata_real_pixel_format_;
    }

    av::Rational frameRate() override {
        if (frame_rate_source_)
            return frame_rate_source_->frameRate();
        return metadata_frame_rate_;
    }

    av::Rational timeBase() override {
        if (time_base_source_)
            return time_base_source_->timeBase();
        return metadata_time_base_;
    }

    static std::shared_ptr<PreheatVideoRouter> create(NodeCreationInfo& nci) {
        const Parameters& params = nci.params;
        auto r = std::make_shared<PreheatVideoRouter>();
        r->createSourcesFromParameters(nci.edges, params);
        r->createSinksFromParameters(nci.edges, params);
        r->initTimeline(nci);

        if (params.count("name"))
            r->label_ = params["name"].get<std::string>();

        r->input_finished_.assign(r->source_edges_.size(), false);
        r->last_output_pts_.assign(r->sink_edges_.size(), NOTS);
        r->frames_drained_per_input_.assign(r->source_edges_.size(), 0);
        r->frames_enqueued_per_output_.assign(r->sink_edges_.size(), 0);
        r->frames_dropped_per_output_.assign(r->sink_edges_.size(), 0);
        r->backward_pts_drops_per_output_.assign(r->sink_edges_.size(), 0);
        r->outputs_seen_by_input_.assign(r->source_edges_.size(), {});

        if (!params.count("labels"))
            throw Error("preheat_video_router[" + r->label_ + "]: labels are required");
        for (const std::string& label : jsonToStringList(params["labels"]))
            r->output_labels_.push_back(label);
        r->validateOutputLabels();

        if (params.count("enforce_monotonic_pts"))
            r->enforce_monotonic_pts_ = params["enforce_monotonic_pts"].get<bool>();
        if (params.count("width"))
            r->metadata_width_ = params["width"].get<int>();
        if (params.count("height"))
            r->metadata_height_ = params["height"].get<int>();
        if (params.count("pixel_format"))
            r->metadata_pixel_format_ = av::PixelFormat(params["pixel_format"].get<std::string>());
        if (params.count("real_pixel_format"))
            r->metadata_real_pixel_format_ = av::PixelFormat(params["real_pixel_format"].get<std::string>());
        if (params.count("frame_rate"))
            r->metadata_frame_rate_ = parseRatio(params["frame_rate"]);
        if (params.count("timebase"))
            r->metadata_time_base_ = parseRatio(params["timebase"]);

        if (params.count("routes"))
            r->routes_ = r->parseRoutes(params["routes"]);
        else
            r->routes_.assign(r->sink_edges_.size(), -1);
        r->effective_routes_ = r->routes_;

        r->validateHomogeneousInputs();

        return r;
    }
};

DECLNODE(preheat_video_router, PreheatVideoRouter);
