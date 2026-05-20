#include "node_common.hpp"
#include "../SharedTimeline.hpp"

#include <algorithm>
#include <atomic>
#include <mutex>
#include <vector>

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
    std::vector<std::string> output_labels_;
    std::vector<av::Timestamp> last_output_pts_;
    std::vector<bool> input_finished_;
    std::vector<uint64_t> frames_drained_per_input_;
    std::vector<uint64_t> frames_enqueued_per_output_;
    std::vector<uint64_t> frames_dropped_per_output_;
    std::vector<uint64_t> backward_pts_drops_per_output_;
    std::string label_ = "<unnamed>";

    template <typename Interface>
    std::shared_ptr<Interface> upstreamInterface(const char* interface_name) {
        for (auto& edge : this->source_edges_) {
            auto src = edge->template findNodeUp<Interface>();
            if (src) return src;
        }
        throw Error("preheat_video_router[" + label_ + "]: no upstream " + interface_name);
    }

    std::vector<int> parseRoutes(const Parameters& value) const {
        if (!value.is_array())
            throw Error("preheat_video_router[" + label_ + "]: routes must be an array");
        if (value.size() != this->sink_edges_.size()) {
            throw Error("preheat_video_router[" + label_ + "]: routes size " +
                        std::to_string(value.size()) + " does not match outputs " +
                        std::to_string(this->sink_edges_.size()));
        }

        std::vector<int> parsed;
        parsed.reserve(value.size());
        for (const auto& item : value) {
            int input_index = item.get<int>();
            if (input_index < -1 || input_index >= (int)this->source_edges_.size()) {
                throw Error("preheat_video_router[" + label_ + "]: route input index " +
                            std::to_string(input_index) + " is outside [-1, " +
                            std::to_string(this->source_edges_.size() - 1) + "]");
            }
            parsed.push_back(input_index);
        }
        return parsed;
    }

    Parameters routesToJson(const std::vector<int>& routes) const {
        Parameters arr = Parameters::array();
        for (int input_index : routes)
            arr.push_back(input_index);
        return arr;
    }

    std::vector<int> currentRoutes() const {
        std::lock_guard<std::mutex> lock(routes_mutex_);
        return routes_;
    }

    std::vector<int> routesAt(const av::Timestamp& pts) const {
        if (pts.isValid()) {
            auto opt = tlGetRaw("routes", pts);
            if (opt)
                return parseRoutes(*opt);
        }
        return currentRoutes();
    }

    void setRoutes(std::vector<int> routes) {
        {
            std::lock_guard<std::mutex> lock(routes_mutex_);
            routes_ = std::move(routes);
        }
        for (auto& edge : this->source_edges_)
            edge->producedEvent().signal();
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
            if (pts.isValid() && last_output_pts_[out_i].isValid() &&
                    pts < last_output_pts_[out_i]) {
                std::lock_guard<std::mutex> lock(stats_mutex_);
                backward_pts_drops_per_output_[out_i]++;
                continue;
            }

            bool enqueued = EdgeSink<av::VideoFrame>(this->sink_edges_[out_i]).put(frame, true);
            std::lock_guard<std::mutex> lock(stats_mutex_);
            if (enqueued) {
                frames_enqueued_per_output_[out_i]++;
                if (pts.isValid())
                    last_output_pts_[out_i] = pts;
            } else {
                frames_dropped_per_output_[out_i]++;
            }
        }
    }

    void propagateEofToRoutedOutputs(const std::vector<int>& routes, int input_index) {
        av::VideoFrame eof = createEofMarker<av::VideoFrame>();
        for (size_t out_i = 0; out_i < this->sink_edges_.size(); ++out_i) {
            if (routes[out_i] != input_index)
                continue;
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

        std::vector<int> routes = routesAt(frame->pts());
        if (isEofMarker(*frame)) {
            propagateEofToRoutedOutputs(routes, srci);
            this->source_edges_[srci]->pop();
            input_finished_[srci] = true;
            if (allInputsFinished())
                this->finished_ = true;
            return;
        }

        copyToRoutedOutputs(*frame, routes, srci);
        this->source_edges_[srci]->pop();
        {
            std::lock_guard<std::mutex> lock(stats_mutex_);
            frames_drained_per_input_[srci]++;
        }
    }

    void setObject(const std::string key, const Parameters& value) override {
        if (key == "routes") {
            setRoutes(parseRoutes(value));
            return;
        }
        throw Error("preheat_video_router[" + label_ + "]: unknown object key: " + key);
    }

    Parameters getObject(const std::string key) override {
        if (key == "routes")
            return routesToJson(currentRoutes());
        if (key != "status")
            throw Error("preheat_video_router[" + label_ + "]: unknown object key: " + key);

        Parameters status;
        status["routes"] = routesToJson(currentRoutes());
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

    av::Rational timeBase() override {
        return upstreamInterface<ITimeBaseSource>("timebase source")->timeBase();
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

        if (params.count("labels")) {
            for (const std::string& label : jsonToStringList(params["labels"]))
                r->output_labels_.push_back(label);
        }
        if (r->output_labels_.size() != r->sink_edges_.size()) {
            r->output_labels_.clear();
            for (size_t i = 0; i < r->sink_edges_.size(); ++i)
                r->output_labels_.push_back(std::to_string(i));
        }

        if (params.count("routes"))
            r->routes_ = r->parseRoutes(params["routes"]);
        else
            r->routes_.assign(r->sink_edges_.size(), -1);

        return r;
    }
};

DECLNODE(preheat_video_router, PreheatVideoRouter);
