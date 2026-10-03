#include "hwaccel/cuda_rect_compositor.hpp"
#include "hwaccel/cuda_stream.hpp"
#include "../mixer/Playout.hpp"
#include "../mixer/primitives/MixerState.hpp"
#include "../mixer/primitives/MonotonicClock.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

/// The mixer's clocked compositor: the scene slots, the wipe overlay and, with `aux_mode`, the
/// AUX buses. Live, monotonic-PTS inputs go through a shared Playout, which picks each input's
/// frame for every `fps` output tick; prewarm, warm resets and an AUX bus's staged
/// `composition` are controlled here.
class MixerCompositor : public CudaRectCompositor,
                        public IInputReset,
                        public IReturnsObjects,
                        public IFlushable {
    std::optional<Parameters> pending_composition_;
    std::optional<Parameters> staged_composition_; // render thread only; latest request replaces it
    avp::mixer::SourceMask staged_mask_;
    int64_t staged_deadline_ns_ = 0;
    std::atomic<bool> composition_preparing_{false};
    std::string composition_error_; // protected by layers_mutex_
    std::string composition_revision_; // of the composition being drawn; protected by layers_mutex_
    bool aux_ = false;
    std::atomic<uint64_t> output_drops_{0};
    // Playout counters snapshotted every 60 frames for the status object; the
    // control thread never touches playout_ itself.
    std::mutex playout_stats_mutex_;
    std::vector<std::array<uint64_t, 3>> playout_stats_published_;   // repeats, discarded, overflow
    uint64_t published_missed_deadlines_ = 0;
    uint64_t published_frames_ = 0;

    avp::mixer::SourceMask prewarm_inputs_;   // under masks_mutex_
    std::pair<avp::mixer::SourceMask, avp::mixer::SourceMask> inputMasks() const {
        std::lock_guard<std::mutex> lock(masks_mutex_);
        return {active_inputs_, prewarm_inputs_};
    }

    std::unique_ptr<avp::mixer::Playout<av::VideoFrame>> playout_;
    av::Rational frame_rate_{0, 1};
    std::atomic<uint64_t> input_generation_{0};
    std::atomic<int64_t> input_valid_from_ns_{0};
    std::atomic<bool> preserve_warm_input_{false};
    uint64_t applied_generation_ = 0;
    avp::mixer::SourceMask applied_active_mask_;
    avp::mixer::SourceMask applied_prewarm_mask_;

    void composite(av::Timestamp pts, const std::vector<const av::VideoFrame *> &sources,
                   const av::VideoFrame *metadata_src) {
        if (!this->sink_->put(compose(pts, sources, metadata_src), aux_)) ++output_drops_;
    }

    void auxInputs(avp::mixer::SourceMask active, avp::mixer::SourceMask preparing = {}) {
        const auto keep = active | preparing;
        for (size_t i = 0; i < subscriptions_.size(); ++i) {
            if (!keep.test(i)) {
                subscriptions_[i]->enable(false);
                source_edges_[i]->clear();
            }
            // Promote a warmed input without discarding its eligible frames.
            playout_->setPrewarm(i, keep.test(i));
            playout_->setActive(i, active.test(i));
            playout_->setPrewarm(i, preparing.test(i));
            if (!stopping_ && keep.test(i)) subscriptions_[i]->enable(true);
        }
        std::lock_guard<std::mutex> lock(masks_mutex_);
        sent_eof_ = false;
        active_inputs_ = applied_active_mask_ = active;
        prewarm_inputs_ = applied_prewarm_mask_ = preparing;
    }

    void applyAuxComposition(const Parameters &value, avp::mixer::SourceMask mask) {
        auxInputs(mask);
        std::lock_guard<std::mutex> lock(layers_mutex_);
        default_layers_ = avp::mixer::parseLayersArray(value.at("layers"));
        composition_revision_ = value.value("revision", std::string());
    }

public:
    explicit MixerCompositor(const Config &config) : CudaRectCompositor("mixer_compositor", config) {}

    ~MixerCompositor() override {
        MixerCompositor::flush();
    }

    void flush() override {
        if (!aux_) return;
        for (auto &subscription : subscriptions_) subscription->close();
        draw_.ensureDevice();
        AVP_CHECK_CU(cuStreamSynchronize(draw_.stream()));
        for (auto &edge : source_edges_) edge->clear();
        playout_.reset();
    }

    Parameters getObject(const std::string key) override {
        if (key != "status") throw Error("mixer_compositor: unknown object " + key);
        Parameters result = {{"output_drops", output_drops_.load()}};
        {
            std::lock_guard<std::mutex> lock(playout_stats_mutex_);
            Parameters playout = Parameters::object();
            playout["frames"] = published_frames_;
            playout["missed_deadlines"] = published_missed_deadlines_;
            uint64_t repeats = 0, discarded = 0, overflow = 0;
            Parameters per_input = Parameters::array();
            for (const auto &st : playout_stats_published_) {
                repeats += st[0]; discarded += st[1]; overflow += st[2];
                per_input.push_back({{"repeats", st[0]}, {"discarded", st[1]}, {"overflow", st[2]}});
            }
            playout["repeats"] = repeats;
            playout["discarded"] = discarded;
            playout["overflow"] = overflow;
            playout["per_input"] = per_input;
            result["playout"] = playout;
        }
        if (aux_) {
            std::lock_guard<std::mutex> lock(layers_mutex_);
            result["composition_pending"] = pending_composition_.has_value() || composition_preparing_.load();
            result["composition_error"] = composition_error_;
            result["composition_revision"] = composition_revision_;
        }
        return result;
    }

    void process() override {
        if (aux_) {
            std::optional<Parameters> update;
            {
                std::lock_guard<std::mutex> lock(layers_mutex_);
                update.swap(pending_composition_);
            }
            if (update) {
                const auto mask = avp::mixer::parseSourceMask(update->at("active_inputs"));
                staged_composition_.reset();
                composition_preparing_ = false;
                if (!frame_counter_) {
                    applyAuxComposition(*update, mask);
                    warmup_started_pts_ = wallclock.pts();
                } else {
                    staged_mask_ = mask;
                    staged_deadline_ns_ = avp::mixer::monotonicNs() +
                        std::max<int64_t>(250000000, 2 * playout_->latencyNs());
                    staged_composition_ = std::move(update);
                    composition_preparing_ = true;
                    auxInputs(applied_active_mask_, mask);
                }
            }
        }
        const int64_t now = avp::mixer::monotonicNs();
        auto [active, prewarm] = inputMasks();
        if (hasTimeline()) {
            const auto deadline = playout_->nextDeadline();
            const int64_t content_time = std::max(now - playout_->latencyNs(),
                deadline ? *deadline - playout_->latencyNs() : now);
            const auto value = tlGetRaw("active_inputs", av::Timestamp(content_time, {1, 1000000000}));
            if (value) active = avp::mixer::parseSourceMask(*value);
        }
        const auto generation = input_generation_.load(std::memory_order_acquire);
        if (generation != applied_generation_ || active != applied_active_mask_ || prewarm != applied_prewarm_mask_) {
            for (size_t i = 0; i < source_edges_.size(); ++i) {
                playout_->setPrewarm(i, prewarm.test((int)i));
                playout_->setActive(i, active.test((int)i));
                if (generation != applied_generation_) {
                    const auto from = input_valid_from_ns_.load(std::memory_order_acquire);
                    playout_->resetInput(i, from ? std::optional<int64_t>(from) : std::nullopt,
                                         preserve_warm_input_.load(std::memory_order_acquire));
                }
            }
            applied_generation_ = generation;
            applied_active_mask_ = active;
            applied_prewarm_mask_ = prewarm;
            warmup_started_pts_ = wallclock.pts();
            sent_eof_ = false;
        }
        if (sent_eof_) return;
        if (!(active | prewarm).any()) {
            this->waitForInput();
            return;
        }
        for (size_t i = 0; i < source_edges_.size(); ++i) {
            if (!(active | prewarm).test((int)i)) continue;
            // Bound ingestion as well as storage; an unpaced producer must not
            // monopolize the output thread before it reaches its deadline.
            for (size_t received = 0; received < 8; ++received) {
                auto *frame = source_edges_[i]->peek();
                if (!frame) break;
                if (isEofMarker(*frame)) {
                    playout_->endInput(i);
                    source_edges_[i]->pop();
                    break;
                }
                if (frameUsable(*frame)) {
                    requireDrawable(*frame);
                    playout_->push(i, *frame, frame->pts().timestamp({1, 1000000000}));
                }
                source_edges_[i]->pop();
            }
        }
        if (staged_composition_) {
            const auto now_ns = avp::mixer::monotonicNs();
            bool ready = true;
            for (size_t i = 0; i < source_edges_.size(); ++i)
                if (staged_mask_.test(i) && !playout_->readyAtNextTick(i, now_ns)) ready = false;
            if (ready || now_ns >= staged_deadline_ns_) {
                if (ready) {
                    applyAuxComposition(*staged_composition_, staged_mask_);
                    active = staged_mask_;
                } else {
                    auxInputs(active);
                    std::lock_guard<std::mutex> lock(layers_mutex_);
                    if (!pending_composition_)
                        composition_error_ = "Aux inputs not ready; keeping the previous layout";
                }
                staged_composition_.reset();
                composition_preparing_ = false;
            }
        }
        if (playout_->finished()) {
            av::VideoFrame eof;
            eof.setPts(NOTS);
            this->sink_->put(eof, aux_);
            sent_eof_ = true;
            return;
        }
        const bool require_all = (!aux_ || !frame_counter_) &&
            (warmup_timeout_ms_ <= 0 || wallclock.pts() - warmup_started_pts_ < warmup_timeout_ms_);
        const auto *decision = playout_->prepare(avp::mixer::monotonicNs(), require_all);
        if (!decision) {
            this->findSourceWithData(avp::mixer::waitMilliseconds(
                playout_->nextDeadline(), avp::mixer::monotonicNs()));
            return;
        }
        std::vector<const av::VideoFrame *> sources;
        const av::VideoFrame *metadata = nullptr;
        for (size_t i = 0; i < decision->frames.size(); ++i) {
            const auto &frame = decision->frames[i];
            const bool visible = active.test((int)i);
            sources.push_back(visible && frame ? &*frame : nullptr);
            if (visible && frame) metadata = &*frame;
        }
        // Warm inputs advance their bounded reference queues without allocating
        // an output surface or issuing any CUDA composition for an idle slot.
        if (active.any()) {
            // A full aux output costs this tick's frame only: the playout still advances, so the
            // next tick the encoder has room for draws current pictures.
            if (aux_ && output_edge_->occupied() >= int(output_edge_->capacity()))
                ++output_drops_;
            else
                composite(av::Timestamp(decision->index, av_inv_q(frame_rate_.getValue())), sources, metadata);
        }
        playout_->commit();
        if (frame_counter_ % 60 == 0) {
            std::vector<std::array<uint64_t, 3>> snapshot;
            snapshot.reserve(sources.size());
            for (size_t i = 0; i < sources.size(); ++i) {
                const auto &st = playout_->stats(i);
                snapshot.push_back({st.repeats, st.discarded, st.overflow});
            }
            std::lock_guard<std::mutex> lock(playout_stats_mutex_);
            playout_stats_published_ = std::move(snapshot);
            published_missed_deadlines_ = playout_->missedDeadlines();
            published_frames_ = frame_counter_;
        }
        if (debug_log_every_n_ > 0 && frame_counter_ % debug_log_every_n_ == 0) {
            std::ostringstream stats;
            stats << "mixer_compositor: frames=" << frame_counter_
                  << " missed_deadlines=" << playout_->missedDeadlines();
            for (size_t i = 0; i < sources.size(); ++i) {
                if (!active.test((int)i)) continue;
                stats << " input" << i << "_reuse=" << playout_->stats(i).repeats
                      << " input" << i << "_discarded=" << playout_->stats(i).discarded;
            }
            logstream << stats.str();
        }
    }

    void resetInput() override {
        if (!playout_) return;
        preserve_warm_input_.store(false, std::memory_order_release);
        input_valid_from_ns_.store(avp::mixer::monotonicNs(), std::memory_order_release);
        input_generation_.fetch_add(1, std::memory_order_release);
        wakeInputs();
    }

    void setObject(const std::string key, const Parameters& value) override {
        if (key == "composition") {
            if (!aux_) throw Error("composition requires aux_mode");
            auto layers = avp::mixer::parseLayersArray(value.at("layers"));
            if (layers.size() > size_t(draw_.maxLayers())) throw Error("aux: too many layers");
            const auto mask = avp::mixer::parseSourceMask(value.at("active_inputs"));
            if (value.contains("revision") && !value.at("revision").is_string()) throw Error("aux: revision must be a string");
            for (const auto &layer : layers)
                if (layer.input < 0 || size_t(layer.input) >= source_edges_.size()) throw Error("aux: invalid input index");
            for (int i = static_cast<int>(source_edges_.size()); i < avp::mixer::kSourceMaskBits; ++i)
                if (mask.test(i)) throw Error("aux: active input out of range");
            {
                std::lock_guard<std::mutex> lock(layers_mutex_);
                pending_composition_ = value;
                composition_error_.clear();
            }
            stop_event_.signal();
        } else if (key == "active_inputs") {
            setActiveInputs(avp::mixer::parseSourceMask(value));
            wakeInputs();
        } else if (key == "prewarm_inputs") {
            if (!playout_) throw Error("mixer_compositor: prewarm_inputs requires clocked playout");
            {
                std::lock_guard<std::mutex> lock(masks_mutex_);
                prewarm_inputs_ = avp::mixer::parseSourceMask(value);
            }
            wakeInputs();
        } else if (key == "warm_reset") {
            if (!playout_) throw Error("mixer_compositor: warm_reset requires clocked playout");
            const auto period = avp::mixer::TickGrid(frame_rate_).time(1);
            input_valid_from_ns_.store(avp::mixer::monotonicNs() - playout_->latencyNs() - period,
                                       std::memory_order_release);
            preserve_warm_input_.store(true, std::memory_order_release);
            input_generation_.fetch_add(1, std::memory_order_release);
            wakeInputs();
        } else {
            CudaRectCompositor::setObject(key, value);
        }
    }

    av::Rational frameRate() override { return frame_rate_; }

    static std::shared_ptr<MixerCompositor> create(NodeCreationInfo &nci);
};
std::shared_ptr<MixerCompositor> MixerCompositor::create(NodeCreationInfo &nci) {
    const Parameters &params = nci.params;
    Config config = parseConfig(nci, "mixer_compositor");
    const bool aux = params.value("aux_mode", false);
    if (aux) {
        config.hw = avp::mixer::makeCudaStreamDevice(config.hw);
        InstanceSharedObjects<HWAccelDevice>::put(nci.instance, params.at("output_hwaccel"), config.hw);
    }
    auto node = std::make_shared<MixerCompositor>(config);
    node->connect(nci);
    node->aux_ = aux;
    node->frame_layers_ = !aux;
    if (!params.contains("fps") || params.contains("clock_input"))
        throw Error("mixer_compositor: fps is required; clock_input needs mixer_keyer");
    node->frame_rate_ = parseRatio(params.at("fps"));
    std::optional<double> latency_ms;
    if (params.contains("latency_ms")) latency_ms = params.at("latency_ms").get<double>();
    node->playout_ = std::make_unique<avp::mixer::Playout<av::VideoFrame>>(
        config.inputs, avp::mixer::TickGrid(node->frame_rate_), latency_ms, avp::mixer::TimestampMode::Presentation);
    node->input_generation_.store(1);
    logstream << "mixer_compositor: latency_ms=" << static_cast<double>(node->playout_->latencyNs()) / 1e6;
    // A Program preview's last input is the finished program, which arrives one frame after
    // the sources it is made of: match it that many frames back instead of raising the
    // latency of every input.
    if (params.contains("pgm_delay_frames")) {
        try {
            node->playout_->setInputOffset(config.inputs - 1, params.at("pgm_delay_frames").get<int64_t>());
        } catch (const std::invalid_argument &e) {
            throw Error(std::string("mixer_compositor: pgm_delay_frames: ") + e.what());
        }
    }
    if (aux) {
        if (params.contains("mixer")) {
            const auto state = InstanceSharedObjects<avp::mixer::MixerState>::get(nci.instance, params.at("mixer"));
            std::lock_guard<std::mutex> lock(state->mutex);
            state->scene_definitions_frozen = true;
        }
        if (!params.contains("subscriptions")) throw Error("aux: subscriptions are required");
    }
    if (params.contains("subscriptions")) {
        if (!aux) throw Error("mixer_compositor: subscriptions need aux_mode");
        node->subscribe(nci, node->frame_rate_, false);
    }
    if (aux)
        node->setObject("composition", {{"layers", params.at("layers")}, {"active_inputs", params.value("active_inputs", Parameters(0))}});

    return node;
}

DECLNODE(mixer_compositor, MixerCompositor)
