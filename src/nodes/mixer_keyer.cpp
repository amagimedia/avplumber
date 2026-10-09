#include "hwaccel/cuda_rect_compositor.hpp"
#include "../mixer/primitives/key_fade.hpp"

#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

/// The mixer's downstream keyer (DSK): keys drawn over the program input (`clock_input`), whose
/// frames drive the output (see process()). Keys switch with `active_inputs` (a cut) or
/// `fade_inputs`.
class MixerKeyer : public CudaRectCompositor {
    size_t clock_input_ = 0;
    av::Rational frame_rate_{0, 1};   // the clock input's; paces the key subscriptions
    // Key frames waiting for the program tick they are stamped for. They arrive
    // about one playout latency ahead of that program frame; the bound stops a key
    // on an unrelated clock from retaining its producer's frames indefinitely.
    static constexpr size_t kKeyQueue = 6;
    std::vector<std::deque<av::VideoFrame>> key_queue_;
    std::vector<av::VideoFrame> held_;   // each key's newest frame at or before the current tick
    std::vector<bool> held_valid_;
    bool keys_subscribed_ = false;   // render thread: every key subscription enabled once
    // Key fades. The control thread records, per key, how the latest command that
    // flipped it asked for the change; the render thread reads that as it steps one
    // envelope per key at each program frame. Both under masks_mutex_.
    struct KeyChange {
        double duration_s = 0;   // 0: cut
        avp::mixer::FadeCurve curve = avp::mixer::FadeCurve::Linear;
    };
    std::vector<KeyChange> key_change_;
    std::vector<avp::mixer::KeyFade> key_fade_;
    std::vector<float> key_opacity_;   // per input, for this program frame; the clock input stays 1
    // Keys whose rectangle may be composed alone, over a copy of the program frame, instead of
    // with the whole canvas (`bounded_keys`): per input, none unless the node names them.
    std::vector<bool> bounded_input_;

    // One envelope step per key for the program frame at `pts`, under masks_mutex_.
    // A key whose fade target differs from `active` starts the change its latest
    // command asked for.
    void stepKeyFades(av::Timestamp pts, avp::mixer::SourceMask active, bool cut) {
        const double t = pts.seconds();
        const double period = av_q2d(av_inv_q(frame_rate_.getValue()));
        for (size_t i = 0; i < key_fade_.size(); ++i) {
            if (i == clock_input_) continue;
            const KeyChange &change = key_change_[i];
            key_fade_[i].retarget(active.test((int)i), cut ? 0.0 : change.duration_s, change.curve, t, period);
            key_opacity_[i] = float(key_fade_[i].level(t));
        }
    }

public:
    explicit MixerKeyer(const Config &config)
        : CudaRectCompositor("mixer_keyer", config),
          key_queue_(config.inputs),
          held_(config.inputs),
          held_valid_(config.inputs),
          key_change_(config.inputs),
          key_opacity_(config.inputs, 1.f),
          bounded_input_(config.inputs, false) {}

    // Each clock-input frame renders at once, at its own PTS, over every key
    // above fade level 0, using its newest frame stamped at or before that tick,
    // as the scene playout matches sources (nearest tick). Matching by timestamp
    // rather than arrival keeps steady motion steady: arrival order races the
    // program frame and alternately repeats and skips key frames. Keys are never
    // waited for, so no playout buffer delays the clock input and a late key
    // cannot stall it. With no key visible the frame passes through untouched.
    // With keys visible the whole canvas is composed, unless every visible key is one of
    // `bounded_keys`: then the output is a copy of the program frame on which only the
    // rectangles those keys cover are composed (while they cover less than most of it).
    //
    // Key fades: each key's opacity follows a KeyFade envelope stepped at every
    // program frame's PTS. A fade starts at the first program frame after the
    // command, whether or not the key has a frame to draw yet.
    //
    // Every key stays subscribed, queued and held while it is off, so a key
    // switched on is drawn from the very next program frame instead of waiting
    // for its source's next frame; only keys above level 0 are drawn. The cost is
    // each off key's frames (held plus queued, as for an on key) and a queue
    // step per key frame, but no GPU work.
    void process() override {
        if (sent_eof_) return;
        const size_t clock = clock_input_;
        av::VideoFrame *program = source_edges_[clock]->peek();
        // Step the envelopes on the frame about to be drawn (EOF markers have no PTS).
        if (program && frameUsable(*program)) {
            std::lock_guard<std::mutex> lock(masks_mutex_);
            stepKeyFades(program->pts(), active_inputs_, false);
        }
        if (!keys_subscribed_) {
            for (auto &subscription : subscriptions_)
                if (subscription) subscription->enable(!stopping_);
            keys_subscribed_ = true;
        }
        for (size_t i = 0; i < source_edges_.size(); ++i) {
            if (i == clock) continue;
            while (auto *frame = source_edges_[i]->peek()) {
                if (frameUsable(*frame)) {
                    requireDrawable(*frame);
                    auto &queue = key_queue_[i];
                    queue.push_back(*frame);
                    if (queue.size() > kKeyQueue) {
                        held_[i] = std::move(queue.front());
                        held_valid_[i] = true;
                        queue.pop_front();
                    }
                }
                source_edges_[i]->pop();
            }
        }
        if (!program) {
            this->waitForInput();
            return;
        }
        av::VideoFrame frame = *program;
        source_edges_[clock]->pop();
        if (isEofMarker(frame)) {
            sent_eof_ = true;
            this->sink_->put(frame);
            return;
        }
        if (!frameUsable(frame)) return;
        requireDrawable(frame);
        // A key frame belongs to the tick nearest its timestamp.
        const av::Timestamp tick_end = addTS(frame.pts(),
            av::Timestamp(1, av::Rational(frame_rate_.getDenominator(), 2 * frame_rate_.getNumerator())));
        std::vector<const av::VideoFrame *> sources(source_edges_.size(), nullptr);
        sources[clock] = &frame;
        bool keyed = false, bounded = true;   // bounded: every key drawn on this frame allows it
        for (size_t i = 0; i < sources.size(); ++i) {
            if (i == clock) continue;
            auto &queue = key_queue_[i];
            while (!queue.empty() && queue.front().pts() < tick_end) {
                held_[i] = std::move(queue.front());
                held_valid_[i] = true;
                queue.pop_front();
            }
            if (!held_valid_[i] || !(key_opacity_[i] > 0.f)) continue;
            sources[i] = &held_[i];
            keyed = true;
            bounded = bounded && bounded_input_[i];
        }
        if (keyed) {
            this->sink_->put(compose(frame.pts(), sources, &frame, &key_opacity_, bounded ? &frame : nullptr));
        } else {
            ++frame_counter_;
            this->sink_->put(frame);
        }
    }

    void setObject(const std::string key, const Parameters& value) override {
        if (key == "active_inputs" || key == "fade_inputs") {
            // fade_inputs {"active_inputs": <mask>, "duration_ms": 0..10000, "curve": <name>}: the keys
            // the mask turns on or off fade over duration_ms, 0 cuts. active_inputs <mask> cuts.
            const bool fade = key == "fade_inputs";
            const auto new_mask = avp::mixer::parseSourceMask(fade ? value.at("active_inputs") : value);
            KeyChange change;
            if (fade) {
                const double ms = value.value("duration_ms", 0.0);
                if (!(ms >= 0 && ms <= 10000))
                    throw Error("mixer_keyer: fade_inputs duration_ms must be 0 to 10000");
                change.duration_s = ms / 1000;
                change.curve = avp::mixer::parseFadeCurve(value.value("curve", std::string("linear")));
            }
            {
                std::lock_guard<std::mutex> lock(masks_mutex_);
                // Only the keys this command flips take its fade: it never restarts another key's, and
                // a key it leaves as it is keeps its running fade, even when the command is a cut.
                for (size_t i = 0; i < key_change_.size(); ++i)
                    if (new_mask.test((int)i) != active_inputs_.test((int)i)) key_change_[i] = change;
                active_inputs_ = new_mask;
            }
            // Unlike cuda_rect_overlay, nothing is reset here: the render thread holds every key's
            // latest frame, on air or not, and ends at the program's EOF. Resetting that state here
            // raced the render thread and blanked keys that stay on air.
            wakeInputs();
        } else {
            CudaRectCompositor::setObject(key, value);
        }
    }

    av::Rational frameRate() override { return frame_rate_; }

    static std::shared_ptr<MixerKeyer> create(NodeCreationInfo &nci);
};
std::shared_ptr<MixerKeyer> MixerKeyer::create(NodeCreationInfo &nci) {
    const Parameters &params = nci.params;
    auto node = std::make_shared<MixerKeyer>(parseConfig(nci, "mixer_keyer"));
    node->connect(nci);
    // fps is the clock input's rate: it paces key subscriptions, not a playout.
    if (!params.contains("clock_input") || !params.contains("fps") || params.value("aux_mode", false) ||
        params.contains("latency_ms") || params.contains("pgm_delay_frames"))
        throw Error("mixer_keyer: clock_input and fps are required; aux_mode, latency_ms and pgm_delay_frames are not keyer parameters");
    const int clock = params.at("clock_input").get<int>();
    if (clock < 0 || size_t(clock) >= node->source_edges_.size())
        throw Error("mixer_keyer: clock_input out of range");
    node->clock_input_ = size_t(clock);
    node->frame_rate_ = parseRatio(params.at("fps"));
    // bounded_keys: true for every key, or the inputs of the keys that allow it.
    if (params.contains("bounded_keys")) {
        const Parameters &bounded = params.at("bounded_keys");
        if (bounded.is_boolean()) {
            node->bounded_input_.assign(node->source_edges_.size(), bounded.get<bool>());
        } else if (bounded.is_array()) {
            for (const auto &input : bounded) {
                if (!input.is_number_integer() || input.get<int>() < 0 || size_t(input.get<int>()) >= node->source_edges_.size() ||
                    input.get<int>() == clock)
                    throw Error("mixer_keyer: bounded_keys names the inputs of keys");
                node->bounded_input_[size_t(input.get<int>())] = true;
            }
        } else {
            throw Error("mixer_keyer: bounded_keys must be a boolean or an array of key inputs");
        }
    }
    // Keys configured on come up fully on, as before fades existed.
    for (size_t i = 0; i < node->source_edges_.size(); ++i)
        node->key_fade_.emplace_back(node->active_inputs_.test((int)i));
    // The clock input is an ordinary edge: its empty name subscribes nothing.
    if (params.contains("subscriptions"))
        node->subscribe(nci, node->frame_rate_, true);

    return node;
}

DECLNODE(mixer_keyer, MixerKeyer)
