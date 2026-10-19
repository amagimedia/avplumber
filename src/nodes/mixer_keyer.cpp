#include "hwaccel/cuda_rect_compositor.hpp"
#include "../mixer/primitives/key_fade.hpp"
#include "../mixer/primitives/MixerState.hpp"

#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

/// The mixer's downstream keyer (DSK): keys drawn over the program input (`clock_input`), whose
/// frames drive the output (see process()). Keys switch with `active_inputs` (a cut),
/// `fade_inputs` or `key_changes`; the Next Transition keys (`prepare_inputs`) switch with the
/// background transition, following its clock; `ftb_input` is the fade to black.
class MixerKeyer : public CudaRectCompositor, public IReturnsObjects {
    size_t clock_input_ = 0;
    av::Rational frame_rate_{0, 1};   // the clock input's grid for key timestamps and fades
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
    std::vector<float> key_opacity_;   // render thread: per input, for this program frame; the clock input stays 1
    // Keys whose rectangle may be composed alone, over a copy of the program frame, instead of
    // with the whole canvas (`bounded_keys`): per input, none unless the node names them.
    std::vector<bool> bounded_input_;
    struct KeyTally {
        bool known = false, available = false, target = false, moving = false;
        double opacity = 0;
        int64_t pts_ns = 0;
    };
    std::vector<KeyTally> tally_; // last emitted program frame, under masks_mutex_
    // Next Transition: the keys a background take carries (prepare_inputs). The orchestrator
    // publishes the take's clock (TakeClock); the render thread reads it, never the mixer's mutex.
    std::shared_ptr<avp::mixer::MixerState> mixer_;
    avp::mixer::SourceMask prepared_;
    // Under masks_mutex_: the take whose keys are switching, those keys (the selection at the
    // take's first frame here) and which of them were on then.
    uint64_t take_generation_ = 0;
    avp::mixer::SourceMask taken_, taken_on_;
    // Fade to black (`ftb_input`): an input nothing feeds, holding one opaque black frame that
    // its layer stretches over the canvas on top of everything; its key fade is the fade.
    std::optional<size_t> ftb_input_;

    // One envelope step per key for the program frame at `pts`, under masks_mutex_. A key whose
    // fade target differs from active_inputs_ starts the change its latest command asked for;
    // the keys a take carries follow that take instead.
    void stepKeyFades(av::Timestamp pts) {
        if (mixer_) followTake(*mixer_->clock(), pts.timestamp({1, 1000000000}));
        const double t = pts.seconds();
        const double period = av_q2d(av_inv_q(frame_rate_.getValue()));
        for (size_t i = 0; i < key_fade_.size(); ++i) {
            if (i == clock_input_ || taken_.test(int(i))) continue;
            const KeyChange &change = key_change_[i];
            key_fade_[i].retarget(active_inputs_.test((int)i), change.duration_s, change.curve, t, period);
            key_opacity_[i] = float(key_fade_[i].level(t));
        }
    }

    // A key the take turns on is at the take's progress, one it turns off at the complement,
    // so they mix, pause, follow the lever and cut with the background. Completed, they keep
    // their new state; cancelled, they return to the state they had.
    void followTake(const avp::mixer::TakeClock &clock, int64_t t_ns) {
        using End = avp::mixer::TakeClock::End;
        if (clock.generation != take_generation_) {
            if (taken_.any())   // the take carrying them ended between two of this keyer's frames
                endTake(clock.previous.generation == take_generation_ && clock.previous.end == End::Completed);
            if (clock.end != End::Cancelled && t_ns < clock.on_air_ns) return;   // not on air here yet
            take_generation_ = clock.generation;
            if (clock.end == End::Cancelled) return;
            taken_ = prepared_;
            taken_on_ = active_inputs_ & taken_;
        }
        if (taken_.none()) return;
        if (clock.end == End::Cancelled || (clock.end == End::Completed && t_ns >= clock.end_ns)) {
            endTake(clock.end == End::Completed);
            return;
        }
        const double progress = clock.progress(t_ns);
        for (size_t i = 0; i < key_fade_.size(); ++i)
            if (taken_.test(int(i))) key_opacity_[i] = float(taken_on_.test(int(i)) ? 1 - progress : progress);
    }

    void endTake(bool completed) {
        for (size_t i = 0; i < key_fade_.size(); ++i) {
            if (!taken_.test(int(i))) continue;
            const bool on = completed != taken_on_.test(int(i));
            active_inputs_.set(int(i), on);
            key_fade_[i] = avp::mixer::KeyFade(on);
            key_opacity_[i] = on ? 1.f : 0.f;
        }
        taken_.reset();
    }

    // A command for a key a take carries ends the take for that key: it returns to the state
    // it had, and the command applies from there. Control thread: key_opacity_ is the render
    // thread's, which reads it unlocked; stepKeyFades sets it from key_fade_ on the next frame.
    void releaseTaken(avp::mixer::SourceMask keys) {
        const auto released = taken_ & keys;
        for (size_t i = 0; i < key_fade_.size(); ++i)
            if (released.test(int(i))) key_fade_[i] = avp::mixer::KeyFade(taken_on_.test(int(i)));
        taken_ &= ~released;
    }

    static av::VideoFrame blackFrame(HWAccelDevice &hw) {
        avp::AvBufferRef frames(av_hwframe_ctx_alloc(hw.deviceContext()));
        if (!frames) throw Error("mixer_keyer: av_hwframe_ctx_alloc failed");
        auto *fc = (AVHWFramesContext *)frames->data;
        fc->format = AV_PIX_FMT_CUDA;
        fc->sw_format = AV_PIX_FMT_BGRA;
        fc->width = fc->height = 16;
        if (int err = av_hwframe_ctx_init(frames.get()); err < 0)
            throw Error("mixer_keyer: av_hwframe_ctx_init (fade to black) failed: " + av::error2string(err));
        av::VideoFrame black = avp::mixer::canvasFrame(frames.get(), "mixer_keyer");
        auto *device = (AVHWDeviceContext *)hw.deviceContext()->data;
        CUcontext ctx = ((AVCUDADeviceContext *)device->hwctx)->cuda_ctx;
        bool ok = !AVP_CHECK_CU(cuCtxPushCurrent(ctx));
        if (ok) {
            // BGRA bytes 0, 0, 0, 255: opaque black.
            ok = !AVP_CHECK_CU(cuMemsetD2D32((CUdeviceptr)(uintptr_t)black.raw()->data[0], black.raw()->linesize[0],
                                             0xFF000000u, 16, 16)) && !AVP_CHECK_CU(cuCtxSynchronize());
            CUcontext popped;
            cuCtxPopCurrent(&popped);
        }
        if (!ok) throw Error("mixer_keyer: cannot clear the fade-to-black frame");
        return black;
    }

public:
    explicit MixerKeyer(const Config &config)
        : CudaRectCompositor("mixer_keyer", config),
          key_queue_(config.inputs),
          held_(config.inputs),
          held_valid_(config.inputs),
          key_change_(config.inputs),
          key_opacity_(config.inputs, 1.f),
          bounded_input_(config.inputs, false), tally_(config.inputs) {}

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
            stepKeyFades(program->pts());
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
        {
            std::lock_guard<std::mutex> lock(masks_mutex_);
            for (size_t i = 0; i < tally_.size(); ++i) {
                if (i == clock) continue;
                const bool taken = taken_.test(int(i));
                tally_[i] = {true, held_valid_[i], taken ? !taken_on_.test(int(i)) : key_fade_[i].target(),
                             taken || key_fade_[i].moving(), key_opacity_[i], frame.pts().timestamp({1, 1000000000})};
            }
        }
    }

    Parameters getObject(const std::string key) override {
        if (key != "keys") throw Error("mixer_keyer: unknown object: " + key);
        std::lock_guard<std::mutex> lock(masks_mutex_);
        Parameters rows = Parameters::array();
        for (size_t i = 0; i < tally_.size(); ++i) {
            if (i == clock_input_) continue;
            const auto &t = tally_[i];
            const char *phase = !t.known ? "unknown" : t.moving ? (t.target ? "fading-in" : "fading-out")
                : t.opacity > 0 && t.available ? "on" : "off";
            rows.push_back({{"input", i}, {"on", active_inputs_.test(int(i))},
                {"on_air", t.known ? Parameters(t.available && t.opacity > 0) : Parameters()},
                {"phase", phase}, {"opacity", t.known ? Parameters(t.opacity) : Parameters()},
                {"available", t.available}, {"pts_ns", t.pts_ns},
                {"pending", !t.known || active_inputs_.test(int(i)) != t.target},
                {"next_take", prepared_.test(int(i))}});
        }
        return rows;
    }

    void setObject(const std::string key, const Parameters& value) override {
        if (key == "prepare_inputs" || key == "key_changes") {
            // prepare_inputs {"inputs": <mask>}: the Next Transition keys. key_changes {"inputs": <mask>,
            // "active_inputs": <mask>, "toggle": bool, "duration_ms": 0..10000, "curve": <name>}: those
            // keys change together on the next program frame; 0 cuts, ending any running fade.
            const auto affected = avp::mixer::parseSourceMask(value.at("inputs"));
            if ((affected >> key_change_.size()).any() || affected.test(int(clock_input_)) ||
                (ftb_input_ && key == "prepare_inputs" && affected.test(int(*ftb_input_))))
                throw Error("mixer_keyer: inputs must name keys, not program or the fade to black");
            std::lock_guard<std::mutex> lock(masks_mutex_);
            if (key == "prepare_inputs") {
                prepared_ = affected;
            } else {
                const auto target = avp::mixer::parseSourceMask(value.value("active_inputs", Parameters(0u)));
                const double ms = value.value("duration_ms", 0.0);
                if (!(ms >= 0 && ms <= 10000)) throw Error("mixer_keyer: duration_ms must be 0 to 10000");
                const KeyChange change{ms / 1000, avp::mixer::parseFadeCurve(value.value("curve", std::string("linear")))};
                releaseTaken(affected);
                for (size_t i = 0; i < key_change_.size(); ++i) {
                    if (i == clock_input_ || !affected.test(int(i))) continue;
                    const bool on = value.value("toggle", false) ? !active_inputs_.test(int(i)) : target.test(int(i));
                    active_inputs_.set(int(i), on);
                    key_change_[i] = change;
                    if (ms == 0) key_fade_[i] = avp::mixer::KeyFade(on);
                }
            }
            wakeInputs();
        } else if (key == "active_inputs" || key == "fade_inputs") {
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
                releaseTaken(new_mask ^ active_inputs_);
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
    const Config config = parseConfig(nci, "mixer_keyer");
    auto node = std::make_shared<MixerKeyer>(config);
    node->connect(nci);
    // fps is the clock input's rate for timestamp matching, not a separate playout.
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
    if (params.contains("mixer"))
        node->mixer_ = InstanceSharedObjects<avp::mixer::MixerState>::get(nci.instance, params.at("mixer"));
    if (params.contains("ftb_input")) {
        const int ftb = params.at("ftb_input").get<int>();
        if (ftb < 0 || size_t(ftb) >= node->source_edges_.size() || ftb == clock)
            throw Error("mixer_keyer: ftb_input names a key input");
        node->ftb_input_ = size_t(ftb);
        node->held_[size_t(ftb)] = blackFrame(*config.hw);   // nothing feeds this input: held for good
        node->held_valid_[size_t(ftb)] = true;
    }
    // The clock input is an ordinary edge: its empty name subscribes nothing.
    // A held copy can arrive before the live paint for the same timestamp. Let
    // the keyer's timestamp queue replace it; first-arrival rate limiting would
    // discard that live paint even when it precedes the program's render deadline.
    if (params.contains("subscriptions"))
        node->subscribe(nci, node->frame_rate_, true, false);

    return node;
}

DECLNODE(mixer_keyer, MixerKeyer)
