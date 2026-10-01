#include "node_common.hpp"
#include "../mixer/primitives/MixerState.hpp"
#include "../mixer/primitives/MonotonicClock.hpp"
#include "../mixer/primitives/PreviewFollow.hpp"
#include "../mixer/primitives/source_mask.hpp"
#include <atomic>
#include <chrono>
#include <optional>
#include <unordered_map>

// One per pgm_pvw_grid AUX bus. The mixer publishes its preview changes
// (MixerState::publishPreview); this node draws them in the bus compositor's PVW cell through
// the compositor's `composition` object, on the multiview frame that leaves the bus when the
// program frame of the take leaves the mixer (`align` "program", the default) or whose PGM
// tile shows that frame ("pgm_tile"), see PreviewFollow.hpp. Every timed change is measured
// from the take command's receipt to the aux deadline it is drawn at (`pvw_latency_ms`), next
// to the program frame's own deadline (`pgm_latency_ms`; the cut probe's `cut_latency` ends at
// the encoder's output instead), and their difference `pvw_minus_pgm_ms`, with the take's
// `kind`, in this node's status and in `mixer.status` `pvw_latency`. The control side
// publishes the layouts once: `pvw`, the PVW-cell layers of every scene, and `base`, the rest
// of the composition (the operator's tiles and the PGM tile); every composition this node sets
// is pvw[shown] followed by base, and it is the only writer of the compositor's composition
// while it runs.
//
// The scene that leaves program becomes the preview (the swap), so a settle pass after every
// take adds the program scene's sources to the active inputs (no layer draws them: one
// subscription push per source per aux tick, no compositing), and every other apply keeps the
// program's inputs that are active already. The timed composition of the next take then only
// drops inputs, which the compositor applies at once instead of staging. Residual: a settle the
// compositor stages and rejects after its deadline (a stalled program source) leaves that input
// in applied_inputs_ without being active, so the next timed apply adds it and is staged too.
//
// The compositor is resolved once, at creation: a lookup by name takes NodeManager's lock, which
// shutdown holds while it joins this thread. A change travels from the feed's lock
// (MixerState::preview_mutex) to the compositor's `composition` without waiting on the mixer's
// `mutex`: the orchestrator holds that while it routes a take, which is when a timed change
// has to reach the compositor, and the compositor's `status` is read under it (the compositor
// reports the mixer's `pvw_scene`). So the status (`suspended`, see `suspended_`) is read only
// on a wake with nothing to apply while no take is in flight (`transition_mode`, lock-free).
// Such a read can still meet a take that started after the check and wait for its routing,
// which takes NodeManager's lock under `mutex`: during shutdown that is a lock-order risk this
// node narrows to that window and cannot remove while the aux compositor holds the mixer.
class MixerPvwFollow : public Node, public IStoppable, public IInputsObjects, public IReturnsObjects {
    struct Layout {
        Parameters layers = Parameters::array();
        avp::mixer::SourceMask inputs;
    };
    struct Change : avp::mixer::MixerState::PreviewChange {
        int64_t target = 0;   // aux tick it changes the PVW tile on; 0: at once
    };
    std::shared_ptr<avp::mixer::MixerState> state_;
    const std::string name_;
    const std::string compositor_name_;
    const std::weak_ptr<NodeWrapper> compositor_;
    const avp::mixer::PreviewFollowTiming timing_;
    // Layouts and status, under layouts_mutex_, which is never held while taking another lock.
    std::mutex layouts_mutex_;
    Layout base_;
    std::string base_revision_;
    std::unordered_map<std::string, Layout> pvw_;
    Parameters status_ = Parameters::object();
    std::string error_;
    // Wake reasons besides the mixer's revision. The base is flagged under the feed's lock; the
    // stop is not, since the framework requests it under locks the orchestrator takes after
    // state_->mutex: a wake lost that way lasts one aux tick.
    bool base_dirty_ = true;
    std::atomic<bool> stopping_{false};
    // Follower thread only.
    std::optional<uint64_t> seen_;
    std::optional<Change> pending_;
    Change applied_;
    avp::mixer::SourceMask applied_inputs_;   // the active inputs the compositor last accepted
    // The compositor's `suspended` as this node last set or read it. Every composition sets it
    // (`enabled`, unset resumes), so after an apply it is what was sent; only the compositor's
    // own suspension (encoder backpressure) diverges it, until the next read between takes.
    // A bus that suspended since then gets one resuming composition (as the 50 ms status
    // poller before this node could send) and suspends again after three blocked ticks; so
    // does one suspended before this node started, as the first composition sent resumes it.
    bool suspended_ = false;
    std::optional<int64_t> settle_at_;   // when to add the program scene's inputs, for the swap
    bool dirty_retry_ = false;           // a base apply failed: again at the next wake
    std::string warning_;                // from the last apply, shown as the status error

    std::shared_ptr<NodeWrapper> compositor() const {
        auto node = compositor_.lock();
        if (!node) throw Error("mixer_pvw_follow: compositor " + compositor_name_ + " is gone");
        return node;
    }

    static Layout parseLayout(const Parameters& value) {
        if (!value.at("layers").is_array()) throw Error("mixer_pvw_follow: layers must be an array");
        return {value.at("layers"), avp::mixer::parseSourceMask(value.at("active_inputs"))};
    }

    // Caller holds layouts_mutex_.
    const Layout* layout(const std::string& scene) const {
        const auto it = pvw_.find(scene);
        return it == pvw_.end() ? nullptr : &it->second;
    }

    /// pvw[shown] ++ base; the program scene's inputs stay warm for the swap: `settle` adds
    /// them all, otherwise those active already are kept, so the composition adds nothing.
    /// `enabled` unset resumes a suspended bus, as an operator's change does; a preview change
    /// keeps it suspended. A scene without a PVW layout draws an empty cell and is reported
    /// (`warning_`), not retried.
    void apply(const Change& change, bool settle, std::optional<bool> enabled) {
        Parameters composition;
        avp::mixer::SourceMask inputs;
        {
            std::lock_guard<std::mutex> lock(layouts_mutex_);
            const Layout* pvw = change.pvw.empty() ? nullptr : layout(change.pvw);
            warning_ = !change.pvw.empty() && !pvw ? "no PVW layout for scene " + change.pvw : "";
            Parameters layers = pvw ? pvw->layers : Parameters::array();
            layers.insert(layers.end(), base_.layers.begin(), base_.layers.end());
            inputs = base_.inputs | (pvw ? pvw->inputs : avp::mixer::SourceMask{});
            if (const Layout* pgm = change.pgm.empty() ? nullptr : layout(change.pgm))
                inputs |= settle ? pgm->inputs : pgm->inputs & applied_inputs_;
            composition = {{"layers", std::move(layers)}, {"active_inputs", avp::mixer::toParameters(inputs)}};
        }
        if (enabled) composition["enabled"] = *enabled;
        compositor()->setObject("composition", composition);
        applied_inputs_ = inputs;
        suspended_ = enabled ? !*enabled : false;
    }

    /// Waits on the compositor's start/stop lock (bounded) and on the mixer's `mutex` (the
    /// compositor reports `pvw_scene` from it), so it is called between takes only, and throws
    /// while the compositor is not created: a failed read must not pass for "not suspended",
    /// which would resume the bus.
    bool compositorSuspended() {
        return compositor()->getObject("status").value("suspended", false);
    }

    void report(const std::string& error) {
        std::lock_guard<std::mutex> lock(layouts_mutex_);
        if (!error.empty() && error != error_) logstream << "mixer_pvw_follow: " << error;
        error_ = error;
    }

    /// The status of `change`, applied at `applied_at`: what it drew and when, and for a timed
    /// change (a cut or fade) its latency from the take's receipt against the program's. Both
    /// end at compositor deadlines: the aux tick the change is first drawn on (the tick its
    /// apply time falls in, so a late apply shows as a later tick) and the program frame's.
    /// `target_unreachable`: the target tick was being drawn, or gone, when the change was
    /// published (a fade's first frame past its end is read after it was presented), so the
    /// tick after it is where the change lands, not a miss of this node's.
    Parameters sample(const Change& change, int64_t applied_at) const {
        Parameters status = {{"pvw_scene", change.pvw}, {"pgm_scene", change.pgm}, {"kind", change.kind},
                             {"applied_revision", change.revision}, {"target_tick", change.target},
                             {"align", timing_.align == avp::mixer::PreviewAlign::Program ? "program" : "pgm_tile"},
                             {"last_change_to_apply_ms", (applied_at - change.published_ns) / 1e6},
                             {"last_target_error_ticks", 0}, {"target_unreachable", false},
                             {"pvw_latency_ms", nullptr}, {"pgm_latency_ms", nullptr}, {"pvw_minus_pgm_ms", nullptr}};
        if (!change.target) return status;
        const int64_t drawn = timing_.tickDrawnAfter(applied_at);
        const int64_t pvw = timing_.deadline(drawn), pgm = timing_.programDeparture(change.effective_ns);
        status["last_target_error_ticks"] = drawn - change.target;
        status["target_unreachable"] = timing_.deadline(change.target) <= change.published_ns;
        status["pvw_minus_pgm_ms"] = (pvw - pgm) / 1e6;
        if (change.received_ns) {
            status["pvw_latency_ms"] = (pvw - change.received_ns) / 1e6;
            status["pgm_latency_ms"] = (pgm - change.received_ns) / 1e6;
        }
        return status;
    }

public:
    MixerPvwFollow(std::shared_ptr<avp::mixer::MixerState> state, std::string name, std::string compositor_name,
                   std::shared_ptr<NodeWrapper> compositor, avp::mixer::PreviewFollowTiming timing)
        : state_(std::move(state)), name_(std::move(name)), compositor_name_(std::move(compositor_name)),
          compositor_(compositor), timing_(timing) {
        ++state_->preview_followers;
    }
    // The last sample stays in the mixer's status: a bus is not removed while the mixer runs.
    ~MixerPvwFollow() override { --state_->preview_followers; }

    // One wake per call: a preview change, a layout change, a stop, the pending change's apply
    // time or one aux tick, whichever comes first. The processing lock is held throughout, so
    // every wait is bounded. A change whose apply time has passed is only still pending after a
    // failed apply; it is retried at the tick, never at once, so a dead compositor cannot spin.
    // A wake that applies nothing and has no change waiting refreshes `suspended_` instead,
    // unless a take is in flight: a timed change is published while its take holds the mixer's
    // `mutex` for the routing, and reading the status then would wait for it (see the header).
    void process() override {
        int64_t now = avp::mixer::monotonicNs();
        int64_t wake_by = now + timing_.aux.time(1);
        if (pending_ && pending_->target && timing_.applyAt(pending_->target) > now)
            wake_by = std::min(wake_by, timing_.applyAt(pending_->target));
        bool dirty;
        std::optional<Change> fresh;
        {
            std::unique_lock<std::mutex> lock(state_->preview_mutex);
            state_->preview_changed.wait_until(lock, std::chrono::steady_clock::time_point(std::chrono::nanoseconds(wake_by)),
                [&] { return stopping_ || base_dirty_ || !seen_ || state_->preview.revision != *seen_; });
            if (stopping_) return;
            dirty = base_dirty_ || dirty_retry_;
            base_dirty_ = dirty_retry_ = false;
            if (!seen_ || state_->preview.revision != *seen_) {
                fresh = Change{state_->preview};
                // The first read takes the state as it is; the change it came from is history.
                if (!seen_) fresh->effective_ns = 0;
                seen_ = state_->preview.revision;
            }
        }
        try {
            if (fresh) {
                fresh->target = fresh->effective_ns ? timing_.targetTick(fresh->effective_ns) : 0;
                pending_ = fresh;   // the latest change wins over one still waiting for its tick
            }
            now = avp::mixer::monotonicNs();
            if (pending_ && (!pending_->target || now >= timing_.applyAt(pending_->target))) {
                apply(*pending_, false, !suspended_);
                const int64_t applied_at = avp::mixer::monotonicNs();
                applied_ = *pending_;
                pending_.reset();
                settle_at_ = applied_.target ? timing_.deadline(applied_.target) : applied_at;
                Parameters status = sample(applied_, applied_at);
                if (applied_.target) {
                    std::lock_guard<std::mutex> lock(state_->preview_mutex);
                    state_->preview_follow_samples[name_] = status;
                }
                std::lock_guard<std::mutex> lock(layouts_mutex_);
                status_ = std::move(status);
            } else if (dirty) {
                // A new base (a tile reassignment; the PVW layouts are only set at creation) keeps
                // the warm inputs; a settle still due stays due.
                apply(applied_, false, std::nullopt);
            } else if (settle_at_ && now >= *settle_at_) {
                apply(applied_, true, !suspended_);
                settle_at_.reset();
            } else if (!pending_ && state_->transition_mode == avp::mixer::MixerState::TransitionMode::Idle) {
                suspended_ = compositorSuspended();
            }
            report(warning_);
        } catch (const std::exception& e) {
            // The pending change and the dirty layouts stay, retried at the next wake: this covers
            // "Node not created" while the bus group is still starting.
            dirty_retry_ = dirty;
            report(e.what());
        }
    }

    void stop() override {
        stopping_ = true;
        state_->preview_changed.notify_all();
    }

    void setObject(const std::string key, const Parameters& value) override {
        if (key == "base") {
            Layout layout = parseLayout(value);
            std::lock_guard<std::mutex> lock(layouts_mutex_);
            base_ = std::move(layout);
            base_revision_ = value.value("revision", std::string());
        } else if (key == "pvw") {
            std::unordered_map<std::string, Layout> layouts;
            for (auto it = value.begin(); it != value.end(); ++it) layouts[it.key()] = parseLayout(it.value());
            std::lock_guard<std::mutex> lock(layouts_mutex_);
            pvw_ = std::move(layouts);
        } else {
            throw Error("mixer_pvw_follow: unknown object " + key);
        }
        std::lock_guard<std::mutex> lock(state_->preview_mutex);
        base_dirty_ = true;
        state_->preview_changed.notify_all();
    }

    Parameters getObject(const std::string key) override {
        if (key != "status") throw Error("mixer_pvw_follow: unknown object " + key);
        std::lock_guard<std::mutex> lock(layouts_mutex_);
        Parameters result = status_;
        result["base_revision"] = base_revision_;
        result["error"] = error_;
        return result;
    }

    static std::shared_ptr<MixerPvwFollow> create(NodeCreationInfo& nci) {
        const Parameters& params = nci.params;
        auto state = InstanceSharedObjects<avp::mixer::MixerState>::get(nci.instance, params.at("mixer").get<std::string>());
        av::Rational main_rate;
        {
            std::lock_guard<std::mutex> lock(state->mutex);
            main_rate = av::Rational(state->fps_num, state->fps_den);
        }
        const auto aux_rate = parseRatio(params.at("fps").get<std::string>());
        const auto ns = [](double ms) { return static_cast<int64_t>(ms * 1000000); };
        const int64_t latency_ns = ns(params.at("latency_ms").get<double>());
        // The main mixer's playout latency: when a program frame leaves its compositor. Without
        // it the bus is assumed to run with the program's.
        const int64_t main_latency_ns = ns(params.value("main_latency_ms", params.at("latency_ms").get<double>()));
        const int64_t delay = params.value("pgm_delay_frames", int64_t(0));
        if (latency_ns < 0 || main_latency_ns < 0 || delay < 0)
            throw Error("mixer_pvw_follow: latency_ms, main_latency_ms and pgm_delay_frames must be nonnegative");
        const auto align_name = params.value("align", std::string("program"));
        if (align_name != "program" && align_name != "pgm_tile")
            throw Error("mixer_pvw_follow: align must be program or pgm_tile");
        const auto align = align_name == "program" ? avp::mixer::PreviewAlign::Program : avp::mixer::PreviewAlign::PgmTile;
        const auto compositor_name = params.at("compositor").get<std::string>();
        auto node = std::make_shared<MixerPvwFollow>(
            std::move(state), params.value("name", compositor_name + "_pvw"), compositor_name, nci.nodes.node(compositor_name),
            avp::mixer::PreviewFollowTiming{avp::mixer::TickGrid(main_rate), avp::mixer::TickGrid(aux_rate), latency_ns, delay,
                                            main_latency_ns, align});
        for (const char* key : {"pvw", "base"})
            if (params.contains(key)) node->setObject(key, params.at(key));
        return node;
    }
};

DECLNODE(mixer_pvw_follow, MixerPvwFollow)
