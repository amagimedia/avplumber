#include "node_common.hpp"
#include "../mixer/primitives/MixerState.hpp"
#include "../mixer/primitives/MonotonicClock.hpp"
#include "../mixer/primitives/PreviewFollow.hpp"
#include "../mixer/primitives/source_mask.hpp"
#include <atomic>
#include <chrono>
#include <optional>
#include <unordered_map>

// One per AUX bus, the only writer of the bus compositor's `composition`: draws the mixer's preview
// changes (MixerState::publishPreview) on the bus frame that leaves it with the take's first
// program frame (`align` "program", the default) or whose last (program) input shows that frame
// ("pgm_tile"), see PreviewFollow.hpp. Each timed change's latency from the take command's receipt
// (`pvw_latency_ms`, next to the program's `pgm_latency_ms`, `pvw_minus_pgm_ms`, `kind`) is in this
// node's status and in `mixer.status` `pvw_latency`.
// The control side sets the `layout` object: `pvw`, every scene's preview layers (empty for a bus
// that draws no preview), `base`, the rest, and their `revision`. Each composition is pvw[shown]
// followed by base, carrying that revision; what the layers draw is the control side's business.
//
// The scene leaving program becomes the preview (the swap), so a settle pass after every take
// keeps the program scene's sources active (no layer draws them) and the next take's timed
// composition only drops inputs, which the compositor applies at once instead of staging. An
// input of a settle the compositor rejects (a stalled program source) stays in applied_inputs_
// without being active, so the next timed apply adds it and is staged too.
//
// The compositor is resolved once, at creation: a lookup by name takes NodeManager's lock, which
// shutdown holds while it joins this thread. A change goes from the feed's lock
// (MixerState::preview_mutex) to the compositor without taking the mixer's `mutex`, which the
// orchestrator holds while it routes a take.
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
    std::string revision_;
    std::unordered_map<std::string, Layout> pvw_;
    Parameters status_ = Parameters::object();
    std::string error_;
    // Wake reasons besides the mixer's revision. The layout is flagged under the feed's lock; the
    // stop is not, since the framework requests it under locks the orchestrator takes after
    // state_->mutex: a wake lost that way lasts one aux tick.
    bool layout_dirty_ = true;
    std::atomic<bool> stopping_{false};
    // Follower thread only.
    std::optional<uint64_t> seen_;
    std::optional<Change> pending_;
    Change applied_;
    avp::mixer::SourceMask applied_inputs_;   // the active inputs the compositor last accepted
    std::optional<int64_t> settle_at_;   // when to add the program scene's inputs, for the swap
    bool dirty_retry_ = false;           // a layout apply failed: again at the next wake
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
    /// A scene without a PVW layout draws an empty cell and is reported (`warning_`), not retried.
    void apply(const Change& change, bool settle) {
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
            composition = {{"layers", std::move(layers)}, {"active_inputs", avp::mixer::toParameters(inputs)},
                           {"revision", revision_}};
        }
        compositor()->setObject("composition", composition);
        applied_inputs_ = inputs;
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
        const auto ms = [](int64_t ns) { return static_cast<double>(ns) / 1e6; };
        Parameters status = {{"pvw_scene", change.pvw}, {"pgm_scene", change.pgm}, {"kind", change.kind},
                             {"applied_revision", change.revision}, {"target_tick", change.target},
                             {"align", timing_.align == avp::mixer::PreviewAlign::Program ? "program" : "pgm_tile"},
                             {"last_change_to_apply_ms", ms(applied_at - change.published_ns)},
                             {"last_target_error_ticks", 0}, {"target_unreachable", false},
                             {"pvw_latency_ms", nullptr}, {"pgm_latency_ms", nullptr}, {"pvw_minus_pgm_ms", nullptr}};
        if (!change.target) return status;
        const int64_t drawn = timing_.tickDrawnAfter(applied_at);
        const int64_t pvw = timing_.deadline(drawn), pgm = timing_.programDeparture(change.effective_ns);
        status["last_target_error_ticks"] = drawn - change.target;
        status["target_unreachable"] = timing_.deadline(change.target) <= change.published_ns;
        status["pvw_minus_pgm_ms"] = ms(pvw - pgm);
        if (change.received_ns) {
            status["pvw_latency_ms"] = ms(pvw - change.received_ns);
            status["pgm_latency_ms"] = ms(pgm - change.received_ns);
        }
        return status;
    }

public:
    MixerPvwFollow(std::shared_ptr<avp::mixer::MixerState> state, std::string name, std::string compositor_name,
                   const std::shared_ptr<NodeWrapper>& compositor, avp::mixer::PreviewFollowTiming timing)
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
                [&] { return stopping_ || layout_dirty_ || !seen_ || state_->preview.revision != *seen_; });
            if (stopping_) return;
            dirty = layout_dirty_ || dirty_retry_;
            layout_dirty_ = dirty_retry_ = false;
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
                const Change change = *pending_;   // stays pending if apply() throws: retried next wake
                apply(change, false);
                const int64_t applied_at = avp::mixer::monotonicNs();
                applied_ = change;
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
                // A new layout keeps the warm inputs; a settle still due stays due.
                apply(applied_, false);
            } else if (settle_at_ && now >= *settle_at_) {
                apply(applied_, true);
                settle_at_.reset();
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

    /// `layout`: {"revision", "pvw": {scene: {"layers", "active_inputs"}}, "base": {...}},
    /// replaced as a whole, so no composition mixes two layouts.
    void setObject(const std::string key, const Parameters& value) override {
        if (key != "layout") throw Error("mixer_pvw_follow: unknown object " + key);
        Layout base = parseLayout(value.at("base"));
        std::unordered_map<std::string, Layout> pvw;
        const Parameters& scenes = value.at("pvw");
        for (auto it = scenes.begin(); it != scenes.end(); ++it) pvw[it.key()] = parseLayout(it.value());
        const auto revision = value.value("revision", std::string());
        {
            std::lock_guard<std::mutex> lock(layouts_mutex_);
            base_ = std::move(base);
            pvw_ = std::move(pvw);
            revision_ = revision;
        }
        std::lock_guard<std::mutex> lock(state_->preview_mutex);
        layout_dirty_ = true;
        state_->preview_changed.notify_all();
    }

    Parameters getObject(const std::string key) override {
        if (key != "status") throw Error("mixer_pvw_follow: unknown object " + key);
        std::lock_guard<std::mutex> lock(layouts_mutex_);
        Parameters result = status_;
        result["layout_revision"] = revision_;
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
        if (params.contains("layout")) node->setObject("layout", params.at("layout"));
        return node;
    }
};

DECLNODE(mixer_pvw_follow, MixerPvwFollow)
