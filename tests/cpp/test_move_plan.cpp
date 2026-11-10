// Planning a move take: the destination slot draws B plus A's sources B lacks, with their routes; a
// routed source both scenes route differently falls back to a fade with the reason.
#include "mixer/move_plan.hpp"
#include "mixer/primitives/layer_move.hpp"
#include <cassert>
#include <stdexcept>

using namespace avp::mixer;

static SceneDefinition scene(const std::string &name, std::initializer_list<std::pair<const char *, int>> sources) {
    SceneDefinition s;
    s.name = name;
    for (const auto &[source, x] : sources) s.sources[source] = Parameters{{"dst_x", x}, {"dst_w", 640}, {"dst_h", 360}};
    return s;
}

static void fill(MixerState &st) {
    for (const auto &[name, input] : std::initializer_list<std::pair<const char *, int>>{{"cam1", 0}, {"cam2", 1}, {"cam3", 2}, {"gfx", 5}}) {
        MixerState::SourceInfo info;
        info.otm_node_name = std::string("otm_") + name;
        info.input_index = input;
        st.sources[name] = info;
    }
    st.sources["cam3"].routed = true;
    st.sources["cam3"].router_node_name = "router";
}

static void union_adds_a_only_sources_with_their_routes() {
    MixerState st;
    fill(st);
    auto a = scene("a", {{"cam1", 0}, {"cam3", 100}});
    a.routes["cam3"] = 4;
    const auto b = scene("b", {{"cam1", 960}, {"cam2", 0}});
    std::string why;
    const auto u = moveUnionScene(a, b, why);
    assert(u && why.empty());
    assert(u->sources.size() == 3 && u->sources.at("cam1") == b.sources.at("cam1"));   // B's layer for a shared source
    assert(u->sources.at("cam3") == a.sources.at("cam3") && u->routes.at("cam3") == 4);
    const SourceMask mask = st.computeActiveInputsMask(*u);
    assert(mask.test(0) && mask.test(1) && mask.test(2) && !mask.test(5));
}

static void spec_carries_a_layers_by_input() {
    MixerState st;
    fill(st);
    const auto a = scene("a", {{"cam2", 0}, {"gfx", 0}}), b = scene("b", {{"cam2", 960}});
    std::string why;
    const Parameters spec = planMove(st, a, b, 7, "cut", why);
    assert(!spec.is_null() && why.empty());
    assert(spec.at("generation") == 7 && spec.at("unmatched") == "cut");
    const auto &from = spec.at("from");
    assert(from.size() == 2 && from[0].at("input") == 1 && from[1].at("input") == 5 && from[0].at("dst_x") == 0);
}

static void routed_change_falls_back_to_fade() {
    MixerState st;
    fill(st);
    auto a = scene("a", {{"cam3", 0}}), b = scene("b", {{"cam3", 960}});
    a.routes["cam3"] = 1;
    b.routes["cam3"] = 2;
    std::string why;
    assert(planMove(st, a, b, 3, "fade", why).is_null() && why.find("cam3") != std::string::npos);
    b.routes["cam3"] = 1;   // the same route glides
    why.clear();
    assert(!planMove(st, a, b, 3, "fade", why).is_null() && why.empty());
}

static void unmatched_is_fade_or_cut() {
    MixerState st;
    fill(st);
    const auto a = scene("a", {{"cam1", 0}}), b = scene("b", {{"cam1", 960}});
    std::string why;
    bool threw = false;
    try { planMove(st, a, b, 1, "slide", why); } catch (const std::exception &) { threw = true; }
    assert(threw);
}

static void spec_carries_the_sheets_pictures() {
    MixerState st;
    fill(st);
    st.sources["cam1"].picture = Parameters{{"reference", {{"x", 1920}, {"y", 0}, {"w", 1920}, {"h", 1080}}},
                                            {"metadata_key", "face_normalize_layers_v1"}, {"mapped", {0, 1, 2}}};
    const auto a = scene("a", {{"cam1", 0}, {"gfx", 0}}), b = scene("b", {{"cam1", 960}, {"cam2", 0}});
    std::string why;
    const Parameters spec = planMove(st, a, b, 4, "fade", why);
    const auto m = parseMoveSpec(spec);
    assert(m.pictures.size() == 1 && m.pictures.count(0));            // cam1 is input 0; the others declare none
    assert(m.pictures.at(0).reference.x == 1920 && m.pictures.at(0).metadata_key == "face_normalize_layers_v1");
    assert(parseMoveSpec(moveSpecJson(m)).pictures.at(0).mapped == std::vector<int>({0, 1, 2}));   // round trip
}

int main() {
    union_adds_a_only_sources_with_their_routes();
    spec_carries_a_layers_by_input();
    routed_change_falls_back_to_fade();
    unmatched_is_fade_or_cut();
    spec_carries_the_sheets_pictures();
    return 0;
}
