// Picture-mapped layer zoom: the geometric glide between two framings of one camera, read from
// the sheet's per-frame layers, and how it rewrites the zooming scene layer per tick.
#include "mixer/primitives/source_picture.hpp"
#include <cassert>
#include <cmath>

void print_stack_trace() {}

using namespace avp::mixer;

static bool near(double a, double b, double eps = 1e-9) { return std::fabs(a - b) <= eps; }
static bool same(const SourceRect &a, const SourceRect &b, double eps = 1e-9) {
    return near(a.x, b.x, eps) && near(a.y, b.y, eps) && near(a.w, b.w, eps) && near(a.h, b.h, eps);
}

// A sheet whose raw tile (the whole 1920x1080 picture, 1:1) is at (0, 0); policy tile "ms" at
// (1920, 0) shows camera rect (480, 120, 960, 540), "cu" at (3840, 0) shows (700, 200, 520, 292);
// a debug tile (input 3) is not mapped.
static const char *kSheet = R"({"layers":[
 {"input":0,"crop":{"x":0,"y":0,"w":1920,"h":1080},"dst_x":0,"dst_y":0,"dst_w":1920,"dst_h":1080,"filter":"bilinear"},
 {"input":1,"crop":{"x":480,"y":120,"w":960,"h":540},"dst_x":1920,"dst_y":0,"dst_w":1920,"dst_h":1080,"filter":"bicubic","bicubic_param":0.5},
 {"input":2,"crop":{"x":700,"y":200,"w":520,"h":292},"dst_x":3840,"dst_y":0,"dst_w":1920,"dst_h":1080,"filter":"bicubic","bicubic_param":0.5},
 {"input":3,"crop":{"x":0,"y":0,"w":1920,"h":1080},"dst_x":0,"dst_y":1080,"dst_w":1920,"dst_h":1080}]})";

static SourcePicture picture() { return {{0, 0, 1920, 1080}, "face_normalize_layers_v1", {0, 1, 2}}; }

static ZoomSpec spec(int64_t duration_ns = 1'000'000'000) {
    ZoomSpec z;
    z.input = 0;
    z.picture = picture();
    z.from_crop = {1920, 0, 1920, 1080};   // the ms tile, whole
    z.from_dst = {0, 0, 1920, 1080};
    z.to_crop = {3840, 0, 1920, 1080};     // the cu tile, whole
    z.to_dst = {0, 0, 1920, 1080};
    z.duration_ns = duration_ns;
    z.curve = FadeCurve::EaseInOut;
    z.generation = 7;
    return z;
}

static av::VideoFrame sheet(const char *json = kSheet) {
    av::VideoFrame frame(AV_PIX_FMT_NV12, 5760, 2160);
    if (json) av_dict_set(&frame.raw()->metadata, "face_normalize_layers_v1", json, 0);
    return frame;
}

static std::vector<LayerSpec> sceneLayer() {   // the target scene's own layer: the cu tile, 1:1 onto the canvas
    LayerSpec l;
    l.input = 0;
    l.crop_x = 3840; l.crop_w = 1920; l.crop_h = 1080;
    l.dst_w = 1920; l.dst_h = 1080;
    l.fit = true;
    return {l};
}

static void geometry() {
    const SourceRect A{100, 50, 1600, 900}, B{400, 200, 1000, 562};
    // endpoints_are_exact
    assert(same(zoomRect(A, B, 0), A) && same(zoomRect(A, B, 1), B));
    // fixed_point_stays_put: P's screen position (P - X) / W is constant
    const double r = B.w / A.w, P = (B.x - r * A.x) / (1 - r);
    for (double s : {0., .25, .5, .75, 1.}) {
        const auto V = zoomRect(A, B, s);
        assert(near((P - V.x) / V.w, (P - A.x) / A.w, 1e-12));
    }
    // scale_is_geometric
    assert(near(zoomRect(A, B, .5).w, std::sqrt(A.w * B.w)));
    // equal_sizes_pan_linearly
    assert(near(zoomRect(A, {300, 50, 1600, 900}, .5).x, 200));
    // edges_stay_between_endpoints
    for (int k = 0; k <= 100; ++k) {
        const auto V = zoomRect(A, B, k / 100.);
        assert(V.x >= A.x - 1e-9 && V.x <= B.x + 1e-9 && V.x + V.w <= A.x + A.w + 1e-9 && V.x + V.w >= B.x + B.w - 1e-9);
    }
}

static void picture_rect_maps_tile_crops() {
    const auto maps = parsePictureMaps(kSheet, picture());
    assert(maps.size() == 3);   // unmapped debug left out
    auto [whole, map] = *pictureRect(maps, {1920, 0, 1920, 1080});
    assert(same(whole, {480, 120, 960, 540}) && map->filter.mode == ScaleFilter::Bicubic);
    assert(same(pictureRect(maps, {1920, 0, 960, 1080})->first, {480, 120, 480, 540}));
    // A clipped tile (letterboxed by the normaliser): the whole tile reaches above the picture.
    const auto clipped = parsePictureMaps(R"({"layers":[{"input":1,"crop":{"x":0,"y":0,"w":1700,"h":868},
        "dst_x":3840,"dst_y":100,"dst_w":1920,"dst_h":980}]})", picture());
    const auto rect = pictureRect(clipped, {3840, 0, 1920, 1080})->first;
    assert(near(rect.y, -100 * 868. / 980) && near(rect.h, 1080 * 868. / 980));
    // unmapped_layers_are_never_matched: a crop over the debug tile only
    assert(!pictureRect(maps, {0, 1080, 1920, 1080}));
}

static void progress_waits_for_on_air_and_eases() {
    auto z = spec(1'500'000'000);
    assert(zoomProgress(z, 5'000'000'000) == 0);
    z.start_ns = 1'000'000'000;
    assert(zoomProgress(z, 900'000'000) == 0);
    assert(near(zoomProgress(z, 1'750'000'000), .5));
    assert(zoomProgress(z, 1'000'000'000 + 1'500'000'000 / 45) < .002);
    assert(zoomProgress(z, 2'500'000'000) == 1);
}

static void applying() {
    const auto frame = sheet();
    auto z = spec();
    z.start_ns = 0;
    ZoomFrameState state;
    std::string why;
    // zero_draws_the_from_tile_verbatim: the ms tile's camera rect from the raw tile, bicubic .5, stretch
    auto layers = sceneLayer();
    assert(applyLayerZoom(layers, z, &frame, 0, state, &why));
    const LayerSpec &l = layers[0];
    assert(l.crop_x == 480 && l.crop_y == 120 && l.crop_w == 960 && l.crop_h == 540 && !l.crop_exact);
    assert(l.dst_x == 0 && l.dst_y == 0 && l.dst_w == 1920 && l.dst_h == 1080);
    assert(l.filter.mode == ScaleFilter::Bicubic && l.filter.bicubic_param == .5f && !l.fit && !l.source_canvas_w);
    // crop_is_exact_and_marks_the_transform: an interior frame is sub-pixel
    layers = sceneLayer();
    assert(applyLayerZoom(layers, z, &frame, 333'000'000, state, &why));
    assert(layers[0].crop_exact && needsTransform(layers[0]));
    const auto V = zoomRect({480, 120, 960, 540}, {700, 200, 520, 292}, zoomProgress(z, 333'000'000));
    assert(same(*layers[0].crop_exact, V, 1e-6));
    // one_leaves_the_scene_layer
    layers = sceneLayer();
    assert(!applyLayerZoom(layers, z, &frame, 1'000'000'000, state, &why));
    assert(layers[0] == sceneLayer()[0]);
}

static void interior_clips_to_the_reference() {
    // The from framing reaches above the picture (a letterboxed tile): the crop stays in the raw tile
    // and the destination shrinks by the same share.
    const auto frame = sheet(R"({"layers":[{"input":1,"crop":{"x":100,"y":0,"w":1700,"h":868},
        "dst_x":1920,"dst_y":100,"dst_w":1920,"dst_h":980,"filter":"bicubic","bicubic_param":0.5},
        {"input":2,"crop":{"x":700,"y":200,"w":520,"h":292},"dst_x":3840,"dst_y":0,"dst_w":1920,"dst_h":1080,
        "filter":"bicubic","bicubic_param":0.5}]})");
    auto z = spec();
    z.start_ns = 0;
    ZoomFrameState state;
    std::string why;
    auto layers = sceneLayer();
    assert(applyLayerZoom(layers, z, &frame, 0, state, &why));
    const LayerSpec &l = layers[0];
    assert(l.crop_y >= 0 && l.dst_y > 0 && l.dst_y + l.dst_h <= 1080);
    assert(l.crop_x >= 0 && l.crop_x + l.crop_w <= 1920);
}

static void missing_and_nested() {
    auto z = spec();
    z.start_ns = 0;
    ZoomFrameState state;
    std::string why;
    // no_metadata_on_the_first_tick_draws_the_scene_layer
    const auto bare = sheet(nullptr);
    auto layers = sceneLayer();
    assert(!applyLayerZoom(layers, z, &bare, 100'000'000, state, &why) && !why.empty());
    assert(layers[0] == sceneLayer()[0]);
    // missing_metadata_holds_the_last_rects
    const auto frame = sheet();
    why.clear();
    layers = sceneLayer();
    assert(applyLayerZoom(layers, z, &frame, 400'000'000, state, &why));
    auto held = sceneLayer();
    assert(applyLayerZoom(held, z, &bare, 400'000'000, state, &why) && state.held == 1);
    assert(held[0] == layers[0]);
    // nested_from_follows_its_own_schedule: the from endpoint is the first glide where it is now
    auto first = std::make_shared<ZoomSpec>(spec(2'000'000'000));
    first->start_ns = 0;
    ZoomSpec back = spec();
    back.from_zoom = first;
    back.to_crop = {1920, 0, 1920, 1080};   // back to ms
    back.start_ns = 1'000'000'000;
    const auto maps = parsePictureMaps(kSheet, picture());
    const auto mid = zoomPose(*first, maps, 1'000'000'000);
    assert(mid && same(mid->picture, zoomRect({480, 120, 960, 540}, {700, 200, 520, 292}, .5)));
    assert(same(zoomPose(back, maps, 1'000'000'000)->picture, mid->picture));
    const auto ended = zoomPose(*first, maps, 9'000'000'000);   // past its end: its to-tile, never "leave"
    assert(ended && same(ended->picture, {700, 200, 520, 292}));
    // parse/serialise round trip, with the nesting
    const ZoomSpec again = parseZoomSpec(zoomSpecJson(back));
    assert(again.from_zoom && again.from_zoom->start_ns == 0 && again.start_ns == back.start_ns);
    assert(again.generation == 7 && again.curve == FadeCurve::EaseInOut && again.picture.mapped.size() == 3);
}

static void plan_refuses_with_a_reason() {
    const SourcePicture pic = picture();
    const auto layer = Parameters::parse(R"({"crop":{"x":1920,"y":0,"w":1920,"h":1080},"dst_x":0,"dst_y":0,"dst_w":1920,"dst_h":1080})");
    std::string why;
    assert(planZoom(layer, layer, 0, 0, &pic, why));
    assert(!planZoom(layer, layer, 0, 1, &pic, why) && why.find("source") != std::string::npos);
    assert(!planZoom(layer, layer, 0, 0, nullptr, why) && why.find("picture") != std::string::npos);
    const auto bare = Parameters::parse(R"({"dst_x":0,"dst_y":0,"dst_w":1920,"dst_h":1080})");
    assert(!planZoom(bare, layer, 0, 0, &pic, why) && why.find("tile") != std::string::npos);
}

// hook_state_machine: a slot's zoom draws its from-framing until the take is on air, then glides,
// and is dropped at its end or by a composition without one.
static void hook_state_machine() {
    const auto frame = sheet();
    const std::vector<const av::VideoFrame *> sources{&frame};
    SlotZoom slot;
    slot.set(spec());
    auto layers = sceneLayer();
    assert(slot.apply(layers, sources, 5'000'000'000) && slot.progress(5'000'000'000) == 0);   // not on air yet
    assert(layers[0].crop_x == 480 && !layers[0].crop_exact);
    slot.onAir(6, 6'000'000'000, 0);                    // another generation: ignored
    assert(slot.progress(6'500'000'000) == 0);
    slot.onAir(7, 6'000'000'000, 0);
    layers = sceneLayer();
    assert(slot.apply(layers, sources, 6'400'000'000) && slot.progress(6'400'000'000) > 0 && layers[0].crop_exact);
    layers = sceneLayer();
    assert(!slot.apply(layers, sources, 7'000'000'000) && !slot.spec && layers[0] == sceneLayer()[0]);   // over: dropped
    slot.set(spec());
    slot.set(std::nullopt);                          // a composition without a zoom
    layers = sceneLayer();
    assert(!slot.apply(layers, sources, 0) && layers[0] == sceneLayer()[0]);
}

// The slot draws ahead of where the take completes: an on-air pts already in the slot's past
// starts the glide at the slot's next tick, never part-way through.
static void late_on_air_starts_at_the_slots_tick() {
    const auto frame = sheet();
    const std::vector<const av::VideoFrame *> sources{&frame};
    SlotZoom slot;
    slot.set(spec(1'500'000'000));
    slot.onAir(7, 1'000'000'000, 4'000'000'000);     // completed at 1 s; the slot is at 4 s
    assert(*slot.spec->start_ns == 4'000'000'000);
    auto layers = sceneLayer();
    assert(slot.apply(layers, sources, 4'500'000'000) && slot.progress(4'500'000'000) > 0 && slot.progress(4'500'000'000) < 1);
    SlotZoom early;
    early.set(spec(1'500'000'000));
    early.onAir(7, 5'000'000'000, 4'000'000'000);    // on-air pts still ahead of the slot: kept
    assert(*early.spec->start_ns == 5'000'000'000);
}


static void direct_source_picture() {
    av::VideoFrame frame(AV_PIX_FMT_NV12, 1920, 1080);
    av_dict_set(&frame.raw()->metadata, "face_normalize_layers_v1", kSheet, 0);
    av_dict_set(&frame.raw()->metadata, "mixer_source_picture_v1",
        R"({"reference":{"x":0,"y":0,"w":1920,"h":1080},"metadata_key":"face_normalize_layers_v1","mapped":[0,1,2]})", 0);
    auto layers = sceneLayer();
    layers[0].rotation_deg = 12;
    layers.push_back(layers[0]);
    layers[1].crop_x = 1920;
    layers[1].dst_x = 960; layers[1].dst_w = 960;
    applySourcePictures(layers, {&frame});
    assert(layers[0].crop_x == 700 && layers[0].crop_y == 200 && layers[0].crop_w == 520);
    assert(layers[0].rotation_deg == 12 && layers[0].dst_w == 1920);
    assert(layers[1].crop_x == 480 && layers[1].crop_w == 960 && layers[1].dst_x == 960 && layers[1].dst_w == 960);
    auto skip = sceneLayer();
    applySourcePictures(skip, {&frame}, 0);
    assert(skip[0] == sceneLayer()[0]);
    av::VideoFrame bare(AV_PIX_FMT_NV12, 1920, 1080);
    applySourcePictures(skip, {&bare});
    assert(skip[0] == sceneLayer()[0]);
    auto half = sceneLayer(); half[0].crop_w = 960;
    applySourcePictures(half, {&frame});
    assert(half[0].crop_x == 700 && half[0].crop_w == 260);
}

int main() {
    direct_source_picture();
    late_on_air_starts_at_the_slots_tick();
    hook_state_machine();
    geometry();
    picture_rect_maps_tile_crops();
    progress_waits_for_on_air_and_eases();
    applying();
    interior_clips_to_the_reference();
    missing_and_nested();
    plan_refuses_with_a_reason();
}
