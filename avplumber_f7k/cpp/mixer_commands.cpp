// mixer.init / source / scene / cut / status / scenes on the Rust graph.
// Fade, wipe, preview and prewarm stay unregistered.
#include "rust_graph.hpp"
#include "mixer/orchestrator/MixerOrchestrator.hpp"
#include "nodes/hwaccel/graphic_color.h"

#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <string>

extern "C" void avp_cpp_ensure_instance(void* rust_instance);
extern "C" InstanceData* avp_cpp_instance(void* rust_instance);
extern "C" int avp_cpp_link_mixer_compositor(void);
extern "C" int avp_cpp_link_mixer_selector(void);
extern "C" int avp_cpp_link_mixer_snapshot(void);

namespace {

std::string trim(std::string text) {
    while (!text.empty() && std::isspace(static_cast<unsigned char>(text.front()))) {
        text.erase(text.begin());
    }
    while (!text.empty() && std::isspace(static_cast<unsigned char>(text.back()))) {
        text.pop_back();
    }
    return text;
}

char* reply_of(const std::string& text) {
    char* buf = static_cast<char*>(std::malloc(text.size() + 1));
    if (buf) {
        std::memcpy(buf, text.c_str(), text.size() + 1);
    }
    return buf;
}

int finish(char** reply, const std::string& text) {
    if (reply) {
        *reply = reply_of(text);
    }
    return 0;
}

int fail(char** reply, const std::string& text) {
    if (reply) {
        *reply = reply_of(text);
    }
    return -1;
}

avp::mixer::MixerOrchestrator orchestrator(AvpCore* core, const std::string& name) {
    avp_cpp_ensure_instance(core);
    auto state = InstanceSharedObjects<avp::mixer::MixerState>::get(*avp_cpp_instance(core), name);
    auto graph = std::make_shared<avp::mixer::RustMixerGraph>(core);
    return avp::mixer::MixerOrchestrator(std::move(graph), std::move(state));
}

extern "C" int mixer_init(AvpCore* core, const char* args, char** reply) {
    try {
        avp_cpp_ensure_instance(core);
        std::stringstream ss(args ? args : "");
        std::string mixer_name, config_str;
        ss >> mixer_name;
        std::getline(ss, config_str);
        config_str = trim(config_str);
        if (mixer_name.empty() || config_str.empty()) {
            return fail(reply, "mixer.init: expected name and JSON");
        }
        json cfg = json::parse(config_str);
        auto state = InstanceSharedObjects<avp::mixer::MixerState>::get(*avp_cpp_instance(core), mixer_name);
        std::lock_guard<std::mutex> lock(state->mutex);
        if (cfg.contains("backend")) {
            state->transition_control = avp::mixer::transitionControl(cfg["backend"].get<std::string>());
        }
        if (cfg.contains("hwaccel")) state->hwaccel_name = cfg["hwaccel"].get<std::string>();
        if (cfg.contains("fps_num")) state->fps_num = cfg["fps_num"].get<int>();
        if (cfg.contains("fps_den")) state->fps_den = cfg["fps_den"].get<int>();
        if (cfg.contains("color")) {
            const auto transfer = avp::mixer::graphicTransfer(cfg["color"].get<std::string>());
            if (transfer == AVCOL_TRC_UNSPECIFIED) {
                throw Error("mixer.init: color must be sdr, hlg or pq");
            }
            state->canvas_transfer = transfer;
        }
        if (cfg.contains("source_switcher")) {
            state->source_switcher_name = cfg["source_switcher"].get<std::string>();
        }
        if (cfg.contains("keyframe_node")) state->keyframe_node_name = cfg["keyframe_node"].get<std::string>();
        if (cfg.contains("initial_pgm_scene")) state->pgm_scene_name = cfg["initial_pgm_scene"].get<std::string>();
        if (cfg.contains("initial_pvw_scene")) {
            state->publishPreview(cfg["initial_pvw_scene"].get<std::string>(), 0);
        }
        if (cfg.contains("initial_pgm_slot")) {
            std::string slot = cfg["initial_pgm_slot"].get<std::string>();
            if (slot == "A") state->pgm_is_slot_a = true;
            else if (slot == "B") state->pgm_is_slot_a = false;
            else throw Error("mixer.init: initial_pgm_slot must be 'A' or 'B'");
        }
        if (cfg.contains("slot_a")) {
            auto& sa = cfg["slot_a"];
            state->slot_a.compositor_name = sa.value("compositor", std::string(""));
            if (sa.contains("norm_ts")) throw Error("mixer.init: norm_ts was removed; slots use clocked compositors");
            state->slot_a.post_otm_name = sa.value("post_otm", std::string(""));
        }
        if (cfg.contains("slot_b")) {
            auto& sb = cfg["slot_b"];
            state->slot_b.compositor_name = sb.value("compositor", std::string(""));
            if (sb.contains("norm_ts")) throw Error("mixer.init: norm_ts was removed; slots use clocked compositors");
            state->slot_b.post_otm_name = sb.value("post_otm", std::string(""));
        }
        if (cfg.contains("wipe_otm")) state->wipe_otm_name = cfg["wipe_otm"].get<std::string>();
        if (cfg.contains("wipe_base_fps")) state->wipe_base_fps_name = cfg["wipe_base_fps"].get<std::string>();
        if (cfg.contains("wipe_selector")) state->wipe_selector_name = cfg["wipe_selector"].get<std::string>();
        if (cfg.contains("wipe_group")) state->wipe_group_name = cfg["wipe_group"].get<std::string>();
        if (cfg.contains("wipe_input_node")) state->wipe_input_node_name = cfg["wipe_input_node"].get<std::string>();
        if (cfg.contains("wipe_cache_store")) state->wipe_cache_store = cfg["wipe_cache_store"].get<std::string>();
        if (cfg.contains("wipe_overlay")) state->wipe_overlay_name = cfg["wipe_overlay"].get<std::string>();
        if (cfg.contains("wipe_flush_edges")) {
            state->wipe_flush_edges.clear();
            for (const auto& edge : cfg["wipe_flush_edges"]) {
                state->wipe_flush_edges.push_back(edge.get<std::string>());
            }
        }
        return finish(reply, "");
    } catch (const std::exception& ex) {
        return fail(reply, ex.what());
    }
}

extern "C" int mixer_source(AvpCore* core, const char* args, char** reply) {
    try {
        std::stringstream ss(args ? args : "");
        std::string mixer_name, src_name, otm_node;
        int input_index = 0;
        if (!(ss >> mixer_name >> src_name >> otm_node >> input_index)) {
            return fail(reply, "mixer.source: expected mixer, source, fanout and input index");
        }
        std::string extra;
        if (ss >> extra) {
            return fail(reply, "mixer.source: per-slot filters were removed; filter the source upstream");
        }
        orchestrator(core, mixer_name).defineSource(src_name, otm_node, input_index);
        return finish(reply, "");
    } catch (const std::exception& ex) {
        return fail(reply, ex.what());
    }
}

extern "C" int mixer_scene(AvpCore* core, const char* args, char** reply) {
    try {
        std::stringstream ss(args ? args : "");
        std::string mixer_name, scene_name, def_str;
        ss >> mixer_name >> scene_name;
        std::getline(ss, def_str);
        def_str = trim(def_str);
        json jdef = json::parse(def_str);
        avp::mixer::SceneDefinition def;
        def.name = scene_name;
        if (jdef.contains("width")) def.width = jdef["width"].get<int>();
        if (jdef.contains("height")) def.height = jdef["height"].get<int>();
        if (!jdef.contains("sources") || !jdef["sources"].is_object()) {
            throw Error("mixer.scene: sources object required (logical source name -> { dst_x, dst_y, ... })");
        }
        for (auto it = jdef["sources"].begin(); it != jdef["sources"].end(); ++it) {
            const json& layer = it.value();
            if (!layer.is_object()) throw Error("mixer.scene: each source layer must be an object");
            if (layer.contains("graph")) {
                throw Error("mixer.scene: scene filter graphs were removed; filter the source upstream");
            }
            def.sources[it.key()] = layer;
        }
        if (jdef.contains("controls")) {
            if (!jdef["controls"].is_array()) throw Error("mixer.scene: controls must be an array");
            for (const auto& control : jdef["controls"]) {
                if (!control.is_object()) throw Error("mixer.scene: control entries must be objects");
                avp::mixer::SceneControl sc;
                sc.node_name = control.at("node").get<std::string>();
                sc.key = control.at("key").get<std::string>();
                sc.value = control.at("value");
                def.controls.push_back(std::move(sc));
            }
        }
        if (jdef.contains("routes")) {
            if (!jdef["routes"].is_object()) throw Error("mixer.scene: routes must be an object");
            for (auto it = jdef["routes"].begin(); it != jdef["routes"].end(); ++it) {
                def.routes[it.key()] = it.value().get<int>();
            }
        }
        orchestrator(core, mixer_name).defineScene(scene_name, def);
        return finish(reply, "");
    } catch (const std::exception& ex) {
        return fail(reply, ex.what());
    }
}

extern "C" int mixer_cut(AvpCore* core, const char* args, char** reply) {
    try {
        json req = json::parse(args ? args : "");
        if (!req.is_object()) throw Error("mixer.cut: expected JSON object");
        if (req.contains("start_pts_ms")) {
            throw Error("mixer: future-scheduled takes were removed; takes are immediate");
        }
        orchestrator(core, req.at("mixer").get<std::string>())
            .cut(req.at("scene").get<std::string>(), avp::mixer::CutLatency::Clock::now());
        return finish(reply, "");
    } catch (const std::exception& ex) {
        return fail(reply, ex.what());
    }
}

extern "C" int mixer_status(AvpCore* core, const char* args, char** reply) {
    try {
        std::string name = trim(args ? args : "");
        std::ostringstream out;
        out << orchestrator(core, name).status() << "\n";
        return finish(reply, out.str());
    } catch (const std::exception& ex) {
        return fail(reply, ex.what());
    }
}

extern "C" int mixer_scenes(AvpCore* core, const char* args, char** reply) {
    try {
        std::string name = trim(args ? args : "");
        json arr = orchestrator(core, name).sceneNames();
        std::ostringstream out;
        out << arr << "\n";
        return finish(reply, out.str());
    } catch (const std::exception& ex) {
        return fail(reply, ex.what());
    }
}

} // namespace

extern "C" int avp_cpp_link_mixer(void) {
    static int ready = 0;
    if (ready) {
        return 1;
    }
    // The node TUs are otherwise unreferenced and dropped under --gc-sections,
    // which drops their factory registrars with them.
    if (avp_cpp_link_mixer_compositor() != 1 || avp_cpp_link_mixer_selector() != 1
        || avp_cpp_link_mixer_snapshot() != 1) {
        return 0;
    }
    const struct {
        const char* name;
        AvpCommandFn fn;
    } commands[] = {
        {"mixer.init", mixer_init},
        {"mixer.source", mixer_source},
        {"mixer.scene", mixer_scene},
        {"mixer.cut", mixer_cut},
        {"mixer.status", mixer_status},
        {"mixer.scenes", mixer_scenes},
    };
    for (const auto& command : commands) {
        if (avp_register_command(command.name, command.fn) != 0) {
            return 0;
        }
    }
    ready = 1;
    return 1;
}
