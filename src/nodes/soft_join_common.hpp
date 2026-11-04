#pragma once

#include "node_common.hpp"

// Shared hold-last auxiliaries for soft_join_metadata_frame / soft_join_video_frame.
// Aux edges are always MetadataFrame; primary/out type is node-specific.

namespace soft_join {

struct AuxState {
    bool eof = false;
    bool has_held = false;
    Parameters held = Parameters::object();
    av::Timestamp held_pts;
    std::string into; // empty => flat-merge held keys into primary
};

inline double ptsSeconds(const av::Timestamp& ts) {
    return ts.seconds();
}

inline bool isUsableMetadata(const MetadataFrame& data) {
    return data.isComplete() && data.pts().isValid();
}

inline std::vector<std::string> parseIntoKeys(const Parameters& params, size_t aux_count,
                                              const char* node_name) {
    std::vector<std::string> into_keys(aux_count);
    if (!params.contains("into")) return into_keys;

    const Parameters& into = params["into"];
    if (into.is_string()) {
        if (aux_count != 1) {
            throw Error(std::string(node_name) +
                        ": string 'into' is only valid with exactly one auxiliary");
        }
        into_keys[0] = into.get<std::string>();
    } else if (into.is_array()) {
        if (into.size() != aux_count) {
            throw Error(std::string(node_name) +
                        ": 'into' length must equal number of auxiliaries");
        }
        for (size_t i = 0; i < aux_count; ++i) {
            if (into[i].is_null()) {
                into_keys[i].clear();
            } else if (into[i].is_string()) {
                into_keys[i] = into[i].get<std::string>();
            } else {
                throw Error(std::string(node_name) +
                            ": each 'into' entry must be a string or null");
            }
        }
    } else if (!into.is_null()) {
        throw Error(std::string(node_name) + ": 'into' must be a string, array, or null");
    }
    return into_keys;
}

inline double parseMaxAge(const Parameters& params, const char* node_name) {
    if (!params.contains("max_age") || params["max_age"].is_null()) return -1.0;
    const double max_age = params["max_age"].get<double>();
    if (max_age < 0.0) {
        throw Error(std::string(node_name) + ": max_age must be >= 0 (omit to disable)");
    }
    return max_age;
}

inline bool allAuxEof(const std::vector<AuxState>& aux) {
    for (const AuxState& a : aux) {
        if (!a.eof) return false;
    }
    return true;
}

inline void drainAuxForPrimary(const std::vector<std::shared_ptr<Edge<MetadataFrame>>>& aux_edges,
                               std::vector<AuxState>& aux,
                               const av::Timestamp& primary_pts,
                               double max_age_sec) {
    const double primary_sec = ptsSeconds(primary_pts);
    for (size_t i = 0; i < aux.size(); ++i) {
        AuxState& state = aux[i];
        while (!state.eof) {
            MetadataFrame* frame = aux_edges[i]->peek();
            if (frame == nullptr) break;
            if (isEofMarker(*frame)) {
                aux_edges[i]->pop();
                state.eof = true;
                break;
            }
            if (!isUsableMetadata(*frame)) {
                aux_edges[i]->pop();
                continue;
            }
            if (ptsSeconds(frame->pts()) > primary_sec) break;
            state.held = frame->metadata();
            state.held_pts = frame->pts();
            state.has_held = true;
            aux_edges[i]->pop();
        }
        if (state.has_held && max_age_sec >= 0.0) {
            const double age = primary_sec - ptsSeconds(state.held_pts);
            if (age > max_age_sec) {
                state.has_held = false;
                state.held = Parameters::object();
                state.held_pts = av::Timestamp();
            }
        }
    }
}

inline void mergeHeldIntoParameters(Parameters& md, const std::vector<AuxState>& aux) {
    if (!md.is_object()) md = Parameters::object();
    for (const AuxState& state : aux) {
        if (!state.has_held) continue;
        if (state.into.empty()) {
            if (!state.held.is_object()) continue;
            for (auto it = state.held.begin(); it != state.held.end(); ++it) {
                md[it.key()] = it.value();
            }
        } else {
            md[state.into] = state.held;
        }
    }
}

inline void mergeHeldIntoVideoFrame(av::VideoFrame& frame, const std::vector<AuxState>& aux) {
    AVFrame* raw = frame.raw();
    if (!raw) return;
    for (const AuxState& state : aux) {
        if (!state.has_held) continue;
        if (!state.into.empty()) {
            av_dict_set(&raw->metadata, state.into.c_str(), state.held.dump().c_str(), 0);
            continue;
        }
        if (!state.held.is_object()) continue;
        for (auto it = state.held.begin(); it != state.held.end(); ++it) {
            const std::string val = it.value().is_string() ? it.value().get<std::string>()
                                                          : it.value().dump();
            av_dict_set(&raw->metadata, it.key().c_str(), val.c_str(), 0);
        }
    }
}

} // namespace soft_join
