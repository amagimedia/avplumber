#pragma once
// A move carried by the frame it belongs to. Source-paced graphs compute linear
// progress from their media timestamps; easing and geometry are the mixer's own.
#include "layer_move.hpp"

namespace avp::mixer {

inline std::vector<LayerSpec> frameMoveLayers(const Parameters &request,
        const std::vector<const av::VideoFrame *> &sources, AVPixelFormat format) {
    const double progress = request.at("progress").get<double>();
    if (!std::isfinite(progress)) throw Error("frame move: progress must be finite");
    const auto curve = parseFadeCurve(request.value("curve", std::string("linear")));
    const auto unmatched = request.value("unmatched", std::string("fade"));
    if (unmatched != "fade" && unmatched != "cut")
        throw Error("frame move: unmatched must be fade or cut");
    return moveLayers(parseLayersArray(request.at("from")), parseLayersArray(request.at("to")),
                      fadeCurveAt(curve, progress), sources, format, unmatched);
}

} // namespace avp::mixer
