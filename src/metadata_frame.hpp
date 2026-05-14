#pragma once

#include <utility>
#include <avcpp/rational.h>
#include <avcpp/timestamp.h>
#include "util.hpp"

class MetadataFrame {
private:
    av::Timestamp pts_;
    Parameters metadata_ = Parameters::object();

public:
    MetadataFrame() = default;
    MetadataFrame(av::Timestamp pts, Parameters metadata = Parameters::object())
        : pts_(pts), metadata_(std::move(metadata)) {}

    av::Timestamp pts() const { return pts_; }
    void setPts(av::Timestamp pts) { pts_ = pts; }
    av::Rational timeBase() const { return pts_.timebase(); }
    void setTimeBase(av::Rational timebase) {
        pts_ = av::Timestamp(pts_.timestamp(), timebase);
    }

    Parameters& metadata() { return metadata_; }
    const Parameters& metadata() const { return metadata_; }
    void setMetadata(Parameters metadata) { metadata_ = std::move(metadata); }

    bool isComplete() const { return pts_.isValid(); }
};
