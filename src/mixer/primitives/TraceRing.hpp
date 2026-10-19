#pragma once

#include "../../util.hpp"
#include <deque>

namespace avp::mixer {

/// The newest 256 diagnostic records of a node, returned by its "trace" object. The node
/// serializes access with its own mutex.
class TraceRing {
    std::deque<Parameters> records_;

public:
    void push(Parameters record) {
        records_.push_back(std::move(record));
        if (records_.size() > 256) records_.pop_front();
    }
    void clear() { records_.clear(); }
    Parameters json() const { return Parameters(records_); }
};

} // namespace avp::mixer
