#pragma once
#include <map>
#include <string>

// Optional, read-only producer telemetry, keyed by destination queue name.
// Subscription demand is independent of observed traffic or queue occupancy.
class IOutputSubscriptions {
public:
    virtual std::map<std::string, bool> outputSubscriptions() const = 0;
    virtual ~IOutputSubscriptions() = default;
};
