#pragma once
#include "instance_shared.hpp"
#include "Event.hpp"

class NamedEvent: public InstanceShared<NamedEvent> {
protected:
    Event event_;
public:
    NamedEvent() {
    }
    Event& event() {
        return event_;
    }
};
