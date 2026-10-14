#include "one_to_many.hpp"

// On an event loop (`event_loop`), run when the input signals a frame: no thread to wake
// for every frame. It must not wait for room in an output, so `drop` is required: a frame
// an output has no room for is dropped for that output.
template <typename T>
class OneToManyNonBlocking : public OneToManyBase<T>, public NonBlockingNode<OneToManyNonBlocking<T>> {
    // Frames handed out in one go before other work on the loop gets its turn.
    static constexpr int kFramesPerTurn = 8;

public:
    using OneToManyBase<T>::OneToManyBase;

    virtual void processNonBlocking(EventLoop&, bool ticks) override {
        for (int handed = 0; handed < kFramesPerTurn; ++handed) {
            T* data = this->source_->peek(0);
            if (!data) {
                if (!ticks) this->processWhenSignalled(this->edgeSource()->edge()->producedEvent());
                return;
            }
            this->handOut(data);
            this->source_->pop();
        }
        if (!ticks) this->yieldAndProcess();
    }

    static std::shared_ptr<OneToManyNonBlocking> create(NodeCreationInfo& nci) {
        auto r = OneToManyBase<T>::template build<OneToManyNonBlocking>(nci);
        if (!r->drop_) throw Error("one_to_many_nb: drop must be true, an event loop cannot wait for room in an output");
        return r;
    }
};

DECLNODE_ATD(one_to_many_nb, OneToManyNonBlocking);
