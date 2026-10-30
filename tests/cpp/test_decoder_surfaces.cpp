#include "decoder_surfaces.hpp"
#include <cassert>
#include <chrono>
#include <thread>

void print_stack_trace() {}

using namespace std::chrono_literals;

static AVFrame *decodedFrame() {
    AVFrame *frame = av_frame_alloc();
    frame->format = AV_PIX_FMT_NV12;
    frame->width = 16;
    frame->height = 16;
    assert(av_frame_get_buffer(frame, 0) == 0);
    return frame;
}

int main() {
    avp::DecoderSurfaceGate gate(2);

    // A frame is held until its last reference is gone, however many consumers share it.
    AVFrame *first = decodedFrame();
    assert(gate.track(first) && gate.held() == 1);
    AVFrame *shared = av_frame_clone(first);
    av_frame_free(&first);
    assert(gate.held() == 1);
    av_frame_free(&shared);
    assert(gate.held() == 0);

    // Below the limit the next packet goes at once; at it, the decoder waits for a release.
    AVFrame *a = decodedFrame(), *b = decodedFrame();
    assert(gate.track(a) && gate.wait(0ms));
    assert(gate.track(b) && gate.held() == 2);
    assert(!gate.wait(20ms));
    std::thread consumer([&] {
        std::this_thread::sleep_for(30ms);
        av_frame_free(&a);
    });
    assert(gate.wait(2s) && gate.held() == 1);
    consumer.join();

    // A frame may outlive its decoder's gate.
    {
        avp::DecoderSurfaceGate gone(1);
        assert(gone.track(b) && gone.held() == 1);
    }
    av_frame_free(&b);

    // No free buffer slot: nothing is counted.
    AVFrame *full = decodedFrame();
    for (int slot = 0; slot < AV_NUM_DATA_POINTERS; ++slot)
        if (!full->buf[slot]) full->buf[slot] = av_buffer_alloc(1);
    assert(!gate.track(full) && gate.held() == 0);
    av_frame_free(&full);

    // Pools against the budget: a pool past it is refused and not recorded; a decoder that
    // reopens replaces its pool; 0 is no budget.
    avp::DecoderPoolLedger ledger;
    bool accepted = false;
    assert(ledger.reserve(&gate, 100, 250, accepted) == 100 && accepted);
    const int second = 0, third = 0;   // owners are identities
    assert(ledger.reserve(&second, 100, 250, accepted) == 200 && accepted);
    assert(ledger.reserve(&third, 100, 250, accepted) == 300 && !accepted);
    assert(ledger.reserve(&gate, 150, 250, accepted) == 250 && accepted);
    ledger.release(&second);
    assert(ledger.reserve(&third, 100, 250, accepted) == 250 && accepted);
    assert(ledger.reserve(&second, 1000, 0, accepted) == 1250 && accepted);
    return 0;
}
