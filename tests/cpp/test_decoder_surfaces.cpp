#include "decoder_surfaces.hpp"
#include <cassert>
#include <chrono>
#include <memory>
#include <thread>

extern "C" {
#include <libavcodec/codec_id.h>
}

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

static void gate() {
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

    // A stream that left less headroom lowers the limit: one held frame is then the most.
    gate.setLimit(1);
    assert(gate.limit() == 1 && !gate.wait(0ms));
    av_frame_free(&b);
    assert(gate.wait(0ms) && gate.held() == 0);
    gate.setLimit(2);

    // A frame may outlive its decoder's gate.
    AVFrame *c = decodedFrame();
    {
        avp::DecoderSurfaceGate gone(1);
        assert(gone.track(c) && gone.held() == 1);
    }
    av_frame_free(&c);

    // No free buffer slot: nothing is counted.
    AVFrame *full = decodedFrame();
    for (int slot = 0; slot < AV_NUM_DATA_POINTERS; ++slot)
        if (!full->buf[slot]) full->buf[slot] = av_buffer_alloc(1);
    assert(!gate.track(full) && gate.held() == 0);
    av_frame_free(&full);
}

static void exhaustion() {
    using avp::SurfaceExhaustion;
    SurfaceExhaustion policy(100ms, 1000ms);
    const auto t0 = SurfaceExhaustion::Clock::now();
    assert(!policy.stuck() && !policy.hiding());

    // The first failure starts an episode: the packet is kept and sent again.
    assert(policy.failed(t0) == SurfaceExhaustion::Next::Retry);
    assert(policy.stuck() && policy.since() == t0 && policy.counters().exhaustions == 1);
    assert(policy.retryAfter() == 100ms);
    policy.retrying();
    assert(policy.failed(t0 + 500ms) == SurfaceExhaustion::Next::Retry);
    assert(policy.counters().retries == 1 && policy.counters().packets_dropped == 0 && !policy.hiding());

    // Still failing when the give-up time is reached: the packet is dropped and, the picture
    // being lost, what follows is hidden until a keyframe. The episode goes on, and the next
    // packet gets a wait of its own.
    assert(policy.failed(t0 + 1000ms) == SurfaceExhaustion::Next::Drop);
    assert(policy.stuck() && policy.hiding() && policy.counters().packets_dropped == 1);
    assert(policy.failed(t0 + 1500ms) == SurfaceExhaustion::Next::Retry);
    assert(policy.failed(t0 + 2500ms) == SurfaceExhaustion::Next::Drop && policy.counters().packets_dropped == 2);
    assert(!policy.showPicture(false) && policy.counters().pictures_hidden == 1);
    assert(!policy.showPicture(false) && policy.counters().pictures_hidden == 2);
    assert(policy.showPicture(true) && !policy.hiding());
    assert(policy.showPicture(false) && policy.counters().pictures_hidden == 2);

    // A picture shown ends the episode, which counts from its first failure, not from the last
    // drop: 2600 ms at 40 ms a picture is 65 pictures repeated. Shown while never stuck changes nothing.
    assert(policy.counters().exhaustions == 1 && policy.stuck());
    assert(policy.since() == t0 + 2500ms && policy.episodeSince() == t0);
    assert(policy.episodePictures(t0 + 2600ms, 40ms) == 65);
    assert(policy.shown(t0 + 2600ms, 40ms) == 65);
    assert(!policy.stuck() && policy.counters().recoveries == 1 && policy.counters().repeats == 65);
    assert(policy.episodePictures(t0 + 2700ms, 40ms) == 0);
    assert(policy.shown(t0 + 2700ms, 40ms) == 0);
    assert(policy.counters().recoveries == 1 && policy.counters().repeats == 65);

    // A decoder that lost the picture on its own (H.264): the episode begins, the picture is
    // counted lost and nothing is shown until a keyframe; a second loss is the same episode.
    policy.lost(t0 + 3000ms);
    assert(policy.stuck() && policy.hiding() && policy.since() == t0 + 3000ms);
    assert(policy.counters().exhaustions == 2 && policy.counters().pictures_lost == 1);
    policy.lost(t0 + 3100ms);
    assert(policy.counters().exhaustions == 2 && policy.counters().pictures_lost == 2 && policy.since() == t0 + 3000ms);
    assert(!policy.showPicture(false) && policy.stuck());
    assert(policy.showPicture(true));
    assert(policy.shown(t0 + 3100ms, 40ms) == 2);
    assert(!policy.stuck() && policy.counters().recoveries == 2 && policy.counters().repeats == 67);
}

static void ledger() {
    // Pools against the budget: a pool past it is refused and not recorded; a decoder that
    // reopens replaces its pool; 0 is no budget.
    avp::DecoderPoolLedger ledger;
    bool accepted = false;
    const int first = 0, second = 0, third = 0;   // owners are identities
    assert(ledger.reserve(&first, 100, 250, accepted) == 100 && accepted);
    assert(ledger.reserve(&second, 100, 250, accepted) == 200 && accepted);
    assert(ledger.reserve(&third, 100, 250, accepted) == 300 && !accepted);
    assert(ledger.total() == 200);
    assert(ledger.reserve(&first, 150, 250, accepted) == 250 && accepted);
    ledger.release(&second);
    assert(ledger.reserve(&third, 100, 250, accepted) == 250 && accepted);
    assert(ledger.reserve(&second, 1000, 0, accepted) == 1250 && accepted);

    // The show's totals: episodes so far and decoders without surfaces or pictures now.
    ledger.exhausted(true);
    ledger.exhausted(true);
    assert(ledger.exhaustion() == std::make_pair(uint64_t(2), 2));
    ledger.exhausted(false);
    assert(ledger.exhaustion() == std::make_pair(uint64_t(2), 1));
    ledger.degraded(true);
    assert(ledger.degradedNow() == 1);
    ledger.degraded(false);
    ledger.degraded(false);
    assert(ledger.degradedNow() == 0);
    // Pictures repeated only add up: an operator counter, never a gauge.
    assert(ledger.repeats() == 0);
    ledger.repeated(65);
    ledger.repeated(0);
    ledger.repeated(2);
    assert(ledger.repeats() == 67);
}

static void parking() {
    auto ledger = std::make_shared<avp::DecoderPoolLedger>();
    avp::DecoderParking lot;

    avp::DecoderParking::Geometry geometry;
    geometry.codec = AV_CODEC_ID_HEVC;
    geometry.width = geometry.height = 16;
    geometry.sw_format = AV_PIX_FMT_NV12;
    geometry.surfaces = 21;

    // The lot takes a reference of its own; the caller's frame stays the caller's. The pool is
    // in the ledger under the entry until it is taken or freed.
    AVFrame *one = decodedFrame();
    assert(lot.park(one, geometry, 100, ledger, "one", 2) == 0);
    assert(av_buffer_get_ref_count(one->buf[0]) == 2);
    av_frame_free(&one);
    assert(ledger->total() == 100 && lot.status().parked == 1);
    AVFrame *two = decodedFrame();
    assert(lot.park(two, geometry, 100, ledger, "two", 2) == 0);
    av_frame_free(&two);

    // Past the limit the oldest is freed at once.
    AVFrame *three = decodedFrame();
    assert(lot.park(three, geometry, 100, ledger, "three", 2) == 1);
    av_frame_free(&three);
    auto status = lot.status();
    assert(status.parked == 2 && status.bytes == 200 && status.parked_total == 3 && status.freed == 1);
    assert(ledger->total() == 200);

    // A live pool that does not fit beside the parked ones frees them, oldest first, as needed.
    const int live = 0;
    bool accepted = false;
    ledger->reserve(&live, 150, 300, accepted);
    assert(!accepted);
    assert(lot.makeRoom(*ledger, 150, 300) == 1);
    assert(lot.status().parked == 1 && ledger->total() == 100);
    ledger->reserve(&live, 150, 300, accepted);
    assert(accepted && ledger->total() == 250);
    assert(lot.makeRoom(*ledger, 0, 0) == 0);   // no budget: nothing to make room for

    // One at a time, when asked; nothing left is not an error.
    assert(lot.freeOne());
    assert(lot.status().parked == 0 && lot.status().freed == 3 && ledger->total() == 150);
    assert(!lot.freeOne());

    // A new source takes over the smallest decoder that fits, out of the ledger.
    avp::DecoderParking::Geometry large = geometry;
    large.surfaces = 24;
    AVFrame *four = decodedFrame();
    assert(lot.park(four, large, 140, ledger, "four", 2) == 0);
    av_frame_free(&four);
    AVFrame *fits = decodedFrame();
    assert(lot.park(fits, geometry, 100, ledger, "fits", 2) == 0);
    av_frame_free(&fits);
    int64_t bytes = 0;
    AVFrame *taken = lot.take(geometry, bytes);
    assert(taken && bytes == 100 && lot.status().parked == 1 && lot.status().reused == 1);
    assert(ledger->total() == 290);
    av_frame_free(&taken);
    taken = lot.take(geometry, bytes);   // a smaller stream can use a larger pool
    assert(taken && bytes == 140 && lot.status().parked == 0 && ledger->total() == 150);
    av_frame_free(&taken);
    assert(!lot.take(geometry, bytes) && lot.status().reused == 2);

    // Anything else is a new decoder: another size, format or codec, or a stream that needs more.
    AVFrame *five = decodedFrame();
    assert(lot.park(five, geometry, 100, ledger, "five", 2) == 0);
    av_frame_free(&five);
    auto other = geometry;
    other.width = 32;
    assert(!lot.take(other, bytes));
    other = geometry;
    other.sw_format = AV_PIX_FMT_P010;
    assert(!lot.take(other, bytes));
    other = geometry;
    other.codec = AV_CODEC_ID_H264;
    assert(!lot.take(other, bytes));
    other = geometry;
    other.surfaces = 22;
    assert(!lot.take(other, bytes) && lot.status().parked == 1);

    // A show that shrinks parks what it ran, past the limit, and frees nothing: the device held
    // all of it at the peak. What a show adds on top of its peak counts against the limit.
    {
        auto pools = std::make_shared<avp::DecoderPoolLedger>();
        avp::DecoderParking shrinking;
        for (int i = 0; i < 5; ++i) shrinking.opened();
        for (int i = 0; i < 5; ++i) {
            shrinking.closed();
            AVFrame *frame = decodedFrame();
            assert(shrinking.park(frame, geometry, 100, pools, "gone", 2) == 0);
            av_frame_free(&frame);
        }
        assert(shrinking.status().parked == 5 && shrinking.status().freed == 0);
        for (int i = 0; i < 5; ++i) shrinking.opened();   // five more of another format
        AVFrame *frame = decodedFrame();
        shrinking.closed();
        // 4 live and 6 parked against a peak of 5 plus 2: three go.
        assert(shrinking.park(frame, geometry, 100, pools, "replaced", 2) == 3);
        av_frame_free(&frame);
        assert(shrinking.status().parked == 3);
    }

    // No limit parks nothing; the lot frees what it holds when it goes.
    AVFrame *six = decodedFrame();
    assert(lot.park(six, geometry, 100, ledger, "six", 0) == 0 && lot.status().parked == 1);
    av_frame_free(&six);
    {
        avp::DecoderParking gone;
        AVFrame *seven = decodedFrame();
        assert(gone.park(seven, geometry, 100, ledger, "seven", 4) == 0);
        av_frame_free(&seven);
        assert(ledger->total() == 350);
    }
    assert(ledger->total() == 250);
}

int main() {
    gate();
    exhaustion();
    ledger();
    parking();
    return 0;
}
