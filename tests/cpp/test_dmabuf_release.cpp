#include "nodes/hwaccel/DmabufReleaseQueue.hpp"
#include <cassert>
#include <memory>

static void expectAck(int fd, DmabufAckKind expected, uint64_t frame = 0) {
    std::array<uint8_t, DMABUF_RELEASE_ACK_BYTES> bytes{};
    assert(recv(fd, bytes.data(), bytes.size(), MSG_WAITALL) == (ssize_t)bytes.size());
    uint64_t got;
    DmabufAckKind kind;
    assert(dmabufDecodeReleaseAck(bytes.data(), got, kind));
    assert(kind == expected && got == frame);
}

int main() {
    int old_pair[2], new_pair[2];
    assert(socketpair(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0, old_pair) == 0);
    assert(socketpair(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0, new_pair) == 0);
    auto receiver = std::make_shared<DmabufReleaseAckQueue>(old_pair[0]);
    auto frame = receiver;
    receiver->enqueue(11);
    assert(receiver->stats().released == 1 && receiver->stats().pending == 1);
    assert(receiver->flush());
    assert(receiver->stats().sent == 1 && receiver->stats().pending == 0);
    expectAck(old_pair[1], DmabufAckKind::Frame, 11);

    receiver->interrupt();
    receiver->retire();
    expectAck(old_pair[1], DmabufAckKind::Drain);
    receiver = std::make_shared<DmabufReleaseAckQueue>(new_pair[0]);
    char byte;
    assert(recv(old_pair[1], &byte, 1, MSG_DONTWAIT) == -1 && errno == EAGAIN);
    // A frame from the retired connection still acknowledges that connection.
    frame->enqueue(12);
    expectAck(old_pair[1], DmabufAckKind::Frame, 12);
    assert(recv(new_pair[1], &byte, 1, MSG_DONTWAIT) == -1 && errno == EAGAIN);
    frame.reset();
    expectAck(old_pair[1], DmabufAckKind::Drained);
    assert(recv(old_pair[1], &byte, 1, 0) == 0);
    receiver.reset();
    expectAck(new_pair[1], DmabufAckKind::Drain);
    expectAck(new_pair[1], DmabufAckKind::Drained);
    assert(recv(new_pair[1], &byte, 1, 0) == 0);
    close(old_pair[1]);
    close(new_pair[1]);

    // Without a return queue
    // every release is acknowledged at once, before any new frame arrives.
    int zero_pair[2];
    assert(socketpair(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0, zero_pair) == 0);
    auto immediate = std::make_shared<DmabufReleaseAckQueue>(zero_pair[0], 0);
    immediate->enqueue(21);
    immediate->enqueue(22);
    assert(immediate->stats().queued == 0 && immediate->stats().pending == 2);
    assert(immediate->flush());
    expectAck(zero_pair[1], DmabufAckKind::Frame, 21);
    expectAck(zero_pair[1], DmabufAckKind::Frame, 22);
    immediate->frameArrived();
    assert(!immediate->hasPending());
    immediate.reset();
    expectAck(zero_pair[1], DmabufAckKind::Drain);
    expectAck(zero_pair[1], DmabufAckKind::Drained);
    assert(recv(zero_pair[1], &byte, 1, 0) == 0);
    close(zero_pair[1]);

    // With a hold of two the sender keeps two frames out here: a frame let go waits until a
    // third is out, one goes back per arrival, a stall's frames let go together go back one
    // per arrival instead of in a bunch, and retiring sends what still waits.
    int held_pair[2];
    assert(socketpair(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0, held_pair) == 0);
    auto paced = std::make_shared<DmabufReleaseAckQueue>(held_pair[0], 2);
    auto silent = [&] { assert(paced->flush()); assert(recv(held_pair[1], &byte, 1, MSG_DONTWAIT) == -1 && errno == EAGAIN); };
    paced->frameArrived();
    paced->frameArrived();
    paced->enqueue(1);
    paced->enqueue(2);
    silent();                // two out, both let go: they stay with the consumer
    assert(paced->stats().released == 2 && paced->stats().queued == 2 && paced->stats().outstanding == 2);
    paced->frameArrived();   // frame 3: one back
    assert(paced->flush());
    expectAck(held_pair[1], DmabufAckKind::Frame, 1);
    assert(paced->stats().queued == 1 && paced->stats().outstanding == 2);
    paced->frameArrived();   // frame 4: one back, and the consumer now holds 3 and 4
    assert(paced->flush());
    expectAck(held_pair[1], DmabufAckKind::Frame, 2);
    paced->frameArrived();   // frame 5 while the consumer stalls: nothing to give back
    silent();
    assert(paced->stats().queued == 0 && paced->stats().outstanding == 3);
    paced->enqueue(3);       // the stall ends: the first frame let go brings the hold back to two
    assert(paced->flush());
    expectAck(held_pair[1], DmabufAckKind::Frame, 3);
    paced->enqueue(4);
    paced->enqueue(5);       // the rest of the catch-up waits instead of going back together
    silent();
    assert(paced->stats().queued == 2 && paced->stats().outstanding == 2);
    paced->frameArrived();   // frame 6: one back
    assert(paced->flush());
    expectAck(held_pair[1], DmabufAckKind::Frame, 4);
    silent();
    paced->retire();
    expectAck(held_pair[1], DmabufAckKind::Frame, 5);
    expectAck(held_pair[1], DmabufAckKind::Drain);
    assert(paced->stats().outstanding == 0);
    paced.reset();
    expectAck(held_pair[1], DmabufAckKind::Drained);
    close(held_pair[1]);
}
