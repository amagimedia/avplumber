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

    // With a return queue an acknowledgement waits for the next frame: one goes per arrival,
    // the oldest goes at once beyond the queue's length, and retiring sends what still waits.
    int held_pair[2];
    assert(socketpair(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0, held_pair) == 0);
    auto paced = std::make_shared<DmabufReleaseAckQueue>(held_pair[0], 2);
    paced->enqueue(1);
    paced->enqueue(2);
    assert(paced->flush());
    assert(paced->stats().released == 2 && paced->stats().queued == 2 && paced->stats().sent == 0);
    assert(recv(held_pair[1], &byte, 1, MSG_DONTWAIT) == -1 && errno == EAGAIN);
    paced->frameArrived();
    assert(paced->flush());
    expectAck(held_pair[1], DmabufAckKind::Frame, 1);
    assert(paced->stats().queued == 1);
    paced->frameArrived();
    paced->frameArrived();   // nothing queued any more: no acknowledgement is invented
    assert(paced->flush());
    expectAck(held_pair[1], DmabufAckKind::Frame, 2);
    assert(recv(held_pair[1], &byte, 1, MSG_DONTWAIT) == -1 && errno == EAGAIN);
    paced->enqueue(3);
    paced->enqueue(4);
    paced->enqueue(5);       // a third beyond the queue of two: the oldest goes at once
    assert(paced->flush());
    expectAck(held_pair[1], DmabufAckKind::Frame, 3);
    assert(paced->stats().queued == 2);
    paced->retire();
    expectAck(held_pair[1], DmabufAckKind::Frame, 4);
    expectAck(held_pair[1], DmabufAckKind::Frame, 5);
    expectAck(held_pair[1], DmabufAckKind::Drain);
    paced.reset();
    expectAck(held_pair[1], DmabufAckKind::Drained);
    close(held_pair[1]);
}
