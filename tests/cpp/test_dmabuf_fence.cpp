// The acquire fence of a DMA-BUF frame: the wait on its sync_file fd, driven with an eventfd in
// its place, and the receive of a TexInfo message whose fence travels as a second fd.
#include "nodes/hwaccel/DmabufFenceWait.hpp"
#include "nodes/hwaccel/TexInfoMessage.hpp"
#include <cassert>
#include <chrono>
#include <cstring>
#include <dirent.h>
#include <sys/eventfd.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>

static int openFds() {
    int count = 0;
    DIR *dir = opendir("/proc/self/fd");
    assert(dir);
    while (readdir(dir)) ++count;
    closedir(dir);
    return count - 3;   // ".", ".." and the directory stream itself
}

static void signalFd(int fd) {
    const uint64_t one = 1;
    assert(write(fd, &one, sizeof(one)) == sizeof(one));
}

static void testWait() {
    FenceWaitStats stats;
    const int signalled = eventfd(0, EFD_CLOEXEC | EFD_NONBLOCK);
    signalFd(signalled);
    FenceWaitResult r = waitForFence(signalled, 100);
    assert(r.outcome == FenceWaitOutcome::Signalled && !r.blocked);
    stats.record(r);
    close(signalled);

    const int late = eventfd(0, EFD_CLOEXEC | EFD_NONBLOCK);
    std::thread producer([late] { std::this_thread::sleep_for(std::chrono::milliseconds(30)); signalFd(late); });
    r = waitForFence(late, 1000);
    producer.join();
    assert(r.outcome == FenceWaitOutcome::Signalled && r.blocked);
    assert(r.waited_us >= 20000 && r.waited_us < 500000);
    stats.record(r);
    close(late);

    const int never = eventfd(0, EFD_CLOEXEC | EFD_NONBLOCK);
    r = waitForFence(never, 50);
    assert(r.outcome == FenceWaitOutcome::TimedOut && r.blocked && r.waited_us >= 50000);
    stats.record(r);

    r = waitForFence(-1, 50);
    assert(r.outcome == FenceWaitOutcome::Invalid);
    stats.record(r);
    const int closed = eventfd(0, EFD_CLOEXEC);
    close(closed);
    assert(waitForFence(closed, 50).outcome == FenceWaitOutcome::Invalid);

    // A wake-up ends the wait only when its owner says so; otherwise the wait goes on.
    const int wake = eventfd(0, EFD_CLOEXEC | EFD_NONBLOCK);
    int wakes = 0;
    auto drain = [&] { uint64_t v; while (read(wake, &v, sizeof(v)) == sizeof(v)) {} return ++wakes >= 2; };
    std::thread waker([wake] {
        std::this_thread::sleep_for(std::chrono::milliseconds(10)); signalFd(wake);
        std::this_thread::sleep_for(std::chrono::milliseconds(10)); signalFd(wake);
    });
    r = waitForFence(never, 1000, wake, drain);
    waker.join();
    assert(r.outcome == FenceWaitOutcome::Interrupted && wakes == 2 && r.waited_us < 500000);
    std::thread spurious([wake] { std::this_thread::sleep_for(std::chrono::milliseconds(5)); signalFd(wake); });
    r = waitForFence(never, 40, wake, [&] { uint64_t v; while (read(wake, &v, sizeof(v)) == sizeof(v)) {} return false; });
    spurious.join();
    assert(r.outcome == FenceWaitOutcome::TimedOut && r.waited_us >= 40000);
    close(never);
    close(wake);

    assert(stats.fenced.load() == 4 && stats.blocked.load() == 2 && stats.timeouts.load() == 1 && stats.invalid.load() == 1);
    assert(stats.wait_us.load() >= 70000 && stats.max_wait_us.load() >= 50000 && stats.max_wait_us.load() <= stats.wait_us.load());
    stats.unfenced();
    assert(stats.plain.load() == 1);
}

static void sendMessage(int sock, const void *bytes, size_t size, const int *fds, int count) {
    char control[CMSG_SPACE(3 * sizeof(int))];
    memset(control, 0, sizeof(control));
    struct iovec iov = {const_cast<void *>(bytes), size};
    struct msghdr msg{};
    msg.msg_iov = &iov;
    msg.msg_iovlen = 1;
    if (count) {
        msg.msg_control = control;
        msg.msg_controllen = CMSG_SPACE(count * sizeof(int));
        struct cmsghdr *cmsg = CMSG_FIRSTHDR(&msg);
        cmsg->cmsg_level = SOL_SOCKET;
        cmsg->cmsg_type = SCM_RIGHTS;
        cmsg->cmsg_len = CMSG_LEN(count * sizeof(int));
        memcpy(CMSG_DATA(cmsg), fds, count * sizeof(int));
    }
    assert(sendmsg(sock, &msg, 0) == (ssize_t)size);
}

static void testReceive() {
    int pair[2];
    assert(socketpair(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0, pair) == 0);
    int fds[3];
    for (int &fd : fds) fd = eventfd(0, EFD_CLOEXEC);
    TexInfo plain{};
    plain.width = 1280; plain.height = 720; plain.stride = 5120;
    plain.pixel_format = TEXINFO_PIX_FMT_BGRA; plain.frame_count = 7;
    TexInfo fenced = plain;
    fenced.pixel_format = TEXINFO_PIX_FMT_BGRA | TEXINFO_PIXEL_FORMAT_ACQUIRE_FENCE;
    assert(texinfoHasAcquireFence(fenced.pixel_format) && !texinfoHasAcquireFence(plain.pixel_format));
    assert(texinfoPixelFormat(fenced.pixel_format) == TEXINFO_PIX_FMT_BGRA);
    const int baseline = openFds();
    TexInfoMessage m;

    sendMessage(pair[0], &plain, sizeof(plain), fds, 1);
    assert(recvTexInfoMessage(pair[1], m) == TexInfoRecvStatus::Ok);
    assert(m.plane_fd >= 0 && m.fence_fd < 0 && m.info.frame_count == 7 && m.info.width == 1280);
    struct stat st;
    assert(fstat(m.plane_fd, &st) == 0);
    close(m.plane_fd);

    sendMessage(pair[0], &fenced, sizeof(fenced), fds, 2);
    assert(recvTexInfoMessage(pair[1], m) == TexInfoRecvStatus::Ok);
    assert(m.plane_fd >= 0 && m.fence_fd >= 0 && m.plane_fd != m.fence_fd);
    assert(texinfoPixelFormat(m.info.pixel_format) == TEXINFO_PIX_FMT_BGRA);
    close(m.plane_fd);
    close(m.fence_fd);
    assert(openFds() == baseline);

    // The header and the fd count disagree: nothing is kept, so nothing can be misread or leak.
    sendMessage(pair[0], &fenced, sizeof(fenced), fds, 1);
    assert(recvTexInfoMessage(pair[1], m) == TexInfoRecvStatus::FdCountMismatch);
    assert(m.plane_fd < 0 && m.fence_fd < 0 && m.info.frame_count == 7);   // the header is still trusted for the ack
    sendMessage(pair[0], &plain, sizeof(plain), fds, 2);
    assert(recvTexInfoMessage(pair[1], m) == TexInfoRecvStatus::FdCountMismatch);
    assert(m.plane_fd < 0 && m.fence_fd < 0);
    sendMessage(pair[0], &fenced, sizeof(fenced), fds, 3);
    assert(recvTexInfoMessage(pair[1], m) == TexInfoRecvStatus::Truncated);
    assert(m.plane_fd < 0 && m.fence_fd < 0);
    sendMessage(pair[0], &plain, sizeof(plain), fds, 0);
    assert(recvTexInfoMessage(pair[1], m) == TexInfoRecvStatus::NoFd);
    sendMessage(pair[0], &plain, 20, fds, 1);
    assert(recvTexInfoMessage(pair[1], m) == TexInfoRecvStatus::Short);
    assert(m.plane_fd < 0 && m.fence_fd < 0);
    assert(openFds() == baseline);

    close(pair[0]);
    assert(recvTexInfoMessage(pair[1], m) == TexInfoRecvStatus::Closed);
    close(pair[1]);
    for (int fd : fds) close(fd);
}

int main() {
    testWait();
    testReceive();
}
