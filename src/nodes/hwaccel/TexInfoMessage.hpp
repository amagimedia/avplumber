#pragma once

// One frame's message on a dma-browser socket: the 48-byte TexInfo (texinfo.h, shared with the
// sender) and the fds its SCM_RIGHTS carries, the DMA-BUF plane and, when the header flags one,
// the acquire fence. Nothing here depends on avplumber: tests/cpp/test_dmabuf_fence.cpp drives
// it over a socketpair.

#include "../../../deps/dma-browser/addons/fdpass/texinfo.h"
#include <cstring>
#include <initializer_list>
#include <sys/socket.h>
#include <unistd.h>

struct TexInfoMessage {
    TexInfo info{};
    int plane_fd = -1;
    int fence_fd = -1;   // -1: none, the pixels are complete on arrival
    void closeFds() {
        for (int *fd : {&plane_fd, &fence_fd}) {
            if (*fd >= 0) ::close(*fd);
            *fd = -1;
        }
    }
};

enum class TexInfoRecvStatus {
    Ok,
    Closed,            // EOF or a socket error: nothing was read
    Short,             // fewer bytes than a header: nothing in it can be trusted
    NoFd,              // a header without a plane fd
    FdCountMismatch,   // the header's fence flag and the number of fds disagree
    Truncated,         // more fds than a message may carry (MSG_CTRUNC)
};

inline const char *texInfoRecvStatusText(TexInfoRecvStatus status) {
    switch (status) {
        case TexInfoRecvStatus::Ok: return "ok";
        case TexInfoRecvStatus::Closed: return "connection closed";
        case TexInfoRecvStatus::Short: return "metadata too small";
        case TexInfoRecvStatus::NoFd: return "message without a DMA-BUF fd";
        case TexInfoRecvStatus::FdCountMismatch: return "fd count does not match the header's fence flag";
        case TexInfoRecvStatus::Truncated: return "more fds than a frame message carries";
    }
    return "unknown";
}

// Reads one message without blocking (call once poll says the socket is readable). On any
// status but Ok every fd the message carried has been closed, and `out.info` is filled whenever
// the whole header arrived, so the frame it names can still be acknowledged. On Ok the caller
// owns `plane_fd` and, when present, `fence_fd`.
inline TexInfoRecvStatus recvTexInfoMessage(int sock, TexInfoMessage &out) {
    out = TexInfoMessage{};
    char control[CMSG_SPACE(2 * sizeof(int))];
    iovec iov{&out.info, sizeof(out.info)};
    msghdr msg{};
    msg.msg_iov = &iov;
    msg.msg_iovlen = 1;
    msg.msg_control = control;
    msg.msg_controllen = sizeof(control);
    const ssize_t received = recvmsg(sock, &msg, MSG_DONTWAIT | MSG_CMSG_CLOEXEC);
    if (received <= 0) return TexInfoRecvStatus::Closed;
    int fds[2];
    int count = 0;
    bool extra = (msg.msg_flags & MSG_CTRUNC) != 0;
    for (cmsghdr *c = CMSG_FIRSTHDR(&msg); c; c = CMSG_NXTHDR(&msg, c)) {
        if (c->cmsg_level != SOL_SOCKET || c->cmsg_type != SCM_RIGHTS) continue;
        const size_t carried = (c->cmsg_len - CMSG_LEN(0)) / sizeof(int);
        for (size_t i = 0; i < carried; ++i) {
            int fd;
            std::memcpy(&fd, CMSG_DATA(c) + i * sizeof(int), sizeof(fd));
            if (count < 2) fds[count++] = fd;
            else { ::close(fd); extra = true; }
        }
    }
    const auto fail = [&](TexInfoRecvStatus status) {
        for (int i = 0; i < count; ++i) ::close(fds[i]);
        return status;
    };
    if (size_t(received) < sizeof(out.info)) return fail(TexInfoRecvStatus::Short);
    if (extra) return fail(TexInfoRecvStatus::Truncated);
    if (count == 0) return fail(TexInfoRecvStatus::NoFd);
    if (count != (texinfoHasAcquireFence(out.info.pixel_format) ? 2 : 1)) return fail(TexInfoRecvStatus::FdCountMismatch);
    out.plane_fd = fds[0];
    if (count == 2) out.fence_fd = fds[1];
    return TexInfoRecvStatus::Ok;
}
