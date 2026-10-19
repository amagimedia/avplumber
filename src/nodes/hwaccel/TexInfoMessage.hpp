#pragma once

// One frame's message on a dma-browser socket: the 48-byte TexInfo (texinfo.h, shared with the
// sender) and the DMA-BUF plane fd its SCM_RIGHTS carries. Nothing here depends on avplumber:
// tests/cpp/test_texinfo_message.cpp drives it over a socketpair.

#include "../../../deps/dma-browser/addons/fdpass/texinfo.h"
#include <cstring>
#include <sys/socket.h>
#include <unistd.h>

struct TexInfoMessage {
    TexInfo info{};
    int plane_fd = -1;
    void closeFd() {
        if (plane_fd >= 0) ::close(plane_fd);
        plane_fd = -1;
    }
};

enum class TexInfoRecvStatus {
    Ok,
    Closed,            // EOF or a socket error: nothing was read
    Short,             // fewer bytes than a header: nothing in it can be trusted
    NoFd,              // a header without a plane fd
    Truncated,         // more than the one fd a message carries
};

inline const char *texInfoRecvStatusText(TexInfoRecvStatus status) {
    switch (status) {
        case TexInfoRecvStatus::Ok: return "ok";
        case TexInfoRecvStatus::Closed: return "connection closed";
        case TexInfoRecvStatus::Short: return "metadata too small";
        case TexInfoRecvStatus::NoFd: return "message without a DMA-BUF fd";
        case TexInfoRecvStatus::Truncated: return "more fds than a frame message carries";
    }
    return "unknown";
}

// Reads one message without blocking (call once poll says the socket is readable). On any
// status but Ok every fd the message carried has been closed, and `out.info` is filled whenever
// the whole header arrived, so the frame it names can still be acknowledged. On Ok the caller
// owns `plane_fd`.
inline TexInfoRecvStatus recvTexInfoMessage(int sock, TexInfoMessage &out) {
    out = TexInfoMessage{};
    char control[CMSG_SPACE(sizeof(int))];
    iovec iov{&out.info, sizeof(out.info)};
    msghdr msg{};
    msg.msg_iov = &iov;
    msg.msg_iovlen = 1;
    msg.msg_control = control;
    msg.msg_controllen = sizeof(control);
    const ssize_t received = recvmsg(sock, &msg, MSG_DONTWAIT | MSG_CMSG_CLOEXEC);
    if (received <= 0) return TexInfoRecvStatus::Closed;
    int plane = -1;
    int count = 0;
    bool extra = (msg.msg_flags & MSG_CTRUNC) != 0;
    for (cmsghdr *c = CMSG_FIRSTHDR(&msg); c; c = CMSG_NXTHDR(&msg, c)) {
        if (c->cmsg_level != SOL_SOCKET || c->cmsg_type != SCM_RIGHTS) continue;
        const size_t carried = (c->cmsg_len - CMSG_LEN(0)) / sizeof(int);
        for (size_t i = 0; i < carried; ++i) {
            int fd;
            std::memcpy(&fd, CMSG_DATA(c) + i * sizeof(int), sizeof(fd));
            if (count++ == 0) plane = fd;
            else { ::close(fd); extra = true; }
        }
    }
    const auto fail = [&](TexInfoRecvStatus status) {
        if (plane >= 0) ::close(plane);
        return status;
    };
    if (size_t(received) < sizeof(out.info)) return fail(TexInfoRecvStatus::Short);
    if (extra) return fail(TexInfoRecvStatus::Truncated);
    if (count == 0) return fail(TexInfoRecvStatus::NoFd);
    out.plane_fd = plane;
    return TexInfoRecvStatus::Ok;
}
