// The receive of a dma-browser frame message: a TexInfo header and its DMA-BUF fd over SCM_RIGHTS,
// with an eventfd in place of the DMA-BUF.
#include "nodes/hwaccel/TexInfoMessage.hpp"
#include <cassert>
#include <cstring>
#include <dirent.h>
#include <sys/eventfd.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <unistd.h>

static int openFds() {
    int count = 0;
    DIR *dir = opendir("/proc/self/fd");
    assert(dir);
    while (readdir(dir)) ++count;
    closedir(dir);
    return count - 3;   // ".", ".." and the directory stream itself
}

static void sendMessage(int sock, const void *bytes, size_t size, const int *fds, int count) {
    char control[CMSG_SPACE(2 * sizeof(int))];
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

int main() {
    int pair[2];
    assert(socketpair(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0, pair) == 0);
    int fds[2];
    for (int &fd : fds) fd = eventfd(0, EFD_CLOEXEC);
    TexInfo plain{};
    plain.width = 1280; plain.height = 720; plain.stride = 5120;
    plain.pixel_format = TEXINFO_PIX_FMT_BGRA; plain.frame_count = 7;
    const int baseline = openFds();
    TexInfoMessage m;

    sendMessage(pair[0], &plain, sizeof(plain), fds, 1);
    assert(recvTexInfoMessage(pair[1], m) == TexInfoRecvStatus::Ok);
    assert(m.plane_fd >= 0 && m.info.frame_count == 7 && m.info.width == 1280);
    struct stat st;
    assert(fstat(m.plane_fd, &st) == 0);
    m.closeFd();
    assert(m.plane_fd < 0 && openFds() == baseline);

    // A malformed message keeps nothing, so nothing can be misread or leak; a whole header is
    // still filled in, so the frame it names can be acknowledged.
    sendMessage(pair[0], &plain, sizeof(plain), fds, 2);
    assert(recvTexInfoMessage(pair[1], m) == TexInfoRecvStatus::Truncated);
    assert(m.plane_fd < 0 && m.info.frame_count == 7);
    sendMessage(pair[0], &plain, sizeof(plain), fds, 0);
    assert(recvTexInfoMessage(pair[1], m) == TexInfoRecvStatus::NoFd);
    sendMessage(pair[0], &plain, 20, fds, 1);
    assert(recvTexInfoMessage(pair[1], m) == TexInfoRecvStatus::Short);
    assert(m.plane_fd < 0);
    assert(openFds() == baseline);

    close(pair[0]);
    assert(recvTexInfoMessage(pair[1], m) == TexInfoRecvStatus::Closed);
    close(pair[1]);
    for (int fd : fds) close(fd);
}
