#pragma once
#include "../../util.hpp"
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <optional>
#include <sys/file.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <unistd.h>
#include <utility>
namespace avp::ipc {
class Fd {
    int fd_ = -1;

public:
    Fd() = default;
    explicit Fd(int fd) : fd_(fd) {}
    Fd(Fd &&other) noexcept : fd_(std::exchange(other.fd_, -1)) {}
    Fd &operator=(Fd &&other) noexcept {
        Fd old(std::move(*this));
        fd_ = std::exchange(other.fd_, -1);
        return *this;
    }
    ~Fd() { if (fd_ >= 0) close(fd_); }
    int get() const { return fd_; }
    explicit operator bool() const { return fd_ >= 0; }
};

inline sockaddr_un socketAddress(const std::string &path) {
    sockaddr_un address{};
    address.sun_family = AF_UNIX;
    if (path.empty() || path.front() != '/' || path.size() >= sizeof(address.sun_path))
        throw Error("Unix IPC: socket must be a short absolute path");
    std::memcpy(address.sun_path, path.c_str(), path.size() + 1);
    return address;
}

inline Fd endpoint() {
    Fd result(socket(AF_UNIX, SOCK_SEQPACKET | SOCK_CLOEXEC | SOCK_NONBLOCK, 0));
    if (!result) throw Error("Unix IPC: cannot create Unix socket");
    return result;
}

class PacketListener {
    std::string path_;
    Fd lock_, listener_;
public:
    explicit PacketListener(std::string path, std::optional<mode_t> mode = {}) : path_(std::move(path)), listener_(endpoint()) {
        const auto address = socketAddress(path_);
        lock_ = Fd(open((path_ + ".lock").c_str(), O_CREAT | O_RDWR | O_CLOEXEC | O_NOFOLLOW, 0600));
        if (!lock_ || flock(lock_.get(), LOCK_EX | LOCK_NB) != 0)
            throw Error("Unix IPC: socket is already owned: " + path_);
        struct stat existing{};
        if (lstat(path_.c_str(), &existing) == 0 && S_ISSOCK(existing.st_mode)) {
            auto probe = endpoint();
            if (connect(probe.get(), reinterpret_cast<const sockaddr *>(&address), sizeof(address)) < 0 &&
                errno == ECONNREFUSED)
                unlink(path_.c_str());
        }
        if (bind(listener_.get(), reinterpret_cast<const sockaddr *>(&address), sizeof(address)) != 0)
            throw Error("Unix IPC: bind " + path_ + ": " + std::strerror(errno));
        if ((mode && chmod(path_.c_str(), *mode) != 0) || listen(listener_.get(), 1) != 0) {
            unlink(path_.c_str());
            throw Error("Unix IPC: listen failed");
        }
    }
    ~PacketListener() { unlink(path_.c_str()); }
    Fd acceptPeer() const { return Fd(accept4(listener_.get(), nullptr, nullptr, SOCK_CLOEXEC | SOCK_NONBLOCK)); }
};
}
