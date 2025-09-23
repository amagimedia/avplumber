#include "node_common.hpp"
extern "C" {
#include <libavutil/buffer.h>
#include <libavutil/hwcontext_drm.h>
#include <libavutil/mem.h>
}
#include <sys/socket.h>
#include <sys/un.h>
#include <sys/poll.h>
#include <sys/eventfd.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include <cstring>
#include <sys/stat.h>

#pragma pack(push, 1)
struct TexInfo {
    uint32_t width;
    uint32_t height;
    uint32_t stride;
    uint32_t pixel_format; // DRM FourCC or compatible value
    uint64_t modifier;
    uint64_t offset;
    uint64_t timestamp; // nanoseconds
    uint64_t frame_count;
};
#pragma pack(pop)

class UnixFdpassReceiver {
protected:
    std::string path_;
    int listen_fd_ = -2;
    int conn_fd_ = -2;
    int event_fd_ = -2;
    struct pollfd pollfds_[3];
    static constexpr size_t LISTEN_INDEX = 0;
    static constexpr size_t CONN_INDEX = 1;
    static constexpr size_t EVENT_INDEX = 2;
public:
    UnixFdpassReceiver(const std::string &path): path_(path) {
        event_fd_ = eventfd(0, 0);
        pollfds_[LISTEN_INDEX].events = POLLIN;
        pollfds_[LISTEN_INDEX].revents = 0;
        pollfds_[CONN_INDEX].events = POLLIN;
        pollfds_[CONN_INDEX].revents = 0;
        pollfds_[EVENT_INDEX].events = POLLIN;
        pollfds_[EVENT_INDEX].revents = 0;
        pollfds_[EVENT_INDEX].fd = event_fd_;
    }
    ~UnixFdpassReceiver() {
        if (conn_fd_ >= 0) close(conn_fd_);
        if (listen_fd_ >= 0) close(listen_fd_);
        if (event_fd_ >= 0) close(event_fd_);
        if (!path_.empty()) unlink(path_.c_str());
    }
    void interrupt() {
        uint64_t v = 1;
        write(event_fd_, &v, sizeof(v));
    }
    bool ensureListening() {
        if (listen_fd_ >= 0) return true;
        int fd = socket(AF_UNIX, SOCK_STREAM, 0);
        if (fd < 0) {
            throw Error("socket() failed for unix path " + path_);
        }
        struct sockaddr_un addr;
        memset(&addr, 0, sizeof(addr));
        addr.sun_family = AF_UNIX;
        if (path_.size() >= sizeof(addr.sun_path)) {
            close(fd);
            throw Error("unix socket path too long: " + path_);
        }
        // remove stale path
        unlink(path_.c_str());
        strncpy(addr.sun_path, path_.c_str(), sizeof(addr.sun_path) - 1);
        if (bind(fd, reinterpret_cast<struct sockaddr*>(&addr), sizeof(addr)) < 0) {
            int e = errno;
            close(fd);
            throw Error(std::string("bind(") + path_ + ") failed: " + strerror(e));
        }
        chmod(path_.c_str(), 0777);
        if (listen(fd, 512) < 0) {
            int e = errno;
            close(fd);
            unlink(path_.c_str());
            throw Error(std::string("listen(") + path_ + ") failed: " + strerror(e));
        }
        // make non-blocking
        int flags = fcntl(fd, F_GETFL, 0);
        if (flags >= 0) fcntl(fd, F_SETFL, flags | O_NONBLOCK);
        listen_fd_ = fd;
        pollfds_[LISTEN_INDEX].fd = listen_fd_;
        pollfds_[CONN_INDEX].fd = -1;
        return true;
    }
    bool recvTexInfoAndFD(TexInfo &info, int &received_fd) {
        received_fd = -1;
        if (!ensureListening()) return false;
        // Prepare to receive payload + a single FD via SCM_RIGHTS
        char control[CMSG_SPACE(sizeof(int))];
        struct iovec iov;
        iov.iov_base = &info;
        iov.iov_len = sizeof(info);
        struct msghdr msg;
        while (true) {
            // Update conn pollfd each iteration
            pollfds_[CONN_INDEX].fd = conn_fd_;
            pollfds_[LISTEN_INDEX].revents = 0;
            pollfds_[CONN_INDEX].revents = 0;
            pollfds_[EVENT_INDEX].revents = 0;
            int pret = poll(pollfds_, 3, -1);
            if (pret < 0) {
                if (errno == EINTR) continue;
                throw Error("poll() failed on unix socket");
            }
            if (pollfds_[EVENT_INDEX].revents & POLLIN) {
                pollfds_[EVENT_INDEX].revents = 0;
                int64_t blackhole;
                ::read(event_fd_, &blackhole, sizeof blackhole);
                return false;
            }
            // Accept new connection if pending
            if (listen_fd_ >= 0 && (pollfds_[LISTEN_INDEX].revents & POLLIN)) {
                int conn = accept(listen_fd_, nullptr, nullptr);
                if (conn >= 0) {
                    // set nonblocking
                    int cflags = fcntl(conn, F_GETFL, 0);
                    if (cflags >= 0) fcntl(conn, F_SETFL, cflags | O_NONBLOCK);
                    if (conn_fd_ >= 0) close(conn_fd_);
                    conn_fd_ = conn;
                    pollfds_[CONN_INDEX].fd = conn_fd_;
                }
                pollfds_[LISTEN_INDEX].revents = 0;
            }
            if (conn_fd_ < 0) {
                continue;
            }
            if (pollfds_[CONN_INDEX].revents & (POLLIN)) {
                memset(&msg, 0, sizeof(msg));
                msg.msg_iov = &iov;
                msg.msg_iovlen = 1;
                msg.msg_control = control;
                msg.msg_controllen = sizeof(control);
                ssize_t r = recvmsg(conn_fd_, &msg, 0);
                if (r <= 0) {
                    // connection reset or EOF: close and wait for next
                    close(conn_fd_);
                    conn_fd_ = -1;
                    pollfds_[CONN_INDEX].fd = -1;
                    continue;
                }
                // Ancillary data
                for (struct cmsghdr *cmsg = CMSG_FIRSTHDR(&msg);
                     cmsg != nullptr; cmsg = CMSG_NXTHDR(&msg, cmsg)) {
                    if (cmsg->cmsg_level == SOL_SOCKET && cmsg->cmsg_type == SCM_RIGHTS) {
                        int *fds = reinterpret_cast<int*>(CMSG_DATA(cmsg));
                        // Expect one fd
                        received_fd = fds[0];
                        break;
                    }
                }
                if (received_fd < 0) {
                    logstream << "fdpass: recvmsg without FD";
                    return false;
                }
                if (static_cast<size_t>(r) < sizeof(TexInfo)) {
                    logstream << "fdpass: metadata too small (" << r << ")";
                    close(received_fd);
                    received_fd = -1;
                    return false;
                }
                return true;
            }
            if (pollfds_[CONN_INDEX].revents & (POLLHUP | POLLERR)) {
                close(conn_fd_);
                conn_fd_ = -1;
                pollfds_[CONN_INDEX].fd = -1;
            }
        }
    }
};

class IPCDMABUFSource: public NodeSingleOutput<av::VideoFrame>, public IStoppable, public ReportsFinishByFlag, public IVideoFormatSource {
protected:
    UnixFdpassReceiver receiver_;
    int width_ = 0;
    int height_ = 0;
public:
    using NodeSingleOutput::NodeSingleOutput;
    IPCDMABUFSource(std::unique_ptr<SinkType> &&sink, const std::string &sock_path):
        NodeSingleOutput(std::move(sink)), receiver_(sock_path) {
    }
    virtual int width() { return width_; }
    virtual int height() { return height_; }
    virtual av::PixelFormat pixelFormat() { return av::PixelFormat(AV_PIX_FMT_DRM_PRIME); }
    virtual void stop() {
        receiver_.interrupt();
        this->finished_ = true;
    }
    virtual void process() {
        TexInfo ti{};
        int dmabuf_fd = -1;
        if (!receiver_.recvTexInfoAndFD(ti, dmabuf_fd)) {
            wallclock.sleepms(5);
            return;
        }

        AVDRMFrameDescriptor *desc = (AVDRMFrameDescriptor*)av_mallocz(sizeof(AVDRMFrameDescriptor));
        if (!desc) {
            if (dmabuf_fd >= 0) close(dmabuf_fd);
            logstream << "av_mallocz failed for DRM descriptor";
            return;
        }

        desc->nb_objects = 1;
        desc->objects[0].fd = dmabuf_fd;
        desc->objects[0].size = 0; // unknown
        desc->objects[0].format_modifier = ti.modifier;

        desc->nb_layers = 1;
        desc->layers[0].format = ti.pixel_format;
        desc->layers[0].nb_planes = 1;
        desc->layers[0].planes[0].object_index = 0;
        desc->layers[0].planes[0].offset = ti.offset;
        desc->layers[0].planes[0].pitch = ti.stride;

        av::VideoFrame vfrm;
        vfrm.raw()->format = AV_PIX_FMT_DRM_PRIME;
        vfrm.raw()->width = ti.width;
        vfrm.raw()->height = ti.height;
        vfrm.setTimeBase({1, 1000000});
        vfrm.raw()->pts = ti.timestamp / 1000; // ns -> us

        AVBufferRef *buf = av_buffer_create(reinterpret_cast<uint8_t*>(desc), sizeof(*desc),
            [](void *opaque, uint8_t *data){
                AVDRMFrameDescriptor *d = reinterpret_cast<AVDRMFrameDescriptor*>(data);
                for (int i=0; i<d->nb_objects; i++) {
                    if (d->objects[i].fd >= 0) close(d->objects[i].fd);
                }
                av_free(d);
            }, desc, 0);
        if (!buf) {
            if (dmabuf_fd >= 0) close(dmabuf_fd);
            av_free(desc);
            logstream << "av_buffer_create failed for DRM descriptor";
            return;
        }
        vfrm.raw()->buf[0] = buf;
        vfrm.raw()->data[0] = reinterpret_cast<uint8_t*>(desc);

        width_ = ti.width;
        height_ = ti.height;

        vfrm.setComplete(true);
        this->sink_->put(vfrm);
    }
    static std::shared_ptr<IPCDMABUFSource> create(NodeCreationInfo &nci) {
        EdgeManager &edges = nci.edges;
        const Parameters &params = nci.params;
        std::shared_ptr<Edge<av::VideoFrame>> edge = edges.find<av::VideoFrame>(params["dst"]);
        auto r = std::make_shared<IPCDMABUFSource>(make_unique<EdgeSink<av::VideoFrame>>(edge), params["socket"]);
        return r;
    }
};

DECLNODE(ipc_dmabuf_source, IPCDMABUFSource);
