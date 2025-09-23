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
    int sock_fd_ = -2;
    int event_fd_ = -2;
    struct pollfd pollfds_[2];
    static constexpr size_t SOCK_INDEX = 0;
    static constexpr size_t EVENT_INDEX = 1;
public:
    UnixFdpassReceiver(const std::string &path): path_(path) {
        event_fd_ = eventfd(1, 0);
        pollfds_[SOCK_INDEX].events = POLLIN;
        pollfds_[SOCK_INDEX].revents = 0;
        pollfds_[EVENT_INDEX].events = POLLIN;
        pollfds_[EVENT_INDEX].revents = 0;
        pollfds_[EVENT_INDEX].fd = event_fd_;
    }
    ~UnixFdpassReceiver() {
        if (sock_fd_ >= 0) close(sock_fd_);
        if (event_fd_ >= 0) close(event_fd_);
    }
    void interrupt() {
        uint64_t v = 1;
        write(event_fd_, &v, sizeof(v));
    }
    bool ensureConnected() {
        if (sock_fd_ >= 0) return true;
        int fd = socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0);
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
        strncpy(addr.sun_path, path_.c_str(), sizeof(addr.sun_path) - 1);
        int ret = connect(fd, reinterpret_cast<struct sockaddr*>(&addr), sizeof(addr));
        if (ret < 0) {
            if (errno==ENOENT || errno==ECONNREFUSED) {
                close(fd);
                logstream << "unix socket " << path_ << " not ready to connect";
                wallclock.sleepms(200);
                return false;
            }
            close(fd);
            throw Error("connect(" + path_ + ") failed");
        }
        sock_fd_ = fd;
        pollfds_[SOCK_INDEX].fd = sock_fd_;
        return true;
    }
    bool recvTexInfoAndFD(TexInfo &info, int &received_fd) {
        received_fd = -1;
        if (!ensureConnected()) return false;
        // Prepare to receive payload + a single FD via SCM_RIGHTS
        char control[CMSG_SPACE(sizeof(int))];
        struct iovec iov;
        iov.iov_base = &info;
        iov.iov_len = sizeof(info);
        struct msghdr msg;
        memset(&msg, 0, sizeof(msg));
        msg.msg_iov = &iov;
        msg.msg_iovlen = 1;
        msg.msg_control = control;
        msg.msg_controllen = sizeof(control);
        while (true) {
            int pret = poll(pollfds_, 2, -1);
            if (pret < 0) {
                if (errno == EINTR) continue;
                throw Error("poll() failed on unix socket");
            }
            if (pollfds_[EVENT_INDEX].revents & POLLIN) {
                return false;
            }
            if (pollfds_[SOCK_INDEX].revents & (POLLIN | POLLHUP | POLLERR)) {
                break;
            }
        }
        ssize_t r = recvmsg(sock_fd_, &msg, 0);
        if (r <= 0) {
            // connection reset or EOF: close and retry later
            close(sock_fd_);
            sock_fd_ = -1;
            return false;
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
            close(dmabuf_fd);
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
            close(dmabuf_fd);
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
