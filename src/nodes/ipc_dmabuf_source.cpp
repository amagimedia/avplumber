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

static constexpr int MAX_DMABUF_OBJECTS = 4;
static constexpr int MAX_DMABUF_LAYERS = 2;
static constexpr int MAX_DMABUF_PLANES = 4;

struct DMABufPlaneDesc {
    uint32_t object_index;
    uint32_t offset;
    uint32_t pitch;
};

struct DMABufLayerDesc {
    uint32_t drm_format; // DRM FourCC
    uint32_t nb_planes;
    DMABufPlaneDesc planes[MAX_DMABUF_PLANES];
};

struct DMABufObjectDesc {
    uint64_t size;
    uint64_t modifier; // DRM format modifier
};

struct DMABufHeader {
    int64_t magic_number;
    uint32_t frame_width;
    uint32_t frame_height;
    struct timespec timestamp; // 1/1e9 timebase

    uint32_t nb_objects;
    DMABufObjectDesc objects[MAX_DMABUF_OBJECTS];

    uint32_t nb_layers;
    DMABufLayerDesc layers[MAX_DMABUF_LAYERS];
};

class UnixRightsReceiver {
protected:
    std::string path_;
    int sock_fd_ = -2;
    int event_fd_ = -2;
    struct pollfd pollfds_[2];
    static constexpr size_t SOCK_INDEX = 0;
    static constexpr size_t EVENT_INDEX = 1;
public:
    UnixRightsReceiver(const std::string &path): path_(path) {
        event_fd_ = eventfd(1, 0);
        pollfds_[SOCK_INDEX].events = POLLIN;
        pollfds_[SOCK_INDEX].revents = 0;
        pollfds_[EVENT_INDEX].events = POLLIN;
        pollfds_[EVENT_INDEX].revents = 0;
        pollfds_[EVENT_INDEX].fd = event_fd_;
    }
    ~UnixRightsReceiver() {
        if (sock_fd_ >= 0) close(sock_fd_);
        if (event_fd_ >= 0) close(event_fd_);
    }
    void interrupt() {
        uint64_t v = 1;
        write(event_fd_, &v, sizeof(v));
    }
    bool ensureConnected() {
        if (sock_fd_ >= 0) return true;
        int fd = socket(AF_UNIX, SOCK_SEQPACKET, 0);
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
                wallclock.sleepms(500);
                return false;
            }
            close(fd);
            throw Error("connect(" + path_ + ") failed");
        }
        sock_fd_ = fd;
        pollfds_[SOCK_INDEX].fd = sock_fd_;
        return true;
    }
    bool recvHeaderAndFDs(DMABufHeader &hdr, int *fds, int fds_capacity, int &fds_received) {
        if (!ensureConnected()) return false;
        struct msghdr msg;
        struct iovec iov;
        iov.iov_base = &hdr;
        iov.iov_len = sizeof(hdr);
        char cmsgbuf[CMSG_SPACE(sizeof(int) * MAX_DMABUF_OBJECTS)];
        memset(&msg, 0, sizeof(msg));
        msg.msg_iov = &iov;
        msg.msg_iovlen = 1;
        msg.msg_control = cmsgbuf;
        msg.msg_controllen = sizeof(cmsgbuf);
        while (true) {
            int pret = poll(pollfds_, 2, -1);
            if (pret < 0) {
                if (errno == EINTR) continue;
                throw Error("poll() failed on unix socket");
            }
            if (pollfds_[EVENT_INDEX].revents & POLLIN) {
                return false;
            }
            if (pollfds_[SOCK_INDEX].revents & POLLIN) {
                break;
            }
        }
        ssize_t r = recvmsg(sock_fd_, &msg, MSG_CMSG_CLOEXEC);
        if (r < 0) {
            if (errno==EAGAIN || errno==EWOULDBLOCK) return false;
            if (errno==ECONNRESET) {
                close(sock_fd_);
                sock_fd_ = -1;
                return false;
            }
            throw Error("recvmsg() failed on unix socket");
        }
        if (static_cast<size_t>(r) < sizeof(DMABufHeader)) {
            logstream << "short recvmsg() for DMABUF header";
            return false;
        }
        fds_received = 0;
        for (struct cmsghdr *cmsg = CMSG_FIRSTHDR(&msg);
             cmsg != nullptr;
             cmsg = CMSG_NXTHDR(&msg, cmsg)) {
            if (cmsg->cmsg_level == SOL_SOCKET && cmsg->cmsg_type == SCM_RIGHTS) {
                size_t count = (cmsg->cmsg_len - CMSG_LEN(0)) / sizeof(int);
                int *cmsg_fds = reinterpret_cast<int*>(CMSG_DATA(cmsg));
                for (size_t i=0; i<count && fds_received < fds_capacity; i++) {
                    fds[fds_received++] = cmsg_fds[i];
                }
            }
        }
        return true;
    }
};

class IPCDMABUFSource: public NodeSingleOutput<av::VideoFrame>, public IStoppable, public ReportsFinishByFlag, public IVideoFormatSource {
protected:
    UnixRightsReceiver receiver_;
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
        DMABufHeader hdr;
        int fds[MAX_DMABUF_OBJECTS] = {-1,-1,-1,-1};
        int nfds = 0;
        if (!receiver_.recvHeaderAndFDs(hdr, fds, MAX_DMABUF_OBJECTS, nfds)) {
            wallclock.sleepms(10);
            return;
        }
        if (hdr.magic_number != 0x12345678abcddd01) {
            for (int i=0; i<nfds; i++) if (fds[i]>=0) close(fds[i]);
            logstream << "invalid magic number, ignoring DMABUF packet";
            return;
        }
        if (hdr.nb_objects > static_cast<uint32_t>(nfds)) {
            for (int i=0; i<nfds; i++) if (fds[i]>=0) close(fds[i]);
            logstream << "not enough fds received for DMABUF objects";
            return;
        }

        AVDRMFrameDescriptor *desc = (AVDRMFrameDescriptor*)av_mallocz(sizeof(AVDRMFrameDescriptor));
        if (!desc) {
            for (int i=0; i<nfds; i++) if (fds[i]>=0) close(fds[i]);
            logstream << "av_mallocz failed for DRM descriptor";
            return;
        }
        desc->nb_objects = hdr.nb_objects;
        for (uint32_t i=0; i<hdr.nb_objects && i<MAX_DMABUF_OBJECTS; i++) {
            desc->objects[i].fd = fds[i];
            desc->objects[i].size = hdr.objects[i].size;
            desc->objects[i].format_modifier = hdr.objects[i].modifier;
        }
        desc->nb_layers = hdr.nb_layers;
        for (uint32_t l=0; l<hdr.nb_layers && l<MAX_DMABUF_LAYERS; l++) {
            desc->layers[l].format = hdr.layers[l].drm_format;
            desc->layers[l].nb_planes = hdr.layers[l].nb_planes;
            for (uint32_t p=0; p<hdr.layers[l].nb_planes && p<MAX_DMABUF_PLANES; p++) {
                desc->layers[l].planes[p].object_index = hdr.layers[l].planes[p].object_index;
                desc->layers[l].planes[p].offset = hdr.layers[l].planes[p].offset;
                desc->layers[l].planes[p].pitch = hdr.layers[l].planes[p].pitch;
            }
        }

        av::VideoFrame vfrm;
        vfrm.raw()->format = AV_PIX_FMT_DRM_PRIME;
        vfrm.raw()->width = hdr.frame_width;
        vfrm.raw()->height = hdr.frame_height;
        vfrm.setTimeBase({1, 1000000});
        vfrm.raw()->pts = hdr.timestamp.tv_sec * 1000000 + hdr.timestamp.tv_nsec / 1000;

        AVBufferRef *buf = av_buffer_create(reinterpret_cast<uint8_t*>(desc), sizeof(*desc),
            [](void *opaque, uint8_t *data){
                AVDRMFrameDescriptor *d = reinterpret_cast<AVDRMFrameDescriptor*>(data);
                for (int i=0; i<d->nb_objects; i++) {
                    if (d->objects[i].fd >= 0) close(d->objects[i].fd);
                }
                av_free(d);
            }, desc, 0);
        if (!buf) {
            for (int i=0; i<nfds; i++) if (fds[i]>=0) close(fds[i]);
            av_free(desc);
            logstream << "av_buffer_create failed for DRM descriptor";
            return;
        }
        vfrm.raw()->buf[0] = buf;
        vfrm.raw()->data[0] = reinterpret_cast<uint8_t*>(desc);

        width_ = hdr.frame_width;
        height_ = hdr.frame_height;

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


