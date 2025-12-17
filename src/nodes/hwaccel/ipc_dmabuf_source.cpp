#include "../node_common.hpp"
extern "C" {
#include <libavutil/buffer.h>
#include <libavutil/hwcontext_drm.h>
#include <libavutil/mem.h>
#include <libdrm/drm_fourcc.h>
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
#include <time.h>
#include "../../hwaccel.hpp"

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

class UnixFdpassClient {
protected:
    std::string path_;
    int conn_fd_ = -2;
    int event_fd_ = -2;
    struct pollfd pollfds_[2];
    static constexpr size_t CONN_INDEX = 0;
    static constexpr size_t EVENT_INDEX = 1;
    bool connecting_ = false;
public:
    UnixFdpassClient(const std::string &path): path_(path) {
        event_fd_ = eventfd(0, 0);
        pollfds_[CONN_INDEX].events = POLLIN;
        pollfds_[CONN_INDEX].revents = 0;
        pollfds_[EVENT_INDEX].events = POLLIN;
        pollfds_[EVENT_INDEX].revents = 0;
        pollfds_[EVENT_INDEX].fd = event_fd_;
        pollfds_[CONN_INDEX].fd = -1;
    }
    ~UnixFdpassClient() {
        if (conn_fd_ >= 0) close(conn_fd_);
        if (event_fd_ >= 0) close(event_fd_);
    }
    void interrupt() {
        uint64_t v = 1;
        write(event_fd_, &v, sizeof(v));
    }
    bool ensureConnected() {
        if (conn_fd_ >= 0) return true;
        int fd = socket(AF_UNIX, SOCK_STREAM, 0);
        if (fd < 0) {
            return false;
        }
        // make non-blocking
        int flags = fcntl(fd, F_GETFL, 0);
        if (flags >= 0) fcntl(fd, F_SETFL, flags | O_NONBLOCK);
        struct sockaddr_un addr;
        memset(&addr, 0, sizeof(addr));
        addr.sun_family = AF_UNIX;
        if (path_.size() >= sizeof(addr.sun_path)) {
            close(fd);
            return false;
        }
        strncpy(addr.sun_path, path_.c_str(), sizeof(addr.sun_path) - 1);
        int rc = connect(fd, reinterpret_cast<struct sockaddr*>(&addr), sizeof(addr));
        if (rc < 0 && errno != EINPROGRESS) {
            int e = errno;
            (void)e;
            close(fd);
            return false;
        }
        conn_fd_ = fd;
        pollfds_[CONN_INDEX].fd = conn_fd_;
        connecting_ = (rc < 0); // EINPROGRESS
        pollfds_[CONN_INDEX].events = connecting_ ? (POLLIN | POLLOUT) : POLLIN;
        return true;
    }
    bool recvTexInfoAndFD(TexInfo &info, int &received_fd) {
        received_fd = -1;
        // Prepare to receive payload + a single FD via SCM_RIGHTS
        char control[CMSG_SPACE(sizeof(int))];
        struct iovec iov;
        iov.iov_base = &info;
        iov.iov_len = sizeof(info);
        struct msghdr msg;
        while (true) {
            // Ensure we have a connection
            if (conn_fd_ < 0) {
                if (!ensureConnected()) {
                    // back off a bit when connect fails
                    struct timespec ts; ts.tv_sec = 0; ts.tv_nsec = 10 * 1000 * 1000; // 10ms
                    nanosleep(&ts, nullptr);
                    // also check for interrupt events
                    // fall through to poll which includes EVENT fd even when conn is -1
                }
                pollfds_[CONN_INDEX].fd = conn_fd_;
            }
            // Update pollfds each iteration
            pollfds_[CONN_INDEX].revents = 0;
            pollfds_[EVENT_INDEX].revents = 0;
            int pret = poll(pollfds_, 2, 50);
            if (pret < 0) {
                if (errno == EINTR) continue;
                throw Error("poll() failed on unix socket");
            }
            if (pret == 0) {
                // timeout
                continue;
            }
            if (pollfds_[EVENT_INDEX].revents & POLLIN) {
                pollfds_[EVENT_INDEX].revents = 0;
                int64_t blackhole;
                ::read(event_fd_, &blackhole, sizeof blackhole);
                return false;
            }
            if (conn_fd_ < 0) {
                continue;
            }
            // Finish connect if it was in progress
            if (connecting_ && (pollfds_[CONN_INDEX].revents & (POLLOUT | POLLIN | POLLERR | POLLHUP | POLLNVAL))) {
                int soerr = 0; socklen_t slen = sizeof(soerr);
                if (getsockopt(conn_fd_, SOL_SOCKET, SO_ERROR, &soerr, &slen) < 0 || soerr != 0) {
                    close(conn_fd_);
                    conn_fd_ = -1;
                    pollfds_[CONN_INDEX].fd = -1;
                    pollfds_[CONN_INDEX].events = POLLIN;
                    connecting_ = false;
                    continue;
                }
                connecting_ = false;
                pollfds_[CONN_INDEX].events = POLLIN;
            }
            if (pollfds_[CONN_INDEX].revents & (POLLIN)) {
                memset(&msg, 0, sizeof(msg));
                msg.msg_iov = &iov;
                msg.msg_iovlen = 1;
                msg.msg_control = control;
                msg.msg_controllen = sizeof(control);
                ssize_t r = recvmsg(conn_fd_, &msg, MSG_DONTWAIT);
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
            if (pollfds_[CONN_INDEX].revents & (POLLHUP | POLLERR | POLLNVAL)) {
                close(conn_fd_);
                conn_fd_ = -1;
                pollfds_[CONN_INDEX].fd = -1;
                connecting_ = false;
            }
        }
    }
};

class IPCDMABUFSource: public NodeSingleOutput<av::VideoFrame>, public IStoppable, public ReportsFinishByFlag, public IVideoFormatSource {
protected:
    UnixFdpassClient receiver_;
    int width_ = 0;
    int height_ = 0;
    std::shared_ptr<HWAccelDevice> hwaccel_;
    AVBufferRef* hw_frames_ctx_ = nullptr;
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

        static constexpr uint32_t PIX_FMT_RGBA = ('R' << 24 | 'G' << 16 | 'B' << 8 | 'A');
        static constexpr uint32_t PIX_FMT_BGRA = ('B' << 24 | 'G' << 16 | 'R' << 8 | 'A');

        // Ensure/refresh HW frames context for filters
        if (hwaccel_) {
            bool need_recreate = false;
            if (!hw_frames_ctx_) need_recreate = true;
            if (!need_recreate && (width_ != (int)ti.width || height_ != (int)ti.height)) need_recreate = true;
            if (need_recreate) {
                if (hw_frames_ctx_) {
                    av_buffer_unref(&hw_frames_ctx_);
                }
                hw_frames_ctx_ = av_hwframe_ctx_alloc(hwaccel_->deviceContext());
                if (!hw_frames_ctx_) {
                    logstream << "failed to alloc hw_frames_ctx";
                } else {
                    AVHWFramesContext *frmctx = (AVHWFramesContext *)(hw_frames_ctx_->data);
                    // Map source sw pixel format
                    frmctx->sw_format = (ti.pixel_format == PIX_FMT_RGBA) ? AV_PIX_FMT_RGBA64LE :
                                        (ti.pixel_format == PIX_FMT_BGRA) ? AV_PIX_FMT_BGRA64LE : AV_PIX_FMT_NONE;
                    frmctx->width = ti.width;
                    frmctx->height = ti.height;

                    AVHWDeviceContext* devctx = (AVHWDeviceContext *)(hwaccel_->deviceContext()->data);
                    switch (devctx->type) {
                        case AV_HWDEVICE_TYPE_DRM: frmctx->format = AV_PIX_FMT_DRM_PRIME; break;
                        case AV_HWDEVICE_TYPE_VAAPI: frmctx->format = AV_PIX_FMT_VAAPI; break;
                        case AV_HWDEVICE_TYPE_CUDA: frmctx->format = AV_PIX_FMT_CUDA; break;
                        default: frmctx->format = AV_PIX_FMT_DRM_PRIME; break;
                    }
                    int r = av_hwframe_ctx_init(hw_frames_ctx_);
                    if (r != 0) {
                        logstream << "av_hwframe_ctx_init failed: " << av::error2string(r);
                        av_buffer_unref(&hw_frames_ctx_);
                        hw_frames_ctx_ = nullptr;
                    }
                }
            }
        }

        AVDRMFrameDescriptor *desc = (AVDRMFrameDescriptor*)av_mallocz(sizeof(AVDRMFrameDescriptor));
        if (!desc) {
            if (dmabuf_fd >= 0) close(dmabuf_fd);
            logstream << "av_mallocz failed for DRM descriptor";
            return;
        }

        desc->nb_objects = 1;
        desc->objects[0].fd = dmabuf_fd;
        desc->objects[0].size = ti.offset + ti.stride * ti.height;
        desc->objects[0].format_modifier = ti.modifier;

        desc->nb_layers = 1;
        desc->layers[0].format = (ti.pixel_format == PIX_FMT_RGBA) ? DRM_FORMAT_ABGR8888 :
            (ti.pixel_format == PIX_FMT_BGRA) ? DRM_FORMAT_ARGB8888 : 0;
        desc->layers[0].nb_planes = 1;
        desc->layers[0].planes[0].object_index = 0;
        desc->layers[0].planes[0].offset = ti.offset;
        desc->layers[0].planes[0].pitch = ti.stride;

        av::VideoFrame vfrm;
        vfrm.raw()->format = AV_PIX_FMT_DRM_PRIME;
        vfrm.raw()->width = ti.width;
        vfrm.raw()->height = ti.height;
        vfrm.setTimeBase({1, 1000000000});
        vfrm.raw()->pts = ti.timestamp;

        if (hw_frames_ctx_) {
            vfrm.raw()->hw_frames_ctx = av_buffer_ref(hw_frames_ctx_);
        }

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
        vfrm.raw()->linesize[0] = ti.stride; // needed for isValid()

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
        if (params.count("hwaccel")) {
            r->hwaccel_ = InstanceSharedObjects<HWAccelDevice>::get(nci.instance, params["hwaccel"]);
        }
        return r;
    }
    ~IPCDMABUFSource() {
        if (hw_frames_ctx_) {
            av_buffer_unref(&hw_frames_ctx_);
            hw_frames_ctx_ = nullptr;
        }
    }
};

DECLNODE(ipc_dmabuf_source, IPCDMABUFSource);
