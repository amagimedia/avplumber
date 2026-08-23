#include "../node_common.hpp"
extern "C" {
#include <libavutil/buffer.h>
#include <libavutil/dict.h>
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
#include <stdint.h>
#include <array>
#include <atomic>
#include <deque>
#include <memory>
#include <mutex>
#include <new>
#include "../../hwaccel.hpp"
#include "../../../deps/dma-browser/addons/fdpass/dmabuf_ack.h"

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

static constexpr uint32_t SCOREBOARD_TRACE_MAGIC = 0x31544253; // "SBT1"
static constexpr uint16_t SCOREBOARD_TRACE_VERSION = 1;

struct TraceTexInfo {
    TexInfo base;
    uint32_t magic;
    uint16_t version;
    uint16_t header_bytes;
    uint64_t trace_id;
    uint64_t sequence;
    int64_t source_pts_ms;
    uint64_t renderer_received_ns;
    uint64_t raf_ns;
    uint64_t dom_applied_ns;
    uint64_t paint_ns;
    uint64_t dmabuf_send_ns;
};
#pragma pack(pop)

static_assert(sizeof(TexInfo) == 48, "default DMA-BUF header changed");
static_assert(sizeof(TraceTexInfo) == 120, "scoreboard trace DMA-BUF header changed");

static uint64_t monotonicNs() {
    struct timespec ts {};
    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) return 0;
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

class DmabufReleaseAckQueue {
    struct PendingAck {
        uint64_t connection_generation;
        std::array<uint8_t, DMABUF_RELEASE_ACK_BYTES> bytes;
    };

    int wake_fd_ = -1;
    mutable std::mutex mutex_;
    std::deque<PendingAck> pending_;
    size_t front_offset_ = 0;
    uint64_t active_generation_ = 0;
    uint64_t next_generation_ = 1;
    std::atomic<bool> interrupted_{false};

    void wake() const {
        const uint64_t value = 1;
        if (wake_fd_ >= 0) {
            (void)::write(wake_fd_, &value, sizeof(value));
        }
    }

public:
    DmabufReleaseAckQueue(): wake_fd_(eventfd(0, EFD_CLOEXEC | EFD_NONBLOCK)) {
        if (wake_fd_ < 0) throw Error("eventfd() failed for DMA-BUF release acknowledgements");
    }

    ~DmabufReleaseAckQueue() {
        if (wake_fd_ >= 0) ::close(wake_fd_);
    }

    int wakeFd() const { return wake_fd_; }
    bool interrupted() const { return interrupted_.load(); }

    uint64_t beginConnection() {
        std::lock_guard<std::mutex> lock(mutex_);
        pending_.clear();
        front_offset_ = 0;
        active_generation_ = next_generation_++;
        return active_generation_;
    }

    void endConnection(uint64_t generation) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (active_generation_ != generation) return;
        active_generation_ = 0;
        pending_.clear();
        front_offset_ = 0;
    }

    void enqueue(uint64_t generation, uint64_t frame_count) {
        if (interrupted()) return;
        PendingAck ack{};
        ack.connection_generation = generation;
        dmabufEncodeReleaseAck(ack.bytes.data(), frame_count);
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (generation == 0 || generation != active_generation_) return;
            pending_.push_back(ack);
        }
        wake();
    }

    void interrupt() {
        interrupted_.store(true);
        wake();
    }

    void drainWakeFd() const {
        uint64_t value;
        while (::read(wake_fd_, &value, sizeof(value)) == sizeof(value)) {}
    }

    bool hasPending() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return !pending_.empty();
    }

    bool flush(int socket_fd, uint64_t generation) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (generation == 0 || generation != active_generation_) return true;
        while (!pending_.empty()) {
            PendingAck &ack = pending_.front();
            if (ack.connection_generation != generation) {
                pending_.pop_front();
                front_offset_ = 0;
                continue;
            }
            const uint8_t *data = ack.bytes.data() + front_offset_;
            const size_t remaining = ack.bytes.size() - front_offset_;
            const ssize_t sent = ::send(socket_fd, data, remaining, MSG_DONTWAIT | MSG_NOSIGNAL);
            if (sent > 0) {
                front_offset_ += static_cast<size_t>(sent);
                if (front_offset_ == ack.bytes.size()) {
                    pending_.pop_front();
                    front_offset_ = 0;
                }
                continue;
            }
            if (sent < 0 && errno == EINTR) continue;
            if (sent < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) return true;
            return false;
        }
        return true;
    }
};

class UnixFdpassClient {
protected:
    std::string path_;
    int conn_fd_ = -2;
    std::shared_ptr<DmabufReleaseAckQueue> ack_queue_;
    uint64_t connection_generation_ = 0;
    struct pollfd pollfds_[2];
    static constexpr size_t CONN_INDEX = 0;
    static constexpr size_t EVENT_INDEX = 1;
    bool connecting_ = false;

    void closeConnection() {
        if (conn_fd_ >= 0) ::close(conn_fd_);
        conn_fd_ = -1;
        pollfds_[CONN_INDEX].fd = -1;
        pollfds_[CONN_INDEX].events = POLLIN;
        connecting_ = false;
        ack_queue_->endConnection(connection_generation_);
        connection_generation_ = 0;
    }
public:
    UnixFdpassClient(const std::string &path):
        path_(path), ack_queue_(std::make_shared<DmabufReleaseAckQueue>()) {
        pollfds_[CONN_INDEX].events = POLLIN;
        pollfds_[CONN_INDEX].revents = 0;
        pollfds_[EVENT_INDEX].events = POLLIN;
        pollfds_[EVENT_INDEX].revents = 0;
        pollfds_[EVENT_INDEX].fd = ack_queue_->wakeFd();
        pollfds_[CONN_INDEX].fd = -1;
    }
    ~UnixFdpassClient() {
        closeConnection();
    }
    void interrupt() {
        ack_queue_->interrupt();
    }
    void releaseFrame(uint64_t generation, uint64_t frame_count) {
        ack_queue_->enqueue(generation, frame_count);
    }
    std::shared_ptr<DmabufReleaseAckQueue> releaseQueue() const {
        return ack_queue_;
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
        connection_generation_ = ack_queue_->beginConnection();
        pollfds_[CONN_INDEX].fd = conn_fd_;
        connecting_ = (rc < 0); // EINPROGRESS
        pollfds_[CONN_INDEX].events = connecting_ ? (POLLIN | POLLOUT) : POLLIN;
        return true;
    }
    bool recvTexInfoAndFD(TexInfo &info, TraceTexInfo *trace_info,
                          int &received_fd, uint64_t &connection_generation) {
        received_fd = -1;
        connection_generation = 0;
        // Prepare to receive payload + a single FD via SCM_RIGHTS
        char control[CMSG_SPACE(sizeof(int))];
        struct iovec iov;
        iov.iov_base = trace_info ? static_cast<void*>(trace_info)
                                  : static_cast<void*>(&info);
        iov.iov_len = trace_info ? sizeof(*trace_info) : sizeof(info);
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
            pollfds_[CONN_INDEX].events = connecting_ ? (POLLIN | POLLOUT) : POLLIN;
            if (!connecting_ && ack_queue_->hasPending()) {
                pollfds_[CONN_INDEX].events |= POLLOUT;
            }
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
                ack_queue_->drainWakeFd();
                if (ack_queue_->interrupted()) return false;
            }
            if (conn_fd_ < 0) {
                continue;
            }
            // Finish connect if it was in progress
            if (connecting_ && (pollfds_[CONN_INDEX].revents & (POLLOUT | POLLIN | POLLERR | POLLHUP | POLLNVAL))) {
                int soerr = 0; socklen_t slen = sizeof(soerr);
                if (getsockopt(conn_fd_, SOL_SOCKET, SO_ERROR, &soerr, &slen) < 0 || soerr != 0) {
                    closeConnection();
                    continue;
                }
                connecting_ = false;
                pollfds_[CONN_INDEX].events = POLLIN;
            }
            if (!connecting_ && !ack_queue_->flush(conn_fd_, connection_generation_)) {
                closeConnection();
                continue;
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
                    closeConnection();
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
                    closeConnection();
                    return false;
                }
                if (static_cast<size_t>(r) != iov.iov_len || (msg.msg_flags & MSG_TRUNC)) {
                    logstream << "fdpass: DMA-BUF protocol mismatch (received=" << r
                              << " expected=" << iov.iov_len << ")";
                    close(received_fd);
                    received_fd = -1;
                    closeConnection();
                    return false;
                }
                if (trace_info) info = trace_info->base;
                connection_generation = connection_generation_;
                return true;
            }
            if (pollfds_[CONN_INDEX].revents & (POLLHUP | POLLERR | POLLNVAL)) {
                closeConnection();
            }
        }
    }
};

struct DmabufFrameOwner {
    AVDRMFrameDescriptor descriptor{};
    std::shared_ptr<DmabufReleaseAckQueue> ack_queue;
    uint64_t connection_generation = 0;
    uint64_t frame_count = 0;
};

static void releaseDmabufFrameOwner(DmabufFrameOwner *owner) {
    if (!owner) return;
    for (int i = 0; i < owner->descriptor.nb_objects; ++i) {
        if (owner->descriptor.objects[i].fd >= 0) {
            ::close(owner->descriptor.objects[i].fd);
            owner->descriptor.objects[i].fd = -1;
        }
    }
    if (owner->ack_queue) {
        owner->ack_queue->enqueue(owner->connection_generation, owner->frame_count);
    }
    delete owner;
}

class IPCDMABUFSource: public NodeSingleOutput<av::VideoFrame>,
                       public IStoppable,
                       public ReportsFinishByFlag,
                       public IVideoFormatSource,
                       public IFrameRateSource,
                       public ITimeBaseSource {
protected:
    static constexpr uint32_t PIX_FMT_RGBA = ('R' << 24 | 'G' << 16 | 'B' << 8 | 'A');
    static constexpr uint32_t PIX_FMT_BGRA = ('B' << 24 | 'G' << 16 | 'R' << 8 | 'A');
    static constexpr uint32_t BYTES_PER_PIXEL = 4;
    static constexpr uint32_t MAX_DIMENSION = 16384;

    UnixFdpassClient receiver_;
    int width_ = 0;
    int height_ = 0;
    av::Rational frame_rate_{0, 1};
    std::shared_ptr<HWAccelDevice> hwaccel_;
    AVBufferRef* hw_frames_ctx_ = nullptr;
    bool trace_protocol_ = false;
    std::string trace_metadata_key_ = "scoreboard_dmabuf_trace_v1";

    static AVPixelFormat swFormatFromTexInfo(uint32_t pixel_format) {
        switch (pixel_format) {
            case PIX_FMT_RGBA: return AV_PIX_FMT_RGBA;
            case PIX_FMT_BGRA: return AV_PIX_FMT_BGRA;
            default: return AV_PIX_FMT_NONE;
        }
    }

    static uint32_t drmFormatFromTexInfo(uint32_t pixel_format) {
        switch (pixel_format) {
            case PIX_FMT_RGBA: return DRM_FORMAT_ABGR8888;
            case PIX_FMT_BGRA: return DRM_FORMAT_ARGB8888;
            default: return 0;
        }
    }

    static bool validateTexInfo(const TexInfo &ti, uint64_t &object_size) {
        const AVPixelFormat sw_format = swFormatFromTexInfo(ti.pixel_format);
        if (sw_format == AV_PIX_FMT_NONE) {
            logstream << "ipc_dmabuf_source: unsupported pixel format " << ti.pixel_format;
            return false;
        }
        if (ti.width == 0 || ti.height == 0 || ti.width > MAX_DIMENSION || ti.height > MAX_DIMENSION) {
            logstream << "ipc_dmabuf_source: invalid dimensions " << ti.width << "x" << ti.height;
            return false;
        }
        const uint64_t min_stride = (uint64_t)ti.width * BYTES_PER_PIXEL;
        if (ti.stride < min_stride) {
            logstream << "ipc_dmabuf_source: invalid stride " << ti.stride
                      << " for width " << ti.width;
            return false;
        }
        if (ti.height > 0 && ti.stride > (UINT64_MAX - ti.offset) / ti.height) {
            logstream << "ipc_dmabuf_source: DMA-BUF size overflow";
            return false;
        }
        object_size = ti.offset + (uint64_t)ti.stride * ti.height;
        if (object_size == 0) {
            logstream << "ipc_dmabuf_source: empty DMA-BUF object";
            return false;
        }
        return true;
    }
public:
    using NodeSingleOutput::NodeSingleOutput;
    IPCDMABUFSource(std::unique_ptr<SinkType> &&sink, const std::string &sock_path):
        NodeSingleOutput(std::move(sink)), receiver_(sock_path) {
    }
    virtual int width() { return width_; }
    virtual int height() { return height_; }
    virtual av::PixelFormat pixelFormat() { return av::PixelFormat(AV_PIX_FMT_DRM_PRIME); }
    virtual av::Rational frameRate() { return frame_rate_; }
    virtual av::Rational timeBase() { return {1, 1000000}; }
    virtual void stop() {
        receiver_.interrupt();
        this->finished_ = true;
    }
    virtual void process() {
        TexInfo ti{};
        TraceTexInfo trace_info{};
        int dmabuf_fd = -1;
        uint64_t connection_generation = 0;
        if (!receiver_.recvTexInfoAndFD(
                ti, trace_protocol_ ? &trace_info : nullptr,
                dmabuf_fd, connection_generation)) {
            wallclock.sleepms(5);
            return;
        }
        const uint64_t receipt_ns = monotonicNs();
        if (trace_protocol_ &&
            (trace_info.magic != SCOREBOARD_TRACE_MAGIC ||
             trace_info.version != SCOREBOARD_TRACE_VERSION ||
             trace_info.header_bytes != sizeof(TraceTexInfo))) {
            close(dmabuf_fd);
            receiver_.releaseFrame(connection_generation, ti.frame_count);
            throw Error("ipc_dmabuf_source: scoreboard trace protocol mismatch");
        }

        uint64_t object_size = 0;
        if (!validateTexInfo(ti, object_size)) {
            close(dmabuf_fd);
            receiver_.releaseFrame(connection_generation, ti.frame_count);
            return;
        }

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
                    frmctx->sw_format = swFormatFromTexInfo(ti.pixel_format);
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

        DmabufFrameOwner *owner = new (std::nothrow) DmabufFrameOwner;
        if (!owner) {
            if (dmabuf_fd >= 0) close(dmabuf_fd);
            receiver_.releaseFrame(connection_generation, ti.frame_count);
            logstream << "allocation failed for DRM descriptor";
            return;
        }
        owner->ack_queue = receiver_.releaseQueue();
        owner->connection_generation = connection_generation;
        owner->frame_count = ti.frame_count;
        AVDRMFrameDescriptor *desc = &owner->descriptor;

        desc->nb_objects = 1;
        desc->objects[0].fd = dmabuf_fd;
        desc->objects[0].size = object_size;
        desc->objects[0].format_modifier = ti.modifier;

        desc->nb_layers = 1;
        desc->layers[0].format = drmFormatFromTexInfo(ti.pixel_format);
        desc->layers[0].nb_planes = 1;
        desc->layers[0].planes[0].object_index = 0;
        desc->layers[0].planes[0].offset = ti.offset;
        desc->layers[0].planes[0].pitch = ti.stride;

        av::VideoFrame vfrm;
        vfrm.raw()->format = AV_PIX_FMT_DRM_PRIME;
        vfrm.raw()->width = ti.width;
        vfrm.raw()->height = ti.height;
        vfrm.setTimeBase({1, 1000000});
        vfrm.raw()->pts = ti.timestamp / 1000;

        if (trace_protocol_ && trace_info.trace_id != 0) {
            Parameters trace_metadata = {
                {"schema", "scoreboard.dmabuf.trace.v1"},
                {"trace_id", trace_info.trace_id},
                {"sequence", trace_info.sequence},
                {"source_pts_ms", trace_info.source_pts_ms},
                {"renderer_received_ns", trace_info.renderer_received_ns},
                {"raf_ns", trace_info.raf_ns},
                {"dom_applied_ns", trace_info.dom_applied_ns},
                {"paint_ns", trace_info.paint_ns},
                {"dmabuf_send_ns", trace_info.dmabuf_send_ns},
                {"receipt_ns", receipt_ns},
                {"frame_number", ti.frame_count},
            };
            av_dict_set(&vfrm.raw()->metadata, trace_metadata_key_.c_str(),
                        trace_metadata.dump().c_str(), 0);
        }

        if (hw_frames_ctx_) {
            vfrm.raw()->hw_frames_ctx = av_buffer_ref(hw_frames_ctx_);
        }

        AVBufferRef *buf = av_buffer_create(reinterpret_cast<uint8_t*>(desc), sizeof(*desc),
            [](void *opaque, uint8_t *data){
                (void)data;
                releaseDmabufFrameOwner(reinterpret_cast<DmabufFrameOwner*>(opaque));
            }, owner, 0);
        if (!buf) {
            releaseDmabufFrameOwner(owner);
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
        if (params.count("fps")) {
            r->frame_rate_ = parseRatio(params["fps"]);
        }
        if (params.count("trace_protocol")) {
            r->trace_protocol_ = params["trace_protocol"].get<bool>();
        }
        if (params.count("trace_metadata_key")) {
            r->trace_metadata_key_ = params["trace_metadata_key"].get<std::string>();
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
