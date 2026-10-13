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
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include <cstring>
#include <sys/stat.h>
#include <time.h>
#include <stdint.h>
#include <atomic>
#include <memory>
#include <new>
#include "../../hwaccel.hpp"
#include "DmabufFenceWait.hpp"
#include "DmabufReleaseQueue.hpp"
#include "TexInfoMessage.hpp"   // the wire format: TexInfo and its fds, documented in texinfo.h

class UnixFdpassClient {
protected:
    std::string path_;
    int conn_fd_ = -2;
    std::shared_ptr<DmabufReleaseAckQueue> ack_queue_;
    struct pollfd pollfds_[2];
    static constexpr size_t CONN_INDEX = 0;
    static constexpr size_t EVENT_INDEX = 1;
    bool connecting_ = false;
    std::atomic<bool> interrupted_{false};
    size_t return_queue_ = 0;

    void closeConnection() {
        if (conn_fd_ >= 0) ack_queue_->retire();
        conn_fd_ = -1;
        pollfds_[CONN_INDEX].fd = -1;
        pollfds_[CONN_INDEX].events = POLLIN;
        connecting_ = false;
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
        interrupted_.store(true);
        std::atomic_load(&ack_queue_)->interrupt();
    }
    /// Applies to connections made from now on (DmabufReleaseAckQueue's return queue).
    void setReturnQueue(size_t frames) { return_queue_ = frames; }
    void releaseFrame(uint64_t frame_count) {
        ack_queue_->enqueue(frame_count);
    }
    std::shared_ptr<DmabufReleaseAckQueue> releaseQueue() const {
        return std::atomic_load(&ack_queue_);
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
        std::atomic_store(&ack_queue_, std::make_shared<DmabufReleaseAckQueue>(fd, return_queue_));
        conn_fd_ = fd;
        pollfds_[EVENT_INDEX].fd = ack_queue_->wakeFd();
        pollfds_[CONN_INDEX].fd = conn_fd_;
        connecting_ = (rc < 0); // EINPROGRESS
        pollfds_[CONN_INDEX].events = connecting_ ? (POLLIN | POLLOUT) : POLLIN;
        return true;
    }
    /// Waits for a frame's acquire fence on the connection that delivered it; interrupt() ends
    /// the wait, as do the queue's own wake-ups when the receiver has been interrupted.
    FenceWaitResult waitForFence(int fence_fd, int timeout_ms) {
        const auto queue = std::atomic_load(&ack_queue_);
        return ::waitForFence(fence_fd, timeout_ms, queue->wakeFd(), [&] {
            queue->drainWakeFd();
            return interrupted_.load() || queue->interrupted();
        });
    }
    /// One frame: its header and the fds it came with, owned by the caller on success.
    bool recvFrame(TexInfoMessage &frame) {
        while (true) {
            if (interrupted_.load()) return false;
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
            if (!connecting_ && !ack_queue_->flush()) {
                closeConnection();
                continue;
            }
            if (pollfds_[CONN_INDEX].revents & (POLLIN)) {
                const TexInfoRecvStatus status = recvTexInfoMessage(conn_fd_, frame);
                if (status == TexInfoRecvStatus::Closed) {
                    // connection reset or EOF: close and wait for next
                    closeConnection();
                    continue;
                }
                if (status != TexInfoRecvStatus::Ok) {
                    // A sender this receiver does not understand (or the reverse): nothing of the
                    // message is kept. A whole header still names the frame, which goes back so
                    // the sender's buffer is not lost before the connection is dropped.
                    logstream << "fdpass: " << texInfoRecvStatusText(status) << "; closing the connection";
                    if (status != TexInfoRecvStatus::Short) ack_queue_->enqueue(frame.info.frame_count);
                    closeConnection();
                    return false;
                }
                ack_queue_->frameArrived();
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
        owner->ack_queue->enqueue(owner->frame_count);
    }
    delete owner;
}

class IPCDMABUFSource: public NodeSingleOutput<av::VideoFrame>,
                       public IStoppable,
                       public ReportsFinishByFlag,
                       public IVideoFormatSource,
                       public IFrameRateSource,
                       public ITimeBaseSource,
                       public IReturnsObjects {
protected:
    static constexpr uint32_t BYTES_PER_PIXEL = 4;
    static constexpr uint32_t MAX_DIMENSION = 16384;
    static constexpr int64_t FENCE_LOG_INTERVAL_US = 1000000;

    UnixFdpassClient receiver_;
    std::atomic<uint64_t> received_{0};
    // A frame whose acquire fence has not signalled after this long is dropped and acknowledged:
    // generous next to the one or two milliseconds a fence takes, short next to a stalled source.
    int fence_timeout_ms_ = 100;
    FenceWaitStats fence_stats_;
    int64_t last_fence_log_us_ = 0;
    int width_ = 0;
    int height_ = 0;
    av::Rational frame_rate_{0, 1};
    std::shared_ptr<HWAccelDevice> hwaccel_;
    AVBufferRef* hw_frames_ctx_ = nullptr;

    static AVPixelFormat swFormatFromTexInfo(uint32_t pixel_format) {
        switch (texinfoPixelFormat(pixel_format)) {
            case TEXINFO_PIX_FMT_RGBA: return AV_PIX_FMT_RGBA;
            case TEXINFO_PIX_FMT_BGRA: return AV_PIX_FMT_BGRA;
            default: return AV_PIX_FMT_NONE;
        }
    }

    static uint32_t drmFormatFromTexInfo(uint32_t pixel_format) {
        switch (texinfoPixelFormat(pixel_format)) {
            case TEXINFO_PIX_FMT_RGBA: return DRM_FORMAT_ABGR8888;
            case TEXINFO_PIX_FMT_BGRA: return DRM_FORMAT_ARGB8888;
            default: return 0;
        }
    }

    /// False when the frame must not be forwarded: its fence timed out, could not be waited on,
    /// or the node is stopping. Logged at most once a second; the counters tell the rest.
    bool fenceSignalled(int fence_fd) {
        const FenceWaitResult waited = receiver_.waitForFence(fence_fd, fence_timeout_ms_);
        fence_stats_.record(waited);
        if (waited.outcome == FenceWaitOutcome::Signalled) return true;
        const int64_t now_us = fenceMonotonicUs();
        if (now_us - last_fence_log_us_ >= FENCE_LOG_INTERVAL_US) {
            last_fence_log_us_ = now_us;
            logstream << "ipc_dmabuf_source: acquire fence "
                      << (waited.outcome == FenceWaitOutcome::TimedOut ? "not signalled within " + std::to_string(fence_timeout_ms_) + " ms"
                          : waited.outcome == FenceWaitOutcome::Invalid ? "cannot be waited on" : "wait interrupted")
                      << "; frame dropped (timeouts " << fence_stats_.timeouts.load()
                      << ", invalid " << fence_stats_.invalid.load() << " so far)";
        }
        return false;
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
    Parameters getObject(const std::string key) override {
        if (key != "frame_stats") throw Error("ipc_dmabuf_source: unknown object " + key);
        const auto stats = receiver_.releaseQueue()->stats();
        return {{"received", received_.load()}, {"released", stats.released},
                {"ack_sent", stats.sent}, {"ack_pending", stats.pending}, {"returns_queued", stats.queued},
                {"fenced", fence_stats_.fenced.load()}, {"unfenced", fence_stats_.plain.load()},
                {"fence_blocked", fence_stats_.blocked.load()}, {"fence_wait_us", fence_stats_.wait_us.load()},
                {"fence_wait_max_us", fence_stats_.max_wait_us.load()},
                {"fence_timeouts", fence_stats_.timeouts.load()}, {"fence_invalid", fence_stats_.invalid.load()}};
    }
    virtual void stop() {
        receiver_.interrupt();
        this->finished_ = true;
    }
    virtual void process() {
        TexInfoMessage frame;
        if (!receiver_.recvFrame(frame)) {
            wallclock.sleepms(5);
            return;
        }
        const TexInfo &ti = frame.info;

        ++received_;
        uint64_t object_size = 0;
        if (!validateTexInfo(ti, object_size)) {
            frame.closeFds();
            receiver_.releaseFrame(ti.frame_count);
            return;
        }
        // The pixels may be read once the fence has signalled; a frame without one is complete
        // on arrival. The fence fd is done with either way.
        if (frame.fence_fd >= 0) {
            const bool signalled = fenceSignalled(frame.fence_fd);
            close(frame.fence_fd);
            frame.fence_fd = -1;
            if (!signalled) {
                frame.closeFds();
                receiver_.releaseFrame(ti.frame_count);
                return;
            }
        } else {
            fence_stats_.unfenced();
        }
        const int dmabuf_fd = frame.plane_fd;

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
            receiver_.releaseFrame(ti.frame_count);
            logstream << "allocation failed for DRM descriptor";
            return;
        }
        owner->ack_queue = receiver_.releaseQueue();
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
        const int64_t return_queue = params.value("return_queue", int64_t(0));
        if (return_queue < 0 || return_queue > 64) throw Error("ipc_dmabuf_source: return_queue must be 0 to 64 frames");
        r->receiver_.setReturnQueue(static_cast<size_t>(return_queue));
        r->fence_timeout_ms_ = params.value("fence_timeout_ms", 100);
        if (r->fence_timeout_ms_ < 1 || r->fence_timeout_ms_ > 10000)
            throw Error("ipc_dmabuf_source: fence_timeout_ms must be 1 to 10000");
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
