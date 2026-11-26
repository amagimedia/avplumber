// One local viewer process per socket. With no reader the node only consumes
// its input. Each accepted connection owns a fresh texture pool, so images a
// crashed reader still holds can never be overwritten on reconnection.
#include "../node_common.hpp"
#include "../../cuda.hpp"
#include "CudaEglFrameRenderer.hpp"
#include "../../hwaccel/CudaEglImagePool.hpp"
#include "UnixPacketSocket.hpp"
#include "../../mixer/primitives/frame_subscription.hpp"
#include "../../../deps/dma-browser/addons/fdpass/texinfo.h"
#include "../../../deps/dma-browser/addons/fdpass/dmabuf_ack.h"
#include <libdrm/drm_fourcc.h>
#include <chrono>
#include <unordered_map>

class IpcDmabufSink: public NodeSingleInput<av::VideoFrame>, public ReportsFinishByFlag {
    using Fd = avp::ipc::Fd;
    avp::ipc::PacketListener listener_;
    Fd peer_;
    unsigned slots_;
    std::shared_ptr<avp::mixer::FrameSubscription> subscription_;
    std::shared_ptr<CudaEglImagePool> pool_;
    std::unique_ptr<CudaEglFrameRenderer> renderer_;
    std::unordered_map<uint64_t, EglImageFrame> held_;
    uint64_t sequence_ = 0;
    PFNEGLEXPORTDMABUFIMAGEQUERYMESAPROC query_ = nullptr;
    PFNEGLEXPORTDMABUFIMAGEMESAPROC export_ = nullptr;
    std::chrono::steady_clock::time_point retry_{};

    void disconnect() {
        if (subscription_) subscription_->enable(false);
        peer_ = Fd();
        held_.clear();
        renderer_.reset();
        pool_.reset();
    }

    void readReleases() {
        uint8_t ack[DMABUF_RELEASE_ACK_BYTES];
        for (unsigned i = 0; i < slots_; ++i) {
            const auto bytes = recv(peer_.get(), ack, sizeof(ack), MSG_TRUNC | MSG_DONTWAIT);
            if (bytes < 0 && (errno == EAGAIN || errno == EWOULDBLOCK || errno == EINTR)) return;
            uint64_t sequence;
            DmabufAckKind kind;
            if (bytes != sizeof(ack) || !dmabufDecodeReleaseAck(ack, sequence, kind) ||
                kind != DmabufAckKind::Frame || !held_.erase(sequence)) {
                disconnect();
                return;
            }
        }
    }

    bool publish(const EglImageFrame &frame) {
        EGLDisplay display = eglGetCurrentDisplay();
        int format = 0, planes = 0;
        EGLuint64KHR modifier = 0;
        if (!query_ || !export_ || !query_(display, frame.image(), &format, &planes, &modifier) || planes != 1)
            throw std::runtime_error("EGL driver cannot export this texture as a single-plane DMA-BUF");
        const bool rgba = format == DRM_FORMAT_ABGR8888 || format == DRM_FORMAT_XBGR8888;
        if (!rgba && format != DRM_FORMAT_ARGB8888 && format != DRM_FORMAT_XRGB8888)
            throw std::runtime_error("unsupported exported DRM format");
        int plane = -1, stride = 0, offset = 0;
        if (!export_(display, frame.image(), &plane, &stride, &offset))
            throw std::runtime_error("EGL DMA-BUF export failed");
        Fd fd(plane);
        if (!fd || stride < frame.width() * 4 || offset < 0) throw std::runtime_error("invalid exported plane");
        const auto now = std::chrono::steady_clock::now().time_since_epoch();
        TexInfo info{uint32_t(frame.width()), uint32_t(frame.height()), uint32_t(stride),
                     rgba ? TEXINFO_PIX_FMT_RGBA : TEXINFO_PIX_FMT_BGRA, modifier, uint64_t(offset),
                     uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(now).count()), ++sequence_};
        iovec io{&info, sizeof(info)};
        alignas(cmsghdr) char control[CMSG_SPACE(sizeof(int))]{};
        msghdr msg{};
        msg.msg_iov = &io;
        msg.msg_iovlen = 1;
        msg.msg_control = control;
        msg.msg_controllen = sizeof(control);
        auto *header = CMSG_FIRSTHDR(&msg);
        header->cmsg_level = SOL_SOCKET;
        header->cmsg_type = SCM_RIGHTS;
        header->cmsg_len = CMSG_LEN(sizeof(plane));
        std::memcpy(CMSG_DATA(header), &plane, sizeof(plane));
        const auto sent = sendmsg(peer_.get(), &msg, MSG_NOSIGNAL | MSG_DONTWAIT);
        if (sent < 0 && (errno == EAGAIN || errno == EWOULDBLOCK || errno == EINTR)) return false;
        if (sent != sizeof(info)) { disconnect(); return false; }
        return true;
    }

public:
    IpcDmabufSink(std::unique_ptr<SourceType> input, std::string path, unsigned slots, mode_t mode)
        : NodeSingleInput(std::move(input)), listener_(std::move(path), mode), slots_(slots) {
        query_ = reinterpret_cast<PFNEGLEXPORTDMABUFIMAGEQUERYMESAPROC>(eglGetProcAddress("eglExportDMABUFImageQueryMESA"));
        export_ = reinterpret_cast<PFNEGLEXPORTDMABUFIMAGEMESAPROC>(eglGetProcAddress("eglExportDMABUFImageMESA"));
    }
    ~IpcDmabufSink() override { if (subscription_) subscription_->close(); }
    void process() override {
        if (peer_) readReleases();
        if (!peer_ && std::chrono::steady_clock::now() >= retry_) peer_ = listener_.acceptPeer();
        if (peer_ && !renderer_) {
            if (subscription_) subscription_->enable(true);
            pool_ = std::make_shared<CudaEglImagePool>();
            renderer_ = std::make_unique<CudaEglFrameRenderer>(pool_, slots_, slots_, 0, CudaEglFrameRenderer::Sync::PrivateStream);
        }
        av::VideoFrame frame;
        if (!source_->tryGet(frame, peer_ ? 10 : 100)) return;
        if (isEofMarker(frame)) { finished_ = true; return; }
        if (!peer_ || !frame || !frame.isComplete() || held_.size() >= slots_) return;
        if (pool_->size() && !pool_->initializedFor(frame.width(), frame.height())) { disconnect(); return; }
        EglImageFrame image;
        if (frame.raw()->format != AV_PIX_FMT_CUDA || !frame.raw()->hw_frames_ctx) return;
        try {
            const auto result = renderer_->render(frame, image);
            if (result == CudaEglFrameRenderer::Result::Invalid) throw std::runtime_error("CUDA to EGL conversion failed");
            if (result == CudaEglFrameRenderer::Result::Ready && publish(image)) held_.emplace(sequence_, std::move(image));
        } catch (const std::exception &err) {
            logstream << "ipc_dmabuf_sink: " << err.what();
            disconnect();
            retry_ = std::chrono::steady_clock::now() + std::chrono::seconds(2);
        }
    }
    static std::shared_ptr<IpcDmabufSink> create(NodeCreationInfo &nci) {
        if (global_cuda.has_errors) throw Error("ipc_dmabuf_sink: CUDA not initialized");
        const int slots = nci.params.value("pool_size", 4);
        if (slots < 2 || slots > 16) throw Error("ipc_dmabuf_sink: pool_size must be 2..16");
        const int mode = nci.params.value("socket_mode", 0600);
        if (mode < 0 || mode > 0777) throw Error("ipc_dmabuf_sink: invalid socket_mode");
        auto sink = std::make_shared<IpcDmabufSink>(make_unique<EdgeSource<av::VideoFrame>>(
            nci.edges.find<av::VideoFrame>(nci.params["src"])), nci.params.at("socket").get<std::string>(), slots, mode);
        if (nci.params.contains("subscription")) {
            sink->subscription_ = InstanceSharedObjects<avp::mixer::FrameSubscription>::get(
                nci.instance, nci.params.at("subscription").get<std::string>());
            sink->subscription_->configure({1, 1}, false);
        }
        return sink;
    }
};

DECLNODE(ipc_dmabuf_sink, IpcDmabufSink)
