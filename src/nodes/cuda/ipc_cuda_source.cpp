#include "../node_common.hpp"
#include "../../cuda.hpp"
#include "../../hwaccel.hpp"
#include "../../InterruptibleReader.hpp"

extern "C" {
#include "libavutil/buffer.h"
#include "libavutil/hwcontext.h"
#include "libavutil/hwcontext_cuda.h"
}

static constexpr size_t MAX_FRAME_WIDTH = 1920;
static constexpr size_t MAX_FRAME_HEIGHT = 1080;
static constexpr size_t MAX_FRAME_SIZE = MAX_FRAME_WIDTH * MAX_FRAME_HEIGHT * 3;

struct Packet {
    int64_t magic_number;

    CUipcMemHandle ipc_handle;
    uint64_t ipc_offset;
    uint32_t frame_width, frame_height;
    struct timespec timestamp; // 1/1e9 timebase

    uint32_t stride_y;
    uint32_t stride_u;
    uint32_t stride_v;
};

class IPCCUDASource: public NodeSingleOutput<av::VideoFrame>, public IStoppable, public ReportsFinishByFlag {
protected:
    InterruptibleReader pipe_reader_;
    CUdeviceptr ipc_memory_ {0};
    CUipcMemHandle cur_ipc_handle_ {0};
    AVCUDADeviceContext* cuda_dev_ctx_ = nullptr;
    int width_ = 0;
    int height_ = 0;
    std::shared_ptr<HWAccelDevice> hwaccel_;
    AVBufferRef* hw_frames_ctx_ = nullptr;
public:
    using NodeSingleOutput::NodeSingleOutput;
    virtual void process() {
        Packet packet;
        if (!pipe_reader_.read(&packet, sizeof(Packet))) {
            logstream << "failed to read packet from pipe (maybe node was stopped)";
            wallclock.sleepms(500);
            return;
        }
        if (packet.magic_number != 0x12345678abcdef01) {
            logstream << "invalid magic number, ignoring packet";
            return;
        }
        int cuda_error = 0;
        cuda_error |= CHECK_CU(cuCtxPushCurrent(cuda_dev_ctx_->cuda_ctx));
        if (cuda_error) {
            logstream << "cuCtxPushCurrent failed";
            return;
        }
        if (ipc_memory_ && (memcmp(&cur_ipc_handle_, &packet.ipc_handle, sizeof(CUipcMemHandle)) != 0)) {
            logstream << "cuda ipc handle changed, closing";
            CHECK_CU(cuIpcCloseMemHandle(ipc_memory_));
            ipc_memory_ = 0;
            cur_ipc_handle_ = {0};
        }
        if (!ipc_memory_) {
            cuda_error |= CHECK_CU(cuIpcOpenMemHandle(&ipc_memory_, packet.ipc_handle, CU_IPC_MEM_LAZY_ENABLE_PEER_ACCESS));
            if (cuda_error) {
                logstream << "failed to open ipc handle";
                return;
            }
            cur_ipc_handle_ = packet.ipc_handle;
            logstream << "opened ipc mem handle";
        }

        if (packet.frame_width != width_ || packet.frame_height != height_) {
            if (hw_frames_ctx_) {
                av_buffer_unref(&hw_frames_ctx_);
            }
            if (packet.frame_width > 0 && packet.frame_height > 0) {
                hw_frames_ctx_ = av_hwframe_ctx_alloc(hwaccel_->deviceContext());
                AVHWFramesContext *frmctx = (AVHWFramesContext *)(hw_frames_ctx_->data);
                frmctx->sw_format = AV_PIX_FMT_YUV420P; // hardcoded in webrtc-ndi-proxy
                frmctx->width = packet.frame_width;
                frmctx->height = packet.frame_height;
                while ((frmctx->height%4) != 0) frmctx->height++;
                /*frmctx->width = 3840;
                frmctx->height = 2160;*/
                frmctx->format = AV_PIX_FMT_CUDA;
                logstream << "set frmctx to " << frmctx->width << "x" << frmctx->height;
                int r = av_hwframe_ctx_init(hw_frames_ctx_);
                if (r != 0) {
                    logstream << "av_hwframe_ctx_init failed: " + av::error2string(r);
                    return;
                }
            }
            width_ = packet.frame_width;
            height_ = packet.frame_height;
            logstream << "new buffer dimensions: " << width_ << "x" << height_;
        }

        av::VideoFrame vfrm;

        vfrm.setTimeBase({1, 1000000});
        vfrm.raw()->pts = packet.timestamp.tv_sec * 1000000 + packet.timestamp.tv_nsec / 1000;

        av_hwframe_get_buffer(hw_frames_ctx_, vfrm.raw(), 0);
        
        //CUdeviceptr dstptr = reinterpret_cast<uint64_t>(vfrm.raw()->data[0]);
        //size_t frame_size = MAX_FRAME_SIZE;
        //logstream << "src " << ipc_memory_ << " dst " << dstptr;

        /*vfrm.raw()->linesize[0] = packet.stride_y;
        vfrm.raw()->linesize[1] = packet.stride_u;
        vfrm.raw()->linesize[2] = packet.stride_v;*/

        //CUstream stream;
        //cuStreamCreate(&stream, 0);

        auto copymem = [&](int datai, uint64_t ptradd, size_t width_in_bytes, size_t height, size_t stride) {
            CUDA_MEMCPY2D cpy = {
                .srcMemoryType = CU_MEMORYTYPE_DEVICE,
                .srcDevice = ipc_memory_ + ptradd,
                .srcPitch = stride,

                .dstMemoryType = CU_MEMORYTYPE_DEVICE,
                .dstDevice = reinterpret_cast<uint64_t>(vfrm.raw()->data[datai]),
                .dstPitch = size_t(vfrm.raw()->linesize[datai]),
                .WidthInBytes = width_in_bytes,
                .Height = height
            };
            cuda_error |= CHECK_CU(cuMemcpy2DAsync(&cpy, cuda_dev_ctx_->stream));
            if (cuda_error) {
                logstream << "cuMemcpy2D failed";
                cuda_error = 0;
            }
        };
        uint64_t shift = packet.ipc_offset;
        copymem(0, shift, packet.frame_width, packet.frame_height, packet.stride_y);
        shift += packet.stride_y * packet.frame_height;
        copymem(1, shift, packet.frame_width/2, packet.frame_height/2, packet.stride_u);
        shift += packet.stride_u * packet.frame_height / 2;
        copymem(2, shift, packet.frame_width/2, packet.frame_height/2, packet.stride_v);

        cuStreamSynchronize(cuda_dev_ctx_->stream);
        //cuStreamSynchronize(stream);
        //cuStreamDestroy(stream);

        //vfrm.raw()->data[1] = vfrm.raw()->data[0] + packet.frame_width * packet.frame_height;
        //vfrm.raw()->data[2] = vfrm.raw()->data[1] + (packet.frame_width * packet.frame_height) / 4;

        vfrm.raw()->width = packet.frame_width;
        vfrm.raw()->height = packet.frame_height;
        vfrm.setComplete(true);
        this->sink_->put(vfrm);
        
        CUcontext dummy;
        cuda_error |= CHECK_CU(cuCtxPopCurrent(&dummy));
    }
    virtual void stop() {
        pipe_reader_.interrupt();
        this->finished_ = true;
    }
    IPCCUDASource(std::unique_ptr<SinkType> &&sink, const std::string pipe_path):
        NodeSingleOutput(std::move(sink)), pipe_reader_(pipe_path) {
    }
    ~IPCCUDASource() {
        if (hw_frames_ctx_) {
            av_buffer_unref(&hw_frames_ctx_);
        }
        if (ipc_memory_) {
            CHECK_CU(cuIpcCloseMemHandle(ipc_memory_));
        }
    }
    static std::shared_ptr<IPCCUDASource> create(NodeCreationInfo &nci) {
        EdgeManager &edges = nci.edges;
        const Parameters &params = nci.params;
        std::shared_ptr<Edge<av::VideoFrame>> edge = edges.find<av::VideoFrame>(params["dst"]);
        auto r = std::make_shared<IPCCUDASource>(make_unique<EdgeSink<av::VideoFrame>>(edge), params["pipe"]);
        r->hwaccel_ = InstanceSharedObjects<HWAccelDevice>::get(nci.instance, params["hwaccel"]);
        AVHWDeviceContext* devctx = (AVHWDeviceContext *)(r->hwaccel_->deviceContext()->data);
        r->cuda_dev_ctx_ = (AVCUDADeviceContext*)(devctx->hwctx);
        return r;
    }
};

DECLNODE(ipc_cuda_source, IPCCUDASource);
