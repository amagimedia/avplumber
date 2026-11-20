#pragma once
#include "instance_shared.hpp"
#include "avutils.hpp"
#include <libavutil/hwcontext.h>
#include <libavutil/pixfmt.h>
#include <mutex>

class HWAccelDevice: public InstanceShared<HWAccelDevice> {
protected:
    AVBufferRef* device_ctx_;
    // Every FilterVideo node sharing this device (looked up by name, e.g. "@gpu")
    // gets the same HWAccelDevice instance. FFmpeg's CUDA filters (hwupload_cuda,
    // scale_cuda, ...) are not documented as safe for concurrent AVFilterGraph
    // alloc/config/free sharing one hw_device_ctx: two sources restarting within
    // the same instant (each source's own auto_restart is independent) raced here
    // and segfaulted during concurrent graph teardown/recreate. Recursive because
    // maybeInitFilterGraph() calls freeFilterGraph() on several of its own error
    // paths while already holding the lock.
    std::recursive_mutex graph_mutex_;
public:
    std::recursive_mutex& graphMutex() { return graph_mutex_; }
    // Adopt an initialized device reference (e.g. a stream sharing an existing CUDA context).
    explicit HWAccelDevice(AVBufferRef *device): device_ctx_(device) {}
    HWAccelDevice(Parameters &params) {
        std::string typestr = params["type"];
        AVHWDeviceType devtype = av_hwdevice_find_type_by_name(typestr.c_str());
        if (devtype==AV_HWDEVICE_TYPE_NONE) {
            throw Error("Unknown device type " + typestr);
        }
        
        std::string dev_string;
        const char* devstr = nullptr;
        if (params.count("device")) {
            dev_string = params["device"].get<std::string>();
            devstr = dev_string.c_str();
        }
        
        av::Dictionary options;
        if (params.count("options")) {
            options = parametersToDict(params["options"]);
        }
        
        if (devstr) {
            logstream << "using device: " << devstr;
        }
        int result = av_hwdevice_ctx_create(&device_ctx_, devtype, devstr, options.raw(), 0);
        if (result < 0) {
            throw Error("av_hwdevice_ctx_create failed: " + av::error2string(result));
        }
    }
    ~HWAccelDevice() {
        av_buffer_unref(&device_ctx_);
    }
    AVBufferRef* deviceContext() {
        return device_ctx_;
    }
    AVBufferRef* refDeviceContext() {
        return av_buffer_ref(device_ctx_);
    }
    AVPixelFormat hardwarePixelFormat() const {
        const AVHWDeviceContext* devctx = device_ctx_ && device_ctx_->data ? (const AVHWDeviceContext*)(device_ctx_->data) : nullptr;
        if (!devctx) return AV_PIX_FMT_NONE;
        switch (devctx->type) {
            case AV_HWDEVICE_TYPE_CUDA:   return AV_PIX_FMT_CUDA;
            case AV_HWDEVICE_TYPE_VAAPI:  return AV_PIX_FMT_VAAPI;
            case AV_HWDEVICE_TYPE_QSV:    return AV_PIX_FMT_QSV;
            case AV_HWDEVICE_TYPE_DRM:    return AV_PIX_FMT_DRM_PRIME;
            case AV_HWDEVICE_TYPE_VDPAU:  return AV_PIX_FMT_VDPAU;
            case AV_HWDEVICE_TYPE_OPENCL: return AV_PIX_FMT_OPENCL;
            default:                      return AV_PIX_FMT_NONE;
        }
    }
};
