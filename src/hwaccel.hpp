#pragma once
#include "instance_shared.hpp"
#include "avutils.hpp"
#include <libavutil/hwcontext.h>
#include <libavutil/pixfmt.h>

class HWAccelDevice: public InstanceShared<HWAccelDevice> {
protected:
    AVBufferRef* device_ctx_;
public:
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
