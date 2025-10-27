#pragma once
#include "instance_shared.hpp"
#include "avutils.hpp"
#include <libavutil/hwcontext.h>

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
        
        const char* devstr = nullptr;
        if (params.count("device")) {
            devstr = params["device"].get<std::string>().c_str();
        }
        
        av::Dictionary options;
        if (params.count("options")) {
            options = parametersToDict(params["options"]);
        }
        
        int result = av_hwdevice_ctx_create(&device_ctx_, devtype, devstr, options.raw(), 0);
        if (result<0) {
            throw Error("av_hwdevice_ctx_create failed");
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
};
