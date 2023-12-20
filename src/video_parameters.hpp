#pragma once
#include <avcpp/pixelformat.h>
#include <avcpp/frame.h>
#include <libavutil/hwcontext.h>

struct VideoParameters {
    int width = -1;
    int height = -1;
    av::PixelFormat pixel_format{AV_PIX_FMT_NONE};
    av::PixelFormat real_pixel_format{AV_PIX_FMT_NONE};
    bool operator==(const VideoParameters &other) const {
        return (width==other.width) && (height==other.height) && (pixel_format==other.pixel_format) &&
            (real_pixel_format==other.real_pixel_format || real_pixel_format==AV_PIX_FMT_NONE || other.real_pixel_format==AV_PIX_FMT_NONE);
            // sometimes real pixel format isn't known, in such case ignore it
    }
    bool operator!=(const VideoParameters &other) const {
        return !( (*this)==other );
    }
    VideoParameters() {
    }
    VideoParameters(const av::VideoFrame &frame):
        width(frame.width()),
        height(frame.height()),
        pixel_format(frame.pixelFormat()),
        real_pixel_format((frame.raw() && frame.raw()->hw_frames_ctx && frame.raw()->hw_frames_ctx->data) ?
                          (((AVHWFramesContext*)(frame.raw()->hw_frames_ctx->data))->sw_format) : AV_PIX_FMT_NONE) {
    }
    av::PixelFormat realPixelFormat() const {
        return real_pixel_format!=AV_PIX_FMT_NONE ? real_pixel_format : pixel_format;
    }
};
