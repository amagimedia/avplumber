#pragma once
#include "instance_shared.hpp"
#include <avcpp/frame.h>

class PictureBuffer: public InstanceShared<PictureBuffer> {
protected:
    av::VideoFrame frame_;
public:
    PictureBuffer(av::VideoFrame frm): frame_(frm) {
        logstream << "Creating picture buffer with video frame " << frm.width() << "x" << frm.height() << " " << frm.pixelFormat();
    }
    av::VideoFrame getFrame() {
        return frame_;
    }
};