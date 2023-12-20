#pragma once
#include "util.hpp"
#include "avutils.hpp"
#include <avcpp/timestamp.h>

class TSEqualizer {
    // this class makes input and output PTSes equal
    // it is intended as workaround for audio filters which reset PTS making them start from 0
protected:
    bool first_ = true;
    av::Timestamp shift_ = NOTS;
    template<typename T, typename = decltype(std::declval<T>().dts())> void shiftDtsIfPossible(T &frm, const av::Timestamp shift) {
        //logstream << "shiftDtsIfPossible: real function called" << std::endl;
        if (!frm.dts()) return;
        frm.setDts(addTS(frm.dts(), shift));
    }
    template<typename T, typename...Args> void shiftDtsIfPossible(Args...) {
        //logstream << "shiftDtsIfPossible: NOOP called" << std::endl;
    }
public:
    template<typename T> void in(T &frm) {
        if (first_ && shift_.isNoPts()) {
            shift_ = frm.pts(); // use shift_ to store input PTS
        }
    }
    template<typename T> void out(T &frm) {
        if (frm.pts().isNoPts() || shift_.isNoPts()) return;
        if (first_) {
            av::Timestamp in_ts = shift_;
            shift_ = addTS(in_ts, negateTS(frm.pts()));
            first_ = false;
            logstream << "EQ: " << in_ts << " -> " << frm.pts() << ", new shift: " << shift_ << std::endl;
        }
        frm.setPts(addTS(frm.pts(), shift_));
        shiftDtsIfPossible<T>(frm, shift_);
    }
    void reset() {
        first_ = true;
    }
};

