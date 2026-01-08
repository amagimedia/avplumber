#pragma once
#include "util.hpp"
#include "avutils.hpp"
#include "instance_shared.hpp"
#include "rest_client.hpp"
#include <mutex>
#include <fstream>

// Forward declaration
struct HistoryTableEntry;

class PTSCorrectorCommon: public InstanceShared<PTSCorrectorCommon> {
protected:
    std::recursive_mutex busy_;
    av::Timestamp timeshift_ = NOTS;
    av::Rational timebase_ = {0, 1};
    bool lock_timeshift_ = false;
    av::Timestamp clk_ = NOTS;
    AVTS clk_wallclock_ = AV_NOPTS_VALUE;
    av::Timestamp next_history_report_ = NOTS;
    av::Timestamp history_reporting_interval_ = NOTS;
    av::Timestamp last_discontinuity_ = NOTS;
    av::Timestamp wallclock_offset_ = NOTS;
    std::ofstream timeshift_history_file_text_;
    std::ofstream timeshift_history_file_;
    ThreadedRESTEndpoint rest_;
    bool reporting_ = false;
    bool first_entry_ = true;
public:
    av::Timestamp start_ts_ = {10, {1,1}};
    void wallclockOffsetChanged(av::Timestamp offset);
    void reportTimeshiftChange();
    void openHistoryFileText(const std::string path);
    void openHistoryFile(const std::string path);
    void addTimebase(av::Rational tb);
    bool hasTimeshift();
    void firstTS(const av::Timestamp ts);
    void setTS(const av::Timestamp ts);
    bool hasTS();
    av::Timestamp rtcTS(const bool warn_if_empty = true);
    av::Timestamp startTS();
    av::Timestamp startTS(const av::Rational tb);
    av::Timestamp timeshift();
    bool shouldReportNow();
    void setTimeshift(const av::Timestamp ts, bool report, bool periodical_report);
    void lockTimeshift();
    av::Timestamp timeshift(const av::Rational tb);
    double timeshiftDiff(const av::Timestamp cmpto);
    void nowDiscontinuity();
    bool wasDiscontinuityRecently();
    std::unique_lock<std::recursive_mutex> getLock();
    std::recursive_mutex& mutex();
    void setReportingURL(const std::string url);
    void setReportingInterval(const av::Timestamp& interval);
    av::Rational getTimebase();
    bool isTimeshiftLocked();
    av::Timestamp getClock();
    AVTS getClockWallclock();
    av::Timestamp getWallclockOffset();
    av::Timestamp getLastDiscontinuity();
    bool isReporting();
    Parameters getStats();
};
