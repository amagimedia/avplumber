#include "../../node_common.hpp"

extern "C" {
#include <libavutil/dict.h>
}

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <deque>
#include <limits>
#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

namespace {

constexpr size_t kMaxDerivOrder = 8;

static bool finitePos(double x) { return std::isfinite(x); }

static std::vector<double> parseDerivativeLimits(const Parameters &p) {
    std::vector<double> lim(kMaxDerivOrder, std::numeric_limits<double>::infinity());
    auto apply = [&lim](int order, double L) {
        if (order < 1 || order > (int)kMaxDerivOrder || !(L > 0.0) || !std::isfinite(L))
            return;
        size_t idx = (size_t)order - 1;
        lim[idx] = std::min(lim[idx], L);
    };
    if (p.contains("derivative_limits") && p["derivative_limits"].is_array()) {
        for (const auto &item : p["derivative_limits"]) {
            if (!item.is_object()) continue;
            int ord = item.value("order", 0);
            double L = NAN;
            if (item.contains("limit"))
                L = item["limit"].get<double>();
            else if (item.contains("limit_px_per_sn"))
                L = item["limit_px_per_sn"].get<double>();
            apply(ord, L);
        }
    }
    if (p.contains("max_velocity_px_per_s"))
        apply(1, p["max_velocity_px_per_s"].get<double>());
    if (p.contains("max_acceleration_px_per_s2"))
        apply(2, p["max_acceleration_px_per_s2"].get<double>());
    if (p.contains("max_jerk_px_per_s3"))
        apply(3, p["max_jerk_px_per_s3"].get<double>());
    if (p.contains("max_snap_px_per_s4"))
        apply(4, p["max_snap_px_per_s4"].get<double>());
    return lim;
}

struct DetectionBox {
    int cls = -1;
    std::string label;
    bool has_label = false;
    double conf = 0.0;
    double x1 = 0.0, y1 = 0.0, x2 = 0.0, y2 = 0.0;
    int model_index = -1;
};

static double centerX(const DetectionBox &b) { return (b.x1 + b.x2) * 0.5; }
static double centerY(const DetectionBox &b) { return (b.y1 + b.y2) * 0.5; }

// Crop center (cx, cy) so fixed viewport [cx±half_w, cy±half_h] contains [ux1,ux2]×[uy1,uy2].
// If the union is wider/taller than the viewport, use union centroid clamped to valid crop centers.
static void centerForFixedViewportContain(double ux1, double uy1, double ux2, double uy2, int fw, int fh, double half_w,
                                          double half_h, double &cx, double &cy) {
    double cx_lo = std::max(ux2 - half_w, half_w);
    double cx_hi = std::min(ux1 + half_w, (double)fw - half_w);
    if (cx_lo <= cx_hi)
        cx = 0.5 * (cx_lo + cx_hi);
    else {
        const double mid = 0.5 * (ux1 + ux2);
        cx = std::max(half_w, std::min((double)fw - half_w, mid));
    }
    double cy_lo = std::max(uy2 - half_h, half_h);
    double cy_hi = std::min(uy1 + half_h, (double)fh - half_h);
    if (cy_lo <= cy_hi)
        cy = 0.5 * (cy_lo + cy_hi);
    else {
        const double mid = 0.5 * (uy1 + uy2);
        cy = std::max(half_h, std::min((double)fh - half_h, mid));
    }
}

// RBJ lowpass; returns coeffs with a0 normalized to 1.
static void rbjLowpass(double w0, double Q, double &b0, double &b1, double &b2, double &a1, double &a2) {
    const double cosw0 = std::cos(w0);
    const double sinw0 = std::sin(w0);
    const double alpha = sinw0 / (2.0 * Q);
    const double a0r = 1.0 + alpha;
    b0 = ((1.0 - cosw0) * 0.5) / a0r;
    b1 = (1.0 - cosw0) / a0r;
    b2 = ((1.0 - cosw0) * 0.5) / a0r;
    a1 = (-2.0 * cosw0) / a0r;
    a2 = (1.0 - alpha) / a0r;
}

// Butterworth Q per biquad section (order = 2 * num_sections).
static const double kButterQ2[1] = {0.7071067811865476};
static const double kButterQ4[2] = {0.541196100146197, 1.3065629648763764};
static const double kButterQ6[3] = {0.5176380902050417, 0.7071067811865476, 1.931851652706137};
static const double kButterQ8[4] = {0.5097955791041592, 0.6013448869350473, 0.8999762231364981, 2.562915447741505};

struct BiquadDF1 {
    double b0 = 1, b1 = 0, b2 = 0, a1 = 0, a2 = 0;
    double x1 = 0, x2 = 0, y1 = 0, y2 = 0;
    void setCoeffs(double tb0, double tb1, double tb2, double ta1, double ta2) {
        b0 = tb0;
        b1 = tb1;
        b2 = tb2;
        a1 = ta1;
        a2 = ta2;
    }
    void reset() {
        x1 = x2 = y1 = y2 = 0;
    }
    double process(double x) {
        const double y = b0 * x + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2;
        x2 = x1;
        x1 = x;
        y2 = y1;
        y1 = y;
        return y;
    }
};

class LowpassBackend {
public:
    virtual ~LowpassBackend() = default;
    virtual void reset() = 0;
    virtual void step(double mx, double my, bool meas_valid, double dt, int frame_w, int frame_h,
                      bool lost_use_frame_center, double &ox, double &oy) = 0;
};

// Kalman1D is now in a shared header
#include "../../../kalman1d.hpp"

class LowpassKalman2 : public LowpassBackend {
    Kalman1D kx_, ky_;
public:
    LowpassKalman2(double q_pos, double q_vel, double r_meas) : kx_(q_pos, q_vel, r_meas), ky_(q_pos, q_vel, r_meas) {}
    void reset() override {
        kx_.reset();
        ky_.reset();
    }
    void step(double mx, double my, bool meas_valid, double dt, int frame_w, int frame_h,
              bool lost_use_frame_center, double &ox, double &oy) override {
        if (dt <= 0 || !std::isfinite(dt))
            dt = 1.0 / 30.0;
        kx_.step(mx, meas_valid, dt, lost_use_frame_center, frame_w);
        ky_.step(my, meas_valid, dt, lost_use_frame_center, frame_h);
        ox = kx_.pos();
        oy = ky_.pos();
    }
};

class LowpassIIR : public LowpassBackend {
    std::vector<BiquadDF1> sx_, sy_;
    double last_x_ = 0, last_y_ = 0;
    bool have_ = false;

public:
    LowpassIIR(double fc_hz, double fs_hz, const double *Qs, size_t nsec) {
        const double w0 = 2.0 * 3.14159265358979323846 * std::min(fc_hz, fs_hz * 0.49) / fs_hz;
        sx_.resize(nsec);
        sy_.resize(nsec);
        for (size_t i = 0; i < nsec; ++i) {
            double b0, b1, b2, a1, a2;
            rbjLowpass(w0, Qs[i], b0, b1, b2, a1, a2);
            sx_[i].setCoeffs(b0, b1, b2, a1, a2);
            sy_[i].setCoeffs(b0, b1, b2, a1, a2);
        }
    }
    void reset() override {
        for (auto &b : sx_)
            b.reset();
        for (auto &b : sy_)
            b.reset();
        last_x_ = last_y_ = 0;
        have_ = false;
    }
    void step(double mx, double my, bool meas_valid, double dt, int frame_w, int frame_h,
              bool lost_use_frame_center, double &ox, double &oy) override {
        (void)dt;
        double ix = mx, iy = my;
        if (!meas_valid) {
            if (lost_use_frame_center) {
                ix = frame_w * 0.5;
                iy = frame_h * 0.5;
            } else if (have_) {
                ix = last_x_;
                iy = last_y_;
            } else {
                ix = frame_w * 0.5;
                iy = frame_h * 0.5;
            }
        }
        double tx = ix, ty = iy;
        for (auto &b : sx_)
            tx = b.process(tx);
        for (auto &b : sy_)
            ty = b.process(ty);
        last_x_ = tx;
        last_y_ = ty;
        have_ = true;
        ox = tx;
        oy = ty;
    }
};

class LowpassFIR : public LowpassBackend {
    std::deque<double> qx_, qy_;
    size_t taps_;
    double last_x_ = 0, last_y_ = 0;
    bool have_ = false;

public:
    explicit LowpassFIR(size_t taps) : taps_(std::max<size_t>(1, taps)) {}
    void reset() override {
        qx_.clear();
        qy_.clear();
        have_ = false;
    }
    void step(double mx, double my, bool meas_valid, double dt, int frame_w, int frame_h,
              bool lost_use_frame_center, double &ox, double &oy) override {
        (void)dt;
        double ix = mx, iy = my;
        if (!meas_valid) {
            if (lost_use_frame_center) {
                ix = frame_w * 0.5;
                iy = frame_h * 0.5;
            } else if (have_) {
                ix = last_x_;
                iy = last_y_;
            } else {
                ix = frame_w * 0.5;
                iy = frame_h * 0.5;
            }
        }
        qx_.push_back(ix);
        qy_.push_back(iy);
        while (qx_.size() > taps_)
            qx_.pop_front();
        while (qy_.size() > taps_)
            qy_.pop_front();
        double sx = 0, sy = 0;
        for (double v : qx_)
            sx += v;
        for (double v : qy_)
            sy += v;
        const double nx = sx / (double)qx_.size();
        const double ny = sy / (double)qy_.size();
        last_x_ = nx;
        last_y_ = ny;
        have_ = true;
        ox = nx;
        oy = ny;
    }
};

// Per-axis state: d[0]=pos, d[1]=v, d[2]=a for derivative limiting (slew chain).
struct DerivSlewAxis {
    std::vector<double> d;

    void reset(double p) {
        d = {p, 0.0, 0.0, 0.0};
    }

    static bool limOk(const std::vector<double> &L, size_t idx) {
        return idx < L.size() && std::isfinite(L[idx]) && L[idx] > 0;
    }

    void step(double p_star, double dt, const std::vector<double> &L) {
        if (dt <= 0 || !std::isfinite(dt))
            dt = 1.0 / 30.0;
        if (d.size() < 4)
            d.resize(4, 0.0);
        double p = d[0], v = d[1], a = d[2], j = d[3];

        const bool limV = limOk(L, 0);
        const bool limA = limOk(L, 1);
        const bool limJ = limOk(L, 2);
        const bool limS = limOk(L, 3);

        if (!limV && !limA && !limJ && !limS) {
            d[0] = p_star;
            d[1] = d[2] = d[3] = 0;
            return;
        }

        double v_cmd = (p_star - p) / dt;
        if (limV)
            v_cmd = std::max(-L[0], std::min(L[0], v_cmd));

        if (limA || limJ || limS) {
            double a_cmd = (v_cmd - v) / dt;
            if (limA)
                a_cmd = std::max(-L[1], std::min(L[1], a_cmd));
            if (limJ || limS) {
                double j_cmd = (a_cmd - a) / dt;
                if (limJ)
                    j_cmd = std::max(-L[2], std::min(L[2], j_cmd));
                if (limS) {
                    const double snap = (j_cmd - j) / dt;
                    const double snap_c = std::max(-L[3], std::min(L[3], snap));
                    j += snap_c * dt;
                } else {
                    j = j_cmd;
                }
                if (limJ)
                    j = std::max(-L[2], std::min(L[2], j));
                a += j * dt;
                if (limA)
                    a = std::max(-L[1], std::min(L[1], a));
            } else {
                a = a_cmd;
            }
            v += a * dt;
            if (limV)
                v = std::max(-L[0], std::min(L[0], v));
            p += v * dt;
        } else {
            v = v_cmd;
            p += v * dt;
            a = j = 0;
        }

        d[0] = p;
        d[1] = v;
        d[2] = a;
        d[3] = j;
    }
};

} // namespace

class SmoothCropViewport : public NodeSISO<av::VideoFrame, av::VideoFrame> {
    std::vector<std::string> metadata_keys_in_;
    std::string metadata_key_out_;
    double model_content_width_ = 0;
    double model_content_height_ = 0;
    double model_content_offset_x_ = 0;
    double model_content_offset_y_ = 0;
    double min_conf_ = 0;
    std::unordered_set<int> allowed_classes_;
    std::unordered_set<std::string> allowed_labels_;
    std::unordered_set<int> allowed_model_indices_;
    std::vector<std::string> label_priority_;
    std::string focus_mode_;
    /// "center": focus_mode centroid / best box only. "contain": place crop so viewport_dst contains union of selected dets.
    std::string viewport_fit_;
    double viewport_contain_margin_px_ = 0;
    std::unordered_set<std::string> label_priority_set_;
    std::string lost_target_mode_;

    std::unique_ptr<LowpassBackend> lowpass_;
    double kalman_q_pos_ = 0.01, kalman_q_vel_ = 0.001, kalman_r_ = 4.0;
    double iir_fc_hz_ = 2.0;
    size_t fir_taps_ = 5;

    bool hold_enabled_ = false;
    double hold_speed_px_per_s_ = 30.0;
    int hold_min_frames_ = 6;
    double hold_break_px_ = 120.0;
    int hold_max_frames_ = 90;
    bool latched_ = false;
    double latch_x_ = 0, latch_y_ = 0;
    int low_speed_frames_ = 0;
    int latched_age_ = 0;
    double prev_meas_x_ = 0, prev_meas_y_ = 0;
    bool have_prev_meas_ = false;

    std::vector<double> derivative_limit_by_order_;
    DerivSlewAxis slew_x_, slew_y_;

    int lookahead_frames_ = 0;
    /// Require this many consecutive frames with a detection on the *output* frame before
    /// measurements drive the lowpass / slew (ignores brief false positives; loss resets streak).
    int min_visible_frames_ = 1;
    int visible_streak_ = 0;
    double viewport_marker_half_extent_ = 1.0;
    int viewport_dst_width_ = 0;
    int viewport_dst_height_ = 0;
    struct BufSlot {
        av::VideoFrame frm;
        double mx = 0, my = 0;
        bool valid = false;
    };
    std::deque<BufSlot> buffer_;

    av::Rational frame_rate_{0, 0};
    int last_w_ = 0, last_h_ = 0;
    uint64_t frame_counter_ = 0;
    int debug_log_every_n_ = 0;

    void resetPipelineState() {
        if (lowpass_)
            lowpass_->reset();
        latched_ = false;
        low_speed_frames_ = latched_age_ = 0;
        have_prev_meas_ = false;
        visible_streak_ = 0;
        buffer_.clear();
        slew_x_.reset(0);
        slew_y_.reset(0);
    }

    bool remapModelCoord(double x, double y, double model_w, double model_h, int fw, int fh, double &out_x,
                         double &out_y) const {
        if (model_content_width_ > 0.0 && model_content_height_ > 0.0) {
            const double content_x = std::max(0.0, std::min(x - model_content_offset_x_, model_content_width_));
            const double content_y = std::max(0.0, std::min(y - model_content_offset_y_, model_content_height_));
            out_x = content_x * ((double)fw / model_content_width_);
            out_y = content_y * ((double)fh / model_content_height_);
            return true;
        }
        const double sx = model_w > 0.0 ? (double)fw / model_w : 1.0;
        const double sy = model_h > 0.0 ? (double)fh / model_h : 1.0;
        out_x = x * sx;
        out_y = y * sy;
        return true;
    }

    bool detAllowed(const DetectionBox &det) const {
        if (det.conf < min_conf_)
            return false;
        if (allowed_classes_.empty() && allowed_labels_.empty() && allowed_model_indices_.empty())
            return true;
        bool ok = false;
        if (!allowed_classes_.empty())
            ok = ok || allowed_classes_.count(det.cls) > 0;
        if (!allowed_labels_.empty() && det.has_label)
            ok = ok || allowed_labels_.count(det.label) > 0;
        if (!allowed_model_indices_.empty())
            ok = ok || allowed_model_indices_.count(det.model_index) > 0;
        return ok;
    }

    bool parseMetadata(const av::VideoFrame &frm, std::vector<DetectionBox> &out) const {
        out.clear();
        const AVFrame *raw = frm.raw();
        if (!raw || !raw->metadata)
            return false;
        for (const std::string &meta_key : metadata_keys_in_) {
            AVDictionaryEntry *entry = av_dict_get(raw->metadata, meta_key.c_str(), nullptr, 0);
            if (!entry || !entry->value)
                continue;
            try {
                Parameters md = Parameters::parse(entry->value);
                if (!md.contains("detections") || !md["detections"].is_array())
                    continue;
                const std::string coord_space = md.value("coord_space", std::string("model"));
                const double model_w = md.value("model_width", (double)frm.width());
                const double model_h = md.value("model_height", (double)frm.height());
                const int fw = frm.width();
                const int fh = frm.height();

                for (const auto &item : md["detections"]) {
                    if (!item.is_object())
                        continue;
                    if (!item.contains("xyxy") || !item["xyxy"].is_array() || item["xyxy"].size() < 4)
                        continue;
                    DetectionBox det;
                    det.cls = item.value("cls", -1);
                    det.conf = item.value("conf", 0.0);
                    det.model_index = item.value("model_index", -1);
                    det.x1 = item["xyxy"][0].get<double>();
                    det.y1 = item["xyxy"][1].get<double>();
                    det.x2 = item["xyxy"][2].get<double>();
                    det.y2 = item["xyxy"][3].get<double>();
                    if (item.contains("label") && item["label"].is_string()) {
                        det.label = item["label"].get<std::string>();
                        det.has_label = true;
                    }
                    if (!detAllowed(det))
                        continue;
                    if (coord_space == "model") {
                        if (!remapModelCoord(det.x1, det.y1, model_w, model_h, fw, fh, det.x1, det.y1))
                            continue;
                        if (!remapModelCoord(det.x2, det.y2, model_w, model_h, fw, fh, det.x2, det.y2))
                            continue;
                    }
                    det.x1 = std::max(0.0, std::min((double)fw, det.x1));
                    det.x2 = std::max(0.0, std::min((double)fw, det.x2));
                    det.y1 = std::max(0.0, std::min((double)fh, det.y1));
                    det.y2 = std::max(0.0, std::min((double)fh, det.y2));
                    if (det.x2 < det.x1)
                        std::swap(det.x1, det.x2);
                    if (det.y2 < det.y1)
                        std::swap(det.y1, det.y2);
                    if (det.x2 > det.x1 && det.y2 > det.y1)
                        out.push_back(det);
                }
                if (!out.empty())
                    return true;
            } catch (const std::exception &) {
                continue;
            }
        }
        return false;
    }

    bool computeFocus(const std::vector<DetectionBox> &dets, int fw, int fh, double &cx, double &cy) const {
        if (dets.empty())
            return false;
        if (focus_mode_ == "label_priority") {
            if (!label_priority_.empty()) {
                for (const std::string &want : label_priority_) {
                    for (const auto &d : dets) {
                        if (d.has_label && d.label == want) {
                            cx = centerX(d);
                            cy = centerY(d);
                            return true;
                        }
                    }
                }
                return false;
            }
            // Empty label_priority with this focus mode: same as best_conf (avoids stuck-at-center filter).
            const DetectionBox *best = &dets[0];
            for (size_t i = 1; i < dets.size(); ++i) {
                if (dets[i].conf > best->conf)
                    best = &dets[i];
            }
            cx = centerX(*best);
            cy = centerY(*best);
            return true;
        }
        if (focus_mode_ == "conf_weighted_centroid") {
            double sw = 0, sx = 0, sy = 0;
            for (const auto &d : dets) {
                const double w = std::max(1e-6, d.conf);
                sx += centerX(d) * w;
                sy += centerY(d) * w;
                sw += w;
            }
            cx = sx / sw;
            cy = sy / sw;
            return true;
        }
        if (focus_mode_ == "union_center") {
            double x1 = dets[0].x1, y1 = dets[0].y1, x2 = dets[0].x2, y2 = dets[0].y2;
            for (size_t i = 1; i < dets.size(); ++i) {
                x1 = std::min(x1, dets[i].x1);
                y1 = std::min(y1, dets[i].y1);
                x2 = std::max(x2, dets[i].x2);
                y2 = std::max(y2, dets[i].y2);
            }
            cx = (x1 + x2) * 0.5;
            cy = (y1 + y2) * 0.5;
            return true;
        }
        // best_conf (default)
        const DetectionBox *best = &dets[0];
        for (size_t i = 1; i < dets.size(); ++i) {
            if (dets[i].conf > best->conf)
                best = &dets[i];
        }
        cx = centerX(*best);
        cy = centerY(*best);
        return true;
    }

    bool collectDetsForContain(const std::vector<DetectionBox> &dets,
                               std::vector<const DetectionBox *> &sel) const {
        sel.clear();
        if (dets.empty())
            return false;
        if (focus_mode_ == "label_priority") {
            if (!label_priority_set_.empty()) {
                for (const auto &d : dets) {
                    if (d.has_label && label_priority_set_.count(d.label))
                        sel.push_back(&d);
                }
                return !sel.empty();
            }
            // Empty label_priority: same as computeFocus — treat as all detections.
            for (const auto &d : dets)
                sel.push_back(&d);
            return true;
        }
        if (focus_mode_ == "best_conf") {
            const DetectionBox *best = &dets[0];
            for (size_t i = 1; i < dets.size(); ++i) {
                if (dets[i].conf > best->conf)
                    best = &dets[i];
            }
            sel.push_back(best);
            return true;
        }
        for (const auto &d : dets)
            sel.push_back(&d);
        return true;
    }

    bool computeViewportMeasurement(const std::vector<DetectionBox> &dets, int fw, int fh, double &mx, double &my) const {
        if (viewport_fit_ != "contain")
            return computeFocus(dets, fw, fh, mx, my);
        std::vector<const DetectionBox *> sel;
        if (!collectDetsForContain(dets, sel))
            return false;
        double x1 = sel[0]->x1, y1 = sel[0]->y1, x2 = sel[0]->x2, y2 = sel[0]->y2;
        for (size_t i = 1; i < sel.size(); ++i) {
            x1 = std::min(x1, sel[i]->x1);
            y1 = std::min(y1, sel[i]->y1);
            x2 = std::max(x2, sel[i]->x2);
            y2 = std::max(y2, sel[i]->y2);
        }
        double m = viewport_contain_margin_px_;
        if (!std::isfinite(m))
            m = 0;
        m = std::max(0.0, m);
        if (m > 0) {
            x1 -= m;
            y1 -= m;
            x2 += m;
            y2 += m;
        }
        x1 = std::max(0.0, std::min((double)fw, x1));
        y1 = std::max(0.0, std::min((double)fh, y1));
        x2 = std::max(0.0, std::min((double)fw, x2));
        y2 = std::max(0.0, std::min((double)fh, y2));
        if (!(x2 > x1 && y2 > y1))
            return false;
        const double half_w = viewport_dst_width_ * 0.5;
        const double half_h = viewport_dst_height_ * 0.5;
        centerForFixedViewportContain(x1, y1, x2, y2, fw, fh, half_w, half_h, mx, my);
        return true;
    }

    double frameDt() const {
        if (frame_rate_.getNumerator() > 0 && frame_rate_.getDenominator() > 0) {
            return (double)frame_rate_.getDenominator() / (double)frame_rate_.getNumerator();
        }
        return 1.0 / 30.0;
    }

    bool anyDerivativeLimit() const {
        for (double L : derivative_limit_by_order_) {
            if (std::isfinite(L) && L > 0)
                return true;
        }
        return false;
    }

    void applyDerivativeLimits(double p_star_x, double p_star_y, double dt, double &ox, double &oy) {
        if (!anyDerivativeLimit()) {
            ox = p_star_x;
            oy = p_star_y;
            return;
        }
        slew_x_.step(p_star_x, dt, derivative_limit_by_order_);
        slew_y_.step(p_star_y, dt, derivative_limit_by_order_);
        ox = slew_x_.d[0];
        oy = slew_y_.d[0];
    }

    void runHoldAndDeriv(double meas_x, double meas_y, bool meas_valid, double lp_x, double lp_y, int fw, int fh,
                         double dt, double &out_x, double &out_y) {
        double hx = lp_x, hy = lp_y;

        if (hold_enabled_) {
            if (meas_valid && have_prev_meas_) {
                const double dx = meas_x - prev_meas_x_;
                const double dy = meas_y - prev_meas_y_;
                const double spd = std::sqrt(dx * dx + dy * dy) / std::max(1e-9, dt);
                if (spd <= hold_speed_px_per_s_)
                    ++low_speed_frames_;
                else
                    low_speed_frames_ = 0;
            } else {
                low_speed_frames_ = 0;
            }
            if (meas_valid) {
                prev_meas_x_ = meas_x;
                prev_meas_y_ = meas_y;
                have_prev_meas_ = true;
            }

            if (!latched_) {
                if (hold_min_frames_ > 0 && low_speed_frames_ >= hold_min_frames_) {
                    latched_ = true;
                    latch_x_ = lp_x;
                    latch_y_ = lp_y;
                    latched_age_ = 0;
                }
            } else {
                ++latched_age_;
                bool brk = latched_age_ >= hold_max_frames_;
                if (meas_valid) {
                    const double ddx = meas_x - latch_x_;
                    const double ddy = meas_y - latch_y_;
                    if (std::sqrt(ddx * ddx + ddy * ddy) > hold_break_px_)
                        brk = true;
                } else if (lost_target_mode_ != "hold_last") {
                    brk = true;
                }
                if (brk) {
                    latched_ = false;
                    low_speed_frames_ = 0;
                    latched_age_ = 0;
                } else {
                    hx = latch_x_;
                    hy = latch_y_;
                }
            }
        }

        applyDerivativeLimits(hx, hy, dt, out_x, out_y);
    }

    void writeViewportMetadata(av::VideoFrame &frm, double cx, double cy) {
        Parameters j;
        const double h = std::max(1.0, viewport_marker_half_extent_);
        j["viewport_bbox"] = {cx - h, cy - h, cx + h, cy + h};
        j["full_frame_width"] = frm.width();
        j["full_frame_height"] = frm.height();
        j["viewport_dst_width"] = viewport_dst_width_;
        j["viewport_dst_height"] = viewport_dst_height_;
        av_dict_set(&frm.raw()->metadata, metadata_key_out_.c_str(), j.dump().c_str(), 0);
    }

    void processOneFrame(av::VideoFrame &frm, double agg_mx, double agg_my, bool current_frame_has_det) {
        const int fw = frm.width();
        const int fh = frm.height();
        const double dt = frameDt();

        if (fw != last_w_ || fh != last_h_) {
            last_w_ = fw;
            last_h_ = fh;
            resetPipelineState();
            slew_x_.reset(fw * 0.5);
            slew_y_.reset(fh * 0.5);
        }
        if (viewport_dst_width_ > fw || viewport_dst_height_ > fh) {
            throw Error("smooth_crop_viewport: viewport_dst_width/height exceed frame dimensions");
        }

        std::vector<DetectionBox> dets;
        double mx = 0, my = 0;
        bool frame_det = false;
        if (current_frame_has_det && finitePos(agg_mx) && finitePos(agg_my)) {
            mx = agg_mx;
            my = agg_my;
            frame_det = true;
        } else if (parseMetadata(frm, dets) && computeViewportMeasurement(dets, fw, fh, mx, my)) {
            frame_det = true;
        }

        if (frame_det)
            ++visible_streak_;
        else
            visible_streak_ = 0;

        const int need = std::max(1, min_visible_frames_);
        const bool meas_valid = frame_det && visible_streak_ >= need;

        const bool lost_center = (lost_target_mode_ == "frame_center");
        double lp_x = 0, lp_y = 0;
        lowpass_->step(mx, my, meas_valid, dt, fw, fh, lost_center, lp_x, lp_y);

        double out_x = 0, out_y = 0;
        runHoldAndDeriv(mx, my, meas_valid, lp_x, lp_y, fw, fh, dt, out_x, out_y);

        {
            const double half_w = viewport_dst_width_ * 0.5;
            const double half_h = viewport_dst_height_ * 0.5;
            out_x = std::max(half_w, std::min((double)fw - half_w, out_x));
            out_y = std::max(half_h, std::min((double)fh - half_h, out_y));
        }

        writeViewportMetadata(frm, out_x, out_y);

        ++frame_counter_;
        if (debug_log_every_n_ > 0 && (frame_counter_ % (uint64_t)debug_log_every_n_) == 0) {
            logstream << "smooth_crop_viewport: frame=" << frame_counter_ << " out=(" << out_x << "," << out_y << ")"
                      << " meas_valid=" << (meas_valid ? 1 : 0) << " streak=" << visible_streak_;
        }
    }

    void flushBufferOnEof() {
        while (!buffer_.empty()) {
            BufSlot slot = std::move(buffer_.front());
            buffer_.pop_front();
            double sx = 0, sy = 0;
            int cnt = 0;
            if (slot.valid) {
                sx += slot.mx;
                sy += slot.my;
                ++cnt;
            }
            const int take = std::min(lookahead_frames_, (int)buffer_.size());
            for (int i = 0; i < take; ++i) {
                if (buffer_[(size_t)i].valid) {
                    sx += buffer_[(size_t)i].mx;
                    sy += buffer_[(size_t)i].my;
                    ++cnt;
                }
            }
            const bool current_det = slot.valid;
            const double ax = current_det && cnt > 0 ? sx / cnt : 0;
            const double ay = current_det && cnt > 0 ? sy / cnt : 0;
            processOneFrame(slot.frm, ax, ay, current_det);
            this->sink_->put(std::move(slot.frm));
        }
    }

public:
    SmoothCropViewport(std::unique_ptr<Source<av::VideoFrame>> &&source, std::unique_ptr<Sink<av::VideoFrame>> &&sink,
                       std::vector<std::string> metadata_keys_in, std::string metadata_key_out, av::Rational frame_rate,
                       double model_content_width, double model_content_height, double model_content_offset_x,
                       double model_content_offset_y, double min_conf, std::unordered_set<int> allowed_classes,
                       std::unordered_set<std::string> allowed_labels, std::unordered_set<int> allowed_model_indices,
                       std::vector<std::string> label_priority, std::string focus_mode, std::string viewport_fit,
                       double viewport_contain_margin_px, std::string lost_target_mode,
                       std::unique_ptr<LowpassBackend> &&lowpass, double kalman_q_pos,
                       double kalman_q_vel, double kalman_r, double iir_fc_hz, size_t fir_taps, bool hold_enabled,
                       double hold_speed_px_per_s, int hold_min_frames, double hold_break_px, int hold_max_frames,
                       std::vector<double> derivative_limit_by_order, double viewport_marker_half_extent,
                       int viewport_dst_width, int viewport_dst_height, int lookahead_frames, int min_visible_frames,
                       int debug_log_every_n)
        : NodeSISO<av::VideoFrame, av::VideoFrame>(std::move(source), std::move(sink)),
          metadata_keys_in_(std::move(metadata_keys_in)), metadata_key_out_(std::move(metadata_key_out)),
          frame_rate_(frame_rate), model_content_width_(model_content_width),
          model_content_height_(model_content_height), model_content_offset_x_(model_content_offset_x),
          model_content_offset_y_(model_content_offset_y), min_conf_(min_conf),
          allowed_classes_(std::move(allowed_classes)), allowed_labels_(std::move(allowed_labels)),
          allowed_model_indices_(std::move(allowed_model_indices)), label_priority_(std::move(label_priority)),
          focus_mode_(std::move(focus_mode)), viewport_fit_(std::move(viewport_fit)),
          viewport_contain_margin_px_(viewport_contain_margin_px), lost_target_mode_(std::move(lost_target_mode)),
          lowpass_(std::move(lowpass)), kalman_q_pos_(kalman_q_pos),
          kalman_q_vel_(kalman_q_vel), kalman_r_(kalman_r), iir_fc_hz_(iir_fc_hz), fir_taps_(fir_taps),
          hold_enabled_(hold_enabled), hold_speed_px_per_s_(hold_speed_px_per_s),
          hold_min_frames_(hold_min_frames), hold_break_px_(hold_break_px), hold_max_frames_(hold_max_frames),
          derivative_limit_by_order_(std::move(derivative_limit_by_order)),
          lookahead_frames_(lookahead_frames), min_visible_frames_(min_visible_frames),
          viewport_marker_half_extent_(viewport_marker_half_extent),
          viewport_dst_width_(viewport_dst_width), viewport_dst_height_(viewport_dst_height),
          debug_log_every_n_(debug_log_every_n) {
        for (const auto &s : label_priority_)
            label_priority_set_.insert(s);
    }

    void process() override {
        av::VideoFrame frm = this->source_->get();
        if (isEofMarker(frm)) {
            flushBufferOnEof();
            this->sink_->put(std::move(frm));
            return;
        }
        if (!frm)
            return;

        BufSlot slot;
        slot.frm = std::move(frm);
        std::vector<DetectionBox> dets;
        if (parseMetadata(slot.frm, dets) &&
            computeViewportMeasurement(dets, slot.frm.width(), slot.frm.height(), slot.mx, slot.my)) {
            slot.valid = true;
        }

        buffer_.push_back(std::move(slot));

        if ((int)buffer_.size() <= lookahead_frames_) {
            return;
        }

        BufSlot out = std::move(buffer_.front());
        buffer_.pop_front();

        double sx = 0, sy = 0;
        int cnt = 0;
        // Average only when the *output* frame has a detection; lookahead smooths target, not validity.
        if (out.valid) {
            sx += out.mx;
            sy += out.my;
            ++cnt;
            for (int i = 0; i < lookahead_frames_ && i < (int)buffer_.size(); ++i) {
                if (buffer_[(size_t)i].valid) {
                    sx += buffer_[(size_t)i].mx;
                    sy += buffer_[(size_t)i].my;
                    ++cnt;
                }
            }
        }
        const bool current_det = out.valid;
        const double ax = current_det && cnt > 0 ? sx / cnt : 0;
        const double ay = current_det && cnt > 0 ? sy / cnt : 0;

        processOneFrame(out.frm, ax, ay, current_det);
        this->sink_->put(std::move(out.frm));
    }

    static std::shared_ptr<SmoothCropViewport> create(NodeCreationInfo &nci) {
        EdgeManager &edges = nci.edges;
        const Parameters &params = nci.params;
        auto src_edge = edges.find<av::VideoFrame>(params["src"]);
        auto frame_rate_src = src_edge->findNodeUp<IFrameRateSource>();
        av::Rational frame_rate = frame_rate_src ? frame_rate_src->frameRate() : av::Rational{0, 0};
        if (params.contains("fps_override")) {
            const double fps = params["fps_override"].get<double>();
            if (fps > 1e-6)
                frame_rate = av::Rational((int)std::lround(fps * 1000), 1000);
        }

        std::vector<std::string> metadata_keys_in;
        if (params.contains("metadata_key_ins") && params["metadata_key_ins"].is_array()) {
            for (const auto &v : params["metadata_key_ins"])
                metadata_keys_in.push_back(v.get<std::string>());
        }
        if (metadata_keys_in.empty())
            metadata_keys_in.push_back(params.value("metadata_key_in", std::string("basketball_analysis_v1")));
        const std::string metadata_key_out = params.value("metadata_key_out", std::string("smoothed_crop_viewport_v1"));
        const double mcw = params.value("model_content_width", 0.0);
        const double mch = params.value("model_content_height", 0.0);
        const double mcx = params.value("model_content_offset_x", 0.0);
        const double mcy = params.value("model_content_offset_y", 0.0);
        const double min_conf = params.value("min_conf", 0.0);
        std::unordered_set<int> allowed_classes;
        if (params.contains("allowed_classes") && params["allowed_classes"].is_array()) {
            for (const auto &v : params["allowed_classes"])
                allowed_classes.insert(v.get<int>());
        }
        std::unordered_set<std::string> allowed_labels;
        if (params.contains("allowed_labels") && params["allowed_labels"].is_array()) {
            for (const auto &v : params["allowed_labels"])
                allowed_labels.insert(v.get<std::string>());
        }
        std::unordered_set<int> allowed_model_indices;
        if (params.contains("allowed_model_indices") && params["allowed_model_indices"].is_array()) {
            for (const auto &v : params["allowed_model_indices"])
                allowed_model_indices.insert(v.get<int>());
        }
        std::vector<std::string> label_priority;
        if (params.contains("label_priority") && params["label_priority"].is_array()) {
            for (const auto &v : params["label_priority"])
                label_priority.push_back(v.get<std::string>());
        }
        const std::string focus_mode = params.value("focus_mode", std::string("best_conf"));
        const std::string viewport_fit = params.value("viewport_fit", std::string("contain"));
        if (viewport_fit != "center" && viewport_fit != "contain") {
            throw Error("smooth_crop_viewport: viewport_fit must be \"center\" or \"contain\"");
        }
        const double contain_margin = params.value("viewport_contain_margin_px", 0.0);
        const std::string lost_target = params.value("lost_target", std::string("hold_last"));
        const std::string filter_type = params.value("filter_type", std::string("kalman"));

        const double kq_pos = params.value("kalman_q_pos", 0.01);
        const double kq_vel = params.value("kalman_q_vel", 0.001);
        const double kr = params.value("kalman_r_meas", 4.0);
        const double iir_fc = params.value("iir_cutoff_hz", 2.0);
        const size_t fir_taps = (size_t)params.value("fir_taps", 5);

        double fs = 30.0;
        if (frame_rate.getNumerator() > 0 && frame_rate.getDenominator() > 0)
            fs = (double)frame_rate.getNumerator() / (double)frame_rate.getDenominator();
        if (params.contains("sample_rate_hz"))
            fs = params["sample_rate_hz"].get<double>();

        std::unique_ptr<LowpassBackend> lp;
        if (filter_type == "kalman") {
            lp = std::make_unique<LowpassKalman2>(kq_pos, kq_vel, kr);
        } else if (filter_type == "iir2") {
            lp = std::make_unique<LowpassIIR>(iir_fc, fs, kButterQ2, 1);
        } else if (filter_type == "iir4") {
            lp = std::make_unique<LowpassIIR>(iir_fc, fs, kButterQ4, 2);
        } else if (filter_type == "iir6") {
            lp = std::make_unique<LowpassIIR>(iir_fc, fs, kButterQ6, 3);
        } else if (filter_type == "iir8") {
            lp = std::make_unique<LowpassIIR>(iir_fc, fs, kButterQ8, 4);
        } else if (filter_type == "fir") {
            lp = std::make_unique<LowpassFIR>(fir_taps);
        } else {
            throw Error("smooth_crop_viewport: unknown filter_type (use kalman, iir2, iir4, iir6, iir8, fir)");
        }

        const bool hold_enabled = params.value("hold_enabled", false);
        const double hold_spd = params.value("hold_speed_px_per_s", 30.0);
        const int hold_min = (int)params.value("hold_min_frames", 6);
        const double hold_break = params.value("hold_break_px", 120.0);
        const int hold_max = (int)params.value("hold_max_frames", 90);

        std::vector<double> deriv_lim = parseDerivativeLimits(params);
        const double viewport_marker_half_extent = params.value("viewport_marker_half_extent", 1.0);
        if (!std::isfinite(viewport_marker_half_extent) || viewport_marker_half_extent < 0.0) {
            throw Error("smooth_crop_viewport: viewport_marker_half_extent must be finite and >= 0");
        }
        const int viewport_dst_width = params.at("viewport_dst_width").get<int>();
        const int viewport_dst_height = params.at("viewport_dst_height").get<int>();
        if (viewport_dst_width <= 0 || viewport_dst_height <= 0) {
            throw Error("smooth_crop_viewport: viewport_dst_width and viewport_dst_height must be positive");
        }
        if ((viewport_dst_width & 1) || (viewport_dst_height & 1)) {
            throw Error("smooth_crop_viewport: viewport_dst_width and viewport_dst_height must be even");
        }
        const int lookahead = (int)params.value("lookahead_frames", 0);
        int min_visible = (int)params.value("min_visible_frames", 3);
        if (min_visible < 1)
            min_visible = 1;
        const int dbg = (int)params.value("debug_log_every_n", 0);

        return NodeSISO<av::VideoFrame, av::VideoFrame>::template createCommon<SmoothCropViewport>(
            edges, params, std::move(metadata_keys_in), metadata_key_out, frame_rate, mcw, mch, mcx, mcy, min_conf,
            std::move(allowed_classes), std::move(allowed_labels), std::move(allowed_model_indices),
            std::move(label_priority), focus_mode, viewport_fit, contain_margin, lost_target, std::move(lp), kq_pos,
            kq_vel, kr,
            iir_fc, fir_taps, hold_enabled, hold_spd, hold_min, hold_break, hold_max, std::move(deriv_lim),
            viewport_marker_half_extent, viewport_dst_width, viewport_dst_height, lookahead, min_visible, dbg);
    }
};

DECLNODE(smooth_crop_viewport, SmoothCropViewport);
