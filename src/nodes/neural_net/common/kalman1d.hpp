#pragma once

#include <cmath>

class Kalman1D {
    double p11_ = 1e3, p12_ = 0, p21_ = 0, p22_ = 1e3;
    double x_ = 0, v_ = 0;
    bool have_ = false;
    double q_pos_, q_vel_, r_meas_;

public:
    Kalman1D(double q_pos, double q_vel, double r_meas) : q_pos_(q_pos), q_vel_(q_vel), r_meas_(r_meas) {}
    Kalman1D() : q_pos_(1.0), q_vel_(0.1), r_meas_(4.0) {}

    void reset() {
        p11_ = 1e3;
        p12_ = p21_ = 0;
        p22_ = 1e3;
        x_ = v_ = 0;
        have_ = false;
    }

    void init(double pos, double vel = 0.0) {
        x_ = pos;
        v_ = vel;
        p11_ = 1e3;
        p12_ = p21_ = 0;
        p22_ = 1e3;
        have_ = true;
    }

    void step(double z, bool valid, double dt, bool use_center, int dim) {
        const double center = dim * 0.5;
        if (dt <= 0 || !std::isfinite(dt))
            dt = 1.0 / 30.0;
        if (!have_) {
            x_ = valid ? z : center;
            v_ = 0;
            p11_ = 1e3;
            p12_ = p21_ = 0;
            p22_ = 1e3;
            have_ = true;
            return;
        }
        // Predict
        x_ += v_ * dt;
        p11_ += dt * (p12_ + p21_) + dt * dt * p22_ + q_pos_;
        p12_ += dt * p22_;
        p21_ += dt * p22_;
        p22_ += q_vel_;

        if (!valid && !use_center)
            return;

        // Correct
        const double zz = valid ? z : center;
        const double S = p11_ + r_meas_;
        const double K1 = p11_ / S;
        const double K2 = p21_ / S;
        const double innov = zz - x_;
        x_ += K1 * innov;
        v_ += K2 * innov;
        const double np11 = p11_ - K1 * p11_;
        const double np12 = p12_ - K1 * p12_;
        const double np21 = p21_ - K2 * p11_;
        const double np22 = p22_ - K2 * p12_;
        p11_ = np11;
        p12_ = np12;
        p21_ = np21;
        p22_ = np22;
    }

    // Predict-only step (no measurement)
    void predict(double dt) {
        if (!have_) return;
        x_ += v_ * dt;
        p11_ += dt * (p12_ + p21_) + dt * dt * p22_ + q_pos_;
        p12_ += dt * p22_;
        p21_ += dt * p22_;
        p22_ += q_vel_;
    }

    // Correct-only step (measurement update)
    void correct(double z) {
        if (!have_) {
            init(z);
            return;
        }
        const double S = p11_ + r_meas_;
        const double K1 = p11_ / S;
        const double K2 = p21_ / S;
        const double innov = z - x_;
        x_ += K1 * innov;
        v_ += K2 * innov;
        const double np11 = p11_ - K1 * p11_;
        const double np12 = p12_ - K1 * p12_;
        const double np21 = p21_ - K2 * p11_;
        const double np22 = p22_ - K2 * p12_;
        p11_ = np11;
        p12_ = np12;
        p21_ = np21;
        p22_ = np22;
    }

    double pos() const { return x_; }
    double vel() const { return v_; }
    bool initialized() const { return have_; }
};
