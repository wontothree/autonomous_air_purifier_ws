#ifndef POSE_H
#define POSE_H

#include <cmath>

namespace mc_localizer {

class Pose {
private:
    double x_, y_, yaw_;

    void normalize_yaw(void) {
        while (yaw_ < -M_PI)
            yaw_ += 2.0 * M_PI;
        while (yaw_ > M_PI)
            yaw_ -= 2.0 * M_PI;
    }

public:
    Pose():
        x_(0.0), y_(0.0), yaw_(0.0) {};

    Pose(double x, double y, double yaw):
        x_(x), y_(y), yaw_(yaw) {};

    ~Pose() {};

    inline void set_x(double x) { x_ = x; }
    inline void set_y(double y) { y_ = y; }
    inline void set_yaw(double yaw) { yaw_ = yaw, normalize_yaw(); }
    inline void set_pose(double x, double y, double yaw) { x_ = x, y_ = y, yaw_ = yaw, normalize_yaw(); }
    inline void set_pose(Pose p) { x_ = p.x_, y_ = p.y_, yaw_ = p.yaw_, normalize_yaw(); }

    inline double get_x(void) const { return x_; }
    inline double get_y(void) const { return y_; }
    inline double get_yaw(void) const { return yaw_; }
    inline Pose get_pose(void) { return Pose(x_, y_, yaw_); }

}; // class Pose

} // namespace mc_localizer

#endif // POSE_H