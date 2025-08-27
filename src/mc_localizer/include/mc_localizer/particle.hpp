#ifndef PARTICLE_H
#define PARTICLE_H

#include <mc_localizer/pose.hpp>

namespace mc_localizer {

class Particle {
private:
    Pose pose_;
    double weight_;

public:
    Particle(): pose_(0.0, 0.0, 0.0), weight_(0.0) {};

    Particle(double x, double y, double yaw, double weight): pose_(x, y, yaw), weight_(weight) {};

    Particle(Pose pose, double weight): pose_(pose), weight_(weight) {};

    ~Particle() {};

    inline double get_x(void) const { return pose_.get_x(); }
    inline double get_y(void) const { return pose_.get_y(); }
    inline double get_yaw(void) const { return pose_.get_yaw(); }
    inline Pose get_pose(void) const { return pose_; }
    inline double get_weight(void) const { return weight_; }

    inline void set_x(double x) { pose_.set_x(x); }
    inline void set_y(double y) { pose_.set_y(y); }
    inline void set_yaw(double yaw) { pose_.set_yaw(yaw); }
    inline void set_pose(double x, double y, double yaw) { pose_.set_pose(x, y, yaw); }
    inline void set_pose(Pose p) { pose_.set_pose(p); }
    inline void set_weight(double weight) { weight_ = weight; }
}; // class Particle

} // namespace mc_localizer

#endif // PARTICLE_H