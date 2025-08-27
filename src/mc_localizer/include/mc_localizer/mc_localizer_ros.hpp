#ifndef MC_LOCALIZER_ROS_H
#define MC_LOCALIZER_ROS_H

#include <rclcpp/rclcpp.hpp>                                  // rclcpp::Node
#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp> // geometry_msgs::msg::PoseWithCovarianceStamped
#include <sensor_msgs/msg/laser_scan.hpp>                     // sensor_msgs::msg::LaserScan
#include <nav_msgs/msg/odometry.hpp>                          // nav_msgs::msg::Odometry

#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>            // tf2::toMsg

#include <visualization_msgs/msg/marker_array.hpp>            // visualization_msgs::msg::MarkerArray

// defined
#include "mc_localizer/particle.hpp"
#include "mc_localizer/mc_localizer.hpp"

namespace mc_localizer {

class MCLocalizerROS : public rclcpp::Node {
public:
    MCLocalizerROS();
    ~MCLocalizerROS() {};

private:
    // subscriber
    std::string initial_pose_topic_name_, scan_topic_name_, odom_topic_name_;
    rclcpp::Subscription<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr initial_pose_subscriber_;
    rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr scan_subscriber_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_subscriber_;

    // publisher
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr particle_set_publisher_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pose_publisher_;

    // callback_timer
    const int timer_period_ = 10;             // timer period (ms)
    rclcpp::TimerBase::SharedPtr timer_;

    // callback_scan
    sensor_msgs::msg::LaserScan scan_;

    // class object
    MCLocalizer mclocalizer_object_;

    // timestamp
    rclcpp::Time odom_pose_timestamp_;

    // flags
    bool is_initialized_;

private: // callback functions
    void callback_timer();

    void callback_initial_pose(const std::shared_ptr<const geometry_msgs::msg::PoseWithCovarianceStamped> initial_pose);

    /**
     * @members scan_
     */
    void callback_scan(const sensor_msgs::msg::LaserScan::SharedPtr &scan);

    void callback_odom(const nav_msgs::msg::Odometry::SharedPtr odom);

private: // visualization functions
    void publish_particle_set(const std::vector<Particle>& particle_set);

    void publish_pose(const Pose& pose);
};

} // namespace mc_localizer

#endif // MC_LOCALIZER_ROS_H