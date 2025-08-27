#include "mc_localizer/mc_localizer_ros.hpp"

namespace mc_localizer {

MCLocalizerROS::MCLocalizerROS() : Node("mc_localizer_node")
{
    // hyper parameter
    scan_topic_name_ = "/robot_0/scan";
    odom_topic_name_ = "/robot_0/odom";
    initial_pose_topic_name_ = "/initialpose";
    // subscribers
    initial_pose_subscriber_ = this->create_subscription<geometry_msgs::msg::PoseWithCovarianceStamped>(
        initial_pose_topic_name_,
        10,
        [this](const std::shared_ptr<const geometry_msgs::msg::PoseWithCovarianceStamped> msg) {
            this->callback_initial_pose(msg);
        }
    ); // initial pose subscriber
    scan_subscriber_ = this->create_subscription<sensor_msgs::msg::LaserScan>(
        scan_topic_name_,
        10,
        [this](const sensor_msgs::msg::LaserScan::SharedPtr msg) {
            this->callback_scan(msg);
        }
    ); // laser scan subscriber
    odom_subscriber_ = this->create_subscription<nav_msgs::msg::Odometry>(
        odom_topic_name_,
        10,
        [this](const nav_msgs::msg::Odometry::SharedPtr msg) {
            this->callback_odom(msg);
        }
    ); // odometry subscriber
    
    // publisher
    particle_set_publisher_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("particle_set", 10);
    pose_publisher_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("pose", 10);

    // timer callback (main execute)
    timer_ = this->create_wall_timer(
        std::chrono::milliseconds(timer_period_),
        [this]() { this->callback_timer(); }
    );

    // flags
    is_initialized_ = false;
}

void MCLocalizerROS::callback_timer()
{
    // step 2
    mclocalizer_object_.update_particles_by_motion_model();

    publish_particle_set(mclocalizer_object_.particle_set_);

    publish_pose(mclocalizer_object_.mcl_estimated_pose_);
}

/**
 * @modifies mclocalizer_object_.mcl_estimated_pose_
 * @modefies particle_set_ indirectly by function initialize_particle_set
 * 
 * @modifies is_initialized_
 */
void MCLocalizerROS::callback_initial_pose(
    const std::shared_ptr<const geometry_msgs::msg::PoseWithCovarianceStamped> initial_pose
)
{
    // declare quaternion
    tf2::Quaternion orientation_quaternion(
        initial_pose->pose.pose.orientation.x,
        initial_pose->pose.pose.orientation.y,
        initial_pose->pose.pose.orientation.z,
        initial_pose->pose.pose.orientation.w
    );

    // calculate roll, pitch, yaw and we only take yaw
    double _, __, yaw;
    tf2::Matrix3x3 rotation_matrix(orientation_quaternion);
    rotation_matrix.getRPY(_, __, yaw);

    // set robot pose
    mclocalizer_object_.mcl_estimated_pose_.set_pose(
        initial_pose->pose.pose.position.x,
        initial_pose->pose.pose.position.y,
        yaw
    );

    // step 1
    // random sampling particles to initialize particle set
    mclocalizer_object_.initialize_particle_set(mclocalizer_object_.mcl_estimated_pose_, mclocalizer_object_.initial_noise_);

    // // 필요 시 신뢰도 초기화
    // if (estimateReliability_)
    //     resetReliabilities();

    // 초기화 완료 flag
    is_initialized_ = true;
}

void MCLocalizerROS::callback_scan(const sensor_msgs::msg::LaserScan::SharedPtr &scan)
{
    if (1) // canUpdateScan_
        scan_ = *scan;
    // if (!gotScan_)
    //     gotScan_ = true;
}

// getter, setter 함수를 사용하는 게 나을까?
/**
 * @modifies is_initialized_
 * @modifies odom_pose_timestamp_
 * 
 * @modifies mclocalizer_object_ : delta_x_, delta_y_, delta_distance_, delta_yaw_
 * @modifies mclocalizer_object_ : delta_time_sum_
 * @modifies mclocalizer_object_ : odom_pose_
 */
void MCLocalizerROS::callback_odom(const nav_msgs::msg::Odometry::SharedPtr odom)
{
    static double previous_time;
    double current_time = rclcpp::Time(odom->header.stamp).seconds();

    // inspection whether initialized
    if (!is_initialized_) {
        previous_time = current_time;
        is_initialized_ = true;
        return;
    }

    // terminate if current_time == previous_time
    double delta_time = current_time - previous_time;
    if (delta_time == 0.0) return;

    // current timestamp
    odom_pose_timestamp_ = rclcpp::Time(odom->header.stamp);

    // update delta values
    mclocalizer_object_.delta_x_ += odom->twist.twist.linear.x * delta_time;
    mclocalizer_object_.delta_y_ += odom->twist.twist.linear.y * delta_time;
    mclocalizer_object_.delta_distance_ += odom->twist.twist.linear.x * delta_time;
    mclocalizer_object_.delta_yaw_ += odom->twist.twist.angular.z * delta_time;

    // normalize yaw
    while (mclocalizer_object_.delta_yaw_ < - M_PI) mclocalizer_object_.delta_yaw_ += 2.0 * M_PI;
    while (mclocalizer_object_.delta_yaw_ > M_PI) mclocalizer_object_.delta_yaw_ -= 2.0 * M_PI;

    mclocalizer_object_.delta_time_sum_ += delta_time;

    // declare quaternion
    tf2::Quaternion orientation_quaternion(
        odom->pose.pose.orientation.x,
        odom->pose.pose.orientation.y,
        odom->pose.pose.orientation.z,
        odom->pose.pose.orientation.w
    );

    // calculate roll, pitch, yaw and we only take yaw
    double roll, pich, yaw;
    tf2::Matrix3x3 rotation_matrix(orientation_quaternion);
    rotation_matrix.getRPY(roll, pich, yaw);

    // set robot pose
    mclocalizer_object_.odom_pose_.set_pose(
        odom->pose.pose.position.x,
        odom->pose.pose.position.y,
        yaw
    );

    previous_time = current_time;
}

/**
 * @brief visualize particle set in RViz
 */
void MCLocalizerROS::publish_particle_set(const std::vector<Particle>& particle_set)
{
    visualization_msgs::msg::MarkerArray marker_array;

    for (size_t i = 0; i < particle_set.size(); ++i) {
        visualization_msgs::msg::Marker marker;
        // marker.header.frame_id = map_frame_id_;  // RViz에서 사용하는 fixed frame
        marker.header.frame_id = "odom"; 
        marker.header.stamp = this->now();       
        marker.ns = "particles";
        marker.id = i;
        marker.type = visualization_msgs::msg::Marker::SPHERE;  // particle를 sphere로 표시
        marker.action = visualization_msgs::msg::Marker::ADD;

        // 위치
        marker.pose.position.x = particle_set[i].get_x();
        marker.pose.position.y = particle_set[i].get_y();
        marker.pose.position.z = 0.05;  // 바닥 위 약간 띄움

        // 방향 (Sphere는 방향 필요 없음)
        marker.pose.orientation.x = 0.0;
        marker.pose.orientation.y = 0.0;
        marker.pose.orientation.z = 0.0;
        marker.pose.orientation.w = 1.0;

        // 크기
        marker.scale.x = 0.05;
        marker.scale.y = 0.05;
        marker.scale.z = 0.05;

        // 색상: weight 기반으로 alpha 설정 가능
        double w = particle_set[i].get_weight();
        marker.color.r = 1.0;
        marker.color.g = 0.0;
        marker.color.b = 0.0;
        marker.color.a = std::min(std::max(w * 10.0, 0.1), 1.0);

        marker_array.markers.push_back(marker);
    }

    // 퍼블리시
    particle_set_publisher_->publish(marker_array);
}

void MCLocalizerROS::publish_pose(const Pose& pose) {
    visualization_msgs::msg::MarkerArray marker_array;

    // --------------------------
    // Arrow: visualize estimated pose
    // --------------------------
    visualization_msgs::msg::Marker arrow_marker;
    std::string mapFrame_ = "odom";
    arrow_marker.header.frame_id = mapFrame_;
    arrow_marker.header.stamp = this->now();
    arrow_marker.ns = "mcl_pose_arrow";
    arrow_marker.id = 0;
    arrow_marker.type = visualization_msgs::msg::Marker::ARROW;
    arrow_marker.action = visualization_msgs::msg::Marker::ADD;

    // Arrow 시작점과 끝점
    geometry_msgs::msg::Point start, end;
    start.x = pose.get_x();
    start.y = pose.get_y();
    start.z = 0.05;  // 살짝 띄워서 particles 위에 표시

    double arrow_length = 0.5; // 화살표 길이
    end.x = start.x + arrow_length * cos(pose.get_yaw());
    end.y = start.y + arrow_length * sin(pose.get_yaw());
    end.z = 0.05;

    arrow_marker.points.push_back(start);
    arrow_marker.points.push_back(end);

    // 색상: 강렬한 빨간색
    arrow_marker.color.r = 1.0;
    arrow_marker.color.g = 0.0;
    arrow_marker.color.b = 0.0;
    arrow_marker.color.a = 1.0;

    // 화살표 두께 및 머리
    arrow_marker.scale.x = 0.05; // shaft diameter
    arrow_marker.scale.y = 0.1;  // head diameter
    arrow_marker.scale.z = 0.1;  // head length

    arrow_marker.lifetime = rclcpp::Duration::from_seconds(0.2);

    marker_array.markers.push_back(arrow_marker);

    // 퍼블리시
    pose_publisher_->publish(marker_array);
}



} // namespace mc_localizer