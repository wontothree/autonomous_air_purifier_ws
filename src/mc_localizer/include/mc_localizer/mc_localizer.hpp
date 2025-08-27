#ifndef MC_LOCALIZER_H
#define MC_LOCALIZER_H

// #include <cmath> // fabs, rand, cos, log, M_PI
#include <vector> // std::vector

#include "sensor_msgs/msg/laser_scan.hpp"

#include <mc_localizer/pose.hpp>
#include <mc_localizer/particle.hpp>
#include <mc_localizer/mae_classifier.hpp>

namespace mc_localizer {

class MCLocalizer {
public:
    MCLocalizer();
    ~MCLocalizer() {};

public:
    // pose
    double initial_pose_x_, initial_pose_y_, initial_pose_yaw_;
    double initial_noise_x_, initial_noise_y_, initial_noise_yaw_;
    Pose mcl_estimated_pose_;  // current estimated pose & finally estimated pose
    Pose odom_pose_;
    Pose initial_noise_;  // noise for initial sampling
    Pose base_link_to_laser_;

    // particles
    int particle_num_, global_localization_particle_num_;
    std::vector<Particle> particle_set_, global_localization_particle_set_;

    // motion
    double delta_x_, delta_y_, delta_distance_, delta_yaw_;
    double delta_x_sum_, delta_y_sum_, delta_distance_sum_, delta_yaw_sum_, delta_time_sum_;

private:
    // map
    Pose map_origin_;
    double map_resolution_;
    int map_width_, map_height_;

    // motion (updating by odom)
    std::vector<double> odom_differential_drive_model_noise_; // distance noise for distance^2, distance noise for yaw^2, yaw noise for distance^2, yaw noise for yaw^2

    // measurements
    std::vector<bool> likelihoodShiftedSteps_;
    sensor_msgs::msg::LaserScan scan_;

    // measurement model
    int scanStep_;

    // localizatoin result
    double total_likelihood_, average_likelihood_, max_likelihood_;
    int max_likelihood_particle_index_;

    // reliability estimation
    double reliability_;
    std::vector<double> reliabilities_;

    // mean absolute error (MAE)-based failure detector
    MAEClassifier mae_classifier_;

    // flags
    bool is_global_localization_sampling_enabled = false;
    bool can_use_global_localization_sample = false;

public:
    void initialize_particle_set(const Pose &initial_pose, const Pose &initial_noise);

    void update_particles_by_motion_model(void);

    // void calculate_likelihoods_by_measurement_model(void);

    // void calculate_likelihoods_by_decision_model(void);

    // void calculate_likelihoods_from_global_localization(void);

    void estimate_robot_pose(void);

    /**
     * @modifies
     */
    void resample_particles(void);

private:
    // template <typename T>
    // std::vector<T> getResidualErrors(Pose pose) {
    //     double yaw = pose.get_yaw();
    //     double sensorX = base_link_to_laser_.get_x() * cos(yaw) - base_link_to_laser_.get_y() * sin(yaw) + pose.get_x();
    //     double sensorY = base_link_to_laser_.get_x() * sin(yaw) + base_link_to_laser_.get_y() * cos(yaw) + pose.get_y();
    //     double sensorYaw = base_link_to_laser_.get_yaw() + yaw;
    //     int size = (int)scan_.ranges.size();
    //     std::vector<T> residualErrors(size);
    //     for (int i = 0; i < size; ++i) {
    //         double r = scan_.ranges[i];
    //         if (r <= scan_.range_min || scan_.range_max <= r) {
    //             residualErrors[i] = -1.0;
    //             continue;
    //         }
    //         double t = (double)i * scan_.angle_increment + scan_.angle_min + sensorYaw;
    //         double x = r * cos(t) + sensorX;
    //         double y = r * sin(t) + sensorY;
    //         int u, v;
    //         xy2uv(x, y, &u, &v);
    //         if (is_on_map(u, v)) {
    //             T dist = (T)distMap_.at<float>(v, u);
    //             residualErrors[i] = dist;
    //         } else {
    //             residualErrors[i] = -1.0;
    //         }
    //     }
    //     return residualErrors;
    // }

private:
    // ---------------------------------------------------------------------------- //
    // util functions ------------------------------------------------------------- //
    // ---------------------------------------------------------------------------- //
    /**
     * @brief 정규 분포를 따르는 무작위 값을 생성합니다.
     *
     * 이 함수는 Box-Muller 변환을 사용하여 평균이 0이고 표준편차가 n인 
     * 정규 분포에서 무작위 값을 생성합니다.
     *
     * @param n 생성된 무작위 값의 표준편차를 결정하는 데 사용되는 값입니다.
     * @return n을 표준편차로 하는 정규 분포에서 샘플링된 무작위 값
     */
    inline double nrand(double n)
    {
        return (n * sqrt(-2.0 * log((double)rand() / RAND_MAX)) * cos(2.0 * M_PI * rand() / RAND_MAX));
    }

    /**
     * @brief 주어진 좌표가 지도의 경계 내에 있는지 확인합니다.
     *
     * 이 함수는 입력된 `u`와 `v` 좌표가 지도의 유효한 범위에 속하는지 검사합니다.
     *
     * @param u 지도의 가로축(column) 좌표입니다.
     * @param v 지도의 세로축(row) 좌표입니다.
     * @return 좌표가 지도 경계 내에 있으면 true, 아니면 false를 반환합니다.
     * @memberof map_width_ 지도의 너비를 나타냅니다.
     * @memberof map_height_ 지도의 높이를 나타냅니다.
     */
    inline bool is_on_map(int u, int v) {
        if (0 <= u && u < map_width_ && 0 <= v && v < map_height_)
            return true;
        else
            return false;
    }

    /**
     * @brief 전역 좌표계(x, y)를 지도 셀 좌표계(u, v)로 변환합니다.
     *
     * 이 함수는 로봇의 전역 좌표계에서 측정된 위치를 
     * 지도의 원점 및 회전을 고려하여 셀 좌표로 변환합니다.
     *
     * @param x 전역 좌표계에서의 x 위치입니다.
     * @param y 전역 좌표계에서의 y 위치입니다.
     * @param u 변환된 지도의 가로축(u) 좌표를 저장할 포인터입니다.
     * @param v 변환된 지도의 세로축(v) 좌표를 저장할 포인터입니다.
     * @member map_origin_ 지도의 원점(origin)과 방향(yaw) 정보를 포함합니다.
     * @member map_resolution_ 지도의 해상도(resolution)를 나타냅니다.
     */
    inline void xy2uv(double x, double y, int *u, int *v) {
        double dx = x - map_origin_.get_x();
        double dy = y - map_origin_.get_y();
        double yaw = -map_origin_.get_yaw();
        double xx = dx * cos(yaw) - dy * sin(yaw);
        double yy = dx * sin(yaw) + dy * cos(yaw);
        *u = (int)(xx / map_resolution_);
        *v = (int)(yy / map_resolution_);
    }

    /**
     * @brief 지도 셀 좌표계(u, v)를 전역 좌표계(x, y)로 변환합니다.
     *
     * 이 함수는 지도의 셀 좌표를 전역 좌표계의 위치로 변환합니다.
     * 이는 `xy2uv` 함수의 역변환입니다.
     *
     * @param u 지도의 가로축(u) 좌표입니다.
     * @param v 지도의 세로축(v) 좌표입니다.
     * @param x 변환된 전역 좌표계의 x 위치를 저장할 포인터입니다.
     * @param y 변환된 전역 좌표계의 y 위치를 저장할 포인터입니다.
     * @member map_origin_ 지도의 원점(origin)과 방향(yaw) 정보를 포함합니다.
     * @member map_resolution_ 지도의 해상도(resolution)를 나타냅니다.
     */
    inline void uv2xy(int u, int v, double *x, double *y) {
        double xx = (double)u * map_resolution_;
        double yy = (double)v * map_resolution_;
        double yaw = -map_origin_.get_yaw();
        double dx = xx * cos(yaw) + yy * sin(yaw);
        double dy = -xx * sin(yaw) + yy * cos(yaw);
        *x = dx + map_origin_.get_x();
        *y = dy + map_origin_.get_y();
    }

}; // class mc_localizer

} // namespace mc_localizer

#endif // MC_LOCALIZER_H