#include "mc_localizer/mc_localizer.hpp"

namespace mc_localizer {

MCLocalizer::MCLocalizer()
{
    // pose
    // hyper parameter
    initial_pose_x_ = 0;
    initial_pose_y_ = 0;
    initial_pose_yaw_ = 0;
    initial_noise_x_ = 0.02;
    initial_noise_y_ = 0.02;
    initial_noise_yaw_ = 0.02;

    // particles
    particle_num_ = 1000;
    global_localization_particle_num_ = 100;

    // motion
    delta_x_ = delta_y_ = delta_distance_ = delta_yaw_ = 0.0;
    delta_x_sum_ = delta_y_sum_ = delta_distance_sum_ = delta_yaw_sum_ = 0.0; // 나중에 언젠간 쓰겠지??
    odom_differential_drive_model_noise_ = {1.0, 0.5, 0.5, 1.0}; // distance noise for distance^2, distance noise for yaw^2, yaw noise for distance^2, yaw noise for yaw^2

    // degree to radian
    initial_pose_yaw_ *= M_PI / 180.0;

    // initialization
    mcl_estimated_pose_.set_pose(initial_pose_x_, initial_pose_y_, initial_pose_yaw_);
    initial_noise_.set_pose(initial_noise_x_, initial_noise_y_, initial_noise_yaw_);
    initialize_particle_set(mcl_estimated_pose_, initial_noise_); // initialize particle_set_
    // resetReliabilities();

    // 임시
    global_localization_particle_set_.resize(global_localization_particle_num_);
}

/**
 * @brief Gaussian random sampling using initial pose (step 1)
 * 
 * @modifies particle_set_
 * 
 * @called callback_initial_pose
 */
void MCLocalizer::initialize_particle_set(
    const Pose &initial_pose,
    const Pose &initial_noise 
)
{
    // set up size of particle set
    particle_set_.resize(particle_num_);

    // extract mean pose and standard deviation
    double mean_x = initial_pose.get_x();
    double mean_y = initial_pose.get_y();
    double mean_yaw = initial_pose.get_yaw();
    double standard_deviation_x = initial_noise.get_x();
    double standard_deviation_y = initial_noise.get_y();
    double standard_deviation_yaw = initial_noise.get_yaw();

    // set same weight (guarantee int)
    double weight = 1.0 / static_cast<double>(particle_num_);

    // generate particle set
    for (int i = 0; i < particle_num_; ++i) {
        double x = mean_x + nrand(standard_deviation_x);
        double y = mean_y + nrand(standard_deviation_y);
        double yaw = mean_yaw + nrand(standard_deviation_yaw);

        // store particle samples
        particle_set_[i].set_pose(x, y, yaw);
        particle_set_[i].set_weight(weight);
    }
}

/**
 * @brief 1. updating delta sum vaiables
 * @brief 2. updating robot pose by differntial drive model
 * @brief 3. updating particles by diffential drive model *****
 * @brief 4. updating reliabilities
 * 
 * @member particleNum_
 * @member useOmniDirectionalModel_, odomNoiseDDM_[], odomNoiseODM_[]
 * @member estimateReliability_
 * 
 * 
 * @modifies deltaX_, deltaY_, deltaDist_, deltaYaw_
 * @modifies deltaXSum_, deltaYSum_, deltaDistSum_, deltaYawSum_
 * 
 * @modifies mcl_estimated_pose_
 * @modifies particle_set_
 * @modifies reliabilities_
 */
void MCLocalizer::update_particles_by_motion_model(void)
{
    // delare and initialize local variables
    double delta_x = delta_x_;
    double delta_y = delta_y_;
    double delta_distance = delta_distance_;
    double delta_yaw = delta_yaw_;

    // initalize member variables
    delta_x_ = 0;
    delta_y_ = 0;
    delta_distance_ = 0;
    delta_yaw_ = 0;

    // update member variables
    delta_x_sum_ += fabs(delta_x);
    delta_y_sum_ += fabs(delta_y);
    delta_distance_sum_ += fabs(delta_distance);
    delta_yaw_sum_ += fabs(delta_yaw);

    // update pose by differential drive model
    double yaw = mcl_estimated_pose_.get_yaw();
    double t = yaw + delta_yaw / 2.0;                                  // midpoint approximation in differntial drive model
    double x = mcl_estimated_pose_.get_x() + delta_distance * cos(t);
    double y = mcl_estimated_pose_.get_y() + delta_distance * sin(t);
    yaw += delta_yaw;
    mcl_estimated_pose_.set_pose(x, y, yaw);

    // update particles by differential drive model
    double squared_delta_distance = delta_distance * delta_distance;
    double squared_delta_yaw = delta_yaw * delta_yaw;
    double distance_noise_variance = squared_delta_distance * odom_differential_drive_model_noise_[0] + squared_delta_yaw * odom_differential_drive_model_noise_[1];
    double yaw_noise_variance = squared_delta_distance * odom_differential_drive_model_noise_[2] + squared_delta_yaw * odom_differential_drive_model_noise_[3];
    for (int i = 0; i < particle_num_; ++i) {
        double ddist = delta_distance + nrand(distance_noise_variance);
        double dyaw = delta_yaw + nrand(yaw_noise_variance);
        double yaw = particle_set_[i].get_yaw();
        double t = yaw + dyaw / 2.0;
        double x = particle_set_[i].get_x() + ddist * cos(t);
        double y = particle_set_[i].get_y() + ddist * sin(t);
        yaw += dyaw;

        // update poarticle set
        particle_set_[i].set_pose(x, y, yaw);

        // estimate reliability
    }
}

// void MCLocalizer::calculate_likelihoods_by_measurement_model(void)
// {
//     // if (scanMightInvalid_)
//     //     return;

//     // if (rejectUnknownScan_ && (measurementModelType_ == 0 || measurementModelType_ == 1))
//     //     rejectUnknownScan();    

//     // sensor pose 계산
//     double x0 = base_link_to_laser_.get_x();
//     double y0 = base_link_to_laser_.get_y();
//     double yaw0 = base_link_to_laser_.get_yaw();
//     std::vector<Pose> sensor_poses(particle_num_);
//     for (int i = 0; i < particle_num_; ++i) {
//         double yaw = particles_[i].get_yaw();
//         double sensorX = x0 * cos(yaw) - y0 * sin(yaw) + particles_[i].get_x();
//         double sensorY = x0 * sin(yaw) - y0 * cos(yaw) + particles_[i].get_y();
//         double sensorYaw = yaw0 + yaw;
//         Pose sensorPose(sensorX, sensorY, sensorYaw);
//         sensor_poses[i] = sensorPose;
//         particles_[i].set_weight(0.0);
//     }

//     // measurement linkelihood
//     likelihoodShiftedSteps_.clear();
//     for (int i = 0; i < (int)scan_.ranges.size(); i += scanStep_) {
//         double range = scan_.ranges[i];
//         double rangeAngle = (double)i * scan_.angle_increment + scan_.angle_min;
//         double max;
//         for (int j = 0; j < particle_num_; ++j) {
//             double p;
//             if (measurementModelType_ == 0)
//                 p = calculateLikelihoodFieldModel(sensor_poses[j], range, rangeAngle);
//             else if (measurementModelType_ == 1)
//                 p = calculateBeamModel(sensor_poses[j], range, rangeAngle);
//             else
//                 p = calculateClassConditionalMeasurementModel(sensor_poses[j], rage, rangeAngle);

//             double w = particles_[j].getW();
//             w += log(p);
//             particles_[j].setW(w);
//             if (j == 0)
//                 max = w;
//             else {
//                 if (max < w)
//                     max = w;
//             }
//         }

//         // numerical stability
//         if (max < -300.0) {
//             for (int j = 0; j < particlesNum_; ++j) {
//                 double w = particles_[j].getW() + 300.0;
//                 particles_[j].set_weight(w);
//             }
//             likelihoodShiftedSteps_.push_back(true);
//         } else {
//             likelihoodShiftedSteps_.push_back(false);
//         }
//     }

//     // Weight normalization
//     double sum = 0.0;
//     double max;
//     int maxIdx;
//     for (int i = 0; i < particle_num_; ++i) {
//         // The log sum is converted to the probability.
//         double w = exp(particles_[i].get_weight());
//         particles_[i].set_weight(w);
//         sum += w;
//         if (i == 0) {
//             max = w;
//             maxIdx = i;
//         } else if (max < w) {
//             max = w;
//             maxIdx = i;
//         }
//     }
//     totalLikelihood_ = sum;
//     averageLikelihood_ = sum / (double)particlesNum_;
//     maxLikelihood_ = max;
//     maxLikelihoodParticleIdx_ = maxIdx;
// }

// void MCLocalizer::calculate_likelihoods_by_decision_model(void)
// {
//     double likelihood_sum = 0.0;
//     double max_likelihood;
//     int max_likelihood_particle_index;
//     for (int i = 0; i < particle_num_; ++i) {
//         Pose particle_pose = particles_[i].get_pose();
//         double measurement_likelihood = particle_set_[i].getW();
//         std::vector<double> residual_errors = getResidualErrors<double>(particle_pose);
//         double decision_likelihood;
//         if (1) { // check
//             double mae = mae_classifier_.getMAE(residual_errors);
//             decision_likelihood = mae_classifier_.calculate_decision_model(mae, &reliabilities_[i]);
//             mae_[i] = mae;
//         }

//         double w = measurement_likelihood * decision_likelihood;
//         particles_[i].set_weight(w);
//         likelihood_sum += w;
//         if (i == 0) {
//             max_likelihood = w;
//             max_likelihood_particle_index = i;
//         } else if (max_likelihood < w) {
//             max_likelihood = w;
//             max_likelihood_particle_index = i
//         }
//     }

//     total_likelihood_ = likelihood_sum;
//     average_likelihood_ = likelihood_sum / (double) particle_num_;
//     max_likelihood_ = max_likelihood;
//     max_likelihood_particle_index_ = max_likelihood_particle_index;
//     reliability_ = reliabilities_[max_likelihood_particle_index];
// }

// void MCLocalizer::calculate_likelihoods_from_global_localization(void)
// {
//     //
// }

/*
    static FILE *fp;
    if (fp == NULL && writePose_) {
        fp = fopen(poseLogFile_.c_str(), "w");
        if (fp == NULL) {
            fprintf(stderr, "Cannot open a pose log file -> %s\\n", poseLogFile_.c_str());
            writePose_ = false;
        }
    }

    if (scanMightInvalid_)
        return;

    double tmpYaw = mclPose_.getYaw();
    double x = 0.0, y = 0.0, yaw = 0.0;
    double sum = 0.0;
    
    for (int i = 0; i < particlesNum_; ++i) {
        double w = particles_[i].getW();
        x += particles_[i].getX() * w;
        y += particles_[i].getY() * w;
        double dyaw = tmpYaw - particles_[i].getYaw();
        while (dyaw < -M_PI)
            dyaw += 2.0 * M_PI;
        while (dyaw > M_PI)
            dyaw -= 2.0 * M_PI;
        yaw += dyaw * w;
        sum += w;
    }

    if (useGLPoseSampler_ && canUseGLSampledPoses_) {
        double x2 = x, y2 = y, yaw2 = yaw;
        for (int i = 0; i < glParticlesNum_; ++i) {
            double w = glParticles_[i].getW();
            x += glParticles_[i].getX() * w;
            y += glParticles_[i].getY() * w;
            double dyaw = tmpYaw - glParticles_[i].getYaw();
            while (dyaw < -M_PI)
                dyaw += 2.0 * M_PI;
            while (dyaw > M_PI)
                dyaw -= 2.0 * M_PI;
            yaw += dyaw * w;
            sum += w;
        }
        if (sum > 1.0)
            x = x2, y = y2, yaw = yaw2;
    }

    yaw = tmpYaw - yaw;
    mclPose_.setPose(x, y, yaw);

    if (writePose_)
        fprintf(fp, "%lf %lf %lf %lf\\n", mclPoseStamp_.toSec(), x, y, yaw);
*/

/**
 * @member particle_set_, global_localization_particle_set_
 * @member particle_num_, global_localization_particle_num_
 * @member is_global_localization_sampling_enabled, can_use_global_localization_sample
 * 
 * @modifies mcl_estimated_pose_
 */
void MCLocalizer::estimate_robot_pose(void)
{
    // if (scanMightInvalid_)
    //     return;

    double reference_yaw = mcl_estimated_pose_.get_yaw();
    double estimated_x = 0.0, estimated_y = 0.0, estimated_yaw = 0.0;
    double weight_sum = 0.0;
    for (int i = 0; i < particle_num_; ++i) {
        double weight = particle_set_[i].get_weight();
        estimated_x += particle_set_[i].get_x() * weight; // weighted sum for x in pose tracking particle set
        estimated_y += particle_set_[i].get_y() * weight; // weighted sum for y in pose tracking particle set

        double dyaw = reference_yaw - particle_set_[i].get_yaw();
        while (dyaw < - M_PI) dyaw += 2.0 * M_PI;         
        while (dyaw > M_PI) dyaw -= 2.0 * M_PI;           
        estimated_yaw += dyaw * weight;                   // weighted sum for yaw in pose tracking particle set

        weight_sum += weight;
    }

    // if (is_global_localization_sampling_enabled && can_use_global_localization_sample) {
    //     double x2 = estimated_x, y2 = estimated_y, yaw2 = estimated_yaw;
    //     for (int i = 0; i < global_localization_particle_num_; ++i) {
    //         double weight = global_localization_particle_set_[i].get_weight();
    //         estimated_x += global_localization_particle_set_[i].get_x() * weight;
    //         estimated_y += global_localization_particle_set_[i].get_y() * weight;

    //         double dyaw = reference_yaw - global_localization_particle_set_[i].get_yaw();
    //         while (dyaw < -M_PI) dyaw += 2.0 * M_PI;
    //         while (dyaw > M_PI) dyaw -= 2.0 * M_PI;
    //         estimated_yaw += dyaw * weight;

    //         weight_sum += weight;
    //     }

    //     // prevent sum of weights is over 1
    //     if (weight_sum > 1.0) estimated_x = x2, estimated_y = y2, estimated_yaw = yaw2;
    // }

    estimated_yaw = reference_yaw - estimated_yaw;

    // update mcl_estimated_pose_
    mcl_estimated_pose_.set_pose(estimated_x, estimated_y, estimated_yaw);
}

void MCLocalizer::resample_particles(void)
{
    //
}

} // namespace mc_localizer
