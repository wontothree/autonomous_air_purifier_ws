import math
import numpy as np
from dataclasses import dataclass, field

@dataclass
class Pose:
    _x: float = 0.0
    _y: float = 0.0
    _yaw: float = 0.0

    def __post_init__(self):
        self.yaw = self._yaw

    def normalize_yaw(self):
        self._yaw = math.atan2(math.sin(self._yaw), math.cos(self._yaw))

    @property
    def x(self):
        return self._x

    @x.setter
    def x(self, value):
        self._x = value

    @property
    def y(self):
        return self._y

    @y.setter
    def y(self, value):
        self._y = value

    @property
    def yaw(self):
        return self._yaw

    @yaw.setter
    def yaw(self, value):
        self._yaw = value
        self.normalize_yaw()

    def update(self, x: float, y: float, yaw: float):
        self.x = x 
        self.y = y 
        self.yaw = yaw

@dataclass
class Particle:
    pose: Pose = field(default_factory=Pose)
    weight: float = 0.0

class MonteCarloLocalizer:
    def __init__(self,
            particle_num=10,
            initial_pose_noise_std=[0.02, 0.02, 0.02], # [x, y, yaw]
            odom_noise_std=[0.01, 0.002, 0.002, 0.01],

            # Weight
            scan_min_range=0.05,
            scan_max_range=8.0,
            scan_min_angle=-1.9,
            scan_angle_increment=0.016,
            scan_step=10,
            sigma_hit=0.2,
            unknown_class_prior=0.5,
            known_class_prior=0.5,
            z_hit=0.9,
            z_max=0.05,
            z_rand=0.05,
            unknown_class_lambda_=1.0,
            baselink_to_laser=[0.147, 0.0, 0.0],  # [x, y, yaw(rad)]

            # Resamping
            omega_slow=0.0,
            omega_fast=0.0,
            alpha_slow=0.001,
            alpha_fast=0.9,
            resample_ess_ratio=0.5,
            delta_distance_resample_threshold=0.2,
            delta_yaw_resample_threshold=0.02,
            random_resampling_noise_std=[0.02, 0.02, 0.02] # [x, y, yaw]
        ):
        
        # Constants
        self.particle_num = particle_num
        self.initial_pose_noise_std = initial_pose_noise_std
        self.odom_noise_std = odom_noise_std
        self.scan_min_angle = scan_min_angle
        self.scan_angle_increment = scan_angle_increment
        self.scan_min_range = scan_min_range
        self.scan_max_range = scan_max_range
        self.scan_step = scan_step
        self.sigma_hit = sigma_hit
        self.unknown_class_prior = unknown_class_prior
        self.known_class_prior = known_class_prior
        self.z_hit = z_hit
        self.z_max = z_max
        self.z_rand = z_rand
        self.unknown_class_lambda_ = unknown_class_lambda_
        self.baselink_to_laser = baselink_to_laser
        self.omega_slow = omega_slow
        self.omega_fast = omega_fast
        self.alpha_slow = alpha_slow
        self.alpha_fast = alpha_fast
        self.resample_ess_ratio = resample_ess_ratio
        self.delta_distance_resample_threshold = delta_distance_resample_threshold
        self.delta_yaw_resample_threshold = delta_yaw_resample_threshold
        self.random_resampling_noise_std = random_resampling_noise_std

        # Particle set
        self.particles = []   

        # estimate_robot_pose (used by resample_particles)
        self.estimated_x = None
        self.estimated_y = None
        self.estimated_yaw = None

        # resample_particles
        self.delta_distance_abs_sum = 0
        self.delta_yaw_abs_sum = 0
        self.average_likelihood = 0

    def initialize_particles(
            self, 
            initial_pose: Pose
        ) -> None:
        """
        Parameters
        ----------
        - initial_pose: initial pose of robot (x, y, yaw)
        
        Update
        ------
        - self.particles: initialized particles

        Using
        -----
        - self.particle_num
        - self.initial_pose_noise_std
        """
        particle_xs = initial_pose.x + np.random.normal(0, self.initial_pose_noise_std[0], self.particle_num)
        particle_ys = initial_pose.y + np.random.normal(0, self.initial_pose_noise_std[1], self.particle_num)
        particle_yaws = initial_pose.yaw + np.random.normal(0, self.initial_pose_noise_std[2], self.particle_num)
        
        initial_weight = 1.0 / self.particle_num
    
        self.particles = [
            Particle(
                pose=Pose(particle_xs[i], particle_ys[i], particle_yaws[i]),
                weight=initial_weight
            )
            for i in range(self.particle_num)
        ]
    
    def update_particles_by_motion_model(self, 
            delta_distance: float, 
            delta_yaw: float
        ) -> None:
        """
        Parameters
        ----------
        - delta_distance: measured by IMU
        - delta_yaw
        
        Update
        ------
        - self.particles

        Using
        -----
        - self.odom_noise_std
        """
        # Update self.delta_distance_abs_sum and self.delta_yaw_abs_sum for resampling
        self.delta_distance_abs_sum += abs(delta_distance)
        self.delta_yaw_abs_sum += abs(delta_yaw)

        # Standard deviation (noise)
        squared_delta_distance = delta_distance * delta_distance
        squared_delta_yaw = delta_yaw * delta_yaw
        std_dev_distance = math.sqrt(self.odom_noise_std[0] * squared_delta_distance + self.odom_noise_std[1] * squared_delta_yaw)
        std_dev_yaw = math.sqrt(self.odom_noise_std[2] * squared_delta_distance + self.odom_noise_std[3] * squared_delta_yaw)
        for particle in self.particles:
            # Differential drive model
            noisy_delta_distance = delta_distance + np.random.normal(0, std_dev_distance)
            noisy_delta_yaw = delta_yaw + np.random.normal(0, std_dev_yaw)
            yaw = particle.pose.yaw
            t = yaw + noisy_delta_yaw / 2.0
            
            updated_x = particle.pose.x + noisy_delta_distance * math.cos(t)
            updated_y = particle.pose.y + noisy_delta_distance * math.sin(t)
            updated_yaw = yaw + noisy_delta_yaw
						
			# update each particle	
            particle.pose.update(updated_x, updated_y, updated_yaw)


    def update_weights_by_measurement_model(
            self,
            scan_ranges: np.ndarray,
            occupancy_grid_map: np.ndarray,
            distance_map: np.ndarray,
            map_origin=(float, float),
            map_resolution=float,
    ) -> None:
        # Fixed parameters for likelihood field model
        gaussian_normalization_constant = 1.0 / math.sqrt(2.0 * math.pi * self.sigma_hit ** 2)
        gaussian_exponent_factor = 1.0 / (2 * self.sigma_hit ** 2)
        probability_rand = 1.0 / self.scan_max_range

        eps = 1e-12

        # Pre-calculate sampled beam angles and ranges
        beam_num = len(scan_ranges)
        sampled_beam_indices = np.arange(0, beam_num, self.scan_step)
        sampled_beam_angles = self.scan_min_angle + sampled_beam_indices * self.scan_angle_increment
        sampled_scan_ranges = scan_ranges[sampled_beam_indices]
        
        # Map constants
        map_height, map_width = occupancy_grid_map.shape
        map_origin_x, map_origin_y = map_origin

        # Pre-calculate particle properties in a vectorized way
        particle_yaws = np.array([p.pose.yaw for p in self.particles])
        particle_xs = np.array([p.pose.x for p in self.particles])
        particle_ys = np.array([p.pose.y for p in self.particles])
        
        # Pre-calculate trigonometric functions for particles and beams
        cos_particle_yaws = np.cos(particle_yaws)
        sin_particle_yaws = np.sin(particle_yaws)
        cos_sampled_beam_angles = np.cos(sampled_beam_angles)
        sin_sampled_beam_angles = np.sin(sampled_beam_angles)
        
        # Calculate sensor pose for all particles at once
        sensor_x = particle_xs + self.baselink_to_laser[0] * cos_particle_yaws - self.baselink_to_laser[1] * sin_particle_yaws
        sensor_y = particle_ys + self.baselink_to_laser[0] * sin_particle_yaws + self.baselink_to_laser[1] * cos_particle_yaws
        sensor_yaw = particle_yaws + self.baselink_to_laser[2]

        # Expand sensor pose to match beam dimensions for vectorized calculation
        sensor_x_expanded = np.repeat(sensor_x, len(sampled_scan_ranges)).reshape(self.particle_num, -1)
        sensor_y_expanded = np.repeat(sensor_y, len(sampled_scan_ranges)).reshape(self.particle_num, -1)
        sensor_yaw_expanded = np.repeat(sensor_yaw, len(sampled_scan_ranges)).reshape(self.particle_num, -1)
        
        # Expand beam properties
        cos_beam_expanded = np.tile(cos_sampled_beam_angles, self.particle_num).reshape(self.particle_num, -1)
        sin_beam_expanded = np.tile(sin_sampled_beam_angles, self.particle_num).reshape(self.particle_num, -1)
        range_expanded = np.tile(sampled_scan_ranges, self.particle_num).reshape(self.particle_num, -1)

        # Calculate Lidar hit points for all particles and beams
        direction_x = np.cos(sensor_yaw_expanded) * cos_beam_expanded - np.sin(sensor_yaw_expanded) * sin_beam_expanded
        direction_y = np.sin(sensor_yaw_expanded) * cos_beam_expanded + np.cos(sensor_yaw_expanded) * sin_beam_expanded

        lidar_hit_x = sensor_x_expanded + range_expanded * direction_x
        lidar_hit_y = sensor_y_expanded + range_expanded * direction_y
        
        # Convert hit points to map indices
        map_index_x = np.round((lidar_hit_x - map_origin_x) / map_resolution).astype(int)
        map_index_y = np.round((lidar_hit_y - map_origin_y) / map_resolution).astype(int)
        
        # Handle out-of-bounds indices
        is_valid_map_index = (map_index_x >= 0) & (map_index_x < map_height) & \
                            (map_index_y >= 0) & (map_index_y < map_width)

        # Get distances from distance map using valid indices
        distance_from_map = np.full(map_index_x.shape, np.inf)
        distance_from_map[is_valid_map_index] = distance_map[map_index_x[is_valid_map_index], map_index_y[is_valid_map_index]]
        
        # Vectorized calculation of known and unknown class probabilities
        probability_hit = gaussian_normalization_constant * np.exp(-(distance_from_map ** 2) * gaussian_exponent_factor) * map_resolution
        
        known_class_probability = (self.z_hit * probability_hit + self.z_rand * probability_rand) * self.known_class_prior
        
        # Exponential distribution for unknown class probability
        exp_factor = -self.unknown_class_lambda_ * range_expanded
        unknown_class_probability = self.unknown_class_lambda_ * np.exp(exp_factor) / (1 - np.exp(-self.unknown_class_lambda_ * self.scan_max_range)) * map_resolution * self.unknown_class_prior
        
        # Combine known and unknown class probabilities
        class_conditional_probability = known_class_probability + unknown_class_probability

        # Handle invalid measurements (too near, too far, inf, nan)
        is_invalid_measurement = ~((self.scan_min_range < range_expanded) & (range_expanded < self.scan_max_range)) | \
                                np.isinf(range_expanded) | np.isnan(range_expanded)
        
        class_conditional_probability[is_invalid_measurement] = self.z_max + self.z_rand * probability_rand
        
        # Sum log-likelihoods for each particle
        log_likelihoods = np.sum(np.log(class_conditional_probability + eps), axis=1)

        # Normalize log-sum-exp
        max_log_weight = np.max(log_likelihoods)
        exp_weights = np.exp(log_likelihoods - max_log_weight)
        normalized_weights = exp_weights / (np.sum(exp_weights) + eps)
        
        # Allocate weights to particles
        for index, particle in enumerate(self.particles):
            particle.weight = max(normalized_weights[index], eps)
        
        # Re-normalize weights to ensure sum is 1 (numerical stability)
        total_weight = sum(p.weight for p in self.particles)
        if total_weight > 0:
            for p in self.particles:
                p.weight /= total_weight
        
        # Average likelihood for resampling
        self.average_likelihood = np.mean([p.weight for p in self.particles])

    def estimate_robot_pose(self):
        """
        Update
        -------
        - self.estimated_x
        - self.estimated_y
        - self.estimated_yaw

        Using
        ------
        - self.particles
        """        
        xs = np.array([particle.pose.x for particle in self.particles])
        ys = np.array([particle.pose.y for particle in self.particles])
        yaws = np.array([particle.pose.yaw for particle in self.particles])
        weights = np.array([particle.weight for particle in self.particles])
        
        # Weighted sum for x and y
        self.estimated_x = np.sum(xs * weights)
        self.estimated_y = np.sum(ys * weights)
        
        # Weighted sum for yaw
        cos_yaw = np.sum(np.cos(yaws) * weights)
        sin_yaw = np.sum(np.sin(yaws) * weights)
        self.estimated_yaw = math.atan2(sin_yaw, cos_yaw)

    def calculate_amcl_random_resampling_particle_rate(self,
            average_likelihood
        ):
        self.omega_slow += self.alpha_slow * (average_likelihood - self.omega_slow)
        self.omega_fast += self.alpha_fast * (average_likelihood - self.omega_fast)
        amcl_random_particle_rate = 1.0 - self.omega_fast / self.omega_slow
        return max(amcl_random_particle_rate, 0.0)

    def calculate_effective_sample_size(self):
        weights = np.array([particle.weight for particle in self.particles])
        weight_sum = np.sum(weights**2)
        effective_sample_size = 1.0 / weight_sum
        return effective_sample_size

    def resample_particles(self):
        """
        Update
        ------
        - self.particles

        Using
        -----
        - self.particle_num
        - self.omega_slow
        - self.omega_fast
        - self.alpha_slow
        - self.alpha_slow
        - self.resample_ess_ratio
        - self.estimated_x
        - self.estimated_y
        - self.estimated_yaw
        - self.random
        """
        # Inspect effective sample size
        threshold = self.particle_num * self.resample_ess_ratio
        effective_sample_size = self.calculate_effective_sample_size()
        if effective_sample_size > threshold:
            return

        # Check movement threshold
        if self.delta_distance_abs_sum < self.delta_distance_resample_threshold and self.delta_yaw_abs_sum < self.delta_yaw_resample_threshold:
            return
        self.delta_distance_abs_sum = 0
        self.delta_yaw_abs_sum = 0

        # Normalize weights
        weights = np.array([particle.weight for particle in self.particles])
        weights /= np.sum(weights)  # normalize
        cumulative_sum = np.cumsum(weights)

        positions = (np.arange(self.particle_num) + np.random.uniform()) / self.particle_num
        indices = np.searchsorted(cumulative_sum, positions)

        # Resampled particles
        self.particles = [
            Particle(pose=Pose(
                _x=self.particles[i].pose.x,
                _y=self.particles[i].pose.y,
                _yaw=self.particles[i].pose.yaw),
                weight=1.0 / self.particle_num
            )
            for i in indices
        ]

    # def resample_particles(self):
    #     """
    #     Update
    #     ------
    #     - self.particles

    #     Using
    #     -----
    #     - self.particle_num
    #     - self.omega_slow
    #     - self.omega_fast
    #     - self.alpha_slow
    #     - self.alpha_slow
    #     - self.resample_ess_ratio
    #     """
    #     # Inspect effective sample size
    #     threshold = self.particle_num * self.resample_ess_ratio
    #     effective_sample_size = self.calculate_effective_sample_size()
    #     if effective_sample_size > threshold:
    #         return

    #     # Check movement threshold
    #     if self.delta_distance_abs_sum < self.delta_distance_resample_threshold and self.delta_yaw_abs_sum < self.delta_yaw_resample_threshold:
    #         return
    #     self.delta_distance_abs_sum = 0
    #     self.delta_yaw_abs_sum = 0

    #     # Normalize weights
    #     weights = np.array([particle.weight for particle in self.particles])
    #     weights /= np.sum(weights)  # normalize
    #     cumulative_sum = np.cumsum(weights)

    #     # Calculate the number of random resampling particles
    #     random_resampling_rate = self.calculate_amcl_random_resampling_particle_rate(self.average_likelihood)
    #     random_resampling_particle_num = int(self.particle_num * random_resampling_rate)
    #     non_random_resampling_particle_num = self.particle_num - random_resampling_particle_num

    #     # Non-random resampling
    #     positions = (np.arange(non_random_resampling_particle_num) + np.random.uniform()) / self.particle_num
    #     indices = np.searchsorted(cumulative_sum, positions)

    #     # Resample non-random particles
    #     self.particles[:non_random_resampling_particle_num] = [
    #         Particle(
    #             pose=Pose(
    #                 _x=self.particles[i].pose.x,
    #                 _y=self.particles[i].pose.y,
    #                 _yaw=self.particles[i].pose.yaw
    #             ),
    #             weight=1.0 / self.particle_num
    #         )
    #         for i in indices
    #     ]

    #     # Resample random particles
    #     xo, yo, yawo = self.estimated_x, self.estimated_y, self.estimated_yaw
    #     self.particles[non_random_resampling_particle_num:] = [
    #         Particle(
    #             pose=Pose(
    #                 _x=xo + np.random.normal(0, self.random_resampling_noise_std[0]),
    #                 _y=yo + np.random.normal(0, self.random_resampling_noise_std[1]),
    #                 _yaw=yawo + np.random.normal(0, self.random_resampling_noise_std[2])
    #             ),
    #             weight=1.0 / self.particle_num
    #         )
    #         for i in range(non_random_resampling_particle_num, self.particle_num)
    #     ]