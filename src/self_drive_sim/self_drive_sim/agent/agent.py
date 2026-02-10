from self_drive_sim.agent.interfaces import Observation, Info, MapInfo

import math
import numpy as np
import itertools
from dataclasses import dataclass, field

from .map import Map, OCCUPANCY_GRID_MAP0, OCCUPANCY_GRID_MAP1, OCCUPANCY_GRID_MAP2, OCCUPANCY_GRID_MAP3
from .reference_waypoints import all_map_reference_waypoints, all_map_distance_matrices

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

class LocalCostMapGenerator:
    def __init__(self,
            scan_point_num=241,
            scan_min_range=0.05,                 # (m)
            scan_max_range=8,                    # (m)
            scan_min_angle=-1.9,                 # (rad)
            scan_angle_increment=0.016,          # (rad)

            baselink_to_front=0.25,              # (m)
            baselink_to_rear=0.25,               # (m)
            baselink_to_right=0.25,              # (m)
            baselink_to_left=0.25,               # (m)
            baselink_to_laser=[0.147, 0.0, 0.0], # [x (m), y (m), yaw (rad)]

            map_x_length: int = 16,              # (m)
            map_y_length: int = 16,              # (m)
            map_center_x_offset: int = 0,        # (m)
            map_center_y_offset: int = 0,        # (m)
            map_resolution: float = 0.02,        # (m)
            max_cost: int = 100
        ) -> None:
        # Constants
        self.baselink_to_front = baselink_to_front
        self.baselink_to_rear = baselink_to_rear
        self.baselink_to_right = baselink_to_right
        self.baselink_to_left = baselink_to_left
        self.baselink_to_laser = baselink_to_laser
        self.scan_point_num = scan_point_num
        self.scan_min_range = scan_min_range
        self.scan_max_range = scan_max_range
        self.scan_min_angle = scan_min_angle
        self.scan_angle_increment = scan_angle_increment
        self.map_x_length = map_x_length
        self.map_y_length = map_y_length
        self.map_center_x_offset = map_center_x_offset
        self.map_center_y_offset = map_center_y_offset
        self.map_resolution = map_resolution
        self.max_cost = max_cost

    def convert_scan_to_pointclouds(self,
            scan_ranges: np.ndarray
        ) -> np.ndarray:
        """
        angle, measurement -> x, y

        Parameter
        ---------
        - scan_distance

        Using
        -----
        - self.scan_point_num
        - self.scan_min_angle
        - self.scan_angle_increment
        """
        scan_angles = self.scan_min_angle + np.arange(self.scan_point_num) * self.scan_angle_increment

        pointcloud_xs = scan_ranges * np.cos(scan_angles)
        pointcloud_ys = scan_ranges * np.sin(scan_angles)

        pointclouds = np.vstack((pointcloud_xs, pointcloud_ys)).T
        
        return pointclouds

    def preprocess_pointclouds(self,
            pointclouds: np.ndarray
        ) -> np.ndarray: 
        """
        Parameters
        ----------
        - pointcloud

        Using
        -----
        - self.scan_min_range
        - self.scan_max_range
        """
        # remove too near or too far points
        distances = np.linalg.norm(pointclouds[:, :2], axis=1)

        is_in_range = (distances >= self.scan_min_range) & (distances <= self.scan_max_range)
        return pointclouds[is_in_range]

    def transform_laser_to_robot_frame(self,
            pointclouds: np.ndarray
        ) -> np.ndarray: 
        """
        Parameter
        ---------
        - pointclouds

        Return
        ------
        np.ndarray of shape (N, 2)
            Points transformed to the robot frame.

        Using
        -----
        - self.baselink_to_laser
        """
        # Laser pose relative to robot frame
        laser_x, laser_y, laser_theta = self.baselink_to_laser

        # Rotation matrix
        cos_theta, sin_theta = np.cos(laser_theta), np.sin(laser_theta)
        rotation_matrix = np.array([[cos_theta, -sin_theta],
                                    [sin_theta,  cos_theta]])

        # Apply rotation and translation
        transformed_pointclouds = pointclouds @ rotation_matrix.T + np.array([laser_x, laser_y])

        return transformed_pointclouds

    def remove_points_within_robot(self,
            pointclouds: np.ndarray
        ) -> np.ndarray: 
        """
        robot frame coordinate
        X axis: forward
        Y axis: left

        Parameter
        ---------
        - pointcloud

        Using
        -----
        - self.baselink_to_rear
        - self.baselink_to_front
        - self.baselink_to_right
        - self.baselink_to_left
        """
        min_x = - self.baselink_to_rear
        max_x = self.baselink_to_front
        min_y = - self.baselink_to_right
        max_y = self.baselink_to_left

        is_inside_box = (
            (pointclouds[:, 0] >= min_x) & (pointclouds[:, 0] <= max_x) &
            (pointclouds[:, 1] >= min_y) & (pointclouds[:, 1] <= max_y)
        )

        return pointclouds[~is_inside_box]

    def convert_pointcloud_to_costmap(self,
            pointclouds: np.ndarray    
        ):
        """
        Parameter
        ---------
        - pointcloud

        Returns
        -------
        - costmap
        - occupied_indices

        Using
        -----
        - self.map_x_length
        - self.map_y_length
        - self.map_center_x_offset 
        - self.map_center_y_offset
        - self.map_resolution 
        - self.max_cost 
        """
        # Map dimensions in pixels
        height = int(self.map_x_length / self.map_resolution)
        width  = int(self.map_y_length / self.map_resolution)

        # Initialize costmap as 0
        costmap = np.zeros((height, width), dtype=np.float32)

        # Convert pointcloud coordinates to map indices
        col_indices = ((pointclouds[:, 1] + self.map_y_length / 2) / self.map_resolution).astype(int) # y
        row_indices = ((pointclouds[:, 0] + self.map_x_length / 2) / self.map_resolution).astype(int) # x

        # Keep only points inside the map
        valid_mask = (row_indices >= 0) & (row_indices < height) & (col_indices >= 0) & (col_indices < width)
        row_indices = row_indices[valid_mask]
        col_indices = col_indices[valid_mask]

        # Fill costmap
        costmap[row_indices, col_indices] = self.max_cost

        # Save occupied indices
        occupied_indices = list(set(zip(row_indices, col_indices)))

        return costmap, occupied_indices

    def inflate_rigid_body(self,
            costmap: np.ndarray,
            occupied_indices: list    
        ):
        """
        Parameters
        ----------
        - costmap
        - occupied_indices

        Return
        - costmap

        Using
        -----
        - self.map_resolution
        - self.max_cost
        - self.baselink_to_front
        - self.baselink_to_rear
        - self.baselink_to_right
        - self.baselink_to_left
        """
        # Offsets (robot half-dimensions in grid cells)
        front_offset = int(np.ceil(self.baselink_to_front / self.map_resolution))
        rear_offset  = int(np.ceil(self.baselink_to_rear  / self.map_resolution))
        right_offset = int(np.ceil(self.baselink_to_right / self.map_resolution))
        left_offset  = int(np.ceil(self.baselink_to_left  / self.map_resolution))

        map_height, map_width = costmap.shape

        for row, col in occupied_indices:
            # Compute inflation bounds (clamped to map size)
            row_start = max(row - right_offset, 0)
            row_end   = min(row + left_offset + 1, map_height)

            col_start = max(col - rear_offset, 0)
            col_end   = min(col + front_offset + 1, map_width)

            # Inflate costmap region
            costmap[row_start:row_end, col_start:col_end] = self.max_cost

        return costmap

    def generate_costmap(self,
            scan_ranges: np.ndarray
        ) -> np.ndarray:

        pointclouds = self.convert_scan_to_pointclouds(
            scan_ranges=scan_ranges
        )
        preprocessed_pointclouds = self.preprocess_pointclouds(
            pointclouds=pointclouds
        )
        no_robot_pointclouds = self.transform_laser_to_robot_frame(
            pointclouds=preprocessed_pointclouds
        )
        processed_pointclouds = self.remove_points_within_robot(
            pointclouds=no_robot_pointclouds
        )
        
        costmap, occupied_indices = self.convert_pointcloud_to_costmap(
            pointclouds=processed_pointclouds
        )
        costmap = self.inflate_rigid_body(
            costmap=costmap.copy(), 
            occupied_indices=occupied_indices
        )

        return costmap

class GoToGoalController:
    def __init__(self,
            max_linear_velocity=1.0,
            max_angular_velocity=2.0,
            linear_velocity_gain=1.0,
            angular_velocity_gain=2.0,
            angle_threshold=0.05
        ) -> None:
        self.max_linear_velocity = max_linear_velocity
        self.max_angular_velocity = max_angular_velocity
        self.linear_velocity_gain = linear_velocity_gain
        self.angular_velocity_gain = angular_velocity_gain
        self.angle_threshold = angle_threshold

    def go_to_goal_controller(self, 
            current_robot_pose, 
            target_position,
            is_obstacle=False,
        ) -> tuple[float, float]:
        """
        Returns
        -------
        - linear_velocity
        - angular_velocity

        Usings
        ------
        - self.angle_threshold
        - self.max_angular_velocity
        - self.max_linear_velocity
        - self.linear_velocity_gain
        - self.angular_velocity_gain
        """
        current_robot_x, current_robot_y, current_robot_yaw = current_robot_pose
        target_x, target_y = target_position

        dx = current_robot_x - target_x
        dy = current_robot_y - target_y
        d_distance = math.hypot(dx, dy)

        target_angle = math.atan2(-dy, -dx)  
        d_yaw = math.atan2(math.sin(target_angle - current_robot_yaw),
                           math.cos(target_angle - current_robot_yaw))

        if abs(d_yaw) > self.angle_threshold:
            linear_velocity = 0.0
            angular_velocity = max(-self.max_angular_velocity,
                min(self.max_angular_velocity, self.angular_velocity_gain * d_yaw))
        elif is_obstacle:
            linear_velocity = 0.0
            angular_velocity = 0.0
        else:
            linear_velocity = max(-self.max_linear_velocity,
                min(self.max_linear_velocity, self.linear_velocity_gain * d_distance))
            angular_velocity = max(-self.max_angular_velocity,
                min(self.max_angular_velocity, self.angular_velocity_gain * d_yaw))

        return linear_velocity, angular_velocity


class AutonomousNavigator:
    def __init__(self,
            initial_robot_pose,
            pollution_threshold=0.05,
            position_threshold=0.05
        ) -> None:
        # Constants
        self.pollution_threshold = pollution_threshold
        self.position_threshold = position_threshold

        # Class object
        self.mcl = MonteCarloLocalizer()
        self.local_costmap_generator = LocalCostMapGenerator()
        self.go_to_goal_controller = GoToGoalController()

        # Monte carlo localization
        self.mcl.initialize_particles(initial_robot_pose)

        # Finite state machine
        self.waypoints = []
        self.current_node_index = None
        self.optimal_next_node_index = None
        self.visited_regions = []
        self.cleaning_holding_time = 0

        # Local Planner
        self.lookahead_position_index = 0
        self.previous_waypoints = None
        self.abc = None                   # logging for debugging
    
        # [test] move_along_nodes
        self.current_waypoint_index = 0
        self.tmp_target_position = None
        self.initial_flag = True
        self.i = 0

    def mission_planner(self, 
            air_pollution_sensor_data, 
            current_node_index,
            current_time,
            pollution_end_time,
            map_id
        ):
        """
        Deadline-Conditinoal Traveling Salmesman Problem Solver

        Parameters:
        - air_pollution_sensor_data
        - current_node_index
        - current_time
        - pollution_end_time
        - map_id: 0, 1, 2, 3

        Return
        ------
        - optimal_visit_order
        """
        selected_map_distance_matrix = all_map_distance_matrices.get(map_id) 
        docking_station_node_index = len(selected_map_distance_matrix) -1 # last index in distanec matrix

        # Polluted regions
        observed_polluted_regions = [
            node_index for node_index in range(len(air_pollution_sensor_data))
            if air_pollution_sensor_data[node_index] > self.pollution_threshold
        ]

        # Regioin with no air pollution sensor
        unobserved_potential_regions = [
            node_index for node_index in range(len(air_pollution_sensor_data))
            if math.isnan(air_pollution_sensor_data[node_index])
            and node_index not in self.visited_regions
        ]

        candidate_regions = []
        if current_time > pollution_end_time: 
            candidate_regions = observed_polluted_regions + unobserved_potential_regions
        else:
            candidate_regions = observed_polluted_regions

        if not candidate_regions:
            return []
        
        # Remove ---------------------------------------------------------------------------------------------------------------------------------------
        exclude_list = []
        candidate_regions = [x for x in candidate_regions if x not in exclude_list]

        # Calculate cost for for every cases and get global minima
        optimal_visit_order = []
        min_cost = float('inf')
        for perm in itertools.permutations(candidate_regions):
            total_cost = 0
            last_visited = current_node_index

            for node_index in perm:
                total_cost += selected_map_distance_matrix[last_visited][node_index]
                last_visited = node_index

            # Add Cost for last_visited to docking station
            total_cost += selected_map_distance_matrix[last_visited][docking_station_node_index]

            if total_cost < min_cost:
                min_cost = total_cost
                optimal_visit_order = list(perm)

        return optimal_visit_order
    
    def global_planner(self, 
            start_node_index, 
            end_node_index, 
            map_id
        ):
        """
        Index rules
        - index of region is a index of matrix
        - last index is for docking station
        - (last - 1) index is for start position
        
        reference_waypoint_matrix[i][j] : waypoints from start node i to end node j

        Parameters
        ----------
        - start_node_index
        - end_node_index
        - map_id

        Returns
        -------
        - selected_map_reference_waypoints[(start_node_index, end_node_index)]
        """
        # Check validity of map_id
        if map_id not in all_map_reference_waypoints:
            return None
        
        selected_map_reference_waypoints = all_map_reference_waypoints[map_id]

        # Check validity of node indexes
        if (start_node_index, end_node_index) not in selected_map_reference_waypoints:
            return None

        return selected_map_reference_waypoints[(start_node_index, end_node_index)]

    # Main Logic
    def finite_state_machine(self,
            air_pollution_sensor_data,
            robot_pollution_sensor_data,
            scan_ranges,
            current_time,
            pollution_end_time,
            current_robot_pose, 
            current_fsm_state,
            map_id,
            room_num,
            position_threshold=0.05
        ) -> tuple[str, tuple[float, float, float]]:
        """
        Parameters
        ----------
        - air_pollution_sensor_data
        - robot_pollution_sensor_data
        - current_time
        - pollution_end_time
        - current_robot_pose
        - current_fsm_state
        - map_id
        - room_num
        - position_threshold

        Returns
        -------
        - next_fsm_state: str
        - action: (float, float, float)

        Updates & Usings
        ---
        - self.current_node_index
        - self.optimal_next_node_index
        - self.waypoints
        - self.visited_regions
        - self.pollution_threshold
        - self.cleaning_holding_time

        - self.mission_planner()
        - self.global_planner()
        - self.local_planner()
        """
        # States of finite state machine
        FSM_READY = "READY"
        FSM_CLEANING = "CLEANING"
        FSM_NAVIGATING = "NAVIGATING"
        FSM_RETURNING = "RETURNING"

        current_robot_position = current_robot_pose[0], current_robot_pose[1]

        # Indexes of initial node and docking station node by map
        initial_node_index = room_num
        docking_station_node_index = room_num + 1

        # -------------------------------------------------------------------------------------------------------------------------------------------------------- #
        # [Start State] READY ------------------------------------------------------------------------------------------------------------------------------------ #
        # -------------------------------------------------------------------------------------------------------------------------------------------------------- #
        if current_fsm_state == FSM_READY:
            # Mission planning
            self.current_node_index = initial_node_index
            optimal_visit_order = self.mission_planner(
                air_pollution_sensor_data, 
                self.current_node_index,
                current_time,
                pollution_end_time,
                map_id
            )

            # Action: stop
            action = (0, 0, 0)

            # [State Transition] READY -> NAVIGATING
            if optimal_visit_order:
                next_fsm_state = FSM_NAVIGATING

                # Global planning
                self.optimal_next_node_index = optimal_visit_order[0]
                self.waypoints = self.global_planner(
                    start_node_index=initial_node_index, 
                    end_node_index=self.optimal_next_node_index, 
                    map_id=map_id
                )
            
            # [State Transition] READY -> READY
            else:
                next_fsm_state = FSM_READY

        # -------------------------------------------------------------------------------------------------------------------------------------------------------- #
        # [Intermediate State] Navigating ------------------------------------------------------------------------------------------------------------------------ #
        # -------------------------------------------------------------------------------------------------------------------------------------------------------- #
        elif current_fsm_state == FSM_NAVIGATING:
            # Local planning
            linear_velocity, angular_velocity = self.local_planner(
                current_robot_pose=current_robot_pose,
                waypoints=self.waypoints,
                scan_ranges=scan_ranges
            )
            action = (0, linear_velocity, angular_velocity)

            # [State Transition] NAVIGATING -> CLEANING
            if self.is_target_reached(current_robot_position, self.waypoints[-1], position_threshold): 
                next_fsm_state = FSM_CLEANING
                self.current_node_index = self.optimal_next_node_index
                self.visited_regions.append(self.current_node_index)
            
            # [State Transition] NAVIGATING -> NAVIGATING
            else:
                next_fsm_state = FSM_NAVIGATING

        # --------------------------------------------------------------------------------------------------------------------------------------------------------  #
        # [Intermediate State] CLEANING --------------------------------------------------------------------------------------------------------------------------  #
        # --------------------------------------------------------------------------------------------------------------------------------------------------------  #
        elif current_fsm_state == FSM_CLEANING:
            # Mission planning
            optimal_visit_order = self.mission_planner(
                air_pollution_sensor_data, 
                self.current_node_index,
                current_time,
                pollution_end_time,
                map_id
            )

            # Action: stop and cleaning
            action = (1, 0, 0)

            # [State Transition] CLEANING -> RETURNING
            if current_time >= pollution_end_time and (air_pollution_sensor_data[self.current_node_index] <= self.pollution_threshold or (robot_pollution_sensor_data != None and robot_pollution_sensor_data <= self.pollution_threshold)) and not optimal_visit_order:
                next_fsm_state = FSM_RETURNING

                # Global planning
                self.waypoints = self.global_planner(
                    start_node_index=self.current_node_index, 
                    end_node_index=docking_station_node_index,
                    map_id=map_id
                )
            
            # [Staet Transition] CLEANING -> RETURNING (early end mode)
            # if current_time >= pollution_end_time and (air_pollution_sensor_data[self.current_node_index] <= self.pollution_threshold or (robot_pollution_sensor_data != None and robot_pollution_sensor_data <= self.pollution_threshold)):
            #     next_fsm_state = FSM_RETURNING

            #     # Global planning
            #     self.waypoints = self.global_planner(
            #         start_node_index=self.current_node_index, 
            #         end_node_index=docking_station_node_index,
            #         map_id=map_id
            #     )

            # [State Transition] CLEANING -> NAVIGATING
            elif (air_pollution_sensor_data[self.current_node_index] <= self.pollution_threshold or (robot_pollution_sensor_data != None and robot_pollution_sensor_data <= self.pollution_threshold)) and optimal_visit_order:
                if self.cleaning_holding_time >= 2:
                    next_fsm_state = FSM_NAVIGATING

                    # Global planning
                    self.optimal_next_node_index = optimal_visit_order[0]
                    self.waypoints = self.global_planner(
                        start_node_index=self.current_node_index,
                        end_node_index=self.optimal_next_node_index,
                        map_id=map_id
                    )

                    self.cleaning_holding_time = 0
                else:
                    next_fsm_state = FSM_CLEANING
                    self.cleaning_holding_time += 1

            # [State Transition] CLEANING -> CLEANING
            else:
                next_fsm_state = FSM_CLEANING

        # -------------------------------------------------------------------------------------------------------------------------------------------------------- #
        # [End State] RETURNING ---------------------------------------------------------------------------------------------------------------------------------- #
        # -------------------------------------------------------------------------------------------------------------------------------------------------------- #
        elif current_fsm_state == FSM_RETURNING:
            next_fsm_state = FSM_RETURNING
            
            # Local planning
            linear_velocity, angular_velocity = self.local_planner(
                current_robot_pose=current_robot_pose,
                waypoints=self.waypoints,
                scan_ranges=scan_ranges
            )
            action = (0, linear_velocity, angular_velocity)

        return next_fsm_state, action
    
    def localizer(self, 
            delta_distance, 
            delta_yaw, 
            scan_ranges, 
            occupancy_grid_map, 
            distance_map, 
            map_origin, 
            map_resolution
        ) -> tuple[float, float, float]:
        """
        Parameters
        ----------
        - delta_distance
        - delta_yaw
        - scan_ranges
        - occupancy_grid_map
        - distance_map
        - map_origin
        - map_resolution

        Returns
        - current_robot_pose

        Usings
        ------
        - self.mcl.update_particles_by_motion_model()
        - self.mcl.update_weights_by_measurement_model()
        - self.mcl.estimate_robto_poes()
        - self.mcl.resample_particles()
        """
        self.mcl.update_particles_by_motion_model(
            delta_distance, 
            delta_yaw
        )
        self.mcl.update_weights_by_measurement_model(
            scan_ranges=scan_ranges,
            occupancy_grid_map=occupancy_grid_map,
            distance_map=distance_map,
            map_origin=map_origin,
            map_resolution=map_resolution
        )
        self.mcl.estimate_robot_pose()
        current_robot_pose = (self.mcl.estimated_x, self.mcl.estimated_y, self.mcl.estimated_yaw)

        self.mcl.resample_particles()

        return current_robot_pose

    def local_planner(self,
            current_robot_pose,
            waypoints,
            scan_ranges: np.ndarray
        ) -> tuple[float, float]:
        """
        Parameters
        ----------
        - current_robot_pose
        - waypoints
        - scan_ranges

        Returns
        -------
        - linear_velocity
        - angular_velocity

        Usings
        ------
        - self.lookahead_position_index

        - self.is_target_reached()
        - self.go_to_goal_controller()
        """
        current_robot_position = current_robot_pose[0], current_robot_pose[1]

        if self.previous_waypoints is None or self.previous_waypoints != waypoints:
            self.lookahead_position_index = 0
            self.previous_waypoints = waypoints

        lookahead_position = waypoints[self.lookahead_position_index]
        if self.is_target_reached(
                current_position=current_robot_position,
                target_position=lookahead_position,
                position_threshold=self.position_threshold
            ) and self.lookahead_position_index < len(waypoints) - 1:
            self.lookahead_position_index += 1
            lookahead_position = waypoints[self.lookahead_position_index]         

        # Local costmap
        local_costmap = self.local_costmap_generator.generate_costmap(
            scan_ranges=scan_ranges
        )

        is_obstacle = self.perception(
            robot_pose_map=current_robot_pose,
            lookahead_position_map=lookahead_position,
            local_costmap=local_costmap
        )
        self.abc = is_obstacle

        # Control
        MAX_LINEAR_VELOCITY = 1.0
        MAX_ANGULAR_VELOCITY = 2.0
        LINEAR_VELOCITY_GAIN = 1.0
        ANGULAR_VELOCITY_GAIN = 2.0
        ANGLE_THRESHOLD = 0.05

        current_robot_x, current_robot_y, current_robot_yaw = current_robot_pose
        target_x, target_y = lookahead_position

        dx = current_robot_x - target_x
        dy = current_robot_y - target_y
        d_distance = math.hypot(dx, dy)

        target_angle = math.atan2(-dy, -dx)  
        d_yaw = math.atan2(
            math.sin(target_angle - current_robot_yaw),
            math.cos(target_angle - current_robot_yaw)
        )

        if abs(d_yaw) > ANGLE_THRESHOLD:
            linear_velocity = 0.0
            angular_velocity = max(-MAX_ANGULAR_VELOCITY,
                min(MAX_ANGULAR_VELOCITY, ANGULAR_VELOCITY_GAIN * d_yaw))
        elif is_obstacle:
            linear_velocity = 0.0
            angular_velocity = 0.0
        else:
            linear_velocity = max(-MAX_LINEAR_VELOCITY,
                min(MAX_LINEAR_VELOCITY, LINEAR_VELOCITY_GAIN * d_distance)
            )
            angular_velocity = max(-MAX_ANGULAR_VELOCITY,
                min(MAX_ANGULAR_VELOCITY, ANGULAR_VELOCITY_GAIN * d_yaw)
            )
        
            linear_velocity = min(linear_velocity, 0.3)
            # linear_velocity = 0.7

        return linear_velocity, angular_velocity

    def perception(self,
            robot_pose_map: tuple,
            lookahead_position_map: tuple,
            local_costmap: np.ndarray
        ) -> bool:
        # Local costmap constants
        height, width = local_costmap.shape
        center_row = height // 2
        center_col = width // 2
        resolution = self.local_costmap_generator.map_resolution
        max_cost = self.local_costmap_generator.max_cost

        # Map frame -> robot frame
        x_map, y_map = lookahead_position_map
        x, y, theta = robot_pose_map
        delta = np.array([x_map - x, y_map - y])
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        rotation_matrix = np.array([
            [cos_t,  sin_t],
            [-sin_t, cos_t]
        ])
        x_robot, y_robot = rotation_matrix @ delta

        # 전방 2m 검사
        pixels_ahead = int(2.0 / resolution)  # 1m / 0.02m (1픽셀 크기)
        front_region = local_costmap[
            center_row : center_row + pixels_ahead,
            center_col - 1 : center_col + 2, 
        ]
        is_1m = np.any(front_region == self.local_costmap_generator.max_cost)

        # -------------------------------
        # Bresenham (row = x, col = y)
        # -------------------------------
        row0 = center_row
        col0 = center_col
        row1 = int(round(center_row + x_robot / resolution))
        col1 = int(round(center_col + y_robot / resolution))

        dx_ = abs(col1 - col0)
        dy_ = abs(row1 - row0)
        sx = 1 if col0 < col1 else -1
        sy = 1 if row0 < row1 else -1
        err = dx_ - dy_

        row, col = row0, col0
        is_obstacle = False
        while True:
            if 0 <= row < height and 0 <= col < width:
                if local_costmap[row, col] >= max_cost:
                    is_obstacle = True
                    break
                # r_start = max(0, y_idx - 1)
                # r_end   = min(height, y_idx + 2)
                # c_start = max(0, x_idx - 1)
                # c_end   = min(width, x_idx + 2)
                # if np.any(local_costmap[r_start:r_end, c_start:c_end] == max_cost):
                #     is_obstacle = True
                #     break

            if row == row1 and col == col1:
                break

            e2 = 2 * err
            if e2 > -dy_:
                err -= dy_
                col += sx
            if e2 < dx_:
                err += dx_
                row += sy

        return is_obstacle & is_1m


    # Util functions
    def calculate_distance_to_target_position(self,
            current_position, 
            target_position
        ) -> float:
        """
        calculaet euclidean distance

        Parameters
        ----------
        - current_position: absolute x, absolute y
        - target_position

        Return
        ------
        - distance
        """
        dx = target_position[0] - current_position[0]
        dy = target_position[1] - current_position[1]
        distance = math.hypot(dx, dy)
        return distance
    
    def is_target_reached(self,
            current_position, 
            target_position, 
            position_threshold
        ):
        """
        Parameters
        ----------
        - current_position
        - target_position
        - position_threshold

        Return
        ------
        - True or False
        """
        return self.calculate_distance_to_target_position(current_position, target_position) < position_threshold

    # global plan test functoin
    def move_along_nodes(self,
            current_robot_pose,
            node_visit_queue,
            map_id,
            room_num
        ):        
        initial_node_index = room_num
        current_robot_position = current_robot_pose[0], current_robot_pose[1]

        if (self.initial_flag == True):
            self.current_node_index = initial_node_index
            self.optimal_next_node_index = node_visit_queue[0]
            self.waypoints = self.global_planner(
                initial_node_index,
                self.optimal_next_node_index,
                map_id
            )
            self.current_waypoint_index = 0
            self.tmp_target_position = self.waypoints[0]

            action = (0, 0, 0)
            self.i += 1
            self.initial_flag = False
        else:
            if self.is_target_reached(current_robot_position, self.waypoints[-1], self.position_threshold):
                self.current_waypoint_index = 0
                self.current_node_index = self.optimal_next_node_index
                self.optimal_next_node_index = node_visit_queue[self.i]
                self.waypoints = self.global_planner(
                    self.current_node_index,
                    self.optimal_next_node_index,
                    map_id
                )
                self.tmp_target_position = self.waypoints[self.current_waypoint_index]
                self.i += 1
            
            if self.is_target_reached(current_robot_position, self.tmp_target_position, self.position_threshold) and self.current_waypoint_index < len(self.waypoints) - 1:
                self.current_waypoint_index += 1
                self.tmp_target_position = self.waypoints[self.current_waypoint_index]

            linear_velocity, angular_velocity = self.go_to_goal_controller.go_to_goal_controller(
                current_robot_pose,
                self.tmp_target_position
            )

            action = (0, linear_velocity, angular_velocity)

        return action

class Agent:
    def __init__(self, logger):
        self.logger = logger
        self.steps = 0

    def initialize_map(self, map_info: MapInfo):
        """
        매핑 정보를 전달받는 함수
        시뮬레이션 (episode) 시작 시점에 호출됨

        MapInfo: 매핑 정보를 저장한 클래스
        ---
        height: int, 맵의 높이 (격자 크기 단위)
        width: int, 맵의 너비 (격자 크기 단위)
        wall_grid: np.ndarray, 각 칸이 벽인지 여부를 저장하는 2차원 배열
        room_grid: np.ndarray, 각 칸이 몇 번 방에 해당하는지 저장하는 2차원 배열
        num_rooms: int, 방의 개수
        grid_size: float, 격자 크기 (m)
        grid_origin: (float, float), 실제 world의 origin이 wall_grid 격자의 어디에 해당하는지 표시
        station_pos: (float, float), 청정 완료 후 복귀해야 하는 도크의 world 좌표
        room_names: list of str, 각 방의 이름
        pollution_end_time: float, 마지막 오염원 활동이 끝나는 시각
        starting_pos: (float, float), 시뮬레이션 시작 시 로봇의 world 좌표
        starting_angle: float, 시뮬레이션 시작 시 로봇의 각도 (x축 기준, 반시계 방향)

        is_wall: (x, y) -> bool, 해당 좌표가 벽인지 여부를 반환하는 함수
        get_room_id: (x, y) -> int, 해당 좌표가 속한 방의 ID를 반환하는 함수 (방에 속하지 않으면 -1 반환)
        get_cells_in_room: (room_id) -> list of (x, y), 해당 방에 속한 모든 격자의 좌표를 반환하는 함수
        grid2pos: (grid) -> (x, y), 격자 좌표를 실제 world 좌표로 변환하는 함수
        pos2grid: (pos) -> (grid_x, grid_y), 실제 world 좌표를 격자 좌표로 변환하는 함수
        ---
        """
        # Constant
        self.resolution = 0.02

        # Identify and initialize map
        self.map = Map()        
        if map_info.num_rooms == 2: 
            self.map_id = 0
            self.map_room_num = 2
            self.map_origin = (-14 * 0.2, -20 * 0.2)
            self.pollution_end_time = 20
            map = OCCUPANCY_GRID_MAP0
        elif map_info.num_rooms == 5: 
            self.map_id = 1
            self.map_room_num = 5
            self.map_origin = (-25 * 0.2, -25 * 0.2)
            self.pollution_end_time = 80
            map = OCCUPANCY_GRID_MAP1
        elif map_info.num_rooms == 8:
            self.map_id = 2
            self.map_room_num = 8
            self.map_origin = (-37 * 0.2, -37 * 0.2)
            self.pollution_end_time = 130
            map = OCCUPANCY_GRID_MAP2
        elif map_info.num_rooms== 13:
            self.map_id = 3
            self.map_room_num = 13
            self.map_origin = (-40 * 0.2, -50 * 0.2)
            self.pollution_end_time = 200
            map = OCCUPANCY_GRID_MAP3

        # Finite state machine
        self.current_fsm_state = "READY"

        # [Localization] Initialize robot pose by IMU sensor data
        initial_robot_pose = Pose(
            _x=map_info.starting_pos[0],
            _y=map_info.starting_pos[1],
            _yaw=map_info.starting_angle
        )
        self.autonomous_navigator = AutonomousNavigator(initial_robot_pose)

        # [Localization] Occupancy grid map and distance map
        original_map = self.map.string_to_np_array_map(map)
        self.occupancy_grid_map = self.map.upscale_occupancy_grid_map(
            original_map, 
            0.2 / self.resolution
        )
        self.distance_map = self.map.occupancy_grid_to_distance_map(
            self.occupancy_grid_map, 
            map_resolution=self.resolution
        )
    
        # True robot pose in only train
        self.true_robot_pose = (None, None, None)

        # Log for debugging
        self.distance_error_sum = 0

    def act(self, observation: Observation):
        """
        env로부터 Observation을 전달받아 action을 반환하는 함수
        매 step마다 호출됨

        Observation: 로봇이 센서로 감지하는 정보를 저장한 dict
        ---
        sensor_lidar_front: np.ndarray, 전방 라이다 (241 x 1)
        sensor_lidar_back: np.ndarray, 후방 라이다 (241 x 1)
        sensor_tof_left: np.ndarray, 좌측 multi-tof (8 x 8)
        sensor_tof_right: np.ndarray, 우측 multi-tof (8 x 8)
        sensor_camera: np.ndarray, 전방 카메라 (480 x 640 x 3)
        sensor_ray: float, 상향 1D 라이다
        sensor_pollution: float, 로봇 내장 오염도 센서
        air_sensor_pollution: np.ndarray, 거치형 오염도 센서 (방 개수 x 1)
        disp_position: (float, float), 이번 step의 로봇의 위치 변위 (오차 포함)
        disp_angle: float, 이번 step의 로봇의 각도 변위 (오차 포함)
        ---

        action: (MODE, LINEAR, ANGULAR)
        MODE가 0인 경우: 이동 명령, Twist 컨트롤로 선속도(LINEAR) 및 각속도(ANGULAR) 조절. 최대값 1m/s, 2rad/s
        MODE가 1인 경우: 청정 명령, 제자리에서 공기를 청정. LINEAR, ANGULAR는 무시됨. 청정 명령을 유지한 후 1초가 지나야 실제로 청정이 시작됨
        """
        # Observation
        air_pollution_sensor_data = observation['air_sensor_pollution'] # pollution data
        robot_pollution_sensor_data =  observation['sensor_pollution']
        delta_distance = np.linalg.norm(observation["disp_position"])   # IMU data
        delta_yaw = observation['disp_angle']
        scan_ranges = observation['sensor_lidar_front']                 # LiDAR data

        # Localization
        current_robot_pose = self.autonomous_navigator.localizer(
            delta_distance, 
            delta_yaw, 
            scan_ranges, 
            self.occupancy_grid_map, 
            self.distance_map, 
            self.map_origin, 
            self.resolution
        )
        # test
        # current_robot_pose = self.true_robot_pose

        # Current time
        dt = 0.1
        current_time = self.steps * dt

        # Finite state machine
        next_state, action = self.autonomous_navigator.finite_state_machine(
            air_pollution_sensor_data=air_pollution_sensor_data,
            robot_pollution_sensor_data=robot_pollution_sensor_data,
            scan_ranges=scan_ranges,
            current_time=current_time,
            pollution_end_time=self.pollution_end_time,
            current_robot_pose=current_robot_pose,
            current_fsm_state=self.current_fsm_state,
            map_id=self.map_id,
            room_num=self.map_room_num
        )
        self.current_fsm_state = next_state 

        # Log
        is_log = True
        if is_log:
            if self.true_robot_pose[0] != None or self.true_robot_pose[1] != None: # Train
                x_error = current_robot_pose[0] - self.true_robot_pose[0]
                y_error = current_robot_pose[1] - self.true_robot_pose[1]
                distance_error = math.hypot(x_error, y_error)
                self.distance_error_sum += distance_error

                if self.current_fsm_state == "NAVIGATING":
                    self.log(f"| {current_time:.1f} | {self.autonomous_navigator.visited_regions} [NAVIGATING] {self.autonomous_navigator.current_node_index} -> {self.autonomous_navigator.optimal_next_node_index} | Loc. Error: {distance_error:.3f}, Avg Loc. Error: {self.distance_error_sum / self.steps :.3f}")
                elif self.current_fsm_state == "CLEANING":
                    node_idx = self.autonomous_navigator.current_node_index
                    sensor_value = air_pollution_sensor_data[node_idx]
                    
                    if math.isnan(sensor_value):
                        sensor_value = robot_pollution_sensor_data

                    self.log(f"| {current_time:.1f} | {self.autonomous_navigator.visited_regions} [CLEANING] {node_idx}: {sensor_value:.3f} | Loc. Error: {distance_error:.3f}, Avg Loc. Error: {self.distance_error_sum / self.steps :.3f}")
                else:
                    self.log(f"| {current_time:.1f} | {self.autonomous_navigator.visited_regions} [{self.current_fsm_state}] | Loc. Error: {distance_error:.3f}, Avg Loc. Error: {self.distance_error_sum / self.steps :.3f}")
            else: # Test
                if self.current_fsm_state == "NAVIGATING":
                    self.log(f"| {current_time:.1f} | {self.autonomous_navigator.visited_regions} [NAVIGATING] {self.autonomous_navigator.current_node_index} -> {self.autonomous_navigator.optimal_next_node_index}")
                elif self.current_fsm_state == "CLEANING":
                    node_idx = self.autonomous_navigator.current_node_index
                    sensor_value = air_pollution_sensor_data[node_idx]
                    
                    if math.isnan(sensor_value):
                        sensor_value = robot_pollution_sensor_data

                    self.log(f"| {current_time:.1f} | {self.autonomous_navigator.visited_regions} [CLEANING] {node_idx}: {sensor_value:.3f}")

                else:
                    self.log(f"| {current_time:.1f}] | {self.autonomous_navigator.visited_regions} [{self.current_fsm_state}]")
        
        self.log(self.autonomous_navigator.abc)

        # --------------------------------------------------
        # For making reference waypoints -------------------
        # --------------------------------------------------
        # node_visit_queue = [8, 13]
        # action = self.autonomous_navigator.move_along_nodes(
        #     current_robot_pose, 
        #     node_visit_queue, 
        #     self.map_id,
        #     self.map_room_num
        # )
        # --------------------------------------------------

        self.steps += 1 
        return action

    def learn(
            self,
            observation: Observation,
            info: Info,
            action,
            next_observation: Observation,
            next_info: Info,
            terminated,
            done,
            ):
        """
        실시간으로 훈련 상태를 전달받고 에이전트를 학습시키는 함수
        training 중에만 매 step마다 호출되며(act 호출 후), test 중에는 호출되지 않음
        강한 충돌(0.7m/s 이상 속도로 충돌)이 발생하면 시뮬레이션이 종료되고 terminated에 true가 전달됨 (실격)
        도킹 스테이션에 도착하면 시뮬레이션이 종료되고 done에 true가 전달됨

        Info: 센서 감지 정보 이외에 학습에 활용할 수 있는 정보 - 오직 training시에만 제공됨
        ---
        robot_position: (float, float), 로봇의 현재 world 좌표
        robot_angle: float, 로봇의 현재 각도
        collided: bool, 현재 로봇이 벽이나 물체에 충돌했는지 여부
        all_pollution: np.ndarray, 거치형 에어 센서가 없는 방까지 포함한 오염도 정보
        ---
        """
        # Only train
        self.true_robot_pose = (info["robot_position"][0], info["robot_position"][1], info["robot_angle"])

    def reset(self):
        """
        모델 상태 등을 초기화하는 함수
        training시, 각 episode가 끝날 때마다 호출됨 (initialize_map 호출 전)
        """
        self.steps = 0

    def log(self, msg):
        """
        터미널에 로깅하는 함수. print를 사용하면 비정상적으로 출력됨.
        ROS Node의 logger를 호출.
        """
        self.logger(str(msg))