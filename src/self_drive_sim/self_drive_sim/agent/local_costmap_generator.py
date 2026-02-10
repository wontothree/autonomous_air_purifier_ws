import numpy as np

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