from self_drive_sim.agent.interfaces import Observation, Info, MapInfo

import math
import numpy as np
import itertools
from dataclasses import dataclass, field
from scipy.ndimage import distance_transform_edt

class Map:
    ORIGINAL_STRING_MAP0 = """
1111111111111111111111111111111111111111
1111111111111111111111111111111111111111
1100000000000000000000001000000000000011
1100000000000000000000000000000000000011
1100000000000000000000000000000000000011
1100000000000000000000000000000000000011
1100000000000000000000000000000000000011
1100000000000000000000000000000000000011
1100000000000000000000000000000000000011
1100000000000000000000000000000000000011
1100000000000000000000000000000000000011
1100000000000000000000000000000000000011
1100000000000000000000001000000000000011
1111111111111111111111111000000000000011
1100000000000000000000000000000000000011
1100000000000000000000000000000000000011
1100000000000000000000000000000000000011
1100000000000000000000000000000000000011
1100000000000000000000000000000000000011
1100000000000000000000000000000000000011
1100000000000000000000000000000000000011
1100000000000000000000000000000000000011
1100000000000000000000000000000000000011
1100000000000000000000000000000000000011
1100000000000000000000000000000000000011
1100000000000000000000000000000000000011
1100000000000000000000000000000000000011
1100000000000000000000000000000000000011
1111111111111111111111111111111111111111
1111111111111111111111111111111111111111
"""

    ORIGINAL_STRING_MAP1 = """
11111111111111111111111111111111111111111111111111
11111111111111111111111111111111111111111111111111
11111100001000000000001111111000000000000000000011
11111100001000000000001111111000000000000000000011
11111100001000000000001111111000000000000000000011
11111100001000000000000001000000000000000000000011
11111100001000000000000001000000000000000000000011
11111100001000000000000001000000000000000000000011
11110000001000000000000001000000000000000000000011
11110000000000000000000001000000000000000000011111
11110000000000000000000001000000000000000000011111
11110000000000000000000001000000000000000000011111
11110000000000000000000001000000000000000000011111
11110000001111111111111111000000000000000000011111
11110000000000000000000001000000000000000000011111
11110000000000000000000000000000000000000000011111
11110000000000000000000000000000000000000000011111
11110000000000000000000000000000000000000000011111
11110000000000000000000000000000000000000000011111
11111000000000000000000001111111111111111111111111
11000000000000000000000000000000000001111111111111
11000000000000000000000000000000000001111111111111
11000000000000000000000000000000000001111111111111
11000000000000000000000000000000000000000000000011
11000000000000000000000001000000000000000000000011
11000000000000000000000001000000000000000000000011
11000000000000000000000001000000000000000000000011
11000000000000000000000001000000000000000000000011
11000000000000000000000001000000000000000000000011
11000000000000000000000001000000000000000000000011
11000000000000000000000001000000000000000000000011
11000000000000000000000001000000000000000000000011
11000000000000000000000001111000000000000000000011
11000000000000000000000001111000000000000000000011
11000000000000000000000001111000000000000000000011
11000000000000000000000001111000000000000000000011
11000000000000000000000001111000000000000000000011
11000000000000000000000001111000000000000000000011
11000000000000000000000001111000000000000000000011
11000000000000000000000001111000000000000000000011
11000000000000000000000001111000000000000000000011
11111111111111110000000011111111111111111111111111
11000000000000000000000000000000000000000000011111
11000000000000000000000000000000000000000000011111
11000000000000000000000000000000000000000000011111
11000000000000000000000000000000000000000000011111
11110000000000000000000000000000000000000000011111
11110000000000000000000000000000000000000000011111
11111111111111111111111111111111111111111111111111
11111111111111111111111111111111111111111111111111
"""

    ORIGINAL_STRING_MAP2 = """
111111111111111111111111111111111111111111111111111111111000000000000000000
111111111111111111111111111111111111111111111111111111111000000000000000000
110000000000000000000000111110000001000000000000000000011000000000000000000
110000000000000000000000111110000001000000000000000000011000000000000000000
110000000000000000000000111110000001000000000000000000011000000000000000000
110000000000000000000000111110000001000000000000000000011000000000000000000
110000000000000000000000111110000000000000000000000000011000000000000000000
110000000000000000000000111110000000000000000000000000011000000000000000000
110000000000000000000000111110000000000000000000000000011000000000000000000
110000000000000000000000111110000000000000000000000000011000000000000000000
110000000000000000000000000010000001000000000000000000011000000000000000000
110000000000000000000000000000000001000000000000000000011000000000000000000
110000000000000000000000000000000001000000000000000000011111111111111111111
110000000000000000000000000000000001000000000000000000011111111111111111111
110000000000000000000000000000000001111111111111111111111111111111111111111
110000000000000000000000000010000000000000000000011111111111111111111111111
111111111111111111111111111110000000000000000000011111111111111111111111111
111111111111000000000000000010000000000000000000011111111111111111111111111
111111111111000000000000000000000000000000000000011111111111111111111111111
111111111111000000000000000000000000000000000000000000000000000000000011111
111111111111000000000000000000000000000000000000000000000000000000000011111
111111111111000000000000000000000000000000000000000000000000000000000011111
111111111111000000000000000010000000000000000000000000000000000000000011111
111000000001000000000000000010000000000000000000000000000000000000000011111
111000000001000000000000000010000000000000000000000000000000000000000011111
111000000001000000000000000010000000000000000000000000000000000000000011111
111000000001000000000000000010000000000000000000000000000000000000000011111
111000000001000000000000000010000000000000000000000000000000000000000011111
111000000001000000000000000010000000000000000000000000000000000000000011111
111000000001000000000000000010000000000000000000000000000000000000000011111
111000000001000000000000000010000000000000000000011111111111111111111111111
111000000001000000000000000010000000000000000000011111111111111111111111111
111000000001000000000000000010000000000000000000011111111111111111111111111
111000000001000000000000000010000000000000000000011111111111111111111111111
111110000111111111111111111110000000000000000000000000000000000000000001111
111100000000000000000000000010000000000000000000000000000000000000000001111
111100000000000000000000000000000000000000000000000000000000000000000001111
111100000000000000000000000000000000000000000000000000000000000000000001111
111100000000000000000000000000000000000000000000000000000000000000000001111
111100000000000000000000000000000000000000000000000000000000000000000001111
111100000000000000000000000010000000000000000000000000000000000000000001111
111100000000000000000000000010000000000000000000000000000000000000000001111
111100000000000000000000000010000000000000000000000000000000000000000001111
111100000000000000000000000010000000000000000000000000000000000000000001111
111100000000000000000000000010000000000000000000000000000000000000000000011
111100000000000000000000000010000000000000000000000000000000000000000000011
111100000000000000000000000010000000000000000000000000000000000000000000011
111100000000000000000000000010000000000000000000000000000000000000000000011
111100000000000000000000000010000000000000000000000000000000000000000000011
111100000000000000000000000010000000000000000000000000000000000000000000011
111111111111111111111111111110000000000000000000000000000000000000000000011
111111111111111111111111111110000000000000000000000000000000000000000000011
000000000000000000000000000110000000000000000000000000000000000000000000011
000000000000000000000000000110000000000000000000000000000000000000000000011
000000000000000000000000000110000000000000000000000000000000000000000000011
000000000000000000000000000110000000000000000000000000000000000000000000011
000000000000000000000000000110000000000000000000000000000000000000000000011
000000000000000000000000000110000000000000000000000000000000000000000000011
000000000000000000000000000110000000000000000000000000000000000000000000011
000000000000000000000000000110000000000000000000000000000000000000000000011
000000000000000000000000000110000000000000000000000000000000000000000000011
000000000000000000000000000110000000000000000000000000000000000000000000011
000000000000000000000000000110000000000000000000000000000000000000000000011
000000000000000000000000000110000000000000000000000000000000000000000000011
000000000000000000000000000110000000000000000000000000000000000000000000011
000000000000000000000000000111111100000000111111111111110000000011111111111
000000000000000000000000000111100000000000000000000000000000000000000001111
000000000000000000000000000111100000000000000000000000000000000000000001111
000000000000000000000000000110000000000000000000000000000000000000000000011
000000000000000000000000000110000000000000000000000000000000000000000000011
000000000000000000000000000110000000000000000000000000000000000000000000011
000000000000000000000000000111100000000000000000000000000000000000000001111
000000000000000000000000000111100000000000000000000000000000000000000001111
000000000000000000000000000111111111111111111111111111111111111111111111111
000000000000000000000000000111111111111111111111111111111111111111111111111
"""

    ORIGINAL_STRING_MAP3 = """
1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111
1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111
1100000000000000000000000000000000111111111111111100000000111100000000000000001111111100000000000011
1100000000000000000000000000000000000000000000000000000000111100000000000000001111111100000000000011
1100000000000000000000000000000000000000000000000000000000111100000000000000001111111100000000000011
1100000000000000000000000000000000000000000000000000000000111100000000000000001111111100000000000011
1100000001000000000100000000000000000000000000000000000000111100000000000000001111111100000000000011
1100000001000000000100000000000000000000000000000000000000111100000000000000001111111100000000000011
1100000001111111111111111111111111111111111111111100000000111100000000000000001111111100000000000011
1100000001111111111111111111111111111111111111111100000000111100000000000000001111111100000000000011
1100000001111111111111111111111111111111111111111100000000111111111111110000111111111100000000000011
1100000000000100000000000000000011111111111111111000000000000000000000000000000000000100000000000011
1100000000000000000000000000000011111111111111111000000000000000000000000000000000000100000000000011
1100000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000011
1100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000011
1100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000011
1100000000000100000000111110000000000000000000000000000000000000000000000000000000000000000000000011
1111111111111100000000111110000000000000000000000000000000111111111111111111100000000000000000000011
1100000000000100000000111110000000000000000000000000000000100000000000000000100000000100000000000011
1100000000000100000000111110000000000000000000000000000000100000000000000000100000000100000000000011
1100000000000100000000111110000000000000000000000000000000000000000000000000000000000100000000000011
1100000000000100000000111110000000000000000000000000000000000000000000000000000000000100000000000011
1100000000000100000000111110000000000000000000000000000000000000000000000000000000000100000000000011
1100000000000100000000111110000000000000000000000000000000000000000000000000000000000100000000000011
1100000000000100000000111110000000000000000000000000000000100000000000000000100000000100000000000011
1100000000000100000000111110000000000000000000000000000000100000000000000000100000000100000000000011
1100000000000100000000111110000000000000000000000000000000100000000000000000100000000100000000000011
1100000000000100000000111110000000000000000000111100000011111111111111111111100000000111111111111111
1100000000000100000000111111111111111111111111111100000010000000000000000000100000000100000000000011
1100000000000100000000111111111111111111111111111100000010000000000000000000000000000000000000000011
1100000000000100000000111111111111111111111111111100000010000000000000000000000000000000000000000011
1100000000000100000000111111111111111111111111111100000010000000000000000000000000000000000000000011
1100000000000100000000100000000000000000000000000100000010000000000000000000000000000000000000000011
1100000000000000000000000000000000000000000000000100000010000000000000000000100000000100000000000011
1100000000000000000000000000000000000000000000000100000010000000000000000000100000000100000000000011
1100000000000000000000000000000000000000000000000100000010000000000000000000100000000100000000000011
1100000000000000000000000000000000000000000000000100000010000000000000000000100000000100000000000011
1100000000000100000000100000000000000000000000000100000010000000000000000000100000000100000000000011
1100000000000100000000100000000000000000000000000100000010000000000000000000100000000100000000000011
1100000000000100000000100000000000000000000000000100000010000000000000000000100000000100000000000011
1100000000000100000000100000000000000000000000000100000010000000000000000000100000000100000000000011
1100000000000100000000100000000000000000000000000100000010000000000000000000100000000100000000000011
1111111111111100000000100000000000000000000000000100000010000000000000000000100000000100000000000011
1100000000000100000000100000000000000000000000000100000010000000000000000000100000000100000000000011
1100000000000100000000100000000000000000000000000100000010000000000000000000100000000100000000000011
1100000000000100000000100000000000000000000000000100000010000000000000000000100000000100000000000011
1100000000000100000000100000000000000000000000000100000010000000000000000000100000000100000000000011
1100000000000100000000111111111111111111111100001100000011000011111111111111100000000100000000000011
1100000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000011
1100000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000011
1100000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000011
1100000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000011
1100000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000011
1100000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000011
1100000000000100000000100000000000000000100000000000000000100000000000000000100000000100000000000011
1100000000000100000000100000000000000000100000000000000000100000000000000000100000000100000000000011
1100000000000100000000100000000000000000100000000000000000100000000000000000100000000100000000000011
1100000000000100000000100000000000000000100000000000000000100000000000000000100000000100000000000011
1100000000000100000000100000000000000000100000000000000000100000000000000000100000000100000000000011
1111111111111100000000100000000000000000100000000000000000100000000000000000100000000100000000000011
1100000000000100000000100000000000000000100000000000000000100000000000000000100000000100000000000011
1100000000000100000000100000000000000000100000000000000000100000000000000000100000000100000000000011
1100000000000100000000100000000000000000100000000000000000100000000000000000100000000100000000000011
1100000000000100000000100000000000000000100000000000000000100000000000000000100000000100000000000011
1100000000000100000000100000000000000000100000000000000000100000000000000000100000000100000000000011
1100000000000100000000100000000000000000100000000000000000100000000000000000100000000100000000000011
1100000000000000000000100000000000000000100000000000000000100000000000000000100000000100000000000011
1100000000000000000000100000000000000000100000000000000000100000000000000000100000000100000000000011
1100000000000000000000100000000000000000100000000000000000100000000000000000100000000100000000000011
1100000000000000000000100000000000000000100000000000000000100000000000000000100000000100000000000011
1100000000000000000000100000000000000000100000000000000000100000000000000000100000000100000000000011
1100000000000000000000100000000000000000100000000000000000100000000000000000100000000100000000000011
1100000000000100000000100000000000000000100000000000000000100000000000000000100000000000000000000011
1100000000000100000000100000000000000000100000000000000000100000000000000000100000000000000000000011
1100000000000100000000100000000000000000100000000000000000100000000000000000100000000000000000000011
1100000000000100000000100000000000000000100000000000000000100000000000000000100000000000000000000011
1100000000000100000000100000000000000000100000000000000000100000000000000000100000000100000000000011
1100000000000100000000100000000000000000100000000000000000100000000000000000100000000100000000000011
1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111
1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111
"""
        
    def make_string_distance_map(self, string_map: str) -> str:
        """
        Generate a distance map from string occupancy map and return as hex string.
        Each cell contains distance to nearest obstacle (0-9, A-F for 10-15).
        """
        # 1. String -> numpy array
        occupancy_grid = self.string_to_numpy_array_map(string_map)
        
        # 2. Distance transform (0=obstacle, 1=free)
        distance_map = distance_transform_edt(occupancy_grid == 0)
        
        # 3. Clip distances to 15 (hex F)
        distance_map_clipped = np.clip(distance_map.astype(int), 0, 15)
        
        # 4. Convert to hex string
        def num_to_hex_char(n):
            if n < 10:
                return str(n)
            else:
                return chr(ord('A') + n - 10)
        
        lines = ["".join(num_to_hex_char(val) for val in row) for row in distance_map_clipped]
        return "\n".join(lines)

    
    def string_to_numpy_array_map(self, string_map: str) -> np.ndarray:
        """
        Multiline string map ('1'=obstacle, '0'=free) -> 2D numpy array
        """
        # 문자열을 '\n'으로 split, 공백 제거, 바로 numpy로 변환
        return np.array([list(map(int, line)) 
                        for line in string_map.strip().splitlines() 
                        if line.strip()], dtype=np.int8)

    def occupancy_grid_to_distance_map(self, occupancy_grid_map: np.ndarray, map_resolution: float = 0.2) -> np.ndarray:
        """
        occupancy grid (0=free, 1=obstacle) → distance map (numpy array, meter 단위)
        """
        # free cell (0)에 대해 거리 계산
        dist_map = distance_transform_edt(occupancy_grid_map == 0)

        # 셀 단위 → 실제 거리(m) 변환
        dist_map = dist_map * map_resolution

        return dist_map


@dataclass
class Pose:
    """A data class representing the 2D pose (x, y) and orientation (yaw) of a robot or particle."""
    x: float = 0.0
    y: float = 0.0
    _yaw: float = 0.0

    def __post_init__(self):
        """Normalizes the yaw value after the object is initialized."""
        self._normalize_yaw()

    def _normalize_yaw(self):
        """Normalizes the yaw value to the range [-pi, +pi]."""
        self._yaw = math.atan2(math.sin(self._yaw), math.cos(self._yaw))

    @property
    def yaw(self):
        return self._yaw

    @yaw.setter
    def yaw(self, value):
        self._yaw = value
        self._normalize_yaw()

    def update(self, x: float, y: float, yaw: float):
        """Updates the pose values at once."""
        self.x = x
        self.y = y
        self.yaw = yaw

@dataclass
class Particle:
    """
    A data class representing a single particle with a pose and a weight
    """
    pose: Pose = field(default_factory=Pose)
    weight: float = 0.0

class ParticleFilter:
    """
    A basic template for a Particle Filter Localization class.
    """
    def __init__(self,
        particle_num=100,
        initial_x_noise=0.02,
        initial_y_noise=0.02,
        initial_yaw_noise=0.02,
        odom_noise=[0.1, 0.01, 0.01, 0.1]
        ):
        
        self.particle_num = particle_num
        self.initial_x_noise=initial_x_noise
        self.initial_y_noise=initial_y_noise
        self.initial_yaw_noise=initial_yaw_noise
        self.odom_noise = odom_noise

        self.particle_set = []

    def initialize_particles(self, initial_pose: Pose):
        """
        Parameters
        ----------
            - initial_pose: initial pose of robot (x, y, yaw)

        Update
        ------
            - particle_set: particle (pose, weight) set
        
        Using
        -----
            - self.particle_num
            - self.initial_x_noise
            - self.initial_y_noise
            - self.initial_yaw_noise
        """
        # 벡터 연산으로 모든 파티클의 Pose 값을 한 번에 계산
        particle_x_set = initial_pose.x + np.random.normal(0, self.initial_x_noise, self.particle_num)
        particle_y_set = initial_pose.y + np.random.normal(0, self.initial_y_noise, self.particle_num)
        particle_yaw_set = initial_pose.yaw + np.random.normal(0, self.initial_yaw_noise, self.particle_num)
        
        initial_weight = 1.0 / self.particle_num

        # for 루프와 .append() 대신 리스트 컴프리헨션 사용
        self.particle_set = [
            Particle(
                pose=Pose(x=particle_x_set[i], y=particle_y_set[i], _yaw=particle_yaw_set[i]),
                weight=initial_weight
            )
            for i in range(self.particle_num)
        ]

    def update_particles_by_motion_model(self,
        delta_distance: float, 
        delta_yaw: float
        ):
        """
        Parameters
        ----------
            - delta_x: measured by IMU
            - delta_y
            - delta_yaw

        Update
        ------
            - self.particle_set
        
        Using
        -----
            - odom_noise
        """    
        # Standard deviation (noise)
        squared_delta_distance = delta_distance * delta_distance
        squared_delta_yaw = delta_yaw * delta_yaw
        std_dev_distance = math.sqrt(self.odom_noise[0] * squared_delta_distance + self.odom_noise[1] * squared_delta_yaw)
        std_dev_yaw = math.sqrt(self.odom_noise[2] * squared_delta_distance + self.odom_noise[2] * squared_delta_yaw)

        for particle in self.particle_set:
            noisy_delta_distance = delta_distance + np.random.normal(0, std_dev_distance)
            noisy_delta_yaw = delta_yaw + np.random.normal(0, std_dev_yaw)

            # A pose of a particle
            yaw = particle.pose.yaw
            t = yaw + noisy_delta_yaw / 2.0
            x = particle.pose.x + noisy_delta_distance * math.cos(t)
            y = particle.pose.y + noisy_delta_distance * math.sin(t)
            yaw += noisy_delta_yaw

            particle.pose.update(x, y, yaw)

    def update_weights_by_measurement_model(
        self,
        scan_ranges: np.ndarray,
        occupancy_grid_map: np.ndarray,
        distance_map: np.ndarray,

        # LiDAR
        min_angle=-1.9,
        angle_increment=0.016,
        min_range=0.05,
        max_range=8.0,
        scan_step=5, # or beam_sample_count
        
        # Map
        map_origin=(14, 20),
        map_resolution=0.2,

        sigma_hit=0.2,
        z_hit=0.95,
        z_rand=0.05
        ):
        """
        Parameters
        ----------
            - scan
            - occupancy_grid_map
            - distance_map

        Update
        ------
            - self.particle_set
        """   
        eps = 1e-12

        beam_num = len(scan_ranges)
        sampled_beam_indices = np.arange(0, beam_num, scan_step)
        sampled_beam_angles = min_angle + sampled_beam_indices * angle_increment
        sampled_scan_ranges = scan_ranges[sampled_beam_indices]
        
        denominator = 2.0 * (sigma_hit ** 2)
        inv_denominator = 1.0 / denominator
        
        # Map constant
        map_height, map_width = occupancy_grid_map.shape
        map_origin_x, map_origin_y = map_origin
        
        # Initialize particle weight
        particle_num = len(self.particle_set)
        log_weights = np.zeros(particle_num, dtype=np.float64)
        
        # previous
        particle_yaws = np.array([particle.pose.yaw for particle in self.particle_set])
        particle_cos_yaws = np.cos(particle_yaws)
        particle_sin_yaws = np.sin(particle_yaws)
        
        particle_xs = np.array([particle.pose.x for particle in self.particle_set])
        particle_ys = np.array([particle.pose.y for particle in self.particle_set])
        
        # Previous calculation cos and sine for beam angle
        cos_sampled_beam_angles = np.cos(sampled_beam_angles)
        sin_sampled_beam_angles = np.sin(sampled_beam_angles)
        
        for particle_index in range(particle_num):
            particle_x = particle_xs[particle_index]
            particle_y = particle_ys[particle_index]
            cos_yaw = particle_cos_yaws[particle_index]
            sin_yaw = particle_sin_yaws[particle_index]
            
            log_likelihood = 0.0
            
            for beam_index in range(len(sampled_scan_ranges)):
                range_measurement = sampled_scan_ranges[beam_index]
                
                # Check validity of range
                if not (min_range < range_measurement < max_range) or np.isinf(range_measurement) or np.isnan(range_measurement):
                    log_likelihood += math.log(z_rand + eps)
                    continue
                
                # LiDAR endpoint
                direction_x = cos_yaw * cos_sampled_beam_angles[beam_index] - sin_yaw * sin_sampled_beam_angles[beam_index]
                direction_y = sin_yaw * cos_sampled_beam_angles[beam_index] + cos_yaw * sin_sampled_beam_angles[beam_index]
                
                lidar_hit_x = particle_x + range_measurement * direction_x
                lidar_hit_y = particle_y + range_measurement * direction_y
                
                # Index
                map_index_x = int(round((lidar_hit_x - map_origin_x) / map_resolution))
                map_index_y = int(round((lidar_hit_y - map_origin_y) / map_resolution))
                
                if 0 <= map_index_x < map_width and 0 <= map_index_y < map_height:
                    distance_in_cells = distance_map[map_index_y, map_index_x]
                    distance_in_meters = float(distance_in_cells) * map_resolution
                    
                    prob_hit = math.exp( -(distance_in_meters ** 2) * inv_denominator)
                    total_prob = z_hit * prob_hit + z_rand * (1.0 / max_range)
                    
                    log_likelihood += math.log(total_prob + eps)
                else:
                    log_likelihood += math.log(z_rand + eps)
        
            log_weights[particle_index] = log_likelihood
            
        # Normalize log-sum-exp
        max_log_weight = np.max(log_weights)
        exp_weights = np.exp(log_weights - max_log_weight)
        normalized_weights = exp_weights / (np.sum(exp_weights) + eps)
        
        # allocate weight to particle
        for index, particle in enumerate(self.particle_set):
            particle.weight = max(normalized_weights[index], eps)
        
        total_weight = sum(particle.weight for particle in self.particle_set)
        if total_weight > 0:
            for particle in self.particle_set:
                particle.weight /= total_weight

    def estimate_robot_pose(self):
        """
        Estimate robot pose from particles.

        Returns
        -------
        pose : dict
            - x: estimated x position
            - y: estimated y position
            - yaw: estimated orientation
        """
        # 파티클 수
        particle_num = len(self.particle_set)
        
        # 각 파티클의 위치와 yaw, 가중치 배열 생성
        xs = np.array([p.pose.x for p in self.particle_set])
        ys = np.array([p.pose.y for p in self.particle_set])
        yaws = np.array([p.pose.yaw for p in self.particle_set])
        weights = np.array([p.weight for p in self.particle_set])
        
        # 가중 평균 계산
        x_est = np.sum(xs * weights)
        y_est = np.sum(ys * weights)
        
        # yaw는 cos/sin 평균 후 atan2
        cos_yaw = np.sum(np.cos(yaws) * weights)
        sin_yaw = np.sum(np.sin(yaws) * weights)
        yaw_est = math.atan2(sin_yaw, cos_yaw)
        
        return {"x": x_est, "y": y_est, "yaw": yaw_est}


    def resample_particles(self):
        """
        Low-variance resampling of particles based on their weights.
        """
        N = len(self.particle_set)
        weights = np.array([p.weight for p in self.particle_set])
        weights /= np.sum(weights)  # normalize

        positions = (np.arange(N) + np.random.uniform()) / N
        cumulative_sum = np.cumsum(weights)
        indexes = np.searchsorted(cumulative_sum, positions)

        # 새로운 파티클 집합 생성
        self.particle_set = [
            Particle(pose=Pose(
                x=self.particle_set[i].pose.x,
                y=self.particle_set[i].pose.y,
                _yaw=self.particle_set[i].pose.yaw),
                weight=1.0 / N
            )
            for i in indexes
        ]

# ----------


class Agent:
    def __init__(self, logger):
        self.logger = logger
        self.steps = 0

        self.current_robot_pose = (0.0, 0.0, 0.0)

        # Finite state machine
        self.current_fsm_state = "READY"
        self.waypoints = []
        self.current_waypoint_index = 0
        self.tmp_target_position = None
        self.optimal_next_node_index = None
        self.current_node_index = None

        # test
        self.particle_filter = ParticleFilter()
        initial_pose = Pose(2.0, 0.0, 1.5708)
        self.particle_filter.initialize_particles(initial_pose)

        map_obj = Map()
        self.occupancy_grid_map = map_obj.string_to_numpy_array_map(Map.ORIGINAL_STRING_MAP0)
        self.distance_map = map_obj.occupancy_grid_to_distance_map(self.occupancy_grid_map)


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
        self.map_info = map_info
        self.pollution_end_time = map_info.pollution_end_time

        # Initialize robot pose
        initial_robot_position = map_info.starting_pos
        initial_robot_yaw = map_info.starting_angle
        self.current_robot_pose = (initial_robot_position[0], initial_robot_position[1], initial_robot_yaw)

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

        # --------------------------------------------------
        # Observation
        air_sensor_pollution_data = observation['air_sensor_pollution']
        robot_sensor_pollution_data =  observation['sensor_pollution']
        pollution_end_time = self.pollution_end_time
        current_robot_pose = self.current_robot_pose

        # localization
        delta_distance = np.linalg.norm(observation["disp_position"])
        self.particle_filter.update_particles_by_motion_model(delta_distance=delta_distance, delta_yaw=observation["disp_angle"])
        self.particle_filter.update_weights_by_measurement_model(
            scan_ranges=observation["sensor_lidar_front"],
            occupancy_grid_map=self.occupancy_grid_map,
            distance_map=self.distance_map
        )
        estimated_pose = self.particle_filter.estimate_robot_pose()
        self.particle_filter.resample_particles()

        # self.log(f"True: {self.current_robot_pose}, estimation: {estimated_pose}")
        # 현재 로봇 포즈와 추정 포즈
        true_x, true_y, true_yaw = self.current_robot_pose[0], self.current_robot_pose[1], self.current_robot_pose[2]
        est_x, est_y, est_yaw = estimated_pose['x'], estimated_pose['y'], estimated_pose['yaw']

        # 거리 차이
        distance_error = math.hypot(est_x - true_x, est_y - true_y)

        # yaw 차이 ([-pi, pi] 범위로 정규화)
        yaw_error = (est_yaw - true_yaw + math.pi) % (2 * math.pi) - math.pi

        # 출력
        self.log(f"Distance error: {distance_error:.3f} m, Yaw error: {math.degrees(yaw_error):.2f} deg")



        # Current time
        dt = 0.1
        current_time = self.steps * dt

        next_state, action = self.finite_state_machine(
            air_sensor_pollution_data,
            robot_sensor_pollution_data,
            current_time,
            pollution_end_time,
            current_robot_pose,
            self.current_fsm_state,
            map_id=0
        )
        self.current_fsm_state = next_state 
        # --------------------------------------------------

        # --------------------------------------------------
        # --------------------------------------------------
        # ------- 여기만 테스트 하세요 아빠 -------------------
        # --------------------------------------------------
        # --------------------------------------------------
        # Global planning
        # waypoints = self.global_planner(5, 1, map_id=1) 
        # self.target_position = waypoints[self.current_waypoint_index]

        # linear_velocity, angular_velocity = self.controller(self.current_robot_pose, self.target_position)
        # action = (0, linear_velocity, angular_velocity)

        # # Distance between current position and target position
        # distance_to_target_position = math.hypot(self.target_position[0] - self.current_robot_pose[0],
        #                                         self.target_position[1] - self.current_robot_pose[1])
        # if distance_to_target_position < 0.1 and self.current_waypoint_index < len(waypoints) - 1:
        #     self.current_waypoint_index += 1
        #     self.target_positionn = waypoints[self.current_waypoint_index]
        # --------------------------------------------------
        # --------------------------------------------------
        # --------------------------------------------------
        # --------------------------------------------------
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
        # Only simulation
        self.current_robot_pose = (info["robot_position"][0], info["robot_position"][1], info["robot_angle"])

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

    # ----------------------------------------------------------------------------------------------------
    # New defined functions

    def mission_planner(self, air_sensor_pollution_data, robot_sensor_pollution_data, current_node_index, map_id):
        """
        미션 플래너: 오염 감지된 방들을 기반으로 TSP 순서에 따라 task queue 생성

        Parameters:
            - air_sensor_pollution_data: list of float, 각 방의 공기 센서 오염 수치
            - robot_sensor_pollution_data: list of float, 로봇 센서 오염 수치 (현재 사용 안 함)
            - current_node_index: int, 현재 방(room)의 ID
            - map_id: 0, 1, 2, 3

        Return:
            - best_path: List[int], 방문해야 할 방의 순서
        """
        distance_matrices = {
            0: np.array([
                [0, 29, 12, 43],
                [29, 0, 33, 27],
                [12, 33, 0, 15],
                [43, 27, 15, 0]
            ]),
            1: np.array([
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
            ]),  
            2: np.array([]),
            3: np.array([]),
        }

        unobserved_potential_regions = []

        # Polluted regions
        observed_polluted_regions = [
            room_id for room_id in range(self.map_info.num_rooms)
            if air_sensor_pollution_data[room_id] > 0
        ]

        if not observed_polluted_regions:
            return

        distance_matrix = distance_matrices.get(map_id) 
        dock_station_id = distance_matrix.shape[0] - 1  # 마지막 인덱스가 도킹 스테이션

        min_cost = float('inf')
        optimal_visit_order = []

        # Calculate cost for every cases
        for perm in itertools.permutations(observed_polluted_regions):
            total_cost = 0
            last_visited = current_node_index

            for room_id in perm:
                total_cost += distance_matrix[last_visited][room_id]
                last_visited = room_id

            # Add Cost for last_visited to docking station
            total_cost += distance_matrix[last_visited][dock_station_id]

            if total_cost < min_cost:
                min_cost = total_cost
                optimal_visit_order = list(perm)

        return optimal_visit_order
    
    def global_planner(self, start_node_index, end_node_index, map_id):
        """
        Index rules
        - index of region is a index of matrix
        - last index is for docking station
        - (last - 1) index is for start position
        
        reference_waypoint_matrix[i][j] : waypoints from start node i to end node j
        """

        reference_waypoints = {
            0: {
                (0, 1): [(-1, 2), (-1, -2)],
                (0, 3): [(1.4, -3)],
                (1, 0): [(-1, 1.6), (1, 1.8)],
                (1, 3): [(-0.8, 1.6), (0.2, 1.2), (1.4, -3)],
                (2, 0): [(1, 1.8)],
                (2, 1): [(0.2, 1.6), (-0.8, 1.6), (-1, -2)],
            },
            1: {
                (5, 1): [(-0.8, -0.8), (0, -0.8)]
            },  # map_id=1의 waypoint 정의 가능
            2: {},  # map_id=2의 waypoint 정의 가능
            3: {},  # map_id=3의 waypoint 정의 가능
        }

        map_reference_waypoints = reference_waypoints[map_id]
        
        # Check validity of map_id
        if map_id not in reference_waypoints:
            return None

        # Check validity of node indexes
        if (start_node_index, end_node_index) not in map_reference_waypoints:
            return None

        return map_reference_waypoints[(start_node_index, end_node_index)]

    def controller(self, current_robot_pose, target_position, 
                linear_gain=1.0, angular_gain=2.0, 
                max_linear=1.0, max_angular=1.0,
                angle_threshold=0.1):
        """
        목표 방향을 먼저 향하도록 하고, 방향이 맞으면 직진.
        """
        x, y, theta = current_robot_pose
        target_x, target_y = target_position

        dx = target_x - x
        dy = target_y - y
        distance = math.hypot(dx, dy)
        target_angle = math.atan2(dy, dx)

        # 방향 오차
        angle_error = target_angle - theta
        angle_error = math.atan2(math.sin(angle_error), math.cos(angle_error))  # -pi ~ pi

        # 방향 우선 제어
        if abs(angle_error) > angle_threshold:
            linear_velocity = 0.0  # 먼저 회전
            angular_velocity = max(-max_angular, min(max_angular, angular_gain * angle_error))
        else:
            linear_velocity = max(-max_linear, min(max_linear, linear_gain * distance))
            angular_velocity = max(-max_angular, min(max_angular, angular_gain * angle_error))

        return linear_velocity, angular_velocity


    # Main Logic
    def finite_state_machine(self,
            air_sensor_pollution_data,
            robot_sensor_pollution_data,
            current_time,
            pollution_end_time,
            current_robot_pose, 
            current_fsm_state,
            map_id 
            ):
        """
        current_fsm_state -> next_fsm_state, action
        """
        # Define states of finite state machine
        FSM_READY = "READY"
        FSM_CLEANING = "CLEANING"
        FSM_NAVIGATING = "NAVIGATING"
        FSM_RETURNING = "RETURNING"

        # Indexes of initial node and docking station node by map
        if map_id == 0:
            initial_node_index = 2
            docking_station_node_index = 3
        elif map_id == 1:
            initial_node_index = 5
            docking_station_node_index = 6
        elif map_id == 2:
            initial_node_index = 8
            docking_station_node_index = 9
        elif map_id == 3:
            initial_node_index = 13
            docking_station_node_index = 14

        def calculate_distance_to_target_position(current_position, target_position):
            """
            Euclidean distance
            """
            dx = target_position[0] - current_position[0]
            dy = target_position[1] - current_position[1]
            distance = math.hypot(dx, dy)
            return distance

        def is_target_reached(current_position, target_position, threshold=0.05):
            """
            Decide if robot reached in target position by threshold
            """
            return calculate_distance_to_target_position(current_position, target_position) < threshold
        
        def are_no_polluted_rooms(air_sensor_pollution_data):
            """
            Decide if there are polluted rooms
            """
            return all(pollution <= 0 for pollution in air_sensor_pollution_data) # True: there are no polluted rooms

        current_robot_position = current_robot_pose[0], current_robot_pose[1]

        # ---------------------------------------------------------------------------- #
        # [State] READY (state state) ------------------------------------------------ #
        # ---------------------------------------------------------------------------- #
        if current_fsm_state == FSM_READY:
            # Mission planning
            optimal_visit_order = self.mission_planner(
                air_sensor_pollution_data, 
                robot_sensor_pollution_data, 
                current_node_index=initial_node_index,
                map_id=map_id
                )

            # State transition
            # READY -> NAVIGATING
            if optimal_visit_order != None: # 목표 구역이 있음
                next_fsm_state = FSM_NAVIGATING
                self.optimal_next_node_index = optimal_visit_order[0]
                self.waypoints = self.global_planner(start_node_index=initial_node_index, end_node_index=self.optimal_next_node_index, map_id=map_id)
                self.current_waypoint_index = 0
                self.tmp_target_position = self.waypoints[0]
            
            # READY -> READY
            else:
                next_fsm_state = FSM_READY

            action = (0, 0, 0) # Stop

        # ---------------------------------------------------------------------------- #
        # [State] Navigating --------------------------------------------------------- #
        # ---------------------------------------------------------------------------- #
        elif current_fsm_state == FSM_NAVIGATING:
            # State transition
            # NAVIGATING -> CLEANING
            if is_target_reached(current_robot_position, self.waypoints[-1]): # 목표 구역에 도달함
                next_fsm_state = FSM_CLEANING
                self.current_waypoint_index = 0
                self.current_node_index = self.optimal_next_node_index
            
            # NAVIGATING -> NAVIGATING
            else:
                next_fsm_state = FSM_NAVIGATING

            # set next tempt target point 이 순서를 바꾸면 왜 문제가 생길까?
            if is_target_reached(current_robot_position, self.tmp_target_position) and self.current_waypoint_index < len(self.waypoints) - 1:
                self.current_waypoint_index += 1
                self.tmp_target_position = self.waypoints[self.current_waypoint_index]

            linear_velocity, angular_velocity = self.controller(current_robot_pose, self.tmp_target_position)
            action = (0, linear_velocity, angular_velocity)

        # ---------------------------------------------------------------------------- #
        # [State] CLEANING ----------------------------------------------------------- #
        # ---------------------------------------------------------------------------- #
        elif current_fsm_state == FSM_CLEANING:
            # Mission planning
            optimal_visit_order = self.mission_planner(
                air_sensor_pollution_data, 
                robot_sensor_pollution_data, 
                current_node_index=self.current_node_index,
                map_id=map_id
                )          

            # State transition
            # CLEANING -> RETURNING
            if are_no_polluted_rooms(air_sensor_pollution_data) and current_time >= pollution_end_time:     # 오염 구역이 없음
                next_fsm_state = FSM_RETURNING

                self.current_waypoint_index = 0
                self.waypoints = self.global_planner(start_node_index=self.current_node_index, end_node_index=docking_station_node_index, map_id=map_id)
                self.tmp_target_position = self.waypoints[0]

            # CLEANING -> NAVIGATING
            elif air_sensor_pollution_data[self.optimal_next_node_index] == 0 and optimal_visit_order != None:       # 청정 완료함
                next_fsm_state = FSM_NAVIGATING

                # 꼭 이때 해야 할까?
                self.optimal_next_node_index = optimal_visit_order[0]
                self.waypoints = self.global_planner(start_node_index=self.current_node_index, end_node_index=self.optimal_next_node_index, map_id=map_id)
                self.current_waypoint_index = 0
                self.tmp_target_position = self.waypoints[0]  

            # CLEANING -> CLEANING
            else:
                next_fsm_state = FSM_CLEANING
            
            action = (1, 0, 0) # Stop and clean

        # ---------------------------------------------------------------------------- #
        # [State] RETURNING (end state) ---------------------------------------------- #
        # ---------------------------------------------------------------------------- #
        elif current_fsm_state == FSM_RETURNING:
            next_fsm_state = FSM_RETURNING
            
            linear_velocity, angular_velocity = self.controller(current_robot_pose, self.tmp_target_position)
            action = (0, linear_velocity, angular_velocity)

            if is_target_reached(current_robot_position, self.tmp_target_position) and self.current_waypoint_index < len(self.waypoints) - 1:
                self.current_waypoint_index += 1
                self.tmp_target_position = self.waypoints[self.current_waypoint_index]

        # log
        self.log(f"{current_time:.1f}: {current_fsm_state}")

        return next_fsm_state, action

# ----------------------------------------------------------------------------------------------------
# import os
# from self_drive_sim.simulation.floor_map import FloorMap

# def print_global_map(map_path, is_coordinate=True, step=5):
#     """
#     주어진 FloorMap 파일을 읽고, 터미널용 맵을 출력합니다.
#     - 벽: '#'
#     - 빈 공간: '.'
#     - 방 ID: 0-9, 10-15 -> A-F (16진수)
#     - 스테이션: 'D'
#     - 좌표계: 상단(y), 좌측(x) 5단위 표시

#     using method ---

#     map_file = './../../worlds/map2.npz'
#     print_global_map(map_file, step=5)

#     """
#     # Generate FloorMap
#     if not os.path.exists(map_path):
#         raise FileNotFoundError(f"Map file not found: {map_path}")
#     floor_map = FloorMap.from_file(map_path)

#     # Generate Mapinfo
#     pollution_end_time = 10.0
#     station_pos = floor_map.station_pos
#     starting_angle = 0.0
#     map_info = floor_map.to_map_info(pollution_end_time, station_pos, starting_angle)

#     wall_grid = map_info.wall_grid
#     station_pos = map_info.station_pos

#     height, width = wall_grid.shape
#     grid_display = np.full((height, width), '0', dtype=str)  # 빈 공간은 '.'

#     # 벽 표시
#     grid_display[wall_grid] = '1'

#     # Room ID (0-9, 10-15 -> A-F)
#     def room_id_to_char(room_id):
#         if room_id < 10:
#             return str(room_id)
#         else:
#             return chr(ord('A') + (room_id - 10) % 6)  # 10->A, 11->B ... 15->F, 16->A 반복

#     for room_id in range(map_info.num_rooms):
#         cells = map_info.get_cells_in_room(room_id)
#         char = room_id_to_char(room_id)
#         for x, y in cells:
#             # 벽이나 스테이션이 아니면 덮어쓰기
#             if grid_display[x, y] not in ('#', 'D', 'O', 'S'):
#                 grid_display[x, y] = '0'

#     # Station
#     gx, gy = map_info.pos2grid(station_pos)
#     row = int(gx)
#     col = int(gy)
#     if 0 <= row < height and 0 <= col < width:
#         grid_display[row, col] = '0'

#     # Origin (grid 좌표 그대로 사용)
#     # ox, oy = map_info.grid_origin
#     # ox, oy = int(ox), int(oy)
#     # if 0 <= ox < height and 0 <= oy < width:
#     #     grid_display[ox, oy] = 'O'

#     # Starting position (world → grid 변환 필요)
#     if (map_info.height == 30 and map_info.width == 40):    # Map0
#         sx, sy = map_info.pos2grid((2.0, 0.0))
#     elif (map_info.height == 50 and map_info.width == 50):  # Map1
#         sx, sy = map_info.pos2grid((0.0, -2.0))
#     elif (map_info.height == 75 and map_info.width == 75):  # Map2
#         sx, sy = map_info.pos2grid((3.0, 3.0))
#     elif (map_info.height == 80 and map_info.width == 100): # Map3
#         sx, sy = map_info.pos2grid((-7.2, 5.0))
#     sx, sy = int(sx), int(sy)
#     if 0 <= sx < height and 0 <= sy < width:
#         grid_display[sx, sy] = '0'

#     # Print
#     if is_coordinate: # Map with coordinate
#         y_labels = "  _|"

#         # Y
#         for j in range(0, width, step):
#             label = f"{int(j-width/2):<{step-1}}|"
#             y_labels += label
#         y_labels += 'Y'
#         print(y_labels)

#         # X
#         x_center = height // 2

#         for i in range(height):
#             x_val = i - x_center

#             # x축 레이블: 숫자만, 공백 없음
#             if i % step == 0:
#                 x_label = f"{x_val:>3} "  # 3칸 폭 안에서 숫자 정렬
#             elif (i + 1) % step == 0:
#                 x_label = "  _ "          # 구분선 3칸
#             else:
#                 x_label = "    "          # 나머지 공백 3칸

#             map_row = ''.join(grid_display[i, :])
#             line = x_label + map_row
#             print(line)
#         print('  X')

            
#     else: # Map without coordinate
#         for i in range(height):
#             line = ''.join(grid_display[i, :])
#             print(line)

# if __name__ == '__main__':
#     map_file = './../../worlds/map1.npz'
#     print_global_map(map_file, is_coordinate=True, step=5)
