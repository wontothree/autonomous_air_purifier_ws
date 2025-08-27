from self_drive_sim.agent.interfaces import Observation, Info, MapInfo

# print global map
import numpy as np
import os
from self_drive_sim.simulation.floor_map import FloorMap

# ROS dependencies
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import PoseWithCovarianceStamped
from nav_msgs.msg import Odometry                          # Odometry
import tf_transformations                                  # quaternion -> euler
from geometry_msgs.msg import Pose as RosPose, PoseArray   # (visualization)


# python dependencies
from dataclasses import dataclass, field
import math
import random                                           # random.gauss

@dataclass
class Pose:
    x: float = 0.0
    y: float = 0.0
    _yaw: float = 0.0

    def __post_init__(self):
        self._normalize_yaw()

    def _normalize_yaw(self):
        """Yaw를 -pi ~ pi 범위로 정규화"""
        while self._yaw < -math.pi:
            self._yaw += 2.0 * math.pi
        while self._yaw > math.pi:
            self._yaw -= 2.0 * math.pi

    @property
    def yaw(self):
        return self._yaw

    @yaw.setter
    def yaw(self, value):
        self._yaw = value
        self._normalize_yaw()

    def set_pose(self, x: float, y: float, yaw: float):
        self.x = x
        self.y = y
        self.yaw = yaw

@dataclass
class Particle:
    pose: Pose = field(default_factory=Pose)
    weight: float = 0.0

    @property
    def x(self):
        return self.pose.x

    @x.setter
    def x(self, value):
        self.pose.x = value

    @property
    def y(self):
        return self.pose.y

    @y.setter
    def y(self, value):
        self.pose.y = value

    @property
    def yaw(self):
        return self.pose.yaw

    @yaw.setter
    def yaw(self, value):
        self.pose.yaw = value

# ----------------------------------------------------------------------------------------------------

class RMCLocalizer:
    def __init__(self, particles_num=100,
                 initial_noise_x=0.02,
                 initial_noise_y=0.02,
                 initial_noise_yaw=0.02
                 ):
        
        # particle
        self._particles_num = particles_num
        self._initial_noise_x = initial_noise_x
        self._initial_noise_y = initial_noise_y
        self._initial_noise_yaw = initial_noise_yaw

        # pose
        self.rmcl_estimated_pose = Pose()

    def initialize_particle_set(self):
        """입자를 초기 포즈 주변에 무작위로 분포시키고 가중치를 균등하게 설정"""
        xo, yo, yawo = self.rmcl_estimated_pose.x, self.rmcl_estimated_pose.y, self.rmcl_estimated_pose.yaw
        wo = 1.0 / self._particles_num

        self.particle_set = []
        for _ in range(self._particles_num):
            x = xo + random.gauss(0, self._initial_noise_x)
            y = yo + random.gauss(0, self._initial_noise_y)
            yaw = yawo + random.gauss(0, self._initial_noise_yaw)

            # set pose and weight
            particle = Particle(Pose(x, y, yaw), wo)
            self.particle_set.append(particle)

    def initialize_reliability_set(self):
        """입자별 신뢰도 배열을 초기값 0.5로 초기화"""
        self.reliability_set = [0.5] * self._particles_num
    
    def update_pose_and_reliability_set(self):
        pass

# ----------------------------------------------------------------------------------------------------

class RMCLocalizerROS(Node):
    def __init__(self, rmclocalizer):
        super().__init__('rmc_localizer_node') # node name

        # MCLocalizer 객체
        self._rmclocalizer = rmclocalizer
        self._is_initialized = False

        # timer
        self.publisher_ = self.create_publisher(String, 'topic', 10)
        self.i = 0
        timer_period = 0.5
        self.timer = self.create_timer(timer_period, self.timer_callback)

        # callback odom
        self._prev_time = 0.0

        # subscribers
        self.sub_initial_pose = self.create_subscription(
            PoseWithCovarianceStamped,
            '/initialpose',
            self.callback_initialpose,
            10
        )

        # publishers
        self.particles_pub = self.create_publisher(PoseArray, 'particle_set', 10)

    def timer_callback(self):
        self.publish_particle_set(self._rmclocalizer.particle_set)

    def callback_initialpose(self, msg: PoseWithCovarianceStamped):
        # extract yaw from quaternion
        q = msg.pose.pose.orientation
        _, _, yaw = tf_transformations.euler_from_quaternion([q.x, q.y, q.z, q.w])

        # set initial pose of robot (used for initialize particle set)
        self._rmclocalizer.rmcl_estimated_pose.set_pose(
            x=msg.pose.pose.position.x,
            y=msg.pose.pose.position.y,
            yaw=yaw
        )

        # initialize particle set
        self._rmclocalizer.initialize_particle_set()

        # initialize reliability set if needed
        if True:
            self._rmclocalizer.initialize_reliability_set()

        # mark as initialized
        self._is_initialized = True

    # def odom_callback(self, msg: Odometry):
    #     """
    #     Odometry 메시지로부터 델타 이동량과 yaw를 업데이트

    #     @param msg: nav_msgs/Odometry 메시지
    #     @details
    #     업데이트되는 멤버 변수:
    #     - self._prev_time : 이전 Odometry 메시지 수신 시각(초 단위)
    #     - self.delta_x : x 방향 이동량 누적
    #     - self.delta_y : y 방향 이동량 누적
    #     - self.delta_dist : 전방 이동거리 누적
    #     - self.delta_yaw : 회전각 누적, -pi ~ pi 범위로 정규화
    #     - self.delta_time_sum : 누적 시간
    #     - self.odom_pose_stamp : 최근 Odometry 메시지의 timestamp
    #     - self.odom_pose : Odometry로부터 추정한 로봇 포즈(Pose)
    #     """
    #     curr_time = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

    #     if not self.is_initialized:
    #         self._prev_time = curr_time
    #         self.is_initialized = True
    #         return

    #     delta_time = curr_time - self._prev_time
    #     if delta_time == 0.0:
    #         return

    #     # extract data from Odometry message
    #     self.odom_pose_stamp = msg.header.stamp
    #     self.delta_x += msg.twist.twist.linear.x * delta_time
    #     self.delta_y += msg.twist.twist.linear.y * delta_time
    #     self.delta_dist += msg.twist.twist.linear.x * delta_time
    #     self.delta_yaw += msg.twist.twist.angular.z * delta_time

    #     # normalize yaw
    #     while self.delta_yaw < -math.pi:
    #         self.delta_yaw += 2.0 * math.pi
    #     while self.delta_yaw > math.pi:
    #         self.delta_yaw -= 2.0 * math.pi

    #     # .
    #     self.delta_time_sum += delta_time

    #     # extract yaw from quaternion
    #     q = msg.pose.pose.orientation
    #     quaternion = [q.x, q.y, q.z, q.w]
    #     _, _, yaw = tf_transformations.euler_from_quaternion(quaternion)

    #     # update odom pose
    #     self.odom_pose.set_pose(
    #         msg.pose.pose.position.x,
    #         msg.pose.pose.position.y,
    #         yaw
    #     )

    #     self._prev_time = curr_time

    def publish_particle_set(self, particle_set):
        """
        입력으로 받은 입자 집합을 PoseArray로 발행하여 RViz에서 시각화
        @param particle_set: List[Particle], 입자 객체 리스트
        """
        if not particle_set:
            return  # 빈 리스트면 발행하지 않음

        pose_array_msg = PoseArray()
        pose_array_msg.header.stamp = self.get_clock().now().to_msg()
        pose_array_msg.header.frame_id = 'odom'  # RViz global frame

        for particle in particle_set:
            pose_msg = RosPose()
            pose_msg.position.x = particle.pose.x
            pose_msg.position.y = particle.pose.y
            pose_msg.position.z = 0.0

            # yaw → quaternion
            q = tf_transformations.quaternion_from_euler(0, 0, particle.pose.yaw)
            pose_msg.orientation.x = q[0]
            pose_msg.orientation.y = q[1]
            pose_msg.orientation.z = q[2]
            pose_msg.orientation.w = q[3]

            pose_array_msg.poses.append(pose_msg)

        self.particles_pub.publish(pose_array_msg)


# ----------------------------------------------------------------------------------------------------

def print_global_map(map_path, step=5):
    """
    주어진 FloorMap 파일을 읽고, 터미널용 맵을 출력합니다.
    - 벽: '#'
    - 빈 공간: '.'
    - 방 ID: 0-9, 10-15 -> A-F (16진수)
    - 스테이션: 'S'
    - 좌표계: 상단(y), 좌측(x) 5단위 표시

    using method ---

    map_file = './../../worlds/map2.npz'
    print_global_map(map_file, step=5)

    """
    # 1. FloorMap 생성
    if not os.path.exists(map_path):
        raise FileNotFoundError(f"Map file not found: {map_path}")
    floor_map = FloorMap.from_file(map_path)

    # 2. MapInfo 생성
    pollution_end_time = 10.0
    starting_pos = floor_map.station_pos
    starting_angle = 0.0
    map_info = floor_map.to_map_info(pollution_end_time, starting_pos, starting_angle)

    wall_grid = map_info.wall_grid
    station_pos = map_info.station_pos

    height, width = wall_grid.shape
    grid_display = np.full((height, width), '.', dtype=str)  # 빈 공간은 '.'

    # 벽 표시
    grid_display[wall_grid] = '#'

    # 방 ID 표시 (0-9, 10-15 -> A-F)
    def room_id_to_char(room_id):
        if room_id < 10:
            return str(room_id)
        else:
            return chr(ord('A') + (room_id - 10) % 6)  # 10->A, 11->B ... 15->F, 16->A 반복

    for room_id in range(map_info.num_rooms):
        cells = map_info.get_cells_in_room(room_id)
        char = room_id_to_char(room_id)
        for x, y in cells:
            # 벽이나 스테이션이 아니면 덮어쓰기
            if grid_display[x, y] not in ('#', 'S'):
                grid_display[x, y] = char

    # 스테이션 표시
    gx, gy = map_info.pos2grid(station_pos)
    row = int(gx)
    col = int(gy)
    if 0 <= row < height and 0 <= col < width:
        grid_display[row, col] = 'S'

    # -----------------------------
    # 좌표계와 함께 출력
    # 상단 y좌표 (열)
    y_labels = '   '
    for j in range(0, width, step):
        label = f"{j:<{step}}"
        y_labels += label
    print(y_labels)

    # 각 행 출력 (왼쪽 x좌표)
    for i in range(height):
        x_label = f"{i:<2d}" if i % step == 0 else "  "
        line = x_label + ' ' + ''.join(grid_display[i, :])
        print(line)

# ----------------------------------------------------------------------------------------------------

def main(args=None):
    rclpy.init(args=args)

    # 
    _rmclocalizer = RMCLocalizer()
    node = RMCLocalizerROS(_rmclocalizer)

    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    # map_file = './../../worlds/map3.npz'
    # print_global_map(map_file, step=5)

    main()
