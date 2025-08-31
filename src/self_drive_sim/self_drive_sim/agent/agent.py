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
from sensor_msgs.msg import LaserScan                      # LaserScan
from geometry_msgs.msg import Pose as RosPose, PoseArray   # (visualization)

# python dependencies
from dataclasses import dataclass, field
import math
import random                                              # random.gauss
from scipy.spatial.transform import Rotation as R          # quaternion -> euler
from itertools import chain                                # chain



def nrand(n: float) -> float:
    """입력 n = 표준편차 σ"""
    u1 = random.random()
    u2 = random.random()
    z0 = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
    return n * z0

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

    def update(self, x: float, y: float, yaw: float):
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
                 initial_noise_yaw=0.02,

                 # class conditional measurment model
                 map_resolution=0.2, # meter
                 map_origin={'x': 0.0, 'y': 0.0, 'yaw': 0.0},
                 map_width=50,
                 map_height=50,
                 known_prior=0.5,
                 unknown_prior=0.5,
                 z_hit=0.9,
                 z_max=0.05,
                 z_rand=0.05,
                 unknown_lambda=1.0,
                 hit_variance=0.1,
                 ):
        
        # particle
        self._particles_num = particles_num
        self._initial_noise_x = initial_noise_x
        self._initial_noise_y = initial_noise_y
        self._initial_noise_yaw = initial_noise_yaw
        self.particle_set = []
        self.reliability_set = []

        # pose
        self.rmcl_estimated_pose = Pose()
        self.odom_pose = Pose()

        # motion
        # obtained by odom and used by updatae_pose_by_motion_model
        self.delta_x = self.delta_y = self.delta_dist = self.delta_yaw = 0
        self.total_abs_delta_x = self.total_abs_delta_y = self.total_abs_delta_dist = self.total_abs_delta_yaw = 0
        self.delta_time_sum = 0
        self.odom_noise = [0.1, 0.05, 0.05, 0.1]
        self.reliability_transition_coef = [0.0, 0.0]

    def initialize_particle_set(self):
        """입자를 초기 포즈 주변에 무작위로 분포시키고 가중치를 균등하게 설정"""
        xo, yo, yawo = self.rmcl_estimated_pose.x, self.rmcl_estimated_pose.y, self.rmcl_estimated_pose.yaw
        wo = 1.0 / self._particles_num

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
        
    def update_pose_by_motion_model(self):
        """
        @brief Odometry 기반 motion model을 이용해 파티클 집합을 업데이트

        이 함수는 differential drive 모델 또는 omni-directional 모델을 기반으로
        odometry 델타를 사용하여 추정 위치(self.rmcl_estimated_pose)와 파티클 집합(self.particle_set)을
        업데이트한다. 또한 파티클 신뢰도(self.reliability_set)를 감소시킨다.

        @details
        - 각 파티클은 Gaussian 노이즈가 추가된 이동량/회전량으로 업데이트됨
        - 신뢰도(reliability)는 이동량/회전량의 제곱에 비례하여 감소함

        @attention 함수 실행 후 delta_x, delta_y, delta_dist, delta_yaw는 0으로 초기화된다.

        @par 변경되는 멤버 변수
        - self.delta_x, self.delta_y, self.delta_dist, self.delta_yaw : 0.0으로 초기화
        - self.total_abs_delta_x, self.total_abs_delta_y, self.total_abs_delta_dist, self.total_abs_delta_yaw : 절댓값 누적
        - self.rmcl_estimated_pose : 로봇의 추정 위치/자세(x, y, yaw) 업데이트
        - self.particle_set : 각 파티클의 위치/자세 업데이트
        - self.reliability_set : 각 파티클 신뢰도 갱신

        @par 사용하는 멤버 변수 (읽기 전용)
        - self.use_omni_directional_model : motion model 종류 선택
        - self.odom_noise_ddm : differential drive 모델 노이즈 파라미터
        - self.odom_noise_odm : omni-directional 모델 노이즈 파라미터
        - self.rel_trans_ddm : differential drive 모델 신뢰도 감소 파라미터
        - self.rel_trans_odm : omni-directional 모델 신뢰도 감소 파라미터
        - self.estimate_reliability : 신뢰도 갱신 여부
        - self.particle_set : 파티클 리스트 (참조하여 업데이트 수행)
        """

        # backup delta values
        delta_x = self.delta_x
        delta_y = self.delta_y
        delta_dist = self.delta_dist
        delta_yaw = self.delta_yaw
        # initialize member variables for delta value
        self.delta_x = self.delta_y = self.delta_dist = self.delta_yaw = 0.0

        # update member variables for delta sum value
        self.total_abs_delta_x += abs(delta_x)
        self.total_abs_delta_y += abs(delta_y)
        self.total_abs_delta_dist += abs(delta_dist)
        self.total_abs_delta_yaw += abs(delta_yaw)

        # ----------------------------
        # Differential drive model
        # ----------------------------
        
        # update estimating robot pose
        yaw = self.rmcl_estimated_pose.yaw
        t = yaw + delta_yaw / 2.0
        x = self.rmcl_estimated_pose.x + delta_dist * math.cos(t)
        y = self.rmcl_estimated_pose.y + delta_dist * math.sin(t)
        yaw += delta_yaw
        self.rmcl_estimated_pose.update(x, y, yaw)

        # noisy dist and yaw
        dist2 = delta_dist * delta_dist
        yaw2 = delta_yaw * delta_yaw
        dist_rand_val = dist2 * self.odom_noise[0] + yaw2 * self.odom_noise[1]
        yaw_rand_val = dist2 * self.odom_noise[2] + yaw2 * self.odom_noise[3]

        # for index, particle in enumerate(self.particle_set):
        #     # differential drive model
        #     ddist = delta_dist + random.gauss(0, math.sqrt(dist_rand_val))
        #     dyaw = delta_yaw + random.gauss(0, math.sqrt(yaw_rand_val))

        #     yaw = particle.pose.yaw
        #     t = yaw + dyaw / 2.0
        #     x = particle.pose.x + ddist * math.cos(t)
        #     y = particle.pose.y + ddist * math.sin(t)
        #     yaw += dyaw
            
        #     # update particle set
        #     particle.pose.update(x, y, yaw)

        #     # heuristic as reliability transition model
        #     decay_rate = 1.0 - (self.reliability_transition_coef[0] * ddist**2 + self.reliability_transition_coef[1] * dyaw**2)
        #     if decay_rate <= 0.0:
        #         decay_rate = 1.0e-6
                
        #     # update reliability set
        #     self.reliability_set[index] *= decay_rate

        for index, particle in enumerate(self.particle_set):
            # differential drive model
            ddist = delta_dist + nrand(dist_rand_val)
            dyaw  = delta_yaw  + nrand(yaw_rand_val)

            yaw = particle.pose.yaw
            t = yaw + dyaw / 2.0
            x = particle.pose.x + ddist * math.cos(t)
            y = particle.pose.y + ddist * math.sin(t)
            yaw += dyaw

            # update particle set
            particle.pose.update(x, y, yaw)

            # heuristic as reliability transition model
            decay_rate = 1.0 - (self.reliability_transition_coef[0] * ddist**2 +
                                self.reliability_transition_coef[1] * dyaw**2)
            if decay_rate <= 0.0:
                decay_rate = 1.0e-5

            # update reliability set
            self.reliability_set[index] *= decay_rate

    def is_on_map(self, u, v, dist_map):
        """
        맵 범위 안인지 확인
        """
        return 0 <= u < dist_map.shape[1] and 0 <= v < dist_map.shape[0]
    
    def xy2uv(self, x, y, map_origin, map_resolution):
        """
        (x, y) 월드 좌표를 맵 좌표(u, v)로 변환
        map_origin: {'x': x_origin, 'y': y_origin, 'yaw': yaw_origin}
        """
        dx = x - map_origin['x']
        dy = y - map_origin['y']
        yaw = -map_origin['yaw']  # C++과 동일하게 회전 반영
        xx = dx * math.cos(yaw) - dy * math.sin(yaw)
        yy = dx * math.sin(yaw) + dy * math.cos(yaw)
        u = int(xx / map_resolution)
        v = int(yy / map_resolution)
        return u, v



# ----------------------------------------------------------------------------------------------------
from geometry_msgs.msg import PoseStamped
class RMCLocalizerROS(Node):
    def __init__(self, rmclocalizer,
                 initialpose_sub_topic_name='/initialpose',
                 odom_sub_topic_name='/robot_0/odom',
                 front_scan_sub_topic_name='/robot_0/base_sensor_lidar_front_controller/out',
                 back_scan_sub_topic_name='/robot_0/base_sensor_lidar_back_controller/out',
                 particles_pub_topic_name='particles',
                 timer_period=0.01, # s
                 ):
        super().__init__('rmc_localizer_node') # node name

        # MCLocalizer 객체
        self._rmclocalizer = rmclocalizer
        self._is_initialized = False

        # timer
        self.timer = self.create_timer(timer_period, self.callback_timer)

        # callback odom
        self._prev_time = 0.0
        self.is_initialized = False

        # subscribers
        self.subscriber_initialpose = self.create_subscription(
            PoseWithCovarianceStamped,
            initialpose_sub_topic_name,
            self.callback_initialpose,
            10
        )

        self.subscriber_odom = self.create_subscription(
            Odometry,
            odom_sub_topic_name,
            self.callback_odom,
            10
        )

        self.subscriber_scan = self.create_subscription(
            LaserScan,
            front_scan_sub_topic_name,
            self.callback_scan,
            10
        )

        # publishers
        self.particles_pub = self.create_publisher(
            PoseArray, 
            particles_pub_topic_name, 
            10
        )

        self.scan_pub = self.create_publisher(
            LaserScan,
            'merged_scan',
            10
        )

        self.pose_publisher = self.create_publisher(
            PoseStamped,
            '/robot_pose_stamped',
            10)

    def callback_timer(self):
        # step 2
        # self._rmclocalizer.update_pose_by_motion_model()

        # if hasattr(self._rmclocalizer, 'particle_set'):
        #     self.publish_particles(self._rmclocalizer.particle_set)
        
        # self.publish_scan()

        # test
        # Info 타입의 객체(딕셔너리) 생성
        info_data: Info = {
            'robot_position': (10.5, 25.1),
            'robot_angle': 1.57,
            'collided': False,
            'all_pollution': np.array([0.2, 0.5, 0.1])
        }

        # robot_position 값 출력
        # print(info_data['robot_position'])

    def callback_initialpose(self, msg: PoseWithCovarianceStamped):
        # extract yaw from quaternion
        q = msg.pose.pose.orientation
        r = R.from_quat([q.x, q.y, q.z, q.w])
        _, _, yaw = r.as_euler('xyz')  # roll, pitch, yaw (radians)

        # set initial pose of robot (used for initialize particle set)
        self._rmclocalizer.rmcl_estimated_pose.update(
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

    def callback_odom(self, msg: Odometry):
        """
        Odometry 메시지로부터 델타 이동량과 yaw를 업데이트

        @param msg: nav_msgs/Odometry 메시지
        @details
        업데이트되는 멤버 변수:
        - self.delta_x : x 방향 이동량 누적
        - self.delta_y : y 방향 이동량 누적
        - self.delta_dist : 전방 이동거리 누적
        - self.delta_yaw : 회전각 누적, -pi ~ pi 범위로 정규화
        - self.delta_time_sum : 누적 시간
        - self.odom_pose : Odometry로부터 추정한 로봇 포즈(Pose)

        - self._prev_time : 이전 Odometry 메시지 수신 시각(초 단위)
        - self.odom_pose_stamp : 최근 Odometry 메시지의 timestamp
        """
        curr_time = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

        if not self.is_initialized:
            self._prev_time = curr_time
            self.is_initialized = True
            return

        delta_time = curr_time - self._prev_time
        if delta_time == 0.0:
            return

        # extract data from Odometry message
        odom_pose_timestamp = msg.header.stamp
        self._rmclocalizer.delta_x += msg.twist.twist.linear.x * delta_time
        self._rmclocalizer.delta_y += msg.twist.twist.linear.y * delta_time
        self._rmclocalizer.delta_dist += msg.twist.twist.linear.x * delta_time
        self._rmclocalizer.delta_yaw += msg.twist.twist.angular.z * delta_time

        # normalize yaw
        while self._rmclocalizer.delta_yaw < -math.pi:
            self._rmclocalizer.delta_yaw += 2.0 * math.pi
        while self._rmclocalizer.delta_yaw > math.pi:
            self._rmclocalizer.delta_yaw -= 2.0 * math.pi

        # 
        self._rmclocalizer.delta_time_sum += delta_time

        # extract yaw from quaternion
        q = msg.pose.pose.orientation
        r = R.from_quat([q.x, q.y, q.z, q.w])
        _, _, yaw = r.as_euler('xyz')  # roll, pitch, yaw (radians)

        # update odom pose
        self._rmclocalizer.odom_pose.update(
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            yaw
        )

        self._prev_time = curr_time

        # test
        """
        Odometry 메시지를 수신했을 때 호출되는 콜백 함수
        """
        # 1. Odometry 메시지에서 x, y 위치 추출
        x_pos = msg.pose.pose.position.x
        y_pos = msg.pose.pose.position.y

        # 2. Odometry 메시지의 Quaternion 방향 정보에서 yaw 각도 추출
        orientation_q = msg.pose.pose.orientation
        (roll, pitch, yaw) = self.euler_from_quaternion(orientation_q)

        # 터미널에 현재 위치와 방향(yaw) 출력 (선택 사항)
        self.get_logger().info(f'Received Odom: x={x_pos:.2f}, y={y_pos:.2f}, yaw={math.degrees(yaw):.2f}°')

        # 3. RViz 시각화를 위한 PoseStamped 메시지 생성 및 발행
        pose_stamped_msg = PoseStamped()
        
        # header 정보는 수신한 odom 메시지의 것을 그대로 사용 (frame_id와 timestamp 일치)
        pose_stamped_msg.header = msg.header
        
        # pose 정보도 odom 메시지의 것을 그대로 복사
        # (x, y, z 위치와 quaternion 방향 정보가 모두 포함됨)
        pose_stamped_msg.pose = msg.pose.pose
        
        self.pose_publisher.publish(pose_stamped_msg)

    def euler_from_quaternion(self, quaternion):
        """
        Quaternion 메시지로부터 Roll, Pitch, Yaw 각도를 계산하는 함수
        """
        x = quaternion.x
        y = quaternion.y
        z = quaternion.z
        w = quaternion.w

        # Roll (x-axis rotation)
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = math.atan2(t0, t1)

        # Pitch (y-axis rotation)
        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch_y = math.asin(t2)

        # Yaw (z-axis rotation)
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = math.atan2(t3, t4)

        return roll_x, pitch_y, yaw_z
    
    def callback_scan(self, front_scan_msg: LaserScan):
        self.scan = front_scan_msg
        self.is_scan_received = True

    def callback_map(self):
        pass

    def publish_particles(self, particle_set):
        """
        Publish particle set as PoseArray for RViz visualization
        """
        if not particle_set:
            return

        pose_array_msg = PoseArray()
        pose_array_msg.header.stamp = self.get_clock().now().to_msg()
        pose_array_msg.header.frame_id = 'odom'

        for particle in particle_set:
            pose_msg = RosPose()
            pose_msg.position.x = particle.pose.x
            pose_msg.position.y = particle.pose.y
            pose_msg.position.z = 0.0

            # convert yaw to quaternion
            r = R.from_euler('xyz', [0, 0, particle.pose.yaw])
            q = r.as_quat()  # [x, y, z, w]
            pose_msg.orientation.x = q[0]
            pose_msg.orientation.y = q[1]
            pose_msg.orientation.z = q[2]
            pose_msg.orientation.w = q[3]

            pose_array_msg.poses.append(pose_msg)

        self.particles_pub.publish(pose_array_msg)

    def publish_scan(self):
        if not hasattr(self, 'scan') or self.scan is None:
            return

        scan_msg = LaserScan()
        scan_msg.header.stamp = self.get_clock().now().to_msg()
        scan_msg.header.frame_id = self.scan.header.frame_id  # 원본 frame 사용

        # 원본 LaserScan 데이터 복사
        scan_msg.angle_min = self.scan.angle_min
        scan_msg.angle_max = self.scan.angle_max
        scan_msg.angle_increment = self.scan.angle_increment
        scan_msg.time_increment = self.scan.time_increment
        scan_msg.scan_time = self.scan.scan_time
        scan_msg.range_min = self.scan.range_min
        scan_msg.range_max = self.scan.range_max

        # list로 변환
        scan_msg.ranges = list(self.scan.ranges)
        scan_msg.intensities = list(self.scan.intensities) if self.scan.intensities else [1.0] * len(self.scan.ranges)

        # 퍼블리시
        self.scan_pub.publish(scan_msg)


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

    # Room ID (0-9, 10-15 -> A-F)
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

    # Station
    gx, gy = map_info.pos2grid(station_pos)
    row = int(gx)
    col = int(gy)
    if 0 <= row < height and 0 <= col < width:
        grid_display[row, col] = 'S'

    # Print
    is_coordinate = True

    if is_coordinate: # Map with coordinate
        x_label_width = 3
        y_labels = " _|"

        for j in range(0, width, step):
            label = f"{j:<{step-1}}|"
            y_labels += label
        print(y_labels)

        # X
        for i in range(height):
            if i % step == 0:
                x_label = f"{i:>2} "
            elif (i + 1) % step == 0:
                x_label = " _ "
            else:
                x_label = "   "

            line = x_label + ''.join(grid_display[i, :])
            print(line)
            
    else: # Map without coordinate
        for i in range(height):
            line = ''.join(grid_display[i, :])
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
    map_file = './../../worlds/map3.npz'
    print_global_map(map_file, step=5)

    # main()
