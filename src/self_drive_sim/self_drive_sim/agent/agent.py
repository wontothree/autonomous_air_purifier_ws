from self_drive_sim.agent.interfaces import Observation, Info, MapInfo

# ROS dependencies
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry                          # Odometry
from geometry_msgs.msg import PoseStamped                  # PoseStamped

# Python dependencies
import math
import numpy as np

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

        # # --- 여기에 맵 정보 출력 코드를 추가합니다 ---
        # self.log("================ MAP INFO (from Agent) ================")
        # self.log(f"Map Dimensions (Height x Width): {map_info.height} x {map_info.width} grids")
        # self.log(f"Grid Size: {map_info.grid_size} m")
        # self.log(f"Grid Origin (in grid coords): {map_info.grid_origin}")
        # self.log(f"Number of Rooms: {map_info.num_rooms}")
        # self.log(f"Room Names: {map_info.room_names}")
        # self.log(f"Station Position (World Coords): {map_info.station_pos}")
        # self.log(f"Robot Starting Position: {map_info.starting_pos}")
        # self.log(f"Robot Starting Angle: {map_info.starting_angle} radians")
        # self.log(f"Last Pollution End Time: {map_info.pollution_end_time} s")
        # self.log(f"Wall Grid Shape: {map_info.wall_grid.shape}")
        # self.log(f"Room Grid Shape: {map_info.room_grid.shape}")
        # self.log("======================================================")
        # # ---------------------------------------------
        
        # # 나중에 다른 함수(예: act)에서도 맵 정보를 사용하기 위해 클래스 변수에 저장합니다.
        # self.map_info = map_info

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
        action = (0, 0.5, 1.0)

        # log 사용법 예시: print를 사용하면 비정상적으로 출력되므로 log를 사용
        if self.steps % 50 == 0:
            self.log(observation['sensor_tof_left'])

        self.steps += 1

        # if self.steps % 10 == 0:
        #     # 1. 로봇 위치의 현재 오염도
        #     robot_pollution = observation['sensor_pollution']

        #     # --- 수정된 부분: None 타입인지 확인 ---
        #     if robot_pollution is not None:
        #         # None이 아닐 때만 소수점 형태로 출력
        #         self.log(f"*****Robot's current pollution reading: {robot_pollution:.4f}")
        #     else:
        #         # None일 경우, 다른 메시지를 출력하여 상황을 알림
        #         self.log("*****Robot's current pollution reading: Not available (None)")
        #     # ------------------------------------

        #     # 2. 방별 고정 센서의 오염도
        #     air_sensor_data = observation['air_sensor_pollution']
        #     self.log(f"*****Stationary air sensor readings: {air_sensor_data}")

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
        pass

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
    

class AgentROS(Node):
    def __init__(self,
                 odom_sub_topic_name='/robot_0/odom',
                 timer_period=0.01, # s
                 ):
        super().__init__('agent_node') # node name

        # Timer and subscribers
        self.timer = self.create_timer(timer_period, self.callback_timer)
        self.subscriber_odom = self.create_subscription(
            Odometry,
            odom_sub_topic_name,
            self.callback_odom,
            10
        )

        # Publishers
        self.pose_publisher = self.create_publisher(
            PoseStamped,
            '/robot_pose_stamped',
            10
        )

    def callback_timer(self):
        pass

    def callback_odom(self, msg: Odometry):
        # Extrack x and y
        x_pos = msg.pose.pose.position.x
        y_pos = msg.pose.pose.position.y

        # Extract yaw
        orientation_q = msg.pose.pose.orientation
        (roll, pitch, yaw) = self.euler_from_quaternion(orientation_q)

        # print
        self.get_logger().info(f'Received Odom: x={x_pos:.2f}, y={y_pos:.2f}, yaw={math.degrees(yaw):.2f}°')

        # Publish for visualizaiton in RViz
        pose_stamped_msg = PoseStamped()
        pose_stamped_msg.header = msg.header
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

def main(args=None):
    rclpy.init(args=args)

    node = AgentROS()

    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


# if __name__ == '__main__':
    # main()