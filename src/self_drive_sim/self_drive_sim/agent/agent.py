from self_drive_sim.agent.interfaces import Observation, Info, MapInfo

import math

from .map import Map, OCCUPANCY_GRID_MAP0, OCCUPANCY_GRID_MAP1, OCCUPANCY_GRID_MAP2, OCCUPANCY_GRID_MAP3
from .autonomous_navigator import AutonomousNavigator
from .monte_carlo_localizer import Pose

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
        delta_distance = math.dist([0, 0], observation["disp_position"]) # IMU data
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