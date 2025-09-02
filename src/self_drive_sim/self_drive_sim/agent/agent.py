from self_drive_sim.agent.interfaces import Observation, Info, MapInfo

import math
import numpy as np
import itertools

class Agent:
    def __init__(self, logger):
        self.logger = logger
        self.steps = 0

        self.current_robot_pose = (0.0, 0.0, 0.0)

        # Finite state machine
        self.current_fsm_state = "WAITING"
        self.waypoints = []
        self.current_waypoint_index = 0
        self.tmp_target_position = None
        self.optimal_next_node_index = None
        self.current_node_index = None

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
        self.get_room_id = map_info.get_room_id

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
        # Mission planning
        air_sensor_pollution_data = observation['air_sensor_pollution']
        robot_sensor_pollution_data =  observation['sensor_pollution']

        # --------------------------------------------------
        next_state, action = self.fsm(
            air_sensor_pollution_data,
            robot_sensor_pollution_data,
            self.current_fsm_state,
            self.current_robot_pose
        )
        self.current_fsm_state = next_state 
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
        self.current_robot_pose = (info["robot_position"][0], info["robot_position"][1], info["robot_angle"])
        # self.log(self.current_robot_pose)

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

    def mission_planner(self, air_sensor_pollution_data, robot_sensor_pollution_data, current_node_index):
        """
        미션 플래너: 오염 감지된 방들을 기반으로 TSP 순서에 따라 task queue 생성

        Parameters:
            - air_sensor_pollution_data: list of float, 각 방의 공기 센서 오염 수치
            - robot_sensor_pollution_data: list of float, 로봇 센서 오염 수치 (현재 사용 안 함)
            - current_node_index: int, 현재 방(room)의 ID

        Return:
            - best_path: List[int], 방문해야 할 방의 순서
        """
        map0_distance_matrix = np.array([
            [0, 29, 12, 43],
            [29, 0, 33, 27],
            [12, 33, 0, 15],
            [43, 27, 15, 0]
        ])

        unobserved_potential_regions = []

        # Polluted regions
        observed_polluted_regions = [
            room_id for room_id in range(self.map_info.num_rooms)
            if air_sensor_pollution_data[room_id] > 0
        ]

        if not observed_polluted_regions:
            return

        distance_matrix = map0_distance_matrix  
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

        # self.log(optimal_visit_order)

        return optimal_visit_order
    
    def global_planner(self, start_node_index, end_node_index):
        """
        Index rules
        - index of region is a index of matrix
        - last index is for docking station
        - (last - 1) index is for start position
        
        reference_waypoint_matrix[i][j] : waypoints from i to j
        """
        map0_reference_waypoint_matrix = [[[] for _ in range(4)] for _ in range(4)]
        map0_reference_waypoint_matrix[0][1] = [(-1, 1), (-1, -2)]
        map0_reference_waypoint_matrix[0][3] = [(1.4, -3)]
        map0_reference_waypoint_matrix[1][0] = [(-1, 1.6), (1, 1.8)]
        map0_reference_waypoint_matrix[1][3] = [(-0.8, 1.6), (0.2, 1.2), (1.4, -3)]
        map0_reference_waypoint_matrix[2][0] = [(1, 1.8)]
        map0_reference_waypoint_matrix[2][1] = [(0.2, 1.6), (-0.8, 1.6), (-1, -2)]
        return map0_reference_waypoint_matrix[start_node_index][end_node_index]


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




    def fsm(self,
            air_sensor_pollution_data,
            robot_sensor_pollution_data,
            current_fsm_state, 
            current_robot_pose, 
            ):
        """
        current_fsm_state -> next_fsm_state, action
        """
        # Define states of finite state machine
        FSM_WAITING = "WAITING"
        FSM_CLEANING = "CLEANING"
        FSM_NAVIGATING = "NAVIGATING"
        FSM_RETURNING = "RETURNING"

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

        if current_fsm_state == FSM_WAITING:
            # Mission planning
            optimal_visit_order = self.mission_planner(air_sensor_pollution_data, robot_sensor_pollution_data, current_node_index=2)

            if optimal_visit_order != None: # 목표 구역이 있음 (상태 전이)
                next_fsm_state = FSM_NAVIGATING
                self.optimal_next_node_index = optimal_visit_order[0]
                self.waypoints = self.global_planner(start_node_index=2, end_node_index=self.optimal_next_node_index)
                self.current_waypoint_index = 0
                self.tmp_target_position = self.waypoints[0]
            else:
                next_fsm_state = FSM_WAITING

            action = (0, 0, 0) # Stop

        elif current_fsm_state == FSM_NAVIGATING:
            if is_target_reached(current_robot_position, self.waypoints[-1]): # 목표 구역에 도달함 (상태 전이):
                next_fsm_state = FSM_CLEANING
                self.current_waypoint_index = 0
                self.current_node_index = self.optimal_next_node_index
            else:
                next_fsm_state = FSM_NAVIGATING

            # set next tempt target point 이 순서를 바꾸면 왜 문제가 생길까?
            if is_target_reached(current_robot_position, self.tmp_target_position) and self.current_waypoint_index < len(self.waypoints) - 1:
                self.current_waypoint_index += 1
                self.tmp_target_position = self.waypoints[self.current_waypoint_index]

            linear_velocity, angular_velocity = self.controller(current_robot_pose, self.tmp_target_position)
            action = (0, linear_velocity, angular_velocity)

        elif current_fsm_state == FSM_CLEANING:
            # Mission planning
            optimal_visit_order = self.mission_planner(air_sensor_pollution_data, robot_sensor_pollution_data, current_node_index=self.current_node_index)          

            if are_no_polluted_rooms(air_sensor_pollution_data):     # 오염 구역이 없음 (상태 전이)
                next_fsm_state = FSM_RETURNING

                self.current_waypoint_index = 0
                self.waypoints = self.global_planner(start_node_index=self.current_node_index, end_node_index=3)
                self.tmp_target_position = self.waypoints[0]

            elif air_sensor_pollution_data[self.optimal_next_node_index] == 0 and optimal_visit_order != None:       # 청정 완료함 (상태 전이)
                next_fsm_state = FSM_NAVIGATING

                # 꼭 이때 해야 할까?
                self.optimal_next_node_index = optimal_visit_order[0]
                self.log(self.current_node_index)
                self.log(self.optimal_next_node_index)
                self.waypoints = self.global_planner(start_node_index=self.current_node_index, end_node_index=self.optimal_next_node_index)
                self.current_waypoint_index = 0
                self.log(self.waypoints)
                self.tmp_target_position = self.waypoints[0]  

            else:
                next_fsm_state = FSM_CLEANING
            
            action = (1, 0, 0) # Stop and clean

        elif current_fsm_state == FSM_RETURNING:
            next_fsm_state = FSM_RETURNING
            
            linear_velocity, angular_velocity = self.controller(current_robot_pose, self.tmp_target_position)
            action = (0, linear_velocity, angular_velocity)

            if is_target_reached(current_robot_position, self.tmp_target_position) and self.current_waypoint_index < len(self.waypoints) - 1:
                self.current_waypoint_index += 1
                self.tmp_target_position = self.waypoints[self.current_waypoint_index]

        self.log(current_fsm_state)

        return next_fsm_state, action