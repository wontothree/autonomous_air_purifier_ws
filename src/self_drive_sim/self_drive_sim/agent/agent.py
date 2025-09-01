from self_drive_sim.agent.interfaces import Observation, Info, MapInfo

import numpy as np
import math
import itertools

class Agent:
    def __init__(self, logger):
        self.logger = logger
        self.steps = 0

        self.current_robot_pose = (0.0, 0.0, 0.0)

        self.current_wp_index = 0   # <-- 멤버 변수로 이동

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
        # action = (0, 0.5, 1.0)

        air_sensor_pollution_data = observation['air_sensor_pollution']
        robot_sensor_pollution_data =  observation['sensor_pollution']
        optimal_visit_order = self.mission_planner(air_sensor_pollution_data, robot_sensor_pollution_data, 2)

        if optimal_visit_order != None:
            # waypoints = self.global_planner(2, optimal_visit_order[0])
            waypoints = self.global_planner(2, 0)

            linear_action, self.current_wp_index = self.follow_waypoints(
                self.current_robot_pose,
                waypoints,
                self.current_wp_index
            )
            action = linear_action
        else:
            action = (0, 0, 0)

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

    def mission_planner(self, air_sensor_pollution_data, robot_sensor_pollution_data, current_node):
        """
        미션 플래너: 오염 감지된 방들을 기반으로 TSP 순서에 따라 task queue 생성

        Parameters:
            - air_sensor_pollution_data: list of float, 각 방의 공기 센서 오염 수치
            - robot_sensor_pollution_data: list of float, 로봇 센서 오염 수치 (현재 사용 안 함)
            - current_node: int, 현재 방(room)의 ID

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
            last_visited = current_node

            for room_id in perm:
                total_cost += distance_matrix[last_visited][room_id]
                last_visited = room_id

            # Add Cost for last_visited to docking station
            total_cost += distance_matrix[last_visited][dock_station_id]

            if total_cost < min_cost:
                min_cost = total_cost
                optimal_visit_order = list(perm)

        self.log(optimal_visit_order)

        return optimal_visit_order

    def global_planner(self, start_node, end_node):

        map0_reference_waypoint = np.empty((4, 4), dtype=object)
        for i in range(4):
            for j in range(4):
                map0_reference_waypoint[i, j] = []
        map0_reference_waypoint[1][0] = np.array([
            (-8.0, -10.0), (-8.0, -9.5), (-7.9, -9.0), (-7.8, -8.5), (-7.6, -8.0), (-7.5, -7.6), (-7.4, -7.1), (-7.3, -6.6), (-7.1, -6.1), (-7.0, -5.6), (-6.9, -5.1), (-6.8, -4.7), (-6.6, -4.2), (-6.5, -3.7), (-6.4, -3.2), (-6.3, -2.7), (-6.1, -2.2), (-6.0, -1.7), (-5.9, -1.3), (-5.8, -0.8), (-5.6, -0.3), (-5.5, 0.2), (-5.4, 0.7), (-5.3, 1.2), (-5.2, 1.6), (-5.0, 2.1), (-4.9, 2.6), (-4.9, 3.1), (-4.8, 3.6), (-4.7, 4.1), (-4.5, 4.6), (-4.4, 5.0), (-4.3, 5.5), (-4.0, 6.0), (-3.7, 6.3), (-3.3, 6.6), (-2.8, 6.8), (-2.3, 6.9), (-1.9, 7.1), (-1.4, 7.2), (-0.9, 7.4), (-0.4, 7.6), (0.0, 7.7), (0.5, 7.9), (1.0, 8.0), (1.4, 8.3), (1.8, 8.6), (2.2, 8.8), (2.7, 9.1), (3.1, 9.4)
        ])
        map0_reference_waypoint[0][1] = map0_reference_waypoint[1][0][::-1]

        # v
        map0_reference_waypoint[2][0] = np.array([
            (1.8, 0.0), (1.8, 0.1), (1.78, 0.2), (1.74, 0.3), (1.68, 0.38), (1.62, 0.46), (1.58, 0.54), (1.52, 0.62), (1.46, 0.7), (1.4, 0.78), (1.34, 0.86), (1.3, 0.96), (1.26, 1.06), (1.22, 1.14), (1.2, 1.24), (1.16, 1.32), (1.12, 1.42), (1.08, 1.52), (1.04, 1.6), (1.0, 1.7), (0.98, 1.8)
        ])
        map0_reference_waypoint[0][2] = map0_reference_waypoint[2][0][::-1]

        map0_reference_waypoint[2][1] = np.array([
            (9.0, 0.0), (9.0, 0.5), (8.9, 1.0), (8.6, 1.4), (8.3, 1.8), (7.9, 2.1), (7.4, 2.2), (6.9, 2.4), (6.5, 2.5), (6.0, 2.7), (5.5, 2.8), (5.0, 3.0), (4.6, 3.2), (4.1, 3.4), (3.7, 3.7), (3.3, 4.0), (2.9, 4.2), (2.4, 4.5), (2.1, 4.9), (1.7, 5.1), (1.2, 5.3), (0.8, 5.6), (0.4, 5.8), (-0.1, 6.0), (-0.6, 6.0), (-1.1, 6.1), (-1.6, 6.1), (-2.1, 6.0), (-2.6, 6.1), (-3.1, 6.1), (-3.6, 6.0), (-4.0, 5.8), (-4.4, 5.5), (-4.7, 5.1), (-4.9, 4.6), (-5.0, 4.1), (-5.1, 3.6), (-5.1, 3.1), (-5.2, 2.6), (-5.3, 2.1), (-5.3, 1.6), (-5.4, 1.2), (-5.5, 0.7), (-5.5, 0.2), (-5.6, -0.3), (-5.7, -0.8), (-5.8, -1.3), (-5.8, -1.8), (-5.9, -2.3), (-6.0, -2.8), (-6.0, -3.3), (-6.2, -3.8), (-6.4, -4.2), (-6.6, -4.7), (-6.8, -5.1), (-7.0, -5.6), (-7.2, -6.1), (-7.4, -6.5), (-7.6, -7.0), (-7.8, -7.5), (-8.0, -7.9), (-8.1, -8.4), (-8.3, -8.8), (-8.6, -9.2)
        ])
        map0_reference_waypoint[1][2] = map0_reference_waypoint[2][1][::-1]

        map0_reference_waypoint[3][0] = np.array([
            (6.0, -15.0), (6.0, -14.5), (5.9, -14.0), (5.8, -13.5), (5.6, -13.0), (5.5, -12.6), (5.4, -12.1), (5.3, -11.6), (5.1, -11.1), (5.0, -10.6), (4.9, -10.1), (4.8, -9.7), (4.6, -9.2), (4.6, -8.7), (4.6, -8.2), (4.6, -7.7), (4.6, -7.2), (4.6, -6.7), (4.6, -6.2), (4.6, -5.7), (4.6, -5.2), (4.6, -4.7), (4.6, -4.2), (4.6, -3.7), (4.6, -3.2), (4.6, -2.7), (4.6, -2.2), (4.6, -1.7), (4.6, -1.2), (4.6, -0.7), (4.6, -0.2), (4.6, 0.3), (4.6, 0.8), (4.6, 1.3), (4.6, 1.8), (4.6, 2.3), (4.6, 2.8), (4.6, 3.3), (4.6, 3.8), (4.6, 4.3), (4.6, 4.8), (4.6, 5.3), (4.6, 5.8), (4.6, 6.3), (4.6, 6.8), (4.6, 7.3), (4.6, 7.8), (4.6, 8.3)
        ])
        map0_reference_waypoint[0][3] = map0_reference_waypoint[3][0][::-1]

        map0_reference_waypoint[3][1] = np.array([
            (6.0, -15.0), (6.0, -14.5), (5.9, -14.0), (5.6, -13.6), (5.5, -13.1), (5.4, -12.6), (5.3, -12.1), (5.1, -11.6), (5.0, -11.2), (4.9, -10.7), (4.8, -10.2), (4.6, -9.7), (4.5, -9.2), (4.4, -8.7), (4.3, -8.2), (4.2, -7.8), (4.0, -7.3), (3.9, -6.8), (3.9, -6.3), (3.9, -5.8), (3.9, -5.3), (3.8, -4.8), (3.5, -4.4), (3.4, -3.9), (3.2, -3.4), (3.1, -3.0), (2.9, -2.5), (2.7, -2.0), (2.6, -1.6), (2.4, -1.1), (2.3, -0.6), (2.1, -0.1), (2.0, 0.3), (1.8, 0.8), (1.6, 1.3), (1.5, 1.7), (1.3, 2.2), (1.2, 2.7), (1.2, 3.2), (1.2, 3.7), (1.2, 4.2), (1.1, 4.7), (0.9, 5.1), (0.5, 5.5), (0.1, 5.8), (-0.3, 6.0), (-0.8, 6.2), (-1.3, 6.2), (-1.8, 6.3), (-2.3, 6.3), (-2.8, 6.3), (-3.3, 6.3), (-3.7, 6.0), (-4.1, 5.7), (-4.4, 5.3), (-4.6, 4.9), (-4.7, 4.4), (-4.8, 3.9), (-4.8, 3.4), (-4.9, 2.9), (-5.0, 2.4), (-5.0, 1.9), (-5.1, 1.4), (-5.2, 0.9), (-5.3, 0.4), (-5.3, -0.1), (-5.4, -0.6), (-5.5, -1.1), (-5.5, -1.6), (-5.6, -2.1), (-5.8, -2.5), (-6.0, -3.0), (-6.2, -3.4), (-6.4, -3.9), (-6.6, -4.4), (-6.8, -4.8), (-6.9, -5.3), (-7.1, -5.8), (-7.3, -6.2), (-7.5, -6.7), (-7.7, -7.1), (-7.9, -7.6), (-8.1, -8.1), (-8.3, -8.5), (-8.5, -9.0), (-8.8, -9.4)
        ])
        map0_reference_waypoint[1][3] = map0_reference_waypoint[3][1][::-1]

        return map0_reference_waypoint[start_node][end_node]

    def go_to_goal(self, current_pose, target_pose):
        ANGLE_TOLERANCE = math.radians(5)
        DISTANCE_TOLERANCE = 0.15
        MAX_LINEAR_VEL = 0.8
        MIN_LINEAR_VEL = 0.05
        MAX_ANGULAR_VEL = 1.5
        KP_ANGLE = 2.0
        KP_DISTANCE = 1.0

        cx, cy, cyaw = current_pose
        tx, ty = target_pose

        dist_error = math.hypot(tx - cx, ty - cy)
        if dist_error < DISTANCE_TOLERANCE:
            return (1, 0.0, 0.0)

        angle_to_target = math.atan2(ty - cy, tx - cx)
        angle_error = angle_to_target - cyaw
        while angle_error > math.pi: angle_error -= 2 * math.pi
        while angle_error < -math.pi: angle_error += 2 * math.pi

        angular_vel = np.clip(KP_ANGLE * angle_error, -MAX_ANGULAR_VEL, MAX_ANGULAR_VEL)
        linear_vel = KP_DISTANCE * dist_error

        angle_error_deg = abs(math.degrees(angle_error))
        if angle_error_deg > 30:
            scaling_factor = max(0, 1 - (angle_error_deg - 30) / 60)
            linear_vel *= scaling_factor

        linear_vel = np.clip(linear_vel, MIN_LINEAR_VEL, MAX_LINEAR_VEL)
        return (0, linear_vel, angular_vel)
        
    def follow_waypoints(self, current_pose, waypoints, wp_index):
        """
        여러 개의 waypoint를 순차적으로 따라가기 위한 함수.

        Args:
            current_pose: (x, y, yaw) 현재 로봇 pose
            waypoints: [(x1, y1), (x2, y2), ...] 따라가야 할 waypoint 리스트
            wp_index: 현재 목표 waypoint 인덱스

        Returns:
            action: (MODE, linear_v, angular_v)
            updated_wp_index: 다음 step에서 사용할 waypoint index
        """
        if wp_index >= len(waypoints):
            # 모든 waypoint 다 도착
            return (0, 0.0, 0.0), wp_index

        # 현재 목표 waypoint
        target_wp = waypoints[wp_index]

        # go_to_goal 사용
        action = self.go_to_goal(current_pose, target_wp)

        # 도착했는지 확인
        DISTANCE_TOLERANCE = 0.15
        dist_error = math.hypot(target_wp[0] - current_pose[0], target_wp[1] - current_pose[1])

        if dist_error < DISTANCE_TOLERANCE:
            wp_index += 1  # 다음 waypoint로 업데이트

        return action, wp_index
