from self_drive_sim.agent.interfaces import Observation, Info, MapInfo

import math
import numpy as np

class Agent:
    def __init__(self, logger):
        self.logger = logger
        self.steps = 0

        self.map_info = None
        self.current_robot_pose = (0.0, 0.0, 0.0)
        #self.pollution_state = (0.0, 0.0)
        self.target_position = None
        self.room_centers = []
        self.pollution_end_time = 0.0

        # Mission planning
        self.polluted_rooms_queue = []      # 처리해야 할 오염된 방의 대기열
        self.previous_pollution_data = None # 이전 스텝의 오염도
        self.CLEAN_THRESHOLD = 0.1        # 청정 완료 기준

        # Finite state machine
        self.state = 'IDLE' # 'IDLE', 'NAVIGATING', 'CLEANING', 'DONE'

        self.clean_start_time = None

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

        self.map_info = map_info
        self.pollution_end_time = map_info.pollution_end_time
        self.target_position = None
        self.state = 'IDLE'
        self.previous_pollution_data = np.zeros(self.map_info.num_rooms)
        
        # initialize robot pose
        start_pos = map_info.starting_pos
        start_angle = map_info.starting_angle
        self.current_robot_pose = (start_pos[0], start_pos[1], start_angle)
        
        # Center world coordinate of every rooms
        self.room_centers = []
        for room_id in range(self.map_info.num_rooms):
            cells = self.map_info.get_cells_in_room(room_id)
            if not cells:
                self.room_centers.append(self.map_info.station_pos)
                continue
            center_grid = (sum(c[0] for c in cells) / len(cells), sum(c[1] for c in cells) / len(cells))
            self.room_centers.append(self.map_info.grid2pos(center_grid))
        # self.log(f"Calculated Room Centers (World Coords): {self.room_centers}")

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

        robot_pollution = observation['sensor_pollution']
        current_time = self.steps * 0.1

        # air_sensor_pollution 없으면 이전 action 유지
        if observation['air_sensor_pollution'] is None:
            self.steps += 1
            if self.state == 'NAVIGATING' and self.target_position:
                return self.go_to_goal(self.current_robot_pose, self.target_position)
            return (0, 0.0, 0.0)

        # 1. 새로운 오염 발생 감지 및 polluted_rooms_queue 업데이트
        current_pollution_data = observation['air_sensor_pollution']
        self.mission_planner(current_pollution_data)

        action = (0, 0.0, 0.0)

        # -------------------
        # FSM + Waypoint Logic
        # -------------------

        # IDLE 상태
        if self.state == 'IDLE':
            if self.polluted_rooms_queue:  # 오염된 방 탐지 시 이동
                target_room_id = self.polluted_rooms_queue[0]
                self.target_position = self.room_centers[target_room_id]
                self.waypoints = self.global_planner(self.current_robot_pose[:2], self.target_position)
                self.current_waypoint_idx = 1
                self.state = 'NAVIGATING'
                self.log(f"Navigating to Room {target_room_id} via {self.waypoints}")
            elif current_time >= self.pollution_end_time:  # 오염원 끝나면 도킹
                self.target_position = self.map_info.station_pos
                self.waypoints = self.global_planner(self.current_robot_pose[:2], self.target_position)
                self.current_waypoint_idx = 1
                self.state = 'NAVIGATING'
                self.log("Returning to station via waypoints.")

        # NAVIGATING 상태
        elif self.state == 'NAVIGATING':
            if self.current_waypoint_idx < len(self.waypoints):
                goal_wp = self.waypoints[self.current_waypoint_idx]
                action = self.go_to_goal(self.current_robot_pose, goal_wp)
                mode, _, _ = action
                if mode == 1:  # waypoint 도착
                    self.current_waypoint_idx += 1
            else:
                # 모든 waypoint 도착
                action = self.go_to_goal(self.current_robot_pose, self.target_position)
                mode, _, _ = action
                if mode == 1:
                    if self.target_position == self.map_info.station_pos:
                        self.state = 'DONE'
                        self.log("Reached station. Task DONE.")
                    else:
                        self.state = 'CLEANING'
                        self.clean_start_time = current_time
                        self.current_cleaning_room_id = self.polluted_rooms_queue[0]  # 현재 청소 방 저장
                        self.log(f"Start cleaning at {self.target_position}")

        # CLEANING 상태
        elif self.state == 'CLEANING':
            action = (1, 0.0, 0.0)
            target_room_id = self.current_cleaning_room_id

            if target_room_id is not None and observation['air_sensor_pollution'] is not None:
                room_pollution = observation['air_sensor_pollution'][target_room_id]
                if room_pollution < self.CLEAN_THRESHOLD:
                    self.log(f"Cleaning complete for target {self.target_position}.")
                    self.polluted_rooms_queue.pop(0)
                    self.state = 'IDLE'
                    self.target_position = None
                    self.waypoints = []
                    self.current_waypoint_idx = 0
                    self.current_cleaning_room_id = None

        # DONE 상태
        elif self.state == 'DONE':
            action = (0, 0.0, 0.0)

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
        #self.pollution_state = info['all_pollution']

        #self.log(f"Robot pose: {self.current_robot_pose}")
        #self.log(f"Pollution State {self.pollution_state}")

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

    # additional functions

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

    def mission_planner(self, current_pollution_data):
        """
        새로운 오염 발생 감지 및 polluted_rooms_queue 업데이트

        사용 멤버 변수:
            - self.map_info.num_rooms : 방 개수 확인
            - self.CLEAN_THRESHOLD : 청정 기준 임계값
            - self.log(msg) : 로그 출력

        변경 멤버 변수:
            - self.polluted_rooms_queue : 새 오염이 발생한 방 ID를 추가
            - self.previous_pollution_data : 현재 오염 데이터를 저장
        """
        for room_id in range(self.map_info.num_rooms):
            prev = self.previous_pollution_data[room_id]
            curr = current_pollution_data[room_id]
            if prev < self.CLEAN_THRESHOLD and curr >= self.CLEAN_THRESHOLD:
                if room_id not in self.polluted_rooms_queue:
                    self.log(f"New pollution detected in Room {room_id}.")
                    self.polluted_rooms_queue.append(room_id)

        # 현재 오염 상태를 다음 step을 위해 저장
        self.previous_pollution_data = current_pollution_data


    def global_planner(self, start_pos, goal_pos):
        """
        현재 위치와 목표 위치에 따라 장애물을 고려한 waypoint 리스트 반환.
        직선 이동 가능한 구간은 직선으로 이동.
        """
        def is_close(pos1, pos2, tol=0.05):
            return math.hypot(pos1[0] - pos2[0], pos1[1] - pos2[1]) < tol

        # 주요 지점 정의
        start = (2, 0)
        outside = (1.6, 2.4)
        room = (-1.4, -2.4)
        station = (1.4, 3)

        # start → goal 경로에 필요한 중간 waypoint 결정
        mid_waypoints = []

        if (is_close(start_pos, start) and is_close(goal_pos, room)) or (is_close(start_pos, room) and is_close(goal_pos, start)):
            mid_waypoints = [(-1.4, 2.4)]
        elif (is_close(start_pos, room) and is_close(goal_pos, station)) or (is_close(start_pos, station) and is_close(goal_pos, room)):
            mid_waypoints = [(1.4, 0), (-1.4, 2.4)]
        elif (is_close(start_pos, outside) and is_close(goal_pos, room)) or (is_close(start_pos, room) and is_close(goal_pos, outside)):
            mid_waypoints = [(-1.4, 2.4)]

        # 나머지는 직선 이동
        waypoints = [start_pos] + mid_waypoints + [goal_pos]

        # -------------------------------------------
        # waypoint 보간(interpolation)
        def interpolate_waypoints(waypoints, step=0.2):
            interpolated = [waypoints[0]]
            for i in range(len(waypoints)-1):
                start = waypoints[i]
                end = waypoints[i+1]
                dist = math.hypot(end[0]-start[0], end[1]-start[1])
                num_steps = max(1, int(dist / step))
                for j in range(1, num_steps+1):
                    x = start[0] + (end[0]-start[0]) * j / num_steps
                    y = start[1] + (end[1]-start[1]) * j / num_steps
                    interpolated.append((x, y))
            return interpolated

        waypoints = interpolate_waypoints(waypoints, step=0.2)
        return waypoints

