import math
import numpy as np
import itertools

from .reference_waypoints import all_map_reference_waypoints, all_map_distance_matrices
from .local_costmap_generator import LocalCostMapGenerator
from .go_to_goal_controller import GoToGoalController
from .monte_carlo_localizer import MonteCarloLocalizer


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