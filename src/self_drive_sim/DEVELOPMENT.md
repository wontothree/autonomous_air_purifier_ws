# Functions

|Function name|Input|Output|Description|
|---|---|---|---|
|mission_planner|`air_sensor_pollution_data`, `robot_sensor_pollution_data`, `current_node_index`, `map_id`, `room_num`, `pollution_threshold`|`optimal_visit_order`||
|global_planner|`start_node_index`, `end_node_index`, `map_id`|`map_reference_waypoints`|start node에서 end node로 가는 reference waypoints를 출력한다. 사전에 `reference_waypoints`라는 거대한 행렬을 작성해두어야 한다.|
|controller|`current_robot_pose`, `target_position`, `linear_gain`, `angular_gain`, `max_linear`, `max_angular`, `angle_threshold`||
|localizer|`delta_distance`, `delta_yaw`, `scan_ranges`, `occupancy_grid_map`, `distance_map`, `self.map_origin`, `self.resolution`||
|finite_state_machine|`air_sensor_pollution_data`, `robot_sensor_pollution_data`, `current_time`, `pollution_end_time`, `current_robot_pose`, `current_fsm_state`, `map_id`, `room_num`, ||