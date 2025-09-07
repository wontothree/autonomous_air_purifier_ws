# Functions

|Function name|Input|Output|
|---|---|---|
|mission_planner|`air_sensor_pollution_data`, `robot_sensor_pollution_data`, `current_node_index`, `map_id`, `room_num`, `pollution_threshold`||
|global_planner|`start_node_index`, `end_node_index`, `map_id`||
|controller|`current_robot_pose`, `target_position`, `linear_gain`, `angular_gain`, `max_linear`, `max_angular`, `angle_threshold`||
|localizer|`delta_distance`, `delta_yaw`, `scan_ranges`, `occupancy_grid_map`, `distance_map`, `self.map_origin`, `self.resolution`||
|finite_state_machine|`air_sensor_pollution_data`, `robot_sensor_pollution_data`, `current_time`, `pollution_end_time`, `current_robot_pose`, `current_fsm_state`, `map_id`, `room_num`, ||