# Functions

RMCLocalizer

|Step|Functions|Description|
|---|---|---|
|1|initialize_particle_set||
|1|initialize_reliability_set||
|2, 3|update_pose_and_reliability_set|update pose set and reliability set|
|5|calculate_likelihoods_by_measurement_model||
|5|calculate_likelihoods_by_decision_model||
|7|estimate_robot_pose||
|8|resample_particle_set||

ROS

|Functions|Description|Called Functions|
|---|---|---|
|callback_timer|||
|callback_initialpose||initialize_particle_set, initialize_reliability_set|
|callback_odom|||
|callback_scan||
||||
|publish_particle_set|||
|publish_pose|||