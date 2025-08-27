# Architecture

---

# `mc_localizer_ros.hpp`

## Member Variables

|Index|Type|Variable|Initialization|Description|
|---|---|---|---|---|
||std::string|scan_topic_name_|||
||std::string|odom_topic_name_|||

## Functions

||Function||
|---|---|---|
|callback|callback_initial_pose|mcl_estimated_pose_를 업데이트하고, 함수 initialize_particle_set를 호출한다. (step 1)|
||callback_scan||
||callback_odom|delta_x_, delta_y_, delta_distance_, delta_yaw_를 업데이트한다.|
||callback_timer||
|||
|visualization|publish_particle_set||
||publish_pose||
|||


---

# `mc_localizer.hpp`

## Member Variables

|Index|Type|Variable|Description|
|---|---|---|---|
||`std::vector<Particle>`|`pose_tracking_particle_set_`||
||`std::vector<Particle>`|`global_localization_particle_set_`||
||int|`pose_tracking_particle_num_`||
||int|`global_localization_particle_num_`||
|||||
||mcl_estimated_pose_|||
||odom_pose_|||

## Functions

||Function|Input|Description|
|---|---|---|---|
|step 1|initialize_particle_set|||
||update_particles_by_motion_model||diffential drive model을 사용하여 `particle_set_` 업데이트||
||calculate_likelihoods_by_measurement_model|||
||calculate_likelihoods_by_decision_model|||
||calculate_likelihoods_from_global_localization|||
|step 7|estimate_robot_pose|||
|step 8|resample_particles|| `particle_set_` 업데이트|

---
