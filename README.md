<div align="center">

  # Autonomous Air Purifier
  
  Indoor Autonomous Navigation of Mobile Robot with 2D LiDAR and Odometry

  [![ROS2 Humble](https://img.shields.io/badge/ROS2-Humble-blue.svg)](https://docs.ros.org/en/humble/)
  [![Gazebo](https://img.shields.io/badge/Gazebo-11-orange.svg)](http://gazebosim.org/)

</div>

https://github.com/user-attachments/assets/5952e121-6102-4518-a836-806de1cbf719

--- 

## 🚀 Quick Start

1. Install `Dev Containers` in Vscode Extentions
2. Click `Dev Containers: Rebuild and Reopen Containers`
3. Build


```bash
# ~/autonomous_agent_ws 
colcon build
source /opt/ros/humble/setup.bash
source install/setup.bash
```

4. Run

```bash
ros2 launch self_drive_sim train_launch_map3.py
# or
ros2 launch self_drive_sim test_launch_map3.py
```

5. Simulation

[http://localhost:8080/vnc.html](http://localhost:8080/vnc.html)

---

## 🏗️ Technical Architecture

### Architecture Diagram

```mermaid
mindmap
  root(autonomous_agent_ws)
    (localizaer)
    (controller)
```

### Project Structure

    autonomous_agent_ws
    └── src/self_drive_sim/self_drive_sim/agent/
        ├── agent.py
        ├── autonomous_navigation.py
        ├── go_to_goal_controller.py
        ├── local_costmap_generator.py
        ├── monte_carlo_localization.py
        ├── mppi.py
        ├── map.py
        └── reference_waypoints.py

---

## Code

https://github.com/wontothree/autonomous_air_purifier_ws/blob/main/src/self_drive_sim/self_drive_sim/agent/agent.py

|Class||Function|
|---|---|---|
|`DistanceMatrixCalculator`|||
|`Map`||
|`Pose`||
|`Particle`||
|`MonteCarloLocalizer`||
|`LocalCostMapGenerator`||
|`GoToGoalController`||
|`AutonomousNavigator`||
|`Agent`||
|`ModelPredictivePathIntegralController`||





