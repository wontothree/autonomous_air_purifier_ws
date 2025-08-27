<div align="center">

  # Autonomous Navigation Workspace
  
  Indoor Autonomous Navigation of Mobile Robot with 2D LiDAR and IMU

  [![ROS2 Humble](https://img.shields.io/badge/ROS2-Humble-blue.svg)](https://docs.ros.org/en/humble/)
  [![Gazebo](https://img.shields.io/badge/Gazebo-11-orange.svg)](http://gazebosim.org/)
  [![C++](https://img.shields.io/badge/C%2B%2B-17-blue.svg)](https://isocpp.org/)

</div>

--- 

## 🚀 Quick Start

```bash
colcon build
source /opt/ros/humble/setup.bash
source install/setup.bash
```

```bash
ros2 launch gazebo_simulator robot_world.launch.py
ros2 run rmc_localizer rmc_localizer_node
```

---

## 🏗️ Technical Architecture

- [Localization] Monte Carlo Localization
- [Control] Model Predictive Path Integral Control

### Architecture Diagram

```mermaid
mindmap
  root(autonomous_navigation_ws)
    (mc_localizaer)
    (mppi_controller)
```

### Project Structure

    autonomous_navigation_ws
    └── src/
        ├── mc_localizater/              # localization
        └── mppi_controller/             # control

---
