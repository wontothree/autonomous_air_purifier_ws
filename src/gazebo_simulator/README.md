# [Package] gazebo_simulator

    gazebo_simulator
    ├── GazeboFiles                     # gazebo files
    │   ├── models
    │   └── world
    ├── meshes
    │   └── hokuyo.dae                  # 3d model blueprint
    ├── urdf                            # robot model
    │   ├── diffbot.gazebo              # gazebo plugin
    │   ├── diffbot.urdf                # useless
    │   └── diffbot.urdf.xacro          # main
    ├── worlds
    │   └── simulator.world             # whole simulation environment (entry point)
    │    
    ├── launch
    ├── include
    ├── src
    ├── CMakeLists.txt
    └── package.xml

    ├── rviz
    │   ├── description.rviz
    │   ├── diffbot.rviz
    │   ├── skibot.rviz
    │   └── tinybot.rviz

# 🚀 Quick Start

```bash
# dependencies
apt update
apt install ros-humble-gazebo-ros-pkgs       # gazebo
apt install ros-humble-joint-state-publisher # ros2 pkg list | grep joint_state_publisher
```

```bash
colcon build
source /opt/ros/humble/setup.bash
source install/setup.bash
```

```bash
gazebo /root/autonomous_navigation_ws/install/gazebo_simulator/share/gazebo_simulator/worlds/simulator.world
ros2 launch gazebo_simulator robot_world.launch.py
```

[http://localhost:8080/vnc.html](http://localhost:8080/vnc.html)

```bash
ps aux | grep -E 'gzserver|gzclient|ros2'

pkill -f gzserver
pkill -f gzclient
pkill -f ros2
```

## Teleoperation

```bash
# dependencies
apt-get update
apt-get install -y ros-humble-teleop-twist-keyboard

ros2 run teleop_twist_keyboard teleop_twist_keyboard --ros-args --remap cmd_vel:=/diffbot/cmd_vel
```

# 🗨️ Published Topics

- `/clicked_point`
- `/clock`
- `/diffbot/cmd_vel`
- `/diffbot/odom`
- `/diffbot/scan`
- `/goal_pose`
- [geometry_msgs/msg/PoseWithCovarianceStamped](https://docs.ros2.org/foxy/api/geometry_msgs/msg/PoseWithCovarianceStamped.html) (`/initialpose`)
- `/joint_states`
- `/performance_metrics`
- `/robot_description`
- `/tf`
- `/tf_static`

# Reference

https://github.com/Road-Balance/gcamp_ros2_basic
