# [Package] rmc_localizer

Reliable Monte Carlo Localization Module

    mc_localizer
    ├── rmc_localizer   
    │   └── rmc_localizer_node.py
    │       ├── point             # dataclass
    │       ├── particle          # dataclass
    │       ├── RMCLocalizer      # main logic
    │       └── RMCLocalizerROS   # callback, visualization, and ROS functions
    │                 
    ├── setup.cfg     
    ├── setup.py                    
    └── package.xml    

# 🚀 Quick Start

Dependencies

```bash
sudo apt update
sudo apt install ros-humble-tf-transformations     # tf_transformations
```

```bash
colcon build
source /opt/ros/humble/setup.bash
source install/setup.bash
```

```bash
ros2 run rmc_localizer rmc_localizer_node
```

# 💬 Subscribed Topics

Following messages (topics) are needed to be published;

- initialpose
- [sensor_msgs/msg/LaserScan](https://docs.ros2.org/foxy/api/sensor_msgs/msg/LaserScan.html) (`/scan`) 
- [nav_msgs/msg/Odometry](https://docs.ros2.org/foxy/api/nav_msgs/msg/Odometry.html) (`/odom`)
- [nav_msgs/msg/OccupancyGrid](https://docs.ros2.org/foxy/api/nav_msgs/msg/OccupancyGrid.html) (`/map`)

Also, static transformation between following two frames is needed to be set.

- origin of a robot (base_link)
- 2D LiDAR (laser)

# 🗨️ Published Topics

# Node

`rmc_localizer_node`

# Reference

https://github.com/NaokiAkai/als_ros