# [Package] self_drive_sim

    self_drive_sim
    ├── launch/
    ├── resources
    ├── self_drive_sim
        └── self_drive_sim
            └── agent
                └── agent.py
    ├── test
    └── worlds
        ├── map0.json
        ├── map0.npz
        ├── map0.world
        ├── map1.json
        ├── map1.npz
        ├── map1.world
        ├── map2.json
        ├── map2.npz
        ├── map2.world
        ├── map3.json
        ├── map3.npz
        └── map3.world

# Quick Start

## Teleoperation

```bash
# dependencies
apt-get update
apt-get install -y ros-humble-teleop-twist-keyboard

ros2 run teleop_twist_keyboard teleop_twist_keyboard --ros-args --remap cmd_vel:=/robot_0/cmd_vel
```
